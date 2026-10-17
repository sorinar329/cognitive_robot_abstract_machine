"""
What the Toyota HSR's RoboCasa tasks share: a RoboCasa kitchen lowered to the HSR's
height, the HSR standing in front of what it reaches for and moving its arm there, and
the task carried out against MuJoCo physics once the scene has settled.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing_extensions import Any, Callable

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from krrood.ormatic.utils import classproperty
from coraplex.datastructures.dataclasses import Context, MotionToleranceConfig
from coraplex.execution_environment import PhysicallySimulatedRobot
from coraplex.plans.plan import Plan
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from segmind.datastructures.events import DetectionEvent
from segmind.event_segmentation import Segmind
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
    RoboCasaTask,
)
from semantic_digital_twin.adapters.robocasa_dataset.mujoco_compat import (
    robocasa_version_assertions_relaxed,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

from experiments.hsr_kitchen_mujoco.demo import (
    FRICTION_CONE,
    IMPEDANCE_RATIO,
    NO_SLIP_ITERATIONS,
    SETTLING_DURATION,
    STEP_SIZE,
    hand_offset,
)

with robocasa_version_assertions_relaxed():
    from robocasa.models.scenes.scene_registry import (
        LayoutType,
        StyleType,
        get_layout_path,
    )

# %% the scene


class KitchenBody(StrEnum):
    """
    The bodies of the RoboCasa kitchen the experiment refers to, by their names.
    """

    FLOOR = "floor_room_main"
    """
    The floor, which the HSR's base stands on and so never keeps clear of.
    """

    FLOOR_BACKING = "floor_backing_room_main"
    """
    The slab under the floor, which the HSR's base comes as close to as to the floor.
    """


class ArmLink(StrEnum):
    """
    The links of the HSR's arm that a task may bring as close to what it handles as
    the hand itself, by the names the HSR's description gives them.
    """

    ARM_ROLL = "arm_roll_link"
    """
    The link the arm rolls, which carries the wrist.
    """

    WRIST_FLEX = "wrist_flex_link"
    """
    The link the wrist flexes.
    """

    WRIST_ROLL = "wrist_roll_link"
    """
    The link the wrist rolls, which carries the hand.
    """


@dataclass
class ArmTilt:
    """
    How far the HSR's hand is tilted in its arm's plane, and where that puts the tool
    frame relative to the wrist.
    """

    tilt: float
    """
    The sum of arm and wrist flex.
    """

    wrist_to_tool: np.ndarray
    """
    The tool frame's position relative to the wrist flex joint, relative to the HSR's
    root.
    """


@dataclass
class WristPlacement:
    """
    A position of the HSR's arm flex and lift that puts the wrist at a height asked
    for.
    """

    flex: float
    """
    The arm flex joint's position.
    """

    lift: float
    """
    The arm lift joint's position.
    """

    wrist_ahead: float
    """
    How far ahead of the HSR's root the wrist flex joint stands.
    """


@dataclass
class HSRBArmPlane:
    """
    The vertical plane the HSR's arm swings its hand in while its arm and wrist roll
    joints stand at zero, and the joint positions that put the hand at a pose in it.
    In that plane, the hand's tilt depends on the sum of arm and wrist flex alone, and
    the wrist's place on arm flex and lift alone.
    """

    search_steps: int = 600
    """
    How many positions each of the two searches tries within a joint's limits.
    """

    reach_resolution: float = 0.005
    """
    How near, in metres, a goal has to come to a distance the arm reaches for the arm
    to count as reaching it: about how far apart the searches' positions put the tool
    frame.
    """

    world: World = field(init=False)
    """
    A world holding nothing but an HSR, whose joints the searches set, so that no
    simulated robot moves.
    """

    robot: HSRB = field(init=False)
    """
    The HSR of :attr:`world`.
    """

    def __post_init__(self):
        self.world = World()
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
        self.robot = RobotSpecification(HSRB).spawn(self.world)

    def joint_positions_for(self, base_T_tool: np.ndarray) -> dict[HSRBJoint, float]:
        """
        :param base_T_tool: Where the tool frame is to be, relative to the HSR's root;
            it has to lie in the arm's plane.
        :return: The positions of the arm's joints that put the tool frame there, as
            near as the searches get, the wrist rolled to the goal's turn about the
            hand's approach.
        """
        arm_tilt = self._tilt_for(base_T_tool[:3, :3])
        wanted_wrist = base_T_tool[:3, 3] - arm_tilt.wrist_to_tool
        placement = min(
            self._wrist_placements(arm_tilt.tilt, wanted_wrist[2]),
            key=lambda candidate: abs(wanted_wrist[0] - candidate.wrist_ahead),
        )
        positions = {
            HSRBJoint.ARM_LIFT: float(placement.lift),
            HSRBJoint.ARM_FLEX: float(placement.flex),
            HSRBJoint.ARM_ROLL: 0.0,
            HSRBJoint.WRIST_FLEX: float(arm_tilt.tilt - placement.flex),
            HSRBJoint.WRIST_ROLL: 0.0,
        }
        positions[HSRBJoint.WRIST_ROLL] = self._wrist_roll_for(positions, base_T_tool)
        return positions

    def reachable_distances(
        self, tool_height: float, base_R_tool: np.ndarray
    ) -> list[float]:
        """
        :param tool_height: How high above the HSR's root the tool frame is to be.
        :param base_R_tool: How the tool frame is to be turned, relative to the HSR's
            root.
        :return: How far ahead of the HSR's root the arm puts the tool frame at that
            height, tilted as near to that turn as the arm allows, one distance per
            arm flex position the search tries that puts it there, in ascending order.
        """
        arm_tilt = self._tilt_for(base_R_tool)
        return sorted(
            float(placement.wrist_ahead + arm_tilt.wrist_to_tool[0])
            for placement in self._wrist_placements(
                arm_tilt.tilt, tool_height - arm_tilt.wrist_to_tool[2]
            )
        )

    def reaches_from(
        self,
        world_T_base: np.ndarray,
        tool_frame_goals: list[np.ndarray],
        reaches: list[list[float]],
    ) -> bool:
        """
        :param world_T_base: Where the HSR's root stands, in the kitchen's frame.
        :param tool_frame_goals: Where the tool frame is to be, in the kitchen's frame.
        :param reaches: For each goal, the :meth:`reachable_distances` at its height
            and turn.
        :return: Whether each goal lies at a distance ahead that the arm reaches it at.
        """
        base_T_world = np.linalg.inv(world_T_base)
        return all(
            min(abs(distance - (base_T_world @ goal)[0, 3]) for distance in reach)
            <= self.reach_resolution
            for goal, reach in zip(tool_frame_goals, reaches)
        )

    def _tilt_for(self, base_R_tool: np.ndarray) -> ArmTilt:
        """
        Tilt the hand, its arm and wrist roll joints at zero, as near to ``base_R_tool``
        as the arm allows.

        :param base_R_tool: How the tool frame is to be turned, relative to the HSR's
            root.
        :return: The tilt, and where it puts the tool frame relative to the wrist.
        """
        end_effector = self.robot.all_arms[0].end_effector
        wanted_approach = base_R_tool @ end_effector.approach_axis.to_np()[:3]
        self._set({HSRBJoint.ARM_ROLL: 0.0, HSRBJoint.WRIST_ROLL: 0.0})
        tilt = max(
            self._tilts(),
            key=lambda candidate: self._approach_at(candidate) @ wanted_approach,
        )
        self._set_tilt(tilt)
        return ArmTilt(
            tilt=float(tilt),
            wrist_to_tool=self._base_T(end_effector.tool_frame)[:3, 3] - self._wrist(),
        )

    def _wrist_placements(
        self, tilt: float, wrist_height: float
    ) -> list[WristPlacement]:
        """
        :param tilt: The sum of arm and wrist flex the hand is tilted by.
        :param wrist_height: How high above the HSR's root the wrist is to be.
        :return: The arm flex positions the search tries that leave the wrist flex
            within its limits, each with the lift that puts the wrist at
            ``wrist_height``, where that lies within the lift's limits.
        """
        lift_limits = self._limits(HSRBJoint.ARM_LIFT)
        wrist_limits = self._limits(HSRBJoint.WRIST_FLEX)
        placements = []
        for flex in np.linspace(*self._limits(HSRBJoint.ARM_FLEX), self.search_steps):
            if not wrist_limits[0] <= tilt - flex <= wrist_limits[1]:
                continue
            self._set({HSRBJoint.ARM_LIFT: 0.0, HSRBJoint.ARM_FLEX: flex})
            wrist = self._wrist()
            lift = wrist_height - wrist[2]
            if lift_limits[0] <= lift <= lift_limits[1]:
                placements.append(
                    WristPlacement(
                        flex=float(flex), lift=float(lift), wrist_ahead=float(wrist[0])
                    )
                )
        return placements

    def _wrist_roll_for(
        self, positions: dict[HSRBJoint, float], base_T_tool: np.ndarray
    ) -> float:
        """
        :param positions: The arm's joint positions, the wrist not rolled.
        :param base_T_tool: Where the tool frame is to be, relative to the HSR's root.
        :return: The wrist roll that turns the tool frame about the hand's approach as
            far as ``base_T_tool`` is turned.
        """
        tool_frame = self.robot.all_arms[0].end_effector.tool_frame
        self._set(positions)
        unrolled = self._base_T(tool_frame)[:3, :3]
        turn = unrolled.T @ base_T_tool[:3, :3]
        angle = math.atan2(turn[1, 0], turn[0, 0])

        def orientation_error(roll: float) -> float:
            self._set({HSRBJoint.WRIST_ROLL: roll})
            return np.linalg.norm(
                self._base_T(tool_frame)[:3, :3] - base_T_tool[:3, :3]
            )

        return min([angle, -angle], key=orientation_error)

    def _tilts(self) -> np.ndarray:
        """
        :return: The sums of arm and wrist flex the joints' limits allow.
        """
        flex = self._limits(HSRBJoint.ARM_FLEX)
        wrist = self._limits(HSRBJoint.WRIST_FLEX)
        return np.linspace(flex[0] + wrist[0], flex[1] + wrist[1], self.search_steps)

    def _set_tilt(self, tilt: float) -> None:
        """
        Tilt the hand by ``tilt``, the sum of arm and wrist flex, with the arm flexed as
        little as the wrist's limits allow.
        """
        flex = float(np.clip(tilt, *self._limits(HSRBJoint.ARM_FLEX)))
        self._set({HSRBJoint.ARM_FLEX: flex, HSRBJoint.WRIST_FLEX: tilt - flex})

    def _approach_at(self, tilt: float) -> np.ndarray:
        """
        :return: The direction the hand approaches along when tilted by ``tilt``,
            relative to the HSR's root.
        """
        self._set_tilt(tilt)
        end_effector = self.robot.all_arms[0].end_effector
        return (
            self._base_T(end_effector.tool_frame)[:3, :3]
            @ end_effector.approach_axis.to_np()[:3]
        )

    def _wrist(self) -> np.ndarray:
        """
        :return: Where the wrist flex joint is, relative to the HSR's root.
        """
        wrist = self.world.get_connection_by_name(HSRBJoint.WRIST_FLEX).child
        return self._base_T(wrist)[:3, 3]

    def _base_T(self, entity: KinematicStructureEntity) -> np.ndarray:
        """
        :return: The pose of ``entity`` relative to the HSR's root.
        """
        return self.world.compute_forward_kinematics_np(self.robot.root, entity)

    def _limits(self, joint: HSRBJoint) -> tuple[float, float]:
        """
        :return: The lower and upper position limit of ``joint``.
        """
        limits = self.world.get_connection_by_name(joint).dof.limits
        return limits.lower.position, limits.upper.position

    def _set(self, positions: dict[HSRBJoint, float]) -> None:
        """
        Set the joints of :attr:`world`'s HSR to ``positions``.
        """
        for joint, position in positions.items():
            self.world.get_connection_by_name(joint).position = position


class FixtureReference(StrEnum):
    """
    The names RoboCasa's tasks refer to their fixtures by.
    """

    DRAWER = "drawer"
    """
    The drawer a drawer task works.
    """

    COUNTER = "counter"
    """
    The counter above the drawer.
    """

    FIXTURE = "fxtr"
    """
    The fixture a door task works.
    """


class LayoutKey(StrEnum):
    """
    The keys of a RoboCasa layout this experiment reads or changes.
    """

    SIZE = "size"
    """
    A fixture's width, depth and height.
    """

    POSITION = "pos"
    """
    Where a fixture stands, along the wall, out from it, and up.
    """

    STACK_HEIGHT = "stack_height"
    """
    How high the fixtures below a stacked fixture reach.
    """

    MAIN_GROUP = "main_group"
    """
    The fixtures along the kitchen's main wall.
    """

    BOTTOM_ROW = "bottom_row"
    """
    The counters and appliances standing on the floor along the main wall.
    """

    TYPE = "type"
    """
    What kind of fixture an entry is.
    """

    NAME = "name"
    """
    The name of a fixture.
    """

    ISLAND_GROUP = "island_group"
    """
    The fixtures of a kitchen island standing free in the room.
    """


class FixtureType(StrEnum):
    """
    The kinds of fixture this experiment looks for in a RoboCasa layout.
    """

    COUNTER = "counter"
    """
    A counter.
    """


class RowOffTheFloor(StrEnum):
    """
    The rows of a RoboCasa layout whose fixtures hang on a wall or make up the room,
    rather than stand on the floor.
    """

    WALLS = "walls"
    """
    The walls themselves.
    """

    FLOOR = "floor"
    """
    The floor itself.
    """

    WALL_ACCESSORIES = "wall_accessories"
    """
    Outlets, switches and the like on the walls.
    """

    TOP_ROW = "top_row"
    """
    The wall cabinets and the appliances hanging between them.
    """

    WALL_STACK = "wall_stack"
    """
    Fixtures stacked on a wall.
    """

    WALL_STACKS = "wall_stacks"
    """
    Fixtures stacked on a wall, as some layouts spell it.
    """


@dataclass
class RoboCasaLayout:
    """
    A RoboCasa kitchen layout, as RoboCasa's layout files describe it.
    """

    configuration: dict[str, Any]
    """
    The layout, as read from its file.
    """

    @classmethod
    def from_layout_type(cls, layout: LayoutType) -> RoboCasaLayout:
        """
        :param layout: One of RoboCasa's layouts.
        :return: That layout, read from RoboCasa's file of it.
        """
        with open(get_layout_path(layout)) as layout_file:
            return cls(yaml.safe_load(layout_file))

    @property
    def counter_height(self) -> float:
        """
        How high the counters along the main wall stand, in metres.
        """
        return next(
            fixture[LayoutKey.SIZE][2]
            for fixture in self.configuration[LayoutKey.MAIN_GROUP][
                LayoutKey.BOTTOM_ROW
            ]
            if fixture[LayoutKey.TYPE] == FixtureType.COUNTER
        )

    def lowered_fixtures(self, counter_height: float) -> dict[str, dict[str, Any]]:
        """
        :param counter_height: How high the counters are to stand, in metres.
        :return: The updates, by fixture name, that scale every height of the fixtures
            standing on the floor by the same factor, so that the counters stand
            ``counter_height`` high; what hangs on the walls keeps its height, and
            where a group stands is left as it is.
        """
        factor = counter_height / self.counter_height
        return {
            fixture[LayoutKey.NAME]: update
            for group in self.configuration.values()
            for row_name, row in group.items()
            if row_name not in RowOffTheFloor and isinstance(row, list)
            for fixture in row
            if isinstance(fixture, dict) and (update := self._lowered(fixture, factor))
        }

    @staticmethod
    def _lowered(fixture: dict[str, Any], factor: float) -> dict[str, Any]:
        """
        :param fixture: A fixture of the layout.
        :param factor: What to multiply each of its heights with.
        :return: The fixture's size, position and stack height with their heights
            scaled, for those it states.
        """
        update = {}
        for key in (LayoutKey.SIZE, LayoutKey.POSITION):
            value = fixture.get(key)
            if (
                isinstance(value, list)
                and len(value) == 3
                and isinstance(value[2], (int, float))
            ):
                update[key] = [value[0], value[1], value[2] * factor]
        if LayoutKey.STACK_HEIGHT in fixture:
            update[LayoutKey.STACK_HEIGHT] = fixture[LayoutKey.STACK_HEIGHT] * factor
        return update


@dataclass
class HSRRoboCasaTask(ABC):
    """
    One of RoboCasa's tasks, performed by the HSR. Each task is a subclass named after
    the RoboCasa task it performs.
    """

    layout: LayoutType = LayoutType.LAYOUT001
    """
    The RoboCasa kitchen layout.
    """

    style: StyleType = StyleType.STYLE001
    """
    The RoboCasa kitchen style.
    """

    seed: int = 0
    """
    The seed RoboCasa samples the task's objects and their places with.
    """

    counter_height: float | None = None
    """
    How high the kitchen's counters are to stand, in metres, everything else standing
    on the floor scaled with them; ``None`` keeps RoboCasa's kitchen as it is.
    """

    tool_frame_tolerance: float = 0.015
    """
    How far, in metres, the tool frame may stop from a goal of the task's motions: the
    base stops within about a centimetre of where it is sent, and the arm, swinging in
    its plane, cannot make up for that sideways.
    """

    collision_avoidance: bool = True
    """
    Whether Giskard keeps the HSR clear of the kitchen and of itself.
    """

    headless: bool = False
    """
    Whether the simulation runs without MuJoCo's own viewer.
    """

    task: RoboCasaTask = field(init=False)
    """
    The task as RoboCasa composed it, built by :meth:`build_world`.
    """

    world: World = field(init=False)
    """
    The kitchen with the task's objects and the HSR, built by :meth:`build_world`.
    """

    detected_events: list[DetectionEvent] = field(init=False, default_factory=list)
    """
    What SegMind detected during the last run, in order.
    """

    @classproperty
    def task_name(cls) -> str:
        """
        The name of the RoboCasa task, which the subclass carries.
        """
        return cls.__name__

    @property
    def task_parameters(self) -> dict[str, Any]:
        """
        The arguments the RoboCasa task is composed with, by name.
        """
        return {}

    @property
    @abstractmethod
    def handled_bodies(self) -> list[Body]:
        """
        The bodies the HSR's hand works next to, which its wrist may touch as the hand
        may.
        """

    @abstractmethod
    def build_plan(self, context: Context) -> Plan:
        """
        :param context: The context the actions are built in.
        :return: The plan that performs the task, or its last stage.
        """

    @abstractmethod
    def build_segmind(self) -> Segmind:
        """
        :return: SegMind watching what the task's success is read from.
        """

    @property
    def stages(self) -> list[Callable[[Context], Plan]]:
        """
        What builds the plans the task is performed in, one after the other, each from
        where the previous one left the scene: :meth:`build_plan` alone.
        """
        return [self.build_plan]

    def build_world(self) -> World:
        """
        Compose the RoboCasa task scene, its kitchen at :attr:`counter_height`, and put
        the HSR at its
        :meth:`start_pose`, its arms parked as it would arrive.

        :return: The world.
        """
        self.task = RoboCasaDatasetLoader().load_task(
            self.task_name,
            self.layout,
            self.style,
            seed=self.seed,
            task_parameters=self.task_parameters,
            fixture_updates=(
                {}
                if self.counter_height is None
                else RoboCasaLayout.from_layout_type(self.layout).lowered_fixtures(
                    self.counter_height
                )
            ),
        )
        self.world = self.task.world
        robot = RobotSpecification(HSRB).spawn(self.world)
        robot.mobile_base.full_body_controlled = False
        with self.world.modify_world():
            self.world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=robot.bodies_with_collision,
                    body_group_b=[
                        self.world.get_body_by_name(floor)
                        for floor in (KitchenBody.FLOOR, KitchenBody.FLOOR_BACKING)
                    ],
                )
            )
            self.world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=self.bodies_touching_handled_bodies,
                    body_group_b=self.handled_bodies,
                )
            )
        self.robot.root.parent_connection.origin = self.start_pose().homogeneous_matrix
        for arm in self.robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(self.world)
        return self.world

    @property
    def bodies_touching_handled_bodies(self) -> list[Body]:
        """
        The HSR's bodies that may touch :attr:`handled_bodies`: the wrist's links,
        which taking an object from where it lies and setting it into its receptacle
        brings as close to them as the hand.
        """
        return self.arm_links([ArmLink.WRIST_FLEX, ArmLink.WRIST_ROLL])

    def arm_links(self, links: list[ArmLink]) -> list[Body]:
        """
        :param links: Links of the HSR's arm.
        :return: The bodies of those links.
        """
        return [
            self.world.get_body_in_branch_by_name(self.robot.root, link)
            for link in links
        ]

    @property
    def robot(self) -> HSRB:
        """
        The HSR.
        """
        return self.world.get_semantic_annotations_by_type(HSRB)[0]

    def colliding_bodies_below(self, root: Body) -> list[Body]:
        """
        :return: The bodies of the object rooted in ``root`` that carry its collision
            geometry: the root itself, or the bodies below it RoboCasa puts it in.
        """
        return [
            body
            for body in [root]
            + list(
                self.world.compute_descendent_child_kinematic_structure_entities(root)
            )
            if isinstance(body, Body) and body.has_collision()
        ]

    def extent(self, body: Body) -> VolumetricBoundingBox:
        """
        :return: The axis-aligned box the collision geometry of ``body`` occupies in the
            kitchen's frame.
        """
        return body.collision.as_bounding_box_collection_in_frame(
            self.world.root
        ).bounding_box()

    @property
    def counter_top_height(self) -> float:
        """
        How high the kitchen's counters stand, in metres.
        """
        if self.counter_height is None:
            return RoboCasaLayout.from_layout_type(self.layout).counter_height
        return self.counter_height

    @property
    def floor(self) -> Floor:
        """
        The annotation of the floor the HSR drives on.
        """
        floor = self.world.get_body_by_name(KitchenBody.FLOOR)
        return [
            annotation
            for annotation in self.world.get_semantic_annotations_by_type(Floor)
            if annotation.root is floor
        ][0]

    @property
    def base_collision_rule(self) -> AvoidExternalCollisions:
        """
        The rule that keeps the HSR's base away from the kitchen.
        """
        return self.world.collision_manager.external_collision_rule(
            self.robot.mobile_base.root
        )

    def start_pose(self) -> Pose:
        """
        :return: Where the HSR starts: on the floor its base stands on without
            entering the buffer zone its collision avoidance keeps it out of, as near
            as it gets to where RoboCasa starts its own robot, and facing as that robot
            does.
        """
        world_T_start = self.task.robot_base_pose.to_np()
        nearest = self.world.transform(
            self.floor.navigable_space(
                self.robot, self.base_collision_rule.buffer_zone_distance
            ).nearest_free_point(
                Point2(
                    float(world_T_start[0, 3]),
                    float(world_T_start[1, 3]),
                    reference_frame=self.world.root,
                )
            ),
            self.world.root,
        )
        return Pose.from_xyz_rpy(
            x=float(nearest.x),
            y=float(nearest.y),
            yaw=math.atan2(world_T_start[1, 0], world_T_start[0, 0]),
            reference_frame=self.world.root,
        )

    def standing_pose(
        self, tool_frame_goals: list[np.ndarray], start: Pose
    ) -> Pose | None:
        """
        :param tool_frame_goals: Where the tool frame is to be, one after the other, in
            the kitchen's frame, all along the first one's approach.
        :param start: Where the HSR stands before it goes there.
        :return: Where the HSR stands to put its tool frame at every goal: facing along
            the first goal's approach, on floor its base stands on without entering the
            buffer zone its collision avoidance keeps it out of, since from closer the
            arm, mounted at the base's front, comes into its own buffer zones as it
            lowers the hand; on floor a path leads to from ``start``, or from the free
            point nearest it,
            from where the arm keeps its buffer zones at every goal, see
            :meth:`arm_keeps_its_buffer_zones`, at a distance from which the arm reaches every
            goal; of those, the one that leaves the first goal nearest the middle of
            the distances the arm reaches it at; ``None`` if there is no such pose.
        """
        end_effector = self.robot.all_arms[0].end_effector
        approach = tool_frame_goals[0][:3, :3] @ end_effector.approach_axis.to_np()[:3]
        heading = math.atan2(approach[1], approach[0])
        world_R_base = Rotation.from_euler("z", heading).as_matrix()
        floor_height = self.robot.root.global_pose.to_np()[2, 3]
        arm_plane = HSRBArmPlane()
        reaches = [
            arm_plane.reachable_distances(
                goal[2, 3] - floor_height, world_R_base.T @ goal[:3, :3]
            )
            for goal in tool_frame_goals
        ]
        if not all(reaches):
            return None
        navigable_space = self.floor.navigable_space(
            self.robot, self.base_collision_rule.buffer_zone_distance
        )
        here = navigable_space.nearest_free_point(Point2.from_pose(start))
        planning_world = self.planning_world()
        first_reach = reaches[0]
        middle = (first_reach[0] + first_reach[-1]) / 2
        for distance in sorted(first_reach, key=lambda reach: abs(reach - middle)):
            candidate = self._standing_pose_at(tool_frame_goals[0], heading, distance)
            standing_point = Point2.from_pose(candidate)
            if (
                arm_plane.reaches_from(candidate.to_np(), tool_frame_goals, reaches)
                and navigable_space.is_free(standing_point)
                and navigable_space.path_from_to(here, standing_point) is not None
                and self.arm_keeps_its_buffer_zones(
                    planning_world, arm_plane, candidate.to_np(), tool_frame_goals
                )
            ):
                return candidate
        return None

    def planning_world(self) -> World:
        """
        :return: A copy of the kitchen, with the HSR, to try arm configurations out in:
            the HSR's base and the bodies it handles are allowed to come as near as they
            come during the task, the base already kept clear by the floor it stands
            on.
        """
        planning_world = deepcopy(self.world)
        robot = planning_world.get_semantic_annotations_by_type(HSRB)[0]
        kitchen = [
            body
            for body in planning_world.bodies_with_collision
            if body not in robot.bodies_with_collision
        ]
        handled = [
            planning_world.get_body_by_name(body.name) for body in self.handled_bodies
        ]
        with planning_world.modify_world():
            planning_world.collision_manager.add_temporary_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=robot.mobile_base.bodies_with_collision,
                    body_group_b=kitchen,
                )
            )
            planning_world.collision_manager.add_temporary_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=robot.all_arms[0].end_effector.bodies_with_collision,
                    body_group_b=handled,
                )
            )
        planning_world.collision_manager.update_collision_matrix()
        return planning_world

    def arm_keeps_its_buffer_zones(
        self,
        planning_world: World,
        arm_plane: HSRBArmPlane,
        world_T_base: np.ndarray,
        tool_frame_goals: list[np.ndarray],
    ) -> bool:
        """
        :param planning_world: A :meth:`planning_world`.
        :param arm_plane: The arm plane that finds the arm's joint positions.
        :param world_T_base: Where the HSR's root stands, in the kitchen's frame.
        :param tool_frame_goals: Where the tool frame is to be, in the kitchen's frame.
        :return: Whether, standing there and with its arm put at each goal in turn, the
            HSR keeps every body out of the buffer zones its collision avoidance keeps
            it out of.
        """
        robot = planning_world.get_semantic_annotations_by_type(HSRB)[0]
        robot.root.parent_connection.origin = HomogeneousTransformationMatrix(
            world_T_base, reference_frame=planning_world.root
        )
        base_T_world = np.linalg.inv(world_T_base)
        for goal in tool_frame_goals:
            for joint, position in arm_plane.joint_positions_for(
                base_T_world @ goal
            ).items():
                planning_world.get_connection_by_name(joint).position = position
            if robot.is_within_a_buffer_zone:
                return False
        return True

    def _standing_pose_at(
        self, tool_frame_goal: np.ndarray, heading: float, distance: float
    ) -> Pose:
        """
        :param tool_frame_goal: Where the tool frame is to be, in the kitchen's frame.
        :param heading: The direction the HSR faces, as a yaw in the kitchen's frame.
        :param distance: How far ahead of the base's centre the goal is to be.
        :return: Where the HSR stands to have the goal that far ahead, with its hand
            rather than its base in line with it.
        """
        facing = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-math.sin(heading), math.cos(heading)])
        x, y = (
            tool_frame_goal[:2, 3] - distance * facing - hand_offset(self.world) * left
        )
        return Pose.from_xyz_rpy(
            x=float(x), y=float(y), yaw=heading, reference_frame=self.world.root
        )

    def in_kitchen_frame(self, poses: list[Pose]) -> list[np.ndarray]:
        """
        :param poses: Poses, in any frame.
        :return: The poses, in the kitchen's frame.
        """
        return [self.world.transform(pose, self.world.root).to_np() for pose in poses]

    def arm_motions_to(
        self, tool_frame_goal: Pose, standing_pose: Pose
    ) -> list[MoveJointsMotion]:
        """
        :param tool_frame_goal: Where the tool frame is to be.
        :param standing_pose: Where the HSR stands, see :meth:`standing_pose`.
        :return: The motions that bring the arm there in joint space: the arm lift
            first, so that the hand leaves its park pose above the counter rather than
            in front of the cabinets below it, then every joint of the arm.
        """
        world_T_base = self.world.transform(standing_pose, self.world.root).to_np()
        world_T_goal = self.world.transform(tool_frame_goal, self.world.root).to_np()
        joints = HSRBArmPlane().joint_positions_for(
            np.linalg.inv(world_T_base) @ world_T_goal
        )
        return [
            MoveJointsMotion(
                names=[HSRBJoint.ARM_LIFT], positions=[joints[HSRBJoint.ARM_LIFT]]
            ),
            MoveJointsMotion(names=list(joints), positions=list(joints.values())),
        ]

    def prepare_contacts(self) -> None:
        """
        Set up how the task's objects make contact in MuJoCo, before the simulation is
        built. Leaves them as RoboCasa gives them.
        """

    def run(self) -> None:
        """
        Let the scene settle, then build each of the :attr:`stages`' plans from where
        everything came to rest and carry it out against MuJoCo physics, watched by
        :meth:`build_segmind`'s SegMind, whose events are kept in
        :attr:`detected_events`.
        """
        context = Context(
            world=self.world,
            robot=self.robot,
            evaluate_conditions=False,
            motion_tolerances=MotionToleranceConfig(
                default_tcp_position_threshold=self.tool_frame_tolerance
            ),
        )
        self.prepare_contacts()
        simulation = MujocoSim(
            world=self.world,
            headless=self.headless,
            step_size=STEP_SIZE,
            cone=FRICTION_CONE,
            impratio=IMPEDANCE_RATIO,
            noslip_iterations=NO_SLIP_ITERATIONS,
        )
        simulation.start_stepped_simulation()
        try:
            simulation.step_simulation(SETTLING_DURATION)
            segmind = self.build_segmind()
            with segmind:
                for build_stage in self.stages:
                    plan = build_stage(context)
                    with PhysicallySimulatedRobot(
                        simulation=simulation,
                        collision_avoidance=self.collision_avoidance,
                    ):
                        plan.perform()
                simulation.step_simulation(SETTLING_DURATION)
            self.detected_events = segmind.event_logger.get_events()
        finally:
            simulation.stop_simulation()
