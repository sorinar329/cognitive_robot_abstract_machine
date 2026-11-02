"""
What the Toyota HSR's RoboCasa tasks share: the kitchen RoboCasa composes for a task,
optionally lowered, with the HSR standing on its free floor, a search for where the HSR
stands for its hand to reach a pose, and the task performed in MuJoCo.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
from cramph.context import ContextExtension
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.robocasa_dataset.layout import RoboCasaLayout
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
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    PlanarGraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.world_entity import Body

from experiments.hsr_robocasa.arm_plane import HSRBArmPlane

with robocasa_version_assertions_relaxed():
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

# %% where the HSR stands


@dataclass
class StanceSearch:
    """
    What the search for where the HSR stands reads, built once for the scene as it
    stands.
    """

    navigable_space: PlanarGraphOfBoundingBoxes
    """
    The floor the HSR's base stands on without entering the buffer zone its collision
    avoidance keeps it out of.
    """

    planning_world: World
    """
    A copy of the kitchen, with the HSR, to try arm configurations out in.
    """

    arm_plane: HSRBArmPlane
    """
    The plane the HSR's arm swings its hand in.
    """


# %% a RoboCasa task, performed by the HSR


@dataclass
class HSRRoboCasaTask(RobotDemonstration, ABC):
    """
    One of RoboCasa's tasks, performed by the HSR in MuJoCo. Each task is a subclass
    named after the RoboCasa task it performs.
    """

    used_robot: type[AbstractRobot] = HSRB

    ros_node_name: str = "hsr_robocasa"

    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor

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

    friction_cone: mujoco.mjtCone = mujoco.mjtCone.mjCONE_ELLIPTIC
    """
    The friction cone the contacts use: the elliptic cone is the exact one, where the
    default pyramid only approximates it and lets a held object creep between the
    fingers.
    """

    impedance_ratio: float = 10.0
    """
    How much harder sliding is than pressing in: MuJoCo's own guidance for grasping with
    an elliptic cone.
    """

    no_slip_iterations: int = 10
    """
    Passes spent removing the slip soft contacts leave, so a held object stays where the
    fingers hold it.
    """

    task: RoboCasaTask = field(init=False)
    """
    The task as RoboCasa composed it, loaded by :meth:`build_scene`.
    """

    @classproperty
    def task_name(cls) -> str:
        """
        The name of the RoboCasa task, which the subclass carries.
        """
        return cls.__name__

    @property
    @abstractmethod
    def handled_bodies(self) -> list[Body]:
        """
        The bodies the HSR's hand works next to.
        """

    # %% the kitchen and the HSR in it

    def build_scene(self) -> World:
        """
        The kitchen RoboCasa composes for the task, at :attr:`counter_height`, with the
        task's objects: RoboCasa's own world, so that the objects' free joints stay at
        the top level of the scene, where MuJoCo needs them.
        """
        self.task = RoboCasaDatasetLoader().load_task(
            self.task_name,
            self.layout,
            self.style,
            seed=self.seed,
            fixture_updates=(
                None
                if self.counter_height is None
                else RoboCasaLayout.from_layout_type(
                    self.layout
                ).fixture_updates_for_counter_height(self.counter_height)
            ),
        )
        return self.task.world

    def scene_specification(self) -> WorldSpecification:
        """
        Not read: :meth:`build_scene` takes RoboCasa's world as the scene.
        """
        return WorldSpecification()

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(self.used_robot)

    def is_scene_populated(self, world: World) -> bool:
        return bool(world.get_semantic_annotations_by_type(Floor))

    def populate_scene(self, world: World) -> None:
        """
        Nothing to add: RoboCasa places the task's objects itself, see
        :meth:`build_scene`.
        """

    def build_simulated_world(self) -> World:
        """
        As :meth:`RobotDemonstration.build_simulated_world`, with the HSR allowed to
        touch the floor, standing at its :meth:`start_pose` with its arm parked.
        """
        world = super().build_simulated_world()
        robot = world.get_semantic_annotations_by_type(HSRB)[0]
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=robot.bodies_with_collision,
                    body_group_b=[
                        floor.root
                        for floor in world.get_semantic_annotations_by_type(Floor)
                    ],
                )
            )
        robot.root.parent_connection.origin = self.start_pose(world).homogeneous_matrix
        for arm in robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        return world

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [RobotAccess(world.get_semantic_annotations_by_type(HSRB)[0])]

    def build_simulation(self, world: World) -> MujocoSim:
        return MujocoSim(
            world=world,
            headless=self.headless,
            cone=self.friction_cone,
            impratio=self.impedance_ratio,
            noslip_iterations=self.no_slip_iterations,
        )

    # %% reading the scene

    @staticmethod
    def robot_in(world: World) -> HSRB:
        """
        :return: The HSR of ``world``.
        """
        return world.get_semantic_annotations_by_type(HSRB)[0]

    @staticmethod
    def floor_in(world: World) -> Floor:
        """
        :return: The floor the HSR drives on: the topmost of the kitchen's floors.
        """
        return max(
            world.get_semantic_annotations_by_type(Floor),
            key=lambda floor: floor.root.collision.as_bounding_box_collection_in_frame(
                world.root
            )
            .bounding_box()
            .max_z,
        )

    @staticmethod
    def extent(body: Body) -> VolumetricBoundingBox:
        """
        :return: The axis-aligned box the collision geometry of ``body`` occupies in its
            world's frame.
        """
        return body.collision.as_bounding_box_collection_in_frame(
            body._world.root
        ).bounding_box()

    @staticmethod
    def colliding_bodies_below(root: Body) -> list[Body]:
        """
        :return: The bodies of the object rooted in ``root`` that carry its collision
            geometry: the root itself, or the bodies below it RoboCasa puts it in.
        """
        world = root._world
        return [
            body
            for body in [root]
            + list(world.compute_descendent_child_kinematic_structure_entities(root))
            if isinstance(body, Body) and body.has_collision()
        ]

    def base_collision_rule(self, world: World) -> AvoidExternalCollisions:
        """
        :return: The rule that keeps the HSR's base away from the kitchen.
        """
        return world.collision_manager.external_collision_rule(
            self.robot_in(world).mobile_base.root
        )

    def navigable_space(self, world: World) -> PlanarGraphOfBoundingBoxes:
        """
        :return: The floor the HSR's base stands on without entering the buffer zone
            its collision avoidance keeps it out of.
        """
        return self.floor_in(world).navigable_space(
            self.robot_in(world), self.base_collision_rule(world).buffer_zone_distance
        )

    def start_pose(self, world: World) -> Pose:
        """
        :return: Where the HSR starts: on the floor its base stands on without entering
            the buffer zone its collision avoidance keeps it out of, as near as it gets
            to where RoboCasa starts its own robot, and facing as that robot does.
        """
        world_T_start = self.task.robot_base_pose.to_np()
        nearest = world.transform(
            self.navigable_space(world).nearest_free_point(
                Point2(
                    float(world_T_start[0, 3]),
                    float(world_T_start[1, 3]),
                    reference_frame=world.root,
                )
            ),
            world.root,
        )
        return Pose.from_xyz_rpy(
            x=float(nearest.x),
            y=float(nearest.y),
            yaw=math.atan2(world_T_start[1, 0], world_T_start[0, 0]),
            reference_frame=world.root,
        )

    # %% the search for where the HSR stands

    def stance_search(self, world: World) -> StanceSearch:
        """
        :return: What :meth:`standing_pose` reads, built for the scene as it stands now.
        """
        return StanceSearch(
            navigable_space=self.navigable_space(world),
            planning_world=self.planning_world(world),
            arm_plane=HSRBArmPlane(),
        )

    def planning_world(self, world: World) -> World:
        """
        :return: A copy of ``world``, with the HSR, to try arm configurations out in:
            the HSR's base and the bodies it handles are allowed to come as near as they
            come during the task, the base already kept clear by the floor it stands on.
        """
        planning_world = deepcopy(world)
        robot = self.robot_in(planning_world)
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

    def standing_pose(
        self,
        world: World,
        tool_frame_goals: list[np.ndarray],
        start: Pose,
        search: StanceSearch,
    ) -> Pose | None:
        """
        :param world: The world the HSR performs the task in.
        :param tool_frame_goals: Where the tool frame is to be, one after the other, in
            the kitchen's frame, all along the first one's approach.
        :param start: Where the HSR stands before it goes there.
        :param search: The :meth:`stance_search` for the scene as it stands.
        :return: Where the HSR stands to put its tool frame at every goal: facing along
            the first goal's approach, on floor its base stands on without entering the
            buffer zone its collision avoidance keeps it out of and a path leads to from
            ``start``, from where the arm reaches every goal and keeps its buffer zones
            at each; of those, the one that leaves the first goal nearest the middle of
            the distances the arm reaches it at; ``None`` if there is no such pose.
        """
        end_effector = self.robot_in(world).all_arms[0].end_effector
        approach = tool_frame_goals[0][:3, :3] @ end_effector.approach_axis.to_np()[:3]
        heading = math.atan2(approach[1], approach[0])
        world_R_base = Rotation.from_euler("z", heading).as_matrix()
        floor_height = self.robot_in(world).root.global_pose.to_np()[2, 3]
        arm_plane = search.arm_plane
        reaches = [
            arm_plane.reachable_distances(
                goal[2, 3] - floor_height, world_R_base.T @ goal[:3, :3]
            )
            for goal in tool_frame_goals
        ]
        if not all(reaches):
            return None
        navigable_space = search.navigable_space
        here = navigable_space.nearest_free_point(Point2.from_pose(start))
        first_reach = reaches[0]
        middle = (first_reach[0] + first_reach[-1]) / 2
        for distance in sorted(first_reach, key=lambda reach: abs(reach - middle)):
            candidate = self._standing_pose_at(
                world, tool_frame_goals[0], heading, distance
            )
            standing_point = Point2.from_pose(candidate)
            if (
                arm_plane.reaches_from(candidate.to_np(), tool_frame_goals, reaches)
                and navigable_space.is_free(standing_point)
                and navigable_space.path_from_to(here, standing_point) is not None
                and self.arm_keeps_its_buffer_zones(
                    search.planning_world,
                    arm_plane,
                    candidate.to_np(),
                    tool_frame_goals,
                )
            ):
                return candidate
        return None

    @staticmethod
    def arm_keeps_its_buffer_zones(
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
        self, world: World, tool_frame_goal: np.ndarray, heading: float, distance: float
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
            tool_frame_goal[:2, 3] - distance * facing - self.hand_offset(world) * left
        )
        return Pose.from_xyz_rpy(
            x=float(x), y=float(y), yaw=heading, reference_frame=world.root
        )

    def hand_offset(self, world: World) -> float:
        """
        :return: How far to its left of its base's centre the HSR carries its hand, in
            metres: the arm swings about a flex joint mounted beside the torso, and the
            base has to make up for it.
        """
        robot = self.robot_in(world)
        flexed_link = world.get_connection_by_name(HSRBJoint.ARM_FLEX).child
        return float(world.compute_forward_kinematics_np(robot.root, flexed_link)[1, 3])

    @staticmethod
    def in_kitchen_frame(world: World, poses: list[Pose]) -> list[np.ndarray]:
        """
        :return: ``poses`` as homogeneous matrices in the frame of ``world``'s root.
        """
        return [world.transform(pose, world.root).to_np() for pose in poses]
