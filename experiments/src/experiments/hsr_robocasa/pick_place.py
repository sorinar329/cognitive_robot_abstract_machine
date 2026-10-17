"""
The Toyota HSR performs RoboCasa's pick-and-place tasks in a RoboCasa kitchen,
simulated in MuJoCo: it picks an object up where the task puts it and sets it into the
receptacle the task names.

RoboCasa composes the kitchen and samples the objects; the HSR replaces the robot
RoboCasa would use. The object is held by nothing but the friction between it and the
fingers.

Run the stove task with::

    python -m experiments.hsr_robocasa.pick_place
"""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass, field
from enum import StrEnum
from typing_extensions import Any, Callable

import numpy as np
from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.navigation import PathPlanningNavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.mixins import GraspPoseSequence
from segmind.detectors.agent_event_detector_nodes import GraspDetector
from segmind.detectors.coarse_event_detector_nodes import PlacingDetector
from segmind.event_segmentation import Segmind
from semantic_digital_twin.adapters.multi_sim import (
    ContactDimensionality,
    MujocoGeom,
)
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    HasGraspCandidates,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Sink,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.spatial_types import Point2, Point3
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    PlanarGraphOfBoundingBoxes,
    VolumetricGraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from experiments.hsr_robocasa.drawers import HSRTaskAtDrawer
from experiments.hsr_robocasa.exceptions import NoApproachError
from experiments.hsr_robocasa.robocasa_task import (
    ArmLink,
    FixtureReference,
    HSRRoboCasaTask,
)

# %% the task's objects


class TaskObject(StrEnum):
    """
    The bodies RoboCasa adds for a pick-and-place task, by the names it gives them.
    """

    FOOD = "obj_main"
    """
    The object to pick up and set down, food in most tasks.
    """

    SECOND_OBJECT = "obj2_main"
    """
    A second object to pick up and set down, where the task has one.
    """

    CONTAINER = "container_main"
    """
    The container the object is to be set into, where the task has one.
    """


@dataclass
class Approach:
    """
    A direction the hand comes in from to take hold of an object.
    """

    heading: float
    """
    The direction the hand comes in towards, as a yaw in the kitchen's frame.
    """

    tilt: float
    """
    How steeply the hand comes down, in radians below the horizontal.
    """


@dataclass
class HSRPickPlaceTask(HSRRoboCasaTask, ABC):
    """
    One of RoboCasa's pick-and-place tasks, performed by the HSR.
    """

    food_category: str | None = None
    """
    The RoboCasa object category the object to pick up is sampled from; ``None``
    leaves the choice to the task.
    """

    lift: float = 0.1
    """
    How far the HSR lifts the food from where it took hold of it, in metres, before
    carrying it to the receptacle.
    """

    release_clearance: float = 0.03
    """
    How far above :attr:`release_height` the HSR lets the food go, in metres.
    """

    floor_margin: float = 0.05
    """
    How far in from the edge of the receptacle's floor the food is set down, in
    metres.
    """

    tilt_steps: int = 7
    """
    How many hand tilts, evenly from level to short of straight down, the search for an
    approach tries: straight down leaves no direction for the HSR to face.
    """

    heading_steps: int = 12
    """
    How many headings, evenly round the full turn, the search for an approach tries at
    each tilt.
    """

    hand_half_width: float = field(default=0.0, init=False)
    """
    How far, at most, the hand reaches out to a side of its approach, in metres, as
    :meth:`build_world` measures it with the gripper closed as the HSR is spawned: the
    gripper is left open after setting an object down.
    """

    already_moved: list[StrEnum] = field(default_factory=list, init=False)
    """
    The names of the objects of :attr:`moved_objects` the stages built so far move.
    """

    moving: StrEnum = field(default=TaskObject.FOOD, init=False)
    """
    The name of the object the stage being built picks up and sets down, which
    :attr:`food` refers to.
    """

    @property
    def moved_objects(self) -> list[StrEnum]:
        """
        The names of the objects the task picks up and sets down, one after the other:
        the food.
        """
        return [TaskObject.FOOD]

    @property
    def moved_together(self) -> list[StrEnum]:
        """
        The names of the objects of :attr:`moved_objects` set down in the same
        receptacle as the one being moved, see :attr:`moving`, it among them: all of
        them.
        """
        return self.moved_objects

    def _distance_to_floor(
        self, navigable_space: PlanarGraphOfBoundingBoxes, moved: StrEnum
    ) -> float:
        """
        :return: How far, horizontally, the object named ``moved`` lies from the
            nearest point of ``navigable_space``.
        """
        middle = self.extent(self.world.get_body_by_name(moved)).center.to_np()[:2]
        nearest = self.world.transform(
            navigable_space.nearest_free_point(
                Point2(*middle, reference_frame=self.world.root)
            ),
            self.world.root,
        )
        return float(np.linalg.norm(nearest.to_np()[:2] - middle))

    def build_world(self) -> World:
        """
        As :meth:`HSRRoboCasaTask.build_world`, with the first of :attr:`moved_objects`
        the one :attr:`moving` names until a stage moves another, and the
        :attr:`hand_half_width` measured.
        """
        world = super().build_world()
        self.moving = self.moved_objects[0]
        self.hand_half_width = self.measure_hand_half_width()
        return world

    @property
    def stages(self) -> list[Callable[[Context], Plan]]:
        """
        One stage per object of :attr:`moved_objects`, each built from where the
        previous one left the scene, see :meth:`build_next_plan`.
        """
        return [self.build_next_plan for _ in self.moved_objects]

    def build_next_plan(self, context: Context) -> Plan:
        """
        :param context: The context the actions are built in.
        :return: The plan that moves, of the objects of :attr:`moved_objects` not
            moved yet, the one nearest the floor the HSR's base can stand on that has
            an approach, see :meth:`plan_if_approachable`: the arm reaching for one
            further back passes over those in front of it, and one in a pile may be
            reachable only once its neighbours are gone.
        :raises NoApproachError: If no object left to move has an approach.
        """
        navigable_space = self.floor.navigable_space(
            self.robot, self.base_collision_rule.buffer_zone_distance
        )
        left_to_move = sorted(
            (moved for moved in self.moved_objects if moved not in self.already_moved),
            key=lambda moved: self._distance_to_floor(navigable_space, moved),
        )
        for moved in left_to_move:
            self.moving = moved
            plan = self.plan_if_approachable(context)
            if plan is not None:
                self.already_moved.append(moved)
                return plan
        raise NoApproachError(self.world.get_body_by_name(left_to_move[0]))

    @property
    def task_parameters(self) -> dict[str, Any]:
        """
        The object category the food is to be sampled from, when :attr:`food_category`
        names one.
        """
        return {} if self.food_category is None else {"obj_groups": self.food_category}

    @property
    def handled_bodies(self) -> list[Body]:
        """
        The task's objects and the receptacle.
        """
        return [
            body
            for root in [*self.task.manipulated_objects, self.receptacle]
            for body in self.colliding_bodies_below(root)
        ]

    @property
    def grasp_point(self) -> np.ndarray:
        """
        Where the food is taken hold of: its middle.
        """
        return self.extent(self.food.root).center.to_np()[:3]

    @property
    def food(self) -> HasGraspCandidates:
        """
        The annotation of the object being moved, see :attr:`moving`.
        """
        food = self.world.get_body_by_name(self.moving)
        return [
            annotation
            for annotation in self.world.get_semantic_annotations_by_type(
                HasGraspCandidates
            )
            if annotation.root is food
        ][0]

    @property
    def receptacle(self) -> Body:
        """
        The root of what the object is to be set into: the task's container.
        """
        return self.world.get_body_by_name(TaskObject.CONTAINER)

    @property
    def receptacle_floor(self) -> VolumetricBoundingBox:
        """
        The floor of the receptacle, in the kitchen's frame: the widest of the convex
        pieces RoboCasa builds its collision geometry from.
        """
        pieces = [
            piece
            for body in self.colliding_bodies_below(self.receptacle)
            for piece in body.collision.as_bounding_box_collection_in_frame(
                self.world.root
            ).bounding_boxes
        ]
        return max(pieces, key=lambda piece: piece.width * piece.depth)

    def build_segmind(self) -> Segmind:
        """
        :return: SegMind watching the objects of :attr:`moved_objects` be taken hold
            of and put down.
        """
        return Segmind.create_for_bodies(
            self.world,
            [self.world.get_body_by_name(moved) for moved in self.moved_objects],
            [GraspDetector, PlacingDetector],
        )

    def approaches(self) -> list[Approach]:
        """
        :return: The approaches the search tries, in order: the tilts from level
            down, and at each tilt the headings nearest the bearing to the food from
            the floor nearest it that the HSR's base can stand on first, along which
            the arm reaches over the least of what the food stands on.
        """
        navigable_space = self.floor.navigable_space(
            self.robot, self.base_collision_rule.violated_distance
        )
        nearest_floor = self.world.transform(
            navigable_space.nearest_free_point(
                Point2(*self.grasp_point[:2], reference_frame=self.world.root)
            ),
            self.world.root,
        )
        to_food = self.grasp_point[:2] - nearest_floor.to_np()[:2]
        bearing = math.atan2(to_food[1], to_food[0])
        headings = sorted(
            (
                bearing + 2 * math.pi * step / self.heading_steps
                for step in range(self.heading_steps)
            ),
            key=lambda heading: abs(math.remainder(heading - bearing, 2 * math.pi)),
        )
        tilts = np.linspace(0.0, math.pi / 2, self.tilt_steps + 1)[:-1]
        return [
            Approach(heading=heading, tilt=float(tilt))
            for tilt in tilts
            for heading in headings
        ]

    def grasp(self, approach: Approach) -> GraspCandidate:
        """
        :return: The grasp that takes the food at :attr:`grasp_point`, coming in as
            ``approach`` says, the fingers closing across its heading.
        """
        food = self.food.root
        world_T_grasp = HomogeneousTransformationMatrix.from_xyz_rpy(
            *self.grasp_point,
            pitch=approach.tilt,
            yaw=approach.heading,
            reference_frame=self.world.root,
        )
        food_T_grasp = self.world.transform(world_T_grasp.pose, food)
        return GraspCandidate(self.food, food_T_grasp)

    def measure_hand_half_width(self) -> float:
        """
        :return: How far, at most, the hand reaches out to a side of its approach, in
            metres, its fingers as they stand.
        """
        end_effector = self.robot.all_arms[0].end_effector
        approach = end_effector.approach_axis.to_np()[:3]
        extents = [
            body.collision.as_bounding_box_collection_in_frame(
                end_effector.tool_frame
            ).bounding_box()
            for body in end_effector.bodies_with_collision
        ]
        corners = np.array(
            [
                [x, y, z]
                for extent in extents
                for x in (extent.min_x, extent.max_x)
                for y in (extent.min_y, extent.max_y)
                for z in (extent.min_z, extent.max_z)
            ],
            dtype=float,
        )
        across = corners - np.outer(corners @ approach, approach)
        return float(np.linalg.norm(across, axis=1).max())

    def reach_space(self, approach_clearance: float) -> VolumetricGraphOfBoundingBoxes:
        """
        :param approach_clearance: How far the hand stays off the food where its
            approach starts.
        :return: The free space around the food that the middle of the hand can pass
            through: everything but the food kept :attr:`hand_half_width` clear of, as
            far around the food as an approach starts and up to :attr:`lift` above it.
        """
        extent = self.extent(self.food.root)
        center = extent.center.to_np()[:3]
        around = (
            np.linalg.norm([extent.width, extent.depth, extent.height]) / 2
            + approach_clearance
            + self.hand_half_width
        )
        search_space = BoundingBoxCollection(
            [
                VolumetricBoundingBox(
                    min_x=center[0] - around,
                    max_x=center[0] + around,
                    min_y=center[1] - around,
                    max_y=center[1] + around,
                    min_z=center[2] - around,
                    max_z=center[2] + max(around, self.lift + extent.height),
                    origin=HomogeneousTransformationMatrix(
                        reference_frame=self.world.root
                    ),
                )
            ],
            self.world.root,
        )
        return VolumetricGraphOfBoundingBoxes.free_space_from_world(
            self.world,
            search_space,
            bloat_obstacles=self.hand_half_width,
            obstacle_height_clearance=self.hand_half_width,
            excluded_bodies=self.colliding_bodies_below(self.food.root),
        )

    def hand_path_is_free(
        self,
        grasp: GraspCandidate,
        picking: GraspPoseSequence,
        reach_space: VolumetricGraphOfBoundingBoxes,
    ) -> bool:
        """
        :param grasp: The grasp the food is taken by.
        :param picking: The tool frame goals of picking the food up by ``grasp``.
        :param reach_space: The :meth:`reach_space` around the food.
        :return: Whether the hand comes in along a straight line through
            ``reach_space`` from where its approach starts to :attr:`hand_half_width`
            short of the food, and the space from that far above the food's top up
            to :attr:`lift` above it is free; the last stretch to the food, where
            the fingers may touch what it rests on, is not checked.
        """
        extent = self.extent(self.food.root)
        center = extent.center.to_np()[:3]
        approach = grasp.world_T_grasp.to_np()[:3, 0]
        half_sizes = np.array([extent.width, extent.depth, extent.height]) / 2
        to_the_surface = min(
            half_size / abs(component)
            for half_size, component in zip(half_sizes, approach)
            if abs(component) > 0.0
        )
        up = np.array([0.0, 0.0, 1.0])
        top = center + half_sizes[2] * up
        segments = [
            (
                self.in_kitchen_frame([picking.pre_grasp])[0][:3, 3],
                center - (to_the_surface + self.hand_half_width) * approach,
            ),
            (top + self.hand_half_width * up, top + self.lift * up),
        ]
        return all(
            self._is_straight_through(reach_space, start, end)
            for start, end in segments
        )

    def _is_straight_through(
        self,
        space: VolumetricGraphOfBoundingBoxes,
        start: np.ndarray,
        end: np.ndarray,
    ) -> bool:
        """
        :return: Whether the straight line from ``start`` to ``end``, in the kitchen's
            frame, lies in ``space``.
        """
        start_point = Point3(*start, reference_frame=self.world.root)
        end_point = Point3(*end, reference_frame=self.world.root)
        return (
            space.is_free(start_point)
            and space.is_free(end_point)
            and len(space.path_from_to(start_point, end_point)) == 2
        )

    @property
    def release_point(self) -> np.ndarray:
        """
        Where the centre of the food is to be let go: its underside
        :attr:`release_clearance` above :attr:`release_height`, over the part of the
        receptacle's floor nearest the HSR but at least :attr:`floor_margin` in from
        its edge, and over a strip of the floor of its own: the floor is cut across its
        width into one strip per object of :attr:`moved_together`, in their order.
        """
        floor = self.receptacle_floor
        food_extent = self.extent(self.food.root)
        floor_center = floor.center.to_np()[:3]
        strip_width = floor.width / len(self.moved_together)
        strip = self.moved_together.index(self.moving)
        return np.array(
            [
                floor.min_x + strip_width * (strip + 0.5),
                min(
                    floor_center[1],
                    floor.min_y + food_extent.width / 2 + self.floor_margin,
                ),
                self.release_height + food_extent.height / 2 + self.release_clearance,
            ]
        )

    @property
    def release_height(self) -> float:
        """
        The height above which the food is let go: the top of the receptacle's floor.
        """
        return self.receptacle_floor.max_z

    def goal(self) -> Pose:
        """
        :return: Where the food is to be put: its centre at :attr:`release_point`,
            turned as it is now.
        """
        food = self.food.root
        world_T_food = food.global_pose.to_np()
        world_T_food[:3, 3] += self.release_point - self.extent(food).center.to_np()[:3]
        return HomogeneousTransformationMatrix(
            world_T_food, reference_frame=self.world.root
        ).pose

    def build_plan(self, context: Context) -> Plan:
        """
        :return: The plan :meth:`plan_if_approachable` builds.

        :raises NoApproachError: If the food has no approach.
        """
        plan = self.plan_if_approachable(context)
        if plan is None:
            raise NoApproachError(self.food.root)
        return plan

    def plan_if_approachable(self, context: Context) -> Plan | None:
        """
        :param context: The context the actions are built in.
        :return: The plan that drives to where the arm reaches the food, moves the arm
            to where the pick up starts its approach, picks the food up, drives holding
            it to where the arm reaches the receptacle, and puts it in; the food taken
            by the first of the :meth:`approaches` whose hand path is free, see
            :meth:`hand_path_is_free`, and that the HSR has somewhere to stand for,
            both picking and placing; ``None`` if no approach is left.
        """
        robot = context.robot
        arm = robot.all_arms[0]
        approaches = self.approaches()
        reach_space = self.reach_space(
            PickUpAction(
                self.grasp(approaches[0]), arm, retreat_distance=self.lift
            ).approach_clearance
        )
        for approach in approaches:
            grasp = self.grasp(approach)
            pick_up = PickUpAction(grasp, arm, retreat_distance=self.lift)
            picking = pick_up.grasp_pose_sequence(
                grasp.world_T_grasp, arm.end_effector, grasp
            )
            if not self.hand_path_is_free(grasp, picking, reach_space):
                continue
            picking_pose = self.standing_pose(
                self.in_kitchen_frame(
                    [picking.pre_grasp, picking.grasp, picking.retreat]
                ),
                robot.root.global_pose,
            )
            if picking_pose is None:
                continue
            place = PlaceAction(self.food, self.goal())
            placing = place.grasp_pose_sequence(
                grasp.moved_to(place.target_location), arm.end_effector, grasp
            )
            placing_pose = self.standing_pose(
                self.in_kitchen_frame(
                    [placing.grasp, placing.retreat, placing.pre_grasp]
                ),
                picking_pose,
            )
            if placing_pose is not None:
                return self._picking_and_placing_plan(
                    context, picking, picking_pose, pick_up, placing_pose, place
                )
        return None

    def _picking_and_placing_plan(
        self,
        context: Context,
        picking: GraspPoseSequence,
        picking_pose: Pose,
        pick_up: PickUpAction,
        placing_pose: Pose,
        place: PlaceAction,
    ) -> Plan:
        """
        :return: The plan that drives to ``picking_pose``, moves the arm to where
            ``picking`` starts, carries out ``pick_up``, drives to ``placing_pose``
            and carries out ``place``, the arms parked before and after.
        """
        robot = context.robot
        return sequential(
            [
                ParkArmsAction(robot.all_arms),
                PathPlanningNavigateAction(picking_pose),
                *self.arm_motions_to(picking.pre_grasp, picking_pose),
                pick_up,
                PathPlanningNavigateAction(placing_pose),
                place,
                ParkArmsAction(robot.all_arms),
            ],
            context=context,
        ).plan

    def prepare_contacts(self) -> None:
        """
        Have the contacts of the objects of :attr:`moved_objects` resolve the twisting
        and rolling friction RoboCasa gives them, so that they neither turn in the hand
        nor roll on without end.
        """
        for moved in self.moved_objects:
            for shape in self.world.get_body_by_name(moved).collision.shapes:
                shape.get_simulator_property_of_type(
                    MujocoGeom
                ).contact_dimensionality = (
                    ContactDimensionality.SLIDING_TWISTING_AND_ROLLING
                )


# %% the tasks


@dataclass
class PickPlaceCounterToStove(HSRPickPlaceTask):
    """
    Pick the food from the plate on the counter and place it in the pan on the stove.
    """

    seed: int = 2
    """
    The seed RoboCasa samples the task with: the plate on the counter left of the
    stove, near its edge.
    """

    food_category: str | None = "cheese"
    """
    A block of cheese, which lies still on its plate.
    """


@dataclass
class PickPlaceStoveToCounter(HSRPickPlaceTask):
    """
    Pick the food from the pan on the stove and place it on the plate or in the bowl
    on the counter.
    """

    seed: int = 12
    """
    The seed RoboCasa samples the task with: the pan on the front left burner.
    """

    food_category: str | None = "lemon"
    """
    A lemon, small enough for the HSR's hand.
    """


@dataclass
class PickPlaceCounterToSink(HSRPickPlaceTask):
    """
    Pick the object from the counter and place it in the sink.
    """

    seed: int = 0
    """
    The seed RoboCasa samples the task with: the object near the counter's edge right
    of the sink.
    """

    food_category: str | None = "canned_food"
    """
    A can, standing upright.
    """

    @property
    def receptacle(self) -> Body:
        return self.world.get_semantic_annotations_by_type(Sink)[0].root

    @property
    def release_height(self) -> float:
        """
        The sink's rim, level with the counter: the object is dropped into the sink
        from just above it.
        """
        return self.counter_top_height


@dataclass
class PickPlaceSinkToCounter(HSRPickPlaceTask):
    """
    Pick the food from the sink and place it on the plate on the counter.
    """

    seed: int = 4
    """
    The seed RoboCasa samples the task with: a flat plate on the counter right of the
    sink.
    """

    food_category: str | None = "yogurt"
    """
    A pot of yogurt.
    """

    lift: float = 0.2
    """
    Out of the sink and above the counter's top: carried lower, the food's underside
    stays below the top and catches on the counter's edge beside the sink.
    """

    tool_frame_tolerance: float = 0.02
    """
    Reaching down into the sink, the arm comes within its collision buffer of the
    counter's edge and the sink's rim 1.6 cm before the grasp, with the food already
    between the fingers.
    """


@dataclass
class PickPlaceDrawerToCounter(HSRPickPlaceTask, HSRTaskAtDrawer):
    """
    Pick the object from the open drawer and place it on the counter.
    """

    seed: int = 9
    """
    The seed RoboCasa samples the task with: a rolling pin lying across the drawer
    diagonally.
    """

    lift: float = 0.2
    """
    Above the drawer's front and the counter's top: carried lower, the long object
    swings into the drawer's front as the HSR drives off.
    """

    tool_frame_tolerance: float = 0.025
    """
    Rolled to close across the object, the wrist moves the tool frame up to 2 cm
    sideways out of the plane the arm swings in, which the arm cannot make up for.
    """

    @property
    def stages(self) -> list[Callable[[Context], Plan]]:
        """
        Pull the drawer fully open, which brings the object out from under the
        counter's top and the drawer's front further from it, then take the object
        from where it came to rest and put it on the counter.
        """
        return [self.build_opening_plan, self.build_plan]

    def build_opening_plan(self, context: Context) -> Plan:
        """
        :param context: The context the actions are built in.
        :return: The plan that pulls the drawer fully open.
        """
        return self.handle_plan(context, OpenAction)

    @property
    def counter(self) -> Body:
        """
        The counter above the drawer.
        """
        return self.task.referenced_fixtures[FixtureReference.COUNTER]

    def top_pieces(self, counter: Body) -> list[VolumetricBoundingBox]:
        """
        :param counter: A counter of the kitchen.
        :return: The pieces of the counter's top, in the kitchen's frame.
        """
        return [
            piece
            for piece in counter.collision.as_bounding_box_collection_in_frame(
                self.world.root
            ).bounding_boxes
            if math.isclose(piece.max_z, self.counter_top_height, abs_tol=0.01)
        ]

    @property
    def counters(self) -> list[Body]:
        """
        Every counter of the kitchen, any of which the object may be placed on.
        """
        return [
            annotation.root
            for annotation in self.world.get_semantic_annotations_by_type(CounterTop)
        ]

    @property
    def long_axis(self) -> np.ndarray:
        """
        The horizontal direction the object is longest along, pointing towards the
        counter.
        """
        food = self.food.root
        extent = food.collision.as_bounding_box_collection_in_frame(food).bounding_box()
        rotation = food.global_pose.to_np()[:3, :3]
        axis = rotation[:2, 0] if extent.depth >= extent.width else rotation[:2, 1]
        axis = axis / np.linalg.norm(axis)
        return axis if axis[1] >= 0.0 else -axis

    def grasp(self, approach: Approach) -> GraspCandidate:
        """
        :return: The grasp that takes the object at :attr:`grasp_point`, coming in as
            ``approach`` says, the fingers closing across the object's long axis; as
            :meth:`HSRPickPlaceTask.grasp` where the approach runs along that axis,
            since closing across the heading then closes across it too.
        """
        direction = np.array(
            [
                math.cos(approach.heading) * math.cos(approach.tilt),
                math.sin(approach.heading) * math.cos(approach.tilt),
                -math.sin(approach.tilt),
            ]
        )
        closing = np.cross(direction, [*self.long_axis, 0.0])
        if np.isclose(np.linalg.norm(closing), 0.0):
            return super().grasp(approach)
        closing /= np.linalg.norm(closing)
        world_T_grasp = np.eye(4)
        world_T_grasp[:3, 0] = direction
        world_T_grasp[:3, 1] = closing
        world_T_grasp[:3, 2] = np.cross(direction, closing)
        world_T_grasp[:3, 3] = self.grasp_point
        food_T_grasp = self.world.transform(
            HomogeneousTransformationMatrix(
                world_T_grasp, reference_frame=self.world.root
            ).pose,
            self.food.root,
        )
        return GraspCandidate(self.food, food_T_grasp)

    @property
    def receptacle(self) -> Body:
        """
        The counter :attr:`receptacle_floor` is a piece of.
        """
        floor = self.receptacle_floor
        return [
            counter for counter in self.counters if floor in self.top_pieces(counter)
        ][0]

    @property
    def receptacle_floor(self) -> VolumetricBoundingBox:
        """
        The widest piece of a counter's top that does not stand above the open
        drawer, which keeps the HSR from reaching across it; of equally wide ones, the
        one nearest the drawer.
        """
        drawer = self.extent(self.drawer.root)
        beside_the_drawer = [
            piece
            for counter in self.counters
            for piece in self.top_pieces(counter)
            if piece.max_x <= drawer.min_x or piece.min_x >= drawer.max_x
        ]
        return max(
            beside_the_drawer,
            key=lambda piece: (
                round(piece.width * piece.depth, 3),
                -abs(piece.center.to_np()[0] - drawer.center.to_np()[0]),
            ),
        )

    @property
    def bodies_touching_handled_bodies(self) -> list[Body]:
        """
        The wrist's links and the forearm, which the drawer's front comes towards as
        the drawer opens.
        """
        return self.arm_links(list(ArmLink))

    @property
    def handled_bodies(self) -> list[Body]:
        """
        The object, the counter it is placed on, and the drawer it is taken out of
        with the counter above it.
        """
        return (
            super().handled_bodies
            + self.colliding_bodies_below(self.drawer.root)
            + self.colliding_bodies_below(self.counter)
        )


if __name__ == "__main__":
    experiment = PickPlaceCounterToStove()
    experiment.build_world()
    experiment.run()
