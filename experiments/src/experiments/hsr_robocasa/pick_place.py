"""
RoboCasa's pick-and-place tasks, performed by the HSR: it finds an approach to the
object that is free and leaves it somewhere to stand, drives there along the floor's
graph of convex sets, picks the object up, drives to where it reaches the receptacle,
and sets the object down there.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from coraplex.robot_plans.actions.core.navigation import PathPlanningNavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from coraplex.robot_plans.mixins import GraspPoseSequence
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.spatial_types import Point2, Point3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import (
    BoundingBoxCollection,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    VolumetricGraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.world_entity import Body

from experiments.hsr_robocasa.exceptions import NoApproachError
from experiments.hsr_robocasa.robocasa_task import HSRRoboCasaTask, StanceSearch

logger = logging.getLogger(__name__)

# %% the task's objects


class TaskObject(StrEnum):
    """
    The bodies RoboCasa adds for a pick-and-place task, by the names it gives them.
    """

    OBJECT = "obj_main"
    """
    The object to pick up and set down.
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
class PickingAndPlacing:
    """
    How the HSR picks the object up and sets it down: the actions, and where it stands
    for each.
    """

    pick_up: PickUpAction
    """
    The pick-up.
    """

    picking_pose: Pose
    """
    Where the HSR stands to pick the object up.
    """

    place: PlaceAction
    """
    The placing.
    """

    placing_pose: Pose
    """
    Where the HSR stands to set the object down.
    """


# %% a pick-and-place task


@dataclass
class HSRPickPlaceTask(HSRRoboCasaTask, ABC):
    """
    One of RoboCasa's pick-and-place tasks, performed by the HSR.
    """

    lift: float = 0.1
    """
    How far the HSR lifts the object from where it took hold of it, in metres, before
    carrying it to the receptacle.
    """

    release_clearance: float = 0.03
    """
    How far above the receptacle's floor the HSR lets the object go, in metres.
    """

    floor_margin: float = 0.05
    """
    How far in from the edge of the receptacle's floor the object is set down, in
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

    @property
    @abstractmethod
    def receptacle(self) -> Body:
        """
        The root of what the object is to be set into.
        """

    @property
    def moved_object(self) -> Body:
        """
        The object to pick up and set down.
        """
        return next(
            body
            for body in self.task.manipulated_objects
            if body.name.name == TaskObject.OBJECT
        )

    @property
    def handled_bodies(self) -> list[Body]:
        """
        The object and the receptacle.
        """
        return [
            body
            for root in [self.moved_object, self.receptacle]
            for body in self.colliding_bodies_below(root)
        ]

    def graspable(self, world: World) -> HasGraspCandidates:
        """
        :return: The annotation of the object to move.
        """
        return next(
            annotation
            for annotation in world.get_semantic_annotations_by_type(HasGraspCandidates)
            if annotation.root is self.moved_object
        )

    def grasp_point(self) -> np.ndarray:
        """
        :return: Where the object is taken hold of: its middle.
        """
        return self.extent(self.moved_object).center.to_np()[:3]

    # %% the approach to the object

    def approaches(self, world: World) -> list[Approach]:
        """
        :return: The approaches the search tries, in order: the tilts from level down,
            and at each tilt the headings nearest the bearing to the object from the
            floor nearest it that the HSR's base can stand on first.
        """
        navigable_space = self.floor_in(world).navigable_space(
            self.robot_in(world), self.base_collision_rule(world).violated_distance
        )
        grasp_point = self.grasp_point()
        nearest_floor = world.transform(
            navigable_space.nearest_free_point(
                Point2(*grasp_point[:2], reference_frame=world.root)
            ),
            world.root,
        )
        to_object = grasp_point[:2] - nearest_floor.to_np()[:2]
        bearing = math.atan2(to_object[1], to_object[0])
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

    def grasp(self, world: World, approach: Approach) -> GraspCandidate:
        """
        :return: The grasp that takes the object at its :meth:`grasp_point`, coming in
            as ``approach`` says, the fingers closing across its heading.
        """
        world_T_grasp = HomogeneousTransformationMatrix.from_xyz_rpy(
            *self.grasp_point(),
            pitch=approach.tilt,
            yaw=approach.heading,
            reference_frame=world.root,
        )
        return GraspCandidate(
            self.graspable(world),
            world.transform(world_T_grasp.pose, self.moved_object),
        )

    def hand_half_width(self, world: World) -> float:
        """
        :return: How far, at most, the hand reaches out to a side of its approach, in
            metres, its fingers as they stand.
        """
        end_effector = self.robot_in(world).all_arms[0].end_effector
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

    def reach_space(
        self, world: World, approach_clearance: float, hand_half_width: float
    ) -> VolumetricGraphOfBoundingBoxes:
        """
        :return: The free space around the object that the middle of the hand can pass
            through: everything but the object kept ``hand_half_width`` clear of, as
            far around the object as an approach starts and up to :attr:`lift` above
            it.
        """
        extent = self.extent(self.moved_object)
        center = extent.center.to_np()[:3]
        around = (
            np.linalg.norm([extent.width, extent.depth, extent.height]) / 2
            + approach_clearance
            + hand_half_width
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
                    origin=HomogeneousTransformationMatrix(reference_frame=world.root),
                )
            ],
            world.root,
        )
        return VolumetricGraphOfBoundingBoxes.free_space_from_world(
            world,
            search_space,
            bloat_obstacles=hand_half_width,
            obstacle_height_clearance=hand_half_width,
            excluded_bodies=self.colliding_bodies_below(self.moved_object),
        )

    def hand_path_is_free(
        self,
        world: World,
        grasp: GraspCandidate,
        picking: GraspPoseSequence,
        reach_space: VolumetricGraphOfBoundingBoxes,
        hand_half_width: float,
    ) -> bool:
        """
        :return: Whether the hand comes in along a straight line through
            ``reach_space`` from where its approach starts to ``hand_half_width`` short
            of the object, and the space from that far above the object's top up to
            :attr:`lift` above it is free; the last stretch to the object, where the
            fingers may touch what it rests on, is not checked.
        """
        extent = self.extent(self.moved_object)
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
                self.in_kitchen_frame(world, [picking.pre_grasp])[0][:3, 3],
                center - (to_the_surface + hand_half_width) * approach,
            ),
            (top + hand_half_width * up, top + self.lift * up),
        ]
        return all(
            self._is_straight_through(world, reach_space, start, end)
            for start, end in segments
        )

    @staticmethod
    def _is_straight_through(
        world: World,
        space: VolumetricGraphOfBoundingBoxes,
        start: np.ndarray,
        end: np.ndarray,
    ) -> bool:
        """
        :return: Whether the straight line from ``start`` to ``end``, in the kitchen's
            frame, lies in ``space``.
        """
        start_point = Point3(*start, reference_frame=world.root)
        end_point = Point3(*end, reference_frame=world.root)
        return (
            space.is_free(start_point)
            and space.is_free(end_point)
            and len(space.path_from_to(start_point, end_point)) == 2
        )

    # %% where the object goes

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
                body._world.root
            ).bounding_boxes
        ]
        return max(pieces, key=lambda piece: piece.width * piece.depth)

    @property
    def release_height(self) -> float:
        """
        The height above which the object is let go: the top of the receptacle's floor.
        """
        return self.receptacle_floor.max_z

    def release_point(self) -> np.ndarray:
        """
        :return: Where the centre of the object is to be let go: its underside
            :attr:`release_clearance` above :attr:`release_height`, over the middle of
            the receptacle's floor across, and over the part of it nearest the HSR but
            at least :attr:`floor_margin` in from its edge.
        """
        floor = self.receptacle_floor
        object_extent = self.extent(self.moved_object)
        floor_center = floor.center.to_np()[:3]
        return np.array(
            [
                floor_center[0],
                min(
                    floor_center[1],
                    floor.min_y + object_extent.width / 2 + self.floor_margin,
                ),
                self.release_height + object_extent.height / 2 + self.release_clearance,
            ]
        )

    def goal(self, world: World) -> Pose:
        """
        :return: Where the object is to be put: its centre at :meth:`release_point`,
            turned as it is now.
        """
        world_T_object = self.moved_object.global_pose.to_np()
        world_T_object[:3, 3] += (
            self.release_point() - self.extent(self.moved_object).center.to_np()[:3]
        )
        return HomogeneousTransformationMatrix(
            world_T_object, reference_frame=world.root
        ).pose

    # %% the plan

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Raise the torso, park the arm, drive to where the arm reaches the object, pick
        it up, drive to where the arm reaches the receptacle, set the object down, and
        park the arm again.
        """
        world = context.world
        robot = self.robot_in(world)
        steps = self.picking_and_placing(world)
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    MoveTorsoAction(torso_state=TorsoState.HIGH),
                    ParkArmsAction(arms=robot.all_arms),
                    PathPlanningNavigateAction(target=steps.picking_pose),
                    steps.pick_up,
                    PathPlanningNavigateAction(target=steps.placing_pose),
                    steps.place,
                    ParkArmsAction(arms=robot.all_arms),
                ]
            )
        )
        return statechart

    def picking_and_placing(self, world: World) -> PickingAndPlacing:
        """
        :return: The object taken by the first of the :meth:`approaches` whose hand
            path is free, see :meth:`hand_path_is_free`, and that the HSR has somewhere
            to stand for picking, set down from where the HSR reaches the receptacle,
            or, where it stands nowhere to reach it, tried from the floor nearest it.
        :raises NoApproachError: If no approach is left.
        """
        robot = self.robot_in(world)
        arm = robot.all_arms[0]
        approaches = self.approaches(world)
        hand_half_width = self.hand_half_width(world)
        reach_space = self.reach_space(
            world,
            PickUpAction(
                grasp=self.grasp(world, approaches[0]),
                arm=arm,
                retreat_distance=self.lift,
            ).approach_clearance,
            hand_half_width,
        )
        search = self.stance_search(world)
        for approach in approaches:
            grasp = self.grasp(world, approach)
            pick_up = PickUpAction(grasp=grasp, arm=arm, retreat_distance=self.lift)
            picking = pick_up.grasp_pose_sequence(
                grasp.world_T_grasp, arm.end_effector, grasp
            )
            if not self.hand_path_is_free(
                world, grasp, picking, reach_space, hand_half_width
            ):
                continue
            picking_pose = self.standing_pose(
                world,
                self.in_kitchen_frame(
                    world, [picking.pre_grasp, picking.grasp, picking.retreat]
                ),
                robot.root.global_pose,
                search,
            )
            if picking_pose is None:
                continue
            place = PlaceAction(
                object_designator=self.graspable(world),
                target_location=self.goal(world),
            )
            placing = place.grasp_pose_sequence(
                grasp.moved_to(place.target_location), arm.end_effector, grasp
            )
            placing_goals = self.in_kitchen_frame(
                world, [placing.grasp, placing.retreat, placing.pre_grasp]
            )
            placing_pose = self.standing_pose(
                world, placing_goals, picking_pose, search
            ) or self.attempting_pose(world, placing_goals[0], search)
            return PickingAndPlacing(
                pick_up=pick_up,
                picking_pose=picking_pose,
                place=place,
                placing_pose=placing_pose,
            )
        raise NoApproachError(self.moved_object)

    def attempting_pose(
        self, world: World, tool_frame_goal: np.ndarray, search: StanceSearch
    ) -> Pose:
        """
        :param tool_frame_goal: Where the tool frame is to be, in the kitchen's frame.
        :return: Where the HSR stands to try a goal it has nowhere to stand to reach
            from: on the free floor nearest the goal, facing it.
        """
        logger.warning(
            "The HSR has nowhere to stand to reach %s; it tries from the floor "
            "nearest it.",
            np.round(tool_frame_goal[:3, 3], 3).tolist(),
        )
        nearest = world.transform(
            search.navigable_space.nearest_free_point(
                Point2(*tool_frame_goal[:2, 3], reference_frame=world.root)
            ),
            world.root,
        )
        to_goal = tool_frame_goal[:2, 3] - nearest.to_np()[:2]
        return Pose.from_xyz_rpy(
            x=float(nearest.x),
            y=float(nearest.y),
            yaw=math.atan2(to_goal[1], to_goal[0]),
            reference_frame=world.root,
        )
