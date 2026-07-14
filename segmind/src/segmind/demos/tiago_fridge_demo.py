"""
Demonstration of a coraplex fetch-and-store plan on the TIAGo++ robot.

The robot opens the fridge door, fetches the milk box from the table, places
it on the lower fridge shelf, and closes the fridge door again.  All motions
are compiled into giskardpy motion statecharts and mirrored into a MuJoCo
viewer.

Run with::

    python -m segmind.demos.tiago_fridge_demo
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy
from typing_extensions import Optional

from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.factories import reachability_location
from coraplex.motion_executor import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.failures import BodyUnfetchable
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from coraplex.view_manager import ViewManager
from krrood.exceptions import DataclassException
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from segmind.demos.tiago_apartment_scene import (
    ApartmentSimulation,
    _add_scene_objects,
    build_context,
    build_tiago_world,
)

MILK_BODY_NAME = "milk_box"

FRIDGE_HANDLE_BODY_NAME = "fridge_door1_handle"

FRIDGE_BASE_BODY_NAME = "fridge_base"

FRIDGE_SHELF_BODY_NAME = "fridge_shelf2"

PLACEMENT_CLEARANCE = 0.02
"""
Vertical clearance in meters between the object bottom and the shelf when placing.
"""

PLACEMENT_DEPTH_OFFSET = 0.08
"""
Distance in meters from the fridge front face into the interior at which the
object is placed, keeping it on the shelf but close to the opening so the
arm does not have to reach deep into the fridge.
"""

FRIDGE_APPROACH_YAW = -2.7
"""
Orientation of the place pose in radians: the gripper approaches the shelf
diagonally from positive y, staying clear of the fridge door which swings
open toward negative y.
"""

PLACE_STANDING_DISTANCE = 0.9
"""
Sampling-ring radius in meters for base poses when placing into the fridge.
Larger than the default reach distance because the robot must stand clear
of the swung-open fridge door.
"""


@dataclass
class StoreObjectInContainerAction(ActionDescription):
    """
    Fetch an object and store it inside a container that is closed by a door.

    The container is opened first, then the object is picked up, placed at the
    target location inside the container, and the container is closed again.
    Navigation poses are grounded lazily right before each stage so they
    account for the current world state (for example the swung-open door).
    """

    object_designator: Body
    """
    The object that should be stored.
    """

    container_handle: Body
    """
    Handle of the container door.
    """

    target_location: Pose
    """
    Pose inside the container at which the object should be placed.
    """

    arm: Arms
    """
    Arm used for fetching and placing the object.
    """

    door_arm: Arms
    """
    Arm used for opening and closing the container door.
    """

    grasp_description: Optional[GraspDescription] = None
    """
    Grasp used for picking the object up; defaults to a front grasp.
    """

    def execute(self) -> None:
        self.grasp_description = self.grasp_description or GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            ViewManager.get_end_effector_view(self.arm, self.robot),
        )
        self.add_subplan(
            sequential(
                [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)]
            )
        ).perform()
        self._operate_container(OpenAction(self.container_handle, self.door_arm))
        self._fetch_object()
        self._store_object()
        self._operate_container(CloseAction(self.container_handle, self.door_arm))

    def _navigation_pose_near_handle(self) -> Pose:
        """
        Ground a base pose from which the container handle is reachable.

        :return: A validated navigation pose near the handle.
        """
        handle_pose = reachability_location(
            self.container_handle.global_pose, self.context, self.door_arm
        ).ground()
        if not handle_pose:
            raise BodyUnfetchable(self.container_handle, self.door_arm)
        return handle_pose

    def _operate_container(self, door_action: OpenAction | CloseAction) -> None:
        """
        Navigate to the container handle and run ``door_action`` on it.

        :param door_action: The door action to perform at the handle.
        """
        self.add_subplan(
            sequential(
                [
                    NavigateAction(self._navigation_pose_near_handle(), True),
                    door_action,
                    ParkArmsAction(Arms.BOTH),
                ]
            )
        ).perform()

    def _fetch_object(self) -> None:
        """
        Navigate to the object and pick it up.
        """
        pickup_pose = reachability_location(
            self.object_designator, self.context, self.arm, self.grasp_description
        ).ground()
        if not pickup_pose:
            raise BodyUnfetchable(self.object_designator, self.arm)
        self.add_subplan(
            sequential(
                [
                    NavigateAction(pickup_pose, True),
                    PickUpAction(
                        self.object_designator, self.arm, self.grasp_description
                    ),
                    ParkArmsAction(Arms.BOTH),
                    MoveTorsoAction(TorsoState.HIGH),
                ]
            )
        ).perform()

    def _store_object(self) -> None:
        """
        Navigate to the container and place the object at the target location.
        """
        place_pose = reachability_location(
            self.target_location,
            self.context,
            self.arm,
            self.grasp_description,
            mean_distance_to_target=PLACE_STANDING_DISTANCE,
        ).ground()
        if not place_pose:
            raise BodyUnfetchable(self.object_designator, self.arm)
        self.add_subplan(
            sequential(
                [
                    NavigateAction(place_pose, True),
                    PlaceAction(
                        self.object_designator, self.target_location, self.arm
                    ),
                    ParkArmsAction(Arms.BOTH),
                ]
            )
        ).perform()


def milk_place_pose(world: World) -> Pose:
    """
    Compute the pose on the lower fridge shelf at which the milk is placed.

    The support surface is found by casting a ray downward inside the fridge
    from the shelf marker height, because the shelf marker bodies carry no
    geometry and sit above the actual shelf plate.  The pose sits above that
    surface by the milk's origin-to-bottom height plus a small clearance and
    is oriented for a diagonal approach clear of the open door.

    :param world: The world containing the fridge and the milk.
    :return: The placement pose for the milk box.
    """
    milk = world.get_body_by_name(MILK_BODY_NAME)
    fridge_boxes = (
        world.get_body_by_name(FRIDGE_BASE_BODY_NAME)
        .collision.as_bounding_box_collection_in_frame(world.root)
        .bounding_boxes
    )
    fridge_front_x = max(box.max_x for box in fridge_boxes)
    fridge_center_y = (
        min(box.min_y for box in fridge_boxes)
        + max(box.max_y for box in fridge_boxes)
    ) / 2
    place_x = fridge_front_x - PLACEMENT_DEPTH_OFFSET
    place_y = fridge_center_y

    shelf_marker_z = float(
        world.get_body_by_name(FRIDGE_SHELF_BODY_NAME).global_pose.to_position().z
    )
    surface_z = _support_surface_height(world, place_x, place_y, shelf_marker_z)

    milk_bottom_z = min(
        box.min_z
        for box in milk.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_boxes
    )
    origin_above_bottom = float(milk.global_pose.to_position().z) - milk_bottom_z
    return Pose.from_xyz_rpy(
        x=place_x,
        y=place_y,
        z=surface_z + origin_above_bottom + PLACEMENT_CLEARANCE,
        yaw=FRIDGE_APPROACH_YAW,
        reference_frame=world.root,
    )


def _support_surface_height(
    world: World, x: float, y: float, start_z: float
) -> float:
    """
    Find the height of the first support surface below ``start_z`` at (x, y).

    :param world: The world to ray-test in.
    :param x: The x-coordinate of the probe in the world frame.
    :param y: The y-coordinate of the probe in the world frame.
    :param start_z: The height to cast the ray downward from.
    :return: The z-coordinate of the surface hit by the ray.
    """
    ray_tracer = RayTracer(world)
    hit_points, hit_ray_indices, _ = ray_tracer.ray_test(
        numpy.array([[x, y, start_z]]),
        numpy.array([[x, y, start_z - 1.0]]),
        max_distance=1.0,
    )
    if len(hit_ray_indices) == 0:
        raise SupportSurfaceNotFound(x=x, y=y, start_z=start_z)
    return float(hit_points[0][2])


@dataclass
class SupportSurfaceNotFound(DataclassException):
    """
    Raised when no support surface exists below a placement probe point.
    """

    x: float
    """
    The x-coordinate of the probe in the world frame.
    """

    y: float
    """
    The y-coordinate of the probe in the world frame.
    """

    start_z: float
    """
    The height the probe ray was cast downward from.
    """

    def error_message(self) -> str:
        return (
            f"No support surface below ({self.x}, {self.y}, {self.start_z})."
        )

    def suggest_correction(self) -> str:
        return "Probe above a body with collision geometry."


if __name__ == "__main__":
    _world = build_tiago_world()
    _add_scene_objects(_world)

    _simulation = ApartmentSimulation(world=_world)
    _simulation.start()

    _context = build_context(_world)

    try:
        with simulated_robot:
            sequential(
                [
                    StoreObjectInContainerAction(
                        object_designator=_world.get_body_by_name(MILK_BODY_NAME),
                        container_handle=_world.get_body_by_name(
                            FRIDGE_HANDLE_BODY_NAME
                        ),
                        target_location=milk_place_pose(_world),
                        arm=Arms.RIGHT,
                        door_arm=Arms.LEFT,
                    )
                ],
                _context,
            ).perform()
        _simulation.freeze_and_wait()
    finally:
        _simulation.stop()
