"""
The HSR performing RoboCasa's pick-and-place tasks in a RoboCasa kitchen lowered to
its height, simulated in MuJoCo with the object held by friction alone.
"""

import math
import os
from enum import StrEnum

import numpy as np
import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from experiments.hsr_robocasa.pick_place import (
    Approach,
    HSRPickPlaceTask,
    PickPlaceCounterToSink,
    PickPlaceCounterToStove,
    PickPlaceDrawerToCounter,
    PickPlaceSinkToCounter,
    PickPlaceStoveToCounter,
    TaskObject,
)
from experiments.hsr_robocasa.robocasa_task import (
    FixtureType,
    HSRBArmPlane,
    LayoutKey,
    RoboCasaLayout,
    RowOffTheFloor,
)
from robocasa.models.scenes.scene_registry import LayoutType
from segmind.datastructures.events import LossOfGraspEvent, PlacingEvent
from semantic_digital_twin.robots.hsrb import HSRBJoint
from semantic_digital_twin.spatial_types import Point2


class HSRLimitation(StrEnum):
    """
    Why a task fails for the HSR in RoboCasa's kitchen as built, with the HSR's
    collision margins as configured.
    """

    PAN_OUT_OF_REACH = (
        "No place on the floor that keeps the base outside its collision buffer zone "
        "lets the arm reach the pan on the stove."
    )
    """
    The stove's top lies too deep beyond its front.
    """

    ARM_MARGIN = (
        "Reaching the object, the arm comes within its collision buffer zone of the "
        "kitchen and collision avoidance stops it."
    )
    """
    At RoboCasa's counter height, the forearm comes down to the counter's edge.
    """

    OBJECTS_TOO_CLOSE = (
        "Objects stand closer together than the hand's width, so no approach to one "
        "keeps the hand clear of its neighbour."
    )
    """
    RoboCasa sets the objects down nearly touching.
    """

    OVEN_BACK_OFF = (
        "The door opens fully, but backing the hand off the released bar stalls."
    )
    """
    The base follows the door out, and the arm cannot draw the hand back from there.
    """


runs_in_ci = pytest.mark.skipif(
    os.environ.get("CI", "false").lower() == "false",
    reason="MuJoCo tests only run in CI",
)

REACHED_POSE_TOLERANCE = 0.005
"""
How far, in metres, the tool frame may end up from a pose the arm plane is asked to
reach: the resolution of its searches.
"""


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            PickPlaceCounterToStove,
            marks=pytest.mark.xfail(strict=True, reason=HSRLimitation.PAN_OUT_OF_REACH),
        ),
        pytest.param(
            PickPlaceStoveToCounter,
            marks=pytest.mark.xfail(strict=True, reason=HSRLimitation.PAN_OUT_OF_REACH),
        ),
        PickPlaceCounterToSink,
        pytest.param(
            PickPlaceSinkToCounter,
            marks=pytest.mark.xfail(strict=True, reason=HSRLimitation.ARM_MARGIN),
        ),
        pytest.param(
            PickPlaceDrawerToCounter,
            marks=pytest.mark.xfail(strict=True, reason=HSRLimitation.ARM_MARGIN),
        ),
    ],
)
def performed_task(request) -> HSRPickPlaceTask:
    """
    A pick-and-place task, performed in MuJoCo.
    """
    experiment = request.param(headless=True)
    experiment.build_world()
    experiment.run()
    return experiment


# %% the lowered kitchen


def test_the_counters_stand_at_the_height_the_kitchen_is_lowered_to():
    experiment = PickPlaceCounterToStove(counter_height=0.75)
    layout = RoboCasaLayout.from_layout_type(experiment.layout)

    lowered = layout.lowered_fixtures(experiment.counter_height)

    counters = [
        fixture[LayoutKey.NAME]
        for group in layout.configuration.values()
        for row in group.values()
        if isinstance(row, list)
        for fixture in row
        if fixture.get(LayoutKey.TYPE) == FixtureType.COUNTER
    ]
    assert [lowered[counter][LayoutKey.SIZE][2] for counter in counters] == (
        pytest.approx([experiment.counter_height] * len(counters))
    )


def test_a_kitchen_island_is_lowered_with_the_counters():
    """
    A layout with a group of its own, such as an island, also states where the group
    stands; that is no fixture to lower, while the island's own fixtures are.
    """
    layout = RoboCasaLayout.from_layout_type(LayoutType.LAYOUT002)

    lowered = layout.lowered_fixtures(0.75)

    island = [
        fixture[LayoutKey.NAME]
        for row in layout.configuration[LayoutKey.ISLAND_GROUP].values()
        if isinstance(row, list)
        for fixture in row
        if isinstance(fixture, dict) and LayoutKey.SIZE in fixture
    ]
    assert island
    assert [name for name in island if name not in lowered] == []


def test_what_hangs_on_the_walls_keeps_its_height():
    experiment = PickPlaceCounterToStove(counter_height=0.75)
    layout = RoboCasaLayout.from_layout_type(experiment.layout)

    lowered = layout.lowered_fixtures(experiment.counter_height)

    on_the_walls = [
        fixture[LayoutKey.NAME]
        for group in layout.configuration.values()
        for row_name, row in group.items()
        if row_name in RowOffTheFloor and isinstance(row, list)
        for fixture in row
    ]
    assert [name for name in on_the_walls if name in lowered] == []


# %% the arm plane


def test_the_arm_plane_puts_the_tool_frame_where_a_configuration_of_it_does():
    arm_plane = HSRBArmPlane()
    tool_frame = arm_plane.robot.all_arms[0].end_effector.tool_frame
    for joint, position in {
        HSRBJoint.ARM_LIFT: 0.4,
        HSRBJoint.ARM_FLEX: -0.8,
        HSRBJoint.WRIST_FLEX: -1.2,
    }.items():
        arm_plane.world.get_connection_by_name(joint).position = position
    base_T_tool = arm_plane.world.compute_forward_kinematics_np(
        arm_plane.robot.root, tool_frame
    )

    for joint, position in arm_plane.joint_positions_for(base_T_tool).items():
        arm_plane.world.get_connection_by_name(joint).position = position
    reached = arm_plane.world.compute_forward_kinematics_np(
        arm_plane.robot.root, tool_frame
    )

    assert reached[:3, 3] == pytest.approx(
        base_T_tool[:3, 3], abs=REACHED_POSE_TOLERANCE
    )


def test_the_arm_plane_turns_the_wrist_to_the_goals_roll_where_it_puts_the_tool_frame():
    arm_plane = HSRBArmPlane()
    tool_frame = arm_plane.robot.all_arms[0].end_effector.tool_frame
    for joint, position in {
        HSRBJoint.ARM_LIFT: 0.4,
        HSRBJoint.ARM_FLEX: -0.8,
        HSRBJoint.WRIST_FLEX: -1.2,
        HSRBJoint.WRIST_ROLL: 0.7,
    }.items():
        arm_plane.world.get_connection_by_name(joint).position = position
    base_T_tool = arm_plane.world.compute_forward_kinematics_np(
        arm_plane.robot.root, tool_frame
    )

    for joint, position in arm_plane.joint_positions_for(base_T_tool).items():
        arm_plane.world.get_connection_by_name(joint).position = position
    reached = arm_plane.world.compute_forward_kinematics_np(
        arm_plane.robot.root, tool_frame
    )

    assert reached == pytest.approx(base_T_tool, abs=REACHED_POSE_TOLERANCE)


def configured_tool_pose(arm_plane: HSRBArmPlane) -> np.ndarray:
    """
    :return: Where a configuration of the arm puts the tool frame, relative to the
        HSR's root.
    """
    tool_frame = arm_plane.robot.all_arms[0].end_effector.tool_frame
    for joint, position in {
        HSRBJoint.ARM_LIFT: 0.4,
        HSRBJoint.ARM_FLEX: -0.8,
        HSRBJoint.WRIST_FLEX: -1.2,
    }.items():
        arm_plane.world.get_connection_by_name(joint).position = position
    return arm_plane.world.compute_forward_kinematics_np(
        arm_plane.robot.root, tool_frame
    )


def test_the_arm_plane_reaches_as_far_ahead_as_a_configuration_of_it_puts_the_tool():
    arm_plane = HSRBArmPlane()
    base_T_tool = configured_tool_pose(arm_plane)

    distances = arm_plane.reachable_distances(base_T_tool[2, 3], base_T_tool[:3, :3])

    assert min(abs(distance - base_T_tool[0, 3]) for distance in distances) <= (
        REACHED_POSE_TOLERANCE
    )


def test_the_arm_plane_puts_the_tool_frame_at_each_distance_it_reaches():
    arm_plane = HSRBArmPlane()
    tool_frame = arm_plane.robot.all_arms[0].end_effector.tool_frame
    base_T_tool = configured_tool_pose(arm_plane)
    distances = arm_plane.reachable_distances(base_T_tool[2, 3], base_T_tool[:3, :3])

    for distance in (min(distances), max(distances)):
        goal = base_T_tool.copy()
        goal[0, 3] = distance
        for joint, position in arm_plane.joint_positions_for(goal).items():
            arm_plane.world.get_connection_by_name(joint).position = position
        reached = arm_plane.world.compute_forward_kinematics_np(
            arm_plane.robot.root, tool_frame
        )
        assert reached[:3, 3] == pytest.approx(goal[:3, 3], abs=REACHED_POSE_TOLERANCE)


# %% choosing the approach


@pytest.fixture(scope="module")
def counter_to_stove() -> PickPlaceCounterToStove:
    """
    The counter-to-stove task's scene, not run.
    """
    experiment = PickPlaceCounterToStove(headless=True)
    experiment.build_world()
    return experiment


def test_approaches_are_tried_level_first_and_from_the_nearest_floor_first(
    counter_to_stove: PickPlaceCounterToStove,
):
    """
    The first heading points at the food from the floor nearest it that the base can
    stand on, so the arm reaches over the least of the counter.
    """
    approaches = counter_to_stove.approaches()
    navigable_space = counter_to_stove.floor.navigable_space(
        counter_to_stove.robot,
        counter_to_stove.base_collision_rule.violated_distance,
    )
    world = counter_to_stove.world
    nearest_floor = world.transform(
        navigable_space.nearest_free_point(
            Point2(*counter_to_stove.grasp_point[:2], reference_frame=world.root)
        ),
        world.root,
    )
    to_food = counter_to_stove.grasp_point[:2] - nearest_floor.to_np()[:2]

    assert approaches[0].tilt == 0.0
    assert approaches[0].heading == pytest.approx(math.atan2(to_food[1], to_food[0]))
    assert [approach.tilt for approach in approaches] == sorted(
        approach.tilt for approach in approaches
    )
    assert len(approaches) == (
        counter_to_stove.tilt_steps * counter_to_stove.heading_steps
    )


def hand_path_is_free_at(experiment: HSRPickPlaceTask, tilt: float) -> bool:
    """
    :return: Whether the hand's path is free for the approach at ``tilt``, heading
        towards the food from where the HSR stands.
    """
    arm = experiment.robot.all_arms[0]
    heading = experiment.approaches()[0].heading
    grasp = experiment.grasp(Approach(heading=heading, tilt=tilt))
    pick_up = PickUpAction(grasp, arm, retreat_distance=experiment.lift)
    picking = pick_up.grasp_pose_sequence(grasp.world_T_grasp, arm.end_effector, grasp)
    return experiment.hand_path_is_free(
        grasp, picking, experiment.reach_space(pick_up.approach_clearance)
    )


def test_a_level_approach_to_food_on_a_plate_is_blocked(
    counter_to_stove: PickPlaceCounterToStove,
):
    """
    Coming in level, the hand passes within its half width of the plate the food lies
    on before it reaches the food.
    """
    assert not hand_path_is_free_at(counter_to_stove, 0.0)


def test_a_steep_approach_to_food_on_a_plate_is_free(
    counter_to_stove: PickPlaceCounterToStove,
):
    assert hand_path_is_free_at(counter_to_stove, math.pi / 3)


# %% the run


@runs_in_ci
def test_segmind_sees_the_object_put_down_in_the_receptacle(
    performed_task: HSRPickPlaceTask,
):
    """
    Held by friction alone, the object travels to the receptacle and is put down in
    it, as SegMind, watching the run, sees it the last time it is let go of. What
    stands right below a thin receptacle, a plate's counter, may hold it up too.
    """
    food = performed_task.food.root
    events_of_the_food = [
        event
        for event in performed_task.detected_events
        if event.tracked_object is food
    ]
    last_release = max(
        event.timestamp
        for event in events_of_the_food
        if isinstance(event, LossOfGraspEvent)
    )

    put_down_on = {
        event.with_object
        for event in events_of_the_food
        if isinstance(event, PlacingEvent) and event.timestamp >= last_release
    }
    assert put_down_on & set(
        performed_task.colliding_bodies_below(performed_task.receptacle)
    )
