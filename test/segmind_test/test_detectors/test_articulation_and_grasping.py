"""
Tests for the events of an agent handling things: a part's joint moving, opening and
closing a part by its handle, and grasping as a grasp followed by a pick-up.
"""

from __future__ import annotations

import pytest
from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import List

from segmind.datastructures.events import (
    ClosingEvent,
    DetectionEvent,
    GraspEvent,
    GraspingEvent,
    JointDirection,
    JointMotionEvent,
    OpeningEvent,
    PickUpEvent,
)
from segmind.detector_set import DetectorSet
from segmind.detectors.base import DetectorStateChart, SegmindContext
from segmind.detectors.coarse_event_detector_nodes import GraspingDetector
from segmind.detectors.joint_detector_nodes import (
    ClosingDetector,
    JointMotionDetector,
    OpeningDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.scene_parts import SceneParts
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from ..dataset.drawer_with_handle import drawer_with_handle
from ..dataset.two_finger_gripper import annotate_two_finger_gripper
from .test_segmind_detectors import _build_grasp_world

JOINT_STEP = 0.05
"""
How far a test moves a joint per tick, in its own unit.
"""

TICKS_OF_MOTION = 6
"""
How many ticks a test moves a joint for; more than a motion window spans.
"""

TICKS_STANDING_STILL = 6
"""
How many ticks a test leaves a joint where it is.
"""

OPENED_POSITION = JOINT_STEP * TICKS_OF_MOTION
"""
Where a part stands after a test has opened it.
"""


# %% scenes


def _gripper_bodies_fixed_to(
    world: World, parent: Body, x: float, y: float, z: float
) -> tuple[Body, Body, Body]:
    """
    Add a two-fingered gripper's bodies to ``world``, fixed to ``parent`` at a position.

    :return: The thumb tip, the finger tip and the tool frame.
    """
    bodies = []
    with world.modify_world():
        for name, has_shape in (
            ("thumb_tip", True),
            ("finger_tip", True),
            ("tool_frame", False),
        ):
            shapes = (
                {
                    "collision": ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
                    "visual": ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
                }
                if has_shape
                else {}
            )
            body = Body(name=PrefixedName(name), **shapes)
            world.add_connection(
                FixedConnection(
                    parent=parent,
                    child=body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x, y, z, reference_frame=parent
                    ),
                )
            )
            bodies.append(body)
    return bodies[0], bodies[1], bodies[2]


def _drawer_scene(gripper_on_handle: bool):
    """
    A drawer with a gripper holding its handle, or standing far from it.

    :return: The world, the drawer and the gripper's tool frame.
    """
    world, drawer = drawer_with_handle()
    parent, x, y, z = (
        (drawer.handle.root, 0.0, 0.0, 0.0)
        if gripper_on_handle
        else (world.root, 5.0, 5.0, 5.0)
    )
    thumb_tip, finger_tip, tool_frame = _gripper_bodies_fixed_to(world, parent, x, y, z)
    annotate_two_finger_gripper(world, "gripper", thumb_tip, finger_tip, tool_frame)
    return world, drawer, tool_frame


def _watching(
    world: World, event_type
) -> tuple[EpisodeSegmenterExecutor, SegmindContext]:
    """
    An executor ticking every detector needed to detect ``event_type`` in ``world``.
    """
    detectors = DetectorSet(SceneParts.of_world(world))
    detectors.add_detecting(event_type)
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    executor.compile(detectors.build_statechart())
    return executor, executor.context.require_extension(SegmindContext)


def _move_joint(executor, drawer, start: float, step: float, ticks: int) -> None:
    """
    Tick ``executor`` once with the drawer at ``start``, then once per step as its joint
    moves by ``step``.
    """
    drawer.mechanical_joint.position = start
    executor.tick()
    for tick in range(1, ticks + 1):
        drawer.mechanical_joint.position = start + step * tick
        executor.tick()


def _events_of(segmind_context: SegmindContext, event_type) -> List[DetectionEvent]:
    return [
        event
        for event in segmind_context.logger.get_events()
        if isinstance(event, event_type)
    ]


# %% what a scene holds


def test_a_scene_names_each_part_moving_on_a_joint_that_has_a_handle():
    world, drawer = drawer_with_handle()

    assert SceneParts.of_world(world).articulated_parts == [drawer]


# %% a joint moving


def test_a_joint_moving_up_is_detected_moving_towards_its_upper_limit():
    world, drawer, _ = _drawer_scene(gripper_on_handle=True)
    executor, segmind_context = _watching(world, JointMotionEvent)

    _move_joint(executor, drawer, start=0.0, step=JOINT_STEP, ticks=TICKS_OF_MOTION)

    [motion] = _events_of(segmind_context, JointMotionEvent)
    assert motion.tracked_object is drawer.root
    assert motion.direction is JointDirection.TOWARDS_UPPER_LIMIT
    assert motion.current_position > motion.start_position


def test_a_joint_moving_down_is_detected_moving_towards_its_lower_limit():
    world, drawer, _ = _drawer_scene(gripper_on_handle=True)
    executor, segmind_context = _watching(world, JointMotionEvent)

    _move_joint(
        executor, drawer, start=OPENED_POSITION, step=-JOINT_STEP, ticks=TICKS_OF_MOTION
    )

    [motion] = _events_of(segmind_context, JointMotionEvent)
    assert motion.direction is JointDirection.TOWARDS_LOWER_LIMIT


def test_a_joint_standing_still_is_not_detected_moving():
    world, drawer, _ = _drawer_scene(gripper_on_handle=True)
    executor, segmind_context = _watching(world, JointMotionEvent)

    _move_joint(executor, drawer, start=0.0, step=0.0, ticks=TICKS_STANDING_STILL)

    assert _events_of(segmind_context, JointMotionEvent) == []


# %% opening and closing a part by its handle


def test_a_part_moving_up_while_its_handle_is_grasped_is_opened():
    world, drawer, tool_frame = _drawer_scene(gripper_on_handle=True)
    executor, segmind_context = _watching(world, OpeningEvent)

    _move_joint(executor, drawer, start=0.0, step=JOINT_STEP, ticks=TICKS_OF_MOTION)

    [opening] = _events_of(segmind_context, OpeningEvent)
    assert opening.tracked_object is drawer.root
    assert opening.with_object is tool_frame
    assert _events_of(segmind_context, ClosingEvent) == []


def test_a_part_moving_down_while_its_handle_is_grasped_is_closed():
    world, drawer, tool_frame = _drawer_scene(gripper_on_handle=True)
    executor, segmind_context = _watching(world, ClosingEvent)

    _move_joint(
        executor, drawer, start=OPENED_POSITION, step=-JOINT_STEP, ticks=TICKS_OF_MOTION
    )

    [closing] = _events_of(segmind_context, ClosingEvent)
    assert closing.tracked_object is drawer.root
    assert closing.with_object is tool_frame


def test_a_part_moving_with_nobody_holding_its_handle_is_not_opened():
    world, drawer, _ = _drawer_scene(gripper_on_handle=False)
    executor, segmind_context = _watching(world, OpeningEvent)

    _move_joint(executor, drawer, start=0.0, step=JOINT_STEP, ticks=TICKS_OF_MOTION)

    assert _events_of(segmind_context, JointMotionEvent) != []
    assert _events_of(segmind_context, OpeningEvent) == []


def test_an_opening_names_the_grasp_and_the_joint_motion_it_was_concluded_from():
    world, drawer, _ = _drawer_scene(gripper_on_handle=True)
    executor, segmind_context = _watching(world, OpeningEvent)
    _move_joint(executor, drawer, start=0.0, step=JOINT_STEP, ticks=TICKS_OF_MOTION)
    [opening] = _events_of(segmind_context, OpeningEvent)

    consumed = opening.participating_events()

    assert {type(event) for event in consumed} == {GraspEvent, JointMotionEvent}


@pytest.mark.parametrize("detector_type", [OpeningDetector, ClosingDetector])
def test_opening_and_closing_need_a_grasp_and_a_joint_motion(detector_type):
    assert detector_type.required_event_types() == (GraspEvent, JointMotionEvent)


def test_a_joint_motion_detector_needs_nothing():
    assert JointMotionDetector.required_event_types() == ()


# %% grasping


def _grasping_detected_from(logged: List[DetectionEvent]) -> List[DetectionEvent]:
    """
    Tick a grasping detector once over events already logged.

    :param logged: The events the detector finds logged.
    :return: The grasping events it concluded.
    """
    world, *_ = _build_grasp_world()
    statechart = DetectorStateChart()
    statechart.add_node(GraspingDetector())
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    segmind_context = executor.context.require_extension(SegmindContext)
    executor.compile(statechart)
    for event in logged:
        segmind_context.logger.log_event(event, segmind_context.tracker_registry)

    executor.tick()

    return _events_of(segmind_context, GraspingEvent)


def test_a_grasp_followed_by_a_pick_up_of_the_same_object_is_grasping():
    shape, tool_frame = Body(name=PrefixedName("shape")), Body(
        name=PrefixedName("tool_frame")
    )
    grasp = GraspEvent(tracked_object=shape, with_object=tool_frame)
    pick_up = PickUpEvent(tracked_object=shape)

    [grasping] = _grasping_detected_from([grasp, pick_up])

    assert grasping.tracked_object is shape
    assert grasping.with_object is tool_frame


def test_a_pick_up_without_a_grasp_is_not_grasping():
    shape = Body(name=PrefixedName("shape"))

    assert _grasping_detected_from([PickUpEvent(tracked_object=shape)]) == []


def test_grasping_needs_a_grasp_and_a_pick_up():
    assert GraspingDetector.required_event_types() == (GraspEvent, PickUpEvent)
