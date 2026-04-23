import pytest
import rclpy

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import (
    ContactEvent,
    LossOfContactEvent,
    SupportEvent,
    LossOfSupportEvent,
    LossOfContainmentEvent,
    ContainmentEvent,
    InsertionEvent,
    TranslationEvent,
    StopTranslationEvent,
    PickUpEvent,
    PlacingEvent,
    RotationEvent,
    StopRotationEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import RotationDetector, StopRotationDetector, ContactDetector, \
    LossOfContactDetector
from segmind.detectors.base import SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_executor(world):
    context = MotionStatechartContext(world=world)
    milk = world.get_body_by_name("milk.stl")
    box1 = world.get_body_by_name("box")
    box2 = world.get_body_by_name("box_2")
    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_context = segmind_executor.context.require_extension(SegmindContext)
    return segmind_executor, segmind_context, milk, box1, box2


def events_of(segmind_context, event_type):
    return [e for e in segmind_context.logger.get_events() if isinstance(e, event_type)]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_contact_detector(simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([ContactDetector(),LossOfContactDetector()])
    segmind_executor.compile(statechart)
    rclpy.init()
    node = rclpy.create_node("segmind_test")
    publisher = VizMarkerPublisher(_world=segmind_executor.context.world, node=node)
    publisher.with_tf_publisher()
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfContactEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box1.global_pose.x, box1.global_pose.y, box1.global_pose.z)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 2
    assert len(events_of(segmind_context, LossOfContactEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box2.global_pose.x, box2.global_pose.y, box2.global_pose.z)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 3
    assert len(events_of(segmind_context, LossOfContactEvent)) == 2

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    print("a")
    assert len(events_of(segmind_context, LossOfContactEvent)) == 3

def test_support_detector(support_setup):
    executor, segmind_context, _, cylinder, table, cabinet, _ = support_setup

    assert len(events_of(segmind_context, SupportEvent)) == 0
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 0

    executor.tick()

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=table.global_pose.x,
        y=table.global_pose.y,
        z=table.global_pose.z + 0.2,
    )
    executor.tick()
    executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 1

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=cabinet.global_pose.x,
        y=cabinet.global_pose.y,
        z=cabinet.global_pose.z,
    )
    executor.tick()
    executor.tick()
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1


def test_containment_detector(support_setup):
    executor, segmind_context, _, cylinder, table, cabinet, _ = support_setup

    assert len(events_of(segmind_context, ContactEvent)) == 0
    assert len(events_of(segmind_context, ContainmentEvent)) == 0

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=cabinet.global_pose.x,
        y=cabinet.global_pose.y,
        z=cabinet.global_pose.z,
    )
    executor.tick()
    assert len(events_of(segmind_context, ContainmentEvent)) == 1

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=table.global_pose.x,
        y=table.global_pose.y,
        z=table.global_pose.z + 0.2,
    )
    executor.tick()
    assert len(events_of(segmind_context, LossOfContainmentEvent)) == 1


def test_insertion_detector(support_setup):
    executor, segmind_context, _, cylinder, _, cabinet, hole = support_setup

    assert len(segmind_context.holes) == 1
    assert len(events_of(segmind_context, InsertionEvent)) == 0

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=hole.global_pose.x,
        y=hole.global_pose.y - 0.03,
        z=hole.global_pose.z,
    )
    executor.tick()

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=cabinet.global_pose.x,
        y=cabinet.global_pose.y,
        z=cabinet.global_pose.z,
    )
    executor.tick()

    contact_events_with_holes = [
        e for e in events_of(segmind_context, ContactEvent)
        if e.with_object in segmind_context.holes
    ]

    assert len(events_of(segmind_context, ContainmentEvent)) == 1
    assert len(contact_events_with_holes) == 1
    assert len(events_of(segmind_context, InsertionEvent)) == 1


def test_pickup(support_setup):
    executor, segmind_context, _, cylinder, table, _, _ = support_setup

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=table.global_pose.x,
        y=table.global_pose.y,
        z=table.global_pose.z + 0.2,
    )
    executor.tick()
    executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 1

    for i in range(5):
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=table.global_pose.x,
            y=table.global_pose.y,
            z=table.global_pose.z + 0.3 + i * 0.1,
        )
        executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) >= 1
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1
    assert len(events_of(segmind_context, PickUpEvent)) == 1


def test_placing(support_setup):
    executor, segmind_context, _, cylinder, table, _, _ = support_setup

    for i in range(5):
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=table.global_pose.x,
            y=table.global_pose.y,
            z=table.global_pose.z + 0.5 - i * 0.05,
        )
        executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) >= 1

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=table.global_pose.x,
        y=table.global_pose.y,
        z=table.global_pose.z + 0.2,
    )
    for _ in range(5):
        executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 1
    assert len(events_of(segmind_context, StopTranslationEvent)) == 1
    assert len(events_of(segmind_context, PlacingEvent)) == 1


def test_translation(contact_setup):
    executor, segmind_context, _, cylinder = contact_setup

    assert len(events_of(segmind_context, TranslationEvent)) == 0

    for i in range(5):
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.1, y=-3, z=0.25
        )
        executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 1


def test_stop_translation(contact_setup):
    executor, segmind_context, _, cylinder = contact_setup

    for i in range(5):
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.1, y=-3, z=0.25
        )
        executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 1

    for _ in range(5):
        executor.tick()

    assert len(events_of(segmind_context, StopTranslationEvent)) == 1


@pytest.mark.skip(reason="Buggy")
def test_rotation():
    world = setup_contact_world()
    context = MotionStatechartContext(world=world)
    segmind_executor = EpisodeSegmenterExecutor(context=context)
    statechart = SegmindStatechart()
    sc = statechart.build_statechart()
    sc.add_node(RotationDetector())
    segmind_executor.compile(sc)
    segmind_context = segmind_executor.context.require_extension(SegmindContext)

    cylinder = world.get_body_by_name("cylinder_body")
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=-3, z=0.25, roll=i*0.1)
        )
        segmind_executor.tick()

    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) >= 1

@pytest.mark.skip(reason="Buggy")
def test_stop_rotation():
    world = setup_contact_world()
    context = MotionStatechartContext(world=world)
    segmind_executor = EpisodeSegmenterExecutor(context=context)
    statechart = SegmindStatechart()
    sc = statechart.build_statechart()
    sc.add_node(RotationDetector())
    sc.add_node(StopRotationDetector())
    segmind_executor.compile(sc)
    segmind_context = segmind_executor.context.require_extension(SegmindContext)



    cylinder = world.get_body_by_name("cylinder_body")

    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=-3, z=0.25, roll=i*0.1)
        )
        segmind_executor.tick()
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) >= 1

    for _ in range(5):
        segmind_executor.tick()
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, StopRotationEvent)]) >= 1


def test_contact_kitchen_world(simple_apartment_setup):
    world = simple_apartment_setup
    context = MotionStatechartContext(world=world)
    segmind_executor = EpisodeSegmenterExecutor(context=context)
    statechart = SegmindStatechart()
    sc = statechart.build_statechart()
    segmind_executor.compile(sc)
    segmind_context = segmind_executor.context.require_extension(SegmindContext)
    rclpy.init()
    node = rclpy.create_node("segmind_test")
    publisher = VizMarkerPublisher(_world=world, node=node)
    publisher.with_tf_publisher()

    milk = world.get_body_by_name("milk.stl")
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfContactEvent)) == 1

