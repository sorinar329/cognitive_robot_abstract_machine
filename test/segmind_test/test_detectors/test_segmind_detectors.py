import numpy as np
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
    HoldingEvent,
    LossOfHoldingEvent,
    LiftingEvent,
    StackingEvent,
)
from segmind.detectors.agent_event_detectors_nodes import HoldingDetector, LossOfHoldingDetector, LiftingDetector
from segmind.detectors.atomic_event_detectors_nodes import RotationDetector, StopRotationDetector, ContactDetector, \
    LossOfContactDetector, TranslationDetector, StopTranslationDetector
from segmind.detectors.base import SegmindContext
from segmind.detectors.coarse_event_detector_nodes import PickUpDetector, PlacingDetector, StackingDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector, LossOfSupportDetector, \
    ContainmentDetector, LossOfContainmentDetector, InsertionDetector
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cube
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection, Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


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

def test_contact_detector(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([ContactDetector(),LossOfContactDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, ContactEvent)) == 0

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfContactEvent)) == 0

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box1.global_pose.x, box1.global_pose.y, box1.global_pose.z)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 1
    assert len(events_of(segmind_context, LossOfContactEvent)) == 0

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box2.global_pose.x, box2.global_pose.y, box2.global_pose.z)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 2
    assert len(events_of(segmind_context, LossOfContactEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfContactEvent)) == 2
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)

def test_support_detector(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([SupportDetector(), LossOfSupportDetector()])
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 0.93)
    segmind_executor.compile(statechart)
    segmind_executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box1.global_pose.x, box1.global_pose.y,
                                                                                 box1.global_pose.z + 0.56)
    segmind_executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 2
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box2.global_pose.x, box2.global_pose.y,
                                                                                 box2.global_pose.z + 0.56)
    segmind_executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 3
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 2

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 3
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)

def test_containment_detector(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([ContainmentDetector(), LossOfContainmentDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, ContainmentEvent)) == 0

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box1.global_pose.x, box1.global_pose.y,
                                                                                 box1.global_pose.z)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContainmentEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box2.global_pose.x, box2.global_pose.y,
                                                                                 box2.global_pose.z)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContainmentEvent)) == 2
    assert len(events_of(segmind_context, LossOfContainmentEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfContainmentEvent)) == 2
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)

def test_pickup(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([PickUpDetector(), SupportDetector(), TranslationDetector(), LossOfSupportDetector()])
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 0.93)

    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 1

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=box2.global_pose.x,
            y=box2.global_pose.y,
            z=box2.global_pose.z + 0.56 + i * 0.1,
        )
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) >= 1
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1
    assert len(events_of(segmind_context, PickUpEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)

def test_placing(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [SupportDetector(), TranslationDetector(), StopTranslationDetector(), PlacingDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=box2.global_pose.x,
            y=box2.global_pose.y,
            z=box2.global_pose.z + 0.97 - i * 0.1,
        )
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) >= 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=box2.global_pose.x,
        y=box2.global_pose.y,
        z=box2.global_pose.z + 0.56,
    )
    for _ in range(5):
        segmind_executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 1
    assert len(events_of(segmind_context, StopTranslationEvent)) == 1
    assert len(events_of(segmind_context, PlacingEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)

def test_pickup_then_place_back_on_same_surface(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [PickUpDetector(), PlacingDetector(), SupportDetector(), LossOfSupportDetector(),
         TranslationDetector(), StopTranslationDetector()])
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=box2.global_pose.x, y=box2.global_pose.y, z=box2.global_pose.z + 0.56,
    )

    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 1

    # Pick the milk up off box2.
    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=box2.global_pose.x,
            y=box2.global_pose.y,
            z=box2.global_pose.z + 0.56 + i * 0.1,
        )
        segmind_executor.tick()

    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1
    assert len(events_of(segmind_context, PickUpEvent)) == 1

    for _ in range(5):
        segmind_executor.tick()

    # Place the milk back down onto the very same surface (box2) it was picked up from.
    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=box2.global_pose.x,
            y=box2.global_pose.y,
            z=box2.global_pose.z + 0.97 - i * 0.1,
        )
        segmind_executor.tick()

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=box2.global_pose.x,
        y=box2.global_pose.y,
        z=box2.global_pose.z + 0.56,
    )
    for _ in range(5):
        segmind_executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 2
    assert len(events_of(segmind_context, PlacingEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)


def test_holding_and_lifting(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    world = _simple_apartment_setup
    milk_pose = milk.global_pose
    parked_pose = HomogeneousTransformationMatrix.from_xyz_rpy(5, 5, 5)

    with world.modify_world():
        gripper = Body(
            name=PrefixedName("gripper_finger"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
            visual=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
        )
        world.add_connection(Connection6DoF.create_with_dofs(
            world=world,
            parent=world.root,
            child=gripper,
        ))
    gripper.parent_connection.origin = parked_pose

    statechart = SegmindStatechart().build_statechart([
        HoldingDetector(gripper_body_names=["gripper_finger"], tracked_object=milk),
        LossOfHoldingDetector(gripper_body_names=["gripper_finger"], tracked_object=milk),
        LiftingDetector(tracked_object=milk),
    ])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, HoldingEvent)) == 0

    # Move the gripper onto the milk to start a hold.
    gripper.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        milk_pose.x, milk_pose.y, milk_pose.z
    )
    segmind_executor.tick()
    assert len(events_of(segmind_context, HoldingEvent)) == 1
    assert len(events_of(segmind_context, LiftingEvent)) == 0

    # Raise both the milk and the gripper together, staying in contact, past the lift threshold.
    for i in range(5):
        z = milk_pose.z + 0.02 * (i + 1)
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(milk_pose.x, milk_pose.y, z)
        gripper.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(milk_pose.x, milk_pose.y, z)
        segmind_executor.tick()

    assert len(events_of(segmind_context, LiftingEvent)) == 1

    # Move the gripper away again to end the hold.
    gripper.parent_connection.origin = parked_pose
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfHoldingEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)


def _add_stack_cube(world, name):
    cube = Body(
        name=PrefixedName(name),
        collision=ShapeCollection([Box(scale=Scale(0.2, 0.2, 0.2))]),
        visual=ShapeCollection([Box(scale=Scale(0.2, 0.2, 0.2))]),
    )
    world.add_connection(Connection6DoF.create_with_dofs(world=world, parent=world.root, child=cube))
    world.add_semantic_annotations([Cube(root=cube)])
    return cube


def test_stacking_detector(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    world = _simple_apartment_setup
    box1_pose = box1.global_pose
    parked_pose = HomogeneousTransformationMatrix.from_xyz_rpy(5, 5, 5)
    # Half of box1's height (0.95) plus half of a stack cube's height (0.2), minus a small
    # deliberate overlap so is_supported_by sees contact rather than two disjoint boxes.
    first_level_offset = 0.475 + 0.1 - 0.02
    level_offset = 0.1 + 0.1 - 0.02

    with world.modify_world():
        cube_a = _add_stack_cube(world, "stack_cube_a")
        cube_b = _add_stack_cube(world, "stack_cube_b")
        cube_c = _add_stack_cube(world, "stack_cube_c")

    for cube in (cube_a, cube_b, cube_c):
        cube.parent_connection.origin = parked_pose

    statechart = SegmindStatechart().build_statechart(
        [SupportDetector(), LossOfSupportDetector(), StackingDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, StackingEvent)) == 0

    # cube_a rests on box1, which is not annotated as a Cube -- alone this must not count.
    cube_a.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        box1_pose.x, box1_pose.y, box1_pose.z + first_level_offset,
    )
    segmind_executor.tick()
    assert len(events_of(segmind_context, StackingEvent)) == 0

    # cube_b rests on cube_a, which is itself supported by box1 -- this must count.
    cube_b.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        box1_pose.x, box1_pose.y, box1_pose.z + first_level_offset + level_offset,
    )
    segmind_executor.tick()
    assert len(events_of(segmind_context, StackingEvent)) == 1

    # cube_c rests on cube_b, extending the chain -- one more event.
    cube_c.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        box1_pose.x, box1_pose.y, box1_pose.z + first_level_offset + 2 * level_offset,
    )
    segmind_executor.tick()
    assert len(events_of(segmind_context, StackingEvent)) == 2

    for cube in (cube_a, cube_b, cube_c):
        cube.parent_connection.origin = parked_pose


def test_stacking_detector_requires_currently_supported_lower_object(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    world = _simple_apartment_setup
    box1_pose = box1.global_pose
    parked_pose = HomogeneousTransformationMatrix.from_xyz_rpy(5, 5, 5)
    first_level_offset = 0.475 + 0.1 - 0.02
    level_offset = 0.1 + 0.1 - 0.02

    with world.modify_world():
        cube_a = _add_stack_cube(world, "stack_cube_live_a")
        cube_b = _add_stack_cube(world, "stack_cube_live_b")

    cube_a.parent_connection.origin = parked_pose
    cube_b.parent_connection.origin = parked_pose

    statechart = SegmindStatechart().build_statechart(
        [SupportDetector(), LossOfSupportDetector(), StackingDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    # cube_a is placed on box1 and then knocked off again before cube_b is placed on it.
    cube_a.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        box1_pose.x, box1_pose.y, box1_pose.z + first_level_offset,
    )
    segmind_executor.tick()
    cube_a.parent_connection.origin = parked_pose
    segmind_executor.tick()

    # cube_b follows cube_a to its new, unsupported spot -- still resting on cube_a, but
    # cube_a itself no longer has any support, so this must not count as stacking.
    cube_b.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        parked_pose.x, parked_pose.y, parked_pose.z + level_offset,
    )
    segmind_executor.tick()
    assert len(events_of(segmind_context, StackingEvent)) == 0

    cube_a.parent_connection.origin = parked_pose
    cube_b.parent_connection.origin = parked_pose


def test_translation(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [TranslationDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 0

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.1, y=-3, z=0.25
        )
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)


def test_stop_translation(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [SupportDetector(), TranslationDetector(), StopTranslationDetector(), PlacingDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.1, y=-3, z=0.25
        )
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 1

    for _ in range(5):
        segmind_executor.tick()

    assert len(events_of(segmind_context, StopTranslationEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)


def test_insertion(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [ContactDetector(), InsertionDetector(), LossOfContactDetector(), ContainmentDetector()])

    with segmind_executor.context.world.modify_world():
        hole = Body(
            name=PrefixedName("box_hole"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
            visual=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
        )
        hole_connection = FixedConnection(
            parent=segmind_executor.context.world.root,
            child=hole,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                2, 2, 2, reference_frame=segmind_executor.context.world.root
            ),
        )
        segmind_executor.context.world.add_connection(hole_connection)
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, InsertionEvent)) == 0
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(hole.global_pose.x, hole.global_pose.y,
                                                                                 hole.global_pose.z)

    segmind_executor.tick()

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box2.global_pose.x,box2.global_pose.y,box2.global_pose.z)
    segmind_executor.tick()

    assert len(events_of(segmind_context, InsertionEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)


def test_rotation(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [RotationDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()


    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        milk.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(roll=i*0.1)
        )
        segmind_executor.tick()

    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) >= 1


def test_stop_rotation(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [RotationDetector(), StopRotationDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        milk.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(roll=i*0.1)
        )
        segmind_executor.tick()
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) >= 1

    for _ in range(5):
        segmind_executor.tick()
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, StopRotationEvent)]) >= 1

