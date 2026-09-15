"""
Tests for building SegMind from the detectors a caller asks for: what each detector
detects and needs, a detector set pulling in what is needed in the order it has to tick,
the one set holding every detector, and the grippers a scene is found to hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import List

from segmind.datastructures.events import (
    ContainmentEvent,
    DetectionEvent,
    EventWithTrackedObjects,
    GraspEvent,
    HoleContactEvent,
    LossOfGraspEvent,
    LossOfSupportEvent,
    PickUpEvent,
    StopTranslationEvent,
    SupportEvent,
    TranslationEvent,
)
from segmind.detector_set import DetectorSet
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    LiftDetector,
    LossOfContactDetector,
    RotationDetector,
    StopLiftDetector,
    StopRotationDetector,
    StopTranslationDetector,
    TranslationDetector,
)
from segmind.detectors.base import AbstractDetector, SegmindContext
from segmind.detectors.coarse_event_detector_nodes import (
    PickUpDetector,
    PlacingDetector,
)
from segmind.detectors.grasp_detector_nodes import GraspDetector, LossOfGraspDetector
from segmind.detectors.spatial_relation_detector_nodes import (
    ContainmentDetector,
    HoleContactDetector,
    InsertionDetector,
    LossOfContainmentDetector,
    LossOfHoleContactDetector,
    LossOfSupportDetector,
    SupportDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.exceptions import NoDetectorDetectsEvent
from segmind.scene_parts import Gripper, SceneParts
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)

from ..dataset.two_finger_gripper import annotate_two_finger_gripper
from .test_segmind_detectors import (
    _build_grasp_world,
    _second_gripper_away_from_the_first,
)

# %% a need nothing can meet


@dataclass(unsafe_hash=True)
class EventNothingDetects(EventWithTrackedObjects):
    """
    An event none of segmind's detectors detects.
    """


@dataclass(eq=False, repr=False)
class DetectorNeedingAnUndetectedEvent(AbstractDetector[TranslationEvent]):
    """
    A detector that needs an event nothing detects.
    """

    @classmethod
    def required_event_types(cls):
        return (EventNothingDetects,)

    def update_context_and_events(
        self, context, segmind_context, tracked_objects
    ) -> List[DetectionEvent]:
        return []


# %% scenes


@pytest.fixture
def scene_with_two_grippers():
    """
    A shape, the gripper that grasps it where it will be put, and a second gripper far
    away, both declared as grippers in the world.
    """
    world, shape, left_finger, right_finger, tool_frame = _build_grasp_world()
    annotate_two_finger_gripper(
        world, "grasping_gripper", left_finger, right_finger, tool_frame
    )
    second_thumb, second_finger, second_tool_frame = (
        _second_gripper_away_from_the_first(world)
    )
    annotate_two_finger_gripper(
        world, "idle_gripper", second_thumb, second_finger, second_tool_frame
    )
    return world, shape, (left_finger, right_finger, tool_frame), second_tool_frame


def _events_of(segmind_context: SegmindContext, event_type) -> List[DetectionEvent]:
    return [
        event
        for event in segmind_context.logger.get_events()
        if isinstance(event, event_type)
    ]


# %% what a detector detects and needs


@pytest.mark.parametrize(
    "detector_type, event_type",
    [
        (PickUpDetector, PickUpEvent),
        (GraspDetector, GraspEvent),
        (HoleContactDetector, HoleContactEvent),
        (TranslationDetector, TranslationEvent),
    ],
)
def test_a_detector_names_the_event_it_detects(detector_type, event_type):
    assert detector_type.detected_event_type() is event_type


@pytest.mark.parametrize(
    "detector_type, needed",
    [
        (PickUpDetector, (TranslationEvent, LossOfSupportEvent)),
        (PlacingDetector, (StopTranslationEvent, SupportEvent)),
        (InsertionDetector, (HoleContactEvent, ContainmentEvent)),
    ],
)
def test_a_rule_detector_needs_the_events_its_rule_reads(detector_type, needed):
    assert detector_type.required_event_types() == needed


@pytest.mark.parametrize(
    "detector_type, needed",
    [
        (LossOfSupportDetector, (SupportEvent,)),
        (StopTranslationDetector, (TranslationEvent,)),
        (LossOfGraspDetector, (GraspEvent,)),
        (LiftDetector, (GraspEvent,)),
    ],
)
def test_a_detector_that_continues_what_another_began_needs_that_one(
    detector_type, needed
):
    assert detector_type.required_event_types() == needed


def test_a_detector_that_starts_something_needs_nothing():
    assert TranslationDetector.required_event_types() == ()


# %% composing a set


def test_adding_a_rule_detector_pulls_in_the_detectors_it_needs():
    world, shape, *_ = _build_grasp_world()
    detectors = DetectorSet(SceneParts.of_world(world))

    detectors.add(PickUpDetector(tracked_object=shape))

    assert {type(detector) for detector in detectors.detectors} == {
        PickUpDetector,
        TranslationDetector,
        LossOfSupportDetector,
        SupportDetector,
    }
    assert all(detector.tracked_object is shape for detector in detectors.detectors)


def test_every_detector_comes_after_the_detectors_it_needs():
    world, shape, *_ = _build_grasp_world()
    detectors = DetectorSet(SceneParts.of_world(world))
    detectors.add(PlacingDetector(tracked_object=shape))
    detectors.add(PickUpDetector(tracked_object=shape))

    ordered = detectors.detectors

    for position, detector in enumerate(ordered):
        for needed in detector.required_event_types():
            [producer_position] = [
                index
                for index, candidate in enumerate(ordered)
                if candidate.detected_event_type() is needed
            ]
            assert producer_position < position


def test_the_same_detector_added_twice_is_kept_once():
    world, shape, *_ = _build_grasp_world()
    detectors = DetectorSet(SceneParts.of_world(world))

    detectors.add(TranslationDetector(tracked_object=shape))
    detectors.add(TranslationDetector(tracked_object=shape))

    assert len(detectors.detectors) == 1


def test_a_detector_the_caller_adds_replaces_one_pulled_in_for_it():
    world, shape, *_ = _build_grasp_world()
    detectors = DetectorSet(SceneParts.of_world(world))
    detectors.add(PickUpDetector(tracked_object=shape))
    own_translation = TranslationDetector(tracked_object=shape)

    detectors.add(own_translation)

    [translation] = [
        detector
        for detector in detectors.detectors
        if isinstance(detector, TranslationDetector)
    ]
    assert translation is own_translation


def test_a_detector_pulled_in_leaves_one_the_caller_added():
    world, shape, *_ = _build_grasp_world()
    detectors = DetectorSet(SceneParts.of_world(world))
    own_translation = TranslationDetector(tracked_object=shape)
    detectors.add(own_translation)

    detectors.add(PickUpDetector(tracked_object=shape))

    [translation] = [
        detector
        for detector in detectors.detectors
        if isinstance(detector, TranslationDetector)
    ]
    assert translation is own_translation


def test_needing_an_event_nothing_detects_is_refused():
    world, shape, *_ = _build_grasp_world()
    detectors = DetectorSet(SceneParts.of_world(world))

    with pytest.raises(NoDetectorDetectsEvent):
        detectors.add(DetectorNeedingAnUndetectedEvent(tracked_object=shape))


def test_asking_for_an_event_adds_a_detector_for_it_per_gripper(
    scene_with_two_grippers,
):
    world, shape, _, _ = scene_with_two_grippers
    detectors = DetectorSet(SceneParts.of_world(world))

    detectors.add_detecting(GraspEvent, shape)

    grasp_detectors = [
        detector
        for detector in detectors.detectors
        if isinstance(detector, GraspDetector)
    ]
    assert {detector.tool_frame for detector in grasp_detectors} == {
        gripper.tool_frame for gripper in SceneParts.of_world(world).grippers
    }


# %% the set holding every detector


def test_the_set_of_all_detectors_holds_every_detector_segmind_defines(
    scene_with_two_grippers,
):
    world, *_ = scene_with_two_grippers

    detectors = DetectorSet.with_all_detectors(SceneParts.of_world(world))

    assert {type(detector) for detector in detectors.detectors} == {
        ContactDetector,
        LossOfContactDetector,
        HoleContactDetector,
        LossOfHoleContactDetector,
        SupportDetector,
        LossOfSupportDetector,
        ContainmentDetector,
        LossOfContainmentDetector,
        TranslationDetector,
        StopTranslationDetector,
        RotationDetector,
        StopRotationDetector,
        GraspDetector,
        LossOfGraspDetector,
        LiftDetector,
        StopLiftDetector,
        PickUpDetector,
        PlacingDetector,
        InsertionDetector,
    }


def test_the_set_of_all_detectors_leaves_out_what_a_scene_without_grippers_cannot_detect():
    world, *_ = _build_grasp_world()

    detectors = DetectorSet.with_all_detectors(SceneParts.of_world(world))

    detected = {detector.detected_event_type() for detector in detectors.detectors}
    assert GraspEvent not in detected
    assert LiftDetector not in {type(detector) for detector in detectors.detectors}


# %% grippers a scene holds


def test_a_scene_names_each_two_finger_grippers_tips_and_tool_frame():
    world, _, left_finger, right_finger, tool_frame = _build_grasp_world()
    annotate_two_finger_gripper(world, "gripper", left_finger, right_finger, tool_frame)

    assert SceneParts.of_world(world).grippers == [
        Gripper(finger_tips=(left_finger, right_finger), tool_frame=tool_frame)
    ]


def test_a_grasp_by_one_gripper_is_not_lost_to_another(scene_with_two_grippers):
    world, shape, (_, _, grasping_tool_frame), _ = scene_with_two_grippers
    detectors = DetectorSet(SceneParts.of_world(world))
    detectors.add_detecting(LossOfGraspEvent, shape)
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    segmind_context = executor.context.require_extension(SegmindContext)
    executor.compile(detectors.build_statechart())

    shape.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0, 0, 0, reference_frame=world.root
    )
    executor.tick()
    executor.tick()

    [grasp] = _events_of(segmind_context, GraspEvent)
    assert grasp.with_object is grasping_tool_frame
    assert _events_of(segmind_context, LossOfGraspEvent) == []
