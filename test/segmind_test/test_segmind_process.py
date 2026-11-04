"""
Tests for SegMind watching a run from a process of its own, which keeps a copy of the
run's world and reports what it detects back into the run's process.
"""

from __future__ import annotations

import threading
from collections import Counter

import pytest
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix

from segmind.datastructures.events import SupportEvent, TranslationEvent
from segmind.detector_selection import DetectorSelection
from segmind.detectors.atomic_event_detectors_nodes import TranslationDetector
from segmind.detectors.coarse_event_detector_nodes import PickUpDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from segmind.exceptions import SegmindProcessFailed
from segmind.segmind_process import SegmindProcess
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from .conftest import RESTING_ON_THE_TABLE
from .dataset.detectors_for_watching import DetectorFailingOnItsFirstTick

DETECTION_TIMEOUT = 20.0
"""
Seconds a test waits for the process to have detected something.
"""

MOVED_ALONG_X = 0.2
"""
How far a test moves a watched body at once.
"""

SHAPES_OF_A_LARGE_BODY = 2000
"""
How many shapes a body has whose addition is a model change too large for the buffer of
a pipe.
"""

MODEL_CHANGE_TIMEOUT = 20.0
"""
Seconds a test gives a change of the world model to be made while it is watched.
"""


@pytest.fixture
def carried_and_carrier() -> tuple[World, Body, Body]:
    """
    A world of its own with two bodies free to move: one to be hung under the other.
    """
    world = World()
    with world.modify_world():
        floor = Body(name=PrefixedName("floor"))
        carried = Body(name=PrefixedName("carried"))
        carrier = Body(name=PrefixedName("carrier"))
        world.add_kinematic_structure_entity(floor)
        for body in (carried, carrier):
            world.add_kinematic_structure_entity(body)
            world.add_connection(
                Connection6DoF.create_with_dofs(parent=floor, child=body, world=world)
            )
    return world, carried, carrier


def _translations_of(segmentation: SegmindProcess) -> list[TranslationEvent]:
    """
    The translations the process reported so far.
    """
    return [
        event
        for event in segmentation.event_logger.get_events()
        if isinstance(event, TranslationEvent)
    ]


def _set_when_translated(segmentation: SegmindProcess) -> threading.Event:
    """
    :return: An event set once ``segmentation`` reports a translation.
    """
    translated = threading.Event()
    segmentation.event_logger.add_callback(
        TranslationEvent, lambda event: translated.set()
    )
    return translated


# %% what a run asks to be watched


def test_a_process_ticks_every_detector_what_it_asks_for_is_read_from(
    milk_in_the_apartment,
):
    """
    A run asks for what it wants detected; the kinds that is read from come along.
    """
    world, milk, _ = milk_in_the_apartment

    segmentation = SegmindProcess.create_for_bodies(
        world, [milk], detector_types=[PickUpDetector]
    )

    assert Counter(segmentation.detector_names) == Counter(
        detector_type.__name__
        for detector_type in DetectorSelection.of(PickUpDetector).detector_types
    )


# %% watching from a process of its own


def test_what_the_world_shows_as_watching_starts_is_detected(milk_in_the_apartment):
    """
    What holds in the world when watching starts, such as the milk resting on a box,
    is reported too.
    """
    world, milk, _ = milk_in_the_apartment
    rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        rest_x, rest_y, rest_z, reference_frame=milk.parent_connection.parent
    )
    segmentation = SegmindProcess.create_for_bodies(
        world, [milk], detector_types=[SupportDetector]
    )

    with segmentation:
        pass

    [support] = [
        event
        for event in segmentation.event_logger.get_events()
        if isinstance(event, SupportEvent)
    ]
    assert support.tracked_object is milk
    assert support.with_object is world.get_body_by_name("box_2")


def test_a_body_moved_while_watched_from_a_process_is_seen_translating(
    milk_in_the_apartment,
):
    """
    A move the run makes reaches the process, and the event names the run's own body.
    """
    world, milk, _ = milk_in_the_apartment
    rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        rest_x, rest_y, rest_z, reference_frame=milk.parent_connection.parent
    )
    segmentation = SegmindProcess.create_for_bodies(
        world, [milk], detector_types=[TranslationDetector]
    )
    translated = _set_when_translated(segmentation)

    with segmentation:
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            rest_x + MOVED_ALONG_X,
            rest_y,
            rest_z,
            reference_frame=milk.parent_connection.parent,
        )
        seen = translated.wait(DETECTION_TIMEOUT)

    assert seen
    [translation] = _translations_of(segmentation)
    assert translation.tracked_object is milk


def test_a_body_hung_under_another_while_watched_is_seen_moving_with_it(
    carried_and_carrier,
):
    """
    A change of the world model reaches the process, so a body hung under another is
    seen moving with it.
    """
    world, carried, carrier = carried_and_carrier
    segmentation = SegmindProcess.create_for_bodies(
        world, [carried], detector_types=[TranslationDetector]
    )
    translated = _set_when_translated(segmentation)

    with segmentation:
        with world.modify_world():
            world.move_branch_with_fixed_connection(carried, carrier)
        carrier.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=MOVED_ALONG_X, reference_frame=carrier.parent_connection.parent
        )
        seen = translated.wait(DETECTION_TIMEOUT)

    assert seen
    [translation] = _translations_of(segmentation)
    assert translation.tracked_object is carried


def _add_a_large_body(world: World) -> None:
    """
    Add a body to ``world`` made of so many shapes that the change of its model is too
    large for the buffer of a pipe.
    """
    body = Body(name=PrefixedName("large"))
    body.collision = ShapeCollection(
        [
            Box(
                scale=Scale(0.01, 0.01, 0.01),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.01 * index, reference_frame=body
                ),
            )
            for index in range(SHAPES_OF_A_LARGE_BODY)
        ],
        reference_frame=body,
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            Connection6DoF.create_with_dofs(parent=world.root, child=body, world=world)
        )


def test_a_model_change_too_large_for_a_pipe_does_not_hold_up_the_run(
    carried_and_carrier,
):
    """
    Changing the world model is not held up by the process, however large the change.
    """
    world, carried, _ = carried_and_carrier
    segmentation = SegmindProcess.create_for_bodies(
        world, [carried], detector_types=[TranslationDetector]
    )

    with segmentation:
        changing = threading.Thread(target=_add_a_large_body, args=(world,))
        changing.start()
        changing.join(MODEL_CHANGE_TIMEOUT)
        changed_in_time = not changing.is_alive()

    assert changed_in_time


def test_a_process_that_fails_while_watching_fails_the_run(milk_in_the_apartment):
    """
    An error in the process is not lost: the run fails when it stops watching.
    """
    world, milk, _ = milk_in_the_apartment
    segmentation = SegmindProcess.create_for_bodies(
        world, [milk], detector_types=[DetectorFailingOnItsFirstTick]
    )

    with pytest.raises(SegmindProcessFailed):
        with segmentation:
            pass
