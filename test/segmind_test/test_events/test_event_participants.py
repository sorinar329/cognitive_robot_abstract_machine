from datetime import datetime

import pytest
from segmind.datastructures.events import (
    InsertionEvent,
    SupportEvent,
    TranslationEvent,
)
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


def body(name: str) -> Body:
    """
    :param name: The name to give the body.
    :return: A body that only carries a name, enough to stand in as a participant.
    """
    return Body(name=PrefixedName(name))


# %% participants beyond the first two


def test_insertion_event_counts_its_containers_as_participants():
    peg, aperture, container = body("peg"), body("aperture"), body("container")

    event = InsertionEvent(
        tracked_object=peg, with_object=aperture, inserted_into_objects=[container]
    )

    assert event.participants == [peg, aperture, container]


def test_insertion_event_reaches_the_tracker_of_its_container():
    peg, aperture, container = body("peg"), body("aperture"), body("container")
    event = InsertionEvent(
        tracked_object=peg, with_object=aperture, inserted_into_objects=[container]
    )
    factory = ObjectTrackerFactory()

    event.update_object_trackers_with_event(factory)

    assert factory.get_tracker(container).get_event_history() == [event]


# %% participants determine identity


def test_motion_events_of_one_body_at_one_time_are_equal_regardless_of_pose():
    moving_body = body("cup")
    timestamp = datetime(2026, 1, 1)

    here = TranslationEvent(
        tracked_object=moving_body,
        timestamp=timestamp,
        current_pose=Pose.from_xyz_rpy(0, 0, 0),
    )
    there = TranslationEvent(
        tracked_object=moving_body,
        timestamp=timestamp,
        current_pose=Pose.from_xyz_rpy(1, 0, 0),
    )

    assert here == there


def test_motion_events_of_one_body_at_different_times_stay_distinct():
    moving_body = body("cup")

    first = TranslationEvent(tracked_object=moving_body, timestamp=datetime(2026, 1, 1))
    second = TranslationEvent(
        tracked_object=moving_body, timestamp=datetime(2026, 1, 2)
    )

    assert first != second


def test_relation_event_cannot_be_built_without_the_body_it_relates_to():
    with pytest.raises(TypeError):
        SupportEvent(tracked_object=body("cup"))
