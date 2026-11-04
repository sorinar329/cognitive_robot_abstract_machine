"""
Tests for detected events sent as JSON to another process, where they name the bodies of
that process's world.
"""

from __future__ import annotations

import json

import pytest
from krrood.adapters.json_serializer import from_json, to_json

from segmind.datastructures.events import (
    ContactEvent,
    PickUpEvent,
    SupportEvent,
    TranslationEvent,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)


def _events_between(milk, box) -> list:
    """
    :return: One event of each shape a detector reports, about ``milk`` and ``box``.
    """
    return [
        SupportEvent(tracked_object=milk, with_object=box),
        PickUpEvent(tracked_object=milk),
        TranslationEvent(
            tracked_object=milk,
            world_T_start_pose=milk.global_pose,
            world_T_current_pose=milk.global_pose,
        ),
        ContactEvent(tracked_object=milk, with_object=box),
    ]


@pytest.mark.parametrize("event_index", range(4))
def test_an_event_sent_as_json_names_the_same_bodies_at_the_same_time(
    milk_in_the_apartment, event_index
):
    """
    An event read back from JSON in the same world is about the same bodies and the same
    moment.
    """
    world, milk, box = milk_in_the_apartment
    event = _events_between(milk, box)[event_index]

    received = from_json(
        json.loads(json.dumps(to_json(event))),
        **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
    )

    assert type(received) is type(event)
    assert received.tracked_object is event.tracked_object
    assert received.with_object is event.with_object
    assert received.timestamp == event.timestamp
