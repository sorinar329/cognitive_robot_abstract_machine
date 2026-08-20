from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import List, Tuple, Type

import pytest

import demo
from coraplex.execution_environment import simulated_robot
from live_segmind_perception import LiveSegmindPerception
from segmind.datastructures.events import DetectionEvent, PickUpEvent, PlacingEvent

# %% ground truth


@dataclass(frozen=True)
class ExpectedEvent:
    """
    One event the demo's plan is expected to produce.
    """

    event_type: Type[DetectionEvent]
    """
    The type of event.
    """

    tracked_object_name: str
    """
    Name of the body the event is about.
    """


def expected_events(
    transport_targets: List[demo.TransportTarget],
) -> List[ExpectedEvent]:
    """
    :param transport_targets: The transports the demo's plan was built from.
    :return: A pick-up immediately followed by a placing for each target, in the
        order the plan carries them out. ``breakfast_cereal.stl``, which the world
        also contains, is deliberately absent: nothing in the plan ever touches it.
    """
    return [
        ExpectedEvent(event_type, target.object_name)
        for target in transport_targets
        for event_type in (PickUpEvent, PlacingEvent)
    ]


# %% the demo's live events match the ground truth


@pytest.mark.slow
def test_live_segmind_events_match_ground_truth():
    world, context, pr2 = demo.setup_demo_world()
    plan, targets = demo.build_plan(world, context, pr2)
    tracked_objects = [world.get_body_by_name(target.object_name) for target in targets]

    with LiveSegmindPerception(world, tracked_objects) as logger:
        with simulated_robot:
            plan.perform()

    actual: List[Tuple[Type[DetectionEvent], str]] = [
        (type(event), event.tracked_object.name.name)
        for event in logger.get_events()
        if isinstance(event, (PickUpEvent, PlacingEvent))
    ]
    expected = [
        (event.event_type, event.tracked_object_name)
        for event in expected_events(targets)
    ]

    assert actual == expected
