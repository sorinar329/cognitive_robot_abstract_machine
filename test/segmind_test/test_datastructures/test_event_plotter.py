from datetime import datetime, timedelta

import pytest

from segmind.datastructures.event_plotter import EventPlotter
from segmind.datastructures.events import PickUpEvent


def _events(world):
    box1 = world.get_body_by_name("box")
    box2 = world.get_body_by_name("box_2")
    return [
        PickUpEvent(tracked_object=box1, timestamp=datetime(2024, 1, 1, 0, 0, 0)),
        PickUpEvent(tracked_object=box2, timestamp=datetime(2024, 1, 1, 0, 0, 1)),
    ]


def test_prepare_data_gives_each_event_a_visible_duration(_simple_apartment_setup):
    data = EventPlotter()._prepare_data(_events(_simple_apartment_setup))

    for start, end in zip(data['start'], data['end']):
        assert end - start == pytest.approx(0.2)


def test_prepare_data_respects_custom_minimum_bar_duration(_simple_apartment_setup):
    data = EventPlotter(minimum_bar_duration=timedelta(seconds=1.5))._prepare_data(
        _events(_simple_apartment_setup)
    )

    for start, end in zip(data['start'], data['end']):
        assert end - start == pytest.approx(1.5)
