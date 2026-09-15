"""
Tests for watching a world while it changes: which thread ticks a monitor and when, what
a monitor reads when watching starts, and what it tells its listeners.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from typing_extensions import List

from segmind.datastructures.events import DetectionEvent, SupportEvent
from segmind.detector_set import DetectorSet
from segmind.detectors.atomic_event_detectors_nodes import ContactDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from segmind.monitor import (
    SegmindMonitor,
    TickedByCaller,
    TickedOnOwnThread,
    TickSpacing,
)
from segmind.scene_parts import SceneParts
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)

TICKS_TO_WATCH = 5
"""
How many ticks a test lets a monitor watch the scene for.
"""

OWN_THREAD_FIRST_TICK_TIMEOUT = 10.0
"""
Seconds a test waits for a monitor on a thread of its own to tick at all.
"""

# %% stand-ins for a monitor and a clock


class TicksItRecords:
    """
    Stands in for a monitor, recording which thread each tick happened on and signalling
    the first one.
    """

    def __init__(self):
        self.tick_threads: List[int] = []
        self.ticked = threading.Event()

    def tick(self) -> None:
        self.tick_threads.append(threading.get_ident())
        self.ticked.set()


class AdvancesOnlyWhenTold:
    """
    A monotonic clock that moves only when a test says it did.
    """

    def __init__(self):
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


class TicksOnAClockItControls:
    """
    Stands in for a monitor whose tick takes real time, without taking any.
    """

    def __init__(self, clock: AdvancesOnlyWhenTold, tick_duration: float):
        self.clock = clock
        self.tick_duration = tick_duration

    def tick(self) -> None:
        self.clock.advance(self.tick_duration)


class TestTicksAreSpacedByTheGapBetweenThem:
    """
    Measuring the rate from a tick's start would let a tick that overran its own
    interval be followed immediately by the next one.
    """

    def test_a_tick_that_overran_its_interval_still_waits_before_the_next(self):
        clock = AdvancesOnlyWhenTold()
        spacing = TickSpacing(tick_rate_hz=10.0, clock=clock)

        spacing.tick(TicksOnAClockItControls(clock, tick_duration=1.0))

        assert not spacing.is_due()

    def test_the_next_tick_comes_once_the_gap_has_passed(self):
        clock = AdvancesOnlyWhenTold()
        spacing = TickSpacing(tick_rate_hz=10.0, clock=clock)
        spacing.tick(TicksOnAClockItControls(clock, tick_duration=1.0))

        clock.advance(spacing.interval)

        assert spacing.is_due()


# %% ticked on a thread of its own, or by the caller


class TestTheMonitorIsTickedOnAThreadOfItsOwn:
    def test_it_ticks_on_another_thread(self):
        monitor = TicksItRecords()
        schedule = TickedOnOwnThread()
        schedule.start(monitor)
        ticked = monitor.ticked.wait(timeout=OWN_THREAD_FIRST_TICK_TIMEOUT)
        schedule.stop()

        assert ticked
        assert threading.get_ident() not in monitor.tick_threads

    def test_stopping_ends_its_thread(self):
        monitor = TicksItRecords()
        threads_before = threading.active_count()
        schedule = TickedOnOwnThread()
        schedule.start(monitor)
        monitor.ticked.wait(timeout=OWN_THREAD_FIRST_TICK_TIMEOUT)

        schedule.stop()

        assert threading.active_count() == threads_before


class TestTheMonitorIsTickedByTheCaller:
    def test_it_never_ticks_on_its_own(self):
        monitor = TicksItRecords()
        threads_before = threading.active_count()
        schedule = TickedByCaller()

        schedule.start(monitor)
        schedule.stop()

        assert monitor.tick_threads == []
        assert threading.active_count() == threads_before


# %% watching a scene


@pytest.fixture
def milk_resting_on_the_table(_simple_apartment_setup):
    """
    The apartment with its milk standing where it rests, put back where it was afterwards.
    """
    world = _simple_apartment_setup
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        -1.7, 0, 0.93, reference_frame=world.root
    )
    yield world, milk
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        -1.7, 0, 1.07, yaw=np.pi, reference_frame=world.root
    )


class RecordsWhatItIsTold:
    """
    Stands in for whatever a run reports its detections to, keeping each handover apart
    so a test can see what a single tick produced.
    """

    def __init__(self):
        self.handovers: List[List[DetectionEvent]] = []

    def receive(self, events: List[DetectionEvent]) -> None:
        self.handovers.append(list(events))


def _monitor_of_the_milk(world, milk, listeners=()) -> SegmindMonitor:
    """
    A monitor detecting what the milk rests on and touches, ticked by the test.

    :param world: The apartment.
    :param milk: The milk body.
    :param listeners: Told what each tick detected.
    """
    detectors = DetectorSet(SceneParts.of_world(world))
    detectors.add(SupportDetector(tracked_object=milk))
    detectors.add(ContactDetector(tracked_object=milk))
    return SegmindMonitor(world=world, detectors=detectors, listeners=list(listeners))


def test_starting_to_watch_reads_the_scene_as_it_is(milk_resting_on_the_table):
    world, milk = milk_resting_on_the_table
    monitor = _monitor_of_the_milk(world, milk)

    monitor.start()
    monitor.stop()

    assert any(
        isinstance(event, SupportEvent) and event.tracked_object is milk
        for event in monitor.events
    )


def test_a_monitor_without_detectors_detects_nothing(milk_resting_on_the_table):
    world, _ = milk_resting_on_the_table
    monitor = SegmindMonitor(
        world=world, detectors=DetectorSet(SceneParts.of_world(world))
    )

    monitor.start()
    for _ in range(TICKS_TO_WATCH):
        monitor.tick()
    monitor.stop()

    assert monitor.events == []


class TestTheListenerHearsWhatEachTickDetected:
    def test_everything_detected_is_handed_over_exactly_once_and_in_order(
        self, milk_resting_on_the_table
    ):
        world, milk = milk_resting_on_the_table
        listener = RecordsWhatItIsTold()
        monitor = _monitor_of_the_milk(world, milk, listeners=[listener])

        for _ in range(TICKS_TO_WATCH):
            monitor.tick()

        handed_over = [event for batch in listener.handovers for event in batch]
        assert handed_over == monitor.events

    def test_the_resting_milk_is_detected_at_all(self, milk_resting_on_the_table):
        """
        Guards the test above: an empty handover list would satisfy it just as well.
        """
        world, milk = milk_resting_on_the_table
        listener = RecordsWhatItIsTold()
        monitor = _monitor_of_the_milk(world, milk, listeners=[listener])

        for _ in range(TICKS_TO_WATCH):
            monitor.tick()

        assert listener.handovers

    def test_a_tick_that_detected_nothing_hands_nothing_over(
        self, milk_resting_on_the_table
    ):
        world, milk = milk_resting_on_the_table
        listener = RecordsWhatItIsTold()
        monitor = _monitor_of_the_milk(world, milk, listeners=[listener])

        for _ in range(TICKS_TO_WATCH):
            monitor.tick()

        assert all(listener.handovers)

    def test_every_listener_hears_the_same_events(self, milk_resting_on_the_table):
        world, milk = milk_resting_on_the_table
        first, second = RecordsWhatItIsTold(), RecordsWhatItIsTold()
        monitor = _monitor_of_the_milk(world, milk, listeners=[first, second])

        for _ in range(TICKS_TO_WATCH):
            monitor.tick()

        assert first.handovers == second.handovers
