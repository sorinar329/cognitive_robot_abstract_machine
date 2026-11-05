"""
Tests for motion being read over a stretch of time, however often the world is ticked.
"""

from __future__ import annotations

import pytest
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix

from segmind.datastructures.events import StopTranslationEvent, TranslationEvent
from segmind.detectors.atomic_event_detectors_nodes import TranslationDetector
from segmind.detectors.base import SegmindContext
from segmind.episode_segmenter import EpisodeSegmentation
from segmind.observation_clock import ClockAdvancingEachTick
from segmind.statecharts.segmind_statechart import DetectorStatechartBuilder
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

SLOW_SPEED = 0.004
"""
Metres per second a test drifts the milk at: too slow to cover the distance threshold
between two ticks a hundredth of a second apart, fast enough across the window.
"""

MOVED_ALONG_X = 0.2
"""
How far a test moves the milk at once.
"""

TICK_RATES = [0.5, 0.01]
"""
Seconds between ticks a test watches the world at: today's default and a hundred hertz.
"""

# %% helpers


def _ticking(
    world: World, detector: TranslationDetector, seconds_per_tick: float
) -> tuple[StatechartExecutor, SegmindContext]:
    """
    :return: An executor ticking ``detector`` on ``world``, each tick observing the
        world ``seconds_per_tick`` after the one before, and the context it logs to.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[
            EpisodeSegmentation(
                clock=ClockAdvancingEachTick(seconds_per_tick=seconds_per_tick)
            )
        ],
    )
    executor.compile(DetectorStatechartBuilder([detector]).build(executor.context))
    return executor, executor.context.require_extension(SegmindContext)


def _put(milk: Body, x: float) -> None:
    """
    Put the milk ``x`` metres along the x-axis of its parent.
    """
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=x, y=-3, z=0.25, reference_frame=milk.parent_connection.parent
    )


def _events_of(segmind_context: SegmindContext, event_type: type) -> list:
    """
    The events of ``event_type`` the detector logged so far.
    """
    return [
        event
        for event in segmind_context.logger.get_events()
        if isinstance(event, event_type)
    ]


# %% the window spans time, not ticks


@pytest.mark.parametrize("seconds_per_tick", TICK_RATES)
def test_a_slow_drift_is_seen_translating_however_often_the_world_is_ticked(
    _simple_apartment_setup, seconds_per_tick
):
    """
    A drift covers the distance threshold across the window whatever the tick rate, so
    it is reported at every rate.
    """
    world = _simple_apartment_setup
    milk = world.get_body_by_name("milk.stl")
    detector = TranslationDetector(tracked_object=milk)
    executor, segmind_context = _ticking(world, detector, seconds_per_tick)
    ticks_across_the_window = round(1.5 * detector.window_duration / seconds_per_tick)

    for tick in range(ticks_across_the_window + 1):
        _put(milk, 1 + SLOW_SPEED * tick * seconds_per_tick)
        executor.tick()

    assert len(_events_of(segmind_context, TranslationEvent)) == 1


@pytest.mark.parametrize("seconds_per_tick", TICK_RATES)
def test_a_stop_is_reported_once_the_body_has_stood_still_for_the_window(
    _simple_apartment_setup, seconds_per_tick
):
    """
    A body that stopped is reported stopped once it has stood still for the window, and
    not before.
    """
    world = _simple_apartment_setup
    milk = world.get_body_by_name("milk.stl")
    detector = TranslationDetector(tracked_object=milk)
    executor, segmind_context = _ticking(world, detector, seconds_per_tick)
    ticks_across_the_window = round(detector.window_duration / seconds_per_tick)
    _put(milk, 1)
    for _ in range(ticks_across_the_window + 1):
        executor.tick()
    _put(milk, 1 + MOVED_ALONG_X)

    for _ in range(ticks_across_the_window):
        executor.tick()
    stops_before_the_window_passed = len(
        _events_of(segmind_context, StopTranslationEvent)
    )
    executor.tick()

    assert len(_events_of(segmind_context, TranslationEvent)) == 1
    assert stops_before_the_window_passed == 0
    assert len(_events_of(segmind_context, StopTranslationEvent)) == 1
