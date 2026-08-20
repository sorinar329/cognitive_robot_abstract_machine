from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing_extensions import Any, Callable, List

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    LossOfContactDetector,
    StopTranslationDetector,
    TranslationDetector,
)
from segmind.detectors.base import AbstractDetector, SegmindContext
from segmind.detectors.coarse_event_detector_nodes import (
    PickUpDetector,
    PlacingDetector,
)
from segmind.detectors.spatial_relation_detector_nodes import (
    LossOfSupportDetector,
    SupportDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.event_logger import EventLogger
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


def build_target_detectors(bodies: List[Body]) -> List[AbstractDetector]:
    """
    :param bodies: The bodies to track.
    :return: One contact/support/motion detector per body, explicitly tied to that
        body, plus one shared pick-up and one shared placing detector correlating the
        events those emit.

    Detectors are tied to a body explicitly rather than left to
    :class:`~segmind.detectors.base.AbstractDetector`'s default of auto-discovering
    bodies attached by a ``Connection6DoF``: a grasp reattaches the body it picks up
    through a different connection type, which would otherwise drop it out of
    tracking for as long as it is held.
    """
    detectors: List[AbstractDetector] = []
    for body in bodies:
        detectors.extend(
            [
                ContactDetector(tracked_object=body),
                LossOfContactDetector(tracked_object=body),
                SupportDetector(tracked_object=body),
                LossOfSupportDetector(tracked_object=body),
                TranslationDetector(tracked_object=body),
                StopTranslationDetector(tracked_object=body),
            ]
        )
    detectors.extend([PickUpDetector(), PlacingDetector()])
    return detectors


@dataclass
class LiveSegmindPerception:
    """
    Observes a world's tracked bodies for segmind events while it is being acted on.

    Ticks its own segmind statechart synchronously, once after every real
    :class:`~giskardpy.executor.Executor` tick, rather than polling from a separate
    thread: every world access a tick performs (e.g. forward kinematics) has to happen
    on the thread driving the simulation, since the underlying solver is not
    thread-safe.
    """

    world: World
    """
    The world the tracked bodies live in.
    """

    tracked_objects: List[Body]
    """
    The bodies to detect events for.
    """

    _segmind_context: MotionStatechartContext = field(init=False)
    """
    Context the segmind statechart ticks against.
    """

    _segmind_executor: EpisodeSegmenterExecutor = field(init=False)
    """
    Executor driving the segmind statechart.
    """

    _original_tick: Callable[..., Any] = field(init=False)
    """
    :class:`~giskardpy.executor.Executor`'s real ``tick``, restored on exit.
    """

    def __enter__(self) -> EventLogger:
        """
        Compile the segmind statechart and start observing every real tick.

        :return: The event logger the segmind statechart logs into.
        """
        self._segmind_context = MotionStatechartContext(world=self.world)
        self._segmind_executor = EpisodeSegmenterExecutor(context=self._segmind_context)
        self._segmind_executor.compile(
            SegmindStatechart().build_statechart(
                build_target_detectors(self.tracked_objects)
            )
        )
        self._original_tick = inspect.getattr_static(Executor, "tick")

        def patched_tick(executor: Executor, *args: Any, **kwargs: Any) -> Any:
            return self._observe_tick(self._original_tick, executor, *args, **kwargs)

        Executor.tick = patched_tick
        return self._segmind_context.require_extension(SegmindContext).logger

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Restore the real ``Executor.tick``.
        """
        Executor.tick = self._original_tick

    def _observe_tick(
        self,
        original: Callable[..., Any],
        executor: Executor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Run the real tick, then tick the segmind statechart once in its wake.

        Skips ticking again when ``executor`` is the segmind executor itself, since it
        is patched too and would otherwise re-enter this method for its own tick.

        :param original: The real, unpatched ``Executor.tick``.
        :param executor: The executor whose tick is running.
        """
        result = original(executor, *args, **kwargs)
        if executor is not self._segmind_executor:
            self._segmind_executor.tick()
        return result
