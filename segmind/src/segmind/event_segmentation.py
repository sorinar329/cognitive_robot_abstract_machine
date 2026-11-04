"""
SegMind itself: what it watches and what it is asked to detect while a run goes on.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from typing_extensions import Self, Sequence

from segmind.detector_selection import DetectorSelection
from segmind.detectors.atomic_event_detectors_nodes import MotionDetector
from segmind.detectors.base import AbstractDetector, SegmindContext
from segmind.episode_segmenter import EpisodeSegmentation
from segmind.event_logger import EventLogger
from segmind.exceptions import NoSemanticAnnotationToWatch
from segmind.observation_clock import ObservationClock, TimeOfTick
from segmind.statecharts.segmind_statechart import DetectorStatechartBuilder
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

logger = logging.getLogger(__name__)


# %% segmenting a run while it goes on


@dataclass(eq=False)
class RunSegmentation(ABC):
    """
    SegMind segmenting a run into events while it goes on.

    Entering it starts watching, and leaving it stops watching and reports what was
    detected, so a run states what it wants watched and detected, and keeps the rest to
    its plan.
    """

    world: World
    """
    The world the run takes place in.
    """

    @classmethod
    @abstractmethod
    def create_for_bodies(
        cls,
        world: World,
        bodies: Sequence[Body],
        detector_types: Sequence[type[AbstractDetector]] = (),
    ) -> Self:
        """
        SegMind detecting what happens to ``bodies``.

        :param world: The world the run takes place in.
        :param bodies: The bodies the run handles, which the detectors watch.
        :param detector_types: The kinds of detector asked for. Every kind they are read
            from is brought along (see
            :class:`~segmind.detector_selection.DetectorSelection`), so asking for what is
            to be detected is enough. Without any, every kind SegMind defines.
        """

    @classmethod
    def create_for_semantic_annotation_types(
        cls,
        world: World,
        semantic_annotation_types: Sequence[type[SemanticAnnotation]],
        detector_types: Sequence[type[AbstractDetector]] = (),
    ) -> Self:
        """
        SegMind watching the bodies of every semantic annotation of the types a run
        names.

        :param world: The world the run takes place in.
        :param semantic_annotation_types: The types whose annotations in ``world`` have
            their bodies watched.
        :param detector_types: The kinds of detector asked for.
        :raises NoSemanticAnnotationToWatch: When ``world`` holds no annotation of a
            type asked for.
        """
        bodies = []
        for semantic_annotation_type in semantic_annotation_types:
            annotations = world.get_semantic_annotations_by_type(
                semantic_annotation_type
            )
            if not annotations:
                raise NoSemanticAnnotationToWatch(semantic_annotation_type)
            bodies.extend(
                body for annotation in annotations for body in annotation.bodies
            )
        return cls.create_for_bodies(world, list(dict.fromkeys(bodies)), detector_types)

    @property
    @abstractmethod
    def watched_bodies(self) -> list[Body]:
        """
        The bodies the detectors watch, each named once, in the order they are ticked.
        """

    @property
    @abstractmethod
    def detector_names(self) -> list[str]:
        """
        The kinds of detector ticked, each named once, in the order they are ticked.
        """

    @property
    @abstractmethod
    def event_logger(self) -> EventLogger:
        """
        The logger holding every event the detectors detected, naming the bodies of
        :attr:`world`.
        """

    @abstractmethod
    def start(self) -> None:
        """
        Start watching the run.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Stop watching the run, once the detectors have seen the world at rest.
        """

    def report_detected_events(self) -> None:
        """
        Put every event detected on the console, which is where a run is read.

        SegMind reports what it detected at debug level, which nothing shows by default,
        and a run that already has a handler of its own would otherwise show each event
        through both.
        """
        detected_events = logging.getLogger(EventLogger.__module__)
        detected_events.setLevel(logging.DEBUG)
        if not detected_events.handlers:
            detected_events.addHandler(logging.StreamHandler())
        propagated = detected_events.propagate
        detected_events.propagate = False
        self.event_logger.print_events()
        detected_events.propagate = propagated

    def __enter__(self) -> Self:
        """
        Start watching, as :meth:`start` does.
        """
        logger.info("SegMind detectors: %s", ", ".join(self.detector_names))
        self.start()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        """
        Stop watching, as :meth:`stop` does, and report what was detected.
        """
        self.stop()
        self.report_detected_events()


# %% watching on a thread of the run's own process


@dataclass(eq=False)
class Segmind(RunSegmentation):
    """
    SegMind ticking its detectors on a thread of the run's own process.

    A tick holds the world's lock, and is followed by a pause at least as long as the
    tick, so a plan changing the world keeps going while it is watched.
    """

    detectors: list[AbstractDetector]
    """
    The detectors ticked.
    """

    tick_period: float = field(kw_only=True, default=0.5)
    """
    The time, in seconds, from the start of one tick to the start of the next.

    A tick that takes longer than half of it is followed by a pause as long as the tick
    took instead, so watching never holds the world for more than half the run.
    """

    pause_between_ticks: float = field(kw_only=True, default=0.01)
    """
    The least time, in seconds, the world is left to other threads after a tick.
    """

    clock: ObservationClock = field(kw_only=True, default_factory=TimeOfTick)
    """
    Tells when the state of the world each tick reads was observed.
    """

    executor: StatechartExecutor = field(init=False)
    """
    Compiles and ticks the detectors.
    """

    stopping: threading.Event = field(init=False, default_factory=threading.Event)
    """
    Set when the detectors are to stop being ticked.
    """

    thread: threading.Thread | None = field(init=False, default=None)
    """
    The thread the detectors are ticked on, once started.
    """

    def __post_init__(self) -> None:
        """
        Compile the detectors, observed with :attr:`clock`.
        """
        self.executor = StatechartExecutor(
            context=StatechartContext(world=self.world),
            extensions=[EpisodeSegmentation(clock=self.clock)],
        )
        self.executor.compile(
            DetectorStatechartBuilder(self.detectors).build(self.executor.context)
        )

    @classmethod
    def create_for_bodies(
        cls,
        world: World,
        bodies: Sequence[Body],
        detector_types: Sequence[type[AbstractDetector]] = (),
    ) -> Self:
        return cls(
            world=world,
            detectors=DetectorSelection.for_run(detector_types).detectors_for(bodies),
        )

    @property
    def watched_bodies(self) -> list[Body]:
        return list(
            dict.fromkeys(
                detector.tracked_object
                for detector in self.detectors
                if detector.tracked_object is not None
            )
        )

    @property
    def detector_names(self) -> list[str]:
        return list(
            dict.fromkeys(type(detector).__name__ for detector in self.detectors)
        )

    @property
    def event_logger(self) -> EventLogger:
        return self.executor.context.require_extension(SegmindContext).logger

    @property
    def time_to_see_the_world_at_rest(self) -> float:
        """
        How long, in seconds, the detectors need to watch a still world to conclude what
        it ended in: the longest motion window among them.
        """
        return max(
            [
                detector.window_duration
                for detector in self.detectors
                if isinstance(detector, MotionDetector)
            ],
            default=0.0,
        )

    @property
    def is_watching(self) -> bool:
        """
        Whether the detectors are being ticked on their thread.
        """
        return self.thread is not None and self.thread.is_alive()

    def tick(self) -> None:
        """
        Tick the detectors once, while no other thread changes the world.
        """
        with self.world._world_lock:
            self.executor.tick()

    def start(self) -> None:
        self.stopping.clear()
        self.thread = threading.Thread(target=self.tick_until_stopped, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.thread.join()

    def _tick_while_the_world_is_at_rest(self) -> None:
        """
        Keep ticking the detectors every :attr:`tick_period` until they have watched the
        world for :attr:`time_to_see_the_world_at_rest`, with a tick at its end.
        """
        watched_at_rest_from = time.monotonic()
        self.tick()
        while (
            time.monotonic() - watched_at_rest_from < self.time_to_see_the_world_at_rest
        ):
            time.sleep(self.tick_period)
            self.tick()

    def tick_until_stopped(self) -> None:
        """
        Tick the detectors until told to stop, then as long as they need to see the
        world at rest.
        """
        while not self.stopping.is_set():
            started = time.monotonic()
            self.tick()
            held_for = time.monotonic() - started
            self.stopping.wait(
                max(self.tick_period - held_for, held_for, self.pause_between_ticks)
            )
        self._tick_while_the_world_is_at_rest()
