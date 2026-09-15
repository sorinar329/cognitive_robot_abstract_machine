"""
Watching a world for SegMind events while it changes: a monitor ticks a set of detectors,
and a tick schedule decides which thread ticks it and when.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from giskardpy.motion_statechart.context import MotionStatechartContext
from semantic_digital_twin.world import World
from typing_extensions import Callable, List, Optional, Protocol

from segmind.datastructures.events import DetectionEvent
from segmind.detector_set import DetectorSet
from segmind.detectors.base import SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor

DEFAULT_TICK_RATE_HZ = 5.0
"""
The default rate a schedule ticks a monitor at.
"""

# %% what a schedule ticks and who hears about it


class TicksDetectors(Protocol):
    """
    Something whose detectors run one cycle at a time.
    """

    def tick(self) -> None:
        """
        Run one detection cycle.
        """


class ReceivesDetectedEvents(Protocol):
    """
    Something a monitor tells what it has just detected.
    """

    def receive(self, events: List[DetectionEvent]) -> None:
        """
        Take what was detected since this was last called, oldest first.

        :param events: The newly detected events.
        """


# %% when a tick is due


@dataclass
class TickSpacing:
    """
    Keeps ticks at least one interval apart, measured from the end of one tick to the
    start of the next.

    A tick that overran its interval is still followed by a full interval, so detection
    never takes over the thread it runs on.
    """

    tick_rate_hz: float = DEFAULT_TICK_RATE_HZ
    """
    The rate ticks are limited to.
    """

    clock: Callable[[], float] = time.monotonic
    """
    Reads the monotonic time the gap between ticks is measured against.
    """

    _last_tick_end: Optional[float] = field(init=False, default=None, repr=False)
    """
    When the last tick ended, or None while none has.
    """

    @property
    def interval(self) -> float:
        """
        The least time between the end of one tick and the start of the next, in seconds.
        """
        return 1.0 / self.tick_rate_hz

    def seconds_until_due(self) -> float:
        """
        :return: How long until the next tick is due; zero if it is due now.
        """
        if self._last_tick_end is None:
            return 0.0
        return max(0.0, self._last_tick_end + self.interval - self.clock())

    def is_due(self) -> bool:
        """
        :return: Whether a tick is due now.
        """
        if self._last_tick_end is None:
            return True
        return self.clock() - self._last_tick_end >= self.interval

    def tick(self, monitor: TicksDetectors) -> None:
        """
        Tick ``monitor`` and note when the tick ended.

        :param monitor: The monitor to tick.
        """
        try:
            monitor.tick()
        finally:
            self._last_tick_end = self.clock()

    def restart(self) -> None:
        """
        Forget the last tick, so the next one is due at once.
        """
        self._last_tick_end = None


# %% which thread ticks


class TickSchedule(ABC):
    """
    Decides which thread ticks a monitor, and when, between :meth:`start` and
    :meth:`stop`.
    """

    @abstractmethod
    def start(self, monitor: TicksDetectors) -> None:
        """
        Start ticking ``monitor``.

        :param monitor: The monitor to tick.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Stop ticking.
        """


@dataclass
class TickedByCaller(TickSchedule):
    """
    Leaves ticking to whoever changes the world, for example after each simulation step:
    patches nothing and starts no thread.
    """

    def start(self, monitor: TicksDetectors) -> None:
        return

    def stop(self) -> None:
        return


@dataclass
class TickedOnOwnThread(TickSchedule):
    """
    Ticks a monitor on a thread of its own, independent of any plan.

    ..warning:: CasADi releases the GIL for the duration of a call and counts its
       expression-node references without atomics, so a tick that builds CasADi objects
       races whatever else in the process evaluates symbolic expressions at the same
       time.
    """

    spacing: TickSpacing = field(default_factory=TickSpacing)
    """
    Keeps ticks apart.
    """

    _thread: Optional[threading.Thread] = field(init=False, default=None, repr=False)
    """
    The thread ticking the monitor, while it runs.
    """

    _stop_requested: threading.Event = field(
        init=False, default_factory=threading.Event, repr=False
    )
    """
    Set by :meth:`stop` to end the thread's loop.
    """

    def start(self, monitor: TicksDetectors) -> None:
        self._stop_requested.clear()
        self.spacing.restart()
        self._thread = threading.Thread(target=self._run, args=(monitor,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop ticking, waiting for the tick in progress to finish.
        """
        self._stop_requested.set()
        if self._thread is None:
            return
        self._thread.join()
        self._thread = None

    def _run(self, monitor: TicksDetectors) -> None:
        """
        Tick ``monitor`` whenever a tick is due, until a stop is requested.

        :param monitor: The monitor to tick.
        """
        while not self._stop_requested.wait(self.spacing.seconds_until_due()):
            self.spacing.tick(monitor)


# %% the monitor


@dataclass
class SegmindMonitor:
    """
    Ticks a set of detectors against a live world, so events are detected while the world
    changes, and tells its listeners what each tick detected.
    """

    world: World
    """
    The world the detectors read.
    """

    detectors: DetectorSet
    """
    The detectors ticked; see :class:`~segmind.detector_set.DetectorSet`.
    """

    schedule: TickSchedule = field(default_factory=TickedByCaller)
    """
    Decides which thread ticks this monitor, and when, between :meth:`start` and
    :meth:`stop`.
    """

    listeners: List[ReceivesDetectedEvents] = field(default_factory=list)
    """
    Told what each tick detected, as it is detected.
    """

    context: MotionStatechartContext = field(init=False)
    """
    The motion statechart context the detectors run against, holding the shared
    :class:`~segmind.detectors.base.SegmindContext`.
    """

    _executor: EpisodeSegmenterExecutor = field(init=False, repr=False)
    """
    Compiles and ticks the detectors' statechart.
    """

    _handed_over_event_count: int = field(init=False, default=0, repr=False)
    """
    How much of :attr:`events` the listeners have been told about.
    """

    _watches_anything: bool = field(init=False, repr=False)
    """
    Whether :attr:`detectors` holds any detector; a statechart without one cannot be
    compiled, and a monitor without one has nothing to tick.
    """

    def __post_init__(self) -> None:
        self.context = MotionStatechartContext(world=self.world)
        self._executor = EpisodeSegmenterExecutor(context=self.context)
        self._watches_anything = bool(self.detectors.detectors)
        if not self._watches_anything:
            return
        self._executor.compile(self.detectors.build_statechart())

    @property
    def events(self) -> List[DetectionEvent]:
        """
        Every event detected so far.
        """
        return self.context.require_extension(SegmindContext).logger.get_events()

    def tick(self) -> None:
        """
        Run one detection cycle against the world as it stands, and tell the listeners
        what it detected.
        """
        if not self._watches_anything:
            return
        self._executor.tick()
        self._hand_over_new_events()

    def start(self) -> None:
        """
        Start watching: read the scene as it is, then leave ticking to :attr:`schedule`.

        The first reading is what later ticks are measured against, so a body found
        elsewhere at a later tick was moved while watched rather than first seen there.
        """
        self._read_geometry_out()
        self.tick()
        self.schedule.start(self)

    def stop(self) -> None:
        """
        Stop watching. Detected events stay readable through :attr:`events`.
        """
        self.schedule.stop()

    def _hand_over_new_events(self) -> None:
        """
        Tell every listener what the last tick added, and nothing else.
        """
        detected = self.events
        new_events = detected[self._handed_over_event_count :]
        if not new_events:
            return
        self._handed_over_event_count = len(detected)
        for listener in self.listeners:
            listener.receive(new_events)

    def _read_geometry_out(self) -> None:
        """
        Read every shape's placement into numbers before watching starts, so that
        one-time cost is not paid inside the first tick of a motion.
        """
        for entity in self.world.kinematic_structure_entities:
            entity.combined_mesh
