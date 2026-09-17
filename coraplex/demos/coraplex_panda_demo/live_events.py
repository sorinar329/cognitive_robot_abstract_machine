"""
The events SegMind detects while the demo stacks, shown on a page beside it.

The detectors tick on a thread of their own against the world the demo is running, so
nothing here changes what the robot does: a tick only reads the world, and the
statechart the detectors tick in carries no constraints, so it commands no motion.

Needs flask, which the demo's other dependencies do not bring along.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from http import HTTPStatus
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, stream_with_context
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.plotters.graphviz import MotionStatechartGraphviz
from typing_extensions import Dict, Iterator, List, Optional, Self, Tuple, Type
from werkzeug.serving import BaseWSGIServer, make_server

from segmind.datastructures.events import (
    ContactEvent,
    DetectionEvent,
    EventWithTrackedObjects,
    LossOfContactEvent,
    RotationEvent,
    StopRotationEvent,
    StopTranslationEvent,
    TranslationEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    MotionDetector,
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
from segmind.statecharts.segmind_statechart import SegmindStatechart
from segmind.utils import PropagatingThread
from semantic_digital_twin.world import World

DETECTORS: Tuple[Type[AbstractDetector], ...] = (
    SupportDetector,
    LossOfSupportDetector,
    TranslationDetector,
    StopTranslationDetector,
    PickUpDetector,
    PlacingDetector,
)
"""
What the demo is watched for: what rests on what, what moves, and the picking up and
placing those two are concluded from.
"""

EVENTS_NOT_SHOWN: Tuple[Type[DetectionEvent], ...] = (
    ContactEvent,
    LossOfContactEvent,
    TranslationEvent,
    StopTranslationEvent,
    RotationEvent,
    StopRotationEvent,
)
"""
The events the page leaves out: touching and moving, which a single carry produces in
numbers, and which the events concluded from them already account for.
"""


# %% the events shown


class FeedField(StrEnum):
    """
    The fields an event is shown by.
    """

    TRACKED_OBJECT = "tracked_object"
    """
    The name of the object the event is about.
    """

    EVENT_TYPE = "event_type"
    """
    The kind of event.
    """

    WITH_OBJECT = "with_object"
    """
    The name of the object it happened with, if any.
    """

    TIMESTAMP = "timestamp"
    """
    When it happened, in ISO 8601.
    """


@dataclass(frozen=True)
class EventRow:
    """
    An event reduced to the fields it is shown by.
    """

    tracked_object: Optional[str]
    """
    See :attr:`FeedField.TRACKED_OBJECT`; None for an event about no object.
    """

    event_type: str
    """
    See :attr:`FeedField.EVENT_TYPE`.
    """

    with_object: Optional[str]
    """
    See :attr:`FeedField.WITH_OBJECT`; None for an event with no other object.
    """

    timestamp: str
    """
    See :attr:`FeedField.TIMESTAMP`.
    """

    @classmethod
    def of(cls, event: DetectionEvent) -> Self:
        """
        :param event: The event to show.
        """
        if not isinstance(event, EventWithTrackedObjects):
            return cls(
                tracked_object=None,
                event_type=type(event).__name__,
                with_object=None,
                timestamp=event.timestamp.isoformat(),
            )
        return cls(
            tracked_object=str(event.tracked_object.name),
            event_type=type(event).__name__,
            with_object=(
                None if event.with_object is None else str(event.with_object.name)
            ),
            timestamp=event.timestamp.isoformat(),
        )

    def to_json(self) -> Dict[FeedField, Optional[str]]:
        """
        :return: The row, keyed by the field each value is.
        """
        return {
            FeedField.TRACKED_OBJECT: self.tracked_object,
            FeedField.EVENT_TYPE: self.event_type,
            FeedField.WITH_OBJECT: self.with_object,
            FeedField.TIMESTAMP: self.timestamp,
        }


@dataclass
class EventFeed:
    """
    Keeps every event it is told about, for any number of readers on any thread.
    """

    _events: List[DetectionEvent] = field(init=False, default_factory=list, repr=False)
    """
    Every event received, oldest first.
    """

    _arrived: threading.Condition = field(
        init=False, default_factory=threading.Condition, repr=False
    )
    """
    Guards :attr:`_events` and wakes readers waiting for more.
    """

    def receive(self, events: List[DetectionEvent]) -> None:
        """
        Keep newly detected events and wake whoever waits for them.

        :param events: The events, oldest first.
        """
        with self._arrived:
            self._events.extend(events)
            self._arrived.notify_all()

    def snapshot(self) -> List[DetectionEvent]:
        """
        :return: Every event received so far, oldest first.
        """
        with self._arrived:
            return list(self._events)

    def subscribe(self) -> Subscription:
        """
        :return: A subscription starting at the first event this feed ever received.
        """
        return Subscription(feed=self)

    def events_after(self, count: int, timeout: float) -> List[DetectionEvent]:
        """
        The events received after the first ``count``, waiting up to ``timeout`` seconds
        for one if there are none yet.

        :param count: How many events the reader has already taken.
        :param timeout: The longest to wait, in seconds.
        :return: The new events, or none if none arrived in time.
        """
        with self._arrived:
            self._arrived.wait_for(lambda: len(self._events) > count, timeout=timeout)
            return self._events[count:]


@dataclass
class Subscription:
    """
    A reader's place in an :class:`EventFeed`: it takes every event exactly once, in the
    order the feed received them, starting with the ones received before it existed.
    """

    feed: EventFeed
    """
    The feed read.
    """

    _taken: int = field(init=False, default=0, repr=False)
    """
    How many of the feed's events this subscription has taken.
    """

    def take(self, timeout: float = 0.0) -> List[DetectionEvent]:
        """
        The events not taken yet, waiting up to ``timeout`` seconds for one.

        :param timeout: The longest to wait, in seconds.
        :return: The new events, or none if none arrived in time.
        """
        events = self.feed.events_after(self._taken, timeout)
        self._taken += len(events)
        return events


# %% watching the run


@dataclass(eq=False)
class EventWatch(PropagatingThread):
    """
    Ticks SegMind's detectors against a world another thread is changing, from when it
    is started until it is stopped, and tells its feed what each tick detected.

    A tick holds the world's lock, so it never reads the world while the simulation or
    a plan is in the middle of changing it. After each tick the world is left to the
    other threads for at least as long as the tick held it, so a run being watched keeps
    going at close to the speed it would run at unwatched.
    """

    world: World
    """
    The world watched.
    """

    executor: EpisodeSegmenterExecutor = field(init=False)
    """
    Compiles and ticks the detectors.
    """

    feed: EventFeed = field(default_factory=EventFeed)
    """
    Told what every tick detected.
    """

    detector_types: Tuple[Type[AbstractDetector], ...] = DETECTORS
    """
    The kinds of detector ticked.
    """

    share_of_the_time_watching: float = 0.15
    """
    The share of the time the detectors may hold the world's lock.

    A tick reads the world while holding that lock, so whatever share of the time it
    takes is a share the run itself does not get: watching without a bound roughly
    halves the speed of the stacking, and at this share it runs at close to the pace it
    has unwatched.
    """

    pause_between_ticks: float = 0.05
    """
    The least time, in seconds, the world is left to the other threads after a tick, so
    that ticking a world nothing is happening in cannot become a busy loop.
    """

    _quiet: bool = field(init=False, default=False, repr=False)
    """
    Whether the feed is being kept out of what is happening; see
    :meth:`not_telling_the_feed`.
    """

    _quiet_ticks_left: int = field(init=False, default=0, repr=False)
    """
    How many more ticks the feed is kept out of, so the detectors can conclude what a
    scene the run has just set up now stands in.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.daemon = True
        self.executor = EpisodeSegmenterExecutor(
            context=MotionStatechartContext(world=self.world)
        )
        self.executor.compile(
            SegmindStatechart().build_statechart(
                [detector_type() for detector_type in self.detector_types]
            )
        )

    @classmethod
    def watching(cls, world: World, **kwargs) -> Self:
        """
        A watch of ``world``, not yet started.

        :param world: The world whose events are detected.
        :param kwargs: Passed on, so a caller can say which detectors to tick or how
            long to leave the world alone between ticks.
        """
        return cls(world=world, **kwargs)

    @property
    def statechart(self) -> MotionStatechart:
        """
        The statechart the detectors tick in.
        """
        return self.executor.motion_statechart

    @property
    def detectors(self) -> List[AbstractDetector]:
        """
        The detectors ticked.
        """
        return self.statechart.nodes

    @property
    def detected(self) -> List[DetectionEvent]:
        """
        Every event detected so far, oldest first.
        """
        return self.executor.context.require_extension(
            SegmindContext
        ).logger.get_events()

    def tick(self) -> None:
        """
        Tick the detectors once and tell the feed what that tick detected, unless the
        feed is being kept out of what is happening.
        """
        detected = self._tick_detectors()
        if self._quiet:
            return
        if self._quiet_ticks_left > 0:
            self._quiet_ticks_left -= 1
            return
        self.feed.receive(detected)

    @contextmanager
    def not_telling_the_feed(self) -> Iterator[None]:
        """
        Keep what is detected while this is held off the feed, and what the detectors
        conclude in the ticks right afterwards, so a change the run makes to its own
        scene is not shown as something that happened in it.

        The scene coming to rest is part of that change: an object settling where it was
        put reads as it having been placed there, and that is concluded a few ticks
        after the object stops. The detectors keep ticking and keep every event either
        way, so what the world then stands in is taken in rather than mistaken for the
        run's own doing the next time something happens.
        """
        self._quiet = True
        try:
            yield
        finally:
            self._quiet = False
            self._quiet_ticks_left = self.ticks_to_see_the_world_at_rest

    def see_the_scene_as_it_stands(self) -> None:
        """
        Tick until the detectors have taken in the scene the run starts from, without
        telling the feed about it.

        A world at rest reads as everything in it having just come to rest where it is,
        so an untouched scene is detected as its objects being supported and placed. The
        run did none of that, so the feed starts with what the run itself does.
        """
        with self.not_telling_the_feed():
            for _ in range(self.ticks_to_see_the_world_at_rest):
                self.tick()

    def _tick_detectors(self) -> List[DetectionEvent]:
        """
        Tick the detectors once while no other thread changes the world.

        :return: What that tick detected, oldest first.
        """
        with self.world._world_lock:
            detected_before = len(self.detected)
            self.executor.tick()
            return self.detected[detected_before:]

    def pause_after(self, held_for: float) -> float:
        """
        How long to leave the world to the other threads after a tick that held its lock
        for ``held_for`` seconds, so the watch keeps to
        :attr:`share_of_the_time_watching`.

        :param held_for: How long the tick took, in seconds.
        """
        watching = self.share_of_the_time_watching
        return max(self.pause_between_ticks, held_for * (1 - watching) / watching)

    def _run(self) -> None:
        self.see_the_scene_as_it_stands()
        while not self.kill_event.is_set():
            started = time.monotonic()
            self.tick()
            self.kill_event.wait(self.pause_after(time.monotonic() - started))
        for _ in range(self.ticks_to_see_the_world_at_rest):
            self.tick()

    @property
    def ticks_to_see_the_world_at_rest(self) -> int:
        """
        How many ticks of a still world the detectors need to conclude what it ended in:
        the longest motion window among them, and at least one.
        """
        return max(
            [
                detector.window_size
                for detector in self.detectors
                if isinstance(detector, MotionDetector)
            ],
            default=1,
        )

    def _join(self, timeout: Optional[float] = None) -> None:
        self.join(timeout)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.stop()


# %% the page


DEFAULT_HOST = "127.0.0.1"
"""
The interface the page is served on unless told otherwise.
"""

DEFAULT_PORT = 5000
"""
The port the page is served on unless told otherwise.
"""

HEARTBEAT_INTERVAL_SECONDS = 15.0
"""
How long the event stream may stay silent before a comment line is sent, so a quiet
stretch does not read as a dropped connection.
"""

PAGE_FILE = Path(__file__).parent / "live_events.html"
"""
The page's template.
"""


class PageRoute(StrEnum):
    """
    The paths the page answers on.
    """

    PAGE = "/"
    """
    The page itself.
    """

    EVENTS = "/events"
    """
    Every event so far, as one JSON list.
    """

    EVENT_STREAM = "/events/stream"
    """
    Every event so far and then each new one, as server-sent events.
    """

    STATECHART = "/statechart.svg"
    """
    The statechart the watch ticks, drawn as it stands at the moment it is asked for.
    """


class PageTab(StrEnum):
    """
    The tabs of the page, each showing one view of the watched run.
    """

    EVENTS = "events"
    """
    The events detected so far, newest first.
    """

    STATECHART = "statechart"
    """
    The statechart the detectors tick in, drawn across the whole page.
    """


class MediaType(StrEnum):
    """
    The media types the page answers with besides JSON and HTML.
    """

    EVENT_STREAM = "text/event-stream"
    """
    A stream of server-sent events.
    """

    DRAWING = "image/svg+xml"
    """
    A drawing a page can scale without losing detail.
    """


class StreamMessage(StrEnum):
    """
    The fixed messages of the event stream.
    """

    HEARTBEAT = ": heartbeat\n\n"
    """
    A comment line keeping a quiet connection open.
    """


@dataclass(frozen=True)
class PageAddress:
    """
    Where the page is served.
    """

    host: str = DEFAULT_HOST
    """
    The interface to bind to.
    """

    port: int = DEFAULT_PORT
    """
    The port to bind to; 0 lets the operating system pick a free one.
    """


@dataclass
class LiveEventPage:
    """
    A page listing the events of an :class:`EventFeed` as they are detected, served on a
    thread of its own between :meth:`start` and :meth:`stop`.
    """

    feed: EventFeed = field(default_factory=EventFeed)
    """
    The events shown.
    """

    address: PageAddress = field(default_factory=PageAddress)
    """
    Where the page is served.
    """

    statechart: Optional[MotionStatechart] = None
    """
    The statechart the detectors tick in, drawn on the page; none when the page watches
    no run.
    """

    hidden_event_types: Tuple[Type[DetectionEvent], ...] = EVENTS_NOT_SHOWN
    """
    The kinds of event the page leaves out; the feed keeps them either way, so anything
    else reading it still sees them.
    """

    app: Flask = field(init=False, repr=False)
    """
    The application answering the page's routes.
    """

    _server: Optional[BaseWSGIServer] = field(init=False, default=None, repr=False)
    """
    The running server, between :meth:`start` and :meth:`stop`.
    """

    _thread: Optional[threading.Thread] = field(init=False, default=None, repr=False)
    """
    The thread serving requests, between :meth:`start` and :meth:`stop`.
    """

    def __post_init__(self) -> None:
        self.app = Flask(__name__)
        self.app.add_url_rule(PageRoute.PAGE, "page", self._page)
        self.app.add_url_rule(PageRoute.EVENTS, "events", self._events)
        self.app.add_url_rule(
            PageRoute.EVENT_STREAM, "event_stream", self._event_stream
        )
        self.app.add_url_rule(PageRoute.STATECHART, "statechart", self._statechart)

    @classmethod
    def watching(cls, watch: EventWatch, address: Optional[PageAddress] = None) -> Self:
        """
        A page showing what ``watch`` detects from now on and before, with the
        statechart it ticks.

        :param watch: The watched run whose events are shown.
        :param address: Where the page is served; the default address when None.
        """
        return cls(
            feed=watch.feed,
            address=address or PageAddress(),
            statechart=watch.statechart,
        )

    @property
    def port(self) -> int:
        """
        The port the running page is bound to.
        """
        return self._server.server_port

    @property
    def url(self) -> str:
        """
        Where the running page can be opened.
        """
        return f"http://{self.address.host}:{self.port}"

    def start(self) -> None:
        """
        Start serving the page on a thread of its own.
        """
        self._server = make_server(
            self.address.host, self.address.port, self.app, threaded=True
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop serving, waiting for the serving loop to end.
        """
        if self._server is None:
            return
        self._server.shutdown()
        self._thread.join()
        self._server.server_close()
        self._server = None
        self._thread = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.stop()

    def _page(self) -> str:
        return render_template_string(
            PAGE_FILE.read_text(),
            fields=FeedField,
            tabs=PageTab,
            event_stream_route=PageRoute.EVENT_STREAM,
            statechart_route=PageRoute.STATECHART if self.statechart else None,
        )

    def _shown(self, events: List[DetectionEvent]) -> List[DetectionEvent]:
        """
        :param events: The events detected.
        :return: The ones the page shows, in the order they were detected.
        """
        return [
            event for event in events if not isinstance(event, self.hidden_event_types)
        ]

    def _events(self) -> Response:
        return jsonify(
            [
                EventRow.of(event).to_json()
                for event in self._shown(self.feed.snapshot())
            ]
        )

    def _statechart(self) -> Response:
        """
        Draw the detectors as they stand, so each shows the state it is in.

        ..note:: The watch keeps ticking on a thread of its own while this draws, so a
            node may show the state it had a tick earlier.
        """
        if self.statechart is None:
            return Response(status=HTTPStatus.NOT_FOUND)
        drawing = MotionStatechartGraphviz(self.statechart).to_dot_graph().create_svg()
        return Response(drawing, mimetype=MediaType.DRAWING)

    def _event_stream(self) -> Response:
        return Response(
            stream_with_context(self._stream_messages()),
            mimetype=MediaType.EVENT_STREAM,
        )

    def _stream_messages(self) -> Iterator[str]:
        """
        Every event so far and then each new one as a server-sent event, with a
        heartbeat whenever none arrives for :data:`HEARTBEAT_INTERVAL_SECONDS`.
        """
        subscription = self.feed.subscribe()
        while True:
            events = subscription.take(timeout=HEARTBEAT_INTERVAL_SECONDS)
            if not events:
                yield StreamMessage.HEARTBEAT
                continue
            for event in self._shown(events):
                yield f"data: {json.dumps(EventRow.of(event).to_json())}\n\n"
