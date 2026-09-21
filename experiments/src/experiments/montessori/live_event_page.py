"""
A page showing what a run's event monitor detects, as it detects it.

Listens to a :class:`~experiments.montessori.event_monitoring.MontessoriEventMonitor`
the way anything else told about its events does, so a run serves the page by handing it
the monitor's listener and nothing about the run itself changes.

Needs flask, which the rest of the experiments do not bring along.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, replace
from enum import StrEnum
from http import HTTPStatus
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, stream_with_context
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.plotters.graphviz import MotionStatechartGraphviz
from typing_extensions import (
    Dict,
    Iterator,
    List,
    Optional,
    Self,
    Sequence,
    Tuple,
    Type,
)
from werkzeug.serving import BaseWSGIServer, make_server

from experiments.montessori.semantics import MontessoriShape
from experiments.montessori.event_monitoring import (
    EventsToldToEach,
    MontessoriEventMonitor,
    ReceivesDetectedEvents,
)

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

REQUEST_LOGGER_NAME = "werkzeug"
"""
The logger the serving library reports every request to.
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

    Told by a monitor directly: it is one of the listeners a monitor hands its events to.
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


# %% the page


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
    The statechart the monitor ticks, drawn as it stands at the moment it is asked for.
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
    The events shown; hand it to a monitor as its listener.
    """

    address: PageAddress = field(default_factory=PageAddress)
    """
    Where the page is served.
    """

    statechart: Optional[MotionStatechart] = None
    """
    The statechart the monitor ticks, drawn on the page; none when the page is shown
    beside no monitor.
    """

    shown_names: Dict[str, str] = field(default_factory=dict)
    """
    What to call a body whose own name says little, keyed by that name: a loose piece is
    a body named after the hole it fits, which reads as the hole having been picked up.
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

    def watch(
        self,
        monitor: MontessoriEventMonitor,
        told_as_well: Sequence[ReceivesDetectedEvents] = (),
    ) -> None:
        """
        Show what ``monitor`` detects from now on, and the statechart it ticks.

        :param monitor: The event monitor watched.
        :param told_as_well: What the monitor was already telling, which keeps being
            told: a run acts on the events it detects, and showing them must not stop
            that.
        """
        self.statechart = monitor.statechart
        self.shown_names = {
            str(piece.root.name): piece.shape_category.value
            for piece in monitor.world.get_semantic_annotations_by_type(MontessoriShape)
        }
        monitor.listener = EventsToldToEach([self.feed, *told_as_well])

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

        The server's own request log is quietened: the page asks for the events
        repeatedly, and a line per request would bury what the run itself reports.
        """
        logging.getLogger(REQUEST_LOGGER_NAME).setLevel(logging.WARNING)
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

    def _row(self, event: DetectionEvent) -> EventRow:
        """
        :param event: The event to show.
        :return: Its row, with every body under the name this page shows it by.
        """
        row = EventRow.of(event)
        return replace(
            row,
            tracked_object=self._shown_name(row.tracked_object),
            with_object=self._shown_name(row.with_object),
        )

    def _shown_name(self, body_name: Optional[str]) -> Optional[str]:
        """
        :param body_name: A body's own name, or None where an event names no body.
        :return: What the page calls it.
        """
        if body_name is None:
            return None
        return self.shown_names.get(body_name, body_name)

    def _events(self) -> Response:
        return jsonify(
            [self._row(event).to_json() for event in self._shown(self.feed.snapshot())]
        )

    def _statechart(self) -> Response:
        """
        Draw the detectors as they stand, so each shows the state it is in.

        ..note:: The run keeps ticking the monitor while this draws, so a node may show
            the state it had a tick earlier.
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
                yield f"data: {json.dumps(self._row(event).to_json())}\n\n"
