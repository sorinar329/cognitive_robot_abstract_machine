"""
The page showing what a run's event monitor detects while the run goes.
"""

import shutil
from dataclasses import dataclass, field

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body

from experiments.montessori.event_monitoring import EventsToldToEach
from experiments.montessori.semantics import CubeShape
from experiments.montessori.live_event_page import (
    EventFeed,
    EventRow,
    FeedField,
    LiveEventPage,
    MediaType,
    PageRoute,
)
from segmind.datastructures.events import (
    DetectionEvent,
    PickUpEvent,
    SupportEvent,
    TranslationEvent,
)
from typing_extensions import List, Optional


@dataclass
class KeepsWhatItIsTold:
    """
    A listener standing for whatever a run already has the monitor telling.
    """

    told: List[DetectionEvent] = field(default_factory=list)
    """
    Every event received, oldest first.
    """

    def receive(self, events: List[DetectionEvent]) -> None:
        self.told.extend(events)


@dataclass
class HoldsThePiecesNamed:
    """
    A world standing for one whose loose pieces are bodies named after the holes they
    fit.
    """

    pieces: List[object] = field(default_factory=list)
    """
    The pieces it holds, each with a body and the kind of piece it is.
    """

    def get_semantic_annotations_by_type(self, annotation_type) -> List[object]:
        return list(self.pieces)


@dataclass
class TellsOneListener:
    """
    A monitor standing for one that hands its events to a single listener.
    """

    world: HoldsThePiecesNamed = field(default_factory=HoldsThePiecesNamed)
    """
    Where the pieces its detectors watch stand.
    """

    statechart: object = "the statechart the detectors tick in"
    """
    What the page draws.
    """

    listener: Optional[object] = None
    """
    Told what each tick detected.
    """


@pytest.fixture
def pieces() -> tuple[Body, Body]:
    """
    Two bodies to name events after, as the run's pieces are.
    """
    return Body(name=PrefixedName("cube")), Body(name=PrefixedName("table"))


# %% the events shown


def test_an_event_is_shown_by_the_names_of_the_bodies_it_is_about(pieces):
    """
    A row states the event's own kind, bodies and time, so the page needs no world.
    """
    piece, table = pieces
    event = SupportEvent(tracked_object=piece, with_object=table)

    row = EventRow.of(event)

    assert row.to_json() == {
        FeedField.TRACKED_OBJECT: str(piece.name),
        FeedField.EVENT_TYPE: type(event).__name__,
        FeedField.WITH_OBJECT: str(table.name),
        FeedField.TIMESTAMP: event.timestamp.isoformat(),
    }


def test_an_event_about_one_body_is_shown_without_a_second(pieces):
    """
    Picking a piece up involves no other body, and the row says so rather than repeating
    the one it is about.
    """
    piece, _ = pieces

    assert EventRow.of(PickUpEvent(tracked_object=piece)).with_object is None


def test_a_reader_takes_every_event_once(pieces):
    """
    A reader that joins late still gets the events detected before it, and no event
    twice.
    """
    piece, table = pieces
    feed = EventFeed()
    detected_before = SupportEvent(tracked_object=piece, with_object=table)
    feed.receive([detected_before])

    reader = feed.subscribe()
    taken_first = reader.take()
    detected_after = PickUpEvent(tracked_object=piece)
    feed.receive([detected_after])

    assert taken_first == [detected_before]
    assert reader.take() == [detected_after]
    assert reader.take() == []


# %% the page


def test_the_page_shows_the_events_it_was_told_about(pieces):
    """
    Every event the page shows is one the feed holds, stated by its row.
    """
    piece, table = pieces
    page = LiveEventPage()
    event = SupportEvent(tracked_object=piece, with_object=table)
    page.feed.receive([event])

    shown = page.app.test_client().get(PageRoute.EVENTS).get_json()

    assert shown == [EventRow.of(event).to_json()]


def test_the_page_leaves_out_the_events_it_hides(pieces):
    """
    Moving is detected many times over a single carry, so the page shows what those
    moves were concluded to be rather than the moves themselves; the feed keeps both.
    """
    piece, _ = pieces
    page = LiveEventPage()
    hidden = TranslationEvent(tracked_object=piece)
    shown = PickUpEvent(tracked_object=piece)
    page.feed.receive([hidden, shown])

    rows = page.app.test_client().get(PageRoute.EVENTS).get_json()

    assert isinstance(hidden, page.hidden_event_types)
    assert rows == [EventRow.of(shown).to_json()]
    assert page.feed.snapshot() == [hidden, shown]


def test_the_page_asks_for_the_events_as_they_come():
    """
    The page follows the run by subscribing to the stream rather than by asking again.
    """
    answer = LiveEventPage().app.test_client().get(PageRoute.PAGE)

    assert answer.status_code == 200
    assert PageRoute.EVENT_STREAM in answer.get_data(as_text=True)


def test_a_page_beside_no_monitor_draws_no_statechart():
    """
    The statechart is what a watched monitor ticks, so a page without one answers that
    it has none rather than drawing an empty one.
    """
    assert (
        LiveEventPage().app.test_client().get(PageRoute.STATECHART).status_code == 404
    )


# %% watching a monitor


def test_watching_a_monitor_keeps_what_it_was_already_telling_told(pieces):
    """
    A run acts on the events its monitor detects, and showing them must not stop that:
    both the page and what the run had listening are told.
    """
    piece, table = pieces
    monitor = TellsOneListener()
    already_told = KeepsWhatItIsTold()
    monitor.listener = already_told
    page = LiveEventPage()

    page.watch(monitor, told_as_well=[already_told])
    detected = [SupportEvent(tracked_object=piece, with_object=table)]
    monitor.listener.receive(detected)

    assert isinstance(monitor.listener, EventsToldToEach)
    assert page.feed.snapshot() == detected
    assert already_told.told == detected


def test_a_piece_is_shown_as_the_kind_of_piece_it_is():
    """
    A loose piece is a body named after the hole it fits, which reads as the hole having
    been picked up, so the page calls it what kind of piece it is instead.
    """
    piece = CubeShape(root=Body(name=PrefixedName("square_hole_shape")))
    page = LiveEventPage()
    page.watch(TellsOneListener(world=HoldsThePiecesNamed(pieces=[piece])))
    page.feed.receive([PickUpEvent(tracked_object=piece.root)])

    [row] = page.app.test_client().get(PageRoute.EVENTS).get_json()

    assert row[FeedField.TRACKED_OBJECT] == piece.shape_category.value


def test_watching_a_monitor_draws_the_statechart_it_ticks():
    """
    The page draws the statechart of the monitor it was pointed at.
    """
    monitor = TellsOneListener()
    page = LiveEventPage()

    page.watch(monitor)

    assert page.statechart is monitor.statechart


@pytest.mark.skipif(
    shutil.which("dot") is None, reason="graphviz's dot is what draws the statechart"
)
def test_the_page_draws_the_statechart_when_it_is_asked_for(_simple_apartment_setup):
    """
    The statechart tab shows the detectors as they stand, so it is drawn when asked for
    rather than once at the start.
    """
    from experiments.montessori.event_monitoring import MontessoriEventMonitor
    from segmind.detectors.spatial_relation_detector_nodes import SupportDetector

    monitor = MontessoriEventMonitor(
        world=_simple_apartment_setup, detectors=[SupportDetector()]
    )
    page = LiveEventPage()
    page.watch(monitor)

    answer = page.app.test_client().get(PageRoute.STATECHART)

    assert answer.status_code == 200
    assert answer.mimetype == MediaType.DRAWING
