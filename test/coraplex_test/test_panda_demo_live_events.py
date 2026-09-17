"""
The page the panda demo serves beside its run, listing what SegMind detects.

The demo is a script rather than a package, so its module is loaded from the file it
lives in; it is registered under its own name first, because a dataclass reads the
module it is defined in while it is being built.
"""

import importlib.util
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body

from segmind.datastructures.events import (
    PickUpEvent,
    SupportEvent,
    TranslationEvent,
)

LIVE_EVENTS_MODULE_NAME = "panda_demo_live_events"
"""
The name the demo's module is registered under while the tests hold it.
"""

LIVE_EVENTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "coraplex"
    / "demos"
    / "coraplex_panda_demo"
    / "live_events.py"
)
"""
The file the demo's module is loaded from.
"""


def load_live_events():
    """
    :return: The demo's ``live_events`` module.
    """
    specification = importlib.util.spec_from_file_location(
        LIVE_EVENTS_MODULE_NAME, LIVE_EVENTS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[LIVE_EVENTS_MODULE_NAME] = module
    specification.loader.exec_module(module)
    return module


live_events = load_live_events()


@pytest.fixture
def apartment(_simple_apartment_setup):
    """
    A world of this test's own.

    The scene it is copied from is shared by the whole session, and these tests move
    things in it to have something detected.
    """
    return deepcopy(_simple_apartment_setup)


@pytest.fixture
def cubes() -> tuple[Body, Body]:
    """
    Two bodies to name events after, as the demo's cubes do.
    """
    return Body(name=PrefixedName("cube1")), Body(name=PrefixedName("cube0"))


# %% the events shown


def test_an_event_is_shown_by_the_names_of_the_bodies_it_is_about(cubes):
    """
    A row states the event's own kind, bodies and time, so the page needs no world.
    """
    carried, below = cubes
    event = SupportEvent(tracked_object=carried, with_object=below)

    row = live_events.EventRow.of(event)

    assert row.tracked_object == str(carried.name)
    assert row.with_object == str(below.name)
    assert row.event_type == type(event).__name__
    assert row.timestamp == event.timestamp.isoformat()


def test_an_event_about_one_body_is_shown_without_a_second(cubes):
    """
    Picking a cube up involves no other body, and the row says so rather than repeating
    the one it is about.
    """
    carried, _ = cubes

    row = live_events.EventRow.of(PickUpEvent(tracked_object=carried))

    assert row.tracked_object == str(carried.name)
    assert row.with_object is None


def test_a_row_is_keyed_by_the_field_each_value_is(cubes):
    """
    The page reads a row by field name, so the keys are the fields themselves.
    """
    carried, below = cubes
    row = live_events.EventRow.of(
        SupportEvent(tracked_object=carried, with_object=below)
    )

    assert row.to_json() == {
        live_events.FeedField.TRACKED_OBJECT: row.tracked_object,
        live_events.FeedField.EVENT_TYPE: row.event_type,
        live_events.FeedField.WITH_OBJECT: row.with_object,
        live_events.FeedField.TIMESTAMP: row.timestamp,
    }


# %% the feed


def test_a_reader_takes_every_event_once(cubes):
    """
    A reader that joins late still gets the events detected before it, and no event
    twice.
    """
    carried, below = cubes
    feed = live_events.EventFeed()
    detected_before = SupportEvent(tracked_object=carried, with_object=below)
    feed.receive([detected_before])

    reader = feed.subscribe()
    taken_first = reader.take()
    detected_after = PickUpEvent(tracked_object=carried)
    feed.receive([detected_after])

    assert taken_first == [detected_before]
    assert reader.take() == [detected_after]
    assert reader.take() == []


def test_the_feed_keeps_every_event_in_the_order_it_received_them(cubes):
    """
    Anything reading the feed as a whole, such as the page's first answer, sees the run
    in the order it happened.
    """
    carried, below = cubes
    feed = live_events.EventFeed()
    detected = [
        SupportEvent(tracked_object=carried, with_object=below),
        PickUpEvent(tracked_object=carried),
    ]

    for event in detected:
        feed.receive([event])

    assert feed.snapshot() == detected


# %% the page


def test_the_page_shows_the_events_it_was_told_about(cubes):
    """
    Every event the page shows is one the feed holds, stated by its row.
    """
    carried, below = cubes
    page = live_events.LiveEventPage()
    event = SupportEvent(tracked_object=carried, with_object=below)
    page.feed.receive([event])

    shown = page.app.test_client().get(live_events.PageRoute.EVENTS).get_json()

    assert shown == [live_events.EventRow.of(event).to_json()]


def test_the_page_leaves_out_the_events_it_hides(cubes):
    """
    Moving is detected many times over a single carry, so the page shows what those
    moves were concluded to be rather than the moves themselves; the feed keeps both.
    """
    carried, below = cubes
    page = live_events.LiveEventPage()
    hidden = TranslationEvent(tracked_object=carried)
    shown = PickUpEvent(tracked_object=carried)
    page.feed.receive([hidden, shown])

    rows = page.app.test_client().get(live_events.PageRoute.EVENTS).get_json()

    assert isinstance(hidden, page.hidden_event_types)
    assert rows == [live_events.EventRow.of(shown).to_json()]
    assert page.feed.snapshot() == [hidden, shown]


def test_the_page_asks_for_the_events_as_they_come(cubes):
    """
    The page follows the run by subscribing to the stream rather than by asking again.
    """
    page = live_events.LiveEventPage()

    answer = page.app.test_client().get(live_events.PageRoute.PAGE)

    assert answer.status_code == 200
    assert live_events.PageRoute.EVENT_STREAM in answer.get_data(as_text=True)


def test_a_page_watching_no_run_draws_no_statechart():
    """
    The statechart is what a watched run ticks, so a page without one answers that it
    has none rather than drawing an empty one.
    """
    page = live_events.LiveEventPage()

    answer = page.app.test_client().get(live_events.PageRoute.STATECHART)

    assert answer.status_code == 404
    assert live_events.PageRoute.STATECHART not in page.app.test_client().get(
        live_events.PageRoute.PAGE
    ).get_data(as_text=True)


# %% watching a run


def test_a_tick_tells_the_feed_what_it_detected(apartment):
    """
    What the detectors find during a tick reaches the page's feed, named by the bodies
    it is about.
    """
    world = apartment
    carried = world.get_body_by_name("milk.stl")
    below = world.get_body_by_name("box")
    watch = live_events.EventWatch.watching(world)

    carried.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        below.global_pose.x,
        below.global_pose.y,
        below.global_pose.z + 0.56,
        reference_frame=world.root,
    )
    watch.tick()

    supports = [
        event for event in watch.feed.snapshot() if isinstance(event, SupportEvent)
    ]
    assert [(event.tracked_object, event.with_object) for event in supports] == [
        (carried, below)
    ]


def test_the_scene_the_run_starts_from_is_not_shown_as_something_that_happened(
    apartment,
):
    """
    Everything in a world at rest reads as having just been put where it is, and the run
    did none of that, so the feed starts with what the run itself does.
    """
    world = apartment
    standing_on_the_box = world.get_body_by_name("milk.stl")
    box = world.get_body_by_name("box")
    standing_on_the_box.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            box.global_pose.x,
            box.global_pose.y,
            box.global_pose.z + 0.56,
            reference_frame=world.root,
        )
    )
    watch = live_events.EventWatch.watching(world)

    watch.see_the_scene_as_it_stands()

    assert watch.feed.snapshot() == []
    assert any(
        isinstance(event, SupportEvent) for event in watch.detected
    ), "the detectors are expected to have taken the standing scene in"


def test_what_the_run_does_to_its_own_scene_is_kept_off_the_feed(apartment):
    """
    Putting an object back where it started reads as it having been picked up and put
    down, and nothing picked it up, so the page is kept out of it while it happens.
    """
    world = apartment
    carried = world.get_body_by_name("milk.stl")
    below = world.get_body_by_name("box")
    watch = live_events.EventWatch.watching(world)
    watch.see_the_scene_as_it_stands()

    with watch.not_telling_the_feed():
        carried.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            below.global_pose.x,
            below.global_pose.y,
            below.global_pose.z + 0.56,
            reference_frame=world.root,
        )
        watch.tick()

    assert watch.feed.snapshot() == []
    assert any(isinstance(event, SupportEvent) for event in watch.detected)


def test_the_scene_coming_to_rest_is_kept_off_the_feed_too(apartment):
    """
    An object settling where the run put it is concluded to have been placed there a few
    ticks after it stops, so the quiet lasts until the detectors have taken the new
    scene in.
    """
    world = apartment
    watch = live_events.EventWatch.watching(world)

    with watch.not_telling_the_feed():
        watch.tick()
    for _ in range(watch.ticks_to_see_the_world_at_rest):
        watch.tick()

    assert watch.feed.snapshot() == []


def test_the_feed_is_told_again_once_the_scene_has_been_taken_in(apartment):
    """
    Keeping the page out of one change does not keep it out of what the run does next.
    """
    world = apartment
    watch = live_events.EventWatch.watching(world)
    carried = world.get_body_by_name("milk.stl")
    below = world.get_body_by_name("box")

    with watch.not_telling_the_feed():
        watch.tick()
    for _ in range(watch.ticks_to_see_the_world_at_rest):
        watch.tick()
    carried.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        below.global_pose.x,
        below.global_pose.y,
        below.global_pose.z + 0.56,
        reference_frame=world.root,
    )
    watch.tick()

    assert [
        (event.tracked_object, event.with_object)
        for event in watch.feed.snapshot()
        if isinstance(event, SupportEvent)
    ] == [(carried, below)]


def test_a_watch_ticks_the_detectors_it_was_asked_for(apartment):
    """
    The demo names which detectors it wants ticked, and those are the ones in the
    statechart drawn on the page.
    """
    watch = live_events.EventWatch.watching(
        apartment, detector_types=live_events.DETECTORS
    )

    assert [type(detector) for detector in watch.detectors] == list(
        live_events.DETECTORS
    )
    assert watch.statechart is watch.executor.motion_statechart


def test_a_watch_leaves_the_run_the_time_it_does_not_watch_in(apartment):
    """
    A tick holds the world while it reads it, so what it costs decides how long the
    world is left to the run afterwards: the run keeps the share the watch does not
    take.
    """
    watch = live_events.EventWatch.watching(apartment, share_of_the_time_watching=0.2)
    held_for = 0.4

    paused_for = watch.pause_after(held_for)

    assert paused_for == pytest.approx(held_for * 4)
    assert held_for / (held_for + paused_for) == pytest.approx(
        watch.share_of_the_time_watching
    )


def test_a_watch_of_a_world_at_rest_does_not_become_a_busy_loop(apartment):
    """
    A tick that costs nothing would otherwise be followed by no pause at all.
    """
    watch = live_events.EventWatch.watching(apartment)

    assert watch.pause_after(0.0) == watch.pause_between_ticks


def test_a_watch_commands_no_motion(apartment):
    """
    The detectors state no constraints, so ticking them drives nothing: the demo's own
    plan stays the only thing moving the robot.
    """
    watch = live_events.EventWatch.watching(apartment)

    assert watch.executor.qp_controller is None


@pytest.mark.skipif(
    shutil.which("dot") is None, reason="graphviz's dot is what draws the statechart"
)
def test_the_page_draws_the_statechart_the_watch_ticks(apartment):
    """
    The statechart tab shows the detectors as they stand, so it is drawn when asked for
    rather than once at the start.
    """
    watch = live_events.EventWatch.watching(apartment)
    page = live_events.LiveEventPage.watching(watch)

    answer = page.app.test_client().get(live_events.PageRoute.STATECHART)

    assert answer.status_code == 200
    assert answer.mimetype == live_events.MediaType.DRAWING
    assert answer.get_data(as_text=True).lstrip().startswith("<?xml")
