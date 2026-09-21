"""
Detectors that track different objects, ticking against one shared context.

A run watching several objects at once has one detector per object, and they all read
and write the same :class:`~segmind.detectors.base.SegmindContext`: what a detector
concludes must therefore be about the object it tracks and no other, or each one undoes
what the others noticed.
"""

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import (
    LossOfContactEvent,
    LossOfSupportEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    LossOfContactDetector,
)
from segmind.detectors.base import SegmindContext
from segmind.detectors.spatial_relation_detector_nodes import (
    LossOfSupportDetector,
    SupportDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


def events_of(segmind_context, event_type):
    """
    :param segmind_context: The context the detectors logged into.
    :param event_type: The kind of event wanted.
    :return: Every event of that kind detected so far.
    """
    return [
        event
        for event in segmind_context.logger.get_events()
        if isinstance(event, event_type)
    ]


def standing_on(world, standing, below):
    """
    Put one body on top of another.

    :param world: The world both stand in.
    :param standing: The body to move.
    :param below: The body it is to stand on.
    """
    standing.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        below.global_pose.x,
        below.global_pose.y,
        below.global_pose.z + 0.56,
        reference_frame=world.root,
    )


def test_a_support_of_an_untracked_object_is_not_reported_lost(_simple_apartment_setup):
    """
    A loss detector tracking one object reads the shared context, which holds what every
    other detector noticed; judging those entries by what it checked declares supports
    lost that nothing has lost.
    """
    world = _simple_apartment_setup
    standing = world.get_body_by_name("milk.stl")
    below = world.get_body_by_name("box")
    elsewhere = world.get_body_by_name("box_2")
    standing_on(world, standing, below)
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    segmind_context = executor.context.require_extension(SegmindContext)
    executor.compile(
        SegmindStatechart().build_statechart(
            [
                SupportDetector(tracked_object=standing),
                LossOfSupportDetector(tracked_object=elsewhere),
            ]
        )
    )

    executor.tick()
    executor.tick()

    assert events_of(segmind_context, LossOfSupportEvent) == []
    assert segmind_context.latest_support[standing] == {below}


def test_a_contact_of_an_untracked_object_is_not_reported_lost(_simple_apartment_setup):
    """
    The same holds for touching: what one object is known to touch is not something
    another object's detector may declare over.
    """
    world = _simple_apartment_setup
    touching = world.get_body_by_name("milk.stl")
    touched = world.get_body_by_name("box")
    elsewhere = world.get_body_by_name("box_2")
    standing_on(world, touching, touched)
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    segmind_context = executor.context.require_extension(SegmindContext)
    executor.compile(
        SegmindStatechart().build_statechart(
            [
                ContactDetector(tracked_object=touching),
                LossOfContactDetector(tracked_object=elsewhere),
            ]
        )
    )

    executor.tick()
    executor.tick()

    assert events_of(segmind_context, LossOfContactEvent) == []
    assert touched in segmind_context.latest_contact_bodies[touching]
