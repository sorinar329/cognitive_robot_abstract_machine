"""
Tests for SegMind annotating a recorded episode on demand: replaying its frames in the
world its plan started in and detecting events there.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from coraplex.datastructures.dataclasses import Context
from coraplex.episodes import Episode, EpisodeRecorder
from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import ActionNode
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from krrood.ormatic.data_access_objects.helper import to_dao
from segmind.datastructures.events import LossOfSupportEvent, SupportEvent
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from segmind.episode_annotation import EpisodeAnnotator
from segmind.exceptions import EpisodeWithoutFrames
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2

from ..conftest import SAMPLING_SEED
from ..coraplex_test.world_snapshot import WorldSnapshot

# %% fixtures


@pytest.fixture
def lifted_cereal(pr2_apartment_world) -> Episode:
    """
    The PR2 taking hold of the breakfast cereal and raising its torso, which lifts the
    cereal off the island countertop, recorded as an episode.
    """
    world = pr2_apartment_world
    snapshot = WorldSnapshot.capture(world)
    [robot] = world.get_semantic_annotations_by_type(PR2)
    plan = sequential(
        [
            ReAttachNode(
                body=world.get_body_by_name("breakfast_cereal.stl"),
                new_parent=robot.left_arm.end_effector.tool_frame,
            ),
            MoveTorsoAction(TorsoState.HIGH),
        ],
        context=Context(world, robot, sampling_seed=SAMPLING_SEED),
    ).plan
    with EpisodeRecorder(plan=plan) as recorder, kinematically_simulated_robot:
        plan.perform()
    yield recorder.episode
    snapshot.restore()


def raising_of(episode: Episode) -> ActionNode:
    """
    :return: The node of the episode's plan that raised the torso.
    """
    [raising] = [
        node
        for node in episode.plan.all_nodes
        if isinstance(node, ActionNode) and isinstance(node.action, MoveTorsoAction)
    ]
    return raising


# %% annotating


def test_an_episode_is_annotated_with_what_segmind_detects_in_its_replay(lifted_cereal):
    annotation = EpisodeAnnotator(detector_types=[SupportDetector]).annotate(
        lifted_cereal
    )

    cereal = lifted_cereal.plan.initial_world.get_body_by_name("breakfast_cereal.stl")
    assert [
        type(event) for event in annotation.events if event.tracked_object is cereal
    ] == [SupportEvent, LossOfSupportEvent]


def test_the_events_of_an_annotation_carry_the_moments_the_episode_recorded(
    lifted_cereal,
):
    annotation = EpisodeAnnotator(detector_types=[SupportDetector]).annotate(
        lifted_cereal
    )

    cereal = lifted_cereal.plan.initial_world.get_body_by_name("breakfast_cereal.stl")
    support, loss = [
        event for event in annotation.events if event.tracked_object is cereal
    ]
    raising = raising_of(lifted_cereal)
    assert support.timestamp == lifted_cereal.frames[0].time
    assert raising.start_time <= loss.timestamp <= lifted_cereal.plan.root.end_time


def test_an_annotation_names_the_detectors_it_was_made_with(lifted_cereal):
    annotation = EpisodeAnnotator(detector_types=[SupportDetector]).annotate(
        lifted_cereal
    )

    assert annotation.detector_names == [SupportDetector.__name__]


def test_an_episode_without_frames_cannot_be_annotated(lifted_cereal):
    lifted_cereal.frames.clear()

    with pytest.raises(EpisodeWithoutFrames) as raised:
        EpisodeAnnotator(detector_types=[SupportDetector]).annotate(lifted_cereal)

    assert raised.value.episode is lifted_cereal


# %% the database


def test_an_annotation_is_kept_in_the_database_with_its_episode(
    lifted_cereal, segmind_testing_session
):
    annotation = EpisodeAnnotator(detector_types=[SupportDetector]).annotate(
        lifted_cereal
    )
    persisted = to_dao(annotation)
    segmind_testing_session.add(persisted)
    segmind_testing_session.commit()
    segmind_testing_session.expire_all()

    restored = segmind_testing_session.scalars(select(type(persisted))).one().from_dao()

    assert [
        (type(event), event.tracked_object.name, event.timestamp)
        for event in restored.events
    ] == [
        (type(event), event.tracked_object.name, event.timestamp)
        for event in annotation.events
    ]
    assert [frame.time for frame in restored.episode.frames] == [
        frame.time for frame in lifted_cereal.frames
    ]
