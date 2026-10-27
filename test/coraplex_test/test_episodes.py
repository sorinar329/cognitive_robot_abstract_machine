"""
Tests for recording the timestamped joint states and free body poses of a plan's run
as an episode, and for keeping it in the database with the plan.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import select

from coraplex.datastructures.dataclasses import Context
from coraplex.episodes import Episode, EpisodeRecorder
from coraplex.execution_environment import (
    PhysicallySimulatedRobot,
    kinematically_simulated_robot,
)
from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

from ..pytest_environment import runs_in_continuous_integration

# %% fixtures


@pytest.fixture
def raised_milk(pr2_apartment_context) -> tuple[World, Episode]:
    """
    The PR2 taking hold of the milk and raising its torso, recorded as an episode.

    :return: The world as the plan left it, and the episode of the plan.
    """
    world, robot, context = pr2_apartment_context
    milk = world.get_body_by_name("milk.stl")
    plan = sequential(
        [
            ReAttachNode(body=milk, new_parent=robot.left_arm.end_effector.tool_frame),
            MoveTorsoAction(TorsoState.HIGH),
        ],
        context=context,
    ).plan

    with EpisodeRecorder(plan=plan) as recorder, kinematically_simulated_robot:
        plan.perform()

    return world, recorder.episode


# %% recording


def test_an_episode_is_recorded_while_its_plan_runs(raised_milk):
    _, episode = raised_milk
    root = episode.plan.root

    assert len(episode.frames) > 1
    assert all(
        root.start_time <= frame.time <= root.end_time for frame in episode.frames
    )


def test_frames_while_the_plan_runs_are_a_sample_period_apart(raised_milk):
    """
    Only the frame the plan ends with may follow the one before it sooner.
    """
    _, episode = raised_milk
    times = [frame.time for frame in episode.frames[:-1]]

    assert all(
        later - earlier >= EpisodeRecorder(plan=episode.plan).sample_period
        for earlier, later in zip(times, times[1:])
    )


def test_the_last_frame_holds_the_joint_states_the_plan_left(raised_milk):
    world, episode = raised_milk
    positions = world.state.to_position_dict()

    np.testing.assert_array_equal(
        episode.frames[-1].joint_positions,
        [positions[name] for name in episode.degree_of_freedom_names],
    )


def test_a_free_body_is_recorded_where_it_is_carried_to(raised_milk):
    """
    The milk leaves its own connection once it is held, so only its pose tells where
    the torso raised it to.
    """
    world, episode = raised_milk
    milk = world.get_body_by_name("milk.stl")
    [milk_index] = [
        index for index, name in enumerate(episode.free_body_names) if name == milk.name
    ]

    np.testing.assert_allclose(
        episode.frames[-1].free_body_poses[milk_index], milk.global_pose.to_np()
    )
    assert (
        episode.frames[-1].free_body_poses[milk_index][2, 3]
        > episode.frames[0].free_body_poses[milk_index][2, 3]
    )


# %% the database


def test_an_episode_is_kept_in_the_database_with_its_plan(
    raised_milk, coraplex_testing_session
):
    _, episode = raised_milk
    persisted = to_dao(episode)
    coraplex_testing_session.add(persisted)
    coraplex_testing_session.commit()
    coraplex_testing_session.expire_all()

    restored = (
        coraplex_testing_session.scalars(select(type(persisted))).one().from_dao()
    )

    assert [frame.time for frame in restored.frames] == [
        frame.time for frame in episode.frames
    ]
    np.testing.assert_array_equal(
        restored.frames[-1].joint_positions, episode.frames[-1].joint_positions
    )
    assert restored.free_body_names == episode.free_body_names
    assert restored.plan.root.end_time == episode.plan.root.end_time


# %% a run in a physical simulation


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_an_episode_is_recorded_while_the_physics_moves_the_robot(simulated_hsrb):
    hsrb, simulation = simulated_hsrb
    world = hsrb._world
    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            NavigateAction(
                Pose.from_xyz_rpy(1.0, 0.5, 0.0, yaw=0.5, reference_frame=world.root)
            ),
        ],
        context=Context(world=world, robot=hsrb, evaluate_conditions=False),
    ).plan

    with (
        EpisodeRecorder(plan=plan) as recorder,
        PhysicallySimulatedRobot(simulation=simulation),
    ):
        plan.perform()

    episode = recorder.episode
    positions = world.state.to_position_dict()
    run_time = plan.root.end_time - plan.root.start_time
    assert len(episode.frames) > run_time / recorder.sample_period / 2
    np.testing.assert_array_equal(
        episode.frames[-1].joint_positions,
        [positions[name] for name in episode.degree_of_freedom_names],
    )
