"""
Tests for the world the HSR performs a RoboCasa task in, and for how it plans to pick
the task's object up there.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_digital_twin.adapters.robocasa_dataset.mujoco_compat import (
    robocasa_version_assertions_relaxed,
)

with robocasa_version_assertions_relaxed():
    pytest.importorskip("robocasa", reason="robocasa is not installed")

from semantic_digital_twin.world import World

from experiments.hsr_robocasa.arm_plane import HSRBArmPlane
from experiments.hsr_robocasa.demonstrations.pick_place_counter_to_microwave import (
    PickPlaceCounterToMicrowave,
)

# %% the world the task runs in


@pytest.fixture(scope="module")
def task() -> PickPlaceCounterToMicrowave:
    return PickPlaceCounterToMicrowave(headless=True)


@pytest.fixture(scope="module")
def task_world(task: PickPlaceCounterToMicrowave) -> World:
    return task.build_simulated_world()


def test_the_hsr_keeps_its_base_where_it_stands_while_it_moves_its_arm(
    task_world: World,
):
    robot = PickPlaceCounterToMicrowave.robot_in(task_world)
    assert not robot.mobile_base.full_body_controlled


# %% bringing the arm to the object


def test_the_arm_is_brought_in_joint_space_to_where_the_pick_up_begins(
    task: PickPlaceCounterToMicrowave, task_world: World
):
    steps = task.picking_and_placing(task_world)
    robot = task.robot_in(task_world)
    end_effector = robot.all_arms[0].end_effector
    picking = steps.pick_up.grasp_pose_sequence(
        steps.pick_up.grasp.world_T_grasp, end_effector, steps.pick_up.grasp
    )
    [pre_grasp] = task.in_kitchen_frame(task_world, [picking.pre_grasp])

    robot.root.parent_connection.origin = steps.picking_pose.homogeneous_matrix
    for joint, position in steps.reaching.positions.items():
        task_world.get_connection_by_name(joint).position = position

    tool_frame = end_effector.tool_frame.global_pose.to_np()
    assert (
        np.linalg.norm(tool_frame[:3, 3] - pre_grasp[:3, 3])
        <= HSRBArmPlane().reach_resolution
    )


def test_the_pick_up_and_the_placing_accept_the_tool_frame_within_the_tasks_tolerance(
    task: PickPlaceCounterToMicrowave, task_world: World
):
    steps = task.picking_and_placing(task_world)

    assert steps.pick_up.position_threshold == task.tool_frame_tolerance
    assert steps.place.position_threshold == task.tool_frame_tolerance
