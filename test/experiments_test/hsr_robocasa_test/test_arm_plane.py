"""
Tests for the plane the HSR's arm swings its hand in, which tells where the HSR has to
stand for its hand to reach a pose.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.hsr_robocasa.arm_plane import HSRBArmPlane
from semantic_digital_twin.robots.hsrb import HSRBJoint

# %% a pose the arm reaches


@pytest.fixture(scope="module")
def arm_plane() -> HSRBArmPlane:
    return HSRBArmPlane()


def set_joints(arm_plane: HSRBArmPlane, positions: dict[HSRBJoint, float]) -> None:
    """
    Set the joints of the arm plane's HSR to ``positions``.
    """
    for joint, position in positions.items():
        arm_plane.world.get_connection_by_name(joint).position = position


def tool_frame_pose(arm_plane: HSRBArmPlane) -> np.ndarray:
    """
    :return: Where the tool frame of the arm plane's HSR stands, relative to its root.
    """
    tool_frame = arm_plane.robot.all_arms[0].end_effector.tool_frame
    return arm_plane.world.compute_forward_kinematics_np(
        arm_plane.robot.root, tool_frame
    )


@pytest.fixture(scope="module")
def reached_pose(arm_plane) -> np.ndarray:
    """
    Where the tool frame stands, relative to the HSR's root, with the arm lifted half
    way, flexed forward and the hand tilted down a little.
    """
    set_joints(
        arm_plane,
        {
            HSRBJoint.ARM_LIFT: 0.3,
            HSRBJoint.ARM_FLEX: -1.2,
            HSRBJoint.ARM_ROLL: 0.0,
            HSRBJoint.WRIST_FLEX: -0.6,
            HSRBJoint.WRIST_ROLL: 0.0,
        },
    )
    return tool_frame_pose(arm_plane)


# %% putting the hand at a pose


def test_the_joint_positions_found_put_the_tool_frame_at_the_pose(
    arm_plane, reached_pose
):
    set_joints(arm_plane, arm_plane.joint_positions_for(reached_pose))

    reached = tool_frame_pose(arm_plane)
    assert (
        np.linalg.norm(reached[:3, 3] - reached_pose[:3, 3])
        <= arm_plane.reach_resolution
    )
    assert reached[:3, :3] == pytest.approx(reached_pose[:3, :3], abs=0.02)


def test_the_distance_of_a_reached_pose_is_among_those_the_arm_reaches(
    arm_plane, reached_pose
):
    distances = arm_plane.reachable_distances(reached_pose[2, 3], reached_pose[:3, :3])

    assert min(abs(distance - reached_pose[0, 3]) for distance in distances) <= (
        arm_plane.reach_resolution
    )
