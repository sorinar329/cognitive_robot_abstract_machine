"""
The i40lab demo that grasps the T-block with the left OnRobot 2FG7.

The 2FG7 approaches the T-block from above and closes its pads on the vertical sides of
the crossbar. The demo parks the arm, opens the gripper, reaches the crossbar, closes
until the pads stall against it, lifts the T-block clear of the table, and holds it
there.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.motions.gripper import (
    HoldToolCenterPointMotion,
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.robots.ur5e import UR5eMujocoDualGripper
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import ContactParameters

# %% scene and motion settings

SCENE = Path(__file__).parent / "mujocoscene" / "scene.xml"

T_BLOCK_NAME = "T_block"
"""
The free body the demo picks up.
"""

APPROACH_LINEAR_VELOCITY = 0.1
"""
Maximum tool linear speed on the approach to the T-block, in metres per second.
"""

GRASP_LINEAR_VELOCITY = 0.05
"""
Maximum tool linear speed on the final grasp approach, in metres per second.
"""

LIFT_LINEAR_VELOCITY = 0.05
"""
Maximum tool linear speed while lifting the grasped T-block, in metres per second.
"""

GRASP_POSITION_THRESHOLD = 0.001
"""
Tool position tolerance for the final grasp approach, in metres.
"""

GRASP_HEIGHT_OFFSET = 0.02
"""
How far above the T-block origin the tool center point grasps, in metres.
"""

FINGER_VELOCITY = 0.05
"""
Maximum 2FG7 finger speed while closing, in metres per second.
"""

STALL_MINIMUM_TIME = 1.0
"""
How long the closing fingers must stay stalled against the T-block, in seconds.
"""

FINAL_HOLD_DURATION = timedelta(seconds=2)
"""
How long the lifted T-block is held at the end of the plan, in simulation time.
"""

# %% building the world the demo runs in


def build_world() -> tuple[World, UR5eMujocoDualGripper]:
    """
    Parse the i40lab scene and annotate the UR5e and T-block.

    The 2FG7 is the end effector of the left arm view; the right arm view carries the
    3FG15 and is left at its park state.

    :return: The parsed world and the annotated robot.
    """
    world = MJCFParser.from_file(str(SCENE)).parse()
    robot = UR5eMujocoDualGripper.from_branch_in_world(world.root)

    for arm in robot.get_arms():
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)

    block_body = world.get_body_by_name(T_BLOCK_NAME)
    with world.modify_world():
        world.add_semantic_annotations([HasRootBody(root=block_body)])
    ContactParameters.create_for_grasped_object().apply_to([block_body])

    return world, robot


# %% the plan, grasp the T-block and lift it


def find_block_annotation(context: Context) -> HasRootBody:
    """
    The T-block annotation built in :func:`build_world`.

    :param context: The context the actions are built in.
    :return: The annotation around the T-block body.
    """
    [annotation] = [
        annotation
        for annotation in context.world.semantic_annotations
        if isinstance(annotation, HasRootBody)
        and annotation.root.name.name == T_BLOCK_NAME
    ]
    return annotation


def build_motion_plan(context: Context) -> Plan:
    """
    Grasp the T-block with the left 2FG7 and hold it above the table.

    The gripper approaches the crossbar from above, closes its pads on the vertical
    sides, then lifts the T-block clear of the table.

    :param context: The context the actions are built in.
    :return: The plan that grasps and lifts the T-block.
    """
    robot = context.robot
    block = find_block_annotation(context)

    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        robot.left_arm.end_effector,
        rotate_gripper=True,
    )
    target_pose = Pose.from_xyz_rpy(
        0.0, 0.0, GRASP_HEIGHT_OFFSET, 0.0, 0.0, 0.0, reference_frame=block.root
    )
    approach_pose, grasp_pose, lift_pose = grasp.pose_sequence(target_pose, block.root)
    lift_pose = context.world.transform(lift_pose, context.world.root)

    actions = [
        ParkArmsAction(arm=Arms.LEFT),
        MoveGripperMotion(
            motion=GripperState.OPEN,
            gripper=Arms.LEFT,
        ),
        MoveToolCenterPointMotion(
            target=approach_pose,
            arm=Arms.LEFT,
            allow_gripper_collision=True,
            max_linear_velocity=APPROACH_LINEAR_VELOCITY,
        ),
        MoveToolCenterPointMotion(
            target=grasp_pose,
            arm=Arms.LEFT,
            allow_gripper_collision=True,
            max_linear_velocity=GRASP_LINEAR_VELOCITY,
            position_threshold=GRASP_POSITION_THRESHOLD,
        ),
        MoveGripperMotion(
            motion=GripperState.CLOSE,
            gripper=Arms.LEFT,
            allow_gripper_collision=True,
            tolerate_stall=True,
            stall_minimum_time=STALL_MINIMUM_TIME,
            finger_velocity=FINGER_VELOCITY,
        ),
        MoveToolCenterPointMotion(
            target=lift_pose,
            arm=Arms.LEFT,
            allow_gripper_collision=True,
            max_linear_velocity=LIFT_LINEAR_VELOCITY,
        ),
        HoldToolCenterPointMotion(
            pose=lift_pose,
            arm=Arms.LEFT,
            duration=FINAL_HOLD_DURATION,
            allow_gripper_collision=True,
        ),
    ]

    return sequential(actions, context=context).plan


# %% running it


def main() -> None:
    """
    Run the T-block demo against MuJoCo physics.
    """
    import mujoco_demo

    mujoco_demo.run(build_world, build_motion_plan)


if __name__ == "__main__":
    main()
