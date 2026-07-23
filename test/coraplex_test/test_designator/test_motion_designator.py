from copy import deepcopy

import numpy as np
import pytest

from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    VerticalAlignment,
    Arms,
    MovementType,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot, real_robot
from coraplex.plans.factories import sequential, execute_single
from coraplex.plans.plan_node import MotionNode, ActionNode
from coraplex.robot_plans import MoveMotion, MoveToolCenterPointMotion
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose

try:
    from coraplex.alternative_motion_mappings.hsrb_motion_mapping import *
    from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
        NavigateActionServerTask,
    )

    skip_tests = False
except (ImportError, ModuleNotFoundError, AttributeError):
    skip_tests = True


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_pick_up_motion(immutable_model_world):
    world, view, context = immutable_model_world
    test_world = deepcopy(world)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        view.left_arm.end_effector,
    )
    pick_up = PickUpAction(
        test_world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
    )

    root = sequential(
        children=[
            ActionNode(
                designator=NavigateAction(
                    Pose(
                        Point3.from_iterable([1.7, 1.5, 0]),
                        Quaternion.from_iterable([0, 0, 0, 1]),
                        test_world.root,
                    ),
                    True,
                )
            ),
            MoveTorsoAction(TorsoState.HIGH),
            pick_up,
        ],
        context=Context.from_world(test_world),
    )
    assert pick_up.plan is not None
    with simulated_robot:
        root.perform()

    pick_up_node = root.plan.get_nodes_by_designator_type(PickUpAction)[0]

    motion_nodes = list(
        filter(lambda x: isinstance(x, MotionNode), pick_up_node.descendants)
    )

    assert len(motion_nodes) == 5

    motion_charts = [type(m.designator.motion_chart) for m in motion_nodes]
    assert all(mc is not None for mc in motion_charts)
    assert CartesianPose in motion_charts
    assert JointPositionList in motion_charts


def test_move_motion_chart(immutable_model_world):
    world, view, context = immutable_model_world
    motion = MoveMotion(
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)
    )
    plan = execute_single(
        motion,
        context=context,
    )

    msc = motion.motion_chart

    assert msc
    np.testing.assert_equal(msc.goal_pose.to_position().to_np(), np.array([1, 1, 1, 1]))


def test_move_tool_center_point_motion_uses_tight_threshold(immutable_model_world):
    """
    MoveToolCenterPointMotion drives grasp approaches, so it must not fall
    back to Giskard's loose default CartesianPose/CartesianPosition threshold
    (0.01m): that tolerance is wide enough to let the gripper stop a
    centimeter away from a small object, e.g. missing or off-center grasps.
    """
    world, view, context = immutable_model_world
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    cartesian_motion = MoveToolCenterPointMotion(
        target, Arms.LEFT, movement_type=MovementType.CARTESIAN
    )
    execute_single(cartesian_motion, context=context)
    assert isinstance(cartesian_motion.motion_chart, CartesianPose)
    assert cartesian_motion.motion_chart.threshold == 0.005

    translation_motion = MoveToolCenterPointMotion(
        target, Arms.LEFT, movement_type=MovementType.TRANSLATION
    )
    execute_single(translation_motion, context=context)
    assert translation_motion.motion_chart.threshold == 0.005


def test_move_gripper_motion_tolerates_stall_only_when_closing(immutable_model_world):
    """
    Closing the gripper must tolerate the fingers stalling against a grasped
    object instead of raising MotionDidNotFinish when they never reach their
    nominal fully-closed target. Opening must not: stalling before reaching
    the open target is a real problem worth surfacing, not a contact
    artifact.
    """
    world, view, context = immutable_model_world

    close_motion = MoveGripperMotion(motion=GripperState.CLOSE, gripper=Arms.LEFT)
    execute_single(close_motion, context=context)
    assert close_motion.motion_chart.tolerate_stall is True

    open_motion = MoveGripperMotion(motion=GripperState.OPEN, gripper=Arms.LEFT)
    execute_single(open_motion, context=context)
    assert open_motion.motion_chart.tolerate_stall is False


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_alternative_mapping(hsr_apartment_world):
    world, view, context = hsr_apartment_world
    context.alternative_motion_mappings = [HSRBMoveMotion]
    move_motion = MoveMotion(
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)
    )

    plan = execute_single(move_motion, context=context)

    with real_robot:
        assert move_motion.get_alternative_motion()
        msc = move_motion.motion_chart
        assert NavigateActionServerTask == type(msc)
