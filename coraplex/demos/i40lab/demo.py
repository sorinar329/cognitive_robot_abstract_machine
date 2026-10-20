"""
The i40lab demo with a UR5e + dual gripper in MuJoCo.

The scene and the plan are defined here; the backend carries it out.
"""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum
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
from semantic_digital_twin.robots.onrobot_3fg15 import OnRobot3FG15Joint
from semantic_digital_twin.robots.ur5e import UR5eMujocoDualGripper
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import ContactParameters

# %% which backend carries the plan out


class Backend(StrEnum):
    """
    Where the demo runs.
    """

    MUJOCO = "mujoco"
    """
    Physics, in a MuJoCo viewer.
    """

    RVIZ = "rviz"
    """
    The world model alone, drawn in RViz, with the robot moved kinematically.
    """


BACKEND = Backend.MUJOCO
"""
The backend this run uses. Edit it to run the demo somewhere else.
"""


SCENE = Path(__file__).parent / "mujocoscene" / "scene.xml"


# %% which objects the demo handles


CYLINDER_NAME = "cylinder"
"""
The free body the demo picks up.
"""

PLACE_XY = (1.0, 0.7)
"""
The table position the cylinder is placed at, clear of the other objects.
"""

TABLE_TOP_Z = 0.74
"""
The height of the table tops.
"""

CYLINDER_HALF_HEIGHT = 0.1
"""
Half the height of the cylinder, so its bottom rests on the table top.
"""

TRANSIT_CLEARANCE_Z = 1.15
"""
Tool height the transit flies at: above the tallest obstacle (the cylinder top
at 0.94) with margin for the gripper hardware, and within the arm's 0.85 m
reach above both the park pose and the cylinder.
"""

RAISED_GRASP_OFFSET = 0.040
"""
How far above the cylinder origin the tool point aims.

The tool frame lies 2.55 cm beyond the fingertips, so aiming it at the rim
would leave the whole claw above the top face. This offset instead puts the
pads on the upper wall while keeping the gripper body clear of the top.
"""

APPROACH_CLEARANCE = 0.10
"""
How far above the grasp point the tool starts its descent, clear of the cylinder top.
"""

LIFT_HEIGHT = 0.08
"""
How far the grasp is lifted after closing, to show the cylinder is held.
"""

FINAL_HOLD_DURATION = timedelta(seconds=2)
"""
How long the lifted cylinder is held at the end of the plan, in simulation time.
"""

GRIPPER_FINGER_VELOCITY = 1.0
"""
Maximum angular velocity of the 3FG15 finger joints, in radians per second.
"""


# %% building the world the demo runs in


def set_3fg15_finger_velocity_limits(world: World) -> None:
    """
    Raise the parsed 3FG15 finger velocity limits to the demo's closing speed.

    The MJCF parser gives the finger hinges a default velocity limit that is
    slower than the closing motion needs.

    :param world: The world whose finger joints should be updated.
    """
    for joint in OnRobot3FG15Joint:
        connection = world.get_connection_by_name(joint)
        connection.raw_dof.limits.upper.velocity = GRIPPER_FINGER_VELOCITY
        connection.raw_dof.limits.lower.velocity = -GRIPPER_FINGER_VELOCITY


def build_world() -> tuple[World, UR5eMujocoDualGripper]:
    """
    Parse the i40lab MJCF scene and annotate the UR5e robot.

    The scene already contains the robot (ur5e_with_dual_onrobot.xml include).
    We locate its root body (ur5e_base) and build the semantic annotation
    around it, which equips joints with servos, gravity compensation, PARK
    joint state, velocity caps, and collision rules. Both arm views span the
    same physical joints; ``LEFT`` carries the 2FG7, ``RIGHT`` the 3FG15.
    """
    world = MJCFParser.from_file(str(SCENE)).parse()
    robot = UR5eMujocoDualGripper.from_branch_in_world(world.root)

    # Apply the parked joint state so the arm starts in a known pose
    for arm in robot.get_arms():
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)

    set_3fg15_finger_velocity_limits(world)

    # Annotate the cylinder as a graspable object and give it contact parameters
    # that let the gripper hold it by friction during the lift.
    cylinder_body = world.get_body_by_name(CYLINDER_NAME)
    with world.modify_world():
        world.add_semantic_annotations([HasRootBody(root=cylinder_body)])
    ContactParameters.create_for_grasped_object().apply_to([cylinder_body])

    return world, robot


# %% the plan, pick up the cylinder and place it elsewhere


def find_cylinder_annotation(context: Context) -> HasRootBody:
    """
    The cylinder annotation built in :func:`build_world`.

    :param context: The context the actions are built in.
    :return: The annotation around the cylinder body.
    """
    [annotation] = [
        annotation
        for annotation in context.world.semantic_annotations
        if isinstance(annotation, HasRootBody)
        and annotation.root.name.name == CYLINDER_NAME
    ]
    return annotation


def build_motion_plan(context: Context) -> Plan:
    """
    Grasp the cylinder with the RIGHT 3FG15: park, reach its upper wall, close until
    the fingers stall against it, then lift it clear of the table.

    The close motion tolerates the stall when the fingers meet the cylinder: the
    closed target lies beyond the cylinder's width, so the goal alone would never
    complete.
    """
    robot = context.robot
    cylinder = find_cylinder_annotation(context)

    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        robot.right_arm.end_effector,
        rotate_gripper=True,
    )
    place_pose = Pose.from_xyz_rpy(
        PLACE_XY[0],
        PLACE_XY[1],
        TABLE_TOP_Z + CYLINDER_HALF_HEIGHT,
        0.0,
        0.0,
        0.0,
        reference_frame=context.world.root,
    )

    cylinder_position = cylinder.root.global_pose.to_position()
    orientation = grasp.grasp_orientation()

    grasp_pose = Pose(
        Point3(
            float(cylinder_position.x),
            float(cylinder_position.y),
            float(cylinder_position.z) + RAISED_GRASP_OFFSET,
            reference_frame=context.world.root,
        ),
        orientation,
        reference_frame=context.world.root,
    )
    approach_pose = Pose(
        Point3(
            float(cylinder_position.x),
            float(cylinder_position.y),
            float(cylinder_position.z) + RAISED_GRASP_OFFSET + APPROACH_CLEARANCE,
            reference_frame=context.world.root,
        ),
        orientation,
        reference_frame=context.world.root,
    )
    lift_pose = Pose(
        Point3(
            float(grasp_pose.to_position().x),
            float(grasp_pose.to_position().y),
            float(grasp_pose.to_position().z) + LIFT_HEIGHT,
            reference_frame=context.world.root,
        ),
        grasp_pose.to_quaternion(),
        reference_frame=context.world.root,
    )

    _transport_pose, _placing_pose, _retract_pose = grasp.pose_sequence(
        place_pose, cylinder.root, reverse=True
    )

    actions = [
        ParkArmsAction(arm=Arms.RIGHT),
        MoveGripperMotion(
            motion=GripperState.OPEN,
            gripper=Arms.RIGHT,
        ),
        MoveToolCenterPointMotion(
            target=approach_pose,
            arm=Arms.RIGHT,
            allow_gripper_collision=True,
            max_linear_velocity=0.1,
        ),
        MoveToolCenterPointMotion(
            target=grasp_pose,
            arm=Arms.RIGHT,
            allow_gripper_collision=True,
            max_linear_velocity=0.05,
            position_threshold=0.001,
        ),
        MoveGripperMotion(
            motion=GripperState.CLOSE,
            gripper=Arms.RIGHT,
            allow_gripper_collision=True,
            tolerate_stall=True,
            stall_minimum_time=1.0,
            finger_velocity=GRIPPER_FINGER_VELOCITY,
        ),
        MoveToolCenterPointMotion(
            target=lift_pose,
            arm=Arms.RIGHT,
            allow_gripper_collision=True,
            max_linear_velocity=0.05,
        ),
        HoldToolCenterPointMotion(
            pose=lift_pose,
            arm=Arms.RIGHT,
            duration=FINAL_HOLD_DURATION,
            allow_gripper_collision=True,
        ),
        # MoveToolCenterPointMotion(
        #     target=_transport_pose,
        #     arm=Arms.RIGHT,
        # ),
        # MoveToolCenterPointMotion(
        #     target=_placing_pose,
        #     arm=Arms.RIGHT,
        # ),
        # MoveGripperMotion(
        #     motion=GripperState.OPEN,
        #     gripper=Arms.RIGHT,
        #     tolerate_stall=True,
        # ),
        # MoveToolCenterPointMotion(
        #     target=_retract_pose,
        #     arm=Arms.RIGHT,
        # ),
        # ParkArmsAction(arm=Arms.RIGHT),
    ]

    return sequential(actions, context=context).plan


# %% running it


def main() -> None:
    """
    Run the demo on the backend :data:`BACKEND` names.

    Each backend is imported only once it is chosen, so a run needs nothing
    installed for the backends it does not use.
    """
    if BACKEND is Backend.MUJOCO:
        import mujoco_demo

        mujoco_demo.run(build_world, build_motion_plan)
        return

    if BACKEND is Backend.RVIZ:
        # Not implemented for i40lab yet
        raise NotImplementedError("RViz backend not implemented for i40lab")

    raise ValueError(f"Unknown backend: {BACKEND}")


if __name__ == "__main__":
    main()
