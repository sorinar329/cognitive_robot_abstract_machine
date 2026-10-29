"""
Tests for the Franka Emika Panda, described by MuJoCo Menagerie's model of it.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
    PositionServo,
)

pytestmark = pytest.mark.skipif(
    not Panda.is_description_installed(),
    reason="The description of the Panda is not installed",
)


@pytest.fixture
def mounted_panda() -> Panda:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    return RobotSpecification(Panda).spawn(world)


# %% reading the description


def test_the_robot_hangs_straight_off_the_connection_that_mounts_it():
    """
    MJCF describes every body inside a world body, and a frame left above the robot's
    own root would sit between the robot and what it is mounted on.
    """
    robot = Panda.from_description()

    assert robot.root is robot._world.root


def test_every_actuator_is_a_servo_a_part_declares():
    """
    The description brings an actuator for every joint, which a simulation would build
    next to the servo each part declares, so the joint would be driven twice.
    """
    robot = Panda.from_description()

    assert all(
        isinstance(actuator, PositionServo) for actuator in robot._world.actuators
    )


def test_a_mounted_robot_stands_fixed_where_it_is_mounted(mounted_panda):
    assert isinstance(mounted_panda.root.parent_connection, FixedConnection)


# %% what drives it


def test_every_joint_the_robot_moves_has_a_servo_to_move_it(mounted_panda):
    servoed = {
        degree_of_freedom.id
        for actuator in mounted_panda._world.actuators
        if isinstance(actuator, PositionServo)
        for degree_of_freedom in actuator.dofs
    }
    undriven = [
        connection.name.name
        for connection in mounted_panda.connections
        if isinstance(connection, ActiveConnection1DOF)
        and connection.raw_dof.id not in servoed
    ]

    assert undriven == []


def test_a_simulation_carries_the_weight_of_the_arm(mounted_panda):
    """
    The arm's own controller cancels its weight, so its servos only move the arm and
    what it carries.
    """
    uncompensated = [
        body.name.name
        for body in mounted_panda.arm.bodies
        if body.get_simulator_property_of_type(GravityCompensation) is None
    ]

    assert uncompensated == []


# %% the hand


def test_the_fingers_meet_at_the_same_point_however_wide_the_hand_stands(
    mounted_panda,
):
    assert mounted_panda.arm.end_effector.travel_when_closing == 0.0


def test_opening_the_hand_frees_the_whole_stroke_of_both_fingers(mounted_panda):
    gripper = mounted_panda.arm.end_effector
    closed = gripper.get_joint_state_by_type(GripperState.CLOSE)
    opened = gripper.get_joint_state_by_type(GripperState.OPEN)
    stroke = sum(
        abs(open_value - closed_value)
        for open_value, closed_value in zip(opened.target_values, closed.target_values)
    )

    closed.apply_to(mounted_panda._world)
    when_closed = gripper.fingertips.separation
    opened.apply_to(mounted_panda._world)

    # a micrometre, which is the single-precision the collision mesh is read at
    assert gripper.fingertips.separation == pytest.approx(
        when_closed + stroke, abs=1e-6
    )


def test_the_hand_approaches_along_the_line_the_fingers_close_on(mounted_panda):
    """
    The tool frame, where a grasp is placed, lies halfway between the fingertips, so
    closing the hand on a grasp centres the object between the fingers.
    """
    gripper = mounted_panda.arm.end_effector
    gripper.get_joint_state_by_type(GripperState.OPEN).apply_to(mounted_panda._world)
    world = mounted_panda._world
    thumb = world.compute_forward_kinematics_np(gripper.tool_frame, gripper.thumb.tip)
    finger = world.compute_forward_kinematics_np(gripper.tool_frame, gripper.finger.tip)

    assert list((thumb[:3, 3] + finger[:3, 3]) / 2)[:2] == pytest.approx(
        [0.0, 0.0], abs=1e-9
    )


# %% the physics


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_servos_hold_the_parked_arm(mounted_panda):
    world = mounted_panda._world
    mounted_panda.arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    world.notify_state_change()
    tool_frame = mounted_panda.arm.end_effector.tool_frame
    parked = world.compute_forward_kinematics_np(world.root, tool_frame)[:3, 3]

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=1))
        simulated = list(
            simulation.simulator.get_body_position(
                body_name=tool_frame.name.name
            ).result
        )
    finally:
        simulation.stop_simulation()

    assert simulated == pytest.approx(list(parked), abs=0.01)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_finger_servos_close_the_gripper(mounted_panda):
    world = mounted_panda._world
    gripper = mounted_panda.arm.end_effector
    gripper.get_joint_state_by_type(GripperState.OPEN).apply_to(world)
    world.notify_state_change()
    finger_joint = gripper.thumb.root.parent_connection

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        gripper.get_joint_state_by_type(GripperState.CLOSE).apply_to(world)
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=1))
        reached = simulation.simulator.get_joint_value(finger_joint.name.name).result
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx(
        world.state[finger_joint.raw_dof.id].position, abs=0.002
    )
