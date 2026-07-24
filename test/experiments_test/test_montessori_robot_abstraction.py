"""
Tests that the Montessori demo's robot-handling code generalizes to a robot other than
:class:`~semantic_digital_twin.robots.hsrb.HSRB`, using a self-contained synthetic robot
(see :mod:`.dataset.synthetic_wheeled_arm_robot`) instead of a second real robot's ROS
package, which this environment cannot resolve.
"""

from experiments.montessori.montessori_demo import (
    ARM_ACTUATOR_POSITION_GAIN,
    BASE_JOINT_ARMATURE,
    BASE_JOINT_DAMPING,
    _base_connections_without_hardware_interface,
    _base_degrees_of_freedom_without_hardware_interface,
    _equip_robot_for_physical_simulation,
    _gripper_drive_degrees_of_freedom,
)
from experiments.montessori.world import MontessoriWorld, robot_installed

from .dataset.synthetic_wheeled_arm_robot import SyntheticWheeledArmRobot

EXPECTED_BASE_DOF_NAMES = {"wheel_left_joint", "wheel_right_joint"}
EXPECTED_CONTROLLED_DOF_NAMES = {"arm_joint", "gripper_joint"}


def test_robot_installed_resolves_a_non_hsrb_robots_local_description():
    assert robot_installed(SyntheticWheeledArmRobot)


def test_spawn_robot_adds_a_non_hsrb_robot_to_the_world():
    montessori = MontessoriWorld()
    body_count_before_spawn = len(montessori.world.bodies)

    robot = montessori.spawn_robot(SyntheticWheeledArmRobot)

    assert isinstance(robot, SyntheticWheeledArmRobot)
    assert robot is montessori.robot
    assert len(montessori.world.bodies) > body_count_before_spawn


def test_base_degrees_of_freedom_without_hardware_interface_finds_the_wheel_joints():
    montessori = MontessoriWorld()
    robot = montessori.spawn_robot(SyntheticWheeledArmRobot)

    base_dof_names = {
        dof.name.name
        for dof in _base_degrees_of_freedom_without_hardware_interface(robot)
    }

    assert base_dof_names == EXPECTED_BASE_DOF_NAMES


def test_base_degrees_of_freedom_without_hardware_interface_excludes_controlled_dofs():
    montessori = MontessoriWorld()
    robot = montessori.spawn_robot(SyntheticWheeledArmRobot)

    base_dofs = set(_base_degrees_of_freedom_without_hardware_interface(robot))
    controlled_dof_names = {
        dof.name.name for dof in robot.degrees_of_freedom_with_hardware_interface
    }

    assert base_dofs.isdisjoint(robot.degrees_of_freedom_with_hardware_interface)
    assert controlled_dof_names == EXPECTED_CONTROLLED_DOF_NAMES


def test_equip_robot_for_physical_simulation_generalizes_to_a_non_hsrb_robot():
    montessori = MontessoriWorld()
    robot = montessori.spawn_robot(SyntheticWheeledArmRobot)
    controlled_dofs = set(robot.degrees_of_freedom_with_hardware_interface)
    base_dofs = set(_base_degrees_of_freedom_without_hardware_interface(robot))

    _equip_robot_for_physical_simulation(robot)

    actuated_dofs = {
        dof for actuator in robot._world.actuators for dof in actuator.dofs
    }
    assert actuated_dofs == controlled_dofs | set(
        _gripper_drive_degrees_of_freedom(robot)
    )
    assert actuated_dofs.isdisjoint(base_dofs)
    for actuator in robot._world.actuators:
        [mujoco_actuator] = actuator.simulator_additional_properties
        assert mujoco_actuator.gain_parameters[0] == ARM_ACTUATOR_POSITION_GAIN


def test_equip_robot_for_physical_simulation_stabilizes_a_non_hsrb_robots_wheels():
    montessori = MontessoriWorld()
    robot = montessori.spawn_robot(SyntheticWheeledArmRobot)

    _equip_robot_for_physical_simulation(robot)

    for connection in _base_connections_without_hardware_interface(robot):
        assert connection.dynamics.damping == BASE_JOINT_DAMPING
        assert connection.dynamics.armature == BASE_JOINT_ARMATURE
