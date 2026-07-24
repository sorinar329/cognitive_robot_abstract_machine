import pytest

from experiments.montessori.montessori_demo import (
    ARM_ACTUATOR_POSITION_GAIN,
    BASE_JOINT_ARMATURE,
    BASE_JOINT_DAMPING,
    BASE_JOINT_DRY_FRICTION,
    GRIPPER_ACTUATOR_POSITION_GAIN,
    GRIPPER_FRICTION,
    MUJOCO_STEP_SIZE,
    RETRY_HORIZONTAL_JITTER,
    _base_connections_without_hardware_interface,
    _base_degrees_of_freedom_without_hardware_interface,
    _connections_driving,
    _enable_robot_table_collision_avoidance,
    _equip_robot_for_physical_simulation,
    _gripper_drive_degrees_of_freedom,
    _physically_simulated_degrees_of_freedom,
    _random_horizontal_jitter,
)
from experiments.montessori.semantics import MontessoriShape
from experiments.montessori.world import MontessoriWorld, robot_installed
from semantic_digital_twin.adapters.multi_sim import MujocoBody, MujocoGeom
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.world_description.connections import Connection6DoF

EXPECTED_BASE_DOF_NAMES = {
    "base_roll_joint",
    "base_l_passive_wheel_x_frame_joint",
    "base_l_passive_wheel_y_frame_joint",
    "base_l_passive_wheel_z_joint",
    "base_r_passive_wheel_x_frame_joint",
    "base_r_passive_wheel_y_frame_joint",
    "base_r_passive_wheel_z_joint",
    "base_l_drive_wheel_joint",
    "base_r_drive_wheel_joint",
}


@pytest.fixture
def montessori_with_robot():
    if not robot_installed(HSRB):
        pytest.skip("hsr_description is not installed")

    montessori = MontessoriWorld()
    montessori.spawn_robot(HSRB)
    return montessori


def test_base_degrees_of_freedom_without_hardware_interface_finds_the_wheel_joints(
    montessori_with_robot,
):
    robot = montessori_with_robot.robot
    base_dof_names = {
        dof.name.name
        for dof in _base_degrees_of_freedom_without_hardware_interface(robot)
    }

    assert base_dof_names == EXPECTED_BASE_DOF_NAMES


def test_base_degrees_of_freedom_without_hardware_interface_excludes_controlled_and_gripper_dofs(
    montessori_with_robot,
):
    robot = montessori_with_robot.robot
    base_dofs = set(_base_degrees_of_freedom_without_hardware_interface(robot))

    assert base_dofs.isdisjoint(robot.degrees_of_freedom_with_hardware_interface)
    assert not any(dof.name.name.startswith("hand_") for dof in base_dofs)


def test_gripper_drive_degrees_of_freedom_excludes_the_passive_spring_joints(
    montessori_with_robot,
):
    """
    Only the joint the gripper actually commands may get an actuator. The HSR's finger
    joints all mimic one motor joint and so share a single dof, while its compliant
    spring joints appear in no declared joint state and must stay free to deflect
    against a grasped object.
    """
    robot = montessori_with_robot.robot

    driven = {dof.name.name for dof in _gripper_drive_degrees_of_freedom(robot)}

    assert driven == {"hand_motor_joint"}


def test_physically_simulated_degrees_of_freedom_cover_the_arm_and_gripper_only(
    montessori_with_robot,
):
    """
    The arm, torso lift, head and the whole gripper (including its passive compliance)
    are driven by physics; the mobile base deliberately stays kinematic, so none of its
    wheel dofs may appear.
    """
    robot = montessori_with_robot.robot
    base_dofs = set(_base_degrees_of_freedom_without_hardware_interface(robot))

    physically_simulated = _physically_simulated_degrees_of_freedom(robot)

    assert set(robot.degrees_of_freedom_with_hardware_interface) <= physically_simulated
    assert {"hand_motor_joint", "hand_l_spring_proximal_joint"} <= {
        dof.name.name for dof in physically_simulated
    }
    assert physically_simulated.isdisjoint(base_dofs)


def test_equip_robot_for_physical_simulation_actuates_every_commanded_joint(
    montessori_with_robot,
):
    """
    Every joint a motion planner commands needs an actuator to physically follow that
    command, and the gripper's passive joints need the absence of one.
    """
    robot = montessori_with_robot.robot
    expected_actuated = set(robot.degrees_of_freedom_with_hardware_interface) | set(
        _gripper_drive_degrees_of_freedom(robot)
    )

    _equip_robot_for_physical_simulation(robot)

    actuated = {dof for actuator in robot._world.actuators for dof in actuator.dofs}
    assert actuated == expected_actuated


def test_equip_robot_for_physical_simulation_uses_the_gripper_gain_for_the_gripper(
    montessori_with_robot,
):
    robot = montessori_with_robot.robot
    gripper_dofs = set(_gripper_drive_degrees_of_freedom(robot))

    _equip_robot_for_physical_simulation(robot)

    for actuator in robot._world.actuators:
        [mujoco_actuator] = actuator.simulator_additional_properties
        expected_gain = (
            GRIPPER_ACTUATOR_POSITION_GAIN
            if actuator.dofs[0] in gripper_dofs
            else ARM_ACTUATOR_POSITION_GAIN
        )
        assert mujoco_actuator.gain_parameters[0] == expected_gain


def test_equip_robot_for_physical_simulation_gravity_compensates_the_driven_links(
    montessori_with_robot,
):
    """
    Without gravity compensation a driven joint settles with a steady-state sag large
    enough to exceed the motion planner's convergence threshold, so a motion merely
    holding the arm never registers as converged.
    """
    robot = montessori_with_robot.robot

    physically_simulated = _equip_robot_for_physical_simulation(robot)

    for connection in _connections_driving(robot, physically_simulated):
        assert any(
            isinstance(prop, MujocoBody) and prop.gravitation_compensation_factor == 1.0
            for prop in connection.child.simulator_additional_properties
        ), f"{connection.child.name.name} is driven but not gravity compensated"


def test_equip_robot_for_physical_simulation_gives_the_fingers_grasp_friction(
    montessori_with_robot,
):
    """
    MuJoCo combines both geoms' friction, so fingers left at the near-zero default
    torsional and rolling friction let a grasped shape spin and roll out of them however
    firmly they squeeze.
    """
    robot = montessori_with_robot.robot
    gripper_dofs = set(_gripper_drive_degrees_of_freedom(robot))

    _equip_robot_for_physical_simulation(robot)

    finger_frictions = [
        prop.friction
        for connection in _connections_driving(robot, gripper_dofs)
        for shape in connection.child.collision
        for prop in shape.simulator_additional_properties
        if isinstance(prop, MujocoGeom)
    ]
    assert finger_frictions, "no finger geometry carries MuJoCo friction settings"
    assert all(friction == list(GRIPPER_FRICTION) for friction in finger_frictions)


def test_equip_robot_for_physical_simulation_leaves_the_base_wheels_unactuated(
    montessori_with_robot,
):
    """
    The base is kinematic, so nothing commands its wheels; a position servo on them only
    fights the floor contacts at whatever position they are teleported to, and diverges
    immediately.
    """
    robot = montessori_with_robot.robot
    base_dofs = set(_base_degrees_of_freedom_without_hardware_interface(robot))

    _equip_robot_for_physical_simulation(robot)

    actuated = {dof for actuator in robot._world.actuators for dof in actuator.dofs}
    assert actuated.isdisjoint(base_dofs)


def test_equip_robot_for_physical_simulation_keeps_the_base_wheels_integrable(
    montessori_with_robot,
):
    """
    MuJoCo integrates joint damping explicitly, so the wheels stay stable only while
    ``damping * step_size / inertia`` is below 1; their own inertia is far too small for
    the damping they need, and their acceleration diverges on the very first step
    without added armature.
    """
    robot = montessori_with_robot.robot
    base_connections = _base_connections_without_hardware_interface(robot)

    _equip_robot_for_physical_simulation(robot)

    for connection in base_connections:
        assert connection.dynamics.damping == BASE_JOINT_DAMPING
        assert connection.dynamics.dry_friction == BASE_JOINT_DRY_FRICTION
        assert connection.dynamics.armature == BASE_JOINT_ARMATURE
        assert connection.dynamics.damping * MUJOCO_STEP_SIZE < (
            connection.dynamics.armature
        )


def test_random_horizontal_jitter_stays_within_the_configured_bound():
    jitter = _random_horizontal_jitter()

    assert abs(float(jitter.x)) <= RETRY_HORIZONTAL_JITTER
    assert abs(float(jitter.y)) <= RETRY_HORIZONTAL_JITTER
    assert float(jitter.z) == 0.0


def test_random_horizontal_jitter_varies_between_calls():
    jitters = {
        (float(jitter.x), float(jitter.y))
        for jitter in (_random_horizontal_jitter() for _ in range(20))
    }

    assert len(jitters) > 1


def test_every_loose_shape_is_spawned_as_a_free_body():
    """
    A shape connected rigidly to the world root is welded in the simulator and cannot be
    picked up, dropped, or fall through a hole at all.
    """
    montessori = MontessoriWorld()
    shapes = list(montessori.world.get_semantic_annotations_by_type(MontessoriShape))

    assert shapes
    assert all(
        isinstance(shape.root.parent_connection, Connection6DoF) for shape in shapes
    )


def test_every_loose_shape_is_spawned_resting_on_the_table():
    """
    A free body whose pose is not baked into its own dof values starts at the world
    origin in the simulator regardless of where it was spawned, and drops straight to
    the floor.
    """
    montessori = MontessoriWorld()
    montessori.world.update_forward_kinematics()

    for shape in montessori.world.get_semantic_annotations_by_type(MontessoriShape):
        position = shape.root.global_transform.to_position()
        assert float(position.z) > 0.5, (
            f"{shape.name.name} was spawned at z={float(position.z)}, not resting on "
            "the table: its pose was not baked into the free joint's dof values"
        )


def test_enable_robot_table_collision_avoidance_checks_the_robot_against_the_table(
    montessori_with_robot,
):
    montessori = montessori_with_robot
    [table] = montessori.world.get_semantic_annotations_by_type(Table)

    _enable_robot_table_collision_avoidance(montessori)
    montessori.world.collision_manager.update_collision_matrix()

    checked_bodies = {
        body
        for check in montessori.world.collision_manager.collision_matrix.collision_checks
        for body in (check.body_a, check.body_b)
    }
    assert set(table.bodies_with_collision) <= checked_bodies
    assert set(montessori.robot.bodies_with_collision) & checked_bodies


def test_enable_robot_table_collision_avoidance_does_not_check_the_board(
    montessori_with_robot,
):
    """
    Only the table is registered, not the shape-sorting board: checking the robot
    against the board's ~40-50-piece CoACD collision decomposition too overloads
    Giskard's QP solver for the tight-clearance pickup motion (a convergence timeout,
    not a detected collision).
    """
    montessori = montessori_with_robot

    _enable_robot_table_collision_avoidance(montessori)
    montessori.world.collision_manager.update_collision_matrix()

    checked_bodies = {
        body
        for check in montessori.world.collision_manager.collision_matrix.collision_checks
        for body in (check.body_a, check.body_b)
    }
    assert set(montessori.board.bodies_with_collision).isdisjoint(checked_bodies)
