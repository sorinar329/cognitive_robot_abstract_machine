"""
Tests for :mod:`experiments.tracy_experiments.real_time_simulation`: a follower world's
joints take the simulated positions after every advance, observers are told how far the
simulation has come, and an unpaced simulation does not wait for the wall clock.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest
from typing_extensions import List

from experiments.tracy_experiments.equipment import (
    equip_arms_with_servos,
    joint_state_of_type,
    parse_tracy,
    tracy_table_mount_position,
)
from experiments.montessori.world import mount_stationary_robot
from experiments.tracy_experiments.real_time_simulation import (
    RealTimeSimulation,
    SimulationObserver,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSynchronizer
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

ADVANCE = 0.2
"""
Simulated seconds each test advances by.
"""


@dataclass
class _ObserverKeepingTheTimes(SimulationObserver):
    """
    Keeps every simulated time it was told.
    """

    told: List[float] = field(default_factory=list)

    def simulation_advanced(self, simulated_time: float) -> None:
        self.told.append(simulated_time)


def _mounted_tracy() -> tuple[World, Tracy]:
    """
    :return: A world with Tracy bolted to its root, and the robot.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(
            Body(name=PrefixedName(name="root", prefix="world"))
        )
    tracy = parse_tracy()
    mount_position, _ = tracy_table_mount_position(tracy, x=0.0, y=0.0)
    robot = mount_stationary_robot(world, Tracy, tracy, mount_position)
    return world, robot


def _first_arm_joint_name(robot: Tracy) -> PrefixedName:
    """
    :return: The name of the left arm's first joint.
    """
    return robot.left_arm.active_connections[0].raw_dof.name


def _position(world: World, name: PrefixedName) -> float:
    """
    :return: A joint's position in a world, by name.
    """
    return world.state[world.get_degree_of_freedom_by_name(name).id].position


def test_a_followers_joints_take_the_simulated_positions_after_an_advance():
    reality, robot = _mounted_tracy()
    actuators = equip_arms_with_servos(reality, robot)
    joint_state_of_type(robot.left_arm, StaticJointState.PARK).apply_to(reality)
    reality.notify_state_change()
    belief = parse_tracy()
    joint = _first_arm_joint_name(robot)
    assert _position(belief, joint) != _position(reality, joint)

    with RealTimeSimulation(
        world=reality, headless=True, paced_to_the_wall_clock=False, followers=[belief]
    ) as simulation:
        parked = joint_state_of_type(robot.left_arm, StaticJointState.PARK)
        for connection, target in zip(parked.connections, parked.target_values):
            simulation.command(actuators[connection.raw_dof.name.name], target)
        simulation.advance(ADVANCE)

    for connection in robot.left_arm.active_connections:
        name = connection.raw_dof.name
        assert _position(belief, name) == _position(reality, name)


def test_an_observer_is_told_the_simulated_time_after_every_advance():
    reality, _ = _mounted_tracy()
    observer = _ObserverKeepingTheTimes()

    with RealTimeSimulation(
        world=reality,
        headless=True,
        paced_to_the_wall_clock=False,
        observers=[observer],
    ) as simulation:
        simulation.advance(ADVANCE)
        simulation.advance(ADVANCE)

    assert observer.told == pytest.approx([ADVANCE, 2 * ADVANCE])


PLANE_SCALE = Scale(2.0, 2.0, 0.1)
"""
Size of the ground plane a free box is dropped onto.
"""

BOX_SCALE = Scale(0.1, 0.1, 0.1)
"""
Size of the free box dropped onto the plane.
"""

BOX_RELEASE_HEIGHT = 0.5
"""
Height a free box is released from above the plane it falls onto, in metres.
"""

BOX_FALL_ADVANCE = 0.5
"""
Simulated seconds advanced to let a released box fall and come to rest, in metres.
"""

BOX_RESTING_HEIGHT = PLANE_SCALE.z / 2 + BOX_SCALE.z / 2
"""
Height the box's own origin settles at once it has fallen onto the plane and stopped,
in metres: the plane's own top surface, plus half the box's height.
"""


def _free_box_above_a_plane() -> tuple[World, Body, Connection6DoF]:
    """
    :return: A world holding a fixed ground plane and a box on a
        :class:`Connection6DoF`, released :data:`BOX_RELEASE_HEIGHT` above the plane.
    """
    world = World()
    root = Body(name=PrefixedName(name="root", prefix="world"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)

        plane = Body(name=PrefixedName(name="plane", prefix="world"))
        plane.collision = ShapeCollection(
            [Box(scale=PLANE_SCALE, color=Color.GREY())], reference_frame=plane
        )
        world.add_connection(FixedConnection(parent=root, child=plane))

        box = Body(name=PrefixedName(name="box", prefix="world"))
        box.collision = ShapeCollection(
            [Box(scale=BOX_SCALE, color=Color.RED())], reference_frame=box
        )
        connection = Connection6DoF.create_with_dofs(
            world=world, parent=root, child=box
        )
        world.add_connection(connection)
    connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=BOX_RELEASE_HEIGHT, reference_frame=root
    )
    world.notify_state_change()
    return world, box, connection


def test_a_physically_simulated_dof_falls_under_gravity_instead_of_staying_pinned():
    """
    A DOF the world never writes to during the advance has to be listed in
    ``physically_simulated_dofs``, or the simulation keeps snapping it back to the
    world's own (unmoving) belief of its position every step instead of letting gravity
    move it -- exactly the situation a body has the instant it is released, before
    anything reads the physics result back into the world.

    Run unthrottled and unpaced, as an unattended settle check needs to: with the
    default (throttled) sync rate, an advance this fast in wall-clock time would read
    the pose back too rarely to see the fall at all (see
    :attr:`~experiments.tracy_experiments.real_time_simulation.RealTimeSimulation.
    sync_rate_hz`).
    """
    world, box, connection = _free_box_above_a_plane()

    with RealTimeSimulation(
        world=world,
        headless=True,
        paced_to_the_wall_clock=False,
        physically_simulated_dofs=set(connection.passive_dofs),
        sync_rate_hz=MujocoSynchronizer.UNTHROTTLED_SYNC_RATE_HZ,
    ) as simulation:
        simulation.advance(BOX_FALL_ADVANCE)

    world.update_forward_kinematics()
    settled_z = float(box.global_transform.to_np()[2, 3])
    assert settled_z == pytest.approx(BOX_RESTING_HEIGHT, abs=1e-3)


def test_an_unpaced_simulation_does_not_wait_for_the_wall_clock():
    reality, _ = _mounted_tracy()

    with RealTimeSimulation(
        world=reality, headless=True, paced_to_the_wall_clock=False
    ) as simulation:
        started = time.monotonic()
        simulation.advance(1.0)
        elapsed = time.monotonic() - started

    assert elapsed < 1.0
