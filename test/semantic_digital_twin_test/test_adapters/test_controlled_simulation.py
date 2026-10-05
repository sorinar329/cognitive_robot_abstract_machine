import os
import threading
import time
from datetime import timedelta

import pytest

from semantic_digital_twin.adapters.controlled_simulation import (
    BodyFrictionChange,
    BodyMassChange,
    BodyPoseChange,
    ControlledSimulation,
    Placement,
    SimulationRunState,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.exceptions import (
    BodyWithoutContactError,
    MasslessBodyError,
    NegativeFrictionError,
    NonPositiveMassError,
    SimulationStoppedError,
    UnmovableBodyError,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.contact import ContactFriction
from semantic_digital_twin.world_description.world_entity import Body

pytestmark = pytest.mark.skipif(
    os.environ.get("CI", "false").lower() == "false",
    reason="MuJoCo tests only run in CI",
)

FALL_DURATION = timedelta(milliseconds=500)
"""
How long the box is left to fall in a test that measures how far it fell.
"""

THREAD_TIMEOUT = timedelta(seconds=5)
"""
How long a test waits for the stepping thread before treating it as stuck.
"""

BLOCKED_THREAD_GRACE = timedelta(milliseconds=200)
"""
How long a paused stepping thread is given to show that it does not return.
"""


def fallen_height(start_height: float, duration: timedelta) -> float:
    """
    :return: Where a body dropped from ``start_height`` is after falling for
        ``duration``.
    """
    return start_height - 0.5 * 9.81 * duration.total_seconds() ** 2


def box_of(world: World) -> Body:
    """
    :return: The box of the falling-box world.
    """
    return world.get_body_by_name("box")


def height_once_it_settles(body: Body, expected: float) -> float:
    """
    Wait until the world holds ``body`` at the ``expected`` height, or until
    :data:`THREAD_TIMEOUT` passes.

    :return: The height the world holds ``body`` at when the wait ends.
    """
    deadline = time.monotonic() + THREAD_TIMEOUT.total_seconds()
    height = float(body.global_pose.to_position().z)
    while height != pytest.approx(expected) and time.monotonic() < deadline:
        time.sleep(BLOCKED_THREAD_GRACE.total_seconds() / 10)
        height = float(body.global_pose.to_position().z)
    return height


def box_height(simulation: MujocoSim) -> float:
    """
    :return: How high MuJoCo holds the box.
    """
    return simulation.simulator.get_body_position(body_name="box").result[2]


@pytest.fixture
def simulation(falling_box_world) -> MujocoSim:
    """
    A stepped simulation of the falling-box world, stopped after the test.
    """
    simulation = MujocoSim(world=falling_box_world, headless=True)
    simulation.start_stepped_simulation()
    yield simulation
    simulation.stop_simulation()


@pytest.fixture
def controlled(simulation) -> ControlledSimulation:
    return ControlledSimulation(simulation)


@pytest.fixture
def world_with_a_table(falling_box_world) -> World:
    """
    The falling-box world with a stand fixed a metre along x from the box's fixed root,
    and a table fixed half a metre above the stand.
    """
    root = falling_box_world.get_body_by_name("root")
    with falling_box_world.modify_world():
        stand = Body(name=PrefixedName("stand"))
        table = Body(name=PrefixedName("table"))
        table.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix(),
                    scale=Scale(0.4, 0.4, 0.05),
                )
            ],
            reference_frame=table,
        )
        falling_box_world.add_connection(
            FixedConnection(
                parent=root,
                child=stand,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.0, reference_frame=root
                ),
            )
        )
        falling_box_world.add_connection(
            FixedConnection(
                parent=stand,
                child=table,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=0.5, reference_frame=stand
                ),
            )
        )
    return falling_box_world


def step_in_thread(controlled: ControlledSimulation) -> threading.Thread:
    """
    Step ``controlled`` once on a thread of its own, as a plan's control loop would.

    :return: The started thread; an error the step raised is kept on it as ``error``.
    """

    def step() -> None:
        try:
            controlled.step_simulation(FALL_DURATION)
        except SimulationStoppedError as error:
            thread.error = error

    thread = threading.Thread(target=step, daemon=True)
    thread.error = None
    thread.start()
    return thread


# %% body physics in MuJoCo


def test_a_set_mass_is_the_mass_mujoco_simulates(simulation, falling_box_world):
    box = box_of(falling_box_world)

    simulation.set_body_mass(box, 2.5)

    assert simulation.body_mass(box) == 2.5


def test_setting_the_mass_keeps_the_mass_distribution(simulation, falling_box_world):
    box = box_of(falling_box_world)
    mass = simulation.body_mass(box)
    inertia = simulation.body_inertia(box)

    simulation.set_body_mass(box, 3 * mass)

    assert simulation.body_inertia(box) == pytest.approx(3 * inertia)


def test_a_set_friction_is_the_friction_mujoco_simulates(simulation, falling_box_world):
    box = box_of(falling_box_world)
    friction = ContactFriction(sliding=0.2, torsional=0.01, rolling=0.001)

    simulation.set_body_friction(box, friction)

    assert simulation.body_friction(box) == friction


def test_a_body_without_colliding_geoms_has_no_friction(simulation, falling_box_world):
    assert simulation.body_friction(falling_box_world.root) is None


# %% stepping


def test_a_running_controlled_simulation_steps_like_the_simulation_itself(
    controlled, simulation
):
    controlled.step_simulation(FALL_DURATION)

    assert box_height(simulation) == pytest.approx(
        fallen_height(1.0, FALL_DURATION), abs=0.01
    )


def test_a_paused_simulation_holds_the_stepping_thread_until_resumed(
    controlled, simulation, falling_box_world
):
    controlled.pause()
    thread = step_in_thread(controlled)
    thread.join(BLOCKED_THREAD_GRACE.total_seconds())
    held = thread.is_alive()
    height_while_paused = float(box_of(falling_box_world).global_pose.to_position().z)

    controlled.resume()
    thread.join(THREAD_TIMEOUT.total_seconds())

    assert held
    assert height_while_paused == pytest.approx(1.0)
    assert not thread.is_alive()
    assert box_height(simulation) == pytest.approx(
        fallen_height(1.0, FALL_DURATION), abs=0.01
    )


def test_stopping_ends_the_next_step_with_an_error(controlled):
    controlled.stop()

    with pytest.raises(SimulationStoppedError):
        controlled.step_simulation(FALL_DURATION)


def test_stopping_a_paused_simulation_releases_the_stepping_thread_with_an_error(
    controlled,
):
    controlled.pause()
    thread = step_in_thread(controlled)

    controlled.stop()
    thread.join(THREAD_TIMEOUT.total_seconds())

    assert not thread.is_alive()
    assert isinstance(thread.error, SimulationStoppedError)


def test_the_run_state_follows_pause_resume_and_stop(controlled):
    states = [controlled.state]
    controlled.pause()
    states.append(controlled.state)
    controlled.resume()
    states.append(controlled.state)
    controlled.stop()
    states.append(controlled.state)

    assert states == [
        SimulationRunState.RUNNING,
        SimulationRunState.PAUSED,
        SimulationRunState.RUNNING,
        SimulationRunState.STOPPED,
    ]


# %% edits


def test_a_pose_change_takes_effect_before_the_next_step(
    controlled, simulation, falling_box_world
):
    box = box_of(falling_box_world)
    controlled.submit(
        BodyPoseChange(
            box,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                z=2.0, reference_frame=falling_box_world.root
            ),
        )
    )

    controlled.step_simulation(FALL_DURATION)

    assert box_height(simulation) == pytest.approx(
        fallen_height(2.0, FALL_DURATION), abs=0.01
    )


def test_an_edit_made_while_paused_is_applied_without_resuming(
    controlled, falling_box_world
):
    box = box_of(falling_box_world)
    controlled.pause()
    thread = step_in_thread(controlled)

    controlled.submit(
        BodyPoseChange(
            box,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                z=2.0, reference_frame=falling_box_world.root
            ),
        )
    )
    height_while_paused = height_once_it_settles(box, 2.0)
    still_held = thread.is_alive()
    controlled.stop()
    thread.join(THREAD_TIMEOUT.total_seconds())

    assert still_held
    assert height_while_paused == pytest.approx(2.0)


def test_a_simulated_body_reflects_an_applied_mass_change(
    controlled, falling_box_world
):
    box = box_of(falling_box_world)
    controlled.submit(BodyMassChange(box, 4.0))

    controlled.step_simulation(FALL_DURATION)

    assert controlled.simulated_body(box).mass == 4.0


def test_a_simulated_body_reflects_an_applied_friction_change(
    controlled, falling_box_world
):
    box = box_of(falling_box_world)
    friction = ContactFriction(sliding=0.3)
    controlled.submit(BodyFrictionChange(box, friction))

    controlled.step_simulation(FALL_DURATION)

    assert controlled.simulated_body(box).friction == friction


def test_each_body_says_how_it_can_be_placed(controlled, falling_box_world):
    placements = {
        body.name.name: controlled.simulated_body(body).placement
        for body in falling_box_world.bodies
    }

    assert placements == {
        "box": Placement.LOOSE,
        "root": Placement.FIXED,
        falling_box_world.root.name.name: Placement.UNMOVABLE,
    }


def test_a_body_fixed_through_other_fixed_bodies_is_fixed(world_with_a_table):
    simulation = MujocoSim(world=world_with_a_table, headless=True)
    simulation.start_stepped_simulation()
    try:
        controlled = ControlledSimulation(simulation)
        placement = controlled.simulated_body(
            world_with_a_table.get_body_by_name("table")
        ).placement
    finally:
        simulation.stop_simulation()

    assert placement is Placement.FIXED


def test_a_moved_nested_fixed_body_stands_at_its_new_pose_in_mujoco(world_with_a_table):
    table = world_with_a_table.get_body_by_name("table")
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.2, y=0.3, z=0.4, yaw=0.5, reference_frame=world_with_a_table.root
    )
    simulation = MujocoSim(world=world_with_a_table, headless=True)
    simulation.start_stepped_simulation()
    try:
        controlled = ControlledSimulation(simulation)
        controlled.submit(BodyPoseChange(table, pose))
        controlled.step_simulation(FALL_DURATION)
        position = simulation.simulator.get_body_position(body_name="table").result
    finally:
        simulation.stop_simulation()

    assert table.global_pose.to_np() == pytest.approx(pose.to_np())
    assert position == pytest.approx(pose.to_position().to_np()[:3])


def test_an_unmovable_body_cannot_be_moved(controlled, falling_box_world):
    with pytest.raises(UnmovableBodyError):
        controlled.submit(
            BodyPoseChange(
                falling_box_world.root,
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=falling_box_world.root
                ),
            )
        )


def test_a_moved_fixed_body_stands_at_its_new_pose_in_the_world_and_in_mujoco(
    controlled, simulation, falling_box_world
):
    fixed = falling_box_world.get_body_by_name("root")
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.5, y=-0.25, z=0.2, yaw=0.3, reference_frame=falling_box_world.root
    )
    controlled.submit(BodyPoseChange(fixed, pose))

    controlled.step_simulation(FALL_DURATION)

    assert fixed.global_pose.to_np() == pytest.approx(pose.to_np())
    assert simulation.simulator.get_body_position(body_name="root").result == (
        pytest.approx(pose.to_position().to_np()[:3])
    )


def test_a_simulated_fixed_body_reports_where_it_stands(controlled, falling_box_world):
    fixed = falling_box_world.get_body_by_name("root")
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.5, reference_frame=falling_box_world.root
    )
    controlled.submit(BodyPoseChange(fixed, pose))

    controlled.step_simulation(FALL_DURATION)

    assert controlled.simulated_body(fixed).pose.to_np() == pytest.approx(pose.to_np())


def test_a_body_mujoco_gives_no_mass_cannot_be_given_one(controlled, falling_box_world):
    with pytest.raises(MasslessBodyError):
        controlled.submit(BodyMassChange(falling_box_world.root, 1.0))


def test_a_body_without_colliding_geoms_cannot_be_given_friction(
    controlled, falling_box_world
):
    with pytest.raises(BodyWithoutContactError):
        controlled.submit(BodyFrictionChange(falling_box_world.root, ContactFriction()))


def test_a_stopped_simulation_takes_no_edits(controlled, falling_box_world):
    controlled.stop()

    with pytest.raises(SimulationStoppedError):
        controlled.submit(BodyMassChange(box_of(falling_box_world), 1.0))


def test_a_mass_has_to_be_positive(falling_box_world):
    with pytest.raises(NonPositiveMassError):
        BodyMassChange(box_of(falling_box_world), 0.0)


def test_friction_cannot_be_negative(falling_box_world):
    with pytest.raises(NegativeFrictionError):
        BodyFrictionChange(box_of(falling_box_world), ContactFriction(sliding=-0.1))
