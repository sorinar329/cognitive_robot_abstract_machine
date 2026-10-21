"""
The i40lab demo carried out against MuJoCo physics.

The physics is stepped from the control loop rather than from a thread of its own:
every tick of the controller writes its command into the world, the servos take it
as their set point, and the physics advances one cycle before the next tick reads
the world back. That is what :class:`~giskardpy.executor.SteppedSimulationPacer`
does, and :data:`~coraplex.plans.executables.GiskardExecutable.simulation_pacer`
is where a plan picks it up.

The simulation is offered to the visualization for control, so a viewer that can
control one (cramera) pauses, resumes and stops the run and moves bodies or
changes their mass and friction while it runs.
"""

from __future__ import annotations

import os
from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Callable

import mujoco

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.plan import Plan
from coraplex.visualization import WorldVisualization
from giskardpy.executor import SteppedSimulationPacer
from semantic_digital_twin.adapters.controlled_simulation import ControlledSimulation
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactImpedance,
    ContactParameters,
    ContactStiffness,
)

if TYPE_CHECKING:
    from semantic_digital_twin.robots.ur5e import UR5eMujocoDualGripper


STEP_SIZE = 0.002
SETTLING_DURATION = timedelta(seconds=1)

HEADLESS = False
"""
Whether the simulation runs without a viewer.
"""


class MujocoDemoOption(StrEnum):
    """
    Environment settings steering the physics demo run.
    """

    HEADLESS = "MUJOCO_HEADLESS"
    """
    Run the simulation without a viewer; the physics is unchanged.
    """


def headless_from_environment() -> bool:
    """
    Whether the simulation runs without a viewer.

    An unset variable keeps the module default; a set one is read as a truth value, so
    ``1`` runs headless and ``0`` opens the viewer.

    :return: Whether to run headless.
    """
    raw = os.environ.get(MujocoDemoOption.HEADLESS)
    if raw is None:
        return HEADLESS
    return raw.strip().lower() in ("1", "true", "yes", "on")


FRICTION_CONE = mujoco.mjtCone.mjCONE_ELLIPTIC
"""
How MuJoCo bounds a contact's friction.

The elliptic cone is the exact one; the default pyramid only approximates it and lets a
squeezed piece creep between the pads.
"""

IMPEDANCE_RATIO = 10.0
"""
How much stiffer MuJoCo makes friction than the push along the contact normal
(``impratio``).

MuJoCo's own guidance for grasping with an elliptic cone; at the default of 1 a held
piece slides out under its own weight despite a firm squeeze.
"""

NO_SLIP_ITERATIONS = 10
"""
How many passes MuJoCo spends removing the slip its soft contacts leave, so that a piece
held as firmly as the real gripper holds it stays where the pads hold it.
"""

RIGID_CONTACT_STIFFNESS = ContactStiffness(
    time_constant=timedelta(seconds=2 * STEP_SIZE), damping_ratio=1.0
)
"""
How stiffly two bodies push back once they touch: as stiffly as MuJoCo stays stable at,
a time constant of two physics steps.

The arm, the mounts and the fingers are solid, so they must not sink into each other;
MuJoCo's default of 20 ms lets a firmly held piece sink several millimetres into the
pads.
"""

RIGID_CONTACT_IMPEDANCE = ContactImpedance(minimum=0.99, maximum=0.999, width=0.001)
"""
How little a contact gives way under load: nearly not at all, as between solid bodies.
"""


def run(
    build_world: Callable[[], tuple[World, UR5eMujocoDualGripper]],
    build_plan: Callable[[Context], Plan],
) -> None:
    """
    Carry the motion plan out against MuJoCo physics.

    :param build_world: Builds the world with the UR5e annotated.
    :param build_plan: Builds the plan that moves the arm.
    """
    world, robot = build_world()
    _make_every_contact_rigid(world)

    headless = headless_from_environment()
    print(
        f"MuJoCo viewer: {'off' if headless else 'on'} ({MujocoDemoOption.HEADLESS}={os.environ.get(MujocoDemoOption.HEADLESS, 'unset')})"
    )

    # The fingers already hold a grasped piece through contact; re-parenting it
    # in the world model as well would stack a second set of degrees of
    # freedom onto it, and MuJoCo rejects the model with more than 6 dofs.
    context = Context(
        world=world,
        robot=robot,
        evaluate_conditions=False,
        update_world_model_attachment=False,
    )
    plan = build_plan(context)

    visualization = WorldVisualization.from_environment(world).start()
    visualization.attach_plan(plan)

    simulation = MujocoSim(
        world=world,
        headless=headless,
        step_size=STEP_SIZE,
        cone=FRICTION_CONE,
        impratio=IMPEDANCE_RATIO,
        noslip_iterations=NO_SLIP_ITERATIONS,
    )
    simulation.start_stepped_simulation()
    controlled = ControlledSimulation(simulation)
    visualization.attach_simulation(controlled)

    previous_pacer = GiskardExecutable.simulation_pacer
    GiskardExecutable.simulation_pacer = SteppedSimulationPacer(controlled)

    try:
        controlled.step_simulation(SETTLING_DURATION)
        with ExecutionEnvironment(execution_type=ExecutionType.SIMULATED):
            plan.perform()
    finally:
        GiskardExecutable.simulation_pacer = previous_pacer
        visualization.stop()
        simulation.stop_simulation()


def _make_every_contact_rigid(world: World) -> None:
    """
    Make every body of ``world`` push back as a solid does, keeping the friction each
    one already has.

    :param world: The world whose bodies are made rigid.
    """
    for body in world.bodies_with_collision:
        for shape in body.collision.shapes:
            contact = shape.get_simulator_property_of_type(ContactParameters)
            if contact is None:
                contact = ContactParameters(friction=ContactFriction())
                shape.add_simulator_property(contact)
            contact.stiffness = RIGID_CONTACT_STIFFNESS
            contact.impedance = RIGID_CONTACT_IMPEDANCE


def _load_world() -> tuple[World, UR5eMujocoDualGripper]:
    """
    Internal helper for visualize-only fallback (no plan).
    """
    import demo_3fg15 as i40lab_demo

    return i40lab_demo.build_world()


def run_visualize_only() -> None:
    """
    Run without a plan — just live view of the physics.
    """
    world, robot = _load_world()
    visualization = WorldVisualization.from_environment(world).start()

    simulation = MujocoSim(
        world=world,
        headless=True,
        step_size=STEP_SIZE,
    )
    simulation.start_stepped_simulation()
    controlled = ControlledSimulation(simulation)
    visualization.attach_simulation(controlled)

    try:
        controlled.step_simulation(SETTLING_DURATION)
        while True:
            controlled.step_simulation(timedelta(seconds=STEP_SIZE))
    except KeyboardInterrupt:
        pass
    finally:
        visualization.stop()
        simulation.stop_simulation()


if __name__ == "__main__":
    # cramera-live puts this directory on sys.path, so demo_3fg15 resolves here
    import demo_3fg15 as i40lab_demo

    i40lab_demo.main()
