"""
The stacking demo carried out against MuJoCo physics, in a MuJoCo viewer.

The physics is stepped from the control loop rather than from a thread of its own: every
tick of the controller writes its command into the world, the servos take it as their set
point, and the physics advances one cycle before the next tick reads the world back. That
is what :class:`~giskardpy.executor.SteppedSimulationPacer` does, and
:data:`~coraplex.plans.executables.GiskardExecutable.simulation_pacer` is where a plan
picks it up.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, List

import mujoco
import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.executables import GiskardExecutable
from giskardpy.executor import SteppedSimulationPacer
from semantic_digital_twin.adapters.multi_sim import (
    ContactDimensionality,
    MujocoGeom,
    MujocoSim,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactImpedance,
    ContactParameters,
    ContactStiffness,
)

if TYPE_CHECKING:
    from demo import BuildsPlan, BuildsScene, BuildsWorld, StackedCube

HEADLESS = False
"""
Whether the simulation runs without a viewer.
"""

STEP_SIZE = 1e-3
"""
How much simulated time one physics step covers, in seconds.
"""

SETTLING_DURATION = timedelta(seconds=1)
"""
How long the scene is left to settle before the robot starts, so the cubes come to rest
on the table rather than being grasped mid-fall.
"""

FRICTION_CONE = mujoco.mjtCone.mjCONE_ELLIPTIC
"""
How MuJoCo bounds a contact's friction.

The elliptic cone is the exact one; the default pyramid only approximates it and lets a
squeezed cube creep between the pads.
"""

IMPEDANCE_RATIO = 10.0
"""
How much stiffer MuJoCo makes friction than the push along the contact normal
(``impratio``).

MuJoCo's own guidance for grasping with an elliptic cone; at the default of 1 a held
cube slides out under its own weight despite a firm squeeze.
"""

NO_SLIP_ITERATIONS = 0
"""
How many passes MuJoCo spends removing the slip its soft contacts leave.

None: the pass applies to every contact in the scene, not only to the pads, so it would
also glue each cube to the one it is stacked on and make the tower steadier than wood
is. The montessori demo uses 10 to keep a held piece where the pads hold it.
"""

RIGID_CONTACT_STIFFNESS = ContactStiffness(
    time_constant=timedelta(seconds=2 * STEP_SIZE), damping_ratio=1.0
)
"""
How stiffly two bodies push back once they touch: as stiffly as MuJoCo stays stable at,
a time constant of two physics steps.

The wooden cubes and the gripper's pads are solid, so they must not sink
into each other; MuJoCo's default of 20 ms lets a firmly held cube sink several
millimetres into the pads.
"""

RIGID_CONTACT_IMPEDANCE = ContactImpedance(minimum=0.99, maximum=0.999, width=0.001)
"""
How little a contact gives way under load: nearly not at all, as between solid bodies.
"""


def run(
    build_world: BuildsWorld, build_scene: BuildsScene, build_plan: BuildsPlan
) -> None:
    """
    Carry the stacking plan out against MuJoCo physics.

    :param build_world: Builds the world Tracy stands in.
    :param build_scene: Stands the cubes in that world.
    :param build_plan: Builds the plan that stacks them.
    """
    world, robot = build_world()
    cubes = build_scene(world)
    _resist_turning_between_the_pads(cubes)
    _make_every_contact_rigid(world)
    context = Context(
        world=world,
        robot=robot,
        evaluate_conditions=False,
        update_world_model_attachment=False,
    )
    plan = build_plan(context, cubes)

    simulation = MujocoSim(
        world=world,
        headless=HEADLESS,
        step_size=STEP_SIZE,
        cone=FRICTION_CONE,
        impratio=IMPEDANCE_RATIO,
        noslip_iterations=NO_SLIP_ITERATIONS,
    )
    simulation.start_stepped_simulation()
    _weigh_the_cubes(simulation, cubes)
    previous_pacer = GiskardExecutable.simulation_pacer
    GiskardExecutable.simulation_pacer = SteppedSimulationPacer(simulation)
    try:
        simulation.step_simulation(SETTLING_DURATION)
        with ExecutionEnvironment(execution_type=ExecutionType.SIMULATED):
            plan.perform()
    finally:
        GiskardExecutable.simulation_pacer = previous_pacer
        simulation.stop_simulation()


def _weigh_the_cubes(simulation: MujocoSim, cubes: List[StackedCube]) -> None:
    """
    Give every cube in the simulation the mass and inertia its body carries.

    MuJoCo is built to work every body's mass out from its geoms at the density of water,
    and it counts the drawn geom as well as the colliding one, which makes a 4 cm cube
    weigh 128 g. The cubes' own inertial properties are therefore set on the compiled
    model.

    Recomputing the model's constants afterwards moves every body to the model's
    reference pose, which for a cube is inside the table, so the simulation's state is
    kept aside and restored around it.

    :param simulation: The running simulation, already compiled.
    :param cubes: The cubes to weigh.
    """
    simulator = simulation.simulator
    with simulator._model_lock:
        model = simulator._mj_model
        for cube in cubes:
            inertial = cube.body.inertial
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, cube.body.name.name
            )
            model.body_mass[body_id] = inertial.mass
            model.body_ipos[body_id] = inertial.center_of_mass.to_np()[:3]
            model.body_iquat[body_id] = (1.0, 0.0, 0.0, 0.0)
            model.body_inertia[body_id] = np.diag(inertial.inertia.data)
        data = simulator._mj_data
        state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        state = np.empty(state_size)
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_setConst(model, data)
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, data)


def _resist_turning_between_the_pads(cubes: List[StackedCube]) -> None:
    """
    Have MuJoCo resolve every cube's twisting friction.

    A cube held between two pads touches them along a line, and without friction
    against turning about the contact normals it swings about that line on the carry.

    :param cubes: The cubes the gripper takes hold of.
    """
    for cube in cubes:
        for shape in cube.body.collision.shapes:
            shape.add_simulator_property(
                MujocoGeom(
                    contact_dimensionality=ContactDimensionality.SLIDING_AND_TWISTING
                )
            )


def _make_every_contact_rigid(world: World) -> None:
    """
    Make every body of ``world`` push back as a solid does, keeping the friction each one
    already has.

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
