"""
The montessori demo carried out against MuJoCo physics, in a MuJoCo viewer.

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
    from demo import MontessoriPiece, MontessoriScene

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
How long the scene is left to settle before the robot starts, so the pieces come to rest
on the table rather than being grasped mid-fall.
"""

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

The wooden pieces, the board and the gripper's pads are solid, so they must not sink
into each other; MuJoCo's default of 20 ms lets a firmly held piece sink several
millimetres into the pads.
"""

RIGID_CONTACT_IMPEDANCE = ContactImpedance(minimum=0.99, maximum=0.999, width=0.001)
"""
How little a contact gives way under load: nearly not at all, as between solid bodies.
"""


def run(scene: MontessoriScene) -> None:
    """
    Carry the sorting plan out against MuJoCo physics.

    :param scene: Tracy, the board and the pieces.
    """
    world = scene.world
    _resist_turning_between_the_pads(scene.pieces)
    _make_every_contact_rigid(world)
    context = Context(
        world=world,
        robot=scene.robot,
        evaluate_conditions=False,
        update_world_model_attachment=False,
    )
    plan = scene.build_plan(context)

    simulation = MujocoSim(
        world=world,
        headless=HEADLESS,
        step_size=STEP_SIZE,
        cone=FRICTION_CONE,
        impratio=IMPEDANCE_RATIO,
        noslip_iterations=NO_SLIP_ITERATIONS,
    )
    simulation.start_stepped_simulation()
    previous_pacer = GiskardExecutable.simulation_pacer
    GiskardExecutable.simulation_pacer = SteppedSimulationPacer(simulation)
    try:
        simulation.step_simulation(SETTLING_DURATION)
        with ExecutionEnvironment(execution_type=ExecutionType.SIMULATED):
            plan.perform()
    finally:
        GiskardExecutable.simulation_pacer = previous_pacer
        simulation.stop_simulation()


def _resist_turning_between_the_pads(pieces: List[MontessoriPiece]) -> None:
    """
    Have MuJoCo resolve every piece's twisting friction.

    A piece held between two pads touches them along a line, and without friction
    against turning about the contact normals it swings about that line on the carry.

    :param pieces: The pieces the gripper takes hold of.
    """
    for piece in pieces:
        for shape in piece.body.collision.shapes:
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
