"""
The stacking demo carried out against MuJoCo physics, in a MuJoCo viewer.

The physics is stepped from the control loop rather than from a thread of its own: every
tick of the controller writes its command into the world, the servos take it as their set
point, and the physics advances one cycle before the next tick reads the world back. That
is what :class:`~coraplex.execution_environment.PhysicallySimulatedRobot` has every
motion of a plan do.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import mujoco

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import PhysicallySimulatedRobot
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from demo import StackingEnvironment


def run(environment: StackingEnvironment, world: World) -> None:
    """
    Carry the stacking plan out against MuJoCo physics.

    :param environment: The scene, with the plan that stacks its cubes.
    :param world: The world the environment stood Tracy and the cubes in.
    """
    [robot] = world.get_semantic_annotations_by_type(Tracy)
    context = Context(
        world=world,
        robot=robot,
        evaluate_conditions=False,
    )
    plan = environment.build_plan(context)

    simulation = MujocoSim(
        world=world,
        headless=False,
        step_size=1e-3,
        # The elliptic cone is the exact one; the default pyramid only approximates it
        # and lets a squeezed cube creep between the pads.
        cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
        # MuJoCo's own guidance for grasping with an elliptic cone; at the default of 1
        # a held cube slides out under its own weight despite a firm squeeze.
        impratio=10.0,
        # No passes removing the slip soft contacts leave: they apply to every contact,
        # so they would also glue each cube to the one below it.
        noslip_iterations=0,
    )
    simulation.start_stepped_simulation()
    try:
        # Let the cubes come to rest on the table before the robot reaches for them.
        simulation.step_simulation(timedelta(seconds=1))
        with PhysicallySimulatedRobot(simulation=simulation):
            plan.perform()
    finally:
        simulation.stop_simulation()
