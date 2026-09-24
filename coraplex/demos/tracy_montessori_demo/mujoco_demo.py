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

import time
from datetime import timedelta
from typing import TYPE_CHECKING

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.executables import GiskardExecutable
from giskardpy.executor import SteppedSimulationPacer
from semantic_digital_twin.adapters.multi_sim import MujocoSim

if TYPE_CHECKING:
    from demo import BuildsPlan, BuildsScene, BuildsWorld

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

VIEWER_POLL_SECONDS = 0.1
"""
How long the demo waits between checks of whether the viewer is still open, once the plan
is done and the scene is left to look at.
"""


def run(
    build_world: BuildsWorld, build_scene: BuildsScene, build_plan: BuildsPlan
) -> None:
    """
    Carry the sorting plan out against MuJoCo physics.

    :param build_world: Builds the world Tracy stands in.
    :param build_scene: Stands the board and the pieces in that world.
    :param build_plan: Builds the plan that sorts them.
    """
    world, robot = build_world()
    pieces = build_scene(world)
    context = Context(
        world=world,
        robot=robot,
        evaluate_conditions=False,
        update_world_model_attachment=False,
    )
    plan = build_plan(context, pieces)

    simulation = MujocoSim(world=world, headless=HEADLESS, step_size=STEP_SIZE)
    simulation.start_stepped_simulation()
    previous_pacer = GiskardExecutable.simulation_pacer
    GiskardExecutable.simulation_pacer = SteppedSimulationPacer(simulation)
    try:
        simulation.step_simulation(SETTLING_DURATION)
        with ExecutionEnvironment(execution_type=ExecutionType.SIMULATED):
            plan.perform()
        _leave_the_scene_to_look_at(simulation)
    finally:
        GiskardExecutable.simulation_pacer = previous_pacer
        simulation.stop_simulation()


def _leave_the_scene_to_look_at(simulation: MujocoSim) -> None:
    """
    Keep stepping the physics until the viewer is closed, so the finished scene can be
    looked at rather than vanishing the moment the plan ends.

    :param simulation: The simulation to keep stepping.
    """
    if HEADLESS:
        return
    while simulation.simulator.renderer.is_running():
        simulation.step_simulation(timedelta(seconds=VIEWER_POLL_SECONDS))
        time.sleep(VIEWER_POLL_SECONDS)
