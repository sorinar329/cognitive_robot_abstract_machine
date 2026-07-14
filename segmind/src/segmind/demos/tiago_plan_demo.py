"""
Demonstration of a coraplex plan on the TIAGo++ robot in an apartment scene.

After the plan completes the final robot pose is frozen in the viewer until
the window is closed.

Run with::

    python -m segmind.demos.tiago_plan_demo
"""
from __future__ import annotations

from coraplex.datastructures.enums import Arms
from coraplex.motion_executor import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

from segmind.demos.tiago_apartment_scene import (
    ApartmentSimulation,
    _add_floor,
    _add_scene_objects,
    build_context,
    build_tiago_world,
)

if __name__ == "__main__":
    _world = build_tiago_world()
    _add_scene_objects(_world)

    _simulation = ApartmentSimulation(world=_world)
    _simulation.start()

    _context = build_context(_world)

    try:
        with simulated_robot:
            sequential(
                [ParkArmsAction(arm=Arms.BOTH)],
                _context,
            ).perform()
        _simulation.freeze_and_wait()
    finally:
        _simulation.stop()
