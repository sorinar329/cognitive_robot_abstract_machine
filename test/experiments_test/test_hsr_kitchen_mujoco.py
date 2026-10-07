"""
The HSR carrying a milk carton and a cereal box between two tables of the predetermined
kitchen, simulated in MuJoCo.
"""

import os

import mujoco
import pytest

from experiments.hsr_kitchen_mujoco.demo import (
    CARRIED_OBJECTS,
    KitchenTable,
    build_world,
    goal_pose,
    run,
    start_pose,
    table_extent,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.predetermined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.world import World

runs_in_ci = pytest.mark.skipif(
    os.environ.get("CI", "false").lower() == "false",
    reason="MuJoCo tests only run in CI",
)

MUJOCO_MISSING_ID = -1
"""
The id MuJoCo answers with for a name its model does not hold.
"""

PLACEMENT_TOLERANCE = 0.03
"""
How far, in metres, an object may come to rest from where it was meant to be set down.
"""


@pytest.fixture(scope="module")
def world() -> World:
    return build_world()


def position_of(world: World, name: str):
    """
    :return: Where the world holds the body named ``name``.
    """
    return world.get_body_by_name(name).global_pose.to_position().to_np()[:3]


# %% the scene


def test_the_scene_is_the_predetermined_kitchen(world: World):
    kitchen = KitchenEnvironment().get_world()

    assert {body.name.name for body in kitchen.bodies} <= {
        body.name.name for body in world.bodies
    }


@pytest.mark.parametrize("table", list(KitchenTable))
def test_the_kitchen_has_both_tables(world: World, table: KitchenTable):
    assert world.get_body_by_name(table).name.name == table


@pytest.mark.parametrize("carried", CARRIED_OBJECTS, ids=lambda carried: carried.name)
def test_every_object_starts_on_the_source_table(world: World, carried):
    extent = table_extent(world, KitchenTable.SOURCE)
    x, y, _ = position_of(world, carried.name)

    assert extent.min_x < x < extent.max_x
    assert extent.min_y < y < extent.max_y
    assert position_of(world, carried.name) == pytest.approx(
        start_pose(world, carried).to_position().to_np()[:3]
    )


@runs_in_ci
def test_every_body_of_the_scene_is_simulated(world: World):
    simulation = MujocoSim(world=world, headless=True)
    model = simulation.simulator._mj_model
    missing = [
        body.name.name
        for body in world.bodies
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.name.name)
        == MUJOCO_MISSING_ID
    ]

    assert missing == []


# %% the run


@runs_in_ci
def test_the_hsr_carries_every_object_to_the_destination_table():
    """
    After the plan has run against MuJoCo physics, each object rests where it was meant
    to be set down on the destination table.
    """
    world = build_world()

    run(world, headless=True)

    for carried in CARRIED_OBJECTS:
        assert position_of(world, carried.name) == pytest.approx(
            goal_pose(world, carried).to_position().to_np()[:3],
            abs=PLACEMENT_TOLERANCE,
        )
