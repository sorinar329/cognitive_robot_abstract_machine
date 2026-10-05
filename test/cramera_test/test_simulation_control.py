"""
Tests of the opt-in simulation control: the scene graph the bridge serves, and the
pause, resume, stop and edit requests it hands to a controlled simulation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
from unittest.mock import Mock

from typing_extensions import Iterator

import pytest

from semantic_digital_twin.adapters.controlled_simulation import (
    BodyPoseChange,
    ControlledSimulation,
    Placement,
    SimulationRunState,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.contact import ContactFriction
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live import visualization as visualization_module
from cramera.body_geometry import rounded_pose
from cramera.live.bridge import Bridge
from cramera.live.http import serve
from cramera.live.simulation_control import (
    PoseField,
    SceneGraphField,
    SimulationChange,
    SimulationEditField,
    SimulationRoute,
)
from cramera.live.transforms import ConnectionKind
from cramera.live.visualization import LiveVisualization, WorldModelSync

from .test_live_http import post
from .test_live_visualization import ServerRecorder
from .test_server import get_json

STEP = timedelta(milliseconds=1)
"""
How far the tests advance the simulation to have its pending changes made.
"""

ROOTED_MASS = 0.0
"""
The mass the stand-in simulation gives every body but the piece, none of which it
moves.
"""

PIECE_MASS = 0.2
"""
The mass the stand-in simulation gives the piece, in kilograms.
"""

PIECE_FRICTION = ContactFriction(sliding=0.8)
"""
The friction the stand-in simulation gives every body.
"""


# %% fixtures


@pytest.fixture()
def world() -> World:
    """
    A table fixed to the world's root, and a piece on a free connection, raised above
    the root.
    """
    world = World()
    root = Body(name=PrefixedName("root", prefix="world"))
    table = Body(name=PrefixedName("table", prefix="world"))
    piece = Body(
        name=PrefixedName("piece", prefix="world"),
        visual=ShapeCollection(shapes=[Box(scale=Scale(0.05, 0.05, 0.05))]),
    )
    with world.modify_world():
        world.add_body(root)
        world.add_connection(FixedConnection(parent=root, child=table))
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=root,
                child=piece,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.3, z=0.8, reference_frame=root
                ),
            )
        )
    return world


def piece_of(world: World) -> Body:
    return world.get_body_by_name("piece")


def table_of(world: World) -> Body:
    return world.get_body_by_name("table")


@pytest.fixture()
def mujoco(world) -> Mock:
    """
    A stand-in for MuJoCo simulating every body of ``world``: the piece with
    :data:`PIECE_MASS`, every other body without mass, and every body with
    :data:`PIECE_FRICTION`.
    """
    simulation = Mock(spec=MujocoSim)
    simulation.world = world
    simulation.simulates.return_value = True
    simulation.body_mass.side_effect = lambda body: (
        PIECE_MASS if body is piece_of(world) else ROOTED_MASS
    )
    simulation.body_friction.return_value = PIECE_FRICTION
    return simulation


@pytest.fixture()
def controlled(mujoco) -> ControlledSimulation:
    return ControlledSimulation(mujoco)


@pytest.fixture()
def bridge(world, controlled) -> Bridge:
    """
    A bridge bound to ``world``, with ``controlled`` attached and one snapshot taken.
    """
    bridge = Bridge()
    bridge.attach(world)
    bridge.simulation = controlled
    bridge.snapshot()
    return bridge


@contextmanager
def served(bridge: Bridge) -> Iterator[str]:
    """
    Serve ``bridge`` on an ephemeral port for the duration of the block.

    :return: The address it is served at.
    """
    httpd = serve(bridge, 0)
    try:
        yield "http://localhost:%d" % httpd.server_address[1]
    finally:
        httpd.shutdown()


@pytest.fixture()
def server(bridge):
    with served(bridge) as address:
        yield address


@pytest.fixture()
def server_without_simulation(world):
    """
    A served bridge bound to ``world`` that no simulation was attached to.
    """
    bridge = Bridge()
    bridge.attach(world)
    with served(bridge) as address:
        yield address


def scene_graph_rows(server: str) -> dict:
    """
    :return: The served scene graph's bodies by name.
    """
    payload = get_json(server + SimulationRoute.SCENE_GRAPH)
    return {row[SceneGraphField.NAME]: row for row in payload[SceneGraphField.BODIES]}


def edit(server: str, body: Body, change: SimulationChange, value: object):
    """
    Post one change of ``body`` and return ``(status, payload)``.
    """
    return post(
        server + SimulationRoute.EDIT,
        {
            SimulationEditField.BODY: str(body.name),
            SimulationEditField.CHANGE: change,
            SimulationEditField.VALUE: value,
        },
    )


# %% the scene graph


def test_without_a_simulation_the_scene_graph_is_unavailable(
    server_without_simulation,
):
    payload = get_json(server_without_simulation + SimulationRoute.SCENE_GRAPH)

    assert payload == {SceneGraphField.AVAILABLE: False}


def test_the_scene_graph_reports_the_run_state(server, controlled):
    controlled.pause()

    payload = get_json(server + SimulationRoute.SCENE_GRAPH)

    assert payload[SceneGraphField.STATE] == SimulationRunState.PAUSED


def test_the_scene_graph_lists_every_simulated_body(server, controlled):
    assert set(scene_graph_rows(server)) == {
        str(simulated.body.name) for simulated in controlled.simulated_bodies()
    }


def test_the_scene_graph_carries_each_bodys_physics(server, world):
    row = scene_graph_rows(server)[str(piece_of(world).name)]

    assert row[SceneGraphField.MASS] == PIECE_MASS
    assert row[SceneGraphField.FRICTION] == asdict(PIECE_FRICTION)
    assert row[SceneGraphField.PLACEMENT] == Placement.LOOSE


def test_the_scene_graph_names_the_connection_holding_each_body(server, world):
    rows = scene_graph_rows(server)

    assert rows[str(piece_of(world).name)][SceneGraphField.PARENT] == str(
        world.root.name
    )
    assert (
        rows[str(piece_of(world).name)][SceneGraphField.CONNECTION]
        == ConnectionKind.FREE
    )
    assert rows[str(table_of(world).name)][SceneGraphField.CONNECTION] == (
        ConnectionKind.FIXED
    )


def test_the_scene_graph_carries_the_published_pose_of_an_object(server, bridge, world):
    published = bridge.get_state()["objects"]["piece"]

    row = scene_graph_rows(server)[str(piece_of(world).name)]

    assert row[SceneGraphField.POSE] == published


def test_the_scene_graph_carries_where_a_fixed_body_stands(server, world):
    row = scene_graph_rows(server)[str(table_of(world).name)]

    assert row[SceneGraphField.PLACEMENT] == Placement.FIXED
    assert row[SceneGraphField.POSE] == rounded_pose(table_of(world))


# %% run control


@pytest.mark.parametrize(
    "route, state",
    [
        (SimulationRoute.PAUSE, SimulationRunState.PAUSED),
        (SimulationRoute.STOP, SimulationRunState.STOPPED),
    ],
)
def test_a_control_request_changes_the_run_state(server, controlled, route, state):
    status, payload = post(server + route)

    assert status == 200
    assert payload[SceneGraphField.STATE] == state
    assert controlled.state is state


def test_resuming_lets_a_paused_simulation_run(server, controlled):
    controlled.pause()

    post(server + SimulationRoute.RESUME)

    assert controlled.state is SimulationRunState.RUNNING


def test_controlling_a_stopped_simulation_is_a_conflict(server, controlled):
    controlled.stop()

    status, payload = post(server + SimulationRoute.RESUME)

    assert status == 409
    assert payload["ok"] is False


@pytest.mark.parametrize(
    "route",
    [
        SimulationRoute.PAUSE,
        SimulationRoute.RESUME,
        SimulationRoute.STOP,
        SimulationRoute.EDIT,
    ],
)
def test_control_requests_without_a_simulation_are_refused(
    server_without_simulation, route
):
    status, payload = post(server_without_simulation + route)

    assert status == 404
    assert payload["ok"] is False


# %% edits


def test_a_posted_pose_moves_the_body_at_the_next_step(server, controlled, world):
    pose = {PoseField.X: 0.1, PoseField.Y: -0.2, PoseField.Z: 1.1}
    pose.update({PoseField.ROLL: 0.0, PoseField.PITCH: 0.0, PoseField.YAW: 0.5})

    status, _ = edit(server, piece_of(world), SimulationChange.POSE, pose)
    controlled.step_simulation(STEP)

    expected = HomogeneousTransformationMatrix.from_xyz_rpy(
        **pose, reference_frame=world.root
    )
    assert status == 200
    assert piece_of(world).global_pose.to_np() == pytest.approx(expected.to_np())


def test_a_posted_pose_moves_a_fixed_body_in_the_world_and_in_the_simulation(
    server, controlled, mujoco, world
):
    pose = {PoseField.X: 0.4, PoseField.Y: 0.1, PoseField.Z: 0.0}
    pose.update({PoseField.ROLL: 0.0, PoseField.PITCH: 0.0, PoseField.YAW: 1.0})

    status, _ = edit(server, table_of(world), SimulationChange.POSE, pose)
    controlled.step_simulation(STEP)

    expected = HomogeneousTransformationMatrix.from_xyz_rpy(
        **pose, reference_frame=world.root
    )
    assert status == 200
    assert table_of(world).global_pose.to_np() == pytest.approx(expected.to_np())
    moved_body, parent_T_body = mujoco.set_fixed_body_pose.call_args.args
    assert moved_body is table_of(world)
    assert parent_T_body.to_np() == pytest.approx(expected.to_np())


def test_moving_a_fixed_body_asks_the_viewer_to_reload_the_scene(
    bridge, controlled, world
):
    WorldModelSync(_world=world, bridge=bridge)
    before = bridge.bundle_signature()
    controlled.submit(
        BodyPoseChange(
            table_of(world),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.4, reference_frame=world.root
            ),
        )
    )

    controlled.step_simulation(STEP)

    assert bridge.bundle_signature() != before


def test_a_posted_mass_is_given_at_the_next_step(server, controlled, mujoco, world):
    status, _ = edit(server, piece_of(world), SimulationChange.MASS, 2.0)
    controlled.step_simulation(STEP)

    assert status == 200
    mujoco.set_body_mass.assert_called_once_with(piece_of(world), 2.0)


def test_a_posted_friction_is_given_at_the_next_step(server, controlled, mujoco, world):
    friction = ContactFriction(sliding=0.1, torsional=0.002, rolling=0.0)

    status, _ = edit(
        server, piece_of(world), SimulationChange.FRICTION, asdict(friction)
    )
    controlled.step_simulation(STEP)

    assert status == 200
    mujoco.set_body_friction.assert_called_once_with(piece_of(world), friction)


def test_an_edit_of_a_body_the_simulation_does_not_know_is_refused(server):
    unknown = Body(name=PrefixedName("ghost", prefix="world"))

    status, payload = edit(server, unknown, SimulationChange.MASS, 1.0)

    assert status == 404
    assert payload["ok"] is False


def test_an_edit_the_body_cannot_take_is_refused(server, world):
    status, payload = edit(server, world.root, SimulationChange.MASS, 1.0)

    assert status == 400
    assert payload["ok"] is False


@pytest.mark.parametrize(
    "change, value",
    [
        ("colour", 1.0),
        (SimulationChange.MASS, "heavy"),
        (SimulationChange.POSE, {PoseField.X: 0.1}),
        (SimulationChange.FRICTION, {"grip": 1.0}),
    ],
)
def test_a_malformed_edit_is_refused(server, world, change, value):
    status, payload = edit(server, piece_of(world), change, value)

    assert status == 400
    assert payload["ok"] is False


# %% the visualization


def test_an_attached_simulation_is_offered_to_the_viewer(
    world, controlled, monkeypatch
):
    monkeypatch.setattr(
        visualization_module, "serve", lambda passed_bridge, port: ServerRecorder()
    )
    live = LiveVisualization(world=world, bridge=Bridge()).start()

    live.attach_simulation(controlled)

    assert live.bridge.simulation is controlled
    live.stop()


def test_stopping_the_visualization_withdraws_the_simulation(
    world, controlled, monkeypatch
):
    monkeypatch.setattr(
        visualization_module, "serve", lambda passed_bridge, port: ServerRecorder()
    )
    live = LiveVisualization(world=world, bridge=Bridge()).start()
    live.attach_simulation(controlled)

    live.stop()

    assert live.bridge.simulation is None
