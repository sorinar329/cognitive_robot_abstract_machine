import numpy as np
import pytest

from semantic_digital_twin.semantic_annotations.wind_turbine import (
    WindTurbine,
    WindTurbineGeometry,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection

# %% fixtures


@pytest.fixture
def wind_turbine_geometry() -> WindTurbineGeometry:
    return WindTurbineGeometry(
        tower_height=80.0,
        tower_diameter=5.0,
        tower_top_to_shaft=2.0,
        overhang=-5.0,
        hub_radius=1.5,
        tip_radius=60.0,
        blade_chord=4.0,
        number_of_blades=3,
    )


@pytest.fixture
def wind_turbine(wind_turbine_geometry: WindTurbineGeometry) -> WindTurbine:
    world = World.create_with_root_body()
    return WindTurbine.create_with_new_bodies_in_world(
        "turbine", world, wind_turbine_geometry
    )


def unit_axis(connection: RevoluteConnection) -> np.ndarray:
    return connection.axis.to_np()[:3]


def blade_direction(wind_turbine: WindTurbine, blade_index: int) -> np.ndarray:
    """
    The world direction from the hub towards the tip of one blade.
    """
    wind_turbine._world.update_forward_kinematics()
    world_T_blade = wind_turbine.blades[blade_index].root.global_transform.to_np()
    return world_T_blade[:3, 2]


# %% kinematic structure


def test_yaw_turns_the_nacelle_about_the_vertical_tower_axis(wind_turbine):
    assert isinstance(wind_turbine.yaw_connection, RevoluteConnection)
    assert np.allclose(unit_axis(wind_turbine.yaw_connection), [0.0, 0.0, 1.0])
    assert wind_turbine.yaw_connection.child == wind_turbine.nacelle.root


def test_rotor_turns_the_hub_about_the_downwind_shaft_axis(wind_turbine):
    assert isinstance(wind_turbine.rotor_connection, RevoluteConnection)
    assert np.allclose(unit_axis(wind_turbine.rotor_connection), [1.0, 0.0, 0.0])
    assert wind_turbine.rotor_connection.child == wind_turbine.hub.root


def test_each_blade_pitches_about_its_own_axis_towards_feather(
    wind_turbine, wind_turbine_geometry
):
    assert len(wind_turbine.pitch_connections) == (
        wind_turbine_geometry.number_of_blades
    )
    for blade, connection in zip(wind_turbine.blades, wind_turbine.pitch_connections):
        assert connection.child == blade.root
        assert np.allclose(unit_axis(connection), [0.0, 0.0, -1.0])


def test_hub_sits_at_the_rotor_apex(wind_turbine, wind_turbine_geometry):
    wind_turbine._world.update_forward_kinematics()
    hub_position = wind_turbine.hub.root.global_transform.to_np()[:3, 3]
    assert np.allclose(
        hub_position,
        [
            wind_turbine_geometry.overhang,
            0.0,
            wind_turbine_geometry.tower_height
            + wind_turbine_geometry.tower_top_to_shaft,
        ],
    )


def test_blades_are_spaced_evenly_with_the_first_blade_pointing_up(
    wind_turbine, wind_turbine_geometry
):
    spacing = 2 * np.pi / wind_turbine_geometry.number_of_blades
    for blade_index in range(wind_turbine_geometry.number_of_blades):
        angle = blade_index * spacing
        assert np.allclose(
            blade_direction(wind_turbine, blade_index),
            [0.0, -np.sin(angle), np.cos(angle)],
        )


def test_turning_the_rotor_moves_every_blade(wind_turbine):
    wind_turbine.rotor_connection.position = np.pi / 2
    assert np.allclose(blade_direction(wind_turbine, 0), [0.0, -1.0, 0.0])


def test_parts_are_registered_as_semantic_annotations(wind_turbine):
    world = wind_turbine._world
    assert world.get_semantic_annotations_by_type(WindTurbine) == [wind_turbine]
    assert wind_turbine.nacelle in world.semantic_annotations
    assert wind_turbine.hub in world.semantic_annotations
    assert all(blade in world.semantic_annotations for blade in wind_turbine.blades)


# %% geometry


def test_blade_length_spans_from_the_hub_to_the_tip(wind_turbine_geometry):
    assert wind_turbine_geometry.blade_length == (
        wind_turbine_geometry.tip_radius - wind_turbine_geometry.hub_radius
    )
