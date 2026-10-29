import numpy as np
import pytest

from semantic_digital_twin.semantic_annotations.wind_farm import (
    NREL_5MW_OC3_MONOPILE,
    OC3_MONOPILE,
    OffshoreSubstation,
    OffshoreWindFarmLayout,
    Sea,
    WindFarm,
)
from semantic_digital_twin.semantic_annotations.wind_turbine import WindTurbine
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures


@pytest.fixture
def layout() -> OffshoreWindFarmLayout:
    return OffshoreWindFarmLayout(
        turbine_model=NREL_5MW_OC3_MONOPILE,
        foundation=OC3_MONOPILE,
        turbine_poses=[
            HomogeneousTransformationMatrix.from_xyz_rpy(y=y) for y in (-900, 0, 900)
        ],
        substation_pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=500),
        substation_topside=Scale(30.0, 20.0, 12.0),
    )


@pytest.fixture
def wind_farm(layout: OffshoreWindFarmLayout) -> WindFarm:
    world = World.create_with_root_body()
    return WindFarm.create_offshore_with_new_bodies_in_world("farm", world, layout)


def world_bounds(body: Body) -> tuple[np.ndarray, np.ndarray]:
    """
    The lowest and highest corner of the box around a body's collision geometry, in the
    world frame.
    """
    world = body._world
    world.update_forward_kinematics()
    bounds = (
        body.collision.as_bounding_box_collection_in_frame(world.root)
        .bounding_box()
        .to_array_bounds()
    )
    return bounds.lower, bounds.upper


def position_of(body: Body) -> np.ndarray:
    body._world.update_forward_kinematics()
    return body.global_transform.to_np()[:3, 3]


def layout_position(pose: HomogeneousTransformationMatrix) -> np.ndarray:
    return pose.to_np()[:3, 3]


# %% turbines


def test_farm_holds_one_turbine_per_layout_position(wind_farm, layout):
    world = wind_farm._world
    assert len(wind_farm.turbines) == len(layout.turbine_poses)
    assert world.get_semantic_annotations_by_type(WindTurbine) == wind_farm.turbines


def test_every_turbine_is_the_layout_model(wind_farm, layout):
    assert all(turbine.model is layout.turbine_model for turbine in wind_farm.turbines)


def test_each_turbine_stands_on_its_own_foundation(wind_farm):
    foundations = [turbine.foundation for turbine in wind_farm.turbines]
    assert len(set(id(foundation) for foundation in foundations)) == len(foundations)
    for turbine in wind_farm.turbines:
        assert turbine.tower.root.parent_connection.parent == (
            turbine.foundation.transition_piece
        )


def test_tower_base_stands_at_platform_height_above_its_layout_position(
    wind_farm, layout
):
    for turbine, pose in zip(wind_farm.turbines, layout.turbine_poses):
        expected = layout_position(pose) + [0.0, 0.0, layout.foundation.platform_height]
        assert np.allclose(position_of(turbine.tower.root), expected)


# %% foundations


def test_monopile_reaches_from_the_seabed_to_sea_level(wind_farm, layout):
    lower, upper = world_bounds(wind_farm.turbines[0].foundation.root)
    assert lower[2] == pytest.approx(-layout.foundation.water_depth)
    assert upper[2] == pytest.approx(0.0)


def test_transition_piece_reaches_from_sea_level_to_the_platform(wind_farm, layout):
    lower, upper = world_bounds(wind_farm.turbines[0].foundation.transition_piece)
    assert lower[2] == pytest.approx(0.0)
    assert upper[2] == pytest.approx(layout.foundation.platform_height)


def test_foundation_is_as_wide_as_the_monopile(wind_farm, layout):
    lower, upper = world_bounds(wind_farm.turbines[0].foundation.root)
    assert upper[0] - lower[0] == pytest.approx(layout.foundation.diameter)


# %% sea


def test_sea_reaches_from_the_seabed_to_sea_level(wind_farm, layout):
    lower, upper = world_bounds(wind_farm.sea.root)
    assert lower[2] == pytest.approx(-layout.foundation.water_depth)
    assert upper[2] == pytest.approx(0.0)


def test_sea_covers_every_structure_with_a_rotor_diameter_to_spare(wind_farm, layout):
    lower, upper = world_bounds(wind_farm.sea.root)
    rotor_diameter = 2 * layout.turbine_model.geometry.tip_radius
    for pose in [*layout.turbine_poses, layout.substation_pose]:
        position = layout_position(pose)
        assert np.all(position[:2] - rotor_diameter >= lower[:2] - 1e-9)
        assert np.all(position[:2] + rotor_diameter <= upper[:2] + 1e-9)


def test_sea_is_registered_as_a_semantic_annotation(wind_farm):
    assert wind_farm._world.get_semantic_annotations_by_type(Sea) == [wind_farm.sea]


# %% substation


def test_substation_stands_on_its_own_foundation(wind_farm, layout):
    substation = wind_farm.substation
    assert substation.root.parent_connection.parent == (
        substation.foundation.transition_piece
    )
    lower, _ = world_bounds(substation.root)
    assert lower[2] == pytest.approx(layout.foundation.platform_height)


def test_substation_topside_has_the_layout_size(wind_farm, layout):
    lower, upper = world_bounds(wind_farm.substation.root)
    topside = layout.substation_topside
    assert np.allclose(upper - lower, [topside.x, topside.y, topside.z])


def test_substation_starts_without_a_power_demand(wind_farm):
    assert wind_farm.substation.power_demand == 0.0


def test_substation_is_registered_as_a_semantic_annotation(wind_farm):
    assert wind_farm._world.get_semantic_annotations_by_type(OffshoreSubstation) == [
        wind_farm.substation
    ]


# %% farm


def test_farm_is_registered_as_a_semantic_annotation(wind_farm):
    assert wind_farm._world.get_semantic_annotations_by_type(WindFarm) == [wind_farm]
