from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING, List, Self

from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.wind_turbine import (
    MonopileGeometry,
    OffshoreFoundation,
    TurbineModel,
    TurbineRating,
    WindTurbine,
    WindTurbineGeometry,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

# %% reference designs


NREL_5MW_OC3_MONOPILE = TurbineModel(
    name="NREL 5 MW reference turbine on the OC3 monopile",
    geometry=WindTurbineGeometry(
        tower_height=77.6,
        tower_diameter=6.0,
        tower_top_to_shaft=1.96256,
        overhang=-5.0191,
        hub_radius=1.5,
        tip_radius=63.0,
        blade_chord=4.652,
        number_of_blades=3,
    ),
    rating=TurbineRating(
        rated_power=5.0e6,
        cut_in_wind_speed=3.0,
        rated_wind_speed=11.4,
        cut_out_wind_speed=25.0,
    ),
)
"""
The NREL 5 MW reference turbine (Jonkman et al., NREL/TP-500-38060, 2009) as the
OpenFAST regression test ``5MW_OC3Mnpl_DLL_WTurb_WavesIrr`` places it on the OC3
monopile: the tower reaches from 10 m to 87.6 m above mean sea level and is drawn with
its 6 m base diameter.
"""

OC3_MONOPILE = MonopileGeometry(water_depth=20.0, diameter=6.0, platform_height=10.0)
"""
The OC3 monopile of the same regression test: 20 m water depth, 6 m diameter, and the
tower base 10 m above mean sea level.
"""

SEA_COLOR = Color(0.11, 0.33, 0.52, 0.75)
"""
Color the sea is drawn in, translucent so the monopiles show through.
"""

SUBSTATION_COLOR = Color(0.82, 0.84, 0.86)
"""
Color the substation topside is drawn in.
"""

# %% sea and substation


@dataclass(eq=False)
class Sea(HasRootBody):
    """
    The water of an offshore site, from the seabed to mean sea level.

    Its root frame lies on the sea surface.
    """

    @classmethod
    def create_with_new_bodies_in_world(
        cls,
        name: str,
        world: World,
        extent: Scale,
        world_root_T_self: HomogeneousTransformationMatrix,
    ) -> Self:
        """
        Create the body and the semantic annotation of the sea.

        :param name: The name of the sea.
        :param world: The world to add the sea to.
        :param extent: The size of the water: its x and y span the surface, its z is the
            water depth.
        :param world_root_T_self: The pose of the middle of the sea surface in the world
            root frame.
        :return: The created sea.
        """
        body = BodySpecification.box(
            name,
            scale=extent,
            color=SEA_COLOR,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=-extent.z / 2),
        ).spawn(world, parent_T_self=world_root_T_self)
        with world.modify_world():
            sea = cls(name=PrefixedName(name), root=body)
            world.add_semantic_annotation(sea)
        return sea


@dataclass(eq=False)
class OffshoreSubstation(HasRootBody):
    """
    The platform that collects the power of the wind farm and passes it on to the grid.

    Its root is the topside, whose frame lies on the bottom of the topside.
    """

    foundation: OffshoreFoundation = field(kw_only=True)
    """
    The foundation the topside stands on.
    """

    power_demand: float = field(default=0.0, kw_only=True)
    """
    Electrical power in watts the grid currently asks the wind farm for.
    """

    @classmethod
    def create_with_new_bodies_in_world(
        cls,
        name: str,
        world: World,
        topside: Scale,
        foundation_geometry: MonopileGeometry,
        world_root_T_sea_level: HomogeneousTransformationMatrix,
    ) -> Self:
        """
        Create a substation topside standing on a new monopile foundation.

        :param name: The name of the substation; part names are derived from it.
        :param world: The world to add the substation to.
        :param topside: The size of the topside.
        :param foundation_geometry: The dimensions of the foundation.
        :param world_root_T_sea_level: The pose of the mean sea level point directly
            above the monopile, in the world root frame.
        :return: The created substation.
        """
        foundation = OffshoreFoundation.create_with_new_bodies_in_world(
            f"{name}_foundation", world, foundation_geometry, world_root_T_sea_level
        )
        topside_body = BodySpecification.box(
            f"{name}_topside",
            scale=topside,
            color=SUBSTATION_COLOR,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=topside.z / 2),
        ).spawn(world, parent=foundation.transition_piece)
        with world.modify_world():
            substation = cls(
                name=PrefixedName(name), root=topside_body, foundation=foundation
            )
            world.add_semantic_annotation(substation)
        return substation


# %% wind farm


@dataclass
class OffshoreWindFarmLayout:
    """
    Where the structures of an offshore wind farm stand and what they are built to.
    """

    turbine_model: TurbineModel
    """
    The type of every turbine in the farm.
    """

    foundation: MonopileGeometry
    """
    The foundation every turbine and the substation stand on.
    """

    turbine_poses: List[HomogeneousTransformationMatrix]
    """
    For each turbine, the pose of the mean sea level point above its monopile, in the
    world root frame.

    Its x axis is the downwind direction of the unyawed nacelle.
    """

    substation_pose: HomogeneousTransformationMatrix
    """
    The pose of the mean sea level point above the substation's monopile, in the world
    root frame.
    """

    substation_topside: Scale
    """
    The size of the substation topside.
    """

    def structure_positions(self) -> np.ndarray:
        """
        :return: The sea level positions of every turbine and the substation, one row
            each.
        """
        return np.array(
            [
                pose.to_np()[:3, 3]
                for pose in [*self.turbine_poses, self.substation_pose]
            ]
        )


@dataclass(eq=False)
class WindFarm(SemanticAnnotation):
    """
    Wind turbines that feed the grid together through one substation.
    """

    turbines: List[WindTurbine] = field(kw_only=True)
    """
    The turbines of the farm, in the order of the layout.
    """

    substation: OffshoreSubstation = field(kw_only=True)
    """
    The substation the turbines feed.
    """

    sea: Sea = field(kw_only=True)
    """
    The sea the farm stands in.
    """

    @classmethod
    def create_offshore_with_new_bodies_in_world(
        cls, name: str, world: World, layout: OffshoreWindFarmLayout
    ) -> Self:
        """
        Create an offshore wind farm: its turbines and substation on monopiles, and the
        sea around them, which reaches one rotor diameter beyond every structure.

        :param name: The name of the farm; part names are derived from it.
        :param world: The world to add the farm to.
        :param layout: Where the structures stand and what they are built to.
        :return: The created wind farm.
        """
        sea = cls._create_sea(name, world, layout)
        turbines = [
            WindTurbine.create_offshore_with_new_bodies_in_world(
                f"{name}_turbine_{turbine_number}",
                world,
                layout.turbine_model,
                layout.foundation,
                pose,
            )
            for turbine_number, pose in enumerate(layout.turbine_poses, start=1)
        ]
        substation = OffshoreSubstation.create_with_new_bodies_in_world(
            f"{name}_substation",
            world,
            layout.substation_topside,
            layout.foundation,
            layout.substation_pose,
        )
        with world.modify_world():
            wind_farm = cls(
                name=PrefixedName(name),
                turbines=turbines,
                substation=substation,
                sea=sea,
            )
            world.add_semantic_annotation(wind_farm)
        return wind_farm

    @staticmethod
    def _create_sea(name: str, world: World, layout: OffshoreWindFarmLayout) -> Sea:
        """
        Create the sea under every structure of ``layout``, with one rotor diameter of
        water beyond the outermost ones.
        """
        positions = layout.structure_positions()
        margin = 2 * layout.turbine_model.geometry.tip_radius
        lower = positions[:, :2].min(axis=0) - margin
        upper = positions[:, :2].max(axis=0) + margin
        middle = (lower + upper) / 2
        span = upper - lower
        return Sea.create_with_new_bodies_in_world(
            f"{name}_sea",
            world,
            Scale(span[0], span[1], layout.foundation.water_depth),
            HomogeneousTransformationMatrix.from_xyz_rpy(x=middle[0], y=middle[1]),
        )
