from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING, List, Optional, Self

from semantic_digital_twin.api import (
    BodySpecification,
    RevoluteConnectionSpecification,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.geometry import Color, Scale

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% turbine parts


@dataclass(eq=False)
class Tower(HasRootBody):
    """
    The tubular structure that carries the nacelle at hub height.
    """


@dataclass(eq=False)
class Nacelle(HasRootBody):
    """
    The housing on top of the tower that holds the drivetrain and yaws to face the
    wind.
    """


@dataclass(eq=False)
class Hub(HasRootBody):
    """
    The part the blades are mounted on; it turns about the rotor shaft.
    """


@dataclass(eq=False)
class RotorBlade(HasRootBody):
    """
    An aerofoil blade that turns wind into torque on the hub and pitches about its own
    long axis.
    """


# %% geometry


STRUCTURE_COLOR = Color(0.92, 0.93, 0.94)
"""
Color the turbine structure is drawn in.
"""

BLADE_THICKNESS_TO_CHORD = 0.25
"""
Ratio of the drawn blade thickness to the blade chord.

Turbine input data describes a blade by its aerofoil sections, not by one thickness,
so the drawn blade uses this fixed ratio.
"""


@dataclass
class WindTurbineGeometry:
    """
    The dimensions of a horizontal-axis wind turbine, in the frame of its tower base:
    x points downwind when the nacelle is not yawed, z points up.
    """

    tower_height: float
    """
    Height of the tower top above the tower base in meters.
    """

    tower_diameter: float
    """
    Diameter of the tower in meters.
    """

    tower_top_to_shaft: float
    """
    Vertical distance from the tower top to the rotor shaft in meters.
    """

    overhang: float
    """
    Distance from the yaw axis to the rotor apex along the shaft in meters, positive
    downwind. Negative for a rotor upwind of the tower.
    """

    hub_radius: float
    """
    Distance from the rotor apex to the blade root in meters.
    """

    tip_radius: float
    """
    Distance from the rotor apex to the blade tip in meters.
    """

    blade_chord: float
    """
    Largest chord of a blade in meters.
    """

    number_of_blades: int
    """
    How many blades the rotor has.
    """

    @property
    def blade_length(self) -> float:
        """
        Length of one blade from its root to its tip in meters.
        """
        return self.tip_radius - self.hub_radius

    @property
    def nacelle_scale(self) -> Scale:
        """
        Size of the drawn nacelle box: it reaches from the yaw axis to the rotor apex on
        both sides of the tower and is as wide as the tower.
        """
        return Scale(
            2 * abs(self.overhang),
            self.tower_diameter,
            2 * self.tower_top_to_shaft,
        )

    @property
    def blade_scale(self) -> Scale:
        """
        Size of the drawn blade box: chord along y, length along the blade axis z.
        """
        return Scale(
            BLADE_THICKNESS_TO_CHORD * self.blade_chord,
            self.blade_chord,
            self.blade_length,
        )


# %% wind turbine


@dataclass(eq=False)
class WindTurbine(HasRootBody):
    """
    A horizontal-axis wind turbine. Its root is the tower body.

    The nacelle yaws about the tower axis, the hub turns about the downwind shaft axis,
    and each blade pitches about its own axis. Positive yaw, rotor and pitch angles
    follow the sign conventions of OpenFAST: rotor angle 0 has the first blade pointing
    up, and positive pitch turns a blade towards feather.
    """

    tower: Tower = field(kw_only=True)
    """
    The tower carrying the nacelle.
    """

    nacelle: Nacelle = field(kw_only=True)
    """
    The nacelle on top of the tower.
    """

    hub: Hub = field(kw_only=True)
    """
    The hub the blades are mounted on.
    """

    blades: List[RotorBlade] = field(kw_only=True)
    """
    The blades, ordered by blade number.
    """

    @property
    def yaw_connection(self) -> RevoluteConnection:
        """
        The connection that turns the nacelle about the tower axis.
        """
        return self.nacelle.root.parent_connection

    @property
    def rotor_connection(self) -> RevoluteConnection:
        """
        The connection that turns the hub about the rotor shaft.
        """
        return self.hub.root.parent_connection

    @property
    def pitch_connections(self) -> List[RevoluteConnection]:
        """
        The connections that pitch the blades, ordered by blade number.
        """
        return [blade.root.parent_connection for blade in self.blades]

    @classmethod
    def create_with_new_bodies_in_world(
        cls,
        name: str,
        world: World,
        geometry: WindTurbineGeometry,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
    ) -> Self:
        """
        Create the bodies, connections and semantic annotations of a wind turbine.

        :param name: The name of the turbine; part names are derived from it.
        :param world: The world to add the turbine to.
        :param geometry: The dimensions of the turbine.
        :param world_root_T_self: The pose of the tower base in the world root frame.
            Its x axis is the downwind direction of the unyawed nacelle.
        :return: The created wind turbine.
        """
        tower_body = cls._tower_specification(name, geometry).spawn(
            world, parent_T_self=world_root_T_self
        )
        nacelle_body = world.get_body_by_name(cls._nacelle_name(name))
        hub_body = world.get_body_by_name(cls._hub_name(name))
        blade_bodies = [
            world.get_body_by_name(cls._blade_name(name, blade_number))
            for blade_number in cls._blade_numbers(geometry)
        ]
        with world.modify_world():
            tower = Tower(name=PrefixedName(f"{name}_tower"), root=tower_body)
            nacelle = Nacelle(name=PrefixedName(f"{name}_nacelle"), root=nacelle_body)
            hub = Hub(name=PrefixedName(f"{name}_hub"), root=hub_body)
            blades = [RotorBlade(name=body.name, root=body) for body in blade_bodies]
            wind_turbine = cls(
                name=PrefixedName(name),
                root=tower_body,
                tower=tower,
                nacelle=nacelle,
                hub=hub,
                blades=blades,
            )
            for annotation in [tower, nacelle, hub, *blades, wind_turbine]:
                world.add_semantic_annotation(annotation)
        return wind_turbine

    @staticmethod
    def _nacelle_name(name: str) -> str:
        return f"{name}_nacelle"

    @staticmethod
    def _hub_name(name: str) -> str:
        return f"{name}_hub"

    @staticmethod
    def _blade_name(name: str, blade_number: int) -> str:
        return f"{name}_blade_{blade_number}"

    @staticmethod
    def _blade_numbers(geometry: WindTurbineGeometry) -> range:
        return range(1, geometry.number_of_blades + 1)

    @classmethod
    def _tower_specification(
        cls, name: str, geometry: WindTurbineGeometry
    ) -> BodySpecification:
        """
        Specification of the tower with the nacelle, hub and blades below it.
        """
        return BodySpecification.cylinder(
            f"{name}_tower",
            width=geometry.tower_diameter,
            height=geometry.tower_height,
            color=STRUCTURE_COLOR,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=geometry.tower_height / 2
            ),
            child_specifications=[cls._nacelle_specification(name, geometry)],
        )

    @classmethod
    def _nacelle_specification(
        cls, name: str, geometry: WindTurbineGeometry
    ) -> BodySpecification:
        """
        Specification of the nacelle, yawing about the tower axis on the tower top.
        """
        return BodySpecification.box(
            cls._nacelle_name(name),
            scale=geometry.nacelle_scale,
            color=STRUCTURE_COLOR,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=geometry.tower_top_to_shaft
            ),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=geometry.tower_height
            ),
            child_specifications=[cls._hub_specification(name, geometry)],
            connection_specification=RevoluteConnectionSpecification(axis=Vector3.Z()),
        )

    @classmethod
    def _hub_specification(
        cls, name: str, geometry: WindTurbineGeometry
    ) -> BodySpecification:
        """
        Specification of the hub at the rotor apex, turning about the downwind shaft
        axis.
        """
        return BodySpecification.sphere(
            cls._hub_name(name),
            radius=geometry.hub_radius,
            color=STRUCTURE_COLOR,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=geometry.overhang, z=geometry.tower_top_to_shaft
            ),
            child_specifications=[
                cls._blade_specification(name, geometry, blade_number)
                for blade_number in cls._blade_numbers(geometry)
            ],
            connection_specification=RevoluteConnectionSpecification(axis=Vector3.X()),
        )

    @classmethod
    def _blade_specification(
        cls, name: str, geometry: WindTurbineGeometry, blade_number: int
    ) -> BodySpecification:
        """
        Specification of one blade. Blade ``k`` sits ``(k - 1) * 360 / n`` degrees
        further along the direction of rotor rotation than the first blade, and pitches
        about the negative z axis of its own frame, whose z axis points to the tip.
        """
        angle_after_first_blade = (
            (blade_number - 1) * 2 * np.pi / geometry.number_of_blades
        )
        return BodySpecification.box(
            cls._blade_name(name, blade_number),
            scale=geometry.blade_scale,
            color=STRUCTURE_COLOR,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=geometry.hub_radius + geometry.blade_length / 2
            ),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                roll=angle_after_first_blade
            ),
            connection_specification=RevoluteConnectionSpecification(
                axis=Vector3.NEGATIVE_Z()
            ),
        )
