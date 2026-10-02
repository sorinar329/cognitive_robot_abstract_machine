from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

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
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

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


@dataclass(eq=False)
class Anemometer(HasRootBody):
    """
    The wind sensor on the nacelle roof. It holds the latest wind it measured.
    """

    wind_speed: float = field(default=0.0, kw_only=True)
    """
    Measured wind speed in meters per second.
    """

    wind_direction: float = field(default=0.0, kw_only=True)
    """
    Measured direction the wind blows towards, in radians, counterclockwise about the
    world z axis from the world x axis.
    """


# %% geometry


STRUCTURE_COLOR = Color(0.92, 0.93, 0.94)
"""
Color the turbine structure is drawn in.
"""

MONOPILE_COLOR = Color(0.35, 0.37, 0.4)
"""
Color the monopile is drawn in.
"""

TRANSITION_PIECE_COLOR = Color(0.98, 0.78, 0.0)
"""
Color the transition piece is drawn in: the yellow offshore transition pieces are
painted in.
"""

FEATHERED_PITCH = np.pi / 2
"""
Blade pitch angle in radians at which the blade chord lies along the wind, so the rotor
produces no torque.
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


@dataclass
class TurbineRating:
    """
    The power and the wind speeds a turbine is designed to operate at.
    """

    rated_power: float
    """
    Electrical power in watts the turbine produces at and above its rated wind speed.
    """

    cut_in_wind_speed: float
    """
    Wind speed in meters per second from which the turbine produces power.
    """

    rated_wind_speed: float
    """
    Wind speed in meters per second at which the turbine reaches its rated power.
    """

    cut_out_wind_speed: float
    """
    Wind speed in meters per second from which the turbine must shut down.
    """


@dataclass
class TurbineModel:
    """
    A type of wind turbine: what every turbine built to it looks like and is rated for.
    """

    name: str
    """
    The name the model is known by.
    """

    geometry: WindTurbineGeometry
    """
    The dimensions of the turbine.
    """

    rating: TurbineRating
    """
    The power and wind speeds the turbine is designed for.
    """


# %% operation


class OperationalState(Enum):
    """
    The phase of operation a turbine reports.
    """

    PARKED = auto()
    """
    Blades feathered, brake engaged, generator disconnected.
    """

    STARTING_UP = auto()
    """
    Yawing into the wind, releasing the brake, pitching the blades in and connecting
    the generator.
    """

    PRODUCING = auto()
    """
    Feeding power into the grid.
    """

    SHUTTING_DOWN = auto()
    """
    Feathering the blades, disconnecting the generator and engaging the brake.
    """


@dataclass
class TurbineStatus:
    """
    What a turbine reports about itself, apart from the angles of its connections.
    """

    operational_state: OperationalState = OperationalState.PARKED
    """
    The phase of operation the turbine is in.
    """

    brake_engaged: bool = True
    """
    Whether the rotor brake holds the rotor.
    """

    generator_connected: bool = False
    """
    Whether the generator feeds the grid.
    """

    electrical_power: float = 0.0
    """
    Electrical power in watts the turbine feeds into the grid.
    """

    in_maintenance: bool = False
    """
    Whether the turbine is taken out of operation for maintenance.
    """

    faulted: bool = False
    """
    Whether the turbine reports a fault.
    """


@dataclass
class TurbineSetpoints:
    """
    What a turbine is commanded to do. The defaults keep it parked.
    """

    yaw: float = 0.0
    """
    Nacelle yaw angle in radians to turn to.
    """

    pitch: float = FEATHERED_PITCH
    """
    Blade pitch angle in radians to turn every blade to.
    """

    brake_engaged: bool = True
    """
    Whether the rotor brake should hold the rotor.
    """

    generator_connected: bool = False
    """
    Whether the generator should feed the grid.
    """


# %% offshore foundation


@dataclass
class MonopileGeometry:
    """
    The dimensions of a monopile foundation, in the frame of the sea level directly
    above it: z points up.
    """

    water_depth: float
    """
    Depth of the seabed below mean sea level in meters.
    """

    diameter: float
    """
    Diameter of the monopile and the transition piece in meters.
    """

    platform_height: float
    """
    Height of the transition piece top above mean sea level in meters, where the tower
    or topside stands.
    """


@dataclass(eq=False)
class OffshoreFoundation(HasRootBody):
    """
    A monopile driven into the seabed with a transition piece on top that carries a
    turbine tower or a topside. Its root is the monopile, whose frame lies on the
    seabed.
    """

    transition_piece: Body = field(kw_only=True)
    """
    The part above the sea between the monopile and what it carries. Its frame lies on
    its top.
    """

    @classmethod
    def create_with_new_bodies_in_world(
        cls,
        name: str,
        world: World,
        geometry: MonopileGeometry,
        world_root_T_sea_level: Optional[HomogeneousTransformationMatrix] = None,
    ) -> Self:
        """
        Create the bodies and the semantic annotation of a monopile foundation.

        :param name: The name of the foundation; part names are derived from it.
        :param world: The world to add the foundation to.
        :param geometry: The dimensions of the foundation.
        :param world_root_T_sea_level: The pose of the mean sea level point directly
            above the monopile, in the world root frame.
        :return: The created foundation.
        """
        sea_level_T_seabed = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=-geometry.water_depth
        )
        world_root_T_seabed = (
            sea_level_T_seabed
            if world_root_T_sea_level is None
            else world_root_T_sea_level @ sea_level_T_seabed
        )
        monopile_body = cls._monopile_specification(name, geometry).spawn(
            world, parent_T_self=world_root_T_seabed
        )
        with world.modify_world():
            foundation = cls(
                name=PrefixedName(name),
                root=monopile_body,
                transition_piece=world.get_body_by_name(
                    cls._transition_piece_name(name)
                ),
            )
            world.add_semantic_annotation(foundation)
        return foundation

    @staticmethod
    def _transition_piece_name(name: str) -> str:
        return f"{name}_transition_piece"

    @classmethod
    def _monopile_specification(
        cls, name: str, geometry: MonopileGeometry
    ) -> BodySpecification:
        """
        Specification of the monopile from the seabed to the sea level, with the
        transition piece on top of it.
        """
        return BodySpecification.cylinder(
            f"{name}_monopile",
            width=geometry.diameter,
            height=geometry.water_depth,
            color=MONOPILE_COLOR,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=geometry.water_depth / 2
            ),
            child_specifications=[
                BodySpecification.cylinder(
                    cls._transition_piece_name(name),
                    width=geometry.diameter,
                    height=geometry.platform_height,
                    color=TRANSITION_PIECE_COLOR,
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=-geometry.platform_height / 2
                    ),
                    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=geometry.water_depth + geometry.platform_height
                    ),
                )
            ],
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

    anemometer: Anemometer = field(kw_only=True)
    """
    The wind sensor on the nacelle roof.
    """

    model: TurbineModel = field(kw_only=True)
    """
    The type of turbine this is.
    """

    foundation: Optional[OffshoreFoundation] = field(default=None, kw_only=True)
    """
    The offshore foundation the turbine stands on, or None for a turbine on land.
    """

    status: TurbineStatus = field(default_factory=TurbineStatus, kw_only=True)
    """
    What the turbine reports about itself.
    """

    setpoints: TurbineSetpoints = field(default_factory=TurbineSetpoints, kw_only=True)
    """
    What the turbine is commanded to do.
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
        model: TurbineModel,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
    ) -> Self:
        """
        Create the bodies, connections and semantic annotations of a parked wind
        turbine on land.

        :param name: The name of the turbine; part names are derived from it.
        :param world: The world to add the turbine to.
        :param model: The type of turbine to build.
        :param world_root_T_self: The pose of the tower base in the world root frame.
            Its x axis is the downwind direction of the unyawed nacelle.
        :return: The created wind turbine.
        """
        return cls._create_with_new_bodies_in_world(
            name, world, model, world.root, world_root_T_self, foundation=None
        )

    @classmethod
    def create_offshore_with_new_bodies_in_world(
        cls,
        name: str,
        world: World,
        model: TurbineModel,
        foundation_geometry: MonopileGeometry,
        world_root_T_sea_level: Optional[HomogeneousTransformationMatrix] = None,
    ) -> Self:
        """
        Create a parked wind turbine standing on a new monopile foundation.

        :param name: The name of the turbine; part names are derived from it.
        :param world: The world to add the turbine to.
        :param model: The type of turbine to build.
        :param foundation_geometry: The dimensions of the foundation.
        :param world_root_T_sea_level: The pose of the mean sea level point directly
            above the monopile, in the world root frame. Its x axis is the downwind
            direction of the unyawed nacelle.
        :return: The created wind turbine.
        """
        foundation = OffshoreFoundation.create_with_new_bodies_in_world(
            f"{name}_foundation", world, foundation_geometry, world_root_T_sea_level
        )
        return cls._create_with_new_bodies_in_world(
            name,
            world,
            model,
            foundation.transition_piece,
            HomogeneousTransformationMatrix(),
            foundation=foundation,
        )

    @classmethod
    def _create_with_new_bodies_in_world(
        cls,
        name: str,
        world: World,
        model: TurbineModel,
        parent: KinematicStructureEntity,
        parent_T_self: Optional[HomogeneousTransformationMatrix],
        foundation: Optional[OffshoreFoundation],
    ) -> Self:
        """
        Create a parked wind turbine whose tower base is attached to ``parent``.
        """
        geometry = model.geometry
        tower_body = cls._tower_specification(name, geometry).spawn(
            world, parent=parent, parent_T_self=parent_T_self
        )
        nacelle_body = world.get_body_by_name(cls._nacelle_name(name))
        hub_body = world.get_body_by_name(cls._hub_name(name))
        anemometer_body = world.get_body_by_name(cls._anemometer_name(name))
        blade_bodies = [
            world.get_body_by_name(cls._blade_name(name, blade_number))
            for blade_number in cls._blade_numbers(geometry)
        ]
        with world.modify_world():
            tower = Tower(name=PrefixedName(f"{name}_tower"), root=tower_body)
            nacelle = Nacelle(name=PrefixedName(f"{name}_nacelle"), root=nacelle_body)
            hub = Hub(name=PrefixedName(f"{name}_hub"), root=hub_body)
            blades = [RotorBlade(name=body.name, root=body) for body in blade_bodies]
            anemometer = Anemometer(name=anemometer_body.name, root=anemometer_body)
            wind_turbine = cls(
                name=PrefixedName(name),
                root=tower_body,
                tower=tower,
                nacelle=nacelle,
                hub=hub,
                blades=blades,
                anemometer=anemometer,
                model=model,
                foundation=foundation,
            )
            for annotation in [tower, nacelle, hub, *blades, anemometer, wind_turbine]:
                world.add_semantic_annotation(annotation)
        for pitch_connection in wind_turbine.pitch_connections:
            pitch_connection.position = wind_turbine.setpoints.pitch
        return wind_turbine

    @staticmethod
    def _nacelle_name(name: str) -> str:
        return f"{name}_nacelle"

    @staticmethod
    def _hub_name(name: str) -> str:
        return f"{name}_hub"

    @staticmethod
    def _anemometer_name(name: str) -> str:
        return f"{name}_anemometer"

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
            child_specifications=[
                cls._hub_specification(name, geometry),
                cls._anemometer_specification(name, geometry),
            ],
            connection_specification=RevoluteConnectionSpecification(axis=Vector3.Z()),
        )

    @classmethod
    def _anemometer_specification(
        cls, name: str, geometry: WindTurbineGeometry
    ) -> BodySpecification:
        """
        Specification of the anemometer on the nacelle roof, on the yaw axis. It is not
        drawn.
        """
        return BodySpecification(
            name=cls._anemometer_name(name),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=2 * geometry.tower_top_to_shaft
            ),
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
