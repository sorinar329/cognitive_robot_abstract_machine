"""
Knowledge about a horizontal-axis wind turbine and the behaviour its controller aims
for.

The model states what the turbine *should* do in each band of wind speed, so that the
behaviour observed in a simulation can be compared against it and explained. All
quantities are in SI units: m, m/s, W, rad and rad/s.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Self

from krrood.adapters.json_serializer import from_json
from krrood.entity_query_language.factories import entity, variable
from krrood.patterns.path_enumeration import PathEnumeration
from krrood.symbol_graph.symbol_graph import Symbol

from experiments.wind_turbine.exceptions import WindSpeedThresholdsOutOfOrderError


# %% actuators and control goals


class Actuator(StrEnum):
    """
    A means by which the turbine controller acts on the turbine.
    """

    GENERATOR_TORQUE = "generator_torque"
    """
    The torque the generator draws from the drivetrain.
    """

    BLADE_PITCH = "blade_pitch"
    """
    The angle of the blades about their long axis.
    """

    YAW_DRIVE = "yaw_drive"
    """
    The drive that turns the nacelle about the tower axis.
    """


class ControlGoal(StrEnum):
    """
    What the turbine controller aims to achieve.
    """

    WAIT_FOR_WIND = "wait_for_wind"
    """
    The wind is too weak to produce power; the rotor is left to idle.
    """

    MAXIMIZE_POWER = "maximize_power"
    """
    Extract as much power as the wind allows.
    """

    LIMIT_TO_RATED_POWER = "limit_to_rated_power"
    """
    Hold rotor speed and power at their rated values.
    """

    ALIGN_WITH_WIND = "align_with_wind"
    """
    Keep the rotor facing into the wind.
    """

    STOP_ROTOR = "stop_rotor"
    """
    Bring the rotor to a standstill to protect the turbine.
    """


class WindRegion(StrEnum):
    """
    A band of hub-height wind speed, bounded by a turbine's cut-in, rated and cut-out
    wind speeds.
    """

    BELOW_CUT_IN = "below_cut_in"
    """
    Too little wind to produce power.
    """

    BELOW_RATED = "below_rated"
    """
    From cut-in up to rated wind speed: power grows with the wind.
    """

    ABOVE_RATED = "above_rated"
    """
    From rated up to cut-out wind speed: power is held at its rated value.
    """

    ABOVE_CUT_OUT = "above_cut_out"
    """
    From cut-out wind speed upwards: the turbine must shut down.
    """


# %% control strategy


@dataclass
class RegionGoal(Symbol):
    """
    The fact that the controller pursues a goal while the wind is in a region.
    """

    region: WindRegion
    """
    The band of wind speed in which the goal is pursued.
    """

    goal: ControlGoal
    """
    The goal pursued in that band.
    """


@dataclass
class GoalActuator(Symbol):
    """
    The fact that the controller pursues a goal by moving an actuator.
    """

    goal: ControlGoal
    """
    The goal that is pursued.
    """

    actuator: Actuator
    """
    The actuator moved to pursue it.
    """


@dataclass(frozen=True)
class ControlStrategy:
    """
    Which goals a turbine controller pursues in each wind region, and with which
    actuators.
    """

    region_goals: list[RegionGoal]
    """
    The goals pursued in each wind region.
    """

    goal_actuators: list[GoalActuator]
    """
    The actuator behind each goal. A goal without an entry is pursued by moving
    nothing.
    """

    def goals_in(self, region: WindRegion) -> list[ControlGoal]:
        """
        :param region: A band of wind speed.
        :return: The goals pursued while the wind is in that band.
        """
        region_goal = variable(RegionGoal, domain=self.region_goals)
        query = entity(region_goal.goal).where(region_goal.region == region)
        return list(query.evaluate())

    def actuator_for(self, goal: ControlGoal) -> Actuator | None:
        """
        :param goal: A goal of the controller.
        :return: The actuator moved to pursue it, or ``None`` if it moves none.
        """
        goal_actuator = variable(GoalActuator, domain=self.goal_actuators)
        query = entity(goal_actuator.actuator).where(goal_actuator.goal == goal)
        return next(iter(query.evaluate()), None)


# %% turbine description


@dataclass(frozen=True)
class TurbineRatings:
    """
    The design point of a turbine and the wind speeds that bound its operation.
    """

    rated_power: float
    """
    Electrical power at and above rated wind speed, in W.
    """

    rated_rotor_speed: float
    """
    Rotor speed at and above rated wind speed, in rad/s.
    """

    cut_in_wind_speed: float
    """
    Hub-height wind speed from which the turbine produces power, in m/s.
    """

    rated_wind_speed: float
    """
    Hub-height wind speed from which the turbine produces its rated power, in m/s.
    """

    cut_out_wind_speed: float
    """
    Hub-height wind speed from which the turbine must shut down, in m/s.
    """

    def __post_init__(self) -> None:
        if not (
            self.cut_in_wind_speed < self.rated_wind_speed < self.cut_out_wind_speed
        ):
            raise WindSpeedThresholdsOutOfOrderError(
                cut_in_wind_speed=self.cut_in_wind_speed,
                rated_wind_speed=self.rated_wind_speed,
                cut_out_wind_speed=self.cut_out_wind_speed,
            )

    def wind_region(self, wind_speed: float) -> WindRegion:
        """
        :param wind_speed: Hub-height wind speed, in m/s.
        :return: The region that wind speed falls in. Each threshold belongs to the
            region above it.
        """
        if wind_speed < self.cut_in_wind_speed:
            return WindRegion.BELOW_CUT_IN
        if wind_speed < self.rated_wind_speed:
            return WindRegion.BELOW_RATED
        if wind_speed < self.cut_out_wind_speed:
            return WindRegion.ABOVE_RATED
        return WindRegion.ABOVE_CUT_OUT


@dataclass(frozen=True)
class TurbineGeometry:
    """
    The main dimensions of a turbine.
    """

    hub_height: float
    """
    Height of the rotor centre above the ground, in m.
    """

    rotor_diameter: float
    """
    Diameter of the circle swept by the blade tips, in m.
    """

    number_of_blades: int
    """
    Number of blades on the rotor.
    """

    gearbox_ratio: float
    """
    Generator speed divided by rotor speed.
    """


@dataclass(frozen=True)
class ControllerLimits:
    """
    The bounds within which the turbine controller moves the actuators.
    """

    minimum_rotor_speed: float
    """
    Lowest rotor speed the controller holds while producing power, in rad/s.
    """

    maximum_rotor_speed: float
    """
    Highest rotor speed the turbine is designed for, in rad/s.
    """

    fine_pitch: float
    """
    Blade pitch for maximum power below rated wind speed, in rad.
    """

    feathered_pitch: float
    """
    Blade pitch at which the blades produce no driving torque, in rad.
    """

    maximum_pitch_rate: float
    """
    Fastest the blades may change pitch, in rad/s.
    """

    yaw_rate: float
    """
    Speed at which the yaw drive turns the nacelle, in rad/s.
    """

    yaw_error_deadband: float
    """
    Misalignment between nacelle and wind below which the controller does not yaw, in
    rad.
    """


@dataclass(frozen=True)
class WindTurbine:
    """
    A horizontal-axis wind turbine: its ratings, its geometry, and the limits and
    strategy of its controller.
    """

    name: str
    """
    The name of the turbine design.
    """

    ratings: TurbineRatings
    """
    The design point and the wind speeds that bound operation.
    """

    geometry: TurbineGeometry
    """
    The main dimensions.
    """

    controller_limits: ControllerLimits
    """
    The bounds within which the controller moves the actuators.
    """

    control_strategy: ControlStrategy
    """
    The goals the controller pursues in each wind region, and the actuators behind them.
    """

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """
        :param path: A JSON file written by :func:`krrood.adapters.json_serializer.to_json`
            from a :class:`WindTurbine`.
        :return: The turbine described by the file.
        """
        return from_json(json.loads(path.read_text()))


class ReferenceTurbineFile(PathEnumeration):
    """
    Published reference turbine designs, as :class:`WindTurbine` JSON files.
    """

    IEA_3_4_130_RWT = Path(__file__).parent / "resources" / "iea_3_4_130_rwt.json"
    """
    The IEA Wind TCP Task 37 3.4 MW land-based reference turbine, with values from its
    windIO definition, power curve and ROSCO tuning.
    """
