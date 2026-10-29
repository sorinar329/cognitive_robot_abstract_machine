"""
Tests for the wind turbine knowledge model: wind regions, the control strategy and the
reference turbine file.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from typing_extensions import Callable

from krrood.adapters.json_serializer import from_json, to_json

from experiments.wind_turbine.exceptions import WindSpeedThresholdsOutOfOrderError
from experiments.wind_turbine.turbine import (
    Actuator,
    ControlGoal,
    ControlStrategy,
    GoalActuator,
    ReferenceTurbineFile,
    RegionGoal,
    TurbineRatings,
    WindRegion,
    WindTurbine,
)


# %% fixtures


@pytest.fixture
def reference_turbine() -> WindTurbine:
    """
    The IEA 3.4 MW reference turbine, read from its knowledge file.
    """
    return WindTurbine.from_file(ReferenceTurbineFile.IEA_3_4_130_RWT)


# %% reference turbine file


def test_reference_turbine_file_round_trips(reference_turbine: WindTurbine):
    round_tripped = from_json(json.loads(json.dumps(to_json(reference_turbine))))
    assert round_tripped == reference_turbine


# %% wind regions


@pytest.mark.parametrize(
    "threshold_of, region_at_threshold, region_just_below",
    [
        (
            lambda ratings: ratings.cut_in_wind_speed,
            WindRegion.BELOW_RATED,
            WindRegion.BELOW_CUT_IN,
        ),
        (
            lambda ratings: ratings.rated_wind_speed,
            WindRegion.ABOVE_RATED,
            WindRegion.BELOW_RATED,
        ),
        (
            lambda ratings: ratings.cut_out_wind_speed,
            WindRegion.ABOVE_CUT_OUT,
            WindRegion.ABOVE_RATED,
        ),
    ],
    ids=["cut_in", "rated", "cut_out"],
)
def test_each_threshold_belongs_to_the_region_above_it(
    reference_turbine: WindTurbine,
    threshold_of: Callable[[TurbineRatings], float],
    region_at_threshold: WindRegion,
    region_just_below: WindRegion,
):
    ratings = reference_turbine.ratings
    wind_speed = threshold_of(ratings)
    assert ratings.wind_region(wind_speed) == region_at_threshold
    assert ratings.wind_region(wind_speed - 1e-6) == region_just_below


def test_calm_wind_is_below_cut_in(reference_turbine: WindTurbine):
    assert reference_turbine.ratings.wind_region(0.0) == WindRegion.BELOW_CUT_IN


def test_wind_speed_thresholds_out_of_order_are_rejected(
    reference_turbine: WindTurbine,
):
    ratings = reference_turbine.ratings
    with pytest.raises(WindSpeedThresholdsOutOfOrderError):
        dataclasses.replace(ratings, rated_wind_speed=ratings.cut_out_wind_speed)


def test_turbine_ratings_reject_equal_cut_in_and_rated_wind_speed():
    with pytest.raises(WindSpeedThresholdsOutOfOrderError):
        TurbineRatings(
            rated_power=1.0,
            rated_rotor_speed=1.0,
            cut_in_wind_speed=5.0,
            rated_wind_speed=5.0,
            cut_out_wind_speed=25.0,
        )


# %% control strategy of the reference turbine


@pytest.mark.parametrize(
    "region, goals",
    [
        (WindRegion.BELOW_CUT_IN, [ControlGoal.WAIT_FOR_WIND]),
        (
            WindRegion.BELOW_RATED,
            [ControlGoal.MAXIMIZE_POWER, ControlGoal.ALIGN_WITH_WIND],
        ),
        (
            WindRegion.ABOVE_RATED,
            [ControlGoal.LIMIT_TO_RATED_POWER, ControlGoal.ALIGN_WITH_WIND],
        ),
        (WindRegion.ABOVE_CUT_OUT, [ControlGoal.STOP_ROTOR]),
    ],
)
def test_reference_goals_per_wind_region(
    reference_turbine: WindTurbine, region: WindRegion, goals: list[ControlGoal]
):
    assert reference_turbine.control_strategy.goals_in(region) == goals


def test_reference_turbine_yaws_only_while_producing_power(
    reference_turbine: WindTurbine,
):
    strategy = reference_turbine.control_strategy
    regions_that_align = {
        region
        for region in WindRegion
        if ControlGoal.ALIGN_WITH_WIND in strategy.goals_in(region)
    }
    assert regions_that_align == {WindRegion.BELOW_RATED, WindRegion.ABOVE_RATED}


@pytest.mark.parametrize(
    "goal, actuator",
    [
        (ControlGoal.WAIT_FOR_WIND, None),
        (ControlGoal.MAXIMIZE_POWER, Actuator.GENERATOR_TORQUE),
        (ControlGoal.LIMIT_TO_RATED_POWER, Actuator.BLADE_PITCH),
        (ControlGoal.ALIGN_WITH_WIND, Actuator.YAW_DRIVE),
        (ControlGoal.STOP_ROTOR, Actuator.BLADE_PITCH),
    ],
)
def test_reference_actuator_per_goal(
    reference_turbine: WindTurbine, goal: ControlGoal, actuator: Actuator | None
):
    assert reference_turbine.control_strategy.actuator_for(goal) == actuator


# %% control strategy as data


def test_goals_follow_the_strategy_facts():
    keep_yawing_when_stopped = ControlStrategy(
        region_goals=[
            RegionGoal(WindRegion.ABOVE_CUT_OUT, ControlGoal.STOP_ROTOR),
            RegionGoal(WindRegion.ABOVE_CUT_OUT, ControlGoal.ALIGN_WITH_WIND),
        ],
        goal_actuators=[],
    )
    assert keep_yawing_when_stopped.goals_in(WindRegion.ABOVE_CUT_OUT) == [
        ControlGoal.STOP_ROTOR,
        ControlGoal.ALIGN_WITH_WIND,
    ]


def test_region_without_facts_has_no_goals():
    strategy = ControlStrategy(region_goals=[], goal_actuators=[])
    assert strategy.goals_in(WindRegion.ABOVE_RATED) == []


def test_actuator_follows_the_strategy_facts():
    stop_by_torque = ControlStrategy(
        region_goals=[],
        goal_actuators=[
            GoalActuator(ControlGoal.STOP_ROTOR, Actuator.GENERATOR_TORQUE)
        ],
    )
    assert (
        stop_by_torque.actuator_for(ControlGoal.STOP_ROTOR) == Actuator.GENERATOR_TORQUE
    )
