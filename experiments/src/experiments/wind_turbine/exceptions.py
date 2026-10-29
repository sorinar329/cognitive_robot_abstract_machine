"""
Exceptions for the wind turbine experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


# %% turbine description


@dataclass
class WindSpeedThresholdsOutOfOrderError(DataclassException):
    """
    Raised when a turbine's cut-in, rated and cut-out wind speeds do not increase in
    that order.
    """

    cut_in_wind_speed: float
    """
    The cut-in wind speed that was given, in m/s.
    """

    rated_wind_speed: float
    """
    The rated wind speed that was given, in m/s.
    """

    cut_out_wind_speed: float
    """
    The cut-out wind speed that was given, in m/s.
    """

    def error_message(self) -> str:
        return (
            f"Wind speed thresholds must increase from cut-in to rated to cut-out, got "
            f"cut-in {self.cut_in_wind_speed} m/s, rated {self.rated_wind_speed} m/s, "
            f"cut-out {self.cut_out_wind_speed} m/s."
        )

    def suggest_correction(self) -> str:
        return "Check the turbine's ratings against its reference documentation."
