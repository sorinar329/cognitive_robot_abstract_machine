"""
knowwind.datastructures
~~~~~~~~~~~~~~~~~~~~~~~
Foundational value types used throughout the knowwind package.

Inspired by semantic_digital_twin's PrefixedName / types pattern, adapted for
wind-turbine SCADA sensor semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Statistic types
# ---------------------------------------------------------------------------

class StatisticType(str, Enum):
    """Aggregation kind stored in a SCADA column."""
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    STD_DEV = "std_dev"

    @classmethod
    def from_csv_token(cls, token: str) -> "StatisticType":
        token = token.strip().lower()
        mapping = {
            "average": cls.AVERAGE,
            "avg": cls.AVERAGE,
            "minimum": cls.MINIMUM,
            "min": cls.MINIMUM,
            "maximum": cls.MAXIMUM,
            "max": cls.MAXIMUM,
            "std_dev": cls.STD_DEV,
            "stddev": cls.STD_DEV,
            "standard deviation": cls.STD_DEV,
        }
        if token not in mapping:
            raise ValueError(f"Unknown statistic type token: '{token}'")
        return mapping[token]


# ---------------------------------------------------------------------------
# Physical unit
# ---------------------------------------------------------------------------

class PhysicalUnit(str, Enum):
    """Physical units appearing in SCADA feature descriptions."""
    CELSIUS       = "°C"
    DEGREE        = "°"
    METER_PER_SEC = "m/s"
    RPM           = "rpm"
    KILOWATT      = "kW"
    WATT_HOUR     = "Wh"
    KVAR          = "kVAr"
    VARH          = "VArh"
    AMPERE        = "A"
    VOLT          = "V"
    HERTZ         = "Hz"
    DIMENSIONLESS = ""

    @classmethod
    def from_string(cls, s: str) -> "PhysicalUnit":
        s = s.strip()
        for member in cls:
            if member.value == s:
                return member
        # Graceful fallback – keep the raw string as DIMENSIONLESS
        return cls.DIMENSIONLESS


# ---------------------------------------------------------------------------
# Sensor identifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorName:
    """Unique identifier for a sensor column, analogous to PrefixedName."""
    base_name: str          # e.g. "wind_speed_3" or "sensor_18"
    statistic: StatisticType = StatisticType.AVERAGE

    @property
    def column_name(self) -> str:
        """Return the CSV column name suffix, e.g. 'wind_speed_3_avg'."""
        suffix_map = {
            StatisticType.AVERAGE: "avg",
            StatisticType.MINIMUM: "min",
            StatisticType.MAXIMUM: "max",
            StatisticType.STD_DEV: "std",
        }
        return f"{self.base_name}_{suffix_map[self.statistic]}"

    def __str__(self) -> str:
        return self.column_name
