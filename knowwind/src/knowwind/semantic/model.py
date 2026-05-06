"""
knowwind.semantic.model
~~~~~~~~~~~~~~~~~~~~~~~
Pydantic models that carry the *meaning* of every sensor column:
physical quantity, unit, statistical aggregation type, turbine subsystem,
and whether the value is an angle or an accumulating counter.

These annotations travel with the data through the pipeline and can be
serialised to / loaded from JSON so a catalogue built once can be reused
across wind farms.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

class StatisticType(str, Enum):
    """10-minute aggregation level available for a signal."""
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    STD_DEV = "std_dev"


class SensorRole(str, Enum):
    """High-level physical role of the measured quantity."""
    TEMPERATURE     = "temperature"
    SPEED           = "speed"
    POWER           = "power"
    REACTIVE_POWER  = "reactive_power"
    CURRENT         = "current"
    VOLTAGE         = "voltage"
    FREQUENCY       = "frequency"
    ANGLE           = "angle"
    DIRECTION       = "direction"
    ENERGY          = "energy"
    COUNTER         = "counter"
    OTHER           = "other"


class SubSystem(str, Enum):
    """Turbine sub-system the sensor belongs to."""
    AMBIENT         = "ambient"
    ROTOR           = "rotor"
    GEARBOX         = "gearbox"
    GENERATOR       = "generator"
    CONVERTER       = "converter"
    TRANSFORMER     = "transformer"
    HYDRAULICS      = "hydraulics"
    NACELLE         = "nacelle"
    GRID            = "grid"
    CONTROL         = "control"
    UNKNOWN         = "unknown"


# ---------------------------------------------------------------------------
# Per-column annotation
# ---------------------------------------------------------------------------

class SensorAnnotation(BaseModel):
    """Semantic annotation attached to one physical sensor signal.

    A *sensor* in this dataset may appear as multiple CSV columns, one per
    ``StatisticType`` (avg / min / max / std).  This model describes the
    underlying sensor; ``column_name`` identifies the specific column.
    """

    # Identity
    sensor_name:   str  = Field(description="Anonymised sensor base-name (e.g. 'wind_speed_3')")
    column_name:   str  = Field(description="Exact CSV column identifier (e.g. 'wind_speed_3_avg')")
    statistic_type: StatisticType

    # Human-readable semantics
    description:   str  = Field(default="", description="Short plain-English description")
    unit:          str  = Field(default="",  description="SI or domain unit string (e.g. '°C', 'm/s')")

    # Semantic flags
    role:          SensorRole = SensorRole.OTHER
    subsystem:     SubSystem  = SubSystem.UNKNOWN
    is_angle:      bool = False
    is_counter:    bool = False

    # Optional range hints for anomaly detection / normalisation
    expected_min:  Optional[float] = None
    expected_max:  Optional[float] = None

    @model_validator(mode="after")
    def _derive_role_and_subsystem(self) -> "SensorAnnotation":
        """Auto-fill role / subsystem from unit + description when not set."""
        if self.role == SensorRole.OTHER:
            self.role = _infer_role(self.unit, self.description, self.is_angle, self.is_counter)
        if self.subsystem == SubSystem.UNKNOWN:
            self.subsystem = _infer_subsystem(self.sensor_name, self.description)
        return self

    # Convenience
    def is_average(self) -> bool:
        return self.statistic_type == StatisticType.AVERAGE

    def __repr__(self) -> str:
        return (
            f"SensorAnnotation({self.column_name!r}, "
            f"role={self.role.value}, unit={self.unit!r}, "
            f"subsystem={self.subsystem.value})"
        )


# ---------------------------------------------------------------------------
# Catalogue: all annotations for one wind farm
# ---------------------------------------------------------------------------

class SensorCatalogue(BaseModel):
    """All ``SensorAnnotation`` objects for a single wind farm dataset.

    Keyed by exact CSV column name for O(1) lookup during pipeline execution.
    """

    farm_id: str = Field(default="unknown")
    annotations: dict[str, SensorAnnotation] = Field(default_factory=dict)

    # ---- Construction helpers ----

    def add(self, annotation: SensorAnnotation) -> None:
        self.annotations[annotation.column_name] = annotation

    def get(self, column_name: str) -> Optional[SensorAnnotation]:
        return self.annotations.get(column_name)

    def by_subsystem(self, subsystem: SubSystem) -> list[SensorAnnotation]:
        return [a for a in self.annotations.values() if a.subsystem == subsystem]

    def by_role(self, role: SensorRole) -> list[SensorAnnotation]:
        return [a for a in self.annotations.values() if a.role == role]

    def average_columns(self) -> list[str]:
        """Return column names for average-type sensors only."""
        return [
            col for col, ann in self.annotations.items()
            if ann.statistic_type == StatisticType.AVERAGE
        ]

    def angle_columns(self) -> list[str]:
        return [col for col, ann in self.annotations.items() if ann.is_angle]

    # ---- Serialisation ----

    def to_json(self, path: str) -> None:
        import json
        with open(path, "w") as fh:
            json.dump(self.model_dump(), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "SensorCatalogue":
        import json
        with open(path) as fh:
            return cls.model_validate(json.load(fh))

    def __len__(self) -> int:
        return len(self.annotations)

    def __repr__(self) -> str:
        return f"SensorCatalogue(farm={self.farm_id!r}, sensors={len(self)})"


# ---------------------------------------------------------------------------
# Heuristic inference helpers (kept private to this module)
# ---------------------------------------------------------------------------

def _infer_role(unit: str, description: str, is_angle: bool, is_counter: bool) -> SensorRole:
    desc_lower  = description.lower()
    unit_lower  = unit.lower()

    if is_counter:
        return SensorRole.COUNTER
    if is_angle or "direction" in desc_lower:
        return SensorRole.DIRECTION if "direction" in desc_lower else SensorRole.ANGLE
    if unit_lower in ("°c", "c", "degc"):
        return SensorRole.TEMPERATURE
    if unit_lower in ("m/s",):
        return SensorRole.SPEED
    if unit_lower in ("rpm",):
        return SensorRole.SPEED
    if unit_lower in ("kw", "w", "wh"):
        return SensorRole.POWER if unit_lower in ("kw", "w") else SensorRole.ENERGY
    if unit_lower in ("kvar", "varh", "kvarh"):
        return SensorRole.REACTIVE_POWER
    if unit_lower in ("a",):
        return SensorRole.CURRENT
    if unit_lower in ("v",):
        return SensorRole.VOLTAGE
    if unit_lower in ("hz",):
        return SensorRole.FREQUENCY
    if "temperature" in desc_lower or "temp" in desc_lower:
        return SensorRole.TEMPERATURE
    if "power" in desc_lower:
        return SensorRole.REACTIVE_POWER if "reactive" in desc_lower else SensorRole.POWER
    if "speed" in desc_lower or "rpm" in desc_lower:
        return SensorRole.SPEED
    if "current" in desc_lower:
        return SensorRole.CURRENT
    if "voltage" in desc_lower:
        return SensorRole.VOLTAGE
    if "frequency" in desc_lower:
        return SensorRole.FREQUENCY
    return SensorRole.OTHER


def _infer_subsystem(sensor_name: str, description: str) -> SubSystem:
    name  = sensor_name.lower()
    desc  = description.lower()
    text  = f"{name} {desc}"

    if any(k in text for k in ("wind", "ambient", "temperature", "outside")) and \
       "gearbox" not in text and "generator" not in text:
        # ambient / met mast sensors
        if "wind" in text or "ambient" in text or "nacelle temp" not in text:
            if "wind" in text:
                return SubSystem.AMBIENT

    if "gearbox" in text:
        return SubSystem.GEARBOX
    if "generator" in text or "stator" in text or "rotor" in text and "side" in text:
        return SubSystem.GENERATOR
    if "pitch" in text or "rotor rpm" in text or "hub" in text:
        return SubSystem.ROTOR
    if "igbt" in text or "inverter" in text or "converter" in text or "vcs" in text or "vcp" in text:
        return SubSystem.CONVERTER
    if "transformer" in text or "hv " in text:
        return SubSystem.TRANSFORMER
    if "hydraulic" in text:
        return SubSystem.HYDRAULICS
    if "nacelle" in text:
        return SubSystem.NACELLE
    if "grid" in text or "phase" in text or "voltage" in text or "current" in text or "frequency" in text:
        return SubSystem.GRID
    if "ambient" in text or "wind speed" in text or "wind direction" in text:
        return SubSystem.AMBIENT
    if "controller" in text or "control" in text:
        return SubSystem.CONTROL

    return SubSystem.UNKNOWN
