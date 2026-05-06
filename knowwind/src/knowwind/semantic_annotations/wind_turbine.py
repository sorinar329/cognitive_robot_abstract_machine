"""
knowwind.semantic_annotations.wind_turbine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Concrete semantic annotations for the wind-turbine SCADA domain.

Each class represents a *kind of physical meaning* – just as
semantic_digital_twin has ``Handle``, ``Container``, ``Drawer``, here we
have ``WindSpeedSensor``, ``TemperatureSensor``, ``PowerSensor``, etc.

Adding a new annotation type:
    1. Subclass ``SensorAnnotation``.
    2. Add ``@dataclass(eq=False)`` decorator.
    3. Optionally add domain-specific fields.
    4. Register it in the ``ANNOTATION_TYPE_MAP`` at the bottom so the
       auto-importer can resolve it from a feature_description.csv row.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowwind.semantic_annotations.base import SensorAnnotation
from knowwind.datastructures import SensorName, PhysicalUnit


# ---------------------------------------------------------------------------
# Meteorological / environmental
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class AmbientTemperatureSensor(SensorAnnotation):
    """Outdoor / ambient temperature measurement."""


@dataclass(eq=False)
class WindSpeedSensor(SensorAnnotation):
    """Anemometer or estimated wind-speed measurement."""


@dataclass(eq=False)
class WindDirectionSensor(SensorAnnotation):
    """Wind direction (absolute or relative to nacelle)."""


# ---------------------------------------------------------------------------
# Mechanical / rotational
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class RotorRpmSensor(SensorAnnotation):
    """Rotor / main-shaft angular velocity."""


@dataclass(eq=False)
class GeneratorRpmSensor(SensorAnnotation):
    """Generator-shaft angular velocity."""


@dataclass(eq=False)
class PitchAngleSensor(SensorAnnotation):
    """Blade-pitch angle (always an angle sensor)."""


@dataclass(eq=False)
class NacelleDirectionSensor(SensorAnnotation):
    """Nacelle yaw direction relative to North."""


# ---------------------------------------------------------------------------
# Electrical – power
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class ActivePowerSensor(SensorAnnotation):
    """Active (real) power measurement."""


@dataclass(eq=False)
class ReactivePowerSensor(SensorAnnotation):
    """Reactive power measurement."""


@dataclass(eq=False)
class GridFrequencySensor(SensorAnnotation):
    """Grid frequency measurement."""


@dataclass(eq=False)
class CurrentSensor(SensorAnnotation):
    """Phase current measurement."""
    phase: int = 0   # 1, 2, or 3; 0 = unknown


@dataclass(eq=False)
class VoltageSensor(SensorAnnotation):
    """Phase voltage measurement."""
    phase: int = 0


# ---------------------------------------------------------------------------
# Thermal – drivetrain & electronics
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class GearboxTemperatureSensor(SensorAnnotation):
    """Temperature sensor located in / around the gearbox."""


@dataclass(eq=False)
class GeneratorTemperatureSensor(SensorAnnotation):
    """Temperature sensor located in / on the generator."""


@dataclass(eq=False)
class TransformerTemperatureSensor(SensorAnnotation):
    """Temperature sensor on the HV/MV transformer."""


@dataclass(eq=False)
class InverterTemperatureSensor(SensorAnnotation):
    """Temperature on IGBT or other power-electronics components."""


@dataclass(eq=False)
class NacelleTemperatureSensor(SensorAnnotation):
    """General nacelle interior temperature."""


@dataclass(eq=False)
class HydraulicTemperatureSensor(SensorAnnotation):
    """Temperature in hydraulic oil circuits."""


@dataclass(eq=False)
class ControllerTemperatureSensor(SensorAnnotation):
    """Temperature inside a PLC / controller enclosure."""


@dataclass(eq=False)
class GenericTemperatureSensor(SensorAnnotation):
    """Catch-all for temperature sensors not covered above."""


# ---------------------------------------------------------------------------
# Generic fall-back
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class GenericSensor(SensorAnnotation):
    """
    Generic sensor annotation used when no domain-specific type can be
    inferred from the description text.
    """


# ---------------------------------------------------------------------------
# Keyword → annotation type mapping (used by the CSV importer)
# ---------------------------------------------------------------------------

# List of (keyword_set, annotation_class) pairs.
# The importer checks description.lower() for *any* keyword in the set.
# Order matters: more specific rules first.
KEYWORD_TO_ANNOTATION: list[tuple[frozenset[str], type[SensorAnnotation]]] = [
    # Meteorological
    (frozenset(["ambient temperature"]),        AmbientTemperatureSensor),
    (frozenset(["windspeed", "wind speed", "estimated windspeed"]), WindSpeedSensor),
    (frozenset(["wind absolute direction", "wind relative direction"]), WindDirectionSensor),

    # Mechanical / rotational
    (frozenset(["rotor rpm"]),                  RotorRpmSensor),
    (frozenset(["generator rpm"]),              GeneratorRpmSensor),
    (frozenset(["pitch angle"]),                PitchAngleSensor),
    (frozenset(["nacelle direction"]),          NacelleDirectionSensor),

    # Power
    (frozenset(["active power", "grid power"]),  ActivePowerSensor),
    (frozenset(["reactive power"]),             ReactivePowerSensor),
    (frozenset(["grid frequency"]),             GridFrequencySensor),
    (frozenset(["current in phase"]),           CurrentSensor),
    (frozenset(["voltage in phase"]),           VoltageSensor),

    # Thermal – specific first
    (frozenset(["gearbox"]),                    GearboxTemperatureSensor),
    (frozenset(["generator bearing", "generator in stator", "stator windings"]), GeneratorTemperatureSensor),
    (frozenset(["hv transformer", "transformer"]), TransformerTemperatureSensor),
    (frozenset(["igbt", "inverter"]),           InverterTemperatureSensor),
    (frozenset(["nacelle temperature"]),        NacelleTemperatureSensor),
    (frozenset(["hydraulic"]),                  HydraulicTemperatureSensor),
    (frozenset(["controller", "hub controller", "nacelle controller"]), ControllerTemperatureSensor),
    # Generic temperature catch-all
    (frozenset(["temperature"]),                GenericTemperatureSensor),
]


def infer_annotation_type(description: str) -> type[SensorAnnotation]:
    """Return the most specific annotation class matching *description*."""
    desc_lower = description.lower()
    for keywords, cls in KEYWORD_TO_ANNOTATION:
        if any(kw in desc_lower for kw in keywords):
            return cls
    return GenericSensor
