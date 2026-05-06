"""
knowwind.semantic_annotations.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Base classes for sensor semantic annotations, mirroring
semantic_digital_twin's SemanticAnnotation / WorldEntity pattern
but adapted for time-series / SCADA data rather than 3-D worlds.

Design principles
-----------------
* Annotations are plain Python *dataclasses* – no hidden magic.
* Every annotation carries a ``sensor_name`` that ties it back to the
  raw CSV column.
* Annotations are registered inside a ``WindTurbine`` (the analogue of
  semantic_digital_twin's ``World``).
* ``eq=False`` on all annotation dataclasses so the UUID-based hash
  defined here is never accidentally overridden.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from knowwind.datastructures import SensorName, PhysicalUnit, StatisticType

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Base annotation
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class SensorAnnotation:
    """
    Abstract base for all sensor semantic annotations.

    Parameters
    ----------
    sensor_name:
        The unique identifier of the sensor this annotation is attached to.
    description:
        Human-readable description (copied from feature_description.csv).
    unit:
        Physical unit of the measurement.
    is_angle:
        Whether the signal represents an angle (wrap-around aware).
    is_counter:
        Whether the signal represents a monotonically increasing counter.
    annotation_id:
        Auto-generated UUID – do *not* set this manually.
    """

    sensor_name: SensorName
    description: str = ""
    unit: PhysicalUnit = PhysicalUnit.DIMENSIONLESS
    is_angle: bool = False
    is_counter: bool = False
    annotation_id: uuid.UUID = field(default_factory=uuid.uuid4, init=False, repr=False)

    def __hash__(self) -> int:  # noqa: D401
        return hash(self.annotation_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SensorAnnotation):
            return NotImplemented
        return self.annotation_id == other.annotation_id

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def column_name(self) -> str:
        """Return the CSV column name this annotation maps to."""
        return self.sensor_name.column_name
