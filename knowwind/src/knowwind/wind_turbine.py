"""
knowwind.wind_turbine
~~~~~~~~~~~~~~~~~~~~~
The ``WindTurbine`` class is the analogue of semantic_digital_twin's
``World``.  It acts as the registry / container for all sensor
annotations that belong to one turbine asset.

Usage
-----
>>> from knowwind.wind_turbine import WindTurbine
>>> turbine = WindTurbine(asset_id="WF_A_T01")
>>> turbine.add_annotation(my_annotation)
>>> sensors = turbine.get_annotations_by_type(WindSpeedSensor)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type, TypeVar

from knowwind.semantic_annotations.base import SensorAnnotation

A = TypeVar("A", bound=SensorAnnotation)


class WindTurbine:
    """
    Registry for semantic annotations attached to a single wind turbine.

    Parameters
    ----------
    asset_id:
        Identifier string for this turbine (maps to ``asset_id`` in the CSV).
    wind_farm:
        Optional label for the wind farm (e.g. "A", "B", "C").
    """

    def __init__(self, asset_id: str, wind_farm: Optional[str] = None) -> None:
        self.asset_id = asset_id
        self.wind_farm = wind_farm
        self._annotations: Dict[str, SensorAnnotation] = {}  # column_name → annotation

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_annotation(self, annotation: SensorAnnotation) -> None:
        """Register a sensor annotation.  Replaces any existing entry for the same column."""
        self._annotations[annotation.column_name] = annotation

    def remove_annotation(self, column_name: str) -> None:
        """Remove the annotation for *column_name* if it exists."""
        self._annotations.pop(column_name, None)

    # ------------------------------------------------------------------
    # Querying  (mirrors semantic_digital_twin query patterns)
    # ------------------------------------------------------------------

    def get_all_annotations(self) -> List[SensorAnnotation]:
        return list(self._annotations.values())

    def get_annotations_by_type(self, annotation_type: Type[A]) -> List[A]:
        """Return all annotations that are instances of *annotation_type*."""
        return [a for a in self._annotations.values() if isinstance(a, annotation_type)]

    def get_annotation_by_column(self, column_name: str) -> Optional[SensorAnnotation]:
        return self._annotations.get(column_name)

    def has_column(self, column_name: str) -> bool:
        return column_name in self._annotations

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"WindTurbine(asset_id={self.asset_id!r}, "
            f"wind_farm={self.wind_farm!r}, "
            f"n_annotations={len(self._annotations)})"
        )

    def __len__(self) -> int:
        return len(self._annotations)
