"""
knowwind.pipeline.annotator
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``AnnotationPipeline`` takes a raw SCADA ``DataFrame`` plus a
``SensorCatalogue`` and returns an ``AnnotatedDataset``:

  * Every recognised sensor column is mapped to its ``SensorAnnotation``.
  * Angle columns are wrapped with their unit (degrees / radians) for
    downstream consumers that need to handle circular statistics.
  * An optional ``StatusFilter`` can restrict rows to desired operational states.
  * Unknown columns (``id``, ``asset_id``, ``time_stamp``, ``status_type``,
    ``train_test``) are forwarded unchanged.

The pipeline is intentionally side-effect-free: the input DataFrame is never
mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from knowwind.semantic.model import (
    SensorAnnotation,
    SensorCatalogue,
    SensorRole,
    StatisticType,
    SubSystem,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedDataset:
    """Output of ``AnnotationPipeline.run()``.

    Attributes
    ----------
    data:
        The (optionally filtered) DataFrame.  Column order is preserved.
    catalogue:
        The ``SensorCatalogue`` used during annotation.
    recognised_columns:
        Sensor columns that were matched to a ``SensorAnnotation``.
    unknown_columns:
        Columns present in the CSV but not in the catalogue (metadata columns
        like ``id``, ``time_stamp``, etc. are always listed here).
    farm_id:
        Identifier of the source wind farm.
    event_id:
        Identifier of the event / dataset file, if known.
    """

    data:                pd.DataFrame
    catalogue:           SensorCatalogue
    recognised_columns:  list[str]         = field(default_factory=list)
    unknown_columns:     list[str]         = field(default_factory=list)
    farm_id:             str               = "unknown"
    event_id:            str               = "unknown"

    # ---- Convenience accessors ----

    def sensor_data(self) -> pd.DataFrame:
        """Return only the recognised sensor columns."""
        return self.data[self.recognised_columns]

    def columns_for_subsystem(self, subsystem: SubSystem) -> list[str]:
        wanted = {a.column_name for a in self.catalogue.by_subsystem(subsystem)}
        return [c for c in self.recognised_columns if c in wanted]

    def columns_for_role(self, role: SensorRole) -> list[str]:
        wanted = {a.column_name for a in self.catalogue.by_role(role)}
        return [c for c in self.recognised_columns if c in wanted]

    def annotation(self, column: str) -> Optional[SensorAnnotation]:
        return self.catalogue.get(column)

    def train_split(self) -> pd.DataFrame:
        if "train_test" in self.data.columns:
            return self.data[self.data["train_test"] == "train"]
        return self.data

    def test_split(self) -> pd.DataFrame:
        if "train_test" in self.data.columns:
            return self.data[self.data["train_test"] == "test"]
        return self.data

    def summary(self) -> dict:
        return {
            "farm_id":           self.farm_id,
            "event_id":          self.event_id,
            "rows":              len(self.data),
            "recognised_sensors": len(self.recognised_columns),
            "unknown_columns":   self.unknown_columns,
            "subsystems":        sorted({
                a.subsystem.value
                for a in self.catalogue.annotations.values()
                if a.column_name in self.recognised_columns
            }),
        }

    def __repr__(self) -> str:
        return (
            f"AnnotatedDataset(event={self.event_id!r}, "
            f"rows={len(self.data)}, "
            f"sensors={len(self.recognised_columns)})"
        )


# ---------------------------------------------------------------------------
# Status filter
# ---------------------------------------------------------------------------

# The dataset README defines these status codes:
#   0 = Normal Operation, 1 = Derated, 2 = Idling,
#   3 = Service, 4 = Downtime, 5 = Other
NORMAL_STATUS_CODES = (0, 2)


class StatusFilter:
    """Filter a DataFrame to only include rows matching given status codes.

    Parameters
    ----------
    keep_codes:
        Iterable of integer status codes to retain.
        Defaults to ``(0, 2)`` — normal operation + idling.
    status_column:
        Name of the status column (default ``"status_type"``).
    """

    def __init__(
        self,
        keep_codes: tuple[int, ...] = NORMAL_STATUS_CODES,
        status_column: str = "status_type",
    ) -> None:
        self.keep_codes    = set(keep_codes)
        self.status_column = status_column

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.status_column not in df.columns:
            return df
        mask = df[self.status_column].isin(self.keep_codes)
        return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Angle normaliser
# ---------------------------------------------------------------------------

class AngleNormalizer:
    """Ensure angle columns live in [0, 360) or [-180, 180).

    Parameters
    ----------
    mode:
        ``"360"`` (default) wraps to [0, 360).
        ``"180"`` wraps to (-180, 180].
    """

    def __init__(self, mode: str = "360") -> None:
        if mode not in ("360", "180"):
            raise ValueError(f"mode must be '360' or '180', got {mode!r}")
        self.mode = mode

    def apply(self, df: pd.DataFrame, angle_columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in angle_columns:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if self.mode == "360":
                df[col] = series % 360
            else:
                df[col] = ((series + 180) % 360) - 180
        return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Metadata columns that are always forwarded unchanged
_META_COLUMNS = frozenset(
    {"id", "asset_id", "time_stamp", "status_type", "train_test", "status_type_id"}
)


class AnnotationPipeline:
    """Attach semantic metadata to raw SCADA DataFrames.

    Parameters
    ----------
    catalogue:
        The ``SensorCatalogue`` built from ``feature_description.csv``.
    status_filter:
        Optional ``StatusFilter``.  Pass ``None`` to skip status filtering.
    angle_normalizer:
        Optional ``AngleNormalizer``.  Pass ``None`` to skip angle wrapping.
    """

    def __init__(
        self,
        catalogue:         SensorCatalogue,
        status_filter:     Optional[StatusFilter]    = None,
        angle_normalizer:  Optional[AngleNormalizer] = None,
    ) -> None:
        self.catalogue        = catalogue
        self.status_filter    = status_filter
        self.angle_normalizer = angle_normalizer

    # ------------------------------------------------------------------ run

    def run(
        self,
        df: pd.DataFrame,
        *,
        farm_id:  str = "unknown",
        event_id: str = "unknown",
    ) -> AnnotatedDataset:
        """Annotate *df* and return an ``AnnotatedDataset``.

        Steps
        -----
        1. Classify each column as recognised sensor / metadata / unknown.
        2. Apply ``StatusFilter`` if configured.
        3. Apply ``AngleNormalizer`` to angle columns if configured.
        4. Return ``AnnotatedDataset``.
        """
        recognised: list[str] = []
        unknown:    list[str] = []

        for col in df.columns:
            if col in _META_COLUMNS:
                continue  # always keep, never annotate
            if self.catalogue.get(col) is not None:
                recognised.append(col)
            else:
                unknown.append(col)

        # --- status filter ---
        processed = df
        if self.status_filter is not None:
            processed = self.status_filter.apply(processed)

        # --- angle normalisation ---
        if self.angle_normalizer is not None:
            angle_cols = [c for c in recognised if self._is_angle(c)]
            if angle_cols:
                processed = self.angle_normalizer.apply(processed, angle_cols)

        return AnnotatedDataset(
            data               = processed,
            catalogue          = self.catalogue,
            recognised_columns = recognised,
            unknown_columns    = unknown,
            farm_id            = farm_id,
            event_id           = event_id,
        )

    # ---------------------------------------------------------------- helpers

    def _is_angle(self, column: str) -> bool:
        ann = self.catalogue.get(column)
        return ann is not None and ann.is_angle

    @classmethod
    def default(cls, catalogue: SensorCatalogue) -> "AnnotationPipeline":
        """Factory: pipeline with status filter (normal + idling) and 360° angle normalisation."""
        return cls(
            catalogue        = catalogue,
            status_filter    = StatusFilter(),
            angle_normalizer = AngleNormalizer(mode="360"),
        )
