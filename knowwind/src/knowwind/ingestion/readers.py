"""
knowwind.ingestion.readers
~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads the three file types that make up a knowwind dataset:

  1. ``feature_description.csv``  →  ``SensorCatalogue``
  2. ``event_info.csv``           →  ``EventRegistry``
  3. ``<event_id>.csv``           →  ``pandas.DataFrame`` (annotated via pipeline)

All readers accept both ``str`` and ``pathlib.Path`` arguments.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import pandas as pd

from knowwind.semantic.model import (
    SensorAnnotation,
    SensorCatalogue,
    StatisticType,
)


# ---------------------------------------------------------------------------
# feature_description.csv  →  SensorCatalogue
# ---------------------------------------------------------------------------

# Map the raw CSV statistics_type strings to our enum
_STAT_MAP: dict[str, StatisticType] = {
    "average":           StatisticType.AVERAGE,
    "avg":               StatisticType.AVERAGE,
    "minimum":           StatisticType.MINIMUM,
    "min":               StatisticType.MINIMUM,
    "maximum":           StatisticType.MAXIMUM,
    "max":               StatisticType.MAXIMUM,
    "std_dev":           StatisticType.STD_DEV,
    "standard deviation":StatisticType.STD_DEV,
    "stddev":            StatisticType.STD_DEV,
}

# Suffix appended to sensor_name to form the column name for each stat type
_STAT_SUFFIX: dict[StatisticType, str] = {
    StatisticType.AVERAGE: "_avg",
    StatisticType.MINIMUM: "_min",
    StatisticType.MAXIMUM: "_max",
    StatisticType.STD_DEV: "_std",
}


def read_feature_description(
    path: str | Path,
    farm_id: str = "unknown",
    *,
    separator: str = ";",
) -> SensorCatalogue:
    """Parse a ``feature_description.csv`` into a ``SensorCatalogue``.

    The CSV uses semicolons and may contain a multi-value ``statistics_type``
    column (comma-separated, e.g. ``"average,minimum,maximum,std_dev"``).
    One ``SensorAnnotation`` is created per (sensor × statistic_type) pair.

    Parameters
    ----------
    path:
        Path to ``feature_description.csv``.
    farm_id:
        Identifier for the wind farm (A / B / C).
    separator:
        Field separator in the CSV (default ``";"``).
    """
    path = Path(path)
    catalogue = SensorCatalogue(farm_id=farm_id)

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=separator)
        # Normalise header names (lowercase, strip whitespace)
        reader.fieldnames = [f.strip().lower() for f in (reader.fieldnames or [])]

        for row in reader:
            sensor_name   = row.get("sensor_name", "").strip()
            stat_raw      = row.get("statistics_type", row.get("statistic_type", "average")).strip()
            description   = row.get("description", "").strip()
            unit          = _clean_unit(row.get("unit", "").strip())
            is_angle      = row.get("is_angle", "False").strip().lower() == "true"
            is_counter    = row.get("is_counter", "False").strip().lower() == "true"

            if not sensor_name:
                continue

            # stat_raw may be a comma-separated list
            stat_tokens = [s.strip().lower() for s in stat_raw.split(",") if s.strip()]

            for token in stat_tokens:
                stat_type = _STAT_MAP.get(token)
                if stat_type is None:
                    continue  # unknown aggregation – skip silently

                suffix      = _STAT_SUFFIX[stat_type]
                column_name = f"{sensor_name}{suffix}"

                annotation = SensorAnnotation(
                    sensor_name    = sensor_name,
                    column_name    = column_name,
                    statistic_type = stat_type,
                    description    = description,
                    unit           = unit,
                    is_angle       = is_angle,
                    is_counter     = is_counter,
                )
                catalogue.add(annotation)

    return catalogue


# ---------------------------------------------------------------------------
# event_info.csv  →  EventRegistry
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class EventInfo:
    event_id:          str
    event_label:       str                   # "anomaly" | "normal"
    event_start:       str
    event_end:         str
    event_start_id:    int | None = None
    event_end_id:      int | None = None
    event_description: str = ""

    @property
    def is_anomaly(self) -> bool:
        return self.event_label.strip().lower() == "anomaly"


@dataclass
class EventRegistry:
    farm_id: str = "unknown"
    events:  dict[str, EventInfo] = field(default_factory=dict)

    def get(self, event_id: str) -> EventInfo | None:
        return self.events.get(event_id)

    def anomalies(self) -> list[EventInfo]:
        return [e for e in self.events.values() if e.is_anomaly]

    def normals(self) -> list[EventInfo]:
        return [e for e in self.events.values() if not e.is_anomaly]

    def __len__(self) -> int:
        return len(self.events)


def read_event_info(
    path: str | Path,
    farm_id: str = "unknown",
    *,
    separator: str = ",",
) -> EventRegistry:
    """Parse ``event_info.csv`` into an ``EventRegistry``.

    The column separator for event_info files is a regular comma (not
    semicolon), but can be overridden if needed.
    """
    path     = Path(path)
    registry = EventRegistry(farm_id=farm_id)

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=separator)
        reader.fieldnames = [f.strip().lower() for f in (reader.fieldnames or [])]

        for row in reader:
            event_id = row.get("event_id", "").strip()
            if not event_id:
                continue

            def _int_or_none(v: str) -> int | None:
                try:
                    return int(v)
                except (ValueError, TypeError):
                    return None

            info = EventInfo(
                event_id          = event_id,
                event_label       = row.get("event_label", "normal").strip(),
                event_start       = row.get("event_start", "").strip(),
                event_end         = row.get("event_end", "").strip(),
                event_start_id    = _int_or_none(row.get("event_start_id", "")),
                event_end_id      = _int_or_none(row.get("event_end_id", "")),
                event_description = row.get("event_description", "").strip(),
            )
            registry.events[event_id] = info

    return registry


# ---------------------------------------------------------------------------
# Dataset CSV  →  pd.DataFrame
# ---------------------------------------------------------------------------

def read_dataset(
    path: str | Path,
    *,
    separator: str = ",",
    parse_timestamps: bool = True,
) -> pd.DataFrame:
    """Load a single event dataset CSV into a ``pandas.DataFrame``.

    Parameters
    ----------
    path:
        Path to the dataset CSV (e.g. ``wind_farm_A/datasets/event_001.csv``).
    separator:
        Field separator (default ``","``).
    parse_timestamps:
        If ``True`` attempts to coerce the ``time_stamp`` column to
        ``datetime64``.  Silently ignores parse errors.
    """
    df = pd.read_csv(Path(path), sep=separator, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if parse_timestamps and "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Convenience: iterate over all dataset files in a farm directory
# ---------------------------------------------------------------------------

def iter_datasets(datasets_dir: str | Path) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(event_id, DataFrame)`` for every CSV in *datasets_dir*."""
    for csv_path in sorted(Path(datasets_dir).glob("*.csv")):
        event_id = csv_path.stem
        yield event_id, read_dataset(csv_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_unit(raw: str) -> str:
    """Normalise common encoding artefacts in unit strings."""
    DEG = "\u00b0"
    replacements = {
        "\ufffd":  DEG,
        "\u00c2\u00b0": DEG,
    }
    for bad, good in replacements.items():
        raw = raw.replace(bad, good)
    return raw
