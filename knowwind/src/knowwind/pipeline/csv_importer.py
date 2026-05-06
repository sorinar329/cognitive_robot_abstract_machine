"""
knowwind.pipeline.csv_importer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads a ``feature_description.csv`` (as described in the README) and
populates a :class:`~knowwind.wind_turbine.WindTurbine` with the
appropriate semantic annotations.

It also provides :func:`annotate_scada_dataframe` which takes a raw
SCADA ``pandas.DataFrame`` (event CSV) and attaches a ``__annotations__``
attribute to each column Series – making the semantic meaning directly
inspectable during analysis.

Example
-------
>>> from knowwind.pipeline.csv_importer import (
...     load_feature_description,
...     annotate_scada_dataframe,
... )
>>> turbine = load_feature_description("feature_description.csv", asset_id="T01")
>>> df = pd.read_csv("event_42.csv", sep=";")
>>> annotated_df = annotate_scada_dataframe(df, turbine)
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from knowwind.datastructures import PhysicalUnit, SensorName, StatisticType
from knowwind.semantic_annotations.base import SensorAnnotation
from knowwind.semantic_annotations.wind_turbine import infer_annotation_type
from knowwind.wind_turbine import WindTurbine


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STAT_TOKEN_RE = re.compile(r"[,\s]+")


def _parse_statistic_types(raw: str) -> List[StatisticType]:
    tokens = [t.strip() for t in _STAT_TOKEN_RE.split(raw) if t.strip()]
    result = []
    for token in tokens:
        try:
            result.append(StatisticType.from_csv_token(token))
        except ValueError:
            pass  # skip unknown tokens
    return result or [StatisticType.AVERAGE]


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_feature_description(
    csv_path: Union[str, Path],
    asset_id: str = "unknown",
    wind_farm: Optional[str] = None,
    separator: str = ";",
) -> WindTurbine:
    """
    Parse *csv_path* (a ``feature_description.csv``) and build a
    :class:`WindTurbine` whose annotations reflect the semantic meaning of
    every sensor column.

    Parameters
    ----------
    csv_path:
        Path to the feature_description CSV file.
    asset_id:
        Turbine identifier to attach to the returned WindTurbine.
    wind_farm:
        Optional wind-farm label (e.g. "A", "B", "C").
    separator:
        CSV field separator (default ``";"``).

    Returns
    -------
    WindTurbine
        Populated with one :class:`~knowwind.semantic_annotations.base.SensorAnnotation`
        per (sensor_name × statistic_type) combination.
    """
    turbine = WindTurbine(asset_id=asset_id, wind_farm=wind_farm)

    path = Path(csv_path)
    with path.open(encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter=separator)
        for row in reader:
            sensor_name_base = row.get("sensor_name", "").strip()
            stats_raw        = row.get("statistics_type", row.get("statistic_type", "average"))
            description      = row.get("description", "").strip()
            unit_raw         = row.get("unit", "").strip()
            is_angle         = _parse_bool(row.get("is_angle", "False"))
            is_counter       = _parse_bool(row.get("is_counter", "False"))

            if not sensor_name_base:
                continue

            unit = PhysicalUnit.from_string(unit_raw)
            stat_types = _parse_statistic_types(stats_raw)
            annotation_cls = infer_annotation_type(description)

            for stat in stat_types:
                sensor_name = SensorName(base_name=sensor_name_base, statistic=stat)
                annotation: SensorAnnotation = annotation_cls(
                    sensor_name=sensor_name,
                    description=description,
                    unit=unit,
                    is_angle=is_angle,
                    is_counter=is_counter,
                )
                turbine.add_annotation(annotation)

    return turbine


def load_feature_description_from_string(
    csv_text: str,
    asset_id: str = "unknown",
    wind_farm: Optional[str] = None,
    separator: str = ";",
) -> WindTurbine:
    """Same as :func:`load_feature_description` but accepts a raw CSV string."""
    turbine = WindTurbine(asset_id=asset_id, wind_farm=wind_farm)

    reader = csv.DictReader(io.StringIO(csv_text), delimiter=separator)
    for row in reader:
        sensor_name_base = row.get("sensor_name", "").strip()
        stats_raw        = row.get("statistics_type", row.get("statistic_type", "average"))
        description      = row.get("description", "").strip()
        unit_raw         = row.get("unit", "").strip()
        is_angle         = _parse_bool(row.get("is_angle", "False"))
        is_counter       = _parse_bool(row.get("is_counter", "False"))

        if not sensor_name_base:
            continue

        unit = PhysicalUnit.from_string(unit_raw)
        stat_types = _parse_statistic_types(stats_raw)
        annotation_cls = infer_annotation_type(description)

        for stat in stat_types:
            sensor_name = SensorName(base_name=sensor_name_base, statistic=stat)
            annotation: SensorAnnotation = annotation_cls(
                sensor_name=sensor_name,
                description=description,
                unit=unit,
                is_angle=is_angle,
                is_counter=is_counter,
            )
            turbine.add_annotation(annotation)

    return turbine


# ---------------------------------------------------------------------------
# SCADA DataFrame annotation
# ---------------------------------------------------------------------------

# Metadata attribute name attached to each pandas Series
_ANNOTATION_ATTR = "knowwind_annotation"


def annotate_scada_dataframe(
    df: pd.DataFrame,
    turbine: WindTurbine,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Attach semantic annotations to every column of a SCADA event DataFrame.

    For each column whose name matches a registered annotation (via suffix
    detection: ``_avg``, ``_min``, ``_max``, ``_std``), the annotation
    object is stored in ``df[col].attrs[_ANNOTATION_ATTR]``.

    Parameters
    ----------
    df:
        Raw SCADA DataFrame (e.g. loaded from an event CSV).
    turbine:
        A :class:`WindTurbine` previously built by :func:`load_feature_description`.
    inplace:
        If *False* (default), operate on a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``.attrs`` populated on annotated columns.
    """
    if not inplace:
        df = df.copy()

    # Build a lookup: column_name → annotation
    annotation_lookup: Dict[str, SensorAnnotation] = {
        ann.column_name: ann for ann in turbine.get_all_annotations()
    }

    for col in df.columns:
        # Try direct match first
        if col in annotation_lookup:
            df[col].attrs[_ANNOTATION_ATTR] = annotation_lookup[col]
        else:
            # Try suffix-normalised name
            # e.g. "sensor_18_avg" → look up annotation whose column_name == "sensor_18_avg"
            # Also handle columns that are just base names (single-stat sensors)
            normalized = _normalize_column(col)
            if normalized in annotation_lookup:
                df[col].attrs[_ANNOTATION_ATTR] = annotation_lookup[normalized]

    return df


def get_column_annotation(
    df: pd.DataFrame,
    column: str,
) -> Optional[SensorAnnotation]:
    """Return the annotation attached to *column* in an annotated DataFrame."""
    return df[column].attrs.get(_ANNOTATION_ATTR)


def get_annotated_columns(
    df: pd.DataFrame,
    annotation_type: Optional[type] = None,
) -> List[str]:
    """
    Return column names that have a semantic annotation attached.

    Parameters
    ----------
    annotation_type:
        If given, only columns whose annotation is an instance of this type
        are returned.
    """
    result = []
    for col in df.columns:
        ann = df[col].attrs.get(_ANNOTATION_ATTR)
        if ann is None:
            continue
        if annotation_type is None or isinstance(ann, annotation_type):
            result.append(col)
    return result


# ---------------------------------------------------------------------------
# Internal normalisation helpers
# ---------------------------------------------------------------------------

_SUFFIX_MAP = {
    "_avg": "_avg",
    "_min": "_min",
    "_max": "_max",
    "_std": "_std",
    # Some datasets use longer suffixes
    "_average": "_avg",
    "_minimum": "_min",
    "_maximum": "_max",
    "_std_dev": "_std",
}


def _normalize_column(col: str) -> str:
    """Try to map a raw column name to our canonical ``sensor_base_stat`` form."""
    for raw_suffix, canonical_suffix in _SUFFIX_MAP.items():
        if col.endswith(raw_suffix):
            base = col[: -len(raw_suffix)]
            return base + canonical_suffix
    # No recognised suffix – assume average
    return col + "_avg"
