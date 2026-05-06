"""
knowwind.pipeline.report
~~~~~~~~~~~~~~~~~~~~~~~~~
Summarise the semantic annotations registered on a WindTurbine or
annotated DataFrame – useful for quick inspection and debugging.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from knowwind.semantic_annotations.base import SensorAnnotation
from knowwind.pipeline.csv_importer import _ANNOTATION_ATTR

if TYPE_CHECKING:
    from knowwind.wind_turbine import WindTurbine


def turbine_annotation_summary(turbine: "WindTurbine") -> pd.DataFrame:
    """
    Return a DataFrame summarising all annotations on *turbine*.

    Columns: column_name, annotation_type, description, unit, is_angle, is_counter
    """
    rows = []
    for ann in sorted(turbine.get_all_annotations(), key=lambda a: a.column_name):
        rows.append({
            "column_name":      ann.column_name,
            "annotation_type":  type(ann).__name__,
            "description":      ann.description,
            "unit":             ann.unit.value,
            "is_angle":         ann.is_angle,
            "is_counter":       ann.is_counter,
        })
    return pd.DataFrame(rows)


def annotation_type_counts(turbine: "WindTurbine") -> Dict[str, int]:
    """Return {annotation_type_name: count} for all annotations in *turbine*."""
    counts: Dict[str, int] = defaultdict(int)
    for ann in turbine.get_all_annotations():
        counts[type(ann).__name__] += 1
    return dict(sorted(counts.items()))


def dataframe_annotation_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame showing which columns of *df* have annotations.

    Columns: column_name, annotation_type, description, unit
    """
    rows = []
    for col in df.columns:
        ann: Optional[SensorAnnotation] = df[col].attrs.get(_ANNOTATION_ATTR)
        if ann is not None:
            rows.append({
                "column_name":     col,
                "annotation_type": type(ann).__name__,
                "description":     ann.description,
                "unit":            ann.unit.value,
            })
        else:
            rows.append({
                "column_name":     col,
                "annotation_type": "—",
                "description":     "",
                "unit":            "",
            })
    return pd.DataFrame(rows)
