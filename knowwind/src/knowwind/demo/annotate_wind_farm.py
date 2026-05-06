"""
knowwind.demo.annotate_wind_farm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end demo:
  1. Load feature_description.csv → build a WindTurbine with semantic annotations
  2. (Optionally) load an event CSV → annotate its columns
  3. Print a summary report

Run from the repo root:
    python -m knowwind.demo.annotate_wind_farm \\
        --feature_desc data/feature_description.csv \\
        --event_csv    data/sample_csvs/event_001.csv \\
        --asset_id     WF_A_T01 \\
        --wind_farm    A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure the package is importable when run as a script from repo root
sys.path.insert(0, str(Path(__file__).parents[2]))

from knowwind.pipeline import (
    load_feature_description,
    annotate_scada_dataframe,
    turbine_annotation_summary,
    annotation_type_counts,
    dataframe_annotation_report,
    get_annotated_columns,
)
from knowwind.semantic_annotations import WindSpeedSensor, ActivePowerSensor


def main() -> None:
    parser = argparse.ArgumentParser(description="KnowWind – semantic annotation demo")
    parser.add_argument("--feature_desc", required=True, help="Path to feature_description.csv")
    parser.add_argument("--event_csv",    default=None,   help="Path to an event dataset CSV (optional)")
    parser.add_argument("--asset_id",     default="T01",  help="Turbine asset ID")
    parser.add_argument("--wind_farm",    default=None,   help="Wind farm label")
    parser.add_argument("--sep",          default=";",    help="CSV separator")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Build annotated WindTurbine from feature description
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  KnowWind – Semantic Annotation Demo")
    print("=" * 60)

    turbine = load_feature_description(
        args.feature_desc,
        asset_id=args.asset_id,
        wind_farm=args.wind_farm,
        separator=args.sep,
    )
    print(f"\n[1] Loaded turbine: {turbine}")

    # Print type counts
    print("\n[2] Annotation type distribution:")
    for ann_type, count in annotation_type_counts(turbine).items():
        print(f"    {ann_type:40s} {count:3d}")

    # Print first 10 rows of the summary table
    summary = turbine_annotation_summary(turbine)
    print(f"\n[3] First 10 annotations (of {len(summary)}):")
    print(summary.head(10).to_string(index=False))

    # Query wind-speed sensors
    ws_sensors = turbine.get_annotations_by_type(WindSpeedSensor)
    print(f"\n[4] WindSpeedSensor columns: {[s.column_name for s in ws_sensors]}")

    power_sensors = turbine.get_annotations_by_type(ActivePowerSensor)
    print(f"    ActivePowerSensor columns: {[s.column_name for s in power_sensors]}")

    # ------------------------------------------------------------------ #
    # 2. Annotate an event DataFrame (optional)
    # ------------------------------------------------------------------ #
    if args.event_csv:
        print(f"\n[5] Annotating event CSV: {args.event_csv}")
        df = pd.read_csv(args.event_csv, sep=args.sep, low_memory=False)
        print(f"    Shape: {df.shape}")

        df_ann = annotate_scada_dataframe(df, turbine)

        report = dataframe_annotation_report(df_ann)
        annotated = report[report["annotation_type"] != "—"]
        unmatched = report[report["annotation_type"] == "—"]

        print(f"    Annotated columns : {len(annotated)}")
        print(f"    Unannotated columns: {len(unmatched)}")

        ws_cols = get_annotated_columns(df_ann, WindSpeedSensor)
        if ws_cols:
            print(f"\n    Wind-speed data (first 5 rows):")
            print(df_ann[ws_cols].head().to_string())
    else:
        print("\n[5] No event CSV provided – skipping SCADA annotation step.")
        print("    Re-run with --event_csv <path> to see column-level annotation.")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
