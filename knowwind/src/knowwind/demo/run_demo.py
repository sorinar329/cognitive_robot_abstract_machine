#!/usr/bin/env python3
"""
demo/run_demo.py
~~~~~~~~~~~~~~~~
Self-contained demonstration of the knowwind annotation pipeline.

Run from the repository root:

    pip install -e ".[dev]"
    python demo/run_demo.py

The demo:
  1. Writes a minimal ``feature_description.csv`` (subset of wind farm A sensors).
  2. Synthesises a realistic 72-hour SCADA event CSV (wind farm A format).
  3. Reads both files through knowwind.
  4. Runs the ``AnnotationPipeline`` (status filter + angle normalisation).
  5. Prints a rich terminal report with catalogue, dataset stats, and sensor stats.
  6. Saves the ``SensorCatalogue`` as JSON for reuse.
"""

from __future__ import annotations

import sys
import os

from knowwind.ingestion import read_feature_description, read_dataset
from knowwind.pipeline.annotator import AnnotationPipeline
from knowwind.semantic import SensorCatalogue, SensorRole, SubSystem

# Allow running without an editable install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.rule import Rule

import knowwind
from knowwind.output.report import (
    print_catalogue_summary,
    print_dataset_summary,
    print_numeric_stats,
)

console = Console()
DEMO_DIR = Path(__file__).parent / "data"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1.  Write a demo feature_description.csv
# ---------------------------------------------------------------------------

FEATURE_ROWS = [
    # sensor_name            ; statistics_type               ; description                                              ; unit ; is_angle ; is_counter
    ("sensor_0",              "average",                      "Ambient temperature",                                     "°C",  "False",  "False"),
    ("sensor_1",              "average",                      "Wind absolute direction",                                 "°",   "True",   "False"),
    ("sensor_2",              "average",                      "Wind relative direction",                                 "°",   "True",   "False"),
    ("wind_speed_3",          "maximum,minimum,average,std_dev", "Windspeed",                                           "m/s", "False",  "False"),
    ("wind_speed_4",          "average",                      "Estimated windspeed",                                     "m/s", "False",  "False"),
    ("sensor_5",              "maximum,minimum,std_dev,average", "Pitch angle",                                         "°",   "True",   "False"),
    ("sensor_11",             "average",                      "Temperature in gearbox bearing on high speed shaft",      "°C",  "False",  "False"),
    ("sensor_12",             "average",                      "Temperature oil in gearbox",                              "°C",  "False",  "False"),
    ("sensor_13",             "average",                      "Temperature in generator bearing 2 (Drive End)",          "°C",  "False",  "False"),
    ("sensor_15",             "average",                      "Temperature inside generator in stator windings phase 1", "°C",  "False",  "False"),
    ("sensor_18",             "maximum,minimum,average,std_dev", "Generator rpm in latest period",                      "rpm", "False",  "False"),
    ("sensor_23",             "average",                      "Averaged current in phase 1",                             "A",   "False",  "False"),
    ("sensor_26",             "average",                      "Grid frequency",                                          "Hz",  "False",  "False"),
    ("power_29",              "maximum,std_dev,average,minimum", "Possible grid active power",                          "kW",  "False",  "False"),
    ("power_30",              "maximum,std_dev,minimum,average", "Grid power",                                          "kW",  "False",  "False"),
    ("sensor_32",             "average",                      "Averaged voltage in phase 1",                             "V",   "False",  "False"),
    ("sensor_42",             "average",                      "Nacelle direction",                                       "°",   "True",   "False"),
    ("sensor_43",             "average",                      "Nacelle temperature",                                     "°C",  "False",  "False"),
    ("sensor_52",             "minimum,average,std_dev,maximum", "Rotor rpm",                                           "rpm", "False",  "False"),
]

FEATURE_CSV = DEMO_DIR / "feature_description.csv"


def write_feature_description() -> None:
    with FEATURE_CSV.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["sensor_name", "statistics_type", "description", "unit", "is_angle", "is_counter"])
        for row in FEATURE_ROWS:
            writer.writerow(row)
    console.print(f"[green]✓[/green] Wrote [bold]{FEATURE_CSV}[/bold]")


# ---------------------------------------------------------------------------
# 2.  Synthesise a 72-hour SCADA dataset
# ---------------------------------------------------------------------------

def _sine_wave(n: int, period: int, amp: float, offset: float, noise: float = 0.0) -> np.ndarray:
    t = np.arange(n)
    sig = offset + amp * np.sin(2 * np.pi * t / period)
    if noise:
        sig += RNG.normal(0, noise, n)
    return sig


def synthesise_dataset(n_rows: int = 432) -> pd.DataFrame:
    """Generate ~72 h of synthetic 10-min SCADA data (432 rows).

    The signals are simplified but physically plausible:
    - Wind speed follows a slow sinusoidal variation + noise.
    - Power tracks wind speed (cubic relationship capped at rated).
    - Temperatures lag wind/power with some inertia.
    - The first 216 rows are training data, the last 216 are test.
    - Status is mostly 0 (normal) with a few service (3) and downtime (4) rows.
    """
    n = n_rows
    timestamps = pd.date_range("2020-01-01", periods=n, freq="10min")

    # --- wind ---
    wind_avg = np.clip(_sine_wave(n, period=144, amp=4.0, offset=8.0, noise=1.5), 0, 25)
    wind_min = np.clip(wind_avg - RNG.uniform(0.5, 1.5, n), 0, None)
    wind_max = wind_avg + RNG.uniform(0.5, 2.0, n)
    wind_std = RNG.uniform(0.2, 1.0, n)

    # --- power (cubic until rated 2000 kW) ---
    rated = 2000.0
    raw_power = np.clip(0.3 * wind_avg ** 3, 0, rated)
    power_avg = raw_power + RNG.normal(0, 30, n)
    power_min = power_avg - RNG.uniform(10, 50, n)
    power_max = power_avg + RNG.uniform(10, 50, n)
    power_std = RNG.uniform(5, 40, n)

    # --- rotor / generator rpm ---
    rotor_avg = np.clip(wind_avg * 1.8, 0, 18.0)
    gen_avg   = rotor_avg * 90.0 + RNG.normal(0, 30, n)

    # --- temperatures ---
    amb_temp  = _sine_wave(n, period=144 * 365, amp=8, offset=10, noise=0.5)
    gearbox_t = amb_temp + 25 + 0.01 * power_avg + RNG.normal(0, 0.8, n)
    gen_bear  = amb_temp + 30 + 0.012 * power_avg + RNG.normal(0, 0.6, n)
    stator_t  = amb_temp + 50 + 0.015 * power_avg + RNG.normal(0, 1.0, n)
    nacelle_t = amb_temp + 15 + RNG.normal(0, 0.5, n)
    gearbox_oil = gearbox_t - 5 + RNG.normal(0, 0.3, n)

    # --- electrical ---
    current_a  = power_avg / (np.sqrt(3) * 690 / 1000 + 1e-9) + RNG.normal(0, 1, n)
    voltage_v  = 690 + RNG.normal(0, 5, n)
    freq_hz    = 50 + RNG.normal(0, 0.05, n)

    # --- directions / angles (0–360°) ---
    wind_dir_abs  = _sine_wave(n, 144 * 7, 60, 200) % 360
    wind_dir_rel  = RNG.uniform(-20, 20, n) % 360
    nacelle_dir   = (wind_dir_abs + RNG.normal(0, 3, n)) % 360
    pitch_avg     = np.clip(5 + 0.5 * wind_avg, 0, 30)
    pitch_min     = pitch_avg - 1
    pitch_max     = pitch_avg + 2
    pitch_std     = RNG.uniform(0.1, 0.5, n)

    # --- status ---
    status = np.zeros(n, dtype=int)
    # inject a few service windows
    for start in RNG.choice(n - 20, 3, replace=False):
        status[start:start + 6] = 3  # service
    for start in RNG.choice(n - 10, 2, replace=False):
        status[start:start + 3] = 4  # downtime

    # --- train/test split ---
    train_test = ["train"] * (n // 2) + ["test"] * (n - n // 2)

    df = pd.DataFrame({
        "id":            range(1, n + 1),
        "time_stamp":    timestamps,
        "asset_id":      "turbine_01",
        "train_test":    train_test,
        "status_type":   status,
        # ambient
        "sensor_0_avg":  np.round(amb_temp, 2),
        "sensor_1_avg":  np.round(wind_dir_abs, 1),
        "sensor_2_avg":  np.round(wind_dir_rel, 1),
        # wind speed
        "wind_speed_3_avg": np.round(wind_avg, 2),
        "wind_speed_3_min": np.round(wind_min, 2),
        "wind_speed_3_max": np.round(wind_max, 2),
        "wind_speed_3_std": np.round(wind_std, 3),
        "wind_speed_4_avg": np.round(wind_avg * 1.02, 2),
        # pitch
        "sensor_5_avg":  np.round(pitch_avg, 2),
        "sensor_5_min":  np.round(pitch_min, 2),
        "sensor_5_max":  np.round(pitch_max, 2),
        "sensor_5_std":  np.round(pitch_std, 3),
        # gearbox
        "sensor_11_avg": np.round(gearbox_t, 2),
        "sensor_12_avg": np.round(gearbox_oil, 2),
        # generator
        "sensor_13_avg": np.round(gen_bear, 2),
        "sensor_15_avg": np.round(stator_t, 2),
        "sensor_18_avg": np.round(gen_avg, 1),
        "sensor_18_min": np.round(gen_avg - 20, 1),
        "sensor_18_max": np.round(gen_avg + 20, 1),
        "sensor_18_std": np.round(RNG.uniform(5, 30, n), 2),
        # electrical
        "sensor_23_avg": np.round(current_a, 2),
        "sensor_26_avg": np.round(freq_hz, 3),
        "power_29_avg":  np.round(power_avg, 1),
        "power_29_min":  np.round(power_min, 1),
        "power_29_max":  np.round(power_max, 1),
        "power_29_std":  np.round(power_std, 2),
        "power_30_avg":  np.round(power_avg * 0.98, 1),
        "power_30_min":  np.round(power_min * 0.98, 1),
        "power_30_max":  np.round(power_max * 0.98, 1),
        "power_30_std":  np.round(power_std, 2),
        "sensor_32_avg": np.round(voltage_v, 1),
        # nacelle
        "sensor_42_avg": np.round(nacelle_dir, 1),
        "sensor_43_avg": np.round(nacelle_t, 2),
        # rotor
        "sensor_52_avg": np.round(rotor_avg, 3),
        "sensor_52_min": np.round(rotor_avg - 0.5, 3),
        "sensor_52_max": np.round(rotor_avg + 0.5, 3),
        "sensor_52_std": np.round(RNG.uniform(0.05, 0.3, n), 4),
    })
    return df


# ---------------------------------------------------------------------------
# 3.  Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    console.print(Rule("[bold cyan]knowwind demo[/bold cyan]"))

    # --- write synthetic data ---
    write_feature_description()

    dataset_csv = DEMO_DIR / "event_demo_01.csv"
    df_raw = synthesise_dataset(n_rows=432)
    df_raw.to_csv(dataset_csv, index=False)
    console.print(
        f"[green]✓[/green] Synthesised [bold]{dataset_csv}[/bold] "
        f"({len(df_raw)} rows × {len(df_raw.columns)} columns)"
    )

    console.print()
    console.print(Rule("[bold]Step 1 — Load feature catalogue[/bold]"))

    # --- build catalogue ---
    catalogue = read_feature_description(FEATURE_CSV, farm_id="A_demo")
    console.print(f"[green]✓[/green] Loaded catalogue: [bold]{catalogue}[/bold]")

    # --- save catalogue as JSON ---
    catalogue_json = DEMO_DIR / "catalogue_A_demo.json"
    catalogue.to_json(str(catalogue_json))
    console.print(f"[green]✓[/green] Saved catalogue → [bold]{catalogue_json}[/bold]")

    # --- reload and verify round-trip ---
    catalogue_rt = SensorCatalogue.from_json(str(catalogue_json))
    assert len(catalogue_rt) == len(catalogue), "JSON round-trip length mismatch!"
    console.print("[green]✓[/green] JSON round-trip verified")

    console.print()
    console.print(Rule("[bold]Step 2 — Load raw dataset[/bold]"))

    df = read_dataset(dataset_csv)
    console.print(f"[green]✓[/green] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    console.print(f"   Columns: {', '.join(df.columns[:8].tolist())} …")

    console.print()
    console.print(Rule("[bold]Step 3 — Run annotation pipeline[/bold]"))

    pipeline = AnnotationPipeline.default(catalogue)
    ads = pipeline.run(df, farm_id="A_demo", event_id="event_demo_01")

    console.print(f"[green]✓[/green] Pipeline complete: [bold]{ads}[/bold]")
    console.print(
        f"   Recognised: {len(ads.recognised_columns)} sensor columns | "
        f"Unknown: {ads.unknown_columns}"
    )
    console.print(
        f"   After status filter (keep codes 0,2): "
        f"{len(ads.data)} / {len(df)} rows retained"
    )
    console.print(
        f"   Angle columns normalised to [0°, 360°): "
        f"{catalogue.angle_columns()}"
    )

    console.print()
    console.print(Rule("[bold]Step 4 — Semantic inspection[/bold]"))

    # Demonstrate subsystem query
    gen_cols = ads.columns_for_subsystem(SubSystem.GENERATOR)
    console.print(f"Generator columns: {gen_cols}")

    temp_cols = ads.columns_for_role(SensorRole.TEMPERATURE)
    console.print(f"Temperature columns: {temp_cols}")

    # Annotate a specific column
    ann = ads.annotation("wind_speed_3_avg")
    if ann:
        console.print(
            f"wind_speed_3_avg → role=[bold]{ann.role.value}[/bold], "
            f"subsystem=[bold]{ann.subsystem.value}[/bold], "
            f"unit=[bold]{ann.unit}[/bold]"
        )

    console.print()
    console.print(Rule("[bold]Step 5 — Reports[/bold]"))

    print_dataset_summary(ads)
    print_catalogue_summary(ads)
    print_numeric_stats(ads, max_sensors=10)

    console.print()
    console.print(Rule("[bold green]Demo complete[/bold green]"))
    console.print(
        f"Outputs saved under [bold]{DEMO_DIR}[/bold]:\n"
        f"  {FEATURE_CSV.name}\n"
        f"  {dataset_csv.name}\n"
        f"  {catalogue_json.name}"
    )


if __name__ == "__main__":
    main()
