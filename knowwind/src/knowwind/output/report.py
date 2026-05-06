"""
knowwind.output.report
~~~~~~~~~~~~~~~~~~~~~~
Pretty-print and structured reporting for ``AnnotatedDataset`` objects.

Uses ``rich`` for terminal output so results look good without any extra
configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from knowwind.semantic.model import SensorRole, SubSystem, StatisticType

if TYPE_CHECKING:
    from knowwind.pipeline.annotator import AnnotatedDataset

_console = Console()


# ---------------------------------------------------------------------------
# Colour maps for rich output
# ---------------------------------------------------------------------------

_SUBSYSTEM_COLORS: dict[SubSystem, str] = {
    SubSystem.AMBIENT:     "cyan",
    SubSystem.ROTOR:       "green",
    SubSystem.GEARBOX:     "yellow",
    SubSystem.GENERATOR:   "magenta",
    SubSystem.CONVERTER:   "blue",
    SubSystem.TRANSFORMER: "bright_blue",
    SubSystem.HYDRAULICS:  "orange1",
    SubSystem.NACELLE:     "white",
    SubSystem.GRID:        "bright_cyan",
    SubSystem.CONTROL:     "bright_magenta",
    SubSystem.UNKNOWN:     "dim",
}

_ROLE_COLORS: dict[SensorRole, str] = {
    SensorRole.TEMPERATURE:    "red",
    SensorRole.SPEED:          "green",
    SensorRole.POWER:          "yellow",
    SensorRole.REACTIVE_POWER: "bright_yellow",
    SensorRole.CURRENT:        "cyan",
    SensorRole.VOLTAGE:        "blue",
    SensorRole.FREQUENCY:      "magenta",
    SensorRole.ANGLE:          "bright_white",
    SensorRole.DIRECTION:      "bright_white",
    SensorRole.ENERGY:         "orange1",
    SensorRole.COUNTER:        "dim",
    SensorRole.OTHER:          "dim",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def print_catalogue_summary(dataset: "AnnotatedDataset") -> None:
    """Print a rich summary of all annotated sensors grouped by sub-system."""
    catalogue = dataset.catalogue
    recognised = set(dataset.recognised_columns)

    _console.rule(f"[bold]Sensor Catalogue — Wind Farm {catalogue.farm_id}[/bold]")

    # Group annotations by subsystem
    by_sub: dict[SubSystem, list] = {}
    for ann in catalogue.annotations.values():
        if ann.column_name not in recognised:
            continue
        by_sub.setdefault(ann.subsystem, []).append(ann)

    for subsystem, annotations in sorted(by_sub.items(), key=lambda x: x[0].value):
        color = _SUBSYSTEM_COLORS.get(subsystem, "white")
        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style=f"bold {color}",
            expand=False,
            padding=(0, 1),
        )
        table.add_column("Column",       style="bold", min_width=30)
        table.add_column("Description",  min_width=35)
        table.add_column("Unit",         min_width=6)
        table.add_column("Role",         min_width=14)
        table.add_column("Statistic",    min_width=8)
        table.add_column("Flags",        min_width=8)

        for ann in sorted(annotations, key=lambda a: a.column_name):
            role_color = _ROLE_COLORS.get(ann.role, "white")
            flags = []
            if ann.is_angle:
                flags.append("∠")
            if ann.is_counter:
                flags.append("Σ")
            table.add_row(
                ann.column_name,
                ann.description or "—",
                ann.unit or "—",
                Text(ann.role.value, style=role_color),
                ann.statistic_type.value,
                " ".join(flags) or "—",
            )

        _console.print(
            Panel(
                table,
                title=f"[{color}]{subsystem.value.upper()}[/{color}]",
                border_style=color,
                expand=False,
            )
        )


def print_dataset_summary(dataset: "AnnotatedDataset") -> None:
    """Print a compact summary of the dataset: shape, splits, unknown columns."""
    s = dataset.summary()

    _console.rule(f"[bold]Dataset — {s['event_id']} (farm {s['farm_id']})[/bold]")

    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim")
    info.add_column(style="bold")
    info.add_row("Rows",              str(s["rows"]))
    info.add_row("Recognised sensors", str(s["recognised_sensors"]))
    info.add_row("Unknown columns",   str(len(s["unknown_columns"])))
    info.add_row("Subsystems",        ", ".join(s["subsystems"]) or "none")

    if "train_test" in dataset.data.columns:
        vc = dataset.data["train_test"].value_counts()
        splits = "  ".join(f"{k}: {v}" for k, v in vc.items())
        info.add_row("Splits", splits)

    if "status_type" in dataset.data.columns:
        vc2 = dataset.data["status_type"].value_counts().sort_index()
        status_str = "  ".join(f"status {k}: {v}" for k, v in vc2.items())
        info.add_row("Status distribution", status_str)

    _console.print(Panel(info, title="[bold]Dataset info[/bold]", expand=False))

    if s["unknown_columns"]:
        _console.print(
            f"[dim]Unknown columns (metadata/unregistered): "
            f"{', '.join(s['unknown_columns'])}[/dim]"
        )


def print_numeric_stats(dataset: "AnnotatedDataset", max_sensors: int = 12) -> None:
    """Print basic numeric statistics for the first *max_sensors* average sensors."""
    avg_cols = [
        c for c in dataset.recognised_columns
        if dataset.annotation(c) and
           dataset.annotation(c).statistic_type == StatisticType.AVERAGE
    ][:max_sensors]

    if not avg_cols:
        _console.print("[dim]No average-type sensor columns found.[/dim]")
        return

    sub = dataset.data[avg_cols].select_dtypes(include="number")
    stats = sub.describe().T[["mean", "std", "min", "max"]]

    table = Table(
        title=f"Numeric statistics (first {len(avg_cols)} avg sensors)",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Column",  style="bold", min_width=28)
    table.add_column("Unit",    min_width=6)
    table.add_column("Mean",    justify="right", min_width=10)
    table.add_column("Std",     justify="right", min_width=10)
    table.add_column("Min",     justify="right", min_width=10)
    table.add_column("Max",     justify="right", min_width=10)

    for col in stats.index:
        ann  = dataset.annotation(col)
        unit = ann.unit if ann else ""
        row  = stats.loc[col]
        table.add_row(
            col,
            unit,
            f"{row['mean']:.2f}",
            f"{row['std']:.2f}",
            f"{row['min']:.2f}",
            f"{row['max']:.2f}",
        )

    _console.print(table)
