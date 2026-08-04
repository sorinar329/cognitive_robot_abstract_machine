"""
Turn the successful stacking attempts into CSV files a joint probability tree can be
learned from.

The attempts are stored one row per step with bookkeeping columns alongside the sampled
parameters (see ``build_merged_dataset.py``). Learning wants neither the bookkeeping nor
the storage order: it wants the parameters alone, in a fixed, meaningful order. This
writes exactly that.

``infer_variables_from_dataframe`` rejects columns it cannot type, so the learning CSV
carries only the float parameters; the bookkeeping is available as a separate file for
tracing a row back to its run.

Run it with the interpreter whose packages point at this checkout, for example::

    /home/sorin/.virtualenvs/cram2-env/bin/python export_successful_parameters.py
    /home/sorin/.virtualenvs/cram2-env/bin/python export_successful_parameters.py --per-step
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas
from sqlalchemy import create_engine
from typing_extensions import Dict, List, Optional

# %% what a learning row consists of

SUCCESSFUL_DATABASE_URI = (
    "postgresql+psycopg://semantic_digital_twin:naren@localhost:5432/"
    "coraplex_panda_demo_successful"
)
"""
Database holding only the attempts from iterations that fully stacked.
"""

PICKUP_PARAMETER_COLUMNS: List[str] = [
    "object_friction",
    "pre_approach_linear_velocity",
    "grasp_linear_velocity",
    "grasp_closing_velocity",
    "grasp_stall_min_time",
    "lift_linear_velocity",
]
"""
``PickUpAction``'s sampled fields, named exactly as the action names them and ordered as
they take effect: the object's friction is applied before the pick, then the arm
approaches, grasps, closes, waits for the grasp to settle and lifts.
"""

PLACE_PARAMETER_COLUMNS: List[str] = [
    "transport_linear_velocity",
    "placing_linear_velocity",
    "release_opening_velocity",
    "retract_linear_velocity",
]
"""
``PlaceAction``'s sampled fields, named exactly as the action names them and ordered as
they take effect: transport, place, release and retract.
"""

PARAMETER_COLUMNS: List[str] = PICKUP_PARAMETER_COLUMNS + PLACE_PARAMETER_COLUMNS
"""
Every sampled parameter of an attempt, in the order the robot moves through them.

A joint probability tree does not need this order to learn, but it keeps the columns, the
learned variables and any plot of them in an order a reader can follow.
"""

STEP_COLUMN = "step_index"
"""
Which cube of the stack an attempt placed: 1 for ``cube1 onto cube0``, 2 and 3 for the
ones above it.

Part of the learning data rather than the bookkeeping. The three steps place onto a
tower of a different height, so their parameters need not be distributed alike, and
without this a tree learned over all of them cannot tell which step a row describes.
Typed as an integer, which ``infer_variables_from_dataframe`` turns into an ``Integer``
variable the tree can split on.
"""

METADATA_COLUMNS: List[str] = [
    "source_database",
    "run_started_at",
    "iteration_index",
    "step_index",
    "step_name",
]
"""
Columns identifying where an attempt came from.

Kept out of the learning data except for :data:`STEP_COLUMN`: the rest are identifiers
rather than parameters, and ``infer_variables_from_dataframe`` raises on the ones it
cannot type. ``iteration_index`` stays out on purpose even though it is an integer --
splitting on it would let a tree separate runs by when they happened rather than by how
they were parameterized.
"""


# %% reading and shaping


def read_successful_attempts(database_uri: str = SUCCESSFUL_DATABASE_URI) -> pandas.DataFrame:
    """
    Read every successful attempt, parameters and bookkeeping alike.

    :param database_uri: Connection string of the database holding the attempts.
    """
    engine = create_engine(database_uri)
    return pandas.read_sql(
        f"SELECT {', '.join(METADATA_COLUMNS + PARAMETER_COLUMNS)} "
        "FROM stacking_dataset ORDER BY source_database, iteration_index, step_index",
        engine,
    )


def parameters_for_learning(
    attempts: pandas.DataFrame,
    columns: Optional[List[str]] = None,
    include_step: bool = True,
) -> pandas.DataFrame:
    """
    Reduce attempts to what a tree is learned from: which step the attempt was, followed
    by the requested parameters in the order they take effect.

    :param attempts: Attempts as read from the database.
    :param columns: Parameters to keep; every parameter of an attempt by default. Pass
        :data:`PICKUP_PARAMETER_COLUMNS` or :data:`PLACE_PARAMETER_COLUMNS` to learn a
        model for one action alone.
    :param include_step: Whether to keep :data:`STEP_COLUMN`. Leave it out when every row
        is the same step already, where a constant column carries nothing.
    :raises MissingParameterColumns: If a requested parameter is absent from ``attempts``.
    """
    requested = PARAMETER_COLUMNS if columns is None else columns
    missing = [column for column in requested if column not in attempts.columns]
    if include_step and STEP_COLUMN not in attempts.columns:
        missing.append(STEP_COLUMN)
    if missing:
        raise MissingParameterColumns(missing)

    learning_data = attempts[requested].astype(float)
    if not include_step:
        return learning_data
    return pandas.concat(
        [attempts[[STEP_COLUMN]].astype(int), learning_data], axis="columns"
    )


def split_by_step(attempts: pandas.DataFrame) -> Dict[str, pandas.DataFrame]:
    """
    Group attempts by which cube was stacked onto which.

    The three steps face a stack of a different height, so their parameters need not be
    distributed alike, and a tree per step can say so where one tree over all of them
    cannot.

    :param attempts: Attempts as read from the database.
    """
    return {
        str(step_name): group for step_name, group in attempts.groupby("step_name")
    }


def csv_name_for_step(step_name: str, action: str) -> str:
    """
    File name holding one action's parameters for one step, for example
    ``pickup_cube2_onto_cube1.csv``.

    :param step_name: The step as recorded on the attempt, such as ``cube2 onto cube1``.
    :param action: Which of the step's actions the file holds, ``pickup`` or ``place``.
    """
    return f"{action}_{step_name.replace(' ', '_')}.csv"


class MissingParameterColumns(KeyError):
    """
    Raised when attempts lack a parameter the learning data is defined to contain.

    Exporting the remaining columns would silently produce a narrower dataset that still
    trains, so the export fails instead.
    """

    def __init__(self, missing_columns: List[str]) -> None:
        super().__init__(
            f"attempts are missing the parameter column(s) {', '.join(missing_columns)}"
        )


# %% writing


ACTION_PARAMETER_COLUMNS: Dict[str, List[str]] = {
    "pickup": PICKUP_PARAMETER_COLUMNS,
    "place": PLACE_PARAMETER_COLUMNS,
}
"""
The parameters belonging to each action a stacking step performs.

A model is registered against one action class, so a tree meant to sample a
``PickUpAction`` must be learned over that action's fields alone -- a tree spanning both
actions matches neither.
"""


def export(output_directory: Path, per_step: bool, database_uri: str) -> None:
    """
    Write the learning CSVs, plus the bookkeeping needed to trace any row back.

    Writes one file per action, which is what a model sampled from in the demo is learned
    from, and one holding every parameter for studying how the two actions relate.

    :param output_directory: Directory the CSVs are written to; created if absent.
    :param per_step: Whether to additionally write a file per action per stacking step.
    :param database_uri: Connection string of the database holding the attempts.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    attempts = read_successful_attempts(database_uri)

    combined_path = output_directory / "successful_parameters.csv"
    parameters_for_learning(attempts).to_csv(combined_path, index=False)
    print(
        f"[export] {combined_path.name}: {len(attempts)} attempts, "
        f"{len(PARAMETER_COLUMNS)} parameters plus {STEP_COLUMN}"
    )

    for action, columns in sorted(ACTION_PARAMETER_COLUMNS.items()):
        action_path = output_directory / f"{action}_parameters.csv"
        parameters_for_learning(attempts, columns).to_csv(action_path, index=False)
        print(
            f"[export] {action_path.name}: {len(attempts)} attempts, "
            f"{len(columns)} parameters plus {STEP_COLUMN}"
        )

    metadata_path = output_directory / "successful_parameters_metadata.csv"
    attempts[METADATA_COLUMNS].to_csv(metadata_path, index=False)
    print(f"[export] {metadata_path.name}: row-for-row provenance of the above")

    if not per_step:
        return
    for step_name, group in sorted(split_by_step(attempts).items()):
        for action, columns in sorted(ACTION_PARAMETER_COLUMNS.items()):
            step_path = output_directory / csv_name_for_step(step_name, action)
            parameters_for_learning(group, columns, include_step=False).to_csv(
                step_path, index=False
            )
            print(f"[export] {step_path.name}: {len(group)} attempts")


def parse_arguments() -> argparse.Namespace:
    """
    Read where to write, whether to split per step, and which database to read.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).parent / "datasets",
        help="where the CSV files are written",
    )
    parser.add_argument(
        "--per-step",
        action="store_true",
        help="also write one CSV per stacking step, so each can be learned separately",
    )
    parser.add_argument(
        "--database-uri",
        default=SUCCESSFUL_DATABASE_URI,
        help="database holding the successful attempts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    export(arguments.output_directory, arguments.per_step, arguments.database_uri)
