"""
Collects the usable attempts of every finished ``demo2.py`` run into one flat dataset.

Each run lives in its own archived database, numbers its iterations from 1, and stores an
attempt's parameters spread over the plan graph that produced them. Training wants none
of that: it wants one row per attempt, holding the sampled parameters and what came of
them. This builds exactly that, in a separate database, leaving every source untouched.

The full plan graphs are deliberately not copied. They are ~2 GB, most of which is a
world snapshot repeated per attempt, and nothing in them is needed once the parameters
are extracted -- the archives remain the place to look for the rest.

Run it with the interpreter whose packages point at this checkout, for example::

    /home/sorin/.virtualenvs/cram2-env/bin/python build_merged_dataset.py
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEMO_DIRECTORY = Path(__file__).parent

POSTGRES_PREFIX = "postgresql+psycopg://semantic_digital_twin:naren@localhost:5432"
"""
Connection prefix shared by every database this demo uses.
"""

DATASET_DATABASE = "coraplex_panda_demo_dataset"
"""
Database the merged dataset is written to, kept apart from both the archives and the
database a run writes into.
"""

SUCCESSFUL_DATABASE = "coraplex_panda_demo_successful"
"""
Database holding only the attempts from iterations that fully stacked.
"""

DATASET_COLUMNS = [
    "source_database", "run_started_at", "iteration_index", "step_index",
    "step_name", "steps_attempted", "simulation_diverged", "iteration_stacked",
    "pre_approach_linear_velocity", "grasp_linear_velocity",
    "grasp_closing_velocity", "lift_linear_velocity", "grasp_stall_min_time",
    "object_friction", "transport_linear_velocity", "placing_linear_velocity",
    "release_opening_velocity", "retract_linear_velocity",
]
"""
Every column of a dataset row, in the order the insert expects them.
"""


def dataset_insert():
    """
    The statement that writes one dataset row, shared by both builds.
    """
    return text(
        f"INSERT INTO stacking_dataset ({', '.join(DATASET_COLUMNS)}) "
        f"VALUES ({', '.join(':' + name for name in DATASET_COLUMNS)})"
    )


@dataclasses.dataclass(frozen=True)
class SourceRun:
    """
    One finished run to take usable attempts from.
    """

    database: str
    """
    Name of the archived database holding the run.
    """

    healthy_iteration_limit: int
    """
    First iteration of the run's dead tail.

    Every run so far has reached a point after which no attempt ever succeeded again;
    attempts from there on describe a broken simulation and are left out.
    """

    report_path: Optional[Path]
    """
    The run's per-iteration report, the only record of whether a stack actually stood.

    ``None`` where the report was overwritten before it was archived, which leaves that
    run's attempts unlabelled.
    """


SOURCE_RUNS = [
    SourceRun("coraplex_panda_demo_2026_08_01", 550, None),
    SourceRun(
        "coraplex_panda_demo_2026_08_02",
        910,
        DEMO_DIRECTORY / "support_report_2026_08_02.md",
    ),
    SourceRun("coraplex_panda_demo", 595, DEMO_DIRECTORY / "support_report.md"),
]
"""
The runs merged into the dataset, oldest first.
"""

ATTEMPT_QUERY = """
SELECT s.run_started_at,
       s.iteration_index,
       s.step_name,
       s.simulation_diverged,
       max(pu.pre_approach_linear_velocity) AS pre_approach_linear_velocity,
       max(pu.grasp_linear_velocity)        AS grasp_linear_velocity,
       max(pu.grasp_closing_velocity)       AS grasp_closing_velocity,
       max(pu.lift_linear_velocity)         AS lift_linear_velocity,
       max(pu.grasp_stall_min_time)         AS grasp_stall_min_time,
       max(pu.object_friction)              AS object_friction,
       max(pl.transport_linear_velocity)    AS transport_linear_velocity,
       max(pl.placing_linear_velocity)      AS placing_linear_velocity,
       max(pl.release_opening_velocity)     AS release_opening_velocity,
       max(pl.retract_linear_velocity)      AS retract_linear_velocity
FROM stacking_attempt s
JOIN "PlanEdgeDAO"        e ON e.parent_id   = s.plan_database_id
JOIN "DesignatorNodeDAO"  d ON d.database_id = e.child_id
LEFT JOIN "PickUpActionDAO" pu ON pu.database_id = d.designator_id
LEFT JOIN "PlaceActionDAO"  pl ON pl.database_id = d.designator_id
WHERE s.iteration_index < :limit
GROUP BY s.database_id, s.run_started_at, s.iteration_index, s.step_name,
         s.simulation_diverged
ORDER BY s.database_id
"""
"""
One row per attempt, folding the pickup and place actions hanging off its plan into a
single record.

The actions are reached through ``DesignatorNodeDAO.designator_id`` rather than
``ActionNodeDAO.execution_data_id``: the latter points at a node's recorded world state,
and only coincides with the action's own key for the very first plan in a database.
"""

CREATE_DATASET_TABLE = """
CREATE TABLE IF NOT EXISTS stacking_dataset (
    database_id                  serial PRIMARY KEY,
    source_database              text        NOT NULL,
    run_started_at               timestamptz NOT NULL,
    iteration_index              integer     NOT NULL,
    step_index                   integer     NOT NULL,
    step_name                    text        NOT NULL,
    steps_attempted              integer     NOT NULL,
    simulation_diverged          boolean     NOT NULL,
    iteration_stacked            boolean,
    pre_approach_linear_velocity double precision,
    grasp_linear_velocity        double precision,
    grasp_closing_velocity       double precision,
    lift_linear_velocity         double precision,
    grasp_stall_min_time         double precision,
    object_friction              double precision,
    transport_linear_velocity    double precision,
    placing_linear_velocity      double precision,
    release_opening_velocity     double precision,
    retract_linear_velocity      double precision
)
"""


class IncompleteDataset(RuntimeError):
    """
    Raised when merged attempts came out without the parameters they exist to record.

    A silent gap here is the worst outcome the merge has: the rows still look like data
    and would be trained on, so the build fails instead of writing them.
    """

    def __init__(self, incomplete_count: int) -> None:
        super().__init__(
            f"{incomplete_count} merged attempt(s) have no pickup or place parameters -- "
            "the join from an attempt to its actions is wrong"
        )


def read_iteration_outcomes(report_path: Optional[Path]) -> dict[int, bool]:
    """
    Read whether each iteration ended with the whole stack standing.

    :param report_path: The run's report, or ``None`` if it was not kept.
    :return: Maps iteration number to whether segmind approved the stack. Empty when
        there is no report.
    """
    if report_path is None or not report_path.exists():
        return {}
    content = report_path.read_text(errors="replace")
    matches = re.findall(
        r"## Iteration (\d+)\s*\n\s*`segmind_approved\(\)`: \*\*(True|False)\*\*",
        content,
    )
    return {int(iteration): approved == "True" for iteration, approved in matches}


def collect_attempts(source: SourceRun) -> list[dict]:
    """
    Read one run's usable attempts, adding each attempt's position within its iteration
    and the iteration's outcome.

    ``step_index`` and ``steps_attempted`` together say where an attempt sat in its
    iteration and whether the iteration was cut short, which is what distinguishes a step
    that was never reached from one that ran.
    """
    engine = create_engine(f"{POSTGRES_PREFIX}/{source.database}")
    with engine.connect() as connection:
        rows = connection.execute(
            text(ATTEMPT_QUERY), {"limit": source.healthy_iteration_limit}
        ).mappings().all()

    outcomes = read_iteration_outcomes(source.report_path)
    steps_per_iteration: dict[int, int] = {}
    for row in rows:
        steps_per_iteration[row["iteration_index"]] = (
            steps_per_iteration.get(row["iteration_index"], 0) + 1
        )

    attempts = []
    seen_in_iteration: dict[int, int] = {}
    for row in rows:
        iteration = row["iteration_index"]
        seen_in_iteration[iteration] = seen_in_iteration.get(iteration, 0) + 1
        attempts.append(
            {
                **dict(row),
                "source_database": source.database,
                "step_index": seen_in_iteration[iteration],
                "steps_attempted": steps_per_iteration[iteration],
                "iteration_stacked": outcomes.get(iteration),
            }
        )
    return attempts


def create_dataset_database(database: str) -> Engine:
    """
    Create a dataset database and its table, dropping any previous build of it.

    The sources are never touched, so a rebuild is always cheap and safe to repeat.
    """
    server = create_engine(f"{POSTGRES_PREFIX}/postgres").execution_options(
        isolation_level="AUTOCOMMIT"
    )
    with server.connect() as connection:
        connection.execute(text(f"DROP DATABASE IF EXISTS {database}"))
        connection.execute(
            text(f"CREATE DATABASE {database} OWNER semantic_digital_twin")
        )
    engine = create_engine(f"{POSTGRES_PREFIX}/{database}")
    with engine.begin() as connection:
        connection.execute(text(CREATE_DATASET_TABLE))
    return engine


def build_successful_subset() -> None:
    """
    Copy the attempts belonging to iterations that fully stacked into their own database.

    Only iterations whose outcome was actually recorded can appear: one archived run's
    report was overwritten before it was kept, so its successes are unidentifiable and
    are left out rather than guessed at.
    """
    source = create_engine(f"{POSTGRES_PREFIX}/{DATASET_DATABASE}")
    with source.connect() as connection:
        rows = connection.execute(
            text(
                f"SELECT {', '.join(DATASET_COLUMNS)} FROM stacking_dataset "
                "WHERE iteration_stacked ORDER BY database_id"
            )
        ).mappings().all()

    engine = create_dataset_database(SUCCESSFUL_DATABASE)
    with engine.begin() as connection:
        connection.execute(dataset_insert(), [dict(row) for row in rows])

    with engine.connect() as connection:
        iterations = connection.execute(
            text(
                "SELECT count(*) FROM (SELECT DISTINCT source_database, iteration_index "
                "FROM stacking_dataset) t"
            )
        ).scalar()
    print(
        f"[dataset] {SUCCESSFUL_DATABASE}: {len(rows)} attempts from {iterations} "
        "iterations that fully stacked"
    )


def build_dataset() -> None:
    """
    Merge every source run's usable attempts into the dataset database and report what
    was written.
    """
    engine = create_dataset_database(DATASET_DATABASE)
    insert = dataset_insert()

    for source in SOURCE_RUNS:
        attempts = collect_attempts(source)
        with engine.begin() as connection:
            connection.execute(
                insert, [{c: a[c] for c in DATASET_COLUMNS} for a in attempts]
            )
        labelled = sum(1 for a in attempts if a["iteration_stacked"] is not None)
        print(
            f"[dataset] {source.database}: {len(attempts)} attempts "
            f"({labelled} with a known iteration outcome)"
        )

    with engine.connect() as connection:
        incomplete = connection.execute(
            text(
                "SELECT count(*) FROM stacking_dataset WHERE object_friction IS NULL "
                "OR retract_linear_velocity IS NULL"
            )
        ).scalar()
        if incomplete:
            raise IncompleteDataset(incomplete)
        total = connection.execute(text("SELECT count(*) FROM stacking_dataset")).scalar()
        labelled = connection.execute(
            text("SELECT count(*) FROM stacking_dataset WHERE iteration_stacked IS NOT NULL")
        ).scalar()
        stacked = connection.execute(
            text("SELECT count(*) FROM stacking_dataset WHERE iteration_stacked")
        ).scalar()
    print(
        f"[dataset] {DATASET_DATABASE}: {total} attempts, {labelled} labelled, "
        f"{stacked} from iterations that fully stacked"
    )


if __name__ == "__main__":
    build_dataset()
    build_successful_subset()
