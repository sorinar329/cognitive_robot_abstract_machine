"""
Third-generation copy of ``build_merged_dataset_v2.py``, collecting a fresh ``demo3.py``
collection run -- one written to its own ``coraplex_panda_demo_v3`` database, not the
earlier ``coraplex_panda_demo_v2`` archive -- into one flat dataset.

Structured exactly like ``build_merged_dataset_v2.py``: each source run lives in its own
database, numbers its iterations from 1, and stores an attempt's parameters spread over
the plan graph that produced them, so this builds one row per attempt holding the sampled
parameters, the final z positions, and what came of them, in a separate database, leaving
every source untouched.

Kept as a separate file and a separate set of output databases rather than pointing
``build_merged_dataset_v2.py`` at the new run: ``coraplex_panda_demo_v2`` is archived
source data, and this generation's dataset/successful databases must not overwrite it.

Only one source run is listed in :data:`SOURCE_RUNS` for now, since this run has not yet
been run more than once -- add further :class:`SourceRun` entries the same way
``build_merged_dataset.py`` does once later runs are archived under their own database
names.

Run it with the interpreter whose packages point at this checkout, for example::

    /home/sorin/.virtualenvs/cram2-env/bin/python build_merged_dataset_v3.py
"""

from __future__ import annotations

import dataclasses
import datetime
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

DATASET_DATABASE = "coraplex_panda_demo_v3_dataset"
"""
Database the merged dataset is written to, kept apart from both the archives and the
database a run writes into.
"""

SUCCESSFUL_DATABASE = "coraplex_panda_demo_v3_successful"
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
    "cube0_final_z", "cube1_final_z", "cube2_final_z", "cube3_final_z",
    "object_final_z",
]
"""
Every column of a dataset row, in the order the insert expects them.

Identical to ``build_merged_dataset_v2.py``'s own :data:`DATASET_COLUMNS`: this is the
same ``demo3.py``/``StackingAttemptRecordV2`` schema, just a fresh collection run kept in
its own database.
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

    ``build_merged_dataset.py``'s own runs each reached a point after which no attempt
    ever succeeded again, found by inspecting that run's report once it was finished; a
    generous placeholder is used here until this run has actually finished and its own
    report inspected the same way.
    """

    report_path: Optional[Path]
    """
    The run's per-iteration report, the only record of whether a stack actually stood.

    ``None`` where the report was overwritten before it was archived, which leaves that
    run's attempts unlabelled.
    """


SOURCE_RUNS = [
    SourceRun("coraplex_panda_demo_v3", 1_000_000, DEMO_DIRECTORY / "support_report_v3.md"),
]
"""
The runs merged into the dataset, oldest first. See :class:`SourceRun` on why
``healthy_iteration_limit`` is a placeholder for now.

Only ``coraplex_panda_demo_v3`` -- this generation's dataset is kept separate from the
``coraplex_panda_demo_v2`` archive rather than combined with it, the same way
``build_merged_dataset_v2.py`` itself never folds ``demo2.py``'s ``coraplex_panda_demo``
in. ``demo3.py``'s ``SUPPORT_REPORT_PATH`` now defaults to ``support_report_v3.md`` to
match, its own fresh report for this fresh database.
"""

ATTEMPT_QUERY = """
SELECT s.run_started_at,
       s.iteration_index,
       s.step_name,
       s.simulation_diverged,
       s.cube0_final_z,
       s.cube1_final_z,
       s.cube2_final_z,
       s.cube3_final_z,
       s.object_final_z,
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
FROM stacking_attempt_v2 s
JOIN "PlanEdgeDAO"        e ON e.parent_id   = s.plan_database_id
JOIN "DesignatorNodeDAO"  d ON d.database_id = e.child_id
LEFT JOIN "PickUpActionDAO" pu ON pu.database_id = d.designator_id
LEFT JOIN "PlaceActionDAO"  pl ON pl.database_id = d.designator_id
WHERE s.iteration_index < :limit
GROUP BY s.database_id, s.run_started_at, s.iteration_index, s.step_name,
         s.simulation_diverged, s.cube0_final_z, s.cube1_final_z, s.cube2_final_z,
         s.cube3_final_z, s.object_final_z
ORDER BY s.database_id
"""
"""
One row per attempt, folding the pickup and place actions hanging off its plan into a
single record, and carrying the z positions straight through from
``stacking_attempt_v2`` -- see ``build_merged_dataset.py``'s own :data:`ATTEMPT_QUERY`
for why the actions are reached through ``DesignatorNodeDAO.designator_id``.

``stacking_attempt_v2`` is still the right table name here even though this database is
named ``coraplex_panda_demo_v3``: it is ``demo3.py``'s own (unchanged)
``StackingAttemptRecordV2`` table, just written into a different database instance.

The z position columns are added to ``GROUP BY`` rather than wrapped in ``max()``: they
live on ``s`` itself, so they are already constant within a group, and ``max()`` would
just be a longer way of saying the same thing.
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
    retract_linear_velocity      double precision,
    cube0_final_z                double precision,
    cube1_final_z                double precision,
    cube2_final_z                double precision,
    cube3_final_z                double precision,
    object_final_z               double precision
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


def read_iteration_outcomes(
    report_path: Optional[Path],
) -> dict[tuple[datetime.datetime, int], bool]:
    """
    Read whether each iteration ended with the whole stack standing.

    :param report_path: The run's report, or ``None`` if it was not kept.
    :return: Maps ``(run_started_at, iteration_index)`` to whether segmind approved the
        stack. Empty when there is no report.

    Keyed by ``(run_started_at, iteration_index)`` rather than bare iteration number:
    unlike ``build_merged_dataset.py``'s own archived sources, each of which held exactly
    one continuous run, ``demo3.py`` can write several separate runs into the same
    database (its own database is not re-archived between invocations), and their
    iteration numbering each restarts at 1. ``demo3.py``'s report heading carries
    ``run_started_at`` for exactly this reason -- see its own ``append_support_report``.
    """
    if report_path is None or not report_path.exists():
        return {}
    content = report_path.read_text(errors="replace")
    matches = re.findall(
        r"## Run (\S+) Iteration (\d+)\s*\n\s*`segmind_approved\(\)`: \*\*(True|False)\*\*",
        content,
    )
    return {
        (datetime.datetime.fromisoformat(run_started_at), int(iteration)): approved == "True"
        for run_started_at, iteration, approved in matches
    }


def collect_attempts(source: SourceRun) -> list[dict]:
    """
    Read one run's usable attempts, adding each attempt's position within its iteration
    and the iteration's outcome.

    ``step_index`` and ``steps_attempted`` together say where an attempt sat in its
    iteration and whether the iteration was cut short, which is what distinguishes a step
    that was never reached from one that ran.

    Grouped by ``(run_started_at, iteration_index)`` rather than bare iteration number --
    see :func:`read_iteration_outcomes` for why bare iteration number is not enough to
    tell two different runs' attempts apart in ``demo3.py``'s database.
    """
    engine = create_engine(f"{POSTGRES_PREFIX}/{source.database}")
    with engine.connect() as connection:
        rows = connection.execute(
            text(ATTEMPT_QUERY), {"limit": source.healthy_iteration_limit}
        ).mappings().all()

    outcomes = read_iteration_outcomes(source.report_path)
    steps_per_iteration: dict[tuple[datetime.datetime, int], int] = {}
    for row in rows:
        key = (row["run_started_at"], row["iteration_index"])
        steps_per_iteration[key] = steps_per_iteration.get(key, 0) + 1

    attempts = []
    seen_in_iteration: dict[tuple[datetime.datetime, int], int] = {}
    for row in rows:
        key = (row["run_started_at"], row["iteration_index"])
        seen_in_iteration[key] = seen_in_iteration.get(key, 0) + 1
        attempts.append(
            {
                **dict(row),
                "source_database": source.database,
                "step_index": seen_in_iteration[key],
                "steps_attempted": steps_per_iteration[key],
                "iteration_stacked": outcomes.get(key),
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
    if rows:
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
                "OR retract_linear_velocity IS NULL OR object_final_z IS NULL"
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
