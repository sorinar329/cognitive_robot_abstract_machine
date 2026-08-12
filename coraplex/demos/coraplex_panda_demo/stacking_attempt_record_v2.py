"""
Per-attempt index over the plans ``demo3.py`` persists, extending
``stacking_attempt_record.py`` with each cube's final z position.

Kept as its own table (``stacking_attempt_v2``) rather than adding columns to
:class:`~stacking_attempt_record.StackingAttemptRecord`: that table is shared with
``demo2.py``, whose rows never recorded a z position and whose plans never park the arm
between pickup and place (see ``demo3.py``'s own docstring), so the two are not
comparable row-for-row and must not be merged.
"""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class StackingAttemptBaseV2(DeclarativeBase):
    """
    Declarative base owning only this demo's v2 bookkeeping tables.
    """


class StackingAttemptRecordV2(StackingAttemptBaseV2):
    """
    One row per stacking step attempted, pointing at the plan persisted for it, plus
    every cube's final z position once the step had settled.

    Joining on :attr:`plan_database_id` recovers which iteration, which cube and which
    simulation state every persisted set of action parameters came from, exactly as
    :class:`~stacking_attempt_record.StackingAttemptRecord` does. The z positions are
    what make this an effect a causal diagnosis can be built against: whether the picked
    cube actually reached stacking height, rather than a bookkeeping field like
    ``step_index`` that carries no information about the outcome (see
    ``causal_diagnosis_v2.py``).
    """

    __tablename__ = "stacking_attempt_v2"

    database_id: Mapped[int] = mapped_column(primary_key=True)
    """
    Surrogate primary key.
    """

    run_started_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    """
    When the run this attempt belongs to started, so several runs sharing the database
    stay separable even though their iteration numbering restarts at 1.

    Timezone-aware, matching the value the demo stamps it with.
    """

    iteration_index: Mapped[int]
    """
    The 1-based iteration this attempt was part of.
    """

    step_name: Mapped[str]
    """
    Which stacking step was attempted, for example ``cube1 onto cube0``.
    """

    plan_database_id: Mapped[int]
    """
    ``database_id`` of the persisted plan's root node, holding the sampled parameters.
    """

    simulation_diverged: Mapped[bool]
    """
    Whether a cube had left the scene's sane bounds by the end of this attempt's
    iteration.

    Rows marked here describe motions planned against nonsense object poses and must be
    excluded from any training data.
    """

    cube0_final_z: Mapped[float]
    """
    ``cube0``'s height above the table once this step's attempt had settled.
    """

    cube1_final_z: Mapped[float]
    """
    ``cube1``'s height above the table once this step's attempt had settled.
    """

    cube2_final_z: Mapped[float]
    """
    ``cube2``'s height above the table once this step's attempt had settled.
    """

    cube3_final_z: Mapped[float]
    """
    ``cube3``'s height above the table once this step's attempt had settled.
    """

    object_final_z: Mapped[float]
    """
    The height of the cube this step actually picked and placed, i.e. whichever of
    :attr:`cube0_final_z`--:attr:`cube3_final_z` belongs to it.

    Recorded separately rather than left for a reader to look up through
    :attr:`step_name`: it is what :mod:`causal_diagnosis_v2` uses as the causal
    circuit's effect variable for both the pickup and the place action, and duplicating
    it here keeps that lookup out of the training data pipeline entirely.
    """

    @classmethod
    def create_table(cls, engine: Engine) -> None:
        """
        Create this demo's v2 bookkeeping tables if they do not exist yet.
        """
        StackingAttemptBaseV2.metadata.create_all(bind=engine, checkfirst=True)
