"""
A per-attempt index over the plans ``demo2.py`` persists.

The generated plan DAOs carry an attempt's parameters but nothing that says which
iteration or which cube they belong to, and nothing that says whether the simulation was
still healthy at the time. Without that, a run's rows cannot be sliced afterwards -- a
stretch of iterations produced by a diverged simulation is indistinguishable from good
data.

Kept in its own declarative base rather than added to the generated ORM's: this is
bookkeeping for one demo, not part of the domain model that ``ormatic`` regenerates.
"""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class StackingAttemptBase(DeclarativeBase):
    """
    Declarative base owning only this demo's bookkeeping tables.
    """


class StackingAttemptRecord(StackingAttemptBase):
    """
    One row per stacking step attempted, pointing at the plan persisted for it.

    Joining on :attr:`plan_database_id` recovers which iteration, which cube and which
    simulation state every persisted set of action parameters came from.
    """

    __tablename__ = "stacking_attempt"

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

    @classmethod
    def create_table(cls, engine: Engine) -> None:
        """
        Create this demo's bookkeeping tables if they do not exist yet.
        """
        StackingAttemptBase.metadata.create_all(bind=engine, checkfirst=True)
