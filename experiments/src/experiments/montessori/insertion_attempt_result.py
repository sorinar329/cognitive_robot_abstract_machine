"""
:class:`InsertionAttemptResult`, shared by every ``*_montessori_demo`` module that
drives a robot kinematically (:mod:`experiments.montessori.montessori_demo`'s own HSRB,
:mod:`experiments.montessori.g1_montessori_demo`'s Unitree G1).

Kept in its own module rather than defined once per demo script: ORMatic generates one
DAO per class it finds across the whole ``experiments`` package, keyed by class name: two
same-named ``InsertionAttemptResult`` classes, one defined in each demo script, collide
in SQLAlchemy's metadata (``Table 'InsertionAttemptResultDAO' is already defined for this
MetaData instance``) the moment both scripts are ever imported (as ORMatic's own
generator does) in the same process. Mirrors
:mod:`experiments.montessori.sorting_results`'s own reasoning for
``ShapeInsertionResult``/``SortingIterationResult``, which is defined outside
:mod:`experiments.montessori.franka_montessori_demo` for the same category of reason
(there, a script re-loaded as both ``__main__`` and its own dotted path, rather than two
distinct scripts sharing a class name, but either way ORMatic ends up with two distinct
class objects of the same name and no way to tell them apart).
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.spatial_types.spatial_types import Point3


@dataclass(frozen=True)
class InsertionAttemptResult:
    """
    Outcome of a single shape-insertion attempt.
    """

    target_horizontal_offset: Point3
    """
    The horizontal offset the attempt was actually released at (see :attr:`~experiments.
    montessori.insert_shape_action.InsertMontessoriShapeAction.target_horizontal_offset`
    ), whether given by the caller or generated internally.
    """

    fell_through_hole: bool
    """
    Whether the shape actually fell through its hole after settling; see :meth:`~experim
    ents.montessori.insert_shape_action.InsertMontessoriShapeAction.has_fallen_through_h
    ole`.
    """
