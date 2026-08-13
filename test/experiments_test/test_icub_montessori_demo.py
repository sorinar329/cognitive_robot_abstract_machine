import dataclasses

import pytest
from typing_extensions import List, Optional

from experiments.montessori.icub_montessori_demo import (
    MAX_INSERTION_ATTEMPTS,
    _insert_all_shapes,
)
from experiments.montessori.semantics import (
    MontessoriShape,
    MontessoriShapeCategory,
    NoMatchingHoleError,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

SKIPPED_CATEGORY = MontessoriShapeCategory.DISK
"""
The coin, which the demo is configured to leave in place.
"""


# %% mimics standing in for the scene the insertion loop walks over


@dataclasses.dataclass
class ShapeWithOptionalHole:
    """
    A loose shape that either has a matching hole in the board or has none.
    """

    name: PrefixedName
    """
    Identifies the shape in assertions and log output, mirroring
    :attr:`~experiments.montessori.semantics.MontessoriShape.name`'s own
    :class:`~semantic_digital_twin.datastructures.prefixed_name.PrefixedName` type,
    which :func:`~experiments.montessori.icub_montessori_demo._insert_all_shapes` reads
    ``.name`` off of.
    """

    has_matching_hole: bool = True
    """
    Whether the board offers a hole this shape can be sorted into.
    """

    shape_category: str = "cube"
    """
    The kind of piece this is, which decides whether the demo attempts it at all.
    """

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = PrefixedName(self.name)


@dataclasses.dataclass
class BoardRejectingUnmatchedShapes:
    """
    A shape-sorting board that refuses shapes it has no hole for, the way the real board
    signals that a shape cannot be sorted at all.
    """

    name: str = "board"
    """
    Identifies the board in the error raised for an unsortable shape.
    """

    def hole_for(self, shape: ShapeWithOptionalHole):
        if shape.has_matching_hole:
            return object()
        raise NoMatchingHoleError(shape, self)


@dataclasses.dataclass
class SceneHoldingShapes:
    """
    The parts of the Montessori scene the insertion loop reads: the shapes to sort and
    the board to sort them into.
    """

    shapes: List[ShapeWithOptionalHole]
    """
    The loose shapes the loop iterates over.
    """

    def __post_init__(self):
        self.board = BoardRejectingUnmatchedShapes()
        self.world = self

    def get_semantic_annotations_by_type(self, annotation_type):
        assert annotation_type is MontessoriShape
        return self.shapes


@dataclasses.dataclass
class PlannedAction:
    """
    Stands in for the :class:`~experiments.montessori.insert_shape_action.InsertMontessoriShapeAction`
    an insertion attempt built, exposing only the ``plan`` attribute
    :func:`~experiments.montessori.icub_montessori_demo._insert_all_shapes` reads off it
    to build a :class:`~experiments.montessori.sorting_results.ShapeInsertionResult`.
    """

    plan: object = None


@dataclasses.dataclass
class InertEventMonitor:
    """
    Stands in for a :class:`~experiments.montessori.event_monitoring.MontessoriEventMonitor`
    that never observes any event, so :func:`~experiments.montessori.icub_montessori_demo._insert_all_shapes`'s
    own start/stop/verdict-logging around each attempt has something to call without a
    real, physically simulated world to detect pick-up/insertion events in.
    """

    events: List[object] = dataclasses.field(default_factory=list)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


@pytest.fixture(autouse=True)
def no_event_monitoring(monkeypatch):
    """
    Every test in this module exercises the insertion-loop control flow, not event
    detection, so :func:`~experiments.montessori.event_monitoring.build_shape_monitor`
    (which needs a real :class:`~experiments.montessori.world.MontessoriWorld` with
    landing regions) is replaced with :class:`InertEventMonitor` for all of them.
    """
    monkeypatch.setattr(
        "experiments.montessori.icub_montessori_demo.build_shape_monitor",
        lambda montessori, shape: InertEventMonitor(),
    )


@dataclasses.dataclass
class RecordingInsertion:
    """
    Stands in for one insertion attempt, recording which shapes it was asked to insert
    and reporting a fixed outcome for each.
    """

    outcome: Optional[bool]
    """
    What every attempt reports: whether the shape fell through, or ``None`` for an
    attempt that failed to run at all.
    """

    calls: List[str] = dataclasses.field(default_factory=list)
    """
    Name of the shape of every attempt made, in order.
    """

    def __call__(self, shape, montessori, context, attempt):
        self.calls.append(shape.name.name)
        return self.outcome, PlannedAction()

    def attempts_for(self, shape_name: str) -> int:
        return self.calls.count(shape_name)


@pytest.fixture
def two_shapes():
    return SceneHoldingShapes(
        [ShapeWithOptionalHole("circle"), ShapeWithOptionalHole("square")]
    )


# %% one attempt per shape unless the attempt never ran


class TestEveryShapeIsAttemptedOnce:
    """
    Exercises the insertion loop moving on to the next shape after a single attempt,
    whatever that attempt made of the shape.
    """

    def test_a_shape_that_does_not_fall_through_is_not_retried(
        self, two_shapes, monkeypatch
    ):
        """
        A shape left resting on the board is not picked up again: the run continues with
        the next shape instead.
        """
        insertion = RecordingInsertion(outcome=False)
        monkeypatch.setattr(
            "experiments.montessori.icub_montessori_demo._insert_shape_or_none",
            insertion,
        )

        _insert_all_shapes(two_shapes, context=None)

        assert insertion.attempts_for("circle") == 1
        assert insertion.calls == ["circle", "square"]

    def test_a_shape_that_falls_through_is_not_retried(self, two_shapes, monkeypatch):
        insertion = RecordingInsertion(outcome=True)
        monkeypatch.setattr(
            "experiments.montessori.icub_montessori_demo._insert_shape_or_none",
            insertion,
        )

        _insert_all_shapes(two_shapes, context=None)

        assert insertion.calls == ["circle", "square"]

    def test_an_attempt_that_never_ran_is_retried(self, two_shapes, monkeypatch):
        """
        An attempt that failed before the shape was ever released says nothing about the
        shape, so it is worth repeating -- unlike one that simply did not sort.
        """
        insertion = RecordingInsertion(outcome=None)
        monkeypatch.setattr(
            "experiments.montessori.icub_montessori_demo._insert_shape_or_none",
            insertion,
        )

        _insert_all_shapes(two_shapes, context=None)

        assert insertion.attempts_for("circle") == MAX_INSERTION_ATTEMPTS
        assert insertion.attempts_for("square") == MAX_INSERTION_ATTEMPTS

    def test_a_shape_without_a_matching_hole_is_never_attempted(self, monkeypatch):
        scene = SceneHoldingShapes(
            [
                ShapeWithOptionalHole("sphere", has_matching_hole=False),
                ShapeWithOptionalHole("square"),
            ]
        )
        insertion = RecordingInsertion(outcome=True)
        monkeypatch.setattr(
            "experiments.montessori.icub_montessori_demo._insert_shape_or_none",
            insertion,
        )

        _insert_all_shapes(scene, context=None)

        assert insertion.calls == ["square"]


# %% shapes the demo leaves alone


class TestSkippedShapesAreNeverAttempted:
    """
    Exercises leaving configured shape categories in place, separately from shapes the
    board simply has no hole for.
    """

    def test_a_skipped_category_is_not_attempted(self, monkeypatch):
        insertion = RecordingInsertion(outcome=True)
        monkeypatch.setattr(
            "experiments.montessori.icub_montessori_demo._insert_shape_or_none",
            insertion,
        )
        scene = SceneHoldingShapes(
            [
                ShapeWithOptionalHole("coin", shape_category=SKIPPED_CATEGORY),
                ShapeWithOptionalHole("square"),
            ]
        )

        _insert_all_shapes(scene, context=None)

        assert insertion.calls == ["square"]

    def test_a_skipped_shape_is_skipped_even_though_it_has_a_hole(self, monkeypatch):
        """
        The board does have a slot for it, so nothing but the skip list keeps the demo
        from trying.
        """
        insertion = RecordingInsertion(outcome=True)
        monkeypatch.setattr(
            "experiments.montessori.icub_montessori_demo._insert_shape_or_none",
            insertion,
        )
        coin = ShapeWithOptionalHole(
            "coin", has_matching_hole=True, shape_category=SKIPPED_CATEGORY
        )

        _insert_all_shapes(SceneHoldingShapes([coin]), context=None)

        assert insertion.calls == []
