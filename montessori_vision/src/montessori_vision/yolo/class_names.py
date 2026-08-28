"""The one mapping between board shapes and the class indices a detector works in."""

from __future__ import annotations

from dataclasses import dataclass

from montessori_vision.board import BoardConfiguration, TargetKind
from montessori_vision.detections import ShapeLabel
from montessori_vision.exceptions import UnknownClassIndex, UnknownShapeLabel


@dataclass(frozen=True)
class YoloClassNames:
    """Turns board shapes into the integer classes a detector predicts, and back.

    Writing the training labels, generating the dataset description and reading a prediction all go
    through this one object, so a class index means the same thing in all three places. The order
    follows the board configuration, which means reordering the configuration invalidates already
    trained weights.
    """

    labels: tuple[ShapeLabel, ...]
    """The shapes a detector distinguishes, in class index order."""

    @property
    def names(self) -> list[str]:
        """The readable class names, in class index order, as a dataset description lists them."""
        return [label.name for label in self.labels]

    def index_of(self, label: ShapeLabel) -> int:
        """Return the class index of a shape.

        :raises UnknownShapeLabel: if the shape is not part of this mapping.
        """
        for index, known in enumerate(self.labels):
            if known.matches(label):
                return index
        raise UnknownShapeLabel(label.name, tuple(self.names))

    def label_at(self, index: int) -> ShapeLabel:
        """Return the shape a class index stands for.

        :raises UnknownClassIndex: if the index is outside this mapping.
        """
        if not 0 <= index < len(self.labels):
            raise UnknownClassIndex(index, len(self.labels))
        return self.labels[index]

    @classmethod
    def from_board(cls, board: BoardConfiguration) -> YoloClassNames:
        """Derive the class list from a board, one class per shape and role."""
        return cls(
            tuple(
                ShapeLabel(category, kind) for category in board.categories for kind in TargetKind
            )
        )
