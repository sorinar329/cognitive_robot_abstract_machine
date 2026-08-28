"""Dataclass based exceptions raised across the package."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# %% base


@dataclass
class MontessoriVisionError(Exception, ABC):
    """Base class for all errors of this package.

    Subclasses implement :func:`error_message` and :func:`suggest_correction`; both are evaluated at
    construction time and composed into the message the exception carries. A non-empty correction is
    rendered as a trailing ``Suggestion: ...`` line.
    """

    def __post_init__(self) -> None:
        message = self.error_message()
        correction = self.suggest_correction()
        if correction:
            message = f"{message}\nSuggestion: {correction}"
        super().__init__(message)

    @abstractmethod
    def error_message(self) -> str:
        """Describe what went wrong."""

    @abstractmethod
    def suggest_correction(self) -> str:
        """Describe how the caller can fix the situation, or return an empty string."""

    def __str__(self) -> str:
        # Stdlib mixins such as KeyError repr their args; always render the composed message.
        return Exception.__str__(self)


# %% board configuration


@dataclass
class UnknownShapeCategory(MontessoriVisionError, KeyError):
    """Raised when a shape category is requested that the board configuration does not define."""

    requested_name: str
    """The category name that was looked up."""

    known_names: tuple[str, ...]
    """The category names the board configuration does define."""

    def error_message(self) -> str:
        return f"The board does not define a shape category named {self.requested_name!r}."

    def suggest_correction(self) -> str:
        return f"use one of {', '.join(self.known_names)}, or add the category to the board configuration."


@dataclass
class UnknownOutlineType(MontessoriVisionError, ValueError):
    """Raised when a board configuration names an outline type that has no implementation."""

    requested_type: str
    """The outline type read from the configuration."""

    known_types: tuple[str, ...]
    """The outline types that can be constructed."""

    def error_message(self) -> str:
        return f"{self.requested_type!r} is not a known shape outline type."

    def suggest_correction(self) -> str:
        return f"use one of {', '.join(self.known_types)}."


@dataclass
class DegenerateOutline(MontessoriVisionError, ValueError):
    """Raised when an outline is configured with too few corners to enclose an area."""

    outline_description: str
    """A description of the offending outline."""

    corner_count: int
    """The number of corners the outline was configured with."""

    minimum_corner_count: int
    """The number of corners an outline needs."""

    def error_message(self) -> str:
        return (
            f"{self.outline_description} has {self.corner_count} corners, fewer than the "
            f"{self.minimum_corner_count} an outline needs."
        )

    def suggest_correction(self) -> str:
        return f"configure the outline with at least {self.minimum_corner_count} corners."


# %% detections


@dataclass
class EmptyBoundingBox(MontessoriVisionError, ValueError):
    """Raised when a bounding box would have a non-positive width or height."""

    left: int
    """Left pixel column of the attempted box."""

    top: int
    """Top pixel row of the attempted box."""

    right: int
    """Right pixel column of the attempted box, exclusive."""

    bottom: int
    """Bottom pixel row of the attempted box, exclusive."""

    def error_message(self) -> str:
        return (
            f"A bounding box spanning ({self.left}, {self.top}) to ({self.right}, {self.bottom}) "
            f"is empty."
        )

    def suggest_correction(self) -> str:
        return "make sure right is greater than left and bottom is greater than top."


@dataclass
class EmptyMask(MontessoriVisionError, ValueError):
    """Raised when a bounding box is derived from a mask that has no set pixel."""

    def error_message(self) -> str:
        return "The mask has no set pixel, so it has no bounding box."

    def suggest_correction(self) -> str:
        return "filter empty masks out before deriving bounding boxes from them."


# %% datasets and models


@dataclass
class MissingAnnotation(MontessoriVisionError, KeyError):
    """Raised when an image of a dataset has no matching ground truth annotation."""

    image_name: str
    """The image whose annotation is missing."""

    annotation_path: str
    """The annotation file that was searched."""

    def error_message(self) -> str:
        return f"{self.annotation_path} holds no annotation for image {self.image_name!r}."

    def suggest_correction(self) -> str:
        return "annotate the image or exclude it from the dataset."


@dataclass
class UnknownShapeLabel(MontessoriVisionError, KeyError):
    """Raised when a shape is asked for a class index that the label mapping does not cover."""

    label_name: str
    """The shape whose class index was looked up."""

    known_names: tuple[str, ...]
    """The shapes the mapping does cover."""

    def error_message(self) -> str:
        return f"{self.label_name!r} is not one of the shapes the detector distinguishes."

    def suggest_correction(self) -> str:
        return f"use one of {', '.join(self.known_names)}."


@dataclass
class UnknownClassIndex(MontessoriVisionError, IndexError):
    """Raised when a detector reports a class index the label mapping does not cover."""

    class_index: int
    """The index reported by the detector."""

    label_count: int
    """The number of labels the mapping knows."""

    def error_message(self) -> str:
        return (
            f"Class index {self.class_index} is outside the {self.label_count} labels "
            f"derived from the board configuration."
        )

    def suggest_correction(self) -> str:
        return "load the weights that were trained on this board configuration."


@dataclass
class MismatchedBatchSize(MontessoriVisionError, ValueError):
    """Raised when a classifier returns a different number of results than it was given crops."""

    crop_count: int
    """The number of crops that were classified."""

    classification_count: int
    """The number of classifications that came back."""

    def error_message(self) -> str:
        return (
            f"{self.crop_count} crops were classified but {self.classification_count} "
            f"classifications came back."
        )

    def suggest_correction(self) -> str:
        return "return exactly one classification per crop, in the order the crops were given."
