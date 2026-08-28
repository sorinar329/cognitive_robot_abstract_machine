"""Naming an image crop as a board shape, or rejecting it."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Optional

from montessori_vision.board import BoardConfiguration, TargetKind
from montessori_vision.detections import ShapeLabel
from montessori_vision.image import Image

# %% prompts


@dataclass(frozen=True)
class LabelledPrompt:
    """One text description together with what a crop matching it would be."""

    text: str
    """The description a text image model compares a crop against."""

    label: Optional[ShapeLabel]
    """The shape the description names, or nothing when it describes the background."""


@dataclass(frozen=True)
class PromptSet:
    """Every description a crop is compared against, built from the board configuration.

    The background descriptions are part of the set on purpose: without something to lose against,
    a crop of the table is forced into the closest shape category instead of being rejected.
    """

    prompts: tuple[LabelledPrompt, ...]
    """The descriptions, shape and background alike, in a fixed order."""

    @property
    def texts(self) -> tuple[str, ...]:
        """The descriptions as plain text, in the order a model should score them."""
        return tuple(prompt.text for prompt in self.prompts)

    @classmethod
    def from_board(cls, board: BoardConfiguration) -> PromptSet:
        """Build the descriptions of every shape and role of a board, plus its background."""
        prompts = [
            LabelledPrompt(text, ShapeLabel(category, kind))
            for category in board.categories
            for kind in TargetKind
            for text in category.prompts_for(kind)
        ]
        prompts.extend(LabelledPrompt(text, None) for text in board.background_prompts)
        return cls(tuple(prompts))


# %% classification


@dataclass(frozen=True)
class CropClassification:
    """What a classifier made of one crop."""

    label: Optional[ShapeLabel]
    """The shape the crop shows, or nothing when a background description matched best."""

    confidence: float
    """How strongly the winning description matched, between zero and one."""


@dataclass
class CropClassifier(ABC):
    """Decides which board shape an image crop shows, or that it shows none.

    Keeping this an interface is what lets the pipeline be tested, and lets a different text image
    model be tried, without loading one.
    """

    board: BoardConfiguration
    """The shape vocabulary the classifier reports against."""

    @abstractmethod
    def classify(self, crops: list[Image]) -> list[CropClassification]:
        """Return one classification per crop, in the order the crops were given.

        Crops are classified together so a model that batches can do so.
        """
