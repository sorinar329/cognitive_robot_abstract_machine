"""CLIP as the classifier that names a crop.

.. warning:: This module imports torch and ``open_clip``, both of which come with the
    ``segment_and_classify`` extra. Import it only where those are installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import open_clip
import torch
from PIL import Image as PillowImage
from typing_extensions import Callable

from montessori_vision.image import Image
from montessori_vision.segment_and_classify.classifier import (
    CropClassification,
    CropClassifier,
    PromptSet,
)
from montessori_vision.segment_and_classify.device import TorchDevice


@dataclass
class LoadedClip:
    """The three pieces open_clip hands out together and that are only useful together."""

    network: torch.nn.Module
    """
    The model that embeds images and text into the same space.
    """

    preprocess: Callable[[PillowImage.Image], torch.Tensor]
    """The transform that turns a crop into the tensor the model expects."""

    tokenize: Callable[[list[str]], torch.Tensor]
    """
    The tokenizer that turns descriptions into the tokens the model expects.
    """


@dataclass
class ClipClassifier(CropClassifier):
    """Names a crop by scoring it against the text descriptions the board configuration provides.

    Nothing is trained: adding a shape to the board means adding its descriptions to the
    configuration, which is the whole appeal of this approach.
    """

    architecture: str = "ViT-B-32"
    """The open_clip model architecture to load."""

    weights: str = "laion2b_s34b_b79k"
    """The name of the pretrained weights open_clip downloads for that architecture."""

    device: TorchDevice = field(default_factory=TorchDevice.available)
    """The device the model runs on."""

    @cached_property
    def prompts(self) -> PromptSet:
        """The descriptions every crop is scored against."""
        return PromptSet.from_board(self.board)

    @cached_property
    def model(self) -> LoadedClip:
        """The loaded model, in evaluation mode on the configured device."""
        network, _, preprocess = open_clip.create_model_and_transforms(
            self.architecture, pretrained=self.weights
        )
        network.to(self.device)
        network.eval()
        return LoadedClip(
            network=network,
            preprocess=preprocess,
            tokenize=open_clip.get_tokenizer(self.architecture),
        )

    @cached_property
    def prompt_features(self) -> torch.Tensor:
        """The normalised text embedding of every description.

        The descriptions never change while a pipeline runs, so they are embedded once.
        """
        tokens = self.model.tokenize(list(self.prompts.texts)).to(self.device)
        with torch.no_grad():
            features = self.model.network.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    def classify(self, crops: list[Image]) -> list[CropClassification]:
        """Score every crop against every description and report the winning one."""
        if not crops:
            return []
        batch = torch.stack(
            [self.model.preprocess(PillowImage.fromarray(crop.pixels)) for crop in crops]
        ).to(self.device)
        with torch.no_grad():
            features = self.model.network.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            # CLIP's own logit scale is what makes the softmax discriminative rather than flat,
            # which is what the pipeline's confidence threshold is compared against.
            scale = self.model.network.logit_scale.exp()
            similarities = (scale * features @ self.prompt_features.T).softmax(dim=-1)
        confidences, winners = similarities.max(dim=-1)
        return [
            CropClassification(
                label=self.prompts.prompts[int(winner)].label, confidence=float(confidence)
            )
            for winner, confidence in zip(winners, confidences)
        ]
