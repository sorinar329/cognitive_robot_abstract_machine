# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the semantic_digital_twin package.
# Dataclasses can be mapped automatically to the ORM model
# using the ORMatic library, they just have to be registered in the classes list.
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
import logging
import os
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np

import segmind

# imported for its alternative mappings, which are collected through a global subclass
# scan: without it segmind's spatial fields degrade to JSON columns
import semantic_digital_twin.orm.model
import coraplex.orm.model
import coraplex.orm.ormatic_interface
from krrood.ormatic.custom_types import NumpyType
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module

import segmind.episode_player
import segmind.players.csv_player
import segmind.players.data_player
import segmind.players.json_player
from segmind.episode_annotation import EpisodeAnnotator
from segmind.event_segmentation import Segmind
from segmind.exceptions import EpisodeWithoutFrames, NoSemanticAnnotationToWatch

ignored_classes = {
    SubclassJSONSerializer,
    Segmind,
    NoSemanticAnnotationToWatch,
    EpisodeAnnotator,
    EpisodeWithoutFrames,
}
# the players are threads that replay recorded files, not data
for player_module in (
    segmind.episode_player,
    segmind.players.data_player,
    segmind.players.csv_player,
    segmind.players.json_player,
):
    ignored_classes |= set(classes_of_module(player_module))

dependencies = [coraplex.orm.ormatic_interface]

type_mappings = {np.ndarray: NumpyType}

# Create an ORMatic object with the classes to be mapped
ormatic = ORMatic.from_package([segmind], dependencies, ignored_classes, type_mappings)
logging.getLogger("krrood").setLevel(logging.DEBUG)


# Generate the ORM classes
ormatic.make_all_tables()

ormatic_interface_path = (
    Path(__file__).parents[1] / "src" / "segmind" / "orm" / "ormatic_interface.py"
)
with open(ormatic_interface_path, "w") as f:
    ormatic.to_sqlalchemy_file(f)
