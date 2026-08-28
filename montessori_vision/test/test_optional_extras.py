"""The package has to stay usable without the model runtimes installed.

Every backend costs gigabytes to install, so a robot that only runs the trained detector must not be
made to install a segmentation model, and a laptop reading annotations must install neither.
"""

from __future__ import annotations

import importlib
import sys

import pytest

HEAVY_DEPENDENCIES = ("torch", "ultralytics", "bpy")
"""The runtimes that only the optional extras bring, none of which a plain import may pull in."""

LIGHT_MODULES = (
    "montessori_vision",
    "montessori_vision.board",
    "montessori_vision.dataset",
    "montessori_vision.detections",
    "montessori_vision.evaluation",
    "montessori_vision.geometry",
    "montessori_vision.image",
    "montessori_vision.pipeline",
    "montessori_vision.segment_and_classify",
    "montessori_vision.synthetic",
    "montessori_vision.yolo",
)
"""The modules that must import with nothing but the core dependencies installed."""


@pytest.mark.parametrize("module_name", LIGHT_MODULES)
def test_a_module_imports_without_the_optional_extras(module_name: str) -> None:
    importlib.import_module(module_name)
    assert not [name for name in HEAVY_DEPENDENCIES if name in sys.modules]
