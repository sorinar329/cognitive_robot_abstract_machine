"""
What goes wrong when the HSR plans a RoboCasa task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from krrood.exceptions import DataclassException
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class NoStandingPoseError(DataclassException):
    """
    Raised when the HSR finds no place on the floor to stand at that its arm reaches
    every tool frame goal from.
    """

    tool_frame_goals: list[np.ndarray]
    """
    The tool frame goals, in the kitchen's frame.
    """

    def error_message(self) -> str:
        return (
            "No place on the floor that the HSR's base can stand on and get to lets "
            "its arm reach the tool frame goals at "
            f"{[goal[:3, 3].round(3).tolist() for goal in self.tool_frame_goals]}."
        )

    def suggest_correction(self) -> str:
        return (
            "Check that the goals lie within the arm's reach of free floor, or that "
            "nothing stands between the HSR and that floor."
        )


@dataclass
class NoApproachError(DataclassException):
    """
    Raised when no approach the HSR searches lets its hand take hold of an object and
    its base stand where its arm reaches.
    """

    taken_object: Body
    """
    The object to be taken hold of.
    """

    def error_message(self) -> str:
        return (
            f"No approach to {self.taken_object.name} has a free path for the hand and "
            "a place on the floor to stand at, for picking it up and setting it down."
        )

    def suggest_correction(self) -> str:
        return "Search more tilts and headings, or check what surrounds the object."
