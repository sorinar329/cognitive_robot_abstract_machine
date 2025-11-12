from dataclasses import dataclass

from giskardpy.motion_statechart.auxilary_variable_manager import (
    AuxiliaryVariableManager,
)
from semantic_digital_twin.world import World


@dataclass
class BuildContext:
    world: World
    auxiliary_variable_manager: AuxiliaryVariableManager


@dataclass
class ExecutionContext:
    world: World
