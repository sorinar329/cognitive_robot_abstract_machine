from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path

import numpy as np
from openfast_io.FAST_reader import InputReader_OpenFAST
from openfast_io.FAST_writer import InputWriter_OpenFAST
from typing_extensions import Any, Self

from semantic_digital_twin.adapters.openfast.simulation import (
    BLADE_PITCH_CHANNELS,
    OutputChannel,
)
from semantic_digital_twin.exceptions import OpenFASTInstallationNotConfiguredError
from semantic_digital_twin.semantic_annotations.wind_turbine import (
    WindTurbineGeometry,
)

# %% installation


class OpenFASTEnvironmentVariable(StrEnum):
    """
    Environment variables that say where the files of an OpenFAST build are.
    """

    LIBRARY = "OPENFAST_LIBRARY"
    """
    Path of the OpenFAST shared library (``libopenfastlib.so``).
    """

    CONTROLLER = "OPENFAST_CONTROLLER"
    """
    Path of the turbine controller library in the Bladed format (``DISCON``).
    """

    TEMPLATE_INPUT_FILE = "OPENFAST_TEMPLATE_INPUT_FILE"
    """
    Path of the main input file (``.fst``) of the turbine to simulate.
    """


@dataclass
class OpenFASTInstallation:
    """
    The files of an OpenFAST build needed to simulate a turbine.
    """

    library: Path
    """
    Path of the OpenFAST shared library (``libopenfastlib.so``).
    """

    controller: Path
    """
    Path of the turbine controller library in the Bladed format (``DISCON``).
    """

    template_input_file: Path
    """
    Path of the main input file (``.fst``) of the turbine to simulate.
    """

    @classmethod
    def is_configured(cls) -> bool:
        """
        :return: Whether every environment variable of :meth:`from_environment` is set.
        """
        return all(variable in os.environ for variable in OpenFASTEnvironmentVariable)

    @classmethod
    def from_environment(cls) -> Self:
        """
        Read the installation from the :class:`OpenFASTEnvironmentVariable` variables.

        :raises OpenFASTInstallationNotConfiguredError: If a variable is not set.
        """
        for variable in OpenFASTEnvironmentVariable:
            if variable not in os.environ:
                raise OpenFASTInstallationNotConfiguredError(
                    environment_variable=variable
                )
        return cls(
            library=Path(os.environ[OpenFASTEnvironmentVariable.LIBRARY]),
            controller=Path(os.environ[OpenFASTEnvironmentVariable.CONTROLLER]),
            template_input_file=Path(
                os.environ[OpenFASTEnvironmentVariable.TEMPLATE_INPUT_FILE]
            ),
        )


# %% input file vocabulary


class InputModule(StrEnum):
    """
    The OpenFAST modules whose input files the deck reads or changes, named as
    ``openfast_io`` names them.
    """

    MAIN = "Fst"
    ELASTODYN = "ElastoDyn"
    INFLOW_WIND = "InflowWind"
    AERODYN = "AeroDyn"
    AERODYN_BLADE = "AeroDynBlade"
    SERVODYN = "ServoDyn"
    OUTPUT_LIST = "outlist"


class MainVariable(StrEnum):
    """
    Variables of the main input file.
    """

    END_TIME = "TMax"


class ElastoDynVariable(StrEnum):
    """
    Variables of the ElastoDyn input file, which describes the turbine structure.
    """

    TOWER_HEIGHT = "TowerHt"
    TOWER_TOP_TO_SHAFT = "Twr2Shft"
    OVERHANG = "OverHang"
    HUB_RADIUS = "HubRad"
    TIP_RADIUS = "TipRad"
    NUMBER_OF_BLADES = "NumBl"


class AeroDynVariable(StrEnum):
    """
    Variables of the AeroDyn input files, which describe the aerodynamic shape.
    """

    TOWER_DIAMETERS = "TwrDiam"
    BLADE_CHORDS = "BlChord"


class InflowWindVariable(StrEnum):
    """
    Variables of the InflowWind input file, which describes the wind.
    """

    WIND_TYPE = "WindType"
    HUB_HEIGHT_WIND_SPEED = "HWindSpeed"
    PROPAGATION_DIRECTION = "PropagationDir"


class ServoDynVariable(StrEnum):
    """
    Variables of the ServoDyn input file, which describes the controller.
    """

    CONTROLLER_LIBRARY = "DLL_FileName"


class WindType(IntEnum):
    """
    Kinds of wind InflowWind can produce.
    """

    STEADY = 1


OUTPUT_CHANNEL_MODULES: dict[OutputChannel, InputModule] = {
    OutputChannel.ROTOR_AZIMUTH: InputModule.ELASTODYN,
    OutputChannel.ROTOR_SPEED: InputModule.ELASTODYN,
    OutputChannel.NACELLE_YAW: InputModule.ELASTODYN,
    **{channel: InputModule.ELASTODYN for channel in BLADE_PITCH_CHANNELS},
    OutputChannel.GENERATOR_POWER: InputModule.SERVODYN,
    OutputChannel.HUB_HEIGHT_WIND_SPEED: InputModule.INFLOW_WIND,
}
"""
The module whose output list provides each output channel.
"""


# %% wind


@dataclass
class SteadyWind:
    """
    Wind of constant speed and direction.
    """

    speed: float
    """
    Wind speed at hub height in meters per second.
    """

    propagation_direction: float = 0.0
    """
    Direction the wind blows towards, in radians, measured from the turbine's downwind
    x axis; positive values turn the direction towards the negative y axis.
    """


# %% input deck


@dataclass
class OpenFASTInputDeck:
    """
    The input files of an OpenFAST turbine simulation, read from a template and written
    as configured copies.
    """

    template_input_file: Path
    """
    The main input file (``.fst``) of the template.
    """

    controller_library: Path
    """
    The turbine controller library in the Bladed format (``DISCON``) the written copies
    use.
    """

    _variables: dict[str, Any] = field(init=False, repr=False)
    """
    The variables of every input file, as ``openfast_io`` reads them.
    """

    def __post_init__(self):
        reader = InputReader_OpenFAST()
        reader.FAST_InputFile = self.template_input_file.name
        reader.FAST_directory = str(self.template_input_file.parent)
        reader.execute()
        self._variables = reader.fst_vt

    def turbine_geometry(self) -> WindTurbineGeometry:
        """
        :return: The dimensions of the turbine the input files describe. The tower is
            drawn with its widest diameter.
        """
        elastodyn = self._variables[InputModule.ELASTODYN]
        return WindTurbineGeometry(
            tower_height=elastodyn[ElastoDynVariable.TOWER_HEIGHT],
            tower_diameter=max(
                self._variables[InputModule.AERODYN][AeroDynVariable.TOWER_DIAMETERS]
            ),
            tower_top_to_shaft=elastodyn[ElastoDynVariable.TOWER_TOP_TO_SHAFT],
            overhang=elastodyn[ElastoDynVariable.OVERHANG],
            hub_radius=elastodyn[ElastoDynVariable.HUB_RADIUS],
            tip_radius=elastodyn[ElastoDynVariable.TIP_RADIUS],
            blade_chord=max(
                self._first_blade_aerodynamics()[AeroDynVariable.BLADE_CHORDS]
            ),
            number_of_blades=elastodyn[ElastoDynVariable.NUMBER_OF_BLADES],
        )

    def write(self, run_directory: Path, wind: SteadyWind, duration: float) -> Path:
        """
        Write a copy of the input files that simulates the turbine in ``wind`` and
        outputs every :class:`OutputChannel`.

        :param run_directory: The directory the copy is written to; OpenFAST also
            writes its output files there.
        :param wind: The wind the turbine stands in.
        :param duration: The simulation time in seconds after which the simulation
            ends.
        :return: The main input file of the copy.
        """
        variables = copy.deepcopy(self._variables)
        variables[InputModule.MAIN][MainVariable.END_TIME] = duration
        inflow_wind = variables[InputModule.INFLOW_WIND]
        inflow_wind[InflowWindVariable.WIND_TYPE] = WindType.STEADY
        inflow_wind[InflowWindVariable.HUB_HEIGHT_WIND_SPEED] = wind.speed
        inflow_wind[InflowWindVariable.PROPAGATION_DIRECTION] = np.rad2deg(
            wind.propagation_direction
        )
        variables[InputModule.SERVODYN][ServoDynVariable.CONTROLLER_LIBRARY] = str(
            self.controller_library.absolute()
        )
        for channel, module in OUTPUT_CHANNEL_MODULES.items():
            variables[InputModule.OUTPUT_LIST][module][channel] = True

        writer = InputWriter_OpenFAST()
        writer.fst_vt = variables
        writer.FAST_runDirectory = str(run_directory)
        writer.FAST_namingOut = self.template_input_file.stem
        writer.execute()
        return run_directory / self.template_input_file.name

    def _first_blade_aerodynamics(self) -> dict[str, Any]:
        """
        :return: The aerodynamic blade variables of the first blade; ``openfast_io``
            keeps a single entry when all blades are alike.
        """
        blades = self._variables[InputModule.AERODYN_BLADE]
        return blades[0] if isinstance(blades, list) else blades
