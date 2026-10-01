from __future__ import annotations

from abc import ABC, abstractmethod
from ctypes import (
    CDLL,
    Array,
    POINTER,
    byref,
    c_bool,
    c_char,
    c_double,
    c_int,
    create_string_buffer,
)
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path

import numpy as np
from typing_extensions import Callable, List, Self, Sequence

from semantic_digital_twin.exceptions import (
    MissingOpenFASTOutputChannelError,
    OpenFASTError,
    OpenFASTSimulationFinishedError,
)

# %% output channels


class OutputChannel(StrEnum):
    """
    The OpenFAST output channels that describe the turbine state, named as OpenFAST
    names them in its outputs.
    """

    TIME = "Time"
    """
    Simulation time in seconds.
    """

    ROTOR_AZIMUTH = "Azimuth"
    """
    Angle of the rotor about the shaft in degrees; 0 has the first blade pointing up.
    """

    ROTOR_SPEED = "RotSpeed"
    """
    Rotor speed in revolutions per minute.
    """

    NACELLE_YAW = "NacYaw"
    """
    Nacelle yaw angle in degrees.
    """

    BLADE_1_PITCH = "BldPitch1"
    """
    Pitch angle of the first blade in degrees, positive towards feather.
    """

    BLADE_2_PITCH = "BldPitch2"
    """
    Pitch angle of the second blade in degrees, positive towards feather.
    """

    BLADE_3_PITCH = "BldPitch3"
    """
    Pitch angle of the third blade in degrees, positive towards feather.
    """

    GENERATOR_POWER = "GenPwr"
    """
    Electrical power of the generator in kilowatts.
    """

    HUB_HEIGHT_WIND_SPEED = "Wind1VelX"
    """
    Wind speed along the inflow x axis at hub height in meters per second.
    """


BLADE_PITCH_CHANNELS = (
    OutputChannel.BLADE_1_PITCH,
    OutputChannel.BLADE_2_PITCH,
    OutputChannel.BLADE_3_PITCH,
)
"""
The blade pitch channels, ordered by blade number.
"""

REVOLUTIONS_PER_MINUTE_TO_RADIANS_PER_SECOND = 2 * np.pi / 60
WATTS_PER_KILOWATT = 1e3


@dataclass
class TurbineState:
    """
    The state of a simulated wind turbine at one moment, in SI units.
    """

    time: float
    """
    Simulation time in seconds.
    """

    rotor_azimuth: float
    """
    Angle of the rotor about the shaft in radians; 0 has the first blade pointing up.
    """

    rotor_speed: float
    """
    Rotor speed in radians per second.
    """

    nacelle_yaw: float
    """
    Nacelle yaw angle in radians.
    """

    blade_pitches: List[float]
    """
    Pitch angle of each blade in radians, positive towards feather, ordered by blade
    number.
    """

    generator_power: float
    """
    Electrical power of the generator in watts.
    """

    hub_height_wind_speed: float
    """
    Wind speed along the inflow x axis at hub height in meters per second.
    """

    @classmethod
    def from_output_row(
        cls, channel_names: Sequence[str], values: Sequence[float]
    ) -> Self:
        """
        Read the turbine state from one row of OpenFAST outputs.

        :param channel_names: The name of each output channel, in output order.
        :param values: The value of each output channel, in output order.
        :return: The turbine state the row describes.
        :raises MissingOpenFASTOutputChannelError: If a channel the state needs is not
            output.
        """
        value_of = cls._channel_reader(channel_names, values)
        return cls(
            time=value_of(OutputChannel.TIME),
            rotor_azimuth=np.deg2rad(value_of(OutputChannel.ROTOR_AZIMUTH)),
            rotor_speed=value_of(OutputChannel.ROTOR_SPEED)
            * REVOLUTIONS_PER_MINUTE_TO_RADIANS_PER_SECOND,
            nacelle_yaw=np.deg2rad(value_of(OutputChannel.NACELLE_YAW)),
            blade_pitches=[
                np.deg2rad(value_of(channel)) for channel in BLADE_PITCH_CHANNELS
            ],
            generator_power=value_of(OutputChannel.GENERATOR_POWER)
            * WATTS_PER_KILOWATT,
            hub_height_wind_speed=value_of(OutputChannel.HUB_HEIGHT_WIND_SPEED),
        )

    @staticmethod
    def _channel_reader(
        channel_names: Sequence[str], values: Sequence[float]
    ) -> Callable[[OutputChannel], float]:
        """
        :return: A function that returns the value of an output channel by name.
        """
        index_of_channel = {name: index for index, name in enumerate(channel_names)}

        def value_of(channel: OutputChannel) -> float:
            if channel not in index_of_channel:
                raise MissingOpenFASTOutputChannelError(
                    channel=channel, available_channels=list(channel_names)
                )
            return float(values[index_of_channel[channel]])

        return value_of


# %% simulations


class TurbineSimulation(ABC):
    """
    A simulation of one wind turbine that is advanced one time step at a time.
    """

    @property
    @abstractmethod
    def time_step(self) -> float:
        """
        Simulated time per step in seconds.
        """

    @abstractmethod
    def start(self) -> TurbineState:
        """
        Initialize the simulation.

        :return: The initial turbine state.
        """

    @abstractmethod
    def step(self) -> TurbineState:
        """
        Advance the simulation by one time step.

        :return: The turbine state after the step.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Release everything the simulation holds.
        """


@dataclass
class SimulationSizes:
    """
    The sizes of an OpenFAST simulation, known once its input files are read.
    """

    channel_names: List[str]
    """
    The name of each output channel, in output order.
    """

    time_step: float
    """
    Simulated time per step in seconds.
    """

    end_time: float
    """
    The simulation time in seconds at which the simulation ends.
    """


class OpenFASTLibrary(ABC):
    """
    The time-stepping interface of the OpenFAST library for a single turbine.
    """

    @abstractmethod
    def allocate_turbines(self, number_of_turbines: int) -> None:
        """
        Allocate the data of the given number of turbines.
        """

    @abstractmethod
    def read_input_file(self, input_file: Path) -> SimulationSizes:
        """
        Read the input files of the turbine.

        :param input_file: The main OpenFAST input file (``.fst``).
        :return: The sizes of the simulation.
        """

    @abstractmethod
    def start(self, outputs: np.ndarray) -> None:
        """
        Initialize the simulation and write the initial outputs into ``outputs``.
        """

    @abstractmethod
    def update(self, outputs: np.ndarray) -> None:
        """
        Advance the simulation by one time step and write the outputs into
        ``outputs``.
        """

    @abstractmethod
    def end(self) -> None:
        """
        End the simulation and release the turbine data.
        """


class OpenFASTErrorLevel(IntEnum):
    """
    Error levels the OpenFAST library reports.
    """

    NONE = 0
    INFO = 1
    WARNING = 2
    SEVERE = 3
    FATAL = 4


ERROR_MESSAGE_LENGTH = 8197
"""
Length of the error message buffer: OpenFAST's ``ErrMsgLen`` plus the terminating null
character.
"""

CHANNEL_NAME_LENGTH = 20
"""
Number of characters OpenFAST reserves for one output channel name.
"""

MAXIMUM_NUMBER_OF_CHANNELS = 4000
"""
The most output channels the OpenFAST library reports (``MAXOUTPUTS``).
"""

NUMBER_OF_EXTERNAL_INPUTS = 51
"""
Length of the external input array the OpenFAST library expects
(``NumFixedInputs``).
"""


@dataclass
class SharedOpenFASTLibrary(OpenFASTLibrary):
    """
    The OpenFAST shared library (``libopenfastlib``), called through ctypes.
    """

    path: Path
    """
    Path of the shared library.
    """

    _library: CDLL = field(init=False, repr=False)
    """
    The loaded shared library.
    """

    _turbine_index: c_int = field(init=False, repr=False, default_factory=c_int)
    """
    Index of the simulated turbine.
    """

    _external_inputs: Array[c_double] = field(init=False, repr=False)
    """
    External control inputs; all zero, since the turbine controls itself.
    """

    def __post_init__(self):
        self._library = CDLL(str(self.path))
        self._external_inputs = (c_double * NUMBER_OF_EXTERNAL_INPUTS)()

    def allocate_turbines(self, number_of_turbines: int) -> None:
        status, message = self._error_buffers()
        self._library.FAST_AllocateTurbines(
            byref(c_int(number_of_turbines)), byref(status), message
        )
        self._raise_on_fatal_error("FAST_AllocateTurbines", status, message)

    def read_input_file(self, input_file: Path) -> SimulationSizes:
        status, message = self._error_buffers()
        abort_level = c_int(OpenFASTErrorLevel.FATAL)
        number_of_outputs, time_step, output_time_step, end_time = (
            c_int(),
            c_double(),
            c_double(),
            c_double(),
        )
        channel_names = create_string_buffer(
            CHANNEL_NAME_LENGTH * MAXIMUM_NUMBER_OF_CHANNELS + 1
        )
        self._library.FAST_Sizes(
            byref(self._turbine_index),
            create_string_buffer(str(input_file.absolute()).encode()),
            byref(abort_level),
            byref(number_of_outputs),
            byref(time_step),
            byref(output_time_step),
            byref(end_time),
            byref(status),
            message,
            channel_names,
            None,
            None,
        )
        self._raise_on_fatal_error("FAST_Sizes", status, message)
        return SimulationSizes(
            channel_names=[name.decode() for name in channel_names.value.split()],
            time_step=time_step.value,
            end_time=end_time.value,
        )

    def start(self, outputs: np.ndarray) -> None:
        status, message = self._error_buffers()
        self._library.FAST_Start(
            byref(self._turbine_index),
            byref(c_int(NUMBER_OF_EXTERNAL_INPUTS)),
            byref(c_int(len(outputs))),
            byref(self._external_inputs),
            outputs.ctypes.data_as(POINTER(c_double)),
            byref(status),
            message,
        )
        self._raise_on_fatal_error("FAST_Start", status, message)

    def update(self, outputs: np.ndarray) -> None:
        status, message = self._error_buffers()
        self._library.FAST_Update(
            byref(self._turbine_index),
            byref(c_int(NUMBER_OF_EXTERNAL_INPUTS)),
            byref(c_int(len(outputs))),
            byref(self._external_inputs),
            outputs.ctypes.data_as(POINTER(c_double)),
            byref(c_bool()),
            byref(status),
            message,
        )
        self._raise_on_fatal_error("FAST_Update", status, message)

    def end(self) -> None:
        status, message = self._error_buffers()
        self._library.FAST_End(byref(self._turbine_index), byref(c_bool(False)))
        self._library.FAST_DeallocateTurbines(byref(status), message)
        self._raise_on_fatal_error("FAST_DeallocateTurbines", status, message)

    @staticmethod
    def _error_buffers() -> tuple[c_int, Array[c_char]]:
        return c_int(OpenFASTErrorLevel.NONE), create_string_buffer(
            ERROR_MESSAGE_LENGTH
        )

    @staticmethod
    def _raise_on_fatal_error(
        routine: str, status: c_int, message: Array[c_char]
    ) -> None:
        if status.value >= OpenFASTErrorLevel.FATAL:
            raise OpenFASTError(routine=routine, message=message.value.decode())


@dataclass
class OpenFASTSimulation(TurbineSimulation):
    """
    An OpenFAST simulation of one wind turbine, advanced one time step at a time.
    """

    library: OpenFASTLibrary
    """
    The OpenFAST library that runs the simulation.
    """

    input_file: Path
    """
    The main OpenFAST input file (``.fst``).
    """

    sizes: SimulationSizes = field(init=False)
    """
    The sizes of the simulation, known after :meth:`start`.
    """

    state: TurbineState = field(init=False)
    """
    The latest turbine state, known after :meth:`start`.
    """

    _outputs: np.ndarray = field(init=False, repr=False)
    """
    The output row the library writes into.
    """

    @property
    def time_step(self) -> float:
        return self.sizes.time_step

    @property
    def is_finished(self) -> bool:
        """
        Whether the simulation has reached the end time set in its input file.
        """
        return self.state.time + self.time_step / 2 >= self.sizes.end_time

    def start(self) -> TurbineState:
        self.library.allocate_turbines(1)
        self.sizes = self.library.read_input_file(self.input_file)
        self._outputs = np.zeros(len(self.sizes.channel_names))
        self.library.start(self._outputs)
        return self._read_state()

    def step(self) -> TurbineState:
        """
        Advance the simulation by one time step.

        :return: The turbine state after the step.
        :raises OpenFASTSimulationFinishedError: If the simulation has reached the end
            time set in its input file.
        """
        if self.is_finished:
            raise OpenFASTSimulationFinishedError(end_time=self.sizes.end_time)
        self.library.update(self._outputs)
        return self._read_state()

    def stop(self) -> None:
        self.library.end()

    def _read_state(self) -> TurbineState:
        self.state = TurbineState.from_output_row(
            self.sizes.channel_names, self._outputs
        )
        return self.state
