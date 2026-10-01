from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from openfast_io.FAST_reader import InputReader_OpenFAST
from typing_extensions import List

from semantic_digital_twin.adapters.openfast.input_deck import (
    ElastoDynVariable,
    InputModule,
    OpenFASTInputDeck,
    OpenFASTInstallation,
    SteadyWind,
)
from semantic_digital_twin.adapters.openfast.simulated_wind_turbine import (
    SimulatedWindTurbine,
)
from semantic_digital_twin.adapters.openfast.simulation import (
    OpenFASTLibrary,
    OpenFASTSimulation,
    OutputChannel,
    SharedOpenFASTLibrary,
    SimulationSizes,
    TurbineSimulation,
    TurbineState,
)
from semantic_digital_twin.exceptions import (
    MissingOpenFASTOutputChannelError,
    OpenFASTSimulationFinishedError,
)
from semantic_digital_twin.semantic_annotations.wind_turbine import (
    WindTurbine,
    WindTurbineGeometry,
)
from semantic_digital_twin.world import World

# %% mimics


@dataclass
class ScriptedOpenFASTLibrary(OpenFASTLibrary):
    """
    Plays back prepared output rows instead of running OpenFAST.
    """

    sizes: SimulationSizes
    rows: List[np.ndarray]
    _next_row: int = field(init=False, default=0)

    def allocate_turbines(self, number_of_turbines: int) -> None:
        pass

    def read_input_file(self, input_file: Path) -> SimulationSizes:
        return self.sizes

    def start(self, outputs: np.ndarray) -> None:
        self._write_next_row(outputs)

    def update(self, outputs: np.ndarray) -> None:
        self._write_next_row(outputs)

    def end(self) -> None:
        pass

    def _write_next_row(self, outputs: np.ndarray) -> None:
        outputs[:] = self.rows[self._next_row]
        self._next_row += 1


@dataclass
class RecordedTurbineSimulation(TurbineSimulation):
    """
    Returns prepared turbine states one after another.
    """

    states: List[TurbineState]
    _next_state: int = field(init=False, default=0)

    @property
    def time_step(self) -> float:
        return self.states[1].time - self.states[0].time

    def start(self) -> TurbineState:
        return self._next()

    def step(self) -> TurbineState:
        return self._next()

    def stop(self) -> None:
        pass

    def _next(self) -> TurbineState:
        state = self.states[self._next_state]
        self._next_state += 1
        return state


# %% fixtures


@pytest.fixture
def channel_values() -> dict[str, float]:
    return {
        "UnrelatedChannel": -1.0,
        OutputChannel.TIME: 1.5,
        OutputChannel.ROTOR_AZIMUTH: 90.0,
        OutputChannel.ROTOR_SPEED: 12.0,
        OutputChannel.NACELLE_YAW: 10.0,
        OutputChannel.BLADE_1_PITCH: 1.0,
        OutputChannel.BLADE_2_PITCH: 2.0,
        OutputChannel.BLADE_3_PITCH: 3.0,
        OutputChannel.GENERATOR_POWER: 1500.0,
        OutputChannel.HUB_HEIGHT_WIND_SPEED: 8.0,
    }


def output_row(channel_values: dict[str, float], **changes: float) -> np.ndarray:
    values = {**channel_values, **changes}
    return np.array(list(values.values()))


@pytest.fixture
def wind_turbine() -> WindTurbine:
    geometry = WindTurbineGeometry(
        tower_height=80.0,
        tower_diameter=5.0,
        tower_top_to_shaft=2.0,
        overhang=-5.0,
        hub_radius=1.5,
        tip_radius=60.0,
        blade_chord=4.0,
        number_of_blades=3,
    )
    return WindTurbine.create_with_new_bodies_in_world(
        "turbine", World.create_with_root_body(), geometry
    )


@pytest.fixture
def openfast_installation() -> OpenFASTInstallation:
    if not OpenFASTInstallation.is_configured():
        pytest.skip("OpenFAST is not installed in this environment.")
    return OpenFASTInstallation.from_environment()


# %% turbine state


def test_turbine_state_converts_openfast_units_to_si_units(channel_values):
    state = TurbineState.from_output_row(
        list(channel_values), list(channel_values.values())
    )
    assert state.time == channel_values[OutputChannel.TIME]
    assert state.rotor_azimuth == np.deg2rad(
        channel_values[OutputChannel.ROTOR_AZIMUTH]
    )
    assert state.rotor_speed == pytest.approx(
        channel_values[OutputChannel.ROTOR_SPEED] * 2 * np.pi / 60
    )
    assert state.nacelle_yaw == np.deg2rad(channel_values[OutputChannel.NACELLE_YAW])
    assert state.blade_pitches == [
        np.deg2rad(channel_values[channel])
        for channel in (
            OutputChannel.BLADE_1_PITCH,
            OutputChannel.BLADE_2_PITCH,
            OutputChannel.BLADE_3_PITCH,
        )
    ]
    assert state.generator_power == channel_values[OutputChannel.GENERATOR_POWER] * 1e3
    assert state.hub_height_wind_speed == (
        channel_values[OutputChannel.HUB_HEIGHT_WIND_SPEED]
    )


def test_turbine_state_requires_every_channel_it_reads(channel_values):
    del channel_values[OutputChannel.ROTOR_SPEED]
    with pytest.raises(MissingOpenFASTOutputChannelError) as error:
        TurbineState.from_output_row(
            list(channel_values), list(channel_values.values())
        )
    assert error.value.channel == OutputChannel.ROTOR_SPEED


# %% simulation


def scripted_simulation(
    channel_values: dict[str, float], number_of_rows: int, end_time: float
) -> OpenFASTSimulation:
    time_step = 0.5
    library = ScriptedOpenFASTLibrary(
        sizes=SimulationSizes(
            channel_names=list(channel_values),
            time_step=time_step,
            end_time=end_time,
        ),
        rows=[
            output_row(channel_values, **{OutputChannel.TIME: row * time_step})
            for row in range(number_of_rows)
        ],
    )
    return OpenFASTSimulation(library=library, input_file=Path("turbine.fst"))


def test_simulation_reports_the_state_of_each_step(channel_values):
    simulation = scripted_simulation(channel_values, number_of_rows=3, end_time=1.0)
    assert simulation.start().time == 0.0
    assert simulation.step().time == simulation.time_step
    assert simulation.step().time == 2 * simulation.time_step


def test_simulation_is_finished_once_it_reaches_its_end_time(channel_values):
    simulation = scripted_simulation(channel_values, number_of_rows=3, end_time=1.0)
    simulation.start()
    simulation.step()
    assert not simulation.is_finished
    simulation.step()
    assert simulation.is_finished


def test_simulation_refuses_to_step_past_its_end_time(channel_values):
    simulation = scripted_simulation(channel_values, number_of_rows=3, end_time=1.0)
    simulation.start()
    simulation.step()
    simulation.step()
    with pytest.raises(OpenFASTSimulationFinishedError):
        simulation.step()


# %% simulated wind turbine


def test_simulated_motion_is_mirrored_into_the_world(wind_turbine, channel_values):
    states = [
        TurbineState.from_output_row(
            list(channel_values),
            output_row(channel_values, **{OutputChannel.TIME: time}),
        )
        for time in (0.0, 0.5)
    ]
    states[1].rotor_azimuth = 1.0
    simulated_wind_turbine = SimulatedWindTurbine(
        wind_turbine=wind_turbine, simulation=RecordedTurbineSimulation(states)
    )
    simulated_wind_turbine.start()
    state = simulated_wind_turbine.step()

    assert state is states[1]
    assert wind_turbine.rotor_connection.position == state.rotor_azimuth
    assert wind_turbine.rotor_connection.velocity == state.rotor_speed
    assert wind_turbine.yaw_connection.position == state.nacelle_yaw
    assert [
        connection.position for connection in wind_turbine.pitch_connections
    ] == state.blade_pitches


# %% live OpenFAST runs


def test_input_deck_takes_the_turbine_geometry_from_its_input_files(
    openfast_installation,
):
    deck = OpenFASTInputDeck(
        template_input_file=openfast_installation.template_input_file,
        controller_library=openfast_installation.controller,
    )
    reader = InputReader_OpenFAST()
    reader.FAST_InputFile = openfast_installation.template_input_file.name
    reader.FAST_directory = str(openfast_installation.template_input_file.parent)
    reader.execute()
    elastodyn = reader.fst_vt[InputModule.ELASTODYN]

    geometry = deck.turbine_geometry()

    assert geometry.tower_height == elastodyn[ElastoDynVariable.TOWER_HEIGHT]
    assert geometry.overhang == elastodyn[ElastoDynVariable.OVERHANG]
    assert geometry.tip_radius == elastodyn[ElastoDynVariable.TIP_RADIUS]
    assert geometry.number_of_blades == elastodyn[ElastoDynVariable.NUMBER_OF_BLADES]


def test_openfast_turns_the_rotor_in_steady_wind(openfast_installation, tmp_path):
    deck = OpenFASTInputDeck(
        template_input_file=openfast_installation.template_input_file,
        controller_library=openfast_installation.controller,
    )
    wind = SteadyWind(speed=8.0)
    simulation = OpenFASTSimulation(
        library=SharedOpenFASTLibrary(path=openfast_installation.library),
        input_file=deck.write(tmp_path, wind, duration=3.0),
    )
    initial_state = simulation.start()
    for _ in range(round(2.0 / simulation.time_step)):
        state = simulation.step()
    simulation.stop()

    assert state.time == pytest.approx(2.0)
    assert state.hub_height_wind_speed == pytest.approx(wind.speed)
    assert state.rotor_speed > 0.0
    assert state.rotor_azimuth != initial_state.rotor_azimuth
