from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.adapters.openfast.simulation import (
    TurbineSimulation,
    TurbineState,
)
from semantic_digital_twin.semantic_annotations.wind_turbine import WindTurbine


@dataclass
class SimulatedWindTurbine:
    """
    A wind turbine in a world whose motion comes from a turbine simulation.

    Every simulated state is written into the turbine's yaw, rotor and pitch
    connections, so the world shows the turbine as the simulation moves it.
    """

    wind_turbine: WindTurbine
    """
    The turbine in the world that mirrors the simulation.
    """

    simulation: TurbineSimulation
    """
    The simulation that moves the turbine.
    """

    state: TurbineState = field(init=False)
    """
    The latest simulated state, known after :meth:`start`.
    """

    def start(self) -> TurbineState:
        """
        Start the simulation and move the turbine to its initial state.

        :return: The initial turbine state.
        """
        return self._mirror(self.simulation.start())

    def step(self) -> TurbineState:
        """
        Advance the simulation by one time step and move the turbine accordingly.

        :return: The turbine state after the step.
        """
        return self._mirror(self.simulation.step())

    def stop(self) -> None:
        """
        Stop the simulation.
        """
        self.simulation.stop()

    def _mirror(self, state: TurbineState) -> TurbineState:
        """
        Write ``state`` into the world state of the turbine's connections.
        """
        world_state = self.wind_turbine._world.state
        rotor_degree_of_freedom = self.wind_turbine.rotor_connection.raw_dof.id
        world_state[rotor_degree_of_freedom].position = state.rotor_azimuth
        world_state[rotor_degree_of_freedom].velocity = state.rotor_speed
        world_state[self.wind_turbine.yaw_connection.raw_dof.id].position = (
            state.nacelle_yaw
        )
        for connection, pitch in zip(
            self.wind_turbine.pitch_connections, state.blade_pitches
        ):
            world_state[connection.raw_dof.id].position = pitch
        self.wind_turbine._world.notify_state_change()
        self.state = state
        return state
