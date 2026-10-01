"""
Simulate a wind turbine with OpenFAST and show it in RViz.

The turbine is built in a semantic digital twin world from the dimensions in its
OpenFAST input files; OpenFAST then turns the rotor, yaws the nacelle and pitches the
blades, and every simulated state is mirrored into the world in real time.

The OpenFAST files default to the ``OPENFAST_*`` environment variables (see
:class:`OpenFASTEnvironmentVariable`). In RViz, set the fixed frame to ``map`` and add a
``MarkerArray`` display on ``/semworld/viz_marker``.
"""

from __future__ import annotations

import argparse
import logging
import signal
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from semantic_digital_twin.adapters.openfast.input_deck import (
    OpenFASTEnvironmentVariable,
    OpenFASTInputDeck,
    OpenFASTInstallation,
    SteadyWind,
)
from semantic_digital_twin.adapters.openfast.simulated_wind_turbine import (
    SimulatedWindTurbine,
)
from semantic_digital_twin.adapters.openfast.simulation import (
    OpenFASTSimulation,
    SharedOpenFASTLibrary,
)
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.semantic_annotations.wind_turbine import WindTurbine
from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

PUBLISH_RATE = 20.0
"""
How often per second of wall-clock time the world is advanced and published.
"""

# %% real-time stepping


@dataclass
class RealTimeTurbineRun:
    """
    Advances a simulated wind turbine at the speed of wall-clock time from a ROS timer.
    """

    simulated_wind_turbine: SimulatedWindTurbine
    """
    The turbine to advance.
    """

    simulation: OpenFASTSimulation
    """
    The simulation that moves the turbine.
    """

    node: Node
    """
    The ROS node whose timer advances the turbine.
    """

    finished: threading.Event = field(default_factory=threading.Event)
    """
    Set once the simulation has reached its end time.
    """

    _last_logged_second: int = field(init=False, default=-1)

    def start(self) -> None:
        """
        Start the simulation and the timer that advances it.
        """
        self.simulated_wind_turbine.start()
        self.node.create_timer(1.0 / PUBLISH_RATE, self._advance)

    def _advance(self) -> None:
        """
        Advance the simulation by the simulated time of one timer period.
        """
        steps_per_period = round(1.0 / PUBLISH_RATE / self.simulation.time_step)
        for _ in range(steps_per_period):
            if self.simulation.is_finished:
                self.finished.set()
                return
            state = self.simulated_wind_turbine.step()
        if int(state.time) != self._last_logged_second:
            self._last_logged_second = int(state.time)
            logger.info(
                "t = %5.1f s  wind %4.1f m/s  rotor %5.2f rpm  pitch %5.2f deg  "
                "power %7.1f kW",
                state.time,
                state.hub_height_wind_speed,
                state.rotor_speed * 60 / (2 * np.pi),
                np.rad2deg(state.blade_pitches[0]),
                state.generator_power / 1e3,
            )


# %% command line


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    installation = (
        OpenFASTInstallation.from_environment()
        if OpenFASTInstallation.is_configured()
        else None
    )
    for name, variable, default in (
        (
            "--library",
            OpenFASTEnvironmentVariable.LIBRARY,
            installation.library if installation else None,
        ),
        (
            "--controller",
            OpenFASTEnvironmentVariable.CONTROLLER,
            installation.controller if installation else None,
        ),
        (
            "--input-file",
            OpenFASTEnvironmentVariable.TEMPLATE_INPUT_FILE,
            installation.template_input_file if installation else None,
        ),
    ):
        parser.add_argument(
            name,
            type=Path,
            default=default,
            required=default is None,
            help=f"defaults to ${variable}",
        )
    parser.add_argument("--wind-speed", type=float, default=8.0, help="m/s")
    parser.add_argument(
        "--wind-direction",
        type=float,
        default=0.0,
        help="direction the wind blows towards, degrees from the downwind x axis",
    )
    parser.add_argument("--duration", type=float, default=120.0, help="seconds")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = parse_arguments()

    deck = OpenFASTInputDeck(
        template_input_file=arguments.input_file,
        controller_library=arguments.controller,
    )
    world = World.create_with_root_body()
    wind_turbine = WindTurbine.create_with_new_bodies_in_world(
        "wind_turbine", world, deck.turbine_geometry()
    )
    run_directory = Path(tempfile.mkdtemp(prefix="openfast_"))
    simulation = OpenFASTSimulation(
        library=SharedOpenFASTLibrary(path=arguments.library),
        input_file=deck.write(
            run_directory,
            SteadyWind(
                speed=arguments.wind_speed,
                propagation_direction=np.deg2rad(arguments.wind_direction),
            ),
            duration=arguments.duration,
        ),
    )
    logger.info("OpenFAST input and output files: %s", run_directory)

    rclpy.init()
    node = rclpy.create_node("openfast_wind_turbine")
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    tf_publisher = TFPublisher(_world=world, node=node)
    VizMarkerPublisher(_world=world, node=node, tf_publisher=tf_publisher)

    run = RealTimeTurbineRun(
        simulated_wind_turbine=SimulatedWindTurbine(
            wind_turbine=wind_turbine, simulation=simulation
        ),
        simulation=simulation,
        node=node,
    )
    # Ending the run however it is asked for lets the shutdown below run in order:
    # ROS first, so no step is in progress, then OpenFAST.
    node.context.on_shutdown(run.finished.set)
    for interruption in (signal.SIGINT, signal.SIGTERM):
        signal.signal(interruption, lambda *_: run.finished.set())
    run.start()
    run.finished.wait()
    if rclpy.ok():
        rclpy.shutdown()
    spin_thread.join()
    simulation.stop()


if __name__ == "__main__":
    main()
