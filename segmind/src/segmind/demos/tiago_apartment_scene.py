"""
Shared world and simulator setup for the TIAGo++ apartment demos.

Builds a semantic world from the Tiago URDF for plan execution and opens a
pre-validated MuJoCo scene (Tiago with velocity actuators, apartment, and
objects).  A :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSynchronizer`
bridges the two: every world-state update from giskardpy is immediately
mirrored into the MuJoCo scene.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy

from coraplex.alternative_motion_mappings.tiago_motion_mapping import TiagoMoveSim
from coraplex.datastructures.dataclasses import Context
from physics_simulators.mujoco_simulator import MujocoSimulator
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from test.conftest import world_with_urdf_factory

_MJCF_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "resources",
        "tiago_episodes",
        "models",
        "assets",
        "mjcf",
    )
)

SCENE_FILE = os.path.join(_MJCF_DIR, "scene_with_apartment.xml")

ENVIRONMENT_JOINT_VELOCITY_LIMIT = 1.0
"""
Velocity limit assigned to environment joints parsed from MJCF, which carries
no velocity limits.  giskardpy requires velocity limits on every controlled
degree of freedom (for example when opening the fridge door).
"""


def build_tiago_world() -> World:
    """
    Build a semantic world for Tiago with a floor and the map → odom → base kinematic chain.

    :return: A world with the Tiago semantic annotation and a ground plane attached.
    """
    world = world_with_urdf_factory(Tiago, DifferentialDrive)
    _add_floor(world)
    return world


def _add_scene_objects(world: World) -> None:
    """
    Merge the apartment, milk box, and bowl into ``world``.

    Each sub-world is parsed from its MJCF file and merged via
    :meth:`~semantic_digital_twin.world.World.merge_world` so that all
    bodies (drawers, cabinets, objects) become queryable in the semantic world.

    :param world: The target world to merge objects into.
    """
    for filename in ("apartment.xml", "milk_box.xml", "bowl.xml"):
        sub_world = MJCFParser(
            file_path=os.path.join(_MJCF_DIR, filename)
        ).parse()
        _assign_missing_velocity_limits(sub_world)
        world.merge_world(
            sub_world,
            root_connection=FixedConnection(
                parent=world.root, child=sub_world.root
            ),
        )


def _assign_missing_velocity_limits(world: World) -> None:
    """
    Give every degree of freedom without velocity limits a default limit.

    ..note:: MJCF does not model joint velocity limits, but giskardpy requires
        them on every degree of freedom it controls.

    :param world: The parsed sub-world whose degrees of freedom are completed.
    """
    for degree_of_freedom in world.degrees_of_freedom:
        if degree_of_freedom.limits.upper.velocity is None:
            degree_of_freedom.limits.upper.velocity = (
                ENVIRONMENT_JOINT_VELOCITY_LIMIT
            )
            degree_of_freedom.limits.lower.velocity = (
                -ENVIRONMENT_JOINT_VELOCITY_LIMIT
            )


def _add_floor(world: World) -> None:
    """
    Add a ground-plane body to the world so MuJoCo has a surface to rest the robot on.

    A :class:`~semantic_digital_twin.world_description.geometry.Box` with z-scale zero
    is converted to a MuJoCo ``<geom type="plane">`` by the builder.

    :param world: The semantic world to add the floor to.
    """
    floor = Body(name=PrefixedName("floor"))
    # The plane sits 0.1m below z=0 so it neither blocks occupancy costmaps
    # (which ray-test down to z=0) nor triggers the 0.05m collision buffer
    # zone around the robot base during base-pose validation.
    floor_geom = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=-0.1, reference_frame=floor
        ),
        scale=Scale(x=10.0, y=10.0, z=0.0),
        color=Color(R=0.8, G=0.8, B=0.8, A=1.0),
    )
    floor.collision = ShapeCollection([floor_geom], reference_frame=floor)
    floor.visual = ShapeCollection([floor_geom], reference_frame=floor)
    with world.modify_world():
        world.add_connection(FixedConnection(parent=world.root, child=floor))


def build_context(world: World) -> Context:
    """
    Create a plan context for Tiago with the simulated navigation alternative motion.

    Pre-condition evaluation is disabled so the plan runs without requiring
    full environment perception.

    :param world: The semantic world containing the Tiago robot.
    :return: A context ready for plan execution.
    """
    context = Context.from_world(
        world, alternative_motion_mappings=[TiagoMoveSim]
    )
    context.evaluate_conditions = False
    return context


@dataclass
class ApartmentSimulation:
    """
    MuJoCo viewer for the apartment scene, mirrored from a semantic world.

    Physics feedback into the semantic world is disabled so giskardpy remains
    the single source of truth for the world state.
    """

    world: World
    """
    The semantic world driving the simulation.
    """

    headless: bool = False
    """
    Run MuJoCo without opening a viewer window.
    """

    simulator: MujocoSimulator = field(init=False)
    """
    The MuJoCo simulator displaying the pre-validated apartment scene.
    """

    synchronizer: MujocoSynchronizer = field(init=False)
    """
    Bridge mirroring semantic-world state updates into the MuJoCo scene.
    """

    def start(self) -> None:
        """
        Open the MuJoCo scene and begin mirroring world-state updates into it.
        """
        self.simulator = MujocoSimulator(
            _headless=self.headless, file_path=SCENE_FILE
        )
        self.synchronizer = MujocoSynchronizer(
            _world=self.world, simulator=self.simulator
        )
        # Disable physics→world feedback so giskardpy tracks world state freely.
        self.synchronizer.sync_rate_hz = 0
        # Zero qvel every physics step so gravity cannot accumulate velocity
        # between giskardpy ticks (50 Hz), which would otherwise cause oscillation.
        self.simulator.read_data_from_simulator = (
            lambda: numpy.copyto(
                self.simulator._mj_data.qvel,
                numpy.zeros_like(self.simulator._mj_data.qvel),
            )
        )
        self.simulator.start()

    def freeze_and_wait(self) -> None:
        """
        Freeze the current simulator pose and block until the viewer is closed.
        """
        frozen_qpos = numpy.copy(self.simulator._mj_data.qpos)
        self.simulator.read_data_from_simulator = (
            lambda: numpy.copyto(self.simulator._mj_data.qpos, frozen_qpos)
        )
        while self.simulator.renderer.is_running():
            time.sleep(0.1)

    def stop(self) -> None:
        """
        Stop the synchronizer and shut the simulator down.
        """
        self.synchronizer.stop()
        self.simulator.stop()
