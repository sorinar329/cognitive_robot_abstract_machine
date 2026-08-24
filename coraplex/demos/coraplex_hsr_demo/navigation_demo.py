"""
HSR moves between two tables purely by teleportation -- no real base motion planning or
control, and no manipulation.

Navigation is driven by ``coraplex.robot_plans.motions.navigation.MoveMotion``, entirely
unmodified: for ``ExecutionType.SIMULATED`` it dispatches to giskardpy's ``SetOdometry``
monitor, which writes the target pose straight onto the robot's ``OmniDrive`` connection.
Under MuJoCo that connection never has a joint of its own (see
``MultiSimBuilder._ignore_connection_types``); ``MujocoSynchronizer`` composes it into the
qpos of the ancestor ``Connection6DoF`` free joint instead (map -> odom), which is what
actually moves the robot. The two tables sit at different positions *and* orientations, so
reaching table B also exercises rotating the base, not just translating it.

Run with (the ``hsr_description`` ROS package must be built and sourced)::

    python coraplex/demos/coraplex_hsr_demo/navigation_demo.py
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

import numpy
from xacro import process_file

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.mujoco_video_recording import MujocoVideoRecorder
from semantic_digital_twin.adapters.package_resolver import CompositePathResolver
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from giskardpy.middleware.ros2.scripts.iai_robots.hsr.configs import WorldWithHSRConfig

logger = logging.getLogger(__name__)

NODE_NAME = "hsr_navigation_demo"

TABLE_SIZE = (0.8, 0.8, 0.75)
"""Width, depth, and height of both tables, in metres."""

TABLE_A_POSE = (0.0, 0.0, 0.0)
"""x, y, yaw (radians) of table A, in the world root frame."""

TABLE_B_POSE = (3.0, 2.0, numpy.pi)
"""
x, y, yaw of table B, in the world root frame -- placed diagonally from table A and
facing back the other way, so navigating between the two tables also has to rotate the
base by roughly 180 degrees, not just translate it.
"""

APPROACH_DISTANCE = 1.0
"""How far in front of a table's own facing direction the robot's approach pose sits, in
metres."""

VIDEO_OUTPUT_PATH = Path(__file__).parent / "hsr_navigation_demo.mp4"
"""Where the recorded video of the run is written."""

GROUND_HALF_THICKNESS = 0.05
"""Half thickness of the ground plane, in metres."""


def _add_ground_plane(world: World) -> Body:
    """
    Add a static ground plane to the world, fixed to the world root, its top surface at
    z=0 -- without it, the base has nothing to rest on and simply falls under gravity.

    :param world: The world to add the ground plane to, modified in place.
    :return: The newly added ground plane.
    """
    ground = Body(name=PrefixedName("ground_plane"))
    shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=-GROUND_HALF_THICKNESS, reference_frame=ground
        ),
        scale=Scale(20.0, 20.0, GROUND_HALF_THICKNESS * 2),
        color=Color(0.8, 0.8, 0.8, 1.0),
    )
    geometry = ShapeCollection([shape], reference_frame=ground)
    ground.collision, ground.visual = geometry, geometry

    with world.modify_world():
        world.add_connection(FixedConnection(parent=world.root, child=ground))
    return ground


def _add_table(world: World, name: str, x: float, y: float, yaw: float) -> Body:
    """
    Add a static table prop to the world, fixed to the world root.

    A table never moves, so it needs no ``Connection6DoF``/physics, unlike a body meant
    to be pushed or picked up.

    :param world: The world to add the table to, modified in place.
    :param name: Name of the table.
    :param x: X position of the table's centre, in the world root frame.
    :param y: Y position of the table's centre, in the world root frame.
    :param yaw: Yaw of the table, in the world root frame.
    :return: The newly added table.
    """
    table = Body(name=PrefixedName(name))
    width, depth, height = TABLE_SIZE
    shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=height / 2, reference_frame=table
        ),
        scale=Scale(width, depth, height),
        color=Color(0.6, 0.5, 0.4, 1.0),
    )
    geometry = ShapeCollection([shape], reference_frame=table)
    table.collision, table.visual = geometry, geometry

    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=table,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=x, y=y, yaw=yaw
                ),
            )
        )
    return table


def _approach_pose(world: World, x: float, y: float, yaw: float) -> Pose:
    """
    The pose the robot's base should reach to stand in front of a table, facing it.

    Offset ``APPROACH_DISTANCE`` back along the table's own facing direction, with the
    robot's own heading matching the table's yaw so it ends up looking straight at it.

    :param world: The world the pose is expressed in.
    :param x: X position of the table's centre.
    :param y: Y position of the table's centre.
    :param yaw: Yaw of the table.
    """
    approach_x = x - APPROACH_DISTANCE * numpy.cos(yaw)
    approach_y = y - APPROACH_DISTANCE * numpy.sin(yaw)
    return Pose.from_xyz_rpy(
        approach_x, approach_y, 0.0, yaw=yaw, reference_frame=world.root
    )


def main(headless: bool = False, record: bool = True) -> None:
    """
    Build the scene and navigate HSR between the two tables purely via teleportation.

    :param headless: Whether to run without opening MuJoCo's viewer window.
    :param record: Whether to record a video of the run to :data:`VIDEO_OUTPUT_PATH`.
    """
    from coraplex.datastructures.dataclasses import Context
    from coraplex.datastructures.enums import ExecutionType
    from coraplex.execution_environment import ExecutionEnvironment
    from coraplex.plans.factories import sequential
    from coraplex.robot_plans.motions.navigation import MoveMotion

    xacro_path = CompositePathResolver().resolve(HSRB.get_ros_file_path())
    urdf_content = process_file(xacro_path).toxml()

    config = WorldWithHSRConfig(urdf=urdf_content)
    with config.world.modify_world():
        config.setup_world()
    world = config.world

    _add_ground_plane(world)
    table_a_x, table_a_y, table_a_yaw = TABLE_A_POSE
    table_b_x, table_b_y, table_b_yaw = TABLE_B_POSE
    _add_table(world, "table_a", table_a_x, table_a_y, table_a_yaw)
    _add_table(world, "table_b", table_b_x, table_b_y, table_b_yaw)

    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
    spinner.start()

    # MujocoVideoRecorder mirrors `world` into its own MujocoSim (see
    # semantic_digital_twin.adapters.mujoco_video_recording). Running that mirror
    # *alongside* a separately started MujocoSim on the same world corrupts the other
    # one's background sync thread (confirmed directly: even a plain Connection6DoF
    # teleport, nothing to do with OmniDrive, silently stops reaching the first MujocoSim
    # once a second one is started on the same world, and its background thread then
    # crashes on its next read with `self._state_callback` having gone ``None``) -- so
    # recording and the interactively-viewable, physically-settling MujocoSim below are
    # mutually exclusive; recording always runs headless, using the recorder's own mirror
    # as the sole simulator instead of starting a second one.
    if record:
        recorder = MujocoVideoRecorder(world=world)
        recorder.start()
        simulator = recorder._multi_sim.simulator
    else:
        multi_sim = MujocoSim(
            world=world, headless=headless, step_size=1e-3, sync_rate_hz=100
        )
        multi_sim.start_simulation()
        time.sleep(0.5)
        recorder = None
        simulator = multi_sim.simulator

    # Fetched only after the simulation has built and started: objects captured from the
    # pre-merge world (e.g. WorldWithHSRConfig.setup_world's own `config.localization`)
    # go stale once MujocoSim's builder reparents bodies under its own synthetic root.
    # WorldWithHSRConfig.setup_world() already built the robot view as config.robot --
    # HSRB.from_world(world) a second time over the same bodies raises
    # DuplicateRobotAssignmentsError, since each robot part may belong to at most one
    # robot view.
    robot = config.robot
    localization = world.get_connections_by_type(Connection6DoF)[0]
    omni_drive = world.get_connections_by_type(OmniDrive)[0]

    context = Context(world, robot, ros_node=node, evaluate_conditions=False)
    context.simulation_clock = lambda: simulator.current_simulation_time

    def log_base_pose(label: str) -> None:
        base_position = simulator.get_body_position("base_footprint").result
        logger.info(
            "%s: base_footprint=%s, localization=%s, omni_drive=%s",
            label,
            base_position,
            localization.origin.to_position(),
            omni_drive.origin.to_position(),
        )

    table_a_approach = _approach_pose(world, table_a_x, table_a_y, table_a_yaw)
    table_b_approach = _approach_pose(world, table_b_x, table_b_y, table_b_yaw)

    plan = sequential(
        [
            MoveMotion(target=table_a_approach),
            MoveMotion(target=table_b_approach),
            MoveMotion(target=table_a_approach),
        ],
        context=context,
    )

    try:
        log_base_pose("start")
        with ExecutionEnvironment(
            execution_type=ExecutionType.SIMULATED,
            collision_avoidance=False,
            real_time_factor=1.0,
            max_ticks_per_motion_mapping=20000,
        ):
            plan.perform()
        logger.info("Navigation plan finished.")
    except Exception as error:
        logger.warning("Navigation plan raised: %r", error)
    finally:
        log_base_pose("end")

        if recorder is not None:
            recorded_video = recorder.stop()
            recorded_video.write(VIDEO_OUTPUT_PATH)
            logger.info(
                "Recorded %d frames to %s", len(recorded_video.frames), VIDEO_OUTPUT_PATH
            )
        elif not headless:
            while True:
                time.sleep(0.1)


if __name__ == "__main__":
    main()
