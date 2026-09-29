"""
The montessori demo carried out by the real Tracy, through giskard.

The world is not built here but fetched from the running stack, so the robot the plan
addresses is the one the controllers already hold, and a synchronizer keeps it in step
with what the robot actually does. The board and the pieces are then stood in that world.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import TYPE_CHECKING

import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.robots.tracy import Tracy

if TYPE_CHECKING:
    from demo import BuildsPlan, BuildsScene

GISKARD_LAUNCH_COMMAND = (
    "ros2",
    "launch",
    "giskardpy_ros",
    "giskardpy_tracy_standalone.launch.py",
)
"""
What brings the robot's controllers up.
"""

GISKARD_STARTUP_SECONDS = 8
"""
How long the launch file is given before anything talks to it.
"""

WORLD_FETCH_TIMEOUT_SECONDS = 300
"""
How long the world is waited for.

This matches giskardpy's own client, which waits as long for the same race: the server is
still parsing the robot's description and starting up while a shorter budget would expire.
"""

NODE_NAME = "tracy_montessori_demo_real"
"""
Name the demo's own ROS node carries.
"""

EXECUTOR_THREAD_NAME = "rclpy-executor"
"""
Name of the thread the node is spun on.
"""

COLLISION_AVOIDANCE = False
"""
Whether the robot avoids collisions while it sorts.
"""


def run(build_scene: BuildsScene, build_plan: BuildsPlan) -> None:
    """
    Carry the sorting plan out on the real robot.

    :param build_scene: Stands the board and the pieces in the fetched world.
    :param build_plan: Builds the plan that sorts them.
    """
    giskard_process = subprocess.Popen(
        list(GISKARD_LAUNCH_COMMAND), start_new_session=True
    )
    time.sleep(GISKARD_STARTUP_SECONDS)
    try:
        rclpy.init()
        node = rclpy.create_node(NODE_NAME)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        threading.Thread(
            target=executor.spin, daemon=True, name=EXECUTOR_THREAD_NAME
        ).start()

        world = fetch_world_from_service(
            node=node, timeout_seconds=WORLD_FETCH_TIMEOUT_SECONDS
        )
        WorldSynchronizer(_world=world, node=node)
        [robot] = world.get_semantic_annotations_by_type(Tracy)

        pieces = build_scene(world)
        context = Context(
            world=world, robot=robot, ros_node=node, evaluate_conditions=False
        )
        plan = build_plan(context, pieces)

        with ExecutionEnvironment(
            execution_type=ExecutionType.REAL,
            collision_avoidance=COLLISION_AVOIDANCE,
        ):
            plan.perform()
    finally:
        os.killpg(os.getpgid(giskard_process.pid), signal.SIGTERM)
        giskard_process.wait()
