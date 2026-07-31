"""
Load the HSR into an otherwise empty world, publish it for RViz, and show it in a MuJoCo
viewer. Nothing else: no scene, no motions.

The robot is parsed from its URDF, not from an MJCF: the semantic model
(:class:`~semantic_digital_twin.robots.hsrb.HSRB`) looks its parts up by the URDF's own
link names, so an MJCF export of the same robot would not reliably resolve them. MJCF
only enters in the opposite direction, when
:class:`~semantic_digital_twin.adapters.multi_sim.MujocoBuilder` writes a scene out for
MuJoCo.

Run with (as a script, like the other demos under ``coraplex/demos``)::

    python coraplex/demos/hsr_montessori/demo.py

Then add a ``MarkerArray`` display in RViz2 for the topic printed at startup, with
``DurabilityPolicy.TRANSIENT_LOCAL``. The MuJoCo viewer opens on its own.
"""

from __future__ import annotations

import logging
import threading
import time

import mujoco
import rclpy
from rclpy.executors import SingleThreadedExecutor

from semantic_digital_twin.adapters.multi_sim import (
    MujocoActuator,
    MujocoBody,
    MujocoSim,
)
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Actuator

logger = logging.getLogger(__name__)

NODE_NAME = "hsr_montessori_demo"
"""
Name of the ROS 2 node the visualization publishes from.
"""

JOINT_POSITION_GAIN = 100.0
"""
Proportional gain of the position-hold actuator given to each of the robot's controlled
joints.
"""

JOINT_VELOCITY_GAIN = 10.0
"""
Derivative (damping) gain of those actuators.
"""

JOINT_FORCE_RANGE = [-100.0, 100.0]
"""
Force clamp of those actuators, taken from the effort limit the HSR's URDF declares for
these joints.
"""

MUJOCO_STEP_SIZE = 2e-4
"""
MuJoCo simulation step size, small enough for the robot's lighter joints to stay
numerically stable.
"""


def hold_joints_upright(robot: AbstractRobot) -> None:
    """
    Give the robot's controlled joints a position-hold actuator and their links gravity
    compensation, so it stands in its spawn pose instead of collapsing.

    An unactuated joint has nothing opposing gravity, so without this the arm simply
    sags to the bottom of its range the moment the simulation starts and there is no
    model left to look at.

    :param robot: The spawned robot, modified in place.
    """
    held_dofs = set(robot.degrees_of_freedom_with_hardware_interface)
    with robot._world.modify_world():
        for dof in held_dofs:
            actuator = Actuator()
            actuator.add_dof(dof=dof)
            actuator.simulator_additional_properties.append(
                MujocoActuator(
                    dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                    gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                    gain_parameters=[JOINT_POSITION_GAIN] + [0.0] * 9,
                    bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                    bias_parameters=[0, -JOINT_POSITION_GAIN, -JOINT_VELOCITY_GAIN]
                    + [0.0] * 7,
                    force_limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                    force_range=list(JOINT_FORCE_RANGE),
                )
            )
            robot._world.add_actuator(actuator=actuator)
        for connection in robot.connections:
            if (
                isinstance(connection, ActiveConnection1DOF)
                and connection.raw_dof in held_dofs
            ):
                connection.child.simulator_additional_properties.append(
                    MujocoBody(gravitation_compensation_factor=1.0)
                )


def build_world() -> World:
    """
    Parse the HSR's URDF into a world and attach its semantic model.

    :return: A world containing nothing but the robot.
    """
    world = URDFParser.from_file(HSRB.get_ros_file_path()).parse()
    HSRB.from_world(world)
    return world


def main() -> None:
    """
    Build the world, publish it for RViz, and keep publishing until interrupted.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    world = build_world()
    logger.info("Loaded the HSR: %d bodies.", len(world.bodies))

    [robot] = world.get_semantic_annotations_by_type(HSRB)
    hold_joints_upright(robot)

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
    spinner.start()

    tf_publisher = TFPublisher(node=node, _world=world)
    viz_marker_publisher = VizMarkerPublisher(_world=world, node=node)
    logger.info("Visualizing on topic '%s'.", viz_marker_publisher.topic_name)

    multi_sim = MujocoSim(world=world, headless=False, step_size=MUJOCO_STEP_SIZE)
    multi_sim.start_simulation()
    logger.info("MuJoCo viewer open. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        multi_sim.stop_simulation()
        viz_marker_publisher.stop()
        tf_publisher.stop()
        executor.shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
