"""
The montessori demo drawn in RViz, with Tracy moved kinematically rather than simulated.

Watch it with a ``MarkerArray`` display on ``/semworld/viz_marker``, whose durability has
to be set to transient local for the markers to arrive.

Neither publisher is driven by a ROS timer: each one publishes from a callback the world
itself calls when its model or its state changes, so the node needs nothing spinning it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import rclpy

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)

if TYPE_CHECKING:
    from demo import MontessoriScene

NODE_NAME = "tracy_montessori_demo_rviz"
"""
Name the demo's own ROS node carries.
"""


def run(scene: MontessoriScene) -> None:
    """
    Draw the scene in RViz and carry the sorting plan out kinematically.

    :param scene: Tracy, the board and the pieces.
    """
    world = scene.world

    rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    TFPublisher(node=node, _world=world)
    VizMarkerPublisher(_world=world, node=node)
    world.notify_state_change()

    context = Context(
        world=world, robot=scene.robot, ros_node=node, evaluate_conditions=False
    )
    with simulated_robot:
        scene.build_plan(context).perform()
