import os
from copy import deepcopy

import pytest
import rclpy

from pycram.datastructures.dataclasses import Context

from pycram.testing import setup_world
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.robots.pr2 import PR2


@pytest.fixture(autouse=True, scope="session")
def cleanup_ros():
    """
    Fixture to ensure that ROS is properly cleaned up after all tests.
    """
    if os.environ.get("ROS_VERSION") == "2":
        import rclpy

        if not rclpy.ok():
            rclpy.init()
    yield
    if os.environ.get("ROS_VERSION") == "2":
        if rclpy.ok():
            rclpy.shutdown()


@pytest.fixture(scope="function")
def mutable_model_world():
    world = setup_world()
    pr2 = PR2.from_world(world)
    return world, pr2, Context(world, pr2)


world = setup_world()
pr2 = PR2.from_world(world)


@pytest.fixture(scope="function")
def immutable_model_world():
    state = deepcopy(world.state.data)
    yield world, pr2, Context(world, pr2)
    world.state.data = state


@pytest.fixture(scope="session")
def viz_marker_publisher():
    rclpy.init()
    node = rclpy.create_node("test_viz_marker_publisher")
    VizMarkerPublisher(world, node)  # Initialize the publisher
    yield
    rclpy.shutdown()
