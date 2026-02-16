import py_trees
import pytest
import rclpy
from rclpy.node import Node

import robokudo.defs


@pytest.fixture(scope="session", autouse=True)
def ros_default():
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    # init once (default/global context)
    if not rclpy.ok():
        print("Calling init")
        rclpy.init()
    yield
    # shutdown once, but don't fail if something already shut it down
    try:
        if rclpy.ok():
            print("Calling shutdown")
            rclpy.shutdown()
    except RuntimeError:
        pass


@pytest.fixture
def node(ros_default):
    n = Node(robokudo.defs.TEST_ROS_NODE_NAME)
    yield n
    n.destroy_node()
