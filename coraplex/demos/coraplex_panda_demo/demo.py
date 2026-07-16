import os
import signal
import subprocess
import threading
import time
import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment, ExecutionType
from coraplex.datastructures.grasp import GraspDescription

from coraplex.execution_environment import simulated_robot, ExecutionEnvironment
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from physics_simulators.base_simulator import SimulatorConstraints

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import SpatialTypeVisualization
from semantic_digital_twin.adapters.ros.visualization.spatial_type_publisher import SpatialTypePublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Spoon,
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from test.coraplex_test.conftest import viz_marker_publisher


time.sleep(8)  # Wait for the launch file to start

execition_mode = ExecutionType.SIMULATED

print("Init ROS")
rclpy.init()
node = rclpy.create_node("stretch_demo_node")

executor = MultiThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

world = MJCFParser("/home/sorin/dev/manipulation_experiments/resources/generated/stacking_scene.xml").parse()
Panda.from_world(world)
publisher = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()


# It is important to have the ros_node in the context for a real robot
context = Context(
    world=world,
    robot=world.get_semantic_annotations_by_type(Panda)[0],
    ros_node=node,
    evaluate_conditions=False,
)

plan = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        # PickUpAction(
        #     box4,
        #     Arms.LEFT,
        #     GraspDescription(
        #         ApproachDirection.FRONT,
        #         VerticalAlignment.TOP,
        #         context.robot.get_arms()[0].end_effector,
        #     ),
],
    context=context
)


s = SpatialTypePublisher(node=node, _world=world)
s.add(SpatialTypeVisualization(context.robot.get_arms()[0].end_effector.tool_frame.global_pose))
#s.add(SpatialTypeVisualization(Pose.from_xyz_quaternion(0.5, 0, 1, 0, 1, 0, 1, reference_frame=world.root)))
s.publish()
print("Perform Plan")

multi_sim = MujocoSim(
    world=world,
    headless=False,
    step_size=0.0001,
)
time_start = time.time()

#constraints = SimulatorConstraints(max_number_of_steps=10000)
multi_sim.start_simulation()
with ExecutionEnvironment(execution_type=execition_mode, collision_avoidance=False):
    plan.perform()
