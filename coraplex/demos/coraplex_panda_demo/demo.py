import os
import signal
import subprocess
import threading
import time
import numpy
import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment, ExecutionType
from coraplex.datastructures.grasp import GraspDescription

from coraplex.execution_environment import simulated_robot, ExecutionEnvironment
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from examples.coraplex_demo import world
from physics_simulators.base_simulator import SimulatorConstraints

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim, MujocoBody
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
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from segmind.detectors.base import SegmindContext

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

box = world.get_body_by_name("cube0")
box2 = world.get_body_by_name("cube1")

# Drop the cube onto the second table ("table2" in the scene, centered at
# x=0.34, y=0.16, top at z=0.1): the object's center should end up one cube
# half-size above that top.
place_location = Pose.from_xyz_rpy(x=0.34, y=0.16, z=0.12, reference_frame=world.root)

plan = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        PickUpAction(
            box,
            Arms.LEFT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.TOP,
                context.robot.get_arms()[0].end_effector,
            ),),
        PlaceAction(box, place_location, Arms.LEFT),
        ParkArmsAction(Arms.BOTH),
],

    context=context
)
box_position = world.get_body_by_name("cube0").global_pose

plan2 = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        PickUpAction(
            box2,
            Arms.LEFT,
            GraspDescription(
                ApproachDirection.LEFT,
                VerticalAlignment.TOP,
                context.robot.get_arms()[0].end_effector,
            ),
        ),
        PlaceAction(box2, box_position, Arms.LEFT),
        ParkArmsAction(Arms.BOTH),
],
context=context
)

#s = SpatialTypePublisher(node=node, _world=world)
#s.add(SpatialTypeVisualization(context.robot.get_arms()[0].end_effector.tool_frame.global_pose))
#s.add(SpatialTypeVisualization(Pose.from_xyz_quaternion(0.5, 0, 1, 0, 1, 0, 1, reference_frame=world.root)))
#s.publish()
print("Perform Plan")

arm = context.robot.get_arms()[0]
gripper = arm.end_effector
physically_simulated_dofs = {c.raw_dof for c in gripper.active_connections} | {
    c.raw_dof for c in arm.active_connections
}

# The arm's actuator gains (parsed from the scene's official mujoco_menagerie
# Franka Panda values) are calibrated assuming gravity is separately cancelled
# out via MuJoCo's own gravcomp mechanism, not held against by the PD gains
# alone. Without it, each joint settles with a steady-state error from gravity
# sag large enough (~0.02 rad) to exceed JointPositionList's default 0.01 rad
# convergence threshold -- so a motion holding the arm under gravity (e.g.
# ParkArmsAction) never registers as converged and Giskard keeps sending
# corrective commands indefinitely, which also stalls the rest of the plan.
for connection in arm.active_connections:
    connection.child.simulator_additional_properties.append(
        MujocoBody(gravitation_compensation_factor=1.0)
    )

multi_sim = MujocoSim(
    world=world,
    headless=False,
    step_size=0.0001,
    real_time_factor=1,
    physically_simulated_dofs=physically_simulated_dofs,
    sync_rate_hz=15
)
time_start = time.time()

tool_frame = gripper.tool_frame


def print_positions():
    """
    Prints the tool_frame's and cube's position as seen by the world model
    (Giskard's kinematic belief) side by side with MuJoCo's own live simulated
    position, so a divergence between "where Giskard thinks it is" and "where
    it actually, physically is" is visible directly.
    """
    tool_frame_kinematic = numpy.array(
        world.compute_forward_kinematics(world.root, tool_frame).to_position().evaluate()[:3],
        dtype=float,
    )
    box_kinematic = numpy.array(
        world.compute_forward_kinematics(world.root, box).to_position().evaluate()[:3],
        dtype=float,
    )
    tool_frame_mujoco = numpy.array(
        multi_sim.simulator.get_body_position(tool_frame.name.name).result[:3],
        dtype=float,
    )
    box_mujoco = numpy.array(
        multi_sim.simulator.get_body_position(box.name.name).result[:3], dtype=float
    )
    print(
        f"tool_frame: kinematic={tool_frame_kinematic} mujoco={tool_frame_mujoco} | "
        f"cube: kinematic={box_kinematic} mujoco={box_mujoco}"
    )


def print_positions_periodically(stop_event: threading.Event):
    while not stop_event.is_set():
        print_positions()
        time.sleep(0.5)


stop_printing = threading.Event()
printing_thread = threading.Thread(
    target=print_positions_periodically, args=(stop_printing,), daemon=True
)
printing_thread.start()

# --- Segmind event detection, running in parallel with execution ---
# A detector statechart adds no QP constraints, so the executor's qp_controller
# stays None and its tick() never calls apply_control_commands -- it is a pure
# read-only observer of the live world. It reads the same world model MuJoCo
# keeps in sync (contacts are computed geometrically from body poses, not from
# MuJoCo's solver), so ticking it in a background thread detects events live
# without interfering with the plan's own control loop.
segmind_executor = EpisodeSegmenterExecutor(
    context=MotionStatechartContext(world=world)
)
segmind_executor.compile(SegmindStatechart().build_statechart())
segmind_context = segmind_executor.context.require_extension(SegmindContext)

stop_detecting = threading.Event()


def detect_events_periodically(stop_event: threading.Event):
    while not stop_event.is_set():
        segmind_executor.tick()
        time.sleep(1 / 60)


detector_thread = threading.Thread(
    target=detect_events_periodically, args=(stop_detecting,), daemon=True
)

#constraints = SimulatorConstraints(max_number_of_steps=10000)
multi_sim.start_simulation()
detector_thread.start()
with ExecutionEnvironment(
    execution_type=execition_mode, collision_avoidance=False, real_time_pacing=True
):
    plan.perform()
    cube2 = world.get_body_by_name("cube0")
    box_position = Pose.from_xyz_rpy(x=cube2.global_pose.x, y=cube2.global_pose.y, z=cube2.global_pose.z + 0.06, reference_frame=world.root)
    plan2 = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            PickUpAction(
                box2,
                Arms.LEFT,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.TOP,
                    context.robot.get_arms()[0].end_effector,
                ),
            ),
            PlaceAction(box2, box_position, Arms.LEFT),
            ParkArmsAction(Arms.BOTH),
        ],
        context=context
    )
    plan2.perform()

# The episode is over: stop the detector thread and read the event logger.
stop_detecting.set()
detector_thread.join()
stop_printing.set()
print("--- final positions ---")
print_positions()

detected_events = segmind_context.logger.get_events()
print(f"--- Segmind detected {len(detected_events)} events ---")
for detected_event in detected_events:
    print(" ", detected_event)

event_timeline_path = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "segmind", "demo", "event_timeline.html",
    )
)
segmind_context.logger.plot_events(show=False, save_path=event_timeline_path)
print(f"Segmind event timeline written to {event_timeline_path}")

print("Plan finished, keeping the viewer open until it is closed")
while multi_sim.simulator.renderer.is_running():
    time.sleep(0.1)
