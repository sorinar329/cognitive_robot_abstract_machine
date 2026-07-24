import threading
import time
import numpy
import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment, ExecutionType
from coraplex.datastructures.grasp import GraspDescription

from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim, MujocoBody
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.spatial_types.spatial_types import Pose


time.sleep(8)  # Wait for the launch file to start

execition_mode = ExecutionType.SIMULATED

print("Init ROS")
rclpy.init()
node = rclpy.create_node("stretch_demo_node")

executor = MultiThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

world = MJCFParser(
    "/home/nvasant/workspace/ros/src/manipulation_experiments/resources/generated/stacking_scene.xml"
).parse()
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
box1 = world.get_body_by_name("cube1")
box2 = world.get_body_by_name("cube2")
box3 = world.get_body_by_name("cube3")

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
    sync_rate_hz=100,
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

NUMBER_OF_ITERATIONS = 10
"""
Number of times the full pickup/stack sequence is repeated, so the demo can
be left running unattended instead of re-started by hand for every trial.
"""

ITERATION_TIME_LIMIT = 60.0
"""
Wall-clock budget (in seconds) for one iteration's cube-stacking attempts,
checked between attempts.

A stuck grasp (fingers touching the cube without closing around it) can
otherwise stall an iteration indefinitely. Giskard's own tick budget (see
``max_ticks_per_motion_mapping`` below) already bounds any single stuck
attempt, but this is an additional, coarser safety net: once the elapsed
time for an iteration exceeds this limit, remaining cube attempts are
skipped and the loop moves on to the next iteration. Because an
already-in-flight attempt is allowed to finish rather than being killed
mid-motion (forcibly interrupting a running Giskard execution is unsafe),
the actual wall-clock time for an iteration that trips this limit can run
somewhat past it.
"""

STACK_HEIGHT_OFFSET = 0.06
"""
Vertical offset (in meters) above a target cube's center at which a placed
cube should end up -- one cube height plus a small clearance margin.
"""

CUBE_SPAWN_POSITIONS = {
    "cube0": numpy.array([0.40, 0.10, 0.06]),
    "cube1": numpy.array([0.40, -0.04, 0.06]),
    "cube2": numpy.array([0.40, -0.14, 0.06]),
    "cube3": numpy.array([0.40, -0.24, 0.06]),
}
"""
Spawn position of every cube, matching the scene's MJCF definition.
"""

CUBE_SPAWN_ORIENTATION = numpy.array([1.0, 0.0, 0.0, 0.0])
"""
Spawn orientation (identity quaternion) of every cube.
"""


def reset_cubes() -> None:
    """
    Teleports every cube back to its spawn pose in MuJoCo, undoing the
    displacement from the previous iteration's stacking attempts.

    Only position and orientation are reset -- MuJoCo exposes no safe,
    synchronized API to reset a body's velocity, so residual velocity from
    the previous iteration can carry over as a minor, known limitation.
    """
    for name, position in CUBE_SPAWN_POSITIONS.items():
        multi_sim.simulator.set_body_position(body_name=name, position=position)
        multi_sim.simulator.set_body_quaternion(
            body_name=name, quaternion=CUBE_SPAWN_ORIENTATION
        )


def stack_on(object_body, target_body, picking_arm) -> None:
    """
    Picks up ``object_body`` and places it centered above ``target_body``,
    one cube height higher.
    """
    target_pose = target_body.global_pose
    place_location = Pose.from_xyz_rpy(
        x=target_pose.x,
        y=target_pose.y,
        z=target_pose.z + STACK_HEIGHT_OFFSET,
        reference_frame=world.root,
    )
    sequential(
        [
            ParkArmsAction(Arms.BOTH),
            PickUpAction(
                object_body,
                picking_arm,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.TOP,
                    context.robot.get_arms()[0].end_effector,
                ),
            ),
            PlaceAction(object_body, place_location, picking_arm),
            ParkArmsAction(Arms.BOTH),
        ],
        context=context,
    ).perform()


def attempt_stack(object_body, target_body, picking_arm, step_name: str) -> None:
    """
    Runs :func:`stack_on` for one cube, logging and swallowing any failure
    instead of letting it propagate.

    A single failed grasp/place should not crash the whole run -- it skips
    to the next cube (or iteration) instead, re-parking the arms first so
    the robot starts the next attempt from a known configuration.
    """
    try:
        stack_on(object_body, target_body, picking_arm)
    except Exception as exc:
        print(f"[warning] {step_name} failed ({type(exc).__name__}: {exc}), moving on")
        try:
            sequential([ParkArmsAction(Arms.BOTH)], context=context).perform()
        except Exception as park_exc:
            print(f"[warning] re-park after {step_name} also failed: {park_exc}")


def print_iteration_summary(iteration_index: int) -> None:
    """
    Prints the final z-height of every cube, a quick visual check of how
    high the stack reached in this iteration.
    """
    heights = {
        name: multi_sim.simulator.get_body_position(name).result[2]
        for name in CUBE_SPAWN_POSITIONS
    }
    print(f"--- iteration {iteration_index} final heights: {heights} ---")


#constraints = SimulatorConstraints(max_number_of_steps=10000)
multi_sim.start_simulation()
iteration_durations = []
with ExecutionEnvironment(
    execution_type=execition_mode,
    collision_avoidance=False,
    real_time_pacing=True,
    # A stuck grasp can otherwise retry for several minutes (2000 ticks per
    # merged motion at 50 Hz = 40 s per motion) before finally giving up --
    # far too slow across many iterations. 250 still gives each individual
    # motion a 5 s budget, comfortably above how long a successful one
    # actually takes at the tuned approach/lift/transport speeds, while
    # capping a single stuck attempt to well under ITERATION_TIME_LIMIT.
    max_ticks_per_motion_mapping=250,
):
    for iteration in range(1, NUMBER_OF_ITERATIONS + 1):
        iteration_start = time.time()
        print(f"=== starting iteration {iteration}/{NUMBER_OF_ITERATIONS} ===")
        reset_cubes()
        time.sleep(1.5)

        for cube_to_pick, cube_to_stack_on, step_label in [
            (box1, box, "cube1 onto cube0"),
            (box2, box1, "cube2 onto cube1"),
            (box3, box2, "cube3 onto cube2"),
        ]:
            elapsed = time.time() - iteration_start
            if elapsed > ITERATION_TIME_LIMIT:
                print(
                    f"[warning] iteration {iteration} already took {elapsed:.1f}s "
                    f"(limit {ITERATION_TIME_LIMIT:.0f}s), skipping remaining "
                    "attempts and moving to the next iteration"
                )
                break
            attempt_stack(cube_to_pick, cube_to_stack_on, Arms.LEFT, step_label)
            time.sleep(1)

        print_iteration_summary(iteration)

        iteration_durations.append(time.time() - iteration_start)
        average_duration = sum(iteration_durations) / len(iteration_durations)
        print(
            f"=== iteration {iteration}/{NUMBER_OF_ITERATIONS} took "
            f"{iteration_durations[-1]:.1f}s (average so far: {average_duration:.1f}s) ==="
        )

stop_printing.set()
print("--- final positions ---")
print_positions()

print("Plan finished, keeping the viewer open until it is closed")
while multi_sim.simulator.renderer.is_running():
    time.sleep(0.1)
