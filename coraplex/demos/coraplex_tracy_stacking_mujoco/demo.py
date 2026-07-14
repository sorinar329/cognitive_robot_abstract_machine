"""Stack N cubes into a tower with Tracy, executed as a live MuJoCo simulation.

Generalizes coraplex_real_tracy's hardcoded 2-box stack into a loop over N cubes,
picked from a grid and placed one level higher each time. Unlike that demo,
execution runs under ExecutionType.SIMULATED with a MujocoSim physics simulation
attached to the same World, so cubes are free-jointed (not welded to the world)
and settle under real gravity/contact instead of teleporting into place -- see
examples/stacking.py in manipulation_experiments for the MuJoCo-viewer original
this ports.

..warning::
    Not yet fully working. The Giskard/coraplex <-> MujocoSim execution wiring
    is confirmed working for the robot's own actuated joints (ParkArmsAction
    moves the arm in both ``world.state`` and MuJoCo's own ``qpos``). However,
    free-jointed (passive) bodies added to a ``MujocoSim``-wrapped world do not
    get real collision response: a cube starting in contact with Tracy's table
    free-falls straight through it, even though loading the exact same exported
    scene directly with plain ``mujoco.MjModel``/``mj_step`` (bypassing
    ``MujocoSim``/``MujocoSimulator``) rests it correctly. This reproduces with
    or without the robot/Giskard in the loop, and is unaffected by switching the
    integrator (RK4 vs Euler), so it is a bug/limitation in
    ``semantic_digital_twin.adapters.multi_sim``'s synchronization of passive
    bodies, not in this demo or in coraplex. Until that is fixed upstream, the
    cubes here will not physically stack.
"""
import colorsys
import time

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

N_CUBES = 4
HALF = 0.05
"""Cube half-size (edge = 0.1 m), matching coraplex_real_tracy's box scale."""

TABLE_SURFACE_Z = 0.88
"""Height of Tracy's built-in table top (its 'table' link's own origin sits
exactly on the surface -- see the table body's collision box: local origin
z=-0.4275, half-height 0.4275, so top face = table origin z)."""

STACK_XY = (0.8, 0.0)
"""Tower base position -- the same spot coraplex_real_tracy places box1 at."""

PICKUP_SLOTS = [
    (0.65, 0.25, Arms.LEFT),
    (0.65, -0.25, Arms.RIGHT),
    (0.95, 0.25, Arms.LEFT),
    (0.95, -0.25, Arms.RIGHT),
    (0.8, 0.4, Arms.LEFT),
    (0.8, -0.4, Arms.RIGHT),
]
"""Pickup grid, anchored to coraplex_real_tracy's proven-reachable (x=0.8,
y=+-0.25) coordinates and spread along x/y for additional cubes."""


def stack_center_z(level: int) -> float:
    """Resting center height of the cube at a given stack level (0 = base)."""
    return TABLE_SURFACE_Z + HALF * (2 * level + 1)


def make_colors(n: int) -> list[Color]:
    """A rainbow of colors so tower levels are visually distinguishable."""
    return [
        Color(*colorsys.hsv_to_rgb(i / max(1, n), 0.65, 0.9), 1.0) for i in range(n)
    ]


world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
tracy = Tracy.from_world(world)

colors = make_colors(N_CUBES)
cubes: list[Body] = []
with world.modify_world():
    for i in range(N_CUBES):
        cube = Body(
            name=PrefixedName(f"cube{i}"),
            collision=ShapeCollection(shapes=[Box(scale=Scale(2 * HALF, 2 * HALF, 2 * HALF))]),
            visual=ShapeCollection(
                shapes=[Box(scale=Scale(2 * HALF, 2 * HALF, 2 * HALF), color=colors[i])]
            ),
        )
        if i == 0:
            x, y, z = STACK_XY[0], STACK_XY[1], stack_center_z(0)
        else:
            x, y, _ = PICKUP_SLOTS[i - 1]
            z = TABLE_SURFACE_Z + HALF
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=cube,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x, y, z, reference_frame=world.root
                ),
            )
        )
        cubes.append(cube)

try:
    import rclpy
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    rclpy.init()
    node = rclpy.create_node("coraplex_tracy_stacking_mujoco")
    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
except ImportError:
    node = None

multi_sim = MujocoSim(world=world, headless=True)
multi_sim.start_simulation()
time.sleep(1.0)  # let the physics thread spin up before Giskard starts ticking

context = Context(world=world, robot=tracy, ros_node=node, evaluate_conditions=False)

actions = [ParkArmsAction(Arms.BOTH)]
for i in range(1, N_CUBES):
    _, _, arm = PICKUP_SLOTS[i - 1]
    end_effector = (
        tracy.left_arm.end_effector if arm == Arms.LEFT else tracy.right_arm.end_effector
    )
    actions.append(
        PickUpAction(
            cubes[i],
            arm,
            GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP, end_effector),
        )
    )
    actions.append(
        PlaceAction(
            cubes[i],
            Pose.from_xyz_rpy(
                STACK_XY[0], STACK_XY[1], stack_center_z(i), yaw=0, reference_frame=world.root
            ),
            arm,
        )
    )

plan = sequential(actions, context=context)

try:
    print("Perform Plan")
    with simulated_robot:
        plan.perform()
finally:
    multi_sim.stop_simulation()
