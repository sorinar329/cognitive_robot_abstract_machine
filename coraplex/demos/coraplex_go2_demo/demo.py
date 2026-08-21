import logging
import os
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.DEBUG)

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.legged_locomotion import (
    ARRIVAL_TOLERANCE,
    WalkAction,
)

from go2_mesh_assets import Go2MeshAssets
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.unitree_go2 import (
    STANDING_CONFIGURATION,
    STANDING_HEIGHT,
    UnitreeGo2,
    UnitreeGo2Joint,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

SCENE_PATH = (
    Path(__file__).parent.parent.parent
    / "resources"
    / "robots"
    / "unitree_go2"
    / "go2.xml"
)

FLOOR_SIZE = 10.0
"""
Width and depth of the floor, in metres.

Generously larger than the route needs. The floor is the only thing holding the robot
up, so walking off its edge drops the robot into an endless fall, which reads in the
final numbers as the simulation having blown up rather than as having run out of floor.
Steering is loose enough that the robot does not hold a straight line exactly, so the
route is given room on every side rather than just enough to fit.
"""

PUBLISH_TO_RVIZ = False
"""
Whether to publish the world to rviz for the length of the run.

Off by default, because it costs the walk its reliability. The simulator feeds physics
back into the world on its own thread, and publishing the tf tree from that thread
slows it down: measured over repeated runs, the patrol went from arriving every time
to arriving about half the time, even with :data:`TF_THROTTLE_STATE_UPDATES` as coarse
as it is. On a machine with other work to do the robot falls over instead of walking
at all, since the gait is open loop and cannot recover its footing. Turn it on to
watch a run, leave it off to depend on one.
"""

TF_THROTTLE_STATE_UPDATES = 50
"""
Publish the tf tree on every nth world state change rather than all of them, when
:data:`PUBLISH_TO_RVIZ` is on. Coarse enough to make the walk usually survive
publishing, which is far coarser than it is smooth to watch.
"""

TABLE_TOP_SIZE = 0.8
"""Width and depth of a table top, in metres."""

TABLE_HEIGHT = 0.5
"""Height of a table's top surface above the floor, in metres."""

TABLE_LEG_SIZE = 0.08
"""Thickness of a square table leg, in metres."""


def build_table(name: str, x: float, y: float, color: Color) -> Body:
    """
    Build a table standing on the floor, as a top slab on four legs.

    :param name: Name for the table's body.
    :param x: Where the centre of the table stands, along the world's x axis.
    :param y: Where the centre of the table stands, along the world's y axis.
    :param color: Colour to draw the whole table in.
    :return: The table, already connected to the world.
    """
    table = Body(name=PrefixedName(name))
    leg_offset = (TABLE_TOP_SIZE - TABLE_LEG_SIZE) / 2
    leg_height = TABLE_HEIGHT - TABLE_LEG_SIZE / 2
    shapes = [
        Box(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=TABLE_HEIGHT, reference_frame=table
            ),
            scale=Scale(TABLE_TOP_SIZE, TABLE_TOP_SIZE, TABLE_LEG_SIZE),
            color=color,
        )
    ] + [
        Box(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=leg_x, y=leg_y, z=leg_height / 2, reference_frame=table
            ),
            scale=Scale(TABLE_LEG_SIZE, TABLE_LEG_SIZE, leg_height),
            color=color,
        )
        for leg_x in (-leg_offset, leg_offset)
        for leg_y in (-leg_offset, leg_offset)
    ]
    geometry = ShapeCollection(shapes, reference_frame=table)
    table.collision, table.visual = geometry, geometry
    world.add_connection(
        FixedConnection(
            parent=world.root,
            child=table,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=x, y=y, reference_frame=world.root
            ),
        )
    )
    return table


# %% building the world

Go2MeshAssets(scene=SCENE_PATH).download_if_missing()
world = MJCFParser(str(SCENE_PATH)).parse()
base = world.get_body_by_name("base")

with world.modify_world():
    world.remove_connection(base.parent_connection)

    # go2.xml commits "base" without a freejoint (see the comment there), so it parses
    # as a fixed body. Replace that with a plain 6-DoF connection: unlike OmniDrive,
    # MuJoCo integrates this physically (multi_sim.py's ignore-list is only
    # OmniDrive/DifferentialDrive/FixedConnection), so the base's pose becomes a
    # consequence of gravity and leg-ground contact under WalkAction's gait, rather
    # than a commanded input.
    world.add_connection(
        Connection6DoF.create_with_dofs(
            world=world,
            parent=world.root,
            child=base,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=STANDING_HEIGHT, reference_frame=world.root
            ),
        )
    )

    ground_plane = Body(name=PrefixedName("ground_plane"))
    ground_plane_geometry = ShapeCollection(
        [
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=ground_plane
                ),
                scale=Scale(FLOOR_SIZE, FLOOR_SIZE, 0.02),
                color=Color(0.6, 0.6, 0.6, 1.0),
            )
        ],
        reference_frame=ground_plane,
    )
    ground_plane.collision, ground_plane.visual = (
        ground_plane_geometry,
        ground_plane_geometry,
    )
    world.add_connection(
        FixedConnection(
            parent=world.root,
            child=ground_plane,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=-0.01, reference_frame=world.root
            ),
        )
    )

    # Two landmarks the patrol runs between, so the robot is visibly somewhere else at
    # the end than it was at the start. Both stand off to the side of the route rather
    # than on it: nothing in this demo avoids obstacles.
    start_table = build_table(
        "start_table", x=-1.0, y=0.0, color=Color(0.55, 0.35, 0.2, 1.0)
    )
    goal_table = build_table(
        "goal_table", x=-1.0, y=2.0, color=Color(0.2, 0.4, 0.7, 1.0)
    )

go2 = UnitreeGo2.from_world(world)

for joint_name, position in STANDING_CONFIGURATION.items():
    world.state[world.get_connection_by_name(joint_name).raw_dof.id].position = position
world.notify_state_change()

# %% the route the robot patrols

WAYPOINTS = [
    Pose.from_xyz_rpy(2.0, 0.0, STANDING_HEIGHT, reference_frame=world.root),
    Pose.from_xyz_rpy(2.0, 2.0, STANDING_HEIGHT, reference_frame=world.root),
    Pose.from_xyz_rpy(0.0, 2.0, STANDING_HEIGHT, reference_frame=world.root),
]
"""
The patrol route: 2m forward, 2m to the side, then 2m back, from beside
:data:`start_table` to beside :data:`goal_table`. Both corners turn the robot a
quarter turn, so walking the whole route means it steered as well as walked.

..note:: The robot swings wide through the corners, because the gait turns much more
    slowly than it walks, and arrives within
    :data:`~coraplex.robot_plans.actions.core.legged_locomotion.ARRIVAL_TOLERANCE`
    rather than on the spot.
"""

headless = os.environ.get("CI", "false").lower() == "true"
multi_sim = MujocoSim(world=world, headless=headless, step_size=1e-3)
multi_sim.start_simulation()

# The legs' actuators are position servos (see go2.xml) so they hold the stance set
# above, but MuJoCo starts every actuator's control target at 0 regardless of the
# joint's initial position, and coraplex's world<->simulator synchronizer only pushes a
# control target on top of a *later* state change. Seeding it once here, right after the
# simulator starts, avoids the legs snapping toward 0 before anything ever touches
# their state again.
mujoco_model = multi_sim.simulator._mj_model
mujoco_data = multi_sim.simulator._mj_data
for actuator_index in range(mujoco_model.nu):
    # go2.xml names each actuator after its joint, minus the "_joint" suffix.
    joint_name = UnitreeGo2Joint(f"{mujoco_model.actuator(actuator_index).name}_joint")
    mujoco_data.ctrl[actuator_index] = STANDING_CONFIGURATION[joint_name]

# %% publishing the world to rviz

visualization_node = None
owns_ros_context = False

if PUBLISH_TO_RVIZ:
    # Imported here rather than at the top so the demo still runs without ROS
    # installed, which is the only thing it needs ROS for.
    import rclpy
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    # The markers describe the bodies once; the tf publisher moves them, on every world
    # state change the simulator raises as it feeds physics back into the world. Seeing
    # them needs a MarkerArray display on the publisher's topic, its durability set to
    # transient local, and the fixed frame set to the tf root -- see VizMarkerPublisher.
    owns_ros_context = not rclpy.ok()
    if owns_ros_context:
        rclpy.init()
    visualization_node = rclpy.create_node("go2_demo_visualization")
    visualization = VizMarkerPublisher(node=visualization_node, _world=world)
    visualization.with_tf_publisher(throttle_state_updates=TF_THROTTLE_STATE_UPDATES)

context = Context(world=world, robot=go2, evaluate_conditions=False)

try:
    with simulated_robot(real_time_factor=1.0):
        sequential(
            [WalkAction(waypoint) for waypoint in WAYPOINTS],
            context=context,
        ).perform()
    time.sleep(2.0)
finally:
    multi_sim.stop_simulation()
    if visualization_node is not None:
        visualization_node.destroy_node()
    if owns_ros_context:
        rclpy.shutdown()

final_position = np.round(go2.root.global_pose.to_position()[:2], 3)
expected_position = np.round(WAYPOINTS[-1].to_position()[:2], 3)
print(f"Go2 ended its patrol at {final_position}")
print(f"Expected it to end at {expected_position}")
# Only where it walked to is checked, not how tall it was standing when it got there:
# a trotting robot's height varies over the gait cycle, and the demo is about the
# route. Much wider than NavigateAction's kinematic 5cm because the base is carried by
# an open-loop gait rather than written to a pose.
assert np.allclose(final_position, expected_position, atol=ARRIVAL_TOLERANCE)
