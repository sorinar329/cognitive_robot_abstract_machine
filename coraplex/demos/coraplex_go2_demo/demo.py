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
                scale=Scale(6.0, 6.0, 0.02),
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

go2 = UnitreeGo2.from_world(world)

for joint_name, position in STANDING_CONFIGURATION.items():
    world.state[world.get_connection_by_name(joint_name).raw_dof.id].position = position
world.notify_state_change()

# %% the route the robot patrols

WAYPOINTS = [
    Pose.from_xyz_rpy(2.0, 0.0, STANDING_HEIGHT, reference_frame=world.root),
]
"""
The route the robot walks: 2m straight ahead, far enough that arriving there can only
be the result of having walked it.

..warning:: A second walk appended here does not work yet. The first one runs to its
    target and stops cleanly, but partway through a second the simulation diverges and
    the robot is thrown far off the map. Steering is also weak - the gait turns much
    more slowly than it walks - so a route with corners gets walked past rather than
    around, which is why this route is a straight line.
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

final_position = np.round(go2.root.global_pose.to_position()[:2], 3)
expected_position = np.round(WAYPOINTS[-1].to_position()[:2], 3)
print(f"Go2 ended its patrol at {final_position}")
print(f"Expected it to end at {expected_position}")
# Only where it walked to is checked, not how tall it was standing when it got there:
# a trotting robot's height varies over the gait cycle, and the demo is about the
# route. Much wider than NavigateAction's kinematic 5cm because the base is carried by
# an open-loop gait rather than written to a pose.
assert np.allclose(final_position, expected_position, atol=ARRIVAL_TOLERANCE)
