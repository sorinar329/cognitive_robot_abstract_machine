import logging
import os
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.DEBUG)

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.navigation import NavigateAction

from go2_mesh_assets import Go2MeshAssets
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.unitree_go2 import UnitreeGo2, UnitreeGo2Joint
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
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

STANDING_HEIGHT = 0.27
"""
Height of the base above the floor with the legs in :data:`STANDING_LEG_JOINT_POSITIONS`,
matching the ``home`` keyframe committed in ``go2.xml``.
"""

HIP_STANCE, THIGH_STANCE, CALF_STANCE = 0.0, 0.9, -1.8
"""Per-leg joint positions of the ``home`` stance, matching ``go2.xml``'s keyframe."""

STANDING_LEG_JOINT_POSITIONS = {
    UnitreeGo2Joint.FRONT_LEFT_HIP: HIP_STANCE,
    UnitreeGo2Joint.FRONT_LEFT_THIGH: THIGH_STANCE,
    UnitreeGo2Joint.FRONT_LEFT_CALF: CALF_STANCE,
    UnitreeGo2Joint.FRONT_RIGHT_HIP: HIP_STANCE,
    UnitreeGo2Joint.FRONT_RIGHT_THIGH: THIGH_STANCE,
    UnitreeGo2Joint.FRONT_RIGHT_CALF: CALF_STANCE,
    UnitreeGo2Joint.REAR_LEFT_HIP: HIP_STANCE,
    UnitreeGo2Joint.REAR_LEFT_THIGH: THIGH_STANCE,
    UnitreeGo2Joint.REAR_LEFT_CALF: CALF_STANCE,
    UnitreeGo2Joint.REAR_RIGHT_HIP: HIP_STANCE,
    UnitreeGo2Joint.REAR_RIGHT_THIGH: THIGH_STANCE,
    UnitreeGo2Joint.REAR_RIGHT_CALF: CALF_STANCE,
}
"""
The 12 leg joint positions of the ``home`` stance, carrying the base at
:data:`STANDING_HEIGHT`. The legs are not gaited, so navigation holds them here
throughout.
"""

# %% building the world

Go2MeshAssets(scene=SCENE_PATH).download_if_missing()
world = MJCFParser(str(SCENE_PATH)).parse()
base = world.get_body_by_name("base")

with world.modify_world():
    world.remove_connection(base.parent_connection)

    # "base" is attached to the world root directly through the OmniDrive itself,
    # rather than through a separate freely-jointed "odom" body: MuJoCo treats a
    # WheeledDrive connection as a weld (it has no torque/inertia representation of
    # its own), so giving it an intermediate body with no mass of its own destabilizes
    # the simulation. Navigating the robot still works: NavigateAction moves the base
    # kinematically (see MoveMotion/SetOdometry) rather than through physical driving.
    world.add_connection(
        OmniDrive.create_with_dofs(
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

for joint_name, position in STANDING_LEG_JOINT_POSITIONS.items():
    world.state[world.get_connection_by_name(joint_name).raw_dof.id].position = position
world.notify_state_change()

# %% the route the robot patrols

WAYPOINTS = [
    Pose.from_xyz_rpy(2.0, 0.0, STANDING_HEIGHT, reference_frame=world.root),
    Pose.from_xyz_rpy(2.0, 2.0, STANDING_HEIGHT, yaw=1.57, reference_frame=world.root),
    Pose.from_xyz_rpy(0.0, 2.0, STANDING_HEIGHT, yaw=3.14, reference_frame=world.root),
]
"""
A small patrol route: 2m forward, 2m to the side, then 2m back -- ending away from
where the robot started, so reaching it proves the route was actually walked.
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
    mujoco_data.ctrl[actuator_index] = STANDING_LEG_JOINT_POSITIONS[joint_name]

context = Context(world=world, robot=go2, evaluate_conditions=False)

try:
    with simulated_robot(real_time_factor=1.0):
        sequential(
            [NavigateAction(waypoint) for waypoint in WAYPOINTS],
            context=context,
        ).perform()
    time.sleep(2.0)
finally:
    multi_sim.stop_simulation()

final_position = np.round(go2.root.global_pose.to_position(), 3)
expected_position = np.round(WAYPOINTS[-1].to_position(), 3)
print(f"Go2 ended its patrol at {final_position}")
print(f"Expected it to end at {expected_position}")
assert np.allclose(final_position, expected_position, atol=0.05)
