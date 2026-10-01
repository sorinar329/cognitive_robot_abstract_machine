"""
Tracy stacking cubes from its table into a tower.

The scene and the plan are the same wherever the demo runs; only how the plan is carried
out differs. Pick :data:`BACKEND`, and for a simulated run :data:`ARM_SPEED_LIMIT`, then
run this file from its own folder::

    python demo.py
"""

from __future__ import annotations

import colorsys
import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, List, Optional, Tuple

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.inertial_properties import (
    InertiaTensor,
    Inertial,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% which backend carries the plan out


class Backend(StrEnum):
    """
    Where the demo runs.
    """

    MUJOCO = "mujoco"
    """
    Physics, in a MuJoCo viewer.
    """

    RVIZ = "rviz"
    """
    The world model alone, drawn in RViz, with the robot moved kinematically.
    """

    REAL = "real"
    """
    The robot itself, through giskard.
    """


BACKEND = Backend.MUJOCO
"""
The backend this run uses. Edit it to run the demo somewhere else.
"""

NUMBER_OF_CUBES = 4
"""
How many cubes Tracy stacks into one tower.
"""


# %% how fast the arms move

ARM_SPEED_LIMIT: Optional[float] = None
"""
The fastest any arm joint of a simulated Tracy may turn, in rad/s, with the other joints
slowed in proportion. ``None`` lets every joint turn as fast as the robot's description
allows, about 2 to 3 rad/s.

Loading Tracy slows its arms to 0.2 rad/s at their fastest joint, which is meant for the
real robot; the simulated backends replace that with this limit. The real backend keeps
the robot's own.
"""


# %% where everything stands

TABLE_TOP_Z = 0.88
"""
Height of the work surface of the table Tracy is bolted to. It is the top of the table's
own collision slab, read off the robot's description rather than guessed.
"""

CUBE_ROW_X = 0.55
"""
How far in front of Tracy the first row of loose cubes stands.
"""

CUBES_PER_ROW = 5
"""
How many loose cubes stand in one row; further cubes start a new row nearer to Tracy,
so that no row runs off the side of the table.
"""

CUBE_ROW_START_Y = 0.1
"""
Where the first loose cube stands sideways, on the side of the stacking arm.
"""

CUBE_ROW_SPACING = 0.1
"""
How far apart the loose cubes stand, within a row and from one row to the next, so
reaching for one does not sweep the arm into its neighbour.
"""

TOWER_X = 0.75
"""
How far in front of Tracy the tower is built, beyond the rows of loose cubes.
"""

TOWER_Y = 0.0
"""
Where the tower is built sideways.
"""

RELEASE_HEIGHT = 0.003
"""
How far above what it is stacked on a cube is let go of, so it settles onto it rather
than being pushed into it.
"""

PICK_ARM = Arms.LEFT
"""
The arm that does the stacking; the arm every Tracy demo in this repository uses.
"""

RESTING_CLEARANCE = 0.001
"""
Gap left between a cube and the table, so it rests rather than intersects.
"""


# %% the cubes


CUBE_EDGE_LENGTH = 0.04
"""
Edge length of every cube.
"""

CUBE_MASS = 0.04
"""
Mass of every cube, in kilograms: a 4 cm cube of wood at about 0.6 g/cm³, as a toy
building block is.
"""

FINGER_PAD_CLOSING_TRAVEL = 0.0135
"""
How far Tracy's finger pads move out along the gripper's reach as it closes.

Measured on the robot's own description: open, the pads end at the tool frame; closed,
13.5 mm past it. The Robotiq linkage swings the pads down onto whatever the hand is
lowered over.
"""

CLOSED_PAD_CLEARANCE = 0.005
"""
How far the closed finger pads stay above the bottom of a grasped cube, so that they
clear whatever the cube is stacked on.
"""

GRASP_HEIGHT = FINGER_PAD_CLOSING_TRAVEL + CLOSED_PAD_CLEARANCE
"""
How far above its bottom face a cube is grasped.

A grasp brings the tool frame to the grasp point, and the pads end there while the hand
is open, so the closed pads stay :data:`CLOSED_PAD_CLEARANCE` above the cube's bottom and
cover most of its sides. Grasped higher, a cube is held by its top few millimetres only
and swings between the pads.
"""


def cube_color(index: int, number_of_cubes: int) -> Color:
    """
    The colour of one cube, so that the tower runs through the hues from red at its
    bottom.

    :param index: Which cube, counted from the bottom of the tower.
    :param number_of_cubes: How many cubes the tower has.
    :return: The colour.
    """
    hue = 0.8 * index / max(number_of_cubes - 1, 1)
    return Color(*colorsys.hsv_to_rgb(hue, 0.85, 0.9))


@dataclass
class StackedCube:
    """
    One cube, and where it is let go of on the tower.
    """

    body: Body
    """
    The cube itself.
    """

    annotation: HasRootBody
    """
    What names the cube for an action that takes an annotation rather than a body.
    """

    release_pose: Pose
    """
    Where the cube is let go of, straight above the cube below it, or above the table for
    the bottom one.
    """


BuildsWorld = Callable[[], Tuple[World, Tracy]]
"""
Builds a world holding Tracy, for a backend that does not get one handed to it.
"""

BuildsScene = Callable[[World], List[StackedCube]]
"""
Stands the cubes in a world.
"""

BuildsPlan = Callable[[Context, List[StackedCube]], Plan]
"""
Builds the stacking plan for the cubes a scene put in a world.
"""


# %% building the world the demo runs in


def build_offline_world() -> Tuple[World, Tracy]:
    """
    Build a world holding nothing but Tracy, read from its own description, with both
    arms parked, both grippers open, and the arms as fast as :data:`ARM_SPEED_LIMIT`
    lets them be.

    Loading Tracy this way also equips its arms and grippers with the position servos and
    the gravity compensation a physical simulation needs, so the same world serves both
    the MuJoCo and the RViz backend.

    :return: The world, and the Tracy standing in it.
    """
    world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
    described_velocity_limits = {
        dof: dof.limits.upper.velocity
        for dof in world.degrees_of_freedom
        if dof.limits.upper.velocity is not None
    }
    robot = Tracy.from_world(world)
    for dof, velocity_limit in described_velocity_limits.items():
        dof.limits.lower.velocity = -velocity_limit
        dof.limits.upper.velocity = velocity_limit
    if ARM_SPEED_LIMIT is not None:
        _slow_the_arms_down(robot, ARM_SPEED_LIMIT)

    for arm in (robot.left_arm, robot.right_arm):
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        arm.end_effector.get_joint_state_by_type(GripperState.OPEN).apply_to(world)
    return world, robot


def _slow_the_arms_down(robot: Tracy, speed_limit: float) -> None:
    """
    Slow Tracy's arms down so that their fastest joint turns at ``speed_limit``, keeping
    the joints' proportions. The grippers keep their speed, as they do on the real robot.

    :param robot: The Tracy to slow down.
    :param speed_limit: The fastest an arm joint may turn, in rad/s.
    """
    gripper_connections = {
        connection
        for arm in robot.get_arms()
        for connection in arm.end_effector.active_connections
    }
    arm_dofs = [
        connection.raw_dof
        for arm in robot.get_arms()
        for connection in arm.active_connections
        if connection not in gripper_connections
        and connection.raw_dof.limits.upper.velocity is not None
    ]
    scale = min(1.0, speed_limit / max(dof.limits.upper.velocity for dof in arm_dofs))
    for dof in arm_dofs:
        dof.limits.lower.velocity *= scale
        dof.limits.upper.velocity *= scale


def build_scene(world: World) -> List[StackedCube]:
    """
    Stand the cubes in rows on Tracy's table.

    Every cube hangs off a connection with six degrees of freedom, which is what lets the
    robot take it somewhere else.

    :param world: The world to build the scene in.
    :return: The cubes, in the order they are stacked.
    """
    cubes = []
    with world.modify_world():
        for level in range(NUMBER_OF_CUBES):
            row, place_in_row = divmod(level, CUBES_PER_ROW)
            release_pose = Pose.from_xyz_rpy(
                TOWER_X,
                TOWER_Y,
                TABLE_TOP_Z + level * CUBE_EDGE_LENGTH + RELEASE_HEIGHT + GRASP_HEIGHT,
                reference_frame=world.root,
            )
            cubes.append(
                _stand_cube_on_the_table(
                    world,
                    f"cube_{level + 1}",
                    cube_color(level, NUMBER_OF_CUBES),
                    CUBE_ROW_X - row * CUBE_ROW_SPACING,
                    CUBE_ROW_START_Y + place_in_row * CUBE_ROW_SPACING,
                    release_pose,
                )
            )
    return cubes


def _stand_cube_on_the_table(
    world: World,
    name: str,
    color: Color,
    row_x: float,
    row_y: float,
    release_pose: Pose,
) -> StackedCube:
    """
    Stand one loose cube in its row in front of Tracy.

    :param world: The world being built, already open for modification.
    :param name: Name the cube's body carries.
    :param color: Colour the cube is drawn in.
    :param row_x: How far in front of Tracy its row stands.
    :param row_y: Where along the row it stands.
    :param release_pose: Where it is let go of on the tower.
    :return: The cube.
    """
    body = Body(name=PrefixedName(name))
    body.collision = ShapeCollection(
        [_cube_around_the_grasp_point(color, body)], reference_frame=body
    )
    body.visual = ShapeCollection(
        [_cube_around_the_grasp_point(color, body)], reference_frame=body
    )
    body.inertial = _solid_cube_inertial(body)
    world.add_kinematic_structure_entity(body)
    world.add_connection(
        Connection6DoF.create_with_dofs(
            world=world,
            parent=world.root,
            child=body,
            name=PrefixedName(f"{name}_connection"),
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                row_x,
                row_y,
                TABLE_TOP_Z + RESTING_CLEARANCE + GRASP_HEIGHT,
            ),
        )
    )

    ContactParameters.create_for_grasped_object().apply_to([body])

    annotation = HasRootBody(root=body)
    world.add_semantic_annotations([annotation])
    return StackedCube(body=body, annotation=annotation, release_pose=release_pose)


def _cube_around_the_grasp_point(color: Color, body: Body) -> Box:
    """
    A cube placed so that ``body``'s origin, which a grasp aims at, lies
    :data:`GRASP_HEIGHT` above the cube's bottom face.

    :param color: Colour the cube is drawn in.
    :param body: The body the cube belongs to.
    :return: The cube.
    """
    cube = Box(
        scale=Scale(CUBE_EDGE_LENGTH, CUBE_EDGE_LENGTH, CUBE_EDGE_LENGTH),
        color=copy.copy(color),
    )
    cube.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=CUBE_EDGE_LENGTH / 2 - GRASP_HEIGHT, reference_frame=body
    )
    return cube


def _solid_cube_inertial(body: Body) -> Inertial:
    """
    The mass and inertia of a solid cube of :data:`CUBE_MASS`, whose centre lies above
    ``body``'s origin as :func:`_cube_around_the_grasp_point` places it.

    :param body: The body the cube belongs to.
    :return: The cube's inertial properties.
    """
    moment_of_inertia = CUBE_MASS * CUBE_EDGE_LENGTH**2 / 6
    return Inertial(
        mass=CUBE_MASS,
        center_of_mass=Point3(
            0.0, 0.0, CUBE_EDGE_LENGTH / 2 - GRASP_HEIGHT, reference_frame=body
        ),
        inertia=InertiaTensor.from_values(
            moment_of_inertia, moment_of_inertia, moment_of_inertia, 0.0, 0.0, 0.0
        ),
    )


# %% the plan, the same wherever it runs


def build_plan(context: Context, cubes: List[StackedCube]) -> Plan:
    """
    Pick every cube up in turn and let it go on top of the one stacked before it.

    The plan still begins by parking the arms: a simulated Tracy already starts parked,
    but the real one starts wherever it was left.

    :param context: The context the actions are built in.
    :param cubes: The cubes to stack, bottom first.
    :return: The plan that stacks them.
    """
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        context.robot.left_arm.end_effector,
        rotate_gripper=True,
    )

    actions = [ParkArmsAction(Arms.BOTH)]
    for cube in cubes:
        actions.append(PickUpAction(cube.annotation, PICK_ARM, grasp))
        actions.append(PlaceAction(cube.body, cube.release_pose, PICK_ARM))
    actions.append(ParkArmsAction(Arms.BOTH))

    return sequential(actions, context=context).plan


# %% running it


def main() -> None:
    """
    Run the demo on the backend :data:`BACKEND` names.

    Each backend is imported only once it is chosen, so a run needs nothing installed for
    the backends it does not use.
    """
    if BACKEND is Backend.MUJOCO:
        import mujoco_demo

        mujoco_demo.run(build_offline_world, build_scene, build_plan)
        return

    if BACKEND is Backend.RVIZ:
        import rviz_demo

        rviz_demo.run(build_offline_world, build_scene, build_plan)
        return

    import real_demo

    real_demo.run(build_scene, build_plan)


if __name__ == "__main__":
    main()
