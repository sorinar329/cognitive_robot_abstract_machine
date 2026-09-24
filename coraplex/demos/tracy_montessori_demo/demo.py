"""
Tracy sorting the Montessori pieces from its table into the holes of the shape-sorting
board.

The scene and the plan are the same wherever the demo runs; only how the plan is carried
out differs. Pick :data:`BACKEND` and run this file from its own folder::

    python demo.py
"""

from __future__ import annotations

import colorsys
import copy
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, List, Tuple

import numpy as np
import trimesh

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from montessori_board import BoardGeometry, HoleShape, board_body
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Scale,
    Shape,
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


# %% where everything stands

TABLE_TOP_Z = 0.88
"""
Height of the work surface of the table Tracy is bolted to, which every piece and the
board rest on. It is the top of the table's own collision slab, read off the robot's
description rather than guessed.
"""

BOARD_X = 0.85
"""
How far in front of Tracy the shape-sorting board stands, straight ahead of it, with
its drawers facing the robot.
"""

BOARD_Y = 0.0
"""
Where the board stands sideways.
"""

PIECE_ROW_X = 0.55
"""
How far in front of Tracy the loose pieces stand: nearer than the board, and far enough
from it that reaching for a piece does not sweep the arm through the drawer handles.
"""

PIECE_ROW_START_Y = 0.1
"""
Where the first loose piece stands sideways, on the side of the sorting arm.
"""

PIECE_ROW_SPACING = 0.1
"""
How far apart the loose pieces stand, so reaching for one does not sweep the arm into
its neighbour.
"""

RELEASE_HEIGHT = 0.01
"""
How far above the board's lid a piece is let go of, so it drops into its hole rather
than being pushed against the lid.
"""

PICK_ARM = Arms.LEFT
"""
The arm that does the sorting; the arm every Tracy demo in this repository uses.
"""

RESTING_CLEARANCE = 0.001
"""
Gap left between a piece and what it stands on, so it rests rather than intersects.
"""


# %% the pieces and the holes they belong in


@dataclass
class MontessoriPiece:
    """
    One loose piece, and where it is let go of over its hole.
    """

    body: Body
    """
    The piece itself.
    """

    annotation: HasRootBody
    """
    What names the piece for an action that takes an annotation rather than a body.
    """

    release_pose: Pose
    """
    Where the piece is let go of, straight above the hole it fits through and turned the
    way that hole is.
    """


BuildsWorld = Callable[[], Tuple[World, Tracy]]
"""
Builds a world holding Tracy, for a backend that does not get one handed to it.
"""

BuildsScene = Callable[[World], List[MontessoriPiece]]
"""
Stands the board and the pieces in a world.
"""

BuildsPlan = Callable[[Context, List[MontessoriPiece]], Plan]
"""
Builds the sorting plan for the pieces a scene put in a world.
"""


def color_of_hue(hue: int) -> Color:
    """
    The colour a hue measured by OpenCV names, at full saturation and brightness.

    :param hue: The hue, on OpenCV's scale of 0 to 180.
    :return: The colour.
    """
    return Color(*colorsys.hsv_to_rgb(hue / 180, 1.0, 1.0))


PALE_BLUE = color_of_hue(86)
"""
Colour of the cube and the cylinder, measured off the real pieces.
"""

YELLOW = color_of_hue(21)
"""
Colour of the two prisms, measured off the real pieces.
"""

PIECE_HEIGHT = 0.03
"""
How tall every piece of the set stands.
"""

CUBE_EDGE_LENGTH = 0.03
"""
Edge length of the cube, which goes through the square hole.
"""

CYLINDER_DIAMETER = 0.028
"""
Diameter of the cylinder, which goes through a circular hole.
"""

RECTANGULAR_PRISM_WIDTH = 0.02
"""
Short side of the rectangular prism, which goes through the rectangular hole.
"""

RECTANGULAR_PRISM_LENGTH = 0.04
"""
Long side of the rectangular prism, along y as the rectangular hole's long side is.
"""

TRIANGULAR_PRISM_SIDE = 0.037
"""
Side of the triangular prism's equilateral cross-section, which goes through the
triangular hole.
"""


TRIANGLE_RELEASE_YAW = -math.pi / 2
"""
Turn that brings the triangular prism's apex, along its own +y, round to the board's +x,
where the triangular hole points.
"""


@dataclass(frozen=True)
class PieceSpecification:
    """
    What a piece is made of and which hole it goes through, before it is built into a
    world.
    """

    name: str
    """
    Name the piece's body carries in the world.
    """

    shape: Shape
    """
    The piece's geometry, turned the way the hole it fits through is.
    """

    hole: HoleShape
    """
    The hole the piece is sorted into.
    """

    release_yaw: float = 0.0
    """
    How far the piece is turned about the vertical, in radians, when it is let go of
    over its hole, so that it lines up with the hole.
    """


def _triangular_prism(side: float, height: float, color: Color) -> Mesh:
    """
    An upright prism of equilateral cross-section, centred on its centroid, its apex
    pointing along its own +y.

    The fingers close along the piece's own y, so they hold it by one flat face and the
    opposite edge; closing on two of its slanted faces would squeeze it out of the hand.

    :param side: Side of the cross-section.
    :param height: How tall the prism stands.
    :param color: Colour it is drawn in.
    :return: The prism.
    """
    circumradius = side / math.sqrt(3)
    inradius = side / (2 * math.sqrt(3))
    outline = np.array(
        [[0.0, circumradius], [-side / 2, -inradius], [side / 2, -inradius]]
    )
    solid = trimesh.creation.extrude_triangulation(
        vertices=outline, faces=np.array([[0, 1, 2]]), height=height
    )
    solid.apply_translation([0.0, 0.0, -height / 2])
    mesh = Mesh.from_trimesh(mesh=solid)
    mesh.color = color
    return mesh


PIECE_SPECIFICATIONS = (
    PieceSpecification(
        name="cube",
        shape=Box(
            scale=Scale(CUBE_EDGE_LENGTH, CUBE_EDGE_LENGTH, PIECE_HEIGHT),
            color=PALE_BLUE,
        ),
        hole=HoleShape.SQUARE,
    ),
    PieceSpecification(
        name="cylinder",
        shape=Cylinder(width=CYLINDER_DIAMETER, height=PIECE_HEIGHT, color=PALE_BLUE),
        hole=HoleShape.CIRCLE,
    ),
    PieceSpecification(
        name="rectangular_prism",
        shape=Box(
            scale=Scale(
                RECTANGULAR_PRISM_WIDTH, RECTANGULAR_PRISM_LENGTH, PIECE_HEIGHT
            ),
            color=YELLOW,
        ),
        hole=HoleShape.RECTANGLE,
    ),
    PieceSpecification(
        name="triangular_prism",
        shape=_triangular_prism(TRIANGULAR_PRISM_SIDE, PIECE_HEIGHT, YELLOW),
        hole=HoleShape.TRIANGLE,
        release_yaw=TRIANGLE_RELEASE_YAW,
    ),
)
"""
The pieces the demo sorts, in the order it sorts them.
"""


# %% building the world the demo runs in


def build_offline_world() -> Tuple[World, Tracy]:
    """
    Build a world holding nothing but Tracy, read from its own description, with both
    arms parked and both grippers open.

    Loading Tracy this way also equips its arms and grippers with the position servos and
    the gravity compensation a physical simulation needs, so the same world serves both
    the MuJoCo and the RViz backend.

    :return: The world, and the Tracy standing in it.
    """
    world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
    robot = Tracy.from_world(world)
    for arm in (robot.left_arm, robot.right_arm):
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        arm.end_effector.get_joint_state_by_type(GripperState.OPEN).apply_to(world)
    return world, robot


def build_scene(world: World) -> List[MontessoriPiece]:
    """
    Stand the shape-sorting board and the loose pieces on Tracy's table.

    The board is bolted down, while every piece hangs off a connection with six degrees
    of freedom, which is what lets the robot take it somewhere else.

    :param world: The world to build the scene in.
    :return: The pieces, in the order they are sorted.
    """
    geometry = BoardGeometry.from_mesh()
    board_z = TABLE_TOP_Z - geometry.bottom
    lid_top_z = board_z + geometry.lid_top

    pieces = []
    with world.modify_world():
        board = board_body(PrefixedName("shape_sorting_board"), geometry)
        world.add_kinematic_structure_entity(board)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=board,
                name=PrefixedName("shape_sorting_board_connection"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    BOARD_X, BOARD_Y, board_z
                ),
            )
        )

        for index, specification in enumerate(PIECE_SPECIFICATIONS):
            hole_x, hole_y = geometry.hole_of_shape(specification.hole).center
            release_pose = Pose.from_xyz_rpy(
                BOARD_X + hole_x,
                BOARD_Y + hole_y,
                lid_top_z + RELEASE_HEIGHT + PIECE_HEIGHT / 2,
                yaw=specification.release_yaw,
                reference_frame=world.root,
            )
            pieces.append(
                _stand_piece_on_the_table(
                    world,
                    specification,
                    PIECE_ROW_START_Y + index * PIECE_ROW_SPACING,
                    release_pose,
                )
            )

    return pieces


def _stand_piece_on_the_table(
    world: World,
    specification: PieceSpecification,
    row_y: float,
    release_pose: Pose,
) -> MontessoriPiece:
    """
    Stand one loose piece in the row in front of Tracy.

    :param world: The world being built, already open for modification.
    :param specification: What the piece is.
    :param row_y: Where along the row it stands.
    :param release_pose: Where it is let go of over its hole.
    :return: The piece.
    """
    body = Body(
        name=PrefixedName(specification.name),
        collision=ShapeCollection([copy.deepcopy(specification.shape)]),
        visual=ShapeCollection([copy.deepcopy(specification.shape)]),
    )
    world.add_kinematic_structure_entity(body)
    world.add_connection(
        Connection6DoF.create_with_dofs(
            world=world,
            parent=world.root,
            child=body,
            name=PrefixedName(f"{specification.name}_connection"),
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                PIECE_ROW_X,
                row_y,
                TABLE_TOP_Z + PIECE_HEIGHT / 2 + RESTING_CLEARANCE,
            ),
        )
    )

    ContactParameters.create_for_grasped_object().apply_to([body])

    annotation = HasRootBody(root=body)
    world.add_semantic_annotations([annotation])
    return MontessoriPiece(body=body, annotation=annotation, release_pose=release_pose)


# %% the plan, the same wherever it runs


def build_plan(context: Context, pieces: List[MontessoriPiece]) -> Plan:
    """
    Pick every piece up in turn and let it go over the hole it fits through.

    The plan still begins by parking the arms: a simulated Tracy already starts parked,
    but the real one starts wherever it was left.

    :param context: The context the actions are built in.
    :param pieces: The pieces to sort.
    :return: The plan that sorts them.
    """
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        context.robot.left_arm.end_effector,
    )

    actions = [ParkArmsAction(Arms.BOTH)]
    for piece in pieces:
        actions.append(PickUpAction(piece.annotation, PICK_ARM, grasp))
        actions.append(PlaceAction(piece.body, piece.release_pose, PICK_ARM))
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
