"""
Tracy sorting the Montessori pieces from its table into the holes of the shape-sorting
board.

The scene and the plan are the same wherever the demo runs; only how the plan is carried
out differs. Pick :data:`BACKEND` and run this file from its own folder::

    python demo.py
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import List, Optional

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from montessori_board import BoardGeometry, HoleShape, board_body
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    HasGraspCandidates,
)
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    Vector3,
)
from semantic_digital_twin.spatial_types.spatial_types import RotationMatrix
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

RESTING_CLEARANCE = 0.001
"""
Gap left between a piece and what it stands on, so it rests rather than intersects.
"""


# %% the pieces and the holes they belong in


@dataclass
class MontessoriPiece:
    """
    One loose piece, how it is grasped, and where it is let go of over its hole.
    """

    body: Body
    """
    The piece itself.
    """

    annotation: HasGraspCandidates
    """
    What names the piece for the actions that take it.
    """

    grasp: GraspCandidate
    """
    How the piece is taken: from above, low on its sides (see :data:`GRASP_HEIGHT`).
    """

    release_pose: Pose
    """
    Where the piece is let go of, straight above the hole it fits through and turned the
    way that hole is.
    """


PIECE_HEIGHT = 0.03
"""
How tall every piece of the set stands.
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
How far the closed finger pads stay above what a grasped piece stands on.
"""

GRASP_HEIGHT = FINGER_PAD_CLOSING_TRAVEL + CLOSED_PAD_CLEARANCE
"""
How far above its bottom face a piece is grasped.

A grasp brings the tool frame to the grasp point, and the pads end there while the hand
is open; closing, they travel :data:`FINGER_PAD_CLOSING_TRAVEL` further down. Grasped
this high, the closed pads stay :data:`CLOSED_PAD_CLEARANCE` above the table and cover
the rest of the piece. Grasped at its middle instead, the closing pads would reach below
the piece's bottom face and press into the table.
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


PIECE_SPECIFICATIONS = (
    PieceSpecification(
        name="cube",
        shape=Box(
            scale=Scale(CUBE_EDGE_LENGTH, CUBE_EDGE_LENGTH, PIECE_HEIGHT),
            color=Color.CYAN(),
        ),
        hole=HoleShape.SQUARE,
    ),
    PieceSpecification(
        name="cylinder",
        shape=Cylinder(
            width=CYLINDER_DIAMETER, height=PIECE_HEIGHT, color=Color.CYAN()
        ),
        hole=HoleShape.CIRCLE,
    ),
    PieceSpecification(
        name="rectangular_prism",
        shape=Box(
            scale=Scale(
                RECTANGULAR_PRISM_WIDTH, RECTANGULAR_PRISM_LENGTH, PIECE_HEIGHT
            ),
            color=Color.YELLOW(),
        ),
        hole=HoleShape.RECTANGLE,
    ),
    PieceSpecification(
        name="triangular_prism",
        shape=Mesh.triangular_prism(TRIANGULAR_PRISM_SIDE, PIECE_HEIGHT),
        hole=HoleShape.TRIANGLE,
    ),
)
"""
The pieces the demo sorts, in the order it sorts them.
"""


# %% the scene the demo runs in

TOP_GRASP_ORIENTATION = RotationMatrix.from_vectors(
    x=Vector3.NEGATIVE_Z(), y=Vector3.X()
)
"""
How every piece is grasped: approached from above, with the fingers closing along the
piece's own x, so they hold the triangular prism by one flat face and the opposite edge;
closing on two of its slanted faces would squeeze it out of the hand.
"""


@dataclass
class MontessoriScene:
    """
    Tracy at its table, with the shape-sorting board and the loose pieces in front of it.
    """

    world: World
    """
    The world everything stands in.
    """

    robot: Tracy
    """
    The robot that sorts the pieces.
    """

    pieces: List[MontessoriPiece]
    """
    The loose pieces, in the order they are sorted.
    """

    def build_plan(self, context: Context) -> Plan:
        """
        Pick every piece up in turn and let it go over the hole it fits through.

        The plan begins by parking the arms: a simulated Tracy starts where its
        description leaves it, the real one wherever it was left.

        :param context: The context the actions are built in.
        :return: The plan that sorts the pieces.
        """
        arm = self.robot.left_arm
        actions = [ParkArmsAction(self.robot.all_arms)]
        for piece in self.pieces:
            actions.append(PickUpAction(piece.grasp, arm))
            actions.append(PlaceAction(piece.annotation, piece.release_pose))
        actions.append(ParkArmsAction(self.robot.all_arms))
        return sequential(actions, context=context).plan


def build_scene(world: Optional[World] = None) -> MontessoriScene:
    """
    Stand the shape-sorting board and the loose pieces on Tracy's table.

    The board is bolted down, while every piece hangs off a connection with six degrees
    of freedom, which is what lets the robot take it somewhere else.

    :param world: The world holding Tracy, as fetched from the real robot. Without one,
        a world holding nothing but Tracy is read from its own description, which also
        equips its arms and grippers with the servos a physical simulation needs.
    :return: The scene.
    """
    if world is None:
        world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
        Tracy.from_world(world)
    [robot] = world.get_semantic_annotations_by_type(Tracy)

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
            hole_x, hole_y = geometry.hole_of_shape(
                specification.hole
            ).bounding_box_center
            release_pose = Pose.from_xyz_rpy(
                BOARD_X + hole_x,
                BOARD_Y + hole_y,
                lid_top_z + RELEASE_HEIGHT + PIECE_HEIGHT / 2,
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

    return MontessoriScene(world=world, robot=robot, pieces=pieces)


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
    body = Body(name=PrefixedName(specification.name))
    collision_shape, visual_shape = (
        copy.deepcopy(specification.shape),
        copy.deepcopy(specification.shape),
    )
    for shape in (collision_shape, visual_shape):
        shape.origin = HomogeneousTransformationMatrix(reference_frame=body)
    body.collision = ShapeCollection([collision_shape], reference_frame=body)
    body.visual = ShapeCollection([visual_shape], reference_frame=body)
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
                TABLE_TOP_Z + RESTING_CLEARANCE + PIECE_HEIGHT / 2,
            ),
        )
    )

    ContactParameters.create_for_grasped_object().apply_to([body])

    annotation = HasGraspCandidates(root=body)
    world.add_semantic_annotations([annotation])
    grasp = GraspCandidate(
        annotation,
        Pose(
            position=Point3(
                0.0, 0.0, GRASP_HEIGHT - PIECE_HEIGHT / 2, reference_frame=body
            ),
            orientation=TOP_GRASP_ORIENTATION.to_quaternion(),
            reference_frame=body,
        ),
    )
    return MontessoriPiece(
        body=body, annotation=annotation, grasp=grasp, release_pose=release_pose
    )


# %% running it


def main() -> None:
    """
    Run the demo on the backend :data:`BACKEND` names.

    Each backend is imported only once it is chosen, so a run needs nothing installed for
    the backends it does not use.
    """
    if BACKEND is Backend.MUJOCO:
        import mujoco_demo

        mujoco_demo.run(build_scene())
        return

    if BACKEND is Backend.RVIZ:
        import rviz_demo

        rviz_demo.run(build_scene())
        return

    import real_demo

    with real_demo.running_robot() as robot:
        real_demo.run(build_scene(robot.world), robot)


if __name__ == "__main__":
    main()
