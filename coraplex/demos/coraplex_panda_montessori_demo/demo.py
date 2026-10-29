"""
The Franka Emika Panda sorts the Montessori pieces from the floor into the holes of the
shape-sorting board, in MuJoCo. Run it from this folder::

    python demo.py
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum

import mujoco

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from montessori_board import HoleShape, ShapeSortingBoard
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    ShapeSortingPiece,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactImpedance,
    ContactParameters,
    ContactStiffness,
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

# %% the scene


class SceneBody(StrEnum):
    """
    The names of the scene's bodies other than the pieces.
    """

    FLOOR = "floor"
    BOARD = "shape_sorting_board"


@dataclass
class ShapeSortingPieceSet:
    """
    The loose pieces of the set, with the sizes and colours measured off the real set.
    """

    piece_height: float = 0.03
    """
    How tall every piece of the set stands.
    """

    cube_edge_length: float = 0.03
    """
    Edge length of the cube, which goes through the square hole.
    """

    cylinder_diameter: float = 0.028
    """
    Diameter of the cylinder, which goes through a circular hole.
    """

    rectangular_prism_width: float = 0.02
    """
    Short side of the rectangular prism, which goes through the rectangular hole.
    """

    rectangular_prism_length: float = 0.04
    """
    Long side of the rectangular prism, along y as the rectangular hole's long side is.
    """

    triangular_prism_side: float = 0.037
    """
    Side of the triangular prism's equilateral cross-section, which goes through the
    triangular hole.
    """

    def shapes(self) -> dict[HoleShape, Shape]:
        """
        :return: The shape of the piece that goes through each hole, turned the way that
            hole is, in the order the pieces are sorted.
        """
        return {
            HoleShape.SQUARE: Box(
                scale=Scale(
                    self.cube_edge_length, self.cube_edge_length, self.piece_height
                ),
                color=Color.CYAN(),
            ),
            HoleShape.CIRCLE: Cylinder(
                width=self.cylinder_diameter,
                height=self.piece_height,
                color=Color.CYAN(),
            ),
            HoleShape.RECTANGLE: Box(
                scale=Scale(
                    self.rectangular_prism_width,
                    self.rectangular_prism_length,
                    self.piece_height,
                ),
                color=Color.YELLOW(),
            ),
            HoleShape.TRIANGLE: Mesh.triangular_prism(
                self.triangular_prism_side, self.piece_height
            ),
        }


# %% the demonstration


@dataclass
class PandaMontessoriDemonstration(RobotDemonstration):
    """
    The Panda picks every loose piece off the floor in turn and lets it go over the hole
    of the shape-sorting board it fits through.
    """

    used_robot: type[AbstractRobot] = Panda

    ros_node_name: str = "panda_montessori_demo"

    execution_type: ExecutionType = ExecutionType.PHYSICALLY_SIMULATED

    floor: BodySpecification = field(
        default_factory=lambda: BodySpecification.box(
            SceneBody.FLOOR,
            Scale(2.0, 2.0, 0.02),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-0.01),
        )
    )
    """
    The floor the robot is bolted to and the board and the pieces stand on.
    """

    board_x: float = 0.6
    """
    How far in front of the robot the board stands, with its drawers facing the robot.
    """

    piece_row_x: float = 0.42
    """
    How far in front of the robot the loose pieces stand: nearer than the board, and far
    enough from it that reaching for a piece does not sweep the arm into the board.
    """

    piece_row_start_y: float = -0.15
    """
    Where the first loose piece stands sideways.
    """

    piece_row_spacing: float = 0.1
    """
    How far apart the loose pieces stand, so reaching for one does not sweep the hand
    into its neighbour.
    """

    resting_clearance: float = 0.001
    """
    Gap left between a piece and the floor, so it rests rather than intersects.
    """

    closed_pad_clearance: float = 0.005
    """
    How far the closed finger pads stay above the floor under a gripped piece.
    """

    release_height: float = 0.01
    """
    How far above the board's lid a piece is let go of, so it drops into its hole rather
    than being pushed against the lid.
    """

    pieces: ShapeSortingPieceSet = field(default_factory=ShapeSortingPieceSet)
    """
    The loose pieces.
    """

    shape_sorting_board: ShapeSortingBoard = field(default_factory=ShapeSortingBoard)
    """
    The board, as read off its mesh, and the body it becomes.
    """

    solid_contact_stiffness: ContactStiffness = field(
        default_factory=lambda: ContactStiffness(
            time_constant=timedelta(milliseconds=2), damping_ratio=1.0
        )
    )
    """
    How stiffly the board and the pieces push back once something touches them: as
    stiffly as physics stepping every millisecond stays stable at, a time constant of
    two steps.
    """

    solid_contact_impedance: ContactImpedance = field(
        default_factory=lambda: ContactImpedance(
            minimum=0.99, maximum=0.999, width=0.001
        )
    )
    """
    How little the board and the pieces give way under load: nearly not at all.
    """

    friction_cone: mujoco.mjtCone = mujoco.mjtCone.mjCONE_ELLIPTIC
    """
    The friction cone the contacts use: the elliptic cone is the exact one, where the
    default pyramid only approximates it and lets a squeezed piece creep between the
    fingers.
    """

    impedance_ratio: float = 10.0
    """
    How much harder sliding is than pressing in: MuJoCo's own guidance for grasping with
    an elliptic cone.
    """

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification(objects=[self.floor])

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(self.used_robot)

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(SceneBody.BOARD)

    def populate_scene(self, world: World) -> None:
        """
        Bolt the board to the floor and stand the loose pieces in a row in front of it,
        each free to be picked up.
        """
        with world.modify_world():
            board = self.shape_sorting_board.body(PrefixedName(SceneBody.BOARD))
            self._solid(ContactFriction()).apply_to([board])
            world.add_kinematic_structure_entity(board)
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=board,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        self.board_x,
                        0.0,
                        -self.shape_sorting_board.geometry.bottom,
                        reference_frame=world.root,
                    ),
                )
            )
            for index, (hole, shape) in enumerate(self.pieces.shapes().items()):
                self._stand_piece(
                    world,
                    PrefixedName(f"{hole}_piece"),
                    shape,
                    self.piece_row_start_y + index * self.piece_row_spacing,
                )

    def build_context(self, world: World) -> Context:
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            _debug=self.debug,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Park the arm, sort every loose piece into its hole in turn, and park again.
        """
        world = context.world
        panda = context.robot
        gripper = panda.arm.end_effector
        actions = [ParkArmsAction(panda.all_arms)]
        for hole in self.pieces.shapes():
            piece = self._piece_for(hole, world)
            piece.grasp_height = self._grasp_height(gripper)
            [grasp] = piece.grasp_candidates()
            actions.append(PickUpAction(grasp=grasp, arm=panda.arm))
            actions.append(
                PlaceAction(
                    object_designator=piece,
                    target_location=self._release_pose_over(hole, world),
                )
            )
        actions.append(ParkArmsAction(panda.all_arms))
        return sequential(actions, context=context).plan

    def build_simulation(self, world: World) -> MujocoSim:
        return MujocoSim(
            world=world,
            headless=self.headless,
            cone=self.friction_cone,
            impratio=self.impedance_ratio,
        )

    def _solid(self, friction: ContactFriction) -> ContactParameters:
        """
        :param friction: The friction the surface offers.
        :return: The contact of a solid surface offering ``friction``.
        """
        return ContactParameters(
            friction=friction,
            stiffness=self.solid_contact_stiffness,
            impedance=self.solid_contact_impedance,
        )

    def _stand_piece(
        self, world: World, name: PrefixedName, shape: Shape, row_y: float
    ) -> None:
        """
        Stand one loose piece in the row in front of the board.

        :param world: The world being built, already open for modification.
        :param name: The name the piece's body carries.
        :param shape: The piece's geometry, centred on its own origin.
        :param row_y: Where along the row it stands.
        """
        body = Body(name=name)
        shape.origin = HomogeneousTransformationMatrix(reference_frame=body)
        body.collision = ShapeCollection([shape], reference_frame=body)
        body.visual = ShapeCollection([copy.deepcopy(shape)], reference_frame=body)
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    self.piece_row_x,
                    row_y,
                    self.resting_clearance + self.pieces.piece_height / 2,
                    reference_frame=world.root,
                ),
            )
        )
        self._solid(ContactParameters.create_for_grasped_object().friction).apply_to(
            [body]
        )
        world.add_semantic_annotations([ShapeSortingPiece(root=body, grasp_height=0.0)])

    def _piece_for(self, hole: HoleShape, world: World) -> ShapeSortingPiece:
        """
        :return: The loose piece that goes through ``hole``.
        """
        body = world.get_body_by_name(f"{hole}_piece")
        [piece] = [
            piece
            for piece in world.get_semantic_annotations_by_type(ShapeSortingPiece)
            if piece.root is body
        ]
        return piece

    def _grasp_height(self, gripper: HasTwoFingers) -> float:
        """
        :param gripper: The gripper that takes hold of the pieces.
        :return: How far above its bottom face a piece is gripped, at the point the
            closed fingers meet, so the closed pads stay :attr:`closed_pad_clearance`
            above the floor.
        """
        tool_frame_above_the_meeting_point = -float(
            gripper.tool_P_grasp_when_closed.vector3.dot(gripper.approach_axis)
        )
        return (
            self.closed_pad_clearance
            + gripper.travel_when_closing
            + tool_frame_above_the_meeting_point
        )

    def _release_pose_over(self, hole: HoleShape, world: World) -> Pose:
        """
        :param hole: The hole a piece is sorted into.
        :param world: The world the board stands in.
        :return: Where that piece is let go of: straight above the hole,
            :attr:`release_height` above the lid. The piece keeps the orientation it
            stood in, which is the hole's.
        """
        geometry = self.shape_sorting_board.geometry
        center = geometry.hole_of_shape(hole).bounds.center
        board_T_release = Pose.from_xyz_rpy(
            center.x,
            center.y,
            geometry.lid_top + self.release_height + self.pieces.piece_height / 2,
            reference_frame=world.get_body_by_name(SceneBody.BOARD),
        )
        return world.transform(board_T_release, world.root)


def main(headless: bool = False) -> World:
    """
    Run the demonstration.

    :param headless: Whether the physics runs without a viewer.
    :return: The world the pieces were sorted in.
    """
    return PandaMontessoriDemonstration(headless=headless).run()


if __name__ == "__main__":
    main()
