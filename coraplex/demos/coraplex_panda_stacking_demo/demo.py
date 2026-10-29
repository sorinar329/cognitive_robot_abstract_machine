"""
The Franka Emika Panda stacks cubes into a tower in MuJoCo. Run it from this folder::

    python demo.py
"""

from __future__ import annotations

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
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import (
    ContactImpedance,
    ContactParameters,
    ContactStiffness,
)
from semantic_digital_twin.world_description.geometry import Color, Scale

# %% the scene


class SceneBody(StrEnum):
    """
    The names of the scene's bodies other than the cubes.
    """

    FLOOR = "floor"


class CubeName(StrEnum):
    """
    The names of the cubes, after their colours.
    """

    RED = "red_cube"
    GREEN = "green_cube"
    BLUE = "blue_cube"


@dataclass
class Cube:
    """
    A cube of the tower: its name and colour, and where it stands before the robot moves
    it.
    """

    name: CubeName
    """
    The name of the cube's body.
    """

    color: Color
    """
    The colour it is drawn in.
    """

    x: float
    """
    How far in front of the robot it stands.
    """

    y: float
    """
    Where it stands sideways.
    """


# %% the demonstration


@dataclass
class PandaStackingDemonstration(RobotDemonstration):
    """
    The Panda picks every loose cube off the floor in turn and lets it go on top of the
    tower, whose bottom cube stands on the floor already.
    """

    used_robot: type[AbstractRobot] = Panda

    ros_node_name: str = "panda_stacking_demo"

    execution_type: ExecutionType = ExecutionType.PHYSICALLY_SIMULATED

    floor: BodySpecification = field(
        default_factory=lambda: BodySpecification.box(
            SceneBody.FLOOR,
            Scale(2.0, 2.0, 0.02),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-0.01),
        )
    )
    """
    The floor the robot is bolted to and the cubes stand on.
    """

    cube_edge_length: float = 0.05
    """
    Edge length of every cube.
    """

    tower: Cube = field(
        default_factory=lambda: Cube(CubeName.RED, Color.RED(), 0.5, -0.15)
    )
    """
    The bottom cube of the tower, which stays where it stands.
    """

    loose_cubes: list[Cube] = field(
        default_factory=lambda: [
            Cube(CubeName.GREEN, Color.GREEN(), 0.45, 0.05),
            Cube(CubeName.BLUE, Color.BLUE(), 0.45, 0.15),
        ]
    )
    """
    The cubes stacked onto the tower, in the order they are stacked.
    """

    grasp_height: float = 0.03
    """
    How far above its bottom face a cube is gripped, which keeps the fingers clear of the
    cube below it when it is let go of on the tower.
    """

    release_height: float = 0.003
    """
    How far above the cube below it a cube is let go of, so it settles onto it rather
    than being pushed into it.
    """

    solid_contact_stiffness: ContactStiffness = field(
        default_factory=lambda: ContactStiffness(
            time_constant=timedelta(milliseconds=2), damping_ratio=1.0
        )
    )
    """
    How stiffly the cubes push back once something touches them: as stiffly as physics
    stepping every millisecond stays stable at, a time constant of two steps.
    """

    solid_contact_impedance: ContactImpedance = field(
        default_factory=lambda: ContactImpedance(
            minimum=0.99, maximum=0.999, width=0.001
        )
    )
    """
    How little the cubes give way under load: nearly not at all.
    """

    friction_cone: mujoco.mjtCone = mujoco.mjtCone.mjCONE_ELLIPTIC
    """
    The friction cone the contacts use: the elliptic cone is the exact one, where the
    default pyramid only approximates it and lets a squeezed cube creep between the
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
        return world.is_kinematic_structure_entity_in_world_by_name(self.tower.name)

    def populate_scene(self, world: World) -> None:
        """
        Stand every cube on the floor, free to be picked up.
        """
        for cube in [self.tower, *self.loose_cubes]:
            annotation = SemanticAnnotationWithRootSpecification(
                name=cube.name,
                semantic_annotation_type=HasGraspCandidates,
                root_specification=BodySpecification.box(
                    cube.name,
                    Scale(
                        self.cube_edge_length,
                        self.cube_edge_length,
                        self.cube_edge_length,
                    ),
                    color=cube.color,
                    connection_specification=Connection6DoFSpecification(),
                ),
            ).spawn(world)
            grasped = ContactParameters.create_for_grasped_object()
            ContactParameters(
                friction=grasped.friction,
                stiffness=self.solid_contact_stiffness,
                impedance=self.solid_contact_impedance,
            ).apply_to([annotation.root])
            annotation.root.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    cube.x,
                    cube.y,
                    self.cube_edge_length / 2,
                    reference_frame=world.root,
                )
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
        Park the arm, stack every loose cube onto the tower in turn, and park again.
        """
        world = context.world
        panda = context.robot
        actions = [ParkArmsAction(panda.all_arms)]
        for level, cube in enumerate(self.loose_cubes, start=1):
            annotation = self._annotation_of(cube, world)
            actions.append(
                PickUpAction(grasp=self._grasp_from_above(annotation), arm=panda.arm)
            )
            actions.append(
                PlaceAction(
                    object_designator=annotation,
                    target_location=self._release_pose_at(level, world),
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

    def _annotation_of(self, cube: Cube, world: World) -> HasGraspCandidates:
        """
        :return: The annotation of ``cube`` in ``world``.
        """
        body = world.get_body_by_name(cube.name)
        [annotation] = [
            annotation
            for annotation in world.get_semantic_annotations_by_type(HasGraspCandidates)
            if annotation.root is body
        ]
        return annotation

    def _grasp_from_above(self, cube: HasGraspCandidates) -> GraspCandidate:
        """
        :return: The grasp that approaches ``cube`` from above, the fingers closing along
            its own x, :attr:`grasp_height` above its bottom face.
        """
        return GraspCandidate(
            cube,
            Pose(
                position=Point3(
                    0.0,
                    0.0,
                    self.grasp_height - self.cube_edge_length / 2,
                    reference_frame=cube.root,
                ),
                orientation=RotationMatrix.from_vectors(
                    x=Vector3.NEGATIVE_Z(), y=Vector3.X()
                ).quaternion,
                reference_frame=cube.root,
            ),
        )

    def _release_pose_at(self, level: int, world: World) -> Pose:
        """
        :param level: How many cubes the tower holds below the released one.
        :param world: The world the tower stands in.
        :return: Where that cube is let go of: straight above the tower,
            :attr:`release_height` above the cube below it.
        """
        return Pose.from_xyz_rpy(
            self.tower.x,
            self.tower.y,
            level * self.cube_edge_length
            + self.release_height
            + self.cube_edge_length / 2,
            reference_frame=world.root,
        )


def main(headless: bool = False) -> World:
    """
    Run the demonstration.

    :param headless: Whether the physics runs without a viewer.
    :return: The world the tower was built in.
    """
    return PandaStackingDemonstration(headless=headless).run()


if __name__ == "__main__":
    main()
