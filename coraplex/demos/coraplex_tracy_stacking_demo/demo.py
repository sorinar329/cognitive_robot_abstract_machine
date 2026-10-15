"""
Tracy stacking cubes from its table onto a tower.

The scene and the plan are the same however the demo is carried out; only the execution
type differs. Choose it at the bottom of this file, then run the file from its own
folder::

    python demo.py
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import timedelta

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from krrood.exceptions import DataclassException
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import StaticJointState
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
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactImpedance,
    ContactParameters,
    ContactStiffness,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% when the demo cannot run


@dataclass
class DemoCannotBeCarriedOut(DataclassException):
    """
    Raised when the demo is asked to run with an execution type it has no runner for.
    """

    execution_type: ExecutionType
    """
    The execution type asked for.
    """

    def error_message(self) -> str:
        return f"The demo cannot be carried out as {self.execution_type.name}."

    def suggest_correction(self) -> str:
        return "Run it as KINEMATICALLY_SIMULATED, PHYSICALLY_SIMULATED or REAL."


# %% the scene the demo runs in


@dataclass
class StackingEnvironment:
    """
    Tracy's table, with the bottom cube of a tower standing on it and a row of loose
    cubes in front of Tracy, each to be stacked onto the tower in turn.
    """

    table_top_z: float = 0.88
    """
    Height of the work surface of the table Tracy is bolted to, which every cube rests
    on. It is the top of the table's own collision slab, read off the robot's
    description rather than guessed.
    """

    tower_x: float = 0.75
    """
    How far in front of Tracy the tower is built, beyond the row of loose cubes.
    """

    tower_y: float = 0.0
    """
    Where the tower is built sideways.
    """

    cube_row_x: float = 0.55
    """
    How far in front of Tracy the loose cubes stand.
    """

    cube_row_start_y: float = 0.1
    """
    Where the first loose cube stands sideways, on the side of the stacking arm.
    """

    cube_row_spacing: float = 0.1
    """
    How far apart the loose cubes stand, so reaching for one does not sweep the arm into
    its neighbour.
    """

    release_height: float = 0.003
    """
    How far above the cube below it a cube is let go of, so it settles onto it rather
    than being pushed into it.
    """

    resting_clearance: float = 0.001
    """
    Gap left between a cube and the table, so it rests rather than intersects.
    """

    cube_edge_length: float = 0.04
    """
    Edge length of every cube.
    """

    cube_colors: list[Color] = field(
        default_factory=lambda: [
            Color.RED(),
            Color.GREEN(),
            Color.BLUE(),
            Color.YELLOW(),
        ]
    )
    """
    The colour of every cube, from the bottom of the tower up; there is one cube per
    colour.
    """

    finger_pad_closing_travel: float = 0.0135
    """
    How far Tracy's finger pads move out along the gripper's reach as it closes.

    Measured on the robot's own description: open, the pads end at the tool frame;
    closed, 13.5 mm past it. The Robotiq linkage swings the pads down onto whatever the
    hand is lowered over.
    """

    closed_pad_clearance: float = 0.005
    """
    How far the closed finger pads stay above the bottom of a grasped cube, so that they
    clear whatever the cube is stacked on.
    """

    solid_contact_stiffness: ContactStiffness = field(
        default_factory=lambda: ContactStiffness(
            time_constant=timedelta(milliseconds=2), damping_ratio=1.0
        )
    )
    """
    How stiffly the solid bodies of the scene push back once something touches them: as
    stiffly as a physics engine stepping every millisecond stays stable at, a time
    constant of two steps.

    The wooden cubes and the robot are solid, so nothing may sink into them; MuJoCo's
    default of 20 ms lets a firmly held cube sink several millimetres into the pads.
    """

    solid_contact_impedance: ContactImpedance = field(
        default_factory=lambda: ContactImpedance(
            minimum=0.99, maximum=0.999, width=0.001
        )
    )
    """
    How little the solid bodies of the scene give way under load: nearly not at all.
    """

    loose_cubes: list[HasGraspCandidates] = field(init=False, default_factory=list)
    """
    The cubes still to be stacked, in the order they are stacked.
    """

    @property
    def grasp_height(self) -> float:
        """
        How far above its bottom face a cube is gripped.

        A grasp brings the tool frame to the grasp point, and the pads end there while
        the hand is open; closing, they travel :attr:`finger_pad_closing_travel` further
        down. Gripped this high, the closed pads stay :attr:`closed_pad_clearance` above
        the cube's bottom and cover most of its sides.
        """
        return self.finger_pad_closing_travel + self.closed_pad_clearance

    def get_world(self) -> World:
        """
        Create a world holding Tracy, read from its own description, which also equips
        its arms and grippers with the servos a physical simulation needs, with its arms
        parked and its body as solid as the cubes it holds, and stand the cubes in it.

        A contact is as soft as the softer of the two shapes in it, so a solid cube held
        between soft pads would still sink into them.
        """
        world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
        robot = Tracy.from_world(world)
        self.solid(ContactFriction()).apply_to(world.bodies_with_collision)
        for arm in robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        self.populate(world)
        return world

    def solid(self, friction: ContactFriction) -> ContactParameters:
        """
        :param friction: The friction the surface offers.
        :return: The contact of a solid surface offering ``friction``.
        """
        return ContactParameters(
            friction=friction,
            stiffness=self.solid_contact_stiffness,
            impedance=self.solid_contact_impedance,
        )

    def populate(self, world: World) -> None:
        """
        Stand the bottom cube of the tower where the tower is built, and the other cubes
        in a row in front of Tracy.

        Every cube hangs off a connection with six degrees of freedom, which is what lets
        the robot take it somewhere else.

        :param world: The world holding Tracy.
        """
        bottom_color, *loose_colors = self.cube_colors
        with world.modify_world():
            self._stand_cube(world, 0, bottom_color, self.tower_x, self.tower_y)
            for index, color in enumerate(loose_colors):
                self.loose_cubes.append(
                    self._stand_cube(
                        world,
                        index + 1,
                        color,
                        self.cube_row_x,
                        self.cube_row_start_y + index * self.cube_row_spacing,
                    )
                )

    def build_plan(self, context: Context) -> Plan:
        """
        Pick every loose cube up in turn and let it go on top of the tower.

        The plan begins by parking the arms: a simulated Tracy starts parked already, the
        real one wherever it was left.

        :param context: The context the actions are built in.
        :return: The plan that stacks the cubes.
        """
        robot = context.robot
        actions = [ParkArmsAction(robot.all_arms)]
        for level, cube in enumerate(self.loose_cubes, start=1):
            actions.append(PickUpAction(self.grasp_from_above(cube), robot.left_arm))
            actions.append(
                PlaceAction(cube, self.release_pose_at(level, context.world))
            )
        actions.append(ParkArmsAction(robot.all_arms))
        return sequential(actions, context=context).plan

    def grasp_from_above(self, cube: HasGraspCandidates) -> GraspCandidate:
        """
        :param cube: The cube to grasp.
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

    def release_pose_at(self, level: int, world: World) -> Pose:
        """
        :param level: How many cubes the tower holds below the released one.
        :param world: The world the tower stands in.
        :return: Where that cube is let go of: straight above the tower,
            :attr:`release_height` above the cube below it.
        """
        return Pose.from_xyz_rpy(
            self.tower_x,
            self.tower_y,
            self.table_top_z
            + level * self.cube_edge_length
            + self.release_height
            + self.cube_edge_length / 2,
            reference_frame=world.root,
        )

    def _stand_cube(
        self, world: World, index: int, color: Color, x: float, y: float
    ) -> HasGraspCandidates:
        """
        Stand one cube on the table.

        :param world: The world being built, already open for modification.
        :param index: Which cube it is, counted from the bottom of the tower.
        :param color: The colour it is drawn in.
        :param x: How far in front of Tracy it stands.
        :param y: Where it stands sideways.
        :return: The cube.
        """
        body = Body(name=PrefixedName(f"cube_{index}"))
        shape = Box(
            origin=HomogeneousTransformationMatrix(reference_frame=body),
            scale=Scale(
                self.cube_edge_length, self.cube_edge_length, self.cube_edge_length
            ),
            color=color,
        )
        body.collision = ShapeCollection([shape], reference_frame=body)
        body.visual = ShapeCollection([copy.deepcopy(shape)], reference_frame=body)
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=body,
                name=PrefixedName(f"cube_{index}_connection"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x,
                    y,
                    self.table_top_z
                    + self.resting_clearance
                    + self.cube_edge_length / 2,
                ),
            )
        )
        self.solid(ContactParameters.create_for_grasped_object().friction).apply_to(
            [body]
        )
        cube = HasGraspCandidates(root=body)
        world.add_semantic_annotations([cube])
        return cube


# %% running it


def main(execution_type: ExecutionType) -> None:
    """
    Carry the demo out the way ``execution_type`` names.

    Each runner is imported only once it is chosen, so a run needs nothing installed for
    the execution types it does not use.

    :param execution_type: How the plan is carried out.
    """
    environment = StackingEnvironment()
    match execution_type:
        case ExecutionType.PHYSICALLY_SIMULATED:
            import mujoco_demo

            mujoco_demo.run(environment, environment.get_world())
        case ExecutionType.KINEMATICALLY_SIMULATED:
            import rviz_demo

            rviz_demo.run(environment, environment.get_world())
        case ExecutionType.REAL:
            import real_demo

            with real_demo.running_robot() as robot:
                environment.populate(robot.world)
                real_demo.run(environment, robot)
        case _:
            raise DemoCannotBeCarriedOut(execution_type)


if __name__ == "__main__":
    main(ExecutionType.KINEMATICALLY_SIMULATED)
