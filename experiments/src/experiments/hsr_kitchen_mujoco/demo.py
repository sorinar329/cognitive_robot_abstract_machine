"""
The Toyota HSR carries a milk carton and a cereal box from one table of the IAI kitchen
to the other, simulated in MuJoCo.

The kitchen is the full IAI kitchen, whose two tables stand at its back: the HSR picks
both objects up from the table behind the low partition and sets them down on the dining
table next to it. Its arm, head and hand are driven by servos; its base is moved by its
drive and MuJoCo follows it there rather than rolling it on its wheels.

Run it with::

    python -m experiments.hsr_kitchen_mujoco.demo
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing_extensions import Optional, Type

import mujoco
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    ExecutionType,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import ActionNode, MotionNode, PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.visualization import WorldVisualization
from giskardpy.executor import SteppedSimulationPacer
from semantic_digital_twin.adapters.controlled_simulation import ControlledSimulation
from semantic_digital_twin.adapters.multi_sim import (
    ContactDimensionality,
    MujocoEquality,
    MujocoGeom,
    MujocoSim,
)
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cereal,
    Food,
    Milk,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Scale,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the kitchen

KITCHEN_DESCRIPTION = "package://iai_kitchen/urdf_obj/iai_kitchen_python.urdf.xacro"
"""
The full IAI kitchen, with the two tables at its back.
"""


class KitchenTable(StrEnum):
    """
    The tables the objects are carried between, by their bodies' names in the kitchen.
    """

    SOURCE = "table_area_main"
    """
    The table behind the low partition, where both objects start.
    """

    DESTINATION = "dining_area_jokkmokk_table_main"
    """
    The dining table, where both objects end up.
    """


def table_top_height(world: World, table: KitchenTable) -> float:
    """
    :return: The height of the top of ``table`` above the kitchen floor.
    """
    return float(table_extent(world, table).max_z)


def table_extent(world: World, table: KitchenTable) -> VolumetricBoundingBox:
    """
    :return: The axis-aligned box ``table`` occupies in the kitchen's frame.
    """
    body = world.get_body_by_name(table)
    return body.collision.as_bounding_box_collection_in_frame(world.root).bounding_box()


# %% what is carried


@dataclass(frozen=True)
class CarriedObject:
    """
    One object the HSR carries, where it starts and where it is set down.
    """

    name: str
    """
    The name of the object's body.
    """

    annotation_type: Type[Food]
    """
    What the object is.
    """

    scale: Scale
    """
    The object's extents: along the gripper's approach, across the gripper, and its
    height.
    """

    color: Color
    """
    The object's colour.
    """

    start_x: float
    """
    Where along the source table the object starts, in the kitchen's x.
    """

    goal_y: float
    """
    Where along the destination table the object is set down, in the kitchen's y.
    """


MILK = CarriedObject(
    name="milk",
    annotation_type=Milk,
    scale=Scale(0.06, 0.06, 0.2),
    color=Color(0.95, 0.95, 0.95),
    start_x=-2.6,
    goal_y=0.1,
)

CEREAL = CarriedObject(
    name="cereal",
    annotation_type=Cereal,
    scale=Scale(0.15, 0.06, 0.2),
    color=Color(0.85, 0.55, 0.15),
    start_x=-2.2,
    goal_y=0.5,
)

CARRIED_OBJECTS = (MILK, CEREAL)
"""
The objects, in the order they are carried.
"""

EDGE_INSET = 0.1
"""
How far in from the edge the HSR faces an object stands or is set down, in metres:
close enough to reach without leaning over the table.
"""

STANDING_DISTANCE = 0.45
"""
How far from a table's edge the HSR's base stands while it works there, in metres:
clear of the edge and of the partition, with the object within the arm's reach.
"""

SETTLING_CLEARANCE = 0.002
"""
How far above a table top an object is put, so it settles onto it rather than starting
inside it.
"""

RELEASE_CLEARANCE = 0.01
"""
How far above the destination table the HSR lets an object go, in metres: pressed onto
the table by a hand that tracks its target only so closely, an object tips in the hand
and falls over once it is let go; dropped from a centimetre it lands upright.
"""


def start_pose(world: World, carried: CarriedObject) -> HomogeneousTransformationMatrix:
    """
    :return: Where ``carried`` stands on the source table, near its edge facing the
        partition, which is where the HSR reaches over to it from. Its x axis points
        away from the HSR, the direction a front grasp approaches it in.
    """
    extent = table_extent(world, KitchenTable.SOURCE)
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x=carried.start_x,
        y=extent.max_y - EDGE_INSET,
        z=extent.max_z + carried.scale.z / 2 + SETTLING_CLEARANCE,
        yaw=-math.pi / 2,
        reference_frame=world.root,
    )


def goal_pose(world: World, carried: CarriedObject) -> Pose:
    """
    :return: Where ``carried`` is set down on the destination table, near its edge
        facing the kitchen, its x axis pointing away from the HSR as it did when it
        was picked up.
    """
    extent = table_extent(world, KitchenTable.DESTINATION)
    return Pose.from_xyz_rpy(
        x=extent.max_x - EDGE_INSET,
        y=carried.goal_y,
        z=extent.max_z + carried.scale.z / 2 + RELEASE_CLEARANCE,
        yaw=math.pi,
        reference_frame=world.root,
    )


def picking_pose(world: World, carried: CarriedObject) -> Pose:
    """
    :return: Where the HSR stands to pick ``carried`` up: across the partition from it,
        facing the source table, with its hand rather than its base in line with it.
    """
    extent = table_extent(world, KitchenTable.SOURCE)
    return Pose.from_xyz_rpy(
        x=carried.start_x - hand_offset(world),
        y=extent.max_y + STANDING_DISTANCE,
        yaw=-math.pi / 2,
        reference_frame=world.root,
    )


def placing_pose(world: World, carried: CarriedObject) -> Pose:
    """
    :return: Where the HSR stands to set ``carried`` down: beside the destination
        table, facing it, with its hand rather than its base in line with the goal.
    """
    extent = table_extent(world, KitchenTable.DESTINATION)
    return Pose.from_xyz_rpy(
        x=extent.max_x + STANDING_DISTANCE,
        y=carried.goal_y + hand_offset(world),
        yaw=math.pi,
        reference_frame=world.root,
    )


def hand_offset(world: World) -> float:
    """
    :return: How far to its left of its base's centre the HSR carries its hand, in
        metres: the arm swings about a flex joint mounted beside the torso, and the
        base, which does not move while the HSR grasps, has to make up for it.
    """
    robot = world.get_semantic_annotations_by_type(HSRB)[0]
    flexed_link = world.get_connection_by_name(HSRBJoint.ARM_FLEX).child
    return float(world.compute_forward_kinematics_np(robot.root, flexed_link)[1, 3])


# %% building the world and the plan


def build_world() -> World:
    """
    :return: The kitchen with the HSR standing across the partition from the source
        table, and the milk and the cereal on that table.
    """
    world = WorldSpecification.from_urdf(
        KITCHEN_DESCRIPTION, robots=[RobotSpecification(HSRB)]
    ).to_domain_object()
    robot = world.get_semantic_annotations_by_type(HSRB)[0]
    robot.mobile_base.full_body_controlled = False
    robot.root.parent_connection.origin = picking_pose(
        world, CARRIED_OBJECTS[0]
    ).to_homogeneous_matrix()
    palm = robot.get_arms()[0].end_effector.root
    with world.modify_world():
        for carried in CARRIED_OBJECTS:
            _stand_on_the_source_table(world, carried, palm)
    return world


def _stand_on_the_source_table(
    world: World, carried: CarriedObject, palm: Body
) -> None:
    """
    Stand ``carried`` on the source table, free to be picked up, and declare the weld
    that holds it in the HSR's hand while it is carried (see
    :class:`HoldWhatTheHandGrips`).

    :param world: The world being built, already open for modification.
    :param carried: The object to stand there.
    :param palm: The HSR's palm, which holds the object while it is carried.
    """
    body = Body(name=PrefixedName(carried.name))
    shapes = ShapeCollection(
        [
            Box(
                origin=HomogeneousTransformationMatrix(reference_frame=body),
                scale=carried.scale,
                color=carried.color,
            )
        ],
        reference_frame=body,
    )
    body.collision = shapes
    body.visual = shapes
    world.add_kinematic_structure_entity(body)
    world.add_connection(
        Connection6DoF.create_with_dofs(
            world=world,
            parent=world.root,
            child=body,
            name=PrefixedName(f"{carried.name}_connection"),
            parent_T_connection_expression=start_pose(world, carried),
        )
    )
    ContactParameters.create_for_grasped_object().apply_to([body])
    world.add_semantic_annotations([carried.annotation_type(root=body)])
    world.add_simulator_property(
        MujocoEquality(
            type=mujoco.mjtEq.mjEQ_WELD,
            object_type=mujoco.mjtObj.mjOBJ_BODY,
            name_1=palm.name.name,
            name_2=body.name.name,
            data=[0.0] * WELD_DATA_LENGTH,
            active=False,
        )
    )


def build_plan(context: Context) -> Plan:
    """
    Carry every object from the source table to the destination table, one at a time.

    :param context: The context the actions are built in.
    :return: The plan.
    """
    world = context.world
    arm = context.robot.get_arms()[0]
    grasp = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, arm.end_effector
    )
    actions = [ParkArmsAction(Arms.BOTH)]
    for carried in CARRIED_OBJECTS:
        annotation = world.get_semantic_annotations_by_type(carried.annotation_type)[0]
        actions += [
            NavigateAction(picking_pose(world, carried)),
            PickUpAction(annotation, Arms.LEFT, grasp),
            NavigateAction(placing_pose(world, carried)),
            PlaceAction(annotation.root, goal_pose(world, carried), Arms.LEFT),
            ParkArmsAction(Arms.BOTH),
        ]
    return sequential(actions, context=context).plan


# %% carrying what the hand holds


WELD_DATA_LENGTH = 11
"""
How many numbers a MuJoCo weld's data holds: an anchor, a relative pose and a torque
scale.
"""


@dataclass
class HoldWhatTheHandGrips(PlanCallback):
    """
    Holds an object in the HSR's hand from the moment the hand closes on it in a pick
    up until it opens to set the object down.

    The HSR's base is moved by putting it somewhere else rather than by driving it
    there, which friction cannot carry a held object through; the weld does.
    """

    simulation: MujocoSim = field(kw_only=True)
    """
    The simulation the object is held in.
    """

    palm: Body = field(kw_only=True)
    """
    The HSR's palm, which holds the object.
    """

    def on_start(self, node: PlanNode) -> None:
        placing = self._gripper_motion_within(node, GripperState.OPEN, PlaceAction)
        if placing is None:
            return
        self.simulation.unfasten(placing.object_designator, self.palm)

    def on_end(self, node: PlanNode) -> None:
        picking = self._gripper_motion_within(node, GripperState.CLOSE, PickUpAction)
        if picking is None:
            return
        self.simulation.fasten(picking.object_designator.root, self.palm)

    @staticmethod
    def _gripper_motion_within(
        node: PlanNode, motion: GripperState, action_type: Type[ActionDescription]
    ) -> Optional[ActionDescription]:
        """
        :return: The action of ``action_type`` that ``node`` belongs to, if ``node``
            moves the gripper as ``motion`` says, else None.
        """
        if not isinstance(node, MotionNode):
            return None
        designator = node.designator
        if not isinstance(designator, MoveGripperMotion) or designator.motion != motion:
            return None
        for ancestor in node.path:
            if isinstance(ancestor, ActionNode) and isinstance(
                ancestor.designator, action_type
            ):
                return ancestor.designator
        return None


# %% running it in MuJoCo

STEP_SIZE = 1e-3
"""
How much simulated time one physics step covers, in seconds.
"""

SETTLING_DURATION = timedelta(seconds=1)
"""
How long the scene is left to settle before the HSR starts, so the objects rest on the
table rather than being grasped mid-fall.
"""

HEADLESS = False
"""
Whether the simulation runs without MuJoCo's own viewer.
"""

FRICTION_CONE = mujoco.mjtCone.mjCONE_ELLIPTIC
"""
How MuJoCo bounds a contact's friction: the elliptic cone is the exact one; the default
pyramid only approximates it and lets a squeezed object creep out of the hand.
"""

IMPEDANCE_RATIO = 10.0
"""
How much stiffer MuJoCo makes friction than the push along the contact normal: at the
default of 1 a held object slides out under its own weight despite a firm squeeze.
"""

NO_SLIP_ITERATIONS = 10
"""
How many passes MuJoCo spends removing the slip its soft contacts leave, so a held
object stays where the fingers hold it.
"""


def run(world: World, headless: bool = HEADLESS) -> None:
    """
    Carry out the plan against MuJoCo physics, offering the simulation to the
    visualization for control.

    :param world: The world :func:`build_world` built.
    :param headless: Whether to run without MuJoCo's viewer.
    """
    robot = world.get_semantic_annotations_by_type(HSRB)[0]
    context = Context(
        world=world,
        robot=robot,
        evaluate_conditions=False,
        update_world_model_attachment=False,
    )
    plan = build_plan(context)
    visualization = WorldVisualization.from_environment(world).start()
    visualization.attach_plan(plan)

    _resist_turning_in_the_hand(world)
    simulation = MujocoSim(
        world=world,
        headless=headless,
        step_size=STEP_SIZE,
        cone=FRICTION_CONE,
        impratio=IMPEDANCE_RATIO,
        noslip_iterations=NO_SLIP_ITERATIONS,
    )
    simulation.start_stepped_simulation()
    plan.node_callbacks.append(
        HoldWhatTheHandGrips(
            plan=plan,
            simulation=simulation,
            palm=robot.get_arms()[0].end_effector.root,
        )
    )
    controlled = ControlledSimulation(simulation)
    visualization.attach_simulation(controlled)
    previous_pacer = GiskardExecutable.simulation_pacer
    GiskardExecutable.simulation_pacer = SteppedSimulationPacer(controlled)
    try:
        controlled.step_simulation(SETTLING_DURATION)
        with ExecutionEnvironment(execution_type=ExecutionType.SIMULATED):
            plan.perform()
    finally:
        GiskardExecutable.simulation_pacer = previous_pacer
        simulation.stop_simulation()


def _resist_turning_in_the_hand(world: World) -> None:
    """
    Have MuJoCo resolve every carried object's twisting friction, so the object does
    not swing about the line the fingers hold it along.

    :param world: The world the objects stand in.
    """
    for carried in CARRIED_OBJECTS:
        for shape in world.get_body_by_name(carried.name).collision.shapes:
            shape.add_simulator_property(
                MujocoGeom(
                    contact_dimensionality=ContactDimensionality.SLIDING_AND_TWISTING
                )
            )


if __name__ == "__main__":
    run(build_world())
