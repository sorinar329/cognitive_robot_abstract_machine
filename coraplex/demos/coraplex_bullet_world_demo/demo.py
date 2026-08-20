from __future__ import annotations

import os
from dataclasses import dataclass
from typing_extensions import List, Optional, Tuple

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription

from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction

from coraplex.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Spoon,
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection

from live_segmind_perception import LiveSegmindPerception

# %% transport targets


@dataclass
class TransportTarget:
    """
    One object the demo's plan transports to a destination pose.
    """

    object_name: str
    """
    Name of the body to transport.
    """

    target_pose: Pose
    """
    Pose the object is placed at.
    """

    arm: Arms
    """
    Arm used to carry the object.
    """

    grasp_description: Optional[GraspDescription] = None
    """
    Grasp to use for picking the object up, if not left to the default.
    """


def build_transport_targets(world: World, pr2: PR2) -> List[TransportTarget]:
    """
    :param world: The world the objects and the robot live in.
    :param pr2: The robot performing the transports.
    :return: The milk, bowl and spoon transports the demo performs, in order.
    """
    return [
        TransportTarget(
            object_name="milk.stl",
            target_pose=Pose.from_xyz_rpy(
                4.9, 3.3, 0.8, yaw=1.57, reference_frame=world.root
            ),
            arm=Arms.LEFT,
        ),
        TransportTarget(
            object_name="bowl.stl",
            target_pose=Pose.from_xyz_rpy(
                5, 3.3, 0.75, yaw=1.57, reference_frame=world.root
            ),
            arm=Arms.LEFT,
        ),
        TransportTarget(
            object_name="spoon.stl",
            target_pose=Pose.from_xyz_rpy(
                5.1, 3.3, 0.75, yaw=1.57, reference_frame=world.root
            ),
            arm=Arms.LEFT,
            grasp_description=GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.TOP,
                pr2.left_arm.end_effector,
            ),
        ),
    ]


# %% world and plan setup


def setup_demo_world() -> Tuple[World, Context, PR2]:
    """
    :return: The apartment world (with spoon and bowl merged in), its context and
        the PR2 robot living in it.
    """
    world = setup_world()

    spoon = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "spoon.stl"
        )
    ).parse()
    bowl = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
        )
    ).parse()

    with world.modify_world():
        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.4, 2.2, 1, reference_frame=world.root
            ),
        )
        connection = FixedConnection(
            parent=world.get_body_by_name("cabinet10_drawer_top"),
            child=spoon.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.05, -0.05, 0
            ),
        )
        world.merge_world(spoon, connection)

    try:
        import rclpy

        rclpy.init()
        from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
            VizMarkerPublisher,
        )

        node = rclpy.create_node("viz_marker")
        VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
    except ImportError:
        node = None

    pr2 = PR2.from_world(world)
    context = Context(world=world, robot=pr2, _debug=False, ros_node=node)

    with world.modify_world():
        world_reasoner = WorldReasoner(world)
        world_reasoner.reason()
        world.add_semantic_annotations(
            [
                Bowl(root=world.get_body_by_name("bowl.stl")),
                Spoon(root=world.get_body_by_name("spoon.stl")),
            ]
        )
        world.add_semantic_annotation_recursively(
            Drawer(
                root=world.get_body_by_name("cabinet10_drawer_top"),
                handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
            )
        )

    context.evaluate_conditions = False

    return world, context, pr2


def build_plan(
    world: World, context: Context, pr2: PR2
) -> Tuple[Plan, List[TransportTarget]]:
    """
    :param world: The world the plan acts in.
    :param context: The context the plan is built with.
    :param pr2: The robot performing the transports.
    :return: The demo's plan and the transport targets it was built from.
    """
    targets = build_transport_targets(world, pr2)
    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            *[
                TransportAction(
                    world.get_body_by_name(target.object_name),
                    target.target_pose,
                    target.arm,
                    target.grasp_description,
                )
                for target in targets
            ],
        ],
        context=context,
    ).plan
    return plan, targets


# %% entry point


def main() -> None:
    world, context, pr2 = setup_demo_world()
    plan, targets = build_plan(world, context, pr2)
    tracked_objects = [world.get_body_by_name(target.object_name) for target in targets]

    with LiveSegmindPerception(world, tracked_objects) as logger:
        with simulated_robot:
            plan.perform()

    for event in logger.get_events():
        print(event)


if __name__ == "__main__":
    main()
