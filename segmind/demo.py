import os

import numpy as np

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def _simple_apartment_setup():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)

        box = Body(
            name=PrefixedName("box"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, 0.95))]),
            visual=ShapeCollection([Box(scale=Scale(1, 1, 0.95))]),
        )

        box_2 = Body(
            name=PrefixedName("box_2"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, 0.95))]),
            visual=ShapeCollection([Box(scale=Scale(1, 1, 0.95))]),
        )

        box_1_connection = FixedConnection(
            parent=world.root,
            child=box,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                2, 0, 0.375, reference_frame=world.root
            ),
        )
        box_2_connection = FixedConnection(
            parent=root,
            child=box_2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -2, 0, 0.375
            ),
        )

        wall1 = Body(
            name=PrefixedName("wall_1"),
            collision=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
            visual=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
        )
        wall2 = Body(
            name=PrefixedName("wall_2"),
            collision=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
            visual=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
        )
        wall3 = Body(
            name=PrefixedName("wall_3"),
            collision=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
            visual=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
        )
        wall4 = Body(
            name=PrefixedName("wall_4"),
            collision=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
            visual=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
        )

        wall_1_connection = FixedConnection(
            parent=root,
            child=wall1,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                0, -4, 1
            ),
        )
        wall_2_connection = FixedConnection(
            parent=root,
            child=wall2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                0, 4, 1
            ),
        )
        wall_3_connection = FixedConnection(
            parent=root,
            child=wall3,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -4, 0, 1, yaw=np.pi / 2
            ),
        )
        wall_4_connection = FixedConnection(
            parent=root,
            child=wall4,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                4, 0, 1, yaw=np.pi / 2
            ),
        )

        world.add_connection(box_1_connection)
        world.add_connection(box_2_connection)
        world.add_connection(wall_1_connection)
        world.add_connection(wall_2_connection)
        world.add_connection(wall_3_connection)
        world.add_connection(wall_4_connection)

    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "coraplex",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()
    world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 0.94, yaw=np.pi),
    )
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box_2.global_pose.x, box_2.global_pose.y,
                                                                                 box_2.global_pose.z + 0.56)
    return world




try:
    import rclpy
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
    rclpy.init()
    node = rclpy.create_node("viz_marker")
    v = VizMarkerPublisher(_world=_simple_apartment_setup(), node=node).with_tf_publisher()
except ImportError:
    node = None