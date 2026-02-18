from semantic_digital_twin.collision_checking.collision_detector import CollisionCheck
from semantic_digital_twin.collision_checking.trimesh_collision_detector import TrimeshCollisionDetector
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection, Connection6DoF
from semantic_digital_twin.world_description.geometry import Cylinder, Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body



    
def setup_contact_world():
    world = World()
    root = Body(name=PrefixedName("root"))

    cylinder = Cylinder(width=0.50, height=0.5)
    shape_geometry = ShapeCollection([cylinder])
    cylinder_body = Body(
        name=PrefixedName("cylinder_body"),
        collision=shape_geometry,
        visual=shape_geometry,
    )

    box1 = Box(scale=Scale(0.60, 0.5, 0.5))
    shape_geometry = ShapeCollection([box1])
    box_body1 = Body(
        name=PrefixedName("box_body1"),
        collision=shape_geometry,
        visual=shape_geometry,
    )

    box2 = Box(scale=Scale(0.60, 0.5, 0.5))
    shape_geometry = ShapeCollection([box2])
    box_body2 = Body(
        name=PrefixedName("box_body2"),
        collision=shape_geometry,
        visual=shape_geometry,
    )

    with world.modify_world():
        root_C_box1 = Connection6DoF.create_with_dofs(
            world=world,
            parent=root,
            child=box_body1,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.5, y=-3.5, z=0.25
            ),
        )

        root_C_box2 = Connection6DoF.create_with_dofs(
            world=world,
            parent=root,
            child=box_body2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.5, y=-3.5, z=0.25
            )
        )

        root_C_cylinder = Connection6DoF.create_with_dofs(
            world=world,
            parent=root,
            child=cylinder_body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=-3, z=0.25
            ),
        )

        world.add_kinematic_structure_entity(root)
        world.add_connection(root_C_cylinder)
        world.add_connection(root_C_box2)
        world.add_connection(root_C_box1)

    return world