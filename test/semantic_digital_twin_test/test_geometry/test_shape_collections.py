import numpy as np

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale, Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def test_post_init_transformation():
    w = World()
    root = Body(name=PrefixedName("root"))
    b1 = Body(name=PrefixedName("b1"))

    with w.modify_world():
        w.add_connection(
            FixedConnection(
                parent=root,
                child=b1,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, reference_frame=root
                ),
            )
        )

    shape = Sphere(
        radius=1,
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=3, reference_frame=root),
    )
    shape_collection = ShapeCollection(
        shapes=[shape],
        reference_frame=b1,
    )
    shape_collection.transform_all_shapes_to_own_frame()
    assert shape.origin.reference_frame == b1
    assert shape.origin.to_position().x == 2.0

    shape = Sphere(
        radius=1,
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=3, reference_frame=root),
    )

    shape_collection = ShapeCollection(reference_frame=b1)
    shape_collection.append(shape)
    shape_collection.transform_all_shapes_to_own_frame()
    assert shape.origin.reference_frame == b1
    assert shape.origin.to_position().x == 2.0


# %% the bounds of a collection placed in the world

ROUNDING_TOLERANCE = 1e-9
"""
How far apart two computations of one placed point may land through rounding alone.
"""


def _body_holding_a_box(yaw: float) -> Body:
    """
    :return: A body two metres along x from the root, turned by ``yaw`` about z, holding
        a box that is offset from the body and longer on each axis than the last.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    body = Body(name=PrefixedName("body"))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2, yaw=yaw, reference_frame=root
                ),
            )
        )
        body.collision.append(
            Box(
                scale=Scale(0.1, 0.2, 0.3),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=0.5, reference_frame=body
                ),
            )
        )
    return body


def _placed_vertices(body: Body) -> np.ndarray:
    """
    :return: The vertices of ``body``'s collision geometry where the world places it.
    """
    mesh = body.collision.combined_mesh.copy()
    mesh.apply_transform(body.global_transform.to_np())
    return mesh.vertices


def test_bounds_in_the_root_frame_fit_a_collection_turned_by_a_right_angle():
    body = _body_holding_a_box(yaw=np.pi / 2)
    vertices = _placed_vertices(body)

    bounds = body.collision.bounds_in_root_frame()

    np.testing.assert_allclose(
        bounds.lower, vertices.min(axis=0), atol=ROUNDING_TOLERANCE
    )
    np.testing.assert_allclose(
        bounds.upper, vertices.max(axis=0), atol=ROUNDING_TOLERANCE
    )


def test_bounds_in_the_root_frame_hold_every_point_of_a_turned_collection():
    body = _body_holding_a_box(yaw=np.pi / 5)
    vertices = _placed_vertices(body)

    bounds = body.collision.bounds_in_root_frame()

    assert np.all(bounds.lower - ROUNDING_TOLERANCE <= vertices)
    assert np.all(vertices <= bounds.upper + ROUNDING_TOLERANCE)
