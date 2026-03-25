import sys
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
import pytest

from robokudo.annotators.static_object_detector import StaticObjectDetectorAnnotator
from robokudo.defs import Region3DWithName
from robokudo.utils.knowledge import get_quaternion_from_rotation_information, \
    get_transform_matrix_from_object_knowledge, get_bb_size_from_object_knowledge, load_world_descriptor, \
    get_obb_for_object_and_transform, get_obb_for_child_object_and_transform, get_obbs_for_object_and_childs
from robokudo.world_descriptor import BaseWorldDescriptor, PredefinedObject
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class ObjectKnowledge(Region3DWithName):
    components: list[object] = field(default_factory=list)
    features: list[object] = field(default_factory=list)
    mesh_ros_package: str = ""
    mesh_relative_path: str = ""

    def is_frame_in_camera_coordinates(self) -> bool:
        return self.frame is None or self.frame == ""


class TestUtilsKnowledge(object):

    def test_get_quaternion_from_rotation_information_unknown_pose_type(self):
        object_knowledge = ObjectKnowledge()
        object_knowledge.pose_type = "unknown_pose_type"
        assert pytest.raises(Exception, get_quaternion_from_rotation_information, object_knowledge)

    @pytest.mark.parametrize(["euler_angles", "quat"], [
        ([0.0, 0.0, 0.0], (0.0, 0.0, 0.0, 1.0)),
        ([90.0, 90.0, 90.0], (-0.1455344, 0.615170, -0.145534, 0.761055)),
    ])
    def test_get_quaternion_from_rotation_information_euler_rotation(self, euler_angles: tuple[float, float, float],
                                                                     quat: tuple[float, float, float, float]):
        object_knowledge = ObjectKnowledge()
        object_knowledge.pose_type = object_knowledge.PoseType.EULER
        object_knowledge.orientation_x = euler_angles[0]
        object_knowledge.orientation_y = euler_angles[1]
        object_knowledge.orientation_z = euler_angles[2]

        result = get_quaternion_from_rotation_information(object_knowledge)

        assert np.allclose(result, quat)

    @pytest.mark.parametrize("quat", [
        (0.0, 0.0, 0.0, 1.0),
        (-0.1455344, 0.615170, -0.145534, 0.761055)
    ])
    def test_get_quaternion_from_rotation_information_quaternion_rotation(self,
                                                                          quat: tuple[float, float, float, float]):
        object_knowledge = ObjectKnowledge()
        object_knowledge.pose_type = object_knowledge.PoseType.QUATERNION
        object_knowledge.orientation_x = quat[0]
        object_knowledge.orientation_y = quat[1]
        object_knowledge.orientation_z = quat[2]
        object_knowledge.orientation_w = quat[3]

        result = get_quaternion_from_rotation_information(object_knowledge)

        assert np.all(result == quat)

    @pytest.mark.parametrize(["euler_angles", "rot_matrix"], [
        ([0.0, 0.0, 0.0], np.eye(3)),
        ([90.0, 90.0, 90.0], [
            [0.200769, 0.042462, 0.978717],
            [-0.400576, 0.915278, 0.042462],
            [-0.893996, -0.400576, 0.200769]
        ]),
    ])
    def test_get_transform_matrix_from_object_knowledge_euler_rotation(self, euler_angles: tuple[float, float, float],
                                                                       rot_matrix: list[list[float]]):
        object_knowledge = ObjectKnowledge()
        object_knowledge.pose_type = object_knowledge.PoseType.EULER
        object_knowledge.orientation_x = euler_angles[0]
        object_knowledge.orientation_y = euler_angles[1]
        object_knowledge.orientation_z = euler_angles[2]
        object_knowledge.position_x = np.random.random()
        object_knowledge.position_y = np.random.random()
        object_knowledge.position_z = np.random.random()

        result = get_transform_matrix_from_object_knowledge(object_knowledge)

        assert np.allclose(result[:3, :3], rot_matrix)
        assert np.all(
            result[:3, 3] == [object_knowledge.position_x, object_knowledge.position_y, object_knowledge.position_z])

    @pytest.mark.parametrize(["quat", "rot_matrix"], [
        ([0.0, 0.0, 0.0, 1.0], np.eye(3)),
        ([-0.1455344, 0.615170, -0.145534, 0.761055], [
            [0.200770, 0.042461, 0.978717],
            [-0.400575, 0.915279, 0.042463],
            [-0.893996, -0.400575, 0.200770]
        ])
    ])
    def test_get_transform_matrix_from_object_knowledge_quaternion_rotation(self,
                                                                            quat: tuple[float, float, float, float],
                                                                            rot_matrix: list[list[float]]):
        object_knowledge = ObjectKnowledge()
        object_knowledge.pose_type = object_knowledge.PoseType.QUATERNION
        object_knowledge.orientation_x = quat[0]
        object_knowledge.orientation_y = quat[1]
        object_knowledge.orientation_z = quat[2]
        object_knowledge.orientation_w = quat[3]
        object_knowledge.position_x = np.random.random()
        object_knowledge.position_y = np.random.random()
        object_knowledge.position_z = np.random.random()

        result = get_transform_matrix_from_object_knowledge(object_knowledge)

        assert np.allclose(result[:3, :3], rot_matrix, atol=1e-6)
        assert np.all(
            result[:3, 3] == [object_knowledge.position_x, object_knowledge.position_y, object_knowledge.position_z])

    def test_get_bb_size_from_object_knowledge(self):
        object_knowledge = ObjectKnowledge()
        object_knowledge.x_size = np.random.randint(sys.maxsize)
        object_knowledge.y_size = np.random.randint(sys.maxsize)
        object_knowledge.z_size = np.random.randint(sys.maxsize)

        bb_size = get_bb_size_from_object_knowledge(object_knowledge)

        assert np.all(bb_size == [object_knowledge.x_size, object_knowledge.y_size, object_knowledge.z_size])

    def test_load_world_descriptor(self):
        ann = StaticObjectDetectorAnnotator()
        assert load_world_descriptor(ann)

    def test_get_obb_for_object_and_transform(self):
        object_knowledge = ObjectKnowledge()
        object_knowledge.x_size = 1.0
        object_knowledge.y_size = 1.0
        object_knowledge.z_size = 1.0

        # Rotated 90 degrees around Z-axis + translation on all axis
        transform = np.array([
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        obb = get_obb_for_object_and_transform(object_knowledge, transform)

        assert isinstance(obb, o3d.geometry.OrientedBoundingBox)
        assert np.allclose(obb.center, [1.0, 2.0, 3.0])
        assert np.allclose(obb.R, transform[:3, :3])
        assert np.allclose(obb.extent, [1.0, 1.0, 1.0])

    def test_get_obb_for_child_object_and_transform(self):
        object_knowledge = ObjectKnowledge()
        object_knowledge.x_size = 1.0
        object_knowledge.y_size = 1.0
        object_knowledge.z_size = 1.0

        object_knowledge.position_x = 3.0
        object_knowledge.position_y = 2.0
        object_knowledge.position_z = 1.0

        # Rotated 90-Degree around Z-Axis
        object_knowledge.orientation_x = 0.0
        object_knowledge.orientation_y = 0.0
        object_knowledge.orientation_z = 0.707106
        object_knowledge.orientation_w = 0.707106

        # Rotated 90 degrees around Z-Axis, Translation on all Axis
        transform = np.array([
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        obb = get_obb_for_child_object_and_transform(object_knowledge, transform)

        # 180 Degree rotation matrix should be the result
        new_rot = transform[:3, :3] @ transform[:3, :3]

        assert isinstance(obb, o3d.geometry.OrientedBoundingBox)
        assert np.allclose(obb.center, [-1.0, 5.0, 4.0]), "translations should be combined"
        assert np.allclose(obb.R, new_rot), "rotation matrices should be combined"
        assert np.allclose(obb.extent, [1.0, 1.0, 1.0]), "obb extends should be the same"

    def test_get_obbs_for_object_and_childs(self):
        feature = ObjectKnowledge()
        feature.name = "test_feature"

        component = ObjectKnowledge()
        component.name = "test_component"

        object_knowledge = ObjectKnowledge()
        object_knowledge.name = "test_object"
        object_knowledge.components = [component]
        object_knowledge.features = [feature]

        # Rotated 90 degrees around Z-Axis, Translation on all Axis
        transform = np.array([
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        obbs = get_obbs_for_object_and_childs(object_knowledge, transform)

        assert len(obbs) == 3

        assert object_knowledge.name in obbs
        assert feature.name in obbs
        assert component.name in obbs

        for obb in obbs.values():
            assert isinstance(obb, o3d.geometry.OrientedBoundingBox)

    def test_get_predefined_objects(self):
        empty_knowledge_base = BaseWorldDescriptor()
        test_world = empty_knowledge_base.world
        root = test_world.root

        foobar1_shape = Box(scale=Scale(0.10, 0.06, 0.05), color=Color(0.1, 0.2, 0.8, 1.0))
        foobar1_body = Body(
            name=PrefixedName(name="foobar", prefix="transform_example"),
            visual=ShapeCollection([foobar1_shape]),
            collision=ShapeCollection([foobar1_shape]),
        )

        foobar2_shape = Box(scale=Scale(0.10, 0.06, 0.05), color=Color(0.1, 0.2, 0.8, 1.0))
        foobar2_body = Body(
            name=PrefixedName(name="foobar2", prefix="transform_example"),
            visual=ShapeCollection([foobar2_shape]),
            collision=ShapeCollection([foobar2_shape]),
        )

        with test_world.modify_world():
            result_world_C_foobar1 = Connection6DoF.create_with_dofs(parent=root, child=foobar1_body, world=test_world)
            result_world_C_foobar2 = Connection6DoF.create_with_dofs(parent=root, child=foobar2_body, world=test_world)
            test_world.add_connection(result_world_C_foobar1)
            test_world.add_connection(result_world_C_foobar2)
            test_world.add_semantic_annotation(PredefinedObject(body=foobar1_body))
            test_world.add_semantic_annotation(PredefinedObject(body=foobar2_body))

        # Set origins in a separate modification block so FK is compiled first
        with test_world.modify_world():
            result_world_C_foobar1.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=0.5, reference_frame=root)
            result_world_C_foobar2.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1.2, reference_frame=root)

        bodies = empty_knowledge_base.get_predefined_object_bodies()
        assert len(bodies) == 2
        assert set(bodies) == {foobar1_body, foobar2_body}
        
