import sys

import numpy as np
import open3d as o3d
import pytest

from robokudo.annotators.static_object_detector import StaticObjectDetectorAnnotator
from robokudo.object_knowledge_base import ObjectKnowledge
from robokudo.utils.knowledge import get_quaternion_from_rotation_information, \
    get_transform_matrix_from_object_knowledge, get_bb_size_from_object_knowledge, load_object_knowledge_base, \
    get_obb_for_object_and_transform, get_obb_for_child_object_and_transform, get_obbs_for_object_and_childs


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

    def test_load_object_knowledge_base(self):
        ann = StaticObjectDetectorAnnotator()
        assert load_object_knowledge_base(ann)

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
