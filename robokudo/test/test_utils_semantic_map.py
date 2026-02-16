import numpy as np
import pytest

from robokudo.semantic_map import SemanticMapEntry
from robokudo.utils.semantic_map import get_obb_from_semantic_map_region, \
    get_obb_from_semantic_map_region_in_cam_coordinates, get_obb_from_semantic_map_region_with_transform_matrix


class TestUtilsSemanticMap:
    @pytest.fixture
    def region(self) -> SemanticMapEntry:
        region = SemanticMapEntry()
        region.x_size = 1.0
        region.y_size = 2.0
        region.z_size = 3.0
        region.orientation_x = 0.5
        region.orientation_y = 0.5
        region.orientation_z = 0.5
        region.orientation_w = 0.5
        region.position_x = 3.0
        region.position_y = 2.0
        region.position_z = 1.0
        return region

    def test_get_obb_from_semantic_map_region(self, region: SemanticMapEntry):
        obb = get_obb_from_semantic_map_region(region)

        assert np.all(obb.extent == [1.0, 2.0, 3.0])
        assert np.all(obb.center == [region.position_x, region.position_y, region.position_z])
        assert np.allclose(obb.R, [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

    def test_get_obb_from_semantic_map_region_in_cam_coordinates_camera_frame_region(self, region: SemanticMapEntry):
        region.frame_id = "camera_link"
        obb = get_obb_from_semantic_map_region_in_cam_coordinates(region, world_frame_name="map",
                                                                  world_to_cam_transform_matrix=np.eye(3))

        assert np.all(obb.extent == [1.0, 2.0, 3.0])
        assert np.all(obb.center == [region.position_x, region.position_y, region.position_z])
        assert np.allclose(obb.R, [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

    def test_get_obb_from_semantic_map_region_in_cam_coordinates_world_frame_region(self, region: SemanticMapEntry):
        region.frame_id = "map"

        world_to_cam = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0, 2.0],
            [0.0, -1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        obb = get_obb_from_semantic_map_region_in_cam_coordinates(region, world_frame_name="map",
                                                                  world_to_cam_transform_matrix=world_to_cam)

        assert np.all(obb.extent == [1.0, 2.0, 3.0])
        assert np.all(obb.center == [region.position_x - 1.0, region.position_y - 3.0, region.position_z])
        assert np.allclose(obb.R, [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0]
        ])

    def test_get_obb_from_semantic_map_region_with_transform_matrix(self, region: SemanticMapEntry):
        world_to_cam = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0, 2.0],
            [0.0, -1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        obb = get_obb_from_semantic_map_region_with_transform_matrix(region, transform_matrix=world_to_cam)

        assert np.all(obb.extent == [1.0, 2.0, 3.0])
        assert np.all(obb.center == [region.position_x - 1.0, region.position_y - 3.0, region.position_z])
        assert np.allclose(obb.R, [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0]
        ])
