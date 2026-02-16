import robokudo.analysis_engine

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator

import robokudo.descriptors.camera_configs.config_realsense

import robokudo.io.camera_interface
import robokudo.idioms


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self):
        return "realsense"

    def implementation(self):
        """
        Create a basic pipeline that does tabletop segmentation
        """
        kinect_camera_config = robokudo.descriptors.camera_configs.config_realsense.CameraConfig()
        kinect_config = CollectionReaderAnnotator.Descriptor(
            camera_config=kinect_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(kinect_camera_config))

        seq = robokudo.pipeline.Pipeline("RealSensePipeline")
        seq.add_children(
            [
                # PipelineTrigger(),
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
            ])
        return seq
