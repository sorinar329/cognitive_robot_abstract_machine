"""Analysis engine for TIAGo robot perception pipeline.

This module provides an analysis engine that demonstrates perception capabilities
using the TIAGo robot's camera system. It implements a query-based pipeline for
tabletop segmentation and object pose estimation.

The pipeline implements the following functionality:

* Query-based perception control
* TIAGo camera data processing
* Point cloud analysis and segmentation
* Object pose estimation using PCA
* Query result generation and response

.. note::
    This engine is specifically designed for the TIAGo robot platform and uses
    its camera configuration. The pipeline can be extended with additional
    perception capabilities as needed.
"""

import robokudo.analysis_engine

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.cluster_pose_pca import ClusterPosePCAAnnotator
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.camera_viewpoint_visualizer import CameraViewpointVisualizer


import robokudo.descriptors.camera_configs.config_tiago

import robokudo.io.camera_interface
import robokudo.idioms
from robokudo.annotators.query import QueryReply, GenerateQueryResult


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    """Analysis engine for TIAGo robot perception.

    This class implements a pipeline that processes camera data from the TIAGo
    robot to perform tabletop segmentation and object pose estimation. It uses
    a query-based approach to control perception tasks.

    The pipeline includes:

    * Query handling for perception control
    * TIAGo camera data collection and preprocessing
    * Point cloud analysis and segmentation
    * Object pose estimation using PCA
    * Query result generation and response

    .. note::
        The pipeline uses PCA-based pose estimation by default, but can be
        configured to use bounding box-based estimation by uncommenting the
        relevant annotator.
    """

    def name(self):
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        :rtype: str
        """
        return "demo"

    def implementation(self):
        """Create a pipeline for TIAGo robot perception.

        This method constructs a processing pipeline that handles perception
        tasks for the TIAGo robot. The pipeline processes camera data to
        perform tabletop segmentation and object pose estimation.

        Pipeline execution sequence:

        1. Initialize pipeline
        2. Wait for query
        3. Read TIAGo camera data
        4. Preprocess image
        5. Crop point cloud
        6. Detect table plane
        7. Extract object clusters
        8. Estimate object poses (PCA)
        9. Generate and send query response

        :return: The configured pipeline for TIAGo perception
        :rtype: robokudo.pipeline.Pipeline
        """
        tiago_camera_config = robokudo.descriptors.camera_configs.config_tiago.CameraConfig()
        tiago_config = CollectionReaderAnnotator.Descriptor(
            camera_config=tiago_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(tiago_camera_config))

        # pc_crop_config = PointcloudCropAnnotator.Descriptor()
        # pc_crop_config.parameters.


        seq = robokudo.pipeline.Pipeline("ContPipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),

                robokudo.annotators.query.QueryAnnotator(),

                CollectionReaderAnnotator(descriptor=tiago_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                # ClusterPoseBBAnnotator(),
                ClusterPosePCAAnnotator(),

                GenerateQueryResult(),
                QueryReply(),
            ])
        return seq
