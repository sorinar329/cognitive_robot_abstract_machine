"""
Finds the Montessori pieces on Tracy's table and answers queries with where they are.

Run it with the camera up::

    ros2 run robokudo main _ae=tracy_montessori

Every query is answered with one object designator per object standing on the table.
Each piece of the set is named in the designator's ``type`` (``square``, ``circle``,
``rectangle``, ``triangle``) and posed at its centre in the ``map`` frame; every other
object, such as the board, comes back without a type.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.montessori_pieces import MontessoriPieceAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import GenerateQueryResult, QueryAnnotator
from robokudo.annotators.shape_estimator import ShapeEstimatorAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline

# %% the pipeline


class AnalysisEngine(AnalysisEngineInterface):
    """
    Table-top segmentation tuned to the small pieces of the Montessori set, followed by
    shape fitting and naming of the pieces.
    """

    def name(self) -> str:
        return "tracy_montessori"

    def implementation(self) -> Pipeline:
        pipeline = Pipeline("TracyMontessoriPipeline")
        pipeline.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                CollectionReaderAnnotator(
                    descriptor=CollectionReaderDescriptorFactory.create_descriptor(
                        "orbbec"
                    )
                ),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(descriptor=self.table_crop()),
                PlaneAnnotator(descriptor=self.table_plane()),
                PointCloudClusterExtractor(descriptor=self.piece_clusters()),
                ShapeEstimatorAnnotator(),
                MontessoriPieceAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return pipeline

    @staticmethod
    def table_crop() -> PointcloudCropAnnotator.Descriptor:
        """
        :return: A crop to the stretch of Tracy's table in front of the robot, in the
            ``map`` frame, keeping the table's surface for the plane to be found in.
        """
        descriptor = PointcloudCropAnnotator.Descriptor()
        descriptor.parameters.relative_to_world = True
        descriptor.parameters.min_x = 0.3
        descriptor.parameters.max_x = 1.18
        descriptor.parameters.min_y = -0.75
        descriptor.parameters.max_y = 0.75
        descriptor.parameters.min_z = 0.85
        descriptor.parameters.max_z = 1.1
        return descriptor

    @staticmethod
    def table_plane() -> PlaneAnnotator.Descriptor:
        """
        :return: A plane fit tight enough to leave most of a 3 cm piece above it.
        """
        descriptor = PlaneAnnotator.Descriptor()
        descriptor.parameters.distance_threshold = 0.008
        return descriptor

    @staticmethod
    def piece_clusters() -> PointCloudClusterExtractor.Descriptor:
        """
        :return: Clustering that keeps clusters as small as a piece, the smallest of
            which covers a few hundred pixels.
        """
        descriptor = PointCloudClusterExtractor.Descriptor()
        descriptor.parameters.min_cluster_count = 100
        descriptor.parameters.dbscan_min_cluster_count = 30
        descriptor.parameters.eps = 0.02
        return descriptor
