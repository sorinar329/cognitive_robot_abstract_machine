"""
Finds the Montessori pieces on Tracy's table and answers queries with where they are.

Run it with the camera up::

    python -m robokudo.scripts.main _ae=tracy_montessori

Every query is answered with one object designator per cyan or yellow blob in view;
each piece is named in its ``type`` and posed at its centre in ``map``.
"""

from semantic_digital_twin.world_description.geometry import Color

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.footprint_shape import FootprintShapeAnnotator
from robokudo.annotators.image_cluster_extractor import ImageClusterExtractor
from robokudo.annotators.montessori_pieces import MontessoriPieceAnnotator
from robokudo.annotators.query import GenerateQueryResult, QueryAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline

# %% the pipeline


class AnalysisEngine(AnalysisEngineInterface):
    """
    Finds the cyan and yellow pieces of the Montessori set in the colour image and names
    each by its colour and the shape of its footprint on the table.
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
                ImageClusterExtractor(
                    name="CyanPieces",
                    descriptor=self.coloured_pieces(lowest_hue=130, highest_hue=146),
                ),
                ImageClusterExtractor(
                    name="YellowPieces",
                    descriptor=self.coloured_pieces(lowest_hue=30, highest_hue=45),
                ),
                FootprintShapeAnnotator(descriptor=self.footprints_on_the_table()),
                MontessoriPieceAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return pipeline

    @staticmethod
    def coloured_pieces(
        lowest_hue: int, highest_hue: int
    ) -> ImageClusterExtractor.Descriptor:
        """
        :param lowest_hue: The lowest OpenCV full-range hue (0 to 255) of the pieces.
        :param highest_hue: The highest OpenCV full-range hue of the pieces.
        :return: Blobs of saturated pixels in that hue range, as large as a piece seen
            from Tracy's camera, up to all four pieces of the set.
        """
        descriptor = ImageClusterExtractor.Descriptor()
        descriptor.parameters.minimum_hue_saturation_value = (
            Color.from_opencv_hue_saturation_value_full(
                hue=lowest_hue, saturation=110, value=60
            )
        )
        descriptor.parameters.maximum_hue_saturation_value = (
            Color.from_opencv_hue_saturation_value_full(
                hue=highest_hue, saturation=255, value=255
            )
        )
        descriptor.parameters.contour_min_size = 300
        descriptor.parameters.num_of_objects = 4
        return descriptor

    @staticmethod
    def footprints_on_the_table() -> FootprintShapeAnnotator.Descriptor:
        """
        :return: Footprints of 3 cm tall pieces on Tracy's table.
        """
        descriptor = FootprintShapeAnnotator.Descriptor()
        descriptor.parameters.table_height = 0.88
        descriptor.parameters.object_height = 0.03
        return descriptor
