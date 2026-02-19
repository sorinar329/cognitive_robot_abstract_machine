import time
from datetime import timedelta

import rclpy

from segmind import set_logger_level, LogLevel, logger
from segmind.datastructures.events import LossOfSupportEvent, SupportEvent, PlacingEvent, LossOfContactEvent, \
    ContainmentEvent
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from segmind.detectors.atomic_event_detectors import ContactDetector, TranslationDetector
from segmind.detectors.coarse_event_detectors import PlacingDetector
from segmind.detectors.spatial_relation_detector import SupportDetector, ContainmentDetector
from segmind.episode_segmenter import NoAgentEpisodeSegmenter
from segmind.event_logger import EventLogger
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.reasoning.predicates import InsideOf, is_supported_by
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from Segmind.test import setup_spatial_world, setup_support_world
from semantic_digital_twin.world_description.world_entity import Body

set_logger_level(LogLevel.DEBUG)
class TestSpatialEvents:
    world: World
    viz_marker_publisher: VizMarkerPublisher

    def test_support_event(self):
        self.world = setup_support_world()
        self.visualize(self.world)
        self.logger = EventLogger()
        self.tracked_obj = self.world.get_body_by_name("cylinder_body")
        object_tracker = ObjectTrackerFactory.get_tracker(self.tracked_obj)
        table = self.world.get_body_by_name("table_body")
        cabinet = self.world.get_body_by_name("cabinet")
        time_between_frames = timedelta(seconds=0.01)
        translation_detector = self.run_and_get_translation_detector(self.tracked_obj, time_between_frames)
        support_detector = SupportDetector(logger=self.logger, world=self.world)
        support_detector.start()
        containment_detector = ContainmentDetector(logger=self.logger, world=self.world)
        containment_detector.start()

        try:
            assert object_tracker.get_latest_event_of_type(SupportEvent) is None
            assert object_tracker.get_latest_event_of_type(LossOfSupportEvent) is None
            assert object_tracker.get_latest_event_of_type(PlacingEvent) is None
            assert object_tracker.get_latest_event_of_type(ContainmentEvent) is None

            translation_detector.update_with_latest_motion_data()
            self.tracked_obj.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=table.global_pose.x, y=table.global_pose.y, z=table.global_pose.z + 0.2
            )
            translation_detector.update_with_latest_motion_data()
            translation_detector.update_with_latest_motion_data()
            translation_detector.update_with_latest_motion_data()
            time.sleep(translation_detector.get_n_changes_wait_time(2))

            assert object_tracker.get_latest_event_of_type(SupportEvent) is not None

            placing_detector = PlacingDetector(logger=self.logger, starter_event=object_tracker.get_latest_event_of_type(SupportEvent))
            placing_detector.start()
            time.sleep(2)

            assert object_tracker.get_latest_event_of_type(PlacingEvent) is not None

            self.tracked_obj.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=cabinet.global_pose.x, y=cabinet.global_pose.y, z=cabinet.global_pose.z
            )
            translation_detector.update_with_latest_motion_data()
            translation_detector.update_with_latest_motion_data()
            translation_detector.update_with_latest_motion_data()
            time.sleep(translation_detector.get_n_changes_wait_time(2))

            assert object_tracker.get_latest_event_of_type(LossOfSupportEvent) is not None
            assert object_tracker.get_latest_event_of_type(ContainmentEvent) is not None

            placing_detector.stop()
        except Exception as e:
            raise e

        finally:
            translation_detector.stop()
            support_detector.stop()
            containment_detector.stop()
            translation_detector.join()
            support_detector.join()
            containment_detector.join()


    @staticmethod
    def run_and_get_translation_detector(obj: Body, time_between_frames: timedelta = timedelta(seconds=0.01))\
            -> TranslationDetector:
        logger = EventLogger()
        translation_detector = TranslationDetector(logger, obj,
                                                   time_between_frames=time_between_frames,
                                                   window_size=2)
        translation_detector.start()
        # wait one timestep to detect the initial state
        time.sleep(translation_detector.get_n_changes_wait_time(1))
        return translation_detector


    def visualize(self, world):
        logger.debug("Starting Visualization")
        rclpy.init()
        self.node = rclpy.create_node("test_node")
        self.world = world
        logger.debug("Node created")
        self.viz_marker_publisher = VizMarkerPublisher(world=self.world, node=self.node)
        self.viz_marker_publisher.with_tf_publisher()