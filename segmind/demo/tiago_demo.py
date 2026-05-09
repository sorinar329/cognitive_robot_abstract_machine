"""
Segmind Episode Segmentation Demo
==================================
Runs a CSV-based episode replay through the Segmind statechart pipeline
and prints the detected events to stdout.

Usage:
    python segmind_demo.py

Dependencies:
    rclpy, giskardpy, segmind, semantic_digital_twin
"""

import rclpy

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    LossOfContactDetector, TranslationDetector, StopTranslationDetector,
)
from segmind.detectors.base import SegmindContext
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector, LossOfSupportDetector
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.players.csv_player import CSVEpisodePlayer
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher


# ---------------------------------------------------------------------------
# Configuration — adjust paths to match your local setup
# ---------------------------------------------------------------------------

SCENE_PATH = (
    "/home/sorin/dev/workspace/cognitive_robot_abstract_machine"
    "/segmind/resources/tiago_episodes/models/assets/mjcf"
    "/iai_tiago_velocity_in_apartment_with_multiverse.xml"
)

CSV_PATH = (
    "/home/sorin/dev/workspace/cognitive_robot_abstract_machine"
    "/segmind/resources/tiago_episodes/data/data.csv"
)

IGNORED_OBJECTS = ["iCub"]
FIXED_OBJECTS = ["scene"]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def build_world(scene_path: str):
    from semantic_digital_twin.world_description.world_entity import Body, KinematicStructureEntity, Region

    for cls in [Body, KinematicStructureEntity, Region]:
        orig = cls.center_of_mass.fget

        def make_patched(orig, cls_name):
            def patched(self):
                print(f"center_of_mass: {cls_name}.{self.name}")
                return orig(self)

            return patched

        cls.center_of_mass = property(make_patched(orig, cls.__name__))

    return MJCFParser(scene_path).parse()

def build_episode_executor(world, csv_path: str) -> tuple:
    """
    Create the CSV player, motion context, and episode executor.

    Returns:
        (file_player, context, episode_executor)
    """
    file_player = CSVEpisodePlayer(file_path=csv_path, world=world)
    context = MotionStatechartContext(world=world)
    episode_executor = EpisodeSegmenterExecutor(
        context=context,
        player=file_player,
        ignored_objects=IGNORED_OBJECTS,
        fixed_objects=FIXED_OBJECTS,
    )
    return file_player, context, episode_executor


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo():
    """Run the full episode segmentation demo and print detected events."""
    print("=== Segmind Demo ===\n")

    # Parse scene
    print(f"Loading scene: {SCENE_PATH}")
    world = build_world(SCENE_PATH)

    # Build pipeline components
    print(f"Loading episode data: {CSV_PATH}")
    file_player, context, episode_executor = build_episode_executor(world, CSV_PATH)

    # Initialise ROS2 and set up visualisation
    rclpy.init()
    node = rclpy.create_node("segmind_demo")
    viz_marker_publisher = VizMarkerPublisher(_world=world, node=node)
    viz_marker_publisher.with_tf_publisher()

    # Build and compile the statechart with desired detectors
    print("Compiling statechart ...\n")
    statechart = SegmindStatechart()
    contact_node = ContactDetector(excluded_objects=["base_footprint", "bowl"])
    loss_of_contact_node = LossOfContactDetector(excluded_objects=["base_footprint", "bowl"])
    support_node = SupportDetector(excluded_objects=["base_footprint", "bowl"])
    loss_of_support_node = LossOfSupportDetector(excluded_objects=["base_footprint", "bowl"])
    translation_node = TranslationDetector(excluded_objects=["base_footprint", "bowl"])
    stop_translation_node = StopTranslationDetector(excluded_objects=["base_footprint", "bowl"])
    statechart.add_node(contact_node)
    statechart.add_node(loss_of_contact_node)
    # statechart.add_node(support_node)
    # statechart.add_node(loss_of_support_node)
    statechart.add_node(translation_node)
    statechart.add_node(stop_translation_node)

    # support_node.start_condition = contact_node.observation_variable
    # loss_of_support_node.start_condition = loss_of_contact_node.observation_variable


    segmind_context = episode_executor.context.require_extension(SegmindContext)
    episode_executor.compile(statechart)

    # Replay episode frame by frame
    print("Replaying episode frames ...")
    frame_count = 0
    for frame_data in file_player.frame_data_generator:
        if frame_count % 1 == 0:
            file_player.process_objects_data(frame_data)
            episode_executor.tick()
        frame_count += 1

    print(f"Processed {frame_count} frames.\n")

    # Print results
    events = segmind_context.logger.get_events()
    print("=== Detected Events ===")
    for event in events:
        print(event)

    rclpy.shutdown()


if __name__ == "__main__":
    run_demo()