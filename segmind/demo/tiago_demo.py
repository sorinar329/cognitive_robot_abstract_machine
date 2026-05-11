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
from krrood.symbolic_math.symbolic_math import trinary_logic_not
from segmind.detectors.agent_event_detectors_nodes import HoldingDetector, LossOfHoldingDetector, LiftingDetector, \
    OpeningDetector
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    LossOfContactDetector, TranslationDetector, StopTranslationDetector,
)
from segmind.detectors.base import SegmindContext
from segmind.detectors.coarse_event_detector_nodes import PlacingDetector, PickUpDetector
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
    contact_node = ContactDetector(excluded_objects=["base_footprint", "bowl",
                                                     "gripper_right_left_inner_knuckle", "gripper_right_right_inner_knuckle",
                                                     "gripper_left_left_inner_knuckle", "gripper_left_right_inner_knuckle"],
                                   tracked_object=[
                                                  world.get_body_by_name("milk_box"),
                                   world.get_body_by_name("fridge_door1_handle")])

    loss_of_contact_node = LossOfContactDetector(excluded_objects=["base_footprint", "bowl"])
    support_node = SupportDetector(excluded_objects=["base_footprint", "bowl"])
    loss_of_support_node = LossOfSupportDetector(excluded_objects=["base_footprint", "bowl"])
    translation_node = TranslationDetector(excluded_objects=["base_footprint", "bowl"])
    stop_translation_node = StopTranslationDetector(excluded_objects=["base_footprint", "bowl"])
    placing_node = PlacingDetector(excluded_objects=["base_footprint", "bowl"])
    pickup_node = PickUpDetector(excluded_objects=["base_footprint", "bowl"])
    holding_node = HoldingDetector(excluded_objects=["base_footprint", "bowl"],
                                   tracked_object=[world.get_body_by_name("milk_box"),
                                                   world.get_body_by_name("fridge_door1_handle")],
                                   gripper_groups=[["gripper_right_left_inner_finger", "gripper_right_right_inner_finger"],
                                                   ["gripper_left_left_inner_finger", "gripper_left_right_inner_finger"]])

    loss_of_holding_node = LossOfHoldingDetector(excluded_objects=["base_footprint", "bowl"],
                                                 tracked_object=[world.get_body_by_name("milk_box"),
                                                                 world.get_body_by_name("fridge_door1_handle")],
                                                 gripper_groups=[["gripper_right_left_inner_finger",
                                                                  "gripper_right_right_inner_finger"],
                                                                 ["gripper_left_left_inner_finger",
                                                                   "gripper_left_right_inner_finger"]])

    lifting_node = LiftingDetector(excluded_objects=["base_footprint", "bowl"])

    opening_detector = OpeningDetector(
        handle_name="fridge_door1_handle",
        gripper_groups=[
            ["gripper_left_right_inner_finger", "gripper_left_left_inner_finger",
             "gripper_left_right_inner_knuckle", "gripper_left_left_inner_knuckle"]
        ],
        observation_window=10,
        distance_threshold=0.01,
        angle_threshold=0.3,
        excluded_objects=["base_footprint", "bowl"],
    )

    statechart.add_node(contact_node)
    statechart.add_node(loss_of_contact_node)
    statechart.add_node(support_node)
    statechart.add_node(loss_of_support_node)
    statechart.add_node(translation_node)
    statechart.add_node(stop_translation_node)
    statechart.add_node(placing_node)
    statechart.add_node(pickup_node)
    statechart.add_node(holding_node)
    statechart.add_node(loss_of_holding_node)
    statechart.add_node(lifting_node)
    statechart.add_node(opening_detector)



    support_node.start_condition = contact_node.observation_variable
    support_node.end_condition = trinary_logic_not(contact_node.observation_variable)
    loss_of_support_node.start_condition = loss_of_contact_node.observation_variable
    loss_of_support_node.end_condition = trinary_logic_not(loss_of_contact_node.observation_variable)
    placing_node.start_condition = support_node.observation_variable
    placing_node.end_condition = trinary_logic_not(support_node.observation_variable)
    pickup_node.start_condition = loss_of_support_node.observation_variable
    pickup_node.end_condition = trinary_logic_not(loss_of_support_node.observation_variable)
    #holding_node.start_condition = contact_node.observation_variable
    #holding_node.end_condition = trinary_logic_not(contact_node.observation_variable)
    #loss_of_holding_node.start_condition = loss_of_contact_node.observation_variable
    #loss_of_holding_node.end_condition = trinary_logic_not(loss_of_contact_node.observation_variable)


    segmind_context = episode_executor.context.require_extension(SegmindContext)
    episode_executor.compile(statechart)

    # Replay episode frame by frame
    import time

    # Replay episode frame by frame
    print("Replaying episode frames ...")
    frame_count = 0
    fps_interval = 100  # print every 50 frames
    t_start = time.perf_counter()
    t_interval = t_start

    for frame_data in file_player.frame_data_generator:
        if frame_count % 2 == 0:
            file_player.process_objects_data(frame_data)
            episode_executor.tick()

        frame_count += 1

        if frame_count % fps_interval == 0:
            now = time.perf_counter()
            fps = fps_interval / (now - t_interval)
            t_interval = now
            print(f"Frame {frame_count} — {fps:.1f} fps")

    total_time = time.perf_counter() - t_start
    print(f"Processed {frame_count} frames in {total_time:.1f}s — avg {frame_count / total_time:.1f} fps\n")

    # Print results
    events = segmind_context.logger.get_events()
    print("=== Detected Events ===")
    for event in events:
        print(event)
    print(f"Number of events:{len(events)}")

    rclpy.shutdown()


if __name__ == "__main__":
    run_demo()