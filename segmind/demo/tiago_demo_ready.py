"""
SegMind Episode Segmentation Demo
==================================
Replays a recorded robot manipulation episode through the SegMind statechart
pipeline and prints the detected event timeline to stdout.

The episode is a TIAGo dual-arm robot performing a domestic kitchen task:
picking up a milk box from a counter and opening a refrigerator door.
The scene is described by an MJCF model and the episode data is stored as
a CSV file containing 6-DoF poses for all scene objects and joint states,
recorded via Multiverse from a MuJoCo simulation.

Usage
-----
    python segmind_demo.py

    Optionally override paths via environment variables:
        SEGMIND_SCENE_PATH   Path to the MJCF scene file
        SEGMIND_CSV_PATH     Path to the episode CSV file

Dependencies
------------
    rclpy, giskardpy, segmind, semantic_digital_twin, krrood

Expected output
---------------
    A printed event timeline of the form:
        HoldingEvent: milk_box held by [...] - <timestamp>
        PickUpEvent:  milk_box - <timestamp>
        ...
"""

import os
import time

import rclpy

from giskardpy.motion_statechart.context import MotionStatechartContext
from krrood.symbolic_math.symbolic_math import trinary_logic_not
from segmind.detectors.agent_event_detectors_nodes import (
    HoldingDetector,
    LossOfHoldingDetector,
    LiftingDetector,
    OpeningDetector,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    LossOfContactDetector,
    TranslationDetector,
    StopTranslationDetector,
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
# Configuration
# ---------------------------------------------------------------------------

# Base directory of the segmind package resources, resolved relative to this
# file so the demo works regardless of the working directory.
_RESOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "resources", "tiago_episodes")

SCENE_PATH: str = os.environ.get(
    "SEGMIND_SCENE_PATH",
    os.path.join(_RESOURCES_DIR, "models", "assets", "mjcf",
                 "iai_tiago_velocity_in_apartment_with_multiverse.xml"),
)
"""
Absolute path to the MJCF scene description file.
Override via the ``SEGMIND_SCENE_PATH`` environment variable.
"""

CSV_PATH: str = os.environ.get(
    "SEGMIND_CSV_PATH",
    os.path.join(_RESOURCES_DIR, "data", "data.csv"),
)
"""
Absolute path to the episode CSV file produced by Multiverse.
Override via the ``SEGMIND_CSV_PATH`` environment variable.
"""

# Objects excluded from contact and support detection to reduce noise.
# These are either non-relevant robot base links or objects not present
# in the episode.
_EXCLUDED_OBJECTS = ["base_footprint", "bowl"]

# Knuckle links are excluded from contact detection because their bounding
# volumes overlap with the finger links and produce spurious contacts.
_EXCLUDED_KNUCKLES = [
    "gripper_right_left_inner_knuckle",
    "gripper_right_right_inner_knuckle",
    "gripper_left_left_inner_knuckle",
    "gripper_left_right_inner_knuckle",
]

# Gripper finger groups used for holding detection.
# Each group contains the finger body names that must all be in contact
# simultaneously for a HoldingEvent to be fired.
_RIGHT_GRIPPER = ["gripper_right_left_inner_finger", "gripper_right_right_inner_finger"]
_LEFT_GRIPPER  = ["gripper_left_left_inner_finger",  "gripper_left_right_inner_finger"]

# Logging interval for FPS output during replay.
_FPS_LOG_INTERVAL = 100


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def build_world(scene_path: str):
    """
    Parse the MJCF scene file and return the world object.

    :param scene_path: Absolute path to the MJCF scene description file.
    :return: Parsed world object containing all bodies, joints, and connections.
    :raises FileNotFoundError: If ``scene_path`` does not exist.
    """
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    return MJCFParser(scene_path).parse()


def build_episode_executor(world, csv_path: str) -> tuple:
    """
    Create the CSV episode player, motion context, and episode executor.

    :param world: Parsed world object returned by :func:`build_world`.
    :param csv_path: Absolute path to the episode CSV file.
    :return: Tuple of ``(file_player, context, episode_executor)``.
    :raises FileNotFoundError: If ``csv_path`` does not exist.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Episode CSV not found: {csv_path}")

    file_player = CSVEpisodePlayer(file_path=csv_path, world=world)
    context = MotionStatechartContext(world=world)
    episode_executor = EpisodeSegmenterExecutor(
        context=context,
        player=file_player,
    )
    return file_player, context, episode_executor


def build_statechart(world) -> SegmindStatechart:
    """
    Construct and wire the SegMind statechart with the full detector stack.

    The detectors are organized in three tiers following the SegMind event
    hierarchy:

    **Atomic** — ContactDetector, LossOfContactDetector,
    TranslationDetector, StopTranslationDetector

    **Spatial** — SupportDetector, LossOfSupportDetector

    **Agent / Coarse** — HoldingDetector, LossOfHoldingDetector,
    LiftingDetector, OpeningDetector, PickUpDetector, PlacingDetector

    Start and end conditions are set so that higher-tier detectors only run
    when the relevant lower-tier conditions are active, reducing unnecessary
    computation.

    :param world: Parsed world object used to look up tracked body references.
    :return: Compiled :class:`SegmindStatechart` ready for ticking.
    """
    statechart = SegmindStatechart()

    # ------------------------------------------------------------------
    # Atomic detectors
    # ------------------------------------------------------------------
    contact_node = ContactDetector(
        excluded_objects=_EXCLUDED_OBJECTS + _EXCLUDED_KNUCKLES,
        tracked_object=[
            world.get_body_by_name("milk_box"),
            world.get_body_by_name("fridge_door1_handle"),
        ],
    )
    loss_of_contact_node = LossOfContactDetector(
        excluded_objects=_EXCLUDED_OBJECTS,
    )
    translation_node = TranslationDetector(
        excluded_objects=_EXCLUDED_OBJECTS,
    )
    stop_translation_node = StopTranslationDetector(
        excluded_objects=_EXCLUDED_OBJECTS,
    )

    # ------------------------------------------------------------------
    # Spatial detectors
    # ------------------------------------------------------------------
    support_node = SupportDetector(
        excluded_objects=_EXCLUDED_OBJECTS,
    )
    loss_of_support_node = LossOfSupportDetector(
        excluded_objects=_EXCLUDED_OBJECTS,
    )

    # ------------------------------------------------------------------
    # Agent detectors
    # ------------------------------------------------------------------
    holding_node = HoldingDetector(
        excluded_objects=_EXCLUDED_OBJECTS,
        tracked_object=[
            world.get_body_by_name("milk_box"),
            world.get_body_by_name("fridge_door1_handle"),
        ],
        gripper_groups=[_RIGHT_GRIPPER, _LEFT_GRIPPER],
    )
    loss_of_holding_node = LossOfHoldingDetector(
        excluded_objects=_EXCLUDED_OBJECTS,
        tracked_object=[
            world.get_body_by_name("milk_box"),
            world.get_body_by_name("fridge_door1_handle"),
        ],
        gripper_groups=[_RIGHT_GRIPPER, _LEFT_GRIPPER],
    )
    lifting_node = LiftingDetector(
        excluded_objects=_EXCLUDED_OBJECTS,
    )
    opening_detector = OpeningDetector(
        handle_name="fridge_door1_handle",
        gripper_groups=[
            ["gripper_left_right_inner_finger", "gripper_left_left_inner_finger",
             "gripper_left_right_inner_knuckle", "gripper_left_left_inner_knuckle"],
        ],
        observation_window=10,
        distance_threshold=0.01,
        angle_threshold=0.3,
        excluded_objects=_EXCLUDED_OBJECTS,
    )

    # ------------------------------------------------------------------
    # Coarse detectors
    # ------------------------------------------------------------------
    placing_node = PlacingDetector(excluded_objects=_EXCLUDED_OBJECTS)
    pickup_node  = PickUpDetector(excluded_objects=_EXCLUDED_OBJECTS)

    # ------------------------------------------------------------------
    # Register nodes
    # ------------------------------------------------------------------
    for node in [
        contact_node, loss_of_contact_node,
        support_node, loss_of_support_node,
        translation_node, stop_translation_node,
        placing_node, pickup_node,
        holding_node, loss_of_holding_node,
        lifting_node, opening_detector,
    ]:
        statechart.add_node(node)

    # ------------------------------------------------------------------
    # Wire start / end conditions
    # ------------------------------------------------------------------
    support_node.start_condition          = contact_node.observation_variable
    support_node.end_condition            = trinary_logic_not(contact_node.observation_variable)

    loss_of_support_node.start_condition  = loss_of_contact_node.observation_variable
    loss_of_support_node.end_condition    = trinary_logic_not(loss_of_contact_node.observation_variable)

    placing_node.start_condition          = support_node.observation_variable
    placing_node.end_condition            = trinary_logic_not(support_node.observation_variable)

    pickup_node.start_condition           = loss_of_support_node.observation_variable
    pickup_node.end_condition             = trinary_logic_not(loss_of_support_node.observation_variable)

    holding_node.start_condition          = contact_node.observation_variable
    holding_node.end_condition            = trinary_logic_not(contact_node.observation_variable)

    loss_of_holding_node.start_condition  = loss_of_contact_node.observation_variable
    loss_of_holding_node.end_condition    = trinary_logic_not(loss_of_contact_node.observation_variable)

    lifting_node.start_condition          = holding_node.observation_variable
    lifting_node.end_condition            = trinary_logic_not(holding_node.observation_variable)

    opening_detector.start_condition      = holding_node.observation_variable
    opening_detector.end_condition        = trinary_logic_not(holding_node.observation_variable)

    return statechart


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """
    Run the full SegMind episode segmentation demo.

    Loads the scene and episode data, initialises ROS2 and the visualisation
    publisher, compiles the statechart, replays the episode frame by frame,
    and prints the detected event timeline to stdout.
    """
    print("=== SegMind Demo ===\n")

    # Parse scene
    print(f"Loading scene:   {SCENE_PATH}")
    world = build_world(SCENE_PATH)

    # Build pipeline
    print(f"Loading episode: {CSV_PATH}")
    file_player, context, episode_executor = build_episode_executor(world, CSV_PATH)

    # Initialise ROS2 and visualisation
    rclpy.init()
    node = rclpy.create_node("segmind_demo")
    viz_marker_publisher = VizMarkerPublisher(_world=world, node=node)
    viz_marker_publisher.with_tf_publisher()

    # Build and compile statechart
    print("Compiling statechart ...\n")
    statechart = build_statechart(world)
    segmind_context = episode_executor.context.require_extension(SegmindContext)
    episode_executor.compile(statechart)

    # Replay episode frame by frame
    print("Replaying episode frames ...")
    frame_count = 0
    t_start    = time.perf_counter()
    t_interval = t_start

    for frame_data in file_player.frame_data_generator:
        file_player.process_objects_data(frame_data)
        episode_executor.tick()
        frame_count += 1

        if frame_count % _FPS_LOG_INTERVAL == 0:
            now = time.perf_counter()
            fps = _FPS_LOG_INTERVAL / (now - t_interval)
            t_interval = now
            print(f"  Frame {frame_count:>5} — {fps:.1f} fps")

    total_time = time.perf_counter() - t_start
    print(f"\nProcessed {frame_count} frames in {total_time:.1f}s "
          f"(avg {frame_count / total_time:.1f} fps)\n")

    # Print results
    events = segmind_context.logger.get_events()
    print("=== Detected Events ===")
    for event in events:
        print(event)
    print(f"\nTotal events: {len(events)}")

    rclpy.shutdown()


if __name__ == "__main__":
    run_demo()