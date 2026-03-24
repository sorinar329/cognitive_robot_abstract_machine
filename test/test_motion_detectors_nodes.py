import time
from typing import List
import numpy as np

from segmind.datastructures.events import (
    TranslationEvent,
    StopTranslationEvent,
    RotationEvent,
    StopRotationEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    TranslationDetector,
    StopTranslationDetector,
    RotationDetector,
    StopRotationDetector,
    SegmindContext,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.event_logger import EventLogger

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from test import setup_contact_world

class TestMotionDetectorsNodes:

    def test_translation_detector(self):
        world = setup_contact_world()
        sc = DetectorStateChart()
        logger = EventLogger()
        cylinder = world.get_body_by_name("cylinder_body")
        
        # Ensure it has a 6DoF connection so it's tracked by default if tracked_object is None
        # In setup_contact_world, it should already have one.
        
        context = SegmindContext(
            world=world,
            logger=logger,
        )

        executor = EpisodeSegmenterExecutor(context=context)

        translation_detector = TranslationDetector(
            name="translation_detector", context=context, tracked_object=cylinder, window_size=2
        )
        stop_translation_detector = StopTranslationDetector(
            name="stop_translation_detector", context=context, tracked_object=cylinder, window_size=2
        )

        sc.add_nodes([translation_detector, stop_translation_detector])
        executor.compile(sc)

        # 1. Initial tick in compile() + 1 explicit tick()
        executor.tick()
        assert len(context.latest_poses[cylinder]) == 2
        assert len(logger.get_events()) == 0

        # 2. Third tick - filling window (window_size=2 needs 3 poses)
        executor.tick()
        assert len(context.latest_poses[cylinder]) == 3 # [p0, p0, p0]
        assert len(logger.get_events()) == 0

        # 3. Fourth tick - movement start
        # window before tick: [p0, p0, p0]
        # window after tick: [p0, p0, p1]
        # is_moving will be True because distance(p0, p1) > 0.005
        current_x = cylinder.parent_connection.origin.x
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=current_x + 0.1)
        executor.tick()
        # Movement detected -> TranslationEvent
        events = logger.get_events()
        assert len(events) == 1
        assert isinstance(events[0], TranslationEvent)
        assert context.latest_motion_events[cylinder] == events[0]
        assert np.allclose(events[0].current_pose.to_position().x, cylinder.global_pose.x)

        # 4. Fifth tick - continued movement
        current_x = cylinder.parent_connection.origin.x
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=current_x + 0.1)
        executor.tick()
        # Still moving -> Update current_pose of the existing TranslationEvent, but NO new event logged
        events = logger.get_events()
        assert len(events) == 1

        # 5. Sixth tick - stop movement (not all window poses same yet)
        # window: [p2, p3, p3] -> is_moving = False (p3 == p3).
        # But not all poses are same (p2 != p3). So no StopTranslationEvent yet.
        executor.tick()
        events = logger.get_events()
        if len(events) != 1:
             print(f"\nDEBUG: Events after 6th tick: {events}")
        assert len(events) == 1

        # 6. Seventh tick - all poses same
        # window: [p3, p3, p3] -> all_poses_same = True -> StopTranslationEvent
        executor.tick()
        events = logger.get_events()
        assert len(events) == 2
        assert isinstance(events[1], StopTranslationEvent)
        assert cylinder not in context.latest_motion_events

    def test_rotation_detector(self):
        world = setup_contact_world()
        sc = DetectorStateChart()
        logger = EventLogger()
        cylinder = world.get_body_by_name("cylinder_body")
        
        context = SegmindContext(
            world=world,
            logger=logger,
        )

        executor = EpisodeSegmenterExecutor(context=context)

        rotation_detector = RotationDetector(
            name="rotation_detector", context=context, tracked_object=cylinder, window_size=2
        )
        stop_rotation_detector = StopRotationDetector(
            name="stop_rotation_detector", context=context, tracked_object=cylinder, window_size=2
        )

        sc.add_nodes([rotation_detector, stop_rotation_detector])
        executor.compile(sc)

        # 1. Initial ticks
        executor.tick()
        assert len(context.latest_poses[cylinder]) == 2
        executor.tick()
        # window is now [p0, p0, p0], window_size is 2, window_size+1 is 3.
        assert len(context.latest_poses[cylinder]) == 3
        assert len(logger.get_events()) == 0

        # 2. Rotation movement start
        # window: [p0, p0, p1] -> is_moving = True (distance > 0)
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(roll=0.1)
        executor.tick()
        # latest_poses: [p0, p1, p2] (p1=p0, p2=p1) -> [p0, p0, p1]
        events = logger.get_events()
        assert len(events) == 1
        assert isinstance(events[0], RotationEvent)

        # 3. Continued rotation
        # latest_poses: [p0, p1, p2]
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(roll=0.2)
        executor.tick()
        events = logger.get_events()
        assert len(events) == 1

        # 4. Stop rotation (not all poses same)
        # latest_poses: [p1, p2, p2] (p2=p2) -> is_moving = False
        executor.tick() 
        events = logger.get_events()
        assert len(events) == 1

        # 5. Stop rotation (all poses same)
        # latest_poses: [p2, p2, p2] -> all_poses_same = True -> StopRotationEvent
        executor.tick() 
        events = logger.get_events()
        assert len(events) == 2
        assert isinstance(events[1], StopRotationEvent)
