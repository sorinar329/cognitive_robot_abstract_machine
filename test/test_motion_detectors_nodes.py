import time
from typing import List

from segmind.datastructures.events import (
    TranslationEvent,
    StopTranslationEvent,
    RotationEvent,
    StopRotationEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    DetectorStateChart,
    TranslationDetector,
    RotationDetector,
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

        sc.add_nodes([translation_detector])
        executor.compile(sc)

        # 1. Initial tick - window not full yet (window_size=2, so we need 3 poses)
        executor.tick()
        assert len(context.latest_poses[cylinder]) == 1
        assert len(logger.get_events()) == 0

        # 2. Second tick - window still not full
        executor.tick()
        assert len(context.latest_poses[cylinder]) == 2
        assert len(logger.get_events()) == 0

        # 3. Third tick - window full, no movement
        executor.tick()
        assert len(context.latest_poses[cylinder]) == 2 # after pop(0)
        assert len(logger.get_events()) == 0

        # 4. Fourth tick - movement
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.1)
        executor.tick()
        # Movement detected -> TranslationEvent
        events = logger.get_events()
        assert len(events) == 1
        assert isinstance(events[0], TranslationEvent)
        assert context.latest_motion_events[cylinder] == events[0]

        # 5. Fifth tick - continued movement
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.2)
        executor.tick()
        # Still moving -> New TranslationEvent replaces old one
        events = logger.get_events()
        assert len(events) == 2
        assert isinstance(events[1], TranslationEvent)
        assert context.latest_motion_events[cylinder] == events[1]
        assert events[1].start_pose != events[1].current_pose

        # 6. Sixth tick - stop movement (last two poses different, but we need ALL window poses same for stop)
        # latest_poses before tick: [x=0.1, x=0.2]
        # after append: [x=0.1, x=0.2, x=0.2]
        # check_obj_movement sees [x=0.1, x=0.2, x=0.2]. is_moving = False (0.2 == 0.2).
        # But not all poses are same (0.1 != 0.2). So no StopTranslationEvent yet.
        executor.tick()
        events = logger.get_events()
        assert len(events) == 2 # No new event

        # 7. Seventh tick - all poses same
        # latest_poses before tick: [x=0.2, x=0.2]
        # after append: [x=0.2, x=0.2, x=0.2]
        # all poses same -> StopTranslationEvent
        executor.tick()
        events = logger.get_events()
        assert len(events) == 3
        assert isinstance(events[2], StopTranslationEvent)
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

        sc.add_nodes([rotation_detector])
        executor.compile(sc)

        # 1. Initial ticks
        executor.tick()
        executor.tick()
        executor.tick()
        assert len(logger.get_events()) == 0

        # 2. Rotation movement
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(roll=0.1)
        executor.tick()
        events = logger.get_events()
        assert len(events) == 1
        assert isinstance(events[0], RotationEvent)

        # 3. Continued rotation
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(roll=0.2)
        executor.tick()
        events = logger.get_events()
        assert len(events) == 2
        assert isinstance(events[1], RotationEvent)

        # 4. Stop rotation
        executor.tick() # No event yet (window [roll=0.1, roll=0.2, roll=0.2])
        executor.tick() # StopRotationEvent
        events = logger.get_events()
        assert len(events) == 3
        assert isinstance(events[2], StopRotationEvent)
