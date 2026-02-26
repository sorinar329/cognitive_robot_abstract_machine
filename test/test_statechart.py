import time

import rclpy

from segmind.datastructures.events import ContactEvent
from segmind.detectors.atomic_event_detectors import (
    DetectorStateChart,
    ContactDetector,
    ContactDetectorNode, LossOfContactDetectorNode,
)
from segmind.event_logger import EventLogger

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.test_nodes.test_nodes import ChangeStateOnEvents
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from test import setup_contact_world


def test_motion_state_chart():
    world = setup_contact_world()
    rclpy.init()
    node = rclpy.create_node("test_node")
    viz = VizMarkerPublisher(_world=world, node=node)
    viz.with_tf_publisher()
    sc = DetectorStateChart()
    logger = EventLogger()
    cylinder = world.get_body_by_name("cylinder_body")
    contact_detector = ContactDetectorNode(
        name="contact_detector", logger=logger, tracked_object=cylinder
    )

    loss_of_contact_detector = LossOfContactDetectorNode(
        name="loss_of_contact_detector", logger=logger, tracked_object=cylinder)

    sc.add_nodes([Sequence([contact_detector]),
                  Sequence([loss_of_contact_detector])
                  ])

    kin_sim = Executor(MotionStatechartContext(world=world))

    kin_sim.compile(motion_statechart=sc)
    kin_sim.tick()
    # No Contact yet
    assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 0
    assert contact_detector.observation_state == 0.0

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        y=-0.4
    )
    kin_sim.tick()

    # Contact with 2 objects
    assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 2
    assert contact_detector.observation_state == 1.0

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=2
    )

    kin_sim.tick()

    kin_sim.motion_statechart.draw("/home/sorin/dev/Segmind/test/img/" + "sony.pdf")
    print(logger.get_events())
