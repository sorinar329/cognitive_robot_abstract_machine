import time

from segmind.detectors.atomic_event_detectors import DetectorStateChart, ContactDetector, ContactDetectorNode
from segmind.event_logger import EventLogger

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.test_nodes.test_nodes import ChangeStateOnEvents
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from test import setup_contact_world


def test_motion_state_chart():
    world = setup_contact_world()
    sc = DetectorStateChart()
    logger = EventLogger()
    cylinder = world.get_body_by_name("cylinder_body")
    contact_detector = ContactDetectorNode(name="contact_detector", logger=logger, tracked_obj=cylinder)

    sc.add_nodes(
        [Sequence(
        [
            contact_detector
        ]
        )]
    )

    kin_sim = Executor(
        MotionStatechartContext(
            world=world
        )
    )

    kin_sim.compile(motion_statechart=sc)
    kin_sim.tick()

    events = logger.get_events()


    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.2)
    time.sleep(2)
    kin_sim.tick()
    kin_sim.tick()
    events = logger.get_events()

    kin_sim.motion_statechart.draw("/home/sorin/dev/cognitive_robot_abstract_machine/scratch/"+"sony.pdf")

