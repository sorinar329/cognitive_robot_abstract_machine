from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from segmind.datastructures.events import Event, ContactEvent, LossOfContactEvent
from semantic_digital_twin.reasoning.predicates import contact
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class SegmindContext(MotionStatechartContext):
    """
    Context object shared across the motion statechart detectors.

    Stores the latest detected contact and support relationships
    between bodies in the simulation as well as the event logger.

    :param latest_contact_bodies: Dictionary mapping each body to the set of
        bodies it is currently in contact with.
    :param latest_support: Dictionary mapping each body to the set of bodies
        that currently support it.
    :param logger: Logger used to record detected events.
    ToDo:  Why circular import for EventLogger? and move this away
    """

    latest_contact_bodies: Dict[Body, Set[Body]] = None
    latest_support: Dict[Body, Set[Body]] = None
    logger: Optional[Any] = None


class DetectorStateChart(MotionStatechart):
    """
    Statechart responsible for running the different motion detectors.

    Currently acts as a container for the detectors and inherits the
    functionality from MotionStatechart.
    """

    pass


@dataclass(eq=False, repr=False)
class BaseContactDetector(MotionStatechartNode, ABC):
    """
    Abstract base class for contact-based detectors.

    Provides shared functionality for detecting contacts between
    bodies and generating events when contact relationships change.

    :param tracked_object: Optional body that should be monitored.
        If None, all trackable objects in the world are checked.
    :param context: Segmind context containing world information,
        contact history and logging utilities.
    """

    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    context: SegmindContext = field(kw_only=True)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Executes one update cycle of the detector.

        Determines the objects that should be checked for contacts,
        computes new contact relationships and triggers events if
        contact changes are detected.

        :param context: The statechart execution context.
        :return: ObservationStateValues.TRUE if events were triggered,
            otherwise ObservationStateValues.FALSE.
        """

        trackable_objects = [
            body
            for body in self.context.world.bodies
            if type(body.parent_connection) is Connection6DoF
        ]
        objects_to_check = (
            [self.tracked_object] if self.tracked_object else trackable_objects
        )
        events = self.update_latest_contact_bodies_and_trigger_events(objects_to_check)
        for e in events:
            self.context.logger.log_event(e)
        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE

    def get_contact_bodies(self, tracked_objects: List[Body]) -> Dict[Body, Set[Body]]:
        """
        Computes the contact relationships for a set of tracked objects.

        Iterates over all bodies with collision enabled and checks whether
        they are in contact with the tracked objects.

        :param tracked_objects: List of bodies that should be checked for contacts.
        :return: Dictionary mapping each tracked body to the set of bodies
            it is currently in contact with.
        """

        contact_bodies: Dict[Body, Set[Body]] = {}
        bodies_with_collision = self.context.world.bodies_with_collision
        for obj in tracked_objects:
            for body in bodies_with_collision:
                if body is obj:
                    continue
                if contact(obj, body):
                    contact_bodies.setdefault(obj, set()).add(body)
        return contact_bodies

    @abstractmethod
    def update_latest_contact_bodies_and_trigger_events(
        self,
        tracked_objects: List[Body],
    ) -> List[Event]:
        """
        Updates the stored contact relationships and generates events
        when changes are detected.

        Implementations define how contact changes are interpreted
        (e.g. new contact or loss of contact).

        :param tracked_objects: List of bodies to check for contact changes.
        :return: List of events generated during the update.
        """

        pass


@dataclass(eq=False, repr=False)
class ContactDetector(BaseContactDetector):
    """
    Detector responsible for identifying newly established contacts
    between bodies.
    """

    def update_latest_contact_bodies_and_trigger_events(
        self,
        tracked_objects: List[Body],
    ) -> List[Event]:
        """
        Detects newly formed contacts and updates the stored contact state.

        Generates a ContactEvent whenever a new contact between two bodies
        is detected.

        :param tracked_objects: List of bodies to check for new contacts.
        :return: List of ContactEvent instances generated during this update.
        """

        new_contact_pairs = self.get_contact_bodies(tracked_objects)
        latest_contact_bodies = self.context.latest_contact_bodies
        events = []
        for obj, contact_list in new_contact_pairs.items():
            if obj not in latest_contact_bodies:
                latest_contact_bodies[obj] = contact_list
                for body in contact_list:
                    events.append(
                        ContactEvent(
                            close_bodies=contact_list,
                            latest_close_bodies=latest_contact_bodies,
                            of_object=obj,
                            with_object=body,
                        )
                    )
            else:
                new_contacts = contact_list - latest_contact_bodies[obj]
                if new_contacts:
                    latest_contact_bodies[obj] |= new_contacts
                    for body in new_contacts:
                        events.append(
                            ContactEvent(
                                close_bodies=contact_list,
                                latest_close_bodies=latest_contact_bodies,
                                of_object=obj,
                                with_object=body,
                            )
                        )

        return events


@dataclass(eq=False, repr=False)
class LossOfContactDetector(BaseContactDetector):
    """
    Detector responsible for identifying when previously existing
    contacts between bodies are lost.
    """

    def update_latest_contact_bodies_and_trigger_events(
        self,
        tracked_objects: List[Body],
    ) -> List[Event]:
        """
        Detects when existing contacts are no longer present and updates
        the stored contact state accordingly.

        Generates a LossOfContactEvent whenever a previously detected
        contact no longer exists.

        :param tracked_objects: List of bodies to check for lost contacts.
        :return: List of LossOfContactEvent instances generated during this update.
        """

        new_contact_pairs = self.get_contact_bodies(tracked_objects)
        latest_contact_bodies = self.context.latest_contact_bodies
        events = []
        for obj, contact_list in list(latest_contact_bodies.items()):
            if obj not in new_contact_pairs:
                latest_contact_bodies.pop(obj)
                for body in contact_list:
                    events.append(
                        LossOfContactEvent(
                            close_bodies=contact_list,
                            latest_close_bodies=latest_contact_bodies,
                            of_object=obj,
                            with_object=body,
                        )
                    )
            else:
                new_contacts = new_contact_pairs[obj]
                lost_contacts = contact_list - new_contacts
                if lost_contacts:
                    latest_contact_bodies[obj] = new_contacts
                    for body in lost_contacts:
                        events.append(
                            LossOfContactEvent(
                                close_bodies=contact_list,
                                latest_close_bodies=latest_contact_bodies,
                                of_object=obj,
                                with_object=body,
                            )
                        )

        return events
