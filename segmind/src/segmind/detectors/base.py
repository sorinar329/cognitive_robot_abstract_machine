import threading
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List, Any

from giskardpy.motion_statechart.context import MotionStatechartContext, ContextExtension
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from segmind.datastructures.events import MotionEvent, DetectionEvent, RotationEvent
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from segmind.event_logger import EventLogger
from semantic_digital_twin.semantic_annotations.semantic_annotations import Aperture
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class DetectorStateChart(MotionStatechart):
    """
    Statechart responsible for running the different motion detectors.

    Currently acts as a container for the detectors and inherits the
    functionality from MotionStatechart.
    """


IndexedBodyPairs = Dict[Body, Set[Body]]
"""
Type hint for dictionaries mapping bodies to sets of bodies
"""

_casadi_lock = threading.Lock()

@dataclass
class SegmindContext(ContextExtension):
    """
    Context object shared across the motion statechart detectors.

    Stores the latest detected contact and support relationships
    between bodies in the simulation as well as the event logger.
    """

    object_moving_status: Dict[Body, bool] = field(default_factory=dict)
    """
    Dictionary mapping each body to a boolean indicating if it is currently moving.
    """

    object_rotation_status: Dict[Body, bool] = field(default_factory=dict)
    """
    Dictionary mapping each body to a boolean indicating if it is currently rotating.
    """

    latest_contact_bodies: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies it is currently in contact with.
    """

    latest_support: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies that currently support it.
    """

    latest_containments: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies that currently contain it.
    """

    latest_poses: Dict[Body, List[Pose]] = field(default_factory=dict)
    """
    Dictionary mapping each body to a list of its recent poses (pose history).
    """

    latest_motion_events: Dict[Body, MotionEvent] = field(default_factory=dict)
    """
    Dictionary mapping each body to its currently active motion event, if any.
    """

    latest_rotation_events: Dict[Body, RotationEvent] = field(default_factory=dict)
    """
    Dictionary mapping each body to its currently active rotation event, if any.
    """

    logger: EventLogger = field(default_factory=EventLogger)
    """
    The event logger used to record detected events.
    """

    placing_pairs: set[Any] = field(default_factory=set)
    """
    Set of placing pairs, to avoid duplicate events
    """

    holes: List[Aperture] = field(default_factory=list)
    """
    List of bodies that can be considered holes
    """

    insertion_pairs: set[Any] = field(default_factory=set)
    """
    List of insertion pairs, to avoid duplicate events
    """

    tracker_registry: ObjectTrackerFactory = field(default_factory=ObjectTrackerFactory)
    """
    The object tracker registry.    
    """

    latest_holding: Dict[Body, Set[Body]] = field(default_factory=dict)
    """
    Dictionary mapping each held body to the set of gripper bodies currently holding it.
    """

    lifting_baselines: Dict[Body, tuple] = field(default_factory=dict)
    """
    Maps each held object to (baseline_z, baseline_pose) recorded at the moment holding started.
    """

    latest_lifting: set = field(default_factory=set)
    """
    Set of objects for which a LiftingEvent has already been fired in the current hold.
    Cleared when holding is lost.
    """

    contact_frame_counter: Dict[tuple, int] = field(default_factory=dict)
    contact_loss_frame_counter: Dict[tuple, int] = field(default_factory=dict)

    opening_tracker: Dict[Body, dict] = field(default_factory=dict)
    """
    Maps handle body to tracking state: pose history of handle and gripper during observation window.
    """

    latest_opening: set = field(default_factory=set)
    """
    Set of handles for which an OpeningEvent has already been fired.
    """

@dataclass(repr=False, eq=False)
class AbstractDetector(MotionStatechartNode, ABC):
    tracked_object: Optional[List[Body]] = field(kw_only=True, default=None)
    excluded_objects: list[str] = field(kw_only=True, default_factory=list)
    """
    :param excluded_objects: List of body names to exclude from tracking.
    """

    def on_tick(self, context: MotionStatechartContext) -> Optional[ObservationStateValues]:
        segmind_context_extension = context.require_extension(SegmindContext)

        objects_to_check = (
            self.tracked_object
            if self.tracked_object
            else [
                body
                for body in context.world.bodies
                if type(body.parent_connection) is Connection6DoF
                and body.name.name not in self.excluded_objects
            ]
        )
        events = self.update_context_and_events(context, segmind_context_extension, objects_to_check)
        for e in events:
            segmind_context_extension.logger.log_event(e, segmind_context_extension.tracker_registry)
        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE


    def get_relation(self, context: MotionStatechartContext, tracked_objects: List[Body], predicate) -> Dict[Body, Set[Body]]:
        """
        Get the relation between tracked objects.

        :param context: The context containing world information.
        :param tracked_objects: List of bodies to check for contact changes.
        :param predicate: Function that returns true if the objects are related.
        :return: Dictionary mapping bodies to sets of related bodies.
        """

        related_bodies: Dict[Body, Set[Body]] = {}
        bodies_with_collision = [
            b for b in context.world.bodies_with_collision
            if b.name.name not in self.excluded_objects
        ]
        for obj in tracked_objects:
            for body in bodies_with_collision:
                if body is obj:
                    continue
                if predicate(obj, body):
                    related_bodies.setdefault(obj, set()).add(body)
        return related_bodies

    @abstractmethod
    def update_context_and_events(self, context:MotionStatechartContext, segmind_context:SegmindContext, tracked_objects: List[Body]) -> List[DetectionEvent]:
        """
        Core detection logic that updates the internal state and identifies new events.

        This method is called during every tick of the detector. Implementations should
        examine the current state of the world (via the context) for the given
        `tracked_objects`, update the relevant fields in `context` (e.g.,
        `latest_contact_bodies`, `latest_support`), and return a list of any
        `DetectionEvent`s that occurred since the last update.

        Specific implementations may detect:
        * State changes: e.g., a new contact (ContactEvent) or loss of contact.
        * Continuous processes: e.g., ongoing motion or containment.
        * Complex interactions: e.g., insertion or picking up objects.

        :param context: The shared SegmindContext containing the world state,
                        history of relationships, and the event logger.
        :param segmind_context: The SegmindContext extension containing additional states.
        :param tracked_objects: A list of bodies that this detector should focus on
                                during this update cycle.
        :return: A list of DetectionEvent objects representing the events detected
                 in this cycle. Returns an empty list if no events were found.
        """
        pass
