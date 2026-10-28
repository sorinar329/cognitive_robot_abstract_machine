"""
The events SegMind is expected to detect while an action of coraplex runs, read from the
action's parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Generic, TypeVar

from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.composite.carrying import CarryAction
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.composite.searching import (
    LookAtAndDetectAction,
    SearchAction,
)
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
    WipingAction,
)
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndOpenAction,
    MoveAndPickUpAction,
    MoveAndPlaceAction,
    PickAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.container import MovesContainerByItsHandle
from coraplex.robot_plans.actions.core.misc import DetectAction, MoveToReach
from coraplex.robot_plans.actions.core.navigation import (
    ElevatorNavigation,
    FaceAtAction,
    LookAtAction,
    NavigateAction,
    PathPlanningNavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import (
    GraspingAction,
    HasGraspChoice,
    PickUpAction,
    ReachAction,
)
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    FollowToolCenterPointPathAction,
    MoveFingersToWidthAction,
    MoveManipulatorAction,
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from krrood.ormatic.utils import classproperty
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from segmind.datastructures.events import (
    ContactEvent,
    EventWithTrackedObjects,
    GraspEvent,
    LossOfGraspEvent,
    LossOfSupportEvent,
    RotationEvent,
    SupportEvent,
    TranslationEvent,
)
from segmind.exceptions import NoEffectsOfAction
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

ActionType = TypeVar("ActionType", bound=Action)
"""
The type of action an :class:`ActionEffects` is of.
"""

# %% expected events


@dataclass
class ExpectedEvent:
    """
    An event SegMind is expected to detect while an action runs.
    """

    event_type: type[EventWithTrackedObjects]
    """
    The type of the event.
    """

    tracked_object: Body
    """
    The body the event is about.
    """

    with_object: Body | None = None
    """
    The other body the event is about; any other body, if None.
    """


# %% effects of one kind of action


@dataclass
class ActionEffects(Generic[ActionType], SubClassSafeGeneric, ABC):
    """
    The events an action of the bound type is expected to cause, read from its
    parameters.
    """

    @property
    def action_type(self) -> type[ActionType]:
        """
        :return: The type of action these effects are of.
        """
        return type(self).get_type_of_generic_parameter(ActionType)

    @abstractmethod
    def expected_events(self, action: ActionType) -> list[ExpectedEvent]:
        """
        :param action: The action, with its parameters grounded.
        :return: The events SegMind is expected to detect while ``action`` runs, in the
            order they are expected in.
        """

    @staticmethod
    def grasp_change(
        event_type: type[GraspEvent | LossOfGraspEvent], action: HasGraspChoice
    ) -> ExpectedEvent:
        """
        :param event_type: Whether the hand takes hold or lets go.
        :param action: The action taking hold by its grasp.
        :return: The hand of the action's arm taking hold of, or letting go of, the body
            its grasp is on.
        """
        return ExpectedEvent(
            event_type,
            action.grasp.graspable.root,
            action.arm.end_effector.tool_frame,
        )


# %% taking hold of and setting down


@dataclass
class GraspingEffects(ActionEffects[GraspingAction]):
    """
    A grasp takes hold of its object.
    """

    def expected_events(self, action: GraspingAction) -> list[ExpectedEvent]:
        return [self.grasp_change(GraspEvent, action)]


@dataclass
class PickUpEffects(ActionEffects[PickUpAction]):
    """
    A pick-up takes hold of its object and lifts it off what supported it.
    """

    def expected_events(self, action: PickUpAction) -> list[ExpectedEvent]:
        return [
            self.grasp_change(GraspEvent, action),
            ExpectedEvent(LossOfSupportEvent, action.grasp.graspable.root),
        ]


@dataclass
class PlaceEffects(ActionEffects[PlaceAction]):
    """
    A placing lets go of its object, which something then supports.

    The hand letting go is any: a placing names no arm.
    """

    def expected_events(self, action: PlaceAction) -> list[ExpectedEvent]:
        placed = action.object_designator.root
        return [
            ExpectedEvent(LossOfGraspEvent, placed),
            ExpectedEvent(SupportEvent, placed),
        ]


# %% containers


@dataclass
class ContainerEffects(ActionEffects[MovesContainerByItsHandle]):
    """
    Opening or closing a container takes hold of its handle, moves the part of the
    container behind its mechanism and lets go of the handle.

    The part slides along a prismatic mechanism, turns about a revolute one, and does
    both along a screw.
    """

    def expected_events(self, action: MovesContainerByItsHandle) -> list[ExpectedEvent]:
        mechanism = action.mechanism
        motions = []
        if not isinstance(mechanism, RevoluteConnection):
            motions.append(ExpectedEvent(TranslationEvent, mechanism.child))
        if not isinstance(mechanism, PrismaticConnection):
            motions.append(ExpectedEvent(RotationEvent, mechanism.child))
        return [
            self.grasp_change(GraspEvent, action),
            *motions,
            self.grasp_change(LossOfGraspEvent, action),
        ]


# %% tools


@dataclass
class CuttingEffects(ActionEffects[CuttingAction]):
    """
    Cutting touches what is cut with the tool.
    """

    def expected_events(self, action: CuttingAction) -> list[ExpectedEvent]:
        return [ExpectedEvent(ContactEvent, action.object_to_cut, action.tool.root)]


@dataclass
class MixingEffects(ActionEffects[MixingAction]):
    """
    Mixing touches the container with the tool.
    """

    def expected_events(self, action: MixingAction) -> list[ExpectedEvent]:
        return [ExpectedEvent(ContactEvent, action.container, action.tool.root)]


@dataclass
class WipingEffects(ActionEffects[WipingAction]):
    """
    Wiping touches its surface with the tool, or touches whatever lies at its target
    pose when it names no surface.
    """

    def expected_events(self, action: WipingAction) -> list[ExpectedEvent]:
        if action.surface is None:
            return [ExpectedEvent(ContactEvent, action.tool.root)]
        return [ExpectedEvent(ContactEvent, action.surface, action.tool.root)]


@dataclass
class PouringEffects(ActionEffects[PouringAction]):
    """
    Pouring tilts the source container.

    What is poured is not among the action's parameters, so nothing is expected of it.
    """

    def expected_events(self, action: PouringAction) -> list[ExpectedEvent]:
        return [ExpectedEvent(RotationEvent, action.source_container.root)]


# %% every action


@dataclass
class EffectsOfActions:
    """
    The events SegMind is expected to detect while any action of coraplex runs.

    An action made of other actions expects what they expect, in the order they run;
    an action that only moves the robot expects nothing, since SegMind watches objects.
    """

    effects: list[ActionEffects] = field(
        default_factory=lambda: [
            GraspingEffects(),
            PickUpEffects(),
            PlaceEffects(),
            ContainerEffects(),
            CuttingEffects(),
            MixingEffects(),
            WipingEffects(),
            PouringEffects(),
        ]
    )
    """
    The effects of the actions that act on objects themselves, one per kind of action.
    """

    @classproperty
    def actions_made_of_actions(cls) -> tuple[type[Action], ...]:
        """
        :return: The types of action whose parameters are the actions they run.
        """
        return (
            CarryAction,
            FaceAndLookAtAction,
            LookAtAndDetectAction,
            SearchAction,
            MoveAndOpenAction,
            MoveAndPickUpAction,
            MoveAndPlaceAction,
            PickAndPlaceAction,
            TransportAction,
        )

    @classproperty
    def actions_moving_only_the_robot(cls) -> tuple[type[Action], ...]:
        """
        :return: The types of action that act on no object.
        """
        return (
            NavigateAction,
            PathPlanningNavigateAction,
            ElevatorNavigation,
            FaceAtAction,
            LookAtAction,
            MoveToReach,
            ReachAction,
            DetectAction,
            FollowToolCenterPointPathAction,
            MoveFingersToWidthAction,
            MoveManipulatorAction,
            MoveTorsoAction,
            ParkArmsAction,
            SetGripperAction,
        )

    def covers(self, action_type: type[Action]) -> bool:
        """
        :param action_type: A type of action.
        :return: Whether the events an action of ``action_type`` expects are known.
        """
        return issubclass(
            action_type,
            (
                *self.actions_made_of_actions,
                *self.actions_moving_only_the_robot,
                *(effects.action_type for effects in self.effects),
            ),
        )

    def expected_events(self, action: Action) -> list[ExpectedEvent]:
        """
        :param action: The action, with its parameters grounded.
        :return: The events SegMind is expected to detect while ``action`` runs, in the
            order they are expected in.
        :raises NoEffectsOfAction: When the events of ``action`` are not known.
        """
        if isinstance(action, self.actions_made_of_actions):
            return [
                event
                for sub_action in self.sub_actions_of(action)
                for event in self.expected_events(sub_action)
            ]
        if isinstance(action, self.actions_moving_only_the_robot):
            return []
        for effects in self.effects:
            if isinstance(action, effects.action_type):
                return effects.expected_events(action)
        raise NoEffectsOfAction(action_type=type(action))

    @staticmethod
    def sub_actions_of(action: Action) -> list[Action]:
        """
        :param action: An action made of other actions.
        :return: The actions ``action`` runs, given as its parameters on their own or in a
            list, in the order of its parameters.
        """
        sub_actions = []
        for parameter in action.designator_parameter.values():
            if isinstance(parameter, Action):
                sub_actions.append(parameter)
            elif isinstance(parameter, list):
                sub_actions.extend(
                    sub_action
                    for sub_action in parameter
                    if isinstance(sub_action, Action)
                )
        return sub_actions
