from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest

import coraplex.robot_plans.actions.composite.carrying
import coraplex.robot_plans.actions.composite.facing
import coraplex.robot_plans.actions.composite.searching
import coraplex.robot_plans.actions.composite.tool_based
import coraplex.robot_plans.actions.composite.transporting
import coraplex.robot_plans.actions.core.container
import coraplex.robot_plans.actions.core.misc
import coraplex.robot_plans.actions.core.navigation
import coraplex.robot_plans.actions.core.pick_up
import coraplex.robot_plans.actions.core.placing
import coraplex.robot_plans.actions.core.robot_body
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
    WipingAction,
)
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    PickAndPlaceAction,
)
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import (
    FaceAtAction,
    LookAtAction,
    NavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import GraspingAction, PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from krrood.utils import recursive_subclasses
from segmind.action_effects import EffectsOfActions, ExpectedEvent
from segmind.datastructures.events import (
    ContactEvent,
    GraspEvent,
    LossOfGraspEvent,
    LossOfSupportEvent,
    RotationEvent,
    StopTranslationEvent,
    SupportEvent,
    TranslationEvent,
)
from segmind.exceptions import NoEffectsOfAction
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CuttingKnife,
    Handle,
    Milk,
    PouringCup,
    Sponge,
    Whisk,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

# %% the scene the actions act in


@dataclass
class ActionScene:
    """
    The PR2 in the apartment with what its actions act on, annotated without changing the
    shared world: the milk, the handle of a drawer, and tools of their own.
    """

    arm: Arm
    """
    The arm every action of a test uses.
    """

    milk: Milk
    """
    The milk standing in the apartment.
    """

    handle: Handle
    """
    The handle of a drawer that slides open.
    """

    table: Body
    """
    A body of the apartment the tools act on.
    """

    @property
    def tool_frame(self) -> Body:
        """
        :return: The tool frame of :attr:`arm`.
        """
        return self.arm.end_effector.tool_frame

    @property
    def somewhere(self) -> Pose:
        """
        :return: A pose in the apartment, for the actions that need one.
        """
        return Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.0, reference_frame=self.table)

    def pick_up(self) -> PickUpAction:
        """
        :return: A pick-up of :attr:`milk` by :attr:`arm`.
        """
        return PickUpAction(
            grasp=GraspCandidate.from_body_origin(self.milk), arm=self.arm
        )

    def place(self) -> PlaceAction:
        """
        :return: A placing of :attr:`milk` at :attr:`somewhere`.
        """
        return PlaceAction(object_designator=self.milk, target_location=self.somewhere)

    @staticmethod
    def tool_body(name: str) -> Body:
        """
        :return: A new body, in no world, for a tool to be made of.
        """
        return Body(name=PrefixedName(name))


@pytest.fixture
def scene(pr2_apartment_world) -> ActionScene:
    [pr2] = pr2_apartment_world.get_semantic_annotations_by_type(PR2)
    return ActionScene(
        arm=pr2.left_arm,
        milk=Milk(root=pr2_apartment_world.get_body_by_name("milk.stl")),
        handle=Handle(root=pr2_apartment_world.get_body_by_name("handle_cab10_m")),
        table=pr2_apartment_world.get_body_by_name("table_area_main"),
    )


# %% every action


def test_every_action_of_coraplex_has_expected_events() -> None:
    concrete_actions = [
        action_type
        for action_type in recursive_subclasses(Action)
        if not inspect.isabstract(action_type)
        and action_type.__module__.startswith("coraplex.")
    ]
    effects = EffectsOfActions()
    assert [
        action_type
        for action_type in concrete_actions
        if not effects.covers(action_type)
    ] == []


def test_an_action_nothing_covers_has_no_expected_events(scene) -> None:
    with pytest.raises(NoEffectsOfAction):
        EffectsOfActions(effects=[]).expected_events(scene.pick_up())


# %% taking hold of and setting down


def test_a_grasp_expects_the_hand_to_take_hold_of_the_object(scene) -> None:
    grasping = GraspingAction(
        grasp=GraspCandidate.from_body_origin(scene.milk), arm=scene.arm
    )
    assert EffectsOfActions().expected_events(grasping) == [
        ExpectedEvent(GraspEvent, scene.milk.root, scene.tool_frame)
    ]


def test_a_pick_up_expects_the_hand_to_take_hold_and_the_object_to_lose_its_support(
    scene,
) -> None:
    assert EffectsOfActions().expected_events(scene.pick_up()) == [
        ExpectedEvent(GraspEvent, scene.milk.root, scene.tool_frame),
        ExpectedEvent(LossOfSupportEvent, scene.milk.root),
    ]


def test_a_placing_expects_the_object_to_be_let_go_of_and_supported(scene) -> None:
    assert EffectsOfActions().expected_events(scene.place()) == [
        ExpectedEvent(LossOfGraspEvent, scene.milk.root),
        ExpectedEvent(SupportEvent, scene.milk.root),
    ]


# %% containers


@pytest.mark.parametrize("action_type", [OpenAction, CloseAction])
def test_moving_a_drawer_expects_its_handle_held_while_the_drawer_slides(
    scene, action_type
) -> None:
    action = action_type(
        grasp=GraspCandidate.from_body_origin(scene.handle), arm=scene.arm
    )
    assert EffectsOfActions().expected_events(action) == [
        ExpectedEvent(GraspEvent, scene.handle.root, scene.tool_frame),
        ExpectedEvent(TranslationEvent, action.mechanism.child),
        ExpectedEvent(LossOfGraspEvent, scene.handle.root, scene.tool_frame),
    ]


# %% tools


def test_cutting_expects_the_tool_to_touch_what_it_cuts(scene) -> None:
    knife = CuttingKnife(root=scene.tool_body("knife"))
    cutting = CuttingAction(arm=scene.arm, tool=knife, object_to_cut=scene.table)
    assert EffectsOfActions().expected_events(cutting) == [
        ExpectedEvent(ContactEvent, scene.table, knife.root)
    ]


def test_mixing_expects_the_tool_to_touch_the_container(scene) -> None:
    whisk = Whisk(root=scene.tool_body("whisk"))
    mixing = MixingAction(arm=scene.arm, tool=whisk, container=scene.table)
    assert EffectsOfActions().expected_events(mixing) == [
        ExpectedEvent(ContactEvent, scene.table, whisk.root)
    ]


def test_wiping_a_surface_expects_the_tool_to_touch_it(scene) -> None:
    sponge = Sponge(root=scene.tool_body("sponge"))
    wiping = WipingAction(arm=scene.arm, tool=sponge, surface=scene.table)
    assert EffectsOfActions().expected_events(wiping) == [
        ExpectedEvent(ContactEvent, scene.table, sponge.root)
    ]


def test_wiping_around_a_pose_expects_the_tool_to_touch_something(scene) -> None:
    sponge = Sponge(root=scene.tool_body("sponge"))
    wiping = WipingAction(arm=scene.arm, tool=sponge, target_pose=scene.somewhere)
    assert EffectsOfActions().expected_events(wiping) == [
        ExpectedEvent(ContactEvent, sponge.root)
    ]


def test_pouring_expects_the_source_to_tilt(scene) -> None:
    cup = PouringCup(root=scene.tool_body("cup"))
    pouring = PouringAction(
        target_container=scene.table, source_container=cup, arm=scene.arm
    )
    assert EffectsOfActions().expected_events(pouring) == [
        ExpectedEvent(RotationEvent, cup.root)
    ]


# %% the robot alone


def test_an_action_moving_only_the_robot_expects_no_events(scene) -> None:
    assert (
        EffectsOfActions().expected_events(
            NavigateAction(target_location=scene.somewhere)
        )
        == []
    )


# %% actions made of other actions


def test_an_action_made_of_actions_expects_their_events_in_order(scene) -> None:
    pick_up, place = scene.pick_up(), scene.place()
    effects = EffectsOfActions()
    assert effects.expected_events(
        PickAndPlaceAction(pick_up=pick_up, place=place)
    ) == effects.expected_events(pick_up) + effects.expected_events(place)


def test_actions_moving_only_the_robot_add_nothing_to_the_action_they_are_part_of(
    scene,
) -> None:
    pick_up = scene.pick_up()
    effects = EffectsOfActions()
    moving_and_picking_up = MoveAndPickUpAction(
        navigate=NavigateAction(target_location=scene.somewhere),
        face_and_look_at=FaceAndLookAtAction(
            face_at=FaceAtAction(target=scene.somewhere),
            look_at=LookAtAction(target=scene.somewhere),
        ),
        pick_up=pick_up,
    )
    assert effects.expected_events(moving_and_picking_up) == effects.expected_events(
        pick_up
    )


# %% meeting an expectation


@dataclass
class BodiesAnEventIsAbout:
    """
    A held object, the surface it stands on, and the tool frames of two hands, with a
    clock that hands out one moment after the other to the events made of them.
    """

    held: Body = field(default_factory=lambda: Body(name=PrefixedName("held")))
    surface: Body = field(default_factory=lambda: Body(name=PrefixedName("surface")))
    hand: Body = field(default_factory=lambda: Body(name=PrefixedName("hand")))
    other_hand: Body = field(
        default_factory=lambda: Body(name=PrefixedName("other_hand"))
    )
    now: datetime = field(default_factory=lambda: datetime(2026, 1, 1))

    def next_moment(self) -> datetime:
        """
        :return: A moment one second after the previous one.
        """
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def bodies() -> BodiesAnEventIsAbout:
    return BodiesAnEventIsAbout()


def test_a_grasp_let_go_of_again_does_not_meet_an_expected_grasp(bodies) -> None:
    expected = ExpectedEvent(GraspEvent, bodies.held, bodies.hand)
    events = [
        GraspEvent(
            tracked_object=bodies.held,
            with_object=bodies.hand,
            timestamp=bodies.next_moment(),
        ),
        LossOfGraspEvent(
            tracked_object=bodies.held,
            with_object=bodies.hand,
            timestamp=bodies.next_moment(),
        ),
    ]

    assert expected.evidence_among(events) == []


def test_a_support_regained_after_its_loss_meets_an_expected_support(bodies) -> None:
    expected = ExpectedEvent(SupportEvent, bodies.held)
    lost = LossOfSupportEvent(
        tracked_object=bodies.held, timestamp=bodies.next_moment()
    )
    regained = SupportEvent(
        tracked_object=bodies.held,
        with_object=bodies.surface,
        timestamp=bodies.next_moment(),
    )

    assert expected.evidence_among([lost, regained]) == [regained]


def test_a_translation_that_stopped_again_meets_an_expected_translation(
    bodies,
) -> None:
    expected = ExpectedEvent(TranslationEvent, bodies.held)
    translation = TranslationEvent(
        tracked_object=bodies.held,
        world_T_start_pose=Pose(),
        world_T_current_pose=Pose(),
        timestamp=bodies.next_moment(),
    )
    stop = StopTranslationEvent(
        tracked_object=bodies.held,
        world_T_start_pose=Pose(),
        world_T_current_pose=Pose(),
        timestamp=bodies.next_moment(),
    )

    assert expected.evidence_among([translation, stop]) == [translation]


def test_a_grasp_by_another_hand_does_not_meet_a_grasp_expected_of_one_hand(
    bodies,
) -> None:
    expected = ExpectedEvent(GraspEvent, bodies.held, bodies.hand)
    events = [
        GraspEvent(
            tracked_object=bodies.held,
            with_object=bodies.other_hand,
            timestamp=bodies.next_moment(),
        )
    ]

    assert expected.evidence_among(events) == []


def test_any_hand_letting_go_meets_a_loss_of_grasp_expected_of_no_hand(
    bodies,
) -> None:
    expected = ExpectedEvent(LossOfGraspEvent, bodies.held)
    letting_go = LossOfGraspEvent(
        tracked_object=bodies.held,
        with_object=bodies.other_hand,
        timestamp=bodies.next_moment(),
    )

    assert expected.evidence_among([letting_go]) == [letting_go]


def test_an_event_about_another_body_does_not_meet_an_expectation(bodies) -> None:
    expected = ExpectedEvent(SupportEvent, bodies.held)
    events = [
        SupportEvent(
            tracked_object=bodies.surface,
            with_object=bodies.held,
            timestamp=bodies.next_moment(),
        )
    ]

    assert expected.evidence_among(events) == []
