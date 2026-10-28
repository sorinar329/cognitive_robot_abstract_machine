from __future__ import annotations

import inspect
from dataclasses import dataclass

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
