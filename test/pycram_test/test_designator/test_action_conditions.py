from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.robot_plans import PickUpAction
from semantic_digital_twin.world_description.world_entity import Body


def test_get_bound_variables(immutable_model_world):
    world, view, context = immutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.manipulator,
        ),
    )

    bound_variables = pick_action.get_bound_variables()

    assert len(bound_variables) == 3
    assert list(bound_variables.keys()) == [
        "object_designator",
        "arm",
        "grasp_description",
    ]
    assert list(bound_variables["arm"]._domain_) == [Arms.LEFT]
    assert bound_variables["arm"]._type_ == Arms
    assert list(bound_variables["object_designator"]._domain_) == [
        world.get_body_by_name("milk.stl")
    ]
    assert bound_variables["object_designator"]._type_ == Body


def test_get_unbound_variables(immutable_model_world):
    world, view, context = immutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.manipulator,
        ),
    )

    unbound_variables = pick_action.get_unbound_variables()

    assert len(unbound_variables) == 3
