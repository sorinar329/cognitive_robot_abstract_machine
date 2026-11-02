from enum import StrEnum

import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

from semantic_digital_twin.adapters.robocasa_dataset.exceptions import (
    RoboCasaTaskNotFoundError,
)
from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
)
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
    RoboCasaObjectCategory,
)
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Dishwasher,
    Floor,
)
from krrood_spatial_math.spatial_types import Point3


@pytest.fixture(scope="session")
def robocasa_loader() -> RoboCasaDatasetLoader:
    loader = RoboCasaDatasetLoader()
    if not loader.directory.exists():
        pytest.skip(
            "RoboCasa assets not downloaded. Run "
            "'python -m robocasa.scripts.download_kitchen_assets' first."
        )
    return loader


def test_load_kitchen(robocasa_loader):
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    world = robocasa_loader.load_kitchen(
        layout_id=next(iter(LayoutType)), style_id=next(iter(StyleType))
    )
    assert len(world.bodies) > 0
    assert len(world.semantic_annotations) > 0


def test_load_kitchen_appliance(robocasa_loader):
    world = robocasa_loader.load_kitchen_appliance(
        RoboCasaKitchenApplianceCategory.CABINET
    )
    assert len(world.bodies) > 0
    assert len(world.semantic_annotations) >= 1


def test_load_kitchen_appliance_attaches_matching_annotation(robocasa_loader):
    """
    The loaded appliance is annotated with the semantic type of the requested category.
    """
    world = robocasa_loader.load_kitchen_appliance(
        RoboCasaKitchenApplianceCategory.DISHWASHER
    )
    assert any(
        isinstance(annotation, Dishwasher) for annotation in world.semantic_annotations
    )


def test_load_object(robocasa_loader):
    world = robocasa_loader.load_object(RoboCasaObjectCategory.APPLE)
    assert len(world.bodies) > 0
    annotations = world.semantic_annotations
    assert len(annotations) == 1
    assert not isinstance(annotations[0], NaturalLanguageWithTypeDescription)


def test_load_object_from_group_without_objaverse_assets(robocasa_loader):
    """
    A category with no objaverse assets still loads from another self-contained group.
    """
    world = robocasa_loader.load_object(RoboCasaObjectCategory.POT)
    assert len(world.bodies) > 0


def test_load_task_binds_instruction_and_world(robocasa_loader):
    """
    The loaded task carries its natural-language instruction bound to the scene it plays
    out in.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    task = robocasa_loader.load_task(
        "TurnOnMicrowave",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
    )
    assert "microwave" in task.instruction.lower()
    assert len(task.world.bodies) > 0
    assert len(task.world.semantic_annotations) > 0


def test_load_task_omits_robot(robocasa_loader):
    """
    The task world contains no robot, since semantic_digital_twin owns the robot.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    task = robocasa_loader.load_task(
        "TurnOnMicrowave",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
    )
    assert not any(
        body.name.name.startswith(RoboCasaDatasetLoader.robot_body_name_prefixes)
        for body in task.world.bodies
    )


def test_load_task_manipulated_objects_are_in_world(robocasa_loader):
    """
    The task's manipulated objects are resolved to bodies present in the task world.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    task = robocasa_loader.load_task(
        "TurnOnMicrowave",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
    )
    assert len(task.manipulated_objects) >= 1
    for manipulated_object in task.manipulated_objects:
        assert manipulated_object in task.world.bodies


def test_load_task_names_the_fixtures_the_task_refers_to(robocasa_loader):
    """
    The fixtures a task refers to, such as the drawer to open, are given by the name
    the task refers to them by, as their bodies in the loaded world.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    environment = robocasa_loader.build_task_environment(
        "OpenDrawer",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
        seed=0,
    )

    task = robocasa_loader.load_task_from_environment(environment)

    assert task.referenced_fixtures["drawer"].name.name.startswith(
        environment.drawer.name
    )


def test_load_task_unknown_name_raises(robocasa_loader):
    """
    Requesting a task that is not registered raises a descriptive error.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    with pytest.raises(RoboCasaTaskNotFoundError):
        robocasa_loader.load_task(
            "ThisTaskDoesNotExist",
            layout_id=LayoutType.LAYOUT001,
            style_id=StyleType.STYLE001,
        )


def _object_positions(task) -> dict[str, list[float]]:
    """
    :return: Where each manipulated object of ``task`` stands, by its name.
    """
    return {
        body.name.name: task.world.compute_forward_kinematics_np(task.world.root, body)[
            :3, 3
        ].tolist()
        for body in task.manipulated_objects
    }


def test_a_task_composed_with_the_same_seed_places_its_objects_the_same(
    robocasa_loader,
):
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    first, second = (
        robocasa_loader.load_task(
            "BreadAndCheese",
            layout_id=LayoutType.LAYOUT001,
            style_id=StyleType.STYLE001,
            seed=6,
        )
        for _ in range(2)
    )

    assert _object_positions(first) == _object_positions(second)


def test_a_task_world_weighs_its_objects_as_robocasa_simulates_them(robocasa_loader):
    """
    Weighed from its geoms alone, an object would also weigh the geoms RoboCasa only
    draws or marks regions with.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    environment = robocasa_loader.build_task_environment(
        "BreadAndCheese",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
        seed=6,
    )
    model = environment.sim.model

    task = robocasa_loader.load_task_from_environment(environment)

    assert {
        body.name.name: body.inertial.mass for body in task.manipulated_objects
    } == pytest.approx(
        {
            body.name.name: float(model.body_mass[model.body_name2id(body.name.name)])
            for body in task.manipulated_objects
        }
    )


def test_a_task_world_marks_the_kitchen_floor_as_a_floor(robocasa_loader):
    """
    A robot drives on the floor rather than around it, so only a floor annotated as one
    leaves room to stand on.
    """
    from robocasa.models.fixtures.others import Floor as RoboCasaFloor
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    environment = robocasa_loader.build_task_environment(
        "BreadAndCheese",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
        seed=6,
    )
    floor_names = {
        fixture.root_body
        for fixture in environment.fixtures.values()
        if isinstance(fixture, RoboCasaFloor)
    }

    task = robocasa_loader.load_task_from_environment(environment)

    assert floor_names
    assert {
        annotation.root.name.name
        for annotation in task.world.get_semantic_annotations_by_type(Floor)
    } == floor_names


def test_load_task_keeps_the_joints_where_robocasa_set_them(robocasa_loader):
    """
    A drawer RoboCasa opens to set a task up stands open in the loaded world, as far
    as RoboCasa opened it.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    environment = robocasa_loader.build_task_environment(
        "PickPlaceCounterToDrawer",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
    )
    model, data = environment.sim.model, environment.sim.data
    [drawer_joint] = [
        model.joint_id2name(joint)
        for joint in range(model.njnt)
        if environment.drawer.name in (model.joint_id2name(joint) or "")
    ]
    opened_to = float(data.qpos[model.jnt_qposadr[model.joint_name2id(drawer_joint)]])

    task = robocasa_loader.load_task_from_environment(environment)

    assert task.world.get_connection_by_name(drawer_joint).position == pytest.approx(
        opened_to
    )


# %% RoboCasa's own success check


class CounterToSinkBody(StrEnum):
    """
    Bodies RoboCasa places for ``PickPlaceCounterToSink``, by the names it gives them.
    """

    OBJECT = "obj_main"
    """
    The object to put into the sink.
    """

    IN_THE_SINK = "distr_sink_main"
    """
    The distractor RoboCasa places inside the sink.
    """


@pytest.fixture
def counter_to_sink(robocasa_loader):
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    return robocasa_loader.load_task(
        "PickPlaceCounterToSink",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
        seed=0,
    )


def put_the_object_into_the_sink(task) -> None:
    """
    Move the task's object to where RoboCasa placed the distractor inside the sink.
    """
    world = task.world
    world.get_body_by_name(
        CounterToSinkBody.OBJECT
    ).parent_connection.origin = world.compute_forward_kinematics(
        world.root, world.get_body_by_name(CounterToSinkBody.IN_THE_SINK)
    )


def far_from_the_objects(task) -> Point3:
    """
    :return: Where RoboCasa places its robot's base on the floor, far from every object.
    """
    return task.robot_base_pose.position


def test_robocasa_does_not_count_the_object_left_on_the_counter_as_success(
    counter_to_sink,
):
    assert not counter_to_sink.succeeds(far_from_the_objects(counter_to_sink))


def test_robocasa_counts_the_object_in_the_sink_and_let_go_as_success(
    counter_to_sink,
):
    put_the_object_into_the_sink(counter_to_sink)

    assert counter_to_sink.succeeds(far_from_the_objects(counter_to_sink))


def test_robocasa_does_not_count_the_object_in_the_sink_still_held_as_success(
    counter_to_sink,
):
    put_the_object_into_the_sink(counter_to_sink)
    world = counter_to_sink.world

    assert not counter_to_sink.succeeds(
        world.get_body_by_name(CounterToSinkBody.OBJECT).global_pose.position
    )
