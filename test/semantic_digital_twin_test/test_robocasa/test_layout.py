import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from semantic_digital_twin.adapters.robocasa_dataset.layout import (
    RoboCasaLayout,
    RoboCasaLayoutKey,
)
from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
)
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
)


@pytest.fixture(scope="session")
def robocasa_loader() -> RoboCasaDatasetLoader:
    loader = RoboCasaDatasetLoader()
    if not loader.directory.exists():
        pytest.skip(
            "RoboCasa assets not downloaded. Run "
            "'python -m robocasa.scripts.download_kitchen_assets' first."
        )
    return loader


@pytest.fixture
def layout() -> RoboCasaLayout:
    return RoboCasaLayout.from_layout_type(LayoutType.LAYOUT001)


def _fixtures_of_rows(layout: RoboCasaLayout, row_names: set[str]) -> list[dict]:
    """
    :return: The fixtures in the rows named ``row_names`` of every group of ``layout``.
    """
    return [
        fixture
        for group in layout.configuration.values()
        if isinstance(group, dict)
        for row_name, row in group.items()
        if row_name in row_names
        for fixture in row
    ]


# %% the layout as RoboCasa's file states it


def test_the_counter_height_is_that_of_the_lowest_counters():
    """
    Layout 11 has a bar counter standing higher than the counters it is worked at.
    """
    layout = RoboCasaLayout.from_layout_type(LayoutType.LAYOUT011)
    counter_heights = {
        fixture[RoboCasaLayoutKey.SIZE][2]
        for fixture in layout.floor_standing_fixtures
        if fixture[RoboCasaLayoutKey.TYPE] == RoboCasaKitchenApplianceCategory.COUNTER
    }

    assert len(counter_heights) > 1
    assert layout.counter_height == min(counter_heights)


@pytest.mark.parametrize(
    "layout_type", [layout_type for layout_type in LayoutType if layout_type >= 0]
)
def test_every_layout_can_be_lowered(layout_type):
    """
    Not every layout has the main group the first layouts have.
    """
    layout = RoboCasaLayout.from_layout_type(layout_type)

    updates = layout.fixture_updates_for_counter_height(0.75)

    assert min(
        updates[fixture[RoboCasaLayoutKey.NAME]][RoboCasaLayoutKey.SIZE][2]
        for fixture in layout.floor_standing_fixtures
        if fixture[RoboCasaLayoutKey.TYPE] == RoboCasaKitchenApplianceCategory.COUNTER
    ) == pytest.approx(0.75)


# %% lowering the kitchen


def test_the_updates_put_every_counter_at_the_asked_height(layout):
    counters = [
        fixture
        for fixture in _fixtures_of_rows(layout, {RoboCasaLayoutKey.BOTTOM_ROW})
        if fixture[RoboCasaLayoutKey.TYPE] == RoboCasaKitchenApplianceCategory.COUNTER
    ]

    updates = layout.fixture_updates_for_counter_height(0.75)

    assert [
        updates[counter[RoboCasaLayoutKey.NAME]][RoboCasaLayoutKey.SIZE][2]
        for counter in counters
    ] == pytest.approx([0.75] * len(counters))


def test_the_updates_scale_where_a_floor_standing_fixture_stands_by_the_same_factor(
    layout,
):
    counter = next(
        fixture
        for fixture in _fixtures_of_rows(layout, {RoboCasaLayoutKey.BOTTOM_ROW})
        if fixture[RoboCasaLayoutKey.TYPE] == RoboCasaKitchenApplianceCategory.COUNTER
    )
    factor = 0.75 / layout.counter_height

    update = layout.fixture_updates_for_counter_height(0.75)[
        counter[RoboCasaLayoutKey.NAME]
    ]

    assert update[RoboCasaLayoutKey.POSITION] == pytest.approx(
        [
            *counter[RoboCasaLayoutKey.POSITION][:2],
            counter[RoboCasaLayoutKey.POSITION][2] * factor,
        ]
    )


def test_the_updates_leave_the_room_and_the_wall_cabinets_as_they_are(layout):
    unchanged = [
        fixture[RoboCasaLayoutKey.NAME]
        for fixture in _fixtures_of_rows(layout, {RoboCasaLayoutKey.TOP_ROW})
    ] + [
        fixture[RoboCasaLayoutKey.NAME]
        for row in layout.configuration[RoboCasaLayoutKey.ROOM].values()
        for fixture in row
    ]

    updates = layout.fixture_updates_for_counter_height(0.75)

    assert unchanged
    assert set(unchanged).isdisjoint(updates)


def test_the_objects_of_a_task_in_a_lowered_kitchen_rest_on_the_lowered_counter(
    robocasa_loader, layout
):
    """
    RoboCasa places a task's objects itself, so they end up on the counters wherever
    those stand.
    """
    task = robocasa_loader.load_task(
        "BreadAndCheese",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
        seed=6,
        fixture_updates=layout.fixture_updates_for_counter_height(0.75),
    )

    lowest_points = [
        body.collision.as_bounding_box_collection_in_frame(task.world.root)
        .bounding_box()
        .min_z
        for body in task.manipulated_objects
    ]

    assert lowest_points == pytest.approx([0.75] * len(lowest_points), abs=0.01)
