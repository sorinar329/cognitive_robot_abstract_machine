from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import yaml
from typing_extensions import Any

from semantic_digital_twin.adapters.robocasa_dataset.mujoco_compat import (
    robocasa_version_assertions_relaxed,
)
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
)

with robocasa_version_assertions_relaxed():
    from robocasa.models.scenes.scene_registry import LayoutType, get_layout_path


# %% the keys of a layout file


class RoboCasaLayoutKey(StrEnum):
    """
    The keys of a RoboCasa layout file that a :class:`RoboCasaLayout` reads.
    """

    NAME = "name"
    """
    A fixture's name, by which a fixture update refers to it.
    """

    TYPE = "type"
    """
    The kind of a fixture, such as a counter.
    """

    SIZE = "size"
    """
    A fixture's width, depth and height.
    """

    POSITION = "pos"
    """
    Where a fixture stands: along the wall, out from it and up.
    """

    STACK_HEIGHT = "stack_height"
    """
    How high the fixtures below a stacked fixture reach.
    """

    ROOM = "room"
    """
    The group of the walls, the floor and what hangs on the walls.
    """

    BOTTOM_ROW = "bottom_row"
    """
    The row of counters, sinks and stoves of a group.
    """

    TOP_ROW = "top_row"
    """
    The row of cabinets a group hangs on the wall, and the start of the name of every
    further such row.
    """


# %% the layout


@dataclass
class RoboCasaLayout:
    """
    A RoboCasa kitchen layout, as its file states it.
    """

    configuration: dict[str, Any]
    """
    The layout file's content: groups of rows of fixtures, and the room.
    """

    @classmethod
    def from_layout_type(cls, layout: LayoutType) -> RoboCasaLayout:
        """
        :param layout: One of RoboCasa's layouts.
        :return: That layout, read from RoboCasa's file of it.
        """
        with open(get_layout_path(layout)) as layout_file:
            return cls(yaml.safe_load(layout_file))

    @property
    def floor_standing_fixtures(self) -> list[dict[str, Any]]:
        """
        The fixtures standing on the floor: those of every row of every group, except
        the room and the cabinets hanging on the walls.
        """
        return [
            fixture
            for group_name, group in self.configuration.items()
            if group_name != RoboCasaLayoutKey.ROOM and isinstance(group, dict)
            for row_name, row in group.items()
            if not row_name.startswith(RoboCasaLayoutKey.TOP_ROW)
            and isinstance(row, list)
            for fixture in row
            if isinstance(fixture, dict)
        ]

    @property
    def counter_height(self) -> float:
        """
        How high the kitchen's lowest counters stand, those it is worked at; a bar or
        an island may stand higher.
        """
        return min(
            fixture[RoboCasaLayoutKey.SIZE][2]
            for fixture in self.floor_standing_fixtures
            if fixture[RoboCasaLayoutKey.TYPE]
            == RoboCasaKitchenApplianceCategory.COUNTER
        )

    def fixture_updates_for_counter_height(
        self, counter_height: float
    ) -> dict[str, dict[str, Any]]:
        """
        :param counter_height: How high the lowest counters are to stand.
        :return: The fixture updates, by fixture name, that scale every height of the
            :attr:`floor_standing_fixtures` by the same factor, so that the lowest
            counters stand ``counter_height`` high. Every other fixture keeps its
            height, and every fixture keeps where it stands along and out from the wall.
        """
        factor = counter_height / self.counter_height
        return {
            fixture[RoboCasaLayoutKey.NAME]: update
            for fixture in self.floor_standing_fixtures
            if (update := self._heights_scaled(fixture, factor))
        }

    @staticmethod
    def _heights_scaled(fixture: dict[str, Any], factor: float) -> dict[str, Any]:
        """
        :param fixture: A fixture of the layout.
        :param factor: What to multiply each of its heights with.
        :return: Those of the fixture's size, position and stack height that state a
            height, with that height scaled.
        """
        update = {}
        for key in (RoboCasaLayoutKey.SIZE, RoboCasaLayoutKey.POSITION):
            value = fixture.get(key)
            if isinstance(value, list) and isinstance(value[2], (int, float)):
                update[key] = [value[0], value[1], value[2] * factor]
        if RoboCasaLayoutKey.STACK_HEIGHT in fixture:
            update[RoboCasaLayoutKey.STACK_HEIGHT] = (
                fixture[RoboCasaLayoutKey.STACK_HEIGHT] * factor
            )
        return update
