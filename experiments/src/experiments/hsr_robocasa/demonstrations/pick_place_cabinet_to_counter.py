"""
RoboCasa's PickPlaceCabinetToCounter, performed by the HSR: pick the object from the
cabinet and place it on the counter.

  python -m experiments.hsr_robocasa.demonstrations.pick_place_cabinet_to_counter [COUNTER_HEIGHT]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum

from semantic_digital_twin.world_description.world_entity import Body

from experiments.hsr_robocasa.pick_place import HSRPickPlaceTask

# %% the task


class FixtureReference(StrEnum):
    """
    The fixtures the task refers to, by the names it gives them.
    """

    COUNTER = "counter"
    """
    The counter the object is placed on.
    """


@dataclass
class PickPlaceCabinetToCounter(HSRPickPlaceTask):
    """
    Pick the object from the cabinet and place it on the counter.
    """

    @property
    def receptacle(self) -> Body:
        return self.task.referenced_fixtures[FixtureReference.COUNTER]


# %% running it

if __name__ == "__main__":
    PickPlaceCabinetToCounter.from_command_line(sys.argv[1:]).run()
