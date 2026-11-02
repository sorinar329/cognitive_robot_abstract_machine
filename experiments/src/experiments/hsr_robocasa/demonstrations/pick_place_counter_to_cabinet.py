"""
RoboCasa's PickPlaceCounterToCabinet, performed by the HSR: pick the object from the
counter and place it in the cabinet.

  python -m experiments.hsr_robocasa.demonstrations.pick_place_counter_to_cabinet [COUNTER_HEIGHT]
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

    CABINET = "cab"
    """
    The cabinet the object is placed in.
    """


@dataclass
class PickPlaceCounterToCabinet(HSRPickPlaceTask):
    """
    Pick the object from the counter and place it in the cabinet.
    """

    @property
    def receptacle(self) -> Body:
        return self.task.referenced_fixtures[FixtureReference.CABINET]


# %% running it

if __name__ == "__main__":
    PickPlaceCounterToCabinet.from_command_line(sys.argv[1:]).run()
