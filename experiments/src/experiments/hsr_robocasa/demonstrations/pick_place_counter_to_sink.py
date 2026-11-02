"""
RoboCasa's PickPlaceCounterToSink, performed by the HSR: pick the object from the
counter and place it in the sink.

  python -m experiments.hsr_robocasa.demonstrations.pick_place_counter_to_sink [COUNTER_HEIGHT]
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

    SINK = "sink"
    """
    The sink the object is placed in.
    """


@dataclass
class PickPlaceCounterToSink(HSRPickPlaceTask):
    """
    Pick the object from the counter and place it in the sink.
    """

    @property
    def receptacle(self) -> Body:
        return self.task.referenced_fixtures[FixtureReference.SINK]


# %% running it

if __name__ == "__main__":
    PickPlaceCounterToSink.from_command_line(sys.argv[1:]).run()
