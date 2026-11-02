"""
RoboCasa's PickPlaceSinkToCounter, performed by the HSR: pick the object from the
sink and place it on the plate on the counter.

  python -m experiments.hsr_robocasa.demonstrations.pick_place_sink_to_counter [COUNTER_HEIGHT]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from semantic_digital_twin.world_description.world_entity import Body

from experiments.hsr_robocasa.pick_place import HSRPickPlaceTask, TaskObject

# %% the task


@dataclass
class PickPlaceSinkToCounter(HSRPickPlaceTask):
    """
    Pick the object from the sink and place it on the plate on the counter.
    """

    @property
    def receptacle(self) -> Body:
        return self.task_object(TaskObject.CONTAINER)


# %% running it

if __name__ == "__main__":
    PickPlaceSinkToCounter.from_command_line(sys.argv[1:]).run()
