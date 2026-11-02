"""
RoboCasa's PickPlaceCounterToMicrowave, performed by the HSR: pick the object from the
counter and place it on the plate in the microwave.

  python -m experiments.hsr_robocasa.demonstrations.pick_place_counter_to_microwave [COUNTER_HEIGHT]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from semantic_digital_twin.world_description.world_entity import Body

from experiments.hsr_robocasa.pick_place import HSRPickPlaceTask, TaskObject

# %% the task


@dataclass
class PickPlaceCounterToMicrowave(HSRPickPlaceTask):
    """
    Pick the object from the counter and place it on the plate in the microwave.
    """

    @property
    def receptacle(self) -> Body:
        return self.task_object(TaskObject.CONTAINER)


# %% running it

if __name__ == "__main__":
    PickPlaceCounterToMicrowave.from_command_line(sys.argv[1:]).run()
