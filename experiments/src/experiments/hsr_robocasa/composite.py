"""
The Toyota HSR performs RoboCasa's composite tasks in a RoboCasa kitchen, simulated in
MuJoCo: tasks that string several of the skills of RoboCasa's atomic tasks together.

Run the snack task with::

    python -m experiments.hsr_robocasa.composite
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from semantic_digital_twin.semantic_annotations.semantic_annotations import Sink
from semantic_digital_twin.world_description.world_entity import Body

from experiments.hsr_robocasa.pick_place import HSRPickPlaceTask, TaskObject

# %% snack preparation


@dataclass
class BreadAndCheese(HSRPickPlaceTask):
    """
    Place the bread and the cheese on the cutting board: RoboCasa's composite task of
    its snack preparation activity.
    """

    seed: int = 6
    """
    The seed RoboCasa samples the task with: a bread roll and a block of cheese at the
    counter's edge left of a cutting board 35 cm wide.
    """

    @property
    def moved_objects(self) -> list[TaskObject]:
        """
        The bread, then the cheese.
        """
        return [TaskObject.FOOD, TaskObject.SECOND_OBJECT]


# %% defrosting food


class DefrostObject(StrEnum):
    """
    The bodies RoboCasa adds for the defrosting task, by the names it gives them.
    """

    FIRST_FRUIT = "obj0_main"
    """
    A fruit of the pile, to go into the sink.
    """

    SECOND_FRUIT = "obj1_main"
    """
    The other fruit of the pile, to go into the sink.
    """

    FIRST_VEGETABLE = "obj2_main"
    """
    A vegetable of the pile, to go into the bowl.
    """

    SECOND_VEGETABLE = "obj3_main"
    """
    The other vegetable of the pile, to go into the bowl.
    """


@dataclass
class DefrostByCategory(HSRPickPlaceTask):
    """
    Put the frozen fruits into the sink to defrost and the frozen vegetables into the
    bowl: RoboCasa's composite task of its defrosting food activity.
    """

    seed: int = 0
    """
    The seed RoboCasa samples the task with: two apples, broccoli and a lime in a pile
    left of the sink, the bowl right of it.
    """

    @property
    def fruits(self) -> list[StrEnum]:
        """
        The fruits, which go into the sink.
        """
        return [DefrostObject.FIRST_FRUIT, DefrostObject.SECOND_FRUIT]

    @property
    def vegetables(self) -> list[StrEnum]:
        """
        The vegetables, which go into the bowl.
        """
        return [DefrostObject.FIRST_VEGETABLE, DefrostObject.SECOND_VEGETABLE]

    @property
    def moved_objects(self) -> list[StrEnum]:
        """
        The fruits, then the vegetables.
        """
        return self.fruits + self.vegetables

    @property
    def moved_together(self) -> list[StrEnum]:
        """
        The fruits, while a fruit is moved; the vegetables otherwise.
        """
        return self.fruits if self.moving in self.fruits else self.vegetables

    @property
    def receptacle(self) -> Body:
        """
        The sink for a fruit, the bowl for a vegetable.
        """
        if self.moving in self.fruits:
            return self.world.get_semantic_annotations_by_type(Sink)[0].root
        return self.world.get_body_by_name(TaskObject.CONTAINER)

    @property
    def release_height(self) -> float:
        """
        For a fruit, the sink's rim, level with the counter: a fruit is dropped into
        the sink from just above it; for a vegetable, the bowl's floor.
        """
        if self.moving in self.fruits:
            return self.counter_top_height
        return super().release_height


if __name__ == "__main__":
    experiment = BreadAndCheese()
    experiment.build_world()
    experiment.run()
