"""
The Toyota HSR performs RoboCasa's composite tasks in a RoboCasa kitchen, simulated in
MuJoCo: tasks that string several of the skills of RoboCasa's atomic tasks together.

Run the snack task with::

    python -m experiments.hsr_robocasa.composite
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.robocasa_dataset.mujoco_compat import (
    robocasa_version_assertions_relaxed,
)
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Sink,
)
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.world_description.world_entity import Body

from experiments.hsr_robocasa.pick_place import HSRPickPlaceTask, TaskObject
from experiments.hsr_robocasa.robocasa_task import FixtureReference

with robocasa_version_assertions_relaxed():
    from robocasa.models.objects.kitchen_objects import get_cats_by_type
    from robocasa.models.scenes.scene_registry import LayoutType

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


# %% organizing recycling


class RecycleBottle(StrEnum):
    """
    The bodies RoboCasa adds for the recycling task, by the names it gives them.
    """

    PLASTIC_SAMPLE = "bottle_plastic1_main"
    """
    A plastic bottle of the group at one end of the counter.
    """

    SECOND_PLASTIC_SAMPLE = "bottle_plastic2_main"
    """
    The other plastic bottle of that group.
    """

    GLASS_SAMPLE = "bottle_glass1_main"
    """
    A glass bottle of the group at the other end.
    """

    SECOND_GLASS_SAMPLE = "bottle_glass2_main"
    """
    The other glass bottle of that group.
    """

    PLASTIC_MIDDLE = "bottle_plastic_middle_main"
    """
    The plastic bottle in the middle, to go to the plastic group.
    """

    GLASS_MIDDLE = "bottle_glass_middle_main"
    """
    The glass bottle in the middle, to go to the glass group.
    """

    MYSTERY_MIDDLE = "mystery_middle_main"
    """
    The bottle in the middle of a kind RoboCasa draws at random, to go to the group
    of its kind.
    """


@dataclass
class RecycleBottlesByType(HSRPickPlaceTask):
    """
    Move the bottles in the middle of the dining counter to the group of bottles of
    their kind, plastic or glass: RoboCasa's composite task of its organizing
    recycling activity.
    """

    layout: LayoutType = LayoutType.LAYOUT002
    """
    A kitchen with a dining counter and three stools, which the task needs.
    """

    seed: int = 1
    """
    The seed RoboCasa samples the task with: middle bottles of 7 cm to 8 cm across,
    within the hand's reach.
    """

    @classproperty
    def cluster_threshold(cls) -> float:
        """
        How near, in metres, a middle bottle has to stand to a bottle of its kind of
        the group for RoboCasa to count it in that group.
        """
        return 0.30

    @property
    def moved_objects(self) -> list[StrEnum]:
        """
        The bottles in the middle.
        """
        return [
            RecycleBottle.PLASTIC_MIDDLE,
            RecycleBottle.GLASS_MIDDLE,
            RecycleBottle.MYSTERY_MIDDLE,
        ]

    def is_glass(self, bottle: StrEnum) -> bool:
        """
        :return: Whether the bottle named ``bottle`` is of a category RoboCasa counts
            as alcohol, which comes in glass; the others are bottled water.
        """
        body = self.world.get_body_by_name(bottle)
        [category] = [
            annotation.description
            for annotation in self.world.get_semantic_annotations_by_type(
                NaturalLanguageWithTypeDescription
            )
            if annotation.root is body
        ]
        return category in get_cats_by_type(["alcohol"])

    @property
    def samples_of_its_kind(self) -> list[StrEnum]:
        """
        The bottles of the group of the kind of the bottle being moved.
        """
        if self.is_glass(self.moving):
            return [RecycleBottle.GLASS_SAMPLE, RecycleBottle.SECOND_GLASS_SAMPLE]
        return [RecycleBottle.PLASTIC_SAMPLE, RecycleBottle.SECOND_PLASTIC_SAMPLE]

    @property
    def receptacle(self) -> Body:
        """
        The dining counter.
        """
        return self.task.referenced_fixtures[FixtureReference.DINING_COUNTER]

    @property
    def bottle_radius(self) -> float:
        """
        Half the bottle being moved's larger width, in metres.
        """
        extent = self.extent(self.food.root)
        return max(extent.width, extent.depth) / 2

    @property
    def release_point(self) -> np.ndarray:
        """
        Where the centre of the bottle is to be let go: its underside
        :attr:`release_clearance` above the counter, at the free spot of the counter's
        top nearest a bottle of its kind of the group, see
        :attr:`samples_of_its_kind`, that keeps the bottle's radius and the hand's half
        width clear of every other bottle.
        """
        counter = [
            annotation
            for annotation in self.world.get_semantic_annotations_by_type(CounterTop)
            if annotation.root is self.receptacle
        ][0]
        extent = self.extent(self.food.root)
        free_top = counter.planar_free_space(
            max_height=extent.height,
            bloat_obstacles=self.bottle_radius + self.hand_half_width,
        )
        spots = []
        for sample in self.samples_of_its_kind:
            sample_middle = self.extent(
                self.world.get_body_by_name(sample)
            ).center.to_np()[:2]
            spot = self.world.transform(
                free_top.nearest_free_point(
                    Point2(*sample_middle, reference_frame=self.world.root)
                ),
                self.world.root,
            ).to_np()[:2]
            spots.append((np.linalg.norm(spot - sample_middle), spot))
        _, spot = min(spots, key=lambda distance_and_spot: distance_and_spot[0])
        return np.array(
            [
                *spot,
                self.release_height + extent.height / 2 + self.release_clearance,
            ]
        )


if __name__ == "__main__":
    experiment = BreadAndCheese()
    experiment.build_world()
    experiment.run()
