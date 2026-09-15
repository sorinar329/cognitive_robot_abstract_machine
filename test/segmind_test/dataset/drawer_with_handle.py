"""
A drawer with a handle on a slider, the smallest scene holding a part that opens and
closes.
"""

from __future__ import annotations

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
    Slider,
)
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale


def drawer_with_handle() -> tuple[World, Drawer]:
    """
    :return: A world holding one drawer, with its handle and a slider along x, and the
        drawer.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        drawer = Drawer.create_with_new_body_in_world(
            name="drawer", scale=Scale(0.2, 0.3, 0.2), world=world
        )
        handle = Handle.create_with_new_body_in_world(name="handle", world=world)
        slider = Slider.create_with_new_body_in_world(
            name="slider",
            world=world,
            parent_connection_specification=Slider.parent_connection_specification(
                axis=Vector3.X()
            ),
        )
        drawer.add(handle)
        drawer.add(slider)
    return world, drawer
