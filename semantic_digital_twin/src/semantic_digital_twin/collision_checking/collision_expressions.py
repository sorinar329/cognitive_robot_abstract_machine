from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
    Collision,
)
from semantic_digital_twin.collision_checking.collision_manager import (
    CollisionGroupConsumer,
    CollisionGroup,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import Vector3, Point3
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@dataclass
class ExternalCollisionExpressionManager(CollisionGroupConsumer):
    """
    Owns symbols and buffer
    """

    registered_bodies: dict[KinematicStructureEntity, int] = field(
        default_factory=dict, init=False
    )
    """
    Maps bodies to the index of point_on_body_a in the collision buffer.
    """

    collision_data: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float), init=False
    )
    """
    All collision data in a single numpy array.
    Repeats blocks of size block_size.
    """

    active_groups: set[CollisionGroup] = field(default_factory=set, init=False)

    block_size: int = field(default=9, init=False)
    """
    block layout:
        9 per collision
        point_on_body_a,  (3)
        contact_normal,  (3)
        contact_distance, (1)
        buffer_distance,  (1)
        violated_distance (1)
    """
    _point_on_a_offset: int = field(init=False, default=0)
    _contact_normal_offset: int = field(init=False, default=3)
    _contact_distance_offset: int = field(init=False, default=6)
    _buffer_distance_offset: int = field(init=False, default=7)
    _violated_distance_offset: int = field(init=False, default=8)

    def __hash__(self):
        return hash(id(self))

    def on_reset(self):
        pass

    def on_collision_matrix_update(self):
        pass

    def on_compute_collisions(self, collision: CollisionCheckingResult):
        """
        Takes collisions, checks if they are external, and inserts them
        into the buffer at the right place.
        """
        closest_contacts: dict[Body, list[Collision]] = defaultdict(list)
        for collision in collision.contacts:
            # 1. check if collision is external
            if (
                collision.body_a not in self.registered_bodies
                and collision.body_b not in self.registered_bodies
            ):
                continue
            if collision.body_a not in self.registered_bodies:
                collision = collision.reverse()
            closest_contacts[collision.body_a].append(collision)

        for body_a, collisions in closest_contacts.items():
            collisions = sorted(collisions, key=lambda c: c.contact_distance)
            for i in range(
                min(
                    len(collisions),
                    self.collision_manager.get_max_avoided_bodies(body_a),
                )
            ):
                collision = collisions[i]
                group1 = self.get_collision_group(collision.body_a)
                group1_T_root = group1.root.global_pose.inverse().to_np()
                group1_P_pa = group1_T_root @ collision.root_P_pa
                self.insert_data_block(
                    body=group1.root,
                    idx=i,
                    group1_P_point_on_a=group1_P_pa,
                    root_V_contact_normal=collision.root_V_n,
                    contact_distance=collision.contact_distance,
                    buffer_distance=self.collision_manager.get_buffer_zone_distance(
                        collision.body_a, collision.body_b
                    ),
                    violated_distance=self.collision_manager.get_violated_distance(
                        collision.body_a, collision.body_b
                    ),
                )

    def insert_data_block(
        self,
        body: KinematicStructureEntity,
        idx: int,
        group1_P_point_on_a: np.ndarray,
        root_V_contact_normal: np.ndarray,
        contact_distance: float,
        buffer_distance: float,
        violated_distance: float,
    ):
        start_idx = self.registered_bodies[body] + idx * self.block_size
        self.collision_data[start_idx : start_idx + self._contact_normal_offset] = (
            group1_P_point_on_a[:3]
        )
        self.collision_data[
            start_idx
            + self._contact_normal_offset : start_idx
            + self._contact_distance_offset
        ] = root_V_contact_normal[:3]
        self.collision_data[start_idx + self._contact_distance_offset] = (
            contact_distance
        )
        self.collision_data[start_idx + self._buffer_distance_offset] = buffer_distance
        self.collision_data[start_idx + self._violated_distance_offset] = (
            violated_distance
        )

    def register_body(self, body: Body):
        """
        Register a body
        """
        self.registered_bodies[body] = len(self.collision_data)
        self.collision_data = np.resize(
            self.collision_data,
            self.collision_data.shape[0]
            + self.block_size * self.collision_manager.get_max_avoided_bodies(body),
        )
        for group in self.collision_groups:
            if body in group:
                self.active_groups.add(group)

    def get_collision_variables(self) -> list[FloatVariable]:
        """

        :return: A list of all external collision variables for registered bodies.
        """
        variables = []
        for body in self.registered_bodies:
            group = self.get_collision_group(body)
            for index in range(group.get_max_avoided_bodies(body)):
                group1_P_point_on_a = self.get_group1_P_point_on_a_symbol(body, index)
                variables.append(group1_P_point_on_a.x.free_variables()[0])
                variables.append(group1_P_point_on_a.y.free_variables()[0])
                variables.append(group1_P_point_on_a.z.free_variables()[0])

                group1_V_contact_normal = self.get_group1_V_contact_normal_symbol(
                    body, index
                )
                variables.append(group1_V_contact_normal.x.free_variables()[0])
                variables.append(group1_V_contact_normal.y.free_variables()[0])
                variables.append(group1_V_contact_normal.z.free_variables()[0])

                contact_distance = self.get_contact_distance_symbol(body, index)
                variables.append(contact_distance.free_variables()[0])
                buffer_distance = self.get_buffer_distance_symbol(body, index)
                variables.append(buffer_distance.free_variables()[0])
                violated_distance = self.get_violated_distance_symbol(body, index)
                variables.append(violated_distance.free_variables()[0])
        return variables

    def get_group1_P_point_on_a_data(
        self, body: KinematicStructureEntity, idx: int
    ) -> np.ndarray:
        start = (
            self.registered_bodies[body]
            + idx * self.block_size
            + self._point_on_a_offset
        )
        end = start + 3
        return self.collision_data[start:end]

    @lru_cache
    def get_group1_P_point_on_a_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> Point3:
        return Point3.create_with_variables(
            name=f"group1_P_point_on_a({body.name}, {idx})",
            resolver=lambda: self.get_group1_P_point_on_a_data(body, idx),
        )

    def get_group1_V_contact_normal_data(
        self, body: KinematicStructureEntity, idx: int
    ) -> np.ndarray:
        start = (
            self.registered_bodies[body]
            + idx * self.block_size
            + self._contact_normal_offset
        )
        end = start + 3
        return self.collision_data[start:end]

    @lru_cache
    def get_group1_V_contact_normal_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> Vector3:
        return Vector3.create_with_variables(
            f"group1_V_contact_normal({body.name}, {idx})",
            resolver=lambda: self.get_group1_V_contact_normal_data(body, idx),
        )

    def get_contact_distance_data(
        self, body: KinematicStructureEntity, idx: int
    ) -> float:
        start = (
            self.registered_bodies[body]
            + idx * self.block_size
            + self._contact_distance_offset
        )
        end = start + 1
        return self.collision_data[start:end][0]

    @lru_cache
    def get_contact_distance_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> FloatVariable:
        return FloatVariable.create_with_resolver(
            f"contact_distance({body.name}, {idx})",
            resolver=lambda: self.get_contact_distance_data(body, idx),
        )

    def get_buffer_distance_data(
        self, body: KinematicStructureEntity, idx: int
    ) -> float:
        start = (
            self.registered_bodies[body]
            + idx * self.block_size
            + self._buffer_distance_offset
        )
        end = start + 1
        return self.collision_data[start:end][0]

    @lru_cache
    def get_buffer_distance_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> FloatVariable:
        return FloatVariable.create_with_resolver(
            f"buffer_distance({body.name}, {idx})",
            resolver=lambda: self.get_buffer_distance_data(body, idx),
        )

    def get_violated_distance_data(
        self, body: KinematicStructureEntity, idx: int
    ) -> float:
        start = (
            self.registered_bodies[body]
            + idx * self.block_size
            + self._violated_distance_offset
        )
        end = start + 1
        return self.collision_data[start:end][0]

    @lru_cache
    def get_violated_distance_symbol(
        self, body: KinematicStructureEntity, idx: int
    ) -> FloatVariable:
        return FloatVariable.create_with_resolver(
            f"violated_distance({body.name}, {idx})",
            resolver=lambda: self.get_violated_distance_data(body, idx),
        )
