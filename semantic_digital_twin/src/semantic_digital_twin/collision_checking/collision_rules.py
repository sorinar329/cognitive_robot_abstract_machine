from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement

import numpy as np
from lxml import etree
from typing_extensions import (
    List,
    TYPE_CHECKING,
    Self,
    Iterable,
    Callable,
)

from .collision_matrix import (
    CollisionRule,
    CollisionMatrix,
    CollisionCheck,
)
from ..robots.abstract_robot import AbstractRobot

if TYPE_CHECKING:
    from ..world import World
    from ..world_description.world_entity import Body


@dataclass
class AvoidCollisionRule(CollisionRule, ABC):
    """
    Base class for collision rules that add collision checks to the collision matrix.
    """

    buffer_zone_distance: float = field(default=0.05)
    """
    Distance defining a buffer zone around the entity. The buffer zone represents a soft boundary where
    proximity should be monitored but minor violations are acceptable.
    """

    violated_distance: float = field(default=0.0)
    """
    Critical distance threshold that must not be violated. Any proximity below this threshold represents
    a severe collision risk requiring immediate attention.
    """

    added_collision_checks: set[CollisionCheck] = field(default_factory=set, init=False)

    def applies_to(self, body_a: Body, body_b: Body) -> bool:
        """
        Returns True if the rule configures collision distances for the given body.
        """
        return (
            CollisionCheck(body_a, body_b) in self.added_collision_checks
            or CollisionCheck(body_b, body_a) in self.added_collision_checks
        )

    def buffer_zone_distance_for(self, body_a: Body, body_b: Body) -> float | None:
        """
        Returns the configured buffer-zone distance for the body or None if not applicable.
        """
        return self.buffer_zone_distance if self.applies_to(body_a, body_b) else None

    def violated_distance_for(self, body_a: Body, body_b: Body) -> float | None:
        """
        Returns the configured violated distance for the body or None if not applicable.
        """
        return self.violated_distance if self.applies_to(body_a, body_b) else None

    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        collision_matrix.add_collision_checks(self.added_collision_checks)


@dataclass
class AllowCollisionRule(CollisionRule, ABC):
    """
    Base class for collision rules that remove collision checks from the collision matrix.
    """

    allowed_collision_pairs: set[CollisionCheck] = field(default_factory=set)
    """
    Set of collision checks that are allowed to occur.
    """
    allowed_collision_bodies: set[Body] = field(default_factory=set)
    """
    Set of bodies that are allowed to collide.
    """

    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        collision_matrix.remove_collision_checks(self.allowed_collision_pairs)
        collision_matrix.collision_checks = {
            collision_check
            for collision_check in collision_matrix.collision_checks
            if collision_check.body_a not in self.allowed_collision_bodies
            and collision_check.body_b not in self.allowed_collision_bodies
        }

    def update(self, world: World):
        self.allowed_collision_pairs = set()
        self.allowed_collision_bodies = set()
        super().update(world)


@dataclass
class AvoidCollisionBetweenGroups(AvoidCollisionRule):
    """
    Adds collision checks between all pairs of bodies in the given groups to the collision matrix.
    """

    body_group_a: List[Body] = field(default_factory=list)
    body_group_b: List[Body] = field(default_factory=list)

    def _update(self, world: World):
        self.added_collision_checks = set()
        for body_a in self.body_group_a:
            for body_b in self.body_group_b:
                if body_a == body_b:
                    continue
                collision_check = CollisionCheck.create_and_validate(
                    body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
                )
                self.added_collision_checks.add(collision_check)


@dataclass
class AvoidAllCollisions(AvoidCollisionRule):
    """
    Adds collision checks between all body pairs of the world managed by the rule to the collision matrix.
    """

    def _update(self, world: World):
        self.added_collision_checks = set()
        for body_a, body_b in combinations(world.bodies_with_collision, 2):
            collision_check = CollisionCheck.create_and_validate(
                body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
            )
            self.added_collision_checks.add(collision_check)


@dataclass
class AvoidExternalCollisions(AvoidCollisionRule):
    """
    Adds collision checks between all bodies managed by the rule and all bodies that do not belong to the robot.
    that are not managed by the rule.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot managed by the rule.
    """
    body_subset: set[Body] | None = field(default=None)
    """
    A subset of bodies managed by the rule. 
    All of them must belong to `robot`.
    If None, all robot bodies are used.
    """

    def _update(self, world: World):
        self.added_collision_checks = set()
        if self.body_subset is None:
            self.body_subset = set(self.robot.bodies_with_collision)
        external_bodies = set(world.bodies_with_collision) - set(self.body_subset)
        for body_a in self.body_subset:
            for body_b in external_bodies:
                collision_check = CollisionCheck.create_and_validate(
                    body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
                )
                self.added_collision_checks.add(collision_check)


@dataclass
class AvoidSelfCollisions(AvoidCollisionRule):
    """
    Adds collision checks between all body pairs of the robot managed by the rule.
    """

    robot: AbstractRobot = field(kw_only=True)

    def _update(self, world: World):
        self.added_collision_checks = set(
            CollisionCheck.create_and_validate(
                body_a, body_b, distance=self.buffer_zone_distance
            )
            for body_a, body_b in combinations(self.robot.bodies_with_collision, 2)
        )


@dataclass
class AllowAllCollisions(AllowCollisionRule):
    """
    Removed all collision checks from the collision matrix.
    """

    world: World = field(kw_only=True)

    def _update(self, world: World):
        self.allowed_collision_bodies = set(world.bodies_with_collision)


@dataclass
class AllowCollisionBetweenGroups(AllowCollisionRule):
    """
    Allows collision checks between two groups of bodies.
    """

    body_group_a: List[Body] = field(default_factory=list)
    body_group_b: List[Body] = field(default_factory=list)

    def _update(self, world: World):
        self.allowed_collision_pairs = set()
        for body_a in self.body_group_a:
            for body_b in self.body_group_b:
                if body_a == body_b:
                    continue
                collision_check = CollisionCheck.create_and_validate(
                    body_a=body_a, body_b=body_b
                )
                self.allowed_collision_pairs.add(collision_check)


@dataclass
class AllowNonRobotCollisions(AllowCollisionRule):
    """
    Allows collision checks between all bodies that do not belong to any robot.
    """

    def _update(self, world: World):
        """
        Disable collision checks between bodies that do not belong to any robot.
        """
        # Bodies that are part of any robot and participate in collisions
        robot_bodies: set[Body] = {
            body
            for robot in world.get_semantic_annotations_by_type(AbstractRobot)
            for body in robot.bodies_with_collision
        }

        # Bodies with collisions that are NOT part of a robot
        non_robot_bodies: set[Body] = set(world.bodies_with_collision) - robot_bodies
        if not non_robot_bodies:
            return

        # Disable every unordered pair (including self-collisions) exactly once
        for a, b in combinations(non_robot_bodies, 2):
            self.allowed_collision_pairs.add(
                CollisionCheck.create_and_validate(body_a=a, body_b=b, distance=0)
            )


@dataclass
class AllowSelfCollisions(AllowCollisionRule):
    """
    Allows collision checks between all body pairs of the robot managed by the rule.
    """

    robot: AbstractRobot = field(kw_only=True)

    def _update(self, world: World):
        self.allowed_collision_pairs = set(
            CollisionCheck.create_and_validate(body_a, body_b)
            for body_a, body_b in combinations(self.robot.bodies_with_collision, 2)
        )


@dataclass
class AllowCollisionForAdjacentPairs(AllowCollisionRule):
    """
    Allow collision between body pairs of a robot that are connected by a chain that has no controllable connection.
    """

    def _update(self, world: World):
        for body_a, body_b in combinations(world.bodies_with_collision, 2):
            if (
                not world.is_controlled_connection_in_chain(body_a, body_b)
                or body_a == body_b.parent_kinematic_structure_entity
                or body_b == body_a.parent_kinematic_structure_entity
            ):
                self.allowed_collision_pairs.add(
                    CollisionCheck.create_and_validate(body_a, body_b)
                )


@dataclass
class SelfCollisionMatrixRule(AllowCollisionRule):
    """
    Used to load collision matrices sorted as srdf, e.g., those created by moveit.
    """

    def update(self, world: World): ...

    def _update(self, world: World): ...

    @classmethod
    def from_collision_srdf(cls, file_path: str, world: World) -> Self:
        """
        Creates a CollisionConfig instance from an SRDF file.

        Parse an SRDF file to configure disabled collision pairs or bodies for a given world.
        Process SRDF elements like `disable_collisions`, `disable_self_collision`,
        or `disable_all_collisions` to update collision configuration
        by referencing bodies in the provided `world`.

        :param file_path: The path to the SRDF file used for collision configuration.
        """
        self = cls()
        SRDF_DISABLE_ALL_COLLISIONS: str = "disable_all_collisions"
        SRDF_DISABLE_SELF_COLLISION: str = "disable_self_collision"
        SRDF_MOVEIT_DISABLE_COLLISIONS: str = "disable_collisions"

        srdf = etree.parse(file_path)
        srdf_root = srdf.getroot()

        children_with_tag = [child for child in srdf_root if hasattr(child, "tag")]

        child_disable_collisions = [
            c for c in children_with_tag if c.tag == SRDF_DISABLE_ALL_COLLISIONS
        ]

        for c in child_disable_collisions:
            body = world.get_body_by_name(c.attrib["link"])
            self.allowed_collision_bodies.add(body)

        child_disable_moveit_and_self_collision = [
            c
            for c in children_with_tag
            if c.tag in {SRDF_MOVEIT_DISABLE_COLLISIONS, SRDF_DISABLE_SELF_COLLISION}
        ]

        disabled_collision_pairs = [
            (body_a, body_b)
            for child in child_disable_moveit_and_self_collision
            if (body_a := world.get_body_by_name(child.attrib["link1"])).has_collision()
            and (
                body_b := world.get_body_by_name(child.attrib["link2"])
            ).has_collision()
        ]

        for body_a, body_b in disabled_collision_pairs:
            if body_a == body_b:
                continue
            self.allowed_collision_pairs.add(
                CollisionCheck.create_and_validate(body_a, body_b)
            )
        return self

    def compute_self_collision_matrix(
        self,
        robot: AbstractRobot,
        body_combinations: Iterable | None = None,
        distance_threshold_zero: float = 0.0,
        distance_threshold_always: float = 0.005,
        distance_threshold_never_max: float = 0.05,
        distance_threshold_never_min: float = -0.02,
        distance_threshold_never_range: float = 0.05,
        distance_threshold_never_zero: float = 0.0,
        number_of_tries_always: int = 200,
        almost_percentage: float = 0.95,
        number_of_tries_never: int = 10000,
        use_collision_checker: bool = True,
        save_to_tmp: bool = True,
        overwrite_old_matrix: bool = True,
        progress_callback: Callable[[int, str], None] | None = None,
    ):
        """
        :param use_collision_checker: if False, only the parts will be called that don't require collision checking.
        :param progress_callback: a function that is used to display the progress. it's called with a value of 0-100 and
                                    a string representing the current action
        """
        if progress_callback is None:
            progress_callback = lambda value, text: None
        np.random.seed(1337)
        remaining_pairs = set()
        if overwrite_old_matrix:
            self_collision_matrix = {}
        else:
            self_collision_matrix = self.self_collision_matrix
        # 0. GENERATE ALL POSSIBLE LINK PAIRS
        if body_combinations is None:
            body_combinations = set(
                combinations_with_replacement(robot.bodies_with_collision, 2)
            )
        for body_a, body_b in list(body_combinations):
            collision_check = CollisionCheck.create_and_validate(body_a, body_b)
            remaining_pairs.add(collision_check)

        # %%
        remaining_pairs, matrix_updates = self.compute_self_collision_matrix_adjacent(
            remaining_pairs, group
        )
        self_collision_matrix.update(matrix_updates)

        if use_collision_checker:
            # %%
            remaining_pairs, matrix_updates = (
                self.compute_self_collision_matrix_default(
                    remaining_pairs, group, distance_threshold_zero
                )
            )
            self_collision_matrix.update(matrix_updates)

            # %%
            remaining_pairs, matrix_updates = self.compute_self_collision_matrix_always(
                link_combinations=remaining_pairs,
                group=group,
                distance_threshold_always=distance_threshold_always,
                number_of_tries=number_of_tries_always,
                almost_percentage=almost_percentage,
            )
            self_collision_matrix.update(matrix_updates)

            # %%
            remaining_pairs, matrix_updates = self.compute_self_collision_matrix_never(
                link_combinations=remaining_pairs,
                group=group,
                distance_threshold_never_initial=distance_threshold_never_max,
                distance_threshold_never_min=distance_threshold_never_min,
                distance_threshold_never_range=distance_threshold_never_range,
                distance_threshold_never_zero=distance_threshold_never_zero,
                number_of_tries=number_of_tries_never,
                progress_callback=progress_callback,
            )
            self_collision_matrix.update(matrix_updates)

        if save_to_tmp:
            self.self_collision_matrix = self_collision_matrix
            self.save_self_collision_matrix(
                group=group,
                self_collision_matrix=self_collision_matrix,
                disabled_links=set(),
            )
        else:
            self.self_collision_matrix.update(self_collision_matrix)
        return self_collision_matrix

    def compute_self_collision_matrix_adjacent(
        self, link_combinations: Set[Tuple[PrefixName, PrefixName]], group: WorldBranch
    ) -> Tuple[
        Set[Tuple[PrefixName, PrefixName]],
        Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason],
    ]:
        """
        Find connecting links and disable all adjacent link collisions
        """
        self_collision_matrix = {}
        remaining_pairs = deepcopy(link_combinations)
        for link_a, link_b in list(link_combinations):
            element = link_a, link_b
            if (
                link_a == link_b
                or god_map.world.are_linked(
                    link_a,
                    link_b,
                    do_not_ignore_non_controlled_joints=False,
                    joints_to_be_assumed_fixed=self.fixed_joints,
                )
                or (
                    not group.is_link_controlled(link_a)
                    and not group.is_link_controlled(link_b)
                )
            ):
                remaining_pairs.remove(element)
                self_collision_matrix[element] = DisableCollisionReason.Adjacent
        return remaining_pairs, self_collision_matrix

    def compute_self_collision_matrix_default(
        self,
        link_combinations: Set[Tuple[PrefixName, PrefixName]],
        group: WorldBranch,
        distance_threshold_zero: float,
    ) -> Tuple[
        Set[Tuple[PrefixName, PrefixName]],
        Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason],
    ]:
        """
        Disable link pairs that are in collision in default state
        """
        with god_map.world.reset_joint_state_context():
            self.set_default_joint_state(group)
            self_collision_matrix = {}
            remaining_pairs = deepcopy(link_combinations)
            for link_a, link_b, _ in self.find_colliding_combinations(
                remaining_pairs, distance_threshold_zero, True
            ):
                link_combination = god_map.world.sort_links(link_a, link_b)
                remaining_pairs.remove(link_combination)
                self_collision_matrix[link_combination] = DisableCollisionReason.Default
        return remaining_pairs, self_collision_matrix

    def compute_self_collision_matrix_always(
        self,
        link_combinations: Set[Tuple[PrefixName, PrefixName]],
        group: WorldBranch,
        distance_threshold_always: float,
        number_of_tries: int = 200,
        almost_percentage: float = 0.95,
    ) -> Tuple[
        Set[Tuple[PrefixName, PrefixName]],
        Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason],
    ]:
        """
        Disable link pairs that are (almost) always in collision.
        """
        if number_of_tries == 0:
            return link_combinations, {}
        with god_map.world.reset_joint_state_context():
            self_collision_matrix = {}
            remaining_pairs = deepcopy(link_combinations)
            counts: DefaultDict[Tuple[PrefixName, PrefixName], int] = defaultdict(int)
            for try_id in range(int(number_of_tries)):
                self.set_rnd_joint_state(group)
                for link_a, link_b, _ in self.find_colliding_combinations(
                    remaining_pairs, distance_threshold_always, True
                ):
                    link_combination = god_map.world.sort_links(link_a, link_b)
                    counts[link_combination] += 1
            for link_combination, count in counts.items():
                if count > number_of_tries * almost_percentage:
                    remaining_pairs.remove(link_combination)
                    self_collision_matrix[link_combination] = (
                        DisableCollisionReason.AlmostAlways
                    )
        return remaining_pairs, self_collision_matrix

    def compute_self_collision_matrix_never(
        self,
        link_combinations: Set[Tuple[PrefixName, PrefixName]],
        group: WorldBranch,
        distance_threshold_never_initial: float,
        distance_threshold_never_min: float,
        distance_threshold_never_range: float,
        distance_threshold_never_zero: float,
        number_of_tries: int = 10000,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[
        Set[Tuple[PrefixName, PrefixName]],
        Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason],
    ]:
        """
        Disable link pairs that are never in collision.
        """
        if number_of_tries == 0:
            return link_combinations, {}
        with god_map.world.reset_joint_state_context():
            one_percent = number_of_tries // 100
            self_collision_matrix = {}
            remaining_pairs = deepcopy(link_combinations)
            update_query = True
            distance_ranges: Dict[
                Tuple[PrefixName, PrefixName], Tuple[float, float]
            ] = {}
            once_without_contact = set()
            for try_id in range(int(number_of_tries)):
                self.set_rnd_joint_state(group)
                contacts = self.find_colliding_combinations(
                    remaining_pairs, distance_threshold_never_initial, update_query
                )
                update_query = False
                contact_keys = set()
                for link_a, link_b, distance in contacts:
                    key = god_map.world.sort_links(link_a, link_b)
                    contact_keys.add(key)
                    if key in distance_ranges:
                        old_min, old_max = distance_ranges[key]
                        distance_ranges[key] = (
                            min(old_min, distance),
                            max(old_max, distance),
                        )
                    else:
                        distance_ranges[key] = (distance, distance)
                    if distance < distance_threshold_never_min:
                        remaining_pairs.remove(key)
                        update_query = True
                        del distance_ranges[key]
                once_without_contact.update(remaining_pairs.difference(contact_keys))
                if try_id % one_percent == 0:
                    progress_callback(try_id // one_percent, "checking collisions")
            never_in_contact = remaining_pairs
            for key in once_without_contact:
                if key in distance_ranges:
                    old_min, old_max = distance_ranges[key]
                    distance_ranges[key] = (old_min, np.inf)
            for key, (min_, max_) in list(distance_ranges.items()):
                if (
                    (max_ - min_) < distance_threshold_never_range
                    or min_ > distance_threshold_never_zero
                ):
                    never_in_contact.add(key)
                    del distance_ranges[key]

            for combi in never_in_contact:
                self_collision_matrix[combi] = DisableCollisionReason.Never
        return remaining_pairs, self_collision_matrix

    def save_self_collision_matrix(
        self,
        group: WorldBranch,
        self_collision_matrix: Dict[
            Tuple[PrefixName, PrefixName], DisableCollisionReason
        ],
        disabled_links: Set[PrefixName],
        file_name: Optional[str] = None,
    ):
        # Create the root element
        root = etree.Element("robot")
        root.set("name", group.name)

        # %% disabled links
        for link_name in sorted(disabled_links):
            child = etree.SubElement(root, self.srdf_disable_all_collisions)
            child.set("link", link_name.short_name)

        # %% self collision matrix
        for (link_a, link_b), reason in sorted(self_collision_matrix.items()):
            child = etree.SubElement(root, self.srdf_disable_self_collision)
            child.set("link1", link_a.short_name)
            child.set("link2", link_b.short_name)
            child.set("reason", reason.name)

        # Create the XML tree
        tree = etree.ElementTree(root)

        if file_name is None:
            file_name = self.get_path_to_self_collision_matrix(group.name)
        get_middleware().loginfo(
            f"Saved self collision matrix for {group.name} in {file_name}."
        )
        tree.write(
            file_name,
            pretty_print=True,
            xml_declaration=True,
            encoding=tree.docinfo.encoding,
        )
        self.self_collision_matrix_cache[group.name] = (
            file_name,
            deepcopy(self_collision_matrix),
            deepcopy(disabled_links),
        )

    def get_path_to_self_collision_matrix(self, group_name: str) -> str:
        path_to_tmp = god_map.tmp_folder
        return f"{path_to_tmp}{group_name}/{group_name}.srdf"

    def blacklist_inter_group_collisions(self) -> None:
        for group_a_name, group_b_name in combinations(
            god_map.world.minimal_group_names, 2
        ):
            one_group_is_robot = (
                group_a_name in self.robot_names or group_b_name in self.robot_names
            )
            if one_group_is_robot:
                if group_a_name in self.robot_names:
                    robot_group = god_map.world.groups[group_a_name]
                    other_group = god_map.world.groups[group_b_name]
                else:
                    robot_group = god_map.world.groups[group_b_name]
                    other_group = god_map.world.groups[group_a_name]
                unmovable_links = robot_group.get_unmovable_links()
                if (
                    len(unmovable_links) > 0
                ):  # ignore collisions between unmovable links of the robot and the env
                    for link_a, link_b in product(
                        unmovable_links, other_group.link_names_with_collisions
                    ):
                        self.self_collision_matrix[
                            god_map.world.sort_links(link_a, link_b)
                        ] = DisableCollisionReason.Unknown
                continue
            # disable all collisions of groups that aren't a robot
            group_a: WorldBranch = god_map.world.groups[group_a_name]
            group_b: WorldBranch = god_map.world.groups[group_b_name]
            for link_a, link_b in product(
                group_a.link_names_with_collisions, group_b.link_names_with_collisions
            ):
                self.self_collision_matrix[god_map.world.sort_links(link_a, link_b)] = (
                    DisableCollisionReason.Unknown
                )
        # disable non actuated groups
        for group in god_map.world.groups.values():
            if group.name not in self.robot_names:
                for link_a, link_b in set(
                    combinations_with_replacement(group.link_names_with_collisions, 2)
                ):
                    key = god_map.world.sort_links(link_a, link_b)
                    self.self_collision_matrix[key] = DisableCollisionReason.Unknown

    # def get_map_T_geometry(self, link_name: PrefixName, collision_id: int = 0) -> Pose:
    #     map_T_geometry = god_map.world.compute_fk_with_collision_offset(god_map.world.root_link_name, link_name,
    #                                                                     collision_id)
    #     return msg_converter.to_ros_message(map_T_geometry).pose

    def set_joint_state_to_zero(self) -> None:
        for free_variable in god_map.world.free_variables:
            god_map.world.state[free_variable].position = 0
        god_map.world.notify_state_change()

    def set_default_joint_state(self, group: WorldBranch):
        for joint_name in group.movable_joint_names:
            free_variable: FreeVariable
            for free_variable in group.joints[joint_name].free_variables:
                if free_variable.has_position_limits():
                    lower_limit = free_variable.get_lower_limit(Derivatives.position)
                    upper_limit = free_variable.get_upper_limit(Derivatives.position)
                    god_map.world.state[free_variable.name].position = (
                        upper_limit + lower_limit
                    ) / 2
        god_map.world.notify_state_change()
