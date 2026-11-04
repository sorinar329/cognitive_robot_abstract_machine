"""
The hierarchy of classes in the sense of an ontology, where a role is a subclass of the
type of its role taker.
"""

from __future__ import annotations

from functools import lru_cache

from typing_extensions import Iterable

from krrood.patterns.role import Role
from krrood.patterns.role_predicates import IsSubClassOrRole


class RoleAwareClassHierarchy:
    """
    Ranks classes by how specific they are, counting a
    :class:`~krrood.patterns.role.Role` as one inheritance step below the type of its
    role taker.
    """

    @staticmethod
    def most_specific(classes: Iterable[type]) -> list[type]:
        """
        :return: The classes no other of the classes is a subclass or role of, without
            duplicates, in their order.
        """
        unique_classes = list(dict.fromkeys(classes))
        return [
            cls
            for cls in unique_classes
            if not any(
                other is not cls and IsSubClassOrRole.holds(other, cls)
                for other in unique_classes
            )
        ]

    @staticmethod
    def role_taker_chain_length(cls: type) -> int:
        """
        :return: The number of role takers between the class and the root role-taker
            type, zero for a class that is no role.
        """
        length = 0
        while issubclass(cls, Role) and cls is not Role:
            cls = cls.get_role_taker_type()
            length += 1
        return length

    @staticmethod
    @lru_cache(maxsize=None)
    def ancestors(cls: type) -> tuple[type, ...]:
        """
        :return: The method resolution order of the class, with
            :class:`~krrood.patterns.role.Role` replaced by the role-taker type for a role.
        """
        if not issubclass(cls, Role) or cls is Role:
            return cls.__mro__
        role_taker_type = cls.get_role_taker_type()
        return tuple(
            role_taker_type if ancestor is Role else ancestor
            for ancestor in cls.__mro__
        )

    @classmethod
    @lru_cache(maxsize=None)
    def nearest_common_ancestor(cls, classes: tuple[type, ...]) -> type | None:
        """
        :return: The first ancestor of the first class, in the order of
            :meth:`ancestors`, that is an ancestor of every class; None for no classes.
        """
        if not classes:
            return None
        other_ancestors = [cls.ancestors(other) for other in classes[1:]]
        return next(
            (
                candidate
                for candidate in cls.ancestors(classes[0])
                if all(candidate in ancestors for ancestors in other_ancestors)
            ),
            None,
        )

    @classmethod
    @lru_cache(maxsize=None)
    def inheritance_path_length(cls, child: type, parent: type) -> float:
        """
        :return: The number of inheritance steps of the shortest path from the child up
            to the parent, a role reaching the type of its role taker in one step;
            infinity if the child is neither a subclass nor a role of the parent.
        """
        if child is parent:
            return 0
        if not IsSubClassOrRole.holds(child, parent):
            return float("inf")
        bases = set(child.__bases__)
        if Role in bases and child is not Role:
            bases.add(child.get_role_taker_type())
        return 1 + min(
            cls.inheritance_path_length(base, parent)
            for base in bases
            if IsSubClassOrRole.holds(base, parent)
        )

    @classmethod
    def sorted_by_inheritance_path_length(
        cls,
        classes: tuple[type, ...],
        common_ancestor: type | None = None,
        classes_without_order: tuple[type, ...] = (),
    ) -> list[type]:
        """
        :param classes: The classes to sort.
        :param common_ancestor: The class the distances are measured to, by default the
            nearest common ancestor of the classes.
        :param classes_without_order: Nearest common ancestors that leave the classes
            unsorted.
        :return: The classes from the nearest to the farthest from the common ancestor;
            among classes equally far, those that are no roles first, then roles with
            shorter role-taker chains.
        """
        if common_ancestor is None:
            common_ancestor = cls.nearest_common_ancestor(tuple(classes))
            if common_ancestor is None or common_ancestor in classes_without_order:
                return list(classes)
        return sorted(
            classes,
            key=lambda sorted_class: (
                cls.inheritance_path_length(sorted_class, common_ancestor),
                cls.role_taker_chain_length(sorted_class),
            ),
        )
