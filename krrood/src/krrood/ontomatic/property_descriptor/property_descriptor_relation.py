"""
Relations between instances held by property descriptors, and what each one implies.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from typing_extensions import TYPE_CHECKING, Iterator

from krrood.class_diagrams.class_diagram import Association, AssociationThroughRoleTaker
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.ontomatic.property_descriptor.mixins import (
    HasEquivalentProperties,
    HasInverseProperty,
    SymmetricProperty,
    TransitiveProperty,
)
from krrood.symbol_graph.symbol_graph import (
    PredicateClassRelation,
    SymbolGraph,
    WrappedInstance,
)

if TYPE_CHECKING:
    from krrood.ontomatic.property_descriptor.property_descriptor import (
        PropertyDescriptor,
    )


@dataclass(unsafe_hash=True)
class PropertyDescriptorRelation(PredicateClassRelation):
    """
    Edge data representing a relation between two wrapped instances that is represented
    structurally by a property descriptor attached to the source instance.
    """

    @cached_property
    def transitive(self) -> bool:
        """
        If the relation is transitive or not.
        """
        if self.property_descriptor_cls:
            return issubclass(self.property_descriptor_cls, TransitiveProperty)
        else:
            return False

    @cached_property
    def inverse_of(self) -> type[PropertyDescriptor] | None:
        """
        The inverse of the relation if it exists.
        """
        if self.property_descriptor_cls and issubclass(
            self.property_descriptor_cls, HasInverseProperty
        ):
            return self.property_descriptor_cls.get_inverse()
        else:
            return None

    def update_source_and_add_to_graph_and_apply_implications(self):
        """
        Update the source wrapped-field value, add this relation to the graph, and apply
        all implications of adding this relation.
        """
        source_updated = not self.inferred or self.update_source_wrapped_field_value()
        if not source_updated:
            # Means that the value was already set, so we don't need to infer anything.
            return
        self.add_to_graph_and_apply_implications()

    def add_to_graph_and_apply_implications(self):
        """
        Add this relation to the graph and apply all implications of this relation.
        """
        if self.add_to_graph():
            self.infer_and_apply_implications()

    def infer_and_apply_implications(self):
        """
        Infer all implications of adding this relation and apply them to the
        corresponding objects.
        """
        self.infer_equivalent_relations()
        self.infer_super_relations()
        self.infer_inverse_relation()
        self.infer_symmetric_relation()
        self.infer_transitive_relations()
        self.infer_chained_relations()

    def infer_equivalent_relations(self):
        """
        Infer the relations of every property equivalent to this relation's property.
        """
        if not issubclass(self.property_descriptor_cls, HasEquivalentProperties):
            return
        for (
            equivalent_descriptor
        ) in self.property_descriptor_cls.get_equivalent_properties():
            self.add_inferred_relation(self.source, equivalent_descriptor, self.target)

    def infer_symmetric_relation(self):
        """
        Infer the relation back from the target to the source if the property is
        symmetric.
        """
        if issubclass(self.property_descriptor_cls, SymmetricProperty):
            self.add_inferred_relation(
                self.target, self.property_descriptor_cls, self.source
            )

    def update_source_wrapped_field_value(self) -> bool:
        """
        Update the wrapped field value for the source instance.

        :return: True if the value of the wrapped field was updated, False otherwise
            (i.e., if the value was already set).
        """
        return self.wrapped_field.property_descriptor.update_value(
            self.source.instance, self.target.instance
        )

    def infer_super_relations(self):
        """
        Infer all super relations of this relation.
        """
        for super_domain, super_field in self.super_relations:
            self.__class__(
                super_domain, self.target, super_field, inferred=True
            ).update_source_and_add_to_graph_and_apply_implications()

    def infer_inverse_relation(self):
        """
        Infer the inverse relation if it exists.
        """
        if self.inverse_of:
            inverse_domain, inverse_field = self.inverse_domain_and_field
            self.__class__(
                inverse_domain, self.source, inverse_field, inferred=True
            ).update_source_and_add_to_graph_and_apply_implications()

    @cached_property
    def super_relations(self) -> Iterator[tuple[WrappedInstance, WrappedField]]:
        """
        Find neighboring symbols connected by super edges.

        This method identifies neighboring symbols that are connected through edge with
        relation types that are superclasses of the current relation type.

        :return: An iterator over neighboring symbols and relations that are super
            relations.
        """
        source_type = self.source.instance_type
        property_descriptor_cls: type[PropertyDescriptor] = (
            self.wrapped_field.property_descriptor.__class__
        )
        for association in property_descriptor_cls.get_superproperties_associations(
            source_type
        ):
            original_source_instance = association.get_original_source_instance_given_this_relation_source_instance(
                self.source.instance
            )
            source = SymbolGraph().get_wrapped_instance(original_source_instance)
            yield source, association.wrapped_field

    @property
    def inverse_domain_and_field(self) -> tuple[WrappedInstance, WrappedField]:
        """
        Get the inverse of the property descriptor.

        :return: The inverse domain instance and property descriptor field.
        """
        if not self.inverse_association:
            raise ValueError(
                f"cannot find a field for the inverse {self.inverse_of} defined for the relation {self}"
            )
        original_source_instance = self.inverse_association.get_original_source_instance_given_this_relation_source_instance(
            self.target.instance
        )
        source = SymbolGraph().get_wrapped_instance(original_source_instance)
        return source, self.inverse_association.wrapped_field

    @cached_property
    def inverse_association(
        self,
    ) -> Association | AssociationThroughRoleTaker | None:
        """
        Return the inverse field (if it exists) stored in the target of this relation.
        """
        return self.inverse_of.get_association_of_source_type(self.target.instance_type)

    def infer_transitive_relations(self):
        """
        Add all transitive relations of this relation type that results from adding this
        relation to the graph.
        """
        if self.transitive:
            self.infer_transitive_relations_outgoing_from_source()
            self.infer_transitive_relations_incoming_to_target()

    def infer_transitive_relations_outgoing_from_source(self):
        """
        Infer transitive relations outgoing from the source.
        """
        for nxt_relation in self.target_outgoing_relations_with_same_descriptor_type:
            self.__class__(
                self.source,
                nxt_relation.target,
                nxt_relation.wrapped_field,
                inferred=True,
            ).update_source_and_add_to_graph_and_apply_implications()

    def infer_transitive_relations_incoming_to_target(self):
        """
        Infer transitive relations incoming to the target.
        """
        for nxt_relation in self.source_incoming_relations_with_same_descriptor_type:
            self.__class__(
                nxt_relation.source,
                self.target,
                nxt_relation.wrapped_field,
                inferred=True,
            ).update_source_and_add_to_graph_and_apply_implications()

    @property
    def target_outgoing_relations_with_same_descriptor_type(
        self,
    ) -> Iterator[PredicateClassRelation]:
        """
        Get the outgoing relations from the target that have the same property
        descriptor type as this relation.
        """
        relation_condition = lambda relation: issubclass(
            relation.property_descriptor_cls, self.property_descriptor_cls
        )
        yield from SymbolGraph().get_outgoing_relations_with_condition(
            self.target, relation_condition
        )

    @property
    def source_incoming_relations_with_same_descriptor_type(
        self,
    ) -> Iterator[PredicateClassRelation]:
        """
        Get the incoming relations from the source that have the same property
        descriptor type as this relation.
        """
        relation_condition = lambda relation: issubclass(
            relation.property_descriptor_cls, self.property_descriptor_cls
        )
        yield from SymbolGraph().get_incoming_relations_with_condition(
            self.source, relation_condition
        )

    def infer_chained_relations(self):
        """
        Infer the relations implied by every property chain this relation completes,
        together with the relations before and after it in the graph.
        """
        chain_positions = self.property_descriptor_cls.chains_with_descriptor[
            self.property_descriptor_cls
        ]
        for chain_position in list(chain_positions):
            for chain_start in self.nodes_starting_paths_to(
                self.source, chain_position.descriptors_before
            ):
                for chain_end in self.nodes_ending_paths_from(
                    self.target, chain_position.descriptors_after
                ):
                    self.add_inferred_relation(
                        chain_start, chain_position.implied_descriptor, chain_end
                    )

    @classmethod
    def nodes_starting_paths_to(
        cls,
        end: WrappedInstance,
        descriptor_classes: tuple[type[PropertyDescriptor], ...],
    ) -> Iterator[WrappedInstance]:
        """
        :param end: The node the paths end in.
        :param descriptor_classes: The descriptor classes of the relations of a path, in order.
        :return: The first node of every path of relations of the descriptor classes that ends in the end node.
        """
        if not descriptor_classes:
            yield end
            return
        *before, last = descriptor_classes
        for relation in list(cls.incoming_relations_of_descriptor(end, last)):
            yield from cls.nodes_starting_paths_to(relation.source, tuple(before))

    @classmethod
    def nodes_ending_paths_from(
        cls,
        start: WrappedInstance,
        descriptor_classes: tuple[type[PropertyDescriptor], ...],
    ) -> Iterator[WrappedInstance]:
        """
        :param start: The node the paths start from.
        :param descriptor_classes: The descriptor classes of the relations of a path, in order.
        :return: The last node of every path of relations of the descriptor classes that starts from the start node.
        """
        if not descriptor_classes:
            yield start
            return
        first, *after = descriptor_classes
        for relation in list(cls.outgoing_relations_of_descriptor(start, first)):
            yield from cls.nodes_ending_paths_from(relation.target, tuple(after))

    @staticmethod
    def incoming_relations_of_descriptor(
        node: WrappedInstance, descriptor_class: type[PropertyDescriptor]
    ) -> Iterator[PropertyDescriptorRelation]:
        """
        :return: The relations of the descriptor class that end in the node.
        """
        yield from SymbolGraph().get_incoming_relations_with_condition(
            node,
            lambda relation: (
                isinstance(relation, PropertyDescriptorRelation)
                and relation.property_descriptor_cls is descriptor_class
            ),
        )

    @staticmethod
    def outgoing_relations_of_descriptor(
        node: WrappedInstance, descriptor_class: type[PropertyDescriptor]
    ) -> Iterator[PropertyDescriptorRelation]:
        """
        :return: The relations of the descriptor class that start from the node.
        """
        yield from SymbolGraph().get_outgoing_relations_with_condition(
            node,
            lambda relation: (
                isinstance(relation, PropertyDescriptorRelation)
                and relation.property_descriptor_cls is descriptor_class
            ),
        )

    def add_inferred_relation(
        self,
        source: WrappedInstance,
        descriptor_class: type[PropertyDescriptor],
        target: WrappedInstance,
    ):
        """
        Add a relation of the descriptor class from the source to the target to the
        graph and to the field that holds it, and apply its implications.

        Nothing is added when the source's type has no such field.
        """
        association = descriptor_class.get_association_of_source_type(
            source.instance_type
        )
        if association is None:
            return
        holder = association.get_original_source_instance_given_this_relation_source_instance(
            source.instance
        )
        self.__class__(
            holder, target, association.wrapped_field, inferred=True
        ).update_source_and_add_to_graph_and_apply_implications()

    @cached_property
    def property_descriptor_cls(self) -> type[PropertyDescriptor]:
        """
        Return the property descriptor class of the relation.
        """
        return type(self.wrapped_field.property_descriptor)
