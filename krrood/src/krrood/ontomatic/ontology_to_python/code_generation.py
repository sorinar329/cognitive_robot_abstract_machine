"""
Rendering the information about an ontology into Python modules with Jinja2 templates.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy, copy
from dataclasses import dataclass, field, asdict

import rdflib
from jinja2 import Environment, FileSystemLoader
from jinja2.ext import loopcontrols
from ordered_set import OrderedSet
from rdflib.namespace import RDFS, XSD
from typing_extensions import Any, Dict, List, Optional

from krrood.ontomatic.naming import NamingRegistry
from krrood.ontomatic.ontology_to_python.property_type import PropertyType
from krrood.ontomatic.property_descriptor.mixins import (
    HasChainAxioms,
)
from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import Symbol

from krrood.ontomatic.ontology_to_python.inference import InferenceEngine
from krrood.ontomatic.ontology_to_python.ontology_info import (
    ClassInfo,
    OntologyInfo,
    PropertyInfo,
)


class JinjaRenderer:
    """
    Renderer for generating Python code using Jinja2 templates.
    """

    def __init__(self, template_dir: str):
        """
        Initialize the renderer.

        :param template_dir: Directory containing Jinja2 templates.
        """
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[loopcontrols],
        )

    def render(self, *, template_name: str, **context) -> str:
        """
        Render a template with the given context.

        :param template_name: Name of the template file.
        :param context: Keyword arguments for the template context.
        :return: Rendered string.
        """
        template = self.env.get_template(template_name)
        return template.render(**context)


@dataclass(kw_only=True)
class CodeGenerator:
    """
    Orchestrates the generation of Python code from extracted ontology information.
    """

    ontology: OntologyInfo
    """
    The ontology the code is generated from.
    """

    engine: InferenceEngine = field(init=False)
    """
    The engine inferring the classes and properties of the ontology.
    """

    renderer: JinjaRenderer = field(init=False)
    """
    The renderer of the templates of the generated modules.
    """

    def __post_init__(self):
        """
        Initialize the code generator.
        """
        self._ensure_ontology_base_class_in_classes()

        self._ensure_uri_in_ontology_properties()

        self._replace_ontology_role_class_with_current_role_class_name()

        self._update_base_classes()

        self.engine, self.renderer = (
            InferenceEngine(ontology=self.ontology),
            JinjaRenderer(os.path.join(os.path.dirname(__file__), "jinja")),
        )

    def generate(self, *, base_file_name: str) -> Dict[str, str]:
        """
        Execute the full generation pipeline.

        :param base_file_name: Base name for generated files.
        :return: Dictionary mapping filenames to their rendered content.
        """
        self._execute_inference_pipeline()

        self._determine_class_properties()

        self.attach_domainless_properties_to_ontology_base_class()

        classes_order, property_order = self._finalize_and_sort()

        return self._perform_rendering(
            base_file_name=base_file_name,
            classes_order=classes_order,
            property_order=property_order,
        )

    def attach_domainless_properties_to_ontology_base_class(self):
        """
        Attach properties without declared domains to the ontology base class.
        """
        for property_information in self.ontology.properties.values():
            if property_information.declared_domains or property_information.domains:
                continue
            property_information.declared_domains = [self.ontology.base_class_name]
            base_class_information = self.ontology.classes[
                self.ontology.base_class_name
            ]
            if (
                property_information.name
                not in base_class_information.declared_properties
            ):
                base_class_information.declared_properties.append(
                    property_information.name
                )

    def _replace_ontology_role_class_with_current_role_class_name(self):
        """
        Replace ontology role class name with current role class name.
        """
        for class_information in self.ontology.classes.values():
            if Role.__name__ not in class_information.base_classes:
                continue
            class_information.base_classes.remove(Role.__name__)
            class_information.base_classes.append(self.ontology.role_class_name)

    def _ensure_uri_in_ontology_properties(self):
        """
        Ensures that the 'uri' property is present in the ontology properties.

        If not present, adds it with appropriate configuration.
        """
        if "uri" in self.ontology.properties:
            return
        self.ontology.properties["uri"] = PropertyInfo(
            name="uri",
            uri="",
            type=PropertyType.DATA_PROPERTY,
            domains=[self.ontology.base_class_name],
            ranges=["str"],
            range_uris=[XSD.anyURI],
            label="URI of the ontology element",
            comment="The unique resource identifier (URI) of the ontology element.",
            field_name="uri",
            descriptor_name="Uri",
            declared_domains=[self.ontology.base_class_name],
        )

    def _update_base_classes(self):
        """
        Make every class inherit from its superclasses, or from the base class; a role
        inherits from the role class and
        :class:`~krrood.symbol_graph.symbol_graph.Symbol` only.
        """
        for class_name, class_information in self.ontology.classes.items():
            if class_name == self.ontology.base_class_name:
                continue
            class_information.base_classes = [
                superclass_name
                for superclass_name in class_information.superclasses
                if superclass_name != Symbol.__name__
            ] or [self.ontology.base_class_name]
            if self.ontology.role_class_name in class_information.base_classes:
                for cls in copy(class_information.base_classes):
                    if cls not in [self.ontology.role_class_name]:
                        class_information.base_classes.remove(cls)
                        class_information.superclasses.remove(cls)
                class_information.base_classes.append(Symbol.__name__)

    def _ensure_ontology_base_class_in_classes(self):
        """
        Add the base class of the ontology to the classes when the ontology does not
        declare it.
        """
        if self.ontology.base_class_name in self.ontology.classes:
            return
        self.ontology.classes[self.ontology.base_class_name] = ClassInfo(
            name=self.ontology.base_class_name,
            uri="",
            superclasses=[Symbol.__name__],
            base_classes=[Symbol.__name__],
            label=f"Base class for {self.ontology.ontology_label}",
        )

    def _execute_inference_pipeline(self):
        """
        Run the inference engine to propagate types and specialized properties.
        """
        self.engine.compute_ancestors()

        self.engine.infer_properties()

        self.engine.apply_predefined_overrides()

        self.attach_domainless_data_properties_to_ontology_base_class()

        self.engine.compute_type_hints()

    def attach_domainless_data_properties_to_ontology_base_class(self):
        """
        Attach properties without declared domains to the ontology base class.
        """
        for property_information in self.ontology.properties.values():
            if (
                property_information.type == PropertyType.DATA_PROPERTY
                and not property_information.declared_domains
            ):
                property_information.declared_domains = [self.ontology.base_class_name]

    def _determine_class_properties(self):
        """
        Decide which properties belong to which class based on inheritance and
        overrides.
        """
        for class_name, class_information in self.ontology.classes.items():
            ancestors = set(class_information.all_base_classes)
            declared: List[str] = []
            for property_name, property_information in self.ontology.properties.items():
                if property_name == "roleFor":
                    continue
                if class_name not in set(property_information.declared_domains):
                    continue
                if (
                    ancestors
                    and class_name not in property_information._overrides_for
                    and any(
                        ancestor
                        in (
                            set(property_information.declared_domains)
                            | set(property_information.domains)
                        )
                        for ancestor in ancestors
                    )
                ):
                    continue
                if property_information.is_specialized:
                    for superproperty in property_information.superproperties:
                        if superproperty in declared:
                            declared.remove(superproperty)
                declared.append(property_name)
            class_information.declared_properties = sorted(set(declared))

    def _finalize_and_sort(self):
        """
        Compute transitive closures and determine final topological order for classes
        and properties.
        """
        for property_information in self.ontology.properties.values():
            property_information.all_superproperties = self._compute_closure(
                initial=property_information.superproperties,
                items=self.ontology.properties,
                key="superproperties",
            )

        for class_information in self.ontology.classes.values():
            initial = set(class_information.all_base_classes)
            if class_information.role_taker:
                initial.add(class_information.role_taker.class_name)
            class_information.all_base_classes_including_role_takers = (
                self._compute_closure(
                    initial=list(initial),
                    items=self.ontology.classes,
                    key="all_base_classes",
                    role_key="role_taker",
                )
            )

        if "roleFor" in self.ontology.properties:
            del self.ontology.properties["roleFor"]

        unspecialized_properties = {
            name: property_information
            for name, property_information in self.ontology.properties.items()
            if not property_information.is_specialized
        }
        property_order = OntologyInfo.topological_order(
            items=unspecialized_properties, dependency_key="superproperties"
        )

        # Subclass and role relations are only taken from the ontology (rdfs:subClassOf, owl:equivalentClass and
        # roleFor restrictions). Implicit subsumptions guessed from shared properties (find_implicit_subtypes) are not
        # entailed and are therefore not applied.

        for property_information in self.ontology.classes.values():
            property_information.base_classes_for_topological_sort = (
                property_information.base_classes[:]
            )
            if property_information.role_taker:
                property_information.base_classes_for_topological_sort.append(
                    property_information.role_taker.class_name
                )

        for property_information in unspecialized_properties.values():
            if (
                not property_information.object_range_hint
                or "Union" not in property_information.object_range_hint
            ):
                continue
            contesting_types = [
                type_name.strip()
                for type_name in property_information.object_range_hint[6:-1].split(",")
            ]
            for index, range_type in enumerate(contesting_types[:-1]):
                for other_range_type in contesting_types[index + 1 :]:
                    if (
                        other_range_type
                        in self.ontology.classes[
                            range_type
                        ].all_base_classes_including_role_takers
                    ):
                        try:
                            contesting_types.remove(range_type)
                        except ValueError:
                            pass
                    elif (
                        range_type
                        in self.ontology.classes[
                            other_range_type
                        ].all_base_classes_including_role_takers
                    ):
                        try:
                            contesting_types.remove(other_range_type)
                        except ValueError:
                            pass
            if len(contesting_types) == 1:
                property_information.object_range_hint = contesting_types[0]

        removed_classes = []
        for class_name, class_information in self.ontology.classes.items():
            if class_name in removed_classes:
                continue
            for equivalent_class in class_information.equivalent_classes:
                removed_classes.append(equivalent_class)
                initial_base_classes = copy(class_information.base_classes)
                for base_class in self.ontology.classes[equivalent_class].base_classes:
                    if base_class not in class_information.base_classes:
                        class_information.base_classes.append(base_class)
                        class_information.all_base_classes.append(base_class)
                        class_information.all_base_classes_including_role_takers.append(
                            base_class
                        )
                        class_information.base_classes_for_topological_sort.append(
                            base_class
                        )
                if (
                    len(initial_base_classes) == 1
                    and initial_base_classes[0] == self.ontology.base_class_name
                    and len(class_information.base_classes) > 1
                ):
                    class_information.base_classes.remove(self.ontology.base_class_name)
                for declared_property in self.ontology.classes[
                    equivalent_class
                ].declared_properties:
                    if declared_property not in class_information.declared_properties:
                        class_information.declared_properties.append(declared_property)
                for axiom in self.ontology.classes[equivalent_class].axioms:
                    if axiom not in class_information.axioms:
                        class_information.axioms.append(axiom)
                for python_axiom in self.ontology.classes[
                    equivalent_class
                ].axioms_python:
                    if python_axiom not in class_information.axioms_python:
                        class_information.axioms_python.append(python_axiom)
                for axiom_setup in self.ontology.classes[equivalent_class].axioms_setup:
                    if axiom_setup not in class_information.axioms_setup:
                        class_information.axioms_setup.append(axiom_setup)
        for removed_class in removed_classes:
            if removed_class in self.ontology.classes:
                del self.ontology.classes[removed_class]
        classes_order = OntologyInfo.topological_order(
            items=self.ontology.classes,
            dependency_key="base_classes_for_topological_sort",
        )

        index_by_property = {
            property_name: index for index, property_name in enumerate(property_order)
        }
        for property_information in unspecialized_properties.values():
            if property_information.inverse_of in unspecialized_properties:
                property_information.inverse_target_is_prior = index_by_property.get(
                    property_information.inverse_of, 1e9
                ) < index_by_property.get(property_information.name, 1e9)

        return classes_order, property_order

    def _classes_annotating_the_base_class(
        self, *, render_classes: Dict[str, Dict], render_properties: Dict[str, Dict]
    ) -> List[str]:
        """
        :return: The generated classes the field annotations of the base class name, which the base module imports
            only for type checking because they are defined in the module of the classes.
        """
        base_class = render_classes[self.ontology.base_class_name]
        annotations = " ".join(
            render_properties[property_name]["object_range_hint"] or ""
            for property_name in base_class["declared_properties"]
        )
        named = set(re.findall(r"\w+", annotations))
        return [
            name
            for name in render_classes
            if name in named and name != self.ontology.base_class_name
        ]

    def _perform_rendering(
        self,
        *,
        base_file_name,
        classes_order,
        property_order,
    ):
        """
        Render all templates and produce the final Python files and stubs.
        """
        stubs_classes = deepcopy(self.ontology.classes)
        for class_name, class_information in self.ontology.classes.items():
            if self.ontology.role_class_name in class_information.base_classes:
                class_information.base_classes.remove(self.ontology.role_class_name)
                class_information.base_classes.insert(
                    0,
                    f"{self.ontology.role_class_name}[{class_information.role_taker.class_name}]",
                )
                stubs_classes[class_name].base_classes.remove(
                    self.ontology.role_class_name
                )
            else:
                class_information.add_role_taker = stubs_classes[
                    class_name
                ].add_role_taker = False

        if Role.__name__ in self.ontology.classes:
            del self.ontology.classes[Role.__name__]

        # topological_order might still have 'Role' name if it was in the items keys
        # We need to filter the order as well
        classes_order = [
            rendered_class
            for rendered_class in classes_order
            if rendered_class != Role.__name__
        ]
        for class_information in self.ontology.classes.values():
            class_information.ontology = None
        render_classes = {
            name: asdict(class_information)
            for name, class_information in self.ontology.classes.items()
        }
        for rendered_class in render_classes.values():
            if rendered_class["role_taker"] is None:
                rendered_class["role_taker"] = {}

        for property_name, property_information in self.ontology.properties.items():
            for equivalent_property_name in property_information.equivalent_properties:
                property_information.equivalent_properties_descriptor_names.append(
                    self.ontology.properties[equivalent_property_name].descriptor_name
                )
            for disjoint_property_name in property_information.disjoint_properties:
                property_information.disjoint_properties_descriptor_names.append(
                    self.ontology.properties[disjoint_property_name].descriptor_name
                )

        for property_name, property_information in self.ontology.properties.items():
            if property_information.chain_axioms:
                property_information.base_descriptors.append(HasChainAxioms.__name__)
            property_uri = rdflib.URIRef(property_information.uri)
            property_information.rdfs_domains = sorted(
                NamingRegistry.uri_to_python_name(uri=domain)
                for domain in self.ontology.graph.objects(property_uri, RDFS.domain)
                if isinstance(domain, rdflib.URIRef)
            )
            property_information.rdfs_ranges = sorted(
                NamingRegistry.uri_to_python_name(uri=range_)
                for range_ in self.ontology.graph.objects(property_uri, RDFS.range)
                if isinstance(range_, rdflib.URIRef)
            )

        for property_information in self.ontology.properties.values():
            property_information.ontology = None
        for stub_information in stubs_classes.values():
            stub_information.ontology = None
        render_properties = {
            name: asdict(property_information)
            for name, property_information in self.ontology.properties.items()
        }
        render_stubs = {
            name: asdict(property_information)
            for name, property_information in stubs_classes.items()
        }
        for rendered_class in render_stubs.values():
            if rendered_class["role_taker"] is None:
                rendered_class["role_taker"] = {}

        properties_module, base_module = (
            f"{base_file_name}_properties",
            f"{base_file_name}_base",
        )
        return {
            f"{properties_module}.py": self.renderer.render(
                template_name="onto_properties.j2",
                properties=render_properties,
                properties_order=property_order,
            ),
            f"{base_module}.py": self.renderer.render(
                template_name="onto_base.j2",
                class_information=render_classes[self.ontology.base_class_name],
                properties=render_properties,
                classes_module_name=base_file_name,
                annotated_classes=self._classes_annotating_the_base_class(
                    render_classes=render_classes, render_properties=render_properties
                ),
            ),
            f"{base_file_name}.py": self.renderer.render(
                template_name="onto_classes.j2",
                ontology_base_module_name=base_module,
                properties_module_name=properties_module,
                classes=render_classes,
                properties=render_properties,
                classes_order=classes_order,
                properties_order=property_order,
                ontology_base_class_name=self.ontology.base_class_name,
            ),
            f"{base_file_name}.pyi": self.renderer.render(
                template_name="onto_stubs.j2",
                ontology_base_module_name=base_module,
                properties_module_name=properties_module,
                role_takers=list(
                    OrderedSet(
                        rendered_class["role_taker"]["class_name"]
                        for rendered_class in render_classes.values()
                        if rendered_class["role_taker"]
                    )
                ),
                classes=render_stubs,
                properties=render_properties,
                classes_order=classes_order,
                ontology_base_class_name=self.ontology.base_class_name,
            ),
        }

    @staticmethod
    def _compute_closure(
        *,
        initial: List[str],
        items: Dict[str, Any],
        key: str,
        role_key: Optional[str] = None,
    ) -> List[str]:
        """
        Compute the transitive closure of a relationship.
        """
        closure, stack = set(), list(initial)
        while stack:
            current = stack.pop()
            if current not in closure:
                closure.add(current)
                if current in items:
                    item = items[current]
                    stack.extend(
                        getattr(item, key, [])
                        if hasattr(item, key)
                        else item.get(key, [])
                    )
                    role_taker = (
                        getattr(item, role_key, None)
                        if role_key and hasattr(item, role_key)
                        else (item.get(role_key) if role_key else None)
                    )
                    if role_taker:
                        stack.append(
                            role_taker.class_name
                            if hasattr(role_taker, "class_name")
                            else role_taker["class_name"]
                        )
        return sorted(closure)
