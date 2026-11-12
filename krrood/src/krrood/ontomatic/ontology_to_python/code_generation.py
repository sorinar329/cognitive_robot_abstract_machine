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

    onto: OntologyInfo
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
            InferenceEngine(onto=self.onto),
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

        classes_order, props_order = self._finalize_and_sort()

        return self._perform_rendering(
            base_file_name=base_file_name,
            classes_order=classes_order,
            props_order=props_order,
        )

    def attach_domainless_properties_to_ontology_base_class(self):
        """
        Attach properties without declared domains to the ontology base class.
        """
        for p in self.onto.properties.values():
            if p.declared_domains or p.domains:
                continue
            p.declared_domains = [self.onto.base_cls_name]
            base_class_info = self.onto.classes[self.onto.base_cls_name]
            if p.name not in base_class_info.declared_properties:
                base_class_info.declared_properties.append(p.name)

    def _replace_ontology_role_class_with_current_role_class_name(self):
        """
        Replace ontology role class name with current role class name.
        """
        for info in self.onto.classes.values():
            if Role.__name__ not in info.base_classes:
                continue
            info.base_classes.remove(Role.__name__)
            info.base_classes.append(self.onto.role_cls_name)

    def _ensure_uri_in_ontology_properties(self):
        """
        Ensures that the 'uri' property is present in the ontology properties.

        If not present, adds it with appropriate configuration.
        """
        if "uri" in self.onto.properties:
            return
        self.onto.properties["uri"] = PropertyInfo(
            name="uri",
            uri="",
            type=PropertyType.DATA_PROPERTY,
            domains=[self.onto.base_cls_name],
            ranges=["str"],
            range_uris=[XSD.anyURI],
            label="URI of the ontology element",
            comment="The unique resource identifier (URI) of the ontology element.",
            field_name="uri",
            descriptor_name="Uri",
            declared_domains=[self.onto.base_cls_name],
        )

    def _update_base_classes(self):
        """
        Make every class inherit from its superclasses, or from the base class; a role
        inherits from the role class and
        :class:`~krrood.symbol_graph.symbol_graph.Symbol` only.
        """
        for n, info in self.onto.classes.items():
            if n == self.onto.base_cls_name:
                continue
            info.base_classes = [
                b for b in info.superclasses if b != Symbol.__name__
            ] or [self.onto.base_cls_name]
            if self.onto.role_cls_name in info.base_classes:
                for cls in copy(info.base_classes):
                    if cls not in [self.onto.role_cls_name]:
                        info.base_classes.remove(cls)
                        info.superclasses.remove(cls)
                info.base_classes.append(Symbol.__name__)

    def _ensure_ontology_base_class_in_classes(self):
        """
        Add the base class of the ontology to the classes when the ontology does not
        declare it.
        """
        if self.onto.base_cls_name in self.onto.classes:
            return
        self.onto.classes[self.onto.base_cls_name] = ClassInfo(
            name=self.onto.base_cls_name,
            uri="",
            superclasses=[Symbol.__name__],
            base_classes=[Symbol.__name__],
            label=f"Base class for {self.onto.ontology_label}",
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
        for p in self.onto.properties.values():
            if p.type == PropertyType.DATA_PROPERTY and not p.declared_domains:
                p.declared_domains = [self.onto.base_cls_name]

    def _determine_class_properties(self):
        """
        Decide which properties belong to which class based on inheritance and
        overrides.
        """
        for cls_name, info in self.onto.classes.items():
            ancestors = set(info.all_base_classes)
            declared: List[str] = []
            for pn, p in self.onto.properties.items():
                if pn == "roleFor":
                    continue
                if cls_name not in set(p.declared_domains):
                    continue
                if (
                    ancestors
                    and cls_name not in p._overrides_for
                    and any(
                        a in (set(p.declared_domains) | set(p.domains))
                        for a in ancestors
                    )
                ):
                    continue
                if p.is_specialized:
                    for sp in p.superproperties:
                        if sp in declared:
                            declared.remove(sp)
                declared.append(pn)
            info.declared_properties = sorted(set(declared))

    def _finalize_and_sort(self):
        """
        Compute transitive closures and determine final topological order for classes
        and properties.
        """
        for p in self.onto.properties.values():
            p.all_superproperties = self._compute_closure(
                initial=p.superproperties,
                items=self.onto.properties,
                key="superproperties",
            )

        for info in self.onto.classes.values():
            initial = set(info.all_base_classes)
            if info.role_taker:
                initial.add(info.role_taker.class_name)
            info.all_base_classes_including_role_takers = self._compute_closure(
                initial=list(initial),
                items=self.onto.classes,
                key="all_base_classes",
                role_key="role_taker",
            )

        if "roleFor" in self.onto.properties:
            del self.onto.properties["roleFor"]

        prop_classes = {
            k: v for k, v in self.onto.properties.items() if not v.is_specialized
        }
        props_order = OntologyInfo.topological_order(
            items=prop_classes, dep_key="superproperties"
        )

        # Subclass and role relations are only taken from the ontology (rdfs:subClassOf, owl:equivalentClass and
        # roleFor restrictions). Implicit subsumptions guessed from shared properties (find_implicit_subtypes) are not
        # entailed and are therefore not applied.

        for info in self.onto.classes.values():
            info.base_classes_for_topological_sort = info.base_classes[:]
            if info.role_taker:
                info.base_classes_for_topological_sort.append(
                    info.role_taker.class_name
                )

        for info in prop_classes.values():
            if not info.object_range_hint or "Union" not in info.object_range_hint:
                continue
            contesting_types = [
                t.strip() for t in info.object_range_hint[6:-1].split(",")
            ]
            for i, r in enumerate(contesting_types[:-1]):
                for r2 in contesting_types[i + 1 :]:
                    if (
                        r2
                        in self.onto.classes[r].all_base_classes_including_role_takers
                    ):
                        try:
                            contesting_types.remove(r)
                        except ValueError:
                            pass
                    elif (
                        r
                        in self.onto.classes[r2].all_base_classes_including_role_takers
                    ):
                        try:
                            contesting_types.remove(r2)
                        except ValueError:
                            pass
            if len(contesting_types) == 1:
                info.object_range_hint = contesting_types[0]

        removed_cls = []
        for cls_name, cls_info in self.onto.classes.items():
            if cls_name in removed_cls:
                continue
            for eq_cls in cls_info.equivalent_classes:
                removed_cls.append(eq_cls)
                initial_base_classes = copy(cls_info.base_classes)
                for base_cls in self.onto.classes[eq_cls].base_classes:
                    if base_cls not in cls_info.base_classes:
                        cls_info.base_classes.append(base_cls)
                        cls_info.all_base_classes.append(base_cls)
                        cls_info.all_base_classes_including_role_takers.append(base_cls)
                        cls_info.base_classes_for_topological_sort.append(base_cls)
                if (
                    len(initial_base_classes) == 1
                    and initial_base_classes[0] == self.onto.base_cls_name
                    and len(cls_info.base_classes) > 1
                ):
                    cls_info.base_classes.remove(self.onto.base_cls_name)
                for prop in self.onto.classes[eq_cls].declared_properties:
                    if prop not in cls_info.declared_properties:
                        cls_info.declared_properties.append(prop)
                for axiom in self.onto.classes[eq_cls].axioms:
                    if axiom not in cls_info.axioms:
                        cls_info.axioms.append(axiom)
                for python_axiom in self.onto.classes[eq_cls].axioms_python:
                    if python_axiom not in cls_info.axioms_python:
                        cls_info.axioms_python.append(python_axiom)
                for axiom_setup in self.onto.classes[eq_cls].axioms_setup:
                    if axiom_setup not in cls_info.axioms_setup:
                        cls_info.axioms_setup.append(axiom_setup)
        for rc in removed_cls:
            if rc in self.onto.classes:
                del self.onto.classes[rc]
        classes_order = OntologyInfo.topological_order(
            items=self.onto.classes, dep_key="base_classes_for_topological_sort"
        )

        idx_map = {n: i for i, n in enumerate(props_order)}
        for info in prop_classes.values():
            if info.inverse_of in prop_classes:
                info.inverse_target_is_prior = idx_map.get(
                    info.inverse_of, 1e9
                ) < idx_map.get(info.name, 1e9)

        return classes_order, props_order

    def _classes_annotating_the_base_class(
        self, *, render_classes: Dict[str, Dict], render_props: Dict[str, Dict]
    ) -> List[str]:
        """
        :return: The generated classes the field annotations of the base class name, which the base module imports
            only for type checking because they are defined in the module of the classes.
        """
        base_class = render_classes[self.onto.base_cls_name]
        annotations = " ".join(
            render_props[prop_name]["object_range_hint"] or ""
            for prop_name in base_class["declared_properties"]
        )
        named = set(re.findall(r"\w+", annotations))
        return [
            name
            for name in render_classes
            if name in named and name != self.onto.base_cls_name
        ]

    def _perform_rendering(
        self,
        *,
        base_file_name,
        classes_order,
        props_order,
    ):
        """
        Render all templates and produce the final Python files and stubs.
        """
        stubs_classes = deepcopy(self.onto.classes)
        for cls_name, info in self.onto.classes.items():
            if self.onto.role_cls_name in info.base_classes:
                info.base_classes.remove(self.onto.role_cls_name)
                info.base_classes.insert(
                    0, f"{self.onto.role_cls_name}[{info.role_taker.class_name}]"
                )
                stubs_classes[cls_name].base_classes.remove(self.onto.role_cls_name)
            else:
                info.add_role_taker = stubs_classes[cls_name].add_role_taker = False

        if Role.__name__ in self.onto.classes:
            del self.onto.classes[Role.__name__]

        # topological_order might still have 'Role' name if it was in the items keys
        # We need to filter the order as well
        classes_order = [c for c in classes_order if c != Role.__name__]
        for cls_info in self.onto.classes.values():
            cls_info.onto = None
        render_classes = {k: asdict(v) for k, v in self.onto.classes.items()}
        for c in render_classes.values():
            if c["role_taker"] is None:
                c["role_taker"] = {}

        for p_name, p_info in self.onto.properties.items():
            for eq_prop_name in p_info.equivalent_properties:
                p_info.equivalent_properties_descriptor_names.append(
                    self.onto.properties[eq_prop_name].descriptor_name
                )
            for dj_prop_name in p_info.disjoint_properties:
                p_info.disjoint_properties_descriptor_names.append(
                    self.onto.properties[dj_prop_name].descriptor_name
                )

        for prop_name, prop_info in self.onto.properties.items():
            if prop_info.chain_axioms:
                prop_info.base_descriptors.append(HasChainAxioms.__name__)
            property_uri = rdflib.URIRef(prop_info.uri)
            prop_info.rdfs_domains = sorted(
                NamingRegistry.uri_to_python_name(uri=domain)
                for domain in self.onto.graph.objects(property_uri, RDFS.domain)
                if isinstance(domain, rdflib.URIRef)
            )
            prop_info.rdfs_ranges = sorted(
                NamingRegistry.uri_to_python_name(uri=range_)
                for range_ in self.onto.graph.objects(property_uri, RDFS.range)
                if isinstance(range_, rdflib.URIRef)
            )

        for prop_info in self.onto.properties.values():
            prop_info.onto = None
        for stubs_info in stubs_classes.values():
            stubs_info.onto = None
        render_props = {k: asdict(v) for k, v in self.onto.properties.items()}
        render_stubs = {k: asdict(v) for k, v in stubs_classes.items()}
        for c in render_stubs.values():
            if c["role_taker"] is None:
                c["role_taker"] = {}

        p_mod, b_mod = f"{base_file_name}_properties", f"{base_file_name}_base"
        return {
            f"{p_mod}.py": self.renderer.render(
                template_name="onto_properties.j2",
                properties=render_props,
                properties_order=props_order,
            ),
            f"{b_mod}.py": self.renderer.render(
                template_name="onto_base.j2",
                cls=render_classes[self.onto.base_cls_name],
                properties=render_props,
                classes_module_name=base_file_name,
                annotated_classes=self._classes_annotating_the_base_class(
                    render_classes=render_classes, render_props=render_props
                ),
            ),
            f"{base_file_name}.py": self.renderer.render(
                template_name="onto_classes.j2",
                ontology_base_module_name=b_mod,
                properties_module_name=p_mod,
                classes=render_classes,
                properties=render_props,
                classes_order=classes_order,
                properties_order=props_order,
                ontology_base_class_name=self.onto.base_cls_name,
            ),
            f"{base_file_name}.pyi": self.renderer.render(
                template_name="onto_stubs.j2",
                ontology_base_module_name=b_mod,
                properties_module_name=p_mod,
                role_takers=list(
                    OrderedSet(
                        c["role_taker"]["class_name"]
                        for c in render_classes.values()
                        if c["role_taker"]
                    )
                ),
                classes=render_stubs,
                properties=render_props,
                classes_order=classes_order,
                ontology_base_class_name=self.onto.base_cls_name,
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
        res, stack = set(), list(initial)
        while stack:
            curr = stack.pop()
            if curr not in res:
                res.add(curr)
                if curr in items:
                    item = items[curr]
                    stack.extend(
                        getattr(item, key, [])
                        if hasattr(item, key)
                        else item.get(key, [])
                    )
                    rt = (
                        getattr(item, role_key, None)
                        if role_key and hasattr(item, role_key)
                        else (item.get(role_key) if role_key else None)
                    )
                    if rt:
                        stack.append(
                            rt.class_name
                            if hasattr(rt, "class_name")
                            else rt["class_name"]
                        )
        return sorted(res)
