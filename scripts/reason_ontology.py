#!/usr/bin/env python3
"""Run the HermiT reasoner over ontology/windturbine.owl (as Protégé's "Start reasoner").

  reason_ontology.py            -> ontology/inferred.json

Checks consistency and unsatisfiable classes, and records what the reasoner adds:
the members of the defined classes (e.g. DrivetrainFault via the transitive partOf),
further inferred types of individuals, and inferred superclasses. Needs owlready2
(bundles HermiT) and Java.
"""
import json
import os

import owlready2 as ow

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ONTO = os.path.join(ROOT, "ontology", "windturbine.owl")


def short(e):
    return e.name if hasattr(e, "name") else str(e)


def named(classes):
    return {c for c in classes if isinstance(c, ow.ThingClass)}


def main():
    world = ow.World()
    onto = world.get_ontology("file://" + ONTO).load()
    classes = list(onto.classes())
    individuals = list(onto.individuals())
    asserted_types = {i: named(i.is_a) for i in individuals}
    asserted_supers = {c: named(c.is_a) for c in classes}
    result = {"reasoner": "HermiT (owlready2 %s)" % ow.VERSION, "consistent": True, "unsatisfiable": []}
    try:
        with onto:
            ow.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
    except ow.OwlReadyInconsistentOntologyError as e:
        result["consistent"] = False
        result["error"] = str(e)[:2000]
    if result["consistent"]:
        result["unsatisfiable"] = sorted(short(c) for c in world.inconsistent_classes())
        defined = [c for c in classes if c.equivalent_to]
        result["defined_class_members"] = {short(c): sorted(short(i) for i in c.instances()) for c in defined}
        inferred_types = {}
        for i in individuals:
            everything = {c for c in i.INDIRECT_is_a if isinstance(c, ow.ThingClass)}
            added = {c for c in everything if c not in asserted_types[i]
                     and not any(issubclass(a, c) for a in asserted_types[i])}
            if added:
                inferred_types[short(i)] = sorted(short(c) for c in added if c is not ow.Thing)
        result["inferred_types"] = inferred_types
        inferred_supers = {}
        for c in classes:
            now = named(c.is_a)
            added = now - asserted_supers[c]
            if added:
                inferred_supers[short(c)] = sorted(short(x) for x in added if x is not ow.Thing)
        result["inferred_superclasses"] = {k: v for k, v in inferred_supers.items() if v}
    with open(os.path.join(ROOT, "ontology", "inferred.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: (v if k != "defined_class_members" else {c: len(m) for c, m in v.items()})
                      for k, v in result.items() if k not in ("inferred_types",)}, indent=1))
    print("individuals with inferred types:", len(result.get("inferred_types", {})))


if __name__ == "__main__":
    main()
