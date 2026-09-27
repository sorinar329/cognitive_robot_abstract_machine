#!/usr/bin/env python3
"""Protégé-style viewer for the wind turbine ontology.

  build_ontology_viewer.py      -> preview/ontology/index.html

Reads ontology/windturbine.ttl (what build_ontology.py wrote), the reasoner results
in ontology/inferred.json (reason_ontology.py) and the AICOR alignment, renders every
class expression in Manchester syntax and fills gallery/ontology.html.
"""
import collections
import json
import os
import sys

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.collection import Collection
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, XSD

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ONTO_DIR = os.path.join(ROOT, "ontology")
OUT = os.path.join(ROOT, "preview", "ontology")
WT = "http://www.semanticweb.org/windturbine-twin/ontology#"

g = Graph().parse(os.path.join(ONTO_DIR, "windturbine.ttl"))
inferred = json.load(open(os.path.join(ONTO_DIR, "inferred.json")))
alignment = Graph().parse(os.path.join(ONTO_DIR, "windturbine-aicor-alignment.ttl"))

classes = sorted({s for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}, key=str)
oprops = sorted(set(g.subjects(RDF.type, OWL.ObjectProperty)), key=str)
dprops = sorted(set(g.subjects(RDF.type, OWL.DatatypeProperty)), key=str)
individuals = sorted(set(g.subjects(RDF.type, OWL.NamedIndividual)), key=str)
kind = {**{c: "c" for c in classes}, **{p: "o" for p in oprops}, **{p: "d" for p in dprops}, **{i: "i" for i in individuals}}


def short(u):
    s = str(u)
    for prefix, name in ((WT, ""), (str(OWL), "owl:"), (str(XSD), "xsd:"), (str(RDFS), "rdfs:"), ("http://aicor.knowledge/l2-ontology.owl#", "aicor:")):
        if s.startswith(prefix):
            return name + s[len(prefix):]
    return s


def tok(u):
    if u == OWL.Thing:
        return [["c", "owl:Thing"]]
    return [[kind.get(u, "t"), short(u)]]


def items(node):
    return list(Collection(g, node))


def render(node):
    """Manchester syntax as tokens [kind, text]: c class, o/d property, i individual, k keyword, t text."""
    if isinstance(node, URIRef):
        return tok(node)
    if isinstance(node, Literal):
        return [["t", f'"{node}"' + (f"^^{short(node.datatype)}" if node.datatype else "")]]
    if (node, RDF.type, OWL.Restriction) in g:
        prop = g.value(node, OWL.onProperty)
        head = tok(prop)
        for pred, word in ((OWL.someValuesFrom, "some"), (OWL.allValuesFrom, "only"), (OWL.hasValue, "value")):
            v = g.value(node, pred)
            if v is not None:
                return head + [["k", f" {word} "]] + wrap(v)
        for pred, word in ((OWL.qualifiedCardinality, "exactly"), (OWL.minQualifiedCardinality, "min"),
                           (OWL.maxQualifiedCardinality, "max")):
            v = g.value(node, pred)
            if v is not None:
                return head + [["k", f" {word} "], ["t", f"{v} "]] + wrap(g.value(node, OWL.onClass))
    for pred, word in ((OWL.intersectionOf, "and"), (OWL.unionOf, "or")):
        lst = g.value(node, pred)
        if lst is not None:
            out = []
            for n, part in enumerate(items(lst)):
                if n:
                    out.append(["k", f" {word} "])
                out += wrap(part)
            return out
    lst = g.value(node, OWL.oneOf)
    if lst is not None:
        out = [["t", "{"]]
        for n, part in enumerate(items(lst)):
            if n:
                out.append(["t", ", "])
            out += tok(part)
        return out + [["t", "}"]]
    return [["t", "?"]]


def wrap(node):
    r = render(node)
    return [["t", "("]] + r + [["t", ")"]] if isinstance(node, BNode) and len(r) > 1 else r


def annotations(s):
    out = []
    for p, pname in ((RDFS.label, "rdfs:label"), (RDFS.comment, "rdfs:comment"), (DCTERMS.source, "dcterms:source")):
        for o in g.objects(s, p):
            out.append([pname, str(o), o.language or ""])
    return out


disjoint = collections.defaultdict(set)
for n in g.subjects(RDF.type, OWL.AllDisjointClasses):
    members = items(g.value(n, OWL.members))
    for a in members:
        disjoint[a] |= {b for b in members if b != a}

def_members = inferred.get("defined_class_members", {})
inf_supers = inferred.get("inferred_superclasses", {})
inf_types = inferred.get("inferred_types", {})

data = {"classes": {}, "objectProperties": {}, "dataProperties": {}, "individuals": {}}
for c in classes:
    name = short(c)
    supers = list(g.objects(c, RDFS.subClassOf))
    asserted_instances = sorted(short(i) for i in g.subjects(RDF.type, c) if i in kind)
    data["classes"][name] = {
        "iri": str(c), "ann": annotations(c),
        "parents": sorted(short(s) for s in supers if isinstance(s, URIRef)),
        "superExpr": [render(s) for s in supers if isinstance(s, BNode)],
        "equivalent": [render(e) for e in g.objects(c, OWL.equivalentClass)],
        "disjoint": sorted(short(x) for x in disjoint[c]),
        "instances": asserted_instances,
        "inferredInstances": sorted(set(def_members.get(name, [])) - set(asserted_instances)),
        "inferredParents": inf_supers.get(name, []),
        "aicor": [short(o) for o in alignment.objects(c, RDFS.subClassOf)],
    }
for i in individuals:
    name = short(i)
    types = sorted(short(t) for t in g.objects(i, RDF.type) if t != OWL.NamedIndividual)
    for t in types:
        data["classes"].get(t, {}).setdefault("instances", [])
    data["individuals"][name] = {
        "iri": str(i), "ann": annotations(i), "types": types, "inferredTypes": inf_types.get(name, []),
        "objects": sorted([short(p), short(o)] for p, o in g.predicate_objects(i) if p in kind and kind[p] == "o"),
        "data": sorted([short(p), str(o), short(o.datatype) if o.datatype else ""] for p, o in g.predicate_objects(i)
                       if p in kind and kind[p] == "d"),
        "usedBy": sorted([short(s), short(p)] for s, p in g.subject_predicates(i) if p in kind and kind[p] == "o"),
    }
CHARS = [("Functional", OWL.FunctionalProperty), ("Inverse functional", OWL.InverseFunctionalProperty),
         ("Transitive", OWL.TransitiveProperty), ("Symmetric", OWL.SymmetricProperty), ("Asymmetric", OWL.AsymmetricProperty),
         ("Reflexive", OWL.ReflexiveProperty), ("Irreflexive", OWL.IrreflexiveProperty)]
for bucket, props in (("objectProperties", oprops), ("dataProperties", dprops)):
    for p in props:
        inverse = [short(x) for x in g.objects(p, OWL.inverseOf)] + [short(x) for x in g.subjects(OWL.inverseOf, p)]
        usage = sum(1 for _ in g.subject_objects(p))
        data[bucket][short(p)] = {
            "iri": str(p), "ann": annotations(p),
            "chars": [[n, (p, RDF.type, t) in g] for n, t in (CHARS if bucket == "objectProperties" else CHARS[:1])],
            "domain": [render(x) for x in g.objects(p, RDFS.domain)], "range": [render(x) for x in g.objects(p, RDFS.range)],
            "inverse": sorted(set(inverse)), "parents": [short(x) for x in g.objects(p, RDFS.subPropertyOf)], "usage": usage,
        }

counts = collections.Counter()
for s, p, o in g:
    if p == RDFS.subClassOf:
        counts["SubClassOf"] += 1
    elif p == OWL.equivalentClass:
        counts["EquivalentClasses"] += 1
    elif p == RDF.type and o in kind and kind[o] == "c":
        counts["ClassAssertion"] += 1
    elif p in kind and kind[p] == "o":
        counts["ObjectPropertyAssertion"] += 1
    elif p in kind and kind[p] == "d":
        counts["DataPropertyAssertion"] += 1
    elif p in (RDFS.label, RDFS.comment, DCTERMS.source) and not isinstance(s, BNode):
        counts["AnnotationAssertion"] += 1
    elif p in (RDFS.domain,):
        counts["Domain"] += 1
    elif p in (RDFS.range,):
        counts["Range"] += 1
    elif p == OWL.inverseOf:
        counts["InverseObjectProperties"] += 1
counts["DisjointClasses"] = sum(1 for _ in g.subjects(RDF.type, OWL.AllDisjointClasses))
counts["DifferentIndividuals"] = sum(1 for _ in g.subjects(RDF.type, OWL.AllDifferent))
counts["Characteristics"] = sum(1 for p in list(oprops) + list(dprops) for _, t in CHARS if (p, RDF.type, t) in g)
onto_iri = next(g.subjects(RDF.type, OWL.Ontology))
data["ontology"] = {
    "iri": str(onto_iri), "version": str(g.value(onto_iri, OWL.versionInfo)),
    "ann": [["dcterms:title", str(g.value(onto_iri, DCTERMS.title)), "en"],
            ["dcterms:description", str(g.value(onto_iri, DCTERMS.description)), "en"],
            ["dcterms:source", str(g.value(onto_iri, DCTERMS.source)), ""]],
    "metrics": {"Axioms": sum(counts.values()), "Classes": len(classes), "Object properties": len(oprops),
                "Data properties": len(dprops), "Individuals": len(individuals), "Triples": len(g)},
    "axioms": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    "expressivity": "SOIQ(D)",
    "reasoner": {k: inferred[k] for k in ("reasoner", "consistent", "unsatisfiable") if k in inferred},
    "alignment": sorted([short(s), short(o)] for s, o in alignment.subject_objects(RDFS.subClassOf)),
    "prefixes": [["wt:", WT], ["aicor:", "http://aicor.knowledge/l2-ontology.owl#"], ["owl:", str(OWL)], ["rdfs:", str(RDFS)],
                 ["xsd:", str(XSD)], ["dcterms:", str(DCTERMS)]],
}

os.makedirs(OUT, exist_ok=True)
html = open(os.path.join(ROOT, "gallery", "ontology.html")).read()
html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
source = {"ttl": open(os.path.join(ONTO_DIR, "windturbine.ttl")).read(), "owl": open(os.path.join(ONTO_DIR, "windturbine.owl")).read()}
html = html.replace("__SOURCE__", json.dumps(source, ensure_ascii=False).replace("</", "<\\/"))
with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print(f"wrote {OUT}/index.html ({len(html) // 1024} KB): {len(classes)} classes, {len(oprops)} object properties, "
      f"{len(dprops)} data properties, {len(individuals)} individuals")
