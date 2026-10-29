# OWLmatic handover

Handover notes for the `owlmatic` branch: converting dataclass hierarchies (for example
`SemanticAnnotation` from `semantic_digital_twin`) into OWL ontologies. OWLmatic is the reverse
of Ontomatic's OWL-to-Python generator (`krrood/ontomatic/ontology_to_python`, on the `dl2`
branch of the `bass` fork, not on `main`): it works on any dataclass hierarchy, and converting
Python that Ontomatic generated back into OWL evaluates what the round trip keeps.

## Status

| Part | State |
|---|---|
| `OWLConverter` on any dataclass hierarchy | Done, tested |
| Ontomatic's conventions in reverse (descriptors, roles, datatype properties, base class) | Done, tested |
| `OntologyComparison` (axiom-level evaluation of two ontologies) | Done, tested |
| Round trip OWL → Ontomatic → Python → OWLmatic → OWL | Done on a small ontology, see [Round-trip evaluation](#round-trip-evaluation-against-ontomatic) |
| Conversion of the real `SemanticAnnotation` hierarchy | Works, no name clashes |
| Narrowing through generic parameters (`HasRootBody`) | Missing, see [Known gaps](#known-gaps) |
| Normalised ontology (mixins, `some`, `hasPart`) | Designed, not implemented, see [Proposed design](#proposed-design-the-mixin-marker) |

## What is on the branch

| File | Content |
|---|---|
| `krrood/src/krrood/owlmatic/owl_converter.py` | `OWLConverter` |
| `krrood/src/krrood/owlmatic/ontology_comparison.py` | `OntologyComparison`, `OntologyAxioms`, `Axiom` |
| `krrood/src/krrood/owlmatic/failures.py` | `DuplicateOWLClassName`, `PropertyIsBothObjectAndDatatype` |
| `test/krrood_test/test_owlmatic/` | Converter, comparison and round-trip tests |
| `test/krrood_test/dataset/owl_conversion/` | Mimic classes (krrood stays self-contained), `university.ttl` and the Python Ontomatic generates from it |

The converter first lived in `krrood.ontomatic`; it moved to its own `krrood.owlmatic` package
beside `ontomatic` and `ormatic`.

## Usage

```python
from rdflib import URIRef
from krrood.owlmatic.owl_converter import OWLConverter
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

# Subclasses are found among loaded classes: import every module defining annotations first.
converter = OWLConverter(
    root_classes=[SemanticAnnotation],
    ontology_iri=URIRef("http://cram2.github.io/semantic_digital_twin/semantic_annotations"),
)
converter.convert().serialize("semantic_annotations.owl", format="xml")  # opens in Protégé
```

`convert()` returns an `rdflib.Graph`; classes and properties are named in the namespace
`<ontology_iri>#`.

Evaluating a round trip through Ontomatic, where `UniversityThing` is the base class Ontomatic
generated for the ontology:

```python
from rdflib import Graph
from krrood.owlmatic.ontology_comparison import OntologyComparison

original = Graph().parse("university.ttl")
converted = OWLConverter(
    root_classes=[UniversityThing],
    ontology_iri=URIRef("http://example.org/university"),
    thing_class=UniversityThing,
).convert()
print(OntologyComparison(original, converted).report())
```

## Current mapping rules

The converter builds on `krrood.class_diagrams.ClassDiagram` (with the descriptor-aware
introspector) and does not introspect dataclasses itself. Each rule is the reverse of what
Ontomatic generates.

| Python | OWL | Ontomatic's direction |
|---|---|---|
| Root classes and all their subclasses | `owl:Class`, `rdfs:subClassOf` | classes, base classes |
| Class docstring | `rdfs:comment` | label and comment into the docstring |
| `thing_class` (for example Ontomatic's `<Label>Thing`) | `owl:Thing`: not declared, no `subClassOf` to it, its fields' properties get no domain | the base class it generates for every ontology |
| Field referring to a dataclass | `owl:ObjectProperty` | object property → `Set[Range]` field |
| Field holding `str`, `int`, `float`, `bool`, `datetime` | `owl:DatatypeProperty` with the XSD range | datatype property → `Optional[type]` field |
| Field `works_for` | property `worksFor` | `to_snake_case(worksFor)` |
| Field managed by descriptor `WorksFor` | property `worksFor`, whatever the field is called | `to_pascal_case(worksFor)` |
| Descriptor subclass `WorksFor(MemberOf)` | `worksFor rdfs:subPropertyOf memberOf` | superproperties → descriptor bases |
| Descriptor with `HasInverseProperty` among its own bases | `owl:inverseOf get_inverse()` | inverse → `HasInverseProperty` |
| Descriptor that is a `TransitiveProperty` | `owl:TransitiveProperty` | transitive → `TransitiveProperty` |
| `Role[Taker]` | `Role SubClassOf roleFor some Taker`, `roleFor` declared | a `roleFor` restriction → role taker |

- A property that one class and its subclasses declare states that class as `rdfs:domain` and
  the field type as `rdfs:range`. A subclass that narrows the type gets `field only Narrower`.
  A property declared by unrelated classes (for example `name` on `Person` and on
  `RecognizedGroup`) gets no domain or range, only `field only Type` on each declaring class,
  because the type can differ between them.
- A field that is not a container also gets `field max 1`.
- Inverses are stated only where a descriptor declares one itself: `WorksFor(MemberOf)` inherits
  `get_inverse()` in Python, but `worksFor inverseOf member` would say that every `member` link
  is a `worksFor` link, which Python does not do. It already follows from
  `worksFor subPropertyOf memberOf` and `memberOf inverseOf member`.
- Dataclasses that fields or roles refer to but that lie outside the hierarchy are declared as
  classes, without their own subclasses and fields.
- Skipped: `Type[X]` fields, private fields, role-taker fields, relations inferred through roles,
  and fields of other types (enums, for example).
- Two converted classes with the same name raise `DuplicateOWLClassName`; a property that would be
  an object property on one class and a datatype property on another raises
  `PropertyIsBothObjectAndDatatype`.
- Roles are converted when they are among the root classes or their subclasses; they are not
  searched for among loaded classes.

Result for `SemanticAnnotation` (against this branch's `semantic_digital_twin`): 148 classes,
19 object properties, 6 datatype properties, 738 triples, under a second. The classes are the
same as the first version of the converter produced; the object properties are the same up to
their new camelCase names.

## Round-trip evaluation against Ontomatic

`test/krrood_test/dataset/owl_conversion/university.ttl` is a small ontology with a class
hierarchy, object properties with domains and ranges, a subproperty, an inverse pair, a transitive
property, a datatype property and a role. `ontomatic_generated_university.py` next to it is what
Ontomatic's `OwlToPythonConverter` generated from it on `dl2` (commit `df5158820e`), joined into
one module and with what `main` does not provide left out (its docstring lists what). Converting
that Python back and comparing it with the original:

```
axiom kind                        recall  precision
class declaration                   1.00       1.00
object property declaration         1.00       1.00
datatype property declaration       1.00       0.50
subclass of                         1.00       1.00
subproperty of                      1.00       1.00
inverse of                          1.00       1.00
domain                              1.00       1.00
range                               1.00       0.86
property characteristic             1.00       1.00
restriction                         0.00       0.00
all                                 0.97       0.91

missing:
restriction: Student roleFor someValuesFrom Person

extra:
datatype property declaration: uri
range: uri string
restriction: Person name maxCardinality 1
```

`test_round_trip_through_ontomatic.py` pins exactly this missing and extra set. What it says about
Ontomatic:

1. **The role is lost.** Ontomatic reads roles from `roleFor` restrictions, but generated
   `Student` as a plain subclass of the base class without a role taker, so nothing can bring the
   restriction back.
2. **Declaring `roleFor` crashes Ontomatic.** With `:roleFor a owl:ObjectProperty` in the ontology,
   rendering fails with `UndefinedError: 'dict object' has no attribute 'roleFor'`. The fixture
   leaves the declaration out for that reason. OWLmatic declares `roleFor` (OWL 2 DL requires it),
   so its output with roles cannot be fed back into Ontomatic until that is fixed.
3. **Ontomatic adds a `uri` field** to the generated base class, which comes back as an extra
   datatype property.
4. **Datatype properties become single optional values** (`Optional[str]`), so every datatype
   property comes back functional (`max 1`), whether or not the ontology said so.

How the comparison works: axioms are compared syntactically, with terms reduced to local names so
the two ontologies may use different namespaces. It compares declarations of named classes and
properties, named `subClassOf`/`equivalentClass`/`disjointWith`, `subPropertyOf`/`inverseOf`/
`equivalentProperty`/`propertyDisjointWith`, domains and ranges, property characteristics, and
restrictions on a property that named classes are declared subclasses of. Symmetric relations
(`inverseOf`, ...) ignore order. Anonymous class expressions other than those restrictions, and
annotations, are not compared, and nothing is inferred: an axiom entailed by others but not
stated counts as missing.

### Reproducing the Ontomatic side

Ontomatic's generator is not on `main` and needs `dl2`'s version of krrood plus the
ripple-down-rules package as it was when `dl2` was written (commit `9fd0230c29` of this repository,
where the package sits at the repository root):

```bash
git worktree add ../dl2 bass/dl2 --detach
mkdir -p rdr/ripple_down_rules && git archive 9fd0230c29 | tar x -C rdr/ripple_down_rules
pip install --target rdr ordered_set
cd ../dl2 && PYTHONPATH=krrood/src:<path to>/rdr python -c "
from krrood.ontomatic.ontology_to_python.owl_to_python import OwlToPythonConverter
converter = OwlToPythonConverter()
converter.load_ontology('university.ttl')
converter.save_to_file('out/university.py')"
```

## Verification

- `test/krrood_test/test_owlmatic/`: 66 passed. With the shared conftest, `test_owlmatic`,
  `test_ontomatic` and `test_class_diagram` run 130 passed.
- Mutation checks: stating inherited inverses, always restricting to the field type, keeping the
  signature `dataclass` writes as documentation, dropping the inherited-field filter and
  converting the role-taker field each make at least one test fail.
- Converted the real `SemanticAnnotation` hierarchy with the first version and this one and
  compared the classes and properties.

### Running the tests in a fresh environment

The shared `test/conftest.py` imports `semantic_digital_twin`, which pulls in the local
`giskardpy` and `physics_simulators`. The uv installed in the development container (0.8.17)
rejected the per-package `override-dependencies` entries in the workspace `[tool.uv]` table, so
plain pip in a Python 3.12 venv was used instead (a current uv should work as usual):

```bash
python3.12 -m venv venv
venv/bin/python -m pip install -e random_events -e probabilistic_model -e krrood \
    -e semantic_digital_twin -e giskardpy -e physics_simulators \
    "pytest>=7,<8" pytest-xdist pytest-rerunfailures "pytest-asyncio<0.24"
venv/bin/python -m pytest test/krrood_test/test_owlmatic
```

With only krrood installed, the tests also run with `--noconftest`, since they use no shared
fixtures.

Running the krrood test suite regenerates
`test/krrood_test/test_eql/test_verbalization/verbalization_results.py` (import order only) and
creates `.living_worlds_tally/`. Neither belongs in a commit.

## Known gaps

1. **Narrowing through generic parameters is lost.** `HasRootBody(HasRootKinematicStructureEntity[Body])`
   narrows `root` to `Body` without redeclaring the field. The class diagram reduces the type
   variable to its bound, so only `root only KinematicStructureEntity` is emitted and the
   `root only Body` restriction is missing for 116 classes. Fix it test-first with a generic mimic
   class, resolving the type each subclass binds the parameter to (see
   `krrood.patterns.subclass_safe_generic.SubClassSafeGeneric`).
2. **Only `only` and `max 1` restrictions.** A universal restriction is satisfied trivially when a
   value is absent, so a reasoner cannot classify anything from the current output. Required
   fields should also get `field some Target`.
3. **`Optional[List[X]]` fields** produce no relation, because the class diagram creates no
   association for them.
4. **The type hints do not always match the defaults.** For example,
   `HasSupportingSurface.supporting_surface: Region = field(default=None)` is typed as required
   but defaults to `None`, so the converter treats it as required.
5. **Only the property mixins `main` has are mapped.** `dl2` adds symmetric, asymmetric,
   reflexive, irreflexive, equivalent, disjoint and chain-axiom mixins; each is one entry in
   `OWLConverter.property_characteristics` (or one more rule) once they land.
6. **Names with digits do not survive Ontomatic.** `door_0` becomes `door0`, which Ontomatic's
   `to_snake_case` turns into `door0`, not back into `door_0`.
7. **Class axioms are not converted.** Ontomatic turns restrictions into EQL `axiom` class
   methods; those, `equivalentClass` aliases and `cls_uri` are not read back.

## Design discussion: mixins are not kinds

In Python, inheriting from a mixin such as `HasHandle` or `HasSupportingSurface` is mostly code
reuse. In OWL, `Drawer SubClassOf HasHandle` says every drawer is a member of the class
`HasHandle`, which tangles the hierarchy: 18 classes have more than one asserted parent, and
`HasRootBody` sits above 116 classes.

Relevant best practices:

- **Rector normalisation ("untangling").** Keep the asserted hierarchy a tree of primitive
  kinds, and express every other grouping as a defined class (`owl:equivalentClass`) so a
  reasoner infers the polyhierarchy.
- **OntoClean (Guarino and Welty).** A dependent class (`HasHandle` depends on a `Handle`) should
  not subsume an independent kind such as `Drawer`.
- **Noy and McGuinness, *Ontology Development 101*.** If a distinction only changes property
  values, model it as a property or restriction, not as a subclass.
- **Rector et al., "OWL Pizzas".** `only` does not imply `some`, and defined classes need
  `some`.
- **W3C note on part-whole relations.** Model parts as sub-properties of one `hasPart` property,
  and use either `hasPart` or `partOf`, not both. The code already marks part fields with
  `IsPartWholeRelationship` field metadata.
- **SOMA / DUL.** SOMA, the robot activity ontology from the same institute, models capabilities
  such as "can support objects" as dispositions rather than subclasses. It is a possible
  alignment target for `HasSupportingSurface`.

Target shape for `Drawer`:

```
# now
Class: Drawer  SubClassOf: Furniture, HasCaseAsRootBody, HasHandle, HasMechanicalJoint

# normalised
ObjectProperty: handle   SubPropertyOf: hasPart
Class: HasHandle         EquivalentTo: handle some Handle
Class: Drawer            SubClassOf: Furniture, handle some Handle, mechanical_joint some MechanicalJoint
# a reasoner infers: Drawer SubClassOf HasHandle
```

## Proposed design: the mixin marker

Add a class decorator `@mixin` in `krrood/patterns`, stating that a class is combined into kinds
and is not a kind itself:

```python
@mixin
@dataclass(eq=False)
class HasHandle(HasRootBody, PartWholeRelationship):
    handle: Optional[Handle] = field(default=None, metadata=IsPartWholeRelationship().as_dict())
```

- **The name describes the Python class, not the OWL output.** It is not called `@owlproperty`,
  because the class does not become a property: its field does. The converter decides what the
  marker means in OWL, and other tools (class diagrams, ORM) can use the same marker.
- **The marker must not be inherited.** A marker base class, a `ClassVar` or a `getattr` lookup
  would make `Drawer` a mixin too. Store the marker in the decorated class's own `__dict__` and
  read it only from there, as `@dataclass` does with `__dataclass_params__`. It is the class-level
  counterpart of `krrood.patterns.field_metadata.FieldMetadata`.
- **One marker covers both kinds of mixin.** The converter tells them apart by structure: relation
  mixins such as `HasHandle` declare fields, while behaviour-only mixins such as
  `HasCaseAsRootBody` declare none.

What the converter would do with a marked class:

1. Leave it out of the asserted hierarchy. Each subclass gets its nearest unmarked ancestors as
   parents, and parents already implied by another parent are dropped.
2. Move its field restrictions down onto those subclasses: `only`, plus `some` for required fields.
3. For relation mixins, optionally emit a defined class, for example
   `HasHandle ≡ handle some Handle`.
4. Make fields carrying `IsPartWholeRelationship` sub-properties of `hasPart`.

Simulated on the exported data by marking the 15 classes in
`semantic_annotations/mixins.py`: classes with more than one asserted parent drop from 18 to 10.
All 10 remaining extra parents are already implied (for example, `Drawer` gets `Furniture` and
`SemanticAnnotation`, and `SemanticAnnotation` is already above `Furniture`), so dropping implied
parents leaves a single tree.

## Open questions for the team

1. **Required or optional?** `HasHandle.handle` is `Optional[Handle]`: in Python a drawer without
   a handle is still a `HasHandle`, but a reasoner would not put it in the defined class
   `handle some Handle`. Should kinds such as `Drawer` require their parts (`some`), or does the
   mixin only mean "may have"?
2. **Which classes are mixins?** `PartWholeRelationship` and `IsStorageSpace` are borderline;
   storage space could be a real category.
3. **Which reasoner?** `owlrl` (already a krrood dependency) only classifies individuals.
   Inferring `Drawer SubClassOf HasHandle` at class level needs HermiT or ELK.
4. **Property naming.** Properties are now the camelCase of the field name (`supporting_surface`
   → `supportingSurface`), because that is what Ontomatic turns back into the field name. Adding a
   `has` prefix (`hasHandle`) would break that, and the property and the defined class
   `HasHandle` would differ only in case. Is camelCase of the field name the convention to keep?
5. **Should the faithful conversion stay** as a separate mode next to the normalised one? It is
   still useful for inspecting the code structure.
6. **Fix Ontomatic's role handling?** The round trip loses roles, and declaring `roleFor` crashes
   Ontomatic (see [Round-trip evaluation](#round-trip-evaluation-against-ontomatic)). Both are on
   `dl2`, not on this branch.

## Suggested next steps

Each step test-first with mimic classes in `test/krrood_test/dataset/owl_conversion/`:

1. Fix narrowing through generic parameters.
2. Add `@mixin` to `krrood/patterns`, and teach the converter to leave marked classes out and drop
   implied parents.
3. Add `some` for required fields, defined classes for relation mixins, and `hasPart`
   sub-properties.
4. In a separate pull request in `semantic_digital_twin`, apply `@mixin` to the classes the team
   agrees on.

## Sources

- [Normalisation ontology design pattern (Manchester ODP catalogue)](http://www.gong.manchester.ac.uk/odp/html/Normalisation.html)
- [Rector, Modularisation of domain ontologies implemented in description logics (K-CAP 2003)](http://www.cs.man.ac.uk/~rector/papers/rector-modularisation-kcap-2003-distrib.pdf)
- [Guarino and Welty, An Overview of OntoClean](https://www.loa.istc.cnr.it/old/Papers/GuarinoWeltyOntoCleanv3.pdf)
- [Noy and McGuinness, Ontology Development 101](https://protege.stanford.edu/publications/ontology_development/ontology101.pdf)
- [Rector et al., OWL Pizzas: Common Errors and Common Patterns](https://www.cs.man.ac.uk/~rector/papers/common_errors_ekaw_2004.pdf)
- [W3C, Simple part-whole relations in OWL Ontologies](https://www.w3.org/2001/sw/BestPractices/OEP/SimplePartWhole/simple-part-whole-relations-v1.3.html)
- [Beßler et al., Foundations of the Socio-physical Model of Activities (SOMA)](https://arxiv.org/pdf/2011.11972)
