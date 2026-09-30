# OWLmatic handover

Handover notes for the `owlmatic` branch: converting dataclass hierarchies (for example
`SemanticAnnotation` from `semantic_digital_twin`) into OWL ontologies.

## Status

| Part | State |
|---|---|
| `OWLConverter` (faithful conversion of the Python structure) | Done, tested |
| Conversion of the real `SemanticAnnotation` hierarchy | Works, no name clashes |
| Narrowing through generic parameters (`HasRootBody`) | Missing, see [Known gaps](#known-gaps) |
| Normalised ontology (mixins, `some`, `hasPart`) | Designed, not implemented, see [Proposed design](#proposed-design-the-mixin-marker) |

## What is on the branch

| File | Content |
|---|---|
| `krrood/src/krrood/ontomatic/owl_converter.py` | `OWLConverter` |
| `krrood/src/krrood/ontomatic/failures.py` | `DuplicateOWLClassName` exception (added) |
| `test/krrood_test/test_ontomatic/test_owl_converter.py` | 13 tests |
| `test/krrood_test/dataset/owl_conversion/` | Mimic classes for the tests (krrood stays self-contained) |

## Usage

```python
from rdflib import URIRef
from krrood.ontomatic.owl_converter import OWLConverter
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

## Current mapping rules

The converter builds on `krrood.class_diagrams.ClassDiagram` and does not introspect
dataclasses itself.

- The root classes and all their subclasses become `owl:Class`. Python inheritance becomes
  `rdfs:subClassOf`.
- A field that refers to instances of another dataclass becomes an `owl:ObjectProperty` named
  after the field.
- The type of a field is stated on the class that declares it:
  `Class SubClassOf field only Target`. It is not stated globally with `rdfs:domain`/`rdfs:range`,
  because the same field name has different types on different classes.
- A field that is not a container also gets `field max 1`.
- A subclass repeats a restriction only when it narrows the field type by redeclaring the field.
- Dataclasses that fields refer to but that lie outside the hierarchy (for `SemanticAnnotation`:
  `Body`, `Region`, `KinematicStructureEntity`, `PrefixedName`, `SimulatorAdditionalProperty`)
  are declared as classes, without their own subclasses and fields.
- Skipped: builtin-typed fields (`str`, `int`, ...), `Type[X]` fields, private fields, and
  relations inferred through roles.
- Two converted classes with the same name raise `DuplicateOWLClassName`.

Result for `SemanticAnnotation`: 151 classes (146 in the hierarchy), 21 object properties,
589 triples, about 3 seconds.

## Verification

- `test_owl_converter.py`: 13 passed. The neighbouring `test_ontomatic` and `test_class_diagram`
  suites pass too (77 in total).
- A mutation check confirmed the inherited-relation test fails when the filter is removed.
- Converted the real `SemanticAnnotation` hierarchy and inspected it in a Protégé-style browser.
  The browser is a private claude.ai artifact and not part of the repository.

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
venv/bin/python -m pytest test/krrood_test/test_ontomatic/test_owl_converter.py
```

With only krrood installed, the new tests also run with `--noconftest`, since they use no shared
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
3. **No datatype properties.** Builtin-typed fields are skipped.
4. **`Optional[List[X]]` fields** produce no relation, because the class diagram creates no
   association for them.
5. **No annotations.** Docstrings are not emitted as `rdfs:comment`.
6. **The type hints do not always match the defaults.** For example,
   `HasSupportingSurface.supporting_surface: Region = field(default=None)` is typed as required
   but defaults to `None`, so the converter treats it as required.

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
4. **Property naming.** Keep field names (`handle`) or use the usual OWL style (`hasHandle`)?
   With `hasHandle`, the property and the defined class `HasHandle` would differ only in case.
5. **Should the faithful conversion stay** as a separate mode next to the normalised one? It is
   still useful for inspecting the code structure.

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
