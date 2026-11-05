# Ontomatic port: standing rules

- Ontomatic's source is AbdelrhmanBassiouny's `aamas27-experiments` (`dl` + 13 `[Ontomatic]` commits),
  the code the paper's experiments ran on. `main` has only `ontomatic/property_descriptor/` and
  `failures.py`, evolved separately (`krrood.patterns.role.Role`, `krrood.symbol_graph`,
  `krrood.patterns.caching`); the port builds on those.
- Rewrite in place, never merge: the branch targets the old EQL API (`entity_query_language.entity`,
  `.symbolic`, `class_diagrams.utils.Role`), uses relative imports, external `ripple_down_rules`,
  `line_profiler` and string-keyed dicts, and upstream's `property_descriptor` diverged by
  +1 025/-236 lines.
- Generated object properties are collections, so the compiler needs the correlated-quantifier EQL
  semantics (decided in the source plan, 2026-10-13); that work is not tracked in this plan.
- TBox domain/range derivation uses explicit OWL rules (prp-dom/rng via inverse, equivalent and
  super-properties; scm-dom2/rng2); the generated RDR files are dropped (decided in the source plan,
  2026-10-13).
- One immutable, typed `Terminology` is read from RDF once; the compiler, the loader and the check
  all read it.
- Every construct is classified as materialized, checked, query-time or unsupported; nothing is
  dropped silently, so an unsupported construct raises or appears in the profile report.
- Each realized OWL 2 RL/RDF rule is a class named for what it does, carrying its W3C rule id; the
  loader and the check iterate a registry of them.
- `OwlLoader` becomes Algorithm 1's steps as single-responsibility classes sharing a load state.
- Reuse upstream: `krrood.code_generation`, `krrood.patterns.role`, `krrood.symbol_graph`,
  `krrood.patterns.subclass_safe_generic`. krrood stays self-contained: test ontologies are
  `.owl`/`.ttl` files and mimic classes under `test/krrood_test/dataset/`, written tests first.
- Gates: after `closure-verification`, Ontomatic's closure equals owlrl's on every dataset ontology
  and the rule-coverage test lists no rule as missing without a reason; after
  `owl2bench-experiments`, OWL2Bench (1 university) matches the supplement's `results/REPORT.md`
  (3 667 individuals, 1.42 M assertions).
