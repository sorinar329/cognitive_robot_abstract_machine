# Code Quality Rules

These rules are carried over from the `cognitive_robot_abstract_machine` workspace so that code
moves between the two repositories without a style change.

## Avoid Behaviour
- Avoid using global variables
- Avoid using mutable objects as default arguments
- If you are unsure why something was done or why specific numbers were chosen, ask the developer
  instead of inventing the reason and writing it as a comment

## Testing
- Run tests with pytest
- Reuse existing fixtures found in `conftest.py`
- Use a test-driven approach: prove a bug with a failing test before fixing it
- When fixing failing tests, never modify the test itself
- All new features and fixes must be covered by tests
- Name test classes (and the stand-in classes used by tests) after the behaviour they exercise, not
  after the concrete external class they happen to replace
- Make assertions as specific as possible: assert equality to the correct expected value rather
  than only a weaker check such as not-None or not-empty
- Assert against the definition rather than a copy of it: compare to the enum member, the named
  constant, or the value read from the fixture the code under test consumed
- Keep each test focused on the one behaviour it names
- Tests must run in CI without model weights, network access or credentials. The
  `segment_and_classify`, `yolo` and `blender` extras are exercised through stub implementations of
  the corresponding interfaces

## Code Style
- Divide a file into logical sections with `# %% <short description>` comment headers
- Create classes instead of using too many primitives. If a return type is always repeated,
  consider whether a dedicated class or type alias would convey more meaningful information
- Minimize duplication of code. Avoid catch-all files like `utils.py`: prefer moving behaviour onto
  the class that owns it
- Comments must be meaningful and adhere to DRY; remove redundant or restating comments
- Do not wrap attribute access in try-except blocks
- Always access attributes via ".", never via `getattr`
- Use existing packages whenever possible
- Always use dataclasses

### Naming
- Names must be technically correct, simple and descriptive, in that order
- Minimize jargon; prefer the plain word every reader already knows
- Do not use abbreviations in identifiers
- Methods are verb phrases for what they do; classes and attributes are noun phrases for what they are
- One operation, one name, throughout a module
- Do not repeat the enclosing type's name in its members
- Never take an identifier the language or something already in scope binds

## Imports
- Imports are absolute and global (top of module)
- Within tests, importing another test module must use a relative import
- Use stdlib type hints where possible, and `typing_extensions` for the rest
- Use `from __future__ import annotations` instead of quoting types
- Use a `TYPE_CHECKING` guard for type-only imports
- Modules that import an optional heavy dependency (`torch`, `ultralytics`, `bpy`) must not be
  imported from a package `__init__.py`, so the package stays importable without those extras

## Design Principles
- Focus on strictly object oriented design and apply the SOLID principles
- Code should be modular and decoupled
- Create meaningful custom exceptions
- Eliminate YAGNI smells
- Make interfaces hard to misuse
- Reduce nesting with guard clauses; the main branch holds the main output
- Do not use try-except blocks; programs in illegal states raise appropriate exceptions
- Prefer structured data over bare strings, hardcoded values and meaningless numbers
  - Never hardcode a string that names a fixed thing; give it a `StrEnum` member
  - Replace a magic number with a named constant or an enum member
  - Mirror payloads whose shape someone else controls in dataclasses with a `from_json`/`from_yaml`
    classmethod, so the access path into the payload is written once
  - Replace a tuple whose positions carry meaning with a dataclass
  - Keep a long literal document in a file of its own type and read it in

## Type Hints
- Classes and methods always have accurate type hints (including `Any`) where applicable

## Documentation
- Classes and methods always have meaningful, non-trivial documentation
- Every field is documented with its own docstring placed directly below the field
- Write docstrings in ReStructuredText, short and to the point: what the code does, not how
- Use Sphinx directives (`..note::`, `..warning::`, `:func:`) where appropriate
- Do not create type information for docstrings
- Do not name a function's current callers in its own docstring
- Do not use all-caps for emphasis; use RST emphasis (`*word*`)
- Format modified files with `black` and `docformatter` (see `.pre-commit-config.yaml`)

## Version Control
- Commits are authored in the name of the human running the tool, using their own configured git
  `user.name` and `user.email`. Never author or amend a commit as an assistant identity, and never
  add a `Co-Authored-By:` trailer for an assistant
- Acknowledging assistant help in the commit body with a plain line such as
  `Made with the help of Claude` is encouraged
