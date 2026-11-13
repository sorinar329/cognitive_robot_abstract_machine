# HSR RoboCasa experiments

- Start from a clean main. The old branch hsr-robocasa-tasks is a read-only reference and is never merged.
- One experiment per pull request, into the fork's hsr-robocasa-integration branch; experiments never go upstream.
- A change to core code (krrood, semantic_digital_twin, coraplex, giskardpy) is never part of an experiment
  pull request: it is its own item and its own pull request from the fork to upstream main.
- Add a core item with /add-plan-item when a task first needs one; an experiment depends on the core
  items it needs and is stacked on their branches until upstream merges them.
- Every pull request follows AGENTS.md and carries its own tests.
