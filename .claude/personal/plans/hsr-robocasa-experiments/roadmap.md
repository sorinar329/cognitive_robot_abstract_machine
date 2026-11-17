# HSR RoboCasa experiments

- Start from a clean main. The old branch hsr-robocasa-tasks is a read-only reference and is never merged.
- The experiments live on one fork branch, hsr-robocasa-integration: every task is committed straight onto it, with no pull request per task, and it never goes upstream.
- A change to core code (krrood, semantic_digital_twin, coraplex, giskardpy) is never committed on the experiment branch directly: it is its own item, branch and draft pull request into the fork's main, opened as soon as the branch is ready.
- Never open a pull request on cram2: the user opens the cram2 pull requests themselves.
- A core branch not yet in main is merged into hsr-robocasa-integration so the experiments can run, and dropped once main has it.
- Add a core item with /add-plan-item when a task first needs one; the task depends on it.
- Every pull request follows AGENTS.md and carries its own tests.
