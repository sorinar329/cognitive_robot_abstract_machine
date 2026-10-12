## tracy-montessori-demo (fork PR #11 -> sorinar329:main): Tracy Montessori demos

The scene-graph/simulation-control work this note used to describe is committed on
`origin/fera_exchange` (bae2ecce26), not on this branch.

2026-10-12: local branch fast-forwarded to origin (b4fcbb209e), then cram2/main merged in
(239 commits); pushed as 7dffe5ca8c (merge) + 4bf737878a (demo migration).
- executables.py conflict: took main's side. Main's PhysicallySimulatedRobot +
  GiskardExecutable.simulation replace this branch's simulation_pacer, and main skips
  MoveBranch under PHYSICALLY_SIMULATED, replacing Context.update_world_model_attachment
  (field dropped, dataclasses.py = main).
- Demos migrated: mujoco_demo uses PhysicallySimulatedRobot; rviz_demo uses
  kinematically_simulated_robot; demo.py `.to_quaternion()` -> `.quaternion` (main
  2a7c5c4178); README "Two changes outside this folder" rewritten as "Running under physics".
- Verified: SDT multi_sim/shape, physics_simulators, coraplex executables + motion
  designator: 137 passed, 33 skipped. Headless MuJoCo demo end to end: cube, cylinder,
  triangular prism in holes; rectangular prism on the lid (z 0.969).

Next: user reviews PR #11; it is still a draft.
