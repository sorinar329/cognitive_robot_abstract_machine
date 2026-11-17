Plan
- Draft PR sorinar329#40 into the fork's main; the user opened cram2#989 from the same branch (2026-11-17)

Done
- 46b6d1c760: is_free / nearest_free_point (+ NoFreeSpaceError), excluded_bodies through the obstacle collection and both graphs, Floor.navigable_space; keyword-only signatures, every caller updated
- ea5fc02c57: merged sorinar329 main; main had added obstacle_height_clearance to the volumetric graph, kept with excluded_bodies beside it
- Tests after the merge: GCS + annotation tests 68 passed 1 skipped; plan-transformation + path-planning tests 45 passed 18 skipped

Next
- Review on the fork; mark ready only when the user says so
- Merge into hsr-robocasa-integration when the first experiment needs it
