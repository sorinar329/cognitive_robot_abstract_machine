Plan
- Review of cram2#962 (draft); answer review comments when asked

Done
- Bootstrap: worktree gcs_wt from main be591627e7, draft PR cram2#962 (new-feature, api-break), item repository set to cram2
- 46b6d1c760: is_free / nearest_free_point (+ NoFreeSpaceError), excluded_bodies through the obstacle collection and both graphs, obstacle_height_clearance on the volumetric graph, Floor.navigable_space (MissingMobileBaseError without a base); PathPlanningNavigateAction uses navigable_space(margin=0.0)
- Changed methods keyword-only, every caller updated; module docstrings on touched modules; krrood format + check clean
- Tests: GCS / annotation / transformation tests 102 passed 2 skipped; path-planning designator tests 44 passed 18 skipped

Next
- Mark ready only when the user says so
- navigation-keeps-base-margin will touch navigation.py and navigable_space: build it on this branch or after it lands
