Plan
- Tests first in test_gcs.py: is_free (free, occupied, outside the search space, any frame), nearest_free_point (itself if free; otherwise free and nearest; error on an empty graph), excluded_bodies frees a body's space, the HSR's navigable_space keeps its margin
- GraphOfBoundingBoxes.is_free / nearest_free_point (keyword-only), NoFreeSpaceError in graph_of_convex_sets/exceptions.py
- excluded_bodies through obstacle_entities, build_bloated_obstacle_collection, both graphs' free_space_from_semantic_annotation, volumetric free_space_from_world (+ obstacle_height_clearance); changed methods keyword-only, every caller updated
- Floor.navigable_space(*, robot, margin)
- krrood check + format, run test_gcs*, test_semantic_annotations, experiments caller

Done
- Worktree gcs_wt from main be591627e7; bootstrap commit pushed to sorin; draft PR cram2#962 (new-feature, api-break); item repository set to cram2

Next
- Write the failing tests
