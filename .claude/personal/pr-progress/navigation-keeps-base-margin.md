## Plan
- obstacle_clearance and floor_top_tolerance on PathPlanningNavigateAction; path starts at the nearest free point.

## Done
- Implemented test-first at 0748bc7dd7 (merged main 2138cbe5f6); designator 300 passed, plan/JSON/mixin tests 239 passed, krrood check clean.
- Description updated; draft PR sorinar329#42.

## Next
- User opens the cram2 PR (labels new-feature, bug-fix).
- Confirm 1 cm floor_top_tolerance covers how far the HSR settles under MuJoCo.
