## Plan
- Margin parameter on PathPlanningNavigateAction (default 0.0) passed to Floor.navigable_space.
- Path starts at nearest_free_point of the robot's position.
- Settled base: floor check uses the floor's bottom, not its top.
- Tests first in test_multi_robot_action_designator.py (old commit 828cafde as reference).

## Done
- Branch from main a3abd8dcf2, bootstrap commit, draft PR sorinar329#42, plan item recorded.

## Next
- Write the failing tests, then implement the three parts.
