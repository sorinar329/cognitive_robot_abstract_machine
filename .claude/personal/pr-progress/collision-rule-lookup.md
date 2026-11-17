Plan
- Port 2be976e0 onto main: CollisionManager.external_collision_rule, AbstractRobot.is_within_a_buffer_zone, NoExternalCollisionRuleError.

Done
- Tests first, then the implementation; test_collision_checking 102 passed, test_robots 256 passed (ORM interface and DAiSy's ur_robot_driver missing locally); krrood format/check clean.
- Draft PR sorinar329#41 into the fork's main (head c4eef85dcf).

Next
- User reviews #41 and opens the cram2 PR.
- Then path-navigation-margin (828cafde) on top: navigation uses the base rule's violated_distance as margin and starts from nearest_free_point.
