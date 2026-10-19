# Segmind judges the post-conditions of coraplex actions

## Why
Every coraplex action declares an EQL `post_condition` (pick_up.py, placing.py,
container.py, navigation.py, robot_body.py) and `ActionDescription.action_plan` wraps it in
a `ConditionNode`, but nothing evaluates them: `ActionDescription.perform` checks only the
pre-condition, and `GiskardExecutable._add_condition_monitors` is documented as currently
unused. Success is judged afterwards, by hand, in each experiment (the HSR RoboCasa tests
read Segmind's events after the run). The goal is a standalone push, independent of
hsr-robocasa: Segmind sees every action and judges its post-condition from events.

## Decisions (user, 2026-10-19)
- Segmind sees the actions, and each action's post-condition is expressed as Segmind events
  (PickUpAction: the picked object loses its support, and so on). This makes segmind depend
  on coraplex (no cycle: coraplex does not import segmind).
- A post-condition that does not hold is recorded as a verdict, not raised; the plan keeps
  running.
- Segmind first, world fallback: actions Segmind has no expectation for (navigation, torso,
  park arms) are judged by evaluating their existing EQL post_condition on the world.

## Design
- `ActionExpectation(Generic[ActionT], SubClassSafeGeneric)`, one subclass per action type
  bound by the generic parameter; `required_detectors`, `watched_bodies(action)`,
  `judge(action, events_in_window)`. Looked up by the action's MRO; new actions add a
  subclass (open/closed).
- Expectations: PickUp = LossOfSupport(object) + Grasp(object, arm's tool frame) with no
  LossOfGrasp after; Place = LossOfGrasp(object) + Support(object) by a non-robot body;
  Open = OpenedEvent(moving part); Close = ClosedEvent(moving part), ratio from the action's
  closed_position/closing_tolerance; Grasping = Grasp(object, tool frame). Composite actions
  have none; their children are judged.
- `PostConditionVerdict` (action_node, holds, judged_by: JudgedBy SEGMIND/WORLD/NOT_WATCHED,
  evidence, action_failed).
- `PostConditionJudge(PlanCallback)`: on_end queues a pending judgement for
  [start_time, end_time] without blocking; the Segmind thread resolves it after
  ticks_to_see_the_world_at_rest ticks past the end; stop() resolves the rest. Unwatched
  objects are judged by the world and marked NOT_WATCHED (detectors compile once per run).
- `Segmind.create_for_plan(world, bodies, expectations=all)` ticks only the detectors the
  expectations need; usage `with Segmind.create_for_plan(...) as segmind,
  segmind.judging(plan): plan.perform()`, then `segmind.verdicts`.
- Robot-agnostic already: GraspDetector finds hands via EndEffector annotations and
  HasTwoFingers (PR2, Tiago, HSR all have them).

## Open points to flag in the PRs
- An event of the next action can fall into the previous one's settle window; the chosen
  evidence makes it harmless (a place produces no LossOfSupport), but it is stated.
- PlacingDetector pairs LossOfGrasp with a support up to +-15 s away, so Place reads the
  atomic events, not PlacingEvent.
- OpeningDetector's default ratios (0.9 / 0.05) differ from the geometric open condition
  (position > 0.3); expectations take thresholds from the action.

## Verification
pytest test/segmind_test and test/coraplex_test/test_designator with --orm-build never, one
at a time under a cgroup memory cap; the bullet world demo prints a SEGMIND verdict that
holds for pick, place and drawer open/close.
