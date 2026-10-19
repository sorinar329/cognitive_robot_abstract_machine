## segmind-post-conditions: Segmind judges the post-conditions of a plan's actions (plan segmind-post-conditions)

Branch from sorin/main b12f67e3a4 (2026-10-19), pushed 120cced7c7. No PR yet: the user opens it
against the fork's main to review first. STALLED by the user: the tests are to move onto a physical
simulation (Segmind is meant for physics; kinematic grasps close fully regardless of the object),
on top of the Tracy Montessori demo (upstream PR cram2#738) once it is on main.
Done: segmind/post_conditions (ActionExpectation[Action] for PickUp, Grasping, Place;
PostConditionJudge(PlanCallback) records verdicts SEGMIND/WORLD/NOT_WATCHED, never raises;
waits after each action until Segmind saw the world at rest - needed, else the next action's
events (move_branch grasp flicker) fell into this one's window: 1/6 Tiago pick-ups failed, now
45/45). Segmind.ticks + seen_at_rest_after; SegmindIsNotWatching; segmind -> coraplex
dependency; new classes ignored in segmind's ORM. 6 tests x PR2/Tiago/HSR kinematic +2 core
tests; segmind suite 115 passed.
Findings: ActionNodes never start/end themselves (actions merge into one motion chart), so
actions are framed by their steps (leaf nodes). OpeningDetector/OpenedEvent/ClosedEvent are
only on hsr-robocasa (e93a12b68e, 941fcfde61): Open/Close expectations need them on main first
(drawer tests kept in scratchpad test_post_conditions_with_drawers.py). Apartment fixture's milk
floats 1.5 cm above island_countertop. move_branch torn read makes Segmind see grasp flicker
(separate bug). ORM regen fails at experiments (control_msgs missing locally).
Next: rebase tests onto physics with Tracy (PR 738), then PR2; Tiago/HSR need servos on main.
