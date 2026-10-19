## containment-quick-fix: minimal containment fix (support before containment, robot excluded)

Branch from sorin/main (0c5a90f8c1), 2026-10-06. containment-detector-fix (filter, bounds
cache, tick period) committed by the user as 891bcc91cd and pushed; this branch is the
minimal alternative.
Done (uncommitted except the cherry-pick): robot exclusion cherry-picked (037efdd944 ->
33988b8743). ContainmentDetector requires SupportDetector; searches all candidates only for
bodies resting on something new since the previous tick (bodies_come_to_rest), and re-checks
only the known containers of bodies already inside something (containers_still_holding).
New tests: test_containment_on_coming_to_rest.py (tray: set down -> contained; held up inside
-> not, failed first; lifted out -> loss) + test_containment_brings_the_supports_it_is_read_from
(failed first). 96 passed, 6 old tests fail because they float the milk inside a solid box
without support: test_insertions_bring_contact_and_containment, test_the_containment_detector_
reports_gaining_and_losing_a_containment, test_an_object_is_contained_in_the_robot_when_it_is_
not_left_out, test_containment_detector, test_insertion, test_a_containment_that_lasts_is_not_
reported_lost_by_another_bodys_detector.
User: rewrite the 6 to the new rule. Shared scene module test/segmind_test/trays.py
(world_with_a_box, add_tray, set_down_in, hold_up_in, lift_out_of); robot tests set the box
down on a shelf inside base_link; insertion uses a tray named tray_hole. segmind suite 102
passed. (Careful: black on the whole test dir reformats unrelated files - format only
touched files.) Demo (old pacing, 1 run): 87.7 s vs main ~100 s, off ~73 s; 761 ticks;
events: 3 pick-ups (+1 extra bowl), placings on table_area_main (+ bowl island_countertop
move_branch artefact), spoon in/out of cabinet10 drawer.
User pushed it as 835632224c + e02b95ded4 (trays.py and new test file - CI segmind
failure on PR #12 was these two files missing) and opened PR sorinar329#12 (not draft).
Merged sorin/main (154 commits, no conflicts, local only, not pushed); ORM regenerated;
segmind 102 passed; test_demo.py exit 0. Merged main itself: demo off 109 s, Segmind on
172 s (plain main) vs 169 s (quick fix); milk/spoon get no support/placing on the table on
plain main too (pre-existing). Breakdown on merged branch: 2678 ticks (old loop), Segmind
CPU 52 s: containment rechecks 16.9 (spoon's 2 containers every tick, 3.5 ms each), contact
15.5, support 12.5, motion 5.4 -> per-tick fixed costs x tick count; tick period (on
containment-detector-fix 891bcc91cd) would cap ticks.
Pushed the merge (6b227281d4) and the tick period (58e72da989, applied from 891bcc91cd:
event_segmentation.py + its 2 tests; failed first with TypeError). segmind 104 passed.
Demo: 114.7 s (off 109, before 169), 210 ticks, held 12%, events as plain main.
PR #12 is not draft and gh is not installed: user to set draft / update description.
CI timing (user asked): merged origin/main -> 95f341eb33 (with changes), pushed revert
ec959185db (segmind == upstream main), restored 6427377059 (tree == 95f341eb33).
CI (1 run each, all green): pipeline wall 18.1 vs 17.7 min; bullet demo job 10.9 vs 10.6,
its Run Script step 6.3 vs 6.2 min; segmind Run tests 0.9 min (reverted steps hit the API
rate limit). No measurable CI difference although local demo is 115 vs 172 s; Segmind does
run in CI (no gating). CI logs need admin/token. Scripts: scratchpad ci_timings.py,
ci_steps.py. Job times vary a lot between runs (experiments 14.3 vs 9.1 min, no change).
Segmind-off CI run (temp 9fdff586af, undone df41dee189): demo Run Script 4.2 min.
Local CI-like (4 cores, QT offscreen, LIBGL software): off 109.4 s, reverted 165.7 s
(Segmind CPU 61.7), with 113.8 s (CPU 5.0). Overlay has no segmind (ruled out).
Instrumented CI run (temp 451b0092b0, undone 2ab9112d76; phases via ::notice annotation):
construct 0.78 s, first tick 1.67 s, 449 ticks 53 s wall / 21.5 s CPU, demo 261 s; Run
Script 4.5 min -> with changes ~= off + 0.3 min; the earlier 6.3 min was runner noise.
Final CI run with changes (2ab9112d76, all 24 green): pipeline wall 20.6 min, bullet demo
Run Script 3.3 min, segmind Run tests 0.8 min. Demo step with changes across 3 runs: 6.3,
4.5, 3.3 min; Segmind off 4.2; reverted 6.2 (one run). Runner noise ~ +-1.5 min.
PR showed 25 files: base was stale fork main 04631c7bbc while branch had upstream merged;
merged synced fork main f38d633f59 (PR #706) -> f05150884c pushed, PR now 10 files.
Tests aligned with existing ones (user, uncommitted): trays.py and
test_containment_on_coming_to_rest.py removed; fixture box_and_trays (world, box, tray,
tray_hole) + position constants (SET_DOWN_IN_THE_TRAY, SET_DOWN_IN_THE_HOLE,
HELD_UP_IN_THE_TRAY, LIFTED_OUT_OF_THE_TRAY, SET_ASIDE) in test/segmind_test/conftest.py;
tests use each file's _place / direct origin like their neighbours; held-up test moved to
test_segmind_detectors.py (fails on main's detector); robot tests keep shelf inside the
original _box_inside (box sunk into base_link never touches its mesh). 102 passed.
Committed + pushed 8cba3a6521; PR #12 shows 9 files.
CI on 8cba3a6521: 24/24 green, wall 16.5 min, demo Run Script 3.9 min, segmind Run tests 0.8 min. Demo step with changes so far: 6.3, 4.5, 3.3, 3.9 (mean 4.5); off 4.2; reverted 6.2.
Merged upstream origin/main (57 new, fork main not synced) -> 096897d472, local only. One
conflict in test_several_watched_bodies.py (upstream: conversions became properties, e.g.
global_pose.position) - kept ours. segmind 102 passed; bullet demo exit 0 in 129 s with 3
pick-ups, 3 placings on table_area_main (height fix now upstream), spoon in/out of drawer;
place-setting test passes. User synced fork main; pushed 096897d472, PR #12 shows 9 files (not draft; gh missing).
2026-10-08: merged fork main (73 new, none touch segmind) -> 167bf8779b, local only.
segmind 102 passed; bullet demo exit 0, 3 pick-ups, 3 placings, spoon in/out. PR has no
description and no labels. Review given in chat (side-effecting bodies_come_to_rest; silent
no-containment without SupportDetector; candidates built every tick; tick_period default
open; motion-window timing change; trade-offs to state in description).
Pushed merge 167bf8779b and fixes for review points 1-3, 8 (split bodies_come_to_rest /
remember_supporters, early return, docstring, 2 tests): segmind 104 passed, demo unchanged.
PR description drafted in chat for the user (gh missing).
Next: period default; PR description (tick period, CI numbers).
2026-10-09: upstream PR is cram2#717; Naren (CHANGES_REQUESTED) 3 comments addressed, uncommitted for
user review: (1) overrunning tick now pauses as long as it held: wait max(period-held, held, pause)
+ test test_a_tick_as_long_as_its_period_is_followed_by_a_pause_as_long_as_it_took (failed first,
4 > 3.5 ticks); (2)+(3) tray constants -> conftest BoxAndTrays dataclass (fields world/box/tray/
tray_hole, poses as properties), tests use attributes, no tuple unpacking. segmind 105 passed.
Not done: main's RESTING_ON_THE_TABLE/WHERE_THE_MILK_STOOD, SHELF_*/TICK_PERIOD module constants;
bullet demo not re-run; replies to Naren not drafted.
2026-10-19: #717 approved by LucaKro and Naren on 1e16e4d96d, CI 24/24 green, but 354 behind
upstream main. Merge of upstream main b29c37fbfd tested locally (segmind + place-setting 106
passed; bullet demo exit 0 in 114 s, 3 pick-ups, 3 placings, spoon in/out) and pushed as
4b8e51360d (user asked). Approvals stayed; CI re-running. User adds labels and merges.

