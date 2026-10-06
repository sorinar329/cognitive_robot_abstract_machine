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
Next: user decides: push the merge; bring the tick period over?

## containment-detector-fix: Segmind's ContainmentDetector slows the bullet world demo

Branch from sorin/main (0c5a90f8c1, after PR #670 segmind-live-1 merge), 2026-10-06.
Very high priority todo. Claim to verify first (pasted by user): bullet world demo
309 s with Segmind vs 195 s without (+115 s, ~60%); a tick ~0.9 s, ~97% in
ContainmentDetector.get_containment_pairs -> InsideOf.compute_containment_ratio 432x per
tick (3 watched objects x 144 bodies outside the hands).

Measured 2026-10-06 (scratchpad segmind_cost.py, quiet machine, 2 runs each):
off 73.4/72.9 s, on 100.0/100.1 s -> +27 s (+37%), not +115 s; absolute times far below
the claim's 195/309 s. Instrumented: 48 ticks, mean 0.98 s (0.54-4.1); containment 80% of
tick time (claim 97%); 3 checks/tick x 1 object x 145-146 candidates = ~435 ratio calls
per tick (claim 432 confirmed), 1.8 ms each; 43 of the candidates are PR2 bodies.
Step 1 done (uncommitted): ContainmentDetector honours exclude_robot (bodies_left_out);
test_robot_bodies.py: test_an_object_is_not_contained_in_the_robot (failed first: base_link,
torso_lift_link) + companion with exclude_robot=False. segmind suite 98 passed.
Effect: ratio calls/check 145 -> 102, tick 0.98 -> 0.74 s, but ticks 48 -> 61 and demo
still 100 s: the tick loop pauses as long as it ticked, so Segmind holds the world ~half
the time whatever a tick costs. Post-fix runs (2/2) show an extra milk PlacingEvent on
island_countertop before the grasp; 0/3 pre-fix runs (event counts vary 38-49 anyway),
suspected tick-timing effect, not proven.
PlacingEvent investigated: (a) World.move_branch for a Connection6DoF (coraplex attach,
executables.py:454) adds the new connection with identity offset inside modify_world, and
sets the correct origin only after the block released the world lock -> a Segmind tick in
that gap sees the milk at 2x its position (trace: 4.74,4.00,2.07 vs 2.37,2.00,1.03) ->
LossOfSupport+Translation, sometimes re-support next tick. (b) PlacingDetector pairs a
LossOfGrasp with any SupportEvent within +-15 s, so a support from before the grasp
becomes a placing. Shorter ticks only made (a) likelier. Proposed: separate branch for (a)
(test: a model-change callback reads the moved body's pose mid-move); (b) belongs to the
detector refactor. User: not fixed here; both added as tasks in todo tab 19 (with notes).
Robot exclusion committed (037efdd944). User's refactor idea (2026-10-06, photo): run all
detectors once at start, then Translation -> Contact -> Support -> Containment cascade,
containment only on a new support, against container annotations or the supporter's parent
chain. My review: good, but needs end-of-containment on movement, misses container-moves /
no-new-support cases, no self-correction, no single Container annotation in sdt
(IsStorageSpace, HasCaseAsRootBody, DrinkingContainer, ...), placed objects reattach to
world root (placing.py:84), duty cycle unchanged. Agreed order: bounds filter -> cascade ->
tick pacing.
Step 2 done (uncommitted): exact bounding-box filter. Bounds.overlaps (geometry.py),
ShapeCollection.bounds_in_root_frame (FK np + shape mesh-bounds corners, superset of what
InsideOf measures), ContainmentDetector skips non-overlapping candidates and objects without
collision. Tests: 3 Bounds, 2 bounds_in_root_frame, 2 detector (measured == {environment};
no-collision object). geometry + segmind suites 208 passed. Existing as_bounding_box_collection
helper was 1.1 ms/body (as slow as a ratio call), hence the numpy path.
Demo (instrumented, 1 run): ratio calls/check 101.8 -> 0.7, tick 0.74 -> 0.17 s, ticks
61 -> 247, demo 97 -> 91 s; containment still 49% of tick (~26 ms/check, mostly building
~100 boxes; not profiled). docformatter missing from venv; ran black only.
User: tick pacing before the cascade.
Step 3 done (uncommitted): Segmind.tick_period (default 0.5 s, my pick - flagged to user);
loop waits max(tick_period - held_for, pause_between_ticks). Tests:
test_detectors_are_ticked_once_a_tick_period (failed first: 48 ticks in 0.5 s at 0.1 s
period), test_changing_the_world_is_not_held_up_by_ticks_longer_than_the_tick_period.
segmind suite 102 passed. Demo (uninstrumented, 1 run each, with bounds filter):
off 73.3 s; period 0.25: 89.0 s/238 ticks/held 50%; 0.5: 81.2 s/120/38% (implemented:
81.7 s/125/38%); 1.0: 78.2 s/65/26%. Spurious island_countertop placings (move_branch gap)
at 0.25/0.5, not at 1.0 (fewer ticks in the gap, luck). Script: scratchpad pacing_cost.py.
No-lock experiment (tick without outer lock, 0.5 s): 83.5 s, no errors, but ~60 bogus
pick-ups/placings (torn reads) -> lock stays; the lock is not the cost.
Breakdown (scratchpad tick_breakdown.py, 0.5 s, demo 81.7 s vs off 73.3 s = +8.4 s):
Segmind thread CPU 8.1 s (wall 30.9 s, rest waiting for the interpreter lock; lock wait
0.4 s) -> demo cost ~= Segmind CPU. CPU: containment 6.7 (bounds_in_root_frame 5.6 over
38844 calls, ratio 1.0), contact 0.5, support 0.5, motion 0.3. cProfile is useless here
(3.12 profiles all threads). Proposed order: cache each body's local corners (bounds) ->
cascade -> period; separate process later.
Step 4 done (uncommitted, user: "start with 1"): bounds_in_root_frame now uses the
already-cached ShapeCollection.combined_mesh bounds (corners -> FK) instead of rebuilding
each shape's mesh (Box.mesh built a trimesh per call). No new test (no behaviour change;
existing bounds tests cover). 210 passed (geometry + segmind). Demo: Segmind CPU 8.1 ->
3.6 s, bounds 5.6 -> 0.9 s; demo 77.7 / 77.2 s (off 73.3). Remaining CPU: containment
2.2 (ratio 1.1, bounds 0.9), contact 0.5, support 0.5, motion 0.3.
Next: user picks default period; commit when asked; then the cascade.
Observations: get_containment_pairs uses bodies_outside_end_effectors directly, so it
skips bodies_left_out (robot bodies) unlike get_relation; each ratio call copies and
transforms both meshes and builds the container's bounding box; Segmind's tick loop
waits as long as the tick held the world lock, so tick cost roughly doubles in plan time.
