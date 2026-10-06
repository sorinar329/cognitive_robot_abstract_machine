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
Next: commit when asked; then cascade, tick pacing.
Observations: get_containment_pairs uses bodies_outside_end_effectors directly, so it
skips bodies_left_out (robot bodies) unlike get_relation; each ratio call copies and
transforms both meshes and builds the container's bounding box; Segmind's tick loop
waits as long as the tick held the world lock, so tick cost roughly doubles in plan time.
