## tracy-montessori-demo (PR sorinar329#11): address the user's 12 review comments

Working on local branch tracy-montessori-review (from sorin/tracy-montessori-demo 487f3fd651),
pushed to tracy-montessori-demo; local tracy-montessori-demo holds fera WiP bae2ecce26, leave it.
Merged sorin/main (e5421d900d): one conflict in coraplex/plans/executables.py resolved by
keeping both main's simulation_time_limit/failures and the PR's simulation_pacer. coraplex
651 passed (7 DAiSy errors: ur_robot_driver not installed locally), sdt multi_sim/mjcf 43
passed. User chose: port to main's grasp API (GraspCandidate), then comments.
Plan: (A) port PickUpAction to GraspCandidate, grasp height in the grasp pose (drops
_shape_around_the_grasp_point); (B) comments: no Callable/Tuple aliases; sdt colors instead
of color_of_hue; answer GRASP_HEIGHT question; triangular prism into sdt geometry; drop
build_offline_world park/open if unneeded, world init in build_scene; explain mjcf condim +
cylinder height (parser doubles sizes, so size[1] is full height - PR change is right) and
multi_sim ContactDimensionality/cylinder converter, with tests; real/rviz need testing
(reply only). No gh/token: replies to be drafted for the user. Verify MuJoCo run headless
(scratchpad tracy_report.py).
Done and pushed b4fcbb209e (FF from 487f3fd651 via merge e5421d900d): GraspCandidate from
above with grasp height in the pose; MontessoriScene + build_scene(world=None); real backend
running_robot()/run(scene, robot); Color.CYAN/YELLOW; Mesh.triangular_prism in sdt geometry
(+3 tests, MESH_FILE_PRECISION); README updated. MuJoCo headless 122 s: cylinder and triangle
through, cube and rect wedged ~7 mm in their holes - identical at 487f3fd651 (pre-existing).
RViz backend runs (8 s). condim/cylinder tests already on main. Replies drafted in chat
(not posted: no gh/token); PR description not updated; PR stays draft.
2026-10-19: PR is upstream cram2#738 (head sorin/tracy-montessori-demo 5eb228fde4). 14 review
comments (Bass 5, Naren 9, Naren CHANGES_REQUESTED) addressed, uncommitted, on a fresh local
tracy-montessori-review (from sorin/tracy-montessori-demo, merged synced sorin/main b12f67e3a4,
clean). montessori_board.py: no module constants (BoardResource path enum, colours/sizes as
ShapeSortingBoard fields, thresholds in HoleShapeClassifier), no tuples (LidPoint, LidArea,
Compartment, HoleOutline), free functions -> methods, trimesh extents for _extent; board
identical (6 holes, 81 boxes). demo.py split: MontessoriPieceSet, SolidContact, PieceGrip,
MontessoriScene, ShapeSortingPlan, DemoRunner(ABC); runners MujocoRunner/RvizRunner/RealRunner
each own entry point, imports on top, no dispatch/exception. Merge regression found: PR #734
aims grasps at the closed meeting point, so pads pressed 9 mm into the table (triangle
wedged): PieceGrip.grasp_height = clearance + travel_when_closing + meeting point offset
(28.2 mm) read off the gripper; test test/coraplex_test/test_tracy_montessori_demo.py (failed
first). MuJoCo headless: cylinder+triangle through, cube+rect wedge (= pre-merge); kinematic all
4 at release poses. Collision avoidance: stalls at first reach - parked upper arms already at the
3 cm camera_pole buffer (not row-dependent; fix belongs in Tracy's collision rules). README axis
fixed (apex +x, released unturned), 4/4 claim corrected. User chose rename MontessoriPiece ->
ShapeSortingPiece (sdt, tests, demo, README). Tests 80 + sdt annotations/orm 244 passed.
Replies drafted in chat (no gh). Next: user reviews; commit/push when asked; PR stays draft.

