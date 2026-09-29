# Tracy montessori demo

Tracy picks the four Montessori pieces (cube, cylinder, rectangular prism, triangular
prism) off its own table and drops each through its hole in the shape-sorting board. The scene, the poses and the plan are the same wherever it runs;
only how the plan is carried out differs. Three backends: MuJoCo physics, RViz, and the
real robot.

The board is the real board's mesh (`resources/board.stl`, taken from the ICRA
branch's `experiments/montessori/resources/`). Its six holes are read off that mesh by
slicing through the lid (`montessori_board.py`), so every piece is released straight
above the hole it actually fits. The pieces use the sizes and colours measured off the
real set. There is no perception: where the board and the pieces start is hardcoded.

Tracy starts with both arms parked (the `PARK` joint states of `TracyLeftArm` and
`TracyRightArm`) and both grippers open. The plan still begins with `ParkArmsAction`,
because the real robot starts wherever it was left.

Branch: `tracy-montessori-demo`, off `cram2/main` at `746302cd9b`.

## Running it

Pick the backend by editing `BACKEND` near the top of `demo.py`:

```python
BACKEND = Backend.RVIZ      # or Backend.MUJOCO, or Backend.REAL
```

Then, **from inside this folder**:

```bash
python demo.py
```

It has to be run from this folder. `coraplex/demos/` is not a package — `coraplex`'s
`pyproject.toml` ships only `src/coraplex*` — so the demo relies on its own directory
being on `sys.path` to find its sibling modules. This is the same mechanism the other
demo folders' `test_demo.py` files rely on for their bare `import demo`.

Backends are imported only once chosen, so you do not need MuJoCo installed to run the
RViz backend, or ROS to run the MuJoCo one.

| Backend | What you need | What you see |
|---|---|---|
| `RVIZ` | ROS 2, RViz2 | A `MarkerArray` display on `/semworld/viz_marker`, durability set to transient local. The robot is moved kinematically. |
| `MUJOCO` | `mujoco` | A MuJoCo viewer. Set `HEADLESS = True` in `mujoco_demo.py` to run without one. The viewer stays open after the plan ends until you close it. |
| `REAL` | The robot, its ROS stack | It launches `giskardpy_tracy_standalone.launch.py` itself and fetches the world from the running stack. |

## File layout

| File | Role |
|---|---|
| `demo.py` | The entry point. Holds `BACKEND`, the pieces, every hardcoded pose, `build_scene()`, `build_plan()`, and the dispatch. |
| `montessori_board.py` | The board: its holes read off the mesh, and a body with the mesh to look at and boxes to collide with. |
| `resources/board.stl` | The board's mesh. |
| `rviz_demo.py` | Kinematic run, published for RViz. |
| `mujoco_demo.py` | Physics run, controller and physics stepped in lockstep. |
| `real_demo.py` | Real robot, through giskard. |

The backends do not import `demo`; `demo` passes its builder callables down to them. That
is what keeps the import from being circular, and it is why each backend's `run()` takes
functions rather than reaching for the scene itself. `demo.py` declares the three callable
types (`BuildsWorld`, `BuildsScene`, `BuildsPlan`) and the backends import them under a
`TYPE_CHECKING` guard, which costs nothing at runtime.

## What the plan does

```
ParkArmsAction(BOTH)
for each piece:
    PickUpAction(piece, LEFT, grasp from the front, aligned to the top)
    PlaceAction(piece, 1 cm above its hole, turned to match it, LEFT)
ParkArmsAction(BOTH)
```

`PICK_ARM = Arms.LEFT`, which is the arm every Tracy demo in this repository uses.

Note this does **not** use `TransportAction` the way the bullet world demo does. Tracy is
fixed-base and has no torso, so the bullet demo's three staples are all unusable here:
`NavigateAction` (and therefore `TransportAction`, which navigates) requires
`context.robot.drive is not None`, and `MoveTorsoAction` calls `robot.get_torso()`. Plain
`PickUpAction` + `PlaceAction` is what the other Tracy code in the repo does too.

## Where the numbers come from

`TABLE_TOP_Z = 0.88` is measured, not guessed. Tracy's kinematic root *is* its table, and
that body carries two collision boxes:

- a `Scale(1.18, 1.6, 0.855)` slab centred at local `(0.59, 0, -0.4275)`, whose top lands
  at exactly `z = 0.88` in world coordinates — this is the work surface;
- a small `0.2 × 0.2 × 0.25` mount at local `(0.13, 0, 0.125)`, top at `z = 1.13`.

The usable surface runs `x` from 0 to 1.18 and `y` from -0.8 to 0.8, with that mount
occupying roughly `x` 0.03 to 0.23. The layout follows the ICRA branch's
`TracyMontessoriWorld`:

| | x | y | z |
|---|---|---|---|
| pieces (start) | 0.55 | 0.1 / 0.2 / 0.3 / 0.4 | on the table |
| board (centre) | 0.85 | 0.0 | on the table, 8 cm tall, drawers facing the robot |
| release poses | over each hole | | 1 cm above the lid |

Holes, as read off the mesh, in the board's frame:

| hole | centre (x, y) | size | piece |
|---|---|---|---|
| circle | -0.0215, -0.087 | Ø 32 mm | cylinder, Ø 28 mm |
| square | 0.0205, -0.087 | 32 × 32 mm | cube, 30 mm |
| triangle | 0.0125, 0.0 | side ≈ 42 mm, apex +x | triangular prism, side 37 mm |
| rectangle | -0.0265, 0.0 | 22 × 42 mm | rectangular prism, 20 × 40 mm |
| circle | -0.0175, 0.091 | Ø 40 mm | none |
| slot | 0.025, 0.091 | 5 × 48 mm | none (no disk in the set) |

The triangular prism is modelled with its apex along its own +y, so the fingers close on
one flat face and the opposite edge, and it is turned -90° when released. MuJoCo collides
with the convex hull of a mesh, which would close every hole, so the board's collision
geometry is boxes: the walls and the base are boxes in the mesh already, and the lid is
tiled around the holes' bounding boxes. The drawer handles are only drawn.

Worth knowing: `coraplex/demos/coraplex_real_tracy/demo.py` and
`test/coraplex_test/test_designator/test_multi_stationary_robot_action_designator.py` put
their boxes at `z = 0.92` and `0.93`. Those float a few centimetres above the real
surface. That is harmless when the robot is moved kinematically, which is all those two do
— but it would matter under physics, which is why this demo uses the measured 0.88.

**These poses are a first pass.** They were measured and then confirmed by running, but
they have not been tuned for margin. Treat them as a starting point.

## Two changes outside this folder

Both live in `coraplex/src/coraplex/`, both are additive, and both default to the existing
behaviour, so nothing that does not opt in changes.

**1. `Context.update_world_model_attachment` (default `True`)**
`coraplex/src/coraplex/datastructures/dataclasses.py`, honoured by
`MoveBranchExecutable.execute()` in `plans/executables.py`.

Taking hold of a body normally moves it under the hand in the world model, because in a
kinematically moved world nothing else would make it follow. Under physics the fingers
already hold it through contact, and re-parenting it as well stacks a second set of
degrees of freedom on it — MuJoCo rejects the model outright with
`ValueError: more than 6 dofs in body 'cube'`. `mujoco_demo.py` sets this to `False`.

**2. `GiskardExecutable.simulation_pacer` (default `None`)**
`coraplex/src/coraplex/plans/executables.py`.

`giskardpy` already has `SteppedSimulationPacer`, which steps a MuJoCo world one control
cycle at a time so the controller and the physics advance in lockstep. There was no way to
reach it from a plan: `_execute_simulation()` built its `Ros2Executor` with no pacer, and
its tick loop never called `pacer.sleep()` (the base `Executor` only calls it inside
`tick_until_end`, which coraplex does not use). The class attribute plus one
`executor.pacer.sleep()` in that loop is the whole hook; with the default `None` it
resolves to `NoPacing`, whose `sleep()` does nothing.

## What has actually been run

| | Result |
|---|---|
| RViz backend (kinematic) | Runs clean. All four pieces reach their release poses over their holes. |
| MuJoCo backend (headless) | Runs clean, about 85 s. Cube and rectangular prism drop into their holes. The cylinder lands on the lid 1 cm off its hole. The triangular prism is pushed about 3 cm instead of being lifted. |
| Real backend | **Never run.** No hardware was available. |

Final positions from the MuJoCo run (the drawer floor is at z = 0.888, the lid top at 0.96):

| piece | ended at | release pose | |
|---|---|---|---|
| cube | `0.871, -0.087, 0.903` | `0.870, -0.087, 0.985` | in its hole |
| cylinder | `0.817, -0.085, 0.973` | `0.828, -0.087, 0.985` | on the lid, 11 mm off |
| rectangular_prism | `0.823, -0.006, 0.908` | `0.823, 0.000, 0.985` | in its hole |
| triangular_prism | `0.615, 0.406, 0.891` | `0.863, 0.000, 0.985` | never lifted |

## Open points for whoever picks this up

- **The triangular prism is not grasped under physics.** The ICRA branch saw the same
  thing: it is held by one face and one edge, which is marginal.
- **The cylinder misses its hole by 11 mm.** The hole leaves 2 mm of clearance each side.
  The cylinder most likely shifts in the hand on the carry.
- **Grasp reliability under physics is the known weak spot for this task.** The earlier
  Tracy montessori work on `icra_final` records that even with dedicated friction and
  solver tuning, typically only one or two of five shapes completed the task. None of that
  tuning exists on `main`. The pieces here only carry `main`'s
  `ContactParameters.create_for_grasped_object()`, and two of four go in.
- **`real_demo.py` is untested scaffolding.** It mirrors `coraplex_real_tracy/demo.py`,
  which is the one real-Tracy path in the repository. Per the `icra_final` notes, no
  `*_real.py` in this codebase has ever been run on hardware.
- **No CI entry and no `test_demo.py`**, by choice. The other demo folders each have a
  `test_demo.py` that just imports the demo, wired into
  `.github/workflows/examples_and_demos.yml`. That would not work unchanged here, since
  importing `demo` does not run anything — `main()` is guarded by `__name__ == "__main__"`.

## Why none of the existing montessori code is reused

There is a much larger Tracy montessori implementation on `icra_final`, at
`experiments/src/experiments/tracy_experiments/montessori/` — roughly 2300 lines across
nine files, with a world builder, hole detection, event monitoring and a dashboard.

None of it exists on `main`. Neither `experiments/src/experiments/tracy_experiments/` nor
`experiments/src/experiments/montessori/` is there at all, so this demo is self-contained
and depends only on `coraplex` and `semantic_digital_twin`. The board and the pieces are
built from `Box` and `Cylinder` primitives rather than the `board.stl` that only exists on
that branch.

The trade is not all bad. `main` has since grown the cleaner API: `TracyLeftArm` and
`TracyRightArm` are `UR10eArm` subclasses, the grippers are `Robotiq85Gripper` subclasses,
and both base classes implement `_setup_servos()` and `_compensate_gravity()`, which
`AbstractRobot.from_world()` calls for every robot part. So `Tracy.from_world(world)`
already equips the arms and grippers with position servos and gravity compensation, and
the whole of `icra_final`'s `equipment.py` is unnecessary here.

## One trap worth recording

The RViz backend originally created a `SingleThreadedExecutor` spinning on a daemon
thread, copied from the pattern in `icra_final`'s `pickup_demo_simulated.py`. It crashed
the process with `terminate called without an active exception` and a core dump.

The executor is not needed. `VizMarkerPublisher` is a model-change callback and
`TFPublisher` a state-change callback — the world calls both directly when it changes.
Neither is driven by a ROS timer, so nothing has to spin the node for markers and
transforms to be published. Removing the executor and its thread fixed it.
