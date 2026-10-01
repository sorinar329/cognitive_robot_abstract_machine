# Tracy stacking demo

Tracy picks cubes off its own table, one after another, and stacks them into a tower,
four by default.
The scene, the poses and the plan are the same wherever it runs; only how the plan is
carried out differs. Three backends: MuJoCo physics, RViz, and the real robot. It is built
the same way as `../tracy_montessori_demo`, and its backends are that demo's with the
board and the pieces swapped for the cubes.

There is no perception: where the cubes start and where the tower goes is hardcoded.

## Running it

Pick the backend, the number of cubes and, for a simulated run, the arm speed near the
top of `demo.py`:

```python
BACKEND = Backend.MUJOCO      # or Backend.RVIZ, or Backend.REAL
NUMBER_OF_CUBES = 4           # 10 has been run too
ARM_SPEED_LIMIT = None        # rad/s at the fastest arm joint; None = full speed
```

Then, **from inside this folder**:

```bash
python demo.py
```

It has to be run from this folder, for the same reason as the montessori demo:
`coraplex/demos/` is not a package, so the demo finds its sibling modules through its own
directory being on `sys.path`.

| Backend | What you need | What you see |
|---|---|---|
| `MUJOCO` | `mujoco` | A MuJoCo viewer. Set `HEADLESS = True` in `mujoco_demo.py` to run without one. |
| `RVIZ` | ROS 2, RViz2 | A `MarkerArray` display on `/semworld/viz_marker`, durability transient local. The robot is moved kinematically. |
| `REAL` | The robot, its ROS stack | It launches `giskardpy_tracy_standalone.launch.py` itself and fetches the world from the running stack. |

## Arm speed

Loading Tracy (`Tracy.from_world`) slows its arms down so that their fastest joint turns at
0.2 rad/s, keeping the joints' proportions (`Tracy._setup_velocity_limits`). That limit is
meant for the real robot. The simulated backends replace it with `ARM_SPEED_LIMIT`:

- `None` gives every joint back the velocity limit of the robot's description, about
  2 rad/s for the shoulders and 3 rad/s for the wrists of the UR10e.
- A number slows the arms down so their fastest joint turns at that speed, the same way
  `Tracy._setup_velocity_limits` does. `0.2` reproduces the default.

The grippers always keep their described speed. The real backend fetches its world from the
running stack and keeps the robot's own limits; `ARM_SPEED_LIMIT` does not apply there.

Cartesian motions of the hand are still normalised by giskard's reference velocity of
0.2 m/s, and parking uses giskard's default joint goal speed of 1 rad/s. The per-phase caps
of `PickUpAction` and `PlaceAction` (`lift_linear_velocity`, `transport_linear_velocity`,
...) are left unset.

## Physics (MuJoCo)

`mujoco_demo.py` takes the montessori demo's contact settings (rigid, critically damped
contacts; elliptic friction cone; `impratio = 10`) with two changes, so that the tower is
not steadier than wooden blocks are:

- **The cubes weigh 40 g** (`CUBE_MASS` in `demo.py`), a 4 cm block of wood at about
  0.6 g/cm³, with the inertia of a solid cube. MuJoCo is built with
  `inertiafromgeom = true`, which works every mass out from the geoms at the density of
  water and counts the drawn geom as well as the colliding one: left alone, a cube weighs
  128 g. `_weigh_the_cubes` sets the cubes' own mass and inertia on the compiled model.
  Recomputing the model's constants (`mj_setConst`) moves every body to its reference
  pose, which for a cube is inside the table, so the state is saved and restored around
  it.
- **No slip correction** (`NO_SLIP_ITERATIONS = 0`, the montessori demo uses 10). MuJoCo
  applies it to every contact in the scene, not only the pads, so it also glued each cube
  to the one below it.

Sliding friction is 0.3 between cubes and 1.0 between a cube and the table (MuJoCo takes
the larger of the two geoms' values; the table keeps MuJoCo's default).

## What the plan does

```
ParkArmsAction(BOTH)
for each cube, bottom of the tower first:
    PickUpAction(cube, LEFT, grasp from the front, aligned to the top)
    PlaceAction(cube, 3 mm above the cube below it, LEFT)
ParkArmsAction(BOTH)
```

## Where the numbers come from

| | x | y | z |
|---|---|---|---|
| cubes (start) | 0.55, then 0.45 for cubes 6–10 | 0.1 / 0.2 / 0.3 / 0.4 / 0.5 | on the table, `TABLE_TOP_Z = 0.88` |
| tower | 0.75 | 0.0 | on the table, 4 cm per level |

The loose cubes stand in rows of five (`CUBES_PER_ROW`), each further row 10 cm nearer
to Tracy, so ten cubes still fit on the table. The cubes have 4 cm edges and run through
the hues from red at the bottom of the tower (`cube_color`). As in
the montessori demo, each cube's body origin, which the grasp aims at, sits 18.5 mm above
its bottom face (`GRASP_HEIGHT`), so the closed finger pads stay 5 mm above whatever the
cube is set down on. A cube is let go of 3 mm above the one below it (`RELEASE_HEIGHT`).

**These poses are a first pass.** Treat them as a starting point.

## What has actually been run

| | Arm speed | Result |
|---|---|---|
| MuJoCo (headless) | full (`None`) | All four cubes stacked. 35.7 s simulated. Same result with and without the two physics changes above. |
| MuJoCo (headless) | 0.2 rad/s | All four cubes stacked. 92.2 s simulated. Run before the physics changes. |
| MuJoCo (headless), 10 cubes | full (`None`) | All ten stacked into a 40 cm tower, each within 2–3 mm of its axis. 86.0 s simulated. Same result with and without the two physics changes above. |
| RViz backend (kinematic) | full (`None`) | Runs clean; every cube ends on its release pose. |
| Real backend | | **Never run.** Same scaffolding as the montessori demo's `real_demo.py`. |

Final positions from the four-cube MuJoCo run at full speed, which was run with the
four colours then fixed as red, green, blue, yellow (4 cm per level, table top at z = 0.88):

| cube | ended at | release pose |
|---|---|---|
| red (cube 1) | `0.747, 0.000, 0.898` | `0.75, 0, 0.902` |
| green (cube 2) | `0.748, 0.000, 0.938` | `0.75, 0, 0.942` |
| blue (cube 3) | `0.748, 0.001, 0.978` | `0.75, 0, 0.981` |
| yellow (cube 4) | `0.748, 0.001, 1.018` | `0.75, 0, 1.021` |

Each cube settles 3–4 mm below where it is let go of, which is the release gap, and within
3 mm of the tower's axis.
