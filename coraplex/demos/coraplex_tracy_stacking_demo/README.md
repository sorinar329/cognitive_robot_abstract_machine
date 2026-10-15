# Tracy stacking demo

Tracy picks three cubes off its own table, one after another, and stacks them onto a
fourth that stands where the tower is built. The scene, the poses and the plan are the
same however it runs; only how the plan is carried out differs, by coraplex's
`ExecutionType`: physically simulated in MuJoCo, kinematically simulated in RViz, or on
the real robot. It is built the same way as `../coraplex_tracy_montessori_demo`.

There is no perception: where the cubes start and where the tower goes is hardcoded.

## Running it

Choose the execution type at the bottom of `demo.py`:

```python
main(ExecutionType.PHYSICALLY_SIMULATED)  # or KINEMATICALLY_SIMULATED, or REAL
```

Then, **from inside this folder**:

```bash
python demo.py
```

| Execution type | What you need | What you see |
|---|---|---|
| `KINEMATICALLY_SIMULATED` | ROS 2, RViz2 | A `MarkerArray` display on `/semworld/viz_marker`, durability set to transient local. The robot is moved kinematically. |
| `PHYSICALLY_SIMULATED` | `mujoco` | A MuJoCo viewer. Pass `headless=True` to the `MujocoSim` in `mujoco_demo.py` to run without one. |
| `REAL` | The robot, its ROS stack | It launches `giskardpy_tracy_standalone.launch.py` itself and fetches the world from the running stack. |

## File layout

| File | Role |
|---|---|
| `demo.py` | The entry point. Holds `StackingEnvironment` — every hardcoded pose and size as a field, the cubes it stands in a world, and the plan that stacks them — and the dispatch on the execution type. |
| `rviz_demo.py` | Kinematic run, published for RViz. |
| `mujoco_demo.py` | Physics run, controller and physics stepped in lockstep. |
| `real_demo.py` | Real robot, through giskard. |

## What the plan does

```
ParkArmsAction(all arms)
for each loose cube:
    PickUpAction(grasp from above, left arm)
    PlaceAction(cube, 3 mm above the top of the tower)
ParkArmsAction(all arms)
```

The grasp approaches a cube from above, the fingers closing along the cube's own x,
`grasp_height` (18.5 mm) above its bottom face, so the closed finger pads stay 5 mm above
whatever the cube is set down on. The number of cubes is the number of `cube_colors`.

## Where the numbers come from

| | x | y | z |
|---|---|---|---|
| loose cubes (start) | 0.55 | 0.1 / 0.2 / 0.3 | on the table, `table_top_z = 0.88` |
| tower | 0.75 | 0.0 | on the table, 4 cm per level |

**These poses are a first pass.** Treat them as a starting point.

## Running under physics

`mujoco_demo.py` performs the plan inside coraplex's `PhysicallySimulatedRobot`, with the
Montessori demo's contact settings, except that it leaves MuJoCo's slip correction off:
that applies to every contact in the scene, so it would also glue each cube to the one
below it.

## What has actually been run

| | Result |
|---|---|
| Kinematically simulated | Runs clean; every cube ends on its release pose. |
| Physically simulated (headless) | All three cubes stacked, each within 1 mm of the tower's axis, at z 0.94 / 0.98 / 1.02 on the bottom cube at 0.90. |
| Real | **Never run.** |
