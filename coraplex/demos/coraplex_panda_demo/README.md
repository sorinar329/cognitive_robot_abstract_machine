# Coraplex Panda Demo

A Franka Panda stacking cubes in MuJoCo, driven by a Coraplex plan.

A MuJoCo viewer opens and the robot picks up `cube1`, stacks it on `cube0`, then
stacks `cube2` on `cube3`. The demo ends when the viewer is closed.

## Files

| file | what it is |
|---|---|
| `demo.py` | The plan and its simulation setup |
| `stacking_scene.xml` | MJCF scene: Panda, table, four cubes |
| `panda_assets.py` | Downloads the Panda meshes on first run |
| `live_events.py` | The SegMind detectors watching the run, and the page showing what they detect |
| `live_events.html` | That page |

Both `demo.py` and `stacking_scene.xml` use paths relative to their own
location, so the demo runs from any working directory and any checkout.

## Meshes

The Panda meshes are about 33 MB, so they are not committed. On first run
`PandaMeshAssets` downloads the 67 meshes the scene references from
[`mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie) into
`franka/assets/`, and later runs reuse them. The list comes from the scene's own
`<mesh>` declarations and the destination from its `meshdir`, so neither can
drift out of step with the scene.

This needs network access the first time only. Pin `PandaMeshAssets.revision`
to a commit if you need the meshes to stay fixed against upstream changes.

## The live event page

While the demo stacks, SegMind's detectors watch the same world on a thread of their
own and a page at <http://127.0.0.1:5000> lists what they detect, newest first:

```
cube1 LossOfSupportEvent floor
cube1 PickUpEvent
cube1 SupportEvent       cube0
cube1 PlacingEvent       cube0
```

Its second tab draws the statechart those detectors tick in, as they stand, so each one
shows the state it is in. Set `SHOW_LIVE_EVENTS = False` in `demo.py` to run without it;
it is the only part of the demo that needs `flask`, and the statechart tab also needs
graphviz's `dot`.

A tick reads the world while holding its lock, so watching costs the run time. The watch
keeps to `EventWatch.share_of_the_time_watching` of it, and at the default the stacking
runs at close to the pace it has unwatched.

What the demo does to its own scene is kept off the page: the cubes being teleported back
between iterations, and their settling afterwards, would otherwise read as them being
picked up and placed by nobody.

## What makes this demo different

The point of this demo is that **the grasp is real**. The cube is held by
nothing but contact and friction between the gripper's fingers, so a grasp that
would not physically work visibly fails instead of being silently rescued.

This required a few changes to how the attached node is handled in the simulation and world model.
Instead of beeing there automatically, the attached node is now an optional part, which can be set in the context. 


Two settings in `demo.py` establish that, and they belong together:

- `MujocoSim(mirror_attachments=False)` — grasping does not weld the object to
  the gripper in MuJoCo.
- `Context(update_world_model_attachment=False)` — grasping does not reparent
  the object to the gripper in the world model either.

Set only the first and the world model still believes the object is rigidly
attached, so its pose becomes forward-kinematics fiction that hides whether the
gripper is really holding anything.

The arm is also genuinely dynamically simulated rather than teleported along the
commanded trajectory:

```python
physically_simulated_dofs = {c.raw_dof for c in gripper.active_connections} | {
    c.raw_dof for c in arm.active_connections
}
```

Every arm joint is driven through MuJoCo's actuators and can lag, sag or be
stopped by contact. `MujocoBody(gravitation_compensation_factor=1.0)` on each
arm link offsets gravity, because the scene's actuator gains assume it is
cancelled separately rather than held against by the gains alone.

Finally, motion execution is paced against MuJoCo's clock rather than the wall
clock:

```python
context.simulation_clock = lambda: multi_sim.simulator.current_simulation_time
```

This simulation runs at roughly 0.7x real time, and without this the controller
issues commands faster than the simulated arm can execute them.

## Tuning knobs

| name | meaning |
|---|---|
| `STACK_HEIGHT_OFFSET` | Height above a target cube's centre to release at. A cube is 0.04 tall, so anything above that is release clearance — and it has to stay above the arm's own vertical positioning error, or the arm drives the carried cube into the one below instead of letting go above it. |
| `max_ticks_per_motion_mapping` | Control cycles a single motion may take before it is declared stalled. A dynamically simulated arm settles far more slowly than a teleported one and needs a larger budget. |
| `step_size` | MuJoCo timestep. Smaller is more accurate but costs real time, and the simulation already does not hold 1.0x. |
| cube `friction` in the scene | All four cubes currently share `1 0.05 0.001`. Lowering a cube's sliding friction is the most direct way to make its grasp harder. |

## Known limitations

Stacking is **not yet reliable**. The demo mostly succeeds, but when its trying to stack all the cubes on top of one another, it mostly fails.
