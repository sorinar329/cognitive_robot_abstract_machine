# Recording real-robot runs

Procedure for recording `stacking_demo_real.py`/`montessori_demo_real.py` runs on the
physical Tracy, so a run can later be sliced per action and replayed against MuJoCo to
tune `equipment.py`'s `ServoGains` (stiffness/damping/torque/velocity limits) against
real, measured behaviour instead of guessed numbers.

## Why

Giskard's real-execution path (`GiskardExecutable._execute_real`) talks to hardware
purely through ROS topics -- `TracyVelocityInterface` (`giskardpy/.../iai_robots/tracy/
configs.py`) subscribes to each arm/gripper's own `joint_states` for feedback and
publishes velocity commands on each arm's `forward_velocity_controller/commands`. Both
sides of that loop are ordinary ROS messages, so recording them gives a faithful log of
what was commanded and what actually happened, with no extra instrumentation needed on
the robot side.

That log is the input to the MuJoCo side of the workflow: replay the *commanded* series
into MuJoCo's own actuators the same way `follow_joint_trajectory`
(`trajectory_planning.py`) already does, and compare MuJoCo's resulting response against
the *measured* series from the real robot for the same segment. The gap between them is
the signal for retuning `ServoGains`.

## What to record

```bash
ros2 bag record -o <output-dir> \
  /left_arm/joint_states \
  /right_arm/joint_states \
  /left_gripper/joint_states \
  /right_gripper/joint_states \
  /left_arm/forward_velocity_controller/commands \
  /right_arm/forward_velocity_controller/commands \
  /tracy_experiments/action_marker
```

- The four `joint_states` topics are the **measured** side of the loop.
- The two `forward_velocity_controller/commands` topics are the **commanded** side --
  what Giskard actually sent. Recording both, not just one, is what lets you compare
  "what we asked for" against "what happened" later, rather than only ever seeing the
  outcome.
- `/tracy_experiments/action_marker` is not a robot topic; see below.

Only the left arm/gripper topics are relevant to `STACK_ARM = Arms.LEFT` /
`PICK_ARM = Arms.LEFT`, but record both arms anyway -- cheap, and rules out having to
re-run because the wrong side was left out.

Start the recording once the Giskard node and world-fetcher are up (after `main()`'s
initial `time.sleep(8)`), stop it in the same `finally:` block that already tears down
`giskard_process`, the same way `stacking_demo_real.py`/`montessori_demo_real.py` launch
and kill that process today.

Store recordings outside version control, e.g.
`experiments/src/experiments/tracy_experiments/stacking/recordings/<timestamp>/` (or
`montessori/recordings/<timestamp>/`) -- add that `recordings/` path to `.gitignore`,
bags are run data, not source.

## Mapping a recording to the action that produced it

A raw bag is just a stream of joint states -- nothing in it says which stretch was "pick
cube_1" vs. "place cube_2" without a marker. Publish a small JSON message on
`/tracy_experiments/action_marker` (`std_msgs/String`) immediately before and after each
`PickUpAction`/`PlaceAction`:

```json
{"phase": "start", "index": 3, "action": "place", "object": "cube_2", "arm": "left", "target_pose": {"x": 0.8, "y": 0.0, "z": 0.175}}
{"phase": "end",   "index": 3, "action": "place", "object": "cube_2", "arm": "left", "success": true}
```

Because the marker is recorded in the *same* bag as the joint topics, both sides use the
bag's own recording clock -- no separate wall-clock log to align by hand later. Concretely:
`ros2 bag record` timestamps every message at the moment it receives it, from the ROS
system clock (this is wall-clock time as long as nothing has `use_sim_time` set, which is
the normal case for a real-hardware run). `JointState` additionally carries its own
`header.stamp`, set the same way by whichever node publishes it; `std_msgs/String` has no
such field, so the marker relies purely on the bag's own recv-timestamp -- which is fine,
since that's the timestamp every other topic in the bag is compared against anyway.

Post-processing: read the bag once, pair up consecutive `start`/`end` markers by
`index`, and everything on the joint/command topics between those two timestamps belongs
to that action.

## Defining "success"

There is no automatic ground-truth signal here -- it has to be a human judgement call,
recorded in the marker's `end` message. Two things rule out relying on this codebase's
own formal success checks:

1. **The cubes/shapes are symbolic anchors, not perceived.** They're added to the live
   world at hand-measured coordinates (see "Table layout" below), never updated by real
   perception. Once `PickUpAction` runs `AttachNode`, the object's world-model pose is
   *defined* by the kinematic attachment to the tool frame, not sensed -- so
   `is_body_gripped` (the ray-cast check `PickUpAction.post_condition` uses) can't
   register a miss even if the real gripper grasped nothing. `PlaceAction`'s pose check
   compares against the *commanded* `target_location`, not a measured one, so it would
   trivially pass regardless of what physically happened.
2. **These checks aren't even running.** Both demos build their `Context` with
   `evaluate_conditions=False` (matching `coraplex_real_tracy/demo.py`), and
   `GiskardExecutable` only wires pre/post-condition monitors in when that flag is true.

The one thing that *is* available for free is whether the action raised
(`MotionDidNotFinish`, a Giskard-side failure) -- that tells you the motion completed
without erroring, not that the grasp/placement was actually correct. Treat it as a floor,
not a substitute for watching the run.

Recommended convention for the operator setting `success` on the `end` marker:

- **Stacking**: `true` only if the cube visibly ended up placed on the stack, not just
  released near it.
- **Montessori**: the action itself only places a shape *above* its hole --
  `build_sorting_actions`'s own docstring is explicit that there is no fall-through check
  or retry. Agree in advance whether `success` means "released within a few cm above the
  correct hole" (what the action actually attempts) or "visibly fell through" (what a
  bystander naturally judges) -- these are different claims, and Montessori additionally
  has more failure modes than a missed grasp (`CollisionViolatedError` from the board/
  drawers, a shape knocked out of the way), so a bare boolean is worth widening to a short
  failure-reason string (`"grasp"`, `"collision"`, `"misplaced"`, `"other"`) rather than
  just `false`.

## Verifying a recording afterward

- `ros2 bag info <output-dir>` -- sanity check topic list and message counts are non-zero
  on every topic above, and duration roughly matches how long the run actually took.
- Read the bag, collect the marker messages, and confirm they pair up start/end by
  `index` with no orphan (a `start` with no matching `end`, or vice versa) -- an orphan
  usually means the process was killed mid-action rather than the run finishing cleanly.
- Per action segment: the `joint_states` message count should roughly match
  `segment_duration * publish_rate`, and the matching `forward_velocity_controller/
  commands` topic should have at least one message in that window -- an empty commands
  segment for an action that has a `success: true` marker is a sign the marker boundaries
  are off, not that the action genuinely sent no commands.
- The real check that the recording is usable for its actual purpose: replay the
  segment's commanded series into MuJoCo and confirm it produces a comparable-looking
  trajectory shape to the segment's measured series -- if MuJoCo's response looks
  qualitatively nothing like the real one even before any gain tuning, something is wrong
  with the slice, not (yet) with the servo gains.

## Table layout and the two hardcoded positions

For the first real-robot tries, no perception is used -- both the pick and the place/
stack position are the two hardcoded points already in `stacking_demo_mujoco.py`/
`stacking_demo_real.py`:

- `_STACK_XY = (0.8, 0.0)` -- place/base position.
- `_PICK_XY_LIST[0] = (0.8, 0.25)` -- pick position.

Both are in the world root frame, which **is** Tracy's own mount point
(`TRACY_MOUNT_X = TRACY_MOUNT_Y = 0.0`, mounted with `mount_yaw=0.0`, i.e. no rotation
between that frame and the table's own edges). `z` for both is
`equipment.table_top_z(robot) + CUBE_SIZE / 2`.

```
                              far edge
        +---------------------------------------------+
        |                                              |
        |                                              |
        |                                              |
      D |                    • pick (0.8, 0.25)        |
        |                    |                         |
        |                  0.25 m                      |
        |                    |                         |
        |         b          • stack (0.8, 0.0)         |
        |<------->           |                         |
        |         ▣----------+-- 0.8 m ----------------|
        |    Tracy mount / root-frame origin (0, 0)     |
        +---------------------------------------------+
        corner  <---   a   --->
       (reference)
                near edge
```

`▣` is Tracy's mount point -- the frame the code above already uses, and the one thing
that's certain. `a`/`b` (distance from whichever table corner is easiest to measure from,
to that mount point, along each edge) and the table's own footprint (`W`/`D`) are **not**
known in this environment: there's no `iai_tracy_description` checkout here to read the
table geometry from, and no table dimensions are documented anywhere in this repo.

Once `a`/`b` are measured by hand tomorrow (tape measure, from the chosen corner to the
mount point, along the table's edges), the corner-frame coordinates of the two points are
just:

```
corner_x = a + root_frame_x
corner_y = b + root_frame_y
```

giving `pick = (a + 0.8, b + 0.25)` and `stack = (a + 0.8, b + 0.0)` from that corner.
