# Tracy PC: how the Montessori demo with cramera is set up, and how to fix it

Everything needed to run the real Tracy Montessori demo and watch it live in cramera on
the Tracy PC (`ai-0050`, user `tracy`), and to put it back together when something
breaks. State as of 13 Nov 2026.

## 1. What lives where

| What | Where | Branch / state |
|---|---|---|
| cram (this repo): the demo | `~/workspace/cognitive_robot_abstract_machine` | `tracy-cramera-demo` on the fork `sorinar329/cognitive_robot_abstract_machine` (remote `sorin`) |
| cramera (the viewer) | `~/workspace/cramera` | `real-robot-plan-snapshots` on `sorinar329/cramera` |
| Tracy's URDF (`iai_tracy_description`) | `~/workspace/ros/src/iai_tracy` | `fix/gripper-udev-composite-adapter` at `910ccb6` **plus** `iai_tracy.patch` (local only, see 5.1) |
| Giskard launch files (`giskardpy_ros`) | `~/workspace/ros/src/cram_ros2_packages` | `main` at `8d2b419` **plus** `cram_ros2_packages.patch` (local only, see 5.2) |
| Python environment | `~/.virtualenvs/cram-env` (`workon cram-env`) | cram packages and cramera installed editable |
| ROS workspace that is actually used | `~/workspace/ros` (merged install in `~/workspace/ros/install`) | **not** `~/ros2_ws`, that copy is never built |

The two `.patch` files next to this README are the local changes to the two ROS repos.
They exist nowhere else; if a checkout is reset, re-apply them (section 5).

## 2. Starting everything

Every terminal: `workon cram-env`, ROS sourced as usual. The robot runs on
`ROS_DOMAIN_ID=2` (set in `~/.bashrc`).

1. **Robot bringup**, as usual (`ros2 launch iai_tracy_bringup tracy_ros2.launch.py`).
2. **Giskard**:
   ```bash
   ros2 launch giskardpy_ros giskardpy_tracy_velocity.launch.py
   ```
   Opens RViz with `tracy.rviz` (fixed frame `table_footprint`).
3. **cramera viewer**:
   ```bash
   cramera --no-browser
   ```
4. **The demo**:
   ```bash
   cd ~/workspace/cognitive_robot_abstract_machine/coraplex/demos/coraplex_tracy_montessori_demo
   python real_demo.py --live-port 8765
   ```
   It stops at "World loaded; check it in RViz, then press Enter". Press Enter to run.
5. **Browser**: http://127.0.0.1:8711/?live=http://127.0.0.1:8765
   It shows nothing until you pressed Enter in step 4; reload the page after that.

**Between two demo runs, restart Giskard (step 2).** The board and pieces of the last run
stay in Giskard's world; a new run without restart adds them twice and fails.

Stop the demo with **Ctrl+C**, never Ctrl+Z (a suspended demo keeps port 8765 busy, see
section 6).

## 3. Quick health checks

Run these with bringup and Giskard up.

| Check | Command | Expected |
|---|---|---|
| Gripper coupler is in the URDF | `ros2 run tf2_ros tf2_echo left_wrist_3_link l_gripper_tool_frame` | z ≈ **0.174** (0.144 means the old URDF without the 3 cm coupler is installed → section 4) |
| `map` is connected to the robot | `ros2 run tf2_ros tf2_echo map l_gripper_tool_frame` | prints a transform (if "not part of the same tree" → section 5.2) |
| Camera calibration is the live one | `ros2 run tf2_ros tf2_echo table camera_link` | translation ≈ `0.429 0.003 0.893` |
| Gripper driver publishes | `ros2 topic echo /left_gripper/joint_states --once` | one joint, `left_robotiq_85_left_knuckle_joint` |
| Demo live server is up | `curl http://127.0.0.1:8765/info` | JSON with `"running":true` |

## 4. The URDF (iai_tracy_description)

What matters in `~/workspace/ros/src/iai_tracy/iai_tracy_description/urdf/tracy.urdf.xacro`:

- **3 cm gripper coupler** between each wrist and its Robotiq gripper
  (`gripper_coupler_length = 0.03`, from commit `9ca13da`). It is real, it is on the
  robot. Without it the model's fingers end 3 cm higher than the real ones, the robot
  only grasps when objects are raised by 3 cm in the demo, and they float in the model.
- **Root link `table_footprint`** (was `map`, commit `09e265f`). Giskard's launch files
  must say so (section 5.2).
- **Camera calibration** `camera_link_joint` origin
  `xyz="0.429417 0.002602 0.892603" rpy="0.040005 1.171431 0.032689"` (Sept 2026). This
  line comes from `iai_tracy.patch`; the committed `910ccb6` has an older one.

Rebuild after **any** change to the URDF, then restart bringup and Giskard:

```bash
cd ~/workspace/ros
colcon build --packages-select iai_tracy_description --merge-install
```

The installed copy is a plain copy, not a link: editing the source without rebuilding
changes nothing.

## 5. Re-applying the local changes

### 5.1 iai_tracy (camera calibration)

```bash
cd ~/workspace/ros/src/iai_tracy
git checkout fix/gripper-udev-composite-adapter   # at 910ccb6
git apply ~/workspace/cognitive_robot_abstract_machine/coraplex/demos/coraplex_tracy_montessori_demo/pc_setup/iai_tracy.patch
cd ~/workspace/ros && colcon build --packages-select iai_tracy_description --merge-install
```

**By hand**, if the patch does not apply (the file changed upstream): in
`iai_tracy_description/urdf/tracy.urdf.xacro`, find `camera_link_joint` and make its
`<origin>` the only one that is not commented out:

```xml
<joint name="camera_link_joint" type="fixed">
    <parent link="table"/>
    <child link="camera_link"/>
    <origin xyz="0.429417 0.002602 0.892603" rpy="0.040005 1.171431 0.032689"/>
</joint>
```

While in that file, check that the gripper coupler is still there: `grep -c
gripper_coupler tracy.urdf.xacro` must print a number above 0, and both
`robotiq_gripper` blocks must have `parent="..._gripper_coupler_link"` with
`<origin xyz="0.0 0.0 ${gripper_coupler_length}" .../>`. Then rebuild as above.

### 5.2 giskardpy_ros (launch files and RViz config)

What the patch changes:
- both Tracy launch files: `root_links=["table_footprint", "table_footprint"]` (the
  interactive markers crash with "No world entity with name 'map'" otherwise), a static
  `map → table_footprint` transform, and RViz started with `tracy.rviz`;
- `launch_helpers.py`: two optional arguments `map_root_link` and `rviz_config` (other
  robots unchanged);
- `launch/rviz_config/tracy.rviz`: fixed frame `table_footprint`, plus a MarkerArray
  "SemanticWorld" on `/semworld/viz_marker` (transient local).

```bash
cd ~/workspace/ros/src/cram_ros2_packages
git checkout main   # at 8d2b419
git apply ~/workspace/cognitive_robot_abstract_machine/coraplex/demos/coraplex_tracy_montessori_demo/pc_setup/cram_ros2_packages.patch
cd ~/workspace/ros && colcon build --packages-select giskardpy_ros --merge-install
```

**By hand**, if the patch does not apply. All paths are under
`~/workspace/ros/src/cram_ros2_packages/giskardpy_ros/`.

1. `launch/giskardpy_tracy_velocity.launch.py` and
   `launch/giskardpy_tracy_standalone.launch.py`: change the arguments to
   ```python
   root_links=["table_footprint", "table_footprint"],
   ...
   map_world_tf=True,
   map_root_link="table_footprint",
   rviz_config="launch/rviz_config/tracy.rviz",
   ```
   This is the minimal fix: `root_links` alone stops the interactive-marker crash.
   The other two lines need step 2.
2. `giskardpy_ros/launch_helpers.py`, in `giskard_launch_description`:
   - add the parameters `rviz_config=None, map_root_link=None` after `map_world_tf=False`;
   - after the `if map_world_tf:` block, add
     ```python
     if map_root_link:
         nodes.append(
             Node(
                 package="tf2_ros",
                 executable="static_transform_publisher",
                 name="map_root_link_transform_publisher",
                 output="screen",
                 arguments=["--frame-id", "map", "--child-frame-id", map_root_link],
             )
         )
     ```
   - in the `if rviz:` block, pass the config to RViz:
     ```python
     rviz_arguments = []
     if rviz_config:
         rviz_arguments = [
             "-d",
             PathJoinSubstitution(
                 [FindPackageShare("giskardpy_ros"), *rviz_config.split("/")]
             ),
         ]
     ```
     and add `arguments=rviz_arguments,` to that `Node(package="rviz2", ...)`.
3. `launch/rviz_config/tracy.rviz`: if it is gone, open RViz, set Global Options → Fixed
   Frame to `table_footprint`, add a MarkerArray display on `/semworld/viz_marker` with
   Durability Policy *Transient Local*, and File → Save Config As… to that path.

Then rebuild as above and restart Giskard.

### 5.3 cramera

```bash
cd ~/workspace && git clone git@github.com:sorinar329/cramera.git   # if missing
cd cramera && git checkout real-robot-plan-snapshots
workon cram-env
pip install -e packages/protocol -e packages/viewer -e packages/bridge
```

`real-robot-plan-snapshots` makes the viewer follow the real robot continuously; without
it, cramera only updates at pick and place.

### 5.4 This repo

```bash
cd ~/workspace/cognitive_robot_abstract_machine
git fetch sorin && git checkout tracy-cramera-demo
```

Pushing to the fork needs the sony key (the repo's default `core.sshCommand` uses
another key, which is why pushing from PyCharm fails):

```bash
GIT_SSH_COMMAND="ssh -i ~/.ssh/sony-ssh -o IdentitiesOnly=yes" git push sorin tracy-cramera-demo
```

## 6. When something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: [Errno 98] Address already in use` when the demo starts | An earlier demo still holds port 8765, often one stopped with Ctrl+Z (`ps` shows state `T`) | `ss -ltnp \| grep 8765` for the PID, then `kill -9 <pid>`. Or use `--live-port 8766` and `?live=http://127.0.0.1:8766` |
| Objects float above the table in the model, or the robot grasps 3 cm too low | Installed URDF is stale (no coupler) | Section 4: rebuild, restart bringup and Giskard, check z ≈ 0.174 |
| `interactive_marker` dies with "No world entity with name 'map'" | Launch files still say `map` | Section 5.2 |
| RViz prints "Sending request for interactive markers" endlessly | RViz cannot place Giskard's arm markers; not fully understood yet | Harmless for the demo: only dragging the arms in RViz is affected |
| `tf2_echo map ...`: "not part of the same tree" | Nothing connects `map` to `table_footprint` | Section 5.2, or by hand: `ros2 run tf2_ros static_transform_publisher --frame-id map --child-frame-id table_footprint` |
| `WorldUpdateReferencesUnknownEntityError ... 'shape_sorting_board'` | Board and pieces of an earlier run are still in Giskard's world | Restart Giskard before every run |
| cramera only moves at pick and place | cramera installed from a branch without the fix | Section 5.3 |
| `[RTPS_TRANSPORT_SHM Error] Failed init_port ...` | Leftover shared-memory files of earlier ROS processes | Harmless, ignore |
| Tests fail at collection with `No module named 'griplink_interfaces'` | The root `test/conftest.py` imports the DAiSy robot, whose ROS package is not on this PC | Put a stub package `griplink_interfaces/action.py` defining `Flexgrip, Flexrelease, Grip, Release` on `PYTHONPATH` for the test run |

## 7. Known open problems

- **The model's fingers do not close on the real robot.** Giskard's own world closes them
  (visible in RViz's Giskard markers), but the value of
  `left_robotiq_85_left_knuckle_joint` does not reach the demo's world. So cramera shows
  the gripper open and segmind never detects a grasp, pick-up or place; only translations
  and rotations. The demo prints a line `Grasp of <piece>: thumb … mm, finger … mm` on
  every grasp: about −18 mm on both sides means the fingers are still open in the model.
- **cramera sometimes stops updating in the middle of a run.** Cause unknown.
- **Nothing falls in the real-robot model.** A released piece stays 1 cm above the lid in
  the model, so there are no containment events on the real robot (in MuJoCo there are).
- **Segmind and the table.** Tracy's table is part of the robot. Since
  `tracy-cramera-demo`, segmind's support detector reads robot bodies too, so "piece
  rests on the table" is detected; contact and containment still leave the robot out.
