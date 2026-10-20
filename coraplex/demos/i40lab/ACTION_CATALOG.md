# Action catalog

Everything that can go into a plan, and what each item needs.
Plans are built as `sequential([...], context=context).plan`; actions and
standalone motions mix freely in the list.

Sources: `coraplex/src/coraplex/robot_plans/` (`actions/` and `motions/`).

---

## Pick and place

### ReachAction

Reaches a pose: first a pre-pose pulled back along the approach axis, then the
final approach. Pass `open_gripper_at_pre_pose=True` instead of adding a
separate open motion.

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `target_pose` | Pose | yes | — | |
| `arm` | Arms | yes | — | |
| `grasp_description` | GraspDescription | yes | — | |
| `object_designator` | HasRootBody | no | None | |
| `reverse_reach_order` | bool | no | False | |
| `open_gripper_at_pre_pose` | bool | no | False | |
| `perceive_before_grasp` | bool | no | False | |
| `position_threshold` | float | no | None | Goal radius, metres |
| `orientation_threshold` | float | no | None | Goal radius, radians |
| `pre_approach_linear_velocity` | float | no | None | |
| `final_approach_linear_velocity` | float | no | None | |

### PickUpAction

Reach, close, attach, lift.

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `object_designator` | HasRootBody | yes | — | |
| `arm` | Arms | yes | — | |
| `grasp_description` | GraspDescription | yes | — | |
| `tolerate_grasp_stall` | bool | no | False | Completes the close on finger stall; verify the grasp afterwards, closing on air also completes |
| `perceive_before_grasp` | bool | no | False | |
| `grasp_closing_velocity` | float | no | None | |
| `lift_linear_velocity` | float | no | None | |
| `grasp_stall_minimum_time` | float | no | None | |
| `grasp_goal_threshold` | float | no | None | Default used here: `GRIPPER_GOAL_THRESHOLD = 0.002` |
| `object_friction` | float | no | None | |
| thresholds / velocities | — | no | None | Same `position_threshold`, `orientation_threshold`, approach velocities as `ReachAction` |

### GraspingAction

The grasp itself, without lift or retreat.

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `object_designator` | Body | yes | — | Plain body, not the annotated designator |
| `arm` | Arms | yes | — | |
| `grasp_description` | GraspDescription | yes | — | |
| thresholds | — | no | None | `position_threshold`, `orientation_threshold` |

### PlaceAction

Lifts first, then transports, descends, opens, retracts.

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `object_designator` | Body | yes | — | |
| `target_location` | Pose | yes | — | |
| `arm` | Arms | yes | — | |
| `grasp_release_threshold` | float | no | 0.1 | |
| velocities / thresholds | — | no | None | `transport_`, `placing_`, `retract_linear_velocity`, `release_opening_velocity`, thresholds as above |

---

## Robot utilities

### ParkArmsAction

Joint-space move to the arm's `PARK` state.

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `arm` | Arms | yes | — | |

### SetGripperAction

Set the gripper open, closed, or in between. No thresholds.

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `gripper` | Arms | yes | — | |
| `motion` | GripperState | yes | — | `OPEN` / `CLOSE` / `MEDIUM` |

### MoveManipulatorAction

Cartesian move of a given end-effector.

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `target_pose` | Pose | yes | — | |
| `end_effector` | EndEffector | yes | — | |
| `allow_gripper_collision` | bool | yes | — | |

### FollowToolCenterPointPathAction

Move the TCP along a dense pose path.

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `target_locations` | PoseTrajectory | yes | — | |
| `arm` | Arms | yes | — | |

### CarryAction

Park the arms; doubles as a safe retract. Optionally axis-aligned.

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `arm` | Arms | yes | — | |
| `align` | bool | no | False | |
| `tip_axis` / `root_axis` | AxisIdentifier | no | None | Used with `tip_link` / `root_link` |

### Not usable on the fixed UR5

`MoveTorsoAction(torso_state)`, `NavigateAction(target_location, …)`,
`MoveAndPickUpAction` / `MoveAndPlaceAction(standing_position, …)`,
`TransportAction`, `PathPlanningNavigateAction(target)` — these need a mobile
base and/or torso and raise on this robot, which is why the demo uses plain
pick and place.

---

## Standalone motions (usable directly in the action list)

### MoveGripperMotion

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `motion` | GripperState | yes | — | |
| `gripper` | Arms | yes | — | |
| `finger_velocity` | float | no | None | |
| `stall_minimum_time` | float | no | None | |
| `tolerate_stall` | bool | no | False | |
| `allow_gripper_collision` | bool | no | None | |
| `goal_threshold` | float | no | None | |

### MoveToolCenterPointMotion

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `target` | Pose | yes | — | |
| `arm` | Arms | yes | — | |
| `movement_type` | MovementType | no | `CARTESIAN` | `TRANSLATION` = position only, orientation frozen; `STRAIGHT_*` variants exist |
| `allow_gripper_collision` | bool | no | None | |
| `max_linear_velocity` | float | no | None | |
| `max_angular_velocity` | float | no | None | |
| thresholds | — | no | None | `position_threshold`, `orientation_threshold` |

### MoveTCPWaypointsMotion

Orientation is enforced only at the final waypoint.

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `waypoints` | List[Pose] | yes | — | |
| `arm` | Arms | yes | — | |
| `movement_type` | WaypointsMovementType | no | `ENFORCE_ORIENTATION_FINAL_POINT` | |
| `allow_gripper_collision` | bool | no | None | |
| thresholds | — | no | None | `position_threshold`, `orientation_threshold` |

### MoveJointsMotion

Direct joint goals; used internally by parking.

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `names` | List[str] | yes | — | Joint names |
| `positions` | List[float] | yes | — | Must match `names` in order |

---

## Perception, containers, composites (reference only)

| Item | Needs | Notes |
|---|---|---|
| `DetectAction(technique, …)` | perception stack | Finds objects; not used here |
| `LookAtAction(target, camera=None)` | camera | Needs a head/camera |
| `OpenAction` / `CloseAction(object, arm, …)` | articulated container | Needs drawers/doors |
| `TransportAction(object, target_location, arm, …)` | mobile base | See fixed-base note above |
| `PickAndPlaceAction(…)` | — | Pick + place without base motion |
| `FaceAtAction` | mobile base | Turn chassis toward a pose |
| `MixingAction` / `CuttingAction` / `WipingAction` / `PouringAction` | tools, containers | Not in this cell |

---

## Key argument types

### GraspDescription

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `approach_direction` | ApproachDirection | yes | — | `FRONT` / `BACK` / `RIGHT` / `LEFT` |
| `vertical_alignment` | VerticalAlignment | yes | — | `NoAlignment` / `TOP` / `BOTTOM` |
| `end_effector` | EndEffector | yes | — | e.g. `robot.left_arm.end_effector` |
| `rotate_gripper` | bool | no | False | |
| `manipulation_offset` | float | no | 0.1 | Sets both the pre-pose pullback and the lift height |

Helpers on it: `grasp_orientation()`, `grasp_pose_sequence()`.

### Arms

`LEFT` (the 2FG7) / `RIGHT` (the 3FG15) / `BOTH`. Both arm views span the
same physical joints.

Thresholds everywhere: `position_threshold` is in metres, `orientation_threshold`
in radians — separate values because the units don't compare.

---

## This demo's plan

`demo.py:build_motion_plan` runs park → transit → orient → reach → close →
lift → transport → place → retract → park, all on `RIGHT` (the 3FG15).
`PickUpAction` / `PlaceAction` are bypassed on purpose: the pick targets a
raised grasp pose the stock pick hardcodes to the body origin, and the place
cannot read the grasp without model attachment or a pick node to borrow from.
The manual sequence uses `ReachAction(target_pose=raised grasp pose, …,
open_gripper_at_pre_pose=True)`, stall-tolerant close, a `+0.12` lift, and
transport / placing / retract poses from
`grasp.pose_sequence(place_pose, cylinder.root, reverse=True)`:

1. `ParkArmsAction(arm=Arms.RIGHT)`
2. Staged `MoveTCPWaypointsMotion` + `MoveToolCenterPointMotion` to above the
   cylinder in grasp orientation (translation and rotation kept separate so
   the wrist cannot wind up)
3. `ReachAction(target_pose=raised grasp pose, arm=Arms.RIGHT, grasp, object_designator=cylinder, open_gripper_at_pre_pose=True)` with the grasp aimed `RAISED_GRASP_OFFSET = 0.10` above the cylinder origin — tool point at the rim plane, pads wrapping the upper wall — because the claw hangs below the tool point and an origin-level grasp would put the pads near the table. Grasp is `GraspDescription(ApproachDirection.FRONT, VerticalAlignment.NoAlignment, robot.right_arm.end_effector, rotate_gripper=False)`: `NoAlignment` keeps the claw upright where `TOP` would lay it sideways; no roll needed on a round body
4. `MoveGripperMotion(CLOSE, RIGHT, tolerate_stall=True)`, lift `+0.12`, `PlaceAction(cylinder.root, place_pose, RIGHT)` at (`PLACE_XY`, `TABLE_TOP_Z + CYLINDER_HALF_HEIGHT`), `ParkArmsAction(RIGHT)`

Rules of thumb from running it:

* Prefer fewer, larger motions — every charted motion is a stall opportunity.
* Keep orientation changes out of loaded, close-quarters phases — the
  orientation servo is what winds the arm up.
* A closed gripper proves nothing — verify the cylinder is between the fingers
  before placing.
