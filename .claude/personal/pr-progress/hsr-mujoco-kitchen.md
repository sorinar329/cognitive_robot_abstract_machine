## hsr-mujoco-kitchen: HSR carries milk + cereal between the kitchen tables in MuJoCo

Plan approved 2026-10-05 (~/.claude/plans/woolly-exploring-frost.md). Branch on
sorin/fera_exchange, no upstream (do NOT push to fera_exchange). Nothing committed.
User decisions: full iai_kitchen xacro (two back tables); Milk + Cereal; servos;
navigation not the focus -> teleported base; 2026-10-06: keep teleport, no base motion
while grasping, carry via attachment (MuJoCo weld) - chosen over a servoed planar base.

Done:
1. branch + URDF axis fix (test_urdf) + scene tests.
2. kinematic base: MujocoSynchronizer.place_welded_body; drive bodies follow the world
   drive pose (test_multi_sim driven_box_world test). set_fixed_body_pose delegates.
3. HSR servos (robots/hsrb.py: HSRBJointDrive table, ServoedHSRBPart mixin, base
   passive-joint armature 0.01 - RK4 diverged on light damped wheels without it).
   OPEN_HAND_ANGLE 0.3 -> 1.2 rad (0.3 gave a 4.6 cm gap < 6 cm carton).
   test_hsr_mujoco.py: 7 tests incl. level-hand carton grasp+lift.
4. weld fastening: MujocoEquality.active, MujocoSim.fasten/unfasten (3 tests).
   demo: elliptic cone/impratio 10/noslip 10, condim 4 on objects,
   full_body_controlled=False, hand_offset from arm_flex link (0.078 m),
   HoldWhatTheHandGrips plan callback toggles the weld. Kinematic dry run OK.
Findings: rigid contacts (montessori style) launch the milk with the HSR hand; squeeze
limit 10 vs 100 made no difference -> kept URDF 100.

5. placement fixed (2026-10-06): weld eq_data[10] torque scale was 0 -> weld held
   position only; fasten now sets it to 1 (test: fastened body turns with holder).
   Teleported base dragged the welded object after it -> fingers locked it 2.4 cm off
   and 5 deg leaning, toppled on release. Synchronizer now owns fasten/unfasten and puts
   fastened free bodies at their weld pose whenever a drive moves (teleport test).
   Full run: milk + cereal upright at goal; demo test + HSR + urdf tests pass (25).

6. wide suites: 4225 passed; failures are env (DAiSy, gazebo warehouse, PR2, ROS command
   publishing), xdist flakes (coraplex pick-up, cramera: pass serially), and
   test_a_geom_carries_its_masks_under_mujocos_names (pre-existing on fera_exchange: condim).
7. recorded via CORAPLEX_VISUALIZATION=cramera; bundle downsampled 1/6 (fps 18.21 = 2x),
   glitch frames + pick markers fixed; artifact Vd6nG1PUmaLhRpJtYD5cDL v2 published;
   todo tab 17 ticked + 6 Oct note. Tally restored, MUJOCO_LOG.TXT + test.srdf removed.

Next: user decides on commit (nothing committed/pushed).
Open: finger spring stiffness 10 lets fingertips fold (weld masks it); Giskard pitches
the hand 16.7 deg during the lift (harmless with the weld).
