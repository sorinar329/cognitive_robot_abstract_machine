## hsr-mujoco-kitchen: HSR carries milk + cereal between the kitchen tables in MuJoCo

Plan approved 2026-10-05 (plan file ~/.claude/plans/woolly-exploring-frost.md). Branch
rebuilt on sorin/fera_exchange (no upstream set - do NOT push to fera_exchange).
User decisions: full iai_kitchen xacro (two back tables); objects Milk + Cereal;
servos needed; navigation not the focus -> teleported (kinematic) base is fine.

Steps:
1. [done] branch on fera_exchange, carried URDF axis fix + parse-only scene + tests.
2. [ ] kinematic base: welded OmniDrive's child body follows the world drive pose in
   MuJoCo (set_fixed_body_pose in world->sim sync). Test first.
3. [ ] HSR servos in robots/hsrb.py (arm, neck, gripper, torso mimic) + gravity comp;
   tuned gains flagged for review. Tests: hold park pose, track a step, grasp+lift.
4. [ ] demo: xacro kitchen + HSR via RobotSpecification + Milk/Cereal on table_area_main;
   plan navigate/pick/navigate/place x2; MuJoCo stepped + ControlledSimulation + cramera.
   Test: both objects end on the jokkmokk table.
5. [ ] record in cramera, republish artifact Vd6nG1PUmaLhRpJtYD5cDL, update todo tab 17.
