## hsr-mujoco-kitchen: HSR in the IAI kitchen, simulated in MuJoCo

Goal (2026-10-05, user): a scene with the HSR in the kitchen that parses into MuJoCo;
the robot does nothing yet. Decisions: free base on a floor; the URDF axis bug fix
stays in this branch (no separate bug PR).

Done (uncommitted):
- URDF parser kept fractional joint axes as ints (map(int, ...)) -> tilted kitchen
  drawers got a zero axis and MuJoCo refused to compile. Fixed to float + test in
  test_urdf.py.
- experiments/hsr_kitchen_mujoco/demo.py: build_world() (kitchen via
  WorldSpecification + floor box + HSR on a 6-DoF connection at (0.3, 1.0)), main()
  runs the MuJoCo viewer. test/experiments_test/test_hsr_kitchen_mujoco.py (CI-gated):
  HSR present, every body in MuJoCo, base stays within 1 cm of its start pose.

Next: wide suites running; then report. Not done: commit/PR (not requested).
Known limits: OmniDrive is welded in MuJoCo, so the HSR uses a passive 6-DoF base;
no servos, so arm/head slump under gravity.
