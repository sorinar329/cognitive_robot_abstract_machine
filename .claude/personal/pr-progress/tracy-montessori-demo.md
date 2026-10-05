## tracy-montessori-demo: live scene graph + simulation control (cramera)

Goal: opt-in cramera panel showing the scene graph (bodies, pose, mass, friction),
editable at runtime (free-body pose, mass, friction), plus play/pause/stop for the
stepped MuJoCo demo. Decisions (2026-10-05, user): opt-in extension; edit poses +
mass + friction.

Plan:
1. SDT: MujocoSim body mass/friction read+write; ControlledSimulation (pause gate in
   step_simulation, edit queue applied on the stepping thread, stop -> raises).
2. giskardpy: SteppedSimulationPacer accepts anything that steps (protocol).
3. coraplex: WorldVisualization/PlanVisualization.attach_simulation (no-op default).
4. cramera: bridge.simulation, GET /simulation, POST /simulation/{pause,resume,stop,edit},
   scene-graph panel (hidden unless a simulation is attached).
5. demo: mujoco_demo wraps MujocoSim in ControlledSimulation and attaches it.

Done: design. Next: TDD step 1.
