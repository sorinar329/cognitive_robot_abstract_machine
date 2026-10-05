## tracy-montessori-demo: live scene graph + simulation control (cramera)

Goal: opt-in cramera panel showing the scene graph (bodies, pose, mass, friction),
editable at runtime, plus play/pause/stop for the stepped MuJoCo demo. Decisions
(2026-10-05, user): opt-in extension; edit poses + mass + friction; must also move
the board (a fixed body) -> "fixed" placement = fixed connections up to the root.

Done (uncommitted, nothing pushed):
1. SDT: MujocoSim body mass/inertia/friction + fixed-body pose setters;
   adapters/controlled_simulation.py (ControlledSimulation, Placement, SimulatedBody,
   BodyPose/Mass/FrictionChange); exceptions; SteppedSimulation protocol.
2. giskardpy: SteppedSimulationPacer typed with SteppedSimulation.
3. coraplex: attach_simulation on PlanVisualization/WorldVisualization/Plugin.
4. cramera: live/simulation_control.py, bridge.scene_graph, /simulation routes,
   bundle signature includes fixed origins, panels/scene_graph + core/scene-graph.js,
   README + core_scope updated.
5. demo: mujoco_demo wraps + attaches ControlledSimulation.
Verified: unit suites green; live headless demo driven over HTTP (pause freezes,
board move, mass/friction, piece move, refusals, stop ends plan).
Observations: pause during the 1 s settling step waits for it to end; unedited
baseline only gets cylinder + triangular prism into holes (cube/rect stay ~0.97 m).

Next: tuple->dataclass in _next_instruction, run format_docstrings, report to user.
Not done: visual browser check of the panel; commit (not requested).
