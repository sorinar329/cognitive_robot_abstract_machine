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

Done also: SteppingInstruction dataclass; black run (docformatter not installed in
venv, so format_docstrings.py could not run). Affected suites: 1059 passed.
Pre-existing failures (not ours): missing ROS pkgs (ur_robot_driver, aws warehouse);
test_simulator_property_dict expects no 'condim' since commit 487f3fd651.

Next: user to try the panel in the browser (not visually checked yet); commit when
asked.
