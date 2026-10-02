# Offshore wind farm demo

## Goal

A demo of event-driven wind farm control in CRAM. Three NREL 5 MW turbines on OC3
monopiles stand in a calm sea. The wind rises and stays up, the grid asks for power,
and the controller turns the turbines on: yaw into the wind, release the brake, pitch
the blades in, connect the generator. It then checks that power is produced. A storm
later makes it shut them down. CRAMERA shows the scene next to the detection and
control statecharts.

## Design decisions (2 October 2026)

- **The semDT is the world model and the only shared state.** The simulation writes
  wind, grid demand, joint states and power into it. The controller writes setpoints
  into it. Neither calls the other.
- **SegMind only detects and emits events.** Its detectors read the semDT and emit
  events through the EventLogger; they never command a turbine.
- **A separate WindFarmController builds the control statecharts** from Giskard
  MotionStatechartNodes. Events trigger, EQL queries over the semDT guard.
- **No OWL or Pellet.** The earlier OWL axiomatization (in the wind farm repo's
  `ontology/`) maps to SegMind detectors (thresholds with hold time and hysteresis),
  EQL queries (policy and constraints) and decisions with outcomes.
- **Decisions are semantic annotations in the semDT**, so the chain from event to
  effect can be queried with EQL, also from CRAMERA's EQL panel. A semantic
  annotation's hash counts only the bodies in its own fields, so Decision hashes by a
  unique name; otherwise every decision after the first would be dropped.
- **`wind_farm_control` is a standalone uv workspace package.** Its simulation modules
  import nothing from its control modules.
- **CRAMERA gets one statechart tab per named chart**: Detection and Control.
- **Time is simulated.** Hold times count control cycles, and Giskard's real-time
  factor compresses a 10-minute wind hold into seconds.
- **Outside only.** The turbines are drawn from primitives at NREL 5 MW scale, from
  the OC3 monopile deck: 20 m water depth, 6 m monopile up to 10 m above sea level,
  tower base at 10 m, tower top at 87.6 m.
- **The robot is out of the loop for now.**

The class diagram is the "Class diagram" tab of the wind farm brainstorm board
(claude.ai artifact Sy8vtC3Lm4KwvVnMWWTi1b).

## Found in the code

- SegMind `EventLogger.add_callback` replaces the callbacks already registered for a
  type and copies them only to direct subclasses (item segmind-event-callbacks).
- `DetectionEvent.timestamp` defaults to the wall clock; wind events need simulated
  time.
- `AbstractDetector.on_tick` only checks bodies on 6-DoF connections; wind detectors
  override it.
- CRAMERA's Bridge observes a single statechart (item cramera-statechart-tabs).
- `OpenFASTInputDeck.turbine_geometry` takes TowerHt as the tower height, which is
  only right onshore; offshore decks also set TowerBsHt (openfast-adapter).

## History

- 1 October 2026: OpenFAST adapter and WindTurbine annotation pushed as
  `openfast_integration`.
- 2 October 2026: OpenFAST stalled; controller design, OWL draft, then the switch to
  semDT, EQL and SegMind; class diagram agreed; plan created.
