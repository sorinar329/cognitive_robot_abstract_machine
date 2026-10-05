# CRAMERA

CRAMERA shows CRAM worlds and plan execution in a browser. Its three panels combine
a 3D scene, EQL questions and execution graphs. You can follow a running demo or
inspect a saved recording.

## Install in the CRAM workspace

Use Python 3.12 and follow the [workspace installation instructions](../README.md),
including the ROS setup needed by live demos. From the repository root:

```bash
uv sync --extra dev --active
cramera
```

The viewer is served at `http://localhost:8711`. All browser libraries are packaged
locally. Their versions, checksums and licenses are documented in
[the vendor directory](src/cramera/web/vendor/README.md).

## Follow a live demo

In another terminal using the same environment:

```bash
cramera-live path/to/demo.py
```

The wrapper selects CRAMERA through CRAM's visualization backend. The live bridge
serves the world's geometry and state on port 8765. Open the viewer and choose Live
to follow the robot, objects, plan progress and motion statecharts. Unless a demo
opts in to simulation control (below), the viewer observes execution; it never offers
robot or plan editing controls.

### Control a stepped simulation

A demo that steps a MuJoCo simulation from its control loop can opt in to being
controlled from the viewer. It wraps the simulation in a `ControlledSimulation`,
paces its plan with it and attaches it to its visualization:

```python
controlled = ControlledSimulation(simulation)
visualization.attach_simulation(controlled)
GiskardExecutable.simulation_pacer = SteppedSimulationPacer(controlled)
```

The *Scene graph · simulation* panel then lists every simulated body with its pose,
mass and friction, and offers pause, play and stop. A pause holds the robot and the
physics together at the next control step. Loose bodies (on a free connection) and
fixed bodies (held by fixed connections up to the world's root) can be moved; mass
and friction can be changed for any body MuJoCo gives them. Changes are made between
two physics steps, right away while paused. Stop ends the plan with a
`SimulationStoppedError`. A demo that attaches no simulation shows the panel as a
one-line note, and every control request is refused.

MuJoCo resolves a contact with the larger friction of the two touching surfaces, so
lowering one body's friction below that of what it touches has no effect.

Use one live session per `CRAMERA_DATA` directory. Separate simultaneous sessions
need separate data directories and ports.

The scene supports orbit, pan, zoom, robot following and click-to-inspect. The graph
panel displays the plan, statecharts, robot kinematics and transforms. EQL results
can highlight entities in the scene and replay a recorded time interval.

The EQL editor executes trusted local Python statements against the loaded world.
The Python server listens only on loopback and rejects requests from remote
origins. Run queries you trust in the same local environment as the viewer.

The 3D panel requires WebGL. If the browser cannot create a WebGL context, open the
viewer address in a browser with WebGL enabled; EQL and graph inspection remain
available.

## Record and replay

Live capture starts with the visualization backend. Stop the recording in the
viewer, choose a name and save it. A demo shutdown also finalizes its capture, so
the viewer can save it after the demo has exited. The save dialog can trim the
recording to a selected frame range; discard removes an unsaved capture.

Playback controls provide play/pause, seeking and speed selection. Recordings carry
model descriptions, resolved mesh assets, joint and object trajectories, and the
captured inspection data. Their files can be served by an ordinary static HTTP
server for 3D playback; EQL and the knowledge panels use the Python server.

Use `?scene=<name>` to select a recording explicitly. The scene picker combines
local recordings with an optional shared bundle directory:

| Variable | Purpose | Default |
| --- | --- | --- |
| `CRAMERA_DATA` | Writable capture and cache directory | `~/.cramera` |
| `CRAMERA_SCENES` | Optional directory of existing scene bundles | Local recordings |
| `CRAMERA_ARCHITECTURE` | Repository inspected by the architecture graph | Current CRAM checkout |

Scene bundles and robot-model assets are data, and are not included in this Python
package. New recordings are written under `CRAMERA_DATA/scenes`.

## Package structure

The frontend mounts independent panels through `web/core/registry.js` and an event
bus. `web/config.js` chooses the layout. The Python `live` package observes existing
world and plan callbacks; `knowledge` answers queries and builds inspection views.
The visualization provider is registered through the `coraplex.visualizations`
entry-point group.

See [core scope](docs/core_scope.md) for the boundaries of this package.

## Tests

```bash
pytest test/cramera_test
```

The browser regressions run through pytest and require Node.js. Python coverage is
measured over the complete `cramera` package; the repository requires at least 85%
coverage for added code. Browser checks cover asset availability, panel contracts,
query rendering, playback helpers and graph behavior.

CRAMERA is licensed under [GPL-3.0-only](LICENSE). Bundled browser dependencies
retain their own licenses.
