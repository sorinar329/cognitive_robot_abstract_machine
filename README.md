# windturbine_model

A whole onshore wind turbine (IEA-3.4-130-RWT: 110 m hub height, 130 m rotor) as an
inspection environment for CRAM, shown in CRAMERA. The plan and task ladder are in
`docs/PLAN.md`; the reference data are in `references/iea34/`.

Unitree G1 tasks (T0 door, T1 tower base round, T2 nacelle walkway round, T3 bring the
tool, T4 crane hoist into the nacelle, T5 tower lift to the yaw deck, T6 flange bolt
check with tightness ratio and actions; the preload model is in `turbine/bolts.py`).
Run them in the full CRAM stack of the `cramera-port` checkout (`uv sync` there), with
ROS, `~/workspace/segmind_ws` (G1 model) and this package sourced:

    PY=~/workspace/cramera-port/.venv/bin/python
    $PY scripts/g1_inspection_round.py --zone ground --points tower_inspect_door        # T0
    $PY scripts/g1_inspection_round.py --zone ground --scenario outside_ground          # T1
    $PY scripts/g1_inspection_round.py --zone nacelle --scenario nacelle_all_faults     # T2
    $PY scripts/g1_bring_the_tool.py                                                    # T3 (scenario bring_the_tool)
    $PY scripts/g1_climb.py --route hoist                                              # T4: crane hoist into the nacelle
    $PY scripts/g1_climb.py --route lift                                               # T5: tower lift to the yaw deck
    $PY scripts/g1_tower_bolts.py                                                      # T6: flange bolt check -> reports/

Record a run for CRAMERA and tidy the bundle (robot naming, GLB materials, camera, and a walking
gait for the legs: in CRAM the G1's base slides, `scripts/add_gait.py` adds steps from the
recorded base motion, visual only):

    $PY -m cramera.onboard.demo scripts/g1_inspection_round.py --name windturbine_g1_t1 -- --zone ground --scenario outside_ground
    python3 scripts/finish_recording.py ~/.cramera/scenes/windturbine_g1_t1 --view g1_ground
    # T2: --name windturbine_g1_t2 -- --zone nacelle --scenario nacelle_all_faults, then --view g1_nacelle
    # T3: $PY -m cramera.onboard.demo scripts/g1_bring_the_tool.py --name windturbine_g1_t3, then --view g1_nacelle_wide
    # T4: $PY -m cramera.onboard.demo scripts/g1_climb.py --name windturbine_g1_t4 -- --route hoist, then --view g1_hoist
    # T5: ... --name windturbine_g1_t5 -- --route lift, then --view g1_tower

Watch a task live in CRAMERA while CRAM executes it (the open viewer tab attaches):

    ~/workspace/cramera-port/.venv/bin/cramera-live scripts/g1_bring_the_tool.py

Render a recording to video (used by the picture tour):

    blender -b --factory-startup -P scripts/render_recording.py -- ~/.cramera/scenes/windturbine_g1_t3 preview/video/g1_t3.mp4 --view g1_nacelle_video

The G1's D435 looks ~48 deg down and the waist pitches at most 0.52 rad, so it only
sees things at or below its camera height (1.27 m); `scripts/check_inspection_points.py`
flags the inspection points above that. Findings are still read from the scenario's
ground truth; recognising them in camera images is the next step.

Interior of a geared ~2 MW wind turbine nacelle as a URDF environment for CRAM
(`semantic_digital_twin`), laid out like `iai_maps/iai_kit_mobile_lab`:
`models/<part>/*.obj` meshes referenced via `package://windturbine_model/...`.

## Picture tour (phone)

https://claude.ai/artifact/GJymeF8rCoQM1yw57dzxfZ shows renders, CRAMERA screenshots and a
healthy-vs-faulty browser. It is rebuilt from `gallery/template.html` after every change:

    ~/cram/cram_venv/bin/python scripts/build_gallery.py   # renders, screenshots (viewer on :8711), page -> preview/site/

and then republished to the same URL.

## Viewing in CRAMERA

The nacelle is packaged like CRAMERA's `precision_lab` (branch `cramera-port` of
sunava/cognitive_robot_abstract_machine). Each part is also exported as GLB with PBR
materials. `scripts/build_cramera_bundle.py` writes one scene bundle per scenario to
`~/.cramera/scenes/windturbine[_<scenario>]`:
`scene.json`, `environment.urdf` (GLB visuals under `assets/`), `trajectory.json`
(initial joint state, e.g. an engaged rotor lock) and `semantics.json` (inspection
points, active faults, sensor signals). The nacelle cover visual is left out so you can
see inside (`--with-cover` keeps it); its collision boxes stay.

    ./view.sh --cramera                                    # healthy
    ./view.sh --cramera windturbine_after_maintenance

The viewer comes from `~/workspace/cramera-port` (its own `.venv`, `pip install -e cramera krrood`).
Movable joints appear under *Doors & drawers* in the scene panel.

## Layout

- `turbine/dims.py`: all dimensions, shared by Blender and the URDF generator
- `blender/parts/<part>.py`: geometry, mesh variants and fault overlays per part
- `turbine/parts/<part>.py`: links, collisions, inspection points, faults, sensor signals
- `scenarios/*.yaml`: scenarios (`faults: [ids]`, optional `extras`, `view` and `joint_states`, e.g. the open crane hatch and the lowered hook in `hoist_access`)

## Workflow (one part at a time)

1. Model the part in `blender/parts/<part>.py` (geometry in its link frame)
2. `./blender/build.sh <part>` → `models/<part>/*.obj` + `blender/parts/<part>.blend`
3. Describe links, inspection points and faults in `turbine/parts/<part>.py`, register it in `turbine/parts/__init__.py`
4. Generate:
   - `python3 scripts/generate_urdf.py`: healthy `urdf/windturbine.urdf` + `urdf/inspection_points.yaml`
   - `python3 scripts/generate_urdf.py --scenario scenarios/gearbox_seal_leak.yaml`
   - `python3 scripts/generate_urdf.py --random 2 --seed 7`
   - `python3 scripts/generate_urdf.py --list`
   Scenario output goes to `urdf/scenarios/<name>.urdf` + `<name>_ground_truth.yaml`
5. Check in CRAM: `~/cram/cram_venv/bin/python scripts/check_cram.py`; check the G1 can take every nacelle view:
   `python3 scripts/check_inspection_points.py`
6. Render overview + one close-up per inspection point:
   `blender -b --factory-startup -P scripts/render_preview.py -- <urdf> <out_dir>`

## Fault model

- **Variant links**: components whose look changes (sight glass, filter indicator,
  bolts, bushing) are their own fixed links with one mesh per state; the first state
  is healthy.
- **Overlay links**: faults that add something (oil streak, puddle) add extra links.
- **Inspection points**: fixed frames named `<part>_inspect_<what>` on the thing to look
  at. `urdf/inspection_points.yaml` also gives `view_from`, a direction for the camera.
- **Joint states**: faults can set joints (e.g. rotor lock left engaged). URDF has no
  initial state, so these go to `initial_joint_states` in the ground truth.
  `check_cram.py` and `render_preview.py` both apply them.
- **Ground truth**: which faults are active, where to see them, their severity, the
  initial joint states, and the non-visual sensor signals (temperature, vibration,
  filter pressure, ...).

## Frame convention

Frame `nacelle`: origin on the yaw axis at the top of the floor, X points upwind
(towards the hub), Z up. The drivetrain axis is at z = 1.5 m; the gearbox output is 0.3 m higher. The real 4–6° shaft
tilt is left out so the floor stays level for mobile robots.

## Reference layout (from research)

Nacelle envelope after the Vestas V90-2MW: 10.4 m long, 3.5 m wide. We use 3.4 m
interior height. Drivetrain uses three-point suspension, following NREL 5 MW:
rotor → main shaft on a main bearing → gearbox on two torque arms on the bedplate →
high-speed shaft with disc brake and coupling → generator. The yaw bearing and yaw
drives sit at the tower/nacelle interface. The controller, converter, hydraulics
and cooling are at the rear. A service crane runs under the roof.

Progress and the part list are tracked in `docs/PLAN.md`.

## Known limitations

- The generator end (fast shaft, brake, coupling, generator, rear frame) is still
  missing, so the gearbox output stub ends in the air.
- The torque arms and girders leave only ~0.37–0.44 m of walkway on each side. That is realistic for a
  2 MW nacelle, but too narrow for most mobile bases.
