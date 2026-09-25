# Plan: the whole wind turbine as an inspection environment

Goal: model a complete onshore wind turbine, from the foundation to the blade tips, as a
CRAM world that CRAMERA shows. Every part robots can reach has inspection points and
faults that can be switched on.

## 1. Reference documents

| Role | Document | What we take from it |
|---|---|---|
| **Geometry (main reference)** | IEA Wind TCP Task 37, *WP2.1 Reference Wind Turbines*, NREL/TP-5000-73492 (2019), land-based **IEA-3.4-130-RWT**; model data: github.com/IEAWindSystems/IEA-3.4-130-RWT | Overall dimensions, drivetrain layout and masses, tower, hub, blade shape |
| Inspection scope | BWE Sachverständigenbeirat, *Grundsätze für die Wiederkehrende Prüfung von Windenergieanlagen* (update of 15 Apr 2025) | What a periodic inspection covers: foundation, tower, nacelle, drivetrain, rotor blades, safety equipment, lightning protection |
| Inspection scope (international) | DNVGL-ST-0262 *Lifetime extension of wind turbines*, Appendix B (inspection list) | Cross-check of the component list |
| Fault priorities | Carroll et al. 2015 (failure rates of offshore turbines); DOE gearbox reliability database | Which faults are frequent or costly, and so modelled first |

Why the IEA 3.4 MW turbine: it is openly documented, and it is a **geared** onshore
design with a DFIG generator. That is the same kind of drivetrain we already started,
and the kind that dominates today's inspection work. It comes with exact numbers:

- Hub height 110 m, tower 108 m. The tower diameter shrinks from 5.99 m at the base
  to 3.0 m at the top; the wall is 57 mm thick at the bottom and 22–27 mm at the top.
- Rotor 130 m across, 3 blades, 63 m each (blade mass 16.4 t). Chord 2.6 m at the root
  and 4.3 m at its widest; the airfoil coordinates are in the YAML file.
- Hub 4.0 m across, cone angle 3°, shaft tilt 5°, overhang 5.0 m, tower top to
  hub height 2.0 m.
- Drivetrain: two main bearings (a CARB bearing and a spherical roller bearing), a
  three-stage gearbox (two planetary stages plus one parallel stage, ratio 1:97), a
  fast shaft 1.5 m long, a DFIG generator at 1200 rpm, and a transformer up in the
  nacelle.
- Masses (Table 5 of the report): bedplate 61 t, gearbox 41 t, main shaft 26.6 t,
  generator 16.9 t, transformer 10.4 t, cover 9.2 t, yaw system 4.5 t; nacelle
  192 t in total.

Not in the reference: detailed nacelle interior dimensions, tower internals, and cabinets.
For those we estimate from typical 3–3.5 MW turbines and mark each value as an estimate
in `dims`.

## 2. What changes compared to today

- **Rescale to the IEA 3.4 MW turbine.** The current nacelle is sized after a 2 MW
  Vestas V90 with a single main bearing. It moves to the IEA 3.4 MW layout with two
  main bearings. The fault system, scenarios, inspection points, CRAMERA bundles and
  CRAM check all stay.
- **`turbine/dims.py` becomes a package**, one module per assembly (`tower.py`,
  `nacelle.py`, `hub.py`, `blade.py`, ...). Each estimated value is marked as an estimate.
- **Zones:** the turbine is split into areas a robot can be in. Each zone gets its own
  CRAMERA bundle, with camera presets, and there is also one bundle for the whole
  turbine.
- **Level of detail:** full detail where a robot can go (interiors). The outside
  (blade surfaces, tower shell) uses simpler meshes, with fault overlays detailed
  enough for drone or camera inspection.

## 3. Kinematic structure

```
world
└─ foundation                (fixed)   cellar, anchor bolt cage, cable entries
   └─ tower_section_1..4     (fixed)   flanges, platforms, ladder, lift rail
      ├─ service_lift        (prismatic Z along the tower)
      ├─ tower_door, platform hatches              (revolute)
      └─ yaw_bearing ─ nacelle (continuous Z = yaw)
         ├─ bedplate, main bearings, gearbox, generator, cabinets, crane ...
         └─ main_shaft (continuous X, tilted 5°)
            └─ hub
               ├─ pitch_bearing_1..3 (revolute = blade pitch)
               │  └─ blade_1..3      root bulkhead + manhole cover (revolute)
               └─ hub hatches, spinner
```

Joints CRAM can use: yaw, rotor, pitch ×3, rotor lock, service lift, crane trolley and hook,
and all doors, hatches and cabinet doors. Initial positions come from `trajectory.json`,
as we already do for the rotor lock.

## 4. Zones, parts and their faults

Faults come from the BWE inspection scope (cracks, corrosion, bolted joints, leaks, wear,
lubrication, play, fluid levels, safety equipment, lightning protection) and are ranked
by the reliability data.

| Zone | Parts | Faults (first set) |
|---|---|---|
| **A Foundation / cellar** | concrete slab and plinth, anchor bolt cage, grout joint, cable ducts, earthing | concrete cracks, broken-out grout, corroded anchor nuts, water in the cellar |
| **B Tower base** | entrance door and stairs, switchgear, converter and controller cabinets (doors), cable trays, lighting, fire extinguisher | open or hot cabinet, burn marks, water ingress, missing extinguisher |
| **C Tower** | 4 steel sections, bolted flanges with torque markings, platforms and hatches, ladder with fall arrest, service lift, cable loop, lights | loose or missing flange bolt, flange gap, weld crack, corrosion, damaged fall-arrest rail, chafed cable |
| **D Yaw deck** | yaw bearing with gear teeth, 6–8 yaw drives, yaw brakes, slip ring, cable twist counter | broken or dry teeth, brake dust, oil leak at a drive, loose bearing bolts |
| **E Nacelle drivetrain** | bedplate, 2 main bearings, main shaft, shrink disc, rotor lock, gearbox (2 planetary + 1 parallel), fast shaft, disc brake, coupling, generator (DFIG), oil and cooling systems, hydraulic unit | *existing 10 faults* plus worn brake pads, blue-heat-marked disc, cracked coupling disc, carbon dust from generator brushes, hot generator terminal box, hydraulic leak |
| **F Nacelle systems** | transformer room, cabinets, cable trays, service crane (trolley and hook), roof hatch, floor hatches, rescue kit, weather mast on the roof | tripped breaker or error light, burn marks, crane hook without safety latch, missing rescue kit, iced or damaged anemometer |
| **G Hub** | hub casting, spinner, 3 pitch bearings, pitch drives and gear teeth, pitch cabinets and batteries, blade root bolts, hub hatch | loose root bolt, broken pitch gear tooth, swollen or leaking battery, grease leak at a pitch bearing |
| **H Blade interior** (first ~15 m) | root bulkhead with manhole, shear webs, bond lines, lightning down-conductor | cracked bond line, loose debris, detached lightning cable |
| **I Outside** | blade surfaces (lofted from the IEA airfoils), lightning receptors, tower paint, nacelle cover, aviation lights | leading-edge erosion, lightning strike damage, trailing-edge crack, paint damage or corrosion, broken aviation light |

Each fault defines, as now: the parts it changes or adds, an inspection point with its
viewing direction and distance, a severity, the sensors that can see it (RGB, thermal,
acoustic, drone), non-visual sensor values, and joint positions it sets.

## 5. Robot: Unitree G1 (the CRAM model)

`semantic_digital_twin.robots.unitree_g1.UnitreeG1` (model files: `iai_offis_g1_description`
in segmind_ws), already used by `coraplex/demos/coraplex_unitree_g1_warehouse_demo`
(Navigate, PickUp, Place, ParkArms).

- **Base:** an omni drive on the pelvis. Walking is abstracted as moving across the
  floor, so the robot needs **flat floor and no stairs**. The pelvis sits 0.79 m above
  the floor.
- **Camera:** a D435 fixed to the torso (field of view 57°×43°) at 1.27 m, tilted about
  48° down (the optical axis is the link's x axis; CRAM's annotation says z, which the
  scripts correct at runtime). With the waist pitch limit of 0.52 rad the G1 only sees
  things at or below its camera height: rotor lock, filter indicator, brake, terminal
  box and converter light are out of its view.
- **Size:** about 1.3 m tall and about 0.45 m across the shoulders. That needs
  **walkways ≥ 0.65 m** and hatches or doors ≥ 0.7 m wide.
- **Arms:** two 7-DoF arms with 3-finger hands, for later tasks (hatches, buttons,
  rotor lock handwheel).

## 6. Robot tasks, simplest first

| # | Task | CRAM actions | Done when |
|---|---|---|---|
| **T0** | **Look at one inspection point**: walk from the start to the standpoint in front of the tower door and aim the camera at it | `NavigateAction`, `LookAtAction` | the target lies inside the D435 field of view (checked geometrically in the world) and CRAMERA shows the run |
| T1 | Walk around the tower base on flat ground: visit every ground-level inspection point and report its state | loop of T0 + reading the state | the report matches the scenario's ground truth (first read from the ground truth, later from rendered camera images) |
| T2 | Nacelle: walk along the drivetrain and check sight glass, grease collector, shrink disc, input seal, torque arm bushing, slip ring | T1 in tight space | correct report on the 0.69 m walkway (straight base moves; the route planner needs more room) |
| ✅ T3 | Assist a technician: find the loose shrink-disc bolt by its torque marking, fetch the torque tool case from the rack, set it on the tray next to the technician | inspection + `PickUpAction`, `PlaceAction` | bolt found; case on the tray (`scripts/g1_bring_the_tool.py`, recorded as `windturbine_g1_t3`) |
| ✅ T4 | Reach the nacelle: walk from the tower door to the lifting platform the service crane lowered through the rear floor hatch, get hoisted 108 m, step off, check the controller cabinet, walk onto the walkway | odom re-parented to the platform + joint motion on `crane_hook_joint`, straight base moves, `LookAtAction` | on the nacelle floor (0 mm step) and walkway (`scripts/g1_climb.py --route hoist`, scenario `hoist_access`, recorded as `windturbine_g1_t4`) |
| ✅ T5 | Up the tower: in through the door, into the service lift car, ride to the yaw deck, check the cable loop | odom re-parented to the car + joint motion on `service_lift_joint` | at the yaw deck; chafed cable found (`scripts/g1_climb.py --route lift`, scenario `tower_climb`, recorded as `windturbine_g1_t5`) |
| T6 | Interact: open the floor hatch, press a button, empty the grease collector | pull, `MoveJointsMotion`, pick and place | joint state changed / part swapped |

Standpoints are computed from each inspection point's `view_from` and `distance`:
project the point onto the floor, set the camera height to 1.27–1.60 m, and turn to
face the target.

## 7. Steps (order agreed: outside first, then the nacelle in full detail)

0. ✅ **Rebaseline.** Put the turbine origin at ground level. Build the kinematic tree
   from the foundation to the blades. Split `dims` by assembly.
1. ✅ **Outside** (zone I plus the visible part of the foundation):
   - a ground area,
   - the foundation plinth,
   - a tower in 4 tapered sections with flanges and a door (at ground level since step 5),
   - the yaw bearing ring,
   - the nacelle cover at IEA size with a weather mast and aviation light,
   - hub and spinner,
   - 3 blades lofted from the IEA chord, twist and airfoil data.

   Joints: yaw, rotor, pitch ×3, tower door. Faults: blade leading-edge erosion,
   lightning strike, trailing-edge crack, tower paint damage and corrosion, plinth
   crack, broken-out grout, damaged anemometer, broken aviation light.
   ✅ **Robot task T0** on the flat ground around the tower (`scripts/g1_inspection_round.py`).
2. ✅ **Nacelle in full detail** (zones E + F), rebuilt to IEA dimensions with 0.69 m
   walkways: two main bearings, a three-stage gearbox, brake, coupling, generator,
   transformer, cabinets, crane, hatches. The existing faults move over (37 faults in total now; every nacelle view
   is reachable from a walkway at G1 camera height, `scripts/check_inspection_points.py`).
   ✅ **Robot tasks T1 and T2** run in CRAM and are recorded for CRAMERA
   (`windturbine_g1_t1`, `windturbine_g1_t2`): 6/6 points seen each.
3. **Hub** (zone G).
4. **Yaw deck** (zone D).
5. ✅ **Tower inside** (zone C) and **access for the G1**: rest platforms under the
   flanges, a yaw deck 2.6 m under the tower top, ladders with fall-arrest rail, cables,
   lamps, ground controller, and a service lift (prismatic joint, 105 m). The door is at
   ground level (plinth flush with the ground, no stairs). The nacelle opening above the
   yaw bearing is blocked by the drivetrain, so the G1 is lifted into the nacelle from
   outside: the service crane lowers a lifting platform through a rear floor hatch
   (`crane_hatch`, hook travel down to the ground). Fault: chafed cable loop.
   ✅ **Robot tasks T4 and T5.**
6. **Foundation cellar** (zone A).
7. **Blade interior** (zone H).
8. **Integration:**
   - a `WindTurbineEnvironment` loader (following `LaboratoryEnvironment`),
   - random fault campaigns,
   - inspection routes,
   - pytest tests.

Every step ends the same way: render close-ups comparing healthy and faulty, pass the
CRAM check, build and look at the CRAMERA bundle, and get your review.

Rough size: about 60–80 parts, about 45 faults, about 70 inspection points.

## 8. Decisions

- **Own repository:** this project stays in its own git repository, not a package
  inside `cramera-port`. Scenes reach CRAMERA as bundles in `~/.cramera/scenes`.
- **Zones were separate start areas at first** (T0–T3 start inside their zone). Since
  step 5 the G1 gets from the ground to the yaw deck by the tower lift (T5) and from the
  ground into the nacelle by the crane hoist (T4). The lift cannot reach the nacelle:
  the opening in the yaw bearing is under the drivetrain.
- **Lift and hoist speed** are 4 m/s in the simulation (real: about 0.3 m/s), so the
  recordings stay short.
