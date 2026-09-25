# Tower flange bolt check (tower_bolt_check)

Tightness ratio = mean estimated preload of the bolts the G1 could read, from the torque-marking offsets (turbine/bolts.py). Readings come from the scenario ground truth (oracle), visibility is computed in the CRAM world.

| Flange | Bolts | Read | Tightness ratio | Worst bolt | Verdict |
|---|---|---|---|---|---|
| 1 (M42) | 132 | 118 | 100.0% | 100% | no action |
| 2 (M42) | 132 | 120 | 98.6% | 0% | re-tension the flange |
| 3 (M36) | 100 | 69 | 99.8% | 88% | re-tighten single bolts |

## Flange 1: no action
- 14 bolts were not readable for the robot (behind the lift or at a flat angle): a technician checks the markings of bolts 59-72 from the platform.
- Tension check on a random 10 % sample (14 bolts) at the next service, since preload lost without turning leaves the markings aligned.

## Flange 2: re-tension the flange
Bolts below 100 %: 20: 76%, 21: 62%, 22: 0%
- Re-tension all 132 bolts of flange 2: M42 10.9, preload 784 kN (about 4281 Nm with a hydraulic torque wrench). Work in a cross pattern, in two passes (50 %, then 100 %).
- Replace bolt, nut and washers at position 22: a bolt that ran loose under the tower's load cycles may be fatigue-damaged. Check the two neighbours on each side and the flange gap before re-tensioning.
- 12 bolts were not readable for the robot (behind the lift or at a flat angle): a technician checks the markings of bolts 60-71 from the platform.
- Afterwards: draw new torque markings and record the values in the turbine log.
- (ground truth check: turned marking [60] not readable for the robot, covered by the technician's list above)

## Flange 3: re-tighten single bolts
Bolts below 100 %: 75: 88%
- Re-tighten bolt 75 to 2676 Nm; the flange as a whole is within limits.
- 31 bolts were not readable for the robot (behind the lift or at a flat angle): a technician checks the markings of bolts 35-65 from the platform.
- Tension check on a random 10 % sample (10 bolts) at the next service, since preload lost without turning leaves the markings aligned.
- Afterwards: draw new torque markings and record the values in the turbine log.
