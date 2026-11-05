# Coraplex Panda demo: diagnose a failure and recover

The IJCAI expo demo. A Franka Emika Panda stacks four cubes into a tower in MuJoCo.
Whenever an attempt fails, it diagnoses why against what successful attempts did, and
retries with the perfect parameters.

## Files

| file | what it is |
|---|---|
| `inference3b.py` | The demonstration: the scene, one attempt as a statechart, the diagnose-and-recover loop |
| `varied_failures.py` | The same demonstration with a realistic friction per cube: only some attempts fail, in different ways |
| `causal_diagnosis.py` | Judges a failed attempt's parameters against the trees fitted on successful attempts |
| `training/models/` | The joint probability trees of the pick-up's and the place's successful parameters |

## Running it

```bash
python inference3b.py --iterations 3
```

| option | meaning |
|---|---|
| `--iterations` | How often the tower is built from the cubes standing in a row (default 20) |
| `--seed` | Seed of the first attempts' parameters (default 42) |
| `--headless` | Step the physics without the MuJoCo window |
| `--live-port` | Port the run is published on for cramera (default 0: publishes nothing) |

Every iteration's attempts and diagnoses are appended to `inference_report.md`.

## Failures as they come

`inference3b.py` makes every cube's first attempt fail the same way: its friction is
always too low to lift it. `varied_failures.py` samples a realistic friction instead:
most cubes grip well, about 30 % are slippery (0.05 to 0.075) and fail at first. A
slippery cube either slides out of the fingers when lifted and stays where it stood, or
drops out of them right after and falls. Frictions from about 0.075 on, where a cube
twists in the fingers and lands tilted, are left out. The velocities alone never make
an attempt fail.

Both are separate programs; run either on its own:

```bash
python inference3b.py --iterations 3      # every cube fails once
python varied_failures.py --iterations 3  # some cubes fail, in different ways
```

## Watching it in cramera

Cramera is optional. Without it the run prints what went wrong (`[failure]`), what
segmind observed during the failed attempt (`[segmind]`), the causal diagnosis
(`[causal]`) and the recovery (`[recovery]`). To publish the run live through
cramera's bridge (`~/cramera`, packages `bridge`, `protocol` and `viewer` on the Python
path), give it a port, start the viewer and open the run:

```bash
python inference3b.py --live-port 8765
cramera --no-browser          # serves http://127.0.0.1:8711
# open http://127.0.0.1:8711/?live=http://127.0.0.1:8765
```

## What happens in one step

1. **First attempt.** The pick-up and place parameters are sampled from a range wider
   than the one successful attempts used. The cube's friction is always too low to
   lift it (0.02 to 0.05), so the cube slides out of the fingers and stays where it
   stood.
2. **Diagnosis.** The pick-up and the place are judged against the final height of the
   cube they stacked. Every parameter is scored by how well the successful attempts
   support its value, relative to the value they support most. A parameter supported
   a quarter as well or less is anomalous, and the least supported one is the primary
   cause.
3. **Recovery.** Every later attempt uses the perfect value of every parameter, which is
   the middle of the region the successful attempts support most. The friction's
   perfect value is 1.31.

A lower cube that an attempt knocks off is re-stacked with the perfect parameters.

## Notes

- **Why friction makes the first attempt fail.** On this branch's controller, a
  velocity parameter only caps a motion that already moves at 0.2 m/s or less, so
  sampled velocities alone never make an attempt fail. The cube's friction is made to
  count in MuJoCo by giving the cube's geoms the higher contact priority. Otherwise the
  finger pads' friction would win.
- **Why the slippery range stops at 0.05.** At 0.08 the fingers carry the cube part of
  the way and drop it somewhere unpredictable, sometimes out of comfortable reach.
- **The models.** These are the trees the IJCAI demo was trained with, converted to the
  current JSON format of `probabilistic_model`. Trees fitted again with the current
  `probabilistic_model` from the same data fail the causal circuit's support
  determinism check.
