# Panda demo with cramera: where it stands and how to continue

State as of 9 Nov 2026.

## Branches

| repository | fork | branch | what it holds |
|---|---|---|---|
| cram | `sorinar329/cognitive_robot_abstract_machine` (remote `origin`) | `panda-demo` | The demo (`inference3b.py`, `varied_failures.py`), publishing live to cramera |
| cramera | `sorinar329/cramera` | `panda-demo-stream` | The bridge and viewer changes the stream needs |

Never push to `cram2` (upstream). The cramera remote is https and has no credentials;
push it with `git push git@github.com:sorinar329/cramera.git panda-demo-stream`.

## Environment

```bash
workon cram-env
```

`cram-env` has cram's packages (coraplex, giskardpy, segmind, semantic_digital_twin)
and cramera's `protocol`, `bridge` and `viewer` installed editable from
`~/workspace`. `piqp` had to be upgraded to 0.7.0 (0.6.3 fails with
`eps_primal_inf`).

## Running it

From this directory, in two terminals:

```bash
python inference3b.py --live-port 8765        # or varied_failures.py
cramera --no-browser                          # serves http://127.0.0.1:8711
```

Open http://127.0.0.1:8711/?live=http://127.0.0.1:8765. Add `--headless` to the demo
to drop the MuJoCo window. Without `--live-port` the demo runs on its own and only
prints.

## What works

- The demo stacks the cubes, diagnoses each failed attempt and recovers. One attempt
  is park, pick up, place, park (the park while carrying was removed).
- The apartment is drawn as a static backdrop: only its visuals are parsed (collisions
  stripped in `VisualDescription`), and its meshes are exported once and hard-linked
  into every live bundle.
- Speed: one tower takes about 89 s live with the viewer open. Without cramera it
  took about 100–110 s before the carrying park was removed.
- The **Segmind** tab lists the events of cramera's Segmind (`detect_events=True`; the
  demo reuses that Segmind instead of starting its own).
- The **Diagnosis** tab shows each failed step: the failure, what Segmind saw, every
  parameter with its verdict and corrected value. It stays empty until the first
  failure.

## Open items

1. **Remaining overhead with the viewer open (~20 %).** The HTTP server runs in the
   demo's process, so serving the viewer's polls competes for the GIL with the
   simulation. Serving from a separate process would remove it.
2. **Recordings can't be saved from `cram-env`.** `torch` is missing there, so saving a
   run fails. Installing the CPU-only build of torch should fix it.
3. **Missing Panda material.** The viewer reports one file it can't load:
   `material_0.mtl` from `iai_franka_panda_description` in `~/ros_ws`. This was already
   missing before these branches; one part of the Panda is drawn without its material.
4. **cramera tests that fail because cram2 changed upstream** (not caused by this
   branch):
   - `test/live/test_names.py::test_every_action_the_workspace_offers_is_pinned`:
     new cram actions to pin in `cramera_bridge/names.py`
   - `test/live/test_recording.py` and
     `test_detected_events.py::test_a_saved_recording_keeps_the_events_of_its_frames`:
     need `torch` (item 2)
   - `test/live/test_perception_images.py::test_a_body_above_and_left_of_the_camera_is_seen_in_the_top_left`
5. **Untracked local files** in the cram checkout were left out of the commits on
   purpose: `franka/`, `render_video.py`, `inference_report.md`,
   `varied_failures_report.md`, and others at the repository root. Decide whether any
   of them belong on the branch.

## Rules to keep in mind

- cram (`AGENTS.md`): no assistant `Co-Authored-By` trailer, end commit messages with
  "Made with AI assistance", format with `krrood format`, and regenerate the ORM
  interfaces with `scripts/regenerate_all_orm.py` after changing data classes.
- cramera (`AGENTS.md`): CRAM imports only in `packages/bridge`; pin new robots and
  actions in `names.py`.
