# Perception for the Tracy Montessori demo: where it stands

Work in progress, paused 2026-11-12. Hacky on purpose; not meant for `main`.

## 1. `perception_demo.py` (works)

Colour-only perception, no RoboKudo. It connects to the running Giskard and stands the
scene as `real_demo.py` does. Then it takes one colour frame and finds the pieces by
colour (yellow: rectangle/triangle, cyan: cube/cylinder). It projects each piece onto
the plane of the pieces' top faces (table 0.88 + 0.03), using the camera pose from TF
(`map` <- `camera_color_optical_frame`). Each piece's model is moved to the perceived
x,y; z keeps the scene's height, including the offset. It writes `perception_seen.png`
with the pieces marked.

Measured on the live camera, without moving anything (hardcoded vs perceived):

| piece | hardcoded | perceived |
|---|---|---|
| cube | (0.79, 0.20) | (0.783, 0.193) |
| cylinder | (0.79, 0.30) | (0.785, 0.296) |
| rectangle | (0.79, 0.40) | (0.782, 0.384) |
| triangle | (0.79, 0.50) | (0.787, 0.486) |

Not yet run end to end with Giskard: placing the pieces and seeing them move in RViz is
still to be checked. It assumes TF `map` is the root of Giskard's world (`map2`).

## 2. RoboKudo pipeline (started, not finished)

Files:
- `robokudo/src/robokudo/descriptors/analysis_engines/tracy_montessori.py`, the engine.
  It runs query → Orbbec reader → crop to the table (in `map`) → plane (threshold 8 mm)
  → clusters (min 100 points, eps 2 cm) → `ShapeEstimatorAnnotator` →
  `MontessoriPieceAnnotator` → query reply.
- `robokudo/src/robokudo/annotators/montessori_pieces.py`, which names each fitted
  cluster by its median colour and shape, and adds a pose at its centre
  (`Classification.classname` becomes the reply's `type`).
- `robokudo/src/robokudo/io/camera_model_adapters.py` drops the zero coefficients the
  Orbbec pads `plumb_bob` with (8 instead of 5). The twin's camera model refused the
  calibration without this.

Run it, then query it:

```
python -m robokudo.scripts.main _ae=tracy_montessori _headless _no3d
ros2 action send_goal /robokudo/query robokudo_msgs/action/Query "{}"
```

(`ros2 run robokudo` does not work: robokudo is not installed as a ROS package here.
The `cv_bridge`/NumPy 2 ImportError printed at start-up is harmless; the pipeline still
runs.)

What the first query showed:
- The plane and clusters take about 2 s, but `ShapeEstimatorAnnotator` takes minutes:
  about 50 s on the first cluster, about 70 s on another. The query did not finish
  within 2 minutes. Big clusters (the board, extents 0.22 x 0.09) probably dominate.
  Next step: drop clusters larger than a piece *before* shape fitting (filter the
  hypotheses, tighten the crop, or cap the cluster size), and lower the estimator's
  multi-start settings (`cylinder_max_initializations`, `cylinder_consensus_trials`).
- The fits are too flat: extents like [0.034, 0.016, 0.010] and
  [0.031, 0.029, 0.007] for pieces 30 mm tall. Only the top of each piece survives
  above the table plane, so the heights (and so the rectangle/triangle elongation test,
  which uses all three sides) are unreliable. Next step: classify from the footprint
  (the two largest extents) and colour, not from the height.
- `MontessoriPieceAnnotator` itself has not run yet (the query never got that far).

Still to do after the pipeline answers:
- Point `perception_demo.py` at `/robokudo/query` instead of its own colour finder.
  Reply poses are in `map`; treat `map` as the world root, as the colour demo does.
- Then use the perceived poses in `real_demo.py` before the plan runs.
