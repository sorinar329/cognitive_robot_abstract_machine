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

## 2. RoboKudo pipeline (answers from the recorded bags)

Colour only, no shape fitting on depth: the depth of the steel table is too coarse and
noisy for 3 cm pieces (a piece is about 5 depth pixels across, the table wobbles by
about 15 mm, and a reflection hides the cube).

Files:
- `robokudo/src/robokudo/descriptors/analysis_engines/tracy_montessori.py`, the engine:
  query → Orbbec reader → `ImageClusterExtractor` for cyan (hue 130-146) and for yellow
  (hue 30-45) → `FootprintShapeAnnotator` → `MontessoriPieceAnnotator` → query reply.
- `robokudo/src/robokudo/annotators/footprint_shape.py`. It takes the brightest part
  of each blob (the top face, lit from above; Otsu on brightness) and casts its outline
  onto the plane of the top faces (table 0.88 + 0.03 in `map`) from the camera pose. It
  then names the outline: fills less than 0.75 of its smallest rectangle → triangle;
  5 or more corners → circle; otherwise square or rectangle by length/width 1.5.
- `ImageClusterExtractor` now keeps each blob's filled outline in `roi.mask`.
- `robokudo/src/robokudo/annotators/montessori_pieces.py` names a piece only for the
  four real pairs (cyan square/circle, yellow rectangle/triangle), and poses it at the
  footprint's centre.
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

Without the robot, play a bag of the ICRA reproduction package instead of the camera.
`/tf_static` needs a transient-local override, or the static camera frames never
arrive:

```
ros2 bag play <episode>/files/bag --qos-profile-overrides-path tf_static_qos.yaml \
  --topics /tf /tf_static /camera/color/image_raw/compressed \
  /camera/depth/image_raw /camera/color/camera_info
```

with `tf_static_qos.yaml`:

```
/tf_static:
  history: keep_all
  durability: transient_local
  reliability: reliable
```

Measured on four real episodes (508e367a, 0a793ded, 57bbcbb2, 25f5161d), 8 s in:
- every query answers in about 1 s;
- all four pieces are named correctly in each;
- every piece lies within 4 mm of where the run's own perception put it
  (`perceived/*` in the bag's TF).

At the end of the sorting run (205 s, table empty) no piece is named; the robot arm's
cyan cap is the only blob and stays unnamed.

Known limits:
- Footprints measure about 20 % small (the cube's top face as 23 x 22 mm), because the
  bright patch leaves out the soft edge of the top face. Shapes and centres are
  unaffected.
- `ImageClusterExtractor` raises `ImageContourMissing` when a colour has no blob at
  all.
- Not run live on the robot yet.

Still to do:
- Point `perception_demo.py` at `/robokudo/query` instead of its own colour finder.
  Reply poses are in `map`; treat `map` as the world root, as the colour demo does.
- Then use the perceived poses in `real_demo.py` before the plan runs.
