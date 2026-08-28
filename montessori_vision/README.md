# Montessori Vision

Two computer vision approaches to the same question, built so they can be compared on the same
pictures: **where are the montessori board's shapes, and where are the holes they fit into?**

A robot arm has to pick a wooden shape up and insert it into the matching cutout. Both steps need
perception, and it is not obvious which approach suits the task better, so this repository builds
both and measures them against each other.

| | Approach one | Approach two |
| --- | --- | --- |
| **Idea** | Segment everything, then name each segment | Render synthetic boards, train a detector on them |
| **Models** | Segment Anything + CLIP | Blender + YOLO |
| **Training** | None | Trained on renders only |
| **A new shape costs** | Two lines in `board.yaml` | A re-render and a re-train |
| **Runs at** | Seconds per frame | Milliseconds per frame |
| **Package** | `montessori_vision.segment_and_classify` | `montessori_vision.synthetic` + `montessori_vision.yolo` |

Both return the same thing — an `ImageDetections` naming every shape, whether it is a `PIECE` or a
`HOLE`, and where it sits — so the robot never has to know which one it is talking to.

## One board configuration, three consumers

`src/montessori_vision/resources/board.yaml` is the single description of the board. It gives every
shape a name, a silhouette and a set of text descriptions, and it feeds:

* the text prompts approach one scores each crop against,
* the class list approach two trains and predicts in,
* the meshes the Blender renderer builds.

Adapt it to the board in front of your robot and both approaches follow; there is no shape
vocabulary hardcoded anywhere else.

```yaml
categories:
  - name: star
    outline: {type: star, points: 5, inner_radius_ratio: 0.45}
    piece_prompts: ["a photo of a star shaped wooden block"]
    hole_prompts:  ["a photo of a star shaped hole cut into a wooden board"]
background_prompts: ["a photo of an empty wooden table", "a photo of a robot gripper"]
```

## Installing

```bash
pip install -e .                          # core: reading pictures, scoring, the shared types
pip install -e ".[segment_and_classify]"  # approach one: torch, segment-anything, open_clip
pip install -e ".[yolo]"                  # approach two, running: ultralytics
pip install -e ".[blender]"               # approach two, rendering: bpy
pip install -e ".[dev]"                   # pytest and the formatters
```

The extras are separate on purpose: each is a multi gigabyte model runtime, and the core package
imports without any of them, so a robot that only runs the trained detector installs only that one.
The `blender` extra pins `bpy`, which is published only for the Python version its Blender release
ships with (3.11 for Blender 4.x and 5.x).

## Approach one: segment, then classify

Segment Anything proposes a mask for everything it can find without being told what to look for, a
size and shape filter throws away the table and the speckle, and CLIP scores each surviving crop
against the descriptions from `board.yaml`. Crops whose best match is a background description are
rejected rather than forced into the nearest shape.

```python
from pathlib import Path

from montessori_vision.board import BoardConfiguration
from montessori_vision.image import Image
from montessori_vision.segment_and_classify.clip import ClipClassifier
from montessori_vision.segment_and_classify.pipeline import SegmentAndClassifyPipeline
from montessori_vision.segment_and_classify.segment_anything import SegmentAnythingMaskGenerator

board = BoardConfiguration.default()
pipeline = SegmentAndClassifyPipeline(
    board=board,
    mask_generator=SegmentAnythingMaskGenerator(checkpoint_path=Path("weights/sam_vit_h.pth")),
    classifier=ClipClassifier(board=board),
)
found = pipeline.detect(Image.read(Path("data/frames/000123.png")))
```

The segmenter and the classifier are interfaces (`MaskGenerator`, `CropClassifier`), so a faster
segmenter or a different text image model is a drop in replacement, and the tests run the whole
pipeline against stand-ins without loading a model.

## Approach two: render, then train

Blender builds a board with real cutouts, scatters loose pieces beside it, and randomises the
viewpoint, the lighting, the materials and the colours. Because the renderer placed every shape, it
knows exactly where each one is, so the training labels are exact and free.

```bash
python -m montessori_vision.synthetic.generate --output data/synthetic --images 5000 --seed 1
```

That writes `images/`, `labels/` and a `data.yaml` in the layout a detector trains from, with the
class list derived from the board configuration. Then:

```python
from pathlib import Path

from montessori_vision.yolo.training import YoloTrainer

weights = YoloTrainer(dataset_description=Path("data/synthetic/data.yaml"), epochs=100).train()
```

and at run time:

```python
from montessori_vision.yolo.pipeline import YoloPipeline

pipeline = YoloPipeline(board=board, weights_path=weights)
```

How wide the randomisation goes is `RandomizationRanges`, and how large the board and its holes are
is `BoardLayout`. Both ship with values for a typical wooden shape sorter — measure your own board
and your own camera and narrow them to what the robot will actually see.

## Comparing the two

Put the frames you extracted from the recording in a folder with an `annotations.json` beside them,
and score both approaches on exactly the same pictures:

```bash
python scripts/compare_pipelines.py --images data/frames \
    --sam-checkpoint weights/sam_vit_h.pth \
    --yolo-weights runs/montessori/weights/best.pt
```

The table it prints has one block per approach, split into pieces and holes — nothing has been
measured yet, so these are the columns rather than results:

```
pipeline                     precision   recall     f1  overlap
------------------------------------------------------------
SegmentAndClassifyPipeline           ?        ?      ?        ?
  piece                              ?        ?      ?        ?
  hole                               ?        ?      ?        ?
YoloPipeline                         ?        ?      ?        ?
  piece                              ?        ?      ?        ?
  hole                               ?        ?      ?        ?
```

A prediction counts as a hit only when it overlaps an annotated shape *and* names the same shape in
the same role, so calling a star a hexagon is a miss and a false alarm rather than a near miss. The
breakdown into pieces and holes is the interesting part: holes are the harder half of the task, and
an overall number hides an approach that never finds them.

## Annotating the validation frames

Ground truth uses the same type the pipelines return, written as json:

```json
[{"image": "000123.png", "width": 1280, "height": 720,
  "detections": [{"category": "star", "kind": "hole",
                  "left": 412, "top": 233, "right": 498, "bottom": 318}]}]
```

## Development

```bash
pytest                      # the whole suite, seconds, no model weights and no network
pre-commit install          # black and docformatter on commit
```

The tests draw their board pictures from the outlines in `board.yaml` itself rather than shipping
recorded images, so they exercise the same shapes the pipelines will be given. Nothing in the suite
downloads a model: the pipelines are tested against stand-in implementations of their interfaces,
and one test asserts that importing the package never pulls in torch, ultralytics or bpy.

Code style follows the `AGENTS.md` carried over from the
[cognitive_robot_abstract_machine](https://github.com/cram2/cognitive_robot_abstract_machine)
workspace, so code moves between the two without a reformat.

## Not here yet

* **Extracting frames from a rosbag.** Drop the pictures in a folder yourself for now; everything
  downstream already reads that layout.
* **Pose rather than boxes.** Both approaches report boxes, and approach one also reports masks.
  Getting from there to a grasp pose needs the depth stream and is the obvious next step.
* **Pieces resting on the board.** The renderer scatters loose pieces beside the board only.

## License

LGPL-3.0-only, as in the workspace this grew out of.
