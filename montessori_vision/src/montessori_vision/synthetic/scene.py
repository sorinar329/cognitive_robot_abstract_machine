"""Building and rendering one montessori board scene in Blender.

.. warning:: This module imports bpy, which comes with the ``blender`` extra and is only published
    for the Python version the matching Blender release ships. Import it only where that is
    installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import bpy

from montessori_vision.board import BoardConfiguration, TargetKind
from montessori_vision.detections import Detection, ImageDetections, ShapeLabel
from montessori_vision.synthetic.layout import BoardLayout
from montessori_vision.synthetic.prism import Prism
from montessori_vision.synthetic.projection import CameraProjection
from montessori_vision.synthetic.randomization import SampledScene

# %% blender names


class SceneObjectName(StrEnum):
    """The names the built objects carry inside the Blender scene."""

    TABLE = "table"
    """The surface the board and the loose pieces rest on."""

    BOARD = "board"
    """The slab the holes are cut into."""

    CAMERA = "camera"
    """The camera the render is taken from."""

    KEY_LIGHT = "key_light"
    """The light the scene is mostly lit by."""

    WORLD = "world"
    """The surroundings that light everything evenly."""


class RenderEngine(StrEnum):
    """The renderers a scene can be drawn with."""

    CYCLES = "CYCLES"
    """Path tracing, which gives the shadowed rims that make a hole read as a hole."""

    EEVEE = "BLENDER_EEVEE_NEXT"
    """Rasterisation, fast enough to render a large dataset but flatter looking."""


class ImageFormat(StrEnum):
    """The file formats a render can be written in."""

    PNG = "PNG"
    """Lossless, so a detector is not trained on compression artefacts the camera will not show."""


class BackgroundInput(StrEnum):
    """The inputs of Blender's background shader that are randomised."""

    COLOR = "Color"
    """The colour the surroundings glow in."""

    STRENGTH = "Strength"
    """How brightly the surroundings glow."""


class MaterialInput(StrEnum):
    """The inputs of Blender's principled shader that are randomised."""

    BASE_COLOR = "Base Color"
    """The colour the surface reflects."""

    ROUGHNESS = "Roughness"
    """How matte the surface is."""


# %% the scene


@dataclass(frozen=True)
class LabelledSolid:
    """A solid in the scene together with the shape it is, so it can be labelled after rendering."""

    label: ShapeLabel
    """The shape and role this solid stands for."""

    prism: Prism
    """The solid in world coordinates, whose corners the label's box is projected from."""

    def rotated(self, angle: float) -> LabelledSolid:
        """Return the solid turned around the world's vertical axis with the board it sits in."""
        return LabelledSolid(label=self.label, prism=self.prism.moved(0.0, 0.0, angle))


@dataclass
class BlenderScene:
    """Builds a board with its holes and loose pieces, renders it, and says where every shape
    landed.

    Because the scene is built rather than loaded, the label of every shape is known exactly, which
    is the point of rendering training data in the first place.
    """

    board: BoardConfiguration
    """The shapes the board holds."""

    layout: BoardLayout = field(default_factory=BoardLayout)
    """The measurements of the board and the grid its holes are cut in."""

    image_width: int = 640
    """Width of the rendered image in pixels."""

    image_height: int = 480
    """Height of the rendered image in pixels."""

    sensor_width: float = 36.0
    """Width of the simulated sensor in millimetres."""

    engine: RenderEngine = RenderEngine.CYCLES
    """The renderer the images are drawn with."""

    samples: int = 32
    """How many samples each pixel is drawn with; more is cleaner and slower."""

    table_size: float = 4.0
    """Metres across of the surface everything rests on, wide enough that its edge stays out of the
    frame at the viewpoints the ranges cover."""

    light_distance: float = 2.0
    """Metres between the key light and the centre of the board."""

    minimum_label_area: int = 64
    """Square pixels a shape has to cover to be labelled, which drops the slivers of a shape that is
    almost entirely outside the frame rather than teaching a detector to find them."""

    def render(self, sampled: SampledScene, image_path: Path) -> ImageDetections:
        """Build the scene, render it to the given path and return the shapes it shows."""
        self.clear()
        self.light_world(sampled)
        self.build_table(sampled)
        solids = self.build_board(sampled)
        solids.extend(self.build_loose_pieces(sampled))
        camera = self.place_camera(sampled)
        self.place_light(sampled)
        self.configure_render(image_path)
        bpy.ops.render.render(write_still=True)
        return self.label(solids, camera, image_path.name)

    # %% building

    def clear(self) -> None:
        """Empty the scene so one render never leaks into the next."""
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.lights):
            for item in list(collection):
                collection.remove(item)

    def build_table(self, sampled: SampledScene) -> None:
        """Lay down the surface the board and the pieces rest on."""
        bpy.ops.mesh.primitive_plane_add(size=self.table_size, location=(0, 0, 0))
        table = bpy.context.active_object
        table.name = SceneObjectName.TABLE
        self.apply_material(table, sampled.table_colour, sampled.roughness)

    def light_world(self, sampled: SampledScene) -> None:
        """Give the surroundings an even glow, so nothing sits against an empty white void."""
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new(SceneObjectName.WORLD)
            bpy.context.scene.world = world
        world.use_nodes = True
        background = world.node_tree.nodes["Background"]
        background.inputs[BackgroundInput.COLOR].default_value = (*sampled.table_colour, 1.0)
        background.inputs[BackgroundInput.STRENGTH].default_value = sampled.world_brightness

    def build_board(self, sampled: SampledScene) -> list[LabelledSolid]:
        """Cut a hole per shape into a slab and return where each hole ended up.

        The holes are cut with a boolean difference so the board really has openings, which is what
        gives the renders the shadowed rims a real board shows.
        """
        size = self.layout.board_size(len(self.board.categories))
        bottom = 0.0
        top = self.layout.board_thickness
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, (bottom + top) / 2))
        slab = bpy.context.active_object
        slab.name = SceneObjectName.BOARD
        slab.scale = (size.width, size.depth, self.layout.board_thickness)
        bpy.ops.object.transform_apply(scale=True)

        holes = []
        positions = self.layout.hole_positions(len(self.board.categories))
        for category, position in zip(self.board.categories, positions):
            # The cutter reaches past both faces so the boolean leaves no paper thin skin behind.
            cutter_prism = Prism.extruded(
                category.outline, self.layout.hole_radius, bottom - top, top * 2
            ).moved(position.x, position.y, 0.0)
            cutter = self.add_solid(cutter_prism, f"{category.name}_cutter")
            self.subtract(slab, cutter)
            holes.append(
                LabelledSolid(
                    label=ShapeLabel(category, TargetKind.HOLE),
                    prism=Prism.extruded(category.outline, self.layout.hole_radius, top, top).moved(
                        position.x, position.y, 0.0
                    ),
                )
            )

        slab.rotation_euler = (0.0, 0.0, sampled.board_rotation)
        bpy.ops.object.select_all(action="DESELECT")
        slab.select_set(True)
        bpy.context.view_layer.objects.active = slab
        bpy.ops.object.transform_apply(rotation=True)
        self.apply_material(slab, sampled.board_colour, sampled.roughness)
        return [hole.rotated(sampled.board_rotation) for hole in holes]

    def build_loose_pieces(self, sampled: SampledScene) -> list[LabelledSolid]:
        """Drop the loose pieces on the table and return where each one landed."""
        placed = []
        for piece, colour in zip(sampled.loose_pieces, sampled.piece_colours):
            prism = Prism.extruded(
                piece.category.outline, self.layout.shape_radius, 0.0, self.layout.piece_thickness
            ).moved(piece.position.x, piece.position.y, piece.rotation)
            solid = self.add_solid(prism, f"{piece.category.name}_piece")
            self.apply_material(solid, colour, sampled.roughness)
            placed.append(
                LabelledSolid(label=ShapeLabel(piece.category, TargetKind.PIECE), prism=prism)
            )
        return placed

    def add_solid(self, prism: Prism, name: str) -> bpy.types.Object:
        """Add a solid to the scene as a new mesh object."""
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(
            [(corner.x, corner.y, corner.z) for corner in prism.corners],
            [],
            [list(face) for face in prism.faces],
        )
        mesh.update()
        solid = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(solid)
        return solid

    def subtract(self, target: bpy.types.Object, cutter: bpy.types.Object) -> None:
        """Cut one solid out of another and remove the cutter."""
        modifier = target.modifiers.new(name=cutter.name, type="BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        bpy.context.view_layer.objects.active = target
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)

    def apply_material(
        self, target: bpy.types.Object, colour: tuple[float, float, float], roughness: float
    ) -> None:
        """Give an object a plain shader in the given colour."""
        material = bpy.data.materials.new(name=f"{target.name}_material")
        material.use_nodes = True
        shader = material.node_tree.nodes["Principled BSDF"]
        shader.inputs[MaterialInput.BASE_COLOR].default_value = (*colour, 1.0)
        shader.inputs[MaterialInput.ROUGHNESS].default_value = roughness
        target.data.materials.append(material)

    # %% camera, light and render settings

    def place_camera(self, sampled: SampledScene) -> CameraProjection:
        """Put the camera on its orbit around the board and return how it projects the world."""
        projection = CameraProjection.looking_at_origin(
            distance=sampled.camera_distance,
            elevation=sampled.camera_elevation,
            azimuth=sampled.camera_azimuth,
            focal_length=sampled.focal_length,
            sensor_width=self.sensor_width,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = SceneObjectName.CAMERA
        camera.data.lens = sampled.focal_length
        camera.data.sensor_width = self.sensor_width
        camera.location = (
            sampled.camera_distance
            * math.cos(sampled.camera_elevation)
            * math.cos(sampled.camera_azimuth),
            sampled.camera_distance
            * math.cos(sampled.camera_elevation)
            * math.sin(sampled.camera_azimuth),
            sampled.camera_distance * math.sin(sampled.camera_elevation),
        )
        camera.rotation_euler = (
            math.pi / 2 - sampled.camera_elevation,
            0.0,
            sampled.camera_azimuth + math.pi / 2,
        )
        bpy.context.scene.camera = camera
        return projection

    def place_light(self, sampled: SampledScene) -> None:
        """Put the key light on its own orbit around the board."""
        bpy.ops.object.light_add(type="AREA")
        light = bpy.context.active_object
        light.name = SceneObjectName.KEY_LIGHT
        light.data.energy = sampled.light_energy
        light.location = (
            self.light_distance
            * math.cos(sampled.light_elevation)
            * math.cos(sampled.light_azimuth),
            self.light_distance
            * math.cos(sampled.light_elevation)
            * math.sin(sampled.light_azimuth),
            self.light_distance * math.sin(sampled.light_elevation),
        )
        light.rotation_euler = (
            math.pi / 2 - sampled.light_elevation,
            0.0,
            sampled.light_azimuth + math.pi / 2,
        )

    def configure_render(self, image_path: Path) -> None:
        """Point the renderer at the output file and set the image size."""
        render = bpy.context.scene.render
        render.engine = self.engine
        render.resolution_x = self.image_width
        render.resolution_y = self.image_height
        render.resolution_percentage = 100
        render.image_settings.file_format = ImageFormat.PNG
        render.filepath = str(image_path)
        if self.engine is RenderEngine.CYCLES:
            bpy.context.scene.cycles.samples = self.samples
            return
        bpy.context.scene.eevee.taa_render_samples = self.samples

    # %% labelling

    def label(
        self, solids: list[LabelledSolid], camera: CameraProjection, image_name: str
    ) -> ImageDetections:
        """Project every solid onto the render and report the shapes that landed on it."""
        detections = []
        for solid in solids:
            box = camera.bounding_box(solid.prism.corners)
            if box is not None and box.area >= self.minimum_label_area:
                detections.append(Detection(label=solid.label, bounding_box=box))
        return ImageDetections(
            image_name=image_name,
            width=self.image_width,
            height=self.image_height,
            detections=detections,
        )
