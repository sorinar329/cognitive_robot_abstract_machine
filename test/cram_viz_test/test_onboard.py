"""
Tests for the onboarder's pure post-processing and the URDF asset bundler.

Recording itself needs a running coraplex demo, but everything that turns a recording
into a scene bundle is plain data work: deciding when an object moved, finding the
attach/detach window of each transport, labelling the resulting segments, and making a
URDF self-contained. Those are covered here against hand-built recordings.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import pytest
from semantic_digital_twin.adapters.gazebo import GazeboParser
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.world_description.geometry import Color, Scale
from typing_extensions import Any, Dict, List

from cram_viz.onboard import bundle_urdf as bundler
from cram_viz.onboard.demo import (
    Recorder,
    SpawnedBox,
    _update_scene_index,
    derive_segments,
    first_base_motion,
    link_set,
    moved,
    object_windows,
    scene_objects,
)

#: a pose that stays put, used wherever a frame's value must not matter
RESTING = [0.0, 0.0, 1.0, 0, 0, 0, 1]

#: an SDF file small enough to parse repeatedly within a single test
SIMPLE_SHAPES_SDF = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "gazebo",
        "simple_shapes.sdf",
    )
)


def pose_at(x: float, y: float, z: float = 1.0) -> List[float]:
    """
    A pose with the given position and no rotation.
    """
    return [x, y, z, 0, 0, 0, 1]


def recording(
    object_frames: List[Dict[str, List[float]]],
    base_frames: List[List[float]] = None,
    actions: List[Dict[str, Any]] = None,
) -> Recorder:
    """
    A recorder holding a finished recording, without having run a demo.
    """
    recorder = Recorder()
    recorder.object_frames = object_frames
    recorder.frames = [{} for _ in object_frames]
    recorder.base_frames = base_frames or [RESTING for _ in object_frames]
    recorder.actions = actions or []
    return recorder


# %% asset and tick hooks
class TestAssetHookMethods:
    """
    The methods ``install_asset_hooks``/``install_tick_hook`` patch in.

    Exercised directly with fake ``original`` callables, so none of the real
    semantic_digital_twin/giskardpy classes need to be monkeypatched here.
    """

    def test_a_resolution_is_remembered_and_returned(self):
        recorder = Recorder()

        result = recorder._remember_resolution(
            lambda resolver, uri: "/opt/pkg/cup.stl",
            "the-resolver",
            "package://pkg/cup.stl",
        )

        assert result == "/opt/pkg/cup.stl"
        assert recorder.resolutions == {"package://pkg/cup.stl": "/opt/pkg/cup.stl"}

    def test_a_urdf_source_is_recorded_once(self):
        recorder = Recorder()
        original = lambda cls, file_path, **kwargs: "parsed"

        first = recorder._remember_urdf_source(original, "the-cls", "robot.urdf")
        recorder._remember_urdf_source(original, "the-cls", "robot.urdf")

        assert first == "parsed"
        assert recorder.urdf_sources == ["robot.urdf"]

    def test_a_gazebo_source_is_recorded_once(self):
        recorder = Recorder()
        original = lambda cls, file_path, **kwargs: "parsed"

        first = recorder._remember_gazebo_source(original, "the-cls", "world.sdf")
        recorder._remember_gazebo_source(original, "the-cls", "world.sdf")

        assert first == "parsed"
        assert recorder.gazebo_sources == ["world.sdf"]

    def test_a_mesh_source_is_recorded_once(self):
        recorder = Recorder()
        original = lambda stl_parser, file_path, *args, **kwargs: None

        recorder._remember_mesh_source(original, "the-parser", "cup.stl")
        recorder._remember_mesh_source(original, "the-parser", "cup.stl")

        assert recorder.mesh_sources == ["cup.stl"]

    def test_a_spawned_box_body_is_recorded_once(self):
        recorder = Recorder()
        specification = BodySpecification.box(
            "parcel", Scale(0.08, 0.08, 0.14), Color(0.8, 0.4, 0.2)
        )
        original = lambda spec, name=None: "materialized"

        first = recorder._remember_spawned_box(original, specification)
        recorder._remember_spawned_box(original, specification)

        assert first == "materialized"
        assert recorder.spawned_boxes == [
            SpawnedBox(name="parcel", scale=[0.08, 0.08, 0.14], color="#cc6633")
        ]

    def test_a_non_box_specification_is_not_recorded(self):
        recorder = Recorder()
        specification = BodySpecification.sphere("ball", 0.1)

        result = recorder._remember_spawned_box(
            lambda spec, name=None: "materialized", specification
        )

        assert result == "materialized"
        assert recorder.spawned_boxes == []

    def test_a_designator_targeting_a_spawned_box_is_matched(self):
        recorder = Recorder()
        recorder.spawned_boxes.append(
            SpawnedBox(name="parcel", scale=[0.08, 0.08, 0.14], color="#cc6633")
        )

        @dataclass
        class TransportsABody:
            target: NamedBody

        assert recorder._target_of(TransportsABody(NamedBody("parcel"))) == "parcel"

    def test_the_tick_hook_forwards_to_the_original_and_records_the_frame(self):
        recorder = Recorder()
        recorded_executors = []
        recorder.record_frame = recorded_executors.append

        result = recorder._record_tick(lambda executor: "ticked", "the-executor")

        assert result == "ticked"
        assert recorded_executors == ["the-executor"]


class TestAssetHookLifecycle:
    """
    ``install_asset_hooks``/``uninstall_asset_hooks`` patch and restore the real parser
    classes, so bundling — which itself re-parses a Gazebo source through
    :class:`GazeboParser` to build a clean, unprefixed URDF — must not be mistaken for
    further recording.
    """

    def test_uninstalling_stops_recording_further_sources(self):
        recorder = Recorder()
        recorder.install_asset_hooks()
        GazeboParser.from_file(SIMPLE_SHAPES_SDF)
        assert recorder.gazebo_sources == [SIMPLE_SHAPES_SDF]

        recorder.uninstall_asset_hooks()
        GazeboParser.from_file(SIMPLE_SHAPES_SDF)

        assert recorder.gazebo_sources == [SIMPLE_SHAPES_SDF]


# %% movement detection
class TestMovementDetection:
    def test_a_pose_is_unmoved_within_the_tolerance(self):
        assert moved(pose_at(0, 0), pose_at(0.01, 0.0)) is False

    def test_planar_travel_counts_as_movement(self):
        assert moved(pose_at(0, 0), pose_at(0.5, 0.0)) is True

    def test_vertical_travel_counts_as_movement(self):
        assert moved(pose_at(0, 0, 1.0), pose_at(0, 0, 1.5)) is True

    def test_the_tolerance_is_configurable(self):
        assert moved(pose_at(0, 0), pose_at(0.5, 0.0), tolerance=1.0) is False


# %% transport windows
class TestObjectWindows:
    def test_an_object_that_never_moves_has_no_window(self):
        recorder = recording([{"milk.stl": RESTING} for _ in range(5)])
        assert object_windows(recorder) == []

    def test_a_transported_object_reports_its_travel_window(self):
        """
        The window starts at the first frame that differs from the spawn pose and ends
        one past the last frame that differs from the final pose.
        """
        frames = [
            {"milk.stl": pose_at(0, 0)},
            {"milk.stl": pose_at(0, 0)},
            {"milk.stl": pose_at(1, 0)},
            {"milk.stl": pose_at(2, 0)},
            {"milk.stl": pose_at(2, 0)},
        ]
        window = object_windows(recording(frames))[0]
        assert window["object"] == "milk.stl"
        assert window["attach"] == 2
        assert window["detach"] == 3
        assert window["place"] == [2, 0, 1.0]

    def test_an_instant_jump_yields_no_window(self):
        """
        An object that is already at its destination the frame after it leaves the spawn
        has an empty window, so it is not reported as a transport.
        """
        frames = [{"milk.stl": pose_at(0, 0)} for _ in range(3)]
        frames += [{"milk.stl": pose_at(2, 0)} for _ in range(3)]
        assert object_windows(recording(frames)) == []

    def test_windows_are_ordered_by_when_they_start(self):
        early = [pose_at(0, 0), pose_at(1, 0), pose_at(2, 0), pose_at(3, 0)]
        early += [pose_at(4, 0), pose_at(4, 0)]
        late = [pose_at(0, 0)] * 3 + [pose_at(0, 1.5), pose_at(0, 3), pose_at(0, 3)]
        frames = [
            {"early.stl": early[index], "late.stl": late[index]} for index in range(6)
        ]
        windows = object_windows(recording(frames))
        assert [window["object"] for window in windows] == ["early.stl", "late.stl"]
        assert [window["attach"] for window in windows] == [1, 3]


class TestFirstBaseMotion:
    def test_a_standing_base_reports_the_upper_bound(self):
        recorder = recording([{} for _ in range(5)])
        assert first_base_motion(recorder, 4) == 4

    def test_the_frame_the_base_leaves_its_spawn_is_found(self):
        recorder = recording([{} for _ in range(5)])
        recorder.base_frames = [
            RESTING,
            RESTING,
            pose_at(1, 0),
            pose_at(2, 0),
            pose_at(2, 0),
        ]
        assert first_base_motion(recorder, 5) == 2

    def test_motion_after_the_bound_is_not_reported(self):
        recorder = recording([{} for _ in range(5)])
        recorder.base_frames = [RESTING, RESTING, RESTING, pose_at(3, 0), pose_at(3, 0)]
        assert first_base_motion(recorder, 2) == 2


# %% segment derivation
class TestDeriveSegments:
    def test_a_recording_without_transports_is_one_segment(self):
        recorder = recording(
            [{"milk.stl": RESTING} for _ in range(4)],
            actions=[{"action": "ParkArmsAction", "arm": None, "target": None}],
        )
        segments = derive_segments(recorder)
        assert [segment["step"] for segment in segments] == ["parkarms"]
        assert segments[0]["start"] == 0

    def test_an_unlabelled_recording_falls_back_to_one_plan_segment(self):
        recorder = recording([{"milk.stl": RESTING} for _ in range(4)])
        assert [segment["step"] for segment in derive_segments(recorder)] == ["plan"]

    def test_a_transport_is_named_after_its_action_and_object(self):
        milk = [pose_at(0, 0), pose_at(0, 0), pose_at(1, 0)]
        milk += [pose_at(2, 0), pose_at(2, 0), pose_at(2, 0)]
        recorder = recording(
            [{"milk.stl": pose} for pose in milk],
            actions=[
                {"action": "TransportAction", "arm": "LEFT", "target": "milk.stl"}
            ],
        )
        transport = derive_segments(recorder)[-1]
        assert transport["step"] == "transport_milk"
        assert transport["picks"] == "milk"
        assert transport["arm"] == "LEFT"

    def test_segments_cover_the_recording_without_gaps(self):
        """
        Playback walks the segments in order, so each must start where the last ended.
        """
        milk = [pose_at(0, 0), pose_at(1, 0), pose_at(2, 0)] + [pose_at(2, 0)] * 5
        cup = [pose_at(5, 0)] * 4 + [pose_at(5, 1), pose_at(5, 2)] + [pose_at(5, 2)] * 2
        recorder = recording(
            [{"milk.stl": milk[index], "cup.stl": cup[index]} for index in range(8)],
            actions=[
                {"action": "TransportAction", "arm": None, "target": "milk.stl"},
                {"action": "TransportAction", "arm": None, "target": "cup.stl"},
            ],
        )
        segments = derive_segments(recorder)
        assert len(segments) == 2
        for earlier, later in zip(segments, segments[1:]):
            assert earlier["end"] == later["start"]


# %% scene objects
class TestSceneObjects:
    def test_a_spawned_box_becomes_a_box_object_entry(self, tmp_path):
        recorder = recording([{"parcel": pose_at(1, 2, 0.8)}])
        recorder.spawned_boxes.append(
            SpawnedBox(name="parcel", scale=[0.08, 0.08, 0.14], color="#cc6633")
        )

        assert scene_objects(recorder, str(tmp_path)) == [
            {
                "id": "parcel",
                "key": "parcel",
                "box": [0.08, 0.08, 0.14],
                "spawn": pose_at(1, 2, 0.8),
                "color": "#cc6633",
                "height": 0.14,
            }
        ]

    def test_a_box_that_was_never_pose_tracked_is_left_out(self, tmp_path):
        recorder = recording([{}])
        recorder.spawned_boxes.append(
            SpawnedBox(name="parcel", scale=[0.08, 0.08, 0.14], color="#cc6633")
        )

        assert scene_objects(recorder, str(tmp_path)) == []


# %% scene index
def write_scene_bundle(bundle_dir, robot_name, model_entries) -> None:
    """
    A minimal ``scene.json`` on disk, with just the keys ``_scan_scenes`` reads.
    """
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "scene.json").write_text(
        json.dumps({"robot": {"name": robot_name}, "models": model_entries})
    )


class TestSceneIndex:
    def test_a_scene_is_indexed_with_its_robot_and_environment(self, tmp_path):
        write_scene_bundle(
            tmp_path / "pr2_kitchen",
            "pr2",
            [{"name": "pr2", "robot": True}, {"name": "apartment", "robot": False}],
        )
        index_path = tmp_path / "index.json"

        _update_scene_index(index_path, "pr2_kitchen")

        index = json.loads(index_path.read_text())
        assert index["scenes"] == [
            {"name": "pr2_kitchen", "robot": "pr2", "environment": "apartment"}
        ]

    def test_a_bench_only_scene_has_no_environment(self, tmp_path):
        write_scene_bundle(
            tmp_path / "tracy_lab", "tracy", [{"name": "tracy", "robot": True}]
        )
        index_path = tmp_path / "index.json"

        _update_scene_index(index_path, "tracy_lab")

        index = json.loads(index_path.read_text())
        assert index["scenes"] == [
            {"name": "tracy_lab", "robot": "tracy", "environment": None}
        ]

    def test_multiple_environment_models_are_joined(self, tmp_path):
        write_scene_bundle(
            tmp_path / "aicor_cell",
            "aicor_cell_arm",
            [
                {"name": "aicor_cell_arm", "robot": True},
                {"name": "bench", "robot": False},
                {"name": "shelf", "robot": False},
            ],
        )
        index_path = tmp_path / "index.json"

        _update_scene_index(index_path, "aicor_cell")

        index = json.loads(index_path.read_text())
        assert index["scenes"][0]["environment"] == "bench+shelf"

    def test_a_removed_bundle_drops_out_of_the_index(self, tmp_path):
        """
        The index is rebuilt from the bundles actually on disk, so a scene folder that
        was deleted or renamed after it was indexed cannot leave a stale entry behind.
        """
        write_scene_bundle(
            tmp_path / "pr2_kitchen",
            "pr2",
            [{"name": "pr2", "robot": True}, {"name": "apartment", "robot": False}],
        )
        index_path = tmp_path / "index.json"
        index_path.write_text(
            json.dumps({"default": "pr2_kitchen", "scenes": ["pr2_kitchen", "gone"]})
        )

        _update_scene_index(index_path, "pr2_kitchen")

        index = json.loads(index_path.read_text())
        assert [entry["name"] for entry in index["scenes"]] == ["pr2_kitchen"]

    def test_the_default_is_set_once_and_then_left_alone(self, tmp_path):
        write_scene_bundle(
            tmp_path / "pr2_kitchen", "pr2", [{"name": "pr2", "robot": True}]
        )
        write_scene_bundle(
            tmp_path / "garmi_apartment", "garmi", [{"name": "garmi", "robot": True}]
        )
        index_path = tmp_path / "index.json"

        _update_scene_index(index_path, "pr2_kitchen")
        _update_scene_index(index_path, "garmi_apartment")

        index = json.loads(index_path.read_text())
        assert index["default"] == "pr2_kitchen"
        assert [entry["name"] for entry in index["scenes"]] == [
            "garmi_apartment",
            "pr2_kitchen",
        ]


# %% robot parts
@dataclass
class RobotPartWithBodies:
    """
    A robot part exposing the bodies whose link names the onboarder records.
    """

    bodies: List[Any] = field(default_factory=list)


@dataclass
class NamedBody:
    """
    A world body carrying a model-prefixed name.
    """

    name: str


class TestLinkSet:
    def test_the_model_prefix_is_stripped(self):
        part = RobotPartWithBodies(bodies=[NamedBody("pr2/l_wrist_link")])
        assert link_set(part) == ["l_wrist_link"]

    def test_an_unprefixed_name_is_kept(self):
        part = RobotPartWithBodies(bodies=[NamedBody("l_wrist_link")])
        assert link_set(part) == ["l_wrist_link"]

    def test_a_part_without_bodies_has_no_links(self):
        assert link_set(RobotPartWithBodies()) == []


# %% URDF reference resolution
class TestResolveUri:
    def test_a_recorded_resolution_wins(self, tmp_path):
        target = tmp_path / "cup.stl"
        target.write_text("solid cup\nendsolid cup\n")
        resolved = bundler.resolve_uri(
            "package://demo/cup.stl", hints={"package://demo/cup.stl": str(target)}
        )
        assert resolved == str(target)

    def test_a_relative_reference_resolves_against_the_urdf(self, tmp_path):
        mesh = tmp_path / "meshes" / "cup.stl"
        mesh.parent.mkdir()
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.resolve_uri("meshes/cup.stl", base_dir=str(tmp_path)) == str(
            mesh
        )

    def test_a_missing_relative_reference_is_unresolved(self, tmp_path):
        assert bundler.resolve_uri("meshes/gone.stl", base_dir=str(tmp_path)) is None

    def test_a_file_uri_resolves_to_its_path(self, tmp_path):
        mesh = tmp_path / "cup.stl"
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.resolve_uri("file://" + str(mesh)) == str(mesh)

    def test_an_absolute_path_that_exists_resolves_to_itself(self, tmp_path):
        mesh = tmp_path / "cup.stl"
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.resolve_uri(str(mesh)) == str(mesh)


class TestReferenceLayout:
    def test_a_package_reference_keeps_its_package_directory(self):
        assert bundler._bundled_relative_path("package://demo/meshes/cup.stl") == (
            "demo/meshes/cup.stl"
        )

    def test_a_local_reference_lands_in_one_flat_directory(self):
        assert bundler._bundled_relative_path("../far/away/cup.stl") == "_local/cup.stl"


# %% bundling a URDF
class TestBundleUrdf:
    @pytest.fixture()
    def source_tree(self, tmp_path):
        """
        A URDF referencing one mesh, both on disk next to each other.
        """
        (tmp_path / "meshes").mkdir()
        (tmp_path / "meshes" / "cup.stl").write_text("solid cup\nendsolid cup\n")
        urdf = tmp_path / "robot.urdf"
        urdf.write_text(
            '<robot name="demo">\n'
            '  <link name="base_link"/>\n'
            '  <link name="cup_link">\n'
            "    <visual><geometry>\n"
            '      <mesh filename="meshes/cup.stl"/>\n'
            "    </geometry></visual>\n"
            "  </link>\n"
            '  <joint name="cup_joint" type="fixed">\n'
            '    <parent link="base_link"/><child link="cup_link"/>\n'
            "  </joint>\n"
            "</robot>\n"
        )
        return urdf

    def test_the_mesh_is_copied_next_to_the_rewritten_urdf(self, source_tree, tmp_path):
        out_dir = tmp_path / "bundle"
        report = bundler.bundle_urdf(str(source_tree), "demo", str(out_dir))
        assert (out_dir / "demo.urdf").is_file()
        assert (out_dir / "meshes" / "_local" / "cup.stl").is_file()
        assert report["meshes_copied"] == 1
        assert report["missing"] == []

    def test_the_reference_is_rewritten_to_the_bundled_copy(
        self, source_tree, tmp_path
    ):
        out_dir = tmp_path / "bundle"
        bundler.bundle_urdf(str(source_tree), "demo", str(out_dir))
        rewritten = (out_dir / "demo.urdf").read_text()
        assert 'filename="meshes/_local/cup.stl"' in rewritten
        assert 'filename="meshes/cup.stl"' not in rewritten

    def test_links_and_joints_are_reported(self, source_tree, tmp_path):
        report = bundler.bundle_urdf(str(source_tree), "demo", str(tmp_path / "bundle"))
        assert report["links"] == ["base_link", "cup_link"]
        assert report["joints"] == ["cup_joint"]
        assert report["movable_joints"] == []

    def test_an_unresolvable_mesh_is_reported_as_missing(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text(
            '<robot name="demo">\n'
            '  <link name="base_link">\n'
            '    <visual><geometry><mesh filename="meshes/gone.stl"/></geometry></visual>\n'
            "  </link>\n"
            "</robot>\n"
        )
        report = bundler.bundle_urdf(str(urdf), "demo", str(tmp_path / "bundle"))
        assert report["missing"] == [bundler.UNRESOLVED_REFERENCE]
        assert report["meshes_copied"] == 0

    def test_a_missing_source_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bundler.bundle_urdf(str(tmp_path / "gone.urdf"), "demo", str(tmp_path))


# %% side assets referenced from inside a mesh
class TestParentRelativeSideAssets:
    """
    A mesh's texture references are resolved relative to the mesh file and must be
    mirrored at that same relative location inside the bundle, because that is where the
    browser resolves them against the bundled mesh's URL.
    """

    @pytest.fixture()
    def source_tree(self, tmp_path):
        """
        A URDF whose mesh keeps its texture in a sibling ``materials`` directory, the
        Gazebo model layout.
        """
        source_directory = tmp_path / "deep" / "source"
        (source_directory / "meshes").mkdir(parents=True)
        (source_directory / "materials" / "textures").mkdir(parents=True)
        (source_directory / "meshes" / "crate.dae").write_text(
            "<library_images><init_from>"
            "../materials/textures/crate.png"
            "</init_from></library_images>\n"
        )
        (source_directory / "materials" / "textures" / "crate.png").write_bytes(
            b"not really a png"
        )
        urdf = source_directory / "robot.urdf"
        urdf.write_text(
            '<robot name="demo">\n'
            '  <link name="crate_link">\n'
            '    <visual><geometry><mesh filename="meshes/crate.dae"/></geometry></visual>\n'
            "  </link>\n"
            "</robot>\n"
        )
        return urdf

    def test_a_parent_relative_texture_is_mirrored_into_the_bundle(
        self, source_tree, tmp_path
    ):
        out_dir = tmp_path / "bundle"
        report = bundler.bundle_urdf(str(source_tree), "demo", str(out_dir))
        # the browser resolves ../materials/... against meshes/_local/crate.dae
        assert (out_dir / "meshes" / "materials" / "textures" / "crate.png").is_file()
        assert ".png" in report["mesh_exts"]

    def test_a_reference_escaping_the_bundle_is_not_copied(self, tmp_path):
        source_directory = tmp_path / "deep" / "source"
        (source_directory / "meshes").mkdir(parents=True)
        (source_directory / "meshes" / "crate.dae").write_text(
            "<init_from>../../../escape.png</init_from>\n"
        )
        (tmp_path / "escape.png").write_bytes(b"not really a png")
        urdf = source_directory / "robot.urdf"
        urdf.write_text(
            '<robot name="demo">\n'
            '  <link name="crate_link">\n'
            '    <visual><geometry><mesh filename="meshes/crate.dae"/></geometry></visual>\n'
            "  </link>\n"
            "</robot>\n"
        )
        out_dir = tmp_path / "out" / "bundle"
        report = bundler.bundle_urdf(str(urdf), "demo", str(out_dir))
        assert not (tmp_path / "out" / "escape.png").exists()
        assert ".png" not in report["mesh_exts"]
