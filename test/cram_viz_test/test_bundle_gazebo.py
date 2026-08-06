"""
Tests for the Gazebo/SDF world bundler.

:func:`bundle_gazebo_world` builds a URDF from a parsed
:class:`~semantic_digital_twin.world.World` rather than rewriting an existing URDF's
text, so these tests exercise every shape and connection type the serializer maps: the
primitive shapes, a revolute joint with limits, a prismatic joint, a continuous joint
(whose position limits GazeboParser itself discards), and a free (non-static) model,
against the fixtures the semantic_digital_twin Gazebo adapter tests already use.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass

import pytest

from cram_viz.onboard import bundle_gazebo as bundler

RESOURCES_DIRECTORY = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "gazebo",
    )
)


@dataclass
class GazeboFixturePaths:
    """
    The paths of the SDF files these tests bundle.
    """

    simple_shapes: str
    hinged_door: str
    drawer: str
    mini_world: str


@pytest.fixture
def gazebo_paths() -> GazeboFixturePaths:
    mini_warehouse = os.path.join(RESOURCES_DIRECTORY, "mini_warehouse")
    return GazeboFixturePaths(
        simple_shapes=os.path.join(RESOURCES_DIRECTORY, "simple_shapes.sdf"),
        hinged_door=os.path.join(RESOURCES_DIRECTORY, "hinged_door.sdf"),
        drawer=os.path.join(RESOURCES_DIRECTORY, "drawer.sdf"),
        mini_world=os.path.join(mini_warehouse, "worlds", "mini.world"),
    )


def joint_types(urdf_path: str) -> list:
    """
    :param urdf_path: Path of a bundled URDF.
    :return: The ``type`` attribute of every ``joint`` element, in document order.
    """
    root = ElementTree.parse(urdf_path).getroot()
    return [joint.get("type") for joint in root.findall("joint")]


# %% primitive shapes
class TestPrimitiveShapes:
    def test_box_cylinder_and_sphere_are_bundled_as_one_link(
        self, gazebo_paths, tmp_path
    ):
        report = bundler.bundle_gazebo_world(
            gazebo_paths.simple_shapes, "simple_shapes", str(tmp_path)
        )
        assert report["links"] == ["simple_shapes/link"]
        assert report["joints"] == []
        assert report["meshes_copied"] == 0
        assert report["missing"] == []

    def test_the_geometries_are_written_with_their_own_tags(
        self, gazebo_paths, tmp_path
    ):
        bundler.bundle_gazebo_world(
            gazebo_paths.simple_shapes, "simple_shapes", str(tmp_path)
        )
        root = ElementTree.parse(tmp_path / "simple_shapes.urdf").getroot()
        geometries = [visual.find("geometry")[0].tag for visual in root.iter("visual")]
        assert geometries == ["box", "cylinder", "sphere"]
        box = root.find(".//box")
        assert box.get("size") == "0.2 0.4 0.6"
        sphere = root.find(".//sphere")
        assert sphere.get("radius") == "0.35"
        cylinder = root.find(".//cylinder")
        assert cylinder.get("radius") == "0.15"
        assert cylinder.get("length") == "0.8"


# %% joints
class TestJointTypes:
    def test_a_revolute_joint_keeps_its_limit_and_axis(self, gazebo_paths, tmp_path):
        report = bundler.bundle_gazebo_world(
            gazebo_paths.hinged_door, "hinged_door", str(tmp_path)
        )
        assert report["links"] == ["hinged_door/frame", "hinged_door/door"]
        assert joint_types(report["urdf"]) == ["revolute"]
        assert report["movable_joints"] == report["joints"]

        joint = ElementTree.parse(report["urdf"]).getroot().find("joint")
        assert joint.find("axis").get("xyz") == "1.0 0.0 0.0"
        limit = joint.find("limit")
        assert limit.get("lower") == "-1.4"
        assert limit.get("upper") == "1.4"
        assert limit.get("velocity") == "2.5"

    def test_a_prismatic_joint_and_a_continuous_joint_are_told_apart(
        self, gazebo_paths, tmp_path
    ):
        """
        ``drawer.sdf`` declares "slide" as prismatic and "spin" as continuous, but
        GazeboParser maps both "revolute" and "continuous" SDF types to the same
        RevoluteConnection class; the continuous joint's declared limits are then
        dropped by GazeboParser itself.

        The bundler must recover "continuous" from the resulting bare RevoluteConnection
        rather than from the source text.
        """
        report = bundler.bundle_gazebo_world(
            gazebo_paths.drawer, "drawer", str(tmp_path)
        )
        assert sorted(joint_types(report["urdf"])) == ["continuous", "prismatic"]

        root = ElementTree.parse(report["urdf"]).getroot()
        continuous_joint = next(
            joint
            for joint in root.findall("joint")
            if joint.get("type") == "continuous"
        )
        assert continuous_joint.find("limit") is None
        assert continuous_joint.find("axis").get("xyz") == "0.0 1.0 0.0"

        prismatic_joint = next(
            joint for joint in root.findall("joint") if joint.get("type") == "prismatic"
        )
        limit = prismatic_joint.find("limit")
        assert limit.get("lower") == "0.0"
        assert limit.get("upper") == "0.45"


# %% a composed world
class TestComposedWorld:
    def test_static_and_free_models_become_fixed_and_floating_joints(
        self, gazebo_paths, tmp_path
    ):
        """
        ``mini.world`` attaches two static shelf instances and a static inline crate
        (fixed joints) plus one non-static pallet include, which the semantic_digital
        _twin world merges in with a free Connection6DoF (a floating joint).
        """
        report = bundler.bundle_gazebo_world(
            gazebo_paths.mini_world, "mini", str(tmp_path)
        )
        assert len(report["links"]) == 5  # world root + 4 model links
        assert sorted(joint_types(report["urdf"])) == [
            "fixed",
            "fixed",
            "fixed",
            "floating",
        ]
        assert report["missing"] == []


# %% meshes
class TestMeshBundling:
    @pytest.fixture()
    def sdf_with_mesh(self, tmp_path):
        """
        A model referencing one mesh, both on disk next to each other.
        """
        (tmp_path / "cup.stl").write_text("solid cup\nendsolid cup\n")
        sdf = tmp_path / "model.sdf"
        sdf.write_text(
            '<?xml version="1.0" ?>\n'
            '<sdf version="1.6">\n'
            '  <model name="cup_model">\n'
            "    <static>1</static>\n"
            '    <link name="link">\n'
            '      <visual name="visual">\n'
            "        <geometry><mesh><uri>cup.stl</uri></mesh></geometry>\n"
            "      </visual>\n"
            "    </link>\n"
            "  </model>\n"
            "</sdf>\n"
        )
        return sdf

    def test_the_mesh_is_copied_next_to_the_written_urdf(self, sdf_with_mesh, tmp_path):
        out_dir = tmp_path / "bundle"
        report = bundler.bundle_gazebo_world(
            str(sdf_with_mesh), "cup_model", str(out_dir)
        )
        assert report["meshes_copied"] == 1
        assert report["missing"] == []
        mesh_element = ElementTree.parse(report["urdf"]).getroot().find(".//mesh")
        bundled_mesh = out_dir / mesh_element.get("filename")
        assert bundled_mesh.is_file()

    def test_a_missing_source_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bundler.bundle_gazebo_world(
                str(tmp_path / "gone.world"), "demo", str(tmp_path)
            )

    def test_a_texture_in_a_sibling_materials_directory_is_bundled(self, tmp_path):
        """
        Gazebo models keep textures in ``materials/textures`` next to ``meshes``, and
        their COLLADA files reference them as ``../materials/textures/<name>.png``.

        The bundle must mirror that layout so the browser's relative lookup from the
        bundled mesh finds the texture.
        """
        model_directory = tmp_path / "crate_model"
        (model_directory / "meshes").mkdir(parents=True)
        (model_directory / "materials" / "textures").mkdir(parents=True)
        (model_directory / "meshes" / "crate.dae").write_text(
            "<library_images><init_from>"
            "../materials/textures/crate.png"
            "</init_from></library_images>\n"
        )
        (model_directory / "materials" / "textures" / "crate.png").write_bytes(
            b"not really a png"
        )
        sdf = model_directory / "model.sdf"
        sdf.write_text(
            '<?xml version="1.0" ?>\n'
            '<sdf version="1.6">\n'
            '  <model name="crate_model">\n'
            "    <static>1</static>\n"
            '    <link name="link">\n'
            '      <visual name="visual">\n'
            "        <geometry><mesh><uri>meshes/crate.dae</uri></mesh></geometry>\n"
            "      </visual>\n"
            "    </link>\n"
            "  </model>\n"
            "</sdf>\n"
        )
        out_dir = tmp_path / "bundle"
        report = bundler.bundle_gazebo_world(str(sdf), "crate_model", str(out_dir))
        assert (
            out_dir / "meshes" / "gazebo" / "materials" / "textures" / "crate.png"
        ).is_file()
        assert ".png" in report["mesh_exts"]
