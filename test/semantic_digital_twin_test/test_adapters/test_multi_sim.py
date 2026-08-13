import logging
import os
import threading
import time

import mujoco
import pytest
import numpy
from PIL.Image import Image
from PIL import Image

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
    Color,
    Cylinder,
    Mesh,
    Texture,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Region, Actuator

from physics_simulators.mujoco_simulator import MujocoSimulator
from physics_simulators.base_simulator import SimulatorState
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import (
    MujocoSim,
    MujocoActuator,
    MujocoBody,
    MujocoSynchronizer,
)
from semantic_digital_twin.adapters.multi_sim import (
    MujocoSim,
    MujocoActuator,
    MujocoBuilder,
    MujocoLight,
)

urdf_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "urdf",
)
mjcf_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "mjcf",
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

headless = os.environ.get("CI", "false").lower() == "true"
only_run_test_in_CI = os.environ.get("CI", "false").lower() == "false"

pytestmark = pytest.mark.skipif(
    only_run_test_in_CI,
    reason="Only run test in CI or multisim could not be imported.",
)

TEST_URDF_1 = os.path.normpath(os.path.join(urdf_dir, "simple_two_arm_robot.urdf"))
TEST_URDF_2 = HSRB.get_ros_file_path()
TEST_URDF_TRACY = Tracy.get_ros_file_path()
TEST_MJCF_1 = os.path.normpath(os.path.join(mjcf_dir, "mjx_single_cube_no_mesh.xml"))
TEST_MJCF_2 = os.path.normpath(os.path.join(mjcf_dir, "jeroen_cups.xml"))
STEP_SIZE = 1e-3


def stop_multisim_if_running(multi_sim: MujocoSim) -> None:
    simulator = getattr(multi_sim, "simulator", None)
    if simulator is None:
        return
    if getattr(simulator, "state", None) is SimulatorState.STOPPED:
        return
    multi_sim.stop_simulation()


@pytest.fixture
def test_urdf_1_world():
    return URDFParser.from_file(file_path=TEST_URDF_1).parse()


@pytest.fixture
def test_mjcf_1_world():
    return MJCFParser(TEST_MJCF_1).parse()


@pytest.fixture
def test_mjcf_2_world():
    return MJCFParser(TEST_MJCF_2).parse()


def test_empty_multi_sim_in_5s():
    world = World()
    multi_sim = MujocoSim(world=world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_world_multi_sim_in_5s(test_urdf_1_world):
    multi_sim = MujocoSim(world=test_urdf_1_world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_apartment_multi_sim_in_5s():
    try:
        test_urdf_2_world = URDFParser.from_file(file_path=TEST_URDF_2).parse()
    except ParsingError:
        pytest.skip("Skipping HSRB krrood_test due to URDF parsing error.")

    multi_sim = MujocoSim(world=test_urdf_2_world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_world_multi_sim_with_change(test_urdf_1_world):
    multi_sim = MujocoSim(world=test_urdf_1_world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        time.sleep(1.0)

        start_time = time.time()

        new_body = Body(name=PrefixedName("test_body"))
        box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.2, y=0.4, z=3.0, roll=0, pitch=0.5, yaw=0, reference_frame=new_body
        )
        box = Box(
            origin=box_origin,
            scale=Scale(1.0, 1.5, 0.5),
            color=Color(1.0, 0.0, 0.0, 1.0),
        )
        new_body.collision = ShapeCollection([box], reference_frame=new_body)

        logger.debug(f"Time before adding new body: {time.time() - start_time}s")
        with test_urdf_1_world.modify_world():
            test_urdf_1_world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=test_urdf_1_world,
                    parent=test_urdf_1_world.root,
                    child=new_body,
                )
            )
        logger.debug(f"Time after adding new body: {time.time() - start_time}s")

        assert new_body.name.name in multi_sim.simulator.get_all_body_names().result

        time.sleep(0.5)

        region = Region(name=PrefixedName("test_region"))
        region_box = Box(
            scale=Scale(0.1, 0.5, 0.2),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=region),
            color=Color(0.0, 1.0, 0.0, 0.8),
        )
        region.area = ShapeCollection([region_box], reference_frame=region)

        logger.debug(f"Time before add adding region: {time.time() - start_time}s")
        with test_urdf_1_world.modify_world():
            test_urdf_1_world.add_connection(
                FixedConnection(
                    parent=test_urdf_1_world.root,
                    child=region,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=0.5
                    ),
                )
            )
        logger.debug(f"Time after add adding region: {time.time() - start_time}s")

        assert region.name.name in multi_sim.simulator.get_all_body_names().result

        time.sleep(0.5)

        T_const = 0.1
        kp = 100
        kv = 10
        actuator = Actuator()
        dof = test_urdf_1_world.get_degree_of_freedom_by_name(name="r_joint_1")
        actuator.add_dof(dof=dof)
        actuator.simulator_additional_properties.append(
            MujocoActuator(
                dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                dynamics_parameters=[T_const] + [0.0] * 9,
                gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                gain_parameters=[kp] + [0.0] * 9,
                bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                bias_parameters=[0, -kp, -kv] + [0.0] * 7,
            )
        )

        logger.debug(f"Time before adding new actuator: {time.time() - start_time}s")
        with test_urdf_1_world.modify_world():
            test_urdf_1_world.add_actuator(actuator=actuator)
        logger.debug(f"Time after adding new actuator: {time.time() - start_time}s")

        assert actuator.name.name in multi_sim.simulator.get_all_actuator_names().result

        time.sleep(4.0)
        multi_sim.stop_simulation()
    finally:
        stop_multisim_if_running(multi_sim)


def test_multi_sim_in_5s(test_mjcf_1_world):
    multi_sim = MujocoSim(
        world=test_mjcf_1_world,
        headless=headless,
        step_size=STEP_SIZE,
    )

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_mesh_scale_and_equality(test_mjcf_2_world):
    multi_sim = MujocoSim(
        world=test_mjcf_2_world,
        headless=headless,
        step_size=STEP_SIZE,
    )

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def _write_textured_tetrahedron(directory, texture_color) -> str:
    """
    Writes a minimal textured OBJ+MTL+PNG mesh (a tetrahedron, so its convex hull is
    non-degenerate) into ``directory``, textured with a solid ``texture_color``, and returns
    the OBJ file's path. Always named "tetra.obj"/"tetra.mtl"/"wood.png", so callers writing
    into different directories can reproduce a texture basename collision between them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=texture_color).save(directory / "wood.png")
    (directory / "tetra.mtl").write_text("newmtl wood\nmap_Kd wood.png\n")
    mesh_file = directory / "tetra.obj"
    mesh_file.write_text(
        "mtllib tetra.mtl\n"
        "o tetra\n"
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "v 0.0 0.0 1.0\n"
        "vt 0.0 0.0\n"
        "vt 1.0 0.0\n"
        "vt 0.0 1.0\n"
        "vt 0.5 0.5\n"
        "usemtl wood\n"
        "f 1/1 2/2 3/3\n"
        "f 1/1 2/2 4/4\n"
        "f 1/1 3/3 4/4\n"
        "f 2/2 3/3 4/4\n"
    )
    return str(mesh_file)


def _build_world_with_two_textured_bodies(
    tmp_path, mesh_file_a: str, mesh_file_b: str
) -> MujocoBuilder:
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        for name, mesh_file in [("quad_0", mesh_file_a), ("quad_1", mesh_file_b)]:
            mesh_shape = Mesh(filename=mesh_file, scale=Scale(1, 1, 1))
            body = Body(
                name=PrefixedName(name),
                visual=ShapeCollection([mesh_shape]),
                collision=ShapeCollection([mesh_shape]),
            )
            world.add_kinematic_structure_entity(body)
            world.add_connection(FixedConnection(parent=root, child=body))

    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))
    return builder


def test_builder_assigns_material_to_every_geom_sharing_a_texture(tmp_path):
    """
    Regression test: MujocoBuilder._parse_geom used to return early - without ever setting
    geom_props["material"] - whenever a geom's texture was already registered by an earlier
    geom. Since most textures in a scene are shared across many geoms (a real RoboCasa
    kitchen reuses a handful of textures across ~90 meshes), this meant only the first geom
    to use a given texture ever got a material; every later reuse silently rendered with
    MuJoCo's default (untextured, gray) material instead.
    """
    mesh_file = _write_textured_tetrahedron(tmp_path, texture_color=(120, 60, 20))

    builder = _build_world_with_two_textured_bodies(tmp_path, mesh_file, mesh_file)

    materials = {
        body.name: geom.material for body in builder.spec.bodies for geom in body.geoms
    }
    assert materials["quad_0"] == materials["quad_1"]
    assert materials["quad_0"] != ""


def test_builder_does_not_confuse_different_textures_sharing_a_basename(tmp_path):
    """
    Regression test: RoboCasa's asset pipeline reuses generic texture basenames (e.g.
    "T_BC001.png") across many unrelated fixtures' own distinct texture files - a real
    kitchen had 14 different fixtures (sink, stove, fridge, dishwasher, ...) all using a
    texture file named exactly "T_BC001.png" in their own directories. Deduplicating by
    basename alone collapsed all of them onto whichever fixture's texture was registered
    first, so most fixtures rendered with the wrong (borrowed) texture image instead of
    their own.
    """
    mesh_file_a = _write_textured_tetrahedron(
        tmp_path / "fixture_a", texture_color=(200, 0, 0)
    )
    mesh_file_b = _write_textured_tetrahedron(
        tmp_path / "fixture_b", texture_color=(0, 200, 0)
    )

    builder = _build_world_with_two_textured_bodies(tmp_path, mesh_file_a, mesh_file_b)

    materials = {
        body.name: geom.material for body in builder.spec.bodies for geom in body.geoms
    }
    assert materials["quad_0"] != materials["quad_1"]
    texture_files = {texture.name: texture.file for texture in builder.spec.textures}
    assert len(texture_files) == 2


def test_builder_writes_a_light_attached_to_a_body(tmp_path):
    """
    Regression test: MujocoBuilder had no handling for MujocoLight additional properties at
    all, so a world's lights were silently dropped when built into a MuJoCo scene - every
    recorded/simulated world fell back to MuJoCo's minimal default camera headlight instead
    of the scene's own intended lighting.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        root.simulator_additional_properties.append(
            MujocoLight(
                name="overview_light",
                body=root,
                directional=True,
                position=[2.0, -2.0, 2.0],
                direction=[0.0, 0.0, -1.0],
                ambient=[0.3, 0.3, 0.3],
                diffuse=[0.5, 0.5, 0.5],
                specular=[0.3, 0.3, 0.3],
            )
        )

    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))

    [light] = [light for body in builder.spec.bodies for light in body.lights]
    assert light.name == "overview_light"
    assert list(light.pos) == pytest.approx([2.0, -2.0, 2.0])
    assert list(light.ambient) == pytest.approx([0.3, 0.3, 0.3])
    assert list(light.diffuse) == pytest.approx([0.5, 0.5, 0.5])


def test_builder_assigns_material_to_a_textured_primitive_shape(tmp_path):
    """
    Regression test: Box/Sphere/Cylinder shapes never carried any texture reference, only a
    flat Color - RoboCasa's countertops and cabinet doors are actual MJCF box geoms with a
    material referencing a marble/wood texture, so this whole texture reference was silently
    discarded on every round-trip and they rendered flat-colored instead of textured.
    """
    texture_directory = tmp_path / "textures"
    texture_directory.mkdir()
    texture_file = texture_directory / "marble.png"
    Image.new("RGB", (4, 4), color=(200, 200, 200)).save(texture_file)

    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        box_shape = Box(
            scale=Scale(1, 1, 1),
            texture=Texture(
                file_path=str(texture_file), repeat=(3.0, 3.0), uniform=True
            ),
        )
        counter = Body(
            name=PrefixedName("counter"),
            visual=ShapeCollection([box_shape]),
            collision=ShapeCollection([box_shape]),
        )
        world.add_kinematic_structure_entity(counter)
        world.add_connection(FixedConnection(parent=root, child=counter))

    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))

    [geom] = [
        geom
        for body in builder.spec.bodies
        for geom in body.geoms
        if body.name == "counter"
    ]
    assert geom.material != ""
    [material] = [
        material
        for material in builder.spec.materials
        if material.name == geom.material
    ]
    assert list(material.texrepeat) == pytest.approx([3.0, 3.0])
    assert bool(material.texuniform) is True
    texture_name = material.textures[0]
    assert texture_name != ""
    [texture] = [
        texture for texture in builder.spec.textures if texture.name == texture_name
    ]
    assert texture.file == str(texture_file)


def test_mujoco_with_tracy_dae_files():
    try:
        dae_world = URDFParser.from_file(file_path=TEST_URDF_TRACY).parse()
    except ParsingError:
        pytest.skip("Skipping tracy test due to URDF parsing error.")

    multi_sim = MujocoSim(world=dae_world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_mujocosim_world_with_added_objects(test_urdf_1_world):
    milk_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "stl",
        "milk.stl",
    )
    stl_parser = STLParser(milk_path)
    mesh_world = stl_parser.parse()
    transformation = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.5, reference_frame=test_urdf_1_world.root
    )

    with test_urdf_1_world.modify_world():
        test_urdf_1_world.merge_world_at_pose(mesh_world, transformation)

    multi_sim = MujocoSim(world=test_urdf_1_world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_spawn_body_with_connections():
    def spawn_robot_body(spawn_world: World) -> Body:
        spawn_body = Body(name=PrefixedName("robot"))
        box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0, y=0, z=0.5, roll=0, pitch=0, yaw=0, reference_frame=spawn_body
        )
        box = Box(
            origin=box_origin,
            scale=Scale(0.4, 0.4, 1.0),
            color=Color(0.9, 0.9, 0.9, 1.0),
        )
        spawn_body.collision = ShapeCollection([box], reference_frame=spawn_body)

        with spawn_world.modify_world():
            spawn_world.add_connection(
                FixedConnection(parent=spawn_world.root, child=spawn_body)
            )

        return spawn_body

    def spawn_shoulder_bodies(spawn_world: World, root_body: Body) -> tuple[Body, Body]:
        spawn_left_shoulder_body = Body(name=PrefixedName("left_shoulder"))
        cylinder = Cylinder(
            width=0.2,
            height=0.1,
            color=Color(0.9, 0.1, 0.1, 1.0),
        )
        spawn_left_shoulder_body.collision = ShapeCollection(
            [cylinder], reference_frame=spawn_left_shoulder_body
        )
        dof = DegreeOfFreedom(name=PrefixedName("left_shoulder_joint"))
        left_shoulder_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0,
            pos_y=0.3,
            pos_z=0.9,
            quat_w=0.707,
            quat_x=0.707,
            quat_y=0,
            quat_z=0,
        )

        with spawn_world.modify_world():
            spawn_world.add_degree_of_freedom(dof)
            spawn_world.add_connection(
                RevoluteConnection(
                    name=dof.name,
                    parent=root_body,
                    child=spawn_left_shoulder_body,
                    axis=Vector3.Z(reference_frame=spawn_left_shoulder_body),
                    raw_dof=dof,
                    parent_T_connection_expression=left_shoulder_origin,
                )
            )

        spawn_right_shoulder_body = Body(name=PrefixedName("right_shoulder"))
        cylinder = Cylinder(
            width=0.2,
            height=0.1,
            color=Color(0.9, 0.1, 0.1, 1.0),
        )
        spawn_right_shoulder_body.collision = ShapeCollection(
            [cylinder], reference_frame=spawn_right_shoulder_body
        )
        dof = DegreeOfFreedom(name=PrefixedName("right_shoulder_joint"))
        right_shoulder_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0,
            pos_y=-0.3,
            pos_z=0.9,
            quat_w=0.707,
            quat_x=0.707,
            quat_y=0,
            quat_z=0,
        )

        with spawn_world.modify_world():
            spawn_world.add_degree_of_freedom(dof)
            spawn_world.add_connection(
                RevoluteConnection(
                    name=dof.name,
                    parent=root_body,
                    child=spawn_right_shoulder_body,
                    axis=Vector3.Z(reference_frame=spawn_right_shoulder_body),
                    raw_dof=dof,
                    parent_T_connection_expression=right_shoulder_origin,
                )
            )

        return spawn_left_shoulder_body, spawn_right_shoulder_body

    world = World()
    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=0.001,
    )

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        robot_body = spawn_robot_body(spawn_world=world)
        spawn_shoulder_bodies(spawn_world=world, root_body=robot_body)

        time.sleep(1)

        assert set(multi_sim.simulator.get_all_body_names().result) == {
            "world",
            "robot",
            "left_shoulder",
            "right_shoulder",
        }

        multi_sim.stop_simulation()
    finally:
        stop_multisim_if_running(multi_sim)


def test_body_frame_excludes_joint_state_at_build_time():
    """A body's static frame must be built at the reference (zero-joint) pose.

    The joint is non-zero while the simulator is built and is evaluated at a
    different angle, so a frame that baked in the build-time angle would have it
    applied twice and drift away from the world forward kinematics.
    """
    world = World()
    base_body = Body(name=PrefixedName("base"))
    rotated_link = Body(name=PrefixedName("rotated_link"))
    # A tip offset from the joint axis, so a rotation actually moves its position
    # (the joint child sits on the axis and would not reveal the bug).
    tip_link = Body(name=PrefixedName("tip"))
    rotated_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.3,
        pos_y=0.0,
        pos_z=0.9,
        quat_w=0.707,
        quat_x=0.707,
        quat_y=0.0,
        quat_z=0.0,
    )
    tip_offset = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5, y=0.2, z=0.0)
    rotated_joint_dof = DegreeOfFreedom(name=PrefixedName("rotated_joint"))
    with world.modify_world():
        world.add_body(base_body)
        world.add_degree_of_freedom(rotated_joint_dof)
        world.add_connection(
            RevoluteConnection(
                name=rotated_joint_dof.name,
                parent=base_body,
                child=rotated_link,
                axis=Vector3.Z(reference_frame=rotated_link),
                raw_dof=rotated_joint_dof,
                parent_T_connection_expression=rotated_origin,
            )
        )
        world.add_connection(
            FixedConnection(
                parent=rotated_link,
                child=tip_link,
                parent_T_connection_expression=tip_offset,
            )
        )

    build_time_angle = 0.7
    with world.modify_world():
        world.state[rotated_joint_dof.id].position = build_time_angle

    multi_sim = MujocoSim(world=world, headless=headless, step_size=0.001)
    try:
        evaluation_angle = 0.3
        with world.modify_world():
            world.state[rotated_joint_dof.id].position = evaluation_angle

        mujoco_model = multi_sim.simulator._mj_model
        mujoco_data = multi_sim.simulator._mj_data
        joint_id = mujoco.mj_name2id(
            mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, rotated_joint_dof.name.name
        )
        mujoco_data.qpos[mujoco_model.jnt_qposadr[joint_id]] = evaluation_angle
        mujoco.mj_forward(mujoco_model, mujoco_data)

        simulated_position = multi_sim.simulator.get_body_position(
            tip_link.name.name
        ).result[:3]
        world_position = world.compute_forward_kinematics_np(world.root, tip_link)[
            :3, 3
        ]
        numpy.testing.assert_allclose(simulated_position, world_position, atol=1e-4)
    finally:
        stop_multisim_if_running(multi_sim)


def test_world_sim_state_sync():
    plane_half_thickness = 0.05
    box_half_size = 0.1
    init_pos = numpy.array([0.3, 0.2, 5.0])
    target_pos = numpy.array(
        [init_pos[0], init_pos[1], plane_half_thickness + box_half_size]
    )

    def spawn_state_sync_scene(
        spawn_world: World,
    ) -> tuple[Body, Connection6DoF]:
        plane_body = Body(name=PrefixedName("ground_plane"))
        plane_body.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        reference_frame=plane_body
                    ),
                    scale=Scale(2.0, 2.0, plane_half_thickness * 2),
                    color=Color(1.0, 1.0, 0.0, 1.0),
                )
            ],
            reference_frame=plane_body,
        )

        falling_box = Body(name=PrefixedName("falling_box"))
        falling_box.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        reference_frame=falling_box
                    ),
                    scale=Scale(
                        box_half_size * 2, box_half_size * 2, box_half_size * 2
                    ),
                    color=Color(1.0, 0.0, 0.0, 1.0),
                )
            ],
            reference_frame=falling_box,
        )

        with spawn_world.modify_world():
            spawn_world.add_connection(
                FixedConnection(parent=spawn_world.root, child=plane_body)
            )
            box_connection = Connection6DoF.create_with_dofs(
                world=spawn_world,
                parent=spawn_world.root,
                child=falling_box,
            )
            spawn_world.add_connection(box_connection)
        return falling_box, box_connection

    world = World()
    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=STEP_SIZE,
    )

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        falling_box, box_connection = spawn_state_sync_scene(world)

        body_names = multi_sim.simulator.get_all_body_names().result
        assert {"ground_plane", "falling_box"}.issubset(
            body_names
        ), f"scene bodies were not spawned in the simulator; bodies={body_names}"

        box_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=float(init_pos[0]),
            y=float(init_pos[1]),
            z=float(init_pos[2]),
            reference_frame=falling_box,
        )
        time.sleep(2.5)

        final_pos = numpy.asarray(
            multi_sim.simulator.get_body_position("falling_box").result[:3],
            dtype=float,
        )

        multi_sim.stop_simulation()

        assert numpy.allclose(final_pos, target_pos, atol=1e-1), (
            f"Box did not settle at target: final_pos={final_pos}, "
            f"expected≈{target_pos}"
        )
    finally:
        stop_multisim_if_running(multi_sim)


def test_attach_node_style_reparent_welds_body_in_mujoco():
    """
    Re-parenting a body in the world model the same way AttachNode/DetachNode
    do (remove its old connection, add a new one, inside a single
    modify_world() block) must also weld/un-weld it in MuJoCo's own
    kinematic tree, not just in the world model -- otherwise a body MuJoCo is
    genuinely, physically simulating (e.g. held only by real contact/
    friction) keeps behaving as an independent free body, oblivious to being
    "attached", and gets left behind the instant whatever carries it moves.

    Proven directly: after the AttachNode-style re-parent, moving the
    "handle" body (via a normal kinematically-teleported connection, exactly
    like the arm's own joints) must move the attached box along with it in
    MuJoCo -- not just in the world model. After a DetachNode-style
    re-parent back to world.root, moving the handle again must NOT move the
    box anymore.
    """
    kp = 2000
    kv = 200

    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        base = Body(name=PrefixedName("handle_base"))
        handle = Body(name=PrefixedName("handle"))
        handle.collision = ShapeCollection(
            [Box(origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=handle),
                 scale=Scale(0.05, 0.05, 0.05), color=Color(0.2, 0.2, 0.8, 1.0))],
            reference_frame=handle,
        )
        box = Body(name=PrefixedName("attachable_box"))
        box.collision = ShapeCollection(
            [Box(origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=box),
                 scale=Scale(0.04, 0.04, 0.04), color=Color(0.8, 0.2, 0.2, 1.0))],
            reference_frame=box,
        )
        dof = DegreeOfFreedom(name=PrefixedName("handle_joint"))

        with world.modify_world():
            world.add_connection(FixedConnection(parent=world.root, child=base))

            handle_connection = PrismaticConnection(
                name=dof.name,
                parent=base,
                child=handle,
                axis=Vector3.Z(reference_frame=handle),
                raw_dof=dof,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=base
                ),
            )
            world.add_degree_of_freedom(dof)
            world.add_connection(handle_connection)

            # A real PD actuator, matching how the arm's own joints hold
            # position via actual force -- without one, the joint has
            # nothing opposing gravity and free-falls from creation, which
            # the arm's actuated joints never do.
            actuator = Actuator()
            actuator.add_dof(dof=dof)
            actuator.simulator_additional_properties.append(
                MujocoActuator(
                    dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                    dynamics_parameters=[0.0] * 10,
                    gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                    gain_parameters=[kp] + [0.0] * 9,
                    bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                    bias_parameters=[0, -kp, -kv] + [0.0] * 7,
                )
            )
            world.add_actuator(actuator=actuator)

            box_connection = Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=0.2, reference_frame=world.root
                ),
            )
            world.add_connection(box_connection)

        time.sleep(1)

        # AttachNode-style re-parent: remove the box's free joint, add a
        # FixedConnection to the handle -- both inside one modify_world().
        with world.modify_world():
            world.remove_connection(box.parent_connection)
            world.add_connection(
                FixedConnection(
                    parent=handle,
                    child=box,
                    parent_T_connection_expression=world.compute_forward_kinematics(
                        handle, box
                    ),
                )
            )
        time.sleep(0.5)

        # Snapshots must be real copies: get_body_position().result is a
        # live view into MuJoCo's own data buffer, continuously overwritten
        # by the physics thread -- numpy.asarray() with a matching dtype
        # does not copy, so "before"/"after" would otherwise silently alias
        # the same, ever-changing memory and the delta would always read
        # as [0, 0, 0] regardless of what actually happened.
        handle_pos_before = numpy.array(
            multi_sim.simulator.get_body_position("handle").result[:3],
            dtype=float, copy=True,
        )
        box_pos_before = numpy.array(
            multi_sim.simulator.get_body_position("attachable_box").result[:3],
            dtype=float, copy=True,
        )

        handle_connection.position = 0.15
        time.sleep(1)

        handle_pos_after = numpy.array(
            multi_sim.simulator.get_body_position("handle").result[:3],
            dtype=float, copy=True,
        )
        box_pos_after = numpy.array(
            multi_sim.simulator.get_body_position("attachable_box").result[:3],
            dtype=float, copy=True,
        )

        handle_delta = handle_pos_after - handle_pos_before
        box_delta = box_pos_after - box_pos_before
        assert numpy.allclose(handle_delta, box_delta, atol=0.02), (
            f"box did not move with the handle after being attached in MuJoCo: "
            f"handle moved by {handle_delta}, box moved by {box_delta}"
        )

        # A welded child has no joint of its own in MuJoCo at all (attach()
        # adds it as a static child body) -- check this directly rather than
        # via position deltas, since the box and handle geometries are
        # vertically stacked and a mere resting contact (no weld at all)
        # can coincidentally reproduce the same delta.
        attached_joints = multi_sim.simulator.get_body_joints("attachable_box").result
        assert attached_joints == [], (
            f"box still has its own MuJoCo joint after being attached: {attached_joints}"
        )

        # DetachNode-style re-parent back to world.root: un-weld it again.
        with world.modify_world():
            world.remove_connection(box.parent_connection)
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=box,
                    parent_T_connection_expression=world.compute_forward_kinematics(
                        world.root, box
                    ),
                )
            )
        time.sleep(0.5)

        multi_sim.stop_simulation()

        detached_joints = multi_sim.simulator.get_body_joints("attachable_box").result
        assert len(detached_joints) == 1 and (
            detached_joints[0].type == mujoco.mjtJoint.mjJNT_FREE
        ), (
            "box did not get a free joint back after being detached to "
            f"world.root, still welded in MuJoCo: joints={detached_joints}"
        )
    finally:
        stop_multisim_if_running(multi_sim)


def test_actuated_joint_ctrl_tracks_commanded_qpos():
    """
    Writing a new commanded position into world.state for a DOF that is
    driven by a strong PD actuator must move the actuator's ctrl setpoint
    along with it. If ctrl is left stale, MuJoCo's actuator keeps servoing
    toward the old setpoint and fights every subsequent qpos write, so the
    joint never settles at the commanded position (it oscillates instead).
    """
    kp = 2000
    kv = 200
    target = 1.0

    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        base = Body(name=PrefixedName("actuated_base"))
        link = Body(name=PrefixedName("actuated_link"))
        link.collision = ShapeCollection(
            [Cylinder(width=0.05, height=0.3, color=Color(0.5, 0.5, 0.5, 1.0))],
            reference_frame=link,
        )
        dof = DegreeOfFreedom(name=PrefixedName("actuated_joint"))

        with world.modify_world():
            world.add_connection(FixedConnection(parent=world.root, child=base))
            world.add_degree_of_freedom(dof)
            connection = RevoluteConnection(
                name=dof.name,
                parent=base,
                child=link,
                axis=Vector3.Z(reference_frame=link),
                raw_dof=dof,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=base
                ),
            )
            world.add_connection(connection)

            actuator = Actuator()
            actuator.add_dof(dof=dof)
            actuator.simulator_additional_properties.append(
                MujocoActuator(
                    dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                    dynamics_parameters=[0.0] * 10,
                    gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                    gain_parameters=[kp] + [0.0] * 9,
                    bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                    bias_parameters=[0, -kp, -kv] + [0.0] * 7,
                )
            )
            world.add_actuator(actuator=actuator)

        time.sleep(1)

        connection.position = target
        time.sleep(2)

        final_position = multi_sim.simulator.get_joint_value(
            dof.name.name
        ).result

        multi_sim.stop_simulation()

        assert numpy.isclose(final_position, target, atol=0.05), (
            f"Joint did not settle at commanded position: got {final_position}, "
            f"expected {target}. The actuator's ctrl setpoint is likely stale "
            "and fighting the qpos write."
        )
    finally:
        stop_multisim_if_running(multi_sim)


def test_ctrl_for_position_matches_actuator_affine_equilibrium():
    """
    _ctrl_for_position must solve MuJoCo's affine actuator equation
    (force = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*length + biasprm[2]*velocity)
    for the zero-force ctrl setpoint at a given position, not just copy the
    position through. For a direct per-joint actuator (gainprm=biasprm chosen
    so ctrl and position share units) this happens to reduce to ctrl ==
    position, but for a tendon-driven actuator remapping to a different
    control range (like the Panda gripper's 0-0.04m -> 0-255 ctrl range) it
    must not.
    """
    arm_actuator = Actuator()
    arm_actuator.simulator_additional_properties.append(
        MujocoActuator(
            bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
            bias_parameters=[0, -2000, -200] + [0.0] * 7,
            gain_type=mujoco.mjtGain.mjGAIN_FIXED,
            gain_parameters=[2000] + [0.0] * 9,
        )
    )
    for position in (0.0, 0.5, -0.3):
        assert numpy.isclose(
            MujocoSynchronizer._ctrl_for_position(arm_actuator, position), position
        )

    gripper_actuator = Actuator()
    gripper_actuator.simulator_additional_properties.append(
        MujocoActuator(
            bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
            bias_parameters=[0, -100, -10] + [0.0] * 7,
            gain_type=mujoco.mjtGain.mjGAIN_FIXED,
            gain_parameters=[0.0156863] + [0.0] * 9,
        )
    )
    for position in (0.0, 0.02, 0.04):
        expected_ctrl = 100 * position / 0.0156863
        assert numpy.isclose(
            MujocoSynchronizer._ctrl_for_position(gripper_actuator, position),
            expected_ctrl,
            rtol=1e-3,
        ), (
            f"ctrl for position {position} should be {expected_ctrl:.2f} "
            "(the tendon actuator's real gain/bias remap), not the raw position."
        )


def test_tendon_actuator_ctrl_uses_correct_unit_conversion():
    """
    Integration-level version of test_ctrl_for_position_matches_actuator_affine_equilibrium:
    a tendon-driven actuator wired up through the real MujocoSim pipeline must
    receive a correctly unit-converted ctrl value when its DOF's commanded
    position changes, not a raw copy of the position.
    """
    gain0 = 0.0156863
    bias1 = -100
    target_position = 0.02

    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        base = Body(name=PrefixedName("tendon_base"))
        link = Body(name=PrefixedName("tendon_link"))
        link.collision = ShapeCollection(
            [Cylinder(width=0.01, height=0.04, color=Color(0.5, 0.5, 0.5, 1.0))],
            reference_frame=link,
        )
        dof = DegreeOfFreedom(name=PrefixedName("tendon_joint"))

        with world.modify_world():
            world.add_connection(FixedConnection(parent=world.root, child=base))
            world.add_degree_of_freedom(dof)
            connection = PrismaticConnection(
                name=dof.name,
                parent=base,
                child=link,
                axis=Vector3.Z(reference_frame=link),
                raw_dof=dof,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=base
                ),
            )
            world.add_connection(connection)

            actuator = Actuator(name=PrefixedName("tendon_actuator"))
            actuator.add_dof(dof=dof)
            actuator.simulator_additional_properties.append(
                MujocoActuator(
                    dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                    dynamics_parameters=[0.0] * 10,
                    gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                    gain_parameters=[gain0] + [0.0] * 9,
                    bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                    bias_parameters=[0, bias1, -10] + [0.0] * 7,
                )
            )
            world.add_actuator(actuator=actuator)

        time.sleep(1)

        connection.position = target_position
        time.sleep(0.5)

        ctrl = multi_sim.simulator.get_actuator(actuator.name.name).result.ctrl[0]
        multi_sim.stop_simulation()

        expected_ctrl = -bias1 * target_position / gain0
        assert numpy.isclose(ctrl, expected_ctrl, rtol=1e-2), (
            f"ctrl was {ctrl}, expected {expected_ctrl:.2f} (the tendon "
            "actuator's own gain/bias remap of the commanded position), "
            "not a raw copy of the position."
        )
    finally:
        stop_multisim_if_running(multi_sim)


def test_physically_simulated_dof_skips_qpos_teleport():
    """
    A DOF marked as physically_simulated must not have its qpos force-written
    by the world->sim sync -- only its actuator's ctrl setpoint. The point of
    this flag is to let MuJoCo's real actuator/contact dynamics decide the
    DOF's actual position (e.g. a gripper finger stopping against a grasped
    object) instead of a kinematic snap fighting/overriding whatever physics
    would otherwise produce.
    """
    kp = 2000
    kv = 200
    target = 1.0

    world = World()
    root = Body(name=PrefixedName("world"))
    base = Body(name=PrefixedName("physsim_base"))
    link = Body(name=PrefixedName("physsim_link"))
    link.collision = ShapeCollection(
        [Cylinder(width=0.05, height=0.3, color=Color(0.5, 0.5, 0.5, 1.0))],
        reference_frame=link,
    )
    dof = DegreeOfFreedom(name=PrefixedName("physsim_joint"))

    with world.modify_world():
        world.add_body(root)
        world.add_connection(FixedConnection(parent=root, child=base))
        world.add_degree_of_freedom(dof)
        connection = RevoluteConnection(
            name=dof.name,
            parent=base,
            child=link,
            axis=Vector3.Z(reference_frame=link),
            raw_dof=dof,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=base
            ),
        )
        world.add_connection(connection)

        actuator = Actuator()
        actuator.add_dof(dof=dof)
        actuator.simulator_additional_properties.append(
            MujocoActuator(
                dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                dynamics_parameters=[0.0] * 10,
                gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                gain_parameters=[kp] + [0.0] * 9,
                bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                bias_parameters=[0, -kp, -kv] + [0.0] * 7,
            )
        )
        world.add_actuator(actuator=actuator)

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=STEP_SIZE,
        physically_simulated_dofs={dof},
    )

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        connection.position = target
        # Deliberately no settling time here: check the write itself, before
        # physics has a chance to converge the joint toward the ctrl setpoint.
        qpos_immediately_after_write = multi_sim.simulator.get_joint_value(
            dof.name.name
        ).result
        ctrl_immediately_after_write = multi_sim.simulator.get_actuator(
            actuator.name.name
        ).result.ctrl[0]

        multi_sim.stop_simulation()

        assert numpy.isclose(ctrl_immediately_after_write, target, atol=1e-6), (
            "ctrl should still track the commanded position for a "
            "physically_simulated DOF."
        )
        assert not numpy.isclose(qpos_immediately_after_write, target, atol=1e-3), (
            f"qpos was snapped to {qpos_immediately_after_write}, matching the "
            f"commanded target {target} -- physically_simulated_dofs should "
            "have skipped the qpos teleport and let physics reach it instead."
        )
    finally:
        stop_multisim_if_running(multi_sim)


def test_physically_simulated_dof_velocity_reads_back_measured_settling():
    """
    The sim->world sync must overwrite a physically_simulated DOF's *velocity*
    in ``world.state`` with the measured simulator velocity, not just its
    position.

    A controller (e.g. Giskard) writes its **commanded** velocity into
    ``world.state`` every tick. A stall detector watching those velocities
    (``JointPositionList(tolerate_stall=True)`` / ``LocalMinimumReached``)
    needs to see the joint's real, physical settling: a gripper finger
    physically stopped by a grasped object otherwise still shows the
    controller's nonzero commanded closing velocity forever, the stall is
    never detected, and every motion queued behind the grasp never starts.
    """
    kp = 2000
    kv = 200

    world = World()
    root = Body(name=PrefixedName("world"))
    base = Body(name=PrefixedName("velsync_base"))
    link = Body(name=PrefixedName("velsync_link"))
    link.collision = ShapeCollection(
        [Cylinder(width=0.05, height=0.3, color=Color(0.5, 0.5, 0.5, 1.0))],
        reference_frame=link,
    )
    dof = DegreeOfFreedom(name=PrefixedName("velsync_joint"))

    with world.modify_world():
        world.add_body(root)
        world.add_connection(FixedConnection(parent=root, child=base))
        world.add_degree_of_freedom(dof)
        connection = RevoluteConnection(
            name=dof.name,
            parent=base,
            child=link,
            axis=Vector3.Z(reference_frame=link),
            raw_dof=dof,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=base
            ),
        )
        world.add_connection(connection)

        actuator = Actuator()
        actuator.add_dof(dof=dof)
        actuator.simulator_additional_properties.append(
            MujocoActuator(
                dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                dynamics_parameters=[0.0] * 10,
                gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                gain_parameters=[kp] + [0.0] * 9,
                bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                bias_parameters=[0, -kp, -kv] + [0.0] * 7,
            )
        )
        world.add_actuator(actuator=actuator)

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=STEP_SIZE,
        physically_simulated_dofs={dof},
    )

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        stale_commanded_velocity = 0.2
        world.state[dof.id].velocity = stale_commanded_velocity
        time.sleep(0.5)

        read_back_velocity = world.state[dof.id].velocity
        assert abs(read_back_velocity) < 0.05, (
            f"world.state still shows the stale commanded velocity "
            f"{read_back_velocity} for a physically settled joint -- the "
            "sim->world sync should have overwritten it with the measured "
            "(near-zero) velocity."
        )
    finally:
        stop_multisim_if_running(multi_sim)


def test_physically_simulated_dof_ctrl_latches_commanded_increments_past_contact():
    """
    A physically_simulated DOF's actuator setpoint must accumulate the
    controller's commanded increments instead of being re-derived from the
    measurement-reset belief position.

    A controller commanding "keep pushing" against a contact (e.g. closing a
    gripper on a grasped object) writes ``measured + one_step_increment``
    into ``world.state`` each tick, because the sim->world sync resets the
    belief to the measured stall position in between. Mapping *that* belief
    straight to ``ctrl`` pins the position servo's setpoint at the contact
    surface, which means near-zero squeeze force -- the grasp cannot hold
    anything. The setpoint must instead integrate the commanded increments,
    so it latches past the contact and the servo keeps pressing.
    """
    kp = 2000
    kv = 200
    contact_position = 0.15

    world = World()
    root = Body(name=PrefixedName("world"))
    base = Body(name=PrefixedName("push_base"))
    slider = Body(name=PrefixedName("push_slider"))
    slider.collision = ShapeCollection(
        [Box(scale=Scale(0.05, 0.05, 0.05), color=Color(0.5, 0.5, 0.5, 1.0))],
        reference_frame=slider,
    )
    wall = Body(name=PrefixedName("push_wall"))
    wall.collision = ShapeCollection(
        [Box(scale=Scale(0.05, 0.05, 0.05), color=Color(0.8, 0.2, 0.2, 1.0))],
        reference_frame=wall,
    )
    dof = DegreeOfFreedom(name=PrefixedName("push_joint"))

    with world.modify_world():
        world.add_body(root)
        world.add_connection(FixedConnection(parent=root, child=base))
        world.add_connection(
            FixedConnection(
                parent=root,
                child=wall,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.2, reference_frame=root
                ),
            )
        )
        world.add_degree_of_freedom(dof)
        connection = PrismaticConnection(
            name=dof.name,
            parent=base,
            child=slider,
            axis=Vector3.X(reference_frame=slider),
            raw_dof=dof,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=base
            ),
        )
        world.add_connection(connection)

        actuator = Actuator()
        actuator.add_dof(dof=dof)
        actuator.simulator_additional_properties.append(
            MujocoActuator(
                dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                dynamics_parameters=[0.0] * 10,
                gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                gain_parameters=[kp] + [0.0] * 9,
                bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                bias_parameters=[0, -kp, -kv] + [0.0] * 7,
                force_limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                force_range=[-87, 87],
            )
        )
        world.add_actuator(actuator=actuator)

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=STEP_SIZE,
        physically_simulated_dofs={dof},
    )

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        # Mimic a measurement-fed controller: each step commands one small
        # increment past the *measured* position (the readback keeps
        # resetting the belief in between, exactly like Giskard's ticks).
        for _ in range(40):
            measured = multi_sim.simulator.get_joint_value(dof.name.name).result
            connection.position = measured + 0.005
            time.sleep(0.05)

        time.sleep(0.5)
        measured_final = multi_sim.simulator.get_joint_value(dof.name.name).result
        ctrl_final = multi_sim.simulator.get_actuator(actuator.name.name).result.ctrl[
            0
        ]

        assert measured_final < contact_position + 0.01, (
            f"slider should have physically stalled against the wall near "
            f"{contact_position}, got {measured_final} -- the scene no longer "
            "reproduces a blocked joint."
        )
        assert ctrl_final > measured_final + 0.02, (
            f"ctrl setpoint {ctrl_final} sits at the measured stall position "
            f"{measured_final} -- the commanded increments were re-derived "
            "from the measurement-reset belief instead of accumulating, so "
            "the position servo exerts no sustained push against the contact."
        )
    finally:
        stop_multisim_if_running(multi_sim)


def test_multiple_physically_simulated_dofs_track_targets_without_oscillating():
    """
    Several physically_simulated DOFs actuated at the same time (mirroring
    several of a multi-joint arm's joints being physically simulated
    simultaneously, not just one isolated DOF) must all converge to their
    commanded targets via their real PD actuators and *stay* converged, not
    merely pass through the target on their way to a sustained oscillation.

    This is the risk that made a fully physically-simulated arm (as opposed
    to kinematically teleporting it every tick) a bigger undertaking than
    making just the gripper's fingers physically simulated: several
    actuator-driven joints settling at once is a materially different
    question -- does the ctrl-sync/qpos-skip machinery
    (physically_simulated_dofs, _write_1dof_to_qpos) hold up across several
    simultaneously-actuated DOFs -- than a single joint settling in isolation.
    Each joint is mounted directly to its own fixed base (independent of the
    others) so this isolates that risk from unrelated kinematic-chain
    dynamics/inertia tuning.
    """
    kp = 2000
    kv = 200
    targets = [0.5, -0.3, 0.2]

    world = World()
    root = Body(name=PrefixedName("world"))
    dofs = []
    connections = []

    with world.modify_world():
        world.add_body(root)
        for i in range(1, 4):
            base = Body(name=PrefixedName(f"physsim_base{i}"))
            link = Body(name=PrefixedName(f"physsim_link{i}"))
            # Sized (and massed, at MuJoCo's default density) to be in the same
            # ballpark as a real Panda arm link -- kp=2000/forcerange=87 (the
            # real joint1-4 actuator values) applied to a much lighter test
            # body produces a huge angular acceleration for its tiny inertia,
            # which blows up numerically at any reasonable step_size.
            link.collision = ShapeCollection(
                [Cylinder(width=0.15, height=0.4, color=Color(0.5, 0.5, 0.5, 1.0))],
                reference_frame=link,
            )
            dof = DegreeOfFreedom(name=PrefixedName(f"physsim_joint{i}"))

            world.add_connection(FixedConnection(parent=root, child=base))
            world.add_degree_of_freedom(dof)
            connection = RevoluteConnection(
                name=dof.name,
                parent=base,
                child=link,
                axis=Vector3.X(reference_frame=link),
                raw_dof=dof,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=i * 0.5, reference_frame=base
                ),
            )
            world.add_connection(connection)
            dofs.append(dof)
            connections.append(connection)

            actuator = Actuator()
            actuator.add_dof(dof=dof)
            actuator.simulator_additional_properties.append(
                MujocoActuator(
                    dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                    dynamics_parameters=[0.0] * 10,
                    gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                    gain_parameters=[kp] + [0.0] * 9,
                    bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                    bias_parameters=[0, -kp, -kv] + [0.0] * 7,
                    # Matches the real Panda joint1-4 actuators' forcerange --
                    # without a force limit, a stiff PD gain applied to a
                    # (relatively light) test body can produce an enormous
                    # instantaneous torque and diverge numerically.
                    force_limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                    force_range=[-87, 87],
                )
            )
            world.add_actuator(actuator=actuator)

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=0.0001,
        physically_simulated_dofs=set(dofs),
        sync_rate_hz=100,
    )

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        # Ramp towards the targets via small incremental writes at roughly
        # Giskard's own ~50Hz control rate, instead of one instantaneous step
        # to the final target -- this mirrors how the real control loop
        # actually drives physically_simulated DOFs (small steps every tick).
        n_steps = 50
        for step in range(1, n_steps + 1):
            for connection, target in zip(connections, targets):
                connection.position = target * step / n_steps
            time.sleep(0.02)
        time.sleep(1)

        settled = [
            multi_sim.simulator.get_joint_value(dof.name.name).result for dof in dofs
        ]
        time.sleep(0.5)
        settled_again = [
            multi_sim.simulator.get_joint_value(dof.name.name).result for dof in dofs
        ]

        multi_sim.stop_simulation()

        for dof, target, value in zip(dofs, targets, settled):
            assert numpy.isclose(value, target, atol=0.05), (
                f"{dof.name.name} did not converge to its target: got {value}, "
                f"expected {target}."
            )
        for dof, first, second in zip(dofs, settled, settled_again):
            assert numpy.isclose(first, second, atol=0.01), (
                f"{dof.name.name} kept moving between two samples 0.5s apart "
                f"({first} -> {second}) -- it settled onto its target but did "
                "not stay settled, suggesting sustained oscillation."
            )
    finally:
        stop_multisim_if_running(multi_sim)


def _settle_gravity_loaded_cantilever(gravity_compensated: bool) -> float:
    """
    Shared setup: a single physically_simulated revolute joint (kp=2000,
    kv=200, forcerange=87 -- the real Panda joint1-4 actuator values) holding
    a horizontally-extended cantilevered link level at position 0 against
    gravity, with or without MuJoCo's gravcomp on that link. Returns the
    joint's settled steady-state position error from 0.
    """
    kp = 2000
    kv = 200

    world = World()
    root = Body(name=PrefixedName("world"))
    base = Body(name=PrefixedName("cantilever_base"))
    link = Body(name=PrefixedName("cantilever_link"))
    link.collision = ShapeCollection(
        [
            Cylinder(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.3, roll=0, pitch=1.5707963267948966, reference_frame=link
                ),
                width=0.2,
                height=0.5,
                color=Color(0.5, 0.5, 0.5, 1.0),
            )
        ],
        reference_frame=link,
    )
    if gravity_compensated:
        link.simulator_additional_properties.append(
            MujocoBody(gravitation_compensation_factor=1.0)
        )
    dof = DegreeOfFreedom(name=PrefixedName("cantilever_joint"))

    with world.modify_world():
        world.add_body(root)
        world.add_connection(FixedConnection(parent=root, child=base))
        world.add_degree_of_freedom(dof)
        connection = RevoluteConnection(
            name=dof.name,
            parent=base,
            child=link,
            axis=Vector3.Y(reference_frame=link),
            raw_dof=dof,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=base
            ),
        )
        world.add_connection(connection)

        actuator = Actuator()
        actuator.add_dof(dof=dof)
        actuator.simulator_additional_properties.append(
            MujocoActuator(
                dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                dynamics_parameters=[0.0] * 10,
                gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                gain_parameters=[kp] + [0.0] * 9,
                bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                bias_parameters=[0, -kp, -kv] + [0.0] * 7,
                force_limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                force_range=[-87, 87],
            )
        )
        world.add_actuator(actuator=actuator)

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=0.0001,
        physically_simulated_dofs={dof},
    )

    try:
        multi_sim.start_simulation()
        time.sleep(1)
        connection.position = 0.0
        time.sleep(3)
        return abs(multi_sim.simulator.get_joint_value(dof.name.name).result)
    finally:
        stop_multisim_if_running(multi_sim)


def test_gravity_compensation_keeps_a_loaded_joint_within_convergence_threshold():
    """
    A physically_simulated joint holding a gravity-loaded link (e.g. the
    Panda arm's own links, once physically simulated rather than
    kinematically teleported) settles with a steady-state position error
    from its PD actuator's gain alone -- without MuJoCo's gravcomp
    countering gravity separately, this error can exceed
    JointPositionList's default 0.01 rad convergence threshold
    (giskardpy/motion_statechart/tasks/joint_tasks.py), so a motion holding
    such a joint (e.g. ParkArmsAction) never registers as converged and
    Giskard keeps sending corrective commands indefinitely -- which also
    stalls the rest of the plan behind it.
    """
    error_without_gravcomp = _settle_gravity_loaded_cantilever(gravity_compensated=False)
    error_with_gravcomp = _settle_gravity_loaded_cantilever(gravity_compensated=True)

    assert error_without_gravcomp > 0.01, (
        f"expected the uncompensated cantilever to sag past the 0.01 rad "
        f"convergence threshold to make this test meaningful, got "
        f"{error_without_gravcomp:.4f} rad -- link/gain setup no longer "
        "produces enough gravity torque to reproduce the issue."
    )
    assert error_with_gravcomp < 0.01, (
        f"gravity-compensated joint still settled {error_with_gravcomp:.4f} rad "
        "from its target, exceeding JointPositionList's 0.01 rad convergence "
        "threshold -- gravcomp did not sufficiently cancel the sag."
    )


def test_concurrent_state_writes_and_reads_do_not_corrupt_the_simulator():
    """
    Writing commanded positions for a physically_simulated DOF (world->sim,
    triggered by world.state changes on whatever thread the control loop
    runs on) and reading the simulator's live state (e.g. get_body_position,
    used for diagnostics/monitoring) must not race with the physics thread's
    own in-flight mj_step.

    Without synchronization this is a real, observed bug: pacing the control
    loop to run at a steady rate (see GiskardExecutable.real_time_pacing)
    combined with several DOFs physically simulated made a previously rare
    race into a reliably-hit one, crashing the whole process with a native
    heap corruption (glibc's "malloc(): unaligned tcache chunk detected",
    SIGABRT) -- not a Python exception, a corrupted C-level allocator, from
    unsynchronized concurrent access to MuJoCo's live _mj_data buffers
    between the physics thread (mj_step) and other threads (world->sim
    writes, get_body_position reads). This test hammers both from separate
    threads while physics steps continuously, and must run to completion
    without the process crashing or either thread raising.
    """
    kp = 2000
    kv = 200
    target = 1.0

    world = World()
    root = Body(name=PrefixedName("world"))
    base = Body(name=PrefixedName("race_base"))
    link = Body(name=PrefixedName("race_link"))
    link.collision = ShapeCollection(
        [Cylinder(width=0.15, height=0.4, color=Color(0.5, 0.5, 0.5, 1.0))],
        reference_frame=link,
    )
    dof = DegreeOfFreedom(name=PrefixedName("race_joint"))

    with world.modify_world():
        world.add_body(root)
        world.add_connection(FixedConnection(parent=root, child=base))
        world.add_degree_of_freedom(dof)
        connection = RevoluteConnection(
            name=dof.name,
            parent=base,
            child=link,
            axis=Vector3.X(reference_frame=link),
            raw_dof=dof,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=base
            ),
        )
        world.add_connection(connection)

        actuator = Actuator()
        actuator.add_dof(dof=dof)
        actuator.simulator_additional_properties.append(
            MujocoActuator(
                dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                dynamics_parameters=[0.0] * 10,
                gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                gain_parameters=[kp] + [0.0] * 9,
                bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                bias_parameters=[0, -kp, -kv] + [0.0] * 7,
                force_limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                force_range=[-87, 87],
            )
        )
        world.add_actuator(actuator=actuator)

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=0.0001,
        physically_simulated_dofs={dof},
        sync_rate_hz=100,
    )

    errors = []
    stop_event = threading.Event()

    def hammer_reads():
        try:
            while not stop_event.is_set():
                multi_sim.simulator.get_body_position("race_link")
                multi_sim.simulator.get_joint_value(dof.name.name)
        except Exception as e:
            errors.append(e)

    try:
        multi_sim.start_simulation()
        time.sleep(0.5)

        reader_thread = threading.Thread(target=hammer_reads, daemon=True)
        reader_thread.start()

        try:
            n_steps = 100
            for step in range(1, n_steps + 1):
                connection.position = target * ((step % 20) / 20)
                time.sleep(0.01)
        except Exception as e:
            errors.append(e)
        finally:
            stop_event.set()
            reader_thread.join(timeout=5)

        final_position = multi_sim.simulator.get_joint_value(dof.name.name).result
        multi_sim.stop_simulation()

        assert not errors, f"Concurrent read/write raised: {errors}"
        assert numpy.isfinite(final_position), (
            f"Joint position became non-finite ({final_position}) -- the "
            "simulator's state was corrupted by the concurrent access."
        )
    finally:
        stop_multisim_if_running(multi_sim)


TENDON_GRIPPER_AND_CUBE_MJCF = """
<mujoco>
  <worldbody>
    <body name="finger1" pos="0.05 0 0">
      <joint name="finger_joint1" type="slide" axis="-1 0 0" range="0 0.06" />
      <geom type="box" size="0.01 0.01 0.03" friction="1 0.5 0.5" />
    </body>
    <body name="finger2" pos="-0.05 0 0">
      <joint name="finger_joint2" type="slide" axis="1 0 0" range="0 0.06" />
      <geom type="box" size="0.01 0.01 0.03" friction="1 0.5 0.5" />
    </body>
    <body name="cube" pos="0 0 0">
      <joint type="free" />
      <geom type="box" size="0.02 0.02 0.02" friction="1 0.5 0.5" />
    </body>
    <body name="floor" pos="0 0 -0.03">
      <geom type="box" size="1 1 0.01" />
    </body>
  </worldbody>
  <tendon>
    <fixed name="split">
      <joint joint="finger_joint1" coef="0.5" />
      <joint joint="finger_joint2" coef="0.5" />
    </fixed>
  </tendon>
  <actuator>
    <general name="gripper_actuator" tendon="split" biastype="affine" gainprm="0.0156863" biasprm="0 -100 -10" />
  </actuator>
</mujoco>
"""


def _close_tendon_gripper_around_cube(physically_simulated: bool):
    """
    Shared setup for the two tests below: parses a minimal two-finger,
    tendon-actuated gripper (mirroring the Panda gripper's real MJCF
    structure -- fixed tendon + a single actuator driving both joints) with
    a cube resting between the fingers, commands both fingers to a fully-
    closed target that would require passing through the cube, lets it
    settle, and returns the final finger positions and cube position.
    """
    world = MJCFParser.from_xml_string(TENDON_GRIPPER_AND_CUBE_MJCF).parse()
    finger1 = world.get_connection_by_name("finger_joint1")
    finger2 = world.get_connection_by_name("finger_joint2")
    physically_simulated_dofs = (
        {finger1.raw_dof, finger2.raw_dof} if physically_simulated else set()
    )

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=STEP_SIZE,
        physically_simulated_dofs=physically_simulated_dofs,
    )
    try:
        multi_sim.start_simulation()
        time.sleep(0.5)

        finger1.position = 0.05
        finger2.position = 0.05
        time.sleep(2)

        q1 = multi_sim.simulator.get_joint_value("finger_joint1").result
        q2 = multi_sim.simulator.get_joint_value("finger_joint2").result
        cube_pos = numpy.asarray(
            multi_sim.simulator.get_body_position("cube").result[:3], dtype=float
        )
        multi_sim.stop_simulation()
        return q1, q2, cube_pos
    finally:
        stop_multisim_if_running(multi_sim)


def test_physically_simulated_gripper_stalls_on_and_holds_cube():
    """
    With both finger DOFs marked physically_simulated, closing the gripper
    onto a cube must stall well short of the (physically unreachable, since
    it requires passing through the cube) commanded target, and the cube
    must stay essentially where it started -- proving real contact/friction,
    not a kinematic snap, is what determines the fingers' resting position
    and holds the object.
    """
    q1, q2, cube_pos = _close_tendon_gripper_around_cube(physically_simulated=True)

    assert q1 < 0.03 and q2 < 0.03, (
        f"fingers should have stalled against the cube well short of the "
        f"commanded 0.05m target, got q1={q1}, q2={q2}"
    )
    assert numpy.allclose(cube_pos, [0, 0, 0], atol=0.01), (
        f"cube should have stayed close to its starting position, held in "
        f"place by the fingers, got {cube_pos}"
    )


def test_kinematically_teleported_gripper_ignores_cube_contact():
    """
    Negative control for test_physically_simulated_gripper_stalls_on_and_holds_cube:
    with physically_simulated_dofs left empty (today's default teleport
    behavior for every DOF), the fingers must reach much closer to the
    commanded target regardless of the cube being in the way, and the cube
    gets displaced/ejected rather than held -- demonstrating that without
    physically_simulated_dofs, closing a gripper on an object does not
    produce a stable, contact-respecting grasp.
    """
    q1, q2, cube_pos = _close_tendon_gripper_around_cube(physically_simulated=False)

    assert q1 > 0.035 and q2 > 0.035, (
        f"fingers should have reached close to the commanded 0.05m target "
        f"regardless of the cube, got q1={q1}, q2={q2}"
    )
    assert not numpy.allclose(cube_pos, [0, 0, 0], atol=0.01), (
        f"cube should have been displaced/ejected by the kinematically "
        f"teleported fingers passing through it, got {cube_pos}"
    )


def test_prebuilt_world_free_body_starts_at_authored_pose():
    """
    A World that already contains a free-jointed body with a non-identity
    parent_T_connection_expression (as MJCFParser produces for e.g.
    ``<body pos="0.34 -0.14 0.02"><joint type="free"/></body>``) must keep
    that body at its authored pose when MujocoSim builds and starts the
    simulation, instead of teleporting it to the world origin.

    Also places the free body as a shallow, direct child of the root while a
    longer fixed-connection chain is inserted before it: the world's
    topologically-sorted body order groups shallow bodies before deep ones
    regardless of insertion order, while plain insertion order does not, so
    this reproduces a mismatch between the two orderings.

    A ground plane is included directly below the box's authored x/y so it
    settles at a deterministic resting height instead of free-falling
    indefinitely (there is nothing else to catch it), which would otherwise
    make the assertion depend on how many physics steps happen to run before
    it's checked.
    """
    box_half_size = 0.02
    offset = numpy.array([0.34, -0.14, box_half_size])

    world = World()
    root = Body(name=PrefixedName("world"))
    with world.modify_world():
        world.add_body(root)

        chain_root = Body(name=PrefixedName("chain_root"))
        chain_middle = Body(name=PrefixedName("chain_middle"))
        chain_tip = Body(name=PrefixedName("chain_tip"))
        world.add_connection(FixedConnection(parent=root, child=chain_root))
        world.add_connection(FixedConnection(parent=chain_root, child=chain_middle))
        world.add_connection(FixedConnection(parent=chain_middle, child=chain_tip))

        ground_plane = Body(name=PrefixedName("ground_plane"))
        ground_plane.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        reference_frame=ground_plane
                    ),
                    scale=Scale(2.0, 2.0, 0.1),
                    color=Color(1.0, 1.0, 0.0, 1.0),
                )
            ],
            reference_frame=ground_plane,
        )
        world.add_connection(
            FixedConnection(
                parent=root,
                child=ground_plane,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-0.05, reference_frame=root
                ),
            )
        )

        floating_box = Body(name=PrefixedName("floating_box"))
        floating_box.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        reference_frame=floating_box
                    ),
                    scale=Scale(
                        box_half_size * 2, box_half_size * 2, box_half_size * 2
                    ),
                    color=Color(0.9, 0.3, 0.3, 1.0),
                )
            ],
            reference_frame=floating_box,
        )
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=root,
                child=floating_box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=float(offset[0]),
                    y=float(offset[1]),
                    z=float(offset[2]),
                    reference_frame=root,
                ),
            )
        )

    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)

    try:
        multi_sim.start_simulation()
        time.sleep(1.0)

        position = numpy.asarray(
            multi_sim.simulator.get_body_position("floating_box").result[:3],
            dtype=float,
        )

        multi_sim.stop_simulation()

        assert numpy.allclose(position, offset, atol=1e-2), (
            f"Free-jointed body did not start/settle at its authored pose: "
            f"got {position}, expected {offset}. It likely spawned at the "
            "origin instead."
        )
    finally:
        stop_multisim_if_running(multi_sim)
