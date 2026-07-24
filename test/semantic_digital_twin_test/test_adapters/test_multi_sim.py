import logging
import os
import threading
import time
from dataclasses import dataclass

import mujoco
import pytest
import numpy
from PIL import Image
from scipy.spatial.transform import Rotation
from typing_extensions import List, Optional, Tuple

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    ParsingError,
    UnsupportedConnection6DoFParentError,
)
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
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
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
    MujocoBuilder,
    MujocoLight,
    MujocoSynchronizer,
    ReparentingMode,
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


def _spawn_revolute_joint(
    world: World, joint_name: str, parent: Body = None
) -> DegreeOfFreedom:
    """
    Adds a single-dof revolute joint, connected to ``parent`` (``world.root`` by
    default), and returns its :class:`DegreeOfFreedom`. ``multi_sim`` must already be
    built so ``world.root`` exists and the connection gets spawned into MuJoCo live.
    """
    body = Body(name=PrefixedName(f"{joint_name}_body"))
    dof = DegreeOfFreedom(name=PrefixedName(joint_name))
    with world.modify_world():
        world.add_degree_of_freedom(dof)
        world.add_connection(
            RevoluteConnection(
                name=dof.name,
                parent=parent or world.root,
                child=body,
                axis=Vector3.Z(reference_frame=body),
                raw_dof=dof,
            )
        )
    return dof


def _qpos_adr(multi_sim: MujocoSim, joint_name: str) -> int:
    mj_model = multi_sim.simulator._mj_model
    joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert joint_id != -1, f"joint {joint_name} not found in the compiled MuJoCo model"
    return mj_model.jnt_qposadr[joint_id]


class _RecordingLogHandler(logging.Handler):
    """
    Collects log records emitted by a specific named logger.

    ``caplog`` relies on records propagating up to the root logger, but the
    ``semantic_digital_twin``/``semantic_digital_twin.adapters.multi_sim`` loggers have
    ``propagate=False`` in this environment (likely set up by the ROS/ament logging
    integration pulled in transitively), so records never reach caplog's root-attached
    handler. Attaching a handler directly to the target logger sidesteps that.
    """

    def __init__(self, logger_name: str, level=logging.WARNING):
        super().__init__(level=level)
        self.records: list = []
        self._logger = logging.getLogger(logger_name)

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def __enter__(self) -> "_RecordingLogHandler":
        self._logger.addHandler(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._logger.removeHandler(self)

    def has_message_containing(self, text: str) -> bool:
        return any(text in record.getMessage() for record in self.records)


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


def test_omni_drive_spawn_pose_is_baked_into_static_body_frame(tmp_path):
    """
    Regression test: OmniDrive (and DifferentialDrive) never get a MuJoCo joint built for
    them (see MultiSimBuilder._ignore_connection_types), so nothing else carries their
    live x/y/yaw state into the exported scene. KinematicStructureEntityConverter._convert
    used to always read reference_origin_as_position_quaternion(), which excludes that
    live state - a robot spawned at a non-identity OmniDrive pose therefore ended up at
    the world origin in MuJoCo, while e.g. RViz (which reads the full
    origin_as_position_quaternion()) showed it at the correct spawn pose.
    """
    world = World()
    with world.modify_world():
        map_root = Body(name=PrefixedName("map"))
        world.add_body(map_root)
        robot_root = Body(name=PrefixedName("robot_root"))
        drive = OmniDrive.create_with_dofs(
            world=world, parent=map_root, child=robot_root
        )
        world.add_connection(drive)

    drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.5, y=-0.5, yaw=0.9, reference_frame=map_root
    )

    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))

    [robot_body] = [body for body in builder.spec.bodies if body.name == "robot_root"]

    expected_pose = world.compute_forward_kinematics_np(world.root, robot_root)
    expected_position = expected_pose[:3, 3]
    expected_quat_wxyz = Rotation.from_matrix(expected_pose[:3, :3]).as_quat(
        scalar_first=True
    )

    assert not numpy.allclose(expected_position, [0.0, 0.0, 0.0])
    numpy.testing.assert_allclose(list(robot_body.pos), expected_position, atol=1e-6)
    numpy.testing.assert_allclose(list(robot_body.quat), expected_quat_wxyz, atol=1e-6)


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


def test_reset_simulation_resyncs_world_state():
    """
    Regression test: MultiSim.reset_simulation used to only call simulator.reset()
    without pulling the reset state back into world.state. The sim -> world direction
    is otherwise driven only by the physics step loop, which a reset does not go
    through, so world.state kept showing the pre-reset joint value until (if ever) the
    simulation was stepped again.
    """
    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)
    try:
        dof = _spawn_revolute_joint(world, "reset_test_joint")

        world.state[dof.id].position = 1.2
        world.notify_state_change()
        assert multi_sim.simulator._mj_data.qpos[
            _qpos_adr(multi_sim, "reset_test_joint")
        ] == pytest.approx(1.2)

        multi_sim.reset_simulation()

        assert world.state[dof.id].position == pytest.approx(0.0)
    finally:
        stop_multisim_if_running(multi_sim)


def test_sim_to_world_sync_does_not_drop_concurrent_edits_during_pause_window():
    """
    Regression test: MujocoSynchronizer._sim_to_world pauses its sibling state-change
    callback, pulls qpos into world.state, then used to rebase the *entire*
    previous-state snapshot to whatever world.state held at that moment - via
    StateChangeCallback.update_previous_world_state(), a full positional copy of
    world.state.positions - before resuming. A dof that a different thread edited
    concurrently, while the callback happened to be paused, was swept up in that same
    full rebase and looked "already synced", so the edit was silently never pushed to
    MuJoCo and never retried.

    joint_a is a real MuJoCo joint that _sim_to_world actually reads on every call, so
    it always triggers the "something changed" rebase path. joint_b's own resolution
    is monkeypatched away for the duration of the _sim_to_world call (simulating, e.g.,
    a moment where a spawn is still in flight) so the loop never touches it, isolating
    the edit made to it during the pause window from being simply overwritten by the
    sim's own read of the same dof.
    """
    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)
    try:
        _spawn_revolute_joint(world, "pause_window_joint_a")
        dof_b = _spawn_revolute_joint(world, "pause_window_joint_b")
        connection_b = next(
            c for c in world.connections if getattr(c, "raw_dof", None) is dof_b
        )

        original_resolve_qpos_adr = multi_sim.synchronizer._resolve_qpos_adr
        skip_joint_b = {"active": True}

        def resolve_qpos_adr_hiding_b(connection):
            if skip_joint_b["active"] and connection is connection_b:
                return None
            return original_resolve_qpos_adr(connection)

        multi_sim.synchronizer._resolve_qpos_adr = resolve_qpos_adr_hiding_b

        original_pause = multi_sim.synchronizer._state_callback.pause

        def pause_and_make_concurrent_edit():
            original_pause()
            # simulates a different thread mutating world.state for joint_b while our
            # callback is paused, without going through notify_state_change (which
            # would be a no-op while paused anyway).
            world.state[dof_b.id].position = 0.42

        multi_sim.synchronizer._state_callback.pause = pause_and_make_concurrent_edit

        multi_sim.synchronizer.sync_rate_hz = (
            MujocoSynchronizer.UNTHROTTLED_SYNC_RATE_HZ
        )
        multi_sim.synchronizer._sim_to_world()

        # joint_b is resolvable again, matching the spawn having finished; the edit
        # made during the pause window must still be pending.
        skip_joint_b["active"] = False
        world.notify_state_change()

        assert multi_sim.simulator._mj_data.qpos[
            _qpos_adr(multi_sim, "pause_window_joint_b")
        ] == pytest.approx(0.42)
    finally:
        stop_multisim_if_running(multi_sim)


def test_world_to_sim_sync_holds_model_lock():
    """
    Regression test: MujocoSynchronizer._on_state_change used to write directly into
    _mj_data.qpos without acquiring the simulator's _model_lock - the very lock
    step_callback holds while running mj_step - so a state push from a user thread
    could race a concurrently running physics step over the same qpos array. This
    proves _on_state_change now blocks on that lock instead of writing straight
    through it.
    """
    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)
    try:
        dof = _spawn_revolute_joint(world, "lock_test_joint")

        lock_acquired_by_holder = threading.Event()
        release_lock = threading.Event()

        def hold_lock():
            with multi_sim.simulator._model_lock:
                lock_acquired_by_holder.set()
                release_lock.wait(timeout=5.0)

        holder_thread = threading.Thread(target=hold_lock)
        holder_thread.start()
        assert lock_acquired_by_holder.wait(timeout=2.0)

        state_change_finished = threading.Event()

        def push_state_change():
            world.state[dof.id].position = 0.9
            world.notify_state_change()
            state_change_finished.set()

        pusher_thread = threading.Thread(target=push_state_change)
        pusher_thread.start()

        # the pusher should be blocked on _model_lock as long as hold_lock holds it
        assert not state_change_finished.wait(timeout=0.5)

        release_lock.set()
        holder_thread.join(timeout=2.0)
        assert state_change_finished.wait(timeout=2.0)
        pusher_thread.join(timeout=2.0)
    finally:
        stop_multisim_if_running(multi_sim)


def test_sim_to_world_sync_does_not_notify_when_nothing_moved():
    """
    Regression test: MujocoSynchronizer._sim_to_world used to call
    world.notify_state_change() - bumping world.state.version and firing every
    registered state-change callback - on every throttled tick as long as at least one
    connection resolved to a MuJoCo joint, regardless of whether any value actually
    changed. On a resting scene this fired a spurious notification on every tick.
    """
    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)
    try:
        _spawn_revolute_joint(world, "resting_joint")

        multi_sim.synchronizer.sync_rate_hz = (
            MujocoSynchronizer.UNTHROTTLED_SYNC_RATE_HZ
        )
        multi_sim.synchronizer._sim_to_world()  # establish the baseline snapshot

        version_before = world.state.version
        multi_sim.synchronizer._sim_to_world()  # nothing moved in between

        assert world.state.version == version_before
    finally:
        stop_multisim_if_running(multi_sim)


def test_on_state_change_logs_when_no_mujoco_joint_is_found(monkeypatch):
    """
    Regression test: both sync directions used to silently skip a connection whenever
    _resolve_qpos_adr found no matching MuJoCo joint, with no log line - unlike the
    neighboring "unsupported connection type" branch, which does warn. A spawn failure
    or a connection/joint name mismatch was therefore invisible.
    """
    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)
    try:
        dof = _spawn_revolute_joint(world, "missing_joint_test")
        monkeypatch.setattr(
            multi_sim.synchronizer, "_resolve_qpos_adr", lambda connection: None
        )

        with _RecordingLogHandler(
            "semantic_digital_twin.adapters.multi_sim"
        ) as log_handler:
            world.state[dof.id].position = 0.3
            world.notify_state_change()

        assert log_handler.has_message_containing("no MuJoCo joint found")
    finally:
        stop_multisim_if_running(multi_sim)


def test_sim_to_world_logs_when_no_mujoco_joint_is_found(monkeypatch):
    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)
    try:
        _spawn_revolute_joint(world, "missing_joint_test_2")
        monkeypatch.setattr(
            multi_sim.synchronizer, "_resolve_qpos_adr", lambda connection: None
        )

        multi_sim.synchronizer.sync_rate_hz = (
            MujocoSynchronizer.UNTHROTTLED_SYNC_RATE_HZ
        )
        with _RecordingLogHandler(
            "semantic_digital_twin.adapters.multi_sim"
        ) as log_handler:
            multi_sim.synchronizer._sim_to_world()

        assert log_handler.has_message_containing("no MuJoCo joint found")
    finally:
        stop_multisim_if_running(multi_sim)


def test_multiple_synchronizers_can_share_one_simulator():
    """
    Regression test: MujocoSynchronizer used to monkeypatch
    simulator.read_data_from_simulator as a single instance attribute - a second
    synchronizer attaching to the same simulator silently overwrote the first's hook,
    and stopping either synchronizer deleted the sole shared attribute regardless of
    which one owned it, breaking the other synchronizer.
    """
    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)
    try:
        simulator = multi_sim.simulator
        first_synchronizer = multi_sim.synchronizer
        second_synchronizer = MujocoSynchronizer(_world=world, simulator=simulator)
        try:
            assert first_synchronizer._sim_to_world in simulator._data_read_hooks
            assert second_synchronizer._sim_to_world in simulator._data_read_hooks

            second_synchronizer.stop()

            assert first_synchronizer._sim_to_world in simulator._data_read_hooks
            assert second_synchronizer._sim_to_world not in simulator._data_read_hooks
        finally:
            if second_synchronizer._sim_to_world in simulator._data_read_hooks:
                second_synchronizer.stop()
    finally:
        stop_multisim_if_running(multi_sim)


def test_resolve_qpos_adr_is_cached_and_invalidated_on_model_change(monkeypatch):
    """
    Regression test: _resolve_qpos_adr called mujoco.mj_name2id (a string joint-name
    lookup) for every connection on every sync call, in both directions, instead of
    caching the resolved qpos address. This asserts a second lookup for the same
    connection is served from cache, and that a later model change (which can shift
    every joint's qpos address after a recompile) invalidates that cache.

    Adding a connection also triggers the framework's own automatic state resync
    (World._notify_model_change calls notify_state_change right after notifying model
    callbacks), which itself re-resolves every connection's qpos address as a side
    effect - so this checks *which joint names* mj_name2id was asked to resolve around
    the second spawn, rather than a raw call count, to isolate what this test is
    actually checking: whether joint_1's now-stale cache entry gets looked up again.
    """
    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)
    try:
        dof_1 = _spawn_revolute_joint(world, "cache_test_joint_1")
        connection_1 = next(
            c for c in world.connections if getattr(c, "raw_dof", None) is dof_1
        )

        looked_up_names = []
        original_mj_name2id = mujoco.mj_name2id

        def recording_mj_name2id(*args, **kwargs):
            looked_up_names.append(args[2] if len(args) > 2 else kwargs.get("name"))
            return original_mj_name2id(*args, **kwargs)

        monkeypatch.setattr(mujoco, "mj_name2id", recording_mj_name2id)

        multi_sim.synchronizer._qpos_adr_cache.clear()
        looked_up_names.clear()
        multi_sim.synchronizer._resolve_qpos_adr(connection_1)
        multi_sim.synchronizer._resolve_qpos_adr(connection_1)
        assert looked_up_names == [
            "cache_test_joint_1"
        ], "second lookup for the same connection was not served from cache"

        looked_up_names.clear()
        _spawn_revolute_joint(world, "cache_test_joint_2")  # a real model change

        assert "cache_test_joint_1" in looked_up_names, (
            "cache_test_joint_1's now-stale cache entry was not invalidated by the "
            "model change"
        )
    finally:
        stop_multisim_if_running(multi_sim)


def test_connection6dof_with_non_root_parent_raises_instead_of_silently_wrong_sync():
    """
    Regression test: MujocoSynchronizer's 6DoF sync assumed a Connection6DoF's parent
    is always the world root. MuJoCo always expresses a free joint's qpos directly in
    the world frame, so converting it into the connection's own dofs needs to also
    fold in the parent's own pose whenever the parent isn't the world root - silently
    skipping that produced a wrong pose instead of failing loudly.

    In practice MuJoCo's own compiler already refuses to build a free joint that isn't
    a direct child of the top-level body ("free joint can only be used on top level"),
    so a Connection6DoF with a non-root parent can never actually reach a live
    MujocoSimulator through the normal build/spawn path - this exercises the
    synchronizer's guard directly instead, so the assumption still fails loudly if that
    ever changes (e.g. via body merging in the builder) rather than silently producing
    a wrong pose.
    """
    world = World()
    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)
    try:
        intermediate_body = Body(name=PrefixedName("intermediate"))
        floating_body = Body(name=PrefixedName("floating_child"))
        with world.modify_world():
            world.add_connection(
                FixedConnection(parent=world.root, child=intermediate_body)
            )
            connection = Connection6DoF.create_with_dofs(
                world=world,
                parent=intermediate_body,
                child=floating_body,
            )
            # deliberately not world.add_connection(connection): see docstring above.

        with pytest.raises(UnsupportedConnection6DoFParentError):
            multi_sim.synchronizer._read_6dof_from_qpos(connection, qpos_adr=0)
    finally:
        stop_multisim_if_running(multi_sim)


# The real Panda joint1-4 actuator values, kept together so every position-servo
# scene below is driven by the same gains the demo scene actually uses.
PANDA_ARM_GAIN = 2000
PANDA_ARM_DAMPING = 200
PANDA_ARM_FORCE_RANGE = [-87, 87]

# The real Panda gripper actuator values: a fixed tendon remapping the fingers'
# 0-0.04m travel onto a 0-255 ctrl range, so ctrl and position do not share units.
PANDA_GRIPPER_GAIN = 0.0156863
PANDA_GRIPPER_STIFFNESS = 100
PANDA_GRIPPER_DAMPING = 10


def _add_affine_position_servo(
    world: World,
    dof: DegreeOfFreedom,
    gain: float,
    stiffness: float,
    damping: float,
    force_range: Optional[List[float]] = None,
    name: Optional[PrefixedName] = None,
) -> Actuator:
    """
    Add a MuJoCo affine position-servo actuator driving ``dof`` to ``world``.

    MuJoCo evaluates such an actuator as ``force = gain * ctrl - stiffness * length -
    damping * velocity``, so the ``ctrl`` value the world→sim sync has to write for a
    commanded position is the one making that force zero -- which only equals the
    position itself when ``gain == stiffness``.

    :param gain: The actuator's ``gainprm[0]``.
    :param stiffness: Positional feedback gain, stored as ``biasprm[1] = -stiffness``.
    :param damping: Velocity feedback gain, stored as ``biasprm[2] = -damping``.
    :param force_range: Force clamping limits, left unlimited when omitted.
    :param name: Actuator name, needed only when the test looks it up by name in the
        compiled model.
    :return: The actuator, already added to ``world``.

    .. note::
        Must be called inside a ``world.modify_world()`` block.
    """
    actuator = Actuator() if name is None else Actuator(name=name)
    actuator.add_dof(dof=dof)
    actuator_properties = MujocoActuator(
        dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
        dynamics_parameters=[0.0] * 10,
        gain_type=mujoco.mjtGain.mjGAIN_FIXED,
        gain_parameters=[gain] + [0.0] * 9,
        bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
        bias_parameters=[0, -stiffness, -damping] + [0.0] * 7,
    )
    if force_range is not None:
        actuator_properties.force_limited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator_properties.force_range = force_range
    actuator.simulator_additional_properties.append(actuator_properties)
    world.add_actuator(actuator=actuator)
    return actuator


def _actuator_ctrl(multi_sim: MujocoSim, actuator: Actuator) -> float:
    """
    :return: The live ``ctrl`` setpoint of ``actuator`` in the compiled MuJoCo model.
    """
    return multi_sim.simulator.get_actuator(actuator.name.name).result.ctrl[0]


def test_actuated_joint_ctrl_tracks_commanded_qpos():
    """
    Writing a new commanded position into world.state for a dof that is driven by a
    strong PD actuator must move the actuator's ctrl setpoint along with it. If ctrl is
    left stale, MuJoCo's actuator keeps servoing toward the old setpoint and fights
    every subsequent qpos write, so the joint never settles at the commanded position
    (it oscillates instead).
    """
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
            _add_affine_position_servo(
                world,
                dof,
                gain=PANDA_ARM_GAIN,
                stiffness=PANDA_ARM_GAIN,
                damping=PANDA_ARM_DAMPING,
            )

        time.sleep(1)

        connection.position = target
        time.sleep(2)

        final_position = multi_sim.simulator.get_joint_value(dof.name.name).result

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
    for the zero-force ctrl setpoint at a given position, not just copy the position
    through. For a direct per-joint actuator (gainprm=biasprm chosen so ctrl and
    position share units) this happens to reduce to ctrl == position, but for a
    tendon-driven actuator remapping to a different control range (like the Panda
    gripper's 0-0.04m -> 0-255 ctrl range) it must not.
    """
    arm_actuator = Actuator()
    arm_actuator.simulator_additional_properties.append(
        MujocoActuator(
            bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
            bias_parameters=[0, -PANDA_ARM_GAIN, -PANDA_ARM_DAMPING] + [0.0] * 7,
            gain_type=mujoco.mjtGain.mjGAIN_FIXED,
            gain_parameters=[PANDA_ARM_GAIN] + [0.0] * 9,
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
            bias_parameters=[0, -PANDA_GRIPPER_STIFFNESS, -PANDA_GRIPPER_DAMPING]
            + [0.0] * 7,
            gain_type=mujoco.mjtGain.mjGAIN_FIXED,
            gain_parameters=[PANDA_GRIPPER_GAIN] + [0.0] * 9,
        )
    )
    for position in (0.0, 0.02, 0.04):
        expected_ctrl = PANDA_GRIPPER_STIFFNESS * position / PANDA_GRIPPER_GAIN
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
    Integration-level version of
    test_ctrl_for_position_matches_actuator_affine_equilibrium: a tendon-driven actuator
    wired up through the real MujocoSim pipeline must receive a correctly unit-converted
    ctrl value when its dof's commanded position changes, not a raw copy of the
    position.
    """
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
            actuator = _add_affine_position_servo(
                world,
                dof,
                gain=PANDA_GRIPPER_GAIN,
                stiffness=PANDA_GRIPPER_STIFFNESS,
                damping=PANDA_GRIPPER_DAMPING,
                name=PrefixedName("tendon_actuator"),
            )

        time.sleep(1)

        connection.position = target_position
        time.sleep(0.5)

        ctrl = _actuator_ctrl(multi_sim, actuator)
        multi_sim.stop_simulation()

        expected_ctrl = PANDA_GRIPPER_STIFFNESS * target_position / PANDA_GRIPPER_GAIN
        assert numpy.isclose(ctrl, expected_ctrl, rtol=1e-2), (
            f"ctrl was {ctrl}, expected {expected_ctrl:.2f} (the tendon "
            "actuator's own gain/bias remap of the commanded position), "
            "not a raw copy of the position."
        )
    finally:
        stop_multisim_if_running(multi_sim)


def _build_physically_simulated_revolute_joint(
    name_prefix: str,
) -> Tuple[World, RevoluteConnection, Actuator]:
    """
    Build a world holding one PD-actuated revolute joint, ready to be marked
    physically simulated.

    :param name_prefix: Prefix for the scene's body, joint and link names.
    :return: The world, the joint's connection and its actuator.
    """
    world = World()
    root = Body(name=PrefixedName("world"))
    base = Body(name=PrefixedName(f"{name_prefix}_base"))
    link = Body(name=PrefixedName(f"{name_prefix}_link"))
    link.collision = ShapeCollection(
        [Cylinder(width=0.05, height=0.3, color=Color(0.5, 0.5, 0.5, 1.0))],
        reference_frame=link,
    )
    dof = DegreeOfFreedom(name=PrefixedName(f"{name_prefix}_joint"))

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
        actuator = _add_affine_position_servo(
            world,
            dof,
            gain=PANDA_ARM_GAIN,
            stiffness=PANDA_ARM_GAIN,
            damping=PANDA_ARM_DAMPING,
        )

    return world, connection, actuator


def test_physically_simulated_dof_skips_qpos_teleport():
    """
    A dof marked as physically_simulated must not have its qpos force-written by the
    world→sim sync -- only its actuator's ctrl setpoint. The point of this flag is to
    let MuJoCo's real actuator/contact dynamics decide the dof's actual position (e.g. a
    gripper finger stopping against a grasped object) instead of a kinematic snap
    fighting/overriding whatever physics would otherwise produce.
    """
    target = 1.0
    world, connection, actuator = _build_physically_simulated_revolute_joint("physsim")

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=STEP_SIZE,
        physically_simulated_dofs={connection.raw_dof},
    )

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        connection.position = target
        # Deliberately no settling time here: check the write itself, before
        # physics has a chance to converge the joint toward the ctrl setpoint.
        qpos_immediately_after_write = multi_sim.simulator.get_joint_value(
            connection.raw_dof.name.name
        ).result
        ctrl_immediately_after_write = _actuator_ctrl(multi_sim, actuator)

        multi_sim.stop_simulation()

        assert numpy.isclose(ctrl_immediately_after_write, target, atol=1e-6), (
            "ctrl should still track the commanded position for a "
            "physically_simulated dof."
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
    The sim→world sync must overwrite a physically_simulated dof's *velocity* in
    ``world.state`` with the measured simulator velocity, not just its position.

    A controller (e.g. Giskard) writes its **commanded** velocity into ``world.state``
    every tick. A stall detector watching those velocities
    (``JointPositionList(tolerate_stall=True)`` / ``LocalMinimumReached``) needs to see
    the joint's real, physical settling: a gripper finger physically stopped by a
    grasped object otherwise still shows the controller's nonzero commanded closing
    velocity forever, the stall is never detected, and every motion queued behind the
    grasp never starts.
    """
    world, connection, _ = _build_physically_simulated_revolute_joint("velsync")
    dof = connection.raw_dof

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
            "sim→world sync should have overwritten it with the measured "
            "(near-zero) velocity."
        )
    finally:
        stop_multisim_if_running(multi_sim)


def test_physically_simulated_dof_ctrl_latches_commanded_increments_past_contact():
    """
    A physically_simulated dof's actuator setpoint must accumulate the controller's
    commanded increments instead of being re-derived from the measurement-reset belief
    position.

    A controller commanding "keep pushing" against a contact (e.g. closing a gripper on
    a grasped object) writes ``measured + one_step_increment`` into ``world.state`` each
    tick, because the sim→world sync resets the belief to the measured stall position in
    between. Mapping *that* belief straight to ``ctrl`` pins the position servo's
    setpoint at the contact surface, which means near-zero squeeze force -- the grasp
    cannot hold anything. The setpoint must instead integrate the commanded increments,
    so it latches past the contact and the servo keeps pressing.
    """
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
        actuator = _add_affine_position_servo(
            world,
            dof,
            gain=PANDA_ARM_GAIN,
            stiffness=PANDA_ARM_GAIN,
            damping=PANDA_ARM_DAMPING,
            force_range=PANDA_ARM_FORCE_RANGE,
        )

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
        ctrl_final = _actuator_ctrl(multi_sim, actuator)

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
    Several physically_simulated dofs actuated at the same time (mirroring several of a
    multi-joint arm's joints being physically simulated simultaneously, not just one
    isolated dof) must all converge to their commanded targets via their real PD
    actuators and *stay* converged, not merely pass through the target on their way to a
    sustained oscillation.

    This is the risk that made a fully physically-simulated arm (as opposed to
    kinematically teleporting it every tick) a bigger undertaking than making just the
    gripper's fingers physically simulated: several actuator-driven joints settling at
    once is a materially different question -- does the ctrl-sync/qpos-skip machinery
    (physically_simulated_dofs, _write_1dof_to_qpos) hold up across several
    simultaneously-actuated dofs -- than a single joint settling in isolation. Each
    joint is mounted directly to its own fixed base (independent of the others) so this
    isolates that risk from unrelated kinematic-chain dynamics/inertia tuning.
    """
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
            # ballpark as a real Panda arm link -- the arm gains applied to a much
            # lighter test body produce a huge angular acceleration for its tiny
            # inertia, which blows up numerically at any reasonable step_size.
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

            _add_affine_position_servo(
                world,
                dof,
                gain=PANDA_ARM_GAIN,
                stiffness=PANDA_ARM_GAIN,
                damping=PANDA_ARM_DAMPING,
                # Without a force limit, a stiff PD gain applied to a (relatively
                # light) test body can produce an enormous instantaneous torque and
                # diverge numerically.
                force_range=PANDA_ARM_FORCE_RANGE,
            )

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
        # actually drives physically_simulated dofs (small steps every tick).
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
    Shared setup: a single physically_simulated revolute joint holding a
    horizontally-extended cantilevered link level at position 0 against gravity, with or
    without MuJoCo's gravcomp on that link.

    :param gravity_compensated: Whether the link carries a MujocoBody gravcomp property.
    :return: The joint's settled steady-state position error from 0.
    """
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
        _add_affine_position_servo(
            world,
            dof,
            gain=PANDA_ARM_GAIN,
            stiffness=PANDA_ARM_GAIN,
            damping=PANDA_ARM_DAMPING,
            force_range=PANDA_ARM_FORCE_RANGE,
        )

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
    A physically_simulated joint holding a gravity-loaded link (e.g. the Panda arm's own
    links, once physically simulated rather than kinematically teleported) settles with
    a steady-state position error from its PD actuator's gain alone -- without MuJoCo's
    gravcomp countering gravity separately, this error can exceed JointPositionList's
    default 0.01 rad convergence threshold
    (giskardpy/motion_statechart/tasks/joint_tasks.py), so a motion holding such a joint
    (e.g. ParkArmsAction) never registers as converged and Giskard keeps sending
    corrective commands indefinitely -- which also stalls the rest of the plan behind
    it.
    """
    error_without_gravcomp = _settle_gravity_loaded_cantilever(
        gravity_compensated=False
    )
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


def _close_tendon_gripper_around_cube(
    physically_simulated: bool,
) -> Tuple[float, float, numpy.ndarray]:
    """
    Shared setup for the two tests below: parses a minimal two-finger, tendon-actuated
    gripper (mirroring the Panda gripper's real MJCF structure -- fixed tendon + a
    single actuator driving both joints) with a cube resting between the fingers,
    commands both fingers to a fully-closed target that would require passing through
    the cube, lets it settle, and reports where everything ended up.

    :param physically_simulated: Whether both finger dofs are marked physically
        simulated.
    :return: The two final finger positions and the cube's final position.
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
        cube_position = numpy.asarray(
            multi_sim.simulator.get_body_position("cube").result[:3], dtype=float
        )
        multi_sim.stop_simulation()
        return q1, q2, cube_position
    finally:
        stop_multisim_if_running(multi_sim)


def test_physically_simulated_gripper_stalls_on_and_holds_cube():
    """
    With both finger dofs marked physically_simulated, closing the gripper onto a cube
    must stall well short of the (physically unreachable, since it requires passing
    through the cube) commanded target, and the cube must stay essentially where it
    started -- proving real contact/friction, not a kinematic snap, is what determines
    the fingers' resting position and holds the object.
    """
    q1, q2, cube_position = _close_tendon_gripper_around_cube(physically_simulated=True)

    assert q1 < 0.03 and q2 < 0.03, (
        f"fingers should have stalled against the cube well short of the "
        f"commanded 0.05m target, got q1={q1}, q2={q2}"
    )
    assert numpy.allclose(cube_position, [0, 0, 0], atol=0.01), (
        f"cube should have stayed close to its starting position, held in "
        f"place by the fingers, got {cube_position}"
    )


def test_mimic_joint_gets_its_own_scaled_range_not_the_joint_it_mimics():
    """
    A mimicking joint's MuJoCo range must be its own, scaled by its multiplier and
    offset, not a copy of the range of the joint it mimics.

    Both joints share one dof, whose limits describe the *mimicked* joint. Copying those
    onto the mimic contradicts the equality constraint that couples them: the mimic is
    then free to travel to positions its multiplier says it can never reach, and the
    solver has to reconcile a limit and an equality that cannot both hold. On the HSR's
    torso this drives the lift to its upper stop on its own.
    """
    world = World()
    root = Body(name=PrefixedName("world"))
    base = Body(name=PrefixedName("mimic_base"))
    driven = Body(name=PrefixedName("driven_link"))
    mimicking = Body(name=PrefixedName("mimicking_link"))
    lower_limits = DerivativeMap[float]()
    lower_limits.position = 0.005
    upper_limits = DerivativeMap[float]()
    upper_limits.position = 0.67
    dof = DegreeOfFreedom(
        name=PrefixedName("driven_joint"),
        limits=DegreeOfFreedomLimits(lower=lower_limits, upper=upper_limits),
    )
    multiplier = 0.5

    with world.modify_world():
        world.add_body(root)
        world.add_connection(FixedConnection(parent=root, child=base))
        world.add_degree_of_freedom(dof)
        world.add_connection(
            RevoluteConnection(
                name=dof.name,
                parent=base,
                child=driven,
                axis=Vector3.Z(reference_frame=driven),
                raw_dof=dof,
            )
        )
        world.add_body(mimicking)
        world.add_connection(
            RevoluteConnection(
                name=PrefixedName("mimicking_joint"),
                parent=base,
                child=mimicking,
                axis=Vector3.Z(reference_frame=mimicking),
                raw_dof=dof,
                multiplier=multiplier,
                offset=0.0,
            )
        )

    builder = MujocoBuilder()
    builder.build_world(world=world, file_path="/tmp/mimic_range_scene.xml")
    model = builder.spec.compile()

    driven_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "driven_joint")
    mimicking_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "mimicking_joint"
    )
    driven_range = list(model.jnt_range[driven_id])
    mimicking_range = list(model.jnt_range[mimicking_id])

    numpy.testing.assert_allclose(
        mimicking_range,
        [limit * multiplier for limit in driven_range],
        atol=1e-9,
        err_msg=(
            f"mimicking joint got range {mimicking_range}, the range of the joint it "
            f"mimics ({driven_range}) rather than its own {multiplier}x scaling"
        ),
    )


def _build_mimic_joint_simulation() -> tuple:
    """
    Start a simulation of one driven revolute joint and a second that mimics it at half
    its travel, mirroring how the HSR's torso follows its arm lift.

    :return: The running simulation, the driven connection and the mimicking connection.
    """
    world = World()
    root = Body(name=PrefixedName("world"))
    base = Body(name=PrefixedName("mimic_base"))
    driven_link = Body(name=PrefixedName("driven_link"))
    mimicking_link = Body(name=PrefixedName("mimicking_link"))
    lower_limits = DerivativeMap[float]()
    lower_limits.position = -1.0
    upper_limits = DerivativeMap[float]()
    upper_limits.position = 1.0
    dof = DegreeOfFreedom(
        name=PrefixedName("driven_joint"),
        limits=DegreeOfFreedomLimits(lower=lower_limits, upper=upper_limits),
    )

    with world.modify_world():
        world.add_body(root)
        world.add_connection(FixedConnection(parent=root, child=base))
        world.add_degree_of_freedom(dof)
        driven = RevoluteConnection(
            name=dof.name,
            parent=base,
            child=driven_link,
            axis=Vector3.Z(reference_frame=driven_link),
            raw_dof=dof,
        )
        world.add_connection(driven)
        world.add_body(mimicking_link)
        mimicking = RevoluteConnection(
            name=PrefixedName("mimicking_joint"),
            parent=base,
            child=mimicking_link,
            axis=Vector3.Z(reference_frame=mimicking_link),
            raw_dof=dof,
            multiplier=0.5,
            offset=0.0,
        )
        world.add_connection(mimicking)

    multi_sim = MujocoSim(world=world, headless=headless, step_size=STEP_SIZE)
    multi_sim.start_simulation()
    time.sleep(0.5)
    return multi_sim, driven, mimicking


def test_world_to_sim_sync_scales_a_mimicking_joints_commanded_position():
    """
    A mimicking joint's qpos must be the shared dof's value scaled by its own multiplier
    and offset, not a raw copy of it.

    Both joints read from one dof, so writing the raw value into both commands the mimic
    to travel as far as the joint it mimics -- which its own limits and the equality
    constraint coupling them both forbid.
    """
    multi_sim, driven, mimicking = _build_mimic_joint_simulation()

    try:
        driven.position = 0.4
        time.sleep(0.5)

        driven_qpos = multi_sim.simulator.get_joint_value("driven_joint").result
        mimicking_qpos = multi_sim.simulator.get_joint_value("mimicking_joint").result

        assert float(driven_qpos) == pytest.approx(0.4, abs=0.02)
        assert float(mimicking_qpos) == pytest.approx(0.2, abs=0.02), (
            f"mimicking joint sits at {float(mimicking_qpos)}, not the "
            "0.5x-scaled 0.2 its multiplier allows"
        )
    finally:
        stop_multisim_if_running(multi_sim)


def test_sim_to_world_sync_does_not_let_a_mimicking_joint_clobber_the_shared_dof():
    """
    Reading a mimicking joint's qpos back must recover the *shared* dof's value, undoing
    its multiplier and offset.

    Every connection sharing a dof writes into the same world-state entry, so a
    mimicking joint that writes its own scaled reading straight in overwrites whatever
    the joint it mimics just reported, and the world model ends up believing a position
    neither joint is actually at.
    """
    multi_sim, driven, mimicking = _build_mimic_joint_simulation()

    try:
        driven.position = 0.4
        time.sleep(0.5)
        multi_sim.synchronizer._sim_to_world(force=True)

        driven_qpos = float(multi_sim.simulator.get_joint_value("driven_joint").result)
        believed = float(
            multi_sim.synchronizer._world.state[driven.raw_dof.id].position
        )

        assert believed == pytest.approx(driven_qpos, abs=1e-3), (
            f"world model believes the shared dof is at {believed} while the joint it "
            f"describes is physically at {driven_qpos}; the mimicking joint's reading "
            "overwrote it"
        )
    finally:
        stop_multisim_if_running(multi_sim)


@dataclass
class CarriedBodyScene:
    """
    A body resting on a movable handle, ready to be re-parented onto it the way
    AttachNode/DetachNode do.
    """

    multi_sim: MujocoSim
    """The running simulation the scene lives in."""

    handle_connection: PrismaticConnection
    """The joint moving the handle the body is carried by."""

    carried_body: Body
    """The body that gets re-parented onto the handle."""

    def reparent_onto(self, parent: Body) -> None:
        """
        Re-parent :attr:`carried_body` onto ``parent`` exactly the way
        AttachNode/DetachNode do: remove its current connection and add a new one,
        both inside a single ``modify_world()`` block.
        """
        world = self.multi_sim.synchronizer._world
        with world.modify_world():
            world.remove_connection(self.carried_body.parent_connection)
            world.add_connection(
                FixedConnection(
                    parent=parent,
                    child=self.carried_body,
                    parent_T_connection_expression=world.compute_forward_kinematics(
                        parent, self.carried_body
                    ),
                )
            )

    def carried_body_joints(self) -> list:
        """
        :return: The MuJoCo joints :attr:`carried_body` currently has. A welded body
            has none of its own; a free body has exactly one free joint.
        """
        return self.multi_sim.simulator.get_body_joints(
            self.carried_body.name.name
        ).result


def _build_carried_body_scene(reparenting_mode: ReparentingMode) -> CarriedBodyScene:
    """
    Build and start a :class:`CarriedBodyScene` whose simulation re-parents bodies
    according to ``reparenting_mode``.

    The handle is driven by a real PD actuator, matching how an arm's own joints hold
    position via actual force: without one the joint has nothing opposing gravity and
    free-falls from creation, which an arm's actuated joints never do.
    """
    world = World()
    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=STEP_SIZE,
        reparenting_mode=reparenting_mode,
    )
    multi_sim.start_simulation()
    time.sleep(1)

    base = Body(name=PrefixedName("handle_base"))
    handle = Body(name=PrefixedName("handle"))
    handle.collision = ShapeCollection(
        [Box(scale=Scale(0.05, 0.05, 0.05), color=Color(0.2, 0.2, 0.8, 1.0))],
        reference_frame=handle,
    )
    carried_body = Body(name=PrefixedName("attachable_box"))
    carried_body.collision = ShapeCollection(
        [Box(scale=Scale(0.04, 0.04, 0.04), color=Color(0.8, 0.2, 0.2, 1.0))],
        reference_frame=carried_body,
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
        _add_affine_position_servo(
            world,
            dof,
            gain=PANDA_ARM_GAIN,
            stiffness=PANDA_ARM_GAIN,
            damping=PANDA_ARM_DAMPING,
        )
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=carried_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=0.2, reference_frame=world.root
                ),
            )
        )

    time.sleep(1)
    return CarriedBodyScene(multi_sim, handle_connection, carried_body)


def test_weld_reparenting_attaches_and_detaches_the_body_in_the_simulator():
    """
    Under :attr:`ReparentingMode.WELD`, re-parenting a body in the world model the way
    AttachNode/DetachNode do must also weld/un-weld it in MuJoCo's own kinematic tree,
    not just in the world model.

    Checked structurally rather than by position deltas: a welded child has no joint of
    its own in MuJoCo at all, while the handle and box geometries are vertically stacked,
    so a mere resting contact (no weld) can coincidentally reproduce the same delta.
    """
    scene = _build_carried_body_scene(ReparentingMode.WELD)
    world = scene.multi_sim.synchronizer._world

    try:
        scene.reparent_onto(scene.handle_connection.child)
        assert scene.carried_body_joints() == [], (
            "box still has its own MuJoCo joint after being attached under "
            "ReparentingMode.WELD; it was not welded to the handle"
        )

        scene.reparent_onto(world.root)
        detached_joints = scene.carried_body_joints()
        assert len(detached_joints) == 1 and (
            detached_joints[0].type == mujoco.mjtJoint.mjJNT_FREE
        ), (
            "box did not get a free joint back after being detached to world.root, "
            f"still welded in MuJoCo: joints={detached_joints}"
        )
    finally:
        stop_multisim_if_running(scene.multi_sim)


def test_contact_only_reparenting_leaves_the_body_free_in_the_simulator():
    """
    Under :attr:`ReparentingMode.CONTACT_ONLY`, the same world-model re-parent must
    leave the body's own free joint untouched in MuJoCo, so nothing but real contact and
    friction holds it to whatever picked it up.

    This is what makes a fully physical grasp possible: with the body welded, a grasp
    can never slip regardless of how the fingers actually contact it, so the simulation
    cannot tell a good grasp from a bad one.
    """
    scene = _build_carried_body_scene(ReparentingMode.CONTACT_ONLY)

    try:
        joints_before = scene.carried_body_joints()
        scene.reparent_onto(scene.handle_connection.child)
        joints_after = scene.carried_body_joints()

        assert len(joints_before) == 1 and (
            joints_before[0].type == mujoco.mjtJoint.mjJNT_FREE
        ), f"box did not start as a free body in MuJoCo: joints={joints_before}"
        assert len(joints_after) == 1 and (
            joints_after[0].type == mujoco.mjtJoint.mjJNT_FREE
        ), (
            "box lost its free joint when it was attached in the world model; "
            "ReparentingMode.CONTACT_ONLY must leave MuJoCo's kinematic tree alone "
            f"so friction alone holds it: joints={joints_after}"
        )
    finally:
        stop_multisim_if_running(scene.multi_sim)


def test_kinematically_teleported_gripper_ignores_cube_contact():
    """
    Negative control for test_physically_simulated_gripper_stalls_on_and_holds_cube:
    with physically_simulated_dofs left empty (the default teleport behaviour for every
    dof), the fingers must reach much closer to the commanded target regardless of the
    cube being in the way, and the cube gets displaced/ejected rather than held --
    demonstrating that without physically_simulated_dofs, closing a gripper on an object
    does not produce a stable, contact-respecting grasp.
    """
    q1, q2, cube_position = _close_tendon_gripper_around_cube(
        physically_simulated=False
    )

    assert q1 > 0.035 and q2 > 0.035, (
        f"fingers should have reached close to the commanded 0.05m target "
        f"regardless of the cube, got q1={q1}, q2={q2}"
    )
    assert not numpy.allclose(cube_position, [0, 0, 0], atol=0.01), (
        f"cube should have been displaced/ejected by the kinematically "
        f"teleported fingers passing through it, got {cube_position}"
    )
