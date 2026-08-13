import os.path
import tempfile

import mujoco
import numpy
import pytest

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoBuilder, MujocoLight
from semantic_digital_twin.world_description.connections import FixedConnection

MJCF_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "mjcf",
)


@pytest.fixture
def table_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "table.xml"))


@pytest.fixture
def kitchen_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "kitchen-small.xml"))


@pytest.fixture
def apartment_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "iai_apartment.xml"))


@pytest.fixture
def pr2_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "pr2_kinematic_tree.xml"))


def test_from_file_carries_the_parsing_parameters():
    file_path = os.path.join(MJCF_DIR, "table.xml")
    parser = MJCFParser.from_file(
        file_path, prefix="env", mimic_joints={"left": "right"}
    )

    assert parser.file_path == file_path
    assert parser.prefix == "env"
    assert parser.mimic_joints == {"left": "right"}


def test_from_file_defaults_the_prefix_to_the_file_name():
    parser = MJCFParser.from_file(os.path.join(MJCF_DIR, "table.xml"))

    assert parser.prefix == "table"
    assert parser.mimic_joints == {}


def test_parsing_twice_yields_independent_worlds(table_xml_parser):
    first = table_xml_parser.parse()
    second = table_xml_parser.parse()

    assert first is not second
    first_ids = {body.id for body in first.bodies}
    second_ids = {body.id for body in second.bodies}
    assert len(first.bodies) == len(second.bodies)
    assert first_ids.isdisjoint(second_ids)


def test_table_parsing(table_xml_parser):
    body_num = 7
    world = table_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) == body_num

    origin_left_front_leg_joint = world.get_connection(
        world.root, world.kinematic_structure_entities[1]
    )
    assert isinstance(origin_left_front_leg_joint, FixedConnection)


def test_kitchen_parsing(kitchen_xml_parser):
    world = kitchen_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0


def test_apartment_parsing(apartment_xml_parser):
    world = apartment_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0


def test_pr2_parsing(pr2_xml_parser):
    world = pr2_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0
    assert world.root.name.name == "world"


HINGED_BODY_MJCF = """
<mujoco>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="door">
        <joint name="hinge" type="hinge" axis="0 0 1" range="-1.57 0"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_joint_position_limits_are_python_floats():
    """
    Parsed joint position limits must be plain Python floats.

    MuJoCo reports them as numpy scalars, which do not interoperate with the symbolic-math layer (``numpy_scalar - symbol`` makes numpy
    try to arrayify the symbol) and break motion planning on the joint.
    """
    world = MJCFParser.from_xml_string(HINGED_BODY_MJCF).parse()
    limits = world.get_degree_of_freedom_by_name("hinge").limits
    assert type(limits.lower.position) is float
    assert type(limits.upper.position) is float


LIT_WORLD_MJCF = """
<mujoco>
  <worldbody>
    <light pos="2.0 -2.0 2.0" dir="0.01 0.01 -1" specular="0.3 0.3 0.3" ambient="0.3 0.3 0.3"
           diffuse="0.3 0.3 0.3" directional="true" castshadow="false"/>
    <body name="base">
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_light_is_parsed_and_attached_to_its_parent_body():
    """
    Regression test: MJCFParser used to have no handling for <light> elements at all, so
    every world built through the parser -> World -> MujocoBuilder round-trip silently
    lost all lighting information, falling back to MuJoCo's minimal default camera
    headlight instead of the scene's own intended lights.
    """
    world = MJCFParser.from_xml_string(LIT_WORLD_MJCF).parse()

    lights = [
        light_property
        for light_property in world.root.simulator_additional_properties
        if isinstance(light_property, MujocoLight)
    ]
    assert len(lights) == 1
    light = lights[0]
    assert light.position == pytest.approx([2.0, -2.0, 2.0])
    assert light.direction == pytest.approx([0.01, 0.01, -1.0], abs=1e-3)
    assert light.ambient == pytest.approx([0.3, 0.3, 0.3])
    assert light.diffuse == pytest.approx([0.3, 0.3, 0.3])
    assert light.specular == pytest.approx([0.3, 0.3, 0.3])
    assert light.directional is True
    assert light.cast_shadow is False


TEXTURED_BOX_MJCF_TEMPLATE = """
<mujoco>
  <asset>
    <texture name="marble_tex" type="2d" file="{texture_file_path}"/>
    <material name="marble" texture="marble_tex" texrepeat="3 3" texuniform="true"/>
  </asset>
  <worldbody>
    <body name="counter">
      <geom type="box" size="0.5 0.5 0.05" material="marble"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_primitive_box_geom_resolves_its_material_texture(tmp_path):
    """
    Regression test: Box/Sphere/Cylinder shapes never carried any texture reference,
    only a flat Color.

    RoboCasa's countertops and cabinet doors are actual MJCF box geoms whose material
    references a marble/wood texture, so this reference was silently discarded on every
    round-trip and they rendered flat-colored instead of textured.
    """
    from PIL import Image

    texture_file_path = tmp_path / "marble.png"
    Image.new("RGB", (4, 4), color=(200, 200, 200)).save(texture_file_path)
    mjcf_file_path = tmp_path / "scene.xml"
    mjcf_file_path.write_text(
        TEXTURED_BOX_MJCF_TEMPLATE.format(texture_file_path=texture_file_path)
    )

    world = MJCFParser(str(mjcf_file_path)).parse()

    [counter] = [
        body
        for body in world.kinematic_structure_entities
        if body.name.name == "counter"
    ]
    [box_shape] = counter.visual.shapes
    assert box_shape.texture is not None
    assert box_shape.texture.file_path == str(texture_file_path)
    assert box_shape.texture.repeat == pytest.approx([3.0, 3.0])
    assert box_shape.texture.uniform is True


TENDON_ACTUATED_GRIPPER_MJCF = """
<mujoco>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="left_finger">
        <joint name="finger_joint1" type="slide" axis="1 0 0" range="0 0.04"/>
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
      <body name="right_finger">
        <joint name="finger_joint2" type="slide" axis="-1 0 0" range="0 0.04"/>
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="split">
      <joint joint="finger_joint1" coef="0.5"/>
      <joint joint="finger_joint2" coef="0.5"/>
    </fixed>
  </tendon>
  <actuator>
    <general name="gripper_actuator" tendon="split" ctrlrange="0 255" gainprm="0.0156863" biasprm="0 -100 -10"/>
  </actuator>
</mujoco>
"""


def test_tendon_actuator_resolves_to_real_joint_dofs():
    """
    A tendon-driven actuator must be associated with the real DegreeOfFreedom
    objects of the joints its tendon couples, not with a synthetic DOF named
    after the tendon.

    The synthetic tendon-named DOF used to be created purely so
    ``get_degree_of_freedom_by_name(mujoco_actuator.target)`` wouldn't crash,
    but it was never referenced by any connection, so ``modify_world()``'s
    orphan cleanup deletes it before parsing finishes -- leaving the actuator
    holding a dangling reference to a DOF that no longer exists in the world,
    and no way to resolve "which real joints does this actuator drive".
    """
    world = MJCFParser.from_xml_string(TENDON_ACTUATED_GRIPPER_MJCF).parse()
    world.validate()

    actuator = next(a for a in world.actuators if a.name.name == "gripper_actuator")
    dof_names = {dof.name.name for dof in actuator.dofs}

    assert dof_names == {"finger_joint1", "finger_joint2"}
    assert not any(dof.name.name == "split" for dof in world.degrees_of_freedom)


MIMIC_JOINT_GRIPPER_MJCF = """
<mujoco>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="left_finger">
        <joint name="finger_joint1" type="slide" axis="1 0 0" range="0 0.04"/>
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
      <body name="right_finger">
        <joint name="finger_joint2" type="slide" axis="-1 0 0" range="0 0.04"/>
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="finger_joint1" joint2="finger_joint2" polycoef="0 1 0 0 0"/>
  </equality>
</mujoco>
"""


def test_mimicked_joint_shares_the_real_degree_of_freedom():
    """
    A joint declared as an <equality>-constrained mimic of another (as the
    Panda gripper's two finger joints are) must resolve to the *same*
    DegreeOfFreedom object as the joint it mimics, not a second, distinct
    object that merely happens to have the same name.

    parse_dof's mimic_joints remap used to build a brand new DegreeOfFreedom
    named after the mimicked joint instead of reusing the one already created
    for it, so world.degrees_of_freedom silently ended up with two DOFs
    sharing one name -- e.g. breaking get_degree_of_freedom_by_name for that
    name, and any actuator/tendon logic that expects mimicked joints to
    genuinely share a single DOF (see test_tendon_actuator_resolves_to_real_joint_dofs).
    """
    world = MJCFParser.from_xml_string(MIMIC_JOINT_GRIPPER_MJCF).parse()

    joint1 = world.get_connection_by_name("finger_joint1")
    joint2 = world.get_connection_by_name("finger_joint2")

    assert joint1.raw_dof is joint2.raw_dof
    assert len([d for d in world.degrees_of_freedom if d.name.name == "finger_joint1"]) == 1
    world.validate()


HINGE_WITH_ZERO_EXCLUDED_FROM_RANGE_MJCF = """
<mujoco>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="link_a" quat="0.707107 0.707107 0 0">
        <joint name="joint_a" type="hinge" axis="0 0 1" range="-1.0 -0.2"/>
        <geom type="box" size="0.05 0.05 0.05"/>
        <body name="link_b" pos="0 0 0.2">
          <geom type="box" size="0.02 0.02 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_rebuilt_body_quat_excludes_a_joints_current_position():
    """
    A joint whose declared range excludes zero (like the Panda's joint4,
    range -3.0718 to -0.0698) gets its DegreeOfFreedom initialized to the
    nearest valid limit instead of 0 (see World._add_degree_of_freedom).
    Rebuilding the MJCF from the parsed world must not bake that nonzero
    joint value into the child body's static quat/pos -- MuJoCo's own joint
    mechanism already applies the DOF's rotation at runtime via qpos, so
    doing it twice (once baked into the body's mounting pose, once via the
    joint itself) doubles the rotation.

    This used to happen because MujocoKinematicStructureEntityConverter
    read origin_as_position_quaternion(), which evaluates the connection's
    full origin_expression (including its DOF-dependent _kinematics) instead
    of just the constant parent_T_connection_expression /
    connection_T_child_expression that a MJCF body's own pos/quat represents.
    """
    world = MJCFParser.from_xml_string(HINGE_WITH_ZERO_EXCLUDED_FROM_RANGE_MJCF).parse()
    joint_a = world.get_connection_by_name("joint_a")
    assert world.state[joint_a.raw_dof.id].position != 0.0

    file_path = tempfile.mktemp(suffix=".xml")
    MujocoBuilder().build_world(world=world, file_path=file_path)
    spec = mujoco.MjSpec.from_file(file_path)

    link_a_spec = spec.body("link_a")
    original_quat = numpy.array([0.707107, 0.707107, 0, 0])
    assert numpy.allclose(link_a_spec.quat, original_quat, atol=1e-4), (
        f"link_a's rebuilt quat {link_a_spec.quat} does not match its authored, "
        f"DOF-independent mounting quat {original_quat} -- joint_a's nonzero "
        "default position has been baked into the static body pose."
    )
