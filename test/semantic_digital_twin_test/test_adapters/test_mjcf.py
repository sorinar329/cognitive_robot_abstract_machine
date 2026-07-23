import os.path
import tempfile

import mujoco
import numpy
import pytest

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoBuilder
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
