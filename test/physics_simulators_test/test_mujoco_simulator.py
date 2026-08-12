import os
import time

import mujoco
import numpy
import pytest

from physics_simulators.mujoco_simulator import MujocoSimulator
from physics_simulators.base_simulator import (
    SimulatorConstraints,
    SimulatorState,
    SimulatorCallbackResult,
)

resources_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "mjcf",
)
headless = os.environ.get("CI", "false").lower() == "true"
# headless = False


class TestMujocoSimulator:
    file_path = os.path.join(resources_path, "floor.xml")
    headless = headless
    step_size = 1e-3

    @pytest.fixture
    def simulator(self):
        sim = MujocoSimulator(
            _headless=self.headless,
            _step_size=self.step_size,
            file_path=os.path.join(resources_path, "mjx_single_cube_no_mesh.xml"),
        )
        yield sim
        try:
            sim.stop()
        except Exception:
            pass

    def test_functions(self, simulator):
        simulator.start(simulate_in_thread=False, render_in_thread=True)

        key_id = None
        save_file_path = None

        for step in range(4000):
            if step < 1000:
                result = simulator.callbacks["get_all_body_names"]()
                assert isinstance(result, SimulatorCallbackResult)
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.result == [
                    "world",
                    "link0",
                    "link1",
                    "link2",
                    "link3",
                    "link4",
                    "link5",
                    "link6",
                    "link7",
                    "hand",
                    "left_finger",
                    "right_finger",
                    "floor",
                    "box",
                ]

                result = simulator.callbacks["get_all_joint_names"]()
                assert isinstance(result, SimulatorCallbackResult)
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.result == [
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                    "joint7",
                    "finger_joint1",
                    "finger_joint2",
                ]

            if step == 1000 or step == 3000:
                result = simulator.callbacks["attach"](body_1_name="abc")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL
                )
                assert result.info == "Body 1 abc not found"

                result = simulator.callbacks["attach"](body_1_name="world")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL
                )
                assert result.info == "Body 1 and body 2 are the same"

                result = simulator.callbacks["attach"](body_1_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.info == "Body 1 box is already attached to body 2 world"

                result = simulator.callbacks["attach"](
                    body_1_name="box", body_2_name="hand"
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
                )
                assert "Attached body 1 box to body 2 hand" in result.info

                result = simulator.callbacks["enable_contact"](
                    body_1_name="box", body_2_name="left_finger"
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
                )

                result = simulator.enable_contact(
                    body_1_name="box", body_2_name="right_finger"
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
                )

                result = simulator.callbacks["attach"](
                    body_1_name="box", body_2_name="hand"
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.info == "Body 1 box is already attached to body 2 hand"

            if step == 1200:
                result = simulator.callbacks["get_joint_value"](joint_name="joint1")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, float)
                joint1_value = result.result

                result = simulator.callbacks["get_joints_values"](
                    joint_names=["joint1", "joint2"]
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, dict)
                assert len(result.result) == 2
                assert "joint1" in result.result
                assert isinstance(result.result["joint1"], float)
                assert "joint2" in result.result
                assert isinstance(result.result["joint2"], float)
                assert joint1_value == result.result["joint1"]

                result = simulator.callbacks["get_body_position"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, numpy.ndarray)
                box_position = result.result

                result = simulator.callbacks["get_body_quaternion"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, numpy.ndarray)
                box_quaternion = result.result

                result = simulator.callbacks["get_bodies_positions"](
                    body_names=["box", "link0"]
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, dict)
                assert len(result.result) == 2
                assert "box" in result.result
                assert isinstance(result.result["box"], numpy.ndarray)
                assert "link0" in result.result
                assert isinstance(result.result["link0"], numpy.ndarray)
                assert numpy.allclose(box_position, result.result["box"])

                result = simulator.callbacks["get_bodies_quaternions"](
                    body_names=["box", "link0"]
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, dict)
                assert len(result.result) == 2
                assert "box" in result.result
                assert isinstance(result.result["box"], numpy.ndarray)
                assert "link0" in result.result
                assert isinstance(result.result["link0"], numpy.ndarray)
                assert numpy.allclose(box_quaternion, result.result["box"])

            if step == 800:
                box_position = numpy.array([0.7, 0.0, 1.0])
                result = simulator.callbacks["set_body_position"](
                    body_name="box", position=box_position
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_body_position"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, numpy.ndarray)
                assert numpy.allclose(box_position, result.result)

                box_position = numpy.array([0.7, 0.0, 2.0])
                result = simulator.callbacks["set_bodies_positions"](
                    bodies_positions={"box": box_position}
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_body_position"](body_name="box")
                assert numpy.allclose(box_position, result.result)

                box_quaternion = numpy.array([0.707, 0.707, 0.0, 0.0])
                box_quaternion /= numpy.linalg.norm(box_quaternion)
                result = simulator.callbacks["set_body_quaternion"](
                    body_name="box", quaternion=box_quaternion
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_body_quaternion"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, numpy.ndarray)
                assert numpy.allclose(box_quaternion, result.result)

                box_quaternion = numpy.array([0.707, 0.0, 0.707, 0.0])
                box_quaternion /= numpy.linalg.norm(box_quaternion)
                result = simulator.callbacks["set_bodies_quaternions"](
                    bodies_quaternions={"box": box_quaternion}
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_body_quaternion"](body_name="box")
                assert numpy.allclose(box_quaternion, result.result)

                joint1_value = 0.3
                result = simulator.callbacks["set_joint_value"](
                    joint_name="joint1", value=joint1_value
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_joint_value"](joint_name="joint1")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, float)
                assert result.result == pytest.approx(joint1_value, abs=1e-3)

                joints_values = {"joint1": joint1_value, "joint2": 0.5}
                result = simulator.callbacks["set_joints_values"](
                    joints_values=joints_values
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_joints_values"](
                    joint_names=["joint1", "joint2"]
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, dict)
                assert len(result.result) == 2
                assert "joint1" in result.result
                assert isinstance(result.result["joint1"], float)
                assert "joint2" in result.result
                assert isinstance(result.result["joint2"], float)
                assert result.result["joint1"] == pytest.approx(joint1_value, abs=1e-3)

            if step == 1550:
                result = simulator.callbacks["save"](key_name="step_1550")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                key_id = result.result

                result = simulator.callbacks["load"](key_id=key_id)
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )
                assert result.result == key_id

            if step == 1570:
                save_file_path = os.path.join(resources_path, "../output/step_1570.xml")
                os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

                result = simulator.callbacks["save"](
                    file_path=save_file_path, key_name="step_1570"
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                key_id = result.result

                result = simulator.callbacks["load"](
                    file_path=save_file_path, key_id=key_id
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )
                assert result.result == key_id

            if step == 2000 or step == 4000:
                result = simulator.callbacks["detach"](body_name="abc")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL
                )
                assert result.info == "Body abc not found"

                result = simulator.callbacks["detach"](body_name="world")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.info == "Body world is already detached"

                result = simulator.callbacks["detach"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
                )
                assert result.info == "Detached body box from body hand"

                result = simulator.callbacks["detach"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.info == "Body box is already detached"

            if step == 8000:
                result = simulator.callbacks["get_contact_bodies"](body_name="abc")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION
                )
                assert result.info == "Body abc not found"

                result = simulator.callbacks["get_contact_bodies"](body_name="hand")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, set)

            if step == 100:
                result = simulator.callbacks["get_contact_points"](body_names=["abc"])
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION
                )
                assert result.info == "Body abc not found"

                result = simulator.callbacks["get_contact_points"](
                    body_names=["box", "hand"]
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, list)
                assert len(result.result) == 0

                result = simulator.callbacks["get_contact_points"](body_names=["world"])
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, list)
                assert len(result.result) == 4

            if step == 500 and mujoco.mj_version() < 3005000:
                result = simulator.callbacks["ray_test"](
                    ray_from_position=[0.7, 0.0, 1.0],
                    ray_to_position=[0.7, 0.0, 0.0],
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )

                result = simulator.callbacks["ray_test"](
                    ray_from_position=[0.7, 0.0, 0.2],
                    ray_to_position=[0.7, 0.0, 0.0],
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )

                result = simulator.callbacks["ray_test_batch"](
                    ray_from_position=[0.7, 0.0, 0.2],
                    ray_to_positions=[[0.7, 0.0, 1.0], [0.7, 0.0, 0.0]],
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.result[1]["hit_position"][2] == pytest.approx(
                    0.0599, abs=1e-3
                )

            simulator.step()
            time.sleep(0.001)

        simulator.stop()

    def test_set_geom_friction_overrides_the_model_value(self, simulator):
        """
        set_geom_friction must write directly into the compiled MuJoCo model, so a
        subsequent get_geom_friction observes the new value -- no simulation step is
        needed, since friction is a static model property rather than simulation state.
        """
        new_friction = numpy.array([1.5, 0.05, 0.0005])

        result = simulator.callbacks["set_geom_friction"](
            geom_name="box", friction=new_friction
        )
        assert (
            result.type
            is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
        )

        result = simulator.callbacks["get_geom_friction"](geom_name="box")
        assert (
            result.type is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        )
        assert numpy.allclose(result.result, new_friction)

    def test_set_geom_friction_fails_for_unknown_geom(self, simulator):
        """
        set_geom_friction must report failure rather than raising or silently doing
        nothing when the requested geom does not exist in the model.
        """
        result = simulator.callbacks["set_geom_friction"](
            geom_name="this_geom_does_not_exist", friction=numpy.array([1.0, 0.0, 0.0])
        )
        assert (
            result.type is SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION
        )


class TestMujocoSimulatorComplex:
    file_path = os.path.join(resources_path, "mjx_single_cube_no_mesh.xml")
    Simulator = MujocoSimulator
    headless = headless
    step_size = 5e-4

    def make_simulator(self, headless=None):
        if headless is None:
            headless = self.headless
        return MujocoSimulator(
            _headless=headless,
            _step_size=self.step_size,
            file_path=self.file_path,
        )

    def test_running_in_10s_in_1(self):
        simulator = self.make_simulator()
        try:
            constraints = SimulatorConstraints(max_real_time=10.0)
            simulator.start(
                constraints=constraints,
                simulate_in_thread=True,
                render_in_thread=True,
            )
            while simulator.state != SimulatorState.STOPPED:
                time.sleep(1)
            assert simulator.state is SimulatorState.STOPPED
        finally:
            try:
                simulator.stop()
            except Exception:
                pass

    def test_running_in_10s_2(self):
        simulator = self.make_simulator()
        try:
            constraints = SimulatorConstraints(max_real_time=10.0)
            simulator.start(
                constraints=constraints,
                simulate_in_thread=True,
                render_in_thread=False,
            )
            while simulator.state != SimulatorState.STOPPED:
                time.sleep(1)
            assert simulator.state is SimulatorState.STOPPED
        finally:
            try:
                simulator.stop()
            except Exception:
                pass

    def test_running_in_10s_3(self):
        simulator = self.make_simulator()
        try:
            constraints = SimulatorConstraints(max_real_time=10.0)
            simulator.start(
                constraints=constraints,
                simulate_in_thread=False,
                render_in_thread=True,
            )
            while simulator.state != SimulatorState.STOPPED:
                simulator.step()
                time.sleep(0.001)
                if simulator.current_number_of_steps == 10000:
                    simulator.stop()
            assert simulator.state is SimulatorState.STOPPED
        finally:
            try:
                simulator.stop()
            except Exception:
                pass

    def test_running_in_10s_4(self):
        simulator = self.make_simulator()
        try:
            constraints = SimulatorConstraints(max_real_time=10.0)
            simulator.start(
                constraints=constraints,
                simulate_in_thread=False,
                render_in_thread=False,
            )
            while simulator.state != SimulatorState.STOPPED:
                simulator.step()
                simulator.render()
                time.sleep(0.001)
                if simulator.current_number_of_steps == 10000:
                    simulator.stop()
            assert simulator.state is SimulatorState.STOPPED
        finally:
            try:
                simulator.stop()
            except Exception:
                pass

    def test_running_2_simulators(self):
        simulator1 = self.make_simulator(headless=self.headless)
        simulator2 = self.make_simulator(headless=True)
        simulator3 = self.make_simulator(headless=True)

        try:
            simulator1.start(simulate_in_thread=False, render_in_thread=True)
            simulator2.start(simulate_in_thread=False)
            simulator3.start(simulate_in_thread=False)

            for _ in range(10000):
                simulator1.step()
                simulator2.step()
                simulator3.step()

            simulator1.stop()
            simulator2.stop()
            simulator3.stop()

            assert simulator1.state is SimulatorState.STOPPED
            assert simulator2.state is SimulatorState.STOPPED
            assert simulator3.state is SimulatorState.STOPPED
        finally:
            for sim in (simulator1, simulator2, simulator3):
                try:
                    sim.stop()
                except Exception:
                    pass


class TestFrictionSetThroughOwningBody:
    """
    Exercises addressing friction through the body that owns the geoms rather than
    through a geom name.

    A world rebuilt from a :class:`~semantic_digital_twin.world.World` object names its
    geoms after the shape's type and object id, since shapes carry no name of their own,
    so the names the original scene file used are gone. Bodies keep their names, which
    makes the body the only stable handle onto a geom's friction.
    """

    scene = """
    <mujoco model="friction_through_body">
      <worldbody>
        <body name="cube">
          <geom type="box" size="0.02 0.02 0.02" friction="1.5 0.05 0.0005"/>
          <geom type="box" size="0.01 0.01 0.01" pos="0 0 0.05" friction="1.5 0.05 0.0005"/>
        </body>
        <body name="marker">
          <inertial mass="0.1" pos="0 0 0" diaginertia="0.001 0.001 0.001"/>
        </body>
      </worldbody>
      <keyframe>
        <key name="home"/>
      </keyframe>
    </mujoco>
    """
    """
    A scene whose geoms are deliberately unnamed, reproducing what a rebuilt world looks
    like, plus a body carrying no geom at all.
    """

    @pytest.fixture
    def simulator(self, tmp_path):
        scene_path = tmp_path / "friction_through_body.xml"
        scene_path.write_text(self.scene)
        return MujocoSimulator(
            _headless=True, _step_size=1e-3, file_path=str(scene_path)
        )

    def test_sets_friction_of_every_geom_of_the_body(self, simulator):
        result = simulator.set_body_friction("cube", numpy.array([0.2, 0.05, 0.0005]))

        assert (
            result.type
            is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
        )
        body_id = mujoco.mj_name2id(
            m=simulator._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, name="cube"
        )
        body = simulator._mj_model.body(body_id)
        first_geom, geom_count = int(body.geomadr[0]), int(body.geomnum[0])
        assert geom_count == 2
        for geom_id in range(first_geom, first_geom + geom_count):
            assert simulator._mj_model.geom_friction[geom_id] == pytest.approx(
                [0.2, 0.05, 0.0005]
            )

    def test_unnamed_geoms_are_unreachable_by_geom_name(self, simulator):
        """
        The geoms this scene's ``cube`` owns carry no name, so the scene-file name a
        caller would reach for does not resolve -- the reason friction has to be set
        through the body.
        """
        result = simulator.set_geom_friction(
            "cube_geom", numpy.array([0.2, 0.05, 0.0005])
        )

        assert (
            result.type is SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION
        )

    def test_unknown_body_reports_failure(self, simulator):
        result = simulator.set_body_friction(
            "not_a_body", numpy.array([0.2, 0.05, 0.0005])
        )

        assert (
            result.type is SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION
        )

    def test_body_without_geoms_reports_failure(self, simulator):
        """
        Reported as a failure rather than a silent no-op: a caller asking for friction on
        a body that has no contact geometry is asking for something that cannot happen.
        """
        result = simulator.set_body_friction("marker", numpy.array([0.2, 0.05, 0.0005]))

        assert (
            result.type is SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION
        )


class TestBodyVelocityReset:
    """
    Exercises clearing a body's velocity, which teleporting it back to a start pose does
    not do.

    A body that has picked up a large velocity keeps it across any number of position
    resets, so a simulation that has diverged once stays diverged until the velocity
    itself is cleared.
    """

    scene = """
    <mujoco model="velocity_reset">
      <worldbody>
        <body name="cube" pos="0 0 1">
          <freejoint/>
          <geom type="box" size="0.02 0.02 0.02"/>
        </body>
        <body name="anchor">
          <geom type="box" size="0.1 0.1 0.01"/>
        </body>
      </worldbody>
      <keyframe>
        <key name="home" qpos="0 0 1 1 0 0 0"/>
      </keyframe>
    </mujoco>
    """
    """
    A freely moving body whose velocity can be set, plus a body bolted to the world that
    owns no degree of freedom at all.
    """

    @pytest.fixture
    def simulator(self, tmp_path):
        scene_path = tmp_path / "velocity_reset.xml"
        scene_path.write_text(self.scene)
        return MujocoSimulator(
            _headless=True, _step_size=1e-3, file_path=str(scene_path)
        )

    def test_clears_the_velocity_of_a_moving_body(self, simulator):
        simulator._mj_data.qvel[:] = 5.0

        result = simulator.reset_body_velocity("cube")

        assert (
            result.type
            is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
        )
        assert simulator._mj_data.qvel == pytest.approx(numpy.zeros(6))

    def test_position_reset_alone_leaves_the_body_moving(self, simulator):
        """
        The reason a velocity reset is needed at all: putting the body back where it
        started leaves it travelling just as fast as before.
        """
        simulator._mj_data.qvel[:] = 5.0

        simulator.set_body_position("cube", numpy.array([0.0, 0.0, 1.0]))

        assert simulator._mj_data.qvel[:3] == pytest.approx([5.0, 5.0, 5.0])

    def test_unknown_body_reports_failure(self, simulator):
        result = simulator.reset_body_velocity("not_a_body")

        assert (
            result.type is SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION
        )

    def test_body_without_degrees_of_freedom_reports_failure(self, simulator):
        """
        A body welded to the world cannot carry a velocity, so asking to clear one is a
        mistake worth reporting rather than a silent success.
        """
        result = simulator.reset_body_velocity("anchor")

        assert (
            result.type is SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION
        )
