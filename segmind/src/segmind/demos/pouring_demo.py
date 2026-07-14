from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy

from physics_simulators.mujoco_simulator import MujocoSimulator

SCENE_FILE = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "resources",
        "mjcf",
        "pouring_demo.xml",
    )
)


@dataclass
class PouringResult:
    """Outcome of a single :meth:`PouringDemo.pour` call."""

    final_tilt_angle: float
    """Tilt angle of the source cup in radians at the end of the pour."""

    cups_in_contact: bool
    """Whether the source and target cups were in direct contact at the end of the pour."""

    steps_executed: int
    """Number of simulation steps performed during the pour."""


@dataclass
class PouringDemo:
    """
    Demonstration of a cup-pouring motion in MuJoCo.

    The source cup is kinematically controlled via a freejoint. The motion
    sequence is: lift the source cup, translate it above the target cup, then
    tilt it so particles flow out under gravity.

    At tilt = 0 the cup is upright. At :attr:`pouring_angle` (π/2) the cup is
    horizontal with its opening directed toward the target cup centre.

    Uses :class:`~physics_simulators.mujoco_simulator.MujocoSimulator`.

    .. note::
        Full particle physics requires calling
        :meth:`~physics_simulators.base_simulator.BaseSimulator.step` after
        each :meth:`tilt_to_angle`. The :meth:`pour` method performs only
        kinematic updates (``mj_step1``) and is designed for fast test use.
        The ``__main__`` block runs the complete physics loop with the
        settle → lift → move → tilt phases.
    """

    simulator: MujocoSimulator
    """The MuJoCo simulator driving the scene."""

    source_cup_name: str = "source_cup"
    """Name of the MuJoCo body that acts as the tilting source cup."""

    target_cup_name: str = "target_cup"
    """Name of the MuJoCo body that acts as the stationary target cup."""

    pouring_angle: float = field(default=2.0 * math.pi / 3.0)
    """
    Target tilt angle in radians at which pouring is considered complete.

    Defaults to 2π/3 (120°): the opening faces down-right so gravity drives
    particles out through the opening into the target cup.
    """

    lift_height: float = 0.25
    """Height in metres to raise the source cup before translating it over the target."""

    cup_half_height: float = 0.14
    """
    Distance in metres from the source cup centre to its rim along the local Z axis.

    When the cup is tilted π/2 around Y, the rim travels this distance in world X.
    Used to offset the source cup so that the rim lands above the target cup centre.
    """

    _home_position: numpy.ndarray = field(init=False, default_factory=lambda: numpy.zeros(3))
    """World-frame position of the source cup captured at construction time."""

    def __post_init__(self) -> None:
        # body.xpos (used by get_body_position) is a forward-kinematics derived
        # quantity and may not be populated yet.  For a freejoint the position is
        # authoritative in qpos, which mj_resetDataKeyframe always initialises from qpos0.
        source_joint = self.simulator.get_body_joints(self.source_cup_name).result[0]
        qpos_adr = source_joint.qposadr[0]
        self._home_position = numpy.array(
            self.simulator._mj_data.qpos[qpos_adr : qpos_adr + 3]
        )

    @property
    def pour_position(self) -> numpy.ndarray:
        """
        Source cup world position from which a π/2 tilt directs particles into the target cup.

        The cup is offset from the target cup centre by :attr:`cup_half_height` along X so
        that after tilting, the rim is directly above the target cup centre.
        """
        target = numpy.array(self.simulator.get_body_position(self.target_cup_name).result)
        return numpy.array([
            target[0] - self.cup_half_height * math.sin(self.pouring_angle),
            self._home_position[1],
            self._home_position[2] + self.lift_height,
        ])

    def tilt_to_angle(self, angle: float) -> None:
        """
        Set the source cup tilt to the given angle via Y-axis rotation.

        Internally calls ``mj_step1`` to update kinematics and contact state.

        :param angle: Desired tilt angle in radians.
        """
        half_angle = angle / 2.0
        quaternion = numpy.array([math.cos(half_angle), 0.0, math.sin(half_angle), 0.0])
        self.simulator.set_body_quaternion(self.source_cup_name, quaternion)

    def get_tilt_angle(self) -> float:
        """
        Return the current tilt angle of the source cup.

        :return: Tilt angle in radians, extracted from the Y-axis rotation component of the body quaternion.
        """
        quaternion = self.simulator.get_body_quaternion(self.source_cup_name).result
        return 2.0 * math.atan2(float(quaternion[2]), float(quaternion[0]))

    @property
    def is_pouring_complete(self) -> bool:
        """``True`` when the source cup tilt has reached :attr:`pouring_angle`."""
        return self.get_tilt_angle() >= self.pouring_angle

    def are_cups_in_contact(self) -> bool:
        """Return ``True`` if any geom of the source cup directly touches the target cup."""
        result = self.simulator.get_contact_bodies(self.source_cup_name)
        return self.target_cup_name in result.result

    def pour(self, steps: int = 100) -> PouringResult:
        """
        Gradually tilt the source cup from upright to :attr:`pouring_angle`.

        Each step updates the tilt angle and advances the kinematic state via
        ``mj_step1``. For full particle physics use the ``__main__`` script which
        also calls :meth:`~physics_simulators.base_simulator.BaseSimulator.step`.

        :param steps: Number of tilt steps to execute.
        :return: A :class:`PouringResult` describing the final state.
        """
        for step_index in range(steps):
            angle = self.pouring_angle * (step_index + 1) / steps
            self.tilt_to_angle(angle)
        return PouringResult(
            final_tilt_angle=self.get_tilt_angle(),
            cups_in_contact=self.are_cups_in_contact(),
            steps_executed=steps,
        )


if __name__ == "__main__":
    _SETTLE_STEPS = 150   # hold upright so particles settle in the cup
    _LIFT_STEPS = 200     # raise cup and particles straight up
    _MOVE_STEPS = 200     # translate cup and particles horizontally to pour position
    _RAMP_STEPS = 300     # tilt cup; particles released to free physics and flow out
    _HOLD_STEPS = 300     # hold tilted while remaining particles finish flowing out
    _STEP_DELAY = 0.004   # sleep between steps to run at roughly real time

    _simulator = MujocoSimulator(_headless=False, file_path=SCENE_FILE)
    _simulator.start(simulate_in_thread=False, render_in_thread=False)
    _demo = PouringDemo(simulator=_simulator)

    # Source cup freejoint addresses.
    _source_joint = _simulator.get_body_joints(_demo.source_cup_name).result[0]
    _qpos_adr = _source_joint.qposadr[0]
    _qdof_adr = _source_joint.dofadr[0]

    # Particle freejoint addresses (particle_0 … particle_7).
    _particle_joint_addrs: list[tuple[int, int]] = [
        (j.qposadr[0], j.dofadr[0])
        for j in (
            _simulator.get_body_joints(f"particle_{i}").result[0] for i in range(8)
        )
    ]

    _home = _demo._home_position.copy()
    _lifted = _home + numpy.array([0.0, 0.0, _demo.lift_height])
    _pour_pos = _demo.pour_position
    _identity_quat = numpy.array([1.0, 0.0, 0.0, 0.0])
    _pour_half = _demo.pouring_angle / 2.0
    _pour_quat = numpy.array([math.cos(_pour_half), 0.0, math.sin(_pour_half), 0.0])

    def _apply_cup_kinematics(position: numpy.ndarray, quaternion: numpy.ndarray) -> None:
        _simulator._mj_data.qpos[_qpos_adr : _qpos_adr + 3] = position
        _simulator._mj_data.qpos[_qpos_adr + 3 : _qpos_adr + 7] = quaternion
        _simulator._mj_data.qvel[_qdof_adr : _qdof_adr + 6] = 0.0

    def _apply_particle_kinematics(positions: list[numpy.ndarray]) -> None:
        for (p_qpos, p_qdof), pos in zip(_particle_joint_addrs, positions):
            _simulator._mj_data.qpos[p_qpos : p_qpos + 3] = pos
            _simulator._mj_data.qpos[p_qpos + 3 : p_qpos + 7] = _identity_quat
            _simulator._mj_data.qvel[p_qdof : p_qdof + 6] = 0.0

    def _step(
        cup_position: numpy.ndarray,
        cup_quaternion: numpy.ndarray,
        particle_positions: list[numpy.ndarray] | None = None,
    ) -> None:
        with _simulator.renderer.lock():
            _apply_cup_kinematics(cup_position, cup_quaternion)
            if particle_positions is not None:
                _apply_particle_kinematics(particle_positions)
            _simulator.step()
        _simulator.renderer.sync()
        time.sleep(_STEP_DELAY)

    # Phase 1 – settle: hold cup upright; let particles settle under gravity.
    for _ in range(_SETTLE_STEPS):
        _step(_home, _identity_quat)

    # Capture each particle's settled position and its offset from the cup centre.
    # These offsets are maintained rigidly during lift and move so the particles
    # travel with the cup instead of being left behind when contact breaks.
    _settle_cup_pos = numpy.array(_simulator._mj_data.qpos[_qpos_adr : _qpos_adr + 3])
    _particle_offsets = [
        numpy.array(_simulator._mj_data.qpos[p_qpos : p_qpos + 3]) - _settle_cup_pos
        for p_qpos, _ in _particle_joint_addrs
    ]

    def _carried_particle_positions(cup_position: numpy.ndarray) -> list[numpy.ndarray]:
        return [cup_position + offset for offset in _particle_offsets]

    def _rotate_y(angle: float, vector: numpy.ndarray) -> numpy.ndarray:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return numpy.array([
            cos_a * vector[0] + sin_a * vector[2],
            vector[1],
            -sin_a * vector[0] + cos_a * vector[2],
        ])

    # Phase 2 – lift: carry particles rigidly with the cup straight up.
    for _i in range(_LIFT_STEPS):
        _fraction = (_i + 1) / _LIFT_STEPS
        _cup_pos = _home + numpy.array([0.0, 0.0, _demo.lift_height * _fraction])
        _step(_cup_pos, _identity_quat, _carried_particle_positions(_cup_pos))

    # Phase 3 – move: carry particles horizontally to the pour position.
    for _i in range(_MOVE_STEPS):
        _fraction = (_i + 1) / _MOVE_STEPS
        _cup_pos = _lifted + (_pour_pos - _lifted) * _fraction
        _step(_cup_pos, _identity_quat, _carried_particle_positions(_cup_pos))

    # Particle positions in the cup's local frame at the start of the tilt.
    # Rotating these by the tilt angle carries particles rigidly with the cup
    # so they cannot escape through the walls during the sweep.
    _particle_local_positions = [
        pos - _pour_pos for pos in _carried_particle_positions(_pour_pos)
    ]

    # Phase 4 – tilt: carry particles by rotating their local-frame positions with the
    # cup.  At 2π/3 the opening faces down-right so gravity drives them out when released.
    for _i in range(_RAMP_STEPS):
        _fraction = (_i + 1) / _RAMP_STEPS
        _angle = _demo.pouring_angle * _fraction
        _half = _angle / 2.0
        _quat = numpy.array([math.cos(_half), 0.0, math.sin(_half), 0.0])
        _tilted_particle_positions = [
            _pour_pos + _rotate_y(_angle, p_local)
            for p_local in _particle_local_positions
        ]
        _step(_pour_pos, _quat, _tilted_particle_positions)

    # Phase 5 – hold: release particles to free physics.  With the cup tilted to 2π/3
    # the opening faces down-right and gravity pulls particles through it into the target.
    for _ in range(_HOLD_STEPS):
        _step(_pour_pos, _pour_quat)

    print(f"Final tilt angle : {_demo.get_tilt_angle():.4f} rad")
    print(f"Cups in contact  : {_demo.are_cups_in_contact()}")

    # Keep stepping physics with the cup pinned until the viewer is closed.
    while _simulator.renderer.is_running():
        _step(_pour_pos, _pour_quat)

    _simulator.stop()
