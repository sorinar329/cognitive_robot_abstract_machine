"""
Visualizes the pouring demo scene statically: two cups on the floor with
particles inside the source cup. The source cup is pinned upright while full
physics runs so particles settle under gravity.

Run with:
    python -m segmind.demos.visualize_pouring_scene
"""
from __future__ import annotations

import time

import numpy

from physics_simulators.mujoco_simulator import MujocoSimulator
from segmind.demos.pouring_demo import PouringDemo, SCENE_FILE

if __name__ == "__main__":
    _simulator = MujocoSimulator(_headless=False, file_path=SCENE_FILE)
    _simulator.start(simulate_in_thread=False, render_in_thread=False)
    _demo = PouringDemo(simulator=_simulator)

    _source_joint = _simulator.get_body_joints(_demo.source_cup_name).result[0]
    _qpos_adr = _source_joint.qposadr[0]
    _qdof_adr = _source_joint.dofadr[0]
    _home = _demo._home_position.copy()
    _identity_quat = numpy.array([1.0, 0.0, 0.0, 0.0])

    while _simulator.renderer.is_running():
        with _simulator.renderer.lock():
            # Pin the source cup upright via freejoint; let particles settle under gravity.
            _simulator._mj_data.qpos[_qpos_adr : _qpos_adr + 3] = _home
            _simulator._mj_data.qpos[_qpos_adr + 3 : _qpos_adr + 7] = _identity_quat
            _simulator._mj_data.qvel[_qdof_adr : _qdof_adr + 6] = 0.0
            _simulator.step()
        _simulator.renderer.sync()
        time.sleep(0.004)

    _simulator.stop()
