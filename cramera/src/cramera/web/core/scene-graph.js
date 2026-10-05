/* ============================================================================
 * core/scene-graph.js — the rules of the scene graph panel, apart from the page.
 *
 * The bridge serves a controlled simulation's bodies as flat rows naming their
 * parents (GET /simulation); this module orders them as a tree, reads a pose's
 * orientation as roll, pitch and yaw, and builds the bodies of the requests that
 * pause, resume, stop and change the simulation. The names below must match
 * cramera.live.simulation_control.
 *
 * Everything here is pure and testable under node; panels/scene_graph/panel.js
 * owns the page.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const ROUTE = {
    SCENE_GRAPH: '/simulation',
    PAUSE: '/simulation/pause',
    RESUME: '/simulation/resume',
    STOP: '/simulation/stop',
    EDIT: '/simulation/edit',
  };
  /* The bridge endpoints; cramera.live.simulation_control.SimulationRoute. */

  const STATE = { RUNNING: 'running', PAUSED: 'paused', STOPPED: 'stopped' };
  /* The run states; semantic_digital_twin's SimulationRunState. */

  const PLACEMENT = { LOOSE: 'loose', FIXED: 'fixed', UNMOVABLE: 'unmovable' };
  /* How a body is held in its world; semantic_digital_twin's Placement. */

  const CHANGE = { POSE: 'pose', MASS: 'mass', FRICTION: 'friction' };
  /* What an edit changes; cramera.live.simulation_control.SimulationChange. */

  const FRICTION_COEFFICIENTS = ['sliding', 'torsional', 'rolling'];
  /* The coefficients of a friction, in the order MuJoCo lists them. */

  const DEGREES_PER_RADIAN = 180 / Math.PI;

  /* The rows in tree order — every body directly after its parent, siblings by name —
     each with its depth below the roots. A row whose parent is not listed is a root. */
  function tree(bodies) {
    const byName = {};
    (bodies || []).forEach(function (body) { byName[body.name] = body; });
    const children = {};
    const roots = [];
    (bodies || []).forEach(function (body) {
      if (body.parent && byName[body.parent]) {
        (children[body.parent] = children[body.parent] || []).push(body);
      } else {
        roots.push(body);
      }
    });
    const byLabel = function (first, second) { return first.name < second.name ? -1 : 1; };
    const ordered = [];
    function visit(body, depth) {
      ordered.push({ body: body, depth: depth });
      (children[body.name] || []).sort(byLabel).forEach(function (child) {
        visit(child, depth + 1);
      });
    }
    roots.sort(byLabel).forEach(function (root) { visit(root, 0); });
    return ordered;
  }

  /* The last segment of a prefixed name, which is what a body is called on screen. */
  function shortName(name) {
    const parts = String(name || '').split('/');
    return parts[parts.length - 1];
  }

  /* Roll, pitch and yaw in radians of a [qx, qy, qz, qw] orientation, rotating about
     the fixed x, y and z axes in that order. */
  function rollPitchYaw(quaternion) {
    const x = quaternion[0], y = quaternion[1], z = quaternion[2], w = quaternion[3];
    const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
    const sine = Math.max(-1, Math.min(1, 2 * (w * y - z * x)));
    const pitch = Math.asin(sine);
    const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    return [roll, pitch, yaw];
  }

  /* A [x, y, z, qx, qy, qz, qw] pose as the panel edits it: metres and degrees. */
  function editablePose(pose) {
    const angles = rollPitchYaw(pose.slice(3)).map(function (angle) {
      return angle * DEGREES_PER_RADIAN;
    });
    return { x: pose[0], y: pose[1], z: pose[2], roll: angles[0], pitch: angles[1], yaw: angles[2] };
  }

  /* Which run controls make sense in a run state. */
  function controlsFor(state) {
    return {
      pause: state === STATE.RUNNING,
      resume: state === STATE.PAUSED,
      stop: state === STATE.RUNNING || state === STATE.PAUSED,
    };
  }

  /* Which of a body's attributes can be changed. */
  function editable(body) {
    return {
      pose: body.placement === PLACEMENT.LOOSE || body.placement === PLACEMENT.FIXED,
      mass: body.mass > 0,
      friction: body.friction !== null && body.friction !== undefined,
    };
  }

  function editRequest(name, change, value) {
    return { body: name, change: change, value: value };
  }

  /* The edit request putting a body at a pose given in metres and degrees. */
  function poseRequest(name, pose) {
    return editRequest(name, CHANGE.POSE, {
      x: pose.x, y: pose.y, z: pose.z,
      roll: pose.roll / DEGREES_PER_RADIAN,
      pitch: pose.pitch / DEGREES_PER_RADIAN,
      yaw: pose.yaw / DEGREES_PER_RADIAN,
    });
  }

  function massRequest(name, mass) {
    return editRequest(name, CHANGE.MASS, mass);
  }

  function frictionRequest(name, friction) {
    const value = {};
    FRICTION_COEFFICIENTS.forEach(function (coefficient) { value[coefficient] = friction[coefficient]; });
    return editRequest(name, CHANGE.FRICTION, value);
  }

  global.SceneGraph = {
    ROUTE: ROUTE,
    STATE: STATE,
    PLACEMENT: PLACEMENT,
    FRICTION_COEFFICIENTS: FRICTION_COEFFICIENTS,
    tree: tree,
    shortName: shortName,
    rollPitchYaw: rollPitchYaw,
    editablePose: editablePose,
    controlsFor: controlsFor,
    editable: editable,
    poseRequest: poseRequest,
    massRequest: massRequest,
    frictionRequest: frictionRequest,
  };
})(typeof window !== 'undefined' ? window : this);
