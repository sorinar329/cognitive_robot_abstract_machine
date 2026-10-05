const BrowserSource = require('./browser_source');
// Unit tests for web/core/scene-graph.js (node:test): the scene graph panel's rules --
// tree order, orientation as roll/pitch/yaw, which controls and edits a body allows,
// and the request bodies posted to the bridge.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', BrowserSource.read(path.join(WEB, 'core/scene-graph.js')))(scope);
  return scope.SceneGraph;
}

function body(name, parent, extra) {
  return Object.assign({ name: name, parent: parent, placement: 'unmovable', mass: 0, friction: null }, extra || {});
}

// %% tree order
test('every body follows its parent, one level deeper', function () {
  const graph = load();
  const rows = graph.tree([
    body('w/piece', 'w/world'),
    body('w/world', null),
    body('w/board', 'w/world'),
    body('w/handle', 'w/board'),
  ]);
  assert.deepStrictEqual(
    rows.map(function (row) { return [row.body.name, row.depth]; }),
    [['w/world', 0], ['w/board', 1], ['w/handle', 2], ['w/piece', 1]]
  );
});

test('a body whose parent is not listed is shown as a root', function () {
  const graph = load();
  const rows = graph.tree([body('w/orphan', 'w/missing')]);
  assert.deepStrictEqual(rows.map(function (row) { return row.depth; }), [0]);
});

test('a body is called by the last segment of its name', function () {
  assert.strictEqual(load().shortName('montessori/shape_sorting_board'), 'shape_sorting_board');
});

// %% orientation
test('a quarter turn about z reads as a yaw of a quarter turn', function () {
  const half = Math.SQRT1_2;
  const angles = load().rollPitchYaw([0, 0, half, half]);
  assert.ok(Math.abs(angles[0]) < 1e-9);
  assert.ok(Math.abs(angles[1]) < 1e-9);
  assert.ok(Math.abs(angles[2] - Math.PI / 2) < 1e-9);
});

test('an editable pose and a pose request turn degrees into radians and back', function () {
  const graph = load();
  const pose = graph.editablePose([0.1, 0.2, 0.3, 0, 0, Math.SQRT1_2, Math.SQRT1_2]);
  const request = graph.poseRequest('w/board', pose);
  assert.ok(Math.abs(pose.yaw - 90) < 1e-9);
  assert.strictEqual(request.body, 'w/board');
  assert.strictEqual(request.change, 'pose');
  assert.ok(Math.abs(request.value.yaw - Math.PI / 2) < 1e-9);
  assert.deepStrictEqual([request.value.x, request.value.y, request.value.z], [0.1, 0.2, 0.3]);
});

// %% what may be done
test('only a running or paused simulation offers the matching controls', function () {
  const graph = load();
  assert.deepStrictEqual(graph.controlsFor(graph.STATE.RUNNING), { pause: true, resume: false, stop: true });
  assert.deepStrictEqual(graph.controlsFor(graph.STATE.PAUSED), { pause: false, resume: true, stop: true });
  assert.deepStrictEqual(graph.controlsFor(graph.STATE.STOPPED), { pause: false, resume: false, stop: false });
});

test('a loose or fixed body can be moved, an unmovable one cannot', function () {
  const graph = load();
  assert.strictEqual(graph.editable(body('a', null, { placement: graph.PLACEMENT.LOOSE })).pose, true);
  assert.strictEqual(graph.editable(body('b', null, { placement: graph.PLACEMENT.FIXED })).pose, true);
  assert.strictEqual(graph.editable(body('c', null, { placement: graph.PLACEMENT.UNMOVABLE })).pose, false);
});

test('a body without mass or contact offers neither to change', function () {
  const graph = load();
  assert.deepStrictEqual(graph.editable(body('a', null)), { pose: false, mass: false, friction: false });
  const solid = body('b', null, { mass: 0.2, friction: { sliding: 1, torsional: 0.005, rolling: 0.0001 } });
  assert.strictEqual(graph.editable(solid).mass, true);
  assert.strictEqual(graph.editable(solid).friction, true);
});

// %% requests
test('a friction request carries exactly the three coefficients', function () {
  const graph = load();
  const request = graph.frictionRequest('w/piece', { sliding: 0.5, torsional: 0.01, rolling: 0.001, extra: 3 });
  assert.deepStrictEqual(request, {
    body: 'w/piece', change: 'friction', value: { sliding: 0.5, torsional: 0.01, rolling: 0.001 },
  });
});

test('a mass request carries the mass as its value', function () {
  assert.deepStrictEqual(load().massRequest('w/piece', 0.4), { body: 'w/piece', change: 'mass', value: 0.4 });
});
