// Unit tests for panels/activity/panel.js (node:test): rendering a demo's
// /activity log entries into cards.
//
// panel.js is loaded with its free variables (Panels, fetch) bound as explicit
// function parameters rather than through global/window stubs, since the file itself
// never touches `window` or `document` directly (it only reaches DOM elements handed to
// it via its own `root` parameter) — see test_graph_panel.js's own header for why.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const SOURCE = fs.readFileSync(path.join(WEB, 'panels/activity/panel.js'), 'utf8');

function flush() {
  return new Promise(function (resolve) { setTimeout(resolve, 0); });
}

// %% stubs of the interfaces panel.js reads
function makeElement() {
  return {
    style: {},
    textContent: '',
    innerHTML: '',
    classList: { toggle() {}, add() {}, remove() {} },
  };
}

function makeRoot() {
  const byId = {
    '#activity-status': makeElement(),
    '#activity-feed': makeElement(),
  };
  return {
    innerHTML: '',
    querySelector(selector) { return byId[selector]; },
    byId: byId,
  };
}

function makeBus() {
  const handlers = {};
  return {
    on(event, cb) { (handlers[event] = handlers[event] || []).push(cb); },
    emit(event, payload) { (handlers[event] || []).forEach(function (cb) { cb(payload); }); },
  };
}

function makeFetch(responses) {
  let calls = 0;
  const fetchFn = async function fetch(url) {
    calls += 1;
    const body = responses[url];
    if (!body) throw new Error('unexpected fetch: ' + url);
    return { status: 200, json: async function () { return body; } };
  };
  fetchFn.callCount = function () { return calls; };
  return fetchFn;
}

function loadPanel(fetchFn) {
  let factory = null;
  const Panels = { define(id, f) { factory = f; } };
  new Function('Panels', 'fetch', SOURCE)(Panels, fetchFn);
  return factory;
}

test('an iteration with a stacked cube and a diagnosed correction is rendered', async function () {
  const fetchFn = makeFetch({
    'http://bridge/activity': {
      entries: [
        {
          iteration: 3,
          totalIterations: 20,
          durationSeconds: 42.345,
          simulationDiverged: false,
          fullStackIntact: true,
          segmindApproved: true,
          cubes: [
            {
              stepLabel: 'cube1 onto cube0',
              finalSucceeded: false,
              correctionAttempts: 1,
              diagnoses: [
                {
                  actionName: 'pickup',
                  primary: {
                    variableName: 'final_approach_linear_velocity',
                    observedValue: 0.2144,
                    observedSupportProbability: 0.0,
                    correctedValue: 0.0937,
                    correctedSupportProbability: 0.0457,
                  },
                },
              ],
            },
          ],
        },
      ],
    },
  });
  const factory = loadPanel(fetchFn);
  const root = makeRoot();
  const bus = makeBus();
  const instance = factory(root, bus);
  try {
    bus.emit('live:changed', { on: true, url: 'http://bridge' });
    await flush();

    const feed = root.byId['#activity-feed'].innerHTML;
    assert.match(feed, /Iteration 3 \/ 20/);
    assert.match(feed, /cube1 onto cube0/);
    assert.match(feed, /on floor/);
    assert.match(feed, /final_approach_linear_velocity/);
    assert.match(feed, /0\.2144/);
    assert.match(feed, /0\.0937/);
    assert.strictEqual(root.byId['#activity-status'].textContent, 'live');
  } finally {
    instance.destroy();
  }
});

test('going live with no entries yet shows the waiting state, not an error', async function () {
  const fetchFn = makeFetch({ 'http://bridge/activity': { entries: [] } });
  const factory = loadPanel(fetchFn);
  const root = makeRoot();
  const bus = makeBus();
  const instance = factory(root, bus);
  try {
    bus.emit('live:changed', { on: true, url: 'http://bridge' });
    await flush();

    assert.match(root.byId['#activity-feed'].innerHTML, /No iterations reported yet/);
  } finally {
    instance.destroy();
  }
});

test('going offline stops polling and updates the status', async function () {
  const fetchFn = makeFetch({ 'http://bridge/activity': { entries: [] } });
  const factory = loadPanel(fetchFn);
  const root = makeRoot();
  const bus = makeBus();
  const instance = factory(root, bus);
  try {
    bus.emit('live:changed', { on: true, url: 'http://bridge' });
    await flush();
    const callsWhileLive = fetchFn.callCount();

    bus.emit('live:changed', { on: false, url: '' });
    await flush();

    assert.strictEqual(root.byId['#activity-status'].textContent, 'not live');
    assert.strictEqual(fetchFn.callCount(), callsWhileLive);   // no further polling
  } finally {
    instance.destroy();
  }
});
