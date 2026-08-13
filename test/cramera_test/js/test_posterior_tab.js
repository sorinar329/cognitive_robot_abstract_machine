// Unit tests for panels/graph/panel.js's Posterior tab (node:test): loading both
// models' variables into one flat, navigable list and rendering each as an SVG chart.
//
// panel.js is loaded with its free variables (Panels, Graph, fetch) bound as explicit
// function parameters rather than through global/window stubs — see test_graph_panel.js's
// own header for why.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const SOURCE = fs.readFileSync(path.join(WEB, 'panels/graph/panel.js'), 'utf8');

function flush() {
  return new Promise(function (resolve) { setTimeout(resolve, 0); });
}

// %% stubs of the interfaces panel.js reads
function makeElement() {
  return {
    style: {},
    textContent: '',
    innerHTML: '',
    disabled: false,
    classList: { toggle() {}, add() {}, remove() {} },
    addEventListener() {},
    querySelectorAll() { return []; },
  };
}

function makeClickable() {
  let onClick = null;
  const el = makeElement();
  el.addEventListener = function (event, cb) { if (event === 'click') onClick = cb; };
  el.click = function () { if (onClick) onClick(); };
  return el;
}

function makeTabButton(view) {
  const el = makeClickable();
  el.dataset = { view: view };
  return el;
}

function makeRoot() {
  const byId = {
    '#graph-empty': makeElement(),
    '#graph-nav': makeElement(),
    '#gnav-up': makeElement(),
    '#gnav-home': makeElement(),
    '#gnav-path': makeElement(),
    '#gt-live': makeElement(),
    '#graph': makeElement(),
    '#legend': makeElement(),
    '#posterior-view': makeElement(),
    '#posterior-title': makeElement(),
    '#posterior-chart': makeElement(),
    '#posterior-legend': makeElement(),
    '#posterior-prev': makeClickable(),
    '#posterior-next': makeClickable(),
    '#posterior-evidence-rows': makeElement(),
    '#posterior-evidence-add': makeClickable(),
    '#posterior-calculate': makeClickable(),
    '#posterior-reset': makeClickable(),
    '#posterior-evidence-status': makeElement(),
  };
  const buttons = ['knowledge', 'kinematics', 'plan', 'chart', 'posterior'].map(makeTabButton);
  byId['#graph-tabs'] = { querySelectorAll() { return buttons; } };
  return {
    innerHTML: '',
    querySelector(selector) { return byId[selector]; },
    byId: byId,
    buttons: buttons,
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
  return async function fetch(url) {
    const body = responses[url];
    if (!body) return { ok: false, status: 404, json: async function () { return {}; } };
    return { ok: true, status: 200, json: async function () { return body; } };
  };
}

// GET fetches answer from `responses` as makeFetch does; a POST to `postUrl` is handed
// its parsed JSON body and answers with whatever it returns.
function makeFetchWithPost(responses, postUrl, postHandler) {
  const get = makeFetch(responses);
  return async function fetch(url, options) {
    if (options && options.method === 'POST' && url === postUrl) {
      const body = postHandler(JSON.parse(options.body));
      return { ok: true, status: 200, json: async function () { return body; } };
    }
    return get(url);
  };
}

function loadPanelWithFetch(fetchFn) {
  let factory = null;
  const Panels = { define(id, f) { factory = f; } };
  const Graph = {
    attach() {}, build() {}, onSelect() {}, onDoubleSelect() {},
    highlight() {}, reset() {}, setStatuses() { return false; },
  };
  new Function('Panels', 'Graph', 'fetch', SOURCE)(Panels, Graph, fetchFn);
  return factory;
}

function loadPanel(responses) {
  return loadPanelWithFetch(makeFetch(responses));
}

function pickupVariable(name, extra) {
  return Object.assign({
    samples: [0.0, 0.5, 1.0],
    pdf: [0.1, 0.9, 0.1],
    cdf: [0.0, 0.5, 1.0],
    expectation: 0.5,
    modes: [[0.4, 0.6]],
    modeHeight: 0.95,
  }, extra || {}, { __name: name });
}

const RESPONSES = {
  '/data/posterior/pickup.json': {
    order: ['object_friction', 'pre_approach_linear_velocity'],
    variables: {
      object_friction: pickupVariable('object_friction'),
      pre_approach_linear_velocity: pickupVariable('pre_approach_linear_velocity'),
    },
  },
  '/data/posterior/place.json': {
    order: ['transport_linear_velocity'],
    variables: {
      transport_linear_velocity: pickupVariable('transport_linear_velocity'),
    },
  },
};

test('clicking the Posterior tab shows the first variable of the first model', async function () {
  const factory = loadPanel(RESPONSES);
  const root = makeRoot();
  const bus = makeBus();
  const instance = factory(root, bus);
  try {
    root.buttons.find(function (b) { return b.dataset.view === 'posterior'; }).click();
    await flush();

    assert.strictEqual(root.byId['#posterior-title'].textContent, 'Pickup · object_friction');
    assert.match(root.byId['#posterior-chart'].innerHTML, /<svg/);
    assert.match(root.byId['#posterior-chart'].innerHTML, /class="posterior-pdf"/);
    assert.match(root.byId['#posterior-chart'].innerHTML, /class="posterior-cdf"/);
    assert.match(root.byId['#posterior-chart'].innerHTML, /class="posterior-mode"/);
    assert.match(root.byId['#posterior-chart'].innerHTML, /class="posterior-expectation"/);
    // graph/legend hidden, posterior view shown
    assert.strictEqual(root.byId['#graph'].style.display, 'none');
    assert.strictEqual(root.byId['#posterior-view'].style.display, '');
  } finally {
    instance.destroy();
  }
});

test('next/prev walk through both models\' variables as one flat list', async function () {
  const factory = loadPanel(RESPONSES);
  const root = makeRoot();
  const bus = makeBus();
  const instance = factory(root, bus);
  try {
    root.buttons.find(function (b) { return b.dataset.view === 'posterior'; }).click();
    await flush();
    assert.strictEqual(root.byId['#posterior-title'].textContent, 'Pickup · object_friction');
    assert.strictEqual(root.byId['#posterior-prev'].disabled, true);

    root.byId['#posterior-next'].click();
    assert.strictEqual(
      root.byId['#posterior-title'].textContent, 'Pickup · pre_approach_linear_velocity'
    );
    assert.strictEqual(root.byId['#posterior-prev'].disabled, false);

    root.byId['#posterior-next'].click();
    assert.strictEqual(
      root.byId['#posterior-title'].textContent, 'Place · transport_linear_velocity'
    );
    assert.strictEqual(root.byId['#posterior-next'].disabled, true);

    // next again does nothing past the end
    root.byId['#posterior-next'].click();
    assert.strictEqual(
      root.byId['#posterior-title'].textContent, 'Place · transport_linear_velocity'
    );

    root.byId['#posterior-prev'].click();
    assert.strictEqual(
      root.byId['#posterior-title'].textContent, 'Pickup · pre_approach_linear_velocity'
    );
  } finally {
    instance.destroy();
  }
});

test('switching away from Posterior and back does not re-fetch (data is cached)', async function () {
  let fetchCount = 0;
  const factory = (function () {
    let factoryFn = null;
    const Panels = { define(id, f) { factoryFn = f; } };
    const Graph = {
      attach() {}, build() {}, onSelect() {}, onDoubleSelect() {},
      highlight() {}, reset() {}, setStatuses() { return false; },
    };
    const fetchFn = async function fetch(url) {
      if (url.indexOf('/data/posterior/') === 0) fetchCount += 1;
      const body = RESPONSES[url] || { ok: true, nodes: [], edges: [], details: {} };
      return { ok: true, status: 200, json: async function () { return body; } };
    };
    new Function('Panels', 'Graph', 'fetch', SOURCE)(Panels, Graph, fetchFn);
    return factoryFn;
  })();
  const root = makeRoot();
  const bus = makeBus();
  const instance = factory(root, bus);
  try {
    await flush();   // let the boot-time knowledge-tab load settle first

    root.buttons.find(function (b) { return b.dataset.view === 'posterior'; }).click();
    await flush();
    const afterFirstVisit = fetchCount;
    assert.strictEqual(afterFirstVisit, 2);   // pickup.json + place.json

    root.buttons.find(function (b) { return b.dataset.view === 'knowledge'; }).click();
    await flush();
    root.buttons.find(function (b) { return b.dataset.view === 'posterior'; }).click();
    await flush();

    assert.strictEqual(fetchCount, afterFirstVisit);   // no additional posterior fetches
  } finally {
    instance.destroy();
  }
});

test('no exported posterior data shows a helpful message instead of an empty chart', async function () {
  const factory = loadPanel({});
  const root = makeRoot();
  const bus = makeBus();
  const instance = factory(root, bus);
  try {
    root.buttons.find(function (b) { return b.dataset.view === 'posterior'; }).click();
    await flush();

    assert.match(root.byId['#posterior-chart'].innerHTML, /export_posterior_plots\.py/);
    assert.strictEqual(root.byId['#posterior-prev'].disabled, true);
    assert.strictEqual(root.byId['#posterior-next'].disabled, true);
  } finally {
    instance.destroy();
  }
});

// %% evidence panel
test('+ variable adds a constraint row scoped to the current model', async function () {
  const factory = loadPanel(RESPONSES);
  const root = makeRoot();
  const instance = factory(root, makeBus());
  try {
    root.buttons.find(function (b) { return b.dataset.view === 'posterior'; }).click();
    await flush();

    root.byId['#posterior-evidence-add'].click();

    const rowsHtml = root.byId['#posterior-evidence-rows'].innerHTML;
    assert.match(rowsHtml, /posterior-evidence-row/);
    assert.match(rowsHtml, /object_friction/);
    // the other pickup variable is offered too, but not the place-only one
    assert.match(rowsHtml, /pre_approach_linear_velocity/);
    assert.doesNotMatch(rowsHtml, /transport_linear_velocity/);
  } finally {
    instance.destroy();
  }
});

test('Calculate Posterior POSTs the current model, its evidence and query variables', async function () {
  let requestBody = null;
  const fetchFn = makeFetchWithPost(RESPONSES, '/api/model/posterior', function (body) {
    requestBody = body;
    return {
      variables: {
        object_friction: pickupVariable('object_friction', { expectation: 0.75 }),
        pre_approach_linear_velocity: pickupVariable('pre_approach_linear_velocity'),
      },
    };
  });
  const factory = loadPanelWithFetch(fetchFn);
  const root = makeRoot();
  const instance = factory(root, makeBus());
  try {
    root.buttons.find(function (b) { return b.dataset.view === 'posterior'; }).click();
    await flush();
    root.byId['#posterior-evidence-add'].click();

    root.byId['#posterior-calculate'].click();
    await flush();

    assert.strictEqual(requestBody.model, 'pickup');
    assert.deepStrictEqual(
      requestBody.queryVariables, ['object_friction', 'pre_approach_linear_velocity']
    );
    assert.strictEqual(requestBody.evidence.length, 1);
    assert.strictEqual(requestBody.evidence[0].variable, 'object_friction');

    // the chart for the currently shown variable reflects the recomputed data
    assert.match(root.byId['#posterior-evidence-status'].textContent, /Conditioned on 1/);
  } finally {
    instance.destroy();
  }
});

test('a calculation error from the server is shown, not thrown', async function () {
  const fetchFn = makeFetchWithPost(RESPONSES, '/api/model/posterior', function () {
    return { ok: false, error: 'EvidenceHasZeroProbability: no mass left' };
  });
  const factory = loadPanelWithFetch(fetchFn);
  const root = makeRoot();
  const instance = factory(root, makeBus());
  try {
    root.buttons.find(function (b) { return b.dataset.view === 'posterior'; }).click();
    await flush();

    root.byId['#posterior-calculate'].click();
    await flush();

    assert.match(
      root.byId['#posterior-evidence-status'].textContent, /EvidenceHasZeroProbability/
    );
  } finally {
    instance.destroy();
  }
});

test('Reset clears the evidence rows and restores the prior', async function () {
  const fetchFn = makeFetchWithPost(RESPONSES, '/api/model/posterior', function () {
    return { variables: { object_friction: pickupVariable('object_friction', { expectation: 0.75 }) } };
  });
  const factory = loadPanelWithFetch(fetchFn);
  const root = makeRoot();
  const instance = factory(root, makeBus());
  try {
    root.buttons.find(function (b) { return b.dataset.view === 'posterior'; }).click();
    await flush();
    root.byId['#posterior-evidence-add'].click();
    root.byId['#posterior-calculate'].click();
    await flush();
    assert.match(root.byId['#posterior-evidence-rows'].innerHTML, /posterior-evidence-row/);

    root.byId['#posterior-reset'].click();

    assert.strictEqual(root.byId['#posterior-evidence-rows'].innerHTML, '');
    assert.strictEqual(root.byId['#posterior-evidence-status'].textContent, '');
  } finally {
    instance.destroy();
  }
});
