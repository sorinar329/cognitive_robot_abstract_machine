/* ============================================================================
 * panels/scene_graph/panel.js — the bodies of a controlled simulation, and the
 * controls that pause, resume, stop and change it.
 *
 * Opt-in: only a demo that attached a ControlledSimulation to its visualization is
 * controllable (see cramera.live.simulation_control). Until the live bridge serves
 * one, the panel shows a one-line note and polls nothing.
 *
 * Changes are made between two physics steps: right away while paused, else before
 * the next step. The inputs are filled when a body is selected, and refilled with
 * "Reload", so a value being typed is never overwritten by the poll.
 *
 * Bus events:
 *   emits    entity:highlight {ids, focus}   the selected body, by its short name
 *   listens  live:changed {on, url}          start/stop polling the bridge
 *
 * The rules (tree order, angles, request bodies) live in core/scene-graph.js.
 * ==========================================================================*/
Panels.define('scene-graph', function (root, bus) {
  root.innerHTML =
    '<div class="panel-head">' +
    '  <h2>Scene graph · simulation</h2>' +
    '  <div class="sg-controls">' +
    '    <span class="sg-state" id="sg-state">not controllable</span>' +
    '    <button id="sg-pause" title="hold the robot and the physics at the next step">⏸ Pause</button>' +
    '    <button id="sg-resume" title="carry on from where the simulation was paused">▶ Play</button>' +
    '    <button id="sg-stop" title="end the plan: its next step raises SimulationStoppedError">⏹ Stop</button>' +
    '  </div>' +
    '</div>' +
    '<div class="sg-note" id="sg-note">Attach the live view of a demo that offers its simulation for control.</div>' +
    '<div class="sg-body hidden" id="sg-body">' +
    '  <div class="sg-tree-wrap">' +
    '    <input id="sg-filter" class="sg-filter" placeholder="filter bodies" spellcheck="false" />' +
    '    <div class="sg-tree" id="sg-tree"></div>' +
    '  </div>' +
    '  <div class="sg-detail" id="sg-detail"><div class="sg-empty">Select a body.</div></div>' +
    '</div>' +
    '<div class="sg-message" id="sg-message"></div>';

  const POLL_MILLISECONDS = 500;
  const stateEl = root.querySelector('#sg-state');
  const noteEl = root.querySelector('#sg-note');
  const bodyEl = root.querySelector('#sg-body');
  const treeEl = root.querySelector('#sg-tree');
  const detailEl = root.querySelector('#sg-detail');
  const filterEl = root.querySelector('#sg-filter');
  const messageEl = root.querySelector('#sg-message');
  const buttons = {
    pause: root.querySelector('#sg-pause'),
    resume: root.querySelector('#sg-resume'),
    stop: root.querySelector('#sg-stop'),
  };

  let bridgeUrl = '';
  let timer = null;
  let graph = null;          // the last served scene graph
  let selected = null;       // name of the selected body
  let filledFor = null;      // name of the body the inputs were filled for
  let destroyed = false;

  // %% talking to the bridge
  function post(route, payload) {
    return fetch(bridgeUrl + route, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload || {}),
    }).then(function (response) { return response.json(); }).then(function (answer) {
      showMessage(answer.ok ? '' : answer.error, !answer.ok);
      refresh();
      return answer;
    }).catch(function (error) { showMessage(String(error), true); });
  }

  function refresh() {
    if (!bridgeUrl) return;
    fetch(bridgeUrl + SceneGraph.ROUTE.SCENE_GRAPH).then(function (response) {
      return response.json();
    }).then(render).catch(function () { render(null); });
  }

  function showMessage(text, isError) {
    messageEl.textContent = text || '';
    messageEl.classList.toggle('error', !!isError);
  }

  // %% rendering
  function render(payload) {
    if (destroyed) return;
    const available = !!(payload && payload.available);
    root.classList.toggle('sg-available', available);
    noteEl.classList.toggle('hidden', available);
    bodyEl.classList.toggle('hidden', !available);
    graph = available ? payload : null;
    const state = available ? payload.state : null;
    stateEl.textContent = state || 'not controllable';
    stateEl.dataset.state = state || '';
    const allowed = SceneGraph.controlsFor(state);
    Object.keys(buttons).forEach(function (control) { buttons[control].disabled = !allowed[control]; });
    if (!available) return;
    renderTree();
    renderDetail();
  }

  function renderTree() {
    const filter = filterEl.value.trim().toLowerCase();
    treeEl.textContent = '';
    SceneGraph.tree(graph.bodies).forEach(function (row) {
      const name = SceneGraph.shortName(row.body.name);
      if (filter && name.toLowerCase().indexOf(filter) === -1) return;
      const item = document.createElement('div');
      item.className = 'sg-row sg-' + row.body.placement + (row.body.name === selected ? ' selected' : '');
      item.style.paddingLeft = (6 + row.depth * 12) + 'px';
      item.textContent = name;
      item.title = row.body.name + ' — ' + row.body.placement;
      item.addEventListener('click', function () { select(row.body.name); });
      treeEl.appendChild(item);
    });
  }

  function selectedBody() {
    if (!graph || !selected) return null;
    return graph.bodies.filter(function (body) { return body.name === selected; })[0] || null;
  }

  function select(name) {
    selected = name;
    filledFor = null;
    const id = SceneGraph.shortName(name);
    bus.emit('entity:highlight', { ids: [id], focus: id });
    renderTree();
    renderDetail();
  }

  function format(value, digits) {
    return (value === null || value === undefined) ? '—' : Number(value).toFixed(digits);
  }

  /* One labelled number field; `current` is shown beside it and refreshed by the poll. */
  function numberField(form, key, label, value, digits) {
    const row = document.createElement('label');
    row.className = 'sg-field';
    const name = document.createElement('span');
    name.textContent = label;
    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.dataset.key = key;
    input.value = format(value, digits);
    const current = document.createElement('span');
    current.className = 'sg-current';
    current.dataset.key = key;
    row.appendChild(name);
    row.appendChild(input);
    row.appendChild(current);
    form.appendChild(row);
  }

  function section(title, apply) {
    const form = document.createElement('form');
    form.className = 'sg-section';
    const heading = document.createElement('div');
    heading.className = 'sg-section-head';
    heading.textContent = title;
    const button = document.createElement('button');
    button.type = 'submit';
    button.textContent = 'Apply';
    heading.appendChild(button);
    form.appendChild(heading);
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      const values = {};
      form.querySelectorAll('input[data-key]').forEach(function (input) {
        values[input.dataset.key] = Number(input.value);
      });
      apply(values);
    });
    detailEl.appendChild(form);
    return form;
  }

  function currentValues(body) {
    const values = { mass: body.mass };
    if (body.pose) Object.assign(values, SceneGraph.editablePose(body.pose));
    if (body.friction) Object.assign(values, body.friction);
    return values;
  }

  function renderDetail() {
    const body = selectedBody();
    if (!body) {
      detailEl.innerHTML = '<div class="sg-empty">Select a body.</div>';
      filledFor = null;
      return;
    }
    if (filledFor !== body.name) fillDetail(body);
    const values = currentValues(body);
    detailEl.querySelectorAll('.sg-current').forEach(function (current) {
      current.textContent = 'now ' + format(values[current.dataset.key], 3);
    });
  }

  function fillDetail(body) {
    filledFor = body.name;
    detailEl.textContent = '';
    const info = document.createElement('div');
    info.className = 'sg-info';
    info.textContent = body.name + ' · ' + body.placement +
      (body.parent ? ' · ' + (body.connection || '') + ' to ' + SceneGraph.shortName(body.parent) : '');
    detailEl.appendChild(info);
    const reload = document.createElement('button');
    reload.className = 'sg-reload';
    reload.textContent = 'Reload';
    reload.title = 'refill the inputs with the current values';
    reload.addEventListener('click', function () { filledFor = null; renderDetail(); });
    info.appendChild(reload);

    const editable = SceneGraph.editable(body);
    const values = currentValues(body);
    if (editable.pose && body.pose) {
      const form = section('Pose in the world (m, °)', function (pose) {
        post(SceneGraph.ROUTE.EDIT, SceneGraph.poseRequest(body.name, pose));
      });
      [['x', 'x'], ['y', 'y'], ['z', 'z'], ['roll', 'roll'], ['pitch', 'pitch'], ['yaw', 'yaw']].forEach(function (pair) {
        numberField(form, pair[0], pair[1], values[pair[0]], pair[0].length === 1 ? 4 : 2);
      });
    }
    if (editable.mass) {
      const form = section('Mass (kg)', function (mass) {
        post(SceneGraph.ROUTE.EDIT, SceneGraph.massRequest(body.name, mass.mass));
      });
      numberField(form, 'mass', 'mass', values.mass, 4);
    }
    if (editable.friction) {
      const form = section('Friction', function (friction) {
        post(SceneGraph.ROUTE.EDIT, SceneGraph.frictionRequest(body.name, friction));
      });
      SceneGraph.FRICTION_COEFFICIENTS.forEach(function (coefficient) {
        numberField(form, coefficient, coefficient, values[coefficient], 4);
      });
    }
    if (!editable.pose && !editable.mass && !editable.friction) {
      const none = document.createElement('div');
      none.className = 'sg-empty';
      none.textContent = 'Nothing about this body can be changed while the simulation runs.';
      detailEl.appendChild(none);
    }
  }

  // %% controls
  buttons.pause.addEventListener('click', function () { post(SceneGraph.ROUTE.PAUSE); });
  buttons.resume.addEventListener('click', function () { post(SceneGraph.ROUTE.RESUME); });
  buttons.stop.addEventListener('click', function () { post(SceneGraph.ROUTE.STOP); });
  filterEl.addEventListener('input', function () { if (graph) renderTree(); });

  bus.on('live:changed', function (live) {
    if (destroyed) return;
    if (timer) { clearInterval(timer); timer = null; }
    bridgeUrl = live && live.on ? (live.url || '') : '';
    if (!bridgeUrl) return render(null);
    timer = setInterval(refresh, POLL_MILLISECONDS);
    refresh();
  });

  render(null);

  return {
    destroy: function () {
      destroyed = true;
      if (timer) clearInterval(timer);
    },
  };
});
