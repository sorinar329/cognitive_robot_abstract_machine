/* ============================================================================
 * panels/graph/panel.js — the graph view with five tabs.
 *
 *   Knowledge   the entity graph + CRAM architecture (double-click drills in)
 *   Kinematics  the robot's URDF tree (links as nodes, joints as edges)
 *   Plan        the executed plan tree, node border = execution status
 *   Statechart  the giskardpy motion statechart of the running motion group
 *   Posterior   PDF/CDF/mode/expectation of every causal-diagnosis model variable
 *
 * Plan and Statechart additionally take LIVE node status from the cramera-live
 * bridge while it is attached: structure changes rebuild the graph, pure
 * status changes only re-colour the rings (no layout jumps).
 *
 * Posterior is unlike the other four: its data starts static (exported once by
 * export_posterior_plots.py, not live or per-scene) and it draws an SVG line chart
 * instead of a node/edge graph, so it renders into its own view rather than through
 * graph.js/vis-network. Its Evidence panel lets you condition a model on interval
 * constraints and recompute -- POST /api/model/posterior, answered live by
 * cramera.live.model_query against the same circuit files.
 *
 * Bus events:
 *   emits    entity:select {id, detail, relations}   node clicked
 *   listens  entity:highlight {ids, focus?}          spotlight matching nodes
 *   listens  scene:step {step}                       highlight the running episode
 *   listens  live:changed {on, url}                  start/stop the status poll
 *
 * Rendering is delegated to graph.js (window.Graph, vis-network wrapper), except
 * for Posterior, which builds its own SVG directly in this file.
 * ==========================================================================*/
Panels.define('graph', function (root, bus) {
  root.innerHTML =
    '<div class="graph-wrap">' +
    '  <div class="graph-tabs" id="graph-tabs">' +
    '    <button data-view="knowledge" class="active" title="the entity graph (EQL / knowledge base)">Knowledge</button>' +
    '    <button data-view="kinematics" title="the robot\'s kinematic structure — URDF links &amp; joints">Kinematics</button>' +
    '    <button data-view="plan" title="the plan tree, with the execution status of every node">Plan</button>' +
    '    <button data-view="chart" title="the giskardpy motion statechart of the running motion group">Statechart</button>' +
    '    <button data-view="posterior" title="PDF / CDF / mode / expectation of every causal-diagnosis model variable">Posterior</button>' +
    '    <span class="gt-live" id="gt-live" title="node status is streaming from the running demo">◉ live status</span>' +
    '  </div>' +
    '  <div id="graph"></div>' +
    '  <div id="graph-empty" class="graph-empty" style="display:none"></div>' +
    '  <div id="graph-nav" class="graph-nav" style="display:none">' +
    '    <button id="gnav-home" title="back to the overview">⌂</button>' +
    '    <button id="gnav-up" title="one level up">↑ back</button>' +
    '    <span id="gnav-path"></span>' +
    '  </div>' +
    '  <div class="legend" id="legend"></div>' +
    '  <div id="posterior-view" class="posterior-view" style="display:none">' +
    '    <div class="posterior-nav">' +
    '      <button id="posterior-prev" title="previous variable">◀</button>' +
    '      <span id="posterior-title" class="posterior-title"></span>' +
    '      <button id="posterior-next" title="next variable">▶</button>' +
    '    </div>' +
    '    <div class="posterior-evidence">' +
    '      <div class="posterior-evidence-header">' +
    '        Evidence' +
    '        <button id="posterior-evidence-add" title="condition on another variable">+ variable</button>' +
    '      </div>' +
    '      <div id="posterior-evidence-rows" class="posterior-evidence-rows"></div>' +
    '      <div class="posterior-evidence-actions">' +
    '        <button id="posterior-calculate">Calculate Posterior</button>' +
    '        <button id="posterior-reset" title="clear evidence, back to the prior">Reset</button>' +
    '        <span id="posterior-evidence-status" class="posterior-evidence-status"></span>' +
    '      </div>' +
    '    </div>' +
    '    <div id="posterior-chart" class="posterior-chart"></div>' +
    '    <div id="posterior-legend" class="posterior-legend"></div>' +
    '  </div>' +
    '</div>';

  const emptyEl = root.querySelector('#graph-empty');
  const navEl = root.querySelector('#graph-nav');
  const navUp = root.querySelector('#gnav-up');
  const navHome = root.querySelector('#gnav-home');
  const navPath = root.querySelector('#gnav-path');
  const tabsEl = root.querySelector('#graph-tabs');
  const liveBadge = root.querySelector('#gt-live');
  const graphEl = root.querySelector('#graph');
  const legendEl = root.querySelector('#legend');
  const posteriorViewEl = root.querySelector('#posterior-view');
  Graph.attach(graphEl, legendEl);

  // %% tabs
  const TABS = {
    knowledge:  { url: '/api/knowledge' },
    kinematics: { url: '/api/knowledge/view?name=kinematics' },
    plan:       { url: '/api/knowledge/view?name=plan' },
    chart:      { url: '/api/knowledge/view?name=chart' },
  };
  let tab = 'knowledge';
  let view = null;            // the currently rendered payload
  const base = {};            // tab -> payload as loaded from the server
  const shown = {};           // tab -> payload currently rendered (drill-downs)
  const stacks = {};          // tab -> parent payloads for the back button
  Object.keys(TABS).forEach(function (t) { stacks[t] = []; });
  let inGraphSet = {};

  function setView(payload) {
    view = payload;
    shown[tab] = payload;
    inGraphSet = {};
    payload.nodes.forEach(function (n) { inGraphSet[n.id] = 1; });
    if (emptyEl) {
      const empty = !payload.nodes.length;
      emptyEl.style.display = empty ? '' : 'none';
      emptyEl.textContent = empty ? (payload.empty || 'Nothing to show in this view.') : '';
    }
    Graph.build({
      nodes: payload.nodes, edges: payload.edges, legend: payload.legend,
      layout: payload.layout, arrows: !!payload.arrows, statusLegend: !!payload.statusLegend,
      key: (payload.key || tab) + '#' + stacks[tab].length,
    });
    updateNav();
  }
  function updateNav() {
    const inside = stacks[tab].length > 0;
    navEl.style.display = inside ? '' : 'none';
    if (inside) {
      const path = stacks[tab].slice(1).map(function (v) { return v.breadcrumb; }).concat([view.breadcrumb]);
      navPath.textContent = path.join(' / ');
    }
  }
  async function drill(id) {
    if (!view.details[id]) return;
    try {
      const r = await fetch('/api/knowledge/expand?node=' + encodeURIComponent(id));
      const p = await r.json();
      if (!p.ok) return;                       // node has no inside view
      stacks[tab].push(view);
      setView(p);
      select(id);
    } catch (err) { /* server unreachable — stay where we are */ }
  }
  function goBack() { if (stacks[tab].length) setView(stacks[tab].pop()); }
  function goHome() {
    if (!stacks[tab].length) return;
    stacks[tab] = [];
    setView(base[tab]);
  }
  navUp.addEventListener('click', goBack);
  navHome.addEventListener('click', goHome);

  async function showTab(name) {
    if (name !== 'posterior' && !TABS[name]) return;
    tab = name;
    tabsEl.querySelectorAll('button').forEach(function (b) {
      b.classList.toggle('active', b.dataset.view === name);
    });

    if (name === 'posterior') {
      graphEl.style.display = 'none';
      legendEl.style.display = 'none';
      emptyEl.style.display = 'none';
      navEl.style.display = 'none';
      posteriorViewEl.style.display = '';
      await showPosterior();
      return;
    }
    graphEl.style.display = '';
    legendEl.style.display = '';
    posteriorViewEl.style.display = 'none';

    if (!base[name]) {
      emptyEl.style.display = '';
      emptyEl.textContent = 'loading…';
      try {
        const r = await fetch(TABS[name].url);
        if (r.status === 404) throw new Error('this build needs the /api/knowledge/view route — restart the server');
        const p = await r.json();
        if (!p.ok) {
          emptyEl.textContent = p.error || 'view unavailable';
          return;
        }
        p.key = name;
        base[name] = p;
      } catch (err) {
        emptyEl.textContent = 'Could not load this view: ' + ((err && err.message) || err);
        return;
      }
    }
    setView(shown[name] || base[name]);
    liveRefresh(true);            // a live tab picks the bridge status up at once
  }
  tabsEl.querySelectorAll('button').forEach(function (b) {
    b.addEventListener('click', function () { showTab(b.dataset.view); });
  });

  // %% node click → describe in whatever panel listens
  function select(id) {
    const d = view && view.details && view.details[id];
    if (!d) return;
    const relations = (view.edges || [])
      .filter(function (e) { return e.from === id || e.to === id; })
      .map(function (e) {
        return { s: labelOf(e.from), p: e.label || e.kind, o: labelOf(e.to) };
      });
    bus.emit('entity:select', { id: id, detail: d, relations: relations });
    spotlight({ ids: [id], focus: id });
  }
  function labelOf(id) { return (view.details[id] && view.details[id].label) || id; }
  Graph.onSelect(select);
  Graph.onDoubleSelect(drill);

  // %% highlights (from EQL results or our own selection)
  function spotlight(p) {
    const ids = (p && p.ids) || [];
    let hi = ids.filter(function (id) { return inGraphSet[id]; });
    if (p && p.focus && inGraphSet[p.focus]) {
      const neighbours = (view.edges || [])
        .filter(function (e) { return e.from === p.focus || e.to === p.focus; })
        .map(function (e) { return e.from === p.focus ? e.to : e.from; });
      hi = hi.concat(neighbours.filter(function (id) { return inGraphSet[id]; }));
    }
    if (hi.length) Graph.highlight(hi); else Graph.reset();
  }
  bus.on('entity:highlight', spotlight);
  bus.on('scene:step', function (p) {
    if (p.step === '__done__') { Graph.reset(); return; }
    if (tab === 'knowledge' && !stacks[tab].length && inGraphSet[p.step]) select(p.step);
  });

  // %% live status overlay (Plan / Statechart tabs)
  // The bridge publishes the plan tree and the executing motion statechart with
  // per-node status. Structure changes (the plan grows as actions expand, a new
  // statechart is compiled per motion group) rebuild the graph; a pure status
  // change only re-colours the rings, so the layout never jumps.
  const CHART_LEGEND = [
    { group: 'task', label: 'Task (motion constraint)' },
    { group: 'monitor', label: 'Monitor / observation' },
    { group: 'motion_goal', label: 'Goal (contains nodes)' },
    { group: 'motion_end', label: 'End / cancel motion' },
  ];
  const liveSig = { plan: '', chart: '' };
  let liveTimer = null;
  let liveState = { on: false, url: '' };

  function liveSource() {
    const p = shown[tab] || base[tab];
    return (p && p.live) || null;               // 'plan' | 'chart' | null
  }

  // drop the redundant 'Action' suffix only — a label that merely contains the word,
  // such as 'ActionNode', must survive intact. Mirrors
  // PlanViewPayload._shorten_action_label: the bridge sends the raw designator name,
  // so the live path shortens it here.
  function shortenActionLabel(label) {
    const shortened = label.replace(/Action$/, '');
    return shortened || label;
  }

  function planPayload(live) {
    const nodes = [], edges = [], details = {};
    (live.nodes || []).forEach(function (n) {
      const label = shortenActionLabel(n.label || '?');
      const lines = ['a ' + n.kind,
                     'status: ' + n.status + (n.derived ? ' (derived from the motion statechart)' : '')];
      if (n.arm) lines.push('arm: ' + n.arm);
      if (n.target) lines.push('target: ' + n.target);
      nodes.push({ id: n.id, label: label, group: n.group,
                   title: [label].concat(lines).join('\n'), status: n.status });
      details[n.id] = { label: label, group: n.group, lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'property', label: 'has step' });
    });
    return { ok: true, breadcrumb: 'live plan', nodes: nodes, edges: edges, details: details,
             legend: live.legend || [], layout: 'hier', arrows: true, statusLegend: true,
             live: 'plan', key: 'plan-live',
             empty: 'The bridge is attached but the demo has not started its plan yet.' };
  }

  function chartPayload(live) {
    const nodes = [], edges = [], details = {}, isParent = {};
    (live.nodes || []).forEach(function (n) { if (n.parent) isParent[n.parent] = 1; });
    (live.nodes || []).forEach(function (n) {
      const group = isParent[n.id] ? 'motion_goal'
        : /EndMotion|CancelMotion/.test(n.class_name) ? 'motion_end'
        : /Monitor|Reached|Observation|Condition/.test(n.class_name + n.name) ? 'monitor' : 'task';
      const lines = ['a ' + n.class_name, 'life cycle: ' + n.life_cycle, 'observation: ' + n.observation];
      nodes.push({ id: n.id, label: n.name, group: group,
                   title: [n.name].concat(lines).join('\n'), status: n.life_cycle });
      details[n.id] = { label: n.name, group: group, lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'type', label: 'contains' });
    });
    (live.edges || []).forEach(function (e) {
      edges.push({ from: e.from, to: e.to, kind: e.kind, label: (e.kind || '').toLowerCase() + ' transition' });
    });
    return { ok: true, breadcrumb: 'statechart' + (live.title ? ' · ' + live.title : ''),
             nodes: nodes, edges: edges, details: details, legend: CHART_LEGEND,
             layout: 'hier', arrows: true, statusLegend: true, live: 'chart', key: 'chart-live',
             empty: 'Attached, but no motion statechart is executing right now.' };
  }

  async function liveRefresh(force) {
    const src = liveSource();
    const active = !!src && liveState.on;
    liveBadge.classList.toggle('on', active);
    if (!active) return;
    if (stacks[tab].length) return;              // inside a drill-down: leave it alone
    let live;
    try {
      live = await fetch(liveState.url + (src === 'plan' ? '/plan' : '/chart'))
        .then(function (r) { return r.json(); });
    } catch (err) { return; }                    // bridge gone — the 3D side handles it
    if (!live || !live.nodes) return;
    const payload = src === 'plan' ? planPayload(live) : chartPayload(live);
    if (force || live.signature !== liveSig[src]) {    // structure changed → rebuild
      liveSig[src] = live.signature;
      base[tab] = payload;
      setView(payload);
      return;
    }
    // same structure: only re-colour, and keep the detail lines in sync
    const map = {};
    payload.nodes.forEach(function (n) { map[n.id] = n.status; });
    if (!Graph.setStatuses(map)) { base[tab] = payload; setView(payload); return; }
    base[tab] = payload;
    if (view && view.details) view.details = payload.details;
  }

  bus.on('live:changed', function (p) {
    liveState = { on: !!p.on, url: p.url || '' };
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    if (liveState.on) {
      liveTimer = setInterval(function () { liveRefresh(false); }, 700);
      liveRefresh(true);
    } else {
      liveBadge.classList.remove('on');
      liveSig.plan = liveSig.chart = '';
      // drop the live payloads so both tabs fall back to the recorded bundle
      ['plan', 'chart'].forEach(function (t) { delete base[t]; delete shown[t]; stacks[t] = []; });
      if (tab === 'plan' || tab === 'chart') showTab(tab);
    }
  });

  // %% posterior — PDF/CDF/mode/expectation of every causal-diagnosis model variable
  //
  // The prior is static data (see export_posterior_plots.py), not live and not tied to
  // the current scene, so it is fetched once and cached for the panel's lifetime. Both
  // models' variables are shown as one flat, navigable list. The Evidence panel below
  // the nav bar conditions the *current* variable's model on interval constraints and
  // POSTs to /api/model/posterior for a live-recomputed distribution, replacing that
  // model's entries in place so navigation position is preserved.
  const POSTERIOR_MODELS = ['pickup', 'place'];
  const POSTERIOR_LEGEND = [
    { label: 'Probability Density Function', cls: 'posterior-pdf' },
    { label: 'Cumulative Distribution Function', cls: 'posterior-cdf' },
    { label: 'Expectation', cls: 'posterior-expectation' },
    { label: 'Mode', cls: 'posterior-mode' },
  ];
  const posteriorTitleEl = root.querySelector('#posterior-title');
  const posteriorChartEl = root.querySelector('#posterior-chart');
  const posteriorLegendEl = root.querySelector('#posterior-legend');
  const posteriorPrevBtn = root.querySelector('#posterior-prev');
  const posteriorNextBtn = root.querySelector('#posterior-next');
  const posteriorEvidenceRowsEl = root.querySelector('#posterior-evidence-rows');
  const posteriorEvidenceAddBtn = root.querySelector('#posterior-evidence-add');
  const posteriorCalculateBtn = root.querySelector('#posterior-calculate');
  const posteriorResetBtn = root.querySelector('#posterior-reset');
  const posteriorEvidenceStatusEl = root.querySelector('#posterior-evidence-status');
  posteriorLegendEl.innerHTML = POSTERIOR_LEGEND.map(function (item) {
    return '<span class="posterior-legend-item"><span class="posterior-legend-swatch ' +
      item.cls + '"></span>' + item.label + '</span>';
  }).join('');

  let posteriorEntries = null;    // [{model, variable, data}], data replaced on Calculate
  let posteriorPriorByKey = {};   // 'model:variable' -> the original, unconditioned data
  let posteriorIndex = 0;
  const posteriorEvidenceByModel = {};   // model -> [{variable, minimum, maximum}]
  POSTERIOR_MODELS.forEach(function (model) { posteriorEvidenceByModel[model] = []; });

  async function loadPosteriorEntries() {
    if (posteriorEntries) return posteriorEntries;
    const entries = [];
    for (const model of POSTERIOR_MODELS) {
      try {
        const r = await fetch('/data/posterior/' + model + '.json');
        if (!r.ok) continue;
        const payload = await r.json();
        (payload.order || []).forEach(function (name) {
          const data = payload.variables[name];
          entries.push({ model: model, variable: name, data: data });
          posteriorPriorByKey[model + ':' + name] = data;
        });
      } catch (err) { /* this model's export is missing — skip it */ }
    }
    posteriorEntries = entries;
    return entries;
  }

  async function showPosterior() {
    const entries = await loadPosteriorEntries();
    if (!entries.length) {
      posteriorTitleEl.textContent = '';
      posteriorChartEl.innerHTML = '<div class="answer-empty">No posterior data found — run ' +
        'coraplex_panda_demo/training/export_posterior_plots.py first.</div>';
      posteriorPrevBtn.disabled = true;
      posteriorNextBtn.disabled = true;
      posteriorEvidenceRowsEl.innerHTML = '';
      return;
    }
    if (posteriorIndex >= entries.length) posteriorIndex = 0;
    renderPosteriorEntry(entries[posteriorIndex]);
  }

  function currentPosteriorModel() {
    return (posteriorEntries && posteriorEntries[posteriorIndex])
      ? posteriorEntries[posteriorIndex].model : null;
  }

  function renderPosteriorEntry(entry) {
    posteriorTitleEl.textContent =
      entry.model.charAt(0).toUpperCase() + entry.model.slice(1) + ' · ' + entry.variable;
    posteriorChartEl.innerHTML = posteriorChartSvg(entry.data);
    posteriorPrevBtn.disabled = posteriorIndex === 0;
    posteriorNextBtn.disabled = posteriorIndex === posteriorEntries.length - 1;
    renderEvidenceRows();
  }

  // %% evidence rows — one per constraint, scoped to the currently viewed model
  function variableBounds(model, variableName) {
    const entry = (posteriorEntries || []).find(
      function (e) { return e.model === model && e.variable === variableName; }
    );
    const samples = entry && entry.data && entry.data.samples;
    return samples && samples.length ? [samples[0], samples[samples.length - 1]] : [0, 1];
  }

  function renderEvidenceRows() {
    const model = currentPosteriorModel();
    if (!model) { posteriorEvidenceRowsEl.innerHTML = ''; return; }
    const modelVariables = posteriorEntries
      .filter(function (e) { return e.model === model; })
      .map(function (e) { return e.variable; });
    const rows = posteriorEvidenceByModel[model];

    posteriorEvidenceRowsEl.innerHTML = rows.map(function (row, index) {
      const options = modelVariables.map(function (name) {
        return '<option value="' + name + '"' + (name === row.variable ? ' selected' : '') +
          '>' + name + '</option>';
      }).join('');
      return '<div class="posterior-evidence-row" data-index="' + index + '">' +
        '<select class="posterior-evidence-variable">' + options + '</select>' +
        '<input type="number" class="posterior-evidence-min" step="any" value="' + row.minimum + '">' +
        '<input type="number" class="posterior-evidence-max" step="any" value="' + row.maximum + '">' +
        '<button class="posterior-evidence-remove" title="remove this constraint">×</button>' +
        '</div>';
    }).join('');

    posteriorEvidenceRowsEl.querySelectorAll('.posterior-evidence-row').forEach(function (rowEl) {
      const index = Number(rowEl.dataset.index);
      rowEl.querySelector('.posterior-evidence-variable').addEventListener('change', function (ev) {
        const bounds = variableBounds(model, ev.target.value);
        rows[index] = { variable: ev.target.value, minimum: bounds[0], maximum: bounds[1] };
        renderEvidenceRows();
      });
      rowEl.querySelector('.posterior-evidence-min').addEventListener('change', function (ev) {
        rows[index].minimum = parseFloat(ev.target.value);
      });
      rowEl.querySelector('.posterior-evidence-max').addEventListener('change', function (ev) {
        rows[index].maximum = parseFloat(ev.target.value);
      });
      rowEl.querySelector('.posterior-evidence-remove').addEventListener('click', function () {
        rows.splice(index, 1);
        renderEvidenceRows();
      });
    });
  }

  posteriorEvidenceAddBtn.addEventListener('click', function () {
    const model = currentPosteriorModel();
    if (!model) return;
    const modelVariables = posteriorEntries
      .filter(function (e) { return e.model === model; })
      .map(function (e) { return e.variable; });
    if (!modelVariables.length) return;
    const variable = modelVariables[0];
    const bounds = variableBounds(model, variable);
    posteriorEvidenceByModel[model].push(
      { variable: variable, minimum: bounds[0], maximum: bounds[1] }
    );
    renderEvidenceRows();
  });

  posteriorCalculateBtn.addEventListener('click', async function () {
    const model = currentPosteriorModel();
    if (!model) return;
    const queryVariables = posteriorEntries
      .filter(function (e) { return e.model === model; })
      .map(function (e) { return e.variable; });
    posteriorEvidenceStatusEl.textContent = 'Calculating…';
    try {
      const r = await fetch('/api/model/posterior', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: model,
          queryVariables: queryVariables,
          evidence: posteriorEvidenceByModel[model],
        }),
      });
      const payload = await r.json();
      if (payload.ok === false) {
        posteriorEvidenceStatusEl.textContent = payload.error || 'calculation failed';
        return;
      }
      posteriorEntries.forEach(function (e) {
        if (e.model === model && payload.variables[e.variable]) {
          e.data = payload.variables[e.variable];
        }
      });
      const count = posteriorEvidenceByModel[model].length;
      posteriorEvidenceStatusEl.textContent =
        count ? 'Conditioned on ' + count + ' constraint(s).' : 'Showing the prior.';
      renderPosteriorEntry(posteriorEntries[posteriorIndex]);
    } catch (err) {
      posteriorEvidenceStatusEl.textContent = 'Could not reach the server.';
    }
  });

  posteriorResetBtn.addEventListener('click', function () {
    const model = currentPosteriorModel();
    if (!model) return;
    posteriorEvidenceByModel[model] = [];
    posteriorEntries.forEach(function (e) {
      if (e.model === model) e.data = posteriorPriorByKey[model + ':' + e.variable];
    });
    posteriorEvidenceStatusEl.textContent = '';
    renderPosteriorEntry(posteriorEntries[posteriorIndex]);
  });

  function posteriorChartSvg(data) {
    if (!data || !data.samples || !data.samples.length) {
      return '<div class="answer-empty">No data for this variable.</div>';
    }
    const width = 720, height = 320;
    const marginLeft = 54, marginRight = 14, marginTop = 14, marginBottom = 32;
    const plotWidth = width - marginLeft - marginRight;
    const plotHeight = height - marginTop - marginBottom;

    const xs = data.samples;
    const xMin = xs[0], xMax = xs[xs.length - 1];
    const xSpan = (xMax - xMin) || 1;
    const yMax = Math.max(
      Math.max.apply(null, data.pdf),
      Math.max.apply(null, data.cdf),
      data.modeHeight || 0,
      1e-9
    ) * 1.08;

    function px(x) { return marginLeft + ((x - xMin) / xSpan) * plotWidth; }
    function py(y) { return marginTop + plotHeight - (y / yMax) * plotHeight; }

    function linePath(ys) {
      return xs.map(function (x, i) {
        return (i === 0 ? 'M' : 'L') + px(x).toFixed(2) + ' ' + py(ys[i]).toFixed(2);
      }).join(' ');
    }

    const parts = [];
    parts.push(
      '<svg viewBox="0 0 ' + width + ' ' + height +
      '" class="posterior-svg" preserveAspectRatio="xMidYMid meet">'
    );

    // axes
    parts.push(
      '<line x1="' + marginLeft + '" y1="' + (marginTop + plotHeight) + '" x2="' +
      (marginLeft + plotWidth) + '" y2="' + (marginTop + plotHeight) + '" class="posterior-axis" />'
    );
    parts.push(
      '<line x1="' + marginLeft + '" y1="' + marginTop + '" x2="' + marginLeft + '" y2="' +
      (marginTop + plotHeight) + '" class="posterior-axis" />'
    );
    parts.push(
      '<text x="' + marginLeft + '" y="' + (height - 8) + '" class="posterior-tick">' +
      xMin.toFixed(3) + '</text>'
    );
    parts.push(
      '<text x="' + (marginLeft + plotWidth) + '" y="' + (height - 8) +
      '" text-anchor="end" class="posterior-tick">' + xMax.toFixed(3) + '</text>'
    );
    parts.push(
      '<text x="' + (marginLeft - 6) + '" y="' + (marginTop + 4) +
      '" text-anchor="end" class="posterior-tick">' + yMax.toFixed(2) + '</text>'
    );
    parts.push(
      '<text x="' + (marginLeft - 6) + '" y="' + (marginTop + plotHeight) +
      '" text-anchor="end" class="posterior-tick">0</text>'
    );

    // mode (drawn first, so PDF/CDF sit on top)
    (data.modes || []).forEach(function (range) {
      const x1 = px(range[0]).toFixed(2), x2 = px(range[1]).toFixed(2);
      const yTop = py(data.modeHeight || 0).toFixed(2), yBase = py(0).toFixed(2);
      parts.push(
        '<polyline points="' + x1 + ',' + yBase + ' ' + x1 + ',' + yTop + ' ' +
        x2 + ',' + yTop + ' ' + x2 + ',' + yBase + '" class="posterior-mode" />'
      );
    });

    // expectation
    if (data.expectation !== null && data.expectation !== undefined) {
      const x = px(data.expectation).toFixed(2);
      parts.push(
        '<line x1="' + x + '" y1="' + py(0).toFixed(2) + '" x2="' + x + '" y2="' +
        py(data.modeHeight || yMax).toFixed(2) + '" class="posterior-expectation" />'
      );
    }

    // PDF / CDF
    parts.push('<path d="' + linePath(data.pdf) + '" class="posterior-pdf" />');
    parts.push('<path d="' + linePath(data.cdf) + '" class="posterior-cdf" />');

    parts.push('</svg>');
    return parts.join('');
  }

  posteriorPrevBtn.addEventListener('click', function () {
    if (posteriorIndex > 0) {
      posteriorIndex -= 1;
      renderPosteriorEntry(posteriorEntries[posteriorIndex]);
    }
  });
  posteriorNextBtn.addEventListener('click', function () {
    if (posteriorEntries && posteriorIndex < posteriorEntries.length - 1) {
      posteriorIndex += 1;
      renderPosteriorEntry(posteriorEntries[posteriorIndex]);
    }
  });

  // %% boot
  showTab('knowledge');

  return {
    destroy: function () {
      if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    },
  };
});
