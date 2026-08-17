/* ============================================================================
 * panels/activity/panel.js — live per-iteration trial outcomes.
 *
 * Polls the live bridge's /activity endpoint (a demo-defined JSON log — see
 * cramera.live.bridge.Bridge.log_activity) while attached, and renders each
 * entry as a card: which iteration, whether the stack stood (geometry ground
 * truth and segmind's own reading), and per cube whether it stacked or ended
 * on the floor, plus what causal diagnosis corrected and why. Demo-agnostic
 * in principle (it renders whatever shape a demo publishes), but its layout
 * is tuned for the coraplex_panda_demo stacking trials.
 *
 * Bus events:
 *   listens  live:changed {on, url}   start/stop polling /activity
 * ==========================================================================*/
Panels.define('activity', function (root, bus) {
  root.innerHTML =
    '<div class="panel-head">' +
    '  <h2>Activity</h2>' +
    '  <span id="activity-status" class="knowledge-status">not live</span>' +
    '</div>' +
    '<div id="activity-feed" class="activity-feed">' +
    '  <div class="answer-empty">Attach to a running demo to see its trial outcomes.</div>' +
    '</div>';

  const statusEl = root.querySelector('#activity-status');
  const feedEl = root.querySelector('#activity-feed');

  let liveUrl = '';
  let timer = null;
  let lastCount = -1;

  function poll() {
    fetch(liveUrl + '/activity').then(function (r) { return r.json(); })
      .then(function (payload) { render(payload.entries || []); })
      .catch(function () {});     // bridge gone — the 3D side already shows that
  }

  function render(entries) {
    if (entries.length === lastCount) return;    // nothing new since the last poll
    lastCount = entries.length;
    if (!entries.length) {
      feedEl.innerHTML = '<div class="answer-empty">No iterations reported yet…</div>';
      return;
    }
    // newest first: this is a live feed, not a scrollback
    feedEl.innerHTML = entries.slice().reverse().map(renderEntry).join('');
  }

  function renderEntry(entry) {
    const overall = entry.simulationDiverged ? 'warn' : (entry.fullStackIntact ? 'ok' : 'bad');
    const title = 'Iteration ' + entry.iteration +
      (entry.totalIterations ? ' / ' + entry.totalIterations : '');
    const duration = entry.durationSeconds != null ? fmtSeconds(entry.durationSeconds) : '';

    let html = '<div class="activity-card ' + overall + '">';
    html += '<div class="activity-card-head">';
    html += '<span class="activity-title">' + esc(title) + '</span>';
    html += '<span class="activity-meta">' + esc(duration) + '</span>';
    html += '</div>';
    html += '<div class="activity-badges">';
    html += badge('geometry', entry.fullStackIntact);
    html += badge('segmind', entry.segmindApproved);
    if (entry.simulationDiverged) html += '<span class="activity-badge warn">diverged</span>';
    html += '</div>';
    (entry.cubes || []).forEach(function (cube) { html += renderCube(cube); });
    html += '</div>';
    return html;
  }

  function renderCube(cube) {
    const ok = !!cube.finalSucceeded;
    let html = '<div class="activity-cube ' + (ok ? 'ok' : 'bad') + '">';
    html += '<span class="activity-cube-label">' + esc(cube.stepLabel) + '</span>';
    html += '<span class="activity-cube-status">' + (ok ? 'stacked' : 'on floor') + '</span>';
    if (cube.correctionAttempts) {
      html += '<span class="activity-cube-corrections">' + cube.correctionAttempts +
        ' correction' + (cube.correctionAttempts === 1 ? '' : 's') + '</span>';
    }
    html += '</div>';
    (cube.diagnoses || []).forEach(function (diagnosis) {
      html += renderDiagnosis(diagnosis);
    });
    return html;
  }

  // "pickup"/"place" are the only action names the demo diagnoses against
  // (see inference3d._diagnosis_payload's docstring)
  const ACTION_PHRASE = { pickup: 'picking the cube up', place: 'placing the cube' };

  function actionPhrase(actionName) {
    return ACTION_PHRASE[actionName] || (actionName ? humanize(actionName) : 'this step');
  }

  // turns a causal-model variable name (e.g. "grasp_closing_velocity",
  // "cube1_final_z") into a plain-English phrase, without a hardcoded
  // per-variable dictionary that would go stale as the causal trees change
  function humanize(name) {
    const cubeHeight = /^cube(\d+)_final_z$/.exec(name || '');
    if (cubeHeight) return "cube " + cubeHeight[1] + "’s final height";
    return String(name || '').replace(/_/g, ' ');
  }

  // one plain-English sentence for a single parameter correction
  function correctionSentence(correction, verb) {
    return verb + ' ' + humanize(correction.variableName) + ' from ' +
      fmtNum(correction.observedValue) + ' to ' + fmtNum(correction.correctedValue) + '.';
  }

  function renderDiagnosis(diagnosis) {
    const primary = diagnosis.primary;
    if (!primary) return '';
    let html = '<div class="activity-diagnosis">';
    html += '<span class="activity-diagnosis-tag">' + esc(diagnosis.actionName) + '</span> ';
    html += '<span class="activity-diagnosis-text">' + esc(
      'While ' + actionPhrase(diagnosis.actionName) +
      (diagnosis.effectVariable ? ', ' + humanize(diagnosis.effectVariable) + ' came out wrong: ' : ', ') +
      correctionSentence(primary, 'the diagnosis corrected')
    ) + '</span>';
    (diagnosis.alsoCorrected || []).forEach(function (extra) {
      html += '<div class="activity-diagnosis-also">' +
        esc('Also ' + correctionSentence(extra, 'corrected')) + '</div>';
    });
    html += '</div>';
    return html;
  }

  function badge(label, on) {
    return '<span class="activity-badge ' + (on ? 'ok' : 'bad') + '">' +
      esc(label) + (on ? ' ✓' : ' ✗') + '</span>';
  }

  function fmtNum(value) {
    return typeof value === 'number' ? value.toFixed(4) : String(value);
  }

  function fmtSeconds(value) {
    return value.toFixed(1) + 's';
  }

  function esc(text) {
    return String(text).replace(/[&<>]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c];
    });
  }

  bus.on('live:changed', function (p) {
    liveUrl = p.url || '';
    if (timer) { clearInterval(timer); timer = null; }
    if (p.on) {
      statusEl.textContent = 'live';
      statusEl.classList.add('ready');
      lastCount = -1;   // force a render on the first poll after (re)attaching
      poll();
      timer = setInterval(poll, 1200);
    } else {
      statusEl.textContent = 'not live';
      statusEl.classList.remove('ready');
    }
  });

  return {
    destroy: function () {
      if (timer) { clearInterval(timer); timer = null; }
    },
  };
});
