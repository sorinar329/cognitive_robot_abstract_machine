// Unit tests for web/core/split-resize.js (node:test).
// A minimal fake DOM: just enough element/selector/event/pointer-capture behaviour for
// the script's own logic to run against, so the resize math is covered without a real
// browser.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cram_viz', 'src', 'cram_viz', 'web');

function makeElement(tag) {
  const handlers = {};
  const el = {
    tagName: tag,
    className: '',
    style: {},
    dataset: {},
    title: '',
    children: [],
    _rect: { top: 0, left: 0, width: 0, height: 0 },
    classList: {
      contains(name) { return (' ' + el.className + ' ').indexOf(' ' + name + ' ') >= 0; },
      add(name) { if (!this.contains(name)) el.className = (el.className + ' ' + name).trim(); },
      remove(name) { el.className = el.className.split(' ').filter(c => c !== name).join(' '); },
    },
    appendChild(child) { el.children.push(child); return child; },
    insertBefore(newEl, refEl) {
      const idx = el.children.indexOf(refEl);
      if (idx < 0) el.children.push(newEl); else el.children.splice(idx, 0, newEl);
      return newEl;
    },
    querySelector(selector) { return queryIn(el, selector); },
    addEventListener(type, fn) { (handlers[type] = handlers[type] || []).push(fn); },
    removeEventListener(type, fn) {
      const list = handlers[type]; if (!list) return;
      const i = list.indexOf(fn); if (i >= 0) list.splice(i, 1);
    },
    dispatch(type, evt) { (handlers[type] || []).slice().forEach(fn => fn(evt)); },
    setPointerCapture() {},
    getBoundingClientRect() { return el._rect; },
  };
  return el;
}

function selectorMatches(el, selector) {
  if (selector.charAt(0) === '[') {
    const m = /\[data-slot="([^"]+)"\]/.exec(selector);
    return !!m && el.dataset.slot === m[1];
  }
  const m = /^([a-zA-Z]*)((?:\.[\w-]+)*)$/.exec(selector);
  const tag = m[1];
  const classes = m[2] ? m[2].slice(1).split('.') : [];
  if (tag && el.tagName.toLowerCase() !== tag.toLowerCase()) return false;
  return classes.every(c => el.classList.contains(c));
}

function queryIn(root, selector) {
  for (const child of root.children) {
    if (selectorMatches(child, selector)) return child;
    const found = queryIn(child, selector);
    if (found) return found;
  }
  return null;
}

// %% one fixture: main.split > [left (1 panel), right (2 panels: eql, graph)]
function setupDom() {
  const store = {};
  global.localStorage = {
    getItem: k => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
  };
  global.location = { pathname: '/index.html' };
  global.window = {};

  const body = makeElement('body');
  const main = makeElement('main'); main.className = 'split';
  const left = makeElement('div'); left.dataset.slot = 'left';
  const right = makeElement('div'); right.dataset.slot = 'right';
  right._rect = { top: 100, left: 0, width: 300, height: 400 };
  body.appendChild(main);
  main.appendChild(left);
  main.appendChild(right);

  const robotScene = makeElement('section'); robotScene.className = 'panel panel-robot-scene';
  left.appendChild(robotScene);

  const eql = makeElement('section'); eql.className = 'panel panel-eql';
  const graph = makeElement('section'); graph.className = 'panel panel-graph';
  right.appendChild(eql);
  right.appendChild(graph);

  global.document = {
    querySelector: selector => queryIn(body, selector),
    createElement: makeElement,
    addEventListener() {},
  };
  return { main, left, right, robotScene, eql, graph, store };
}

function load() {
  new Function(fs.readFileSync(path.join(WEB, 'core/split-resize.js'), 'utf8'))();
}

function rowDividerOf(slotEl) {
  return slotEl.children.find(c => c.classList.contains('split-divider-row'));
}

function drag(divider, rect, fromClientY, toClientY) {
  divider.dispatch('pointerdown', { preventDefault() {}, pointerId: 1, clientY: fromClientY });
  divider.dispatch('pointermove', { clientY: toClientY });
  divider.dispatch('pointerup', {});
}

test('a slot with two stacked panels gets a row divider, sized 35/65 by default', function () {
  const { right, eql, graph } = setupDom();
  load();
  assert.strictEqual(eql.style.flex, '0 0 35%');
  assert.strictEqual(graph.style.flex, '1 1 0');
  assert.ok(rowDividerOf(right));
});

test('a slot with only one panel gets no row divider', function () {
  const { left, robotScene } = setupDom();
  load();
  assert.strictEqual(rowDividerOf(left), undefined);
  assert.strictEqual(robotScene.style.flex, undefined);
});

test('dragging the row divider resizes the top panel and persists the split', function () {
  const { right, eql, graph, store } = setupDom();
  load();
  const divider = rowDividerOf(right);
  drag(divider, right._rect, 100, 300); // clientY 300 of a [100, 500] rect = 50%
  assert.strictEqual(eql.style.flex, '0 0 50%');
  assert.strictEqual(graph.style.flex, '1 1 0');
  assert.strictEqual(store['splitRows:right:index.html'], '50.0');
});

test('the row split is clamped to [15, 80] percent', function () {
  const { right, eql } = setupDom();
  load();
  const divider = rowDividerOf(right);
  drag(divider, right._rect, 100, 100); // clientY at the very top = 0%
  assert.strictEqual(eql.style.flex, '0 0 15%');
  drag(divider, right._rect, 100, 900); // far past the bottom = way over 100%
  assert.strictEqual(eql.style.flex, '0 0 80%');
});

test('double-clicking the row divider resets to the 35/65 default', function () {
  const { right, eql, store } = setupDom();
  load();
  const divider = rowDividerOf(right);
  drag(divider, right._rect, 100, 300);
  assert.strictEqual(eql.style.flex, '0 0 50%');
  divider.dispatch('dblclick', {});
  assert.strictEqual(eql.style.flex, '0 0 35%');
  assert.strictEqual(store['splitRows:right:index.html'], '35');
});

test('a persisted split percentage is applied on load, without needing to drag again', function () {
  const fixture = setupDom();
  fixture.store['splitRows:right:index.html'] = '60';
  load();
  assert.strictEqual(fixture.eql.style.flex, '0 0 60%');
});
