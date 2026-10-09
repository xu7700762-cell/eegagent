const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'eeg_agent/web/brain.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'eeg_agent/web/brain.html'), 'utf8');

test('all statically referenced DOM ids exist in the current webpage', () => {
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]));
  for (const match of source.matchAll(/\$\('([^']+)'\)/g)) assert.ok(ids.has(match[1]), match[1]);
});

test('evaluation table renders only current GPT and VRMSModel metrics', () => {
  function node(tag, cls, text) {
    return {tag, cls, text, children: [], append(...values) { this.children.push(...values); }};
  }
  const start = source.indexOf('function buildEvaluationTable(');
  const end = source.indexOf('function renderEvaluation(', start);
  const context = {el: node};
  vm.createContext(context);
  vm.runInContext(source.slice(start, end), context);
  const metric = JSON.parse(fs.readFileSync(path.join(root, 'results/current_seed2026.json'), 'utf8'));
  const rendered = JSON.stringify(context.buildEvaluationTable(metric.metrics, 'completed'));
  assert.match(rendered, /73\.97%/);
  assert.match(rendered, /64\.38%/);
  assert.match(rendered, /108 \/ 146/);
  assert.match(rendered, /94 \/ 146/);
  assert.doesNotMatch(rendered, /数值融合|云端二次判断/);
});
