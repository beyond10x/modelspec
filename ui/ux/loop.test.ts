// L1 checks of the loop's own files: the core imports no project code, every scenario's truth is
// its recording's, every finding record has its header, and no closed S1 or S2 finding lacks a
// regression asset (the ui-iteration-loop skill § 1 and § 5).

import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { test } from 'node:test';
import { KEYS, SCENARIOS, UX_DIR, loadScenario, truthOf, truthValue } from './adapter.ts';
import type { Baseline } from './core/metrics.ts';
import { findingHeader as header } from './core/round.ts';

test('the core imports nothing from the adapter, the scenarios or the app', () => {
  for (const name of readdirSync(`${UX_DIR}core`).filter((n) => n.endsWith('.ts'))) {
    const text = readFileSync(`${UX_DIR}core/${name}`, 'utf8');
    for (const [, from] of text.matchAll(/from\s+'([^']+)'/g)) {
      assert.ok(
        from !== undefined && !from.startsWith('../') && !from.includes('adapter'),
        `core/${name} imports ${from}; the core stays free of project code`,
      );
    }
  }
});

test('every scenario has its truth from its recording and tasks that name truth keys', () => {
  for (const scenario of SCENARIOS) {
    const loaded = loadScenario(scenario);
    assert.deepEqual(truthOf(loaded.truth.recording, loaded.events), loaded.truth, `${scenario.name}/truth.json`);
    assert.ok(loaded.tasks.length > 0, `${scenario.name}/tasks.json has tasks`);
    for (const task of loaded.tasks) {
      assert.ok(task.name !== undefined || task.text !== undefined, `${scenario.name} ${task.id}: a name or a text`);
      for (const key of Object.values(task.answer)) {
        assert.notEqual(truthValue(loaded.truth, key), undefined, `${scenario.name} ${task.id}: truth key ${key}`);
      }
    }
  }
});

test('the baseline holds every key of every scenario', () => {
  const baseline = JSON.parse(readFileSync(`${UX_DIR}baseline.json`, 'utf8')) as Baseline;
  for (const scenario of SCENARIOS) {
    for (const key of Object.keys(KEYS)) {
      const entry = baseline.scenarios[scenario.name]?.[key as keyof typeof KEYS];
      assert.ok(entry !== undefined && Number.isFinite(entry.value), `baseline.json ${scenario.name} ${key}`);
    }
  }
});

const FIELDS = [
  'id',
  'title',
  'class',
  'severity',
  'metric',
  'status',
  'found_by',
  'evidence_recording',
  'evidence_screenshot',
  'evidence_lines',
  'regression_asset',
];

test('every finding has its header, and no closed S1 or S2 finding lacks a regression asset', () => {
  const dir = `${UX_DIR}findings`;
  const names = readdirSync(dir).filter((n) => /^UXF-\d{3}\.md$/.test(n));
  for (const name of names) {
    const fields = header(readFileSync(`${dir}/${name}`, 'utf8'));
    for (const field of FIELDS) {
      assert.ok(fields[field] !== undefined && fields[field] !== '', `${name}: header field ${field}`);
    }
    assert.equal(`${fields.id}.md`, name, `${name}: id`);
    assert.match(fields.class ?? '', /^(layout|cue|affordance|copy|noise|engine|input)$/, `${name}: class`);
    assert.match(fields.severity ?? '', /^S[123]$/, `${name}: severity`);
    assert.match(fields.status ?? '', /^(open|closed)$/, `${name}: status`);
    if (fields.status === 'closed' && /^S[12]$/.test(fields.severity ?? '')) {
      assert.doesNotMatch(fields.regression_asset ?? 'none', /^none\b/, `${name}: a closed ${fields.severity} finding needs a regression asset`);
    }
  }
});
