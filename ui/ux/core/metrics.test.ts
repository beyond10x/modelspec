import assert from 'node:assert/strict';
import { test } from 'node:test';
import { exposed } from './bot.ts';
import { type Baseline, glance, median, per100Events, percentile, ratchet, update, words } from './metrics.ts';
import { frames } from './replay.ts';

test('median and the nearest-rank percentile', () => {
  assert.equal(median([]), 0);
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 2, 3]), 2.5);
  assert.equal(percentile([1, 1, 1, 3], 95), 3);
  assert.equal(percentile([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3], 95), 2);
});

test('words counts runs of non-whitespace', () => {
  assert.equal(words('  step 200 of 200 '), 4);
  assert.equal(words(''), 0);
});

test('layout shift per 100 replayed events', () => {
  assert.equal(per100Events(0.06, 600), 0.01);
  assert.equal(per100Events(0.06, 300), 0.02);
  assert.equal(per100Events(1, 0), 0);
});

test('glance: 1 in the primary card, 2 elsewhere in view, 3 out of view', () => {
  const view = { width: 100, height: 100 };
  const primary = { left: 0, top: 10, right: 100, bottom: 50 };
  assert.equal(glance({ left: 10, top: 20, right: 30, bottom: 30 }, primary, view), 1);
  assert.equal(glance({ left: 10, top: 60, right: 30, bottom: 70 }, primary, view), 2);
  assert.equal(glance({ left: 10, top: 90, right: 30, bottom: 120 }, primary, view), 3);
});

test('the exposed name or text of an aria snapshot line', () => {
  assert.equal(exposed('- figure "loss step 200: 0.0184982":\n  - img "loss per step"'), 'loss step 200: 0.0184982');
  assert.equal(exposed('- paragraph: finished step 200 of 200'), 'finished step 200 of 200');
  assert.equal(exposed('- figure "a \\"b\\""'), 'a "b"');
  assert.equal(exposed(`- 'figure "loss step 20: 0.0952381"':\n  - text: "loss step 20: 0.0952381"`), 'loss step 20: 0.0952381');
  assert.equal(exposed(`- paragraph: 'run a: b'`), 'run a: b');
  assert.equal(exposed('- heading "ModelSpec observatory" [level=1]'), 'ModelSpec observatory');
});

test('SSE frames keep their text and parse their data', () => {
  const parsed = frames('event: RunStarted\nid: 0\ndata: {"a":1}\n\nevent: RunFinished\nid: 1\ndata: {"b":2}\n\n');
  assert.equal(parsed.length, 2);
  assert.equal(parsed[0]?.event, 'RunStarted');
  assert.deepEqual(parsed[1]?.data, { b: 2 });
  assert.equal(parsed[0]?.text, 'event: RunStarted\nid: 0\ndata: {"a":1}\n\n');
});

const KEYS = { 'reading_load.max': 'counts', 'screen_only.time_to_act_steps.max': 'steps', 'layout_shift.per_100_events': 'layout_shift' } as const;
const keys = KEYS as unknown as Parameters<typeof ratchet>[1];
const tolerances: Baseline['tolerances'] = { counts: { absolute: 0 }, steps: { relative: 0.05 }, layout_shift: { absolute: 0.01 } };

function base(values: Record<string, number>, decision?: string): Baseline {
  return {
    about: '',
    tolerances,
    scenarios: {
      s: Object.fromEntries(
        Object.entries(values).map(([k, v]) => [k, { value: v, tolerance: KEYS[k as keyof typeof KEYS], ...(decision ? { decision } : {}) }]),
      ),
    },
  };
}

function kinds(problems: ReturnType<typeof ratchet>): string[] {
  return problems.map((p) => `${p.key} ${p.kind}`);
}

test('the ratchet passes on values within tolerance', () => {
  const b = base({ 'reading_load.max': 40, 'screen_only.time_to_act_steps.max': 20, 'layout_shift.per_100_events': 0.1 });
  const m = { s: { 'reading_load.max': 40, 'screen_only.time_to_act_steps.max': 21, 'layout_shift.per_100_events': 0.105 } };
  assert.deepEqual(ratchet(['s'], keys, tolerances, b, m, null), []);
});

test('the ratchet fails on worse, on an improvement not taken, and on a missing value', () => {
  const b = base({ 'reading_load.max': 40, 'screen_only.time_to_act_steps.max': 20, 'layout_shift.per_100_events': 0.1 });
  const m = { s: { 'reading_load.max': 41, 'screen_only.time_to_act_steps.max': 18 } };
  assert.deepEqual(kinds(ratchet(['s'], keys, tolerances, b, m, null)), [
    'reading_load.max worse',
    'screen_only.time_to_act_steps.max improved',
    'layout_shift.per_100_events missing',
  ]);
});

test('the ratchet fails on a baseline made worse than main without a decision', () => {
  const main = base({ 'reading_load.max': 40, 'screen_only.time_to_act_steps.max': 0, 'layout_shift.per_100_events': 0 });
  const worse = base({ 'reading_load.max': 41, 'screen_only.time_to_act_steps.max': 0, 'layout_shift.per_100_events': 0 });
  const m = { s: { 'reading_load.max': 41, 'screen_only.time_to_act_steps.max': 0, 'layout_shift.per_100_events': 0 } };
  assert.deepEqual(kinds(ratchet(['s'], keys, tolerances, worse, m, main)), ['reading_load.max baseline-worse']);
  const decided = base({ 'reading_load.max': 41, 'screen_only.time_to_act_steps.max': 0, 'layout_shift.per_100_events': 0 }, 'D99');
  assert.deepEqual(ratchet(['s'], keys, tolerances, decided, m, main), []);
});

test('update takes improvements and missing values, and a worse value only with a decision', () => {
  const b = base({ 'reading_load.max': 40, 'screen_only.time_to_act_steps.max': 20 });
  const m = { s: { 'reading_load.max': 39, 'screen_only.time_to_act_steps.max': 30, 'layout_shift.per_100_events': 0 } };
  const taken = update(['s'], keys, tolerances, b, m, null);
  assert.equal(taken.scenarios.s?.['reading_load.max']?.value, 39);
  assert.equal(taken.scenarios.s?.['screen_only.time_to_act_steps.max']?.value, 20);
  assert.equal(taken.scenarios.s?.['layout_shift.per_100_events']?.value, 0);
  const decided = update(['s'], keys, tolerances, b, m, 'D99');
  assert.deepEqual(decided.scenarios.s?.['screen_only.time_to_act_steps.max'], { value: 30, tolerance: 'steps', decision: 'D99' });
});
