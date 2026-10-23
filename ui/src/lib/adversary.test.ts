import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { MetricSample, RunEvent, RunEventBody } from '../generated/observatory.ts';
import { FRAME, line } from './chart.ts';
import { apply, charts, emptyRun } from './series.ts';

const RUN = '00000000-0000-4000-8000-000000000001';

function event(sequence: number, body: RunEventBody): RunEvent {
  return {
    run_event_id: `00000000-0000-4000-8000-${String(sequence).padStart(12, '0')}`,
    training_run_id: RUN,
    sequence,
    body,
    state: 'Emitted',
  };
}

function step(step: number, loss: number | null, metrics: MetricSample[] = []): RunEventBody {
  return { kind: 'StepCompleted', value: { step, loss, metrics, parameters: [] } };
}

// A metric value is any finite Binary64. Two finite values whose difference is not finite make the
// chart's value range `high - low` infinite, and the point at `high` projects to (inf / inf), NaN.
test('a series of finite values far apart draws a line with no NaN in it', () => {
  const d = line(FRAME, [
    { step: 0, value: -1e308 },
    { step: 1, value: 1e308 },
  ]);
  assert.ok(!d.includes('NaN'), `the line of two finite points is drawable: ${d}`);
});

// `charts` names its series `loss` and then each metric's name; a metric named `loss` (the spec's
// distinct rule covers only the metrics among themselves) gives two series of one name, so two
// `MetricChart`s with the same Vue `key` and two `svg[data-metric="loss"]`.
test('the charts of a run have distinct names when a metric is called loss', () => {
  const run = emptyRun();
  apply(run, event(0, { kind: 'RunStarted', value: { steps: 1, start_step: 0 } }));
  apply(run, event(1, step(0, 2, [{ name: 'loss', value: 0.5 }])));
  const names = charts(run).map((s) => s.name);
  assert.equal(new Set(names).size, names.length, `chart names: ${JSON.stringify(names)}`);
});
