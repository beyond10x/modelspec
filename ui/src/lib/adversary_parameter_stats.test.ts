// Adversary cases for the parameter statistics the page folds (story:observatory-parameter-statistics):
// runs whose every list is empty, as a Vulkan run's are; a stream resent from its start on a
// reconnect; a resumed run; and chart names shared by a metric and a parameter norm.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import type {
  MetricSample,
  ParameterStatistics,
  RunEvent,
  RunEventBody,
  StatisticsSubject,
} from '../generated/observatory.ts';
import { apply, charts, emptyRun, sampleAt } from './series.ts';

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

function stats(path: string, subject: StatisticsSubject, norm: number | null): ParameterStatistics {
  return {
    path,
    subject,
    norm,
    histogram: {
      bin_edges: [0, 1],
      counts: [4],
      nan_count: 0,
      positive_infinity_count: 0,
      negative_infinity_count: 0,
    },
  };
}

function step(
  step: number,
  parameters: ParameterStatistics[] = [],
  metrics: MetricSample[] = [],
): RunEventBody {
  return { kind: 'StepCompleted', value: { step, loss: 1, metrics, parameters } };
}

test('a run whose every list is empty has no parameter series, no sample and nothing to show', () => {
  const run = emptyRun();
  apply(run, event(0, { kind: 'RunStarted', value: { steps: 2, start_step: 0 } }));
  apply(run, event(1, step(0)));
  apply(run, event(2, step(1)));
  apply(run, event(3, step(2)));
  apply(run, event(4, { kind: 'RunFinished' }));
  assert.equal(run.state, 'finished');
  assert.equal(run.loss.points.length, 3);
  assert.deepEqual(run.parameters, []);
  assert.deepEqual(run.samples, []);
  assert.equal(sampleAt(run, null), null);
  assert.equal(sampleAt(run, 0), null);
});

test('a stream resent from its start on a reconnect adds no sample and no norm point twice', () => {
  const run = emptyRun();
  const events = [
    event(0, { kind: 'RunStarted', value: { steps: 2, start_step: 0 } }),
    event(1, step(0, [stats('M.w', 'Value', 2), stats('M.w', 'Gradient', 0.5)])),
    event(2, step(1)),
    event(3, step(2, [stats('M.w', 'Value', 1)])),
  ];
  for (const e of events) {
    apply(run, e);
  }
  for (const e of events) {
    assert.equal(apply(run, e), false, `sequence ${e.sequence} is folded once`);
  }
  apply(run, event(4, { kind: 'RunFinished' }));
  assert.deepEqual(
    run.samples.map((s) => s.step),
    [0, 2],
  );
  assert.equal(run.parameters.length, 1);
  assert.deepEqual(run.parameters[0]?.value.points, [
    { step: 0, value: 2 },
    { step: 2, value: 1 },
  ]);
  assert.deepEqual(run.parameters[0]?.gradient.points, [{ step: 0, value: 0.5 }]);
});

test('a resumed run samples from its start step, and its last sample has values only', () => {
  const run = emptyRun();
  apply(run, event(0, { kind: 'RunStarted', value: { steps: 5, start_step: 7 } }));
  apply(run, event(1, step(7, [stats('M.w', 'Value', 3), stats('M.w', 'Gradient', 1)])));
  for (let k = 8; k < 12; k += 1) {
    apply(run, event(k - 6, step(k)));
  }
  apply(run, event(6, step(12, [stats('M.w', 'Value', 2)])));
  assert.deepEqual(
    run.samples.map((s) => s.step),
    [7, 12],
  );
  assert.equal(sampleAt(run, null)?.step, 12);
  assert.equal(sampleAt(run, 9)?.step, 12, 'an unsampled step shows the latest sample');
  assert.deepEqual(
    sampleAt(run, 12)?.statistics.map((s) => s.subject),
    ['Value'],
  );
  assert.deepEqual(run.parameters[0]?.gradient.points, [{ step: 7, value: 1 }]);
});

// `Series.name` is "distinct among the charts of a run" (series.ts:14). The norm charts of a
// parameter are named `<path> value norm` and `<path> gradient norm` (series.ts `parameter`),
// and a metric keeps its own name, so a metric called `M.w value norm` and the parameter `M.w`
// give two charts on the page with one name.
test('a metric and a parameter norm chart never share a name', () => {
  const run = emptyRun();
  apply(run, event(0, { kind: 'RunStarted', value: { steps: 1, start_step: 0 } }));
  apply(
    run,
    event(1, step(0, [stats('M.w', 'Value', 2), stats('M.w', 'Gradient', 1)], [{ name: 'M.w value norm', value: 0.5 }])),
  );
  const names = [
    ...charts(run).map((s) => s.name),
    ...run.parameters.flatMap((p) => [p.value.name, p.gradient.name]),
  ];
  assert.equal(new Set(names).size, names.length, `chart names: ${JSON.stringify(names)}`);
});
