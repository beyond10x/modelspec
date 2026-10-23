import assert from 'node:assert/strict';
import { test } from 'node:test';
import type {
  Histogram,
  MetricSample,
  ParameterStatistics,
  RunEvent,
  RunEventBody,
  StatisticsSubject,
} from '../generated/observatory.ts';
import { LOSS, apply, chartName, charts, emptyRun, ends, sampleAt } from './series.ts';

const RUN = '00000000-0000-4000-8000-000000000001';

function event(sequence: number, body: RunEventBody, run = RUN): RunEvent {
  return {
    run_event_id: `00000000-0000-4000-8000-${String(sequence).padStart(12, '0')}`,
    training_run_id: run,
    sequence,
    body,
    state: 'Emitted',
  };
}

function step(step: number, loss: number | null, metrics: MetricSample[] = []): RunEventBody {
  return { kind: 'StepCompleted', value: { step, loss, metrics, parameters: [] } };
}

const started: RunEventBody = { kind: 'RunStarted', value: { steps: 2, start_step: 0 } };

test('a run folds to one loss point per StepCompleted, in order', () => {
  const run = emptyRun();
  assert.equal(run.state, 'connecting');
  apply(run, event(0, started));
  assert.equal(run.state, 'streaming');
  assert.equal(run.steps, 2);
  apply(run, event(1, step(0, 2.5)));
  apply(run, event(2, step(1, 0.125)));
  apply(run, event(3, { kind: 'RunFinished' }));
  assert.equal(run.state, 'finished');
  assert.deepEqual(run.loss.points, [
    { step: 0, value: 2.5 },
    { step: 1, value: 0.125 },
  ]);
  assert.deepEqual(
    charts(run).map((s) => s.name),
    ['loss'],
  );
});

test('every metric name gets its own series in the order it first appears', () => {
  const run = emptyRun();
  apply(run, event(0, started));
  apply(run, event(1, step(0, 1, [{ name: 'accuracy', value: 0.5 }])));
  apply(
    run,
    event(2, step(1, 0.5, [
      { name: 'perplexity', value: 9 },
      { name: 'accuracy', value: 0.75 },
    ])),
  );
  assert.deepEqual(
    charts(run).map((s) => s.name),
    ['loss', 'accuracy', 'perplexity'],
  );
  assert.deepEqual(run.metrics[0]?.points, [
    { step: 0, value: 0.5 },
    { step: 1, value: 0.75 },
  ]);
  assert.deepEqual(run.metrics[1]?.points, [{ step: 1, value: 9 }]);
});

test('a null loss is a point with no value', () => {
  const run = emptyRun();
  apply(run, event(0, started));
  apply(run, event(1, step(0, null)));
  assert.deepEqual(run.loss.points, [{ step: 0, value: null }]);
});

test('an event already folded and an event of another run are skipped', () => {
  const run = emptyRun();
  assert.equal(apply(run, event(0, started)), true);
  assert.equal(apply(run, event(1, step(0, 1))), true);
  assert.equal(apply(run, event(1, step(0, 1))), false);
  assert.equal(apply(run, event(0, started)), false);
  assert.equal(apply(run, event(2, step(1, 1), '00000000-0000-4000-8000-000000000002')), false);
  assert.equal(run.loss.points.length, 1);
});

test('a RunStarted of another run starts the page over on that run', () => {
  const other = '00000000-0000-4000-8000-000000000002';
  const run = emptyRun();
  apply(run, event(0, started));
  apply(run, event(1, step(0, 1, [{ name: 'accuracy', value: 0.5 }])));
  apply(run, event(2, step(1, 1)));
  assert.equal(apply(run, event(0, { kind: 'RunStarted', value: { steps: 5, start_step: 3 } }, other)), true);
  assert.equal(run.trainingRunId, other);
  assert.equal(run.state, 'streaming');
  assert.equal(run.steps, 5);
  assert.equal(run.startStep, 3);
  assert.deepEqual(run.loss.points, []);
  assert.deepEqual(run.metrics, []);
  assert.equal(apply(run, event(1, step(3, 0.25), other)), true);
  assert.deepEqual(run.loss.points, [{ step: 3, value: 0.25 }]);
  assert.equal(apply(run, event(3, step(2, 1))), false, 'the old run is skipped from then on');
});

test('no metric name gives the chart name of the loss or of another metric', () => {
  const names = ['loss', 'loss (metric)', 'loss (metric) (metric)', 'accuracy', 'Loss', 'loss ', ' loss'];
  const shown = names.map(chartName);
  assert.deepEqual(shown, [
    'loss (metric)',
    'loss (metric) (metric)',
    'loss (metric) (metric) (metric)',
    'accuracy',
    'Loss',
    'loss ',
    ' loss',
  ]);
  assert.equal(new Set([LOSS, ...shown]).size, names.length + 1);
  const run = emptyRun();
  apply(run, event(0, started));
  apply(run, event(1, step(0, 1, names.map((name) => ({ name, value: 1 })))));
  assert.deepEqual(
    charts(run).map((s) => s.name),
    [LOSS, ...shown],
  );
});

test('a failure keeps its code and message', () => {
  const run = emptyRun();
  apply(run, event(0, started));
  apply(run, event(1, { kind: 'RunFailed', value: { code: 'E-DATA-003', message: 'm' } }));
  assert.equal(run.state, 'failed');
  assert.equal(run.failure, 'E-DATA-003: m');
  const plain = emptyRun();
  apply(plain, event(0, { kind: 'RunFailed', value: { message: 'out of memory' } }));
  assert.equal(plain.failure, 'out of memory');
});

test('RunFinished and RunFailed end the stream, nothing else does', () => {
  assert.equal(ends(event(0, started)), false);
  assert.equal(ends(event(0, step(0, 1))), false);
  assert.equal(ends(event(0, { kind: 'RunFinished' })), true);
  assert.equal(ends(event(0, { kind: 'RunFailed', value: { message: 'm' } })), true);
});

function histogram(counts: number[]): Histogram {
  return {
    bin_edges: counts.length === 0 ? [] : counts.map((_, i) => i).concat([counts.length]),
    counts,
    nan_count: 0,
    positive_infinity_count: 0,
    negative_infinity_count: 0,
  };
}

function stats(path: string, subject: StatisticsSubject, norm: number | null, counts = [1]): ParameterStatistics {
  return { path, subject, norm, histogram: histogram(counts) };
}

function sampled(step: number, parameters: ParameterStatistics[]): RunEventBody {
  return { kind: 'StepCompleted', value: { step, loss: 1, metrics: [], parameters } };
}

test('parameter statistics fold into a value and a gradient norm series per path, in the order paths first appear', () => {
  const run = emptyRun();
  apply(run, event(0, started));
  apply(
    run,
    event(1, sampled(0, [stats('M.b', 'Value', 2), stats('M.b', 'Gradient', 0.5), stats('M.a', 'Value', 3)])),
  );
  apply(run, event(2, step(1, 0.5)));
  apply(run, event(3, sampled(2, [stats('M.b', 'Value', 1.5), stats('M.a', 'Value', null)])));
  assert.deepEqual(
    run.parameters.map((p) => p.path),
    ['M.b', 'M.a'],
  );
  const [b, a] = run.parameters;
  assert.deepEqual(b?.value.points, [
    { step: 0, value: 2 },
    { step: 2, value: 1.5 },
  ]);
  assert.deepEqual(b?.gradient.points, [{ step: 0, value: 0.5 }], 'the last step has values only');
  assert.deepEqual(a?.value.points, [
    { step: 0, value: 3 },
    { step: 2, value: null },
  ]);
  assert.deepEqual(a?.gradient.points, [], 'a parameter the loss does not read has no gradient');
  assert.notEqual(b?.value.name, b?.gradient.name);
  assert.deepEqual(
    charts(run).map((s) => s.name),
    ['loss'],
    'norms are not metric charts',
  );
});

test('only a step with statistics is a sample, and the sample shown is the selected one, else the latest', () => {
  const run = emptyRun();
  apply(run, event(0, started));
  assert.equal(sampleAt(run, null), null, 'no sample before the first');
  apply(run, event(1, sampled(0, [stats('M.w', 'Value', 2, [1, 0, 3])])));
  apply(run, event(2, step(1, 0.5)));
  apply(run, event(3, sampled(2, [stats('M.w', 'Value', 1, [2, 2])])));
  assert.deepEqual(
    run.samples.map((s) => s.step),
    [0, 2],
  );
  assert.equal(sampleAt(run, null)?.step, 2);
  assert.deepEqual(sampleAt(run, 0)?.statistics[0]?.histogram.counts, [1, 0, 3]);
  assert.equal(sampleAt(run, 1)?.step, 2, 'a step without statistics shows the latest sample');
});

test('a RunStarted of another run clears the parameter series and the samples', () => {
  const run = emptyRun();
  apply(run, event(0, started));
  apply(run, event(1, sampled(0, [stats('M.w', 'Value', 2)])));
  apply(run, event(0, started, '00000000-0000-4000-8000-000000000002'));
  assert.deepEqual(run.parameters, []);
  assert.deepEqual(run.samples, []);
});

test('every chart of a run has its own name, metrics named like a norm chart included', () => {
  const paths = ['M.w', 'M', 'M.w value', 'M.w gradient', ''];
  const metrics = [
    'loss',
    'loss (metric)',
    'M.w value norm',
    'M.w gradient norm',
    'M.w value norm (metric)',
    'M value norm',
    ' gradient norm',
    'M.w value norm value norm',
    'value norm',
    'M.w value norms',
    'accuracy',
  ];
  const run = emptyRun();
  apply(run, event(0, started));
  apply(run, event(1, sampled(0, paths.flatMap((p) => [stats(p, 'Value', 1), stats(p, 'Gradient', 1)]))));
  apply(run, event(2, step(1, 1, metrics.map((name) => ({ name, value: 1 })))));
  const names = [
    ...charts(run).map((s) => s.name),
    ...run.parameters.flatMap((p) => [p.value.name, p.gradient.name]),
  ];
  assert.equal(names.length, 1 + metrics.length + 2 * paths.length);
  assert.equal(new Set(names).size, names.length, `chart names: ${JSON.stringify(names)}`);
  assert.equal(chartName('value norm'), 'value norm', 'a name without the leading space is no norm chart name');
  assert.equal(chartName('M.w value norms'), 'M.w value norms');
  assert.equal(chartName('accuracy'), 'accuracy');
});
