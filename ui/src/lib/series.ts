import type { ParameterStatistics, RunEvent } from '../generated/observatory.ts';

/** Where the stream of a run is: before its first event, between its first and last, or ended. */
export type RunState = 'connecting' | 'streaming' | 'finished' | 'failed';

/** One value at one step; `null` is a value that was NaN or ±inf on the server. */
export interface Point {
  step: number;
  value: number | null;
}

/** The values of one named quantity, one point per `StepCompleted` that carried it, in step order. */
export interface Series {
  /** The chart's name, distinct among the charts of a run (`chartName`). */
  name: string;
  points: Point[];
}

/** What the charts draw: the events of one run folded in `sequence` order. */
export interface Run {
  state: RunState;
  trainingRunId: string | null;
  /** `RunStarted.steps` and `RunStarted.start_step`, once the run has started. */
  steps: number | null;
  startStep: number | null;
  loss: Series;
  /** One series per metric name, in the order the names first appear. */
  metrics: Series[];
  /** The norms of each parameter, in the order the paths first appear. */
  parameters: ParameterSeries[];
  /** Every `StepCompleted` that carried statistics, in step order. */
  samples: Sample[];
  /** The `RunFailed` message, with its code when it has one. */
  failure: string | null;
  /** The `sequence` the next event must carry. */
  next: number;
}

/**
 * The norms of one parameter over the sampled steps: of its values at every sample, and of its
 * gradient at every sample that has one. The last step of a run makes no update and so has no
 * gradient, and a parameter the loss does not read never has one.
 */
export interface ParameterSeries {
  path: string;
  value: Series;
  gradient: Series;
}

/** The statistics of one sampled step, as the stream sent them. */
export interface Sample {
  step: number;
  statistics: ParameterStatistics[];
}

export const LOSS = 'loss';

/** The names that `chartName` gives one more ` (metric)`: `loss`, `loss (metric)`, ... */
const RESERVED = /^loss( \(metric\))*$/;

/**
 * The chart name of the metric `name`: the name itself, except that a name of the form `loss`,
 * `loss (metric)`, ... gets one more ` (metric)`. Metric names are distinct within a step
 * (`StepCompleted`), and the map is one-to-one and never gives `loss`, so the loss chart and every
 * metric chart have distinct names, and so distinct Vue keys and `svg[data-metric]` values.
 */
export function chartName(name: string): string {
  return RESERVED.test(name) ? `${name} (metric)` : name;
}

export function emptyRun(): Run {
  return {
    state: 'connecting',
    trainingRunId: null,
    steps: null,
    startStep: null,
    loss: { name: LOSS, points: [] },
    metrics: [],
    parameters: [],
    samples: [],
    failure: null,
    next: 0,
  };
}

/** The series of the parameter `path`, added after the others the first time it is seen. */
function parameter(run: Run, path: string): ParameterSeries {
  let series = run.parameters.find((p) => p.path === path);
  if (series === undefined) {
    series = {
      path,
      value: { name: `${path} value norm`, points: [] },
      gradient: { name: `${path} gradient norm`, points: [] },
    };
    run.parameters.push(series);
  }
  return series;
}

/**
 * Folds `event` into `run` in place and returns whether it changed. A `RunStarted` of another run
 * (the server at the page's address was restarted with a new run) starts `run` over on that run.
 * An event whose `sequence` was already folded (the stream resends a run from its start on every
 * reconnect) and any other event of another run are skipped.
 */
export function apply(run: Run, event: RunEvent): boolean {
  if (
    run.trainingRunId !== null &&
    event.training_run_id !== run.trainingRunId &&
    event.body.kind === 'RunStarted'
  ) {
    Object.assign(run, emptyRun());
  }
  if (event.sequence < run.next) {
    return false;
  }
  if (run.trainingRunId !== null && event.training_run_id !== run.trainingRunId) {
    return false;
  }
  run.trainingRunId = event.training_run_id;
  run.next = event.sequence + 1;
  const body = event.body;
  switch (body.kind) {
    case 'RunStarted':
      run.state = 'streaming';
      run.steps = body.value.steps;
      run.startStep = body.value.start_step;
      break;
    case 'StepCompleted': {
      run.state = 'streaming';
      const step = body.value.step;
      run.loss.points.push({ step, value: body.value.loss });
      for (const sample of body.value.metrics) {
        const name = chartName(sample.name);
        let series = run.metrics.find((s) => s.name === name);
        if (series === undefined) {
          series = { name, points: [] };
          run.metrics.push(series);
        }
        series.points.push({ step, value: sample.value });
      }
      const statistics = body.value.parameters;
      if (statistics.length > 0) {
        run.samples.push({ step, statistics });
        for (const entry of statistics) {
          const series = parameter(run, entry.path);
          const norms = entry.subject === 'Value' ? series.value : series.gradient;
          norms.points.push({ step, value: entry.norm });
        }
      }
      break;
    }
    case 'RunFinished':
      run.state = 'finished';
      break;
    case 'RunFailed':
      run.state = 'failed';
      run.failure =
        body.value.code === undefined ? body.value.message : `${body.value.code}: ${body.value.message}`;
      break;
  }
  return true;
}

/** The series the page draws: the loss first, then every metric. */
export function charts(run: Run): Series[] {
  return [run.loss, ...run.metrics];
}

/**
 * The sample the histograms show: the one of `step` when that step was sampled, else the latest,
 * or `null` before the first.
 */
export function sampleAt(run: Run, step: number | null): Sample | null {
  const chosen = step === null ? undefined : run.samples.find((s) => s.step === step);
  return chosen ?? run.samples.at(-1) ?? null;
}

/** Whether `event` ends its run's stream. */
export function ends(event: RunEvent): boolean {
  return event.body.kind === 'RunFinished' || event.body.kind === 'RunFailed';
}
