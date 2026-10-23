import type { RunEvent } from '../generated/observatory.ts';

/** Where the stream of a run is: before its first event, between its first and last, or ended. */
export type RunState = 'connecting' | 'streaming' | 'finished' | 'failed';

/** One value at one step; `null` is a value that was NaN or ±inf on the server. */
export interface Point {
  step: number;
  value: number | null;
}

/** The values of one named quantity, one point per `StepCompleted` that carried it, in step order. */
export interface Series {
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
  /** The `RunFailed` message, with its code when it has one. */
  failure: string | null;
  /** The `sequence` the next event must carry. */
  next: number;
}

export const LOSS = 'loss';

export function emptyRun(): Run {
  return {
    state: 'connecting',
    trainingRunId: null,
    steps: null,
    startStep: null,
    loss: { name: LOSS, points: [] },
    metrics: [],
    failure: null,
    next: 0,
  };
}

/**
 * Folds `event` into `run` in place and returns whether it changed. An event whose `sequence` was
 * already folded (a reconnecting stream sends from `Last-Event-ID` on, but a replay of the page may
 * not) and an event of another run are skipped.
 */
export function apply(run: Run, event: RunEvent): boolean {
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
        let series = run.metrics.find((s) => s.name === sample.name);
        if (series === undefined) {
          series = { name: sample.name, points: [] };
          run.metrics.push(series);
        }
        series.points.push({ step, value: sample.value });
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

/** Whether `event` ends its run's stream. */
export function ends(event: RunEvent): boolean {
  return event.body.kind === 'RunFinished' || event.body.kind === 'RunFailed';
}
