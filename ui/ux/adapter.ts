// The observatory's adapter for the ui-iteration-loop (story:observatory-ui-iteration-loop):
// every slot the loop's generic core (`core/`) needs, filled from `src/App.vue` and its
// components. The core imports nothing from here or from `src/`; the specs join the two.

import { readFileSync } from 'node:fs';
import type { RunEvent } from '../src/generated/observatory.ts';
import type { MetricKey, Theme, ToleranceClass } from './core/metrics.ts';

/** The directory of `ui/`, from this file. */
export const UI_DIR = new URL('..', import.meta.url).pathname;
/** The repository root, from this file. */
export const REPO_ROOT = new URL('../..', import.meta.url).pathname;
/** The served app, `task ui` output. */
export const UI_DIST = 'ui/dist';
/** Where a run writes what it measures; `node_modules/` is never committed. */
export const OUT_DIR = new URL('../node_modules/.cache/ux/', import.meta.url).pathname;
/** The loop's tracked files. */
export const UX_DIR = new URL('./', import.meta.url).pathname;

/**
 * A scenario: one recorded run the server replays (`modelspec observe <recording>`), with the
 * truth of the end state in `scenarios/<name>/truth.json` and the screen-only user's tasks in
 * `scenarios/<name>/tasks.json`. Two anchors make rounds comparable; the rotating card changes per
 * round to find new problems.
 */
export interface Scenario {
  name: string;
  card: 'anchor' | 'rotating';
}

export const SCENARIOS: readonly Scenario[] = [
  { name: 'linear-regression', card: 'anchor' },
  { name: 'two-layer-relu', card: 'anchor' },
  { name: 'accuracy', card: 'rotating' },
];

export interface Viewport {
  width: number;
  height: number;
}

export function viewportName(v: Viewport): string {
  return `${v.width}x${v.height}`;
}

/** Every merge checks these. */
export const GATING_VIEWPORTS: readonly Viewport[] = [
  { width: 1366, height: 768 },
  { width: 1280, height: 720 },
];

/** Reported in a round, never gated. */
export const MEASURED_VIEWPORTS: readonly Viewport[] = [
  { width: 1440, height: 900 },
  { width: 1920, height: 1080 },
  { width: 390, height: 844 },
];

/** The app follows `prefers-color-scheme` (`src/style.css`); both are emulated. */
export const THEMES: readonly Theme[] = ['light', 'dark'];

/**
 * The `data-ux-region` marks of `src/` and what each is. The loss chart is the primary card: a
 * value read there costs one glance.
 */
export const REGIONS = {
  header: 'the run header: title, run state, progress, run id',
  failure: 'the failure alert of a failed run',
  'loss-chart': 'the loss chart, the primary card',
  'metric-chart': 'one chart per metric the run reports',
  parameters: 'the parameter section: the histogram step picker and one panel per parameter',
  'parameter-panel': 'one parameter: its value and gradient norm charts and histograms',
  histogram: 'one histogram of a parameter at the shown step',
} as const;

export type Region = keyof typeof REGIONS;

export const PRIMARY_REGION: Region = 'loss-chart';

/**
 * An element that must be 100 % inside the viewport, not clipped and not covered, at every state
 * change of every replay at the gating viewports (gate 2). `when` limits it to the states where
 * the app draws it.
 */
export interface AlwaysVisible {
  name: string;
  selector: string;
  when: 'always' | 'streamed' | 'parameters';
}

/**
 * The run header with its run state from the first paint; the loss chart once the first step is
 * drawn; and, for a run that carries parameter statistics, the parameter section's head: its step
 * picker and the name of the first parameter. The parameter panels themselves are 960 px each and
 * cannot be in view at once; the app has no list of them (UXF-001).
 */
export const ALWAYS_VISIBLE: readonly AlwaysVisible[] = [
  { name: 'run header', selector: '[data-ux-region="header"]', when: 'always' },
  { name: 'run state', selector: '[data-ux-region="header"] .state', when: 'always' },
  { name: 'loss chart', selector: '[data-ux-region="loss-chart"]', when: 'streamed' },
  { name: 'histogram step picker', selector: '[data-ux-region="parameters"] > .pick', when: 'parameters' },
  {
    name: 'first parameter name',
    selector: '[data-ux-region="parameters"] > [data-ux-region="parameter-panel"]:first-of-type > h2',
    when: 'parameters',
  },
];

/** What "done" looks like: the state word in the header, and for a failure also the alert. */
export const COMPLETION = {
  finished: { selector: '[data-ux-region="header"] .state', text: 'finished' },
  failed: { selector: '[data-ux-region="failure"][role="alert"]', text: null },
  /** The words the header's state may read; a completion word before the end is `false_complete`. */
  stateSelector: '[data-ux-region="header"] .state',
  completionWords: ['finished', 'failed'],
} as const;

/** The moments the contact sheet shows and the screen-only user reads every task at. */
export type Moment = 'connecting' | 'started' | 'first-step' | 'middle-step' | 'last-step' | 'end';

export const MOMENTS: readonly Moment[] = ['connecting', 'started', 'first-step', 'middle-step', 'last-step', 'end'];

/**
 * The compressed clock of a replay: the harness steps the recording one event at a time and the
 * app draws each before the next is sent, so time on the replay is counted in events. A minute is
 * 600 events (10 per second), the unit of `layout_shift.per_minute`.
 */
export const EVENTS_PER_MINUTE = 600;

/**
 * `reading_load`: the words a reader faces at one moment, the skill's references leave it open.
 * Defined here: the whitespace-separated words of every rendered text node, HTML or SVG, with at
 * least one client rectangle inside the viewport, counted at each key moment of each gating replay
 * (the closed `<select>` shows only its chosen option, and `<title>` tooltips are not rendered).
 * Lower is better.
 */
export const READING_LOAD_DEFINITION = 'words of rendered text nodes inside the viewport at a key moment';

/** The tolerance class of each ratchet key; the skill's numbers. */
export const TOLERANCES: Record<ToleranceClass, { absolute?: number; relative?: number }> = {
  counts: { absolute: 0 },
  steps: { relative: 0.05 },
  layout_shift: { absolute: 0.01 },
};

/** Every ratchet key of a scenario (skill § 6 without the cue keys: the app has no guidance cue). */
export const KEYS: Record<MetricKey, ToleranceClass> = {
  'actions_per_fix.max': 'counts',
  'actions_per_fix.median': 'counts',
  'glance_cost.median': 'counts',
  'glance_cost.p95': 'counts',
  'layout_shift.per_minute': 'layout_shift',
  'reading_load.max': 'counts',
  'reading_load.median': 'counts',
  'screen_only.actions': 'counts',
  'screen_only.looks': 'counts',
  'screen_only.steers': 'counts',
  'screen_only.time_to_act_steps.max': 'steps',
};

/** Gate 4: a fix takes at most this many actions. */
export const MAX_ACTIONS_PER_FIX = 2;
/** The screen-only user gives up on a task after this many actions. */
export const ACTION_LIMIT = 8;

/** The axe-core tags gate 5 runs. */
export const AXE_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'];

/** `scenarios/<name>/truth.json`: the end state the screen must show, from the recording. */
export interface Truth {
  /** The recording the scenario replays, from the repository root; referenced, never copied. */
  recording: string;
  run_state: 'finished' | 'failed';
  steps: number;
  start_step: number;
  /** The step and loss of the last `StepCompleted`. */
  last_step: number;
  last_loss: number;
  /** The last value of each metric, in the order the names first appear. */
  metrics: { name: string; last_value: number }[];
  /** The parameter paths, in the order they first appear. */
  parameters: string[];
  /** The last `StepCompleted` that carried statistics, or `null` when none did. */
  last_statistics_step: number | null;
  /**
   * The gradient norm of the first parameter at the last statistics step that carries one: the
   * last step of a run makes no update and has no gradient. `null` without parameters.
   */
  first_parameter_gradient: { path: string; step: number; norm: number } | null;
}

/**
 * One task of the screen-only user. It finds an element by `role` and accessible `name` (a regular
 * expression), or, for a role that takes no name from its content (`paragraph`, `status`, `alert`),
 * by `role` and the text the accessibility tree exposes; reads `read`'s named groups from that
 * name or text; and compares each group with the truth key it names in `answer`.
 */
export interface Task {
  id: string;
  ask: string;
  role: 'figure' | 'paragraph' | 'status' | 'alert' | 'heading';
  name?: string;
  text?: string;
  read: string;
  answer: Record<string, string>;
  /** `true` when the answer changes with every step, so it is read at every replayed event. */
  live?: boolean;
}

export interface Tasks {
  tasks: Task[];
}

/** The truth of `events` computed from the recording itself, to check `truth.json` against. */
export function truthOf(recording: string, events: readonly RunEvent[]): Truth {
  const first = events[0];
  const last = events.at(-1);
  if (first?.body.kind !== 'RunStarted' || last === undefined) {
    throw new Error(`${recording}: not a recording that starts with RunStarted`);
  }
  let lastStep = -1;
  let lastLoss = Number.NaN;
  const metrics: { name: string; last_value: number }[] = [];
  const parameters: string[] = [];
  let lastStatistics: number | null = null;
  const gradients = new Map<string, { step: number; norm: number }>();
  for (const event of events) {
    if (event.body.kind !== 'StepCompleted') {
      continue;
    }
    const v = event.body.value;
    lastStep = v.step;
    lastLoss = v.loss ?? Number.NaN;
    for (const m of v.metrics) {
      const known = metrics.find((x) => x.name === m.name);
      if (known === undefined) {
        metrics.push({ name: m.name, last_value: m.value ?? Number.NaN });
      } else {
        known.last_value = m.value ?? Number.NaN;
      }
    }
    if (v.parameters.length > 0) {
      lastStatistics = v.step;
    }
    for (const p of v.parameters) {
      if (!parameters.includes(p.path)) {
        parameters.push(p.path);
      }
      if (p.subject === 'Gradient' && p.norm !== null) {
        gradients.set(p.path, { step: v.step, norm: p.norm });
      }
    }
  }
  const firstPath = parameters[0];
  const gradient = firstPath === undefined ? undefined : gradients.get(firstPath);
  return {
    recording,
    run_state: last.body.kind === 'RunFailed' ? 'failed' : 'finished',
    steps: first.body.value.steps,
    start_step: first.body.value.start_step,
    last_step: lastStep,
    last_loss: lastLoss,
    metrics,
    parameters,
    last_statistics_step: lastStatistics,
    first_parameter_gradient:
      firstPath === undefined || gradient === undefined ? null : { path: firstPath, ...gradient },
  };
}

/** The value `truth` holds at the dotted `key`, such as `first_parameter_gradient.norm`. */
export function truthValue(truth: Truth, key: string): unknown {
  let at: unknown = truth;
  for (const part of key.split('.')) {
    if (at === null || typeof at !== 'object') {
      return undefined;
    }
    at = (at as Record<string, unknown>)[part];
  }
  return at;
}

/** A scenario's files and the recording's events. */
export interface Loaded {
  scenario: Scenario;
  truth: Truth;
  tasks: Task[];
  events: RunEvent[];
}

export function loadScenario(scenario: Scenario): Loaded {
  const dir = `${UX_DIR}scenarios/${scenario.name}/`;
  const truth = JSON.parse(readFileSync(`${dir}truth.json`, 'utf8')) as Truth;
  const tasks = (JSON.parse(readFileSync(`${dir}tasks.json`, 'utf8')) as Tasks).tasks;
  const events = readFileSync(`${REPO_ROOT}${truth.recording}`, 'utf8')
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line) => JSON.parse(line) as RunEvent);
  return { scenario, truth, tasks, events };
}

/**
 * The selector that matches once the app has drawn `event`: the run state after `RunStarted`,
 * the loss chart's last step after a `StepCompleted`, the end state after `RunFinished` or
 * `RunFailed`; `null` before the first event.
 */
export function drawnSelector(event: RunEvent | null): string | null {
  if (event === null) {
    return null;
  }
  switch (event.body.kind) {
    case 'RunStarted':
      return 'main[data-run-state="streaming"]';
    case 'StepCompleted':
      return `[data-ux-region="loss-chart"] svg[data-metric="loss"][data-last-step="${event.body.value.step}"]`;
    case 'RunFinished':
      return 'main[data-run-state="finished"]';
    case 'RunFailed':
      return 'main[data-run-state="failed"]';
  }
}

/** The index in `events` of each moment; `connecting` is before the first event (-1). */
export function momentIndex(events: readonly RunEvent[]): Record<Moment, number> {
  const steps = events.flatMap((e, i) => (e.body.kind === 'StepCompleted' ? [i] : []));
  const firstStep = steps[0] ?? 1;
  const lastStep = steps.at(-1) ?? firstStep;
  return {
    connecting: -1,
    started: 0,
    'first-step': firstStep,
    'middle-step': steps[Math.floor((steps.length - 1) / 2)] ?? firstStep,
    'last-step': lastStep,
    end: events.length - 1,
  };
}

/** The index of the first event that carries parameter statistics, or `null`. */
export function firstStatistics(events: readonly RunEvent[]): number | null {
  const i = events.findIndex((e) => e.body.kind === 'StepCompleted' && e.body.value.parameters.length > 0);
  return i < 0 ? null : i;
}

/** The always-visible elements the app draws after `events[index]` has been drawn. */
export function alwaysVisibleAt(events: readonly RunEvent[], index: number): AlwaysVisible[] {
  const firstStep = events.findIndex((e) => e.body.kind === 'StepCompleted');
  const stats = firstStatistics(events);
  return ALWAYS_VISIBLE.filter(
    (a) =>
      a.when === 'always' ||
      (a.when === 'streamed' && firstStep >= 0 && index >= firstStep) ||
      (a.when === 'parameters' && stats !== null && index >= stats),
  );
}

/** `events` ending in `RunFailed` instead of their `RunFinished`: the failed end state. */
export function failedVariant(events: readonly RunEvent[]): RunEvent[] {
  const last = events.at(-1);
  if (last === undefined) {
    return [];
  }
  return [
    ...events.slice(0, -1),
    { ...last, body: { kind: 'RunFailed', value: { message: 'the loss is not finite after step 20' } } },
  ];
}
