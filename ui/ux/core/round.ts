// What one run of the specs leaves behind, and the round files made from it: fragments per
// replay, the aggregated metrics per scenario, `round.json`, `report.md` and the trend row.

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { type Baseline, type Measured, type MetricKey, type Problem, max, median, per100Events, percentile } from './metrics.ts';

/** A gating or measured-only replay of the layout contract. */
export interface LayoutFragment {
  kind: 'layout';
  scenario: string;
  viewport: string;
  theme: string;
  gating: boolean;
  /** A gating replay of the failed end state (a run ending in `RunFailed`): gated, never ratcheted. */
  failedEnd: boolean;
  events: number;
  shift: number;
  worstShift: { index: number; event: string; value: number; sources: string[]; shot: string | null } | null;
  readingLoad: { moment: string; words: number; shot: string | null }[];
  violations: string[];
  completionMiss: number;
  falseComplete: number;
  axe: { moment: string; serious: number; ids: string[] }[];
  shots: { moment: string; file: string }[];
}

/** One task of the screen-only user at the end state. */
export interface AttemptRecord {
  task: string;
  ask: string;
  actions: number;
  looks: number;
  steers: number;
  glance: number;
  keys: string[];
  read: string | null;
  expected: Record<string, unknown>;
  correct: boolean;
  shot: string | null;
}

/** A screen-only replay. */
export interface ScreenFragment {
  kind: 'screen';
  scenario: string;
  viewport: string;
  theme: string;
  attempts: AttemptRecord[];
  glances: { moment: string; task: string; glance: number }[];
  timeToAct: Record<string, number>;
}

export type Fragment = LayoutFragment | ScreenFragment;

export function writeFragment(dir: string, fragment: Fragment): void {
  mkdirSync(`${dir}fragments`, { recursive: true });
  const name = `${fragment.kind}-${fragment.scenario}-${fragment.viewport}-${fragment.theme}.json`;
  writeFileSync(`${dir}fragments/${name}`, `${JSON.stringify(fragment, null, 2)}\n`);
}

export function readFragments(dir: string): Fragment[] {
  let names: string[];
  try {
    names = readdirSync(`${dir}fragments`);
  } catch {
    return [];
  }
  return names
    .filter((n) => n.endsWith('.json'))
    .sort()
    .map((n) => JSON.parse(readFileSync(`${dir}fragments/${n}`, 'utf8')) as Fragment);
}

/**
 * The ratchet keys of each scenario from the gating fragments: layout shift per 100 events and reading
 * load from the layout replays, the rest from the screen-only replays; the worst over viewports
 * and themes.
 */
export function aggregate(scenarios: readonly string[], fragments: readonly Fragment[]): Measured {
  const measured: Measured = {};
  for (const scenario of scenarios) {
    const layout = fragments.filter((f): f is LayoutFragment => f.kind === 'layout' && f.gating && !f.failedEnd && f.scenario === scenario);
    const screen = fragments.filter((f): f is ScreenFragment => f.kind === 'screen' && f.scenario === scenario);
    const m: Partial<Record<MetricKey, number>> = {};
    if (layout.length > 0) {
      m['layout_shift.per_100_events'] = max(layout.map((f) => per100Events(f.shift, f.events)));
      const loads = layout.flatMap((f) => f.readingLoad.map((r) => r.words));
      m['reading_load.max'] = max(loads);
      m['reading_load.median'] = median(loads);
    }
    if (screen.length > 0) {
      const actions = screen.flatMap((f) => f.attempts.map((a) => a.actions));
      m['actions_per_fix.max'] = max(actions);
      m['actions_per_fix.median'] = median(actions);
      const glances = screen.flatMap((f) => f.glances.map((g) => g.glance));
      m['glance_cost.median'] = median(glances);
      m['glance_cost.p95'] = percentile(glances, 95);
      const sum = (f: ScreenFragment, k: 'actions' | 'looks' | 'steers') => f.attempts.reduce((n, a) => n + a[k], 0);
      m['screen_only.actions'] = max(screen.map((f) => sum(f, 'actions')));
      m['screen_only.looks'] = max(screen.map((f) => sum(f, 'looks')));
      m['screen_only.steers'] = max(screen.map((f) => sum(f, 'steers')));
      m['screen_only.time_to_act_steps.max'] = max(screen.flatMap((f) => Object.values(f.timeToAct)));
    }
    measured[scenario] = m;
  }
  return measured;
}

/** The metric table: one row per key, one column per scenario, `measured (baseline)`. */
export function table(scenarios: readonly string[], keys: readonly MetricKey[], measured: Measured, baseline: Baseline | null): string {
  const head = `| key | ${scenarios.join(' | ')} |`;
  const rule = `|---|${scenarios.map(() => '---:').join('|')}|`;
  const rows = keys.map((key) => {
    const cells = scenarios.map((s) => {
      const m = measured[s]?.[key];
      const b = baseline?.scenarios[s]?.[key]?.value;
      return `${m ?? 'missing'}${b === undefined ? '' : ` (${b})`}`;
    });
    return `| ${key} | ${cells.join(' | ')} |`;
  });
  return [head, rule, ...rows].join('\n');
}

/** The code a round measured: HEAD, whether the tree differed from it, and the served app's hash. */
export interface RoundCommit {
  head: string;
  dirty: boolean;
  ui_dist_sha256: string;
}

export interface RoundInput {
  round: string;
  date: string;
  commit: RoundCommit;
  scenarios: { name: string; card: string }[];
  keys: readonly MetricKey[];
  measured: Measured;
  baseline: Baseline | null;
  previous: Measured | null;
  problems: Problem[];
  fragments: readonly Fragment[];
  bars: string[];
  notRun: string[];
  findings: Record<string, string>[];
  definitions: string[];
  /** What a reader of the round must know before using its numbers; first in `report.md`. */
  caveats: string[];
}

/** The gate counts of `layout` replays. */
function gateCounts(layout: readonly LayoutFragment[]) {
  return {
    replays: layout.length,
    violations: layout.reduce((n, f) => n + f.violations.length, 0),
    completion_miss: layout.reduce((n, f) => n + f.completionMiss, 0),
    false_complete: layout.reduce((n, f) => n + f.falseComplete, 0),
    axe_serious: layout.reduce((n, f) => n + f.axe.reduce((k, a) => k + a.serious, 0), 0),
  };
}

/** `round.json`: the numbers of the round, machine-readable. */
export function roundJson(input: RoundInput): unknown {
  const layout = input.fragments.filter((f): f is LayoutFragment => f.kind === 'layout');
  const screen = input.fragments.filter((f): f is ScreenFragment => f.kind === 'screen');
  return {
    round: input.round,
    date: input.date,
    commit: input.commit,
    scenarios: Object.fromEntries(input.scenarios.map((s) => [s.name, { card: s.card, metrics: input.measured[s.name] ?? {} }])),
    gates: {
      ...gateCounts(layout.filter((f) => f.gating && !f.failedEnd)),
      screen_only_wrong: screen.reduce((n, f) => n + f.attempts.filter((a) => !a.correct).length, 0),
    },
    failed_end: {
      scenarios: [...new Set(layout.filter((f) => f.failedEnd).map((f) => f.scenario))],
      gates: gateCounts(layout.filter((f) => f.failedEnd)),
    },
    measured_only: layout
      .filter((f) => !f.gating)
      .map((f) => ({ scenario: f.scenario, viewport: f.viewport, theme: f.theme, violations: f.violations, reading_load: max(f.readingLoad.map((r) => r.words)) })),
    ratchet: input.problems,
    not_run: input.notRun,
  };
}

/** `report.md`: the metric table against the bars and the baseline, worst moments, misses. */
export function report(input: RoundInput, worst: { scenario: string; what: string; shot: string }[], contact: { scenario: string; shot: string }[]): string {
  const layout = input.fragments.filter((f): f is LayoutFragment => f.kind === 'layout');
  const screen = input.fragments.filter((f): f is ScreenFragment => f.kind === 'screen');
  const names = input.scenarios.map((s) => s.name);
  const lines: string[] = [];
  lines.push(`# ${input.round}: observatory UI round`, '');
  const c = input.commit;
  lines.push(
    `Date ${input.date}, HEAD \`${c.head}\`${c.dirty ? ' with uncommitted changes outside `ui/ux/rounds/` (dirty)' : ' (clean)'}, \`ui/dist\` SHA-256 \`${c.ui_dist_sha256}\`. Scenarios: ${input.scenarios.map((s) => `\`${s.name}\` (${s.card})`).join(', ')}.`,
    '',
  );
  if (input.caveats.length > 0) {
    lines.push('## Caveats', '', ...input.caveats.map((v, i) => `${i + 1}. ${v}`), '');
  }
  lines.push('## Metrics', '', 'Measured, with the baseline in parentheses. Lower is better.', '');
  lines.push(table(names, input.keys, input.measured, input.baseline), '');
  if (input.previous !== null) {
    lines.push('Last round:', '', table(names, input.keys, input.previous, null), '');
  } else {
    lines.push('Last round: none; this is the first.', '');
  }
  lines.push('## Bars (hard gates)', '', ...input.bars.map((b) => `- ${b}`), '');
  const gating = layout.filter((f) => f.gating);
  lines.push('| replay | violations | completion_miss | false_complete | axe serious+critical |', '|---|---:|---:|---:|---:|');
  for (const f of gating) {
    const name = `${f.scenario} ${f.viewport} ${f.theme}${f.failedEnd ? ' (failed end, not ratcheted)' : ''}`;
    lines.push(`| ${name} | ${f.violations.length} | ${f.completionMiss} | ${f.falseComplete} | ${f.axe.reduce((n, a) => n + a.serious, 0)} |`);
  }
  lines.push('');
  lines.push('## Screen-only user', '', '| scenario | viewport | task | read | correct | actions | looks | steers | glance |', '|---|---|---|---|---|---:|---:|---:|---:|');
  for (const f of screen) {
    for (const a of f.attempts) {
      lines.push(`| ${f.scenario} | ${f.viewport} | ${a.task} | ${a.read === null ? 'nothing' : `\`${a.read}\``} | ${a.correct ? 'yes' : 'no'} | ${a.actions} | ${a.looks} | ${a.steers} | ${a.glance} |`);
    }
  }
  lines.push('');
  lines.push('## Worst moments', '');
  for (const w of worst) {
    lines.push(`- ${w.scenario}: ${w.what}. ![${w.scenario} ${w.what}](${w.shot})`);
  }
  lines.push('', '## Contact sheets', '', 'Every key moment in both themes, each frame 0.75 s after its moment, at the first gating viewport.', '');
  for (const c of contact) {
    lines.push(`- ${c.scenario}: ![${c.scenario} contact sheet](${c.shot})`);
  }
  lines.push('', '## Measured-only viewports', '', '| scenario | viewport | theme | layout findings | reading_load.max |', '|---|---|---|---|---:|');
  for (const f of layout.filter((x) => !x.gating)) {
    lines.push(`| ${f.scenario} | ${f.viewport} | ${f.theme} | ${f.violations.length === 0 ? 'none' : f.violations.slice(0, 3).join('; ').replace(/\|/g, '/')} | ${max(f.readingLoad.map((r) => r.words))} |`);
  }
  lines.push('', '## Findings', '', '| id | title | class | severity | status | regression asset |', '|---|---|---|---|---|---|');
  for (const f of input.findings) {
    lines.push(`| ${f.id} | ${f.title} | ${f.class} | ${f.severity} | ${f.status} | ${(f.regression_asset ?? '').replace(/\|/g, '/')} |`);
  }
  lines.push('', '## Ratchet', '');
  lines.push(input.problems.length === 0 ? 'No value worse than, better than or missing from the baseline.' : input.problems.map((p) => `- ${p.message}`).join('\n'));
  lines.push('', '## Definitions', '', ...input.definitions.map((d) => `- ${d}`));
  lines.push('', '## Not run', '', ...input.notRun.map((n) => `- ${n}`), '');
  return lines.join('\n');
}

/** The `key: value` header lines of a finding record, up to its first `## ` section. */
export function findingHeader(text: string): Record<string, string> {
  const fields: Record<string, string> = {};
  for (const line of text.split('\n')) {
    if (line.startsWith('## ')) {
      break;
    }
    const found = /^([a-z_]+): (.*)$/.exec(line);
    if (found?.[1] !== undefined && found[2] !== undefined) {
      fields[found[1]] = found[2].trim();
    }
  }
  return fields;
}

/** The headers of every `UXF-nnn.md` in `dir`, in id order. */
export function readFindings(dir: string): Record<string, string>[] {
  let names: string[];
  try {
    names = readdirSync(dir);
  } catch {
    return [];
  }
  return names
    .filter((n) => /^UXF-\d{3}\.md$/.test(n))
    .sort()
    .map((n) => findingHeader(readFileSync(`${dir}/${n}`, 'utf8')));
}

/** The keys `rounds/trend.md` follows, one column per scenario and key. */
export const TREND_KEYS = [
  'actions_per_fix.max',
  'glance_cost.p95',
  'layout_shift.per_100_events',
  'reading_load.max',
  'screen_only.actions',
] as const satisfies readonly MetricKey[];

/**
 * `rounds/trend.md` with the row of `input`: its columns are named per scenario, the header is the
 * union of the scenarios of `existing` and of `input` (a rotating card adds its own columns), a
 * row without a value for a column holds `-`, and a rerun of a round replaces its row.
 */
export function trend(existing: string | null, input: RoundInput): string {
  const cells = (line: string) => line.split('|').slice(1, -1).map((c) => c.trim());
  const table = (existing ?? '').split('\n').filter((l) => l.startsWith('|'));
  const head = table[0] === undefined ? [] : cells(table[0]);
  const scenarios: string[] = [];
  for (const h of head.slice(2)) {
    const name = h.split(' ')[0];
    if (name !== undefined && !scenarios.includes(name)) {
      scenarios.push(name);
    }
  }
  for (const s of input.scenarios) {
    if (!scenarios.includes(s.name)) {
      scenarios.push(s.name);
    }
  }
  const rows = new Map<string, Map<string, string>>();
  for (const line of table.slice(2)) {
    const values = cells(line);
    const round = values[0];
    if (round !== undefined && round !== input.round) {
      rows.set(round, new Map(head.map((h, i) => [h, values[i] ?? '-'])));
    }
  }
  const mine = new Map<string, string>([
    ['round', input.round],
    ['date', input.date],
  ]);
  for (const s of input.scenarios) {
    for (const k of TREND_KEYS) {
      mine.set(`${s.name} ${k}`, String(input.measured[s.name]?.[k] ?? 'missing'));
    }
  }
  rows.set(input.round, mine);
  const columns = ['round', 'date', ...scenarios.flatMap((s) => TREND_KEYS.map((k) => `${s} ${k}`))];
  const out = [
    `| ${columns.join(' | ')} |`,
    `|---|---|${columns.slice(2).map(() => '---:').join('|')}|`,
    ...[...rows.keys()].sort().map((r) => `| ${columns.map((c) => rows.get(r)?.get(c) ?? '-').join(' | ')} |`),
  ];
  return `# Trend\n\nOne row per round; lower is better. Columns are named per scenario; \`-\` is a scenario the round did not run.\n\n${out.join('\n')}\n`;
}

/**
 * The files of a round's `shots/` that a rerun of the round removes: the ones it generates, each
 * named after a scenario (`<scenario>-...jpg`). Anything else, `review-*` above all, is kept.
 */
export function roundShotsToRemove(files: readonly string[], scenarios: readonly string[]): string[] {
  return files.filter((f) => f.endsWith('.jpg') && !f.startsWith('review-') && scenarios.some((s) => f.startsWith(`${s}-`)));
}

/** The HTML of a contact sheet: one row per theme, one cell per moment, from data URLs. */
export function contactSheet(title: string, rows: { label: string; cells: { label: string; src: string }[] }[]): string {
  const cell = (c: { label: string; src: string }) =>
    `<figure><img src="${c.src}"><figcaption>${c.label}</figcaption></figure>`;
  return `<!doctype html><html><head><style>
body{margin:0;padding:12px;background:#fff;color:#111;font:13px system-ui,sans-serif}
h1{font-size:15px;margin:0 0 8px}.row{display:flex;gap:8px;margin-bottom:8px;align-items:flex-start}
.label{width:48px;font-weight:600}figure{margin:0;width:300px}img{width:300px;border:1px solid #999;display:block}
figcaption{padding:2px 0}
</style></head><body><h1>${title}</h1>${rows
    .map((r) => `<div class="row"><div class="label">${r.label}</div>${r.cells.map(cell).join('')}</div>`)
    .join('')}</body></html>`;
}
