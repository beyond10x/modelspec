// The ratchet, the last step of `pnpm ux` (story:observatory-ui-iteration-loop; the
// ui-iteration-loop skill § 6). It aggregates the fragments the gating replays wrote into the
// metric keys per scenario, prints the table, writes the run's `round.json` under
// `node_modules/.cache/ux/`, and fails on a value worse than `baseline.json` beyond its tolerance,
// an improvement not taken into it, a missing value, or a baseline made worse than main's without a
// decision id. `UX_UPDATE=1` takes improvements and missing values into the baseline,
// `UX_DECISION=<id>` also a worse value, recorded with the id. `UX_ROUND=R-nn` writes the round's
// files under `rounds/R-nn/` and its row in `rounds/trend.md`.

import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import {
  EVENTS_PER_MINUTE,
  GATING_VIEWPORTS,
  KEYS,
  MOMENTS,
  OUT_DIR,
  READING_LOAD_DEFINITION,
  REPO_ROOT,
  SCENARIOS,
  THEMES,
  TOLERANCES,
  UX_DIR,
  viewportName,
} from './adapter.ts';
import { SHOTS } from './harness.ts';
import { type Baseline, type Measured, type MetricKey, ratchet, update } from './core/metrics.ts';
import {
  type LayoutFragment,
  type RoundInput,
  type ScreenFragment,
  aggregate,
  contactSheet,
  readFindings,
  readFragments,
  report,
  roundJson,
  table,
  trendRow,
} from './core/round.ts';

const BASELINE = `${UX_DIR}baseline.json`;
const ABOUT =
  'The ratchet of the observatory UI loop (story:observatory-ui-iteration-loop): every key is lower-is-better; pnpm ux fails on a value worse than here beyond its tolerance, an improvement not taken in, a missing value, or a value made worse than main has without a decision id.';

function mainBaseline(): Baseline | null {
  try {
    const text = execFileSync('git', ['show', 'origin/main:ui/ux/baseline.json'], {
      cwd: REPO_ROOT,
      stdio: ['ignore', 'pipe', 'ignore'],
    }).toString();
    return JSON.parse(text) as Baseline;
  } catch {
    return null;
  }
}

function commit(): string {
  try {
    return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: REPO_ROOT }).toString().trim();
  } catch {
    return 'unknown';
  }
}

test('ratchet', async ({ page }) => {
  const names = SCENARIOS.map((s) => s.name);
  const keys = Object.keys(KEYS) as MetricKey[];
  const fragments = readFragments(OUT_DIR);
  const measured: Measured = aggregate(names, fragments, EVENTS_PER_MINUTE);
  const empty: Baseline = { about: ABOUT, scenarios: {}, tolerances: TOLERANCES };
  const current: Baseline = existsSync(BASELINE) ? (JSON.parse(readFileSync(BASELINE, 'utf8')) as Baseline) : empty;
  let baseline = current;
  if (process.env.UX_UPDATE === '1') {
    baseline = update(names, KEYS, current, measured, process.env.UX_DECISION ?? null);
    baseline.about = ABOUT;
    baseline.tolerances = TOLERANCES;
    writeFileSync(BASELINE, `${JSON.stringify(baseline, null, 2)}\n`);
  }
  const main = mainBaseline();
  const problems = ratchet(names, KEYS, baseline, measured, main);
  const printed = table(names, keys, measured, baseline);
  console.log(`\nux metrics, measured (baseline):\n${printed}\n`);
  mkdirSync(OUT_DIR, { recursive: true });

  const roundName = process.env.UX_ROUND ?? 'run';
  const date = new Date().toISOString().slice(0, 10);
  const previousFile = (() => {
    const dir = `${UX_DIR}rounds/`;
    if (!existsSync(dir)) {
      return null;
    }
    const earlier = readdirSync(dir)
      .filter((d) => /^R-\d+$/.test(d) && d < roundName)
      .sort();
    const last = earlier.at(-1);
    return last === undefined ? null : `${dir}${last}/round.json`;
  })();
  const previous: Measured | null =
    previousFile !== null && existsSync(previousFile)
      ? Object.fromEntries(
          Object.entries(
            (JSON.parse(readFileSync(previousFile, 'utf8')) as { scenarios: Record<string, { metrics: Measured[string] }> }).scenarios,
          ).map(([k, v]) => [k, v.metrics]),
        )
      : null;
  const input: RoundInput = {
    round: roundName,
    date,
    commit: commit(),
    scenarios: SCENARIOS.map((s) => ({ name: s.name, card: s.card })),
    keys,
    measured,
    baseline,
    previous,
    problems,
    fragments,
    findings: readFindings(`${UX_DIR}findings`),
    bars: [
      `gate 1, no scroll: no scrollable element and no horizontal page scroll at ${GATING_VIEWPORTS.map(viewportName).join(' and ')} in ${THEMES.join(' and ')} after every replayed event (the page's own vertical scroll is allowed)`,
      'gate 2, always visible: the run header and its state, the loss chart, and with parameters the histogram step picker and the first parameter name wholly in view after every event',
      'gate 3, unmissable completion: completion_miss = 0 and false_complete = 0 (the failed end state is checked on the rotating card ending in RunFailed)',
      `gate 4, screen-only user: every task read equal to the truth, each in at most 2 actions`,
      'gate 5, accessibility: axe-core 0 serious or critical violations (wcag2a, wcag2aa, wcag21a, wcag21aa, wcag22aa) at every key moment in both themes; every focusable reached by Tab with a focus indicator; no animation under reduced motion',
      'gate 6, cue truth: not applicable, the app has no guidance cue',
    ],
    notRun: [
      'L0 engine scenarios and gate 6: the app has no guidance cue',
      'L4 keyed replay, L5 discoverability probe: paid, operator-only',
      'L6 recorded live rounds, L8 real-user sessions: no people in R-00',
      'L7 heuristic and report review agents: started by the coordinator after this round command',
    ],
    definitions: [
      `reading_load: ${READING_LOAD_DEFINITION} (${MOMENTS.join(', ')}), over both gating viewports and themes`,
      `layout_shift.per_minute: the Layout Instability API's shift sum over a replay, per minute of the compressed clock (${EVENTS_PER_MINUTE} events a minute); the worst replay`,
      'actions_per_fix: the key presses the screen-only user needs for one task from the top of the page at the end state (a fix: bringing an answer into view); every task at both gating viewports',
      'glance_cost: the glance level of every task the page can answer at every key moment, both gating viewports: 1 in the loss chart (the primary card), 2 elsewhere in view, 3 not in view',
      'screen_only.actions, looks, steers: the sums over one viewport\'s tasks (key presses; screens read; reversals of course, Home from the bottom), the worse viewport',
      'screen_only.time_to_act_steps.max: the events the screen lagged behind the stream before a live task read the current value',
    ],
  };
  writeFileSync(`${OUT_DIR}round.json`, `${JSON.stringify(roundJson(input), null, 2)}\n`);

  if (process.env.UX_ROUND !== undefined) {
    const dir = `${UX_DIR}rounds/${roundName}/`;
    rmSync(`${dir}shots`, { recursive: true, force: true });
    mkdirSync(`${dir}shots`, { recursive: true });
    const layout = fragments.filter((f): f is LayoutFragment => f.kind === 'layout');
    const screen = fragments.filter((f): f is ScreenFragment => f.kind === 'screen');
    const copy = (file: string | null) => {
      if (file === null || !existsSync(`${SHOTS}${file}`)) {
        return null;
      }
      copyFileSync(`${SHOTS}${file}`, `${dir}shots/${file}`);
      return `shots/${file}`;
    };
    const worst: { scenario: string; what: string; shot: string }[] = [];
    const contact: { scenario: string; shot: string }[] = [];
    const first = viewportName(GATING_VIEWPORTS[0] ?? { width: 0, height: 0 });
    for (const name of names) {
      const mine = layout.filter((f) => f.scenario === name && f.gating);
      const load = mine
        .filter((f) => f.viewport === first)
        .flatMap((f) => f.readingLoad.map((r) => ({ ...r, theme: f.theme })))
        .sort((a, b) => b.words - a.words)[0];
      const loadShot = copy(load?.shot ?? null);
      if (load !== undefined && loadShot !== null) {
        worst.push({ scenario: name, what: `the highest reading load, ${load.words} words in view at ${load.moment} (${first} ${load.theme})`, shot: loadShot });
      }
      const shift = mine
        .filter((f) => f.worstShift !== null && f.worstShift.shot !== null)
        .sort((a, b) => (b.worstShift?.value ?? 0) - (a.worstShift?.value ?? 0))[0]?.worstShift;
      const shiftShot = copy(shift?.shot ?? null);
      if (shift && shiftShot !== null) {
        worst.push({ scenario: name, what: `the largest layout shift, ${shift.value.toFixed(5)} at ${shift.event} (${shift.sources.join(', ')})`, shot: shiftShot });
      }
      const costly = screen
        .filter((f) => f.scenario === name)
        .flatMap((f) => f.attempts.filter((a) => a.shot !== null))
        .sort((a, b) => b.actions - a.actions)[0];
      const costlyShot = copy(costly?.shot ?? null);
      if (costly && costly.actions > 0 && costlyShot !== null) {
        worst.push({ scenario: name, what: `the costliest task, ${costly.task}: ${costly.actions} actions (${costly.keys.join(', ')}), glance ${costly.glance} at the start`, shot: costlyShot });
      }
      const rows = THEMES.map((theme) => ({
        label: theme,
        cells: MOMENTS.flatMap((moment) => {
          const file = `${name}-${first}-${theme}-${moment}.jpg`;
          if (!existsSync(`${SHOTS}${file}`)) {
            return [];
          }
          return [{ label: moment, src: `data:image/jpeg;base64,${readFileSync(`${SHOTS}${file}`).toString('base64')}` }];
        }),
      }));
      await page.setViewportSize({ width: 1920, height: 400 });
      await page.setContent(contactSheet(`${name}, ${first}, every key moment + 0.75 s`, rows));
      const sheet = `${name}-contact.jpg`;
      await page.screenshot({ path: `${dir}shots/${sheet}`, type: 'jpeg', quality: 75, fullPage: true });
      contact.push({ scenario: name, shot: `shots/${sheet}` });
    }
    writeFileSync(`${dir}round.json`, `${JSON.stringify(roundJson(input), null, 2)}\n`);
    writeFileSync(`${dir}report.md`, report(input, worst, contact));
    const trend = `${UX_DIR}rounds/trend.md`;
    if (existsSync(trend)) {
      const kept = readFileSync(trend, 'utf8')
        .split('\n')
        .filter((l) => !l.startsWith(`| ${roundName} |`))
        .join('\n')
        .trimEnd();
      writeFileSync(trend, `${kept}\n${trendRow(input, false)}`);
    } else {
      writeFileSync(trend, `# Trend\n\nOne row per round; lower is better.\n\n${trendRow(input, true)}`);
    }
  }

  expect(problems.map((p) => p.message), `ratchet:\n${problems.map((p) => p.message).join('\n')}`).toEqual([]);
});
