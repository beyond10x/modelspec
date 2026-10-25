// The ratchet, the last step of `pnpm ux` (story:observatory-ui-iteration-loop; the
// ui-iteration-loop skill § 6). It aggregates the fragments the gating replays wrote into the
// metric keys per scenario, prints the table, writes the run's `round.json` under
// `node_modules/.cache/ux/`, and fails on a value worse than `baseline.json` beyond its tolerance,
// an improvement not taken into it, a missing value, a tolerance in it that disagrees with the
// code, or a baseline made worse than main's without a decision id; `origin/main` missing fails it
// too. `UX_UPDATE=1` takes improvements and missing values into the baseline, `UX_DECISION=<id>`
// also a worse value, recorded with the id. `UX_ROUND=R-nn` writes the round's files under
// `rounds/R-nn/` and its row in `rounds/trend.md`.

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import {
  CADENCE_MS,
  FONTS,
  GATING_VIEWPORTS,
  KEYS,
  LAYOUT_SHIFT_DEFINITION,
  MOMENTS,
  OUT_DIR,
  READING_LOAD_DEFINITION,
  RENDER_BUDGET_MS,
  REPO_ROOT,
  SCENARIOS,
  THEMES,
  TOLERANCES,
  UI_DIST,
  UX_DIR,
  viewportName,
} from './adapter.ts';
import { SHOTS } from './harness.ts';
import { commitInfo, mainBaseline, treeSha256 } from './core/git.ts';
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
  roundShotsToRemove,
  table,
  trend,
} from './core/round.ts';

const BASELINE = `${UX_DIR}baseline.json`;
const ABOUT =
  'The ratchet of the observatory UI loop (story:observatory-ui-iteration-loop): every key is lower-is-better; pnpm ux fails on a value worse than here beyond its tolerance, an improvement not taken in, a missing value, a tolerance that disagrees with the code, or a value made worse than main has without a decision id.';

/** What a reader of the round must know first, from the round's own numbers where it can. */
function caveats(names: readonly string[], measured: Measured, main: Baseline | null): string[] {
  const paged = names.filter((s) => (measured[s]?.['actions_per_fix.max'] ?? 0) > 0);
  const flat = names.filter((s) => (measured[s]?.['layout_shift.per_100_events'] ?? 0) < (TOLERANCES.layout_shift.absolute ?? 0));
  const list = [
    `Hermetic and synthetic: no people, no questionnaire, no L4, L5 or L6. Time is counted in events, because the recordings carry no timestamps: the layout replays send one event at a time, the live replay of the screen-only user one every ${CADENCE_MS} ms.`,
    `Gate 1 allows the page's own vertical scroll (the story's acceptance); skill § 4 allows it only for docs sites. Under the skill's gate the scenarios whose tasks need a key press (actions_per_fix.max above 0: ${paged.length === 0 ? 'none' : paged.join(', ')}) would fail at both gating viewports, because that key press scrolls the page.`,
    `layout_shift.per_100_events divides a replay's shift sum by its event count, so the same shift weighs less the longer the run: the values are not comparable across scenarios, and where the value is below the tolerance ${TOLERANCES.layout_shift.absolute} (${flat.length === 0 ? 'none' : flat.join(', ')}) an improvement to 0 is not reported and a regression of up to ${TOLERANCES.layout_shift.absolute} passes.`,
    'screen_only.steers carries no information: the app has no guidance cue, and a steer (Home from the bottom) fires only when an answer is missing from the page.',
    'reading_load gets worse when UXF-001 is fixed, because more of the parameter panels come into view; that fix will need a decision id for the ratchet.',
    'The screen-only user runs in the light theme only, and the failed end state is replayed on the rotating card only (round.json failed_end).',
    `Every page runs under the injected font stack "${FONTS.text}" and "${FONTS.code}", so that this machine and the CI runner measure the same words; a reader with the app's own system font sees other line breaks.`,
  ];
  if (main === null) {
    list.push('The rule that a baseline is never made worse than main\'s did not run: origin/main carries no ui/ux/baseline.json (the first landing).');
  }
  return list;
}

test('ratchet', async ({ page }) => {
  const names = SCENARIOS.map((s) => s.name);
  const keys = Object.keys(KEYS) as MetricKey[];
  const fragments = readFragments(OUT_DIR);
  const measured: Measured = aggregate(names, fragments);
  const empty: Baseline = { about: ABOUT, scenarios: {}, tolerances: TOLERANCES };
  const current: Baseline = existsSync(BASELINE) ? (JSON.parse(readFileSync(BASELINE, 'utf8')) as Baseline) : empty;
  let baseline = current;
  if (process.env.UX_UPDATE === '1') {
    baseline = update(names, KEYS, TOLERANCES, current, measured, process.env.UX_DECISION ?? null);
    baseline.about = ABOUT;
    baseline.tolerances = TOLERANCES;
    writeFileSync(BASELINE, `${JSON.stringify(baseline, null, 2)}\n`);
  }
  const main = mainBaseline(REPO_ROOT, 'ui/ux/baseline.json') as Baseline | null;
  const problems = ratchet(names, KEYS, TOLERANCES, baseline, measured, main);
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
    commit: { ...commitInfo(REPO_ROOT, ['ui/ux/rounds/']), ui_dist_sha256: treeSha256(`${REPO_ROOT}${UI_DIST}`) },
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
      'gate 3, unmissable completion: completion_miss = 0 and false_complete = 0 (the failed end state is checked on the rotating card ending in RunFailed, recorded under failed_end)',
      `gate 4, screen-only user: every task read equal to the truth, each in at most 2 actions, and every live reading equal to the truth at the step it shows`,
      'gate 5, accessibility: axe-core 0 serious or critical violations (wcag2a, wcag2aa, wcag21a, wcag21aa, wcag22aa) at every key moment in both themes; every focusable reached by Tab with a focus indicator; no animation under reduced motion after any event',
      'gate 6, cue truth: not applicable, the app has no guidance cue',
    ],
    notRun: [
      'L0 engine scenarios and gate 6: the app has no guidance cue',
      'L4 keyed replay, L5 discoverability probe: paid, operator-only',
      'L6 recorded live rounds, L8 real-user sessions, the questionnaire: no people in this round',
      'L7 heuristic and report review agents: started by the coordinator after this round command',
    ],
    definitions: [
      `reading_load: ${READING_LOAD_DEFINITION} (${MOMENTS.join(', ')}), over both gating viewports and themes`,
      `layout_shift.per_100_events: ${LAYOUT_SHIFT_DEFINITION}`,
      'actions_per_fix: the key presses the screen-only user needs for one task from the top of the page at the end state (a fix: bringing an answer into view); every task at both gating viewports',
      'glance_cost: the glance level of every task the page can answer at every key moment, both gating viewports: 1 in the loss chart (the primary card), 2 elsewhere in view, 3 not in view; the end-state task is answerable only once the state reads finished or failed',
      'screen_only.actions, looks, steers: the sums over one viewport\'s tasks (key presses; screens read, waits for the end state included; reversals of course, Home from the bottom), the worse viewport',
      `screen_only.time_to_act_steps.max: with the stream running at one event every ${CADENCE_MS} ms while the user reads, the steps sent at least ${RENDER_BUDGET_MS} ms before a reading of a live task began minus the steps up to the one the screen shows; the most over every reading, task and gating viewport`,
      'the screen-only user reads a value only when its own text is in view, uncovered and not transparent',
    ],
    caveats: caveats(names, measured, main),
  };
  writeFileSync(`${OUT_DIR}round.json`, `${JSON.stringify(roundJson(input), null, 2)}\n`);

  if (process.env.UX_ROUND !== undefined) {
    const dir = `${UX_DIR}rounds/${roundName}/`;
    mkdirSync(`${dir}shots`, { recursive: true });
    // A rerun replaces only the shots it generates; review-* and anything else stay.
    for (const file of roundShotsToRemove(readdirSync(`${dir}shots`), names)) {
      rmSync(`${dir}shots/${file}`);
    }
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
      const mine = layout.filter((f) => f.scenario === name && f.gating && !f.failedEnd);
      const load = mine
        .filter((f) => f.viewport === first)
        .flatMap((f) => f.readingLoad.map((r) => ({ ...r, theme: f.theme })))
        .sort((a, b) => b.words - a.words)[0];
      const loadShot = copy(load?.shot ?? null);
      if (load !== undefined && loadShot !== null) {
        worst.push({ scenario: name, what: `the highest reading load, ${load.words} words in view at ${load.moment} (${first} ${load.theme})`, shot: loadShot });
      }
      // The largest single shift over every gating replay of the scenario, each of which keeps a shot of its own.
      const shifted = mine
        .filter((f) => f.worstShift !== null)
        .sort((a, b) => (b.worstShift?.value ?? 0) - (a.worstShift?.value ?? 0))[0];
      const shift = shifted?.worstShift;
      const shiftShot = copy(shift?.shot ?? null);
      if (shifted !== undefined && shift && shiftShot !== null) {
        worst.push({
          scenario: name,
          what: `the largest layout shift, ${shift.value.toFixed(5)} at ${shift.event} (${shifted.viewport} ${shifted.theme}; ${shift.sources.join(', ')})`,
          shot: shiftShot,
        });
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
    const trendFile = `${UX_DIR}rounds/trend.md`;
    writeFileSync(trendFile, trend(existsSync(trendFile) ? readFileSync(trendFile, 'utf8') : null, input));
  }

  expect(problems.map((p) => p.message), `ratchet:\n${problems.map((p) => p.message).join('\n')}`).toEqual([]);
});
