// The ratchet of `core/metrics.ts` driven from the committed `baseline.json` and the skill's rule
// (§ 6) that a baseline is never made worse without a decision id; the main branch's baseline
// read from git; the round files a rerun replaces. The cases named ADV1-nn are the adversary's
// (story:observatory-ui-iteration-loop R-00, adversary pass 1), each red before its fix.

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { test } from 'node:test';
import { AXE_TAGS, KEYS, MAX_ACTIONS_PER_FIX, OUT_DIR, SCENARIOS, TOLERANCES, UX_DIR } from './adapter.ts';
import { mainBaseline, treeSha256 } from './core/git.ts';
import { type Baseline, type Measured, type MetricKey, isDecision, ratchet, update } from './core/metrics.ts';
import { type RoundInput, roundShotsToRemove, trend } from './core/round.ts';

const NAMES = SCENARIOS.map((s) => s.name);

function committed(): Baseline {
  return JSON.parse(readFileSync(`${UX_DIR}baseline.json`, 'utf8')) as Baseline;
}

function measuredAsBaseline(baseline: Baseline): Measured {
  return Object.fromEntries(
    Object.entries(baseline.scenarios).map(([s, entries]) => [
      s,
      Object.fromEntries(Object.entries(entries).map(([k, e]) => [k, e?.value ?? Number.NaN])),
    ]),
  );
}

function one(scenario: string, key: MetricKey, value: number): Measured {
  const m = measuredAsBaseline(committed());
  (m[scenario] as Record<string, number>)[key] = value;
  return m;
}

function kinds(problems: ReturnType<typeof ratchet>): string[] {
  return problems.map((p) => `${p.scenario} ${p.key} ${p.kind}`);
}

test('the committed baseline measured unchanged passes the ratchet, with and without main', () => {
  const b = committed();
  assert.deepEqual(ratchet(NAMES, KEYS, TOLERANCES, b, measuredAsBaseline(b), null), []);
  assert.deepEqual(ratchet(NAMES, KEYS, TOLERANCES, b, measuredAsBaseline(b), committed()), []);
});

test('counts: equal passes, one more is worse, one fewer is an improvement not taken', () => {
  const b = committed();
  assert.deepEqual(kinds(ratchet(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'reading_load.max', 25), null)), []);
  assert.deepEqual(kinds(ratchet(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'reading_load.max', 26), null)), [
    'accuracy reading_load.max worse',
  ]);
  assert.deepEqual(kinds(ratchet(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'reading_load.max', 24), null)), [
    'accuracy reading_load.max improved',
  ]);
});

test('steps 5 %: the band is the same on both sides of a baseline of 20', () => {
  const b = committed();
  const s = b.scenarios.accuracy as Record<string, { value: number; tolerance: 'steps' }>;
  s['screen_only.time_to_act_steps.max'] = { value: 20, tolerance: 'steps' };
  const at = (v: number) => kinds(ratchet(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'screen_only.time_to_act_steps.max', v), null));
  assert.deepEqual(at(21), []);
  assert.deepEqual(at(19), []);
  assert.deepEqual(at(21.01), ['accuracy screen_only.time_to_act_steps.max worse']);
  assert.deepEqual(at(18.99), ['accuracy screen_only.time_to_act_steps.max improved']);
});

test('steps 5 % of a baseline of 0 is no band: one event of lag fails', () => {
  const b = committed();
  assert.deepEqual(kinds(ratchet(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'screen_only.time_to_act_steps.max', 1), null)), [
    'accuracy screen_only.time_to_act_steps.max worse',
  ]);
});

test('layout_shift 0.01: worse by 0.0099 passes, by 0.0101 fails, and the same below', () => {
  const b = committed();
  const v = b.scenarios.accuracy?.['layout_shift.per_100_events']?.value as number;
  const at = (x: number) => kinds(ratchet(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'layout_shift.per_100_events', x), null));
  assert.deepEqual(at(v + 0.0099), []);
  assert.deepEqual(at(v + 0.0101), ['accuracy layout_shift.per_100_events worse']);
  assert.deepEqual(at(v - 0.0099), []);
  assert.deepEqual(at(v - 0.0101), ['accuracy layout_shift.per_100_events improved']);
});

test('layout_shift 0.01: a value exactly 0.01 from the baseline passes on both sides', () => {
  const b = committed();
  const v = b.scenarios.accuracy?.['layout_shift.per_100_events']?.value as number;
  const above = Math.round((v + 0.01) * 1e6) / 1e6;
  const below = Math.round((v - 0.01) * 1e6) / 1e6;
  const at = (x: number) => kinds(ratchet(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'layout_shift.per_100_events', x), null));
  assert.deepEqual(at(above), [], `${above} is the baseline ${v} plus the tolerance`);
  assert.deepEqual(at(below), [], `${below} is the baseline ${v} minus the tolerance`);
});

test('a scenario missing from the measured values is a missing value for each of its keys', () => {
  const b = committed();
  const m = measuredAsBaseline(b);
  delete m['two-layer-relu'];
  const problems = ratchet(NAMES, KEYS, TOLERANCES, b, m, null);
  assert.equal(problems.length, Object.keys(KEYS).length);
  assert.ok(problems.every((p) => p.scenario === 'two-layer-relu' && p.kind === 'missing'));
});

test('a value measured for a key the baseline lacks is a missing baseline value', () => {
  const b = committed();
  delete b.scenarios.accuracy?.['screen_only.looks'];
  assert.deepEqual(kinds(ratchet(NAMES, KEYS, TOLERANCES, b, measuredAsBaseline(committed()), null)), ['accuracy screen_only.looks missing']);
});

test('first landing: main without the baseline does not fail the ratchet', () => {
  const b = committed();
  const main: Baseline = { about: '', scenarios: {}, tolerances: TOLERANCES };
  assert.deepEqual(ratchet(NAMES, KEYS, TOLERANCES, b, measuredAsBaseline(b), main), []);
  assert.deepEqual(ratchet(NAMES, KEYS, TOLERANCES, b, measuredAsBaseline(b), null), []);
});

test('a baseline value made worse than main without a decision fails, and with one passes', () => {
  const b = committed();
  const entry = b.scenarios.accuracy?.['reading_load.max'];
  assert.ok(entry !== undefined);
  entry.value = 26;
  assert.deepEqual(kinds(ratchet(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'reading_load.max', 26), committed())), [
    'accuracy reading_load.max baseline-worse',
  ]);
  entry.decision = 'D99';
  assert.deepEqual(ratchet(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'reading_load.max', 26), committed()), []);
});

test('the tolerances are the skill numbers (§ 6) in the adapter, in baseline.json and in each entry', () => {
  const skill = { counts: { absolute: 0 }, steps: { relative: 0.05 }, layout_shift: { absolute: 0.01 } };
  assert.deepEqual(TOLERANCES, skill);
  const b = committed();
  assert.deepEqual(b.tolerances, skill);
  for (const name of NAMES) {
    for (const [key, cls] of Object.entries(KEYS)) {
      assert.equal(b.scenarios[name]?.[key as MetricKey]?.tolerance, cls, `${name} ${key}`);
    }
  }
});

test('ADV1-01 a baseline whose counts tolerance is loosened lets a worse value through, main unchanged', () => {
  const b = committed();
  b.tolerances.counts = { absolute: 5 };
  const problems = ratchet(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'reading_load.max', 28), committed());
  assert.notDeepEqual(problems, [], 'reading_load.max 28 against 25 with counts tolerance 0 on main');
});

test('ADV1-01 a baseline entry moved from counts to steps lets a worse value through, main unchanged', () => {
  const b = committed();
  const entry = b.scenarios['linear-regression']?.['reading_load.max'];
  assert.ok(entry !== undefined);
  (entry as { tolerance: string }).tolerance = 'steps';
  const problems = ratchet(NAMES, KEYS, TOLERANCES, b, one('linear-regression', 'reading_load.max', 32), committed());
  assert.notDeepEqual(problems, [], 'reading_load.max is a count (KEYS), tolerance 0: 32 against 31 is worse');
});

test('ADV1-02 UX_DECISION set to the empty string does not take a worse value into the baseline', () => {
  const b = committed();
  const next = update(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'reading_load.max', 40), '');
  assert.equal(next.scenarios.accuracy?.['reading_load.max']?.value, 25, 'an empty decision id is no decision id');
});

test('ADV1-01 a tolerance class in baseline.json that disagrees with KEYS is a ratchet error naming the key', () => {
  const b = committed();
  const entry = b.scenarios['linear-regression']?.['reading_load.max'];
  assert.ok(entry !== undefined);
  (entry as { tolerance: string }).tolerance = 'steps';
  const problems = ratchet(NAMES, KEYS, TOLERANCES, b, measuredAsBaseline(committed()), null);
  assert.deepEqual(kinds(problems), ['linear-regression reading_load.max tolerance']);
  assert.match(problems[0]?.message ?? '', /reading_load\.max.*steps.*counts/);
});

test('ADV1-01 a tolerances block in baseline.json that disagrees with the code is a ratchet error naming the class', () => {
  const b = committed();
  b.tolerances.layout_shift = { absolute: 1 };
  const problems = ratchet(NAMES, KEYS, TOLERANCES, b, measuredAsBaseline(committed()), null);
  assert.deepEqual(
    problems.map((p) => `${p.key} ${p.kind}`),
    ['tolerances.layout_shift tolerance'],
  );
});

test('ADV1-01 the band is the code tolerance even when baseline.json is loosened: layout_shift 0.02 worse fails', () => {
  const b = committed();
  b.tolerances.layout_shift = { absolute: 1 };
  const v = b.scenarios.accuracy?.['layout_shift.per_100_events']?.value as number;
  const problems = ratchet(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'layout_shift.per_100_events', v + 0.02), null);
  assert.ok(problems.some((p) => p.key === 'layout_shift.per_100_events' && p.kind === 'worse'), JSON.stringify(problems));
});

test('ADV1-02 one predicate decides a decision id: empty and whitespace are none, in update and in the ratchet', () => {
  assert.equal(isDecision(''), false);
  assert.equal(isDecision('  \t'), false);
  assert.equal(isDecision(null), false);
  assert.equal(isDecision(undefined), false);
  assert.equal(isDecision('D59'), true);
  const b = committed();
  const kept = update(NAMES, KEYS, TOLERANCES, b, one('accuracy', 'reading_load.max', 40), '   ');
  assert.equal(kept.scenarios.accuracy?.['reading_load.max']?.value, b.scenarios.accuracy?.['reading_load.max']?.value);
  const worse = committed();
  const entry = worse.scenarios.accuracy?.['reading_load.max'];
  assert.ok(entry !== undefined);
  entry.value += 1;
  entry.decision = ' ';
  assert.deepEqual(kinds(ratchet(NAMES, KEYS, TOLERANCES, worse, one('accuracy', 'reading_load.max', entry.value), committed())), [
    'accuracy reading_load.max baseline-worse',
  ]);
});

test('gate 4 and gate 5 constants are the skill numbers: 2 actions a fix, the five axe tags', () => {
  assert.equal(MAX_ACTIONS_PER_FIX, 2);
  assert.deepEqual(AXE_TAGS, ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']);
});

/** A throwaway git repository under the run's cache, never committed anywhere. */
function scratchRepo(): { dir: string; git: (...args: string[]) => string } {
  mkdirSync(OUT_DIR, { recursive: true });
  const dir = mkdtempSync(`${OUT_DIR}git-`);
  const env = {
    ...process.env,
    GIT_AUTHOR_NAME: 'ux',
    GIT_AUTHOR_EMAIL: 'ux@example.invalid',
    GIT_COMMITTER_NAME: 'ux',
    GIT_COMMITTER_EMAIL: 'ux@example.invalid',
    GIT_CONFIG_NOSYSTEM: '1',
    GIT_CONFIG_GLOBAL: '/dev/null',
  };
  const git = (...args: string[]) => execFileSync('git', args, { cwd: dir, env, stdio: ['ignore', 'pipe', 'pipe'] }).toString().trim();
  git('init', '-q');
  return { dir, git };
}

/** Points `refs/remotes/origin/main` at a commit whose tree holds `files`. */
function mainWith(repo: ReturnType<typeof scratchRepo>, files: Record<string, string>): void {
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(`${repo.dir}/${path.split('/').slice(0, -1).join('/')}`, { recursive: true });
    writeFileSync(`${repo.dir}/${path}`, text);
    const blob = repo.git('hash-object', '-w', path);
    repo.git('update-index', '--add', '--cacheinfo', `100644,${blob},${path}`);
  }
  const tree = repo.git('write-tree');
  const commit = repo.git('commit-tree', tree, '-m', 'main');
  repo.git('update-ref', 'refs/remotes/origin/main', commit);
}

test('ADV1-05 origin/main missing is an error naming the ref, as in a depth-1 checkout', () => {
  const repo = scratchRepo();
  try {
    assert.throws(() => mainBaseline(repo.dir, 'ui/ux/baseline.json'), /origin\/main/);
  } finally {
    rmSync(repo.dir, { recursive: true, force: true });
  }
});

test('ADV1-05 baseline.json absent on origin/main is the first landing (null)', () => {
  const repo = scratchRepo();
  try {
    mainWith(repo, { 'README.md': 'x\n' });
    assert.equal(mainBaseline(repo.dir, 'ui/ux/baseline.json'), null);
  } finally {
    rmSync(repo.dir, { recursive: true, force: true });
  }
});

test('ADV1-05 baseline.json on origin/main is read from the ref, not from the working tree', () => {
  const repo = scratchRepo();
  try {
    mainWith(repo, { 'ui/ux/baseline.json': '{"on":"main"}\n' });
    writeFileSync(`${repo.dir}/ui/ux/baseline.json`, '{"on":"branch"}\n');
    assert.deepEqual(mainBaseline(repo.dir, 'ui/ux/baseline.json'), { on: 'main' });
  } finally {
    rmSync(repo.dir, { recursive: true, force: true });
  }
});

test('T6 the ui/dist hash covers the sorted file list and the contents', () => {
  mkdirSync(OUT_DIR, { recursive: true });
  const dir = mkdtempSync(`${OUT_DIR}dist-`);
  try {
    mkdirSync(`${dir}/assets`);
    writeFileSync(`${dir}/index.html`, 'a');
    writeFileSync(`${dir}/assets/x.js`, 'b');
    const one = treeSha256(dir);
    assert.match(one, /^[0-9a-f]{64}$/);
    assert.equal(treeSha256(dir), one);
    writeFileSync(`${dir}/assets/x.js`, 'c');
    const two = treeSha256(dir);
    assert.notEqual(two, one);
    rmSync(`${dir}/assets/x.js`);
    writeFileSync(`${dir}/assets/y.js`, 'c');
    assert.notEqual(treeSha256(dir), two, 'a renamed file changes the hash');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('T7 a round rerun removes only the shots it generates, never review-* or other files', () => {
  const files = [
    'accuracy-1366x768-light-end.jpg',
    'accuracy-failed-1366x768-dark-worst-shift.jpg',
    'linear-regression-contact.jpg',
    'two-layer-relu-1280x720-light-first-parameter-gradient-found.jpg',
    'review-accuracy-finished-dark.png',
    'review-relu-end-full-light.png',
    'notes.txt',
    'old-card-contact.jpg',
  ];
  assert.deepEqual(roundShotsToRemove(files, NAMES), [
    'accuracy-1366x768-light-end.jpg',
    'accuracy-failed-1366x768-dark-worst-shift.jpg',
    'linear-regression-contact.jpg',
    'two-layer-relu-1280x720-light-first-parameter-gradient-found.jpg',
  ]);
});

function roundOf(round: string, scenarios: string[], value: number): RoundInput {
  return {
    round,
    date: '2026-10-25',
    commit: { head: 'h', dirty: false, ui_dist_sha256: 'd' },
    scenarios: scenarios.map((name) => ({ name, card: 'anchor' })),
    keys: Object.keys(KEYS) as MetricKey[],
    measured: Object.fromEntries(scenarios.map((s) => [s, Object.fromEntries(Object.keys(KEYS).map((k) => [k, value]))])),
    baseline: null,
    previous: null,
    problems: [],
    fragments: [],
    bars: [],
    notRun: [],
    findings: [],
    definitions: [],
    caveats: [],
  };
}

test('T8 the trend names its columns per scenario: a new rotating card widens the header and pads the rows', () => {
  const first = trend(null, roundOf('R-00', ['a', 'rot1'], 1));
  const second = trend(first, roundOf('R-01', ['a', 'rot2'], 2));
  const lines = second.split('\n').filter((l) => l.startsWith('|'));
  const head = (lines[0] ?? '').split('|').map((c) => c.trim()).filter((c) => c !== '');
  assert.ok(head.includes('rot1 reading_load.max') && head.includes('rot2 reading_load.max'), lines[0]);
  const row = (name: string) => {
    const cells = (lines.find((l) => l.startsWith(`| ${name} |`)) ?? '').split('|').map((c) => c.trim()).slice(1, -1);
    return Object.fromEntries(head.map((h, i) => [h, cells[i]]));
  };
  assert.equal(row('R-00')['rot1 reading_load.max'], '1');
  assert.equal(row('R-00')['rot2 reading_load.max'], '-');
  assert.equal(row('R-01')['rot1 reading_load.max'], '-');
  assert.equal(row('R-01')['rot2 reading_load.max'], '2');
  assert.equal(row('R-01')['a reading_load.max'], '2');
  const again = trend(second, roundOf('R-01', ['a', 'rot2'], 3));
  assert.equal(again.split('\n').filter((l) => l.startsWith('| R-01 |')).length, 1, 'a rerun replaces its row');
});
