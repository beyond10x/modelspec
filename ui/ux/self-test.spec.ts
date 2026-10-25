// The gates' self-tests, the `self-test` project of `pnpm ux` (story:observatory-ui-iteration-loop;
// the adversary's cases of R-00 pass 1, adopted). Each plants one known defect into the served
// page, in the test and never in the app, and asserts that the gate's own code reports it, so a
// weakened gate fails here; the `gates` project runs the gates only on the app as it is. The
// screen-only user (`core/bot.ts`, skill § 7) may use only what is visible; the ADV1-03 cases hide
// the answer from the eye while leaving it in the accessibility tree, and the user must then read
// nothing. The cases named ADV1-nn were red before their fix.

import { AxeBuilder } from '@axe-core/playwright';
import { type Browser, type Page, expect, test } from '@playwright/test';
import {
  ACTION_LIMIT,
  AXE_TAGS,
  COMPLETION,
  GATING_VIEWPORTS,
  PRIMARY_REGION,
  SCENARIOS,
  type Viewport,
  alwaysVisibleAt,
  drawnSelector,
  loadScenario,
} from './adapter.ts';
import { replay } from './harness.ts';
import { contract } from './contract.ts';
import { type BotTask, glanceNow, perform } from './core/bot.ts';
import { animations, probe } from './core/page.ts';
import { bot, live } from './screen.ts';

const PRIMARY = `[data-ux-region="${PRIMARY_REGION}"]`;

async function endState(browser: Browser, css: string | null, name = 'accuracy') {
  const scenario = SCENARIOS.find((s) => s.name === name);
  if (scenario === undefined) {
    throw new Error(`no ${name} scenario`);
  }
  const loaded = loadScenario(scenario);
  const run = await replay(browser, loaded, GATING_VIEWPORTS[0] as Viewport, 'light');
  run.pacer.push(loaded.events.length);
  await run.page.waitForSelector('main[data-run-state="finished"]', { state: 'attached', timeout: 10_000 });
  if (css !== null) {
    await run.page.addStyleTag({ content: css });
  }
  const task = loaded.tasks.find((t) => t.id === 'last-loss');
  if (task === undefined) {
    throw new Error('no last-loss task');
  }
  const bot: BotTask = { id: task.id, role: task.role, name: task.name, text: task.text, read: task.read };
  return { run, bot, loaded };
}

async function probeEnd(page: Page, events: Parameters<typeof alwaysVisibleAt>[0]) {
  return page.evaluate(probe, {
    until: null,
    timeoutMs: 0,
    alwaysVisible: alwaysVisibleAt(events, events.length - 1).map(({ name, selector }) => ({ name, selector })),
    stateSelector: COMPLETION.stateSelector,
    alertSelector: COMPLETION.failed.selector,
  });
}

async function seriousAxe(page: Page): Promise<string[]> {
  const axe = await new AxeBuilder({ page }).withTags(AXE_TAGS).analyze();
  return axe.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical').map((v) => v.id);
}

test('gate 1 and 2: the probe passes the finished accuracy page and reports .charts at 110vw', async ({ browser }) => {
  const { run, loaded } = await endState(browser, null);
  try {
    const clean = await probeEnd(run.page, loaded.events);
    expect(clean.scroll).toEqual([]);
    expect(clean.hidden).toEqual([]);
    await run.page.addStyleTag({ content: '.charts { width: 110vw }' });
    const planted = await probeEnd(run.page, loaded.events);
    expect(planted.scroll.some((s) => s.startsWith('horizontal page scroll'))).toBe(true);
    expect(planted.hidden.some((h) => h.startsWith('loss chart: not wholly in the viewport'))).toBe(true);
  } finally {
    await run.close();
  }
});

test('gate 1: the probe reports a parameter panel that scrolls', async ({ browser }) => {
  const { run, loaded } = await endState(browser, null, 'two-layer-relu');
  try {
    await run.page.waitForSelector('main[data-run-state="finished"]', { state: 'attached', timeout: 10_000 });
    await run.page.addStyleTag({ content: '[data-ux-region="parameter-panel"] { overflow: auto; max-height: 300px }' });
    const planted = await probeEnd(run.page, loaded.events);
    expect(planted.scroll.some((s) => s.includes('scrollable element section.parameter') || s.includes('[data-ux-region=parameter-panel]'))).toBe(true);
  } finally {
    await run.close();
  }
});

test('gate 2: the probe reports the run state 2 px below the viewport', async ({ browser }) => {
  const { run, loaded } = await endState(browser, null);
  try {
    const vh = (GATING_VIEWPORTS[0] as Viewport).height;
    await run.page.addStyleTag({ content: `.status .state { position: fixed; left: 8px; top: ${vh - 19}px; height: 21px; line-height: 21px }` });
    const planted = await probeEnd(run.page, loaded.events);
    expect(planted.hidden.some((h) => h.startsWith('run state: not wholly in the viewport'))).toBe(true);
    expect(planted.stateInView).toBe(false);
  } finally {
    await run.close();
  }
});

test('gate 5: AXE_TAGS pass the finished page and report a nameless svg[role=img]', async ({ browser }) => {
  const { run } = await endState(browser, null);
  try {
    expect(await seriousAxe(run.page)).toEqual([]);
    await run.page.evaluate(() => document.querySelector('svg[role="img"]')?.removeAttribute('aria-label'));
    expect(await seriousAxe(run.page)).toContain('svg-img-alt');
  } finally {
    await run.close();
  }
});

test('gate 5: AXE_TAGS report an invalid autocomplete, a wcag21aa rule', async ({ browser }) => {
  const { run } = await endState(browser, null);
  try {
    await run.page.evaluate(() => {
      const input = document.createElement('input');
      input.setAttribute('aria-label', 'planted');
      input.setAttribute('autocomplete', 'planted-token');
      document.querySelector('main')?.prepend(input);
    });
    expect(await seriousAxe(run.page)).toContain('autocomplete-valid');
  } finally {
    await run.close();
  }
});

test('gate 5: AXE_TAGS report text #888 on #999', async ({ browser }) => {
  const { run } = await endState(browser, '.chart .name { color: #888; background: #999 }');
  try {
    expect(await seriousAxe(run.page)).toContain('color-contrast');
  } finally {
    await run.close();
  }
});

test('control: the screen-only user reads the last loss at the end of accuracy', async ({ browser }) => {
  const { run, bot } = await endState(browser, null);
  try {
    const attempt = await perform(run.page, bot, PRIMARY, ACTION_LIMIT);
    expect(attempt.groups?.value).toBe('0.0952381');
  } finally {
    await run.close();
  }
});

test('ADV1-03 the screen-only user reads nothing when the loss value is transparent', async ({ browser }) => {
  const { run, bot } = await endState(browser, '.chart .last { opacity: 0 }');
  try {
    const attempt = await perform(run.page, bot, PRIMARY, ACTION_LIMIT);
    expect(attempt.groups, `read ${attempt.source} with ${attempt.actions} actions`).toBeNull();
  } finally {
    await run.close();
  }
});

test('ADV1-03 the screen-only user reads nothing when the loss value is covered', async ({ browser }) => {
  const { run, bot } = await endState(
    browser,
    '.chart .last { position: relative } .chart .last::after { content: ""; position: absolute; inset: -3px; background: var(--panel) }',
  );
  try {
    const attempt = await perform(run.page, bot, PRIMARY, ACTION_LIMIT);
    expect(attempt.groups, `read ${attempt.source} with ${attempt.actions} actions`).toBeNull();
  } finally {
    await run.close();
  }
});


function accuracy() {
  const scenario = SCENARIOS.find((s) => s.name === 'accuracy');
  if (scenario === undefined) {
    throw new Error('no accuracy scenario');
  }
  return loadScenario(scenario);
}

const SLIDE =
  '@media (prefers-reduced-motion: reduce) { .chart { animation: adv-slide 2s linear } } @keyframes adv-slide { from { transform: translateY(8px) } to { transform: none } }';

test('ADV1-04 no animation runs under reduced motion at any replayed event, not only 2 s after the end', async ({ browser }) => {
  const loaded = accuracy();
  // The plant animates at the events, and is over 2.5 s after the end, where the gate used to look.
  const run = await replay(browser, loaded, GATING_VIEWPORTS[0] as Viewport, 'light', loaded.events, { css: SLIDE });
  const running: number[] = [];
  let atEnd = 0;
  try {
    for (const event of loaded.events) {
      run.pacer.push(1);
      await run.page.waitForSelector(drawnSelector(event) as string, { state: 'attached', timeout: 5_000 });
      running.push(await run.page.evaluate(animations));
    }
    await run.page.waitForTimeout(2_500);
    atEnd = await run.page.evaluate(animations);
  } finally {
    await run.close();
  }
  expect(Math.max(...running), `the plant animates: per event ${running.join(',')}`).toBeGreaterThan(0);
  expect(atEnd, 'the plant is over 2.5 s after the end').toBe(0);
  const { failures } = await contract(browser, loaded, GATING_VIEWPORTS[0] as Viewport, 'light', { gating: true, plant: SLIDE, shots: false });
  expect(failures.some((f) => f.startsWith('gate 5 (reduced motion): an animation or transition runs')), failures.join('\n')).toBe(true);
});

test('gate 3: the contract counts a completion_miss when the finished state is not in view at the end', async ({ browser }) => {
  const loaded = accuracy();
  const { fragment, failures } = await contract(browser, loaded, GATING_VIEWPORTS[0] as Viewport, 'light', {
    gating: true,
    plant: 'main[data-run-state="finished"] .status .state { visibility: hidden }',
    shots: false,
  });
  expect(fragment.completionMiss, failures.join('\n')).toBe(1);
  expect(failures.some((f) => f.startsWith('gate 3 (completion_miss)')), failures.join('\n')).toBe(true);
});

test('gate 5: the contract reports a focusable without a focus indicator', async ({ browser }) => {
  const loaded = SCENARIOS.find((s) => s.name === 'two-layer-relu');
  if (loaded === undefined) {
    throw new Error('no two-layer-relu scenario');
  }
  const { failures } = await contract(browser, loadScenario(loaded), GATING_VIEWPORTS[0] as Viewport, 'light', {
    gating: false,
    plant: ':focus, :focus-visible { outline: none !important; box-shadow: none !important }',
    shots: false,
  });
  expect(failures.some((f) => /^gate 5 \(keyboard\): focusable \d+ shows no focus indicator/.test(f)), failures.join('\n')).toBe(true);
});

test('fonts: the contract reports a page drawn without the loop font stack', async ({ browser }) => {
  const loaded = accuracy();
  const { failures } = await contract(browser, loaded, GATING_VIEWPORTS[0] as Viewport, 'light', {
    gating: false,
    plant: 'html, body, body * { font-family: serif !important }',
    shots: false,
  });
  expect(failures.some((f) => f.startsWith('fonts: [data-ux-region="header"] h1 is drawn with')), failures.join('\n')).toBe(true);
});

test('T4 the end-state task is not answerable while the run streams, and is once it has finished', async ({ browser }) => {
  const loaded = accuracy();
  const task = loaded.tasks.find((t) => t.id === 'end-state');
  if (task === undefined) {
    throw new Error('no end-state task');
  }
  const run = await replay(browser, loaded, GATING_VIEWPORTS[0] as Viewport, 'light');
  try {
    run.pacer.push(5);
    await run.page.waitForSelector(drawnSelector(loaded.events[4] ?? null) as string, { state: 'attached', timeout: 5_000 });
    expect(await glanceNow(run.page, bot(task), PRIMARY), 'streaming: nothing answers how the run ended').toBeNull();
    run.pacer.push(loaded.events.length);
    await run.page.waitForSelector('main[data-run-state="finished"]', { state: 'attached', timeout: 5_000 });
    expect(await glanceNow(run.page, bot(task), PRIMARY)).not.toBeNull();
  } finally {
    await run.close();
  }
});

test('ADV1-07 a render lag of 1.5 s per event gives time_to_act above 0, and none gives 0', async ({ browser }) => {
  // two-layer-relu streams 203 events, 8 s at 40 ms, so the screen is behind for longer than a
  // reading takes on a loaded machine; accuracy's 23 events can all be drawn before the first one.
  const scenario = SCENARIOS.find((s) => s.name === 'two-layer-relu');
  if (scenario === undefined) {
    throw new Error('no two-layer-relu scenario');
  }
  const loaded = loadScenario(scenario);
  const viewport = GATING_VIEWPORTS[0] as Viewport;
  const clean = await live(browser, loaded, viewport);
  expect(clean.wrong).toEqual([]);
  expect(Math.max(...Object.values(clean.timeToAct)), JSON.stringify(clean.timeToAct)).toBe(0);
  const lagged = await live(browser, loaded, viewport, { lagMs: 1_500 });
  expect(Math.max(...Object.values(lagged.timeToAct)), JSON.stringify(lagged.timeToAct)).toBeGreaterThan(0);
});
