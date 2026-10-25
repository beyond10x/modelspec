// L2, the layout contract (story:observatory-ui-iteration-loop; the ui-iteration-loop skill § 2
// and § 4): every scenario at every gating viewport in both themes, one recorded event at a time,
// checked after the app has drawn each. Gate 1 no scroll, gate 2 always visible, gate 3
// unmissable completion, gate 5 accessibility (axe-core at every key moment, keyboard reach and
// focus indicators at the end). The measured-only viewports are replayed at their key moments
// and reported, never gated.

import { AxeBuilder } from '@axe-core/playwright';
import { type Browser, expect, test } from '@playwright/test';
import type { RunEvent } from '../src/generated/observatory.ts';
import {
  AXE_TAGS,
  COMPLETION,
  GATING_VIEWPORTS,
  type Loaded,
  MEASURED_VIEWPORTS,
  MOMENTS,
  type Moment,
  OUT_DIR,
  SCENARIOS,
  THEMES,
  type Viewport,
  alwaysVisibleAt,
  drawnSelector,
  failedVariant,
  loadScenario,
  momentIndex,
  viewportName,
} from './adapter.ts';
import { ROUND, checkTruth, replay, shot } from './harness.ts';
import type { Theme } from './core/metrics.ts';
import { animations, focusState, focusables, probe } from './core/page.ts';
import { type LayoutFragment, writeFragment } from './core/round.ts';

const DRAW_MS = 5_000;

function describeEvent(index: number, event: RunEvent | null): string {
  if (event === null) {
    return 'before the first event';
  }
  const what = event.body.kind === 'StepCompleted' ? `StepCompleted step ${event.body.value.step}` : event.body.kind;
  return `event ${index} (${what})`;
}

interface Contract {
  fragment: LayoutFragment;
  failures: string[];
}

/** Replays `loaded` (or `events`) and checks the contract after each event, or only at moments. */
async function contract(
  browser: Browser,
  loaded: Loaded,
  viewport: Viewport,
  theme: Theme,
  options: { gating: boolean; events?: RunEvent[]; label?: string },
): Promise<Contract> {
  const where = `${viewportName(viewport)} ${theme}`;
  const run = await replay(browser, loaded, viewport, theme, options.events);
  const { page, events, pacer } = run;
  const moments = momentIndex(events);
  const momentAt = new Map<number, Moment>(MOMENTS.map((m) => [moments[m], m]));
  const end = events.length - 1;
  const endKind = events[end]?.body.kind;
  const scenario = options.label ?? loaded.scenario.name;
  const failures = new Map<string, { first: string; count: number }>();
  const fail = (gate: string, at: string, message: string) => {
    const key = `${gate}: ${message}`;
    const seen = failures.get(key);
    if (seen === undefined) {
      failures.set(key, { first: at, count: 1 });
    } else {
      seen.count += 1;
    }
  };
  const fragment: LayoutFragment = {
    kind: 'layout',
    scenario,
    viewport: viewportName(viewport),
    theme,
    gating: options.gating,
    events: events.length,
    shift: 0,
    worstShift: null,
    readingLoad: [],
    violations: [],
    completionMiss: 0,
    falseComplete: 0,
    axe: [],
    shots: [],
  };
  try {
    for (let index = -1; index < events.length; index++) {
      const moment = momentAt.get(index);
      if (!options.gating && moment === undefined) {
        continue;
      }
      if (index >= 0) {
        // Measured-only replays send everything up to the next moment at once.
        const count = options.gating ? 1 : index + 1 - (events.length - pacer.remaining());
        pacer.push(count);
      }
      const event = index >= 0 ? (events[index] ?? null) : null;
      const at = `${where}, ${describeEvent(index, event)}`;
      const result = await page.evaluate(probe, {
        until: drawnSelector(event),
        timeoutMs: DRAW_MS,
        alwaysVisible: alwaysVisibleAt(events, index).map(({ name, selector }) => ({ name, selector })),
        stateSelector: COMPLETION.stateSelector,
        alertSelector: COMPLETION.failed.selector,
      });
      if (!result.settled) {
        fail('replay', at, `the app did not draw the event within ${DRAW_MS} ms`);
      }
      for (const s of result.scroll) {
        fail('gate 1 (no scroll)', at, s);
      }
      for (const h of result.hidden) {
        fail('gate 2 (always visible)', at, h);
      }
      const completed =
        (result.stateText !== null && (COMPLETION.completionWords as readonly string[]).includes(result.stateText)) ||
        result.alertInView;
      if (index < end && completed) {
        fragment.falseComplete += 1;
        fail('gate 3 (false_complete)', at, `the screen shows the run as ${result.stateText} before its stream ended`);
      }
      if (index === end) {
        const shown =
          endKind === 'RunFailed'
            ? result.alertInView && result.stateText === 'failed' && result.stateInView
            : result.stateText === COMPLETION.finished.text && result.stateInView;
        if (!shown) {
          fragment.completionMiss += 1;
          fail('gate 3 (completion_miss)', at, `the end state ${endKind} is not shown in view (state reads ${result.stateText})`);
        }
      }
      fragment.shift += result.shift;
      if (result.shift > 0 && (fragment.worstShift === null || result.shift > fragment.worstShift.value)) {
        fragment.worstShift = {
          index,
          event: describeEvent(index, event),
          value: result.shift,
          sources: result.shiftSources,
          shot: null,
        };
        if (ROUND !== null && options.gating && viewport === GATING_VIEWPORTS[0] && theme === 'light') {
          fragment.worstShift.shot = await shot(page, `${scenario}-${fragment.viewport}-${theme}-worst-shift`);
        }
      }
      if (moment !== undefined) {
        let file: string | null = null;
        if (ROUND !== null && options.gating && viewport === GATING_VIEWPORTS[0]) {
          file = await shot(page, `${scenario}-${fragment.viewport}-${theme}-${moment}`);
          fragment.shots.push({ moment, file });
        }
        fragment.readingLoad.push({ moment, words: result.words, shot: file });
        const axe = await new AxeBuilder({ page }).withTags(AXE_TAGS).analyze();
        const serious = axe.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical');
        fragment.axe.push({ moment, serious: serious.length, ids: serious.map((v) => v.id) });
        for (const v of serious) {
          fail(
            'gate 5 (accessibility)',
            `${where}, ${moment}`,
            `axe ${v.impact} ${v.id}: ${v.help} at ${v.nodes.map((n) => n.target.join(' ')).slice(0, 3).join(', ')}`,
          );
        }
      }
    }
    // Gate 5, keyboard: every interactive element is reached by Tab from the top and shows focus.
    const ids = await page.evaluate(focusables);
    const reached = new Set<string>();
    for (let i = 0; i < ids.length + 2; i++) {
      await page.keyboard.press('Tab');
      const state = await page.evaluate(focusState);
      if (state.id !== null) {
        reached.add(state.id);
        if (!state.indicated) {
          fail('gate 5 (keyboard)', `${where}, end`, `focusable ${state.id} shows no focus indicator`);
        }
      }
    }
    for (const id of ids) {
      if (!reached.has(id)) {
        fail('gate 5 (keyboard)', `${where}, end`, `focusable ${id} is not reached by Tab`);
      }
    }
    const running = await page.evaluate(animations);
    if (running > 0) {
      fail('gate 5 (reduced motion)', `${where}, end`, `${running} animations run under prefers-reduced-motion`);
    }
  } finally {
    await run.close();
  }
  fragment.violations = [...failures.entries()].map(
    ([message, { first, count }]) => `${message} (first at ${first}${count > 1 ? `, ${count} times` : ''})`,
  );
  return { fragment, failures: fragment.violations };
}

for (const scenario of SCENARIOS) {
  test.describe(`L2 layout ${scenario.name}`, () => {
    for (const viewport of GATING_VIEWPORTS) {
      for (const theme of THEMES) {
        test(`${viewportName(viewport)} ${theme}`, async ({ browser }) => {
          const loaded = loadScenario(scenario);
          checkTruth(loaded);
          const { fragment, failures } = await contract(browser, loaded, viewport, theme, { gating: true });
          writeFragment(OUT_DIR, fragment);
          expect(failures, `${scenario.name} at ${viewportName(viewport)} ${theme}:\n${failures.join('\n')}`).toEqual([]);
        });
      }
    }

    test('measured-only viewports', async ({ browser }) => {
      const loaded = loadScenario(scenario);
      for (const viewport of MEASURED_VIEWPORTS) {
        for (const theme of THEMES) {
          const { fragment } = await contract(browser, loaded, viewport, theme, { gating: false });
          writeFragment(OUT_DIR, fragment);
        }
      }
    });
  });
}

test.describe('L2 completion failed', () => {
  const scenario = SCENARIOS.find((s) => s.card === 'rotating') ?? SCENARIOS[0];
  for (const theme of THEMES) {
    test(`${scenario?.name} ending in RunFailed ${theme}`, async ({ browser }) => {
      if (scenario === undefined) {
        throw new Error('no scenario');
      }
      const loaded = loadScenario(scenario);
      const viewport = GATING_VIEWPORTS[0] as Viewport;
      const { failures } = await contract(browser, loaded, viewport, theme, {
        gating: true,
        events: failedVariant(loaded.events),
        label: `${scenario.name}-failed`,
      });
      expect(failures, `${scenario.name} failed variant at ${viewportName(viewport)} ${theme}:\n${failures.join('\n')}`).toEqual([]);
    });
  }
});
