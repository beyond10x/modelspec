// The layout contract of L2 (story:observatory-ui-iteration-loop; the ui-iteration-loop skill
// § 2 and § 4), one replay at a time: gate 1 no scroll, gate 2 always visible, gate 3 unmissable
// completion and gate 5 accessibility (axe-core at every key moment, no animation under reduced
// motion after any event, keyboard reach and focus indicators at the end), and the fonts the page
// draws with. `layout.spec.ts` runs it on the app; `self-test.spec.ts` runs it on planted defects.

import { AxeBuilder } from '@axe-core/playwright';
import type { Browser } from '@playwright/test';
import type { RunEvent } from '../src/generated/observatory.ts';
import {
  AXE_TAGS,
  COMPLETION,
  GATING_VIEWPORTS,
  type Loaded,
  MOMENTS,
  type Moment,
  type Viewport,
  alwaysVisibleAt,
  drawnSelector,
  momentIndex,
  viewportName,
} from './adapter.ts';
import { ROUND, replay, shot, wrongFonts } from './harness.ts';
import type { Theme } from './core/metrics.ts';
import { animations, focusState, focusables, probe } from './core/page.ts';
import type { LayoutFragment } from './core/round.ts';

export const DRAW_MS = 5_000;

function describeEvent(index: number, event: RunEvent | null): string {
  if (event === null) {
    return 'before the first event';
  }
  const what = event.body.kind === 'StepCompleted' ? `StepCompleted step ${event.body.value.step}` : event.body.kind;
  return `event ${index} (${what})`;
}

export interface Contract {
  fragment: LayoutFragment;
  failures: string[];
}

export interface ContractOptions {
  /** Every event (a gating replay), or only the key moments (measured-only). */
  gating: boolean;
  /** The events to send instead of the recording's (the failed end state). */
  events?: RunEvent[];
  /** The scenario name the fragment carries, when not the scenario's own. */
  label?: string;
  /** A replay of the failed end state: gated, never ratcheted. */
  failedEnd?: boolean;
  /** `false`: no screenshots even in a round (the gates' self-tests). */
  shots?: boolean;
  /** A stylesheet planted before the first paint (the gates' self-tests). */
  plant?: string;
}

/** Replays `loaded` (or `options.events`) and checks the contract after each event, or only at moments. */
export async function contract(browser: Browser, loaded: Loaded, viewport: Viewport, theme: Theme, options: ContractOptions): Promise<Contract> {
  const where = `${viewportName(viewport)} ${theme}`;
  const run = await replay(browser, loaded, viewport, theme, options.events, options.plant === undefined ? null : { css: options.plant });
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
    failedEnd: options.failedEnd ?? false,
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
      if (result.animations > 0) {
        fail('gate 5 (reduced motion)', at, 'an animation or transition runs under prefers-reduced-motion');
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
        // Every gating replay keeps a shot of its own worst moment; the round shows the largest.
        if (ROUND !== null && options.gating && options.shots !== false) {
          fragment.worstShift.shot = await shot(page, `${scenario}-${fragment.viewport}-${theme}-worst-shift`);
        }
      }
      if (moment !== undefined) {
        let file: string | null = null;
        if (ROUND !== null && options.gating && options.shots !== false && viewport === GATING_VIEWPORTS[0]) {
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
    for (const wrong of await wrongFonts(page)) {
      fail('fonts', `${where}, end`, wrong);
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
