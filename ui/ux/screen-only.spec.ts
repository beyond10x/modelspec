// L3, the screen-only user (story:observatory-ui-iteration-loop, Acceptance 5; the
// ui-iteration-loop skill § 7 and gate 4): every scenario at every gating viewport. While the
// run streams, after each event, it reads the live tasks (those that change every step) without
// an action and counts how many events the screen lags behind (`time_to_act`); at each key moment
// it notes the glance level of every task the page can answer; at the end state it performs every
// task from the top of the page, keyboard only, and compares its reading with the truth. The app
// has no guidance cue, so there is no cue to follow or steer; a steer is a reversal of its course.

import { expect, test } from '@playwright/test';
import {
  ACTION_LIMIT,
  GATING_VIEWPORTS,
  MAX_ACTIONS_PER_FIX,
  MOMENTS,
  type Moment,
  OUT_DIR,
  PRIMARY_REGION,
  SCENARIOS,
  type Task,
  type Truth,
  drawnSelector,
  loadScenario,
  momentIndex,
  truthOf,
  truthValue,
  viewportName,
} from './adapter.ts';
import { ROUND, checkTruth, replay, shot } from './harness.ts';
import { type BotTask, glanceNow, look, perform } from './core/bot.ts';
import { type AttemptRecord, type ScreenFragment, writeFragment } from './core/round.ts';

const PRIMARY = `[data-ux-region="${PRIMARY_REGION}"]`;
const LIVE_BUDGET_MS = 1_000;

/**
 * Whether `read` (text from the screen) is `expected`. A number is compared at the precision the
 * screen shows it: equal to the truth rounded to 4, 5 or 6 significant digits, no fewer than
 * `read` shows (the app shows 6, and 4 in exponent form).
 */
function same(read: string | undefined, expected: unknown): boolean {
  if (read === undefined) {
    return false;
  }
  if (typeof expected === 'number') {
    if (!Number.isFinite(expected)) {
      return read === 'non-finite';
    }
    const n = Number(read);
    if (!Number.isFinite(n)) {
      return false;
    }
    if (n === expected) {
      return true;
    }
    // Trailing zeros are not shown, so `0.1` may be the truth at 4 to 6 digits.
    const digits = read.replace(/^[-+]/, '').replace(/e.*$/i, '').replace('.', '').replace(/^0+/, '').length;
    for (let p = Math.max(4, digits); p <= 6; p++) {
      if (Number(expected.toPrecision(p)) === n) {
        return true;
      }
    }
    return false;
  }
  return read === String(expected);
}

function check(task: Task, truth: Truth, groups: Record<string, string> | null): { correct: boolean; expected: Record<string, unknown> } {
  const expected = Object.fromEntries(Object.entries(task.answer).map(([group, key]) => [group, truthValue(truth, key)]));
  const correct = groups !== null && Object.entries(expected).every(([group, value]) => same(groups[group], value));
  return { correct, expected };
}

function bot(task: Task): BotTask {
  return { id: task.id, role: task.role, name: task.name, text: task.text, read: task.read };
}

for (const scenario of SCENARIOS) {
  test.describe(`L3 screen-only ${scenario.name}`, () => {
    for (const viewport of GATING_VIEWPORTS) {
      test(`${viewportName(viewport)}`, async ({ browser }) => {
        const loaded = loadScenario(scenario);
        checkTruth(loaded);
        const theme = 'light';
        const run = await replay(browser, loaded, viewport, theme);
        const { page, events, pacer } = run;
        const moments = momentIndex(events);
        const momentAt = new Map<number, Moment>(MOMENTS.map((m) => [moments[m], m]));
        const live = loaded.tasks.filter((t) => t.live === true);
        const fragment: ScreenFragment = {
          kind: 'screen',
          scenario: scenario.name,
          viewport: viewportName(viewport),
          theme,
          attempts: [],
          glances: [],
          timeToAct: Object.fromEntries(live.map((t) => [t.id, 0])),
        };
        const wrong: string[] = [];
        try {
          // The steps each live task still waits to see: the step it must show, from the event index.
          const waiting = new Map<string, { step: number; since: number }>();
          for (let index = -1; index < events.length; index++) {
            if (index >= 0) {
              pacer.push(1);
            }
            const event = index >= 0 ? (events[index] ?? null) : null;
            const selector = drawnSelector(event);
            if (selector !== null) {
              await page.waitForSelector(selector, { state: 'attached', timeout: 5_000 }).catch(() => undefined);
            }
            if (event?.body.kind === 'StepCompleted') {
              const truthNow = truthOf(loaded.truth.recording, events.slice(0, index + 1));
              for (const task of live) {
                const step = event.body.value.step;
                const behind = waiting.get(task.id);
                const pending = behind ?? { step, since: index };
                waiting.set(task.id, pending);
                // The render budget is waited once per lag; a screen already behind is read once per event.
                const deadline = behind === undefined ? Date.now() + LIVE_BUDGET_MS : 0;
                for (;;) {
                  const seen = await look(page, bot(task));
                  const shown = Number(seen.groups?.step ?? Number.NaN);
                  if (shown >= pending.step && check(task, truthNow, seen.groups).correct) {
                    fragment.timeToAct[task.id] = Math.max(fragment.timeToAct[task.id] ?? 0, index - pending.since);
                    waiting.delete(task.id);
                    break;
                  }
                  if (Date.now() > deadline) {
                    break;
                  }
                  await page.waitForTimeout(16);
                }
              }
            }
            const moment = momentAt.get(index);
            if (moment !== undefined) {
              for (const task of loaded.tasks) {
                const level = await glanceNow(page, bot(task), PRIMARY);
                if (level !== null) {
                  fragment.glances.push({ moment, task: task.id, glance: level });
                }
              }
            }
          }
          for (const [task, pending] of waiting) {
            fragment.timeToAct[task] = Math.max(fragment.timeToAct[task] ?? 0, events.length - pending.since);
            wrong.push(`${task}: the screen never showed step ${pending.step} while the run streamed`);
          }
          // The end state: every task from the top of the page, keyboard only.
          for (const task of loaded.tasks) {
            await page.evaluate(() => window.scrollTo(0, 0));
            const attempt = await perform(page, bot(task), PRIMARY, ACTION_LIMIT);
            const { correct, expected } = check(task, loaded.truth, attempt.groups);
            let file: string | null = null;
            if (ROUND !== null && attempt.actions > 0 && viewport === GATING_VIEWPORTS[0]) {
              file = await shot(page, `${scenario.name}-${viewportName(viewport)}-${theme}-${task.id}-found`);
            }
            const record: AttemptRecord = {
              task: task.id,
              ask: task.ask,
              actions: attempt.actions,
              looks: attempt.looks,
              steers: attempt.steers,
              glance: attempt.glance,
              keys: attempt.keys,
              read: attempt.source,
              expected,
              correct,
              shot: file,
            };
            fragment.attempts.push(record);
            if (!correct) {
              wrong.push(`gate 4 (screen-only user): ${task.id} read ${attempt.source ?? 'nothing'}, the truth is ${JSON.stringify(expected)}`);
            }
            if (attempt.actions > MAX_ACTIONS_PER_FIX) {
              wrong.push(`gate 4 (screen-only user): ${task.id} took ${attempt.actions} actions (${attempt.keys.join(', ')}), at most ${MAX_ACTIONS_PER_FIX}`);
            }
          }
        } finally {
          await run.close();
        }
        writeFragment(OUT_DIR, fragment);
        expect(wrong, `${scenario.name} at ${viewportName(viewport)}:\n${wrong.join('\n')}`).toEqual([]);
      });
    }
  });
}
