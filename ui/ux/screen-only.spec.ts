// L3, the screen-only user (story:observatory-ui-iteration-loop, Acceptance 5; the
// ui-iteration-loop skill § 7 and gate 4): every scenario at every gating viewport, in two
// replays. The stepped replay sends one event at a time and waits until the app has drawn it; at
// each key moment it notes the glance level of every task the page can answer, and at the end
// state it performs every task from the top of the page, keyboard only, and compares its reading
// with the truth. The live replay sends one event every `CADENCE_MS` and never waits for the
// screen or the user; the user reads the live tasks (those that change every step) over and over
// while it runs, and `time_to_act` is how many steps the screen owes at each reading: the steps
// sent at least `RENDER_BUDGET_MS` before the reading began, minus the steps up to the one the
// screen shows. The app has no guidance cue, so there is no cue to follow or steer; a steer is a
// reversal of its course.

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
  drawnSelector,
  loadScenario,
  momentIndex,
  viewportName,
} from './adapter.ts';
import { ROUND, checkTruth, replay, shot, wrongFonts } from './harness.ts';
import { glanceNow, perform } from './core/bot.ts';
import { type AttemptRecord, type ScreenFragment, writeFragment } from './core/round.ts';
import { bot, check, live } from './screen.ts';

const PRIMARY = `[data-ux-region="${PRIMARY_REGION}"]`;

for (const scenario of SCENARIOS) {
  test.describe(`L3 screen-only ${scenario.name}`, () => {
    for (const viewport of GATING_VIEWPORTS) {
      test(`${viewportName(viewport)}`, async ({ browser }) => {
        const loaded = loadScenario(scenario);
        checkTruth(loaded);
        const theme = 'light';
        const fragment: ScreenFragment = {
          kind: 'screen',
          scenario: scenario.name,
          viewport: viewportName(viewport),
          theme,
          attempts: [],
          glances: [],
          timeToAct: {},
        };
        const wrong: string[] = [];
        const run = await replay(browser, loaded, viewport, theme);
        const { page, events, pacer } = run;
        const moments = momentIndex(events);
        const momentAt = new Map<number, Moment>(MOMENTS.map((m) => [moments[m], m]));
        try {
          for (let index = -1; index < events.length; index++) {
            if (index >= 0) {
              pacer.push(1);
            }
            const event = index >= 0 ? (events[index] ?? null) : null;
            const selector = drawnSelector(event);
            if (selector !== null) {
              await page.waitForSelector(selector, { state: 'attached', timeout: 5_000 }).catch(() => {
                wrong.push(`replay: the app did not draw event ${index} within 5000 ms`);
              });
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
          wrong.push(...(await wrongFonts(page)).map((w) => `fonts: ${w}`));
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
        const streamed = await live(browser, loaded, viewport);
        fragment.timeToAct = streamed.timeToAct;
        wrong.push(...streamed.wrong);
        writeFragment(OUT_DIR, fragment);
        expect(wrong, `${scenario.name} at ${viewportName(viewport)}:\n${wrong.join('\n')}`).toEqual([]);
      });
    }
  });
}
