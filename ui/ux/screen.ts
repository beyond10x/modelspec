// The screen-only user's checks and its live replay (story:observatory-ui-iteration-loop; the
// ui-iteration-loop skill § 7): a reading compared with the truth at the precision the screen
// shows, and `time_to_act` measured with the stream running at `CADENCE_MS` while the user reads.
// `screen-only.spec.ts` runs them on the app; `self-test.spec.ts` on a planted render lag.

import type { Browser } from '@playwright/test';
import { CADENCE_MS, type Loaded, RENDER_BUDGET_MS, type Task, type Truth, type Viewport, truthOf, truthValue } from './adapter.ts';
import { type Plant, replay } from './harness.ts';
import { type BotTask, look } from './core/bot.ts';

/** How long the live replay waits, after its last event, for the screen to show the last step. */
export const CATCH_UP_MS = 5_000;

/**
 * Whether `read` (text from the screen) is `expected`. A number is compared at the precision the
 * screen shows it: equal to the truth rounded to 4, 5 or 6 significant digits, no fewer than
 * `read` shows (the app shows 6, and 4 in exponent form).
 */
export function same(read: string | undefined, expected: unknown): boolean {
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

export function check(task: Task, truth: Truth, groups: Record<string, string> | null): { correct: boolean; expected: Record<string, unknown> } {
  const expected = Object.fromEntries(Object.entries(task.answer).map(([group, key]) => [group, truthValue(truth, key)]));
  const correct = groups !== null && Object.entries(expected).every(([group, value]) => same(groups[group], value));
  return { correct, expected };
}

export function bot(task: Task): BotTask {
  return { id: task.id, role: task.role, name: task.name, text: task.text, read: task.read };
}

/**
 * The live replay: the stream runs at `CADENCE_MS` while the user reads every live task in turn.
 * Returns the most steps the screen owed at a reading, per task, and every reading that differs
 * from the truth at the step it shows.
 */
export async function live(browser: Browser, loaded: Loaded, viewport: Viewport, plant: Plant | null = null): Promise<{ timeToAct: Record<string, number>; wrong: string[] }> {
  const tasks = loaded.tasks.filter((t) => t.live === true);
  const timeToAct: Record<string, number> = Object.fromEntries(tasks.map((t) => [t.id, 0]));
  const wrong = new Set<string>();
  const run = await replay(browser, loaded, viewport, 'light', loaded.events, plant);
  const { page, events, pacer } = run;
  // The index of each step's `StepCompleted`, and the truth once it is drawn. `stepsIn[n]` is the
  // steps among the first n events: the screen shows steps, so the events it owes are counted in
  // steps, and the run's last event after its last step is never owed to a reading.
  const stepAt = new Map<number, number>();
  const stepsIn = [0];
  events.forEach((e, i) => {
    if (e.body.kind === 'StepCompleted') {
      stepAt.set(e.body.value.step, i);
    }
    stepsIn.push((stepsIn[i] ?? 0) + (e.body.kind === 'StepCompleted' ? 1 : 0));
  });
  const lastStep = Math.max(...stepAt.keys());
  const truths = new Map<number, Truth>();
  const truthAt = (index: number) => {
    let t = truths.get(index);
    if (t === undefined) {
      t = truthOf(loaded.truth.recording, events.slice(0, index + 1));
      truths.set(index, t);
    }
    return t;
  };
  const caught = new Set<string>();
  try {
    let ended: number | null = null;
    const streaming = pacer.start(CADENCE_MS).then(() => {
      ended = performance.now();
    });
    for (;;) {
      for (const task of tasks) {
        const began = performance.now();
        const seen = await look(page, bot(task));
        const shown = seen.groups?.step;
        if (seen.groups === null || shown === undefined) {
          continue;
        }
        const index = stepAt.get(Number(shown));
        if (index === undefined) {
          wrong.add(`${task.id}: the screen shows step ${shown}, which the recording does not have`);
          continue;
        }
        const { correct, expected } = check(task, truthAt(index), seen.groups);
        if (!correct) {
          wrong.add(`${task.id}: read ${seen.source} while the run streamed, the truth at step ${shown} is ${JSON.stringify(expected)}`);
        }
        const owed = (stepsIn[pacer.sentAt(began - RENDER_BUDGET_MS)] ?? 0) - (stepsIn[index + 1] ?? 0);
        timeToAct[task.id] = Math.max(timeToAct[task.id] ?? 0, owed);
        if (Number(shown) === lastStep) {
          caught.add(task.id);
        }
      }
      const end: number | null = ended;
      if (end !== null && (caught.size === tasks.length || performance.now() - end > CATCH_UP_MS)) {
        break;
      }
    }
    await streaming;
  } finally {
    await run.close();
  }
  for (const task of tasks) {
    if (!caught.has(task.id)) {
      wrong.add(`${task.id}: the screen did not show the last step ${lastStep} within ${CATCH_UP_MS} ms of the stream's end`);
    }
  }
  return { timeToAct, wrong: [...wrong] };
}

