// What both specs share: a scenario replayed through the pacing proxy into one page at one
// viewport and theme, and the run's settings from the environment.

import { deepStrictEqual } from 'node:assert';
import type { Browser, Page } from '@playwright/test';
import type { RunEvent } from '../src/generated/observatory.ts';
import { type Loaded, OUT_DIR, REPO_ROOT, UI_DIST, type Viewport, truthOf } from './adapter.ts';
import { installShiftObserver } from './core/page.ts';
import type { Theme } from './core/metrics.ts';
import { type Frame, type Origin, type Pacer, frame, observe, pace, recorded } from './core/replay.ts';

/** The `modelspec` binary the replays run: `MODELSPEC`, else this tree's debug build. */
export const MODELSPEC = process.env.MODELSPEC ?? `${REPO_ROOT}target/debug/modelspec`;
/** `UX_ROUND=R-nn`: also take the round's screenshots. */
export const ROUND = process.env.UX_ROUND ?? null;
export const SHOTS = `${OUT_DIR}shots/`;

export interface Replay {
  page: Page;
  events: RunEvent[];
  pacer: Pacer;
  close(): Promise<void>;
}

/** Fails when `truth.json` is not what the recording says. */
export function checkTruth(loaded: Loaded): void {
  deepStrictEqual(
    truthOf(loaded.truth.recording, loaded.events),
    loaded.truth,
    `scenarios/${loaded.scenario.name}/truth.json differs from ${loaded.truth.recording}`,
  );
}

/**
 * `loaded` served by `modelspec observe` and paced into a fresh page at `viewport` and `theme`;
 * `events` replaces the recorded ones (the failed variant), keeping their frames' ids.
 */
export async function replay(
  browser: Browser,
  loaded: Loaded,
  viewport: Viewport,
  theme: Theme,
  events: RunEvent[] = loaded.events,
): Promise<Replay> {
  const origin: Origin = await observe(MODELSPEC, REPO_ROOT, loaded.truth.recording, UI_DIST);
  const served = await recorded(origin);
  if (served.length !== loaded.events.length) {
    await origin.stop();
    throw new Error(`${loaded.truth.recording}: served ${served.length} frames for ${loaded.events.length} events`);
  }
  const script: Frame[] = events.map((event, i) =>
    event === loaded.events[i] ? (served[i] as Frame) : frame(event.body.kind, event.sequence, event),
  );
  const pacer = await pace(origin, script);
  const context = await browser.newContext({ viewport, reducedMotion: 'reduce' });
  const page = await context.newPage();
  await page.emulateMedia({ colorScheme: theme, reducedMotion: 'reduce' });
  await page.addInitScript(installShiftObserver);
  await page.goto(pacer.url);
  await pacer.connected;
  return {
    page,
    events,
    pacer,
    close: async () => {
      await context.close();
      await pacer.stop();
      await origin.stop();
    },
  };
}

/** A screenshot of the viewport 0.75 s after its moment, as a small JPEG under the run's shots. */
export async function shot(page: Page, name: string): Promise<string> {
  await page.waitForTimeout(750);
  const file = `${name}.jpg`;
  await page.screenshot({ path: `${SHOTS}${file}`, type: 'jpeg', quality: 70 });
  return file;
}
