// What both specs share: a scenario replayed through the pacing proxy into one page at one
// viewport and theme under the loop's font stack, and the run's settings from the environment.

import { execFileSync } from 'node:child_process';
import { deepStrictEqual } from 'node:assert';
import type { Browser, Page } from '@playwright/test';
import type { RunEvent } from '../src/generated/observatory.ts';
import { FONTS, type Loaded, OUT_DIR, REPO_ROOT, UI_DIST, type Viewport, truthOf } from './adapter.ts';
import { installShiftObserver } from './core/page.ts';
import type { Theme } from './core/metrics.ts';
import { type Frame, type Origin, type Pacer, frame, observe, pace, recorded } from './core/replay.ts';

/** The `modelspec` binary the replays run: `MODELSPEC`, else this tree's debug build. */
export const MODELSPEC = process.env.MODELSPEC ?? `${REPO_ROOT}target/debug/modelspec`;
/** `UX_ROUND=R-nn`: also take the round's screenshots. */
export const ROUND = process.env.UX_ROUND ?? null;
export const SHOTS = `${OUT_DIR}shots/`;

/** The stylesheet of `FONTS`, forced over the app's own font families. */
export const FONT_CSS = `html, body, body * { font-family: "${FONTS.text}" !important; } :is(${FONTS.codeSelector}), :is(${FONTS.codeSelector}) * { font-family: "${FONTS.code}" !important; }`;

/** One line naming what `fc-list` lacks of `FONTS`, or `null` when both families are installed. */
export function missingFonts(): string | null {
  let listed: string[];
  try {
    listed = execFileSync('fc-list', [':', 'family'], { stdio: ['ignore', 'pipe', 'ignore'] })
      .toString()
      .split('\n')
      .flatMap((line) => line.split(','))
      .map((f) => f.trim());
  } catch {
    return `pnpm ux: fc-list did not run; the gates need the fonts "${FONTS.text}" and "${FONTS.code}" installed (fontconfig)`;
  }
  const lacking = [FONTS.text, FONTS.code].filter((f) => !listed.includes(f));
  return lacking.length === 0 ? null : `pnpm ux: fc-list lists no ${lacking.map((f) => `"${f}"`).join(' and no ')}; install them (fonts-dejavu-core) so every machine measures the same words`;
}

/** Adds `css` to the document before its first paint (runs in the page, from an init script). */
function addStyle(css: string): void {
  const style = document.createElement('style');
  style.textContent = css;
  const put = () => (document.head ?? document.documentElement).appendChild(style);
  if (document.documentElement !== null) {
    put();
  } else {
    document.addEventListener('readystatechange', put, { once: true });
  }
}

/**
 * The fonts the page draws with, from the browser itself (`CSS.getPlatformFontsForNode`): one line
 * per element of `FONTS` that is drawn with another family; the header title for text, the run id
 * for code when the page shows one.
 */
export async function wrongFonts(page: Page): Promise<string[]> {
  const cdp = await page.context().newCDPSession(page);
  try {
    await cdp.send('DOM.enable');
    await cdp.send('CSS.enable');
    const { root } = await cdp.send('DOM.getDocument', { depth: -1 });
    const wrong: string[] = [];
    const probes: { selector: string; family: string }[] = [
      { selector: '[data-ux-region="header"] h1', family: FONTS.text },
      { selector: '[data-ux-region="header"] .run', family: FONTS.code },
    ];
    for (const { selector, family } of probes) {
      const { nodeId } = await cdp.send('DOM.querySelector', { nodeId: root.nodeId, selector });
      if (nodeId === 0) {
        continue;
      }
      const { fonts } = await cdp.send('CSS.getPlatformFontsForNode', { nodeId });
      const used = fonts.map((f) => f.familyName);
      if (used.length === 0 || used.some((f) => f !== family)) {
        wrong.push(`${selector} is drawn with ${used.join(', ') || 'no font'}, not "${family}"`);
      }
    }
    return wrong;
  } finally {
    await cdp.detach();
  }
}

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

/** A defect planted into the page by the gates' self-tests, never into the app. */
export interface Plant {
  /** A stylesheet added after `FONT_CSS`, before the first paint. */
  css?: string;
  /** Every server-sent event reaches the app this many milliseconds late: a render lag. */
  lagMs?: number;
}

/** Delays every `EventSource` message by `ms` (runs in the page, from an init script). */
function delayEvents(ms: number): void {
  const add = EventTarget.prototype.addEventListener;
  EventSource.prototype.addEventListener = function (
    this: EventSource,
    type: string,
    listener: EventListenerOrEventListenerObject | null,
    options?: boolean | AddEventListenerOptions,
  ): void {
    if (listener === null || type === 'open' || type === 'error') {
      add.call(this, type, listener, options);
      return;
    }
    const late = (event: Event) => {
      setTimeout(() => (typeof listener === 'function' ? listener.call(this, event) : listener.handleEvent(event)), ms);
    };
    add.call(this, type, late, options);
  } as typeof EventSource.prototype.addEventListener;
}

/**
 * `loaded` served by `modelspec observe` and paced into a fresh page at `viewport` and `theme`
 * under `FONT_CSS`; `events` replaces the recorded ones (the failed variant), keeping their
 * frames' ids; `plant` is a defect of a self-test.
 */
export async function replay(
  browser: Browser,
  loaded: Loaded,
  viewport: Viewport,
  theme: Theme,
  events: RunEvent[] = loaded.events,
  plant: Plant | null = null,
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
  await page.addInitScript(addStyle, plant?.css === undefined ? FONT_CSS : `${FONT_CSS}\n${plant.css}`);
  if (plant?.lagMs !== undefined) {
    await page.addInitScript(delayEvents, plant.lagMs);
  }
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
