// The screen-only user of the ui-iteration-loop skill (§ 7): deterministic code, no model. It
// reads the page only through the accessibility tree, by role and accessible name, or by role
// and the text the tree exposes for a role that takes no name from its content; it uses only
// elements wholly inside the viewport and not covered at their centre, and only values drawn as
// text the eye can see (`textSeen`: in view, uncovered, not transparent); a key press is one action.
// Its only movement is the keyboard: `PageDown`, and `Home` from the bottom of the page, which
// reverses its course and so counts as a steer.

import type { Locator, Page } from '@playwright/test';
import { type Box, glance, inside } from './metrics.ts';

export interface BotTask {
  id: string;
  role: string;
  /** A regular expression the accessible name matches. */
  name?: string;
  /** A regular expression the exposed text matches, for a role without a name from content. */
  text?: string;
  /** A regular expression with named groups, matched against that name or text. */
  read: string;
}

export interface Reading {
  /** The named groups of `read`, or `null` when no visible element answered. */
  groups: Record<string, string> | null;
  /** The name or text the groups were read from. */
  source: string | null;
  box: Box | null;
}

export interface Attempt extends Reading {
  task: string;
  actions: number;
  looks: number;
  steers: number;
  /** Where the answer was at the first look: 1 primary card, 2 elsewhere in view, 3 not in view. */
  glance: 1 | 2 | 3;
  keys: string[];
}

type AriaRole = Parameters<Page['getByRole']>[0];

/** The first line of an aria snapshot, `- role "name": text`, as `name: text` without the role. */
export function exposed(snapshot: string): string {
  let line = (snapshot.split('\n')[0] ?? '').replace(/^\s*-\s*/, '');
  // A YAML key with a colon in it is single-quoted: `- 'figure "a: b"':`.
  const key = /^'((?:[^']|'')*)'(.*)$/.exec(line);
  if (key !== null) {
    line = (key[1] ?? '').replace(/''/g, "'") + (key[2] ?? '');
  }
  const scalar = (s: string) => {
    const t = s.trim();
    if (t.startsWith('"') && t.endsWith('"')) {
      return t.slice(1, -1).replace(/\\(.)/g, '$1');
    }
    if (t.startsWith("'") && t.endsWith("'")) {
      return t.slice(1, -1).replace(/''/g, "'");
    }
    return t;
  };
  const body = line.replace(/^[\w-]+\s*/, '');
  const named = /^"((?:[^"\\]|\\.)*)"(?:\s*\[[^\]]*\])*(?::\s*(.*))?$/.exec(body);
  if (named !== null) {
    const name = (named[1] ?? '').replace(/\\(.)/g, '$1');
    const text = scalar(named[2] ?? '');
    return text === '' ? name : `${name}: ${text}`;
  }
  return scalar(body.replace(/^(?:\[[^\]]*\]\s*)*:?/, ''));
}

function candidates(page: Page, task: BotTask): Locator {
  const role = task.role as AriaRole;
  return task.name === undefined ? page.getByRole(role) : page.getByRole(role, { name: new RegExp(task.name) });
}

/** Whether `locator` is wholly in the viewport and not covered at its centre; its box. */
async function visibleBox(page: Page, locator: Locator): Promise<Box | null> {
  const b = await locator.boundingBox();
  const viewport = page.viewportSize();
  if (b === null || viewport === null) {
    return null;
  }
  const box = { left: b.x, top: b.y, right: b.x + b.width, bottom: b.y + b.height };
  if (!inside(box, { left: 0, top: 0, right: viewport.width, bottom: viewport.height })) {
    return null;
  }
  const uncovered = await locator.evaluate((el, [x, y]) => {
    const hit = document.elementFromPoint(x as number, y as number);
    return hit !== null && (hit === el || el.contains(hit));
  }, [b.x + b.width / 2, b.y + b.height / 2]);
  return uncovered ? box : null;
}

/** Every element answering `task`, visible or not: its exposed name or text and its box. */
async function answers(page: Page, task: BotTask): Promise<{ locator: Locator; source: string; groups: Record<string, string> }[]> {
  const all = candidates(page, task);
  const count = Math.min(await all.count(), 50);
  const read = new RegExp(task.read);
  const text = task.text === undefined ? null : new RegExp(task.text);
  const found = [];
  for (let i = 0; i < count; i++) {
    const locator = all.nth(i);
    const source = exposed(await locator.ariaSnapshot());
    if (text !== null && !text.test(source)) {
      continue;
    }
    const match = read.exec(source);
    if (match !== null) {
      found.push({ locator, source, groups: { ...(match.groups ?? {}) } });
    }
  }
  return found;
}

/**
 * Whether every value of `values` is drawn inside `el` as text the eye can see (runs in the page).
 * A value is seen when some text node of the element's name or text holds it as a whole token and
 * the range of exactly those characters (`Range.getBoundingClientRect`) has a size and lies wholly
 * in the viewport; `elementFromPoint` at its centre is the text's element or inside it; the caret
 * positions at its left and right edges (`caretPositionFromPoint`) fall on that text node at the
 * value's first and last character, which a cover painted over it, a pseudo-element included,
 * moves; no element from the text's up to the root has opacity 0; and the text's element is
 * `visibility: visible` and its colour is not transparent. A figure's name is its `figcaption`.
 */
export function textSeen(el: Element, values: string[]): boolean {
  const root =
    el.localName === 'figure' && !el.hasAttribute('aria-label') && !el.hasAttribute('aria-labelledby')
      ? (el.querySelector(':scope > figcaption') ?? el)
      : el;
  const vw = document.documentElement.clientWidth;
  const vh = document.documentElement.clientHeight;
  const caret = (x: number, y: number): { node: Node; offset: number } | null => {
    const d = document as Document & { caretPositionFromPoint?: (x: number, y: number) => { offsetNode: Node; offset: number } | null };
    if (typeof d.caretPositionFromPoint === 'function') {
      const p = d.caretPositionFromPoint(x, y);
      return p === null ? null : { node: p.offsetNode, offset: p.offset };
    }
    const r = document.caretRangeFromPoint(x, y);
    return r === null ? null : { node: r.startContainer, offset: r.startOffset };
  };
  const seen = (node: Text, start: number, end: number): boolean => {
    const parent = node.parentElement;
    if (parent === null) {
      return false;
    }
    const range = document.createRange();
    range.setStart(node, start);
    range.setEnd(node, end);
    const b = range.getBoundingClientRect();
    if (b.width <= 0 || b.height <= 0 || b.left < -0.5 || b.top < -0.5 || b.right > vw + 0.5 || b.bottom > vh + 0.5) {
      return false;
    }
    const y = (b.top + b.bottom) / 2;
    const hit = document.elementFromPoint((b.left + b.right) / 2, y);
    if (hit === null || !(hit === parent || parent.contains(hit))) {
      return false;
    }
    const left = caret(b.left + 1, y);
    const right = caret(b.right - 1, y);
    if (left?.node !== node || right?.node !== node || left.offset !== start || right.offset !== end) {
      return false;
    }
    for (let a: Element | null = parent; a !== null; a = a.parentElement) {
      if (Number.parseFloat(getComputedStyle(a).opacity) === 0) {
        return false;
      }
    }
    const style = getComputedStyle(parent);
    const transparent = /^rgba\(.*,\s*0\)$/.test(style.color) || style.color === 'transparent';
    return style.visibility === 'visible' && !transparent;
  };
  const nodes: Text[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n !== null; n = walker.nextNode()) {
    const parent = n.parentElement;
    if (parent !== null && parent.closest('title, script, style, template, [aria-hidden="true"]') === null) {
      nodes.push(n as Text);
    }
  }
  const token = (c: string | undefined) => c !== undefined && /[0-9A-Za-z.]/.test(c);
  return values.every((value) => {
    if (value === '') {
      return true;
    }
    return nodes.some((node) => {
      const text = node.data;
      for (let i = text.indexOf(value); i >= 0; i = text.indexOf(value, i + 1)) {
        const end = i + value.length;
        if (!token(text[i - 1]) && !token(text[end]) && seen(node, i, end)) {
          return true;
        }
      }
      return false;
    });
  });
}

/**
 * What the screen shows for `task` without an action: the first answer whose element is wholly in
 * view and uncovered at its centre and whose every read value is text the eye can see
 * (`textSeen`), if any.
 */
export async function look(page: Page, task: BotTask): Promise<Reading> {
  for (const answer of await answers(page, task)) {
    const box = await visibleBox(page, answer.locator);
    if (box !== null && (await answer.locator.evaluate(textSeen, Object.values(answer.groups)))) {
      return { groups: answer.groups, source: answer.source, box };
    }
  }
  return { groups: null, source: null, box: null };
}

/** Whether any element of the page answers `task`, in view or not. */
export async function exists(page: Page, task: BotTask): Promise<boolean> {
  return (await answers(page, task)).length > 0;
}

/** The glance level of `task` now, or `null` when nothing on the page answers it. */
export async function glanceNow(page: Page, task: BotTask, primary: string): Promise<1 | 2 | 3 | null> {
  if (!(await exists(page, task))) {
    return null;
  }
  const seen = await look(page, task);
  if (seen.box === null) {
    return 3;
  }
  const viewport = page.viewportSize() ?? { width: 0, height: 0 };
  return glance(seen.box, await page.locator(primary).first().boundingBox().then((b) => (b === null ? null : { left: b.x, top: b.y, right: b.x + b.width, bottom: b.y + b.height })), viewport);
}

async function settleScroll(page: Page): Promise<void> {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        let last = -1;
        let still = 0;
        const tick = () => {
          if (window.scrollY === last) {
            still += 1;
          } else {
            still = 0;
            last = window.scrollY;
          }
          if (still >= 3) {
            resolve();
          } else {
            requestAnimationFrame(tick);
          }
        };
        requestAnimationFrame(tick);
      }),
  );
}

/**
 * Performs `task` from the page as it is: looks, and while no visible element answers, presses
 * `PageDown`, or `Home` at the bottom of the page, until one does or `limit` actions are spent.
 * While nothing on the page answers at all (the end state the task asks about is not there yet),
 * it waits `waitMs` and looks again instead of pressing a key, at most `limit` times; each wait is
 * a look.
 */
export async function perform(page: Page, task: BotTask, primary: string, limit: number, waitMs = 100): Promise<Attempt> {
  let actions = 0;
  let looks = 0;
  let steers = 0;
  let waits = 0;
  let first: 1 | 2 | 3 | null = null;
  const keys: string[] = [];
  for (;;) {
    looks += 1;
    const seen = await look(page, task);
    if (seen.groups === null && waits < limit && !(await exists(page, task))) {
      waits += 1;
      await page.waitForTimeout(waitMs);
      continue;
    }
    if (first === null) {
      first = (await glanceNow(page, task, primary)) ?? 3;
    }
    if (seen.groups !== null || actions >= limit) {
      return { task: task.id, ...seen, actions, looks, steers, glance: first, keys };
    }
    const atBottom = await page.evaluate(
      () => window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 1,
    );
    const key = atBottom ? 'Home' : 'PageDown';
    if (atBottom) {
      steers += 1;
    }
    await page.keyboard.press(key);
    keys.push(key);
    actions += 1;
    await settleScroll(page);
  }
}
