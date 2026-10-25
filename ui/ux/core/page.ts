// The in-page probes of the layout contract. Each function here runs inside the page through
// `page.evaluate` or `page.addInitScript`, so it closes over nothing and imports nothing.

import type { Box } from './metrics.ts';

declare global {
  interface Window {
    __ux?: { shifts: { value: number; sources: string[] }[] };
  }
}

/** Collects every layout shift (Layout Instability API) without recent input into `window.__ux`. */
export function installShiftObserver(): void {
  const store: { shifts: { value: number; sources: string[] }[] } = { shifts: [] };
  window.__ux = store;
  const describe = (node: Node | null): string => {
    if (!(node instanceof Element)) {
      return 'text';
    }
    const region = node.closest('[data-ux-region]')?.getAttribute('data-ux-region') ?? 'page';
    const cls = node.getAttribute('class');
    return `${node.tagName.toLowerCase()}${cls ? `.${cls.split(/\s+/).join('.')}` : ''} in ${region}`;
  };
  try {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const shift = entry as PerformanceEntry & {
          value: number;
          hadRecentInput: boolean;
          sources?: { node: Node | null }[];
        };
        if (!shift.hadRecentInput) {
          store.shifts.push({ value: shift.value, sources: (shift.sources ?? []).map((s) => describe(s.node)) });
        }
      }
    }).observe({ type: 'layout-shift', buffered: true });
  } catch {
    // A browser without the API reports no shift; the gate then measures 0.
  }
}

export interface ProbeArgs {
  /** Wait until this selector matches (at most `timeoutMs`), then for the frame to paint. */
  until: string | null;
  timeoutMs: number;
  alwaysVisible: { name: string; selector: string }[];
  stateSelector: string;
  alertSelector: string;
}

export interface ProbeResult {
  settled: boolean;
  /** Gate 1: every scroll or overflow, named. */
  scroll: string[];
  /** Gate 2: every always-visible element that is not wholly in view, named with the reason. */
  hidden: string[];
  /** The header's state word and whether the failure alert is in view. */
  stateText: string | null;
  stateInView: boolean;
  alertInView: boolean;
  /** The layout shift since the last probe, and its largest entry's sources. */
  shift: number;
  shiftSources: string[];
  /** `reading_load` at this moment. */
  words: number;
  viewport: { width: number; height: number };
}

/** The layout contract at this moment (runs in the page). */
export async function probe(args: ProbeArgs): Promise<ProbeResult> {
  const frame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  let settled = true;
  if (args.until !== null) {
    const start = performance.now();
    settled = false;
    while (performance.now() - start < args.timeoutMs) {
      if (document.querySelector(args.until) !== null) {
        settled = true;
        break;
      }
      await frame();
    }
  }
  await frame();
  await frame();
  await new Promise<void>((resolve) => setTimeout(resolve, 0));

  const vw = document.documentElement.clientWidth;
  const vh = document.documentElement.clientHeight;
  const describe = (el: Element): string => {
    const region = el.closest('[data-ux-region]')?.getAttribute('data-ux-region') ?? 'page';
    const own = el.getAttribute('data-ux-region');
    const cls = el.getAttribute('class');
    return `${el.tagName.toLowerCase()}${own ? `[data-ux-region=${own}]` : cls ? `.${cls.split(/\s+/).join('.')}` : ''} in ${region}`;
  };
  const rendered = (el: Element) => el.checkVisibility({ opacityProperty: true, visibilityProperty: true });

  const scroll: string[] = [];
  const root = document.scrollingElement ?? document.documentElement;
  if (root.scrollWidth > root.clientWidth + 0.5) {
    scroll.push(`horizontal page scroll: the page is ${root.scrollWidth} px wide in a ${root.clientWidth} px viewport`);
  }
  for (const el of Array.from(document.body.querySelectorAll('*'))) {
    if (!rendered(el)) {
      continue;
    }
    const style = getComputedStyle(el);
    const scrollsX = /auto|scroll/.test(style.overflowX) && el.scrollWidth > el.clientWidth + 1;
    const scrollsY = /auto|scroll/.test(style.overflowY) && el.scrollHeight > el.clientHeight + 1;
    if (scrollsX || scrollsY) {
      scroll.push(`scrollable element ${describe(el)} (${scrollsX ? 'x' : ''}${scrollsY ? 'y' : ''})`);
    }
    if (el instanceof HTMLElement && /hidden|clip/.test(style.overflowX) && el.scrollWidth > el.clientWidth + 1 && el.textContent?.trim() && !el.title) {
      scroll.push(`text cut off without its full value: ${describe(el)}`);
    }
    if (el instanceof SVGElement && !(el instanceof SVGSVGElement)) {
      // An SVG's own box clips what it draws; the box itself is checked.
      continue;
    }
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0 && (r.right > vw + 0.5 || r.left < -0.5)) {
      scroll.push(
        r.right > vw + 0.5
          ? `element past the viewport's right edge: ${describe(el)} ends at ${Math.round(r.right)} of ${vw} px`
          : `element past the viewport's left edge: ${describe(el)} starts at ${Math.round(r.left)} px`,
      );
    }
  }

  const inView = (el: Element): string | null => {
    if (!rendered(el)) {
      return 'not rendered';
    }
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) {
      return 'has no size';
    }
    if (r.left < -0.5 || r.top < -0.5 || r.right > vw + 0.5 || r.bottom > vh + 0.5) {
      return `not wholly in the viewport (${Math.round(r.left)},${Math.round(r.top)}..${Math.round(r.right)},${Math.round(r.bottom)} of ${vw}x${vh})`;
    }
    for (let a = el.parentElement; a !== null && a !== document.body; a = a.parentElement) {
      const s = getComputedStyle(a);
      if (s.overflowX !== 'visible' || s.overflowY !== 'visible') {
        const c = a.getBoundingClientRect();
        if (r.left < c.left - 0.5 || r.right > c.right + 0.5 || r.top < c.top - 0.5 || r.bottom > c.bottom + 0.5) {
          return `clipped by ${describe(a)}`;
        }
      }
    }
    const inset = 1;
    const points: [number, number][] = [
      [(r.left + r.right) / 2, (r.top + r.bottom) / 2],
      [r.left + inset, r.top + inset],
      [r.right - inset, r.top + inset],
      [r.left + inset, r.bottom - inset],
      [r.right - inset, r.bottom - inset],
    ];
    for (const [x, y] of points) {
      const hit = document.elementFromPoint(x, y);
      if (hit !== null && hit !== el && !el.contains(hit) && !hit.contains(el)) {
        return `covered at ${Math.round(x)},${Math.round(y)} by ${describe(hit)}`;
      }
    }
    return null;
  };

  const hidden: string[] = [];
  for (const { name, selector } of args.alwaysVisible) {
    const el = document.querySelector(selector);
    const problem = el === null ? 'missing' : inView(el);
    if (problem !== null) {
      hidden.push(`${name}: ${problem}`);
    }
  }

  const state = document.querySelector(args.stateSelector);
  const alert = document.querySelector(args.alertSelector);

  const store = window.__ux ?? { shifts: [] };
  const taken = store.shifts.splice(0);
  let shift = 0;
  let largest: { value: number; sources: string[] } | null = null;
  for (const s of taken) {
    shift += s.value;
    if (largest === null || s.value > largest.value) {
      largest = s;
    }
  }

  let words = 0;
  const count = (text: string) => (text.trim().match(/\S+/g) ?? []).length;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const range = document.createRange();
  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) {
    const text = node.textContent ?? '';
    const parent = node.parentElement;
    if (text.trim() === '' || parent === null || parent.closest('title, option, script, style, template') !== null || !rendered(parent)) {
      continue;
    }
    range.selectNodeContents(node);
    const seen = Array.from(range.getClientRects()).some((r) => r.width > 0 && r.height > 0 && r.right > 0 && r.bottom > 0 && r.left < vw && r.top < vh);
    if (seen) {
      words += count(text);
    }
  }
  for (const select of Array.from(document.querySelectorAll('select'))) {
    const r = select.getBoundingClientRect();
    if (rendered(select) && r.right > 0 && r.bottom > 0 && r.left < vw && r.top < vh) {
      words += count(select.selectedOptions[0]?.textContent ?? '');
    }
  }

  return {
    settled,
    scroll,
    hidden,
    stateText: state?.textContent?.trim() ?? null,
    stateInView: state !== null && inView(state) === null,
    alertInView: alert !== null && inView(alert) === null,
    shift,
    shiftSources: largest?.sources ?? [],
    words,
    viewport: { width: vw, height: vh },
  };
}

/** The bounding box of the first element matching `selector`, or `null` (runs in the page). */
export function boxOf(selector: string): Box | null {
  const el = document.querySelector(selector);
  if (el === null) {
    return null;
  }
  const r = el.getBoundingClientRect();
  return { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
}

/**
 * Every interactive element (`a[href]`, `button`, `input`, `select`, `textarea`, `[tabindex]`)
 * that `Tab` presses from the top of the page do not reach, and every one reached without a
 * visible focus indicator (runs in the page, between the presses, through `focusState`).
 */
export function focusables(): string[] {
  const all = Array.from(
    document.querySelectorAll('a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])'),
  ).filter((el) => el.checkVisibility() && !(el as HTMLButtonElement).disabled);
  return all.map((el, i) => {
    el.setAttribute('data-ux-focus', String(i));
    return String(i);
  });
}

/** The `data-ux-focus` of the focused element and whether it shows a focus indicator. */
export function focusState(): { id: string | null; indicated: boolean } {
  const el = document.activeElement;
  if (el === null || el === document.body) {
    return { id: null, indicated: false };
  }
  const s = getComputedStyle(el);
  const outline = s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0;
  const ring = s.boxShadow !== 'none';
  return { id: el.getAttribute('data-ux-focus'), indicated: outline || ring };
}

/** Whether any animation or transition runs (runs in the page). */
export function animations(): number {
  return document.getAnimations().length;
}
