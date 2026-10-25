// The loop's generic core: the metric functions and the ratchet of the ui-iteration-loop skill
// (§ 6), free of any project code. Nothing under `core/` imports the adapter or the app.

export type Theme = 'light' | 'dark';

/** A rectangle in CSS pixels of the viewport. */
export interface Box {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** The ratchet keys of one scenario; all lower is better. */
export type MetricKey =
  | 'actions_per_fix.max'
  | 'actions_per_fix.median'
  | 'glance_cost.median'
  | 'glance_cost.p95'
  | 'layout_shift.per_100_events'
  | 'reading_load.max'
  | 'reading_load.median'
  | 'screen_only.actions'
  | 'screen_only.looks'
  | 'screen_only.steers'
  | 'screen_only.time_to_act_steps.max';

export type ToleranceClass = 'counts' | 'steps' | 'layout_shift';

export interface Tolerance {
  absolute?: number;
  relative?: number;
}

export interface BaselineEntry {
  value: number;
  /** The tolerance class of the key, as the code's `KEYS` names it; a record, never read for the band. */
  tolerance: ToleranceClass;
  /** The decision that allowed this value to be worse than the one before it. */
  decision?: string;
}

export interface Baseline {
  about: string;
  scenarios: Record<string, Partial<Record<MetricKey, BaselineEntry>>>;
  tolerances: Record<ToleranceClass, Tolerance>;
}

export type Measured = Record<string, Partial<Record<MetricKey, number>>>;

/** The middle value of `values`, the mean of the two middle ones for an even count; 0 for none. */
export function median(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? (sorted[mid] as number) : ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
}

/** The nearest-rank `p`-th percentile of `values` (`p` in 0..100); 0 for none. */
export function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.max(1, Math.ceil((p / 100) * sorted.length));
  return sorted[rank - 1] as number;
}

export function max(values: readonly number[]): number {
  return values.length === 0 ? 0 : Math.max(...values);
}

/** The words of `text`: its runs of non-whitespace. */
export function words(text: string): number {
  const found = text.trim().match(/\S+/g);
  return found === null ? 0 : found.length;
}

/** Layout shift per 100 replayed events of a replay of `events` events. */
export function per100Events(shift: number, events: number): number {
  if (events === 0) {
    return 0;
  }
  return round((shift / events) * 100, 6);
}

export function round(n: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(n * f) / f;
}

/**
 * Where an answer was read: glance 1 inside the primary card, 2 elsewhere in the viewport, 3 not
 * in the viewport (the skill § 7).
 */
export function glance(answer: Box, primary: Box | null, viewport: { width: number; height: number }): 1 | 2 | 3 {
  const view = { left: 0, top: 0, right: viewport.width, bottom: viewport.height };
  if (!inside(answer, view)) {
    return 3;
  }
  if (primary !== null && inside(answer, primary)) {
    return 1;
  }
  return 2;
}

/** Whether `b` lies wholly within `outer`, to half a pixel. */
export function inside(b: Box, outer: Box): boolean {
  const e = 0.5;
  return b.left >= outer.left - e && b.right <= outer.right + e && b.top >= outer.top - e && b.bottom <= outer.bottom + e;
}

/** The band around a baseline value within which a measured value counts as unchanged. */
export function band(value: number, tolerance: Tolerance): number {
  return Math.max(tolerance.absolute ?? 0, (tolerance.relative ?? 0) * Math.abs(value));
}

export interface Problem {
  scenario: string;
  key: string;
  kind: 'missing' | 'worse' | 'improved' | 'baseline-worse' | 'tolerance';
  message: string;
}

/** Whether `decision` names a decision: a string with something other than whitespace in it. */
export function isDecision(decision: string | null | undefined): decision is string {
  return typeof decision === 'string' && decision.trim() !== '';
}

/**
 * The ratchet (skill § 6): every problem of `measured` against `baseline`, and of `baseline`
 * against `main`, the baseline on the main branch when there is one. A value worse than the
 * baseline beyond its band, an improvement beyond its band not taken into the baseline, a value
 * missing from either side, a baseline value made worse than `main`'s without a decision id, and
 * a tolerance in `baseline` that disagrees with the code. The band of a key comes only from
 * `tolerances[keys[key]]`, the code's numbers, never from the file under test.
 */
export function ratchet(
  scenarios: readonly string[],
  keys: Record<MetricKey, ToleranceClass>,
  tolerances: Record<ToleranceClass, Tolerance>,
  baseline: Baseline,
  measured: Measured,
  main: Baseline | null,
): Problem[] {
  const problems: Problem[] = [];
  for (const cls of Object.keys(tolerances) as ToleranceClass[]) {
    const recorded = baseline.tolerances?.[cls];
    if (JSON.stringify(recorded ?? null) !== JSON.stringify(tolerances[cls])) {
      problems.push({
        scenario: '*',
        key: `tolerances.${cls}`,
        kind: 'tolerance',
        message: `baseline.json tolerances.${cls} is ${JSON.stringify(recorded ?? null)}, the code's is ${JSON.stringify(tolerances[cls])}`,
      });
    }
  }
  for (const scenario of scenarios) {
    for (const key of Object.keys(keys) as MetricKey[]) {
      const m = measured[scenario]?.[key];
      const b = baseline.scenarios[scenario]?.[key];
      if (m === undefined || !Number.isFinite(m)) {
        problems.push({ scenario, key, kind: 'missing', message: `${scenario} ${key}: no measured value` });
        continue;
      }
      if (b === undefined) {
        problems.push({ scenario, key, kind: 'missing', message: `${scenario} ${key}: no baseline value (measured ${m})` });
        continue;
      }
      if (b.tolerance !== keys[key]) {
        problems.push({
          scenario,
          key,
          kind: 'tolerance',
          message: `${scenario} ${key}: baseline.json names the tolerance class ${b.tolerance}, the code's KEYS names ${keys[key]}`,
        });
      }
      const tolerance = tolerances[keys[key]];
      const width = band(b.value, tolerance);
      if (m > b.value + width) {
        problems.push({
          scenario,
          key,
          kind: 'worse',
          message: `${scenario} ${key}: ${m} is worse than the baseline ${b.value} beyond its tolerance ${width}`,
        });
      } else if (m < b.value - width) {
        problems.push({
          scenario,
          key,
          kind: 'improved',
          message: `${scenario} ${key}: ${m} improves on the baseline ${b.value}; take it into baseline.json in this merge (UX_UPDATE=1 pnpm ux)`,
        });
      }
      const was = main?.scenarios[scenario]?.[key];
      if (main !== null && was !== undefined) {
        const mainWidth = band(was.value, tolerance);
        if (b.value > was.value + mainWidth && !isDecision(b.decision)) {
          problems.push({
            scenario,
            key,
            kind: 'baseline-worse',
            message: `${scenario} ${key}: the baseline ${b.value} is worse than main's ${was.value} and names no decision`,
          });
        }
      }
    }
  }
  return problems;
}

/**
 * `baseline` with every improvement of `measured` taken in and every missing value added; a worse
 * value is taken only with a `decision` (`isDecision`), which it records. The band is the code's
 * (`tolerances[keys[key]]`), and a value it writes carries the class `keys` names.
 */
export function update(
  scenarios: readonly string[],
  keys: Record<MetricKey, ToleranceClass>,
  tolerances: Record<ToleranceClass, Tolerance>,
  baseline: Baseline,
  measured: Measured,
  decision: string | null,
): Baseline {
  const next: Baseline = structuredClone(baseline);
  for (const scenario of scenarios) {
    const entries = (next.scenarios[scenario] ??= {});
    for (const key of Object.keys(keys) as MetricKey[]) {
      const m = measured[scenario]?.[key];
      if (m === undefined || !Number.isFinite(m)) {
        continue;
      }
      const b = entries[key];
      const tolerance = keys[key];
      if (b === undefined) {
        entries[key] = { value: m, tolerance };
        continue;
      }
      const width = band(b.value, tolerances[tolerance]);
      if (m < b.value - width) {
        entries[key] = { value: m, tolerance };
      } else if (m > b.value + width && isDecision(decision)) {
        entries[key] = { value: m, tolerance, decision: decision.trim() };
      }
    }
  }
  return next;
}
