import type { Point } from './series.ts';

/** The drawing area of a chart, in SVG user units. */
export interface Frame {
  width: number;
  height: number;
  /** Space kept free on every side for the axis labels. */
  margin: { top: number; right: number; bottom: number; left: number };
}

export const FRAME: Frame = {
  width: 640,
  height: 220,
  margin: { top: 12, right: 16, bottom: 28, left: 64 },
};

/** The ranges a chart maps: steps across, values up. */
export interface Domain {
  steps: [number, number];
  values: [number, number];
}

/**
 * The step and value ranges of `points`, values from their finite ones. A range of one number is
 * widened so it still has a height; `null` when no point has a finite value.
 */
export function domain(points: readonly Point[]): Domain | null {
  let first = Infinity;
  let last = -Infinity;
  let low = Infinity;
  let high = -Infinity;
  for (const { step, value } of points) {
    first = Math.min(first, step);
    last = Math.max(last, step);
    if (value !== null && Number.isFinite(value)) {
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
  }
  if (low > high) {
    return null;
  }
  if (first === last) {
    first -= 1;
    last += 1;
  }
  if (low === high) {
    // At least the smallest step, so a subnormal value still widens; at most to ±MAX_VALUE.
    const pad = low === 0 ? 1 : Math.max(Math.abs(low) / 2, Number.MIN_VALUE);
    low = Math.max(low - pad, -Number.MAX_VALUE);
    high = Math.min(high + pad, Number.MAX_VALUE);
  }
  return { steps: [first, last], values: [low, high] };
}

/** Maps a step to its x and a value to its y within `frame`. */
export function project(frame: Frame, range: Domain): { x(step: number): number; y(value: number): number } {
  const { width, height, margin } = frame;
  const across = width - margin.left - margin.right;
  const up = height - margin.top - margin.bottom;
  const [first, last] = range.steps;
  const [low, high] = range.values;
  // Two finite values can be further apart than the largest finite one; their halves cannot.
  const span = high - low;
  const above = Number.isFinite(span)
    ? (value: number) => (value - low) / span
    : (value: number) => (value / 2 - low / 2) / (high / 2 - low / 2);
  return {
    x: (step) => margin.left + ((step - first) / (last - first)) * across,
    y: (value) => margin.top + (1 - above(value)) * up,
  };
}

/**
 * The `d` of the line through `points` in order: one `M` or `L` command per finite point, and a
 * new `M` after a point with no finite value, so the line has a gap there instead of a guess.
 */
export function line(frame: Frame, points: readonly Point[]): string {
  const range = domain(points);
  if (range === null) {
    return '';
  }
  const { x, y } = project(frame, range);
  const commands: string[] = [];
  let pen = false;
  for (const { step, value } of points) {
    if (value === null || !Number.isFinite(value)) {
      pen = false;
      continue;
    }
    commands.push(`${pen ? 'L' : 'M'}${round(x(step))},${round(y(value))}`);
    pen = true;
  }
  return commands.join(' ');
}

/** A value as the chart's labels show it: six significant digits, no trailing zeros. */
export function label(value: number | null): string {
  if (value === null) {
    return 'non-finite';
  }
  if (value === 0 || (Math.abs(value) >= 1e-3 && Math.abs(value) < 1e6)) {
    return String(Number(value.toPrecision(6)));
  }
  return value.toExponential(3);
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}
