import type { Histogram } from '../generated/observatory.ts';
import type { Frame } from './chart.ts';

/** One bin drawn as a bar, in SVG user units. */
export interface Bar {
  x: number;
  y: number;
  width: number;
  height: number;
  count: number;
}

/**
 * The bars of `counts` across the drawing area of `frame`, one per bin and all of one width, as
 * the bins are (`Histogram`: the edges are evenly spaced), each standing on the bottom of the
 * area and the fullest bin at its full height.
 */
export function bars(frame: Frame, counts: readonly number[]): Bar[] {
  const { width, height, margin } = frame;
  const across = width - margin.left - margin.right;
  const up = height - margin.top - margin.bottom;
  const bottom = height - margin.bottom;
  const most = Math.max(0, ...counts);
  const each = across / Math.max(counts.length, 1);
  return counts.map((count, i) => {
    const tall = most === 0 ? 0 : (count / most) * up;
    return { x: margin.left + i * each, y: bottom - tall, width: each, height: tall, count };
  });
}

/** The elements of `histogram` in no bin, by kind, or `null` when every element is in one. */
export function nonFinite(histogram: Histogram): string | null {
  const parts = [
    [histogram.nan_count, 'NaN'],
    [histogram.positive_infinity_count, '+inf'],
    [histogram.negative_infinity_count, '−inf'],
  ] as const;
  const named = parts.filter(([count]) => count > 0).map(([count, kind]) => `${count} ${kind}`);
  return named.length === 0 ? null : named.join(', ');
}
