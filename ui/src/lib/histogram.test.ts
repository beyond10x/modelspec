import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { Histogram } from '../generated/observatory.ts';
import { FRAME } from './chart.ts';
import { bars, nonFinite } from './histogram.ts';

function histogram(counts: number[], nan = 0, positive = 0, negative = 0): Histogram {
  return {
    bin_edges: counts.length === 0 ? [] : counts.map((_, i) => i).concat([counts.length]),
    counts,
    nan_count: nan,
    positive_infinity_count: positive,
    negative_infinity_count: negative,
  };
}

test('32 bins are 32 bars of one width across the drawing area, the fullest bin at full height', () => {
  const counts = Array.from({ length: 32 }, (_, i) => (i === 7 ? 10 : i % 3));
  const drawn = bars(FRAME, counts);
  const { width, height, margin } = FRAME;
  const across = width - margin.left - margin.right;
  const up = height - margin.top - margin.bottom;
  assert.equal(drawn.length, 32);
  assert.deepEqual(
    drawn.map((b) => b.count),
    counts,
  );
  for (const [i, bar] of drawn.entries()) {
    assert.ok(Math.abs(bar.width - across / 32) < 1e-9, `bar ${i} width ${bar.width}`);
    assert.ok(Math.abs(bar.x - (margin.left + (i * across) / 32)) < 1e-9, `bar ${i} x ${bar.x}`);
    assert.ok(Math.abs(bar.y + bar.height - (height - margin.bottom)) < 1e-9, `bar ${i} stands on the axis`);
  }
  assert.ok(Math.abs((drawn[7]?.height ?? 0) - up) < 1e-9, 'the fullest bin is full height');
  assert.equal(drawn[0]?.height, 0, 'an empty bin has no height');
  assert.ok(Math.abs((drawn[1]?.height ?? 0) - up / 10) < 1e-9);
});

test('one bin is one bar the width of the area, and no bin is no bar', () => {
  const { width, margin } = FRAME;
  const one = bars(FRAME, [4]);
  assert.equal(one.length, 1);
  assert.ok(Math.abs((one[0]?.width ?? 0) - (width - margin.left - margin.right)) < 1e-9);
  assert.deepEqual(bars(FRAME, []), []);
  assert.deepEqual(
    bars(FRAME, [0, 0]).map((b) => b.height),
    [0, 0],
    'bins that are all empty draw nothing above the axis',
  );
});

test('the elements outside every bin are named by kind, and nothing when there are none', () => {
  assert.equal(nonFinite(histogram([1, 2])), null);
  assert.equal(nonFinite(histogram([1], 2)), '2 NaN');
  assert.equal(nonFinite(histogram([], 1, 3, 2)), '1 NaN, 3 +inf, 2 −inf');
  assert.equal(nonFinite(histogram([5], 0, 0, 1)), '1 −inf');
});
