import assert from 'node:assert/strict';
import { test } from 'node:test';
import { FRAME, domain, label, line } from './chart.ts';

function commands(d: string): string[] {
  return d.split(' ').filter((c) => c.length > 0);
}

test('the line has one command per finite point', () => {
  const d = line(FRAME, [
    { step: 0, value: 3 },
    { step: 1, value: 2 },
    { step: 2, value: 1 },
  ]);
  const all = commands(d);
  assert.equal(all.length, 3);
  assert.ok(all[0]?.startsWith('M'));
  assert.ok(all[1]?.startsWith('L'));
  assert.ok(all[2]?.startsWith('L'));
});

test('a point with no value breaks the line instead of being drawn', () => {
  const all = commands(
    line(FRAME, [
      { step: 0, value: 3 },
      { step: 1, value: null },
      { step: 2, value: 1 },
      { step: 3, value: 0.5 },
    ]),
  );
  assert.deepEqual(
    all.map((c) => c[0]),
    ['M', 'M', 'L'],
  );
});

test('the first and last steps span the drawing area, the low and high values its height', () => {
  const { width, height, margin } = FRAME;
  const all = commands(
    line(FRAME, [
      { step: 10, value: 4 },
      { step: 20, value: 2 },
    ]),
  );
  assert.equal(all[0], `M${margin.left},${margin.top}`);
  assert.equal(all[1], `L${width - margin.right},${height - margin.bottom}`);
});

test('one point or one value still has a range', () => {
  assert.deepEqual(domain([{ step: 5, value: 2 }]), { steps: [4, 6], values: [1, 3] });
  assert.deepEqual(domain([{ step: 5, value: 0 }]), { steps: [4, 6], values: [-1, 1] });
  assert.equal(domain([{ step: 5, value: null }]), null);
  assert.equal(line(FRAME, []), '');
});

test('every finite value, however large, small or far apart, is drawn inside the frame', () => {
  const max = Number.MAX_VALUE;
  const tiny = Number.MIN_VALUE;
  const cases: number[][] = [
    [-max, max],
    [max],
    [-max],
    [max, max / 2],
    [tiny],
    [0, tiny],
    [-tiny, tiny],
    [-max, 0, tiny, max],
  ];
  const { width, height, margin } = FRAME;
  for (const values of cases) {
    const all = commands(line(FRAME, values.map((value, step) => ({ step, value }))));
    assert.equal(all.length, values.length, `one command per value of ${values}`);
    for (const command of all) {
      const [x, y] = command.slice(1).split(',').map(Number);
      assert.ok(
        x >= margin.left && x <= width - margin.right && y >= margin.top && y <= height - margin.bottom,
        `${command} of ${values} is inside the frame`,
      );
    }
  }
});

test('labels keep six significant digits', () => {
  assert.equal(label(0.123456789), '0.123457');
  assert.equal(label(0), '0');
  assert.equal(label(5.610489850482736e-11), '5.610e-11');
  assert.equal(label(null), 'non-finite');
});
