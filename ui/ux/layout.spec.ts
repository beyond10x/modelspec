// L2, the layout contract (story:observatory-ui-iteration-loop; the ui-iteration-loop skill § 2
// and § 4): every scenario at every gating viewport in both themes, one recorded event at a time,
// checked after the app has drawn each (`contract.ts`). Gate 1 no scroll, gate 2 always visible,
// gate 3 unmissable completion, gate 5 accessibility (axe-core at every key moment, no animation
// under reduced motion after any event, keyboard reach and focus indicators at the end). The
// failed end state is replayed on the rotating card and recorded, never ratcheted. The
// measured-only viewports are replayed at their key moments and reported, never gated.

import { expect, test } from '@playwright/test';
import {
  GATING_VIEWPORTS,
  MEASURED_VIEWPORTS,
  OUT_DIR,
  SCENARIOS,
  THEMES,
  type Viewport,
  failedVariant,
  loadScenario,
  viewportName,
} from './adapter.ts';
import { checkTruth } from './harness.ts';
import { contract } from './contract.ts';
import { writeFragment } from './core/round.ts';

for (const scenario of SCENARIOS) {
  test.describe(`L2 layout ${scenario.name}`, () => {
    for (const viewport of GATING_VIEWPORTS) {
      for (const theme of THEMES) {
        test(`${viewportName(viewport)} ${theme}`, async ({ browser }) => {
          const loaded = loadScenario(scenario);
          checkTruth(loaded);
          const { fragment, failures } = await contract(browser, loaded, viewport, theme, { gating: true });
          writeFragment(OUT_DIR, fragment);
          expect(failures, `${scenario.name} at ${viewportName(viewport)} ${theme}:\n${failures.join('\n')}`).toEqual([]);
        });
      }
    }

    test('measured-only viewports', async ({ browser }) => {
      const loaded = loadScenario(scenario);
      for (const viewport of MEASURED_VIEWPORTS) {
        for (const theme of THEMES) {
          const { fragment } = await contract(browser, loaded, viewport, theme, { gating: false });
          writeFragment(OUT_DIR, fragment);
        }
      }
    });
  });
}

test.describe('L2 completion failed', () => {
  const scenario = SCENARIOS.find((s) => s.card === 'rotating') ?? SCENARIOS[0];
  for (const theme of THEMES) {
    test(`${scenario?.name} ending in RunFailed ${theme}`, async ({ browser }) => {
      if (scenario === undefined) {
        throw new Error('no scenario');
      }
      const loaded = loadScenario(scenario);
      const viewport = GATING_VIEWPORTS[0] as Viewport;
      const { fragment, failures } = await contract(browser, loaded, viewport, theme, {
        gating: true,
        events: failedVariant(loaded.events),
        label: `${scenario.name}-failed`,
        failedEnd: true,
      });
      writeFragment(OUT_DIR, fragment);
      expect(failures, `${scenario.name} failed variant at ${viewportName(viewport)} ${theme}:\n${failures.join('\n')}`).toEqual([]);
    });
  }
});
