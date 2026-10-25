// `pnpm ux`: the hermetic levels L2 and L3 of the observatory UI loop and the ratchet after them
// (story:observatory-ui-iteration-loop). The browser is the executable `CHROME` names, as the Rust
// browser tests take it; Playwright downloads nothing. No network beyond 127.0.0.1.

import { rmSync } from 'node:fs';
import { defineConfig } from '@playwright/test';
import { OUT_DIR } from './adapter.ts';

const chrome = process.env.CHROME;
if (chrome === undefined || chrome === '') {
  console.error('pnpm ux: set CHROME to a Chrome or Chromium executable, as task ui-browser takes it');
  process.exit(2);
}

// Each run starts from no fragments and no shots, so the ratchet reads only this run's;
// `pnpm ux --project ratchet --no-deps` runs the ratchet alone on the last run's.
if (process.env.TEST_WORKER_INDEX === undefined && !process.argv.includes('--no-deps')) {
  rmSync(`${OUT_DIR}fragments`, { recursive: true, force: true });
  rmSync(`${OUT_DIR}shots`, { recursive: true, force: true });
}

export default defineConfig({
  testDir: '.',
  outputDir: `${OUT_DIR}test-results`,
  fullyParallel: true,
  forbidOnly: true,
  retries: 0,
  timeout: 300_000,
  reporter: [['list']],
  use: {
    headless: true,
    launchOptions: { executablePath: chrome },
    trace: 'off',
    screenshot: 'off',
    video: 'off',
  },
  projects: [
    { name: 'gates', testMatch: ['layout.spec.ts', 'screen-only.spec.ts'] },
    { name: 'ratchet', testMatch: ['ratchet.spec.ts'], dependencies: ['gates'] },
  ],
});
