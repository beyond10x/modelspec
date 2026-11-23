---
format: aep.planning-md/3
id: story:benchmark-docs-drift-budget
kind: story
status: implemented
title: Keep benchmark documentation drift checks within the test budget
summary: Use the generator inputs once and retain every benchmark drift diagnostic within 60 seconds.
relations:
- serves: vision:modelspec
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-11-23T02:08:11Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-11-23T02:08:11Z", actor: "human:timo", revision: 3}
- {from: "active", to: "implemented", at: "2026-11-23T02:13:12Z", actor: "human:timo", revision: 4, decided_on: {"recorded":{"test_result":1}}}
---
# Fit the benchmark documentation drift check within the test budget

The documentation test for stale, missing and stray benchmark output timed out at
60.638 seconds in the 0.28.0 release checkout. The same test took 47.564 seconds
in the preceding 0.27.0 release check. Commit `bb15aaa2` introduced its five
repository copies and nine invocations of the documentation generator. The
generator reads benchmark records from `benchmarks/runs.yaml`. The raw timing
captures under `benchmarks/raw` are independent of the generated page.

## Acceptance

- A checked-in documentation fixture passes the generator's clean check.
- Editing a chart, editing the page, removing a chart, adding an unrecognized chart
  and changing a source measurement each produce the expected drift diagnostic.
- The benchmark drift test finishes within 60 seconds and the documentation suite
  finishes within 120 seconds.
