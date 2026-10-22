---
format: aep.planning-md/3
id: review-result:adversary-m1-mlp-example-pass-1
kind: review-result
status: active
title: Adversary pass 1, story:m1-mlp-example
relations:
- reviews: story:m1-mlp-example
revision: 1
---
# Adversary pass 1 on story:m1-mlp-example

Verdict: CONFIRMED, 2 findings, both notes; no red case. `task ci` exit 0, 640 passed, 1 ignored.

- F1, `crates/modelspec-cli/tests/mnist_mlp.rs:178`: `trains_on_cpu` passed with lr 0.001 (final
  0.129) and 0.0001 (final 1.50), since it asserted only `last < first`. Fixed by the coordinator
  before the unit commit: `last < first * 1e-6` (observed ratio 2.8e-13); the release test passes
  (0.69 s).
- F2, `crates/modelspec-runtime/tests/gradients.rs:665`: f64 error 1.06e-7 against `TOL_F64` 1e-6,
  a 9.4x margin from finite-difference rounding; deterministic with the fixed seed. Not changed.

Attacked without a break: the golden derived by hand (loss scale 2/1000, `ReluGrad`, both
updates), the kink redraw and coordinate choice, the example against the benchmark shape, the CI
wiring of `check-examples`.

```findings
[
 {
  "file": "crates/modelspec-cli/tests/mnist_mlp.rs",
  "line": 178,
  "category": "mutant",
  "severity": "note",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "trains_on_cpu passes unchanged with lr 0.001 (final 0.129) or 0.0001 (final 1.50) since it asserts only last < first, so a runtime SGD scaling fault goes unseen; assert last < 1e-6 * first (observed ratio 2.8e-13)."
 },
 {
  "file": "crates/modelspec-runtime/tests/gradients.rs",
  "line": 665,
  "category": "property",
  "severity": "note",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "mnist_mlp_sampled f64 error is 1.06e-7 against TOL_F64 1e-6, a 9.4x margin made entirely of finite-difference rounding on a large loss, so a change of seed or value scale can make it flaky."
 }
]
```
