---
format: aep.planning-md/3
id: verification-report:independent-audit-0-1-1
kind: verification-report
status: draft
title: Independent audit of release 0.1.1
summary: 531+3 tests pass, checksum and docs current; train exits 0 on a NaN run
relations:
- reviews: epic:m0-semantic-core
revision: 1
---
# Independent audit of release 0.1.1

An independent audit ran on 2026-10-21 against the published release `0.1.1` at public commit
`16d3dd2f576450fcc71b07cfea2d7c2b01d1fb4a`. It changed nothing in the repository; one independent
Rust test (`independent_audit.rs`) ran in a managed audit tree that is archived. The evidence is
kept outside the repository, beside the audit's `results.json`; this report carries the figures.

## Results

| check | result |
|---|---|
| release binary checksum | matches the published `.sha256` asset (`9f0e57d4…4089`), re-read by the coordinator |
| upstream test suite | 531 passed, 0 failed, 0 ignored |
| independent tests | 3 passed, 0 failed |
| `modelspec-docs generate --check` | 11 generated files current |
| release binary and source build, example output | byte-identical |
| same-seed rerun | byte-identical on one Linux host |
| `linear_regression.mspec` | loss 9.693205 → 5.61049e-11 in 500 steps |
| two-layer relu demo | loss 1.3817453 → 0.018498179 in 200 steps |
| shared-parameter gradient, f64 | max scaled error 5.93e-11 |
| reduction gradient | max scaled error 4.83e-8 (f64), 3.11e-7 (f32) |
| custom dataset through the Rust API (`compile_training` + `sgd_step`) | loss 14.072917 → 5.87e-13 in 400 steps; weights [1.2499992, -1.9999996, 0.7500003] against [1.25, -2.0, 0.75]; 6 held-out examples, max absolute error 2.38e-6 |

CLI exit statuses observed: divergence 0, seed 43 0, bad shape 1, `--backend vulkan` 1, `tiny_mlp`
target syntax 1, external data flag 2, checkpoint flag 2, infer command 2.

## Finding

`modelspec train` on `linear_regression.mspec` with `sgd(lr = 100.0)` instead of `lr = 0.1` prints
`step 50 loss NaN` through `step 500 loss NaN`, ends with every parameter `NaN`, and exits 0. The
coordinator reproduced it with the 0.1.1 release binary on 2026-10-21: the same last two lines and
exit status 0. DESIGN.md § 8 lists no diagnostic for a run whose loss or parameters stop being
finite. Filed as `story:train-refuses-non-finite-loss`.

## Boundaries the audit confirmed

- `--backend vulkan` exits 1 with E-TARGET-001; the runtime `vulkan` feature is empty.
- `examples/tiny_mlp.mspec` exits 1; modules and generics are beyond M0.
- The CLI trains on generated teacher labels; it has no dataset import, checkpoint or resume, and no
  inference command. Custom data and held-out prediction work through the Rust library.
- No GPU performance, large-model or self-improvement result is established.
