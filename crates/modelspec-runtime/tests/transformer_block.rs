//! The pure part of the transformer-block benchmark (story:transformer-block-benchmark), shared
//! with the bench `benches/transformer_block.rs`: the three points and their sources, read from
//! `examples/transformer_block.mspec` with the batch substituted, the matmul-only FLOP count of
//! the block, and the result line.
//!
//! With the `vulkan` feature, every point's source compiles for Vulkan on the RTX 3090 profile of
//! `tests/fixtures/rtx3090.device.json` without a device, has a SPIR-V module per kernel, counts
//! every dispatch of its step plan (no `uncounted` site in `train --profile`), its step plan's
//! dense matmuls count the dense part of the FLOP rule, and its plan does not depend on the step
//! count the bench gives each run.

#[allow(dead_code)]
#[path = "../benches/mlp_sweep/sweep.rs"]
mod sweep;

#[allow(dead_code)]
#[path = "../benches/transformer_block/block.rs"]
mod block;

use block::{BATCHES, BlockResult, EXAMPLE, Outcome, Point, points};
use sweep::{Build, TimedRun};

fn example_file() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/transformer_block.mspec"
    );
    std::fs::read_to_string(path).expect("examples/transformer_block.mspec")
}

#[test]
fn the_points_are_batches_8_32_and_128_sequences_of_128() {
    assert_eq!(BATCHES, [8, 32, 128]);
    let points = points();
    assert_eq!(
        points.iter().map(|p| p.batch).collect::<Vec<_>>(),
        [8, 32, 128]
    );
    for point in points {
        assert_eq!(point.tokens_per_step(), point.batch * 128);
    }
}

/// The bench runs the example itself: its source at the file's batch is the file, byte for byte,
/// and [`EXAMPLE`] is what the file holds now.
#[test]
fn the_source_at_batch_8_is_the_example_file() {
    let file = example_file();
    assert_eq!(EXAMPLE, file);
    assert_eq!(Point { batch: 8 }.source().unwrap(), file);
}

#[test]
fn a_source_substitutes_the_batch_in_every_declaration_and_nothing_else() {
    let file = example_file();
    for batch in [32, 128] {
        let source = Point { batch }.source().unwrap();
        let rows = format!("[{}, 128]", batch * 128);
        let mask = format!("[{batch}, 128]");
        assert!(!source.contains("[1024, 128]"), "{batch}");
        assert!(!source.contains("[8, 128]"), "{batch}");
        assert_eq!(source.lines().count(), file.lines().count(), "{batch}");
        let (mut row_uses, mut mask_uses) = (0, 0);
        for (was, now) in file.lines().zip(source.lines()) {
            row_uses += was.matches("[1024, 128]").count();
            let masks = was.matches("[8, 128]").count();
            assert!(masks == 0 || was.contains("key_mask"), "{was}");
            mask_uses += masks;
            let expected = was.replace("[1024, 128]", &rows).replace("[8, 128]", &mask);
            assert_eq!(expected, now, "{batch}");
        }
        assert_eq!((row_uses, mask_uses), (4, 2), "{batch}");
        assert_eq!(source.matches(&rows).count(), 4, "{batch}");
    }
}

/// The closed form `N (42 H² + 24 H F + 24 S H)` with `N = 128 B`, `H = 128`, `F = 512`,
/// `S = 128`: 2,654,208 FLOP per row, 339,738,624 per sequence.
#[test]
fn flop_per_step_is_the_block_s_matmul_count() {
    let at_8 = Point { batch: 8 };
    assert_eq!(at_8.dense_flop(), 1024 * (42 * 128 * 128 + 24 * 128 * 512));
    assert_eq!(at_8.attention_flop(), 1024 * 24 * 128 * 128);
    assert_eq!(at_8.flop_per_step(), 2_717_908_992);
    assert_eq!(Point { batch: 32 }.flop_per_step(), 10_871_635_968);
    assert_eq!(Point { batch: 128 }.flop_per_step(), 43_486_543_872);
    for point in points() {
        assert_eq!(point.flop_per_step(), 339_738_624 * point.batch);
    }
}

fn measured(batch: u64, median_us: f64) -> BlockResult {
    let runs = vec![
        TimedRun {
            us_per_step: median_us,
            seconds: 5.0,
            span_unix_ms: [0, 5000],
        };
        3
    ];
    BlockResult::measured("modelspec-vulkan", Point { batch }, 1000, runs, None)
}

/// Three distinct runs whose median, 2000 µs, is not their mean (3000 µs), in two orders that
/// put the median last and first, so neither the mean nor a run picked by position gives these
/// figures (ADV-TB-1).
#[test]
fn tokens_per_second_and_gflops_are_over_the_median_step_time() {
    for order in [[6500.0, 500.0, 2000.0], [2000.0, 6500.0, 500.0]] {
        let runs = order
            .iter()
            .map(|&us_per_step| TimedRun {
                us_per_step,
                seconds: 5.0,
                span_unix_ms: [0, 5000],
            })
            .collect();
        let result =
            BlockResult::measured("modelspec-vulkan", Point { batch: 8 }, 1000, runs, None);
        let Outcome::Measured {
            median_us_per_step,
            tokens_per_s,
            gflops,
            ..
        } = result.outcome
        else {
            panic!("measured");
        };
        assert_eq!(median_us_per_step, 2000.0, "{order:?}");
        assert_eq!(tokens_per_s, 1024.0 * 1e6 / 2000.0, "{order:?}");
        assert_eq!(gflops, 2_717_908_992.0 / (2000.0 * 1e3), "{order:?}");
    }
}

/// The line the bench writes and the opponent writes too: its fields in this order.
#[test]
fn a_result_line_has_the_fields_of_the_brief_in_order_and_round_trips() {
    let result = measured(32, 5000.0).with_build(Build {
        version: Some("0.20.0".into()),
        commit: Some("abc".into()),
        dirty: Some(false),
    });
    let line = result.to_line();
    let value: serde_json::Value = serde_json::from_str(&line).unwrap();
    let keys: Vec<&str> = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    let mut expected = vec![
        "side",
        "batch",
        "seq",
        "tokens_per_step",
        "flop_per_step",
        "flop_rule",
        "build",
        "status",
        "steps",
        "runs",
        "median_us_per_step",
        "tokens_per_s",
        "gflops",
    ];
    expected.sort_unstable();
    assert_eq!(sorted, expected, "{line}");
    let order: Vec<usize> = [
        "\"side\"",
        "\"batch\"",
        "\"seq\"",
        "\"tokens_per_step\"",
        "\"flop_per_step\"",
        "\"flop_rule\"",
        "\"build\"",
        "\"status\"",
        "\"steps\"",
        "\"runs\"",
        "\"median_us_per_step\"",
        "\"tokens_per_s\"",
        "\"gflops\"",
    ]
    .iter()
    .map(|key| line.find(key).unwrap_or_else(|| panic!("{key} in {line}")))
    .collect();
    assert!(order.windows(2).all(|w| w[0] < w[1]), "{line}");
    assert_eq!(value["seq"], 128);
    assert_eq!(value["tokens_per_step"], 4096);
    assert_eq!(value["flop_rule"], "matmul-only");
    assert_eq!(value["status"], "measured");
    assert_eq!(BlockResult::from_line(&line).unwrap(), result);
}

/// A line of the opponent's shape, written by hand, parses as a result.
#[test]
fn an_opponent_line_written_by_hand_parses() {
    let line = r#"{"side": "pytorch-compile", "batch": 128, "seq": 128, "tokens_per_step": 16384, "flop_per_step": 43486543872, "flop_rule": "matmul-only", "device": "a GPU", "build": {"commit": "abc", "dirty": true}, "status": "measured", "steps": 50, "runs": [{"us_per_step": 1.0, "seconds": 5.0, "span_unix_ms": [1, 2]}], "median_us_per_step": 1.0, "tokens_per_s": 16384000000.0, "gflops": 43486.5}"#;
    let result = BlockResult::from_line(line).unwrap();
    assert_eq!(result.side, "pytorch-compile");
    assert_eq!(result.tokens_per_step, 16384);
}

#[test]
fn a_refusal_is_a_result_with_the_point_s_counts() {
    let result = BlockResult::refused(
        "modelspec-vulkan",
        Point { batch: 128 },
        "E-TARGET-010".into(),
        None,
    );
    let line = result.to_line();
    assert!(line.contains(r#""status":"refused""#), "{line}");
    assert!(line.contains(r#""refusal":"E-TARGET-010""#), "{line}");
    assert!(line.contains(r#""flop_per_step":43486543872"#), "{line}");
    assert!(line.contains(r#""tokens_per_step":16384"#), "{line}");
}

#[cfg(feature = "vulkan")]
mod rtx3090 {
    use modelspec_core::artifact::Backend;
    use modelspec_core::compile::{compile_artifact_for, compile_training};
    use modelspec_core::device::DeviceProfile;
    use modelspec_core::ir::kernel::{DispatchCount, Kernel, KernelKind, PlanOp};
    use modelspec_runtime::spirv::emit::emit_kernels;
    use modelspec_runtime::vulkan::negotiate::negotiate;

    use super::block::{LAYERS, Point, points};
    use super::sweep::WARMUP_STEPS;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/rtx3090.device.json"
    ));

    fn profile() -> DeviceProfile {
        DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile")
    }

    /// Every point compiles for Vulkan on the profile, negotiates, has a SPIR-V module per kernel
    /// and buffers within a storage-buffer binding and the device-local heap; every dispatch of
    /// its step plan is counted; its `MatMul` dispatches are the dense part of the FLOP rule; and
    /// it has one `Attention`, `AttentionGradQ`, `AttentionGradK` and `AttentionGradV` dispatch
    /// per layer, the sites the attention part of the rule counts.
    #[test]
    fn every_point_compiles_for_the_rtx3090_profile_and_counts_every_dispatch() {
        let profile = profile();
        negotiate(&profile).expect("the fixture negotiates");
        let budget = profile
            .memory_heaps
            .iter()
            .filter(|heap| heap.device_local)
            .map(|heap| heap.budget_bytes.unwrap_or(heap.size_bytes))
            .max()
            .expect("a device-local heap");
        for point in points() {
            let name = point.file_name();
            let source = point.source().expect("a source");
            let artifact = compile_artifact_for(&name, &source, Backend::Vulkan, Some(&profile))
                .unwrap_or_else(|d| panic!("{name}: {d:?}"));
            let plan = &artifact.execution;
            let modules = emit_kernels(&artifact, &profile)
                .unwrap_or_else(|error| panic!("{name}: no SPIR-V: {error}"));
            assert_eq!(modules.len(), plan.kernels.len(), "{name}");
            let largest = plan.buffers.iter().copied().max().unwrap_or(0);
            assert!(
                largest <= u64::from(profile.max_storage_buffer_range),
                "{name}: a buffer of {largest} bytes"
            );
            let total: u64 = plan.buffers.iter().sum();
            assert!(total <= budget, "{name}: {total} bytes of buffers");
            let dispatches: Vec<_> = plan
                .step
                .iter()
                .filter(|step| step.op == PlanOp::Dispatch)
                .collect();
            let uncounted: Vec<String> = dispatches
                .iter()
                .filter(|step| plan.dispatch_count(step) == Some(DispatchCount::Uncounted))
                .map(|step| {
                    let kernel = step.kernel_module_id.and_then(|id| plan.kernel(id));
                    format!("{:?}", kernel.map(|k| (k.kind, k.op)))
                })
                .collect();
            assert!(uncounted.is_empty(), "{name}: uncounted {uncounted:?}");
            let kernels: Vec<&Kernel> = dispatches
                .iter()
                .filter_map(|step| plan.kernel(step.kernel_module_id?))
                .collect();
            let matmul_flop: u64 = kernels
                .iter()
                .filter(|kernel| kernel.kind == KernelKind::MatMul)
                .map(|kernel| {
                    let (m, k, n) = Kernel::matmul_shape(&kernel.operands)
                        .unwrap_or_else(|| panic!("{name}: a matmul of static shapes"));
                    2 * m * k * n
                })
                .sum();
            assert_eq!(matmul_flop, point.dense_flop(), "{name}");
            for op in [
                "Attention",
                "AttentionGradQ",
                "AttentionGradK",
                "AttentionGradV",
            ] {
                let count = kernels.iter().filter(|kernel| kernel.op == op).count();
                assert_eq!(count as u64, LAYERS, "{name}: {op}");
            }
        }
    }

    /// The bench compiles a point once and gives the warm-up its [`WARMUP_STEPS`] in the
    /// configuration, not in the source, and each later run its own step count. That is the run
    /// of the source at those step counts because nothing else differs: the training step, the
    /// forward program, the device plan and its matmul limits are those of the source with its
    /// `steps` rewritten.
    #[test]
    fn a_point_s_plan_is_the_same_at_every_step_count() {
        let profile = profile();
        for point in [Point { batch: 8 }, Point { batch: 128 }] {
            let name = point.file_name();
            let source = point.source().expect("a source");
            assert_eq!(source.matches("    steps = 100;\n").count(), 1);
            let warm_source = source.replace("    steps = 100;\n", "    steps = 20;\n");
            let mut file = compile_training(&name, &source).expect("compiles");
            let warm = compile_training(&name, &warm_source).expect("compiles");
            file.config.steps = WARMUP_STEPS;
            assert_eq!(file.training, warm.training, "{name}");
            assert_eq!(file.forward, warm.forward, "{name}");
            assert_eq!(file.config, warm.config, "{name}");
            let artifact = |source: &str| {
                compile_artifact_for(&name, source, Backend::Vulkan, Some(&profile))
                    .expect("compiles for Vulkan")
            };
            let (a, b) = (artifact(&source), artifact(&warm_source));
            assert_eq!(a.execution, b.execution, "{name}");
            assert_eq!(a.matmul_limits, b.matmul_limits, "{name}");
        }
    }
}
