//! The pure part of the MLP size sweep (`story:mlp-size-sweep-vs-pytorch-cuda`), shared with the
//! bench `benches/mlp_sweep.rs`: the 12 points and their sources, the matmul-only FLOP count,
//! the median, the result lines both sides write, and the launch-bound crossover rule of the
//! story's third acceptance line, checked on hand-made numbers.
//!
//! With the `vulkan` feature, every one of the 12 sources compiles for Vulkan, negotiates and has
//! a SPIR-V module for each kernel on the RTX 3090 profile of `tests/fixtures/rtx3090.device.json`,
//! without a device, and its step plan's matmuls count the FLOP the sweep reports.

#[allow(dead_code)]
#[path = "../benches/mlp_sweep/sweep.rs"]
mod sweep;

use sweep::{
    BATCHES, Crossover, HIDDEN, Outcome, PointResult, TimedRun, crossover, crossover_lines, median,
    points,
};

fn measured(side: &str, batch: u64, hidden: u64, median_us: f64) -> PointResult {
    let point = sweep::Point { batch, hidden };
    let runs = vec![
        TimedRun {
            us_per_step: median_us,
            seconds: 5.0,
            span_unix_ms: [0, 5000],
        };
        3
    ];
    PointResult::measured(side, point, 1000, runs, None)
}

#[test]
fn the_sweep_has_twelve_points_batch_major_widths_ascending() {
    let points = points();
    assert_eq!(points.len(), 12);
    assert_eq!(HIDDEN, [64, 256, 1024, 4096]);
    assert_eq!(BATCHES, [100, 1000, 8192]);
    let pairs: Vec<(u64, u64)> = points.iter().map(|p| (p.batch, p.hidden)).collect();
    let mut expected = Vec::new();
    for batch in BATCHES {
        for hidden in HIDDEN {
            expected.push((batch, hidden));
        }
    }
    assert_eq!(pairs, expected);
}

#[test]
fn flop_per_step_is_the_matmul_count_3196_b_h() {
    let point = sweep::Point {
        batch: 100,
        hidden: 64,
    };
    assert_eq!(point.flop_per_step(), 3196 * 100 * 64);
    assert_eq!(point.flop_per_step(), 20_454_400);
    let largest = sweep::Point {
        batch: 8192,
        hidden: 4096,
    };
    assert_eq!(largest.flop_per_step(), 107_239_964_672);
}

#[test]
fn a_source_names_the_point_shapes_and_the_step_count() {
    let source = sweep::Point {
        batch: 1000,
        hidden: 256,
    }
    .source(1234);
    for needle in [
        "Tensor<f32, [784, 256]>",
        "Tensor<f32, [256, 10]>",
        "Tensor<f32, [1000, 784]>",
        "Tensor<f32, [1000, 10]>",
        "steps = 1234;",
        "seed = 42;",
        "sgd(lr = 0.001)",
        "mse(model.forward(x), y)",
    ] {
        assert!(source.contains(needle), "{needle} not in\n{source}");
    }
}

#[test]
fn the_median_of_three_is_the_middle_value() {
    assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    assert_eq!(median(&[10.0, 30.0, 20.0, 40.0]), Some(25.0));
    assert_eq!(median(&[]), None);
}

#[test]
fn gflops_is_flop_per_step_over_the_median_step_time() {
    let point = sweep::Point {
        batch: 1000,
        hidden: 1024,
    };
    let runs = [120.0, 100.0, 110.0]
        .map(|us| TimedRun {
            us_per_step: us,
            seconds: 6.0,
            span_unix_ms: [1, 6001],
        })
        .to_vec();
    let result = PointResult::measured("modelspec-vulkan", point, 50_000, runs, None);
    assert_eq!(result.flop_per_step, 3196 * 1000 * 1024);
    assert_eq!(result.flop_rule, "matmul-only");
    let Outcome::Measured {
        median_us_per_step,
        gflops,
        steps,
        ..
    } = result.outcome
    else {
        panic!("a measured point: {result:?}");
    };
    assert_eq!(steps, 50_000);
    assert_eq!(median_us_per_step, 110.0);
    let expected = 3196.0 * 1000.0 * 1024.0 / (110.0 * 1e3);
    assert!((gflops - expected).abs() < 1e-9, "{gflops} vs {expected}");
}

#[test]
fn a_result_line_round_trips_and_has_the_shape_the_opponent_writes() {
    let line = measured("pytorch-eager", 100, 64, 50.0).to_line();
    assert!(!line.contains('\n'), "{line}");
    let value: serde_json::Value = serde_json::from_str(&line).unwrap();
    for key in [
        "side",
        "batch",
        "hidden",
        "flop_per_step",
        "flop_rule",
        "status",
        "steps",
        "runs",
        "median_us_per_step",
        "gflops",
    ] {
        assert!(value.get(key).is_some(), "{key} missing in {line}");
    }
    assert_eq!(value["status"], "measured");
    assert_eq!(value["runs"][0]["us_per_step"], 50.0);
    assert_eq!(value["runs"][0]["seconds"], 5.0);
    assert_eq!(value["runs"][0]["span_unix_ms"][1], 5000);
    assert_eq!(
        PointResult::from_line(&line).unwrap(),
        measured("pytorch-eager", 100, 64, 50.0)
    );

    let refused = PointResult::refused(
        "modelspec-vulkan",
        sweep::Point {
            batch: 8192,
            hidden: 4096,
        },
        "E-TARGET-010: a tensor is too large for the vulkan target".into(),
        None,
    );
    let line = refused.to_line();
    let value: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(value["status"], "refused");
    assert_eq!(
        value["refusal"],
        "E-TARGET-010: a tensor is too large for the vulkan target"
    );
    assert_eq!(PointResult::from_line(&line).unwrap(), refused);
}

/// A line as `opponents/mlp_sweep/mlp_sweep.py` writes it, by hand.
#[test]
fn an_opponent_line_written_by_hand_parses() {
    let line = r#"{"side": "pytorch-compile", "batch": 1000, "hidden": 4096, "flop_per_step": 13090816000, "flop_rule": "matmul-only", "device": "NVIDIA GeForce RTX 3090; torch 2.9.0; cuda 12.8", "status": "measured", "steps": 2000, "runs": [{"us_per_step": 2500.0, "seconds": 5.0, "span_unix_ms": [1, 5001]}, {"us_per_step": 2400.0, "seconds": 4.8, "span_unix_ms": [6000, 10800]}, {"us_per_step": 2600.0, "seconds": 5.2, "span_unix_ms": [11000, 16200]}], "median_us_per_step": 2500.0, "gflops": 5236.3264}"#;
    let result = PointResult::from_line(line).unwrap();
    assert_eq!(result.side, "pytorch-compile");
    assert_eq!((result.batch, result.hidden), (1000, 4096));
    assert_eq!(result.median_us_per_step(), Some(2500.0));
}

#[test]
fn the_crossover_is_the_first_width_more_than_ten_percent_slower() {
    // Flat while launch-bound, then 25 % up at 1024.
    let widths = [
        (64, Some(100.0)),
        (256, Some(105.0)),
        (1024, Some(131.25)),
        (4096, Some(400.0)),
    ];
    assert_eq!(crossover(&widths), Crossover::At(1024));
    // Up at the first comparison.
    let widths = [
        (64, Some(100.0)),
        (256, Some(120.0)),
        (1024, Some(131.0)),
        (4096, Some(400.0)),
    ];
    assert_eq!(crossover(&widths), Crossover::At(256));
}

#[test]
fn exactly_ten_percent_is_not_more_than_ten_percent() {
    let widths = [
        (64, Some(100.0)),
        (256, Some(110.0)),
        (1024, Some(121.0)),
        (4096, Some(133.0)),
    ];
    assert_eq!(crossover(&widths), Crossover::NotReached);
    let widths = [
        (64, Some(100.0)),
        (256, Some(110.0)),
        (1024, Some(121.0)),
        (4096, Some(133.2)),
    ];
    assert_eq!(crossover(&widths), Crossover::At(4096));
}

#[test]
fn a_faster_larger_width_is_not_a_crossover() {
    let widths = [
        (64, Some(100.0)),
        (256, Some(90.0)),
        (1024, Some(95.0)),
        (4096, Some(99.0)),
    ];
    assert_eq!(crossover(&widths), Crossover::NotReached);
}

#[test]
fn a_width_without_a_figure_before_the_crossover_leaves_it_undetermined() {
    // 256 refused: neither 64 -> 256 nor 256 -> 1024 can be compared, so 1024 is not known to be
    // the first.
    let widths = [
        (64, Some(100.0)),
        (256, None),
        (1024, Some(300.0)),
        (4096, Some(900.0)),
    ];
    assert_eq!(crossover(&widths), Crossover::Undetermined { hidden: 256 });
    // A width missing after the crossover does not change it.
    let widths = [
        (64, Some(100.0)),
        (256, Some(200.0)),
        (1024, None),
        (4096, None),
    ];
    assert_eq!(crossover(&widths), Crossover::At(256));
    // The largest width missing with no crossover before it.
    let widths = [
        (64, Some(100.0)),
        (256, Some(101.0)),
        (1024, Some(102.0)),
        (4096, None),
    ];
    assert_eq!(crossover(&widths), Crossover::Undetermined { hidden: 4096 });
}

#[test]
fn crossover_lines_cover_each_side_and_batch_from_both_outputs() {
    let mut results = Vec::new();
    // ModelSpec at batch 1000: flat, then up at 1024; batch 100 never up; batch 8192 has a refusal.
    for (hidden, us) in [(64, 30.0), (256, 31.0), (1024, 60.0), (4096, 200.0)] {
        results.push(measured("modelspec-vulkan", 1000, hidden, us));
    }
    for (hidden, us) in [(64, 30.0), (256, 30.5), (1024, 31.0), (4096, 32.0)] {
        results.push(measured("modelspec-vulkan", 100, hidden, us));
    }
    for (hidden, us) in [(64, 30.0), (256, 90.0), (1024, 300.0)] {
        results.push(measured("modelspec-vulkan", 8192, hidden, us));
    }
    results.push(PointResult::refused(
        "modelspec-vulkan",
        sweep::Point {
            batch: 8192,
            hidden: 4096,
        },
        "E-TARGET-010".into(),
        None,
    ));
    // PyTorch eager at batch 1000: up at 4096.
    for (hidden, us) in [(64, 80.0), (256, 80.0), (1024, 85.0), (4096, 120.0)] {
        results.push(measured("pytorch-eager", 1000, hidden, us));
    }

    let lines = crossover_lines(&results);
    let find = |side: &str, batch: u64| {
        lines
            .iter()
            .find(|line| line.side == side && line.batch == batch)
            .unwrap_or_else(|| panic!("no line for {side} at batch {batch}: {lines:?}"))
    };
    assert_eq!(
        find("modelspec-vulkan", 1000).crossover,
        Crossover::At(1024)
    );
    assert!(find("modelspec-vulkan", 1000).pitch_slide);
    assert_eq!(
        find("modelspec-vulkan", 100).crossover,
        Crossover::NotReached
    );
    assert!(!find("modelspec-vulkan", 100).pitch_slide);
    assert_eq!(find("modelspec-vulkan", 8192).crossover, Crossover::At(256));
    assert_eq!(find("pytorch-eager", 1000).crossover, Crossover::At(4096));
    // A side's batch it has no point for at all is undetermined at the first width.
    assert_eq!(
        find("pytorch-eager", 100).crossover,
        Crossover::Undetermined { hidden: 64 }
    );
    // Two sides, three batches each, in the order the sides first appear.
    assert_eq!(lines.len(), 6);
    assert_eq!(lines[0].side, "modelspec-vulkan");
    assert_eq!(lines[3].side, "pytorch-eager");

    let text = find("modelspec-vulkan", 1000).to_line();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["verdict"], "reached");
    assert_eq!(value["hidden"], 1024);
    assert_eq!(value["pitch_slide"], true);
    let text = find("modelspec-vulkan", 100).to_line();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["verdict"], "not reached");
    assert!(value["hidden"].is_null());
}

#[test]
fn a_refusal_is_a_result_and_has_no_median() {
    let refused = PointResult::refused(
        "modelspec-vulkan",
        sweep::Point {
            batch: 100,
            hidden: 64,
        },
        "E-TARGET-003: no device".into(),
        None,
    );
    assert_eq!(refused.median_us_per_step(), None);
    assert_eq!(refused.flop_per_step, 3196 * 100 * 64);
}

#[test]
fn steps_for_a_timed_run_reach_the_minimum_duration() {
    // 20 steps took 1 ms: 50 µs per step; 6 s at that rate is 120,000 steps.
    assert_eq!(sweep::steps_for(1e-3, 20, 6.0), 120_000);
    // Rounded up, never below the warm-up count.
    assert_eq!(sweep::steps_for(1.0, 20, 6.0), 120);
    assert_eq!(sweep::steps_for(100.0, 20, 6.0), sweep::WARMUP_STEPS);
}

#[cfg(feature = "vulkan")]
mod rtx3090 {
    use modelspec_core::artifact::Backend;
    use modelspec_core::compile::compile_artifact;
    use modelspec_core::device::DeviceProfile;
    use modelspec_core::ir::kernel::{Kernel, KernelKind, PlanOp};
    use modelspec_runtime::spirv::emit::emit_kernels;
    use modelspec_runtime::vulkan::negotiate::negotiate;

    use super::sweep::points;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/rtx3090.device.json"
    ));

    /// Every point's source compiles for Vulkan, the profile negotiates, every kernel has a
    /// SPIR-V module, every buffer fits a storage-buffer binding, all buffers fit the
    /// device-local heap's budget, and the step plan's matmuls are `3196·B·H` FLOP.
    #[test]
    fn every_sweep_source_compiles_for_the_rtx3090_profile() {
        let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile");
        negotiate(&profile).expect("the fixture negotiates");
        let budget = profile
            .memory_heaps
            .iter()
            .filter(|heap| heap.device_local)
            .map(|heap| heap.budget_bytes.unwrap_or(heap.size_bytes))
            .max()
            .expect("a device-local heap");
        let mut refused = Vec::new();
        for point in points() {
            let name = point.file_name();
            let source = point.source(1000);
            let artifact = match compile_artifact(&name, &source, Backend::Vulkan) {
                Ok(artifact) => artifact,
                Err(diagnostics) => {
                    refused.push(format!("{name}: {diagnostics:?}"));
                    continue;
                }
            };
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
            let matmul_flop: u64 = plan
                .step
                .iter()
                .filter(|step| step.op == PlanOp::Dispatch)
                .filter_map(|step| plan.kernel(step.kernel_module_id?))
                .filter(|kernel| kernel.kind == KernelKind::MatMul)
                .map(|kernel| {
                    let (m, k, n) = Kernel::matmul_shape(&kernel.operands)
                        .unwrap_or_else(|| panic!("{name}: a matmul of static shapes"));
                    2 * m * k * n
                })
                .sum();
            assert_eq!(matmul_flop, point.flop_per_step(), "{name}");
        }
        assert!(refused.is_empty(), "{refused:#?}");
    }
}
