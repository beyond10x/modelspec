//! The matmul-only FLOP rule of BERT-tiny pre-training (story:bert-tiny-vs-pytorch-cuda),
//! `benches/bert_tiny/flop.rs`, checked without a device: its value at the example's B = 32 and
//! S = 128 and its relation to the transformer block's rule. With the `vulkan` feature, its
//! dense, head and decoder parts are checked against the `MatMul` dispatches of the step plan
//! `examples/bert_tiny.mspec` compiles to on the RTX 3090 profile of
//! `tests/fixtures/rtx3090.device.json`.

#[path = "../benches/bert_tiny/flop.rs"]
mod flop;

/// At B = 32 and S = 128: 9,142,272 FLOP per row, 1,170,210,816 per sequence and
/// 37,446,746,112 per step, of which 1,610,612,736 are the attention products and 0 the
/// embedding lookups.
#[test]
fn bert_tiny_flop_at_the_example_batch() {
    assert_eq!(flop::flop_per_step(32, 128), 37_446_746_112);
    assert_eq!(flop::flop_per_step(1, 128), 1_170_210_816);
    assert_eq!(flop::flop_per_step(32, 128), 4096 * 9_142_272);
    assert_eq!(flop::attention_flop(32, 128), 1_610_612_736);
    assert_eq!(flop::embedding_flop(32, 128), 0);
    assert_eq!(
        flop::flop_per_step(32, 128),
        flop::dense_flop(32, 128)
            + flop::head_flop(32, 128)
            + flop::decoder_flop(32, 128)
            + flop::attention_flop(32, 128)
            + flop::embedding_flop(32, 128)
    );
    assert_eq!(
        (flop::BATCH, flop::SEQ, flop::TOKENS_PER_STEP),
        (32, 128, 4096)
    );
    assert_eq!(flop::FLOP_PER_SEQUENCE, 1_170_210_816);
    assert_eq!(flop::FLOP_RULE, "matmul-only");
}

/// The rule is the transformer block's, `N·(42·H² + 24·H·F + 24·S·H)` per layer stack, plus the
/// input adjoints of the first layer's three projections (`6·N·H²`), the MLM transform
/// (`6·N·H²`) and the tied decoder (`6·N·H·V`), at every B and S, and linear in B.
#[test]
fn bert_tiny_flop_is_the_block_plus_the_first_adjoints_and_the_head() {
    let (h, f, v) = (flop::HIDDEN, flop::FFN, flop::VOCAB);
    for (b, s) in [(1u64, 16u64), (8, 128), (32, 128), (128, 64)] {
        let n = b * s;
        let block = n * (42 * h * h + 24 * h * f + 24 * s * h);
        assert_eq!(
            flop::flop_per_step(b, s),
            block + 6 * n * h * h + 6 * n * h * h + 6 * n * h * v,
            "B {b} S {s}"
        );
        assert_eq!(flop::flop_per_step(b, s), b * flop::flop_per_step(1, s));
    }
}

#[cfg(feature = "vulkan")]
mod rtx3090 {
    use modelspec_core::artifact::Backend;
    use modelspec_core::compile::compile_artifact_for;
    use modelspec_core::device::DeviceProfile;
    use modelspec_core::ir::kernel::{DispatchCount, Kernel, KernelKind, PlanOp};

    use super::flop;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/rtx3090.device.json"
    ));

    const EXAMPLE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/bert_tiny.mspec"
    ));

    /// Every dispatch of the example's step plan is counted, its `MatMul` dispatches are the
    /// dense, head and decoder parts of the rule at B = 32 and S = 128, and it has one
    /// `Attention`, `AttentionGradQ`, `AttentionGradK` and `AttentionGradV` dispatch per layer,
    /// the sites the attention part counts.
    #[test]
    fn bert_tiny_flop_matches_the_step_plan_matmul_dispatches() {
        let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile");
        let artifact = compile_artifact_for(
            "examples/bert_tiny.mspec",
            EXAMPLE,
            Backend::Vulkan,
            Some(&profile),
        )
        .unwrap_or_else(|d| panic!("{d:?}"));
        let plan = &artifact.execution;
        let dispatches: Vec<_> = plan
            .step
            .iter()
            .filter(|step| step.op == PlanOp::Dispatch)
            .collect();
        let uncounted = dispatches
            .iter()
            .filter(|step| plan.dispatch_count(step) == Some(DispatchCount::Uncounted))
            .count();
        assert_eq!(uncounted, 0);
        let kernels: Vec<&Kernel> = dispatches
            .iter()
            .filter_map(|step| plan.kernel(step.kernel_module_id?))
            .collect();
        let matmul: u64 = kernels
            .iter()
            .filter(|kernel| kernel.kind == KernelKind::MatMul)
            .map(|kernel| {
                let (m, k, n) =
                    Kernel::matmul_shape(&kernel.operands).expect("a matmul of static shapes");
                2 * m * k * n
            })
            .sum();
        let (b, s) = (flop::BATCH, flop::SEQ);
        assert_eq!(
            matmul,
            flop::dense_flop(b, s) + flop::head_flop(b, s) + flop::decoder_flop(b, s)
        );
        for op in [
            "Attention",
            "AttentionGradQ",
            "AttentionGradK",
            "AttentionGradV",
        ] {
            let count = kernels.iter().filter(|kernel| kernel.op == op).count();
            assert_eq!(count as u64, flop::LAYERS, "{op}");
        }
    }
}
