//! story:vulkan-mlm-masked-row-gather: the masked-row gather of the BERT-tiny MLM loss
//! (DESIGN.md § 3.3). The training step computes the head, the vocabulary projection, its bias,
//! the cross-entropy loss and its gradient on the gathered rows only; the full head stays for the
//! prediction, which no recorded training step computes.

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile;
use modelspec_core::ir::kernel::{Kernel, Operand, PlanOp};
use modelspec_core::ir::tensor::TensorType;
use modelspec_core::passes::masked_rows::{self, MaskedRowsSite};

const BERT_TINY: &str = include_str!("../../../examples/bert_tiny.mspec");

/// The examples that compile; `tiny_mlp.mspec` uses syntax past the M0 subset.
const EXAMPLES: [(&str, &str); 4] = [
    ("bert_tiny.mspec", BERT_TINY),
    (
        "linear_regression.mspec",
        include_str!("../../../examples/linear_regression.mspec"),
    ),
    (
        "mnist_mlp.mspec",
        include_str!("../../../examples/mnist_mlp.mspec"),
    ),
    (
        "transformer_block.mspec",
        include_str!("../../../examples/transformer_block.mspec"),
    ),
];

fn vulkan(masked_rows: Option<u64>) -> CompiledArtifact {
    compile::compile_artifact_with(
        "bert_tiny.mspec",
        BERT_TINY,
        Backend::Vulkan,
        None,
        masked_rows,
    )
    .expect("bert_tiny compiles for vulkan")
}

/// Whether `kernel` reads or writes a tensor of type `ty`.
fn touches(kernel: &Kernel, ty: &TensorType) -> bool {
    kernel.result == *ty
        || kernel
            .operands
            .iter()
            .any(|operand| matches!(operand, Operand::Buffer(t) if t == ty))
}

/// The kernels the recorded steps `positions` of the step plan dispatch.
fn dispatched(artifact: &CompiledArtifact, positions: &[usize]) -> Vec<Kernel> {
    let plan = &artifact.execution;
    positions
        .iter()
        .map(|&at| &plan.step[at])
        .filter(|step| step.op == PlanOp::Dispatch)
        .map(|step| {
            plan.kernel(step.kernel_module_id.expect("a dispatch names its kernel"))
                .expect("the plan has the kernel")
                .clone()
        })
        .collect()
}

/// At a capacity of 768 the training step's recorded dispatches touch no `[4096, 8192]` logits:
/// the cross-entropy and its gradient run on `[768, 8192]`. The plan still writes the full
/// prediction, in the step plan for a loss evaluation and in the forward plan.
#[test]
fn bert_tiny_trains_on_768_gathered_rows() {
    let artifact = vulkan(Some(768));
    let full = TensorType::f32(&[4096, 8192]);
    let gathered = TensorType::f32(&[768, 8192]);
    let plan = &artifact.execution;
    let trained = dispatched(&artifact, &plan.training_steps());
    let touching: Vec<String> = trained
        .iter()
        .filter(|k| touches(k, &full))
        .map(|k| k.op.to_owned())
        .collect();
    assert!(
        touching.is_empty(),
        "the training step touches the full logits: {touching:?}"
    );
    for op in ["CrossEntropy", "CrossEntropyGrad"] {
        assert!(
            trained.iter().any(|k| k.op == op && touches(k, &gathered)),
            "no {op} kernel on [768, 8192]"
        );
    }
    let loss = dispatched(&artifact, &plan.loss_steps());
    assert!(loss.iter().any(|k| touches(k, &full)));
    assert_eq!(plan.prediction_view.element_count, 4096 * 8192);
    assert!(
        plan.forward
            .iter()
            .any(|step| step.writes.contains(&plan.prediction_view))
    );
}

/// The gathered step names its site: the capacity and the declared labels and weights the host
/// gathers from.
#[test]
fn the_gathered_step_names_its_site() {
    let compiled = compile::compile_training_with("bert_tiny.mspec", BERT_TINY, Some(768))
        .expect("bert_tiny compiles");
    assert_eq!(
        masked_rows::site(&compiled.training.training.program),
        Some(MaskedRowsSite {
            capacity: 768,
            labels: "labels".into(),
            weights: "weights".into(),
        })
    );
    assert_eq!(masked_rows::site(&compiled.forward.program), None);
}

/// Without a capacity, or with one of every row, the step is the ungathered one, and its
/// training step records every Dispatch and Copy of the step plan: no example's plan has a step
/// the training step skips.
#[test]
fn without_a_gather_every_step_is_recorded() {
    let plain = compile::compile_training("bert_tiny.mspec", BERT_TINY).unwrap();
    for capacity in [None, Some(4096), Some(5000), Some(0)] {
        let compiled =
            compile::compile_training_with("bert_tiny.mspec", BERT_TINY, capacity).unwrap();
        assert_eq!(compiled, plain, "capacity {capacity:?}");
    }
    for (name, source) in EXAMPLES {
        let artifact = compile::compile_artifact(name, source, Backend::Vulkan)
            .unwrap_or_else(|d| panic!("{name}: {d:?}"));
        let plan = &artifact.execution;
        let every: Vec<usize> = plan
            .step
            .iter()
            .enumerate()
            .filter(|(_, step)| matches!(step.op, PlanOp::Dispatch | PlanOp::Copy))
            .map(|(at, _)| at)
            .collect();
        let recorded: Vec<usize> = plan
            .training_steps()
            .into_iter()
            .filter(|&at| matches!(plan.step[at].op, PlanOp::Dispatch | PlanOp::Copy))
            .collect();
        assert_eq!(recorded, every, "{name}");
    }
}

/// A model whose loss is no weighted `cross_entropy` has no site at any capacity.
#[test]
fn a_model_without_cross_entropy_has_no_site() {
    let (name, source) = EXAMPLES[1];
    let compiled = compile::compile_training_with(name, source, Some(1)).unwrap();
    assert_eq!(masked_rows::site(&compiled.training.training.program), None);
    assert_eq!(compiled, compile::compile_training(name, source).unwrap());
}
