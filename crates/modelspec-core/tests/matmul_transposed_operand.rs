//! `story:matmul-reads-transposed-operand`, acceptance `no-transpose-dispatch`: on the MNIST
//! example and on the `mlp_sweep` sources, the step plan dispatches no `transpose` whose output
//! only matmuls read; each such matmul reads the untransposed view in place. A transpose whose
//! output another kernel reads stays a dispatch, and its matmul reader reads its output.
//!
//! The cases read the plan's views, not the kernel record, so they say the same thing of the
//! plan before the change and after it.

use std::path::{Path, PathBuf};

use modelspec_core::artifact::Backend;
use modelspec_core::compile::compile_artifact;
use modelspec_core::ir::kernel::{BufferView, ExecutionPlan, Kernel, KernelKind, PlanOp, PlanStep};
use modelspec_core::ir::tensor::{Dim, TensorType};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn mnist() -> String {
    std::fs::read_to_string(repo_root().join("examples/mnist_mlp.mspec"))
        .expect("the example reads")
}

/// The source `benches/mlp_sweep.rs` compiles at batch `batch` and hidden width `hidden`.
fn sweep(batch: u64, hidden: u64) -> String {
    format!(
        "model SweepMlp {{
    param w1: Tensor<f32, [784, {hidden}]> = xavier_uniform();
    param w2: Tensor<f32, [{hidden}, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [{batch}, 784]>)
        -> Tensor<f32, [{batch}, 10]> {{
        return matmul(relu(matmul(x, w1)), w2);
    }}
}}

train Fit {{
    model = SweepMlp;
    input x: Tensor<f32, [{batch}, 784]>;
    target y: Tensor<f32, [{batch}, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 220;
    seed = 42;
    device = vulkan(0);
}}
"
    )
}

/// `t = transpose(x)` read by a matmul and by a `relu`: the transpose stays a dispatch.
fn transpose_read_elsewhere() -> String {
    "model ReadElsewhere {
    param w1: Tensor<f32, [10, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, [12, 3]> {
        let t = transpose(x);
        return add(matmul(t, w1), matmul(relu(t), w1));
    }
}

train Fit {
    model = ReadElsewhere;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [12, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 3;
    seed = 7;
    device = vulkan(0);
}
"
    .to_owned()
}

fn plan(name: &str, source: &str) -> ExecutionPlan {
    compile_artifact(name, source, Backend::Vulkan)
        .unwrap_or_else(|d| panic!("{name} compiles: {d:?}"))
        .execution
}

/// Every source the story names, by name.
fn sources() -> Vec<(String, String)> {
    let mut sources = vec![("mnist_mlp".to_owned(), mnist())];
    for (batch, hidden) in [(1000, 1024), (8192, 1024), (8192, 4096)] {
        sources.push((format!("sweep_b{batch}_h{hidden}"), sweep(batch, hidden)));
    }
    sources
}

fn kernel_of<'a>(plan: &'a ExecutionPlan, step: &PlanStep) -> &'a Kernel {
    plan.kernel(step.kernel_module_id.expect("a dispatch names its kernel"))
        .expect("the plan has the kernel")
}

/// The dispatch steps of `plan` that run a kernel of `kind`, each with its kernel.
fn dispatches(
    plan: &ExecutionPlan,
    kind: KernelKind,
) -> impl Iterator<Item = (&PlanStep, &Kernel)> {
    plan.step
        .iter()
        .filter(|s| s.op == PlanOp::Dispatch)
        .map(|s| (s, kernel_of(plan, s)))
        .filter(move |(_, k)| k.kind == kind)
}

fn dims(ty: &TensorType) -> Vec<u64> {
    ty.shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            Dim::Symbol(_) => panic!("{ty} is static"),
        })
        .collect()
}

/// The one matmul dispatch of the step plan whose result has shape `result`.
fn matmul_of<'a>(plan: &'a ExecutionPlan, result: &[u64]) -> &'a PlanStep {
    let found: Vec<&PlanStep> = dispatches(plan, KernelKind::MatMul)
        .filter(|(_, k)| dims(&k.result) == result)
        .map(|(s, _)| s)
        .collect();
    assert_eq!(found.len(), 1, "one matmul writes {result:?}");
    found[0]
}

fn data_view(plan: &ExecutionPlan, name: &str) -> BufferView {
    plan.data
        .iter()
        .find(|d| d.name == name)
        .unwrap_or_else(|| panic!("the plan binds {name}"))
        .view
}

fn parameter_view(plan: &ExecutionPlan, path_suffix: &str) -> BufferView {
    plan.parameters
        .iter()
        .find(|p| p.path.ends_with(path_suffix))
        .unwrap_or_else(|| panic!("the plan has a parameter {path_suffix}"))
        .views[0]
}

#[test]
fn no_step_plan_of_the_story_dispatches_a_transpose() {
    for (name, source) in sources() {
        let plan = plan(&name, &source);
        let transposes: Vec<String> = plan
            .kernels
            .iter()
            .filter(|k| k.kind == KernelKind::Transpose)
            .map(Kernel::signature)
            .collect();
        assert!(transposes.is_empty(), "{name}: {transposes:#?}");
        assert_eq!(
            dispatches(&plan, KernelKind::Transpose).count(),
            0,
            "{name}: no transpose dispatch"
        );
        assert_eq!(
            dispatches(&plan, KernelKind::MatMul).count(),
            5,
            "{name}: the five matmuls stay"
        );
    }
}

/// The weight gradient of `w1` reads the input `x` in place, that of `w2` the activation the
/// forward matmul writes, and the hidden gradient reads `w2` in place, each a view the step
/// plan already binds or writes.
#[test]
fn each_matmul_reads_the_untransposed_view_in_place() {
    for (name, source, batch, hidden) in [
        ("mnist_mlp".to_owned(), mnist(), 100, 64),
        (
            "sweep_b8192_h4096".to_owned(),
            sweep(8192, 4096),
            8192,
            4096,
        ),
        (
            "sweep_b1000_h1024".to_owned(),
            sweep(1000, 1024),
            1000,
            1024,
        ),
    ] {
        let plan = plan(&name, &source);
        let (forward, _) = dispatches(&plan, KernelKind::MatMul)
            .find(|(_, k)| dims(&k.result) == [batch, hidden] && k.epilogue.is_some())
            .expect("the forward matmul with its relu epilogue");
        let activation = *forward.writes.last().expect("the forward matmul writes");
        let w1_gradient = matmul_of(&plan, &[784, hidden]);
        assert_eq!(
            w1_gradient.reads[0],
            data_view(&plan, "x"),
            "{name}: the w1 gradient reads x"
        );
        let w2_gradient = matmul_of(&plan, &[hidden, 10]);
        assert_eq!(
            w2_gradient.reads[0], activation,
            "{name}: the w2 gradient reads the activation"
        );
        let hidden_gradient = dispatches(&plan, KernelKind::MatMul)
            .find(|(_, k)| dims(&k.result) == [batch, hidden] && k.epilogue.is_none())
            .map(|(s, _)| s)
            .expect("the hidden gradient matmul");
        assert_eq!(
            hidden_gradient.reads[1],
            parameter_view(&plan, "w2"),
            "{name}: the hidden gradient reads w2"
        );
    }
}

#[test]
fn a_transpose_another_kernel_reads_stays_a_dispatch_and_its_matmul_reads_its_output() {
    let plan = plan("read_elsewhere", &transpose_read_elsewhere());
    let x = data_view(&plan, "x");
    let (transpose, _) = dispatches(&plan, KernelKind::Transpose)
        .find(|(s, _)| s.reads == [x])
        .expect("transpose(x) is dispatched");
    let output = transpose.writes[0];
    let relu = dispatches(&plan, KernelKind::Elementwise)
        .find(|(s, k)| k.op == "Relu" && s.reads == [output]);
    assert!(relu.is_some(), "the relu reads the transpose's output");
    let matmul_reads_it = dispatches(&plan, KernelKind::MatMul).any(|(s, _)| s.reads[0] == output);
    assert!(matmul_reads_it, "a matmul reads the transpose's output");
    let matmul_reads_x = dispatches(&plan, KernelKind::MatMul).any(|(s, _)| s.reads.contains(&x));
    assert!(
        !matmul_reads_x,
        "no matmul reads x in place while the transpose is dispatched"
    );
}
