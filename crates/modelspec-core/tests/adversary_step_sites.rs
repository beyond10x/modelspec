//! Adversary pass 1 of story:step-sites-beat-pytorch, without a device: the `ReluGrad` epilogue,
//! the place of the `mse` loss sum and `ExecutionPlan::loss_steps` on models other than the two
//! the unit's own tests use (`tests/step_sites_plan.rs` in `modelspec-runtime`: the MNIST example
//! and the `mlp_sweep` b8192 h4096 point).
//!
//! The models: three layers (two `ReluGrad` epilogues), a `relu` of a parameter (the epilogue
//! reads a parameter view that is updated in place later in the step), a bias `add` before the
//! `relu` (the pre-activation is an elementwise output), two `relu`s of one matmul output (two
//! epilogues reading one pre-activation), transposed parameters, a wide second layer (the hidden
//! gradient and the `w2` gradient tie on FLOP), no `relu` at all, a scaled loss
//! (`mul(mse(..), 2.0)`, the loss view written after the sum) and `relu(relu(matmul))`.
//!
//! Each case states the rule it drives from: DESIGN.md § 3.1 (revisions 0.45 and 0.46), the
//! module documentation of `schedule.rs` and the 0.46 CHANGELOG line.

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{self, CompiledTraining};
use modelspec_core::ir::kernel::{
    BufferView, Epilogue, ExecutionPlan, Kernel, KernelKind, PlanOp, PlanStep,
};
use modelspec_core::ir::print::print_kernel;
use modelspec_core::ir::tensor::{TensorNode, TensorOp, ValueId};
use modelspec_core::ir::verify::verify_plan;

/// A model `Adv` with the parameter lines `params`, input `x: [b, i]`, the forward `body`
/// returning `[b, o]`, trained on `loss`.
fn model(params: &str, (b, i, o): (u64, u64, u64), body: &str, loss: &str) -> String {
    format!(
        "model Adv {{
{params}
    fn forward(x: Tensor<f32, [{b}, {i}]>)
        -> Tensor<f32, [{b}, {o}]> {{
        {body}
    }}
}}

train Fit {{
    model = Adv;
    input x: Tensor<f32, [{b}, {i}]>;
    target y: Tensor<f32, [{b}, {o}]>;
    loss = {loss};
    optimizer = sgd(lr = 0.05);
    steps = 30;
    seed = 7;
    device = vulkan(0);
}}
"
    )
}

const MSE: &str = "mse(model.forward(x), y)";

/// The probe models, by name.
fn models() -> Vec<(&'static str, String)> {
    let p = |lines: &[&str]| {
        lines
            .iter()
            .map(|l| format!("    param {l} = xavier_uniform();\n"))
            .collect::<String>()
    };
    vec![
        (
            "deep3",
            model(
                &p(&[
                    "w1: Tensor<f32, [12, 9]>",
                    "w2: Tensor<f32, [9, 7]>",
                    "w3: Tensor<f32, [7, 5]>",
                ]),
                (10, 12, 5),
                "return matmul(relu(matmul(relu(matmul(x, w1)), w2)), w3);",
                MSE,
            ),
        ),
        (
            "relu_of_a_parameter",
            model(
                &p(&["w1: Tensor<f32, [12, 9]>", "w2: Tensor<f32, [9, 5]>"]),
                (10, 12, 5),
                "return matmul(matmul(x, relu(w1)), w2);",
                MSE,
            ),
        ),
        (
            "bias_before_relu",
            model(
                &p(&[
                    "w1: Tensor<f32, [12, 9]>",
                    "b1: Tensor<f32, [10, 9]>",
                    "w2: Tensor<f32, [9, 5]>",
                ]),
                (10, 12, 5),
                "return matmul(relu(add(matmul(x, w1), b1)), w2);",
                MSE,
            ),
        ),
        (
            "two_relus_of_one_matmul",
            model(
                &p(&[
                    "w1: Tensor<f32, [12, 9]>",
                    "w2: Tensor<f32, [9, 5]>",
                    "w3: Tensor<f32, [9, 5]>",
                ]),
                (10, 12, 5),
                "let z = matmul(x, w1);\n        return add(matmul(relu(z), w2), matmul(relu(z), w3));",
                MSE,
            ),
        ),
        (
            "transposed_parameters",
            model(
                &p(&["v1: Tensor<f32, [9, 12]>", "v2: Tensor<f32, [5, 9]>"]),
                (10, 12, 5),
                "return matmul(relu(matmul(x, transpose(v1))), transpose(v2));",
                MSE,
            ),
        ),
        (
            "wide_second_layer",
            model(
                &p(&["w1: Tensor<f32, [16, 8]>", "w2: Tensor<f32, [8, 512]>"]),
                (64, 16, 512),
                "return matmul(relu(matmul(x, w1)), w2);",
                MSE,
            ),
        ),
        (
            "no_relu",
            model(
                &p(&["w1: Tensor<f32, [12, 9]>", "w2: Tensor<f32, [9, 5]>"]),
                (10, 12, 5),
                "return matmul(matmul(x, w1), w2);",
                MSE,
            ),
        ),
        (
            "scaled_loss",
            model(
                &p(&["w1: Tensor<f32, [12, 9]>", "w2: Tensor<f32, [9, 5]>"]),
                (10, 12, 5),
                "return matmul(relu(matmul(x, w1)), w2);",
                "mul(mse(model.forward(x), y), 2.0)",
            ),
        ),
    ]
}

/// `relu(relu(matmul(x, w1)))`: the inner `relu` is the forward matmul's epilogue, which writes
/// the activation `a1` to a buffer of its own; the outer `relu`'s gradient is
/// `ReluGrad(a1, matmul(g, transpose(w2)))`.
fn relu_of_relu() -> String {
    model(
        "    param w1: Tensor<f32, [12, 9]> = xavier_uniform();\n    param w2: Tensor<f32, [9, 5]> = xavier_uniform();\n",
        (10, 12, 5),
        "return matmul(relu(relu(matmul(x, w1))), w2);",
        MSE,
    )
}

struct Compiled {
    training: CompiledTraining,
    artifact: CompiledArtifact,
}

fn compiled(name: &str, source: &str) -> Compiled {
    let file = format!("{name}.mspec");
    let training = compile::compile_training(&file, source)
        .unwrap_or_else(|d| panic!("{name}: compile_training refused: {d:#?}"));
    let artifact = compile::compile_artifact(&file, source, Backend::Vulkan)
        .unwrap_or_else(|d| panic!("{name}: compile_artifact refused: {d:#?}"));
    verify_plan(&artifact.execution).unwrap_or_else(|e| panic!("{name}: {e}"));
    Compiled { training, artifact }
}

fn nodes(c: &Compiled) -> &[TensorNode] {
    &c.training.training.training.program.nodes
}

fn node(c: &Compiled, value: ValueId) -> &TensorNode {
    nodes(c)
        .iter()
        .find(|n| n.id == value)
        .unwrap_or_else(|| panic!("no node {value}"))
}

fn readers(c: &Compiled, value: ValueId) -> Vec<&TensorNode> {
    nodes(c)
        .iter()
        .filter(|n| n.operands.contains(&value))
        .collect()
}

fn kernel_of<'p>(plan: &'p ExecutionPlan, step: &PlanStep) -> &'p Kernel {
    plan.kernel(step.kernel_module_id.expect("a dispatch names its module"))
        .expect("the plan holds the kernel")
}

fn dispatches_of(plan: &ExecutionPlan, value: ValueId) -> Vec<usize> {
    plan.step
        .iter()
        .enumerate()
        .filter(|(_, s)| s.op == PlanOp::Dispatch && s.value == Some(value))
        .map(|(i, _)| i)
        .collect()
}

/// The view a step of the plan reads for the value `x`: its parameter or data view, the first
/// view of its own dispatch (the pre-activation of a `relu` epilogue that keeps it), or the
/// last view of the matmul whose `relu` epilogue `x` is.
fn view_of(c: &Compiled, x: ValueId) -> BufferView {
    let plan = &c.artifact.execution;
    match &node(c, x).op {
        TensorOp::ParamRead { param } => {
            plan.parameters
                .iter()
                .find(|p| p.param == *param)
                .expect("a declared parameter")
                .views[0]
        }
        TensorOp::Input { name } => {
            plan.data
                .iter()
                .find(|d| &d.name == name)
                .expect("a declared input")
                .view
        }
        _ => match dispatches_of(plan, x)[..] {
            [at] => plan.step[at].writes[0],
            _ => {
                let operand = node(c, x).operands[0];
                let [at] = dispatches_of(plan, operand)[..] else {
                    panic!("{x} has no dispatch and no matmul whose epilogue it is");
                };
                *plan.step[at].writes.last().expect("the matmul writes")
            }
        },
    }
}

/// DESIGN.md § 3.1 (0.46) and the 0.46 CHANGELOG line: a `relu_grad(x, g)` whose `g` is a
/// matmul output that no other op reads, neither the loss, the prediction nor an update, with `x`
/// stored before the matmul, is that matmul's `epilogue ReluGrad`: no dispatch of its own, the
/// matmul reads `x` as its third operand, of the output's type, and writes one view, read by the
/// readers of the `relu_grad`.
fn assert_every_matmul_gated_relu_grad_is_an_epilogue(name: &str, c: &Compiled) -> usize {
    let plan = &c.artifact.execution;
    let mut fused = 0;
    for rg in nodes(c).iter().filter(|n| n.op == TensorOp::ReluGrad) {
        let [x, g] = rg.operands[..] else {
            panic!("{name}: ReluGrad reads two values");
        };
        if node(c, g).op != TensorOp::MatMul || readers(c, g).len() != 1 {
            continue;
        }
        assert!(
            dispatches_of(plan, rg.id).is_empty(),
            "{name}: {} has a dispatch of its own\n{}",
            rg.id,
            print_kernel(plan)
        );
        let [at] = dispatches_of(plan, g)[..] else {
            panic!(
                "{name}: the matmul {g} has one dispatch\n{}",
                print_kernel(plan)
            );
        };
        let step = &plan.step[at];
        let kernel = kernel_of(plan, step);
        assert_eq!(kernel.kind, KernelKind::MatMul, "{name}");
        assert_eq!(kernel.epilogue, Some(Epilogue::ReluGrad), "{name}");
        assert_eq!(kernel.operands.len(), 3, "{name}");
        assert_eq!(
            kernel.operands[2].to_string(),
            kernel.result.to_string(),
            "{name}"
        );
        assert_eq!(
            step.reads[2],
            view_of(c, x),
            "{name}: the epilogue of {g} reads x = {x}\n{}",
            print_kernel(plan)
        );
        assert_eq!(step.writes.len(), 1, "{name}");
        assert!(
            plan.step[..at]
                .iter()
                .any(|s| s.writes.contains(&step.reads[2]))
                || plan.parameters.iter().any(|p| p.views[0] == step.reads[2])
                || plan.data.iter().any(|d| d.view == step.reads[2]),
            "{name}: x is stored before the matmul runs"
        );
        fused += 1;
    }
    fused
}

#[test]
fn every_matmul_gated_relu_grad_is_an_epilogue_that_reads_its_own_pre_activation() {
    let expected: BTreeMap<&str, usize> = [
        ("deep3", 2),
        ("relu_of_a_parameter", 1),
        ("bias_before_relu", 1),
        ("two_relus_of_one_matmul", 2),
        ("transposed_parameters", 1),
        ("wide_second_layer", 1),
        ("no_relu", 0),
        ("scaled_loss", 1),
    ]
    .into_iter()
    .collect();
    for (name, source) in models() {
        let c = compiled(name, &source);
        let fused = assert_every_matmul_gated_relu_grad_is_an_epilogue(name, &c);
        assert_eq!(Some(&fused), expected.get(name), "{name}");
        let epilogues = c
            .artifact
            .execution
            .step
            .iter()
            .filter(|s| s.op == PlanOp::Dispatch)
            .filter(|s| kernel_of(&c.artifact.execution, s).epilogue == Some(Epilogue::ReluGrad))
            .count();
        assert_eq!(
            epilogues, fused,
            "{name}: no other dispatch has the epilogue"
        );
    }
}

/// The epilogue that reads a parameter: the matmul that gates `relu(w1)`'s gradient reads `w1`'s
/// view, and `w1`'s in-place update comes after a barrier that follows that matmul.
#[test]
fn an_epilogue_reading_a_parameter_runs_before_its_update_behind_a_barrier() {
    let models = models();
    let source = &models
        .iter()
        .find(|(n, _)| *n == "relu_of_a_parameter")
        .expect("the model")
        .1;
    let c = compiled("relu_of_a_parameter", source);
    let plan = &c.artifact.execution;
    let w1 = plan.parameters[0].views[0];
    let reader = plan
        .step
        .iter()
        .position(|s| {
            s.op == PlanOp::Dispatch && kernel_of(plan, s).epilogue == Some(Epilogue::ReluGrad)
        })
        .expect("the epilogue dispatch");
    assert_eq!(plan.step[reader].reads[2], w1, "{}", print_kernel(plan));
    let update = plan
        .step
        .iter()
        .position(|s| s.op == PlanOp::Dispatch && s.writes.contains(&w1))
        .expect("w1 is updated in place");
    assert!(reader < update);
    assert!(
        plan.step[reader..update]
            .iter()
            .any(|s| s.op == PlanOp::Barrier),
        "{}",
        print_kernel(plan)
    );
}

/// DESIGN.md § 3.1 (0.45), `step_order`: the `mse` loss sum is recorded right after the matmul
/// with the most FLOP, the first of equals, among the steps after it up to the first conflict,
/// with no barrier between the two; the last writer of each buffer it reads is behind a barrier
/// before it. The expected result shape of that matmul per model.
#[test]
fn the_loss_sum_sits_after_the_first_largest_matmul_and_behind_its_inputs_barrier() {
    let expected: BTreeMap<&str, Vec<u64>> = [
        ("deep3", vec![12, 9]),
        ("relu_of_a_parameter", vec![12, 9]),
        ("bias_before_relu", vec![12, 9]),
        ("two_relus_of_one_matmul", vec![12, 9]),
        // The gradient of v1 is the transpose of the [12, 10] x [10, 9] matmul's output.
        ("transposed_parameters", vec![12, 9]),
        // The hidden gradient [64, 512] x [512, 8] and the w2 gradient [8, 64] x [64, 512]
        // both have 524,288 FLOP; the first of equals is the hidden gradient.
        ("wide_second_layer", vec![64, 8]),
        ("no_relu", vec![12, 9]),
    ]
    .into_iter()
    .collect();
    for (name, source) in models() {
        let c = compiled(name, &source);
        let plan = &c.artifact.execution;
        let sums: Vec<usize> = plan
            .step
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.op == PlanOp::Dispatch && kernel_of(plan, s).kind == KernelKind::LossSum
            })
            .map(|(i, _)| i)
            .collect();
        let [at] = sums[..] else {
            panic!("{name}: one loss sum");
        };
        let sum = &plan.step[at];
        let last_writer = plan.step[..at]
            .iter()
            .rposition(|s| {
                s.op != PlanOp::Barrier
                    && s.writes
                        .iter()
                        .any(|w| sum.reads.iter().any(|r| r.buffer == w.buffer))
            })
            .expect("the prediction is written before the sum");
        assert!(
            plan.step[last_writer..at]
                .iter()
                .any(|s| s.op == PlanOp::Barrier),
            "{name}: a barrier between the sum and its inputs' last writer\n{}",
            print_kernel(plan)
        );
        match expected.get(name) {
            Some(shape) => {
                let before = &plan.step[at - 1];
                assert_eq!(
                    before.op,
                    PlanOp::Dispatch,
                    "{name}\n{}",
                    print_kernel(plan)
                );
                let matmul = kernel_of(plan, before);
                assert_eq!(matmul.kind, KernelKind::MatMul, "{name}");
                let dims: Vec<String> = matmul.result.shape.iter().map(|d| d.to_string()).collect();
                let want: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                assert_eq!(dims, want, "{name}\n{}", print_kernel(plan));
            }
            None => {
                // scaled_loss: the loss view is written by the `mul` after the sum, which reads
                // the sum's buffer, so the sum keeps its Training IR place.
                assert_ne!(sum.writes, vec![plan.loss_view], "{name}");
            }
        }
    }
}

/// `ExecutionPlan::loss_steps` on every probe model: the chosen steps, run alone in order from
/// the bound parameter and data views, have every buffer they read written before they read it,
/// write the loss and the prediction views, write no parameter, compute only values the loss
/// or the prediction depends on, and keep a chosen barrier between any two chosen steps the
/// plan orders through a hazard.
#[test]
fn the_loss_steps_alone_compute_the_loss_and_the_prediction_and_nothing_else() {
    for (name, source) in models() {
        let c = compiled(name, &source);
        let plan = &c.artifact.execution;
        let chosen = plan.loss_steps();
        assert!(!chosen.is_empty(), "{name}");
        let mut written: BTreeSet<u32> = plan
            .parameters
            .iter()
            .flat_map(|p| p.views.iter().map(|v| v.buffer))
            .chain(plan.data.iter().map(|d| d.view.buffer))
            .collect();
        let parameters: BTreeSet<u32> = plan
            .parameters
            .iter()
            .flat_map(|p| p.views.iter().map(|v| v.buffer))
            .collect();
        let needed = ancestors(
            &c,
            &[
                c.training.training.training.loss,
                c.training.forward.prediction,
            ],
        );
        for &at in &chosen {
            let step = &plan.step[at];
            assert_ne!(step.op, PlanOp::Allocate, "{name}");
            if step.op == PlanOp::Barrier {
                continue;
            }
            for r in &step.reads {
                assert!(
                    written.contains(&r.buffer),
                    "{name}: step {at} reads b{} before a chosen step writes it\n{}",
                    r.buffer,
                    print_kernel(plan)
                );
            }
            for w in &step.writes {
                assert!(
                    !parameters.contains(&w.buffer),
                    "{name}: step {at} writes a parameter"
                );
                written.insert(w.buffer);
            }
            let value = step.value.expect("a dispatch names its value");
            assert!(
                needed.contains(&value),
                "{name}: step {at} computes {value}, which neither the loss nor the prediction \
                 reads\n{}",
                print_kernel(plan)
            );
        }
        assert!(written.contains(&plan.loss_view.buffer), "{name}");
        assert!(written.contains(&plan.prediction_view.buffer), "{name}");
        for (i, &a) in chosen.iter().enumerate() {
            for &b in &chosen[i + 1..] {
                let (sa, sb) = (&plan.step[a], &plan.step[b]);
                if sa.op == PlanOp::Barrier || sb.op == PlanOp::Barrier {
                    continue;
                }
                let touches = |x: &[BufferView], y: &[BufferView]| {
                    x.iter().any(|v| y.iter().any(|w| v.buffer == w.buffer))
                };
                let hazard = touches(&sa.writes, &sb.reads)
                    || touches(&sa.writes, &sb.writes)
                    || touches(&sa.reads, &sb.writes);
                if hazard {
                    assert!(
                        chosen
                            .iter()
                            .any(|&k| a < k && k < b && plan.step[k].op == PlanOp::Barrier),
                        "{name}: chosen steps {a} and {b} share a buffer with no chosen barrier \
                         between them\n{}",
                        print_kernel(plan)
                    );
                }
            }
        }
    }
}

/// The values `roots` depend on, themselves included.
fn ancestors(c: &Compiled, roots: &[ValueId]) -> BTreeSet<ValueId> {
    let mut reached: BTreeSet<ValueId> = BTreeSet::new();
    let mut stack: Vec<ValueId> = roots.to_vec();
    while let Some(v) = stack.pop() {
        if reached.insert(v) {
            stack.extend(node(c, v).operands.iter().copied());
        }
    }
    reached
}

/// Finding ADV-S1-F1. DESIGN.md § 3.1 (0.46): "A `relu_grad(x, g)` whose `g` is a `matmul`
/// output that no other op reads, and that is neither the loss, the prediction nor an update, is
/// an epilogue of that matmul's kernel (`epilogue ReluGrad`) when `x` is stored before the matmul
/// and is no parameter updated in place before it"; the CHANGELOG line: "A `relu_grad` whose
/// gradient operand is a matmul output that nothing else reads is now an epilogue of that
/// matmul". In `relu(relu(matmul(x, w1)))` the outer gradient `ReluGrad(a1, g)` has such a `g`,
/// and `a1`, the inner `relu`, is stored before it: the forward matmul's `relu` epilogue writes
/// it to a buffer of its own. `schedule.rs` refused the epilogue because `a1` was in `absorbed`
/// (`!absorbed.contains(&x)`), which also holds a `relu` epilogue that has a buffer; fix pass 1
/// (coordinator decision D65) refuses only an `x` without a buffer, a fused `lr * g` product.
#[test]
fn adv_s1_f1_relu_grad_of_an_epilogue_activation_is_an_epilogue_as_design_states() {
    let c = compiled("relu_of_relu", &relu_of_relu());
    let fused = assert_every_matmul_gated_relu_grad_is_an_epilogue("relu_of_relu", &c);
    assert_eq!(fused, 1);
}
