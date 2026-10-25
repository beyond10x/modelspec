//! Adversary pass 1 of story:fuse-elementwise-chains-and-gradient-sums: the fused plan computes
//! the Training IR, read as terms.
//!
//! Each value of the Training IR is a term: an input, a parameter, an `f32` immediate, or an op
//! over the terms of its operands, in operand order. Each plan is run symbolically: a dispatch
//! writes, into each view it writes, the term its kernel computes from the terms its views hold
//! (a fused chain its ops in order, each reading the result before it at the position its name
//! gives). The step plan must leave the term of each parameter's update in its view and the term
//! of the loss in the loss view; the forward plan the term of the prediction in the prediction
//! view; and every view a dispatch writes must hold the term of some Training IR value. A
//! reassociated adjoint sum, a swapped operand, a chain that reads a parameter after its update
//! or a value read from a view nobody wrote fails one of them. Constant ops over two immediates
//! are folded on both sides by the schedule's rule, so a folded scalar compares with its op.
//!
//! The models are the chain conditions of the adversary brief: an intermediate read twice, by a
//! reduction, by a matmul, by an SGD update, after a `relu` epilogue, a loss chain of rank 0,
//! adjoint sums of 2 to 8 contributions, and the two examples.

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::compile::{self, CompiledTraining};
use modelspec_core::ir::kernel::{
    Epilogue, ExecutionPlan, Kernel, KernelKind, Operand, PlanOp, PlanStep,
};
use modelspec_core::ir::print::print_kernel;
use modelspec_core::ir::tensor::{TensorOp, ValueId};
use modelspec_core::ir::verify::verify_plan;
use modelspec_core::schedule;

/// Hash-consed terms: each distinct `(head, children)` has one id.
#[derive(Default)]
struct Terms {
    ids: BTreeMap<(String, Vec<u32>), u32>,
    heads: Vec<(String, Vec<u32>)>,
}

impl Terms {
    fn make(&mut self, head: String, children: Vec<u32>) -> u32 {
        if let Some(&id) = self.ids.get(&(head.clone(), children.clone())) {
            return id;
        }
        let id = u32::try_from(self.heads.len()).expect("few terms");
        self.heads.push((head.clone(), children.clone()));
        self.ids.insert((head, children), id);
        id
    }

    fn imm(&mut self, value: f32) -> u32 {
        self.make(format!("imm:{:#010x}", value.to_bits()), Vec::new())
    }

    fn immediate(&self, id: u32) -> Option<f32> {
        let (head, children) = &self.heads[id as usize];
        let bits = head.strip_prefix("imm:0x")?;
        children
            .is_empty()
            .then(|| f32::from_bits(u32::from_str_radix(bits, 16).expect("hex bits")))
    }

    /// `op` over `children`, an `Add`, `Sub` or `Mul` of two immediates folded as the schedule
    /// folds it (finite, normal operands and result).
    fn op(&mut self, op: &str, children: Vec<u32>) -> u32 {
        if let [a, b] = children[..]
            && let (Some(a), Some(b)) = (self.immediate(a), self.immediate(b))
        {
            let value = match op {
                "Add" => Some(a + b),
                "Sub" => Some(a - b),
                "Mul" => Some(a * b),
                _ => None,
            };
            let exact = |x: f32| x.is_finite() && !x.is_subnormal();
            if let Some(value) = value.filter(|&v| exact(a) && exact(b) && exact(v)) {
                return self.imm(value);
            }
        }
        self.make(op.to_owned(), children)
    }

    fn render(&self, id: u32, depth: usize) -> String {
        let (head, children) = &self.heads[id as usize];
        if children.is_empty() {
            return head.clone();
        }
        if depth == 0 {
            return format!("{head}(…)");
        }
        let inner: Vec<String> = children
            .iter()
            .map(|&c| self.render(c, depth - 1))
            .collect();
        format!("{head}({})", inner.join(", "))
    }
}

fn training(source: &str) -> CompiledTraining {
    compile::compile_training("adv1.mspec", source).expect("the source compiles for training")
}

/// The term of every Training IR value.
fn ir_terms(terms: &mut Terms, training: &CompiledTraining) -> BTreeMap<ValueId, u32> {
    let program = &training.training.training.program;
    let mut of: BTreeMap<ValueId, u32> = BTreeMap::new();
    for node in &program.nodes {
        let children: Vec<u32> = node.operands.iter().map(|o| of[o]).collect();
        let term = match &node.op {
            TensorOp::Input { name } => terms.make(format!("in:{name}"), Vec::new()),
            TensorOp::ParamRead { param } => terms.make(format!("param:{}", param.0), Vec::new()),
            TensorOp::Scalar(value) => terms.imm(*value),
            op => terms.op(op.name(), children),
        };
        of.insert(node.id, term);
    }
    of
}

fn arity(op: &str) -> usize {
    if op == "Relu" { 1 } else { 2 }
}

/// The terms one dispatch writes, in the order of its writes.
fn dispatch_terms(terms: &mut Terms, kernel: &Kernel, read: &[u32]) -> Vec<u32> {
    let mut reads = read.iter().copied();
    let operands: Vec<u32> = kernel
        .operands
        .iter()
        .map(|operand| match operand {
            Operand::Buffer(_) => reads.next().expect("a view per buffer operand"),
            Operand::Immediate(value) => terms.imm(*value),
        })
        .collect();
    assert!(reads.next().is_none(), "{}", kernel.signature());
    match kernel.kind {
        KernelKind::Elementwise if kernel.op.contains('>') => {
            let mut leaves = operands.into_iter();
            let mut previous = None;
            for (i, link) in kernel.op.split('>').enumerate() {
                let (op, at) = match link.split_once('@') {
                    Some((op, at)) => (op, Some(at.parse::<usize>().expect("a position"))),
                    None => {
                        assert_eq!(i, 0, "{}", kernel.op);
                        (link, None)
                    }
                };
                let children: Vec<u32> = (0..arity(op))
                    .map(|j| {
                        if Some(j) == at {
                            previous.expect("a later op reads the one before")
                        } else {
                            leaves.next().expect("a leaf per operand")
                        }
                    })
                    .collect();
                previous = Some(terms.op(op, children));
            }
            assert!(leaves.next().is_none(), "{}", kernel.signature());
            vec![previous.expect("a chain has ops")]
        }
        KernelKind::Elementwise => vec![terms.op(kernel.op, operands)],
        KernelKind::Broadcast => vec![terms.op("BroadcastScalar", operands)],
        KernelKind::SgdUpdate => match operands[..] {
            [p, lr, g] => {
                let product = terms.op("Mul", vec![lr, g]);
                vec![terms.op("Sub", vec![p, product])]
            }
            [p, d] => vec![terms.op("Sub", vec![p, d])],
            _ => panic!("{}", kernel.signature()),
        },
        KernelKind::MatMul => {
            let children: Vec<u32> = operands
                .iter()
                .zip(kernel.transposed)
                .map(|(&t, flag)| {
                    if flag {
                        terms.op("Transpose", vec![t])
                    } else {
                        t
                    }
                })
                .collect();
            let product = terms.op("MatMul", children);
            match kernel.epilogue {
                None => vec![product],
                Some(Epilogue::Relu {
                    writes_pre_activation,
                }) => {
                    let relu = terms.op("Relu", vec![product]);
                    if writes_pre_activation {
                        vec![product, relu]
                    } else {
                        vec![relu]
                    }
                }
                Some(Epilogue::ReluGrad) => {
                    let [_, _, pre_activation] = operands[..] else {
                        panic!("{}", kernel.signature());
                    };
                    vec![terms.op("ReluGrad", vec![pre_activation, product])]
                }
            }
        }
        KernelKind::PerElement | KernelKind::PerRow | KernelKind::PerColumn => {
            vec![terms.op(kernel.op, operands)]
        }
        KernelKind::LossSum => vec![terms.op("Mse", operands)],
        KernelKind::Transpose => vec![terms.op("Transpose", operands)],
        KernelKind::ReduceStage1 => vec![terms.op(&format!("{}:stage1", kernel.op), operands)],
        KernelKind::ReduceStage2 => {
            let [partials] = operands[..] else {
                panic!("{}", kernel.signature());
            };
            let (head, children) = terms.heads[partials as usize].clone();
            assert_eq!(
                head,
                format!("{}:stage1", kernel.op),
                "stage 2 reads stage 1"
            );
            vec![terms.op(kernel.op, children)]
        }
    }
}

/// Runs `steps` over `views` (buffer to term), checking that no barrier group holds a hazard and
/// that every view a dispatch writes holds the term of some value in `known`.
fn run(
    terms: &mut Terms,
    plan: &ExecutionPlan,
    steps: &[PlanStep],
    views: &mut BTreeMap<u32, u32>,
    known: &BTreeSet<u32>,
    name: &str,
) {
    let mut group: Vec<(BTreeSet<u32>, BTreeSet<u32>)> = Vec::new();
    for (at, step) in steps.iter().enumerate() {
        let reads: BTreeSet<u32> = step.reads.iter().map(|v| v.buffer).collect();
        let writes: BTreeSet<u32> = step.writes.iter().map(|v| v.buffer).collect();
        match step.op {
            PlanOp::Barrier => {
                group.clear();
                continue;
            }
            PlanOp::Allocate | PlanOp::Complete => continue,
            PlanOp::Dispatch | PlanOp::Copy => {
                for (r, w) in &group {
                    assert!(
                        reads.is_disjoint(w) && writes.is_disjoint(r) && writes.is_disjoint(w),
                        "{name} step {at}: a hazard in one barrier group\n{}",
                        print_kernel(plan)
                    );
                }
                group.push((reads, writes));
            }
        }
        for view in step.reads.iter().chain(&step.writes) {
            assert_eq!(
                (view.offset, view.size_bytes),
                (0, plan.buffers[view.buffer as usize]),
                "M1 views are whole buffers"
            );
        }
        let read: Vec<u32> = step
            .reads
            .iter()
            .map(|v| {
                *views.get(&v.buffer).unwrap_or_else(|| {
                    panic!(
                        "{name} step {at} reads b{} before anything wrote it\n{}",
                        v.buffer,
                        print_kernel(plan)
                    )
                })
            })
            .collect();
        let written = match step.op {
            PlanOp::Copy => read.clone(),
            _ => {
                let kernel = step
                    .kernel_module_id
                    .and_then(|id| plan.kernel(id))
                    .expect("a dispatch names a kernel");
                dispatch_terms(terms, kernel, &read)
            }
        };
        assert_eq!(written.len(), step.writes.len(), "{name} step {at}");
        for (view, term) in step.writes.iter().zip(written) {
            if step.op == PlanOp::Dispatch && !known.contains(&term) {
                let reduce = matches!(
                    terms.heads[term as usize].0.as_str(),
                    "ReduceSum:stage1" | "ReduceMean:stage1"
                );
                assert!(
                    reduce,
                    "{name} step {at} writes {} into b{}, the term of no Training IR value\n{}",
                    terms.render(term, 4),
                    view.buffer,
                    print_kernel(plan)
                );
            }
            views.insert(view.buffer, term);
        }
    }
}

/// The step plan and the forward plan of `source` compute its Training IR.
fn assert_plan_computes_the_ir(label: &str, source: &str) -> ExecutionPlan {
    let training = training(source);
    let plan = schedule::schedule(
        &training.training,
        training.forward.prediction,
        &training.config,
    )
    .unwrap_or_else(|e| panic!("{label}: the schedule refused: {e}"));
    verify_plan(&plan).unwrap_or_else(|e| panic!("{label}: {e}\n{}", print_kernel(&plan)));

    let mut terms = Terms::default();
    let ir = ir_terms(&mut terms, &training);
    let known: BTreeSet<u32> = ir.values().copied().collect();
    let mut start = BTreeMap::new();
    for data in &plan.data {
        start.insert(
            data.view.buffer,
            terms.make(format!("in:{}", data.name), Vec::new()),
        );
    }
    for parameter in &plan.parameters {
        let term = terms.make(format!("param:{}", parameter.param.0), Vec::new());
        for view in &parameter.views {
            start.insert(view.buffer, term);
        }
    }

    let mut views = start.clone();
    run(
        &mut terms,
        &plan,
        &plan.step,
        &mut views,
        &known,
        &format!("{label} step"),
    );
    let step = &training.training;
    assert_eq!(
        views.get(&plan.loss_view.buffer),
        Some(&ir[&step.training.loss]),
        "{label}: the loss view\n{}",
        print_kernel(&plan)
    );
    for parameter in &plan.parameters {
        let expected = match step.optimizer.updates.get(&parameter.param) {
            Some(update) => ir[update],
            None => start[&parameter.views[0].buffer],
        };
        let got = views[&parameter.views[0].buffer];
        assert_eq!(
            got,
            expected,
            "{label}: parameter {} after the step holds {}, the IR's update is {}\n{}",
            parameter.path,
            terms.render(got, 3),
            terms.render(expected, 3),
            print_kernel(&plan)
        );
    }

    let mut views = start;
    run(
        &mut terms,
        &plan,
        &plan.forward,
        &mut views,
        &known,
        &format!("{label} forward"),
    );
    assert_eq!(
        views.get(&plan.prediction_view.buffer),
        Some(&ir[&training.forward.prediction]),
        "{label}: the forward plan's prediction\n{}",
        print_kernel(&plan)
    );
    plan
}

fn model(params: &str, body: &str, out: &str, loss: &str) -> String {
    format!(
        "model Adv {{
{params}
    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, {out}> {{
{body}
    }}
}}

train Fit {{
    model = Adv;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, {out}>;
    loss = {loss};
    optimizer = sgd(lr = 0.05);
    steps = 30;
    seed = 7;
    device = cpu;
}}
"
    )
}

const W1W2: &str = "    param w1: Tensor<f32, [12, 9]> = xavier_uniform();
    param w2: Tensor<f32, [9, 5]> = xavier_uniform();
";

const MSE: &str = "mse(model.forward(x), y)";

/// `h = relu(matmul(x, w1))` read by `n` matmuls added left to right: the adjoint of `h` sums
/// `n` contributions.
fn h_read_by(n: usize) -> String {
    let mut params = "    param w1: Tensor<f32, [12, 9]> = xavier_uniform();\n".to_owned();
    let mut sum = "matmul(h, w2)".to_owned();
    for i in 2..=n + 1 {
        params.push_str(&format!(
            "    param w{i}: Tensor<f32, [9, 5]> = xavier_uniform();\n"
        ));
        if i > 2 {
            sum = format!("add({sum}, matmul(h, w{i}))");
        }
    }
    model(
        &params,
        &format!("        let h = relu(matmul(x, w1));\n        return {sum};"),
        "[10, 5]",
        MSE,
    )
}

/// The models of the brief's chain conditions, by name.
fn models() -> Vec<(String, String)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut models = vec![
        (
            "read twice".to_owned(),
            model(
                W1W2,
                "        let d = sub(matmul(x, w1), 1.0);
        let e = mul(d, 2.0);
        return matmul(add(e, d), w2);",
                "[10, 5]",
                MSE,
            ),
        ),
        (
            "read by a reduction".to_owned(),
            model(
                W1W2,
                "        let h = relu(matmul(x, w1));
        return add(matmul(h, w2), mul(0.01, sum(mul(add(h, 0.5), 2.0))));",
                "[10, 5]",
                MSE,
            ),
        ),
        (
            "read by a matmul after a relu epilogue".to_owned(),
            model(
                W1W2,
                "        let h = mul(add(relu(matmul(x, w1)), 0.5), 2.0);
        return matmul(h, w2);",
                "[10, 5]",
                MSE,
            ),
        ),
        (
            "a loss chain of rank 0".to_owned(),
            model(
                W1W2,
                "        return matmul(relu(matmul(x, w1)), w2);",
                "[10, 5]",
                "mul(2.0, sub(mse(model.forward(x), y), 0.5))",
            ),
        ),
        (
            "a prediction that ends a chain".to_owned(),
            model(
                W1W2,
                "        return sub(mul(matmul(relu(matmul(x, w1)), w2), 3.0), 0.25);",
                "[10, 5]",
                MSE,
            ),
        ),
        (
            "a parameter leaf".to_owned(),
            "model Adv {
    param w1: Tensor<f32, [12, 9]> = xavier_uniform();
    param s: Tensor<f32, [10, 9]> = ones();

    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, [10, 9]> {
        return add(mul(sub(matmul(x, w1), s), 0.5), s);
    }
}

train Fit {
    model = Adv;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [10, 9]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 30;
    seed = 7;
    device = cpu;
}
"
            .to_owned(),
        ),
        (
            "a transposed chain".to_owned(),
            "model Adv {
    param w1: Tensor<f32, [10, 7]> = xavier_uniform();

    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, [12, 7]> {
        return mul(matmul(transpose(x), w1), 0.5);
    }
}

train Fit {
    model = Adv;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [12, 7]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 30;
    seed = 7;
    device = cpu;
}
"
            .to_owned(),
        ),
    ];
    for n in 2..=8 {
        models.push((format!("an adjoint of {n} contributions"), h_read_by(n)));
    }
    for example in ["mnist_mlp", "linear_regression"] {
        models.push((
            example.to_owned(),
            std::fs::read_to_string(root.join(format!("examples/{example}.mspec")))
                .expect("the example reads"),
        ));
    }
    models
}

#[test]
fn every_fused_plan_computes_the_terms_of_its_training_ir() {
    let mut chains = BTreeSet::new();
    for (label, source) in models() {
        let plan = assert_plan_computes_the_ir(&label, &source);
        for kernel in plan.kernels.iter().filter(|k| k.op.contains('>')) {
            chains.insert(kernel.op.to_owned());
        }
    }
    for expected in ["Sub>Mul@0", "Add>Add@0", "Add>ReluGrad@1"] {
        assert!(
            chains.contains(expected),
            "{expected} among the chains {chains:?}"
        );
    }
}

/// A value read by the next op and by a later op is stored, and the next op starts no chain
/// that absorbs it: `d` of the `read twice` model has a dispatch of its own.
#[test]
fn a_value_read_twice_keeps_its_dispatch() {
    let (_, source) = models().into_iter().next().expect("the first model");
    let training = training(&source);
    let plan = assert_plan_computes_the_ir("read twice", &source);
    let program = &training.training.training.program;
    let d = program
        .nodes
        .iter()
        .find(|n| {
            n.op == TensorOp::Sub
                && n.ty.rank() == 2
                && program
                    .node(n.operands[0])
                    .is_some_and(|m| m.op == TensorOp::MatMul)
        })
        .expect("d");
    assert!(
        plan.step
            .iter()
            .any(|s| s.op == PlanOp::Dispatch && s.value == Some(d.id)),
        "{}",
        print_kernel(&plan)
    );
}
