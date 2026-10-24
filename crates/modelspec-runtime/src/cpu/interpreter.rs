//! The CPU reference interpreter: evaluates a verified [`TensorProgram`] on host tensors
//! (DESIGN.md § 6, § 9 gate 2). Every other backend is compared against it.
//!
//! It runs the core verifier first and executes nothing it rejects; it never sees source
//! (DESIGN.md § 7.1). The element type is the interpreter's type parameter, so a program typed
//! `f32` also runs in `f64`.
//!
//! Reduction order (DESIGN.md § 2.2): a sum accumulates sequentially in row-major index order in
//! the element type, starting from its first term; `matmul` accumulates each output element over
//! the contracting index in ascending order the same way; `mean` is that sum divided by the
//! element count converted to the element type; `mse` is the same sum of `(a - b)^2` divided by
//! the element count converted to the element type. Every product, difference and partial sum is
//! rounded to the element type; nothing is fused.
//!
//! Every value is allocated through one checked path; a value the host cannot hold is a
//! [`RuntimeError::Allocation`].
//!
//! A [`Prepared`] program is verified and planned once and then run any number of times
//! (story:cpu-prepared-training-step): each intermediate is freed after its last consumer, its
//! storage is reused by a later value of the same element count and by the same value in the
//! next run, and the outputs are moved out, not copied. [`evaluate`] is one run of a new plan.

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::ir::tensor::{Dim, ParamId, TensorOp, TensorProgram, TensorType, ValueId};
use modelspec_core::ir::verify::{VerifyError, verify};
use thiserror::Error;

use super::matmul::{self, Kernel, Lhs};
use super::tensor::{Element, HostTensor};

/// Why the interpreter refused a program or its inputs (DESIGN.md § 8.1, runtime class).
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum RuntimeError {
    #[error("program does not verify: {0}")]
    Verify(#[from] VerifyError),
    #[error("no value supplied for input `{name}`")]
    MissingInput { name: String },
    #[error("a value is supplied for `{name}`, which is not an input of the program")]
    UnexpectedInput { name: String },
    #[error("input `{name}` is declared {declared}; the supplied tensor has shape {got:?}")]
    InputShape {
        name: String,
        declared: TensorType,
        got: Vec<u64>,
    },
    #[error("no value supplied for parameter {param} `{path}`")]
    MissingParam { param: ParamId, path: String },
    #[error("a value is supplied for parameter {param}, which the program does not declare")]
    UndeclaredParam { param: ParamId },
    #[error(
        "parameter {param} `{path}` is declared {declared}; the supplied tensor has shape {got:?}"
    )]
    ParamShape {
        param: ParamId,
        path: String,
        declared: TensorType,
        got: Vec<u64>,
    },
    /// The host cannot hold a value the program defines (DESIGN.md § 2.2, "allocations over
    /// backend limits are hard errors"; § 8.1, "runtime allocation failure").
    #[error("cannot allocate {bytes} bytes for {value}")]
    Allocation { value: ValueId, bytes: u128 },
}

/// The result of [`evaluate_all`]: the program outputs and the value of every node.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation<T> {
    /// In the order of [`TensorProgram::outputs`]; a value listed twice appears twice.
    pub outputs: Vec<HostTensor<T>>,
    /// Every node's value, keyed by the value it defines.
    pub values: BTreeMap<ValueId, HostTensor<T>>,
}

/// Per-thread counts of the interpreter's work, for the tests of story:cpu-prepared-training-step.
#[doc(hidden)]
pub mod counters {
    use std::cell::Cell;

    thread_local! {
        pub(super) static PEAK_LIVE_VALUES: Cell<usize> = const { Cell::new(0) };
        pub(super) static IN_PLACE_TRANSPOSES: Cell<u64> = const { Cell::new(0) };
    }

    /// The largest number of values live at once during the last run on this thread that
    /// finished: a value is live from its node until its last consumer has run, or to the end
    /// when the run keeps it. An input or parameter read counts though it is not a copy; a
    /// `transpose` read in place has no value.
    pub fn peak_live_values() -> usize {
        PEAK_LIVE_VALUES.get()
    }

    /// Builds of the in-place transpose table on this thread: one per plan that reads in place.
    pub fn in_place_transposes() -> u64 {
        IN_PLACE_TRANSPOSES.get()
    }
}

/// Evaluates `program` and returns its outputs, in the order of [`TensorProgram::outputs`].
///
/// `inputs` holds a tensor for exactly the program's inputs, by name, and `params` one for
/// exactly the entries of its parameter table; each has its declared shape.
pub fn evaluate<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
    params: &BTreeMap<ParamId, HostTensor<T>>,
) -> Result<Vec<HostTensor<T>>, RuntimeError> {
    Prepared::new(program)?.run(inputs, params)
}

/// [`evaluate`], also returning every intermediate value.
pub fn evaluate_all<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
    params: &BTreeMap<ParamId, HostTensor<T>>,
) -> Result<Evaluation<T>, RuntimeError> {
    verify(program)?;
    let mut prepared = Prepared::planned(program, Keep::Every);
    prepared.execute(inputs, params)?;
    let mut outputs = Vec::new();
    prepared.outputs_into(inputs, params, &mut outputs)?;
    let values = program
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let Hold::Stored(slot) = prepared.plan.hold[i] else {
                unreachable!("a plan that keeps every value stores every node")
            };
            let value = prepared.storage[slot]
                .take()
                .expect("a finished run holds every stored value");
            (node.id, value)
        })
        .collect();
    Ok(Evaluation { outputs, values })
}

/// Which values a run must hold when it ends.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Keep {
    /// Every node's value, each its own copy, as [`evaluate_all`] promises.
    Every,
    /// Only what the outputs need: an input or parameter read is the supplied tensor itself, not
    /// a copy, a `transpose` read only as a `matmul`'s left operand has no value (the `matmul`
    /// reads its operand in place), and every other value is freed after its last consumer.
    Outputs,
}

/// Where a run holds a node's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hold {
    /// The supplied input or parameter tensor itself.
    Supplied,
    /// No value: a `transpose` that the `matmul`s reading it read in place, from the node at
    /// this index.
    InPlace(usize),
    /// Computed into this storage slot.
    Stored(usize),
}

/// What every run of a program does, decided once from the program alone.
struct Plan {
    /// Each node's static shape.
    shapes: Vec<Vec<u64>>,
    /// Each node's operands, as node indices.
    operands: Vec<Vec<usize>>,
    hold: Vec<Hold>,
    /// After each node has run, the nodes whose value it was the last consumer of.
    frees: Vec<Vec<usize>>,
    /// The node of each output, by position.
    outputs: Vec<usize>,
    /// Per output position: whether the output moves its node's storage out, as the last
    /// listing of a stored value does under [`Keep::Outputs`]; every other output is a copy.
    moves: Vec<bool>,
    /// The number of storage slots.
    slots: usize,
}

impl Plan {
    fn new(program: &TensorProgram, keep: Keep) -> Self {
        let index: BTreeMap<ValueId, usize> = program
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id, i))
            .collect();
        let shapes: Vec<Vec<u64>> = program
            .nodes
            .iter()
            .map(|node| static_shape(&node.ty))
            .collect();
        let operands: Vec<Vec<usize>> = program
            .nodes
            .iter()
            .map(|node| node.operands.iter().map(|o| index[o]).collect())
            .collect();
        let outputs: Vec<usize> = program.outputs.iter().map(|o| index[o]).collect();

        let mut hold = vec![Hold::Stored(0); program.nodes.len()];
        if keep == Keep::Outputs {
            for (&transpose, &source) in &in_place_transposes(program) {
                hold[index[&transpose]] = Hold::InPlace(index[&source]);
            }
            for (node, hold) in program.nodes.iter().zip(&mut hold) {
                if matches!(node.op, TensorOp::Input { .. } | TensorOp::ParamRead { .. }) {
                    *hold = Hold::Supplied;
                }
            }
        }

        // The last consumer of each value; a `matmul` reading an in-place transpose consumes
        // the transposed value. A value nothing reads is freed after its own node.
        let mut last: Vec<usize> = (0..program.nodes.len()).collect();
        for (j, reads) in operands.iter().enumerate() {
            if matches!(hold[j], Hold::InPlace(_)) {
                continue;
            }
            for &i in reads {
                let i = match hold[i] {
                    Hold::InPlace(source) => source,
                    _ => i,
                };
                last[i] = last[i].max(j);
            }
        }
        let mut frees = vec![Vec::new(); program.nodes.len()];
        for (i, (&held, &last)) in hold.iter().zip(&last).enumerate() {
            let kept = keep == Keep::Every || outputs.contains(&i);
            if !matches!(held, Hold::InPlace(_)) && !kept {
                frees[last].push(i);
            }
        }

        // A stored value takes a free slot of its element count, else a new slot; a freed
        // value's slot is free once its last consumer has run.
        let counts: Vec<u64> = shapes.iter().map(|s| s.iter().product()).collect();
        let mut free: Vec<(u64, usize)> = Vec::new();
        let mut slots = 0;
        for (i, freed) in frees.iter().enumerate() {
            if let Hold::Stored(_) = hold[i] {
                let slot = match free.iter().position(|&(count, _)| count == counts[i]) {
                    Some(at) => free.swap_remove(at).1,
                    None => {
                        slots += 1;
                        slots - 1
                    }
                };
                hold[i] = Hold::Stored(slot);
            }
            for &f in freed {
                if let Hold::Stored(slot) = hold[f] {
                    free.push((counts[f], slot));
                }
            }
        }

        let moves = outputs
            .iter()
            .enumerate()
            .map(|(p, node)| {
                keep == Keep::Outputs
                    && matches!(hold[*node], Hold::Stored(_))
                    && !outputs[p + 1..].contains(node)
            })
            .collect();
        Self {
            shapes,
            operands,
            hold,
            frees,
            outputs,
            moves,
            slots,
        }
    }
}

/// A program verified and planned once, to be run any number of times.
///
/// Each run checks the supplied tensors, evaluates every node in definition order but a
/// `transpose` read in place, frees each intermediate after its last consumer and moves the
/// outputs out. The storage of the values stays with the plan between runs, so a run after the
/// first allocates only for the outputs it hands out; an output given back with
/// [`Prepared::recycle`] is that output's storage in the next run, shape and elements both, so
/// that a moved output then allocates nothing and only a copy (an output listed again, or one
/// that is a supplied tensor) does.
pub struct Prepared<'p, T> {
    program: &'p TensorProgram,
    plan: Plan,
    /// One tensor per storage slot, refilled in place by each value the slot holds: `None` until
    /// first used, and after its value was moved out and not given back.
    storage: Vec<Option<HostTensor<T>>>,
}

/// A value an operation reads: a supplied tensor or a stored one.
#[derive(Clone, Copy)]
struct View<'v, T> {
    shape: &'v [u64],
    data: &'v [T],
}

impl<'v, T: Element> View<'v, T> {
    fn of(tensor: &'v HostTensor<T>) -> Self {
        Self {
            shape: tensor.shape(),
            data: tensor.data(),
        }
    }

    fn rank(&self) -> usize {
        self.shape.len()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<'p, T: Element> Prepared<'p, T> {
    /// Verifies `program` and plans its runs.
    pub fn new(program: &'p TensorProgram) -> Result<Self, RuntimeError> {
        verify(program)?;
        Ok(Self::planned(program, Keep::Outputs))
    }

    /// Plans the runs of `program`, which the caller has verified.
    pub(crate) fn verified(program: &'p TensorProgram) -> Self {
        Self::planned(program, Keep::Outputs)
    }

    fn planned(program: &'p TensorProgram, keep: Keep) -> Self {
        let plan = Plan::new(program, keep);
        let storage = (0..plan.slots).map(|_| None).collect();
        Self {
            program,
            plan,
            storage,
        }
    }

    /// Runs the program and returns its outputs, in the order of [`TensorProgram::outputs`].
    ///
    /// `inputs` holds a tensor for exactly the program's inputs, by name, and `params` one for
    /// exactly the entries of its parameter table; each has its declared shape.
    pub fn run(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<T>>,
        params: &BTreeMap<ParamId, HostTensor<T>>,
    ) -> Result<Vec<HostTensor<T>>, RuntimeError> {
        let mut outputs = Vec::new();
        self.run_into(inputs, params, &mut outputs)?;
        Ok(outputs)
    }

    /// [`Prepared::run`], replacing the contents of `outputs` with the outputs.
    pub(crate) fn run_into(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<T>>,
        params: &BTreeMap<ParamId, HostTensor<T>>,
        outputs: &mut Vec<HostTensor<T>>,
    ) -> Result<(), RuntimeError> {
        self.execute(inputs, params)?;
        self.outputs_into(inputs, params, outputs)
    }

    /// Gives back `tensor`, the output at `position` of an earlier run, as that output's storage
    /// in the next run. It is dropped instead when the output is a copy, when the output has
    /// storage already, or when its element count is not the output's.
    pub fn recycle(&mut self, position: usize, tensor: HostTensor<T>) {
        if let Some(slot) = self.moved_slot(position)
            && self.storage[slot].is_none()
            && tensor.len() as u64 == self.count(self.plan.outputs[position])
        {
            self.storage[slot] = Some(tensor);
        }
    }

    /// Allocates now the storage of the output at `position` in the next run, unless it has
    /// storage already or is a copy, so that the next run does not allocate it.
    pub(crate) fn reserve_output(&mut self, position: usize) -> Result<(), RuntimeError> {
        if let Some(slot) = self.moved_slot(position)
            && self.storage[slot].is_none()
        {
            let node = self.plan.outputs[position];
            let id = self.program.nodes[node].id;
            let (mut data, n) = allocate(id, self.count(node))?;
            data.resize(n, T::ZERO);
            let shape = self.plan.shapes[node].clone();
            self.storage[slot] = Some(HostTensor::from_parts(shape, data));
        }
        Ok(())
    }

    /// The storage slot the output at `position` moves out, if it moves one.
    fn moved_slot(&self, position: usize) -> Option<usize> {
        if !self.plan.moves.get(position).copied().unwrap_or(false) {
            return None;
        }
        match self.plan.hold[self.plan.outputs[position]] {
            Hold::Stored(slot) => Some(slot),
            _ => None,
        }
    }

    /// The element count of node `node`.
    fn count(&self, node: usize) -> u64 {
        self.plan.shapes[node].iter().product()
    }

    /// Checks the supplied tensors, then evaluates every node in definition order.
    fn execute(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<T>>,
        params: &BTreeMap<ParamId, HostTensor<T>>,
    ) -> Result<(), RuntimeError> {
        check_inputs(self.program, inputs)?;
        check_params(self.program, params)?;
        let (mut live, mut peak) = (0usize, 0usize);
        for i in 0..self.program.nodes.len() {
            match self.plan.hold[i] {
                Hold::InPlace(_) => continue,
                Hold::Supplied => {}
                Hold::Stored(slot) => {
                    // A slot only ever holds values of one element count, so a tensor it holds
                    // has room for this one: refilling it allocates nothing, and copies the
                    // shape only when it differs.
                    let value = match self.storage[slot].take() {
                        Some(mut value) => {
                            value.refill(&self.plan.shapes[i], |data| {
                                data.clear();
                                self.eval_node(i, data, inputs, params);
                            });
                            value
                        }
                        None => {
                            let id = self.program.nodes[i].id;
                            let (mut data, _) = allocate(id, self.count(i))?;
                            self.eval_node(i, &mut data, inputs, params);
                            HostTensor::from_parts(self.plan.shapes[i].clone(), data)
                        }
                    };
                    self.storage[slot] = Some(value);
                }
            }
            live += 1;
            peak = peak.max(live);
            live -= self.plan.frees[i].len();
        }
        counters::PEAK_LIVE_VALUES.set(peak);
        Ok(())
    }

    /// The outputs, into `outputs`: a moved output is its storage's tensor, shape and elements,
    /// and allocates nothing; any other is a copy allocated through [`allocate`].
    fn outputs_into(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<T>>,
        params: &BTreeMap<ParamId, HostTensor<T>>,
        outputs: &mut Vec<HostTensor<T>>,
    ) -> Result<(), RuntimeError> {
        outputs.clear();
        for position in 0..self.plan.outputs.len() {
            let node = self.plan.outputs[position];
            let output = match self.moved_slot(position) {
                Some(slot) => self.storage[slot]
                    .take()
                    .expect("a finished run holds every output it moves"),
                None => {
                    let value = self.view(node, inputs, params);
                    let id = self.program.nodes[node].id;
                    let (mut data, _) = allocate(id, value.len() as u64)?;
                    data.extend_from_slice(value.data);
                    HostTensor::from_parts(value.shape.to_vec(), data)
                }
            };
            outputs.push(output);
        }
        Ok(())
    }

    /// The value of node `i`, which is not a `transpose` read in place.
    fn view<'v>(
        &'v self,
        i: usize,
        inputs: &'v BTreeMap<String, HostTensor<T>>,
        params: &'v BTreeMap<ParamId, HostTensor<T>>,
    ) -> View<'v, T> {
        match self.plan.hold[i] {
            Hold::Stored(slot) => View::of(
                self.storage[slot]
                    .as_ref()
                    .expect("a value is held from its node through its last consumer"),
            ),
            Hold::Supplied => match &self.program.nodes[i].op {
                TensorOp::Input { name } => View::of(&inputs[name]),
                TensorOp::ParamRead { param } => View::of(&params[param]),
                _ => unreachable!("only an input or a parameter read is supplied"),
            },
            Hold::InPlace(_) => unreachable!("an in-place transpose is read only by a matmul"),
        }
    }

    /// Evaluates node `i` into `data`, which it empties first and which has room for the value's
    /// elements already, so this allocates nothing and cannot fail: the run checked the inputs
    /// and parameters before its first node.
    fn eval_node(
        &self,
        i: usize,
        data: &mut Vec<T>,
        inputs: &BTreeMap<String, HostTensor<T>>,
        params: &BTreeMap<ParamId, HostTensor<T>>,
    ) {
        let node = &self.program.nodes[i];
        let operands = &self.plan.operands[i];
        // The verifier guarantees every operand is defined by an earlier node, and that the
        // element count of every type fits `u64`. An in-place transpose is read only as a
        // `matmul`'s left operand, below, so every other operand has a value.
        let arg = |k: usize| self.view(operands[k], inputs, params);
        let lhs = || match self.plan.hold[operands[0]] {
            Hold::InPlace(source) => (self.view(source, inputs, params), Lhs::Transposed),
            _ => (arg(0), Lhs::RowMajor),
        };
        data.clear();
        let n = self.count(i) as usize;
        debug_assert!(data.capacity() >= n, "a value's storage has room for it");
        match &node.op {
            TensorOp::Input { name } => data.extend_from_slice(inputs[name].data()),
            TensorOp::ParamRead { param: id } => data.extend_from_slice(params[id].data()),
            TensorOp::Scalar(v) => data.push(T::from_f32(*v)),
            TensorOp::Add => elementwise(data, arg(0), arg(1), |a, b| a + b),
            TensorOp::Sub => elementwise(data, arg(0), arg(1), |a, b| a - b),
            TensorOp::Mul => elementwise(data, arg(0), arg(1), |a, b| a * b),
            TensorOp::MatMul => {
                let (a, layout) = lhs();
                matmul(data, a, layout, arg(1));
            }
            TensorOp::Relu => data.extend(arg(0).data.iter().map(|&x| relu_where(x, x))),
            TensorOp::Transpose => transpose(data, arg(0)),
            TensorOp::ReduceSum => data.push(sum(arg(0).data.iter().copied())),
            TensorOp::ReduceMean => {
                let a = arg(0);
                data.push(sum(a.data.iter().copied()) / count(a));
            }
            TensorOp::Mse => {
                let (a, b) = (arg(0), arg(1));
                let squares = a.data.iter().zip(b.data).map(|(&x, &y)| {
                    let d = x - y;
                    d * d
                });
                data.push(sum(squares) / count(a));
            }
            TensorOp::ReluGrad => {
                let (a, g) = (arg(0), arg(1));
                data.extend(a.data.iter().zip(g.data).map(|(&a, &g)| relu_where(a, g)));
            }
            TensorOp::BroadcastScalar { .. } => data.resize(n, arg(0).data[0]),
        }
    }
}

/// Each `transpose` whose every reader is a `matmul` reading it as its left operand, and which is
/// not an output, mapped to the value it transposes.
fn in_place_transposes(program: &TensorProgram) -> BTreeMap<ValueId, ValueId> {
    counters::IN_PLACE_TRANSPOSES.set(counters::IN_PLACE_TRANSPOSES.get() + 1);
    let mut readers_elsewhere: BTreeSet<ValueId> = program.outputs.iter().copied().collect();
    for node in &program.nodes {
        for (i, &operand) in node.operands.iter().enumerate() {
            if !(matches!(node.op, TensorOp::MatMul) && i == 0) {
                readers_elsewhere.insert(operand);
            }
        }
    }
    program
        .nodes
        .iter()
        .filter(|node| {
            matches!(node.op, TensorOp::Transpose) && !readers_elsewhere.contains(&node.id)
        })
        .map(|node| (node.id, node.operands[0]))
        .collect()
}

/// Every Input node has a supplied tensor of its declared shape, and nothing else is supplied.
fn check_inputs<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
) -> Result<(), RuntimeError> {
    let mut declared = 0;
    for node in &program.nodes {
        if let TensorOp::Input { name } = &node.op {
            let value = input(inputs, name)?;
            if !has_shape(value, &node.ty) {
                return Err(RuntimeError::InputShape {
                    name: name.clone(),
                    declared: node.ty.clone(),
                    got: value.shape().to_vec(),
                });
            }
            declared += 1;
        }
    }
    // Input names are unique (the verifier refuses a repeated one) and each was found above, so
    // a supplied tensor is unexpected exactly when there are more of them.
    if inputs.len() == declared {
        return Ok(());
    }
    let names: BTreeSet<&str> = program
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            TensorOp::Input { name } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    match inputs.keys().find(|name| !names.contains(name.as_str())) {
        Some(name) => Err(RuntimeError::UnexpectedInput { name: name.clone() }),
        None => Ok(()),
    }
}

/// Every parameter table entry has a supplied tensor of its declared shape, read by a node or
/// not, and no undeclared parameter is supplied.
fn check_params<T: Element>(
    program: &TensorProgram,
    params: &BTreeMap<ParamId, HostTensor<T>>,
) -> Result<(), RuntimeError> {
    for (&id, decl) in &program.params {
        let value = param(program, params, id)?;
        if !has_shape(value, &decl.ty) {
            return Err(RuntimeError::ParamShape {
                param: id,
                path: decl.path.clone(),
                declared: decl.ty.clone(),
                got: value.shape().to_vec(),
            });
        }
    }
    match params.keys().find(|id| !program.params.contains_key(id)) {
        Some(&param) => Err(RuntimeError::UndeclaredParam { param }),
        None => Ok(()),
    }
}

fn input<'a, T>(
    inputs: &'a BTreeMap<String, HostTensor<T>>,
    name: &str,
) -> Result<&'a HostTensor<T>, RuntimeError> {
    inputs.get(name).ok_or_else(|| RuntimeError::MissingInput {
        name: name.to_string(),
    })
}

fn param<'a, T>(
    program: &TensorProgram,
    params: &'a BTreeMap<ParamId, HostTensor<T>>,
    id: ParamId,
) -> Result<&'a HostTensor<T>, RuntimeError> {
    params.get(&id).ok_or_else(|| RuntimeError::MissingParam {
        param: id,
        path: program
            .params
            .get(&id)
            .map(|p| p.path.clone())
            .unwrap_or_default(),
    })
}

fn has_shape<T: Element>(value: &HostTensor<T>, ty: &TensorType) -> bool {
    value.rank() == ty.rank()
        && ty
            .shape
            .iter()
            .zip(value.shape())
            .all(|(dim, &n)| *dim == Dim::Static(n))
}

/// The static shape of a verified type.
fn static_shape(ty: &TensorType) -> Vec<u64> {
    ty.shape
        .iter()
        .map(|dim| match *dim {
            Dim::Static(n) => n,
            Dim::Symbol(_) => unreachable!("a verified program has no symbolic dimension"),
        })
        .collect()
}

/// Room for exactly `count` elements of the value `value`: the one path every value the
/// interpreter makes is allocated through. A count whose bytes exceed what the host can address
/// or provide is a [`RuntimeError::Allocation`], not a panic or an abort.
fn allocate<T>(value: ValueId, count: u64) -> Result<(Vec<T>, usize), RuntimeError> {
    let mut data = Vec::new();
    let n = reserve(&mut data, value, count)?;
    Ok((data, n))
}

/// Room for `count` elements of the value `value` in the empty `data`, through the checked path
/// of [`allocate`]; a buffer that has the room already is not reallocated.
fn reserve<T>(data: &mut Vec<T>, value: ValueId, count: u64) -> Result<usize, RuntimeError> {
    let refused = || RuntimeError::Allocation {
        value,
        bytes: u128::from(count) * size_of::<T>() as u128,
    };
    let n = usize::try_from(count).map_err(|_| refused())?;
    data.try_reserve_exact(n).map_err(|_| refused())?;
    Ok(n)
}

/// `where(a > 0, g, 0)`: `relu(x)` is `relu_where(x, x)`; the derivative at 0 is 0 (DESIGN.md § 4).
fn relu_where<T: Element>(a: T, g: T) -> T {
    if a > T::ZERO { g } else { T::ZERO }
}

/// `Add`, `Sub`, `Mul`: same shapes element by element, or one rank-0 operand against every
/// element of the other, operand order kept.
fn elementwise<T: Element>(
    data: &mut Vec<T>,
    a: View<'_, T>,
    b: View<'_, T>,
    f: impl Fn(T, T) -> T,
) {
    if a.shape == b.shape {
        data.extend(a.data.iter().zip(b.data).map(|(&x, &y)| f(x, y)));
    } else if b.rank() == 0 {
        let y = b.data[0];
        data.extend(a.data.iter().map(|&x| f(x, y)));
    } else {
        let x = a.data[0];
        data.extend(b.data.iter().map(|&y| f(x, y)));
    }
}

/// `[M, K] x [K, N] -> [M, N]`; each element sums its `K` products in ascending `k`, by the
/// kernel [`Kernel::detected`] names (the module [`super::matmul`] holds both kernels). `a` is
/// the `[M, K]` operand, or under [`Lhs::Transposed`] the `[K, M]` value it is the transpose of.
fn matmul<T: Element>(data: &mut Vec<T>, a: View<'_, T>, lhs: Lhs, b: View<'_, T>) {
    let (m, k) = match lhs {
        Lhs::RowMajor => (a.shape[0], a.shape[1]),
        Lhs::Transposed => (a.shape[1], a.shape[0]),
    };
    let (m, k, n) = (m as usize, k as usize, b.shape[1] as usize);
    let (a, b) = (a.data, b.data);
    if !matmul::matmul_into(Kernel::detected(), data, a, lhs, b, m, k, n) {
        let portable = matmul::matmul_into(Kernel::Portable, data, a, lhs, b, m, k, n);
        debug_assert!(portable, "the portable kernel runs everywhere");
    }
}

/// `[M, N] -> [N, M]`.
fn transpose<T: Element>(data: &mut Vec<T>, a: View<'_, T>) {
    let m = a.shape[0] as usize;
    let n = a.shape[1] as usize;
    let a = a.data;
    for j in 0..n {
        for i in 0..m {
            data.push(a[i * n + j]);
        }
    }
}

/// Sequential sum in iteration order, starting from the first term (`+0` for no terms).
fn sum<T: Element>(mut terms: impl Iterator<Item = T>) -> T {
    match terms.next() {
        Some(first) => terms.fold(first, |acc, x| acc + x),
        None => T::ZERO,
    }
}

fn count<T: Element>(a: View<'_, T>) -> T {
    T::from_count(a.len() as u64)
}
