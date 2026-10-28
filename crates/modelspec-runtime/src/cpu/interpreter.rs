//! The CPU reference interpreter: evaluates a verified [`TensorProgram`] on host tensors
//! (DESIGN.md § 6, § 9 gate 2). Every other backend is compared against it.
//!
//! It runs the core verifier first and executes nothing it rejects; it never sees source
//! (DESIGN.md § 7.1). The element type is the interpreter's type parameter, so a program typed
//! `f32` also runs in `f64`.
//!
//! Reduction order (DESIGN.md § 2.2): a sum accumulates sequentially in row-major index order,
//! starting from its first term; `matmul` accumulates each output element over the contracting
//! index in ascending order in the element type. A sum of a whole tensor into one scalar,
//! `reduce_sum`, `mean`, `mse` and the numerator and denominator of `cross_entropy` (and the
//! denominator of its gradient), accumulates in `f64` and is rounded to the element type once at
//! the end (coordinator decision D84): each term is widened to `f64`, an `mse` difference is
//! taken in `f64`, every partial sum is rounded in `f64`, and `mean` and `mse` divide by the
//! element count in `f64` before that one rounding. The first product of a `matmul`, `mse` or
//! other sum of products is rounded and starts the sum, and each later product is fused into
//! the sum with [`Element::mul_add`], rounded once, as the GPU's `Fma` (DESIGN.md § 12
//! ADR-009). Every reduction over a row or a column, and every per-element op, stays in the
//! element type.
//! `SumRows`, the bias gradient of `add_rows`, starts each column from row 0 and adds every later
//! row in ascending order, each partial sum rounded; `EmbeddingGrad` adds rows in ascending `n`.
//! `gelu` and its gradient are computed in the element type, with its `exp` (DESIGN.md § 4).
//! `layer_norm`, `softmax` and their gradient ops work row by row: each row reduction sums in
//! ascending column order from the first column, and the gradient of `gamma` sums rows in
//! ascending order from row 0; a sum of products fuses each product after the first, as the
//! `mse` sum does (DESIGN.md § 4). `attention` and its gradient ops work per sequence, head and
//! query over the unmasked keys only, every reduction in ascending index order and every sum of
//! products fused after its first product; the gradient ops recompute the probabilities through
//! the forward's code path, and a masked key's rows of the `k` and `v` gradients stay `+0`
//! (DESIGN.md § 4, coordinator decision D37). The key mask is checked before the op runs: a
//! value other than 0 or 1 is [`RuntimeError::KeyMaskValue`], a sequence with no unmasked key
//! [`RuntimeError::NoUnmaskedKey`]. `cross_entropy` and its gradient op recompute each row's
//! maximum and sum of `exp(x - max)` through one code path, every reduction in ascending index
//! order; the loss is the weighted sum of the row losses, its first product rounded and each
//! later one fused, over the plain sum of the weights, both sums in `f64`, and a weight sum of 0
//! gives a loss and a
//! gradient of `+0` (DESIGN.md § 4, coordinator decision D43). Its labels and weights are checked
//! before the op runs, row by row: a label outside the classes is
//! [`RuntimeError::LabelOutOfRange`], a negative or non-finite weight
//! [`RuntimeError::LossWeight`].
//! `SigregColumns` and its gradient op work column by column, in the element type: each column
//! mean of `cos(t_j P)` and `sin(t_j P)` sums in ascending row order from row 0, with `sin` and
//! `cos` from [`Element::sin_cos`], and the sum over the eight knots runs in ascending `j`, its
//! first product rounded and each later one fused (DESIGN.md § 4, story:jepa-sigreg-cpu).
//! `sigreg` itself never runs: lowering expands it and the verifier refuses it.
//!
//! `dropout` and its gradient op read the `i32[2]` RNG token of their site and the run's seed
//! ([`Prepared::set_seed`], [`evaluate_seeded`]): element `i` in row-major order is kept when
//! [`crate::train::init::dropout_keeps`] of the seed, the token's draw id, `i` and `p` holds,
//! Philox stream tag 4. A kept element is `x * scale` (`g * scale` for the gradient) with
//! `scale = 1.0f32 / (1.0f32 - p)` computed once per op in `f32` and converted to the element
//! type; a dropped one is `+0`. Both ops build the mask through one function, so the gradient
//! recomputes the forward's mask and no mask is stored (DESIGN.md § 2.3, § 4, coordinator
//! decision D47). A program with a dropout run without a seed is [`RuntimeError::NoSeed`]
//! before any node runs.
//!
//! Every value is allocated through one checked path; a value the host cannot hold is a
//! [`RuntimeError::Allocation`].
//!
//! A [`Prepared`] program is verified and planned once and then run any number of times
//! (story:cpu-prepared-training-step): each intermediate is freed after its last consumer, its
//! storage is reused by a later value of the same element count and by the same value in the
//! next run, and the outputs are moved out, not copied. [`evaluate`] is one run of a new plan.

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::ir::tensor::{
    DType, Dim, ParamId, TensorOp, TensorProgram, TensorType, ValueId,
};
use modelspec_core::ir::verify::{VerifyError, verify};
use thiserror::Error;

use super::matmul::{self, Kernel, Lhs};
use super::tensor::{Element, HostTensor};
use crate::train::init::{dropout_keeps, token_id};

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
    /// A tensor supplied among the values of the other element type: an `f32` tensor for an
    /// `i32` input, or ids for an `f32` input.
    #[error("input `{name}` is declared {declared}; it is supplied as the other element type")]
    InputDType { name: String, declared: TensorType },
    /// A row id outside the table an `embedding` or its gradient reads (DESIGN.md § 4).
    #[error(
        "input `{name}` holds id {id} at position {index}, outside the {rows} rows of its table"
    )]
    IdOutOfRange {
        name: String,
        index: usize,
        id: i32,
        rows: u64,
    },
    /// A key mask of `attention` or one of its gradient ops holding a value other than 0 and 1
    /// (DESIGN.md § 4); NaN is such a value.
    #[error(
        "the key mask of {value} holds a value other than 0 or 1 at sequence {sequence}, key {key}"
    )]
    KeyMaskValue {
        value: ValueId,
        sequence: usize,
        key: usize,
    },
    /// A key mask of `attention` or one of its gradient ops that masks every key of a sequence,
    /// whose softmax would divide by zero (DESIGN.md § 4).
    #[error("the key mask of {value} masks every key of sequence {sequence}")]
    NoUnmaskedKey { value: ValueId, sequence: usize },
    /// A label of `cross_entropy` or its gradient op outside `[0, V)`, `V` the classes of its
    /// logits (DESIGN.md § 4, coordinator decision D43).
    #[error(
        "input `{name}` holds label {label} at row {row} of {value}, outside the {classes} \
         classes of its logits"
    )]
    LabelOutOfRange {
        name: String,
        value: ValueId,
        row: usize,
        label: i32,
        classes: u64,
    },
    /// A weight of `cross_entropy` or its gradient op that is negative or not finite; `-0.0` is
    /// not negative (DESIGN.md § 4, coordinator decision D43).
    #[error("the weight of row {row} of {value} is negative or not finite")]
    LossWeight { value: ValueId, row: usize },
    /// A `Dropout` or `DropoutGrad` in a run without a seed: its mask has no key (DESIGN.md
    /// § 2.3).
    #[error("{value} is a dropout, and the run has no seed to key its mask")]
    NoSeed { value: ValueId },
    /// A `Dropout` or `DropoutGrad` over more than `2^32` elements: the last ones have no Philox
    /// counter (DESIGN.md § 2.3).
    #[error("{value} is a dropout over {count} elements; a draw addresses at most 2^32")]
    DropoutElements { value: ValueId, count: u64 },
}

/// No ids, for the programs that read none.
static NO_IDS: BTreeMap<String, HostTensor<i32>> = BTreeMap::new();

/// The result of [`evaluate_all`]: the program outputs and the value of every node.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation<T> {
    /// In the order of [`TensorProgram::outputs`]; a value listed twice appears twice.
    pub outputs: Vec<HostTensor<T>>,
    /// Every node's value, keyed by the value it defines, except the `i32` inputs: their values
    /// are the supplied ids.
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
/// `inputs` holds a tensor for exactly the program's `f32` inputs, by name, and `params` one for
/// exactly the entries of its parameter table; each has its declared shape. A program with an
/// `i32` input runs through [`evaluate_with_ids`].
pub fn evaluate<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
    params: &BTreeMap<ParamId, HostTensor<T>>,
) -> Result<Vec<HostTensor<T>>, RuntimeError> {
    evaluate_with_ids(program, inputs, &NO_IDS, params)
}

/// [`evaluate`] with `ids`, a tensor for exactly the program's `i32` inputs, by name, each of
/// its declared shape. Every id an `embedding` or its gradient reads is checked against the rows
/// of its table before the first node runs: an id outside them is
/// [`RuntimeError::IdOutOfRange`].
pub fn evaluate_with_ids<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
    ids: &BTreeMap<String, HostTensor<i32>>,
    params: &BTreeMap<ParamId, HostTensor<T>>,
) -> Result<Vec<HostTensor<T>>, RuntimeError> {
    Prepared::new(program)?.run_with_ids(inputs, ids, params)
}

/// [`evaluate_with_ids`] under the run's `seed`, the key of every dropout mask: `ids` holds the
/// RNG token of each dropout site beside any row ids or labels.
pub fn evaluate_seeded<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
    ids: &BTreeMap<String, HostTensor<i32>>,
    params: &BTreeMap<ParamId, HostTensor<T>>,
    seed: u64,
) -> Result<Vec<HostTensor<T>>, RuntimeError> {
    let mut prepared = Prepared::new(program)?;
    prepared.set_seed(seed);
    prepared.run_with_ids(inputs, ids, params)
}

/// [`evaluate`], also returning every intermediate value.
pub fn evaluate_all<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
    params: &BTreeMap<ParamId, HostTensor<T>>,
) -> Result<Evaluation<T>, RuntimeError> {
    evaluate_all_with_ids(program, inputs, &NO_IDS, params)
}

/// [`evaluate_with_ids`], also returning every intermediate value.
pub fn evaluate_all_with_ids<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
    ids: &BTreeMap<String, HostTensor<i32>>,
    params: &BTreeMap<ParamId, HostTensor<T>>,
) -> Result<Evaluation<T>, RuntimeError> {
    verify(program)?;
    let mut prepared = Prepared::planned(program, Keep::Every, &[]);
    prepared.execute(inputs, ids, params)?;
    let mut outputs = Vec::new();
    prepared.outputs_into(inputs, params, &mut outputs)?;
    let values = program
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| {
            let slot = match prepared.plan.hold[i] {
                Hold::Stored(slot) => slot,
                Hold::Ids => return None,
                Hold::Supplied | Hold::InPlace(_) => {
                    unreachable!("a plan that keeps every value stores every f32 node")
                }
            };
            let value = prepared.storage[slot]
                .take()
                .expect("a finished run holds every stored value");
            Some((node.id, value))
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
    /// The supplied ids of an `i32` input, read only by `Embedding` and `EmbeddingGrad`, as the
    /// labels of `CrossEntropy` and `CrossEntropyGrad` and as the RNG token of `Dropout` and
    /// `DropoutGrad`, under either [`Keep`].
    Ids,
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
    /// Each read of ids, in node order: the `i32` input node and the rows of the table it
    /// indexes.
    id_reads: Vec<(usize, u64)>,
    /// Each `Dropout` and `DropoutGrad` node, in node order.
    dropouts: Vec<usize>,
    /// The nodes whose value a run holds in its own storage when it ends, beside the outputs,
    /// for [`Prepared::kept`]: never supplied, read in place, freed or moved out.
    kept: Vec<usize>,
}

impl Plan {
    fn new(program: &TensorProgram, keep: Keep, kept: &[ValueId]) -> Self {
        let index: BTreeMap<ValueId, usize> = program
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id, i))
            .collect();
        let kept: Vec<usize> = kept.iter().filter_map(|v| index.get(v).copied()).collect();
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
                if !kept.contains(&index[&transpose]) {
                    hold[index[&transpose]] = Hold::InPlace(index[&source]);
                }
            }
            for (i, (node, hold)) in program.nodes.iter().zip(&mut hold).enumerate() {
                if matches!(node.op, TensorOp::Input { .. } | TensorOp::ParamRead { .. })
                    && !kept.contains(&i)
                {
                    *hold = Hold::Supplied;
                }
            }
        }
        for (node, hold) in program.nodes.iter().zip(&mut hold) {
            if node.ty.dtype == DType::I32 {
                *hold = Hold::Ids;
            }
        }
        let id_reads = program
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| match node.op {
                TensorOp::Embedding => Some((operands[i][1], shapes[operands[i][0]][0])),
                TensorOp::EmbeddingGrad { .. } => Some((operands[i][0], shapes[i][0])),
                _ => None,
            })
            .collect();
        let dropouts = program
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.op.dropout_rate().is_some())
            .map(|(i, _)| i)
            .collect();

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
            let held_to_end = keep == Keep::Every || outputs.contains(&i) || kept.contains(&i);
            if !matches!(held, Hold::InPlace(_)) && !held_to_end {
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
                    && !kept.contains(node)
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
            id_reads,
            dropouts,
            kept,
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
    /// The run's seed, the key of every dropout mask; `None` until [`Prepared::set_seed`].
    seed: Option<u64>,
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
        Ok(Self::planned(program, Keep::Outputs, &[]))
    }

    /// Plans the runs of `program`, which the caller has verified, keeping each value of `kept`
    /// as [`Prepared::keeping`] does.
    pub(crate) fn verified(program: &'p TensorProgram, kept: &[ValueId]) -> Self {
        Self::planned(program, Keep::Outputs, kept)
    }

    /// Verifies `program` and plans its runs as [`Prepared::new`] does, except that each value
    /// of `kept` is computed into the plan's own storage and held there after a run, until the
    /// next run refills it, for [`Prepared::kept`]. Every other intermediate is still freed after
    /// its last consumer.
    pub(crate) fn keeping(
        program: &'p TensorProgram,
        kept: &[ValueId],
    ) -> Result<Self, RuntimeError> {
        verify(program)?;
        Ok(Self::planned(program, Keep::Outputs, kept))
    }

    /// The value of `value` in the last run that finished, if the plan keeps it
    /// ([`Prepared::keeping`]); `None` before the first run and for a value it does not keep.
    pub(crate) fn kept(&self, value: ValueId) -> Option<&HostTensor<T>> {
        let node = self.program.nodes.iter().position(|n| n.id == value)?;
        if !self.plan.kept.contains(&node) {
            return None;
        }
        match self.plan.hold[node] {
            Hold::Stored(slot) => self.storage[slot].as_ref(),
            _ => None,
        }
    }

    /// The program this plan runs.
    pub(crate) fn program(&self) -> &'p TensorProgram {
        self.program
    }

    fn planned(program: &'p TensorProgram, keep: Keep, kept: &[ValueId]) -> Self {
        let plan = Plan::new(program, keep, kept);
        let storage = (0..plan.slots).map(|_| None).collect();
        Self {
            program,
            plan,
            storage,
            seed: None,
        }
    }

    /// Sets the seed every later run keys its dropout masks with, the `train` block's seed.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
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
        self.run_with_ids(inputs, &NO_IDS, params)
    }

    /// [`Prepared::run`] with `ids`, a tensor for exactly the program's `i32` inputs, as
    /// [`evaluate_with_ids`] takes them.
    pub fn run_with_ids(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<T>>,
        ids: &BTreeMap<String, HostTensor<i32>>,
        params: &BTreeMap<ParamId, HostTensor<T>>,
    ) -> Result<Vec<HostTensor<T>>, RuntimeError> {
        let mut outputs = Vec::new();
        self.execute(inputs, ids, params)?;
        self.outputs_into(inputs, params, &mut outputs)?;
        Ok(outputs)
    }

    /// [`Prepared::run_with_ids`], replacing the contents of `outputs` with the outputs.
    pub(crate) fn run_into(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<T>>,
        ids: &BTreeMap<String, HostTensor<i32>>,
        params: &BTreeMap<ParamId, HostTensor<T>>,
        outputs: &mut Vec<HostTensor<T>>,
    ) -> Result<(), RuntimeError> {
        self.execute(inputs, ids, params)?;
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

    /// Checks the supplied tensors and every id read, then evaluates every node in definition
    /// order.
    fn execute(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<T>>,
        ids: &BTreeMap<String, HostTensor<i32>>,
        params: &BTreeMap<ParamId, HostTensor<T>>,
    ) -> Result<(), RuntimeError> {
        check_inputs(self.program, inputs, ids)?;
        check_params(self.program, params)?;
        for &node in &self.plan.dropouts {
            let value = self.program.nodes[node].id;
            if self.seed.is_none() {
                return Err(RuntimeError::NoSeed { value });
            }
            let count = self.count(node);
            if count > 1 << 32 {
                return Err(RuntimeError::DropoutElements { value, count });
            }
        }
        for &(node, rows) in &self.plan.id_reads {
            let name = input_name(self.program, node);
            let outside = ids[name]
                .data()
                .iter()
                .enumerate()
                .find(|&(_, &id)| u64::try_from(id).map_or(true, |id| id >= rows));
            if let Some((index, &id)) = outside {
                return Err(RuntimeError::IdOutOfRange {
                    name: name.to_owned(),
                    index,
                    id,
                    rows,
                });
            }
        }
        let (mut live, mut peak) = (0usize, 0usize);
        for i in 0..self.program.nodes.len() {
            let node = &self.program.nodes[i];
            if node.op.attention_heads().is_some() {
                let mask = self.view(self.plan.operands[i][3], inputs, params);
                check_key_mask(node.id, mask)?;
            }
            if matches!(node.op, TensorOp::CrossEntropy | TensorOp::CrossEntropyGrad) {
                let reads = &self.plan.operands[i];
                check_loss_rows(
                    node.id,
                    input_name(self.program, reads[1]),
                    self.ids(reads[1], ids),
                    self.view(reads[2], inputs, params).data,
                    self.plan.shapes[reads[0]][1],
                )?;
            }
            match self.plan.hold[i] {
                Hold::InPlace(_) => continue,
                Hold::Supplied | Hold::Ids => {}
                Hold::Stored(slot) => {
                    // A slot only ever holds values of one element count, so a tensor it holds
                    // has room for this one: refilling it allocates nothing, and copies the
                    // shape only when it differs.
                    let value = match self.storage[slot].take() {
                        Some(mut value) => {
                            value.refill(&self.plan.shapes[i], |data| {
                                data.clear();
                                self.eval_node(i, data, inputs, ids, params);
                            });
                            value
                        }
                        None => {
                            let id = self.program.nodes[i].id;
                            let (mut data, _) = allocate(id, self.count(i))?;
                            self.eval_node(i, &mut data, inputs, ids, params);
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
            Hold::Ids => unreachable!("ids are read only as the ids of an embedding"),
        }
    }

    /// The ids of the `i32` input node `i`.
    fn ids<'v>(&self, i: usize, ids: &'v BTreeMap<String, HostTensor<i32>>) -> &'v [i32] {
        ids[input_name(self.program, i)].data()
    }

    /// `values` under the dropout mask of the token node `token` at drop probability `p`, into
    /// `data`: each kept element times `1.0f32 / (1.0f32 - p)`, each dropped one `+0`. The one
    /// mask path of `Dropout` and `DropoutGrad`.
    fn masked(
        &self,
        data: &mut Vec<T>,
        values: &[T],
        token: usize,
        p: f32,
        ids: &BTreeMap<String, HostTensor<i32>>,
    ) {
        let seed = self
            .seed
            .expect("the run checked that a dropout has a seed");
        let token = self.ids(token, ids);
        let id = token_id([token[0], token[1]]);
        let scale = T::from_f32(1.0f32 / (1.0f32 - p));
        data.extend(values.iter().enumerate().map(|(i, &v)| {
            if dropout_keeps(seed, id, i as u32, p) {
                v * scale
            } else {
                T::ZERO
            }
        }));
    }

    /// Evaluates node `i` into `data`, which it empties first and which has room for the value's
    /// elements already, so this allocates nothing and cannot fail: the run checked the inputs,
    /// the parameters and every id read before its first node.
    fn eval_node(
        &self,
        i: usize,
        data: &mut Vec<T>,
        inputs: &BTreeMap<String, HostTensor<T>>,
        ids: &BTreeMap<String, HostTensor<i32>>,
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
            TensorOp::ReduceSum => data.push(T::from_f64(wide_sum(arg(0).data))),
            TensorOp::ReduceMean => {
                let a = arg(0);
                data.push(T::from_f64(wide_sum(a.data) / wide_count(a)));
            }
            TensorOp::Mse => {
                let (a, b) = (arg(0), arg(1));
                let diffs = a.data.iter().zip(b.data).map(|(&x, &y)| {
                    let d = x.to_f64() - y.to_f64();
                    (d, d)
                });
                data.push(T::from_f64(sum_of_products(diffs) / wide_count(a)));
            }
            TensorOp::ReluGrad => {
                let (a, g) = (arg(0), arg(1));
                data.extend(a.data.iter().zip(g.data).map(|(&a, &g)| relu_where(a, g)));
            }
            TensorOp::BroadcastScalar { .. } => data.resize(n, arg(0).data[0]),
            TensorOp::Embedding => {
                let table = arg(0);
                let h = table.shape[1] as usize;
                for &id in self.ids(operands[1], ids) {
                    let row = id as usize * h;
                    data.extend_from_slice(&table.data[row..row + h]);
                }
            }
            TensorOp::EmbeddingGrad { .. } => {
                let g = arg(1);
                let h = g.shape[1] as usize;
                data.resize(n, T::ZERO);
                for (k, &id) in self.ids(operands[0], ids).iter().enumerate() {
                    let row = id as usize * h;
                    for (out, &x) in data[row..row + h].iter_mut().zip(&g.data[k * h..]) {
                        *out = *out + x;
                    }
                }
            }
            TensorOp::AddRows => {
                let (x, b) = (arg(0), arg(1));
                let h = b.len();
                data.extend(
                    x.data
                        .chunks_exact(h)
                        .flat_map(|row| row.iter().zip(b.data).map(|(&x, &b)| x + b)),
                );
            }
            TensorOp::SumRows => {
                let g = arg(0);
                let h = n;
                let mut rows = g.data.chunks_exact(h);
                match rows.next() {
                    Some(first) => data.extend_from_slice(first),
                    None => data.resize(n, T::ZERO),
                }
                for row in rows {
                    for (out, &x) in data.iter_mut().zip(row) {
                        *out = *out + x;
                    }
                }
            }
            TensorOp::Gelu => data.extend(arg(0).data.iter().map(|&x| gelu(x))),
            TensorOp::GeluGrad => {
                let (a, g) = (arg(0), arg(1));
                data.extend(a.data.iter().zip(g.data).map(|(&a, &g)| gelu_grad(a, g)));
            }
            TensorOp::LayerNorm => {
                let (x, gamma, beta) = (arg(0), arg(1), arg(2));
                for row in x.data.chunks_exact(gamma.len()) {
                    let (mean, rstd) = row_moments(row);
                    data.extend(
                        row.iter()
                            .zip(gamma.data.iter().zip(beta.data))
                            .map(|(&x, (&gamma, &beta))| (x - mean) * rstd * gamma + beta),
                    );
                }
            }
            TensorOp::LayerNormGradX => {
                let (x, gamma, g) = (arg(0), arg(1), arg(2));
                let h = gamma.len();
                let width = T::from_count(h as u64);
                for (row, g_row) in x.data.chunks_exact(h).zip(g.data.chunks_exact(h)) {
                    let (mean, rstd) = row_moments(row);
                    let xhat = |j: usize| (row[j] - mean) * rstd;
                    let gx = |j: usize| g_row[j] * gamma.data[j];
                    let mean_gx = sum((0..h).map(gx)) / width;
                    let mean_gx_xhat = sum_of_products((0..h).map(|j| (gx(j), xhat(j)))) / width;
                    data.extend(
                        (0..h).map(|j| rstd * ((gx(j) - mean_gx) - xhat(j) * mean_gx_xhat)),
                    );
                }
            }
            TensorOp::LayerNormGradGamma => {
                let (x, g) = (arg(0), arg(1));
                let h = n;
                let rows = x.data.chunks_exact(h).zip(g.data.chunks_exact(h));
                for (k, (row, g_row)) in rows.enumerate() {
                    let (mean, rstd) = row_moments(row);
                    let terms = row.iter().zip(g_row).map(|(&x, &g)| (g, (x - mean) * rstd));
                    if k == 0 {
                        data.extend(terms.map(|(g, xhat)| g * xhat));
                    } else {
                        for (out, (g, xhat)) in data.iter_mut().zip(terms) {
                            *out = g.mul_add(xhat, *out);
                        }
                    }
                }
                data.resize(n, T::ZERO);
            }
            TensorOp::Softmax => {
                let x = arg(0);
                for row in x.data.chunks_exact(x.shape[1] as usize) {
                    let max = row[1..]
                        .iter()
                        .fold(row[0], |m, &v| if v > m { v } else { m });
                    let start = data.len();
                    data.extend(row.iter().map(|&v| (v - max).exp()));
                    let total = sum(data[start..].iter().copied());
                    for e in &mut data[start..] {
                        *e = *e / total;
                    }
                }
            }
            TensorOp::SoftmaxGrad => {
                let (y, g) = (arg(0), arg(1));
                let k = y.shape[1] as usize;
                for (y_row, g_row) in y.data.chunks_exact(k).zip(g.data.chunks_exact(k)) {
                    let d = sum_of_products(g_row.iter().copied().zip(y_row.iter().copied()));
                    data.extend(y_row.iter().zip(g_row).map(|(&y, &g)| y * (g - d)));
                }
            }
            TensorOp::Attention { heads } => {
                let at = Attend::new(arg(0), arg(1), arg(2), arg(3), *heads);
                data.resize(n, T::ZERO);
                for i in 0..at.rows() {
                    for head in 0..at.heads {
                        let softmax = at.softmax(i, head);
                        let mut first = true;
                        for j in at.keys(i) {
                            let p = at.probability(i, j, head, softmax);
                            for c in at.columns(head) {
                                let out = &mut data[i * at.h + c];
                                *out = fused(first, p, at.v.data[j * at.h + c], *out);
                            }
                            first = false;
                        }
                    }
                }
            }
            TensorOp::AttentionGradQ { heads } => {
                let at = Attend::new(arg(0), arg(1), arg(2), arg(3), *heads);
                let g = arg(4);
                data.resize(n, T::ZERO);
                for i in 0..at.rows() {
                    for head in 0..at.heads {
                        let softmax = at.softmax(i, head);
                        let d = at.row_dot(i, head, softmax, g);
                        let mut first = true;
                        for j in at.keys(i) {
                            let ds = at.score_grad(i, j, head, softmax, d, g);
                            for c in at.columns(head) {
                                let out = &mut data[i * at.h + c];
                                *out = fused(first, ds, at.k.data[j * at.h + c], *out);
                            }
                            first = false;
                        }
                        for c in at.columns(head) {
                            data[i * at.h + c] = data[i * at.h + c] * at.scale;
                        }
                    }
                }
            }
            TensorOp::AttentionGradK { heads } => {
                let at = Attend::new(arg(0), arg(1), arg(2), arg(3), *heads);
                let g = arg(4);
                data.resize(n, T::ZERO);
                for i in 0..at.rows() {
                    let first = i % at.s == 0;
                    for head in 0..at.heads {
                        let softmax = at.softmax(i, head);
                        let d = at.row_dot(i, head, softmax, g);
                        for j in at.keys(i) {
                            let ds = at.score_grad(i, j, head, softmax, d, g);
                            for c in at.columns(head) {
                                let out = &mut data[j * at.h + c];
                                *out = fused(first, ds, at.q.data[i * at.h + c], *out);
                            }
                        }
                    }
                }
                for j in 0..at.rows() {
                    if at.unmasked(j) {
                        for out in &mut data[j * at.h..(j + 1) * at.h] {
                            *out = *out * at.scale;
                        }
                    }
                }
            }
            TensorOp::AttentionGradV { heads } => {
                let at = Attend::new(arg(0), arg(1), arg(2), arg(3), *heads);
                let g = arg(4);
                data.resize(n, T::ZERO);
                for i in 0..at.rows() {
                    let first = i % at.s == 0;
                    for head in 0..at.heads {
                        let softmax = at.softmax(i, head);
                        for j in at.keys(i) {
                            let p = at.probability(i, j, head, softmax);
                            for c in at.columns(head) {
                                let out = &mut data[j * at.h + c];
                                *out = fused(first, p, g.data[i * at.h + c], *out);
                            }
                        }
                    }
                }
            }
            TensorOp::CrossEntropy => {
                let labels = self.ids(operands[1], ids);
                data.push(cross_entropy(arg(0), labels, arg(2).data));
            }
            TensorOp::CrossEntropyGrad => {
                let labels = self.ids(operands[1], ids);
                cross_entropy_grad(data, arg(0), labels, arg(2).data, arg(3).data[0]);
            }
            TensorOp::Dropout { p } => self.masked(data, arg(0).data, operands[1], *p, ids),
            TensorOp::DropoutGrad { p } => self.masked(data, arg(1).data, operands[0], *p, ids),
            TensorOp::StopGradient => data.extend_from_slice(arg(0).data),
            TensorOp::Sqrt => data.extend(arg(0).data.iter().map(|&x| x.sqrt())),
            TensorOp::Div => elementwise(data, arg(0), arg(1), |a, b| a / b),
            TensorOp::SigregColumns => sigreg_columns(data, arg(0)),
            TensorOp::SigregColumnsGrad => sigreg_columns_grad(data, arg(0), arg(1).data),
            TensorOp::Sigreg { .. } => {
                unreachable!("the verifier refuses `sigreg`, which lowering expands")
            }
        }
    }
}

/// Checks the labels and weights of the `cross_entropy` op or gradient op `value` before it
/// runs, in ascending row order and the label of a row before its weight: each label of the
/// input `name` in `[0, classes)`, each weight finite and not negative, `-0.0` passing
/// (coordinator decision D43).
fn check_loss_rows<T: Element>(
    value: ValueId,
    name: &str,
    labels: &[i32],
    weights: &[T],
    classes: u64,
) -> Result<(), RuntimeError> {
    for (row, (&label, &weight)) in labels.iter().zip(weights).enumerate() {
        if u64::try_from(label).map_or(true, |label| label >= classes) {
            return Err(RuntimeError::LabelOutOfRange {
                name: name.to_owned(),
                value,
                row,
                label,
                classes,
            });
        }
        // A NaN or an infinity times 0 is NaN, which equals nothing.
        let finite = weight * T::ZERO == T::ZERO;
        if !finite || weight < T::ZERO {
            return Err(RuntimeError::LossWeight { value, row });
        }
    }
    Ok(())
}

/// The maximum of `row`, the first element replaced by each later greater one in ascending
/// order, and the sum of `exp(x - max)` over `row` in ascending order from its first term: the
/// row softmax of `cross_entropy` and its gradient op (coordinator decision D43).
fn row_softmax<T: Element>(row: &[T]) -> (T, T) {
    let max = row[1..]
        .iter()
        .fold(row[0], |m, &v| if v > m { v } else { m });
    (max, sum(row.iter().map(|&v| (v - max).exp())))
}

/// `sum_n weights[n] * loss_n / sum_n weights[n]` over the rows of `logits` (D43): the row loss
/// `max + ln(sum) - logits[n, labels[n]]` in the element type, the numerator a
/// [`sum_of_products`] and the denominator a [`wide_sum`], both whole-tensor sums in `f64` in
/// ascending `n`, their quotient rounded to the element type once, and `+0` when the
/// denominator is 0 (D30, D84).
fn cross_entropy<T: Element>(logits: View<'_, T>, labels: &[i32], weights: &[T]) -> T {
    let v = logits.shape[1] as usize;
    let losses = logits
        .data
        .chunks_exact(v)
        .zip(labels)
        .map(|(row, &label)| {
            let (max, total) = row_softmax(row);
            (max + total.ln() - row[label as usize]).to_f64()
        });
    let numerator = sum_of_products(weights.iter().map(|w| w.to_f64()).zip(losses));
    let denominator = wide_sum(weights);
    if denominator == 0.0 {
        T::ZERO
    } else {
        T::from_f64(numerator / denominator)
    }
}

/// The gradient of [`cross_entropy`] with respect to `logits`, into `data`: per element
/// `g * weights[n] / sum(weights) * (p[n, v] - [v == labels[n]])` left to right, `p` recomputed
/// through [`row_softmax`], `sum(weights)` the [`wide_sum`] rounded to the element type once;
/// every element `+0` when the weight sum is 0 (D30, D43, D84).
fn cross_entropy_grad<T: Element>(
    data: &mut Vec<T>,
    logits: View<'_, T>,
    labels: &[i32],
    weights: &[T],
    g: T,
) {
    let denominator = T::from_f64(wide_sum(weights));
    if denominator == T::ZERO {
        data.resize(logits.len(), T::ZERO);
        return;
    }
    let one = T::from_f32(1.0);
    let v = logits.shape[1] as usize;
    let rows = logits.data.chunks_exact(v).zip(labels).zip(weights);
    for ((row, &label), &weight) in rows {
        let (max, total) = row_softmax(row);
        let scale = g * weight / denominator;
        data.extend(row.iter().enumerate().map(|(c, &x)| {
            let target = if c == label as usize { one } else { T::ZERO };
            scale * ((x - max).exp() / total - target)
        }));
    }
}

/// Checks the key mask `mask` of the `attention` op or gradient op `value` before it runs:
/// every value 0 or 1, and at least one 1 in each sequence (a row of `mask`), in ascending
/// sequence order and, within one, ascending key order (coordinator decision D37).
fn check_key_mask<T: Element>(value: ValueId, mask: View<'_, T>) -> Result<(), RuntimeError> {
    let one = T::from_f32(1.0);
    let s = mask.shape[1] as usize;
    for (sequence, row) in mask.data.chunks_exact(s).enumerate() {
        if let Some(key) = row.iter().position(|&m| m != T::ZERO && m != one) {
            return Err(RuntimeError::KeyMaskValue {
                value,
                sequence,
                key,
            });
        }
        if !row.contains(&one) {
            return Err(RuntimeError::NoUnmaskedKey { value, sequence });
        }
    }
    Ok(())
}

/// `a * b` when `first`, else `a * b + acc` fused and rounded once: one step of an accumulation
/// that starts from its first product, as [`sum_of_products`] does (DESIGN.md § 12 ADR-009).
fn fused<T: Element>(first: bool, a: T, b: T, acc: T) -> T {
    if first { a * b } else { a.mul_add(b, acc) }
}

/// The operands of one `attention` or gradient op, `q`, `k`, `v` of shape `[N, H]` and a key
/// mask of shape `[B, S]` that [`check_key_mask`] passed, with `heads` dividing `H` (DESIGN.md
/// § 4, coordinator decision D37). Row `i` is token `i % S` of sequence `i / S`, and the key mask
/// entry of row `j` is element `j` of the mask; head `a` reads columns `a * H/A` to
/// `(a + 1) * H/A - 1`. A masked key is skipped: it adds nothing to any sum, so no value of its
/// `k` or `v` row reaches a result.
struct Attend<'v, T> {
    q: View<'v, T>,
    k: View<'v, T>,
    v: View<'v, T>,
    mask: View<'v, T>,
    heads: usize,
    /// `S`, the tokens per sequence.
    s: usize,
    /// `H`, the width of `q`, `k` and `v`.
    h: usize,
    /// `H / heads`, the width of one head.
    d: usize,
    /// `1 / sqrt(H / heads)`, computed once per call.
    scale: T,
}

/// The maximum score over the unmasked keys of one query and head, and the sum of
/// `exp(score - max)` over them.
#[derive(Clone, Copy)]
struct Softmax<T> {
    max: T,
    total: T,
}

impl<'v, T: Element> Attend<'v, T> {
    fn new(q: View<'v, T>, k: View<'v, T>, v: View<'v, T>, mask: View<'v, T>, heads: u64) -> Self {
        let (h, heads) = (q.shape[1] as usize, heads as usize);
        let d = h / heads;
        Self {
            s: mask.shape[1] as usize,
            h,
            d,
            heads,
            scale: T::from_f32(1.0) / T::from_count(d as u64).sqrt(),
            q,
            k,
            v,
            mask,
        }
    }

    /// `N`, the rows of `q`.
    fn rows(&self) -> usize {
        self.q.shape[0] as usize
    }

    /// The columns head `head` reads.
    fn columns(&self, head: usize) -> std::ops::Range<usize> {
        head * self.d..(head + 1) * self.d
    }

    /// Whether the key mask keeps the key of row `j`.
    fn unmasked(&self, j: usize) -> bool {
        self.mask.data[j] != T::ZERO
    }

    /// The rows of the unmasked keys of the sequence of row `i`, ascending.
    fn keys(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let start = i / self.s * self.s;
        (start..start + self.s).filter(|&j| self.unmasked(j))
    }

    /// The score of query `i` and key `j` in head `head`: their dot product over the head's
    /// columns in ascending order, fused after its first product, times the scale.
    fn score(&self, i: usize, j: usize, head: usize) -> T {
        let (q, k) = (&self.q.data[i * self.h..], &self.k.data[j * self.h..]);
        sum_of_products(self.columns(head).map(|c| (q[c], k[c]))) * self.scale
    }

    /// The softmax of query `i` in head `head` over its unmasked keys: the maximum score in
    /// ascending key order, then the sum of `exp(score - max)` in ascending key order.
    fn softmax(&self, i: usize, head: usize) -> Softmax<T> {
        let mut scores = self.keys(i).map(|j| self.score(i, j, head));
        let first = scores
            .next()
            .expect("the key mask keeps a key of every sequence");
        let max = scores.fold(first, |m, x| if x > m { x } else { m });
        let total = sum(self.keys(i).map(|j| (self.score(i, j, head) - max).exp()));
        Softmax { max, total }
    }

    /// The probability of key `j` for query `i` in head `head`: `exp(score - max) / total`.
    fn probability(&self, i: usize, j: usize, head: usize, softmax: Softmax<T>) -> T {
        (self.score(i, j, head) - softmax.max).exp() / softmax.total
    }

    /// `dP[i, j]`, the gradient of the probability of key `j` for query `i` in head `head`:
    /// the dot product of row `i` of `g` and row `j` of `v` over the head's columns.
    fn probability_grad(&self, i: usize, j: usize, head: usize, g: View<'_, T>) -> T {
        let (g, v) = (&g.data[i * self.h..], &self.v.data[j * self.h..]);
        sum_of_products(self.columns(head).map(|c| (g[c], v[c])))
    }

    /// `sum_j P[i, j] dP[i, j]` over the unmasked keys of query `i` in head `head`, ascending.
    fn row_dot(&self, i: usize, head: usize, softmax: Softmax<T>, g: View<'_, T>) -> T {
        sum_of_products(self.keys(i).map(|j| {
            let p = self.probability(i, j, head, softmax);
            (p, self.probability_grad(i, j, head, g))
        }))
    }

    /// `dS[i, j] = P[i, j] * (dP[i, j] - d)`, the gradient of the score of query `i` and key
    /// `j` in head `head`, `d` its row's [`Self::row_dot`]; the scale is applied by the caller.
    fn score_grad(
        &self,
        i: usize,
        j: usize,
        head: usize,
        softmax: Softmax<T>,
        d: T,
        g: View<'_, T>,
    ) -> T {
        let p = self.probability(i, j, head, softmax);
        p * (self.probability_grad(i, j, head, g) - d)
    }
}

/// `eps` of `layer_norm`, `1e-12` as an `f32` constant (DESIGN.md § 4).
const LAYER_NORM_EPS: f32 = 1e-12;

/// The mean and `1 / sqrt(var + eps)` of one `layer_norm` row, in two passes (coordinator
/// decision D32): the mean is the row summed in ascending column order from its first element and
/// divided by its length; the biased variance is the sum of the squared deviations from that
/// mean, as [`sum_of_products`] adds them, divided by the length. The square root and the
/// division are each rounded once.
fn row_moments<T: Element>(row: &[T]) -> (T, T) {
    let width = T::from_count(row.len() as u64);
    let mean = sum(row.iter().copied()) / width;
    let variance = sum_of_products(row.iter().map(|&x| (x - mean, x - mean))) / width;
    let rstd = T::from_f32(1.0) / (variance + T::from_f32(LAYER_NORM_EPS)).sqrt();
    (mean, rstd)
}

/// The sum of the products `a * b` in iteration order: the first product rounded and the start
/// of the sum, each later one fused into it with [`Element::mul_add`], rounded once, as the
/// `mse` loss sum is (DESIGN.md § 2.2, § 12 ADR-009); `+0` for no terms.
fn sum_of_products<T: Element>(mut terms: impl Iterator<Item = (T, T)>) -> T {
    match terms.next() {
        Some((a, b)) => terms.fold(a * b, |acc, (a, b)| a.mul_add(b, acc)),
        None => T::ZERO,
    }
}

/// `sqrt(2 / pi)`, the scale of the tanh form of GELU.
const GELU_SCALE: f64 = 0.797_884_560_802_865_4;
/// `0.044715`, the cubic coefficient of the tanh form of GELU.
const GELU_CUBIC: f64 = 0.044_715;

/// `value` in `T` as the sum of two `f32` parts: in `f64` it is `value` to about `2^-48`
/// relative, in `f32` the nearest `f32` to `value` (the low part is under half an `f32` ulp of
/// the high part, so their `f32` sum rounds to the high part).
fn constant<T: Element>(value: f64) -> T {
    let high = value as f32;
    let low = (value - f64::from(high)) as f32;
    T::from_f32(high) + T::from_f32(low)
}

/// `2 u`, for `u = sqrt(2/pi) (x + 0.044715 x^3)` the argument of the tanh in the tanh form of
/// GELU.
fn gelu_twice_inner<T: Element>(x: T) -> T {
    let two = T::from_f32(2.0);
    two * constant::<T>(GELU_SCALE) * (x + constant::<T>(GELU_CUBIC) * x * x * x)
}

/// The logistic function at `v`, `1 / (1 + exp(-v))`, which is `0.5 (1 + tanh(v / 2))`.
fn logistic<T: Element>(v: T) -> T {
    let one = T::from_f32(1.0);
    one / (one + (T::ZERO - v).exp())
}

/// The tanh form of GELU, `0.5 x (1 + tanh(u))` with `u = sqrt(2/pi) (x + 0.044715 x^3)`
/// (DESIGN.md § 4), evaluated as the identical `x / (1 + exp(-2u))`: in the left tail `tanh(u)`
/// rounds to within an ulp of `-1` and `1 + tanh(u)` keeps only a few correct bits in `f32`
/// (at `x = -3` the tanh evaluation is off by `1.1e-5` relative), where `exp(-2u)` loses none.
fn gelu<T: Element>(x: T) -> T {
    x * logistic(gelu_twice_inner(x))
}

/// The derivative of [`gelu`] at `x` times `g`. With `s = 1 / (1 + exp(-2u))` and
/// `r = 1 - s = 1 / (1 + exp(2u))`, `gelu(x) = x s` and
/// `gelu'(x) = s + 2 x s r u'(x)`, `u'(x) = sqrt(2/pi) (1 + 3 * 0.044715 x^2)`; this is
/// `0.5 (1 + t) + 0.5 x (1 - t^2) u'(x)` with `t = tanh(u)`, since `1 + t = 2 s` and
/// `1 - t^2 = 4 s r`. `r` is computed from its own exponential, not as `1 - s`, so that neither
/// tail cancels.
fn gelu_grad<T: Element>(x: T, g: T) -> T {
    let one = T::from_f32(1.0);
    let two = T::from_f32(2.0);
    let three = T::from_f32(3.0);
    let v = gelu_twice_inner(x);
    let (s, r) = (logistic(v), logistic(T::ZERO - v));
    let slope = constant::<T>(GELU_SCALE) * (one + three * constant::<T>(GELU_CUBIC) * x * x);
    (s + two * (s * x) * r * slope) * g
}

/// The number of positive knots of the `sigreg` quadrature, `t_j = 0.625 j` for `j = 1..=8`.
const SIGREG_KNOTS: usize = 8;
/// The knot spacing of the `sigreg` quadrature: 17 knots on `[-5, 5]` (LeJEPA, arXiv 2511.08544,
/// § 4.2.3 listing and § 6.1).
const SIGREG_SPACING: f64 = 0.625;

/// The knot `t_j`, the standard normal characteristic function and window
/// `phi_j = exp(-t_j^2 / 2)`, and the trapezoid weight `c_j` of the `sigreg` quadrature, for
/// `j = 1..=8`, all in `f64`. The trapezoid on the 17 knots of `[-5, 5]` weighs an interior knot
/// by the spacing and an end knot by half of it; the integrand `|ecf(t) - phi(t)|^2 phi(t)` is
/// even and 0 at `t = 0`, so the positive knots carry twice their weight: `c_j = 1.25 phi_j` for
/// `j < 8` and `c_8 = 0.625 phi_8` (LeJEPA, arXiv 2511.08544, § 4.2.3 listing and § 5.1).
fn sigreg_knot(j: usize) -> (f64, f64, f64) {
    let t = SIGREG_SPACING * (j as f64);
    let phi = (-0.5 * t * t).exp();
    let share = if j == SIGREG_KNOTS { 1.0 } else { 2.0 };
    (t, phi, share * SIGREG_SPACING * phi)
}

/// The column means `(C_j, S_j)` of `cos(t_j P)` and `sin(t_j P)` of column `m` of the `[N, M]`
/// tensor `p`, for `j = 1..=8`: each argument `t_j * P[n, m]` rounded in `T`, each sum over `n`
/// ascending from row 0 in `T`, then divided by `N` in `T`. `SigregColumns` and its gradient op
/// both use it, so the gradient sees the forward's means.
fn sigreg_means<T: Element>(p: &View<'_, T>, m: usize) -> [(T, T); SIGREG_KNOTS] {
    let (rows, columns) = (p.shape[0], p.shape[1] as usize);
    let count = T::from_count(rows);
    std::array::from_fn(|k| {
        let t = constant::<T>(sigreg_knot(k + 1).0);
        let mut column = p
            .data
            .iter()
            .skip(m)
            .step_by(columns)
            .map(|&x| (t * x).sin_cos());
        let (sin, cos) = match column.next() {
            Some(first) => column.fold(first, |(sin, cos), (s, c)| (sin + s, cos + c)),
            None => (T::ZERO, T::ZERO),
        };
        (cos / count, sin / count)
    })
}

/// `SigregColumns`: for the `[N, M]` tensor `p`, the `[M]` Epps–Pulley statistics
/// `T_m = N sum_j c_j ((C_j - phi_j)^2 + S_j^2)` (DESIGN.md § 4). Per column, with
/// `d_j = C_j - phi_j`, `q_j = fma(S_j, S_j, d_j * d_j)`; the sum over `j` ascending is
/// `c_1 * q_1` and then `fma(c_j, q_j, sum)`; the result is `N * sum`.
fn sigreg_columns<T: Element>(data: &mut Vec<T>, p: View<'_, T>) {
    let count = T::from_count(p.shape[0]);
    for m in 0..p.shape[1] as usize {
        let means = sigreg_means(&p, m);
        let terms = means.iter().enumerate().map(|(k, &(cos, sin))| {
            let (_, phi, weight) = sigreg_knot(k + 1);
            let d = cos - constant::<T>(phi);
            (constant::<T>(weight), sin.mul_add(sin, d * d))
        });
        data.push(count * sum_of_products(terms));
    }
}

/// `SigregColumnsGrad`: the gradient of [`sigreg_columns`] of the `[N, M]` tensor `p` times the
/// `[M]` upstream gradient `g`,
/// `dP[n, m] = 2 g[m] sum_j c_j t_j (S_j cos(t_j P[n, m]) - (C_j - phi_j) sin(t_j P[n, m]))`
/// (DESIGN.md § 4). The means are recomputed by [`sigreg_means`]; per element, with
/// `(s_j, c_j) = sin_cos(t_j * P[n, m])`, the term is `fma(-(C_j - phi_j), s_j, S_j * c_j)`, the
/// sum over `j` ascending is `(c_j t_j) * term_1` and then `fma(c_j t_j, term_j, sum)` with
/// `c_j t_j` one constant, and the element is `(2 * g[m]) * sum`.
fn sigreg_columns_grad<T: Element>(data: &mut Vec<T>, p: View<'_, T>, g: &[T]) {
    let columns = p.shape[1] as usize;
    let knots: [(T, T, T); SIGREG_KNOTS] = std::array::from_fn(|k| {
        let (t, phi, weight) = sigreg_knot(k + 1);
        (
            constant::<T>(t),
            constant::<T>(phi),
            constant::<T>(weight * t),
        )
    });
    let means: Vec<_> = (0..columns).map(|m| sigreg_means(&p, m)).collect();
    let two = T::from_f32(2.0);
    for (i, &x) in p.data.iter().enumerate() {
        let m = i % columns;
        let terms = knots
            .iter()
            .zip(&means[m])
            .map(|(&(t, phi, slope), &(cos, sin))| {
                let (s, c) = (t * x).sin_cos();
                (slope, (T::ZERO - (cos - phi)).mul_add(s, sin * c))
            });
        data.push((two * g[m]) * sum_of_products(terms));
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

/// The name of the Input node `i`.
fn input_name(program: &TensorProgram, i: usize) -> &str {
    match &program.nodes[i].op {
        TensorOp::Input { name } => name,
        _ => unreachable!("only an Input node is i32"),
    }
}

/// Every Input node has a supplied tensor of its declared shape, an `f32` one among `inputs` and
/// an `i32` one among `ids`, and nothing else is supplied. A name supplied among the tensors of
/// the other element type is [`RuntimeError::InputDType`].
fn check_inputs<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
    ids: &BTreeMap<String, HostTensor<i32>>,
) -> Result<(), RuntimeError> {
    let (mut floats, mut integers) = (0, 0);
    for node in &program.nodes {
        if let TensorOp::Input { name } = &node.op {
            let integer = node.ty.dtype == DType::I32;
            let (shape, elsewhere) = if integer {
                (
                    ids.get(name).map(HostTensor::shape),
                    inputs.contains_key(name),
                )
            } else {
                (
                    inputs.get(name).map(HostTensor::shape),
                    ids.contains_key(name),
                )
            };
            if elsewhere {
                return Err(RuntimeError::InputDType {
                    name: name.clone(),
                    declared: node.ty.clone(),
                });
            }
            let shape = shape.ok_or_else(|| RuntimeError::MissingInput { name: name.clone() })?;
            if !has_shape(shape, &node.ty) {
                return Err(RuntimeError::InputShape {
                    name: name.clone(),
                    declared: node.ty.clone(),
                    got: shape.to_vec(),
                });
            }
            if integer {
                integers += 1;
            } else {
                floats += 1;
            }
        }
    }
    // Input names are unique (the verifier refuses a repeated one) and each was found above in
    // its own map only, so a supplied tensor is unexpected exactly when there are more of them.
    if inputs.len() == floats && ids.len() == integers {
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
    match inputs
        .keys()
        .chain(ids.keys())
        .find(|name| !names.contains(name.as_str()))
    {
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
        if !has_shape(value.shape(), &decl.ty) {
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

fn has_shape(shape: &[u64], ty: &TensorType) -> bool {
    shape.len() == ty.rank()
        && ty
            .shape
            .iter()
            .zip(shape)
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

/// `Add`, `Sub`, `Mul`, `Div`: same shapes element by element, or one rank-0 operand against
/// every element of the other, operand order kept.
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

/// A whole-tensor sum (DESIGN.md § 2.2, coordinator decision D84): the terms widened to `f64`,
/// each exactly, and added in iteration order from the first term, each partial sum rounded in
/// `f64`; `+0` for no terms. The caller rounds the result to the element type once.
fn wide_sum<T: Element>(terms: &[T]) -> f64 {
    sum(terms.iter().map(|&x| x.to_f64()))
}

/// The element count of `a` in `f64`, the divisor of a whole-tensor `mean` or `mse`.
fn wide_count<T: Element>(a: View<'_, T>) -> f64 {
    a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GELU constants are the nearest `f32` in `f32`, and within `2^-48` relative in `f64`.
    #[test]
    fn gelu_constants_round_once_in_f32_and_keep_f64_precision() {
        for value in [GELU_SCALE, GELU_CUBIC] {
            assert_eq!(constant::<f32>(value), value as f32, "{value}");
            let wide = constant::<f64>(value);
            assert!(
                ((wide - value) / value).abs() <= f64::powi(2.0, -48),
                "{value}: {wide}"
            );
        }
    }
}
