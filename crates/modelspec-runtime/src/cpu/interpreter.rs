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

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::ir::tensor::{
    Dim, ParamId, TensorNode, TensorOp, TensorProgram, TensorType, ValueId,
};
use modelspec_core::ir::verify::{VerifyError, verify};
use thiserror::Error;

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

/// Evaluates `program` and returns its outputs, in the order of [`TensorProgram::outputs`].
///
/// `inputs` holds a tensor for exactly the program's inputs, by name, and `params` one for
/// exactly the entries of its parameter table; each has its declared shape.
pub fn evaluate<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
    params: &BTreeMap<ParamId, HostTensor<T>>,
) -> Result<Vec<HostTensor<T>>, RuntimeError> {
    let values = run(program, inputs, params)?;
    outputs(program, &values)
}

/// [`evaluate`], also returning every intermediate value.
pub fn evaluate_all<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
    params: &BTreeMap<ParamId, HostTensor<T>>,
) -> Result<Evaluation<T>, RuntimeError> {
    let values = run(program, inputs, params)?;
    Ok(Evaluation {
        outputs: outputs(program, &values)?,
        values,
    })
}

/// Copies of the output values, each allocated through [`allocate`].
fn outputs<T: Element>(
    program: &TensorProgram,
    values: &BTreeMap<ValueId, HostTensor<T>>,
) -> Result<Vec<HostTensor<T>>, RuntimeError> {
    program
        .outputs
        .iter()
        .map(|&id| {
            let value = &values[&id];
            let (mut data, _) = allocate(id, value.len() as u64)?;
            data.extend_from_slice(value.data());
            Ok(HostTensor::from_parts(value.shape().to_vec(), data))
        })
        .collect()
}

/// Verifies, checks the supplied tensors, then evaluates every node in definition order.
fn run<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
    params: &BTreeMap<ParamId, HostTensor<T>>,
) -> Result<BTreeMap<ValueId, HostTensor<T>>, RuntimeError> {
    verify(program)?;
    check_inputs(program, inputs)?;
    check_params(program, params)?;
    let mut values = BTreeMap::new();
    for node in &program.nodes {
        let value = eval_node(program, node, &values, inputs, params)?;
        values.insert(node.id, value);
    }
    Ok(values)
}

/// Every Input node has a supplied tensor of its declared shape, and nothing else is supplied.
fn check_inputs<T: Element>(
    program: &TensorProgram,
    inputs: &BTreeMap<String, HostTensor<T>>,
) -> Result<(), RuntimeError> {
    let mut names = BTreeSet::new();
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
            names.insert(name.as_str());
        }
    }
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

fn eval_node<T: Element>(
    program: &TensorProgram,
    node: &TensorNode,
    values: &BTreeMap<ValueId, HostTensor<T>>,
    inputs: &BTreeMap<String, HostTensor<T>>,
    params: &BTreeMap<ParamId, HostTensor<T>>,
) -> Result<HostTensor<T>, RuntimeError> {
    // The verifier guarantees every operand is defined by an earlier node, and that the element
    // count of every type fits `u64`.
    let arg = |i: usize| &values[&node.operands[i]];
    let shape = static_shape(&node.ty);
    let (mut data, n) = allocate::<T>(node.id, shape.iter().product())?;
    match &node.op {
        TensorOp::Input { name } => data.extend_from_slice(input(inputs, name)?.data()),
        TensorOp::ParamRead { param: id } => {
            data.extend_from_slice(param(program, params, *id)?.data())
        }
        TensorOp::Scalar(v) => data.push(T::from_f32(*v)),
        TensorOp::Add => elementwise(&mut data, arg(0), arg(1), |a, b| a + b),
        TensorOp::Sub => elementwise(&mut data, arg(0), arg(1), |a, b| a - b),
        TensorOp::Mul => elementwise(&mut data, arg(0), arg(1), |a, b| a * b),
        TensorOp::MatMul => matmul(&mut data, arg(0), arg(1)),
        TensorOp::Relu => data.extend(arg(0).data().iter().map(|&x| relu_where(x, x))),
        TensorOp::Transpose => transpose(&mut data, arg(0)),
        TensorOp::ReduceSum => data.push(sum(arg(0).data().iter().copied())),
        TensorOp::ReduceMean => {
            let a = arg(0);
            data.push(sum(a.data().iter().copied()) / count(a));
        }
        TensorOp::Mse => {
            let (a, b) = (arg(0), arg(1));
            let squares = a.data().iter().zip(b.data()).map(|(&x, &y)| {
                let d = x - y;
                d * d
            });
            data.push(sum(squares) / count(a));
        }
        TensorOp::ReluGrad => {
            let (a, g) = (arg(0), arg(1));
            data.extend(
                a.data()
                    .iter()
                    .zip(g.data())
                    .map(|(&a, &g)| relu_where(a, g)),
            );
        }
        TensorOp::BroadcastScalar { .. } => data.resize(n, arg(0).data()[0]),
    }
    Ok(HostTensor::from_parts(shape, data))
}

/// Room for exactly `count` elements of the value `value`: the one path every value the
/// interpreter makes is allocated through. A count whose bytes exceed what the host can address
/// or provide is a [`RuntimeError::Allocation`], not a panic or an abort.
fn allocate<T>(value: ValueId, count: u64) -> Result<(Vec<T>, usize), RuntimeError> {
    let refused = || RuntimeError::Allocation {
        value,
        bytes: u128::from(count) * size_of::<T>() as u128,
    };
    let n = usize::try_from(count).map_err(|_| refused())?;
    let mut data = Vec::new();
    data.try_reserve_exact(n).map_err(|_| refused())?;
    Ok((data, n))
}

/// `where(a > 0, g, 0)`: `relu(x)` is `relu_where(x, x)`; the derivative at 0 is 0 (DESIGN.md § 4).
fn relu_where<T: Element>(a: T, g: T) -> T {
    if a > T::ZERO { g } else { T::ZERO }
}

/// `Add`, `Sub`, `Mul`: same shapes element by element, or one rank-0 operand against every
/// element of the other, operand order kept.
fn elementwise<T: Element>(
    data: &mut Vec<T>,
    a: &HostTensor<T>,
    b: &HostTensor<T>,
    f: impl Fn(T, T) -> T,
) {
    if a.shape() == b.shape() {
        data.extend(a.data().iter().zip(b.data()).map(|(&x, &y)| f(x, y)));
    } else if b.rank() == 0 {
        let y = b.data()[0];
        data.extend(a.data().iter().map(|&x| f(x, y)));
    } else {
        let x = a.data()[0];
        data.extend(b.data().iter().map(|&y| f(x, y)));
    }
}

/// `[M, K] x [K, N] -> [M, N]`; each element sums its `K` products in ascending `k`.
///
/// Loop order `i`, `k`, `j`, so the inner loop reads a row of `b`: each output row starts as its
/// `k = 0` products (not `+0`, so an all-`-0` sum stays `-0`) and every later `k` adds its
/// products in turn, the same sequence of roundings per element as summing over `k` directly.
fn matmul<T: Element>(data: &mut Vec<T>, a: &HostTensor<T>, b: &HostTensor<T>) {
    let m = a.shape()[0] as usize;
    let k = a.shape()[1] as usize;
    let n = b.shape()[1] as usize;
    let (a, b) = (a.data(), b.data());
    if k == 0 || n == 0 {
        data.extend(std::iter::repeat_n(T::ZERO, m * n));
        return;
    }
    for a_row in a.chunks_exact(k) {
        let start = data.len();
        let (&first, rest) = a_row.split_first().expect("k > 0");
        data.extend(b[..n].iter().map(|&y| first * y));
        let row = &mut data[start..];
        for (&x, b_row) in rest.iter().zip(b[n..].chunks_exact(n)) {
            for (acc, &y) in row.iter_mut().zip(b_row) {
                *acc = *acc + x * y;
            }
        }
    }
}

/// `[M, N] -> [N, M]`.
fn transpose<T: Element>(data: &mut Vec<T>, a: &HostTensor<T>) {
    let m = a.shape()[0] as usize;
    let n = a.shape()[1] as usize;
    let a = a.data();
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

fn count<T: Element>(a: &HostTensor<T>) -> T {
    T::from_count(a.len() as u64)
}
