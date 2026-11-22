//! Domain checks for arbitrary, caller-bound tensor programs. A negative or oversized host ID
//! must be rejected before any shader can use it as an address. The IR verifier proves bounds
//! for derived integer indices. Float masks and weights can themselves be computed by the
//! program: they never address memory, so retain their device values and check them on download.

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::{DType, ParamId, TensorOp, TensorProgram, ValueId};

use crate::cpu::interpreter::{check_key_mask, check_key_spans, static_shape};
use crate::cpu::{HostTensor, RuntimeError};
use crate::execution::Output;

enum Domain {
    TableRows(u64),
    Classes(u64),
    Spans(u64),
}

enum FloatDomain {
    KeyMask,
    LossWeights,
}

impl FloatDomain {
    fn check(&self, value: ValueId, data: &HostTensor<f32>) -> Result<(), RuntimeError> {
        match self {
            Self::KeyMask => check_key_mask(value, data.shape(), data.data()),
            Self::LossWeights => {
                if let Some(row) = data.data().iter().position(|&w| w < 0.0 || !w.is_finite()) {
                    return Err(RuntimeError::LossWeight { value, row });
                }
                Ok(())
            }
        }
    }
}

enum Source {
    Input(String),
    Param(ParamId),
    Output(usize),
}

#[derive(Default)]
pub(super) struct Checks {
    integers: Vec<(ValueId, String, Domain)>,
    floats: Vec<(ValueId, Source, FloatDomain)>,
    dropouts: Vec<(ValueId, u64)>,
}

impl Checks {
    /// Append each computed control tensor once. The caller truncates these private outputs.
    pub(super) fn prepare(program: &mut TensorProgram) -> Self {
        let mut checks = Self::default();
        let mut outputs = program.outputs.clone();
        for node in &program.nodes {
            let operand = |i: usize| program.node(node.operands[i]).expect("verified operand");
            let mut integer = |slot, domain| {
                if let TensorOp::Input { name } = &operand(slot).op {
                    checks.integers.push((node.id, name.clone(), domain));
                }
            };
            let mut float = |slot, domain| {
                let control = operand(slot);
                let source = match &control.op {
                    TensorOp::Input { name } => Source::Input(name.clone()),
                    TensorOp::ParamRead { param } => Source::Param(*param),
                    _ => {
                        let i = outputs
                            .iter()
                            .position(|id| *id == control.id)
                            .unwrap_or_else(|| {
                                outputs.push(control.id);
                                outputs.len() - 1
                            });
                        Source::Output(i)
                    }
                };
                checks.floats.push((node.id, source, domain));
            };
            match &node.op {
                TensorOp::Embedding => {
                    integer(1, Domain::TableRows(static_shape(&operand(0).ty)[0]))
                }
                TensorOp::EmbeddingGrad { .. } => {
                    integer(0, Domain::TableRows(static_shape(&node.ty)[0]))
                }
                TensorOp::CrossEntropy | TensorOp::CrossEntropyGrad => {
                    integer(1, Domain::Classes(static_shape(&operand(0).ty)[1]));
                    float(2, FloatDomain::LossWeights);
                }
                op if op.attention_heads().is_some() => {
                    if operand(3).ty.dtype == DType::I32 {
                        integer(3, Domain::Spans(static_shape(&operand(1).ty)[0]));
                    } else {
                        float(3, FloatDomain::KeyMask);
                    }
                }
                TensorOp::Dropout { .. } | TensorOp::DropoutGrad { .. } => {
                    checks
                        .dropouts
                        .push((node.id, static_shape(&node.ty).iter().product()));
                }
                _ => {}
            }
        }
        program.outputs = outputs;
        checks
    }

    pub(super) fn before(
        &self,
        inputs: &BTreeMap<String, HostTensor<f32>>,
        ids: &BTreeMap<String, HostTensor<i32>>,
        params: &BTreeMap<ParamId, HostTensor<f32>>,
        seed: Option<u64>,
    ) -> Result<(), RuntimeError> {
        for &(value, count) in &self.dropouts {
            if seed.is_none() {
                return Err(RuntimeError::NoSeed { value });
            }
            if count > 1 << 32 {
                return Err(RuntimeError::DropoutElements { value, count });
            }
        }
        for (value, name, domain) in &self.integers {
            let data = ids[name].data();
            let limit = match domain {
                Domain::Spans(rows) => {
                    check_key_spans(*value, data, *rows)?;
                    continue;
                }
                Domain::TableRows(rows) | Domain::Classes(rows) => *rows,
            };
            if let Some((index, &id)) = data
                .iter()
                .enumerate()
                .find(|&(_, &id)| u64::try_from(id).map_or(true, |n| n >= limit))
            {
                return Err(match domain {
                    Domain::Classes(_) => RuntimeError::LabelOutOfRange {
                        name: name.clone(),
                        value: *value,
                        row: index,
                        label: id,
                        classes: limit,
                    },
                    Domain::TableRows(_) => RuntimeError::IdOutOfRange {
                        name: name.clone(),
                        index,
                        id,
                        rows: limit,
                    },
                    Domain::Spans(_) => unreachable!(),
                });
            }
        }
        for (value, source, domain) in &self.floats {
            let data = match source {
                Source::Input(name) => &inputs[name],
                Source::Param(id) => &params[id],
                Source::Output(_) => continue,
            };
            domain.check(*value, data)?;
        }
        Ok(())
    }

    pub(super) fn after(&self, outputs: &[Output]) -> Result<(), RuntimeError> {
        for (value, source, domain) in &self.floats {
            if let Source::Output(i) = source {
                let Output::F32(data) = &outputs[*i] else {
                    unreachable!("verified float control")
                };
                domain.check(*value, data)?;
            }
        }
        Ok(())
    }
}
