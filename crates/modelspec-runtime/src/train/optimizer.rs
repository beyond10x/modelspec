//! SGD: its configuration and one training step on the CPU interpreter (DESIGN.md § 3.1, § 6).
//!
//! A step runs `&step.training.program` on the interpreter unchanged, reading the current
//! version of every parameter, then applies each update output as its parameter's next version,
//! in effect-token order. All writes come after the evaluation, so none overwrites a value a read
//! of the old version still needs (DESIGN.md § 4). A parameter without an update keeps its value
//! and its version.

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::ParamId;
use modelspec_core::ir::training::{Access, StateVersion, TrainConfig, TrainingStep};
use modelspec_core::ir::verify::{TrainingVerifyError, verify_step};
use thiserror::Error;

use crate::cpu::{HostTensor, RuntimeError, evaluate};

/// The SGD configuration of a run, from its `train` block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SgdConfig {
    pub lr: f32,
    pub steps: u64,
    pub seed: u64,
}

impl SgdConfig {
    pub fn from_config(config: &TrainConfig) -> Self {
        Self {
            lr: config.lr,
            steps: config.steps,
            seed: config.seed,
        }
    }
}

/// The persistent parameter state of a run: each declared parameter's value and version.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamState {
    pub values: BTreeMap<ParamId, HostTensor<f32>>,
    pub versions: BTreeMap<ParamId, StateVersion>,
}

impl ParamState {
    /// The state before the first step: `values` at the versions `step` is compiled against.
    pub fn new(step: &TrainingStep, values: BTreeMap<ParamId, HostTensor<f32>>) -> Self {
        Self {
            values,
            versions: step.optimizer.versions.clone(),
        }
    }
}

/// Why a step failed.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum StepError {
    #[error("training step does not verify: {0}")]
    Verify(#[from] TrainingVerifyError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    /// The parameter is at the last version; no update can follow.
    #[error("parameter {param} has no version after {current}")]
    VersionExhausted {
        param: ParamId,
        current: StateVersion,
    },
}

/// Runs one step of `step` on `inputs` and `state`, applies its updates and returns the loss of
/// the old parameter values.
pub fn sgd_step(
    step: &TrainingStep,
    inputs: &BTreeMap<String, HostTensor<f32>>,
    state: &mut ParamState,
) -> Result<f32, StepError> {
    verify_step(step)?;
    let program = &step.training.program;
    let writes: Vec<(ParamId, usize)> = step
        .optimizer
        .effects_in_order()
        .into_iter()
        .filter(|(_, effect)| effect.access == Access::Write)
        .map(|(value, effect)| {
            let position = program
                .outputs
                .iter()
                .position(|&o| o == value)
                .expect("a verified update is an output");
            (effect.param, position)
        })
        .collect();
    for &(param, _) in &writes {
        let current = state
            .versions
            .get(&param)
            .copied()
            .unwrap_or(StateVersion(0));
        if current.0.checked_add(1).is_none() {
            return Err(StepError::VersionExhausted { param, current });
        }
    }

    let outputs = evaluate(program, inputs, &state.values)?;
    let loss = outputs[0].data()[0];
    for (param, position) in writes {
        state.values.insert(param, outputs[position].clone());
        let version = state.versions.entry(param).or_insert(StateVersion(0));
        version.0 += 1;
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::evaluate;
    use modelspec_core::autodiff::differentiate;
    use modelspec_core::diag::SourceSpan;
    use modelspec_core::ir::tensor::{Initializer, TensorProgramBuilder, TensorType};
    use modelspec_core::ir::training::{Effect, EffectToken, attach_sgd};

    const LR: f32 = 0.1;
    const W: ParamId = ParamId(0);
    const U: ParamId = ParamId(1);

    fn tensor(shape: &[u64], data: &[f32]) -> HostTensor<f32> {
        HostTensor::new(shape.to_vec(), data.to_vec()).unwrap()
    }

    /// `mse(matmul(x, w), y)`; with `unused`, a parameter `LinearReg.u` nothing reads.
    fn linear_step(unused: bool) -> TrainingStep {
        let mut b = TensorProgramBuilder::new();
        let w = b
            .declare_param(
                "LinearReg.w",
                TensorType::f32(&[2, 1]),
                Initializer::XavierUniform,
            )
            .unwrap();
        if unused {
            b.declare_param(
                "LinearReg.u",
                TensorType::f32(&[2, 1]),
                Initializer::XavierUniform,
            )
            .unwrap();
        }
        let x = b.input("x", TensorType::f32(&[3, 2])).unwrap();
        let y = b.input("y", TensorType::f32(&[3, 1])).unwrap();
        let w = b.param_read(w).unwrap();
        let h = b.matmul(x, w).unwrap();
        let loss = b.mse(h, y).unwrap();
        let training = differentiate(&b.finish(), loss).unwrap();
        attach_sgd(training, LR).unwrap()
    }

    fn inputs() -> BTreeMap<String, HostTensor<f32>> {
        BTreeMap::from([
            (
                "x".into(),
                tensor(&[3, 2], &[1.0, 2.0, -1.0, 0.5, 0.25, -2.0]),
            ),
            ("y".into(), tensor(&[3, 1], &[1.0, -1.0, 0.5])),
        ])
    }

    fn values(unused: bool) -> BTreeMap<ParamId, HostTensor<f32>> {
        let mut v = BTreeMap::from([(W, tensor(&[2, 1], &[0.3, -0.7]))]);
        if unused {
            v.insert(U, tensor(&[2, 1], &[0.9, 0.1]));
        }
        v
    }

    #[test]
    fn sgd_config_reads_the_train_block() {
        let config = TrainConfig {
            steps: 500,
            lr: 0.1,
            seed: 42,
            inputs: vec![],
            targets: vec![],
            span: SourceSpan {
                file: "f.mspec".into(),
                line: 1,
                column: 1,
            },
        };
        assert_eq!(
            SgdConfig::from_config(&config),
            SgdConfig {
                lr: 0.1,
                steps: 500,
                seed: 42
            }
        );
    }

    /// One step writes `w - lr * grad`, computed in `f32` from the gradient of the old value,
    /// and moves `w` from version 0 to 1.
    #[test]
    fn a_step_writes_p_minus_lr_times_grad_as_the_next_version() {
        let step = linear_step(false);
        let mut state = ParamState::new(&step, values(false));
        assert_eq!(state.versions[&W], StateVersion(0));
        let outputs = evaluate(&step.training.program, &inputs(), &values(false)).unwrap();
        let loss = sgd_step(&step, &inputs(), &mut state).unwrap();
        assert_eq!(loss, outputs[0].data()[0]);
        let g = outputs[1].data();
        let old = values(false)[&W].data().to_vec();
        let expected: Vec<f32> = old.iter().zip(g).map(|(&p, &g)| p - LR * g).collect();
        assert_eq!(state.values[&W].data(), &expected[..]);
        assert_ne!(state.values[&W].data(), &old[..]);
        assert_eq!(state.versions[&W], StateVersion(1));
    }

    #[test]
    fn each_step_advances_the_version_by_one_and_lowers_the_loss() {
        let step = linear_step(false);
        let mut state = ParamState::new(&step, values(false));
        let mut losses = Vec::new();
        for _ in 0..3 {
            losses.push(sgd_step(&step, &inputs(), &mut state).unwrap());
        }
        assert_eq!(state.versions[&W], StateVersion(3));
        assert!(losses[0] > losses[1] && losses[1] > losses[2], "{losses:?}");
    }

    /// A parameter the loss does not read keeps its value and its version.
    #[test]
    fn a_step_leaves_an_unused_parameter_at_its_value_and_version() {
        let step = linear_step(true);
        let mut state = ParamState::new(&step, values(true));
        sgd_step(&step, &inputs(), &mut state).unwrap();
        sgd_step(&step, &inputs(), &mut state).unwrap();
        assert_eq!(state.values[&U], values(true)[&U]);
        assert_eq!(state.versions[&U], StateVersion(0));
        assert_eq!(state.versions[&W], StateVersion(2));
    }

    /// A step the verifier refuses is not run: the state is untouched.
    #[test]
    fn a_step_with_illegal_effects_is_refused_before_it_runs() {
        let mut step = linear_step(false);
        let update = step.optimizer.updates[&W];
        let effect = step.optimizer.effects[&update];
        step.optimizer.effects.insert(
            update,
            Effect {
                token: EffectToken(0),
                ..effect
            },
        );
        let mut state = ParamState::new(&step, values(false));
        let before = state.clone();
        let result = sgd_step(&step, &inputs(), &mut state);
        assert!(matches!(result, Err(StepError::Verify(_))), "{result:?}");
        assert_eq!(state, before);
    }

    /// At the last version a step is refused, not wrapped to version 0, and nothing is written.
    #[test]
    fn a_step_at_the_last_version_is_refused() {
        let step = linear_step(false);
        let mut state = ParamState::new(&step, values(false));
        state.versions.insert(W, StateVersion(u64::MAX));
        let before = state.clone();
        assert_eq!(
            sgd_step(&step, &inputs(), &mut state),
            Err(StepError::VersionExhausted {
                param: W,
                current: StateVersion(u64::MAX)
            })
        );
        assert_eq!(state, before);
    }
}
