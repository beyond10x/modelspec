//! A native host supplies each minibatch and its learning rate. Training state remains under
//! one backend's ownership until the caller requests a checkpoint.

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::ir::tensor::{ParamId, TensorOp, TensorProgram};
use modelspec_core::ir::training::{
    AdamW, C1_INPUT, C2_INPUT, LR_T_INPUT, Moment, OptimizerRule, TrainingStep, moment_input,
};
use modelspec_core::ir::verify::verify_step;

use super::optimizer::{MomentValues, ParamState, validate_moments};
use crate::cpu::HostTensor;
use crate::cpu::interpreter::{Prepared, check_inputs, check_params};
use crate::execution::{Backend, ExecutionError};

type Parameters = BTreeMap<ParamId, HostTensor<f32>>;
pub(crate) type Floats = BTreeMap<String, HostTensor<f32>>;
pub(crate) type Ids = BTreeMap<String, HostTensor<i32>>;

/// State after exactly `step` committed updates. Every optimizer slot has its parameter's
/// shape. A fresh session may omit moments; a resumed session supplies all of them.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingSnapshot {
    pub step: u64,
    pub parameters: BTreeMap<ParamId, HostTensor<f32>>,
    pub moments: MomentValues,
}

/// Cumulative Vulkan queue and staging counters for a single training session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingDeviceStats {
    pub recordings: u64,
    pub submits: u64,
    pub host_waits: u64,
    pub dispatches: u64,
    pub bytes_uploaded: u64,
    pub bytes_downloaded: u64,
}

impl TrainingSnapshot {
    pub fn fresh(parameters: BTreeMap<ParamId, HostTensor<f32>>) -> Self {
        Self {
            step: 0,
            parameters,
            moments: BTreeMap::new(),
        }
    }
}

pub(crate) fn invalid(reason: impl ToString) -> ExecutionError {
    ExecutionError::Training(reason.to_string())
}

fn check_values(state: &TrainingSnapshot) -> Result<(), ExecutionError> {
    if state
        .parameters
        .values()
        .any(|v| v.data().iter().any(|x| !x.is_finite()))
        || state.moments.iter().any(|((_, moment), v)| {
            v.data()
                .iter()
                .any(|x| !x.is_finite() || (*moment == Moment::Second && *x < 0.))
        })
    {
        return Err(invalid("invalid parameter or optimizer moment"));
    }
    Ok(())
}

enum Target {
    Cpu(Box<CpuSession>),
    #[cfg(feature = "vulkan")]
    Vulkan(crate::vulkan::session::DeviceSession),
}

/// A compiled AdamW update with persistent weights and moments. One update returns the loss
/// evaluated before its write. Named inputs are bound by exact dtype and shape.
pub struct TrainingSession {
    batch: TensorProgram,
    rule: AdamW,
    step: u64,
    target: Target,
}

impl TrainingSession {
    /// Read host-maintained device counters without submitting work. CPU sessions return
    /// `None`. Checkpoint transfers count only when `snapshot` is requested.
    pub fn device_stats(&self) -> Result<Option<TrainingDeviceStats>, ExecutionError> {
        match &self.target {
            Target::Cpu(_) => Ok(None),
            #[cfg(feature = "vulkan")]
            Target::Vulkan(gpu) => gpu.stats().map(Some),
        }
    }

    /// Open a session from a verified training step. The default dropout run key is zero.
    pub fn new(
        step: TrainingStep,
        backend: Backend,
        device: Option<usize>,
        initial: TrainingSnapshot,
    ) -> Result<Self, ExecutionError> {
        Self::new_seeded(step, backend, device, initial, 0)
    }

    /// Open or restore a session with an explicit dropout run key. A resumed caller uses the
    /// original key and supplies the same batch and dropout token at each absolute step.
    pub fn new_seeded(
        step: TrainingStep,
        backend: Backend,
        device: Option<usize>,
        mut initial: TrainingSnapshot,
        seed: u64,
    ) -> Result<Self, ExecutionError> {
        verify_step(&step).map_err(invalid)?;
        let OptimizerRule::AdamW(rule) = step.optimizer.rule else {
            return Err(invalid("expected a compiled AdamW step"));
        };
        if backend == Backend::Cpu && device.is_some() {
            return Err(ExecutionError::CpuDevice);
        }
        let p = &step.training.program;
        check_params(p, &initial.parameters)?;
        if initial.step == 0 && initial.moments.is_empty() {
            let fresh = ParamState::new(&step, initial.parameters);
            initial.parameters = fresh.values;
            initial.moments = fresh.moments;
        }
        validate_moments(&step, &initial.parameters, &initial.moments).map_err(invalid)?;
        check_values(&initial)?;
        let private: BTreeSet<_> = step.optimizer.host_inputs().into_iter().collect();
        let batch = TensorProgram {
            nodes: p
                .nodes
                .iter()
                .filter(|n| matches!(n.op, TensorOp::Input { .. }) && !private.contains(&n.id))
                .cloned()
                .collect(),
            ..TensorProgram::default()
        };
        let cursor = initial.step;
        let target = match backend {
            Backend::Cpu => Target::Cpu(Box::new(CpuSession::new(step, initial, seed))),
            #[cfg(feature = "vulkan")]
            Backend::Vulkan => Target::Vulkan(crate::vulkan::session::DeviceSession::new(
                device.unwrap_or(0),
                step,
                initial,
                seed,
            )?),
            #[cfg(not(feature = "vulkan"))]
            Backend::Vulkan => return Err(ExecutionError::VulkanDisabled),
        };
        Ok(Self {
            batch,
            rule,
            step: cursor,
            target,
        })
    }

    /// Execute one update at the supplied rate. The optimizer uses the next absolute
    /// step's AdamW bias corrections. The trainer does not schedule or decay the rate.
    pub fn update(
        &mut self,
        floats: &BTreeMap<String, HostTensor<f32>>,
        ids: &BTreeMap<String, HostTensor<i32>>,
        lr: f32,
    ) -> Result<f32, ExecutionError> {
        if !lr.is_finite() || lr <= 0. {
            return Err(invalid("learning rate must be positive and finite"));
        }
        let next = self
            .step
            .checked_add(1)
            .ok_or_else(|| invalid("optimizer step overflow"))?;
        check_inputs(&self.batch, floats, ids)?;
        let scalars = self.rule.scalars(lr, next, next);
        let loss = match &mut self.target {
            Target::Cpu(cpu) => cpu.update(floats, ids, [lr, scalars.c1, scalars.c2])?,
            #[cfg(feature = "vulkan")]
            Target::Vulkan(gpu) => gpu.update(floats, ids, [lr, scalars.c1, scalars.c2])?,
        };
        self.step = next;
        Ok(loss)
    }

    /// Copy the current parameters and both moments to host memory. The Vulkan update path
    /// does not download these tensors between snapshots.
    pub fn snapshot(&mut self) -> Result<TrainingSnapshot, ExecutionError> {
        let state = match &mut self.target {
            Target::Cpu(cpu) => TrainingSnapshot {
                step: self.step,
                parameters: cpu.values.clone(),
                moments: cpu.moments.clone(),
            },
            #[cfg(feature = "vulkan")]
            Target::Vulkan(gpu) => gpu.snapshot(self.step)?,
        };
        check_values(&state)?;
        Ok(state)
    }
}

struct CpuSession {
    plan: Prepared<'static, f32>,
    values: Parameters,
    moments: MomentValues,
    writes: Vec<ParamId>,
    slot_names: Vec<((ParamId, Moment), String)>,
}

impl CpuSession {
    fn new(step: TrainingStep, initial: TrainingSnapshot, seed: u64) -> Self {
        let mut p = step.training.program;
        let slots = step.optimizer.state_slots();
        p.outputs = std::iter::once(step.training.loss)
            .chain(step.optimizer.updates.values().copied())
            .chain(slots.iter().map(|s| s.next))
            .collect();
        let slot_names = slots
            .iter()
            .map(|s| {
                (
                    (s.param, s.moment),
                    moment_input(s.moment, &p.params[&s.param].path),
                )
            })
            .collect();
        let mut plan = Prepared::owned_verified(p);
        plan.set_seed(seed);
        Self {
            plan,
            values: initial.parameters,
            moments: initial.moments,
            writes: step.optimizer.updates.into_keys().collect(),
            slot_names,
        }
    }

    fn update(
        &mut self,
        floats: &Floats,
        ids: &Ids,
        scalars: [f32; 3],
    ) -> Result<f32, ExecutionError> {
        let mut bound = floats.clone();
        if !self.writes.is_empty() {
            for (k, v) in [LR_T_INPUT, C1_INPUT, C2_INPUT].into_iter().zip(scalars) {
                bound.insert(k.to_owned(), HostTensor::scalar(v));
            }
        }
        for (key, name) in &self.slot_names {
            bound.insert(name.clone(), self.moments[key].clone());
        }
        let mut outputs = self
            .plan
            .run_with_ids(&bound, ids, &self.values)?
            .into_iter();
        let loss = outputs.next().expect("verified scalar loss").data()[0];
        if !loss.is_finite()
            || outputs
                .as_slice()
                .iter()
                .any(|t| t.data().iter().any(|x| !x.is_finite()))
        {
            return Err(invalid("non-finite training update"));
        }
        for id in &self.writes {
            self.values
                .insert(*id, outputs.next().expect("verified parameter output"));
        }
        for (key, _) in &self.slot_names {
            self.moments
                .insert(*key, outputs.next().expect("verified moment output"));
        }
        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelspec_core::compile::compile_training;

    const MODEL: &str = r#"
model Tiny {
    param table: Tensor<f32, [3, 2]> = zeros();
    param bias: Tensor<f32, [2]> = zeros();
    fn forward(ids: Tensor<i32, [2]>) -> Tensor<f32, [2, 2]> {
        return add_rows(embedding(table, ids), bias);
    }
}
train Fit {
    model = Tiny;
    input ids: Tensor<i32, [2]>;
    target labels: Tensor<i32, [2]>;
    target weights: Tensor<f32, [2]>;
    loss = cross_entropy(model.forward(ids), labels, weights);
    optimizer = adamw(lr = 0.001, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, warmup_steps = 0);
    steps = 3;
    seed = 11;
    device = cpu;
}"#;

    fn setup() -> (TrainingStep, TrainingSnapshot) {
        let step = compile_training("tiny.mspec", MODEL).unwrap().training;
        let params = step
            .training
            .program
            .params
            .iter()
            .map(|(&id, p)| {
                let shape = crate::cpu::interpreter::static_shape(&p.ty);
                let size = shape.iter().product::<u64>() as usize;
                (
                    id,
                    HostTensor::new(shape, (0..size).map(|n| n as f32 / 9. - 0.2).collect())
                        .unwrap(),
                )
            })
            .collect();
        (step, TrainingSnapshot::fresh(params))
    }

    fn batch(a: i32, b: i32) -> (Floats, Ids) {
        let f = BTreeMap::from([(
            "weights".into(),
            HostTensor::new(vec![2], vec![1., 0.5]).unwrap(),
        )]);
        let i = BTreeMap::from([
            ("ids".into(), HostTensor::new(vec![2], vec![a, b]).unwrap()),
            (
                "labels".into(),
                HostTensor::new(vec![2], vec![0, 1]).unwrap(),
            ),
        ]);
        (f, i)
    }

    fn close(a: &TrainingSnapshot, b: &TrainingSnapshot) {
        assert_eq!(a.step, b.step);
        assert_eq!(
            a.parameters.keys().collect::<Vec<_>>(),
            b.parameters.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            a.moments.keys().collect::<Vec<_>>(),
            b.moments.keys().collect::<Vec<_>>()
        );
        for (x, y) in a
            .parameters
            .values()
            .chain(a.moments.values())
            .zip(b.parameters.values().chain(b.moments.values()))
        {
            assert_eq!(x.shape(), y.shape());
            for (&x, &y) in x.data().iter().zip(y.data()) {
                assert!((x - y).abs() < 2e-6, "{x} and {y}");
            }
        }
    }

    fn proof(backend: Backend) {
        let (step, initial) = setup();
        let mut cpu =
            TrainingSession::new(step.clone(), Backend::Cpu, None, initial.clone()).unwrap();
        let mut subject =
            TrainingSession::new(step.clone(), backend, None, initial.clone()).unwrap();
        let (f, i) = batch(0, 1);
        let before = subject.device_stats().unwrap();
        let expected = cpu.update(&f, &i, 0.02).unwrap();
        assert!((subject.update(&f, &i, 0.02).unwrap() - expected).abs() < 2e-6);
        if let (Some(a), Some(b)) = (before, subject.device_stats().unwrap()) {
            assert_eq!(b.recordings, a.recordings);
            assert_eq!(b.submits - a.submits, 1);
            assert_eq!(b.host_waits - a.host_waits, 1);
            assert_eq!(b.bytes_downloaded - a.bytes_downloaded, 4);
            assert_eq!(b.bytes_uploaded - a.bytes_uploaded, 36);
        }
        let checkpoint = subject.snapshot().unwrap();
        assert_eq!(checkpoint.step, 1);
        assert_ne!(checkpoint.parameters, initial.parameters);
        for tensor in checkpoint.moments.values() {
            assert!(tensor.data().iter().any(|x| *x != 0.));
        }
        close(&checkpoint, &cpu.snapshot().unwrap());
        let (f, bad) = batch(-1, 2);
        assert!(subject.update(&f, &bad, 0.005).is_err());
        assert!(subject.update(&f, &i, f32::NAN).is_err());
        assert!(subject.update(&f, &i, 0.).is_err());
        assert!(subject.update(&f, &Ids::new(), 0.005).is_err());
        assert_eq!(checkpoint, subject.snapshot().unwrap());
        let bad = TrainingSnapshot {
            step: 1,
            moments: BTreeMap::new(),
            ..initial
        };
        assert!(TrainingSession::new(step.clone(), backend, None, bad).is_err());
        let mut restored = TrainingSession::new(step.clone(), backend, None, checkpoint).unwrap();
        let mut migrated: Option<TrainingSession> = None;
        for (a, b, lr) in [(1, 2, 0.005), (2, 0, 0.03)] {
            let (f, i) = batch(a, b);
            let expected = cpu.update(&f, &i, lr).unwrap();
            let actual = subject.update(&f, &i, lr).unwrap();
            assert!((actual - expected).abs() < 2e-6);
            assert_eq!(actual, restored.update(&f, &i, lr).unwrap());
            if let Some(migrated) = &mut migrated {
                assert!((migrated.update(&f, &i, lr).unwrap() - expected).abs() < 2e-6);
                close(&migrated.snapshot().unwrap(), &cpu.snapshot().unwrap());
            } else {
                let second = restored.snapshot().unwrap();
                assert_eq!(second.step, 2);
                migrated =
                    Some(TrainingSession::new(step.clone(), Backend::Cpu, None, second).unwrap());
            }
        }
        let final_state = subject.snapshot().unwrap();
        assert_eq!(final_state, restored.snapshot().unwrap());
        close(&final_state, &cpu.snapshot().unwrap());
        assert_eq!(final_state.step, 3);
    }

    #[test]
    fn application_batches_resume_on_cpu() {
        proof(Backend::Cpu);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan compute device"]
    fn resident_vulkan_updates_and_resume_match_cpu() {
        proof(Backend::Vulkan);
    }
}
