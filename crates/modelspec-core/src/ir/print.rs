//! Readable IR printer with stable IDs (DESIGN.md § 8).
//!
//! One line per parameter, `param @<path>: <type> = <initializer>`, in `ParamId` order; one line
//! per node, `%2: Tensor<f32, [8, 1]> = MatMul %0, %1`, in definition order, with parameters named
//! by canonical path; then one `output %n` line per output. Training programs and steps print
//! the same lines with their subgraph tags, output roles, effects and state versions
//! ([`print_training`], [`print_step`]).

use std::fmt::{self, Write};

use crate::ir::kernel::{ExecutionPlan, PlanStep};
use crate::ir::tensor::{ParamId, TensorNode, TensorOp, TensorProgram};
use crate::ir::training::{Optimizer, TrainingProgram, TrainingStep};

/// The text form of `program`, one line each, newline-terminated.
pub fn print(program: &TensorProgram) -> String {
    program.to_string()
}

/// The text form of a training program: as [`print`], each node line led by its subgraph tag
/// (`gradient %7: ...`), and each output line followed by its role (`output %4 loss`,
/// `output %7 gradient @LinearReg.w`).
pub fn print_training(training: &TrainingProgram) -> String {
    training.to_string()
}

/// The text form of a training step: as [`print_training`], with each parameter line followed by
/// its state version (`param @LinearReg.w: ... = xavier_uniform {v0}`), one line
/// `optimizer sgd lr = <lr>` after the parameters, each node that carries an effect followed by
/// it (`{!0 read @LinearReg.w v0}`, `{!1 write @LinearReg.w v1}`), and each update output line
/// `output %10 update @LinearReg.w`.
pub fn print_step(step: &TrainingStep) -> String {
    step.to_string()
}

impl fmt::Display for TrainingProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_training(f, self, None)
    }
}

impl fmt::Display for TrainingStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_training(f, &self.training, Some(&self.optimizer))
    }
}

fn write_training(
    f: &mut fmt::Formatter<'_>,
    training: &TrainingProgram,
    optimizer: Option<&Optimizer>,
) -> fmt::Result {
    let program = &training.program;
    let path = |param: &ParamId| match program.params.get(param) {
        Some(decl) => format!("@{}", decl.path),
        None => param.to_string(),
    };
    for (id, param) in &program.params {
        write!(f, "param @{}: {} = {}", param.path, param.ty, param.init)?;
        if let Some(version) = optimizer.and_then(|o| o.versions.get(id)) {
            write!(f, " {{{version}}}")?;
        }
        f.write_char('\n')?;
    }
    if let Some(optimizer) = optimizer {
        writeln!(f, "optimizer sgd lr = {:?}", optimizer.lr)?;
    }
    for node in &program.nodes {
        if let Some(tag) = training.subgraphs.get(&node.id) {
            write!(f, "{tag} ")?;
        }
        write_node(f, program, node)?;
        if let Some(effect) = optimizer.and_then(|o| o.effects.get(&node.id)) {
            write!(
                f,
                " {{{} {} {} {}}}",
                effect.token,
                effect.access,
                path(&effect.param),
                effect.version
            )?;
        }
        f.write_char('\n')?;
    }
    for output in &program.outputs {
        write!(f, "output {output}")?;
        if *output == training.loss {
            f.write_str(" loss")?;
        } else if let Some((param, _)) = training.gradients.iter().find(|(_, v)| *v == output) {
            write!(f, " gradient {}", path(param))?;
        } else if let Some((param, _)) =
            optimizer.and_then(|o| o.updates.iter().find(|(_, v)| *v == output))
        {
            write!(f, " update {}", path(param))?;
        }
        f.write_char('\n')?;
    }
    Ok(())
}

impl fmt::Display for TensorProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for param in self.params.values() {
            writeln!(f, "param @{}: {} = {}", param.path, param.ty, param.init)?;
        }
        for node in &self.nodes {
            write_node(f, self, node)?;
            f.write_char('\n')?;
        }
        for output in &self.outputs {
            writeln!(f, "output {output}")?;
        }
        Ok(())
    }
}

/// The text form of Kernel IR and its execution plans, one line each, newline-terminated.
///
/// One line per buffer, `buffer b0: 128 bytes`; per `input` and `target`
/// (`input x: Tensor<f32, [8, 4]> view b0[f32 x 32, bytes 0..128, align 4]`); per parameter, with
/// its views; `loss <view>` and `prediction <view>`. Then one line per kernel, numbered `k0`, `k1`
/// in the order the step plan first dispatches them: `kernel k0 <module id> <kind> <op> <dtype>
/// (<operands>) -> <result> [over <reduced>] workgroup_size <n> workgroups [x, y, z]`, the
/// workgroup size being the constant of the kind. Then `plan step` and `plan forward`, each
/// followed by one indented line per step: its position, op, kernel, value, views, workgroups and
/// dependencies.
pub fn print_kernel(plan: &ExecutionPlan) -> String {
    let mut out = String::new();
    write_kernel(&mut out, plan).expect("writing to a String does not fail");
    out
}

fn write_kernel(out: &mut String, plan: &ExecutionPlan) -> fmt::Result {
    for (i, size) in plan.buffers.iter().enumerate() {
        writeln!(out, "buffer b{i}: {size} bytes")?;
    }
    for data in &plan.data {
        writeln!(
            out,
            "{} {}: {} view {}",
            data.role, data.name, data.ty, data.view
        )?;
    }
    for parameter in &plan.parameters {
        write!(out, "param @{}: {} views", parameter.path, parameter.ty)?;
        for view in &parameter.views {
            write!(out, " {view}")?;
        }
        out.write_char('\n')?;
    }
    writeln!(out, "loss {}", plan.loss_view)?;
    writeln!(out, "prediction {}", plan.prediction_view)?;
    for (i, kernel) in plan.kernels.iter().enumerate() {
        writeln!(out, "kernel k{i} {} {}", kernel.id, kernel.signature())?;
    }
    for (name, steps) in [("step", &plan.step), ("forward", &plan.forward)] {
        writeln!(out, "plan {name}")?;
        for (i, step) in steps.iter().enumerate() {
            write_plan_step(out, plan, i, step)?;
        }
    }
    Ok(())
}

fn write_plan_step(
    out: &mut String,
    plan: &ExecutionPlan,
    at: usize,
    step: &PlanStep,
) -> fmt::Result {
    write!(out, "  {at}: {}", step.op)?;
    if let Some(id) = step.kernel_module_id {
        match plan.kernels.iter().position(|k| k.id == id) {
            Some(k) => write!(out, " k{k}")?,
            None => write!(out, " {id}")?,
        }
    }
    if let Some(value) = step.value {
        write!(out, " {value}")?;
    }
    for (label, views) in [("reads", &step.reads), ("writes", &step.writes)] {
        if !views.is_empty() {
            write!(out, " {label}")?;
            for (i, view) in views.iter().enumerate() {
                out.write_str(if i == 0 { " " } else { ", " })?;
                write!(out, "{view}")?;
            }
        }
    }
    if let Some([x, y, z]) = step.workgroups {
        write!(out, " workgroups [{x}, {y}, {z}]")?;
    }
    if !step.depends_on.is_empty() {
        out.write_str(" after [")?;
        for (i, on) in step.depends_on.iter().enumerate() {
            if i > 0 {
                out.write_str(", ")?;
            }
            write!(out, "{on}")?;
        }
        out.write_char(']')?;
    }
    out.write_char('\n')
}

/// Writes one node without a trailing newline.
pub fn write_node(out: &mut impl Write, program: &TensorProgram, node: &TensorNode) -> fmt::Result {
    write!(out, "{}: {} = {}", node.id, node.ty, node.op.name())?;
    match &node.op {
        TensorOp::Input { name } => write!(out, " {name:?}")?,
        TensorOp::ParamRead { param } => match program.params.get(param) {
            Some(decl) => write!(out, " @{}", decl.path)?,
            None => write!(out, " {param}")?,
        },
        TensorOp::Scalar(value) => write!(out, " {value:?}")?,
        TensorOp::BroadcastScalar { shape } => {
            out.write_str(" [")?;
            for (i, dim) in shape.iter().enumerate() {
                if i > 0 {
                    out.write_str(", ")?;
                }
                write!(out, "{dim}")?;
            }
            out.write_char(']')?;
        }
        _ => {}
    }
    for (i, operand) in node.operands.iter().enumerate() {
        out.write_str(if i == 0 { " " } else { ", " })?;
        write!(out, "{operand}")?;
    }
    Ok(())
}
