//! Readable IR printer with stable IDs (DESIGN.md § 8).
//!
//! One line per parameter, `param @<path>: <type> = <initializer>`, in `ParamId` order; one line
//! per node, `%2: Tensor<f32, [8, 1]> = MatMul %0, %1`, in definition order, with parameters named
//! by canonical path; then one `output %n` line per output.

use std::fmt::{self, Write};

use crate::ir::tensor::{TensorNode, TensorOp, TensorProgram};

/// The text form of `program`, one line each, newline-terminated.
pub fn print(program: &TensorProgram) -> String {
    program.to_string()
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
