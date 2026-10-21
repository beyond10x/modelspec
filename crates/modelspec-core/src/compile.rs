//! The compiler driver: `.mspec` source to Typed HIR and to verified Tensor IR (DESIGN.md § 3,
//! § 3.3 passes 1–3).
//!
//! [`compile_hir`] parses, resolves and checks; [`compile_tensor`] then lowers, removes dead code
//! and verifies. A source fault is reported as the diagnostics of the stage that found it. A
//! program the verifier refuses after lowering, or a lowering failure on checked HIR, is a compiler
//! bug and is reported under [`codes::COMPILER_BUG`], never as a user code (DESIGN.md § 8.1).

use crate::diag::Diagnostic;
use crate::hir::Hir;
use crate::ir::tensor::TensorProgram;
use crate::ir::verify::verify;
use crate::lower::{self, LoweredProgram};
use crate::passes::dce;

/// Compiler-bug diagnostic codes.
pub mod codes {
    use crate::diag::Code;

    /// A compiler verification failure in the tensor pipeline, not a user error (DESIGN.md
    /// § 8.1): lowering refused checked HIR, or the program its passes produced does not verify.
    /// No source file is meant to reach it.
    pub const COMPILER_BUG: Code = Code::new("E-INTERNAL-002");
}

/// Parses, resolves and checks one source file; `path` names it in diagnostic spans. Returns
/// the Typed HIR, or every diagnostic in source order.
pub fn compile_hir(path: &str, source: &str) -> Result<Hir, Vec<Diagnostic>> {
    crate::types::check_source(path, source)
}

/// [`compile_hir`], then lowering ([`lower::lower`]), dead-code elimination ([`dce::eliminate`])
/// and verification ([`verify`]).
pub fn compile_tensor(path: &str, source: &str) -> Result<LoweredProgram, Vec<Diagnostic>> {
    let hir = compile_hir(path, source)?;
    tensor_stage(&hir, dce::eliminate)
}

/// Lowers `hir`, runs `passes` over the program, then verifies the result and that the
/// prediction is still defined.
fn tensor_stage(
    hir: &Hir,
    passes: fn(TensorProgram) -> TensorProgram,
) -> Result<LoweredProgram, Vec<Diagnostic>> {
    let lowered = lower::lower(hir)
        .map_err(|error| vec![compiler_bug("lowering refused checked HIR", &error)])?;
    let program = passes(lowered.program);
    if let Err(error) = verify(&program) {
        return Err(vec![compiler_bug(
            "the lowered Tensor IR does not verify",
            &error,
        )]);
    }
    if program.node(lowered.prediction).is_none() {
        return Err(vec![compiler_bug(
            "the lowered Tensor IR lost its prediction",
            &format!("prediction {} is not defined", lowered.prediction),
        )]);
    }
    Ok(LoweredProgram {
        program,
        prediction: lowered.prediction,
        declared_inputs: lowered.declared_inputs,
    })
}

fn compiler_bug(what: &str, error: &dyn std::fmt::Display) -> Diagnostic {
    Diagnostic::new(codes::COMPILER_BUG, format!("compiler bug: {what}"))
        .with("error", error.to_string())
        .note("this is a fault in modelspec, not in the specification")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::tensor::{Dim, TensorOp, ValueId};

    const EXAMPLE: &str = include_str!("../../../examples/linear_regression.mspec");

    fn example_hir() -> Hir {
        compile_hir("linear_regression.mspec", EXAMPLE).expect("the example checks")
    }

    fn the_compiler_bug(result: Result<LoweredProgram, Vec<Diagnostic>>) -> Diagnostic {
        let diagnostics = result.expect_err("the stage refuses the program");
        assert_eq!(diagnostics.len(), 1, "{diagnostics:#?}");
        let diagnostic = diagnostics.into_iter().next().unwrap();
        assert_eq!(diagnostic.code, codes::COMPILER_BUG, "{diagnostic}");
        assert!(
            !crate::types::codes::ALL.contains(&diagnostic.code),
            "a compiler bug is not a user code"
        );
        assert!(
            diagnostic.message.starts_with("compiler bug: "),
            "{diagnostic}"
        );
        diagnostic
    }

    #[test]
    fn the_tensor_stage_with_dce_accepts_the_example() {
        let lowered = tensor_stage(&example_hir(), crate::passes::dce::eliminate)
            .expect("the example lowers");
        assert_eq!(lowered.program.nodes.len(), 5);
    }

    #[test]
    fn a_program_that_fails_verification_after_its_passes_is_reported_as_a_compiler_bug() {
        let diagnostic = the_compiler_bug(tensor_stage(&example_hir(), |mut program| {
            program.outputs.push(ValueId(99));
            program
        }));
        assert!(
            diagnostic.to_string().contains("output %99 is not defined"),
            "{diagnostic}"
        );
    }

    #[test]
    fn a_pass_that_removes_the_prediction_is_reported_as_a_compiler_bug() {
        let diagnostic = the_compiler_bug(tensor_stage(&example_hir(), |mut program| {
            program
                .nodes
                .retain(|n| !matches!(n.op, TensorOp::MatMul | TensorOp::Mse));
            program.outputs = vec![program.inputs[1]];
            program
        }));
        assert!(
            diagnostic.to_string().contains("prediction"),
            "{diagnostic}"
        );
    }

    #[test]
    fn a_lowering_failure_on_checked_hir_is_reported_as_a_compiler_bug() {
        let mut hir = example_hir();
        hir.model.params[0].ty.shape[0] = Dim::Static(3);
        let diagnostic = the_compiler_bug(tensor_stage(&hir, crate::passes::dce::eliminate));
        assert!(
            diagnostic
                .to_string()
                .contains("matmul contracting dimension mismatch"),
            "{diagnostic}"
        );
    }

    #[test]
    fn checked_hir_whose_loss_does_not_call_forward_is_a_compiler_bug() {
        let mut hir = example_hir();
        let crate::hir::HirExprKind::Op { args, .. } = &mut hir.train.loss.kind else {
            panic!("the example's loss is mse(...)");
        };
        args[0] = args[1].clone();
        let diagnostic = the_compiler_bug(tensor_stage(&hir, crate::passes::dce::eliminate));
        assert!(
            diagnostic
                .to_string()
                .contains("calls `model.forward` 0 time(s), not once"),
            "{diagnostic}"
        );
    }

    #[test]
    fn checked_hir_whose_loss_calls_forward_twice_is_a_compiler_bug() {
        let mut hir = example_hir();
        let once = hir.train.loss.clone();
        hir.train.loss = crate::hir::HirExpr {
            kind: crate::hir::HirExprKind::Op {
                op: TensorOp::Add,
                args: vec![once.clone(), once.clone()],
            },
            ty: once.ty.clone(),
            span: once.span.clone(),
        };
        let diagnostic = the_compiler_bug(tensor_stage(&hir, crate::passes::dce::eliminate));
        assert!(
            diagnostic
                .to_string()
                .contains("calls `model.forward` 2 time(s), not once"),
            "{diagnostic}"
        );
    }
}
