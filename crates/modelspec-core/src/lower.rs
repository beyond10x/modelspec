//! Lowering: Typed HIR to Tensor IR (DESIGN.md § 2.1, § 2.3, § 3, § 3.3 pass 2).
//!
//! [`lower`] turns the train block's loss into one SSA graph, in this order:
//!
//! 1. the parameter table, one entry per `param` in source order, keyed by its canonical path
//!    (DESIGN.md § 2.3) with its initializer;
//! 2. one `Input` node per `input` and `target` declaration, in source order;
//! 3. one `ParamRead` per `param`, in source order, which every use of the parameter shares;
//! 4. the loss, operands before the operation, left to right. A `model.<function>(args)` call is
//!    inlined (DESIGN.md § 2.1): its arguments are lowered in the caller, then each `let` of the
//!    function in order, then its `return` expression. Each call is inlined afresh.
//!
//! The loss is the program's single output. Every node carries the span of the source construct
//! it came from: the declaration for `Input` and `ParamRead`, the expression otherwise. Nothing
//! is removed here, so a `let`, input or parameter the loss does not use is a dead node until
//! [`crate::passes::dce`] removes it. An `input` or `target` the loss does not read therefore
//! leaves the graph's inputs, but stays in [`LoweredProgram::declared_inputs`], which lists every
//! declaration with its name and type.
//!
//! The checker guarantees the loss calls `model.forward` exactly once (E-TYPE-017), and that
//! call's value is the prediction. Any [`LowerError`] is therefore a compiler bug.

use thiserror::Error;

use crate::diag::SourceSpan;
use crate::hir::{FnRef, Hir, HirData, HirExpr, HirExprKind, HirModel};
use crate::ir::tensor::{BuildError, TensorOp, TensorProgram, TensorProgramBuilder, ValueId};

/// A lowered train block: its Tensor IR and the value of its model's prediction.
#[derive(Clone, Debug, PartialEq)]
pub struct LoweredProgram {
    /// Outputs: the loss alone.
    pub program: TensorProgram,
    /// The value of the loss's one inlined `model.forward(...)` call (E-TYPE-017 refuses a loss
    /// with none or several). The loss depends on it, so dead-code elimination keeps it.
    pub prediction: ValueId,
    /// Every `input` and `target` declaration of the train block, in declaration order, whether
    /// or not the IR reads it. After dead-code elimination `program.inputs` lists only the read
    /// ones; this list is the data interface the source declares.
    pub declared_inputs: Vec<HirData>,
}

/// Why lowering refused a Typed HIR.
///
/// Every variant means the HIR broke an invariant [`crate::types::check`] guarantees, so each
/// is a compiler bug (DESIGN.md § 8.1), never a fault of the source.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LowerError {
    /// The loss does not call `model.forward` exactly once, which E-TYPE-017 guarantees.
    #[error(
        "{span}: the loss of train block `{train}` calls `model.forward` {count} time(s), not once"
    )]
    ForwardCalls {
        train: String,
        count: usize,
        span: SourceSpan,
    },
    #[error("{span}: {source}")]
    Build {
        span: SourceSpan,
        source: BuildError,
    },
    #[error("{span}: {what}")]
    Malformed { span: SourceSpan, what: String },
}

/// Lowers the train block of `hir` with its model inlined.
pub fn lower(hir: &Hir) -> Result<LoweredProgram, LowerError> {
    let model = &hir.model;
    let train = &hir.train;
    let mut builder = TensorProgramBuilder::new();

    let mut params = Vec::with_capacity(model.params.len());
    for param in &model.params {
        let id = builder
            .declare_param(&param.path, param.ty.clone(), param.init)
            .map_err(build_error(&param.span))?;
        params.push(id);
    }
    let mut data = Vec::with_capacity(train.data.len());
    for decl in &train.data {
        let value = builder
            .input_at(&decl.name, decl.ty.clone(), Some(decl.span.clone()))
            .map_err(build_error(&decl.span))?;
        data.push(value);
    }
    let mut reads = Vec::with_capacity(params.len());
    for (param, &id) in model.params.iter().zip(&params) {
        let value = builder
            .push_at(
                TensorOp::ParamRead { param: id },
                &[],
                Some(param.span.clone()),
            )
            .map_err(build_error(&param.span))?;
        reads.push(value);
    }

    let mut lowering = Lowering {
        builder,
        model,
        reads,
        forward: model.function("forward").map(|(f, _)| f),
        predictions: Vec::new(),
        inlining: Vec::new(),
    };
    let loss = lowering.expr(&train.loss, &Frame::Train { data: &data })?;
    lowering
        .builder
        .output(loss)
        .map_err(build_error(&train.loss.span))?;
    let prediction = match lowering.predictions.as_slice() {
        [prediction] => *prediction,
        calls => {
            return Err(LowerError::ForwardCalls {
                train: train.name.clone(),
                count: calls.len(),
                span: train.loss.span.clone(),
            });
        }
    };
    Ok(LoweredProgram {
        program: lowering.builder.finish(),
        prediction,
        declared_inputs: train.data.clone(),
    })
}

fn build_error(span: &SourceSpan) -> impl FnOnce(BuildError) -> LowerError + '_ {
    move |source| LowerError::Build {
        span: span.clone(),
        source,
    }
}

/// What the names of an expression refer to.
enum Frame<'a> {
    /// The train block: its `input` and `target` values.
    Train { data: &'a [ValueId] },
    /// An inlined model function: its argument values and the `let` values lowered so far.
    Function {
        args: Vec<ValueId>,
        lets: Vec<ValueId>,
    },
}

struct Lowering<'h> {
    builder: TensorProgramBuilder,
    model: &'h HirModel,
    /// The `ParamRead` of each parameter, by [`crate::hir::ParamRef`].
    reads: Vec<ValueId>,
    forward: Option<FnRef>,
    /// The value of each inlined `model.forward(...)` call, in evaluation order.
    predictions: Vec<ValueId>,
    /// The functions being inlined, innermost last; a call of one of them would not terminate.
    inlining: Vec<FnRef>,
}

impl Lowering<'_> {
    fn expr(&mut self, expr: &HirExpr, frame: &Frame<'_>) -> Result<ValueId, LowerError> {
        let span = &expr.span;
        let malformed = |what: String| LowerError::Malformed {
            span: span.clone(),
            what,
        };
        match &expr.kind {
            HirExprKind::Param(p) => self
                .reads
                .get(p.0)
                .copied()
                .ok_or_else(|| malformed(format!("parameter {} is not declared", p.0))),
            HirExprKind::Arg(a) => match frame {
                Frame::Function { args, .. } => args
                    .get(a.0)
                    .copied()
                    .ok_or_else(|| malformed(format!("argument {} is not bound", a.0))),
                Frame::Train { .. } => Err(malformed("an argument outside a function".into())),
            },
            HirExprKind::Let(l) => match frame {
                Frame::Function { lets, .. } => lets
                    .get(l.0)
                    .copied()
                    .ok_or_else(|| malformed(format!("let {} is not bound yet", l.0))),
                Frame::Train { .. } => Err(malformed("a let outside a function".into())),
            },
            HirExprKind::Data(d) => match frame {
                Frame::Train { data } => data
                    .get(d.0)
                    .copied()
                    .ok_or_else(|| malformed(format!("data declaration {} is not declared", d.0))),
                Frame::Function { .. } => {
                    Err(malformed("an input or target inside a function".into()))
                }
            },
            HirExprKind::Scalar(value) => self
                .builder
                .push_at(TensorOp::Scalar(*value), &[], Some(span.clone()))
                .map_err(build_error(span)),
            HirExprKind::Op { op, args } => {
                let mut operands = Vec::with_capacity(args.len());
                for arg in args {
                    operands.push(self.expr(arg, frame)?);
                }
                self.builder
                    .push_at(op.clone(), &operands, Some(span.clone()))
                    .map_err(build_error(span))
            }
            HirExprKind::Call { function, args } => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(self.expr(arg, frame)?);
                }
                let value = self.inline(*function, values, span)?;
                if Some(*function) == self.forward {
                    self.predictions.push(value);
                }
                Ok(value)
            }
        }
    }

    fn inline(
        &mut self,
        function: FnRef,
        args: Vec<ValueId>,
        span: &SourceSpan,
    ) -> Result<ValueId, LowerError> {
        let model = self.model;
        let Some(f) = model.functions.get(function.0) else {
            return Err(LowerError::Malformed {
                span: span.clone(),
                what: format!("function {} is not declared", function.0),
            });
        };
        if self.inlining.contains(&function) {
            return Err(LowerError::Malformed {
                span: span.clone(),
                what: format!("`{}` calls itself", f.name),
            });
        }
        if args.len() != f.args.len() {
            return Err(LowerError::Malformed {
                span: span.clone(),
                what: format!(
                    "`{}` takes {} argument(s); got {}",
                    f.name,
                    f.args.len(),
                    args.len()
                ),
            });
        }
        self.inlining.push(function);
        let mut frame = Frame::Function {
            args,
            lets: Vec::with_capacity(f.lets.len()),
        };
        for binding in &f.lets {
            let value = self.expr(&binding.value, &frame)?;
            if let Frame::Function { lets, .. } = &mut frame {
                lets.push(value);
            }
        }
        let result = self.expr(&f.result, &frame)?;
        self.inlining.pop();
        Ok(result)
    }
}
