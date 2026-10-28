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
//! A `dropout(x, p = P)` call with `P > 0` is a `Dropout` node over `x` and the RNG token of its
//! site (DESIGN.md § 2.3): an `i32[2]` `Input` named `<Model>.<fn>.dropout#<k>`, where `<fn>` is
//! the function the call is written in (`train` for the loss itself, a keyword no function can
//! take) and `k` the call's index among the `dropout` calls of that function in source order,
//! from 0, counting calls with `p = 0.0`. Every inlining of a function is its own site, so two
//! calls of one function draw two masks: the first inlining of `<fn>` in the program keeps the
//! name `<Model>.<fn>.dropout#<k>`, and the second and later ones are
//! `<Model>.<fn>@<c>.dropout#<k>`, `c` counting from 1 in the order the calls are inlined
//! (coordinator decision D54, 2026-10-25). `p = 0.0` lowers to the operand itself, no node and
//! no token. The training host supplies every site's token each step; [`without_dropout`] makes
//! the program every other run reads, dropout the identity (coordinator decision D47,
//! 2026-10-25).
//!
//! A `sigreg(z, slices = M)` call, `z: f32[N, K]`, is `ReduceMean(SigregColumns(MatMul(z,
//! proj)))`, every node at the call's span, `proj` the projection input of its site: an `f32[K,
//! M]` `Input` named `<Model>.<fn>.sigreg#<k>` by the rule of the dropout sites, `k` the call's
//! index among the `sigreg` calls of its function or loss in source order. The training host
//! supplies every site's projection each step (DESIGN.md § 2.3, story:jepa-sigreg-cpu).
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

use std::collections::BTreeMap;

use thiserror::Error;

use crate::diag::SourceSpan;
use crate::hir::{FnRef, Hir, HirData, HirExpr, HirExprKind, HirModel};
use crate::ir::tensor::{
    BuildError, DType, Dim, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use crate::ops;
use crate::passes::dce;

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
        source: Box<BuildError>,
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
        inlined: vec![0; model.functions.len()],
        loss: &train.loss,
        tokens: BTreeMap::new(),
        projections: BTreeMap::new(),
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

/// `lowered` with every `Dropout` the identity: each use of a `Dropout` reads its operand
/// instead, the `Dropout` nodes are removed, and dead-code elimination removes the token inputs
/// nothing reads any more. Every other node keeps its id; a prediction that was a `Dropout` is
/// its operand. This is the program of the final loss line, evaluation and inference, which run
/// without dropout (coordinator decision D47, 2026-10-25).
pub fn without_dropout(lowered: LoweredProgram) -> LoweredProgram {
    let LoweredProgram {
        mut program,
        prediction,
        declared_inputs,
    } = lowered;
    let mut identity: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let mut nodes = Vec::with_capacity(program.nodes.len());
    for mut node in std::mem::take(&mut program.nodes) {
        for operand in &mut node.operands {
            *operand = identity.get(operand).copied().unwrap_or(*operand);
        }
        if matches!(node.op, TensorOp::Dropout { .. }) {
            identity.insert(node.id, node.operands[0]);
        } else {
            nodes.push(node);
        }
    }
    program.nodes = nodes;
    for output in &mut program.outputs {
        *output = identity.get(output).copied().unwrap_or(*output);
    }
    LoweredProgram {
        program: dce::eliminate(program),
        prediction: identity.get(&prediction).copied().unwrap_or(prediction),
        declared_inputs,
    }
}

/// The source positions of the `sigreg` calls in `expr`, not entering the bodies of the model
/// functions it calls.
fn sigreg_calls(expr: &HirExpr, out: &mut Vec<(u32, u32)>) {
    match &expr.kind {
        HirExprKind::Op { op, args } => {
            if matches!(op, TensorOp::Sigreg { .. }) {
                out.push((expr.span.line, expr.span.column));
            }
            for arg in args {
                sigreg_calls(arg, out);
            }
        }
        HirExprKind::Call { args, .. } => {
            for arg in args {
                sigreg_calls(arg, out);
            }
        }
        HirExprKind::Param(_)
        | HirExprKind::Arg(_)
        | HirExprKind::Let(_)
        | HirExprKind::Data(_)
        | HirExprKind::Scalar(_) => {}
    }
}

/// The source positions of the `dropout` calls in `expr`, not entering the bodies of the model
/// functions it calls.
fn dropout_calls(expr: &HirExpr, out: &mut Vec<(u32, u32)>) {
    match &expr.kind {
        HirExprKind::Op { op, args } => {
            if matches!(op, TensorOp::Dropout { .. }) {
                out.push((expr.span.line, expr.span.column));
            }
            for arg in args {
                dropout_calls(arg, out);
            }
        }
        HirExprKind::Call { args, .. } => {
            for arg in args {
                dropout_calls(arg, out);
            }
        }
        HirExprKind::Param(_)
        | HirExprKind::Arg(_)
        | HirExprKind::Let(_)
        | HirExprKind::Data(_)
        | HirExprKind::Scalar(_) => {}
    }
}

fn build_error(span: &SourceSpan) -> impl FnOnce(BuildError) -> LowerError + '_ {
    move |source| LowerError::Build {
        span: span.clone(),
        source: Box::new(source),
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
    /// The functions being inlined, innermost last, each with its inlining instance `c` (0 for
    /// the first inlining of the function in the program); a call of one of them would not
    /// terminate.
    inlining: Vec<(FnRef, usize)>,
    /// How many times each function has been inlined so far, by [`FnRef`] index.
    inlined: Vec<usize>,
    /// The train block's loss, the body of the dropout sites named `<Model>.train.dropout#<k>`.
    loss: &'h HirExpr,
    /// The RNG token input of each dropout site lowered so far, by site name.
    tokens: BTreeMap<String, ValueId>,
    /// The projection input of each `sigreg` site lowered so far, by site name.
    projections: BTreeMap<String, ValueId>,
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
                let mut operands = Vec::with_capacity(args.len() + 1);
                for arg in args {
                    operands.push(self.expr(arg, frame)?);
                }
                if let TensorOp::Sigreg { slices } = op {
                    return self.sigreg(&operands, *slices, span);
                }
                if let TensorOp::Dropout { p } = op {
                    let [x] = operands[..] else {
                        return Err(malformed(format!(
                            "dropout takes 1 operand; got {}",
                            operands.len()
                        )));
                    };
                    if *p == 0.0 {
                        return Ok(x);
                    }
                    let token = self.token(span)?;
                    operands.insert(ops::token_operand(op).unwrap_or(1), token);
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

    /// The RNG token input of the dropout site of the call at `span`, made at its first call:
    /// `<Model>.<fn>.dropout#<k>` for the first inlining of the function being inlined,
    /// `<Model>.<fn>@<c>.dropout#<k>` for its inlining `c` from 1, or `train` for the loss
    /// (coordinator decision D54, 2026-10-25).
    fn token(&mut self, span: &SourceSpan) -> Result<ValueId, LowerError> {
        let model = self.model;
        let (function, mut calls) = match self.inlining.last() {
            Some(&(f, instance)) => {
                let f = &model.functions[f.0];
                let mut calls = Vec::new();
                for binding in &f.lets {
                    dropout_calls(&binding.value, &mut calls);
                }
                dropout_calls(&f.result, &mut calls);
                let name = match instance {
                    0 => f.name.clone(),
                    c => format!("{}@{c}", f.name),
                };
                (name, calls)
            }
            None => {
                let mut calls = Vec::new();
                dropout_calls(self.loss, &mut calls);
                ("train".to_owned(), calls)
            }
        };
        calls.sort_unstable();
        let Ok(k) = calls.binary_search(&(span.line, span.column)) else {
            return Err(LowerError::Malformed {
                span: span.clone(),
                what: format!("the dropout call is not in the body of `{function}`"),
            });
        };
        let site = format!("{}.{function}.dropout#{k}", model.name);
        if let Some(&token) = self.tokens.get(&site) {
            return Ok(token);
        }
        let token = self
            .builder
            .input_at(&site, ops::token_type(), Some(span.clone()))
            .map_err(build_error(span))?;
        self.tokens.insert(site, token);
        Ok(token)
    }

    /// `sigreg(z, slices = M)` at `span`, `z: f32[N, K]`: `ReduceMean(SigregColumns(MatMul(z,
    /// proj)))`, `proj` the `f32[K, M]` projection input of the call's site ([`Self::projection`]),
    /// every node at the call's span (DESIGN.md § 2.2, § 2.3).
    fn sigreg(
        &mut self,
        operands: &[ValueId],
        slices: u64,
        span: &SourceSpan,
    ) -> Result<ValueId, LowerError> {
        let malformed = |what: String| LowerError::Malformed {
            span: span.clone(),
            what,
        };
        let [z] = operands[..] else {
            return Err(malformed(format!(
                "sigreg takes 1 operand; got {}",
                operands.len()
            )));
        };
        let k = match self.builder.type_of(z).map(|ty| ty.shape.as_slice()) {
            Some(&[_, k]) => k,
            _ => return Err(malformed("the operand of sigreg is not of rank 2".into())),
        };
        let ty = TensorType {
            dtype: DType::F32,
            shape: vec![k, Dim::Static(slices)],
        };
        let proj = self.projection(span, ty)?;
        let at = Some(span.clone());
        let p = self
            .builder
            .push_at(TensorOp::MatMul, &[z, proj], at.clone())
            .map_err(build_error(span))?;
        let columns = self
            .builder
            .push_at(TensorOp::SigregColumns, &[p], at.clone())
            .map_err(build_error(span))?;
        self.builder
            .push_at(TensorOp::ReduceMean, &[columns], at)
            .map_err(build_error(span))
    }

    /// The projection input of the `sigreg` site of the call at `span`, of type `ty`, made at its
    /// first call: `<Model>.<fn>.sigreg#<k>`, `k` the call's index among the `sigreg` calls of
    /// the function in source order, named by the dropout site rule ([`Self::token`]): `<fn>@<c>`
    /// for the inlining `c` from 1, `train` for the loss.
    fn projection(&mut self, span: &SourceSpan, ty: TensorType) -> Result<ValueId, LowerError> {
        let model = self.model;
        let (function, mut calls) = match self.inlining.last() {
            Some(&(f, instance)) => {
                let f = &model.functions[f.0];
                let mut calls = Vec::new();
                for binding in &f.lets {
                    sigreg_calls(&binding.value, &mut calls);
                }
                sigreg_calls(&f.result, &mut calls);
                let name = match instance {
                    0 => f.name.clone(),
                    c => format!("{}@{c}", f.name),
                };
                (name, calls)
            }
            None => {
                let mut calls = Vec::new();
                sigreg_calls(self.loss, &mut calls);
                ("train".to_owned(), calls)
            }
        };
        calls.sort_unstable();
        let Ok(k) = calls.binary_search(&(span.line, span.column)) else {
            return Err(LowerError::Malformed {
                span: span.clone(),
                what: format!("the sigreg call is not in the body of `{function}`"),
            });
        };
        let site = format!("{}.{function}.sigreg#{k}", model.name);
        if let Some(&proj) = self.projections.get(&site) {
            return Ok(proj);
        }
        let proj = self
            .builder
            .input_at(&site, ty, Some(span.clone()))
            .map_err(build_error(span))?;
        self.projections.insert(site, proj);
        Ok(proj)
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
        if self.inlining.iter().any(|&(f, _)| f == function) {
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
        let instance = self.inlined[function.0];
        self.inlined[function.0] += 1;
        self.inlining.push((function, instance));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::tensor::TensorType;
    use crate::ir::verify::verify;

    /// Dropout in `forward` (nested, and one with `p = 0.0`), in `extra`, inlined twice, and in
    /// the loss itself.
    const SITES: &str = "model Reg {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        let h = matmul(dropout(x, p = 0.5), w);
        return dropout(relu(dropout(h, p = 0.25)), p = 0.0);
    }

    fn extra(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        return dropout(matmul(x, w), p = 0.5);
    }
}

train Fit {
    model = Reg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(dropout(add(model.forward(x), add(model.extra(x), model.extra(x))), p = 0.1), y);
    optimizer = sgd(lr = 0.05);
    steps = 2;
    seed = 1;
    device = cpu;
}
";

    fn lowered(source: &str) -> LoweredProgram {
        let hir = crate::types::check_source("sites.mspec", source).expect("it checks");
        lower(&hir).expect("it lowers")
    }

    /// The name of every `Input` node, in node order.
    fn input_names(program: &TensorProgram) -> Vec<&str> {
        program
            .nodes
            .iter()
            .filter_map(|n| match &n.op {
                TensorOp::Input { name } => Some(name.as_str()),
                _ => None,
            })
            .collect()
    }

    /// A site is `<Model>.<fn>.dropout#<k>`, `k` counting every `dropout` call of the function in
    /// source order, `p = 0.0` included; the second inlining of a function is a site of its own,
    /// `<Model>.<fn>@1.dropout#<k>`, with its own token, and a call in the loss is a `train`
    /// site. Each token is an `i32[2]` input.
    #[test]
    fn dropout_sites_are_named_by_function_and_source_order() {
        let lowered = lowered(SITES);
        let program = &lowered.program;
        assert_eq!(
            input_names(program),
            [
                "x",
                "y",
                "Reg.forward.dropout#0",
                "Reg.forward.dropout#2",
                "Reg.extra.dropout#0",
                "Reg.extra@1.dropout#0",
                "Reg.train.dropout#0",
            ]
        );
        for node in &program.nodes {
            if let TensorOp::Input { name } = &node.op
                && name.contains(".dropout#")
            {
                assert_eq!(node.ty, TensorType::i32(&[2]), "{name}");
            }
        }
        let token_of = |name: &str| {
            program
                .nodes
                .iter()
                .find(|n| n.op == TensorOp::Input { name: name.into() })
                .unwrap()
                .id
        };
        let dropouts: Vec<(f32, ValueId)> = program
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                TensorOp::Dropout { p } => Some((p, n.operands[1])),
                _ => None,
            })
            .collect();
        assert_eq!(
            dropouts,
            [
                (0.5, token_of("Reg.forward.dropout#0")),
                (0.25, token_of("Reg.forward.dropout#2")),
                (0.5, token_of("Reg.extra.dropout#0")),
                (0.5, token_of("Reg.extra@1.dropout#0")),
                (0.1, token_of("Reg.train.dropout#0")),
            ]
        );
        verify(program).expect("the lowered program verifies");
    }

    /// `p = 0.0` lowers to its operand: no node, no token.
    #[test]
    fn dropout_with_p_0_is_its_operand() {
        let source = SITES
            .replace("p = 0.5", "p = 0.0")
            .replace("p = 0.25", "p = 0.0");
        let source = source.replace("p = 0.1", "p = 0.0");
        let lowered = lowered(&source);
        assert!(
            lowered
                .program
                .nodes
                .iter()
                .all(|n| !matches!(n.op, TensorOp::Dropout { .. }))
        );
        assert_eq!(input_names(&lowered.program), ["x", "y"]);
    }

    /// [`without_dropout`] removes every `Dropout` and every token, keeps the id of every other
    /// node, and verifies.
    #[test]
    fn without_dropout_is_the_identity_and_keeps_node_ids() {
        let with = dce::eliminate(lowered(SITES).program);
        let with = LoweredProgram {
            program: with,
            ..lowered(SITES)
        };
        let without = without_dropout(with.clone());
        verify(&without.program).expect("the program without dropout verifies");
        assert_eq!(input_names(&without.program), ["x", "y"]);
        for node in &without.program.nodes {
            assert!(!matches!(node.op, TensorOp::Dropout { .. }));
            let before = with.program.node(node.id).expect("the id is kept");
            assert_eq!(before.op, node.op);
        }
        assert_eq!(without.prediction, with.prediction);
        assert!(without.program.node(without.prediction).is_some());
    }

    /// A prediction that is itself a `Dropout` becomes its operand.
    #[test]
    fn without_dropout_maps_a_dropout_prediction_to_its_operand() {
        let source = SITES.replace(
            "return dropout(relu(dropout(h, p = 0.25)), p = 0.0);",
            "return dropout(h, p = 0.25);",
        );
        let with = lowered(&source);
        let node = with.program.node(with.prediction).unwrap();
        assert!(matches!(node.op, TensorOp::Dropout { .. }));
        let h = node.operands[0];
        let without = without_dropout(with);
        assert_eq!(without.prediction, h);
        verify(&without.program).expect("it verifies");
    }

    const SIGREG_SITES: &str = "model Enc {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    param v: Tensor<f32, [4, 6]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        return matmul(x, w);
    }

    fn spread(x: Tensor<f32, [8, 4]>) -> Tensor<f32, []> {
        let z = matmul(x, w);
        return add(sigreg(z, slices = 2), sigreg(z, slices = 5));
    }

    fn wide(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 6]> {
        return matmul(x, v);
    }
}

train Fit {
    model = Enc;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = add(add(mse(model.forward(x), y), model.spread(x)), add(model.spread(x), sigreg(model.wide(x), slices = 4)));
    optimizer = sgd(lr = 0.05);
    steps = 2;
    seed = 1;
    device = cpu;
}
";

    /// A `sigreg` call lowers to `ReduceMean(SigregColumns(MatMul(z, proj)))`, `proj` an `f32[K,
    /// M]` input named by the dropout site rule with `sigreg#<k>`: the second inlining of a
    /// function is `<fn>@1`, a call in the loss is a `train` site. No `Sigreg` node remains, and
    /// the program verifies.
    #[test]
    fn sigreg_sites_are_named_by_function_and_source_order() {
        let lowered = lowered(SIGREG_SITES);
        let program = &lowered.program;
        assert_eq!(
            input_names(program),
            [
                "x",
                "y",
                "Enc.spread.sigreg#0",
                "Enc.spread.sigreg#1",
                "Enc.spread@1.sigreg#0",
                "Enc.spread@1.sigreg#1",
                "Enc.train.sigreg#0",
            ]
        );
        let ty = |name: &str| {
            program
                .nodes
                .iter()
                .find(|n| n.op == TensorOp::Input { name: name.into() })
                .map(|n| n.ty.clone())
                .unwrap()
        };
        assert_eq!(ty("Enc.spread.sigreg#0"), TensorType::f32(&[3, 2]));
        assert_eq!(ty("Enc.spread.sigreg#1"), TensorType::f32(&[3, 5]));
        assert_eq!(ty("Enc.train.sigreg#0"), TensorType::f32(&[6, 4]));
        assert!(program.nodes.iter().all(|n| !ops::expanded(&n.op)));
        let columns: Vec<_> = program
            .nodes
            .iter()
            .filter(|n| n.op == TensorOp::SigregColumns)
            .collect();
        assert_eq!(columns.len(), 5);
        for node in columns {
            let matmul = program.node(node.operands[0]).unwrap();
            assert_eq!(matmul.op, TensorOp::MatMul);
            let proj = program.node(matmul.operands[1]).unwrap();
            assert!(matches!(&proj.op, TensorOp::Input { name } if name.contains(".sigreg#")));
            let mean = program
                .nodes
                .iter()
                .find(|n| n.operands == [node.id])
                .unwrap();
            assert_eq!(mean.op, TensorOp::ReduceMean);
        }
        verify(program).expect("it verifies");
    }
}
