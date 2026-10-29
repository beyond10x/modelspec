//! Type and shape checking: Source AST to Typed HIR (DESIGN.md § 2.2, § 3, § 3.3 passes 1–2).
//!
//! [`check`] resolves names through [`crate::resolve`], types every expression through the
//! operator registry ([`crate::ops::infer`]) and maps each registry error to an `E-TYPE`
//! diagnostic with the span of the call. It reports every diagnostic of the file, not only the
//! first: an expression that fails is reported once and is not typed further, so its failure does
//! not cascade into the expressions that contain it. Diagnostics are returned in source order.
//!
//! Beyond the registry's operator rules, the checker enforces the M0 rules of DESIGN.md § 2.2
//! and the train-block rules: every declared tensor type has positive dimensions and element and
//! byte counts within `u64`; an `i32` tensor is declared only by an `input`, a `target` or a
//! function argument, and read only as the ids of `embedding` or the labels of `cross_entropy`
//! (E-TYPE-018 for a parameter or a function result of `i32`; the registry refuses every other
//! operand); the third use of `i32`, the RNG token of `dropout` ([`ops::token_operand`]), is no
//! source value: the checker types the call with an `i32[2]` token in its place and the lowering
//! supplies it; a parameter is initialized by `xavier_uniform()` and is rank 2; a
//! function's `return` has its declared type; a `model.<function>(args)` call matches the
//! function's arguments; the loss is rank 0 and calls `model.forward` exactly once, counting calls
//! nested in the arguments of another (the prediction is that call's value, DESIGN.md § 2.1);
//! the learning rate of `sgd`, `adamw` and `muon` is a positive finite `f32`, and each other
//! `adamw` or `muon` argument is given once, named, and in range (E-TYPE-023, E-TYPE-024,
//! E-TYPE-025, coordinator decisions D50 and D67);
//! `steps` is positive. An operator that takes named compile-time arguments
//! ([`ops::named_parameters`], `heads` of `attention`, `p` of `dropout`) gets each exactly once,
//! as a literal of its kind, after the operands (E-TYPE-001 when omitted, E-TYPE-002 when
//! unknown, repeated or before an operand, E-TYPE-019 for another value, E-TYPE-022 for a float
//! literal `p` outside `[0, 1)`, coordinator decision D47); every other call takes positional arguments
//! only. An operand that takes no gradient ([`ops::constant_operands`], the `key_mask` of
//! `attention`, the `labels` and `weights` of `cross_entropy`) must not read a parameter by
//! dataflow (E-TYPE-021, coordinator decisions D41 and D43):
//! not through a `let`, an operator, the result of a `model.<function>(...)` call whose body
//! reads one, nor an argument of such a call that reaches the result or reaches such an operand
//! inside the function, which is refused at that argument. Each parameter the loss reads must
//! reach it on at least one path without a `stop_gradient`: a parameter read only through
//! `stop_gradient` would get no gradient, and is refused at that call (E-TYPE-026,
//! story:jepa-stop-gradient). The parser already enforces that `seed` fits `u64`, that the optimizer is
//! `sgd`, `adamw` or `muon`, that `sgd` takes `lr` alone, and that literals are finite `f64`s. A
//! `vulkan(<index>)` device is not validated.

use std::collections::{BTreeMap, BTreeSet};

use crate::diag::{Code, Diagnostic, SourceSpan};
use crate::hir::{
    ArgRef, DataRef, Device, FnRef, Hir, HirArg, HirData, HirExpr, HirExprKind, HirFunction,
    HirLet, HirModel, HirParam, HirTrain, LetRef,
};
use crate::ir::tensor::{Initializer, TensorOp, TensorType};
use crate::ir::training::{AdamW, MUON_MAX_NS_STEPS, Muon, OptimizerRule};
use crate::ops::{self, OpError};
use crate::resolve::{self, Binding, ModelNames, Scope};
use crate::syntax::{self, ast};

/// The named arguments of `adamw(...)`, each required exactly once, in any order (DESIGN.md
/// § 2.1, coordinator decision D50).
pub const ADAMW_ARGUMENTS: [&str; 6] = [
    "lr",
    "weight_decay",
    "beta1",
    "beta2",
    "eps",
    "warmup_steps",
];

/// The named arguments of `muon(...)`, each required exactly once, in any order (DESIGN.md
/// § 2.1, coordinator decision D67).
pub const MUON_ARGUMENTS: [&str; 9] = [
    "lr",
    "momentum",
    "weight_decay",
    "ns_steps",
    "warmup_steps",
    "adam_lr",
    "beta1",
    "beta2",
    "eps",
];

/// A literal optimizer argument as written, at its argument's span.
#[derive(Clone, Copy, Debug)]
enum Literal<'a> {
    Float(f64, &'a SourceSpan),
    Int(u64, &'a SourceSpan),
}

/// Type and shape diagnostic codes.
pub mod codes {
    use crate::diag::Code;

    macro_rules! declare {
        ($($(#[doc = $doc:literal])* $name:ident = $text:literal;)*) => {
            $(
                $(#[doc = $doc])*
                pub const $name: Code = Code::new($text);
            )*

            /// Every type and shape code, in numeric order.
            pub const ALL: &[Code] = &[$($name),*];
        };
    }

    declare! {
        /// An operator or model function called with the wrong number of arguments, or an
        /// operator call that omits a named argument the operator takes, such as `heads` of
        /// `attention`, or an `adamw` or `muon` optimizer that omits one of its six or nine named
        /// arguments.
        ARGUMENT_COUNT = "E-TYPE-001";
        /// A named argument (`a = x`) to a model function or to an operator that takes no named
        /// argument of that name, a named argument given twice, or a named argument before an
        /// operand: the named arguments follow the operands. An `adamw` or `muon` argument that
        /// is unnamed, not one of its own, or given twice.
        NAMED_ARGUMENT = "E-TYPE-002";
        /// `add`, `sub`, `mul` or `mse` operands of different shapes, neither of them rank 0, an
        /// `add_rows(x, b)` whose `b` is not as long as a row of `x`, a
        /// `layer_norm(x, gamma, beta)` whose `gamma` or `beta` is not as long as a row of `x`,
        /// an `attention(q, k, v, key_mask, heads = A)` whose `k` differs from `q` in width, whose
        /// `v` differs from `k` in shape, or whose `key_mask` of shape `[B, S]` has `B * S` other
        /// than the rows of `k` or a `B` that does not divide the rows of `q` into at most `S`, or whose
        /// `i32` spans operand of shape `[U, 2]` has `U` of 0 or a `q` of other than the rows of `k`
        /// or `U` rows, or a
        /// `cross_entropy(logits, labels, weights)` whose `labels` or `weights` is not as long as
        /// the rows of `logits`.
        SHAPE_MISMATCH = "E-TYPE-003";
        /// A `matmul`, `transpose` or `softmax` operand that is not rank 2, an `embedding` whose
        /// table is not rank 2 or whose ids are not rank 1, an `add_rows(x, b)` whose `x` is not
        /// rank 2 or whose `b` is not rank 1, a `layer_norm(x, gamma, beta)` whose `x` is not
        /// rank 2 or whose `gamma` or `beta` is not rank 1, an `attention` operand that is not
        /// rank 2, or a `cross_entropy` whose `logits` is not rank 2 or whose `labels` or
        /// `weights` is not rank 1.
        RANK = "E-TYPE-004";
        /// An integer literal where a tensor is expected.
        INTEGER_LITERAL = "E-TYPE-005";
        /// A float literal that is not a finite `f32`.
        FLOAT_LITERAL_RANGE = "E-TYPE-006";
        /// A function whose `return` expression's type differs from its declared result type.
        RETURN_TYPE = "E-TYPE-007";
        /// A model function argument whose type differs from the parameter's declared type.
        ARGUMENT_TYPE = "E-TYPE-008";
        /// A train block's `loss` that is not rank 0.
        LOSS_RANK = "E-TYPE-009";
        /// An `sgd`, `adamw` or `muon` learning rate `lr`, or an `adam_lr` of `muon`, that is not
        /// a positive finite `f32`.
        LEARNING_RATE = "E-TYPE-010";
        /// A train block's `steps` of 0.
        STEPS = "E-TYPE-011";
        /// `matmul([M, K], [K', N])` with `K != K'` (DESIGN.md § 8.1).
        MATMUL_CONTRACTING = "E-TYPE-012";
        /// A parameter initializer other than `xavier_uniform()`, `zeros()` and `ones()`, or one
        /// of them called with arguments.
        INITIALIZER = "E-TYPE-013";
        /// `xavier_uniform()` on a parameter that is not rank 2, or `zeros()` or `ones()` on a
        /// rank-0 parameter.
        INITIALIZER_RANK = "E-TYPE-014";
        /// A tensor type with a dimension of 0 (DESIGN.md § 2.2).
        ZERO_DIMENSION = "E-TYPE-015";
        /// A tensor type whose element or byte count overflows `u64`, declared or derived by an
        /// operator from a named argument, such as the projection `sigreg` derives from `slices`
        /// (DESIGN.md § 2.2).
        COUNT_OVERFLOW = "E-TYPE-016";
        /// A train block's loss that does not call `model.forward` exactly once: the prediction
        /// is the value of that one call.
        FORWARD_CALLS = "E-TYPE-017";
        /// An `i32` tensor used other than as the ids of `embedding` or the labels of
        /// `cross_entropy`: an operand of any other operator, the table of `embedding`, the
        /// `logits` or `weights` of `cross_entropy`, an `f32` where those ids or labels go, a
        /// `param` of type `i32`, or a function declared to return `i32` (DESIGN.md § 2.2). An
        /// `i32` fourth operand of `attention` is its key spans only at rank 2 and width 2
        /// (`[U, 2]`); of any other shape it is a key mask of the wrong element type. The
        /// RNG token of `dropout`, the third use of `i32`, is supplied by the compiler, never by
        /// a source value.
        DTYPE = "E-TYPE-018";
        /// A named operator argument whose value is not a literal of its kind, such as
        /// `heads = x` or `heads = 0` of `attention`, whose `heads` is an integer literal of at
        /// least 1, or `p = x` or `p = 0` of `dropout`, whose `p` is a float literal
        /// (DESIGN.md § 2.1), or an `adamw` or `muon` argument other than a float literal, or
        /// than an integer literal for `warmup_steps` and `ns_steps`.
        NAMED_VALUE = "E-TYPE-019";
        /// An `attention` whose `heads` does not divide the width `H` of `q`, `k` and `v`.
        HEADS_DIVIDE = "E-TYPE-020";
        /// An operand that takes no gradient, the `key_mask` of `attention` or the `labels` or
        /// `weights` of `cross_entropy`, whose value reads a model parameter by dataflow:
        /// directly, through a `let`, through a `model.<function>(...)` call whose result reads
        /// one, or through an argument of a model function call that reaches the operand inside
        /// the function, reported at that argument. A model function that reads no parameter is
        /// no reason to refuse.
        CONSTANT_OPERAND = "E-TYPE-021";
        /// A `dropout` whose `p` is a float literal outside `[0, 1)` once rounded to `f32`, such
        /// as `p = 1.0`, whose scale `1 / (1 - p)` is infinite, or `p = -0.1` (DESIGN.md
        /// § 2.3).
        DROPOUT_RATE = "E-TYPE-022";
        /// An `adamw` or `muon` argument out of range: a `weight_decay` below 0 or not finite, a
        /// `beta1` or `beta2` outside `[0, 1)` once rounded to `f32`, such as `beta1 = 1.0`, or an
        /// `eps` that is not a positive finite `f32`, such as `eps = 0.0` (DESIGN.md § 2.1).
        OPTIMIZER_HYPERPARAMETER = "E-TYPE-023";
        /// An `adamw` or `muon` whose `warmup_steps` is not below the train block's `steps`,
        /// which would leave its learning rate no step to decay over (DESIGN.md § 2.1,
        /// coordinator decision D50).
        WARMUP_STEPS = "E-TYPE-024";
        /// A `muon` argument out of range: a `momentum` outside `[0, 1)` once rounded to `f32`,
        /// such as `momentum = 1.0`, or an `ns_steps` outside `[1, 10]`, such as `ns_steps = 0`
        /// or `ns_steps = 11` (DESIGN.md § 2.1, coordinator decision D67).
        MUON_HYPERPARAMETER = "E-TYPE-025";
        /// A parameter that reaches the train block's loss only through `stop_gradient`: it
        /// would receive no gradient, and training needs one for every parameter the loss reads.
        /// Reported at the `stop_gradient` call (DESIGN.md § 2.2, § 8.1).
        PARAMETER_ONLY_THROUGH_STOP_GRADIENT = "E-TYPE-026";
    }

    /// A compiler verification failure, not a user error (DESIGN.md § 8.1): the checker reached a
    /// state its own rules exclude, such as a registry dtype mismatch between two operands that
    /// each passed their own dtype check.
    /// Not in [`ALL`]; no source file is meant to reach it.
    pub const INTERNAL: Code = Code::new("E-INTERNAL-001");
}

/// Parses and checks one `.mspec` source file; `file` names it in diagnostic spans. Returns the
/// Typed HIR, or every diagnostic in source order: the parser's when the file does not parse,
/// otherwise the checker's.
pub fn check_source(file: &str, source: &str) -> Result<Hir, Vec<Diagnostic>> {
    let ast = syntax::parse(file, source).map_err(source_order)?;
    check(&ast)
}

/// Checks a parsed file: name resolution, then types and shapes. Returns the Typed HIR, or every
/// diagnostic in source order.
pub fn check(ast: &ast::Ast) -> Result<Hir, Vec<Diagnostic>> {
    let mut checker = Checker {
        diagnostics: Vec::new(),
    };
    let hir = checker.file(ast);
    let diagnostics = source_order(checker.diagnostics);
    match hir {
        Some(hir) if diagnostics.is_empty() => Ok(hir),
        _ if !diagnostics.is_empty() => Err(diagnostics),
        _ => Err(vec![
            Diagnostic::new(
                codes::INTERNAL,
                "the checker produced neither Typed HIR nor a diagnostic",
            )
            .at(ast.span.clone()),
        ]),
    }
}

/// Sorts diagnostics by line, then column; a stable sort, so diagnostics at one position keep
/// the order they were reported in.
/// Whether the element and byte counts of `ty` fit `u64` (DESIGN.md § 2.2), a symbolic
/// dimension counting as 1.
fn counts_fit_u64(ty: &TensorType) -> bool {
    ty.shape
        .iter()
        .try_fold(1u64, |count, dim| match dim {
            crate::ir::tensor::Dim::Static(n) => count.checked_mul(*n),
            crate::ir::tensor::Dim::Symbol(_) => Some(count),
        })
        .and_then(|elements| elements.checked_mul(ty.dtype.size_bytes()))
        .is_some()
}

fn source_order(mut diagnostics: Vec<Diagnostic>) -> Vec<Diagnostic> {
    diagnostics.sort_by_key(|d| d.span.as_ref().map(|s| (s.line, s.column)));
    diagnostics
}

/// Appends the span of every call of `forward` in `expr`, nested calls included.
fn forward_calls(expr: &HirExpr, forward: FnRef, out: &mut Vec<SourceSpan>) {
    match &expr.kind {
        HirExprKind::Call { function, args } => {
            if *function == forward {
                out.push(expr.span.clone());
            }
            for arg in args {
                forward_calls(arg, forward, out);
            }
        }
        HirExprKind::Op { args, .. } => {
            for arg in args {
                forward_calls(arg, forward, out);
            }
        }
        _ => {}
    }
}

/// The source name of an operator call, for messages: `matmul` rather than the IR's `MatMul`.
type OpName<'a> = &'a str;

/// The declared types of a model function; `None` where a declared type was itself invalid.
struct Signature {
    name: String,
    args: Vec<Option<TensorType>>,
    output: Option<TensorType>,
}

/// What an expression can see: its scope and the types of what the scope binds.
struct Env<'a> {
    scope: &'a Scope,
    params: &'a [Option<TensorType>],
    args: &'a [Option<TensorType>],
    lets: &'a [Option<TensorType>],
    /// What each earlier `let` reads by dataflow.
    let_reads: &'a [Reads],
    /// The dataflow of each model function, in the train block.
    flows: &'a [Flow],
    data: &'a [Option<TensorType>],
    in_train: bool,
    /// The train block's model, when its `model` setting resolved.
    model: Option<(&'a ModelNames, &'a [Signature])>,
}

struct Checker {
    diagnostics: Vec<Diagnostic>,
}

impl Checker {
    fn report(&mut self, code: Code, message: impl Into<String>, span: &SourceSpan) {
        self.diagnostics
            .push(Diagnostic::new(code, message).at(span.clone()));
    }

    fn file(&mut self, ast: &ast::Ast) -> Option<Hir> {
        let names = ModelNames::collect(&ast.model, &mut self.diagnostics);
        let mut params = Vec::new();
        let mut param_types = Vec::new();
        let mut signatures = Vec::new();
        for item in &ast.model.items {
            match item {
                ast::ModelItem::Param(p) => {
                    let (ty, param) = self.param(&ast.model.name.name, p);
                    param_types.push(ty);
                    params.push(param);
                }
                ast::ModelItem::Fn(f) => signatures.push(Signature {
                    name: f.name.name.clone(),
                    args: f.params.iter().map(|a| self.declared(&a.ty)).collect(),
                    output: self.output(f),
                }),
            }
        }
        let (functions, flows): (Vec<Option<HirFunction>>, Vec<Flow>) = ast
            .model
            .items
            .iter()
            .filter_map(|item| match item {
                ast::ModelItem::Fn(f) => Some(f),
                ast::ModelItem::Param(_) => None,
            })
            .zip(&signatures)
            .map(|(f, sig)| self.function(f, sig, &names, &param_types))
            .unzip();
        let train = self.train(&ast.train, &names, &signatures, &flows);
        let hir = Hir {
            model: HirModel {
                name: ast.model.name.name.clone(),
                params: params.into_iter().collect::<Option<_>>()?,
                functions: functions.into_iter().collect::<Option<_>>()?,
                span: ast.model.span.clone(),
            },
            train: train?,
        };
        self.stopped_params(&hir);
        Some(hir)
    }

    /// E-TYPE-026 at the `stop_gradient` call for each parameter that the loss of `hir` reads
    /// only through `stop_gradient` ([`GradientReads`]), in parameter order.
    fn stopped_params(&mut self, hir: &Hir) {
        let mut summaries: Vec<GradientReads> = Vec::new();
        for function in &hir.model.functions {
            let mut lets: Vec<GradientReads> = Vec::new();
            for l in &function.lets {
                let value = gradient_reads(&l.value, &lets, &summaries);
                lets.push(value);
            }
            let result = gradient_reads(&function.result, &lets, &summaries);
            summaries.push(result);
        }
        let loss = gradient_reads(&hir.train.loss, &[], &summaries);
        for (source, span) in &loss.stopped {
            let Source::Param(index) = *source else {
                continue;
            };
            if loss.live.contains(source) {
                continue;
            }
            let name = hir
                .model
                .params
                .get(index)
                .map_or("a parameter", |p| p.name.as_str());
            self.report(
                codes::PARAMETER_ONLY_THROUGH_STOP_GRADIENT,
                format!(
                    "parameter `{name}` reaches the loss only through `stop_gradient`, so it \
                     would receive no gradient; training needs one for every parameter the loss \
                     reads"
                ),
                span,
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // Declarations.
    // -----------------------------------------------------------------------------------------

    /// A declared tensor type: positive dimensions (E-TYPE-015), counts within `u64`
    /// (E-TYPE-016).
    fn declared(&mut self, ty: &ast::TensorType) -> Option<TensorType> {
        let sizes: Vec<u64> = ty.dims.iter().map(|d| d.size).collect();
        let converted = match ty.dtype.kind {
            ast::DType::F32 => TensorType::f32(&sizes),
            ast::DType::I32 => TensorType::i32(&sizes),
        };
        let mut valid = true;
        for dim in ty.dims.iter().filter(|d| d.size == 0) {
            self.report(
                codes::ZERO_DIMENSION,
                format!("zero dimension in {converted}; dimensions are positive"),
                &dim.span,
            );
            valid = false;
        }
        if valid && self.counts_fit(&converted, &ty.span) {
            Some(converted)
        } else {
            None
        }
    }

    /// A function's declared result type; a result of `i32` is E-TYPE-018 at its dtype.
    fn output(&mut self, f: &ast::Function) -> Option<TensorType> {
        if f.output.dtype.kind != ast::DType::F32 {
            self.report(
                codes::DTYPE,
                format!(
                    "`{}` must return f32; `i32` is for the ids of `embedding` and the labels of \
                     `cross_entropy` only",
                    f.name.name
                ),
                &f.output.dtype.span,
            );
            return None;
        }
        self.declared(&f.output)
    }

    /// Whether the element and byte counts of `ty` fit `u64`; E-TYPE-016 at `span` if not.
    fn counts_fit(&mut self, ty: &TensorType, span: &SourceSpan) -> bool {
        let fits = counts_fit_u64(ty);
        if !fits {
            self.report(
                codes::COUNT_OVERFLOW,
                format!("element or byte count of {ty} overflows u64"),
                span,
            );
        }
        fits
    }

    /// A parameter: its declared type, which its uses see whenever the type itself is valid, and
    /// its HIR, only when its initializer is valid as well. An initializer error is independent
    /// of the parameter's uses, so it does not hide their diagnostics.
    fn param(&mut self, model: &str, p: &ast::Param) -> (Option<TensorType>, Option<HirParam>) {
        if p.ty.dtype.kind != ast::DType::F32 {
            self.report(
                codes::DTYPE,
                format!(
                    "parameter `{}` must be f32; `i32` is for the ids of `embedding` and the labels \
                     of `cross_entropy` only",
                    p.name.name
                ),
                &p.ty.dtype.span,
            );
            return (None, None);
        }
        let ty = self.declared(&p.ty);
        let call = &p.init;
        let init = Initializer::from_name(&call.callee.name).filter(|_| call.args.is_empty());
        let Some(init) = init else {
            self.report(
                codes::INITIALIZER,
                format!(
                    "initializer `{}(...)` is not supported; the initializers are \
                     `xavier_uniform()`, `zeros()` and `ones()`",
                    call.callee.name
                ),
                &call.span,
            );
            return (ty, None);
        };
        let Some(ty) = ty else {
            return (None, None);
        };
        let (fits, rule, other) = match init {
            Initializer::XavierUniform => (
                ty.rank() == 2,
                "needs a rank-2 parameter",
                "`zeros()` and `ones()` take a parameter of any rank from 1",
            ),
            Initializer::Zeros | Initializer::Ones => (
                ty.rank() >= 1,
                "needs a parameter of rank 1 or more",
                "`xavier_uniform()` needs rank 2",
            ),
        };
        if !fits {
            self.diagnostics.push(
                Diagnostic::new(
                    codes::INITIALIZER_RANK,
                    format!("`{init}()` {rule}; `{}` has type {ty}", p.name.name),
                )
                .at(call.span.clone())
                .note(other),
            );
            return (Some(ty), None);
        }
        let param = HirParam {
            name: p.name.name.clone(),
            path: format!("{model}.{}", p.name.name),
            ty: ty.clone(),
            init,
            span: p.span.clone(),
        };
        (Some(ty), Some(param))
    }

    /// The checked function, and its dataflow for the calls to it ([`Flow`]), taken from the
    /// parts that checked when the function itself does not.
    fn function(
        &mut self,
        f: &ast::Function,
        sig: &Signature,
        names: &ModelNames,
        params: &[Option<TensorType>],
    ) -> (Option<HirFunction>, Flow) {
        let mut scope = Scope::function(names, &f.name);
        for (i, arg) in f.params.iter().enumerate() {
            scope.bind_local(&arg.name, Binding::Arg(ArgRef(i)), &mut self.diagnostics);
        }
        let mut let_types: Vec<Option<TensorType>> = Vec::new();
        let mut let_reads: Vec<Reads> = Vec::new();
        let mut flow = Flow::default();
        let mut lets = Vec::new();
        for (i, binding) in f.lets.iter().enumerate() {
            let env = Env {
                scope: &scope,
                params,
                args: &sig.args,
                lets: &let_types,
                let_reads: &let_reads,
                flows: &[],
                data: &[],
                in_train: false,
                model: None,
            };
            let value = self.expr(&binding.value, &env);
            let_types.push(value.as_ref().map(|v| v.ty.clone()));
            let read = value.as_ref().map_or_else(Reads::default, |v| {
                constant_args(v, &let_reads, &mut flow.constant);
                reads(v, &let_reads, &[])
            });
            let_reads.push(read);
            scope.bind_local(
                &binding.name,
                Binding::Let(LetRef(i)),
                &mut self.diagnostics,
            );
            lets.push(value.map(|value| HirLet {
                name: binding.name.name.clone(),
                value,
                span: binding.span.clone(),
            }));
        }
        let env = Env {
            scope: &scope,
            params,
            args: &sig.args,
            lets: &let_types,
            let_reads: &let_reads,
            flows: &[],
            data: &[],
            in_train: false,
            model: None,
        };
        let result = self.expr(&f.result.value, &env);
        if let Some(result) = &result {
            constant_args(result, &let_reads, &mut flow.constant);
            flow.result = reads(result, &let_reads, &[]);
        }
        let result = match (result, &sig.output) {
            (Some(result), Some(output)) if result.ty != *output => {
                self.report(
                    codes::RETURN_TYPE,
                    format!(
                        "`{}` returns {} but is declared to return {output}",
                        f.name.name, result.ty
                    ),
                    &result.span,
                );
                None
            }
            (result, _) => result,
        };
        let args = f
            .params
            .iter()
            .zip(&sig.args)
            .map(|(arg, ty)| {
                ty.clone().map(|ty| HirArg {
                    name: arg.name.name.clone(),
                    ty,
                    span: arg.span.clone(),
                })
            })
            .collect::<Option<Vec<_>>>();
        let function = (|| {
            Some(HirFunction {
                name: f.name.name.clone(),
                args: args?,
                output: sig.output.clone()?,
                lets: lets.into_iter().collect::<Option<_>>()?,
                result: result?,
                span: f.span.clone(),
            })
        })();
        (function, flow)
    }

    fn train(
        &mut self,
        t: &ast::Train,
        names: &ModelNames,
        signatures: &[Signature],
        flows: &[Flow],
    ) -> Option<HirTrain> {
        let model_known = resolve::resolve_train_model(t, names, &mut self.diagnostics);
        let mut scope = Scope::train(t);
        let mut data_types = Vec::new();
        for (i, decl) in t.data.iter().enumerate() {
            data_types.push(self.declared(&decl.ty));
            scope.bind_data(&decl.name, Binding::Data(DataRef(i)), &mut self.diagnostics);
        }
        let env = Env {
            scope: &scope,
            params: &[],
            args: &[],
            lets: &[],
            let_reads: &[],
            flows,
            data: &data_types,
            in_train: true,
            model: model_known.then_some((names, signatures)),
        };
        let loss = self.expr(&t.loss.value, &env);
        let loss = match loss {
            Some(loss) if loss.ty.rank() != 0 => {
                self.report(
                    codes::LOSS_RANK,
                    format!("loss has type {}; it must be rank 0", loss.ty),
                    &loss.span,
                );
                None
            }
            Some(loss) => self.forward_called_once(loss, &t.name.name, names),
            None => None,
        };

        let (lr, optimizer) = match &t.optimizer.value {
            ast::Optimizer::Sgd { lr, lr_span } => {
                (self.learning_rate(*lr, lr_span), Some(OptimizerRule::Sgd))
            }
            ast::Optimizer::AdamW { args, span } => self.adamw(args, span, t.steps.value),
            ast::Optimizer::Muon { args, span } => self.muon(args, span, t.steps.value),
        };

        let steps = if t.steps.value == 0 {
            self.report(
                codes::STEPS,
                "`steps` must be positive; got 0",
                &t.steps.span,
            );
            None
        } else {
            Some(t.steps.value)
        };

        let data = t
            .data
            .iter()
            .zip(&data_types)
            .map(|(decl, ty)| {
                ty.clone().map(|ty| HirData {
                    role: decl.role,
                    name: decl.name.name.clone(),
                    ty,
                    span: decl.span.clone(),
                })
            })
            .collect::<Option<Vec<_>>>();
        let device = match t.device.value {
            ast::Device::Cpu => Device::Cpu,
            ast::Device::Vulkan { index, .. } => Device::Vulkan(index),
        };
        if !model_known {
            return None;
        }
        Some(HirTrain {
            name: t.name.name.clone(),
            model: names.name.clone(),
            data: data?,
            loss: loss?,
            lr: lr?,
            optimizer: optimizer?,
            steps: steps?,
            seed: t.seed.value,
            device,
            span: t.span.clone(),
        })
    }

    /// The learning rate `lr` of `sgd` or `adamw`, written at `span`: a positive finite `f32`,
    /// E-TYPE-010 otherwise.
    fn learning_rate(&mut self, lr: f64, span: &SourceSpan) -> Option<f32> {
        let rounded = lr as f32;
        if rounded.is_finite() && rounded > 0.0 {
            return Some(rounded);
        }
        self.report(
            codes::LEARNING_RATE,
            format!("learning rate `{lr:?}` is not a positive finite f32"),
            span,
        );
        None
    }

    /// The literal value of each of `names`, the named arguments of the optimizer `optimizer`
    /// written as `args` at `span`, and whether they were written well (DESIGN.md § 2.1). It takes
    /// each of `names` exactly once, named, in any order: an unnamed, unknown or repeated argument
    /// is E-TYPE-002 at it, an omitted one E-TYPE-001 at the optimizer's name, and a value other
    /// than a float literal (an integer literal for the names whose `integer` entry is `true`)
    /// E-TYPE-019 at its argument. Every such argument is reported.
    fn optimizer_arguments<'a>(
        &mut self,
        optimizer: &str,
        names: &[&str],
        integer: &[bool],
        args: &'a [ast::Arg],
        span: &SourceSpan,
    ) -> (Vec<Option<Literal<'a>>>, bool) {
        let takes = || {
            let names: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
            names.join(", ")
        };
        let kind = |index: usize| match integer[index] {
            true => "an integer literal",
            false => "a float literal",
        };
        let mut given: Vec<Option<&ast::Arg>> = vec![None; names.len()];
        let mut valid = true;
        for arg in args {
            let Some(name) = &arg.name else {
                self.report(
                    codes::NAMED_ARGUMENT,
                    format!(
                        "an unnamed argument; `{optimizer}` takes the named arguments {}",
                        takes()
                    ),
                    &arg.span,
                );
                valid = false;
                continue;
            };
            let Some(index) = names.iter().position(|n| *n == name.name) else {
                self.report(
                    codes::NAMED_ARGUMENT,
                    format!(
                        "named argument `{}`; `{optimizer}` takes the named arguments {}",
                        name.name,
                        takes()
                    ),
                    &arg.span,
                );
                valid = false;
                continue;
            };
            if given[index].is_some() {
                self.report(
                    codes::NAMED_ARGUMENT,
                    format!(
                        "named argument `{}` of `{optimizer}` is given twice",
                        name.name
                    ),
                    &arg.span,
                );
                valid = false;
                continue;
            }
            given[index] = Some(arg);
        }
        let missing: Vec<String> = names
            .iter()
            .enumerate()
            .filter(|(index, _)| given[*index].is_none())
            .map(|(index, name)| format!("`{name}` ({})", kind(index)))
            .collect();
        if !missing.is_empty() {
            self.report(
                codes::ARGUMENT_COUNT,
                format!(
                    "`{optimizer}` needs the named argument(s) {}",
                    missing.join(", ")
                ),
                span,
            );
            valid = false;
        }

        let mut values = vec![None; names.len()];
        for (index, arg) in given.iter().enumerate() {
            let Some(arg) = arg else { continue };
            match (integer[index], &arg.value.kind) {
                (true, ast::ExprKind::Int(value)) => {
                    values[index] = Some(Literal::Int(*value, &arg.span))
                }
                (false, ast::ExprKind::Float(value)) => {
                    values[index] = Some(Literal::Float(*value, &arg.span))
                }
                _ => {
                    self.report(
                        codes::NAMED_VALUE,
                        format!("`{}` of `{optimizer}` takes {}", names[index], kind(index)),
                        &arg.span,
                    );
                    valid = false;
                }
            }
        }
        (values, valid)
    }

    /// The float argument `name` of `optimizer`, `value` as written, once rounded to `f32`, when
    /// `holds` of it; `code` at the argument, saying it must be `rule`, otherwise.
    fn hyperparameter(
        &mut self,
        code: Code,
        optimizer: &str,
        name: &str,
        value: Option<Literal<'_>>,
        rule: &str,
        holds: fn(f32) -> bool,
    ) -> Option<f32> {
        let Some(Literal::Float(value, at)) = value else {
            return None;
        };
        let rounded = value as f32;
        if holds(rounded) {
            return Some(rounded);
        }
        self.report(
            code,
            format!("`{name}` of `{optimizer}` must be {rule}; got {value:?}"),
            at,
        );
        None
    }

    /// `warmup_steps` of `optimizer`, as written, when below `steps`; E-TYPE-024 at it otherwise,
    /// unless `steps` is 0, which is E-TYPE-011 alone.
    fn warmup_steps(
        &mut self,
        optimizer: &str,
        value: Option<Literal<'_>>,
        steps: u64,
    ) -> Option<u64> {
        let Some(Literal::Int(value, at)) = value else {
            return None;
        };
        if steps > 0 && value >= steps {
            self.report(
                codes::WARMUP_STEPS,
                format!(
                    "`warmup_steps` of `{optimizer}` is {value}; it must be below `steps`, \
                     {steps}, so the learning rate has steps to decay over"
                ),
                at,
            );
            return None;
        }
        Some(value)
    }

    /// The learning rate and the rule of `adamw(args)`, whose name is at `span`, in a train block
    /// of `steps` steps (DESIGN.md § 2.1, coordinator decision D50). It takes each of
    /// [`ADAMW_ARGUMENTS`] exactly once, named, in any order ([`Self::optimizer_arguments`]).
    /// `lr` that is not a positive finite `f32` is E-TYPE-010; `weight_decay` below 0 or not
    /// finite, a beta outside `[0, 1)` once rounded to `f32`, and `eps` that is not a positive
    /// finite `f32` are E-TYPE-023; `warmup_steps` not below `steps` is E-TYPE-024, unless
    /// `steps` is 0, which is E-TYPE-011 alone. Every such argument is reported.
    fn adamw(
        &mut self,
        args: &[ast::Arg],
        span: &SourceSpan,
        steps: u64,
    ) -> (Option<f32>, Option<OptimizerRule>) {
        let integer = [false, false, false, false, false, true];
        let (values, valid) =
            self.optimizer_arguments("adamw", &ADAMW_ARGUMENTS, &integer, args, span);
        let lr = self.optimizer_rate(values[0]);
        let adamw = self.adamw_hyperparameters(
            "adamw",
            [values[1], values[2], values[3], values[4], values[5]],
            steps,
        );
        let rule = adamw.filter(|_| valid).map(OptimizerRule::AdamW);
        (lr, rule)
    }

    /// The learning rate and the rule of `muon(args)`, whose name is at `span`, in a train block
    /// of `steps` steps (DESIGN.md § 2.1, coordinator decision D67). It takes each of
    /// [`MUON_ARGUMENTS`] exactly once, named, in any order ([`Self::optimizer_arguments`]).
    /// `lr` and `adam_lr` that are not positive finite `f32`s are E-TYPE-010; `weight_decay`, the
    /// betas and `eps` are E-TYPE-023 as for `adamw`, and `warmup_steps` E-TYPE-024; `momentum`
    /// outside `[0, 1)` once rounded to `f32` and `ns_steps` outside `[1, 10]` are E-TYPE-025.
    /// Every such argument is reported.
    fn muon(
        &mut self,
        args: &[ast::Arg],
        span: &SourceSpan,
        steps: u64,
    ) -> (Option<f32>, Option<OptimizerRule>) {
        let integer = [false, false, false, true, true, false, false, false, false];
        let (values, valid) =
            self.optimizer_arguments("muon", &MUON_ARGUMENTS, &integer, args, span);
        let lr = self.optimizer_rate(values[0]);
        let momentum = self.hyperparameter(
            codes::MUON_HYPERPARAMETER,
            "muon",
            "momentum",
            values[1],
            "in [0, 1)",
            |x| (0.0..1.0).contains(&x),
        );
        let ns_steps = match values[3] {
            Some(Literal::Int(value, at)) => match u32::try_from(value) {
                Ok(n) if (1..=MUON_MAX_NS_STEPS).contains(&n) => Some(n),
                _ => {
                    self.report(
                        codes::MUON_HYPERPARAMETER,
                        format!(
                            "`ns_steps` of `muon` must be in [1, {MUON_MAX_NS_STEPS}]; got {value}"
                        ),
                        at,
                    );
                    None
                }
            },
            _ => None,
        };
        let adam_lr = self.optimizer_rate(values[5]);
        let adamw = self.adamw_hyperparameters(
            "muon",
            [values[2], values[6], values[7], values[8], values[4]],
            steps,
        );
        let rule = match (momentum, ns_steps, adam_lr, adamw) {
            (Some(momentum), Some(ns_steps), Some(adam_lr), Some(adamw)) if valid => {
                Some(OptimizerRule::Muon(Muon {
                    momentum,
                    ns_steps,
                    adam_lr,
                    adamw,
                }))
            }
            _ => None,
        };
        (lr, rule)
    }

    /// A learning rate argument as [`Self::learning_rate`] checks it.
    fn optimizer_rate(&mut self, value: Option<Literal<'_>>) -> Option<f32> {
        match value {
            Some(Literal::Float(value, at)) => self.learning_rate(value, at),
            _ => None,
        }
    }

    /// The AdamW hyperparameters of `optimizer` from its `weight_decay`, `beta1`, `beta2`, `eps`
    /// and `warmup_steps` arguments, in that order: E-TYPE-023 for the first four out of range,
    /// E-TYPE-024 for the last.
    fn adamw_hyperparameters(
        &mut self,
        optimizer: &str,
        [weight_decay, beta1, beta2, eps, warmup]: [Option<Literal<'_>>; 5],
        steps: u64,
    ) -> Option<AdamW> {
        let code = codes::OPTIMIZER_HYPERPARAMETER;
        let weight_decay = self.hyperparameter(
            code,
            optimizer,
            "weight_decay",
            weight_decay,
            "a finite f32 of at least 0",
            |x| x.is_finite() && x >= 0.0,
        );
        let beta = |x: f32| (0.0..1.0).contains(&x);
        let beta1 = self.hyperparameter(code, optimizer, "beta1", beta1, "in [0, 1)", beta);
        let beta2 = self.hyperparameter(code, optimizer, "beta2", beta2, "in [0, 1)", beta);
        let eps = self.hyperparameter(code, optimizer, "eps", eps, "a positive finite f32", |x| {
            x.is_finite() && x > 0.0
        });
        let warmup_steps = self.warmup_steps(optimizer, warmup, steps);
        Some(AdamW {
            weight_decay: weight_decay?,
            beta1: beta1?,
            beta2: beta2?,
            eps: eps?,
            warmup_steps: warmup_steps?,
        })
    }

    /// E-TYPE-017 unless the typed, rank-0 `loss` calls `model.forward` exactly once. No call is
    /// reported at the loss; a second call at the second one in source order.
    fn forward_called_once(
        &mut self,
        loss: HirExpr,
        train: &str,
        names: &ModelNames,
    ) -> Option<HirExpr> {
        let mut calls = Vec::new();
        if let Some(forward) = names.function("forward") {
            forward_calls(&loss, forward, &mut calls);
        }
        calls.sort_by_key(|span| (span.line, span.column));
        match calls.as_slice() {
            [_] => Some(loss),
            [] => {
                let reason = if names.function("forward").is_some() {
                    "it does not call it".to_owned()
                } else {
                    format!("model `{}` has no function `forward`", names.name)
                };
                self.report(
                    codes::FORWARD_CALLS,
                    format!(
                        "the loss of train block `{train}` must call `model.forward` exactly \
                         once; {reason}"
                    ),
                    &loss.span,
                );
                None
            }
            [_, second, ..] => {
                let second = second.clone();
                self.report(
                    codes::FORWARD_CALLS,
                    format!(
                        "the loss of train block `{train}` must call `model.forward` exactly \
                         once; it calls it {} times",
                        calls.len()
                    ),
                    &second,
                );
                None
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // Expressions.
    // -----------------------------------------------------------------------------------------

    fn expr(&mut self, e: &ast::Expr, env: &Env<'_>) -> Option<HirExpr> {
        let span = e.span.clone();
        match &e.kind {
            ast::ExprKind::Var(name) => {
                let binding = env.scope.lookup(name, &e.span, &mut self.diagnostics)?;
                let (kind, ty) = match binding {
                    Binding::Param(r) => (HirExprKind::Param(r), &env.params[r.0]),
                    Binding::Arg(r) => (HirExprKind::Arg(r), &env.args[r.0]),
                    Binding::Let(r) => (HirExprKind::Let(r), &env.lets[r.0]),
                    Binding::Data(r) => (HirExprKind::Data(r), &env.data[r.0]),
                };
                Some(HirExpr {
                    kind,
                    ty: ty.clone()?,
                    span,
                })
            }
            ast::ExprKind::Int(value) => {
                self.report(
                    codes::INTEGER_LITERAL,
                    format!(
                        "integer literal `{value}` where a tensor is expected; write `{value}.0`"
                    ),
                    &e.span,
                );
                None
            }
            ast::ExprKind::Float(value) => {
                let scalar = *value as f32;
                if !scalar.is_finite() {
                    self.report(
                        codes::FLOAT_LITERAL_RANGE,
                        format!("float literal `{value:?}` is not a finite f32"),
                        &e.span,
                    );
                    return None;
                }
                Some(HirExpr {
                    kind: HirExprKind::Scalar(scalar),
                    ty: TensorType::f32_scalar(),
                    span,
                })
            }
            ast::ExprKind::Call(call) => self.op_call(call, env),
            ast::ExprKind::MemberCall(call) => self.member_call(call, env),
        }
    }

    /// The arguments of a call, each typed; `None` if any is named (E-TYPE-002) or fails.
    fn args(&mut self, callee: &str, args: &[ast::Arg], env: &Env<'_>) -> Option<Vec<HirExpr>> {
        let mut typed = Vec::new();
        let mut valid = true;
        for arg in args {
            if let Some(name) = &arg.name {
                self.report(
                    codes::NAMED_ARGUMENT,
                    format!(
                        "named argument `{}`; `{callee}` takes positional arguments",
                        name.name
                    ),
                    &arg.span,
                );
                valid = false;
            }
            match self.expr(&arg.value, env) {
                Some(value) => typed.push(value),
                None => valid = false,
            }
        }
        valid.then_some(typed)
    }

    /// The operands and named compile-time arguments of a call to `op`, which takes the named
    /// arguments `named` (DESIGN.md § 2.1). A positional argument is an operand; a named one
    /// before an operand, that `op` does not take, or that the call gives twice, is E-TYPE-002
    /// at the argument; a
    /// value other than a literal of the parameter's kind is E-TYPE-019 at the argument and is
    /// not typed; a named argument the call omits is E-TYPE-001 at the call. The values come
    /// back in the order of `named`.
    fn named_args(
        &mut self,
        call: &ast::Call,
        named: &[ops::NamedParameter],
        env: &Env<'_>,
    ) -> Option<(Vec<HirExpr>, Vec<ops::NamedValue>)> {
        let callee = &call.callee.name;
        let mut operands = Vec::new();
        let mut values: Vec<Option<ops::NamedValue>> = vec![None; named.len()];
        let mut given = vec![false; named.len()];
        let mut valid = true;
        for (at, arg) in call.args.iter().enumerate() {
            let Some(name) = &arg.name else {
                match self.expr(&arg.value, env) {
                    Some(value) => operands.push(value),
                    None => valid = false,
                }
                continue;
            };
            let position = named.iter().position(|p| p.name == name.name);
            if call.args[at + 1..].iter().any(|a| a.name.is_none()) {
                self.report(
                    codes::NAMED_ARGUMENT,
                    format!(
                        "named argument `{}` of `{callee}` comes before an operand; named \
                         arguments follow the operands",
                        name.name
                    ),
                    &arg.span,
                );
                if let Some(index) = position {
                    given[index] = true;
                }
                valid = false;
                continue;
            }
            let Some(index) = position else {
                let takes: Vec<String> = named.iter().map(|p| format!("`{}`", p.name)).collect();
                self.report(
                    codes::NAMED_ARGUMENT,
                    format!(
                        "named argument `{}`; `{callee}` takes the named argument(s) {}",
                        name.name,
                        takes.join(", ")
                    ),
                    &arg.span,
                );
                valid = false;
                continue;
            };
            if std::mem::replace(&mut given[index], true) {
                self.report(
                    codes::NAMED_ARGUMENT,
                    format!(
                        "named argument `{}` of `{callee}` is given twice",
                        name.name
                    ),
                    &arg.span,
                );
                valid = false;
                continue;
            }
            let parameter = named[index];
            let value = match arg.value.kind {
                ast::ExprKind::Int(literal) => parameter.kind.integer(literal),
                ast::ExprKind::Float(literal) if parameter.kind == ops::NamedKind::Probability => {
                    let value = parameter.kind.float(literal);
                    if value.is_none() {
                        self.report(
                            codes::DROPOUT_RATE,
                            format!(
                                "`{}` of `{callee}` must be in [0, 1); got {literal:?}",
                                parameter.name
                            ),
                            &arg.span,
                        );
                        valid = false;
                        continue;
                    }
                    value
                }
                _ => None,
            };
            if value.is_none() {
                self.report(
                    codes::NAMED_VALUE,
                    format!(
                        "`{}` of `{callee}` takes {}",
                        parameter.name,
                        parameter.kind.describe()
                    ),
                    &arg.span,
                );
                valid = false;
            }
            values[index] = value;
        }
        let missing: Vec<String> = named
            .iter()
            .zip(&given)
            .filter(|(_, given)| !**given)
            .map(|(p, _)| format!("`{}` ({})", p.name, p.kind.describe()))
            .collect();
        if !missing.is_empty() {
            self.report(
                codes::ARGUMENT_COUNT,
                format!(
                    "`{callee}` needs the named argument(s) {}",
                    missing.join(", ")
                ),
                &call.span,
            );
            valid = false;
        }
        if !valid {
            return None;
        }
        Some((operands, values.into_iter().collect::<Option<_>>()?))
    }

    fn op_call(&mut self, call: &ast::Call, env: &Env<'_>) -> Option<HirExpr> {
        let op = resolve::resolve_operator(&call.callee, &mut self.diagnostics);
        let named = op.as_ref().map_or(&[][..], ops::named_parameters);
        let args = if named.is_empty() {
            self.args(&call.callee.name, &call.args, env)
                .map(|args| (args, Vec::new()))
        } else {
            self.named_args(call, named, env)
        };
        let (op, (args, values)) = (op?, args?);
        let op = ops::with_named(&op, &values);
        let mut types: Vec<&TensorType> = args.iter().map(|a| &a.ty).collect();
        let token = ops::token_type();
        if let Some(index) = ops::token_operand(&op) {
            let expected = ops::source_arity(&op);
            if types.len() != expected {
                self.report(
                    codes::ARGUMENT_COUNT,
                    format!(
                        "`{}` takes {expected} argument(s); got {}",
                        call.callee.name,
                        types.len()
                    ),
                    &call.span,
                );
                return None;
            }
            types.insert(index, &token);
        }
        match ops::infer(&op, &types, None) {
            Ok(ty) => {
                if !self.counts_fit(&ty, &call.span) {
                    return None;
                }
                if let Some(derived) = ops::derived_types(&op, &types)
                    .into_iter()
                    .find(|derived| !counts_fit_u64(derived))
                {
                    let argument = call.args.iter().rev().find(|a| a.name.is_some());
                    let (name, span) = argument.map_or(("", &call.span), |a| {
                        (a.name.as_ref().map_or("", |n| n.name.as_str()), &a.span)
                    });
                    self.report(
                        codes::COUNT_OVERFLOW,
                        format!(
                            "`{name}` of `{}` makes it compute {derived}, whose element or byte \
                             count overflows u64",
                            call.callee.name
                        ),
                        span,
                    );
                    return None;
                }
                for &(index, operand) in ops::constant_operands(&op) {
                    if reads(&args[index], env.let_reads, env.flows).param {
                        self.report(
                            codes::CONSTANT_OPERAND,
                            format!(
                                "`{operand}` of `{}` reads a model parameter; it takes no \
                                 gradient, so it must not depend on a `param`",
                                call.callee.name
                            ),
                            &args[index].span,
                        );
                        return None;
                    }
                }
                Some(HirExpr {
                    kind: HirExprKind::Op { op, args },
                    ty,
                    span: call.span.clone(),
                })
            }
            Err(error) => {
                self.diagnostics
                    .push(op_diagnostic(&call.callee.name, error, &types, &call.span));
                None
            }
        }
    }

    fn member_call(&mut self, call: &ast::MemberCall, env: &Env<'_>) -> Option<HirExpr> {
        let function = resolve::resolve_method(
            call,
            env.in_train,
            env.model.map(|(names, _)| names),
            &mut self.diagnostics,
        );
        let args = self.args(&call.method.name, &call.args, env);
        let function = function?;
        let (_, signatures) = env.model?;
        let sig = &signatures[function.0];
        if call.args.len() != sig.args.len() {
            self.report(
                codes::ARGUMENT_COUNT,
                format!(
                    "`{}` takes {} argument(s); got {}",
                    sig.name,
                    sig.args.len(),
                    call.args.len()
                ),
                &call.span,
            );
            return None;
        }
        let args = args?;
        let mut valid = true;
        for (i, (arg, want)) in args.iter().zip(&sig.args).enumerate() {
            match want {
                Some(want) if arg.ty != *want => {
                    self.report(
                        codes::ARGUMENT_TYPE,
                        format!(
                            "argument {} of `{}` has type {}; the function takes {want}",
                            i + 1,
                            sig.name,
                            arg.ty
                        ),
                        &arg.span,
                    );
                    valid = false;
                }
                Some(_) => {}
                None => valid = false,
            }
        }
        if !valid {
            return None;
        }
        let constant = env.flows.get(function.0).map(|flow| &flow.constant);
        for (&index, operand) in constant.into_iter().flatten() {
            if reads(&args[index], env.let_reads, env.flows).param {
                self.report(
                    codes::CONSTANT_OPERAND,
                    format!(
                        "argument {} of `{}` reads a model parameter and reaches {operand} in \
                         it; that operand takes no gradient, so it must not depend on a `param`",
                        index + 1,
                        sig.name
                    ),
                    &args[index].span,
                );
                valid = false;
            }
        }
        if !valid {
            return None;
        }
        Some(HirExpr {
            kind: HirExprKind::Call { function, args },
            ty: sig.output.clone()?,
            span: call.span.clone(),
        })
    }
}

/// What a value reads by dataflow (coordinator decision D41): whether a model parameter reaches
/// it, and which arguments of the enclosing model function do.
#[derive(Clone, Debug, Default, PartialEq)]
struct Reads {
    param: bool,
    args: BTreeSet<usize>,
}

impl Reads {
    fn join(mut self, other: Reads) -> Reads {
        self.param |= other.param;
        self.args.extend(other.args);
        self
    }
}

/// The dataflow of a model function that a call to it needs (D41): what its result reads, and
/// which of its arguments reach an operand that takes no gradient, each with that operand's
/// name for the message (`` `key_mask` of `attention` ``).
#[derive(Clone, Debug, Default)]
struct Flow {
    result: Reads,
    constant: BTreeMap<usize, String>,
}

/// What `expr` reads by dataflow: a `param` reads a parameter and an argument itself; a `let`
/// reads what its value reads (`let_reads`); an operator reads what any operand reads; a
/// `model.<function>(...)` call reads what the function's result reads, with each argument of
/// the function that reaches its result replaced by what the call's argument reads (`flows`).
fn reads(expr: &HirExpr, let_reads: &[Reads], flows: &[Flow]) -> Reads {
    match &expr.kind {
        HirExprKind::Param(_) => Reads {
            param: true,
            args: BTreeSet::new(),
        },
        HirExprKind::Arg(r) => Reads {
            param: false,
            args: BTreeSet::from([r.0]),
        },
        HirExprKind::Let(r) => let_reads.get(r.0).cloned().unwrap_or_default(),
        HirExprKind::Op { args, .. } => args
            .iter()
            .map(|a| reads(a, let_reads, flows))
            .fold(Reads::default(), Reads::join),
        HirExprKind::Call { function, args } => {
            let Some(flow) = flows.get(function.0) else {
                return Reads::default();
            };
            let through = flow.result.args.iter().filter_map(|&i| args.get(i));
            through.map(|a| reads(a, let_reads, flows)).fold(
                Reads {
                    param: flow.result.param,
                    args: BTreeSet::new(),
                },
                Reads::join,
            )
        }
        HirExprKind::Data(_) | HirExprKind::Scalar(_) => Reads::default(),
    }
}

/// Adds to `out` each argument of the enclosing model function that reaches an operand of an
/// operator in `expr` that takes no gradient ([`ops::constant_operands`]), with that operand's
/// name; the first such operand an argument reaches names it.
fn constant_args(expr: &HirExpr, let_reads: &[Reads], out: &mut BTreeMap<usize, String>) {
    let HirExprKind::Op { op, args } = &expr.kind else {
        return;
    };
    for &(index, operand) in ops::constant_operands(op) {
        let Some(arg) = args.get(index) else {
            continue;
        };
        for i in reads(arg, let_reads, &[]).args {
            out.entry(i)
                .or_insert_with(|| format!("`{operand}` of `{}`", source_name(op)));
        }
    }
    for arg in args {
        constant_args(arg, let_reads, out);
    }
}

/// A model parameter or an argument of the enclosing model function, as a value's source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Source {
    Param(usize),
    Arg(usize),
}

/// What a value reads by dataflow, split by whether a gradient flows back along the read:
/// `live` reads carry one, `stopped` reads pass through a `stop_gradient` on every path that
/// reaches them from this value, each at the span of one such call.
#[derive(Clone, Debug, Default)]
struct GradientReads {
    live: BTreeSet<Source>,
    stopped: BTreeMap<Source, SourceSpan>,
}

impl GradientReads {
    /// Both reads; a source stopped on both sides keeps the earlier span in source order.
    fn join(mut self, other: GradientReads) -> GradientReads {
        self.live.extend(other.live);
        for (source, span) in other.stopped {
            match self.stopped.get(&source) {
                Some(kept) if (kept.line, kept.column) <= (span.line, span.column) => {}
                _ => {
                    self.stopped.insert(source, span);
                }
            }
        }
        self
    }

    /// Every read, stopped at `span`: what a `stop_gradient` call at `span` reads.
    fn stopped_at(self, span: &SourceSpan) -> GradientReads {
        let stopped = self
            .live
            .into_iter()
            .chain(self.stopped.into_keys())
            .map(|source| (source, span.clone()))
            .collect();
        GradientReads {
            live: BTreeSet::new(),
            stopped,
        }
    }
}

/// What `expr` reads by dataflow, as [`GradientReads`]: a `param` and an argument read
/// themselves, live; a `let` reads what its value reads (`lets`); `stop_gradient` reads what its
/// operand reads, all stopped at the call; another operator reads what any operand reads; a
/// `model.<function>(...)` call reads the parameters its function's result reads (`summaries`),
/// and for each argument of the function that the result reads, what the call's argument reads,
/// stopped at the function's `stop_gradient` when the function stops it.
fn gradient_reads(
    expr: &HirExpr,
    lets: &[GradientReads],
    summaries: &[GradientReads],
) -> GradientReads {
    match &expr.kind {
        HirExprKind::Param(r) => GradientReads {
            live: BTreeSet::from([Source::Param(r.0)]),
            stopped: BTreeMap::new(),
        },
        HirExprKind::Arg(r) => GradientReads {
            live: BTreeSet::from([Source::Arg(r.0)]),
            stopped: BTreeMap::new(),
        },
        HirExprKind::Let(r) => lets.get(r.0).cloned().unwrap_or_default(),
        HirExprKind::Op {
            op: TensorOp::StopGradient,
            args,
        } => args
            .iter()
            .map(|a| gradient_reads(a, lets, summaries))
            .fold(GradientReads::default(), GradientReads::join)
            .stopped_at(&expr.span),
        HirExprKind::Op { args, .. } => args
            .iter()
            .map(|a| gradient_reads(a, lets, summaries))
            .fold(GradientReads::default(), GradientReads::join),
        HirExprKind::Call { function, args } => {
            let Some(summary) = summaries.get(function.0) else {
                return GradientReads::default();
            };
            let mut out = GradientReads {
                live: summary
                    .live
                    .iter()
                    .filter(|s| matches!(s, Source::Param(_)))
                    .copied()
                    .collect(),
                stopped: summary
                    .stopped
                    .iter()
                    .filter(|(s, _)| matches!(s, Source::Param(_)))
                    .map(|(s, span)| (*s, span.clone()))
                    .collect(),
            };
            for (index, arg) in args.iter().enumerate() {
                let source = Source::Arg(index);
                let live = summary.live.contains(&source);
                let stopped = summary.stopped.get(&source);
                if !live && stopped.is_none() {
                    continue;
                }
                let value = gradient_reads(arg, lets, summaries);
                if let Some(span) = stopped {
                    out = out.join(value.clone().stopped_at(span));
                }
                if live {
                    out = out.join(value);
                }
            }
            out
        }
        HirExprKind::Data(_) | HirExprKind::Scalar(_) => GradientReads::default(),
    }
}

/// The source name of `op`, `attention` for any `Attention { heads }`.
fn source_name(op: &TensorOp) -> &'static str {
    resolve::OPERATORS
        .iter()
        .find(|(_, o)| std::mem::discriminant(o) == std::mem::discriminant(op))
        .map_or("an operator", |(name, _)| name)
}

/// The `layer_norm` operand that [`ops::infer`] refuses first, in the order it checks them: `x`
/// of rank 2, then `gamma`, then `beta`, each of the dtype of `x`, of rank 1 and as long as a row
/// of `x`. `None` unless `types` holds the three operand types.
fn layer_norm_operand(types: &[&TensorType]) -> Option<&'static str> {
    let [x, gamma, beta] = types else {
        return None;
    };
    if x.rank() != 2 {
        return Some("x");
    }
    [("gamma", gamma), ("beta", beta)]
        .into_iter()
        .find(|(_, row)| row.dtype != x.dtype || row.rank() != 1 || row.shape[0] != x.shape[1])
        .map(|(operand, _)| operand)
}

/// The `cross_entropy` operand that [`ops::infer`] refuses first for its rank, in the order it
/// checks them: `logits` of rank 2, `labels` of rank 1, then `weights` of rank 1. `None` unless
/// `types` holds the three operand types and one of them has another rank.
fn cross_entropy_rank_operand(types: &[&TensorType]) -> Option<&'static str> {
    let [logits, labels, weights] = types else {
        return None;
    };
    [
        ("logits", logits, 2),
        ("labels", labels, 1),
        ("weights", weights, 1),
    ]
    .into_iter()
    .find(|(_, ty, rank)| ty.rank() != *rank)
    .map(|(operand, _, _)| operand)
}

/// The operands an E-TYPE-003 of `cross_entropy` names (adversary finding N1): when `labels` and
/// `weights` agree on `N` and `logits` has another row count, `logits` and its rows against
/// `N`; when `labels` and `weights` disagree with each other, both, beside the type of `logits`.
/// `None` unless `types` holds the three operand types, each of its rank, and a length differs.
fn cross_entropy_shape_operands(types: &[&TensorType]) -> Option<String> {
    let [logits, labels, weights] = types else {
        return None;
    };
    if cross_entropy_rank_operand(types).is_some() {
        return None;
    }
    let (rows, n) = (logits.shape[0], labels.shape[0]);
    if n != weights.shape[0] {
        return Some(format!(
            "`labels` has type {labels} and `weights` has type {weights} for `logits` of type \
             {logits}"
        ));
    }
    (n != rows).then(|| {
        format!("`logits` has type {logits}, {rows} rows against N = {n} of `labels` and `weights`")
    })
}

/// The user diagnostic for a registry error of the operator called `name` at `span`, its operands
/// of the types `types` (empty where they are not known).
fn op_diagnostic(
    name: OpName<'_>,
    error: OpError,
    types: &[&TensorType],
    span: &SourceSpan,
) -> Diagnostic {
    match error {
        OpError::Arity { expected, got, .. } => Diagnostic::new(
            codes::ARGUMENT_COUNT,
            format!("`{name}` takes {expected} argument(s); got {got}"),
        ),
        OpError::ShapeMismatch { lhs, rhs, .. } => {
            let rule = if name == "mse" {
                "`mse` needs operands of identical shape".to_owned()
            } else if name == "add_rows" {
                "`add_rows(x, b)` needs `b` of shape [H] for `x` of shape [N, H]".to_owned()
            } else if name == "attention" {
                "`attention(q, k, v, key_mask, heads = A)` needs `k` and `v` of one shape \
                 [B * S, H], `key_mask` of shape [B, S] and `q` of shape [B * L, H] with \
                 1 <= L <= S, or an `i32` spans operand of shape [U, 2] with `q` of the rows of `k` \
                 or of U rows"
                    .to_owned()
            } else if name == "cross_entropy" {
                let rule = "`cross_entropy(logits, labels, weights)` needs `labels` and `weights` \
                            of shape [N] for `logits` of shape [N, V]";
                match cross_entropy_shape_operands(types) {
                    Some(operands) => format!("{rule}; {operands}"),
                    None => rule.to_owned(),
                }
            } else if name == "layer_norm" {
                let rule = "`layer_norm(x, gamma, beta)` needs `gamma` and `beta` of shape [H] \
                            for `x` of shape [N, H]";
                match layer_norm_operand(types) {
                    Some(operand) => format!("{rule}; `{operand}` has type {rhs}"),
                    None => rule.to_owned(),
                }
            } else {
                "operands need the same shape, or one of them rank 0; M0 has no broadcasting"
                    .to_owned()
            };
            Diagnostic::new(
                codes::SHAPE_MISMATCH,
                format!("`{name}` operand shapes differ"),
            )
            .with("lhs", lhs.to_string())
            .with("rhs", rhs.to_string())
            .note(rule)
        }
        OpError::Rank { expected, got, .. } => {
            let operands = match name {
                "add_rows" => Some([("x", 2), ("b", 1)]),
                "embedding" => Some([("table", 2), ("ids", 1)]),
                _ => None,
            };
            let wrong = operands.and_then(|ops| ops.into_iter().find(|&(_, r)| r == expected));
            if name == "cross_entropy" {
                let rule =
                    "`cross_entropy` needs `logits` of rank 2 and `labels` and `weights` of rank 1";
                let message = match cross_entropy_rank_operand(types) {
                    Some(operand) => format!("{rule}; `{operand}` has type {got}"),
                    None => format!("{rule}; got {got}"),
                };
                return Diagnostic::new(codes::RANK, message).at(span.clone());
            }
            if name == "layer_norm" {
                let rule = "`layer_norm` needs `x` of rank 2 and `gamma` and `beta` of rank 1";
                let message = match layer_norm_operand(types) {
                    Some(operand) => format!("{rule}; `{operand}` has type {got}"),
                    None => format!("{rule}; got {got}"),
                };
                return Diagnostic::new(codes::RANK, message).at(span.clone());
            }
            match (operands, wrong) {
                (Some([(first, r1), (second, r2)]), Some((operand, _))) => Diagnostic::new(
                    codes::RANK,
                    format!(
                        "`{name}` needs `{first}` of rank {r1} and `{second}` of rank {r2}; \
                         `{operand}` has type {got}"
                    ),
                ),
                _ => Diagnostic::new(
                    codes::RANK,
                    format!("`{name}` needs rank-{expected} operands"),
                )
                .with("got", got.to_string()),
            }
        }
        OpError::MatMulContracting {
            lhs,
            rhs,
            expected,
            got,
        } => Diagnostic::new(
            codes::MATMUL_CONTRACTING,
            "matmul contracting dimension mismatch",
        )
        .with("lhs", lhs.to_string())
        .with("rhs", rhs.to_string())
        .note(format!("expected rhs dimension[0] = {expected}; got {got}")),
        OpError::HeadsDivide { heads, width, .. } => Diagnostic::new(
            codes::HEADS_DIVIDE,
            format!("`{name}` needs `heads` to divide the width {width} of `q`, `k` and `v`; got heads = {heads}"),
        )
        .note("each head reads an equal slice of the width: H / heads columns"),
        OpError::OperandDType {
            index,
            expected,
            got,
            ..
        } => Diagnostic::new(
            codes::DTYPE,
            format!("`{name}` operand {} must be {expected}", index + 1),
        )
        .with("got", got.to_string())
        .note(
            "`i32` tensors are row ids, class labels or key spans: `embedding` and `cross_entropy` read \
             them as their second operand, `attention` as its fourth, of shape [U, 2]",
        ),
        error @ (OpError::DTypeMismatch { .. }
        | OpError::MissingDeclaredType { .. }
        | OpError::ParamDType { .. }
        | OpError::DropoutRate { .. }
        | OpError::Token { .. }) => Diagnostic::new(
            codes::INTERNAL,
            format!("compiler verification failure in `{name}`: {error}"),
        ),
    }
    .at(span.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{DataRole, ParamRef};
    use crate::ir::tensor::{DType, TensorOp};

    const LINEAR_REGRESSION: &str = include_str!("../../../examples/linear_regression.mspec");

    fn span() -> SourceSpan {
        SourceSpan {
            file: "m.mspec".to_owned(),
            line: 3,
            column: 7,
        }
    }

    #[test]
    fn linear_regression_typed_hir() {
        let hir = check_source("linear_regression.mspec", LINEAR_REGRESSION).expect("checks");
        let model = &hir.model;
        assert_eq!(model.name, "LinearReg");
        assert_eq!(model.params.len(), 1);
        assert_eq!(model.params[0].path, "LinearReg.w");
        assert_eq!(model.params[0].ty, TensorType::f32(&[4, 1]));
        assert_eq!(model.params[0].init, Initializer::XavierUniform);

        let (_, forward) = model.function("forward").expect("forward");
        assert_eq!(forward.args.len(), 1);
        assert_eq!(forward.args[0].ty, TensorType::f32(&[8, 4]));
        assert_eq!(forward.output, TensorType::f32(&[8, 1]));
        match &forward.result.kind {
            HirExprKind::Op { op, args } => {
                assert_eq!(*op, TensorOp::MatMul);
                assert_eq!(args[0].kind, HirExprKind::Arg(ArgRef(0)));
                assert_eq!(args[1].kind, HirExprKind::Param(ParamRef(0)));
            }
            other => panic!("forward returns {other:?}"),
        }
        assert_eq!(forward.result.ty, forward.output);

        let train = &hir.train;
        assert_eq!(train.name, "Fit");
        assert_eq!(train.model, "LinearReg");
        let inputs: Vec<_> = train.inputs().map(|d| (&*d.name, &d.ty)).collect();
        let targets: Vec<_> = train.targets().map(|d| (&*d.name, &d.ty)).collect();
        assert_eq!(inputs, [("x", &TensorType::f32(&[8, 4]))]);
        assert_eq!(targets, [("y", &TensorType::f32(&[8, 1]))]);
        assert_eq!(train.data[1].role, DataRole::Target);
        assert_eq!(train.loss.ty, TensorType::f32_scalar());
        match &train.loss.kind {
            HirExprKind::Op { op, args } => {
                assert_eq!(*op, TensorOp::Mse);
                assert!(matches!(args[0].kind, HirExprKind::Call { .. }));
                assert_eq!(args[1].kind, HirExprKind::Data(DataRef(1)));
            }
            other => panic!("loss is {other:?}"),
        }
        assert_eq!(train.lr, 0.1f32);
        assert_eq!(train.steps, 500);
        assert_eq!(train.seed, 42);
        assert_eq!(train.device, Device::Cpu);
        assert_eq!(train.loss.ty.dtype, DType::F32);
    }

    #[test]
    fn every_source_operator_is_a_registry_op() {
        let names: Vec<&str> = resolve::OPERATORS.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            [
                "matmul",
                "add",
                "sub",
                "mul",
                "relu",
                "transpose",
                "sum",
                "mean",
                "mse",
                "embedding",
                "add_rows",
                "gelu",
                "layer_norm",
                "softmax",
                "attention",
                "cross_entropy",
                "dropout",
                "stop_gradient",
                "sigreg"
            ]
        );
    }

    /// A file whose model `M` has the table `t: [5, 4]` and the parameter `p: [6]`, the model
    /// functions `functions` and `forward(ids: i32[6]) -> [6, 5]`, the tied decoder; its train
    /// block declares `ids`, `labels` and `weights` of the types given and has the loss `loss`.
    fn cross_entropy_source(functions: &str, labels: &str, weights: &str, loss: &str) -> String {
        format!(
            "model M {{
    param t: Tensor<f32, [5, 4]> = xavier_uniform();
    param p: Tensor<f32, [6]> = ones();
    {functions}
    fn forward(ids: Tensor<i32, [6]>) -> Tensor<f32, [6, 5]> {{
        return matmul(embedding(t, ids), transpose(t));
    }}
}}
train T {{
    model = M;
    input ids: Tensor<i32, [6]>;
    target labels: {labels};
    target weights: {weights};
    loss = {loss};
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}}
"
        )
    }

    /// The codes `check` reports for `cross_entropy_source(...)`.
    fn cross_entropy_codes(
        functions: &str,
        labels: &str,
        weights: &str,
        loss: &str,
    ) -> Vec<String> {
        match check_source(
            "m.mspec",
            &cross_entropy_source(functions, labels, weights, loss),
        ) {
            Ok(_) => Vec::new(),
            Err(diagnostics) => diagnostics
                .iter()
                .map(|d| d.code.as_str().to_owned())
                .collect(),
        }
    }

    const LABELS: &str = "Tensor<i32, [6]>";
    const WEIGHTS: &str = "Tensor<f32, [6]>";
    const LOSS: &str = "cross_entropy(model.forward(ids), labels, weights)";

    /// `i32[6]` labels are the second permitted use of an `i32` tensor: the tied decoder checks
    /// and its loss is a rank-0 `CrossEntropy`.
    #[test]
    fn cross_entropy_over_i32_labels_checks() {
        let source = cross_entropy_source("", LABELS, WEIGHTS, LOSS);
        let hir = check_source("m.mspec", &source).expect("checks");
        assert!(matches!(
            &hir.train.loss.kind,
            HirExprKind::Op { op: TensorOp::CrossEntropy, args } if args.len() == 3
        ));
        assert_eq!(hir.train.loss.ty, TensorType::f32_scalar());
    }

    /// `f32` labels are E-TYPE-018, labels or weights of another length E-TYPE-003 and logits of
    /// rank 1 E-TYPE-004, each once.
    #[test]
    fn cross_entropy_operand_errors_are_e_type_018_003_and_004() {
        assert_eq!(
            cross_entropy_codes("", "Tensor<f32, [6]>", WEIGHTS, LOSS),
            ["E-TYPE-018"]
        );
        assert_eq!(
            cross_entropy_codes("", "Tensor<i32, [7]>", WEIGHTS, LOSS),
            ["E-TYPE-003"]
        );
        assert_eq!(
            cross_entropy_codes("", LABELS, "Tensor<f32, [5]>", LOSS),
            ["E-TYPE-003"]
        );
        assert_eq!(
            cross_entropy_codes(
                "",
                LABELS,
                WEIGHTS,
                "cross_entropy(sum(model.forward(ids)), labels, weights)"
            ),
            ["E-TYPE-004"]
        );
    }

    /// The one E-TYPE-003 diagnostic of `cross_entropy` with `labels`, `weights` and `loss`,
    /// rendered.
    fn cross_entropy_shape_diagnostic(labels: &str, weights: &str, loss: &str) -> String {
        let source = cross_entropy_source("", labels, weights, loss);
        let diagnostics = check_source("m.mspec", &source).expect_err(loss);
        assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
        assert_eq!(
            diagnostics[0].code,
            codes::SHAPE_MISMATCH,
            "{diagnostics:?}"
        );
        diagnostics[0].to_string()
    }

    /// An E-TYPE-003 of `cross_entropy` names the operand a reader can fix (adversary finding N1):
    /// `logits` with its row count when `labels` and `weights` agree on `N`, and both `labels`
    /// and `weights` when they disagree with each other.
    #[test]
    fn cross_entropy_shape_mismatch_names_the_operand_that_disagrees() {
        let transposed = "cross_entropy(transpose(model.forward(ids)), labels, weights)";
        let text = cross_entropy_shape_diagnostic(LABELS, WEIGHTS, transposed);
        assert!(
            text.contains(
                "`logits` has type Tensor<f32, [5, 6]>, 5 rows against N = 6 of `labels` and \
                 `weights`"
            ),
            "{text}"
        );
        assert!(!text.contains("`labels` has type"), "{text}");
        assert!(!text.contains("`weights` has type"), "{text}");

        for (labels, weights) in [
            ("Tensor<i32, [7]>", WEIGHTS),
            (LABELS, "Tensor<f32, [5]>"),
            ("Tensor<i32, [7]>", "Tensor<f32, [5]>"),
        ] {
            let text = cross_entropy_shape_diagnostic(labels, weights, LOSS);
            let want = format!(
                "`labels` has type {labels} and `weights` has type {weights} for `logits` of \
                 type Tensor<f32, [6, 5]>"
            );
            assert!(text.contains(&want), "{labels}, {weights}: {text}");
        }
    }

    /// The weights take no gradient: weights that read a parameter, directly in a model function
    /// or through the result of a model function call, are E-TYPE-021, its message naming
    /// `weights`; weights that pass through a parameter-free model function check.
    #[test]
    fn cross_entropy_weights_that_read_a_parameter_are_e_type_021() {
        let scaled = "fn scaled(a: Tensor<f32, [6]>) -> Tensor<f32, [6]> { return mul(a, p); }";
        let free = "fn free(a: Tensor<f32, [6]>) -> Tensor<f32, [6]> { return relu(a); }";
        let inner = "fn inner(z: Tensor<f32, [6, 5]>, l: Tensor<i32, [6]>, a: Tensor<f32, [6]>) -> Tensor<f32, []> { return cross_entropy(z, l, a); }";
        let functions = [scaled, free, inner].join(" ");
        let refused = [
            "cross_entropy(model.forward(ids), labels, model.scaled(weights))",
            "cross_entropy(model.forward(ids), labels, mul(weights, model.scaled(weights)))",
            "model.inner(model.forward(ids), labels, model.scaled(weights))",
        ];
        for loss in refused {
            let source = cross_entropy_source(&functions, LABELS, WEIGHTS, loss);
            let diagnostics = check_source("m.mspec", &source).expect_err(loss);
            assert_eq!(diagnostics.len(), 1, "{loss}: {diagnostics:?}");
            assert_eq!(diagnostics[0].code, codes::CONSTANT_OPERAND, "{loss}");
            assert!(
                diagnostics[0].message.contains("`weights`"),
                "{loss}: {}",
                diagnostics[0].message
            );
        }
        for loss in [
            "cross_entropy(model.forward(ids), labels, model.free(weights))",
            "model.inner(model.forward(ids), labels, weights)",
        ] {
            assert!(
                cross_entropy_codes(&functions, LABELS, WEIGHTS, loss).is_empty(),
                "{loss}"
            );
        }
    }

    /// `layer_norm` takes `x` of rank 2 and two rows of rank 1: its E-TYPE-004 message states
    /// each rank, and its E-TYPE-003 note states its rule, not `add`'s.
    #[test]
    fn layer_norm_rank_and_shape_errors_state_its_rule() {
        let error = OpError::Rank {
            op: "LayerNorm",
            expected: 1,
            got: TensorType::f32(&[1, 5]),
        };
        let diagnostic = op_diagnostic("layer_norm", error, &[], &span());
        assert_eq!(diagnostic.code, codes::RANK);
        assert_eq!(
            diagnostic.message,
            "`layer_norm` needs `x` of rank 2 and `gamma` and `beta` of rank 1; \
             got Tensor<f32, [1, 5]>"
        );
        let error = OpError::ShapeMismatch {
            op: "LayerNorm",
            lhs: TensorType::f32(&[4, 5]),
            rhs: TensorType::f32(&[6]),
        };
        let diagnostic = op_diagnostic("layer_norm", error, &[], &span());
        assert_eq!(diagnostic.code, codes::SHAPE_MISMATCH);
        let notes = format!("{diagnostic:?}");
        assert!(
            notes.contains("`layer_norm(x, gamma, beta)` needs `gamma` and `beta` of shape [H]"),
            "{notes}"
        );
        let error = OpError::Rank {
            op: "Softmax",
            expected: 2,
            got: TensorType::f32(&[3]),
        };
        let diagnostic = op_diagnostic("softmax", error, &[], &span());
        assert_eq!(diagnostic.code, codes::RANK);
        assert_eq!(diagnostic.message, "`softmax` needs rank-2 operands");
    }

    /// `layer_norm`'s E-TYPE-003 note and E-TYPE-004 message name the operand the registry
    /// refuses, `x`, `gamma` or `beta`, with its type, as `add_rows` does (coordinator decision
    /// D33), so a long `gamma` and a long `beta` read differently.
    #[test]
    fn layer_norm_errors_name_the_refused_operand() {
        let f32 = TensorType::f32;
        let cases = [
            (
                [f32(&[4, 5]), f32(&[6]), f32(&[5])],
                codes::SHAPE_MISMATCH,
                "`gamma` has type Tensor<f32, [6]>",
            ),
            (
                [f32(&[4, 5]), f32(&[5]), f32(&[6])],
                codes::SHAPE_MISMATCH,
                "`beta` has type Tensor<f32, [6]>",
            ),
            (
                [f32(&[5]), f32(&[5]), f32(&[5])],
                codes::RANK,
                "`x` has type Tensor<f32, [5]>",
            ),
            (
                [f32(&[4, 5]), f32(&[1, 5]), f32(&[5])],
                codes::RANK,
                "`gamma` has type Tensor<f32, [1, 5]>",
            ),
            (
                [f32(&[4, 5]), f32(&[5]), f32(&[1, 5])],
                codes::RANK,
                "`beta` has type Tensor<f32, [1, 5]>",
            ),
        ];
        for (operands, code, named) in cases {
            let types: Vec<&TensorType> = operands.iter().collect();
            let error = crate::ops::infer(&crate::ir::tensor::TensorOp::LayerNorm, &types, None)
                .unwrap_err();
            let diagnostic = op_diagnostic("layer_norm", error, &types, &span());
            assert_eq!(diagnostic.code, code, "{diagnostic:?}");
            let text = format!("{diagnostic:?}");
            assert!(text.contains(named), "{named}: {text}");
        }
    }

    #[test]
    fn registry_errors_the_checker_rules_out_are_internal_diagnostics() {
        let ty = TensorType::f32(&[2]);
        for error in [
            OpError::DTypeMismatch {
                op: "Add",
                lhs: ty.clone(),
                rhs: ty.clone(),
            },
            OpError::MissingDeclaredType { op: "Input" },
        ] {
            let diagnostic = op_diagnostic("add", error, &[], &span());
            assert_eq!(diagnostic.code, codes::INTERNAL);
            assert_eq!(diagnostic.span, Some(span()));
            assert!(!codes::ALL.contains(&diagnostic.code));
        }
    }

    /// An operand of the wrong element type is the user's error, E-TYPE-018 at the call.
    #[test]
    fn an_operand_of_the_wrong_dtype_is_e_type_018() {
        let error = OpError::OperandDType {
            op: "Add",
            index: 0,
            expected: DType::F32,
            got: TensorType::i32(&[4]),
        };
        let diagnostic = op_diagnostic("add", error, &[], &span());
        assert_eq!(diagnostic.code, codes::DTYPE);
        assert_eq!(diagnostic.code.as_str(), "E-TYPE-018");
        assert_eq!(diagnostic.span, Some(span()));
        assert!(
            diagnostic.message.contains("`add` operand 1 must be f32"),
            "{}",
            diagnostic.message
        );
    }

    /// `add_rows` and `embedding` take operands of two ranks, so their E-TYPE-004 message states
    /// the rank of each operand and names the operand that has another.
    #[test]
    fn a_rank_error_of_an_operator_with_two_operand_ranks_states_each_rank() {
        let add_rows = "`add_rows` needs `x` of rank 2 and `b` of rank 1";
        let embedding = "`embedding` needs `table` of rank 2 and `ids` of rank 1";
        let cases = [
            (
                "add_rows",
                "AddRows",
                2,
                TensorType::f32(&[3]),
                add_rows,
                "`x` has type Tensor<f32, [3]>",
            ),
            (
                "add_rows",
                "AddRows",
                1,
                TensorType::f32(&[1, 3]),
                add_rows,
                "`b` has type Tensor<f32, [1, 3]>",
            ),
            (
                "embedding",
                "Embedding",
                2,
                TensorType::f32(&[6]),
                embedding,
                "`table` has type Tensor<f32, [6]>",
            ),
            (
                "embedding",
                "Embedding",
                1,
                TensorType::i32(&[2, 4]),
                embedding,
                "`ids` has type Tensor<i32, [2, 4]>",
            ),
        ];
        for (name, op, expected, got, rule, operand) in cases {
            let diagnostic = op_diagnostic(name, OpError::Rank { op, expected, got }, &[], &span());
            assert_eq!(diagnostic.code, codes::RANK);
            assert_eq!(diagnostic.message, format!("{rule}; {operand}"));
        }
        let error = OpError::Rank {
            op: "MatMul",
            expected: 2,
            got: TensorType::f32(&[3]),
        };
        let matmul = op_diagnostic("matmul", error, &[], &span());
        assert_eq!(matmul.message, "`matmul` needs rank-2 operands");
    }

    /// `i32` is declared by inputs, targets and function arguments and read by `embedding`.
    #[test]
    fn i32_ids_read_by_embedding_check_and_are_typed_i32() {
        let source = "model M {
    param table: Tensor<f32, [6, 3]> = xavier_uniform();
    fn forward(ids: Tensor<i32, [4]>) -> Tensor<f32, [4, 3]> {
        return embedding(table, ids);
    }
}
train T {
    model = M;
    input ids: Tensor<i32, [4]>;
    target y: Tensor<f32, [4, 3]>;
    loss = mse(model.forward(ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}
";
        let hir = check_source("m.mspec", source).expect("checks");
        assert_eq!(hir.train.data[0].ty, TensorType::i32(&[4]));
        let (_, forward) = hir.model.function("forward").expect("forward");
        assert_eq!(forward.args[0].ty, TensorType::i32(&[4]));
        assert_eq!(forward.result.ty, TensorType::f32(&[4, 3]));
    }

    /// A file whose `forward` binds `lets` and returns `result`, over `x: [6, 4]`, a key mask
    /// argument `m: [2, 3]` and parameters `w: [4, 4]` and `p: [2, 3]`.
    fn attention_source(lets: &str, result: &str) -> String {
        format!(
            "model M {{
    param w: Tensor<f32, [4, 4]> = xavier_uniform();
    param p: Tensor<f32, [2, 3]> = ones();
    fn forward(x: Tensor<f32, [6, 4]>, m: Tensor<f32, [2, 3]>) -> Tensor<f32, [6, 4]> {{
        {lets}
        return {result};
    }}
}}
train T {{
    model = M;
    input x: Tensor<f32, [6, 4]>;
    input m: Tensor<f32, [2, 3]>;
    target y: Tensor<f32, [6, 4]>;
    loss = mse(model.forward(x, m), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}}
"
        )
    }

    /// The codes `check` reports for `attention_source(lets, result)`.
    fn attention_codes(lets: &str, result: &str) -> Vec<String> {
        match check_source("m.mspec", &attention_source(lets, result)) {
            Ok(_) => Vec::new(),
            Err(diagnostics) => diagnostics
                .iter()
                .map(|d| d.code.as_str().to_owned())
                .collect(),
        }
    }

    #[test]
    fn attention_with_literal_heads_checks_and_carries_them_in_its_op() {
        let source = attention_source("", "attention(matmul(x, w), x, x, m, heads = 4)");
        let hir = check_source("m.mspec", &source).expect("checks");
        let (_, forward) = hir.model.function("forward").expect("forward");
        assert!(matches!(
            &forward.result.kind,
            HirExprKind::Op { op: TensorOp::Attention { heads: 4 }, args } if args.len() == 4
        ));
        assert_eq!(forward.result.ty, TensorType::f32(&[6, 4]));
    }

    /// `heads` is an integer literal of at least 1: 0, a float literal and a name are each one
    /// E-TYPE-019; given twice it is E-TYPE-002; on an operator without named arguments a named
    /// argument stays E-TYPE-002.
    #[test]
    fn named_operator_arguments_are_integer_literals_given_once() {
        for heads in ["0", "2.0", "x"] {
            let result = format!("attention(x, x, x, m, heads = {heads})");
            assert_eq!(attention_codes("", &result), ["E-TYPE-019"], "{heads}");
        }
        assert_eq!(
            attention_codes("", "attention(x, x, x, m, heads = 2, heads = 2)"),
            ["E-TYPE-002"]
        );
        assert_eq!(
            attention_codes("", "matmul(x, w, heads = x)"),
            ["E-TYPE-002"]
        );
    }

    /// `dropout(x, p = P)` checks at any rank with one operand; the HIR holds `p` in the op and
    /// the one source operand, the token being the lowering's.
    #[test]
    fn dropout_with_a_literal_p_checks_and_carries_it_in_its_op() {
        let source = attention_source("let d = dropout(m, p = 0.25);", "dropout(x, p = 0.5)");
        let hir = check_source("m.mspec", &source).expect("checks");
        let (_, forward) = hir.model.function("forward").expect("forward");
        assert!(matches!(
            &forward.result.kind,
            HirExprKind::Op { op: TensorOp::Dropout { p }, args } if *p == 0.5 && args.len() == 1
        ));
        assert_eq!(forward.result.ty, TensorType::f32(&[6, 4]));
        assert!(matches!(
            &forward.lets[0].value.kind,
            HirExprKind::Op { op: TensorOp::Dropout { p }, .. } if *p == 0.25
        ));
        assert_eq!(
            attention_codes("", "dropout(x, p = 0.0)"),
            Vec::<String>::new()
        );
    }

    /// `p` outside `[0, 1)` is E-TYPE-022, a value that is not a float literal E-TYPE-019, a
    /// missing `p` or a second operand E-TYPE-001.
    #[test]
    fn dropout_p_is_a_float_literal_in_0_1() {
        for p in ["1.0", "1.5", "0.9999999999", "-0.1"] {
            let result = format!("dropout(x, p = {p})");
            assert_eq!(attention_codes("", &result), ["E-TYPE-022"], "{p}");
        }
        for p in ["0", "1", "x"] {
            let result = format!("dropout(x, p = {p})");
            assert_eq!(attention_codes("", &result), ["E-TYPE-019"], "{p}");
        }
        assert_eq!(attention_codes("", "dropout(x)"), ["E-TYPE-001"]);
        assert_eq!(
            attention_codes("", "dropout(x, x, p = 0.5)"),
            ["E-TYPE-001"]
        );
        assert_eq!(
            attention_codes("", "dropout(x, p = 0.5, p = 0.5)"),
            ["E-TYPE-002"]
        );
    }

    /// A key mask that reads a parameter, directly, through an operator or through a `let`, is
    /// E-TYPE-021; one built from arguments alone checks.
    #[test]
    fn a_key_mask_that_reads_a_parameter_is_e_type_021() {
        let call = "attention(x, x, x, k, heads = 2)";
        assert_eq!(attention_codes("let k = p;", call), ["E-TYPE-021"]);
        assert_eq!(attention_codes("let k = mul(m, p);", call), ["E-TYPE-021"]);
        assert_eq!(
            attention_codes("let q = add(m, p); let k = relu(q);", call),
            ["E-TYPE-021"]
        );
        assert!(attention_codes("let k = relu(m);", call).is_empty());
    }

    /// Each diagnostic `check` reports for `source`, as its code, line and column.
    fn located(source: &str) -> Vec<(String, u32, u32)> {
        match check_source("m.mspec", source) {
            Ok(_) => Vec::new(),
            Err(diagnostics) => diagnostics
                .iter()
                .map(|d| {
                    let span = d.span.as_ref().expect("a located diagnostic");
                    (d.code.as_str().to_owned(), span.line, span.column)
                })
                .collect(),
        }
    }

    /// DESIGN.md § 2.1: the named arguments follow the operands. One before an operand is
    /// E-TYPE-002 at that named argument, wherever among the operands it stands.
    #[test]
    fn a_named_argument_before_an_operand_is_e_type_002_at_the_argument() {
        for result in [
            "attention(heads = 2, x, x, x, m)",
            "attention(x, x, heads = 2, x, m)",
        ] {
            let column = 16 + result.find("heads").expect("heads") as u32;
            assert_eq!(
                located(&attention_source("", result)),
                [("E-TYPE-002".to_owned(), 6, column)],
                "{result}"
            );
        }
        let source = attention_source("", "attention(heads = 2, x, x, x, m)");
        let message = check_source("m.mspec", &source).expect_err("refused")[0]
            .message
            .clone();
        assert!(
            message.contains("named arguments follow the operands"),
            "{message}"
        );
        assert!(located(&attention_source("", "attention(x, x, x, m, heads = 2)")).is_empty());
    }

    /// A model function of `[2, 3]` to `[2, 3]` reading `p`.
    const PMASK: &str =
        "fn pmask(a: Tensor<f32, [2, 3]>) -> Tensor<f32, [2, 3]> { return mul(a, p); }";
    /// A model function of `[2, 3]` to `[2, 3]` reading no parameter.
    const FREE: &str = "fn free(a: Tensor<f32, [2, 3]>) -> Tensor<f32, [2, 3]> { return relu(a); }";
    /// A model function whose argument reaches its result through a `let`.
    const PASS: &str =
        "fn pass(a: Tensor<f32, [2, 3]>) -> Tensor<f32, [2, 3]> { let b = relu(a); return b; }";
    /// A model function whose first argument does not reach its result.
    const DROP: &str = "fn drop(a: Tensor<f32, [2, 3]>, b: Tensor<f32, [2, 3]>) -> Tensor<f32, [2, 3]> { return relu(b); }";
    /// A model function that binds `p` in a `let` its result does not read.
    const UNUSED: &str =
        "fn unused(a: Tensor<f32, [2, 3]>) -> Tensor<f32, [2, 3]> { let q = p; return relu(a); }";
    /// A model function of `[6, 4]` to `[6, 4]` reading `w`.
    const SCALED: &str =
        "fn scaled(a: Tensor<f32, [6, 4]>) -> Tensor<f32, [6, 4]> { return matmul(a, w); }";

    /// A file whose model has the parameters `w: [4, 4]` and `p: [2, 3]`, the one-line model
    /// functions `functions`, then `forward(x: [6, 4], m: [2, 3])` binding `lets` and returning
    /// `forward`; the train block's loss, on line 14 at column 12, is `loss`.
    fn dataflow_source(functions: &str, lets: &str, forward: &str, loss: &str) -> String {
        format!(
            "model M {{
    param w: Tensor<f32, [4, 4]> = xavier_uniform();
    param p: Tensor<f32, [2, 3]> = ones();
    {functions}
    fn forward(x: Tensor<f32, [6, 4]>, m: Tensor<f32, [2, 3]>) -> Tensor<f32, [6, 4]> {{
        {lets} return {forward};
    }}
}}
train T {{
    model = M;
    input x: Tensor<f32, [6, 4]>;
    input m: Tensor<f32, [2, 3]>;
    target y: Tensor<f32, [6, 4]>;
    loss = {loss};
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}}
"
        )
    }

    /// The diagnostics of `dataflow_source` with `forward` the attention of `x` under the key
    /// mask `forward_mask`.
    fn dataflow(
        functions: &[&str],
        lets: &str,
        forward_mask: &str,
        loss: &str,
    ) -> Vec<(String, u32, u32)> {
        let forward = format!("attention(x, x, x, {forward_mask}, heads = 2)");
        located(&dataflow_source(&functions.join(" "), lets, &forward, loss))
    }

    /// E-TYPE-021 on line 14 at `needle` of `loss`.
    fn refused_at(loss: &str, needle: &str) -> Vec<(String, u32, u32)> {
        let column = 12 + loss.find(needle).expect("needle in loss") as u32;
        vec![("E-TYPE-021".to_owned(), 14, column)]
    }

    /// Coordinator decision D41, 2026-10-25, `.engineering/waves/bert-1.md`: a key mask is
    /// E-TYPE-021 exactly when its value reads a `param` by dataflow, through `let`s, model
    /// function bodies and model function arguments alike. A train block argument that reaches
    /// the key mask inside the callee is refused at the argument.
    #[test]
    fn a_key_mask_reading_a_parameter_through_a_function_argument_is_e_type_021() {
        let all = [PMASK, FREE, PASS, DROP, UNUSED, SCALED];
        let loss = "mse(model.forward(x, model.pmask(m)), y)";
        assert_eq!(
            dataflow(&all, "", "m", loss),
            refused_at(loss, "model.pmask")
        );
        assert_eq!(
            dataflow(&all, "", "relu(m)", loss),
            refused_at(loss, "model.pmask")
        );
        assert_eq!(
            dataflow(&all, "let k = relu(m);", "k", loss),
            refused_at(loss, "model.pmask")
        );
        let pass = "mse(model.forward(x, model.pass(model.pmask(m))), y)";
        assert_eq!(
            dataflow(&all, "", "m", pass),
            refused_at(pass, "model.pass")
        );
        let operator = "mse(model.forward(x, add(m, model.pmask(m))), y)";
        assert_eq!(
            dataflow(&all, "", "m", operator),
            refused_at(operator, "add")
        );
        let direct = "mse(attention(model.forward(x, m), x, x, model.pmask(m), heads = 2), y)";
        assert_eq!(
            dataflow(&all, "", "m", direct),
            refused_at(direct, "model.pmask")
        );
    }

    /// D41: a model function that reads no parameter, or whose parameter-reading argument does
    /// not reach its result, is no reason to refuse; nor is a parameter-reading argument that
    /// reaches only an operand with a gradient.
    #[test]
    fn a_key_mask_that_reads_no_parameter_by_dataflow_checks() {
        let all = [PMASK, FREE, PASS, DROP, UNUSED, SCALED];
        assert!(dataflow(&all, "", "m", "mse(model.forward(x, m), y)").is_empty());
        for loss in [
            "mse(model.forward(x, model.free(m)), y)",
            "mse(attention(model.forward(x, m), x, x, model.free(m), heads = 2), y)",
            "mse(model.forward(x, model.pass(m)), y)",
            "mse(model.forward(x, model.drop(model.pmask(m), m)), y)",
            "mse(model.forward(x, model.unused(m)), y)",
            "mse(model.forward(model.scaled(x), m), y)",
        ] {
            assert!(dataflow(&all, "", "m", loss).is_empty(), "{loss}");
        }
    }
}
