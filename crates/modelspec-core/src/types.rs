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
//! function argument, and read only as the ids of `embedding` (E-TYPE-018 for a parameter or a
//! function result of `i32`; the registry refuses every other operand); a parameter is
//! initialized by `xavier_uniform()` and is rank 2; a
//! function's `return` has its declared type; a `model.<function>(args)` call matches the
//! function's arguments; the loss is rank 0 and calls `model.forward` exactly once, counting calls
//! nested in the arguments of another (the prediction is that call's value, DESIGN.md § 2.1);
//! the `sgd` learning rate is a positive finite `f32`;
//! `steps` is positive. The parser already enforces that `seed` fits `u64`, that the optimizer is
//! `sgd` and that literals are finite `f64`s. A `vulkan(<index>)` device is not validated.

use crate::diag::{Code, Diagnostic, SourceSpan};
use crate::hir::{
    ArgRef, DataRef, Device, FnRef, Hir, HirArg, HirData, HirExpr, HirExprKind, HirFunction,
    HirLet, HirModel, HirParam, HirTrain, LetRef,
};
use crate::ir::tensor::{Initializer, TensorType};
use crate::ops::{self, OpError};
use crate::resolve::{self, Binding, ModelNames, Scope};
use crate::syntax::{self, ast};

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
        /// An operator or model function called with the wrong number of arguments.
        ARGUMENT_COUNT = "E-TYPE-001";
        /// A named argument (`a = x`) to an operator or a model function.
        NAMED_ARGUMENT = "E-TYPE-002";
        /// `add`, `sub`, `mul` or `mse` operands of different shapes, neither of them rank 0, an
        /// `add_rows(x, b)` whose `b` is not as long as a row of `x`, or a
        /// `layer_norm(x, gamma, beta)` whose `gamma` or `beta` is not as long as a row of `x`.
        SHAPE_MISMATCH = "E-TYPE-003";
        /// A `matmul`, `transpose` or `softmax` operand that is not rank 2, an `embedding` whose
        /// table is not rank 2 or whose ids are not rank 1, an `add_rows(x, b)` whose `x` is not
        /// rank 2 or whose `b` is not rank 1, or a `layer_norm(x, gamma, beta)` whose `x` is not
        /// rank 2 or whose `gamma` or `beta` is not rank 1.
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
        /// An `sgd` learning rate that is not a positive finite `f32`.
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
        /// A tensor type whose element or byte count overflows `u64` (DESIGN.md § 2.2).
        COUNT_OVERFLOW = "E-TYPE-016";
        /// A train block's loss that does not call `model.forward` exactly once: the prediction
        /// is the value of that one call.
        FORWARD_CALLS = "E-TYPE-017";
        /// An `i32` tensor used other than as the ids of `embedding`: an operand of any other
        /// operator or the table of `embedding`, a `param` of type `i32`, or a function declared
        /// to return `i32` (DESIGN.md § 2.2).
        DTYPE = "E-TYPE-018";
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
        let functions: Vec<Option<HirFunction>> = ast
            .model
            .items
            .iter()
            .filter_map(|item| match item {
                ast::ModelItem::Fn(f) => Some(f),
                ast::ModelItem::Param(_) => None,
            })
            .zip(&signatures)
            .map(|(f, sig)| self.function(f, sig, &names, &param_types))
            .collect();
        let train = self.train(&ast.train, &names, &signatures);
        Some(Hir {
            model: HirModel {
                name: ast.model.name.name.clone(),
                params: params.into_iter().collect::<Option<_>>()?,
                functions: functions.into_iter().collect::<Option<_>>()?,
                span: ast.model.span.clone(),
            },
            train: train?,
        })
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
                    "`{}` must return f32; `i32` is for the ids of `embedding` only",
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
        let fits = ty
            .shape
            .iter()
            .try_fold(1u64, |count, dim| match dim {
                crate::ir::tensor::Dim::Static(n) => count.checked_mul(*n),
                crate::ir::tensor::Dim::Symbol(_) => Some(count),
            })
            .and_then(|elements| elements.checked_mul(ty.dtype.size_bytes()))
            .is_some();
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
                    "parameter `{}` must be f32; `i32` is for the ids of `embedding` only",
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

    fn function(
        &mut self,
        f: &ast::Function,
        sig: &Signature,
        names: &ModelNames,
        params: &[Option<TensorType>],
    ) -> Option<HirFunction> {
        let mut scope = Scope::function(names, &f.name);
        for (i, arg) in f.params.iter().enumerate() {
            scope.bind_local(&arg.name, Binding::Arg(ArgRef(i)), &mut self.diagnostics);
        }
        let mut let_types: Vec<Option<TensorType>> = Vec::new();
        let mut lets = Vec::new();
        for (i, binding) in f.lets.iter().enumerate() {
            let env = Env {
                scope: &scope,
                params,
                args: &sig.args,
                lets: &let_types,
                data: &[],
                in_train: false,
                model: None,
            };
            let value = self.expr(&binding.value, &env);
            let_types.push(value.as_ref().map(|v| v.ty.clone()));
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
            data: &[],
            in_train: false,
            model: None,
        };
        let result = self.expr(&f.result.value, &env);
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
        Some(HirFunction {
            name: f.name.name.clone(),
            args: args?,
            output: sig.output.clone()?,
            lets: lets.into_iter().collect::<Option<_>>()?,
            result: result?,
            span: f.span.clone(),
        })
    }

    fn train(
        &mut self,
        t: &ast::Train,
        names: &ModelNames,
        signatures: &[Signature],
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

        let ast::Optimizer::Sgd { lr, lr_span } = &t.optimizer.value;
        let lr_f32 = *lr as f32;
        let lr = if lr_f32.is_finite() && lr_f32 > 0.0 {
            Some(lr_f32)
        } else {
            self.report(
                codes::LEARNING_RATE,
                format!("learning rate `{lr:?}` is not a positive finite f32"),
                lr_span,
            );
            None
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
            steps: steps?,
            seed: t.seed.value,
            device,
            span: t.span.clone(),
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

    fn op_call(&mut self, call: &ast::Call, env: &Env<'_>) -> Option<HirExpr> {
        let op = resolve::resolve_operator(&call.callee, &mut self.diagnostics);
        let args = self.args(&call.callee.name, &call.args, env);
        let (op, args) = (op?, args?);
        let types: Vec<&TensorType> = args.iter().map(|a| &a.ty).collect();
        match ops::infer(&op, &types, None) {
            Ok(ty) => {
                if !self.counts_fit(&ty, &call.span) {
                    return None;
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
        Some(HirExpr {
            kind: HirExprKind::Call { function, args },
            ty: sig.output.clone()?,
            span: call.span.clone(),
        })
    }
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
        .note("`i32` tensors are row ids; only `embedding` reads them, as its second operand"),
        error @ (OpError::DTypeMismatch { .. }
        | OpError::MissingDeclaredType { .. }
        | OpError::ParamDType { .. }) => Diagnostic::new(
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
                "softmax"
            ]
        );
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
}
