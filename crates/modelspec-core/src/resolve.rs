//! Name resolution: binds every identifier to its declaration (DESIGN.md § 3, § 3.3 pass 1).
//!
//! The M0 subset has three kinds of scope:
//!
//! - the model: its parameters and functions share one namespace;
//! - a model function: the model's parameters, then the function's arguments, then its `let`
//!   bindings in order; a name is bound at most once, so neither an argument nor a `let` may
//!   reuse a parameter's, an argument's or an earlier `let`'s name;
//! - the train block: its `input` and `target` declarations, and `model`, whose functions are
//!   called as `model.<function>(args)`. Model parameters are not visible here.
//!
//! A call `name(args)` names a built-in operator ([`operator`]); model functions are reached only
//! through `model.` in the train block, so a model function cannot call itself or another.
//! Each failure is one `E-NAME` diagnostic, and a duplicate never cascades: after a duplicate
//! argument, `let` or data declaration the later binding wins; after a duplicate model item the
//! first parameter and the first function of the name both stay reachable.

use std::collections::BTreeMap;

use crate::diag::{Code, Diagnostic, SourceSpan};
use crate::hir::{ArgRef, DataRef, FnRef, LetRef, ParamRef};
use crate::ir::tensor::TensorOp;
use crate::syntax::ast::{Ident, MemberCall, Model, ModelItem, Train};

/// Name-resolution diagnostic codes.
pub mod codes {
    use crate::diag::Code;

    macro_rules! declare {
        ($($(#[doc = $doc:literal])* $name:ident = $text:literal;)*) => {
            $(
                $(#[doc = $doc])*
                pub const $name: Code = Code::new($text);
            )*

            /// Every name-resolution code, in numeric order.
            pub const ALL: &[Code] = &[$($name),*];
        };
    }

    declare! {
        /// A variable no enclosing scope binds.
        UNKNOWN_NAME = "E-NAME-001";
        /// A call whose callee is not a built-in operator.
        UNKNOWN_FUNCTION = "E-NAME-002";
        /// A model parameter or function whose name an earlier model item already declares.
        DUPLICATE_MODEL_ITEM = "E-NAME-003";
        /// A function argument or `let` binding a name its function's scope already binds.
        DUPLICATE_BINDING = "E-NAME-004";
        /// An `input` or `target` declaration whose name an earlier one already declares.
        DUPLICATE_DATA = "E-NAME-005";
        /// A train block's `model` setting naming no model of the file.
        UNKNOWN_MODEL = "E-NAME-006";
        /// A member call whose receiver is not `model` in a train block.
        UNKNOWN_RECEIVER = "E-NAME-007";
        /// `model.<name>(...)` where the model has no function `<name>`.
        UNKNOWN_METHOD = "E-NAME-008";
    }
}

/// The built-in operators a source call may name, and the Tensor IR op each one is.
pub static OPERATORS: [(&str, TensorOp); 9] = [
    ("matmul", TensorOp::MatMul),
    ("add", TensorOp::Add),
    ("sub", TensorOp::Sub),
    ("mul", TensorOp::Mul),
    ("relu", TensorOp::Relu),
    ("transpose", TensorOp::Transpose),
    ("sum", TensorOp::ReduceSum),
    ("mean", TensorOp::ReduceMean),
    ("mse", TensorOp::Mse),
];

/// The Tensor IR op the source operator `name` is.
pub fn operator(name: &str) -> Option<TensorOp> {
    OPERATORS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, op)| op.clone())
}

/// What a variable refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    Param(ParamRef),
    Arg(ArgRef),
    Let(LetRef),
    Data(DataRef),
}

/// The model's item names. Parameters and functions are numbered in source order, each kind on
/// its own, as [`crate::hir::HirModel`] lists them.
#[derive(Clone, Debug)]
pub struct ModelNames {
    pub name: String,
    params: BTreeMap<String, (ParamRef, SourceSpan)>,
    functions: BTreeMap<String, FnRef>,
}

impl ModelNames {
    /// Collects the item names of `model`, reporting each name declared twice (E-NAME-003).
    ///
    /// A duplicate is reported once and still registers where it does not collide: the first
    /// parameter and the first function of each name are both reachable, so a `param x` and a
    /// `fn x` leave every later use of either resolved and nothing cascades.
    pub fn collect(model: &Model, diagnostics: &mut Vec<Diagnostic>) -> Self {
        let mut names = Self {
            name: model.name.name.clone(),
            params: BTreeMap::new(),
            functions: BTreeMap::new(),
        };
        let mut declared: BTreeMap<&str, &SourceSpan> = BTreeMap::new();
        let (mut params, mut functions) = (0, 0);
        for item in &model.items {
            let name = match item {
                ModelItem::Param(p) => &p.name,
                ModelItem::Fn(f) => &f.name,
            };
            let first = declared.get(name.name.as_str()).copied();
            if let Some(first) = first {
                diagnostics.push(
                    Diagnostic::new(
                        codes::DUPLICATE_MODEL_ITEM,
                        format!(
                            "`{}` is already declared in model `{}`",
                            name.name, model.name.name
                        ),
                    )
                    .at(name.span.clone())
                    .with("first declared at", first.to_string()),
                );
            } else {
                declared.insert(&name.name, &name.span);
            }
            match item {
                ModelItem::Param(_) => {
                    names
                        .params
                        .entry(name.name.clone())
                        .or_insert_with(|| (ParamRef(params), name.span.clone()));
                    params += 1;
                }
                ModelItem::Fn(_) => {
                    names
                        .functions
                        .entry(name.name.clone())
                        .or_insert(FnRef(functions));
                    functions += 1;
                }
            }
        }
        names
    }

    /// The function named `name`.
    pub fn function(&self, name: &str) -> Option<FnRef> {
        self.functions.get(name).copied()
    }
}

/// The variables visible in one model function or in the train block.
#[derive(Clone, Debug)]
pub struct Scope {
    /// What the scope belongs to, as diagnostics name it: "function `forward`".
    owner: String,
    bindings: BTreeMap<String, (Binding, SourceSpan)>,
}

impl Scope {
    /// The scope of model function `function`: the model's parameters, before any argument.
    pub fn function(model: &ModelNames, function: &Ident) -> Self {
        Self {
            owner: format!("function `{}`", function.name),
            bindings: model
                .params
                .iter()
                .map(|(name, (r, span))| (name.clone(), (Binding::Param(*r), span.clone())))
                .collect(),
        }
    }

    /// The scope of train block `train`, before its data declarations.
    pub fn train(train: &Train) -> Self {
        Self {
            owner: format!("train block `{}`", train.name.name),
            bindings: BTreeMap::new(),
        }
    }

    /// Binds a function argument or `let`; a name already bound is E-NAME-004.
    pub fn bind_local(
        &mut self,
        name: &Ident,
        binding: Binding,
        diagnostics: &mut Vec<Diagnostic>,
    ) {
        self.bind(
            name,
            binding,
            codes::DUPLICATE_BINDING,
            "bound",
            diagnostics,
        );
    }

    /// Binds an `input` or `target`; a name already declared is E-NAME-005.
    pub fn bind_data(&mut self, name: &Ident, binding: Binding, diagnostics: &mut Vec<Diagnostic>) {
        self.bind(
            name,
            binding,
            codes::DUPLICATE_DATA,
            "declared",
            diagnostics,
        );
    }

    fn bind(
        &mut self,
        name: &Ident,
        binding: Binding,
        code: Code,
        verb: &str,
        diagnostics: &mut Vec<Diagnostic>,
    ) {
        if let Some((_, first)) = self.bindings.get(&name.name) {
            diagnostics.push(
                Diagnostic::new(
                    code,
                    format!("`{}` is already {verb} in {}", name.name, self.owner),
                )
                .at(name.span.clone())
                .with("first declared at", first.to_string()),
            );
        }
        self.bindings
            .insert(name.name.clone(), (binding, name.span.clone()));
    }

    /// What variable `name`, used at `span`, refers to; E-NAME-001 when nothing binds it.
    pub fn lookup(
        &self,
        name: &str,
        span: &SourceSpan,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Option<Binding> {
        match self.bindings.get(name) {
            Some((binding, _)) => Some(*binding),
            None => {
                diagnostics.push(
                    Diagnostic::new(
                        codes::UNKNOWN_NAME,
                        format!("unknown name `{name}` in {}", self.owner),
                    )
                    .at(span.clone()),
                );
                None
            }
        }
    }
}

/// The operator a call names; E-NAME-002 when it names none.
pub fn resolve_operator(callee: &Ident, diagnostics: &mut Vec<Diagnostic>) -> Option<TensorOp> {
    let op = operator(&callee.name);
    if op.is_none() {
        let known: Vec<&str> = OPERATORS.iter().map(|(n, _)| *n).collect();
        diagnostics.push(
            Diagnostic::new(
                codes::UNKNOWN_FUNCTION,
                format!("unknown function `{}`", callee.name),
            )
            .at(callee.span.clone())
            .with("M0 operators", known.join(", ")),
        );
    }
    op
}

/// Whether the train block's `model` setting names `model`; E-NAME-006 when it does not.
pub fn resolve_train_model(
    train: &Train,
    model: &ModelNames,
    diagnostics: &mut Vec<Diagnostic>,
) -> bool {
    let named = &train.model.value;
    if named.name == model.name {
        return true;
    }
    diagnostics.push(
        Diagnostic::new(
            codes::UNKNOWN_MODEL,
            format!("unknown model `{}`", named.name),
        )
        .at(named.span.clone())
        .with("the file's model", model.name.clone()),
    );
    false
}

/// The model function a member call reaches.
///
/// `model` is the train block's model when the call is in the train block and its `model`
/// setting resolved; `in_train` says whether the call is in the train block at all. A receiver
/// other than `model` in the train block, or any member call in a model function, is E-NAME-007;
/// a function the model lacks is E-NAME-008. A call on a model the train block could not resolve
/// reports nothing more.
pub fn resolve_method(
    call: &MemberCall,
    in_train: bool,
    model: Option<&ModelNames>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<FnRef> {
    if !in_train || call.receiver.name != "model" {
        diagnostics.push(
            Diagnostic::new(
                codes::UNKNOWN_RECEIVER,
                format!("unknown receiver `{}`", call.receiver.name),
            )
            .at(call.receiver.span.clone())
            .note("only `model.<function>(...)` in a train block calls a model function"),
        );
        return None;
    }
    let model = model?;
    let function = model.function(&call.method.name);
    if function.is_none() {
        diagnostics.push(
            Diagnostic::new(
                codes::UNKNOWN_METHOD,
                format!(
                    "model `{}` has no function `{}`",
                    model.name, call.method.name
                ),
            )
            .at(call.method.span.clone()),
        );
    }
    function
}
