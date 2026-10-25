//! Abstract syntax tree of a `.mspec` source file, with source spans.
//!
//! This is the Source AST of DESIGN.md § 3 for the M0 subset of DESIGN.md § 2: exactly one
//! `model` and one `train` item. Every node carries the [`SourceSpan`] of its first token.
//! Equality compares spans too; [`Ast::erase_spans`] compares structure alone.

use crate::diag::SourceSpan;

/// A parsed source file.
#[derive(Clone, Debug, PartialEq)]
pub struct Ast {
    pub model: Model,
    pub train: Train,
    /// The start of the file.
    pub span: SourceSpan,
}

/// A name as written, such as a model, parameter or function name.
#[derive(Clone, Debug, PartialEq)]
pub struct Ident {
    pub name: String,
    pub span: SourceSpan,
}

/// `model Name { ... }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub name: Ident,
    /// Items in source order.
    pub items: Vec<ModelItem>,
    pub span: SourceSpan,
}

/// One item of a model body.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelItem {
    Param(Param),
    Fn(Function),
}

/// `param name: Type = init();`.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub ty: TensorType,
    pub init: Call,
    pub span: SourceSpan,
}

/// `fn name(args) -> Type { let ...; return expr; }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub name: Ident,
    pub params: Vec<FnParam>,
    pub output: TensorType,
    pub lets: Vec<Let>,
    pub result: Return,
    pub span: SourceSpan,
}

/// `name: Type` in a function's parameter list.
#[derive(Clone, Debug, PartialEq)]
pub struct FnParam {
    pub name: Ident,
    pub ty: TensorType,
    pub span: SourceSpan,
}

/// `let name = expr;`.
#[derive(Clone, Debug, PartialEq)]
pub struct Let {
    pub name: Ident,
    pub value: Expr,
    pub span: SourceSpan,
}

/// `return expr;`, the last statement of a function body.
#[derive(Clone, Debug, PartialEq)]
pub struct Return {
    pub value: Expr,
    pub span: SourceSpan,
}

/// `Tensor<f32, [d0, d1, ...]>`; no dimensions means rank 0.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorType {
    pub dtype: Dtype,
    pub dims: Vec<Dim>,
    pub span: SourceSpan,
}

/// The element type written in a tensor type.
#[derive(Clone, Debug, PartialEq)]
pub struct Dtype {
    pub kind: DType,
    pub span: SourceSpan,
}

/// Element types of the M0 subset. `i32` is for row ids only; the checker refuses it anywhere
/// else (`E-TYPE-018`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    I32,
}

/// One integer literal dimension of a tensor type.
#[derive(Clone, Debug, PartialEq)]
pub struct Dim {
    pub size: u64,
    pub span: SourceSpan,
}

/// An expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: SourceSpan,
}

/// The forms of an expression; M0 has no operators.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    /// A name such as `x`.
    Var(String),
    /// An integer literal.
    Int(u64),
    /// A float literal.
    Float(f64),
    /// `callee(args)`.
    Call(Call),
    /// `receiver.method(args)`, such as `model.forward(x)`.
    MemberCall(MemberCall),
}

/// `callee(args)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub callee: Ident,
    pub args: Vec<Arg>,
    pub span: SourceSpan,
}

/// `receiver.method(args)`.
#[derive(Clone, Debug, PartialEq)]
pub struct MemberCall {
    pub receiver: Ident,
    pub method: Ident,
    pub args: Vec<Arg>,
    pub span: SourceSpan,
}

/// A call argument, positional (`x`) or named (`lr = 0.1`).
#[derive(Clone, Debug, PartialEq)]
pub struct Arg {
    pub name: Option<Ident>,
    pub value: Expr,
    pub span: SourceSpan,
}

/// `train Name { ... }`. Each setting appears exactly once; `data` holds the `input` and
/// `target` declarations in source order.
#[derive(Clone, Debug, PartialEq)]
pub struct Train {
    pub name: Ident,
    pub model: Setting<Ident>,
    pub data: Vec<DataDecl>,
    pub loss: Setting<Expr>,
    pub optimizer: Setting<Optimizer>,
    pub steps: Setting<u64>,
    pub seed: Setting<u64>,
    pub device: Setting<Device>,
    pub span: SourceSpan,
}

/// A `name = value;` setting of a train block; `span` is the setting's name.
#[derive(Clone, Debug, PartialEq)]
pub struct Setting<T> {
    pub value: T,
    pub span: SourceSpan,
}

/// `input name: Type;` or `target name: Type;`.
#[derive(Clone, Debug, PartialEq)]
pub struct DataDecl {
    pub role: DataRole,
    pub name: Ident,
    pub ty: TensorType,
    pub span: SourceSpan,
}

/// Whether a data declaration is a model input or a training target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataRole {
    Input,
    Target,
}

/// The optimizer of a train block.
#[derive(Clone, Debug, PartialEq)]
pub enum Optimizer {
    /// `sgd(lr = <float>)`.
    Sgd { lr: f64, lr_span: SourceSpan },
    /// `adamw(<args>)`, at `span`, the `adamw` name: the arguments as written, which the checker
    /// requires to be `lr`, `weight_decay`, `beta1`, `beta2`, `eps` and `warmup_steps`, each
    /// once and named (DESIGN.md § 2.1).
    AdamW { args: Vec<Arg>, span: SourceSpan },
    /// `muon(<args>)`, at `span`, the `muon` name: the arguments as written, which the checker
    /// requires to be `lr`, `momentum`, `weight_decay`, `ns_steps`, `warmup_steps`, `adam_lr`,
    /// `beta1`, `beta2` and `eps`, each once and named (DESIGN.md § 2.1).
    Muon { args: Vec<Arg>, span: SourceSpan },
}

/// The execution device of a train block.
#[derive(Clone, Debug, PartialEq)]
pub enum Device {
    /// `cpu`.
    Cpu,
    /// `vulkan(<index>)`.
    Vulkan { index: u32, index_span: SourceSpan },
}

impl Ast {
    /// Replaces every span in the tree with the same empty span, so that two trees compare
    /// equal exactly when their structure and values are equal.
    pub fn erase_spans(&mut self) {
        self.erase();
    }
}

fn erase_span(span: &mut SourceSpan) {
    *span = SourceSpan {
        file: String::new(),
        line: 0,
        column: 0,
    };
}

trait Erase {
    fn erase(&mut self);
}

impl Erase for SourceSpan {
    fn erase(&mut self) {
        erase_span(self);
    }
}

impl Erase for u64 {
    fn erase(&mut self) {}
}

impl<T: Erase> Erase for Vec<T> {
    fn erase(&mut self) {
        self.iter_mut().for_each(Erase::erase);
    }
}

impl Erase for Ast {
    fn erase(&mut self) {
        self.model.erase();
        self.train.erase();
        self.span.erase();
    }
}

impl Erase for Ident {
    fn erase(&mut self) {
        self.span.erase();
    }
}

impl Erase for Model {
    fn erase(&mut self) {
        self.name.erase();
        self.items.erase();
        self.span.erase();
    }
}

impl Erase for ModelItem {
    fn erase(&mut self) {
        match self {
            ModelItem::Param(p) => p.erase(),
            ModelItem::Fn(f) => f.erase(),
        }
    }
}

impl Erase for Param {
    fn erase(&mut self) {
        self.name.erase();
        self.ty.erase();
        self.init.erase();
        self.span.erase();
    }
}

impl Erase for Function {
    fn erase(&mut self) {
        self.name.erase();
        self.params.erase();
        self.output.erase();
        self.lets.erase();
        self.result.erase();
        self.span.erase();
    }
}

impl Erase for FnParam {
    fn erase(&mut self) {
        self.name.erase();
        self.ty.erase();
        self.span.erase();
    }
}

impl Erase for Let {
    fn erase(&mut self) {
        self.name.erase();
        self.value.erase();
        self.span.erase();
    }
}

impl Erase for Return {
    fn erase(&mut self) {
        self.value.erase();
        self.span.erase();
    }
}

impl Erase for TensorType {
    fn erase(&mut self) {
        self.dtype.span.erase();
        self.dims.erase();
        self.span.erase();
    }
}

impl Erase for Dim {
    fn erase(&mut self) {
        self.span.erase();
    }
}

impl Erase for Expr {
    fn erase(&mut self) {
        match &mut self.kind {
            ExprKind::Var(_) | ExprKind::Int(_) | ExprKind::Float(_) => {}
            ExprKind::Call(c) => c.erase(),
            ExprKind::MemberCall(m) => m.erase(),
        }
        self.span.erase();
    }
}

impl Erase for Call {
    fn erase(&mut self) {
        self.callee.erase();
        self.args.erase();
        self.span.erase();
    }
}

impl Erase for MemberCall {
    fn erase(&mut self) {
        self.receiver.erase();
        self.method.erase();
        self.args.erase();
        self.span.erase();
    }
}

impl Erase for Arg {
    fn erase(&mut self) {
        if let Some(name) = &mut self.name {
            name.erase();
        }
        self.value.erase();
        self.span.erase();
    }
}

impl Erase for Train {
    fn erase(&mut self) {
        self.name.erase();
        self.model.erase();
        self.data.erase();
        self.loss.erase();
        self.optimizer.erase();
        self.steps.erase();
        self.seed.erase();
        self.device.erase();
        self.span.erase();
    }
}

impl<T: Erase> Erase for Setting<T> {
    fn erase(&mut self) {
        self.value.erase();
        self.span.erase();
    }
}

impl Erase for DataDecl {
    fn erase(&mut self) {
        self.name.erase();
        self.ty.erase();
        self.span.erase();
    }
}

impl Erase for Optimizer {
    fn erase(&mut self) {
        match self {
            Optimizer::Sgd { lr_span, .. } => lr_span.erase(),
            Optimizer::AdamW { args, span } | Optimizer::Muon { args, span } => {
                args.erase();
                span.erase();
            }
        }
    }
}

impl Erase for Device {
    fn erase(&mut self) {
        match self {
            Device::Cpu => {}
            Device::Vulkan { index_span, .. } => index_span.erase(),
        }
    }
}
