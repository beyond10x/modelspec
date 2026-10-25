//! Typed HIR: the resolved and typed program (DESIGN.md § 3.1).
//!
//! [`crate::types::check`] produces a [`Hir`] only for a file with no diagnostics, so every name
//! here is resolved, every expression carries its tensor type, and every declared type is a
//! valid M0 type (static, no zero dimension, element and byte counts within `u64`). Each node
//! keeps the [`SourceSpan`] of the source construct it came from. M0 has no generics, so there
//! are no dimension bindings yet.

use crate::diag::SourceSpan;
use crate::ir::tensor::{Initializer, TensorOp, TensorType};
use crate::ir::training::OptimizerRule;

/// A checked `.mspec` file: one model and its train block.
#[derive(Clone, Debug, PartialEq)]
pub struct Hir {
    pub model: HirModel,
    pub train: HirTrain,
}

/// Index of a parameter in [`HirModel::params`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParamRef(pub usize);

/// Index of a function in [`HirModel::functions`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FnRef(pub usize);

/// Index of an argument in [`HirFunction::args`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArgRef(pub usize);

/// Index of a binding in [`HirFunction::lets`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LetRef(pub usize);

/// Index of a declaration in [`HirTrain::data`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DataRef(pub usize);

/// `model Name { ... }`, with its items in source order.
#[derive(Clone, Debug, PartialEq)]
pub struct HirModel {
    pub name: String,
    pub params: Vec<HirParam>,
    pub functions: Vec<HirFunction>,
    pub span: SourceSpan,
}

impl HirModel {
    /// The function named `name`.
    pub fn function(&self, name: &str) -> Option<(FnRef, &HirFunction)> {
        self.functions
            .iter()
            .enumerate()
            .find(|(_, f)| f.name == name)
            .map(|(i, f)| (FnRef(i), f))
    }
}

/// A trainable parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct HirParam {
    pub name: String,
    /// Stable canonical path `<Model>.<param>`, such as `LinearReg.w` (DESIGN.md § 2.3).
    pub path: String,
    pub ty: TensorType,
    pub init: Initializer,
    pub span: SourceSpan,
}

/// A pure model function.
#[derive(Clone, Debug, PartialEq)]
pub struct HirFunction {
    pub name: String,
    pub args: Vec<HirArg>,
    pub output: TensorType,
    /// `let` bindings in source order; a binding only refers to earlier ones.
    pub lets: Vec<HirLet>,
    /// The `return` expression; its type equals [`Self::output`].
    pub result: HirExpr,
    pub span: SourceSpan,
}

/// A function argument.
#[derive(Clone, Debug, PartialEq)]
pub struct HirArg {
    pub name: String,
    pub ty: TensorType,
    pub span: SourceSpan,
}

/// `let name = value;`.
#[derive(Clone, Debug, PartialEq)]
pub struct HirLet {
    pub name: String,
    pub value: HirExpr,
    pub span: SourceSpan,
}

/// A typed expression.
#[derive(Clone, Debug, PartialEq)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: TensorType,
    pub span: SourceSpan,
}

/// The resolved forms of an expression.
#[derive(Clone, Debug, PartialEq)]
pub enum HirExprKind {
    /// A model parameter, inside a model function.
    Param(ParamRef),
    /// An argument of the enclosing function.
    Arg(ArgRef),
    /// An earlier `let` of the enclosing function.
    Let(LetRef),
    /// An `input` or `target` of the train block, inside the train block.
    Data(DataRef),
    /// A rank-0 `f32` constant.
    Scalar(f32),
    /// A built-in operator applied to `args`, typed through [`crate::ops::infer`].
    Op { op: TensorOp, args: Vec<HirExpr> },
    /// `model.<function>(args)`, inside the train block.
    Call { function: FnRef, args: Vec<HirExpr> },
}

/// The checked `train` block. Later stages read the training configuration from here.
#[derive(Clone, Debug, PartialEq)]
pub struct HirTrain {
    pub name: String,
    /// The model the block trains, [`Hir::model`] by name.
    pub model: String,
    /// The `input` and `target` declarations, in source order.
    pub data: Vec<HirData>,
    /// The loss; its type is rank 0.
    pub loss: HirExpr,
    /// The learning rate of `sgd`, or the peak rate of `adamw`: positive and finite.
    pub lr: f32,
    /// The update rule of the `optimizer` setting, with the `adamw` hyperparameters.
    pub optimizer: OptimizerRule,
    /// The number of training steps: positive.
    pub steps: u64,
    pub seed: u64,
    pub device: Device,
    pub span: SourceSpan,
}

impl HirTrain {
    /// The `input` declarations, in source order.
    pub fn inputs(&self) -> impl Iterator<Item = &HirData> {
        self.data.iter().filter(|d| d.role == DataRole::Input)
    }

    /// The `target` declarations, in source order.
    pub fn targets(&self) -> impl Iterator<Item = &HirData> {
        self.data.iter().filter(|d| d.role == DataRole::Target)
    }
}

/// `input name: Type;` or `target name: Type;`.
#[derive(Clone, Debug, PartialEq)]
pub struct HirData {
    pub role: DataRole,
    pub name: String,
    pub ty: TensorType,
    pub span: SourceSpan,
}

pub use crate::syntax::ast::DataRole;

/// The execution device a train block names. M0 does not validate a Vulkan index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Vulkan(u32),
}
