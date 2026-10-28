//! The glyph set (DESIGN.md § 13.1): one SVG `<symbol>` per Tensor IR operation, in eleven
//! families, and the gradient, scope, stage and endpoint glyphs, themed by CSS custom properties.
//!
//! Every glyph is drawn on a 24-unit grid with integer coordinates and one stroke width, 2 units:
//! the family fixes the outline and the accent colour, the operator fixes the mark inside it, in
//! the ink colour. A gradient op is its forward operator's glyph with the outline dashed and the
//! `∇` badge in the top right; one that is the gradient with respect to one of several operands
//! (`LayerNormGradX`, `AttentionGradK`, ...) also marks that operand's slot along the bottom.
//! Composites (scopes) share a stacked-card outline, stages a chevron and endpoints a document.

use std::fmt;

use modelspec_core::ir::tensor::{Dim, ParamId, TensorOp};
use serde::{Deserialize, Serialize};

/// `modelspec.diagram.GlyphId`: the id of one symbol of the sprite, without its `ms-` prefix.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(transparent)]
pub struct GlyphId(pub String);

impl GlyphId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for GlyphId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// `modelspec.diagram.OperatorFamily`: the class of an operator, which fixes its glyph's outline
/// and accent colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub enum OperatorFamily {
    Elementwise,
    Contraction,
    Reduction,
    Normalization,
    Attention,
    Embedding,
    Layout,
    Stochastic,
    Loss,
    Optimizer,
    DataIo,
}

impl OperatorFamily {
    /// Every family, in the order of the specification.
    pub const ALL: [Self; 11] = [
        Self::Elementwise,
        Self::Contraction,
        Self::Reduction,
        Self::Normalization,
        Self::Attention,
        Self::Embedding,
        Self::Layout,
        Self::Stochastic,
        Self::Loss,
        Self::Optimizer,
        Self::DataIo,
    ];

    /// The family's name in a CSS custom property, `--ms-family-<name>`.
    pub fn css_name(self) -> &'static str {
        match self {
            Self::Elementwise => "elementwise",
            Self::Contraction => "contraction",
            Self::Reduction => "reduction",
            Self::Normalization => "normalization",
            Self::Attention => "attention",
            Self::Embedding => "embedding",
            Self::Layout => "layout",
            Self::Stochastic => "stochastic",
            Self::Loss => "loss",
            Self::Optimizer => "optimizer",
            Self::DataIo => "data-io",
        }
    }

    /// The outline every glyph of the family shares, and its dash pattern.
    fn outline(self) -> (&'static str, Option<&'static str>) {
        match self {
            Self::Elementwise => (CIRCLE, None),
            Self::Contraction => ("M3 3H21V21H3Z", None),
            Self::Reduction => ("M2 4H22L15 20H9Z", None),
            Self::Normalization => (
                "M7 3H17A4 4 0 0 1 21 7V17A4 4 0 0 1 17 21H7A4 4 0 0 1 3 17V7A4 4 0 0 1 7 3Z",
                None,
            ),
            Self::Attention => ("M2 12L7 3H17L22 12L17 21H7Z", None),
            Self::Embedding => ("M3 4H21V20H3ZM3 9H21M9 9V20", None),
            Self::Layout => ("M7 4H22L17 20H2Z", None),
            Self::Stochastic => (CIRCLE, Some(DOTTED)),
            Self::Loss => ("M12 1L23 12L12 23L1 12Z", None),
            Self::Optimizer => ("M8 2H16L22 8V16L16 22H8L2 16V8Z", None),
            Self::DataIo => (
                "M4 6A8 3 0 0 1 20 6A8 3 0 0 1 4 6ZM4 6V18A8 3 0 0 0 20 18V6",
                None,
            ),
        }
    }
}

/// `modelspec.diagram.PipelineStage`: a stage of a pipeline view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub enum PipelineStage {
    Data,
    Parameters,
    Forward,
    Loss,
    Backward,
    Optimizer,
}

impl PipelineStage {
    /// Every stage, in the order of the specification.
    pub const ALL: [Self; 6] = [
        Self::Data,
        Self::Parameters,
        Self::Forward,
        Self::Loss,
        Self::Backward,
        Self::Optimizer,
    ];

    /// The stage's glyph, `stage-<stage>`.
    pub fn glyph(self) -> GlyphId {
        GlyphId(format!("stage-{}", self.css_name()))
    }

    fn css_name(self) -> &'static str {
        match self {
            Self::Data => "data",
            Self::Parameters => "parameters",
            Self::Forward => "forward",
            Self::Loss => "loss",
            Self::Backward => "backward",
            Self::Optimizer => "optimizer",
        }
    }
}

/// `modelspec.diagram.EndpointRole`: what an Endpoint node of a pipeline view stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub enum EndpointRole {
    DataSource,
    CheckpointSource,
    CheckpointSink,
    PredictionSink,
}

impl EndpointRole {
    /// Every role, in the order of the specification.
    pub const ALL: [Self; 4] = [
        Self::DataSource,
        Self::CheckpointSource,
        Self::CheckpointSink,
        Self::PredictionSink,
    ];

    /// The role's glyph, `endpoint-<role>`.
    pub fn glyph(self) -> GlyphId {
        GlyphId(format!("endpoint-{}", self.css_name()))
    }

    fn css_name(self) -> &'static str {
        match self {
            Self::DataSource => "data-source",
            Self::CheckpointSource => "checkpoint-source",
            Self::CheckpointSink => "checkpoint-sink",
            Self::PredictionSink => "prediction-sink",
        }
    }
}

/// What an `Input` node is fed, which selects its glyph (DESIGN.md § 13.1). The graph builder
/// classifies an `Input` by the training step's dropout sites and the AdamW step's host inputs;
/// every other node passes `Data`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRole {
    /// A program input, a target or a batch tensor.
    Data,
    /// The RNG token of a dropout site.
    DropoutToken,
    /// A host input of an AdamW step.
    HostStep,
}

/// The glyphs of composite nodes (DESIGN.md § 13.1): a Function, Adjoint, Module or Layer scope
/// and the Update scope of each optimizer.
pub const SCOPE_GLYPHS: [&str; 7] = [
    "scope-function",
    "scope-adjoint",
    "scope-update-sgd",
    "scope-update-adamw",
    "scope-update-muon",
    "scope-module",
    "scope-layer",
];

/// One entry of the glyph set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glyph {
    pub id: GlyphId,
    /// The operator family whose outline and accent the glyph has; `None` for a scope, stage or
    /// endpoint glyph.
    pub family: Option<OperatorFamily>,
    /// For a gradient op's glyph, the glyph of its forward operator.
    pub gradient_of: Option<GlyphId>,
}

/// One op of each `TensorOp` variant, in declaration order (the order of
/// `modelspec.observatory.TensorOperator`). A variant added to `TensorOp` fails to compile in
/// [`glyph_of`] and [`family_of`] until it has a glyph, and `tests/glyphs.rs` fails until it is
/// listed here and in the specification.
pub fn operators() -> Vec<TensorOp> {
    vec![
        TensorOp::Input {
            name: String::new(),
        },
        TensorOp::ParamRead { param: ParamId(0) },
        TensorOp::Scalar(0.0),
        TensorOp::Add,
        TensorOp::Sub,
        TensorOp::Mul,
        TensorOp::MatMul,
        TensorOp::Relu,
        TensorOp::Transpose,
        TensorOp::ReduceSum,
        TensorOp::ReduceMean,
        TensorOp::Mse,
        TensorOp::ReluGrad,
        TensorOp::BroadcastScalar { shape: Vec::new() },
        TensorOp::Embedding,
        TensorOp::EmbeddingGrad {
            rows: Dim::Static(1),
        },
        TensorOp::AddRows,
        TensorOp::SumRows,
        TensorOp::Gelu,
        TensorOp::GeluGrad,
        TensorOp::LayerNorm,
        TensorOp::LayerNormGradX,
        TensorOp::LayerNormGradGamma,
        TensorOp::Softmax,
        TensorOp::SoftmaxGrad,
        TensorOp::Attention { heads: 1 },
        TensorOp::AttentionGradQ { heads: 1 },
        TensorOp::AttentionGradK { heads: 1 },
        TensorOp::AttentionGradV { heads: 1 },
        TensorOp::CrossEntropy,
        TensorOp::CrossEntropyGrad,
        TensorOp::Dropout { p: 0.0 },
        TensorOp::DropoutGrad { p: 0.0 },
        TensorOp::StopGradient,
        TensorOp::Sqrt,
        TensorOp::Div,
        TensorOp::Sigreg { slices: 1 },
        TensorOp::SigregColumns,
        TensorOp::SigregColumnsGrad,
    ]
}

/// The glyph of a Tensor IR operation: `op-` and its name in kebab case; an `Input` by `role`,
/// which every other op ignores.
pub fn glyph_of(op: &TensorOp, role: InputRole) -> GlyphId {
    let id = match op {
        TensorOp::Input { .. } => match role {
            InputRole::Data => "op-input",
            InputRole::DropoutToken => "op-input-token",
            InputRole::HostStep => "op-input-step",
        },
        TensorOp::ParamRead { .. } => "op-param-read",
        TensorOp::Scalar(_) => "op-scalar",
        TensorOp::Add => "op-add",
        TensorOp::Sub => "op-sub",
        TensorOp::Mul => "op-mul",
        TensorOp::MatMul => "op-mat-mul",
        TensorOp::Relu => "op-relu",
        TensorOp::Transpose => "op-transpose",
        TensorOp::ReduceSum => "op-reduce-sum",
        TensorOp::ReduceMean => "op-reduce-mean",
        TensorOp::Mse => "op-mse",
        TensorOp::ReluGrad => "op-relu-grad",
        TensorOp::BroadcastScalar { .. } => "op-broadcast-scalar",
        TensorOp::Embedding => "op-embedding",
        TensorOp::EmbeddingGrad { .. } => "op-embedding-grad",
        TensorOp::AddRows => "op-add-rows",
        TensorOp::SumRows => "op-sum-rows",
        TensorOp::Gelu => "op-gelu",
        TensorOp::GeluGrad => "op-gelu-grad",
        TensorOp::LayerNorm => "op-layer-norm",
        TensorOp::LayerNormGradX => "op-layer-norm-grad-x",
        TensorOp::LayerNormGradGamma => "op-layer-norm-grad-gamma",
        TensorOp::Softmax => "op-softmax",
        TensorOp::SoftmaxGrad => "op-softmax-grad",
        TensorOp::Attention { .. } => "op-attention",
        TensorOp::AttentionGradQ { .. } => "op-attention-grad-q",
        TensorOp::AttentionGradK { .. } => "op-attention-grad-k",
        TensorOp::AttentionGradV { .. } => "op-attention-grad-v",
        TensorOp::CrossEntropy => "op-cross-entropy",
        TensorOp::CrossEntropyGrad => "op-cross-entropy-grad",
        TensorOp::Dropout { .. } => "op-dropout",
        TensorOp::DropoutGrad { .. } => "op-dropout-grad",
        TensorOp::StopGradient => "op-stop-gradient",
        TensorOp::Sqrt => "op-sqrt",
        TensorOp::Div => "op-div",
        TensorOp::Sigreg { .. } => "op-sigreg",
        TensorOp::SigregColumns => "op-sigreg-columns",
        TensorOp::SigregColumnsGrad => "op-sigreg-columns-grad",
    };
    GlyphId(id.to_owned())
}

/// The family of a Tensor IR operation (DESIGN.md § 13.1); a gradient op is in its forward
/// operator's family.
pub fn family_of(op: &TensorOp) -> OperatorFamily {
    use OperatorFamily::*;
    match op {
        TensorOp::Input { .. } | TensorOp::ParamRead { .. } => DataIo,
        TensorOp::Scalar(_)
        | TensorOp::Add
        | TensorOp::Sub
        | TensorOp::Mul
        | TensorOp::Relu
        | TensorOp::ReluGrad
        | TensorOp::Gelu
        | TensorOp::GeluGrad
        | TensorOp::Sqrt
        | TensorOp::Div => Elementwise,
        TensorOp::MatMul => Contraction,
        TensorOp::ReduceSum | TensorOp::ReduceMean | TensorOp::SumRows => Reduction,
        TensorOp::LayerNorm
        | TensorOp::LayerNormGradX
        | TensorOp::LayerNormGradGamma
        | TensorOp::Softmax
        | TensorOp::SoftmaxGrad => Normalization,
        TensorOp::Attention { .. }
        | TensorOp::AttentionGradQ { .. }
        | TensorOp::AttentionGradK { .. }
        | TensorOp::AttentionGradV { .. } => Attention,
        TensorOp::Embedding | TensorOp::EmbeddingGrad { .. } | TensorOp::AddRows => Embedding,
        TensorOp::Transpose | TensorOp::BroadcastScalar { .. } | TensorOp::StopGradient => Layout,
        TensorOp::Dropout { .. } | TensorOp::DropoutGrad { .. } => Stochastic,
        TensorOp::Mse
        | TensorOp::CrossEntropy
        | TensorOp::CrossEntropyGrad
        | TensorOp::Sigreg { .. }
        | TensorOp::SigregColumns
        | TensorOp::SigregColumnsGrad => Loss,
    }
}

/// The glyph set, in id order.
pub fn glyphs() -> Vec<Glyph> {
    drawings()
        .into_iter()
        .map(|(id, drawing)| Glyph {
            id: GlyphId(id),
            family: match drawing.frame {
                Frame::Family(family) => Some(family),
                _ => None,
            },
            gradient_of: drawing.gradient_of.map(GlyphId),
        })
        .collect()
}

/// A colour scheme of the theme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    Light,
    Dark,
}

/// The custom properties of the theme and their values in `scheme`, in the order the sprite's
/// `<style>` sets them: `--ms-diagram-ink`, `--ms-diagram-paper`, `--ms-diagram-muted`, one
/// `--ms-family-<family>` per family and one `--ms-subgraph-<subgraph>` per Training IR subgraph.
pub fn theme(scheme: Scheme) -> Vec<(String, &'static str)> {
    let pick = |light: &'static str, dark: &'static str| match scheme {
        Scheme::Light => light,
        Scheme::Dark => dark,
    };
    let mut properties = vec![
        ("--ms-diagram-ink".to_owned(), pick(INK, "#e3e3e3")),
        ("--ms-diagram-paper".to_owned(), pick(PAPER, "#1b1b1d")),
        ("--ms-diagram-muted".to_owned(), pick(MUTED, "#a0a4ab")),
    ];
    for family in OperatorFamily::ALL {
        let (light, dark) = family_colours(family);
        properties.push((
            format!("--ms-family-{}", family.css_name()),
            pick(light, dark),
        ));
    }
    for (name, light, dark) in SUBGRAPHS {
        properties.push((format!("--ms-subgraph-{name}"), pick(light, dark)));
    }
    properties
}

/// The sprite: one `<svg>` holding a `<symbol id="ms-<id>" viewBox="0 0 24 24">` per glyph in id
/// order, after the `<style>` that sets the theme's light defaults and, under
/// `@media (prefers-color-scheme: dark)`, its dark ones. Every fill and stroke is a custom
/// property with a fallback, so a page that sets the properties themes every `<use>` of a symbol.
pub fn sprite() -> String {
    let mut out = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" class=\"ms-glyphs\" width=\"0\" height=\"0\" aria-hidden=\"true\"><style>",
    );
    out.push_str(&properties_rule(Scheme::Light));
    out.push_str("@media (prefers-color-scheme: dark){");
    out.push_str(&properties_rule(Scheme::Dark));
    out.push_str("}</style>");
    for (id, drawing) in drawings() {
        out.push_str(&symbol(&id, &drawing));
    }
    out.push_str("</svg>");
    out
}

const CIRCLE: &str = "M12 2A10 10 0 0 1 12 22A10 10 0 0 1 12 2Z";
/// The Stochastic outline: dots, 16 of them on a circle measured as 64 units.
const DOTTED: &str = "0 4";
/// A gradient op's outline: dashes, 8 on an outline measured as 64 units.
const DASHED: &str = "5 3";
/// The `∇` badge of a gradient op, in the top right.
const NABLA: &str = "M14 3H22L18 10Z";

const INK: &str = "#1c1e21";
const PAPER: &str = "#ffffff";
const MUTED: &str = "#606770";

/// The Training IR subgraphs (`modelspec.observatory.Subgraph`) and their light and dark colours.
const SUBGRAPHS: [(&str, &str, &str); 4] = [
    ("forward", "#0369a1", "#38bdf8"),
    ("loss", "#be123c", "#fb7185"),
    ("gradient", "#a21caf", "#e879f9"),
    ("optimizer", "#c2410c", "#fb923c"),
];

fn family_colours(family: OperatorFamily) -> (&'static str, &'static str) {
    match family {
        OperatorFamily::Elementwise => ("#2563eb", "#60a5fa"),
        OperatorFamily::Contraction => ("#4f46e5", "#a5b4fc"),
        OperatorFamily::Reduction => ("#0f766e", "#2dd4bf"),
        OperatorFamily::Normalization => ("#0e7490", "#67e8f9"),
        OperatorFamily::Attention => ("#9333ea", "#c084fc"),
        OperatorFamily::Embedding => ("#15803d", "#4ade80"),
        OperatorFamily::Layout => ("#a16207", "#facc15"),
        OperatorFamily::Stochastic => ("#db2777", "#f472b6"),
        OperatorFamily::Loss => ("#dc2626", "#f87171"),
        OperatorFamily::Optimizer => ("#ea580c", "#fdba74"),
        OperatorFamily::DataIo => ("#475569", "#94a3b8"),
    }
}

fn properties_rule(scheme: Scheme) -> String {
    let mut rule = String::from(":where(.ms-glyphs){");
    for (property, value) in theme(scheme) {
        rule.push_str(&format!("{property}:{value};"));
    }
    rule.push('}');
    rule
}

fn ink() -> String {
    format!("var(--ms-diagram-ink, var(--ifm-font-color-base, {INK}))")
}

fn paper() -> String {
    format!("var(--ms-diagram-paper, var(--ifm-background-color, {PAPER}))")
}

/// Where a glyph's outline and accent come from.
#[derive(Clone, Copy)]
enum Frame {
    Family(OperatorFamily),
    /// A composite, accented by the subgraph (or muted) it belongs to.
    Scope(Accent),
    Stage(PipelineStage),
    Endpoint,
}

#[derive(Clone, Copy)]
enum Accent {
    Subgraph(usize),
    Muted,
}

impl Frame {
    fn outline(self) -> (&'static str, Option<&'static str>) {
        match self {
            Frame::Family(family) => family.outline(),
            Frame::Scope(_) => ("M3 7H17V21H3ZM7 7V3H21V17H17", None),
            Frame::Stage(_) => ("M2 5H16L22 12L16 19H2L6 12Z", None),
            Frame::Endpoint => ("M5 2H14L19 7V22H5ZM14 2V7H19", None),
        }
    }

    fn accent(self) -> String {
        let subgraph = |i: usize| {
            let (name, light, _) = SUBGRAPHS[i];
            format!("var(--ms-subgraph-{name}, {light})")
        };
        let muted = || format!("var(--ms-diagram-muted, var(--ifm-color-emphasis-600, {MUTED}))");
        let family = |family: OperatorFamily| {
            format!(
                "var(--ms-family-{}, {})",
                family.css_name(),
                family_colours(family).0
            )
        };
        match self {
            Frame::Family(f) => family(f),
            Frame::Scope(Accent::Subgraph(i)) => subgraph(i),
            Frame::Scope(Accent::Muted) => muted(),
            Frame::Stage(stage) => match stage {
                PipelineStage::Data | PipelineStage::Parameters => family(OperatorFamily::DataIo),
                PipelineStage::Forward => subgraph(0),
                PipelineStage::Loss => subgraph(1),
                PipelineStage::Backward => subgraph(2),
                PipelineStage::Optimizer => subgraph(3),
            },
            Frame::Endpoint => family(OperatorFamily::DataIo),
        }
    }
}

/// The mark inside an outline: `line` is stroked in ink, `solid` filled in ink.
#[derive(Clone, Copy)]
struct Mark {
    line: &'static str,
    solid: &'static str,
}

const fn line(line: &'static str) -> Mark {
    Mark { line, solid: "" }
}

const fn mark(line: &'static str, solid: &'static str) -> Mark {
    Mark { line, solid }
}

/// One glyph to draw.
struct Drawing {
    frame: Frame,
    mark: Mark,
    /// A gradient op: dashed outline, `∇` badge, and the slot of the operand it differentiates
    /// as `(index, operand count)` when there are several.
    gradient: Option<Option<(u32, u32)>>,
    gradient_of: Option<String>,
}

impl Drawing {
    fn plain(frame: Frame, mark: Mark) -> Self {
        Drawing {
            frame,
            mark,
            gradient: None,
            gradient_of: None,
        }
    }
}

/// The mark of a forward operator, or of a gradient op's forward operator.
fn op_mark(op: &TensorOp, role: InputRole) -> Mark {
    match op {
        TensorOp::Input { .. } => match role {
            InputRole::Data => line("M12 10V18M9 15L12 18L15 15"),
            InputRole::DropoutToken => mark(
                "M9 7H15A2 2 0 0 1 17 9V15A2 2 0 0 1 15 17H9A2 2 0 0 1 7 15V9A2 2 0 0 1 9 7Z",
                "M9 10A1 1 0 0 1 11 10A1 1 0 0 1 9 10ZM13 14A1 1 0 0 1 15 14A1 1 0 0 1 13 14Z",
            ),
            InputRole::HostStep => line("M6 17H10V13H14V9H18"),
        },
        TensorOp::ParamRead { .. } => line("M12 10A3 4 0 0 1 12 18A3 4 0 0 1 12 10ZM9 14H15"),
        TensorOp::Scalar(_) => mark("", "M9 12A3 3 0 0 1 15 12A3 3 0 0 1 9 12Z"),
        TensorOp::Add => line("M12 7V17M7 12H17"),
        TensorOp::Sub => line("M7 12H17"),
        TensorOp::Mul => line("M8 8L16 16M16 8L8 16"),
        TensorOp::MatMul => mark(
            "M6 12H18",
            "M5 12A3 3 0 0 1 11 12A3 3 0 0 1 5 12ZM13 12A3 3 0 0 1 19 12A3 3 0 0 1 13 12Z",
        ),
        TensorOp::Relu | TensorOp::ReluGrad => line("M6 15H12L17 8"),
        TensorOp::Transpose => line("M7 8H17M12 8V17"),
        TensorOp::ReduceSum => line("M15 7H9L12 10L9 13H15"),
        TensorOp::ReduceMean => line("M9 6H15M10 9L14 13M14 9L10 13"),
        TensorOp::Mse => line("M8 8Q12 22 16 8"),
        TensorOp::BroadcastScalar { .. } => mark(
            "M10 12L16 8M10 12H17M10 12L16 16",
            "M6 12A2 2 0 0 1 10 12A2 2 0 0 1 6 12Z",
        ),
        TensorOp::Embedding | TensorOp::EmbeddingGrad { .. } => {
            mark("", "M5 15A1 1 0 0 1 7 15A1 1 0 0 1 5 15ZM12 13H19V17H12Z")
        }
        TensorOp::AddRows => line("M15 11V19M11 15H19"),
        TensorOp::SumRows => line("M7 7H17M8 10H16M12 12V17M10 15L12 17L14 15"),
        TensorOp::Gelu | TensorOp::GeluGrad => line("M5 12C8 12 8 16 11 16S15 10 18 6"),
        TensorOp::LayerNorm | TensorOp::LayerNormGradX | TensorOp::LayerNormGradGamma => {
            line("M6 16C9 16 9 8 12 8S15 16 18 16")
        }
        TensorOp::Softmax | TensorOp::SoftmaxGrad => line("M8 17V13M12 17V7M16 17V11"),
        TensorOp::Attention { .. }
        | TensorOp::AttentionGradQ { .. }
        | TensorOp::AttentionGradK { .. }
        | TensorOp::AttentionGradV { .. } => mark(
            "M5 12C8 7 16 7 19 12C16 17 8 17 5 12Z",
            "M10 12A2 2 0 0 1 14 12A2 2 0 0 1 10 12Z",
        ),
        TensorOp::CrossEntropy | TensorOp::CrossEntropyGrad => line("M8 7Q9 16 17 16"),
        TensorOp::Dropout { .. } | TensorOp::DropoutGrad { .. } => {
            line("M7 7V17M17 7V17M10 10L14 14M14 10L10 14")
        }
        TensorOp::StopGradient => line("M5 12H14M11 9L14 12L11 15M18 7V17"),
        TensorOp::Sqrt => line("M6 13L8 12L10 17L13 7H18"),
        TensorOp::Div => mark(
            "M7 12H17",
            "M11 8A1 1 0 0 1 13 8A1 1 0 0 1 11 8ZM11 16A1 1 0 0 1 13 16A1 1 0 0 1 11 16Z",
        ),
        TensorOp::Sigreg { .. } => line("M7 15C10 15 10 9 12 9S14 15 17 15M7 17H17"),
        TensorOp::SigregColumns | TensorOp::SigregColumnsGrad => {
            line("M7 15C10 15 10 9 12 9S14 15 17 15M9 17V19M12 17V19M15 17V19")
        }
    }
}

/// For a gradient op, its forward operator and the slot of the operand it differentiates when the
/// forward operator has several differentiable operands; `None` for every other op.
fn gradient(op: &TensorOp) -> Option<(TensorOp, Option<(u32, u32)>)> {
    match op {
        TensorOp::ReluGrad => Some((TensorOp::Relu, None)),
        TensorOp::EmbeddingGrad { .. } => Some((TensorOp::Embedding, None)),
        TensorOp::GeluGrad => Some((TensorOp::Gelu, None)),
        TensorOp::LayerNormGradX => Some((TensorOp::LayerNorm, Some((0, 2)))),
        TensorOp::LayerNormGradGamma => Some((TensorOp::LayerNorm, Some((1, 2)))),
        TensorOp::SoftmaxGrad => Some((TensorOp::Softmax, None)),
        TensorOp::AttentionGradQ { heads } => {
            Some((TensorOp::Attention { heads: *heads }, Some((0, 3))))
        }
        TensorOp::AttentionGradK { heads } => {
            Some((TensorOp::Attention { heads: *heads }, Some((1, 3))))
        }
        TensorOp::AttentionGradV { heads } => {
            Some((TensorOp::Attention { heads: *heads }, Some((2, 3))))
        }
        TensorOp::CrossEntropyGrad => Some((TensorOp::CrossEntropy, None)),
        TensorOp::DropoutGrad { p } => Some((TensorOp::Dropout { p: *p }, None)),
        TensorOp::SigregColumnsGrad => Some((TensorOp::SigregColumns, None)),
        TensorOp::Input { .. }
        | TensorOp::ParamRead { .. }
        | TensorOp::Scalar(_)
        | TensorOp::Add
        | TensorOp::Sub
        | TensorOp::Mul
        | TensorOp::MatMul
        | TensorOp::Relu
        | TensorOp::Transpose
        | TensorOp::ReduceSum
        | TensorOp::ReduceMean
        | TensorOp::Mse
        | TensorOp::BroadcastScalar { .. }
        | TensorOp::Embedding
        | TensorOp::AddRows
        | TensorOp::SumRows
        | TensorOp::Gelu
        | TensorOp::LayerNorm
        | TensorOp::Softmax
        | TensorOp::Attention { .. }
        | TensorOp::CrossEntropy
        | TensorOp::Dropout { .. }
        | TensorOp::StopGradient
        | TensorOp::Sqrt
        | TensorOp::Div
        | TensorOp::Sigreg { .. }
        | TensorOp::SigregColumns => None,
    }
}

/// Every glyph, sorted by id.
fn drawings() -> Vec<(String, Drawing)> {
    let mut all = Vec::new();
    for op in operators() {
        let roles: &[InputRole] = match op {
            TensorOp::Input { .. } => &[
                InputRole::Data,
                InputRole::DropoutToken,
                InputRole::HostStep,
            ],
            _ => &[InputRole::Data],
        };
        for &role in roles {
            let id = glyph_of(&op, role).0;
            let family = match role {
                InputRole::Data => family_of(&op),
                InputRole::DropoutToken => OperatorFamily::Stochastic,
                InputRole::HostStep => OperatorFamily::Optimizer,
            };
            let drawing = match gradient(&op) {
                Some((forward, slot)) => Drawing {
                    frame: Frame::Family(family),
                    mark: op_mark(&forward, role),
                    gradient: Some(slot),
                    gradient_of: Some(glyph_of(&forward, role).0),
                },
                None => Drawing::plain(Frame::Family(family), op_mark(&op, role)),
            };
            all.push((id, drawing));
        }
    }
    let scopes = [
        (
            SCOPE_GLYPHS[0],
            Accent::Subgraph(0),
            line("M12 10C10 9 9 10 9 12V17C9 19 8 19 6 18M7 13H12"),
        ),
        (
            SCOPE_GLYPHS[1],
            Accent::Subgraph(2),
            line("M6 11H14L10 18Z"),
        ),
        (
            SCOPE_GLYPHS[2],
            Accent::Subgraph(3),
            line("M6 10L13 17M9 17H13V13"),
        ),
        (
            SCOPE_GLYPHS[3],
            Accent::Subgraph(3),
            line("M6 10C6 15 8 17 13 17M11 15L13 17L11 19"),
        ),
        (
            SCOPE_GLYPHS[4],
            Accent::Subgraph(3),
            line("M10 10V17M6 17H14"),
        ),
        (
            SCOPE_GLYPHS[5],
            Accent::Muted,
            line("M6 10H14V18H6ZM10 10V18M6 14H14"),
        ),
        (
            SCOPE_GLYPHS[6],
            Accent::Muted,
            line("M6 11H14M6 14H14M6 17H14"),
        ),
    ];
    for (id, accent, m) in scopes {
        all.push((id.to_owned(), Drawing::plain(Frame::Scope(accent), m)));
    }
    for stage in PipelineStage::ALL {
        let m = match stage {
            PipelineStage::Data => {
                line("M7 9A4 2 0 0 1 15 9A4 2 0 0 1 7 9ZM7 9V15A4 2 0 0 0 15 15V9")
            }
            PipelineStage::Parameters => line("M11 8A3 4 0 0 1 11 16A3 4 0 0 1 11 8ZM8 12H14"),
            PipelineStage::Forward => line("M7 12H14M11 9L14 12L11 15"),
            PipelineStage::Loss => line("M11 8L15 12L11 16L7 12Z"),
            PipelineStage::Backward => line("M7 9H15L11 15Z"),
            PipelineStage::Optimizer => line("M7 9L13 15M9 15H13V11"),
        };
        all.push((stage.glyph().0, Drawing::plain(Frame::Stage(stage), m)));
    }
    for role in EndpointRole::ALL {
        let m = match role {
            EndpointRole::DataSource => line("M8 11H12M8 14H16M8 17H16"),
            EndpointRole::CheckpointSource => line("M12 17V10M9 13L12 10L15 13M8 19H16"),
            EndpointRole::CheckpointSink => line("M12 10V17M9 14L12 17L15 14M8 19H16"),
            EndpointRole::PredictionSink => line("M9 19V15M12 19V11M15 19V13"),
        };
        all.push((role.glyph().0, Drawing::plain(Frame::Endpoint, m)));
    }
    all.sort_by(|a, b| a.0.cmp(&b.0));
    all
}

fn symbol(id: &str, drawing: &Drawing) -> String {
    let (d, family_dash) = drawing.frame.outline();
    let accent = drawing.frame.accent();
    let dash = match drawing.gradient {
        Some(_) => Some(DASHED),
        None => family_dash,
    };
    let dash = match dash {
        Some(pattern) => format!(" pathLength=\"64\" stroke-dasharray=\"{pattern}\""),
        None => String::new(),
    };
    let mut out = format!(
        "<symbol id=\"ms-{id}\" viewBox=\"0 0 24 24\"><path class=\"ms-outline\" d=\"{d}\" fill=\"{accent}\" fill-opacity=\"14%\" stroke=\"{accent}\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"{dash}/>"
    );
    let ink = ink();
    out.push_str(&format!(
        "<g class=\"ms-mark\" fill=\"var(--ms-diagram-mark-fill, none)\" stroke=\"{ink}\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\">"
    ));
    if !drawing.mark.line.is_empty() {
        out.push_str(&format!("<path d=\"{}\"/>", drawing.mark.line));
    }
    if !drawing.mark.solid.is_empty() {
        out.push_str(&format!(
            "<path d=\"{}\" fill=\"{ink}\" stroke-width=\"0\"/>",
            drawing.mark.solid
        ));
    }
    out.push_str("</g>");
    if let Some(slot) = drawing.gradient {
        if let Some((index, count)) = slot {
            out.push_str(&slots(index, count));
        }
        let gradient = format!("var(--ms-subgraph-gradient, {})", SUBGRAPHS[2].1);
        out.push_str(&format!(
            "<path class=\"ms-grad\" d=\"{NABLA}\" fill=\"{}\" stroke=\"{gradient}\" stroke-width=\"2\" stroke-linejoin=\"round\"/>",
            paper()
        ));
    }
    out.push_str("</symbol>");
    out
}

/// The operand slots of a gradient op along the bottom: the differentiated operand's slot in the
/// gradient colour, the others muted.
fn slots(index: u32, count: u32) -> String {
    let mut out = String::from("<g class=\"ms-slots\" stroke-width=\"0\">");
    let first = 12 - 2 * (count - 1);
    for i in 0..count {
        let x = first + 4 * i;
        let colour = if i == index {
            format!("var(--ms-subgraph-gradient, {})", SUBGRAPHS[2].1)
        } else {
            format!("var(--ms-diagram-muted, var(--ifm-color-emphasis-600, {MUTED}))")
        };
        out.push_str(&format!(
            "<path d=\"M{} 19A1 1 0 0 1 {} 19A1 1 0 0 1 {} 19Z\" fill=\"{colour}\"/>",
            x - 1,
            x + 1,
            x - 1
        ));
    }
    out.push_str("</g>");
    out
}
