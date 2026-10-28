//! The records of `spec/domains/diagram.yaml` (`modelspec.diagram`) as Rust types, hand-written
//! beside the schema `ess generate` writes under `schema/` and held to it by
//! `tests/render.rs` (`diagram.json-validates`). Field names are the domain's; an optional field
//! is absent from the JSON, never `null`. [`GlyphId`], [`OperatorFamily`], [`PipelineStage`] and
//! [`EndpointRole`] are the glyph set's.

use serde::{Deserialize, Serialize};

pub use crate::glyph::{EndpointRole, GlyphId, OperatorFamily, PipelineStage};

/// `modelspec.diagram.DiagramGraphId`: the view's lower-case name, a slash and the graph's
/// `ir_hash` (`model/<ir_hash>`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(transparent)]
pub struct DiagramGraphId(pub String);

/// `modelspec.diagram.DiagramRenderId`: derived from the graph id, root, depth and expanded
/// scopes, `<graph_id>#<root>@<depth>` and `+<scope>` per expanded scope.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(transparent)]
pub struct DiagramRenderId(pub String);

/// `modelspec.diagram.ScopeKey`: the canonical text of a scope path.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(transparent)]
pub struct ScopeKey(pub String);

/// `modelspec.diagram.DiagramView`: what a graph shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub enum DiagramView {
    Model,
    Train,
    Eval,
    Infer,
}

impl DiagramView {
    /// The view's lower-case name, as a graph id and `--view` write it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Train => "train",
            Self::Eval => "eval",
            Self::Infer => "infer",
        }
    }
}

/// `modelspec.diagram.ScopeKind`: what one segment of a scope path names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub enum ScopeKind {
    Program,
    Stage,
    Function,
    Adjoint,
    Update,
    Module,
    Layer,
}

/// `modelspec.diagram.ScopeSegment`: one step of a scope path, outermost first.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct ScopeSegment {
    pub kind: ScopeKind,
    pub name: String,
    pub instance: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub call_site: Option<String>,
}

/// `modelspec.checkpoint.DType`: a tensor element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub enum DType {
    F32,
    I32,
}

impl DType {
    /// The element type as Tensor IR prints it, `f32` or `i32`.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
        }
    }
}

/// `modelspec.observatory.Subgraph`: the Training IR subgraph of a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub enum Subgraph {
    Forward,
    Loss,
    Gradient,
    Optimizer,
}

/// `modelspec.diagram.NodeKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub enum NodeKind {
    Operator,
    Composite,
    Endpoint,
}

/// `modelspec.diagram.DiagramNode`: one node of a graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct DiagramNode {
    /// `%<k>` for an Operator node, the scope key for a Composite, `endpoint:<role>` for an
    /// Endpoint.
    pub node_key: String,
    pub kind: NodeKind,
    pub scope: ScopeKey,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub composite_of: Option<ScopeKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub value_id: Option<i64>,
    /// The `modelspec.observatory.TensorOperator`, the Tensor IR op's name (`TensorOp::name`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub op: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub family: Option<OperatorFamily>,
    pub glyph: GlyphId,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub dtype: Option<DType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub shape: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub subgraph: Option<Subgraph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub endpoint: Option<EndpointRole>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub source_span: Option<String>,
}

/// `modelspec.diagram.DiagramEdge`: one operand, from the node that defines it to the reader.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct DiagramEdge {
    pub producer: String,
    pub consumer: String,
    pub operand: i64,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

/// `modelspec.diagram.PortDirection`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub enum PortDirection {
    In,
    Out,
}

/// `modelspec.diagram.Port`: a tensor crossing a scope's boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct Port {
    pub direction: PortDirection,
    pub value_id: i64,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

/// `modelspec.diagram.Scope`: one scope of the graph's scope tree.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct Scope {
    pub key: ScopeKey,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub parent: Option<ScopeKey>,
    pub path: Vec<ScopeSegment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub stage: Option<PipelineStage>,
    pub glyph: GlyphId,
    pub ports: Vec<Port>,
}

/// `modelspec.diagram.DiagramGraph.State`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub enum DiagramGraphState {
    #[default]
    Derived,
}

/// `modelspec.diagram.DiagramGraph`: the whole diagram of one program in one view.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct DiagramGraph {
    pub graph_id: DiagramGraphId,
    pub view: DiagramView,
    pub program: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub function: Option<String>,
    pub source_hash: String,
    pub ir_hash: String,
    pub scopes: Vec<Scope>,
    pub nodes: Vec<DiagramNode>,
    pub edges: Vec<DiagramEdge>,
    pub state: DiagramGraphState,
}

/// `modelspec.diagram.NodeBox`: where a visible node is drawn, in integer units.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct NodeBox {
    pub node_key: String,
    pub x: i64,
    pub y: i64,
    pub width: i64,
    pub height: i64,
    pub rank: i64,
}

/// `modelspec.diagram.EdgeRoute`: the polyline of one visible edge, `x0, y0, x1, y1, …`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct EdgeRoute {
    pub producer: String,
    pub consumer: String,
    pub operand: i64,
    pub points: Vec<i64>,
}

/// `modelspec.diagram.DiagramRender.State`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub enum DiagramRenderState {
    #[default]
    LaidOut,
}

/// `modelspec.diagram.DiagramRender`: one laid-out projection of a graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct DiagramRender {
    pub render_id: DiagramRenderId,
    pub graph_id: DiagramGraphId,
    pub root: ScopeKey,
    pub depth: i64,
    pub expanded: Vec<ScopeKey>,
    pub width: i64,
    pub height: i64,
    pub boxes: Vec<NodeBox>,
    pub routes: Vec<EdgeRoute>,
    pub state: DiagramRenderState,
}
