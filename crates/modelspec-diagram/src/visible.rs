//! The visible graph of a render request (DESIGN.md § 13.3): what one level of a graph draws
//! before it is laid out.
//!
//! A request names a root scope, a depth and the scopes expanded beyond it. A scope inside the
//! root is open when it lies within `depth` levels of the root, or it or a scope inside it is
//! listed in `expanded`, so expanding a nested scope opens its ancestors too. Every node of the
//! root whose scopes below the root are all open is drawn as itself, an Operator node; the
//! outermost scope below the root that is not open is drawn instead, as one Composite node with
//! the scope's ports. When the root is not the Program scope, each of its own ports is a Boundary
//! node: an In port above the nodes that read the value, an Out port below the node that
//! defines it.
//!
//! Each edge of the graph becomes an edge between the visible nodes standing for its producer
//! and its consumer: into a composite at the In port of its value, out of one at the Out port of
//! its value; an edge inside one composite is not drawn. Edges with the same producer, port,
//! consumer and slot merge into one with a count, which the SVG labels `×n`.

use std::collections::{BTreeMap, BTreeSet};

use crate::graph::within;
use crate::layout::{RenderRequest, default_depth};
use crate::model::{DiagramGraph, DiagramNode, NodeKind, Port, PortDirection, Scope, ScopeKey};

/// What a visible node stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// One Tensor IR node of the root or an open scope.
    Operator,
    /// A collapsed scope.
    Composite,
    /// A port of the root scope, the value entering (In) or leaving (Out) the level.
    Boundary(PortDirection),
}

/// One node a level draws.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleNode {
    /// The graph's node for an Operator; for a Composite a node of kind Composite, its key the
    /// scope key; for a Boundary a node of kind Endpoint keyed `in:%<k>` or `out:%<k>`, labelled
    /// `in %<k>` or `out %<k>` over the second line of the defining node's label.
    pub node: DiagramNode,
    pub role: Role,
    /// The ValueId the node orders by: an Operator's own, the lowest one a Composite holds, the
    /// value a Boundary carries.
    pub value: i64,
    /// A Composite's In ports in port order; an Out Boundary's one port.
    pub inputs: Vec<Port>,
    /// A Composite's Out ports in port order; an In Boundary's one port.
    pub outputs: Vec<Port>,
    /// The slots of the node's top edge: an Operator's operands, a Composite's In ports, 1 for a
    /// Boundary.
    pub slots: i64,
}

/// One edge a level draws, from a visible node to a visible node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleEdge {
    /// Indices into [`Visible::nodes`].
    pub producer: usize,
    pub consumer: usize,
    /// The ValueId the edge carries.
    pub value: i64,
    /// The producer's Out port the edge leaves at, when the producer is a Composite.
    pub out_port: Option<usize>,
    /// The slot it enters: the operand of an Operator, the In port of a Composite, 0 on a
    /// Boundary.
    pub slot: i64,
    /// How many edges of the graph it stands for; 1 into an Out boundary, which a value leaves once.
    pub count: usize,
    /// The first of them, an index into `graph.edges`.
    pub edge: usize,
}

/// The visible graph of one request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Visible {
    pub root: ScopeKey,
    pub depth: u32,
    /// The request's expanded scopes, sorted and without repeats.
    pub expanded: Vec<ScopeKey>,
    /// In the graph's node order, a Composite where its first node stands, a Boundary In where
    /// its value's producer stands and a Boundary Out right after its producer.
    pub nodes: Vec<VisibleNode>,
    /// In the graph's edge order, by the first edge each stands for.
    pub edges: Vec<VisibleEdge>,
}

/// The root a request names: its root, else the graph's Program scope.
pub fn root_of(graph: &DiagramGraph, request: &RenderRequest) -> ScopeKey {
    request.root.clone().unwrap_or_else(|| {
        graph
            .scopes
            .iter()
            .find(|s| s.parent.is_none())
            .map(|s| s.key.clone())
            .unwrap_or_else(|| ScopeKey(graph.program.clone()))
    })
}

/// The visible graph of `graph` that `request` asks for.
pub fn visible(graph: &DiagramGraph, request: &RenderRequest) -> Visible {
    let root = root_of(graph, request);
    let depth = request.depth.unwrap_or_else(|| default_depth(graph.view));
    let expanded: Vec<ScopeKey> = request
        .expanded
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let scopes: BTreeMap<&ScopeKey, &Scope> = graph.scopes.iter().map(|s| (&s.key, s)).collect();
    let levels_below =
        |scope: &ScopeKey| -> u32 { scope.0[root.0.len()..].matches('/').count() as u32 };
    let open = |scope: &ScopeKey| -> bool {
        levels_below(scope) <= depth || expanded.iter().any(|e| within(e, scope))
    };
    // The visible stand-in of a scope inside the root: None when it is open all the way,
    // else the outermost scope below the root that is not open.
    let collapsed_at = |scope: &ScopeKey| -> Option<ScopeKey> {
        let mut chain = Vec::new();
        let mut at = Some(scope.clone());
        while let Some(key) = at {
            if key == root {
                break;
            }
            at = scopes.get(&key).and_then(|s| s.parent.clone());
            chain.push(key);
        }
        chain.into_iter().rev().find(|key| !open(key))
    };
    let root_ports: &[Port] = scopes.get(&root).map_or(&[], |s| s.ports.as_slice());
    let root_port = |direction: PortDirection, value: i64| {
        root_ports
            .iter()
            .find(|p| p.direction == direction && p.value_id == value)
    };
    let mut slots: BTreeMap<&str, i64> = BTreeMap::new();
    for edge in &graph.edges {
        let n = slots.entry(edge.consumer.as_str()).or_insert(0);
        *n = (*n).max(edge.operand + 1);
    }

    let mut nodes: Vec<VisibleNode> = Vec::new();
    let mut by_key: BTreeMap<String, usize> = BTreeMap::new();
    // The visible node standing for each graph node inside the root.
    let mut stand_in: BTreeMap<usize, usize> = BTreeMap::new();
    let mut boundary_in: BTreeMap<i64, usize> = BTreeMap::new();
    let mut boundary_out: BTreeMap<i64, usize> = BTreeMap::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        let value = node.value_id.unwrap_or(i64::MAX);
        if !within(&node.scope, &root) {
            if let Some(port) = root_port(PortDirection::In, value) {
                boundary_in.insert(value, nodes.len());
                nodes.push(boundary(node, port, &root));
            }
            continue;
        }
        let at = match collapsed_at(&node.scope) {
            None => {
                nodes.push(VisibleNode {
                    node: node.clone(),
                    role: Role::Operator,
                    value,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    slots: slots
                        .get(node.node_key.as_str())
                        .copied()
                        .unwrap_or(1)
                        .max(1),
                });
                nodes.len() - 1
            }
            Some(scope) => match by_key.get(&scope.0) {
                Some(&at) => {
                    nodes[at].value = nodes[at].value.min(value);
                    at
                }
                None => {
                    let composite = composite(scopes[&scope], value);
                    by_key.insert(scope.0.clone(), nodes.len());
                    nodes.push(composite);
                    nodes.len() - 1
                }
            },
        };
        stand_in.insert(i, at);
        if let Some(port) = root_port(PortDirection::Out, value) {
            boundary_out.insert(value, nodes.len());
            nodes.push(boundary(node, port, &root));
        }
    }

    let index: BTreeMap<&str, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.node_key.as_str(), i))
        .collect();
    let mut edges: Vec<VisibleEdge> = Vec::new();
    let mut merged: BTreeMap<(usize, Option<usize>, usize, i64), usize> = BTreeMap::new();
    for (e, edge) in graph.edges.iter().enumerate() {
        let (Some(&p), Some(&c)) = (
            index.get(edge.producer.as_str()),
            index.get(edge.consumer.as_str()),
        ) else {
            continue;
        };
        let value = graph.nodes[p].value_id.unwrap_or(i64::MAX);
        let producer = match stand_in.get(&p) {
            Some(&v) => v,
            None => match boundary_in.get(&value) {
                Some(&v) => v,
                None => continue,
            },
        };
        let consumer = match stand_in.get(&c) {
            Some(&v) => v,
            None => match boundary_out.get(&value) {
                Some(&v) if stand_in.contains_key(&p) => v,
                _ => continue,
            },
        };
        if producer == consumer {
            continue;
        }
        let out_port = (nodes[producer].role == Role::Composite)
            .then(|| {
                nodes[producer]
                    .outputs
                    .iter()
                    .position(|port| port.value_id == value)
            })
            .flatten();
        let slot = match nodes[consumer].role {
            Role::Operator => edge.operand,
            Role::Composite => nodes[consumer]
                .inputs
                .iter()
                .position(|port| port.value_id == value)
                .map_or(0, |k| k as i64),
            Role::Boundary(_) => 0,
        };
        let key = (producer, out_port, consumer, slot);
        match merged.get(&key) {
            // A value leaving the level is one exit, however many readers it has outside.
            Some(_) if matches!(nodes[consumer].role, Role::Boundary(_)) => {}
            Some(&k) => edges[k].count += 1,
            None => {
                merged.insert(key, edges.len());
                edges.push(VisibleEdge {
                    producer,
                    consumer,
                    value,
                    out_port,
                    slot,
                    count: 1,
                    edge: e,
                });
            }
        }
    }
    Visible {
        root,
        depth,
        expanded,
        nodes,
        edges,
    }
}

/// The Composite node of a collapsed `scope`, `value` the first ValueId seen in it.
fn composite(scope: &Scope, value: i64) -> VisibleNode {
    let (inputs, outputs): (Vec<Port>, Vec<Port>) = scope
        .ports
        .iter()
        .cloned()
        .partition(|p| p.direction == PortDirection::In);
    let segment = scope.path.last();
    let name = segment.map_or(scope.key.0.as_str(), |s| s.name.as_str());
    let instance = segment.map_or(0, |s| s.instance);
    let label = format!(
        "{name}\ninstance {instance}, {} in, {} out",
        inputs.len(),
        outputs.len()
    );
    VisibleNode {
        node: DiagramNode {
            node_key: scope.key.0.clone(),
            kind: NodeKind::Composite,
            scope: scope.parent.clone().unwrap_or_else(|| scope.key.clone()),
            composite_of: Some(scope.key.clone()),
            value_id: None,
            op: None,
            family: None,
            glyph: scope.glyph.clone(),
            label,
            dtype: None,
            shape: None,
            subgraph: None,
            endpoint: None,
            source_span: segment.and_then(|s| s.call_site.clone()),
        },
        role: Role::Composite,
        value,
        slots: inputs.len().max(1) as i64,
        inputs,
        outputs,
    }
}

/// The Boundary node of the root's `port`, whose value `producer` defines.
fn boundary(producer: &DiagramNode, port: &Port, root: &ScopeKey) -> VisibleNode {
    let word = match port.direction {
        PortDirection::In => "in",
        PortDirection::Out => "out",
    };
    let (_, second) = producer
        .label
        .split_once('\n')
        .unwrap_or((&producer.label, ""));
    let value = port.value_id;
    let (inputs, outputs) = match port.direction {
        PortDirection::In => (Vec::new(), vec![port.clone()]),
        PortDirection::Out => (vec![port.clone()], Vec::new()),
    };
    VisibleNode {
        node: DiagramNode {
            node_key: format!("{word}:%{value}"),
            kind: NodeKind::Endpoint,
            scope: root.clone(),
            composite_of: None,
            value_id: None,
            op: None,
            family: None,
            glyph: producer.glyph.clone(),
            label: format!("{word} %{value}\n{second}"),
            dtype: Some(port.dtype),
            shape: Some(port.shape.clone()),
            subgraph: None,
            endpoint: None,
            source_span: None,
        },
        role: Role::Boundary(port.direction),
        value,
        inputs,
        outputs,
        slots: 1,
    }
}
