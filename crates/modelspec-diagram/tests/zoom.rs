//! Zoom (DESIGN.md § 13.3, story:diagram-zoom): a collapsed scope is one Composite node with its
//! boundary ports, depth and `expanded` open scopes, and [`levels`] draws one SVG per scope whose
//! composites and breadcrumb link to their own levels.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use modelspec_diagram::layout::{RenderRequest, render};
use modelspec_diagram::model::{
    DiagramGraph, DiagramRender, NodeKind, PortDirection, ScopeKey, ScopeKind,
};
use modelspec_diagram::visible::{Role, visible};
use modelspec_diagram::{json, levels, model_graph};

const FIXTURE: &str = "crates/modelspec-diagram/tests/fixtures/two_functions.mspec";
const ATTENTION: &str = "crates/modelspec-docs/models/layers/self_attention.mspec";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn graph(file: &str) -> DiagramGraph {
    let source = std::fs::read_to_string(repo().join(file)).unwrap();
    model_graph(file, &source).unwrap_or_else(|d| panic!("{file}: {d:?}"))
}

fn root(graph: &DiagramGraph) -> ScopeKey {
    graph
        .scopes
        .iter()
        .find(|s| s.parent.is_none())
        .expect("a root scope")
        .key
        .clone()
}

fn request(root: Option<&ScopeKey>, depth: u32, expanded: &[&ScopeKey]) -> RenderRequest {
    RenderRequest {
        root: root.cloned(),
        depth: Some(depth),
        expanded: expanded.iter().map(|&k| k.clone()).collect(),
    }
}

/// `key` is `scope` or inside it.
fn within(key: &ScopeKey, scope: &ScopeKey) -> bool {
    key == scope || key.0.starts_with(&format!("{}/", scope.0))
}

/// The scope of each node, by node key.
fn scopes_of(graph: &DiagramGraph) -> BTreeMap<&str, &ScopeKey> {
    graph
        .nodes
        .iter()
        .map(|n| (n.node_key.as_str(), &n.scope))
        .collect()
}

/// The values crossing the boundary of `scope`, computed from the graph's edges:
/// `(direction, value_id)` in the scope's port order.
fn crossing(graph: &DiagramGraph, scope: &ScopeKey) -> Vec<(PortDirection, i64)> {
    let of = scopes_of(graph);
    let mut ins = BTreeSet::new();
    let mut outs = BTreeSet::new();
    for edge in &graph.edges {
        let value: i64 = edge.producer[1..].parse().unwrap();
        match (
            within(of[edge.producer.as_str()], scope),
            within(of[edge.consumer.as_str()], scope),
        ) {
            (false, true) => {
                ins.insert(value);
            }
            (true, false) => {
                outs.insert(value);
            }
            _ => {}
        }
    }
    ins.into_iter()
        .map(|v| (PortDirection::In, v))
        .chain(outs.into_iter().map(|v| (PortDirection::Out, v)))
        .collect()
}

fn child_keys(graph: &DiagramGraph, parent: &ScopeKey) -> Vec<ScopeKey> {
    graph
        .scopes
        .iter()
        .filter(|s| s.parent.as_ref() == Some(parent))
        .map(|s| s.key.clone())
        .collect()
}

/// Every route of `render` starts and ends on a box, enters a composite at one of its In ports and
/// leaves one at one of its Out ports.
fn routes_land_on_ports(graph: &DiagramGraph, render: &DiagramRender) {
    let shown = visible(graph, &request_of(render));
    let boxes: BTreeMap<&str, _> = render
        .boxes
        .iter()
        .map(|b| (b.node_key.as_str(), b))
        .collect();
    let node = |key: &str| {
        shown
            .nodes
            .iter()
            .find(|n| n.node.node_key == key)
            .unwrap_or_else(|| panic!("{key} is not a visible node"))
    };
    for route in &render.routes {
        let from = boxes
            .get(route.producer.as_str())
            .unwrap_or_else(|| panic!("route from {} has no box", route.producer));
        let to = boxes
            .get(route.consumer.as_str())
            .unwrap_or_else(|| panic!("route to {} has no box", route.consumer));
        let n = route.points.len();
        let (x0, y0, x1, y1) = (
            route.points[0],
            route.points[1],
            route.points[n - 2],
            route.points[n - 1],
        );
        assert_eq!(
            y0,
            from.y + from.height,
            "{route:?} leaves its producer's bottom"
        );
        assert_eq!(y1, to.y, "{route:?} enters its consumer's top");
        let consumer = node(&route.consumer);
        if consumer.node.kind == NodeKind::Composite {
            let ports = consumer.inputs.len() as i64;
            assert!(
                (0..ports).contains(&route.operand),
                "{route:?} names In port {} of {ports}",
                route.operand
            );
            assert_eq!(
                x1,
                to.x + to.width * (2 * route.operand + 1) / (2 * ports),
                "{route:?} lands on In port {}",
                route.operand
            );
        }
        let producer = node(&route.producer);
        if producer.node.kind == NodeKind::Composite {
            let m = producer.outputs.len() as i64;
            assert!(
                (0..m).any(|j| x0 == from.x + from.width * (2 * j + 1) / (2 * m)),
                "{route:?} leaves at one of the {m} Out ports of {}",
                route.producer
            );
        }
    }
}

fn request_of(render: &DiagramRender) -> RenderRequest {
    RenderRequest {
        root: Some(render.root.clone()),
        depth: Some(render.depth as u32),
        expanded: render.expanded.clone(),
    }
}

/// The graph: one scope per function inlining in the model view, each with its ports.
#[test]
fn scopes_and_ports() {
    let graph = graph(FIXTURE);
    let root = root(&graph);
    assert_eq!(root.0, "Fit");
    let children: Vec<String> = child_keys(&graph, &root).into_iter().map(|k| k.0).collect();
    assert_eq!(children, ["Fit/forward", "Fit/head", "Fit/head@1"]);
    for scope in &graph.scopes {
        let last = scope.path.last().unwrap();
        if scope.parent.is_some() {
            assert_eq!(last.kind, ScopeKind::Function, "{}", scope.key.0);
            assert_eq!(scope.glyph.as_str(), "scope-function");
            assert!(last.call_site.is_some(), "{}: the call's span", scope.key.0);
        }
        let ports: Vec<(PortDirection, i64)> = scope
            .ports
            .iter()
            .map(|p| (p.direction, p.value_id))
            .collect();
        assert_eq!(ports, crossing(&graph, &scope.key), "{}", scope.key.0);
    }
    let head1 = graph
        .scopes
        .iter()
        .find(|s| s.key.0 == "Fit/head@1")
        .unwrap();
    assert_eq!(head1.path.last().unwrap().name, "head");
    assert_eq!(head1.path.last().unwrap().instance, 1);
    let ins = head1
        .ports
        .iter()
        .filter(|p| p.direction == PortDirection::In);
    assert!(ins.count() >= 2, "head@1 reads head's result and v");
    let of = scopes_of(&graph);
    assert!(
        graph.nodes.iter().all(|n| graph
            .scopes
            .iter()
            .any(|s| &s.key == of[n.node_key.as_str()])),
        "every node's scope is in the graph"
    );
}

/// zoom.collapsed-is-one-node
#[test]
fn collapsed_is_one_node() {
    let graph = graph(FIXTURE);
    let root = root(&graph);
    let request = request(None, 0, &[]);
    let shown = visible(&graph, &request);
    let composites: Vec<&str> = shown
        .nodes
        .iter()
        .filter(|n| n.node.kind == NodeKind::Composite)
        .map(|n| n.node.node_key.as_str())
        .collect();
    assert_eq!(composites, ["Fit/forward", "Fit/head", "Fit/head@1"]);
    for node in &shown.nodes {
        match node.node.kind {
            NodeKind::Operator => {
                assert_eq!(
                    node.node.scope, root,
                    "{} is inside a collapsed scope",
                    node.node.node_key
                );
                assert_eq!(node.role, Role::Operator);
            }
            NodeKind::Composite => {
                let key = node.node.composite_of.clone().unwrap();
                assert_eq!(key.0, node.node.node_key);
                assert_eq!(node.node.glyph.as_str(), "scope-function");
                let ports: Vec<(PortDirection, i64)> = node
                    .inputs
                    .iter()
                    .chain(&node.outputs)
                    .map(|p| (p.direction, p.value_id))
                    .collect();
                assert_eq!(ports, crossing(&graph, &key), "{}", key.0);
            }
            NodeKind::Endpoint => panic!("no boundary port at the root"),
        }
    }
    let keys: BTreeSet<&str> = shown
        .nodes
        .iter()
        .map(|n| n.node.node_key.as_str())
        .collect();
    for edge in &shown.edges {
        assert!(keys.contains(shown.nodes[edge.producer].node.node_key.as_str()));
        assert!(keys.contains(shown.nodes[edge.consumer].node.node_key.as_str()));
    }
    let merged: Vec<(&str, &str, usize)> = shown
        .edges
        .iter()
        .filter(|e| e.count > 1)
        .map(|e| {
            (
                shown.nodes[e.producer].node.node_key.as_str(),
                shown.nodes[e.consumer].node.node_key.as_str(),
                e.count,
            )
        })
        .collect();
    assert!(
        merged.contains(&("Fit/head", "Fit/head@1", 2)),
        "head's result read twice in head@1 is one edge ×2: {merged:?}"
    );

    let drawn = render(&graph, &request);
    routes_land_on_ports(&graph, &drawn);
    let svg = &levels(&graph, "{scope}.svg")
        .into_iter()
        .find(|(key, _)| *key == root)
        .unwrap()
        .1;
    assert!(svg.contains(">×2</text>"), "a merged edge is labelled ×2");
    for key in composites {
        let open = format!("<g class=\"node\" data-key=\"{key}\"");
        let group = &svg[svg.find(&open).unwrap()..];
        assert!(
            group.contains("<use href=\"#ms-scope-function\""),
            "{key}: its glyph"
        );
        assert!(svg.contains("<symbol id=\"ms-scope-function\""));
        assert!(group.contains("class=\"port\""), "{key}: its port row");
    }
}

/// A nested level draws the root's boundary ports, and its routes land on ports too.
#[test]
fn nested_level_draws_its_boundary() {
    let graph = graph(FIXTURE);
    let head1 = ScopeKey("Fit/head@1".to_owned());
    let request = request(Some(&head1), 0, &[]);
    let shown = visible(&graph, &request);
    let scope = graph.scopes.iter().find(|s| s.key == head1).unwrap();
    let boundary: Vec<(PortDirection, i64)> = shown
        .nodes
        .iter()
        .filter_map(|n| match n.role {
            Role::Boundary(direction) => Some((direction, n.value)),
            _ => None,
        })
        .collect();
    let ports: Vec<(PortDirection, i64)> = scope
        .ports
        .iter()
        .map(|p| (p.direction, p.value_id))
        .collect();
    let mut sorted = boundary.clone();
    sorted.sort();
    let mut expected = ports.clone();
    expected.sort();
    assert_eq!(sorted, expected, "one boundary node per port of the root");
    assert!(
        shown
            .nodes
            .iter()
            .filter(|n| n.role == Role::Operator)
            .all(|n| n.node.scope == head1)
    );
    routes_land_on_ports(&graph, &render(&graph, &request));
}

/// zoom.expand-shows-children
#[test]
fn expand_shows_children() {
    let graph = graph(FIXTURE);
    let head = ScopeKey("Fit/head".to_owned());
    let head1 = ScopeKey("Fit/head@1".to_owned());
    let shown = visible(&graph, &request(None, 0, &[&head1]));
    let operators: BTreeSet<&str> = shown
        .nodes
        .iter()
        .filter(|n| n.node.kind == NodeKind::Operator)
        .map(|n| n.node.node_key.as_str())
        .collect();
    let inside: BTreeSet<&str> = graph
        .nodes
        .iter()
        .filter(|n| n.scope == head1)
        .map(|n| n.node_key.as_str())
        .collect();
    assert!(!inside.is_empty());
    assert!(
        inside.is_subset(&operators),
        "head@1's operators are visible"
    );
    assert!(
        graph
            .nodes
            .iter()
            .filter(|n| n.scope == head)
            .all(|n| !operators.contains(n.node_key.as_str()))
    );
    let composites: Vec<&str> = shown
        .nodes
        .iter()
        .filter(|n| n.node.kind == NodeKind::Composite)
        .map(|n| n.node.node_key.as_str())
        .collect();
    assert_eq!(
        composites,
        ["Fit/forward", "Fit/head"],
        "head stays one node"
    );
    routes_land_on_ports(&graph, &render(&graph, &request(None, 0, &[&head1])));

    let every: Vec<&ScopeKey> = graph.scopes.iter().map(|s| &s.key).collect();
    let mut union = BTreeSet::new();
    for scope in &graph.scopes {
        let shown = visible(&graph, &request(Some(&scope.key), 0, &every));
        union.extend(
            shown
                .nodes
                .iter()
                .filter(|n| n.node.kind == NodeKind::Operator)
                .map(|n| n.node.node_key.clone()),
        );
    }
    let all: BTreeSet<String> = graph
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Operator)
        .map(|n| n.node_key.clone())
        .collect();
    assert_eq!(union, all);
}

/// zoom.depth
#[test]
fn depth() {
    for file in [FIXTURE, ATTENTION] {
        let graph = graph(file);
        let root = root(&graph);
        let children = child_keys(&graph, &root);
        let by_depth = render(&graph, &request(None, 1, &[]));
        let listed: Vec<&ScopeKey> = children.iter().collect();
        let by_list = render(&graph, &request(None, 0, &listed));
        assert_eq!(by_depth.boxes, by_list.boxes, "{file}");
        assert_eq!(by_depth.routes, by_list.routes, "{file}");
        assert_eq!(
            (by_depth.width, by_depth.height),
            (by_list.width, by_list.height),
            "{file}"
        );
    }
}

/// The values of every `<a class="{class}" href="…">` in `svg`.
fn links<'a>(svg: &'a str, class: &str) -> Vec<&'a str> {
    let open = format!("<a class=\"{class}\" href=\"");
    svg.split(open.as_str())
        .skip(1)
        .map(|rest| &rest[..rest.find('"').unwrap()])
        .collect()
}

/// zoom.links
#[test]
fn links_name_levels() {
    let graph = graph(FIXTURE);
    let drawn = levels(&graph, "{scope}.svg");
    let keys: Vec<&ScopeKey> = drawn.iter().map(|(k, _)| k).collect();
    let scopes: Vec<&ScopeKey> = graph.scopes.iter().map(|s| &s.key).collect();
    assert_eq!(keys, scopes, "one level per scope, in scope order");
    let files: BTreeSet<String> = keys.iter().map(|k| format!("{}.svg", k.0)).collect();
    let parent: BTreeMap<&ScopeKey, Option<&ScopeKey>> = graph
        .scopes
        .iter()
        .map(|s| (&s.key, s.parent.as_ref()))
        .collect();
    let mut composites_seen = 0;
    for (key, svg) in &drawn {
        let level = render(&graph, &request(Some(key), 0, &[]));
        let composites = visible(&graph, &request_of(&level))
            .nodes
            .iter()
            .filter(|n| n.node.kind == NodeKind::Composite)
            .count();
        let hrefs = links(svg, "composite");
        assert_eq!(
            hrefs.len(),
            composites,
            "{}: every composite is a link",
            key.0
        );
        composites_seen += composites;
        for href in hrefs {
            assert!(files.contains(href), "{}: {href} names no level", key.0);
        }
        let mut ancestors = Vec::new();
        let mut at = parent[key];
        while let Some(up) = at {
            ancestors.push(format!("{}.svg", up.0));
            at = parent[up];
        }
        ancestors.reverse();
        let crumbs: Vec<String> = links(svg, "crumb").into_iter().map(str::to_owned).collect();
        assert_eq!(
            crumbs, ancestors,
            "{}: the breadcrumb links each ancestor",
            key.0
        );
    }
    assert!(composites_seen >= 3);
}

/// zoom.deterministic
#[test]
fn deterministic() {
    let graph = graph(FIXTURE);
    let forward = ScopeKey("Fit/forward".to_owned());
    let head1 = ScopeKey("Fit/head@1".to_owned());
    let one = json::render_json(&render(&graph, &request(None, 0, &[&head1, &forward])));
    let two = json::render_json(&render(&graph, &request(None, 0, &[&head1, &forward])));
    let swapped = json::render_json(&render(
        &graph,
        &request(None, 0, &[&forward, &head1, &forward]),
    ));
    assert_eq!(one, two);
    assert_eq!(one, swapped, "the order of expanded does not matter");
    assert_eq!(levels(&graph, "#{scope}"), levels(&graph, "#{scope}"));
}

/// zoom.validates
#[test]
fn validates() {
    let renders = validator("DiagramRender");
    let graphs = validator("DiagramGraph");
    for file in [FIXTURE, ATTENTION] {
        let graph = graph(file);
        check(&graphs, &json::graph_json(&graph), file);
        let every: Vec<&ScopeKey> = graph.scopes.iter().map(|s| &s.key).collect();
        for scope in &graph.scopes {
            for (depth, expanded) in [(0, &[][..]), (1, &[][..]), (0, &every[..])] {
                let render = render(&graph, &request(Some(&scope.key), depth, expanded));
                check(&renders, &json::render_json(&render), file);
            }
        }
    }
}

/// The collapsed level of a layer block: one `scope-function` composite whose In ports are the
/// block's arguments and parameters.
#[test]
fn block_collapses_to_one_composite() {
    let graph = graph(ATTENTION);
    let shown = visible(&graph, &request(None, 0, &[]));
    let composites: Vec<_> = shown
        .nodes
        .iter()
        .filter(|n| n.node.kind == NodeKind::Composite)
        .collect();
    assert_eq!(composites.len(), 1);
    let block = composites[0];
    assert_eq!(block.node.node_key, "FitSelfAttention/forward");
    assert_eq!(block.node.glyph.as_str(), "scope-function");
    assert_eq!(block.inputs.len(), 5, "x, key_mask, wq, wk and wv");
    let drawn = render(&graph, &request(None, 0, &[]));
    assert_eq!(drawn.routes.len(), 5, "one route into each In port");
    routes_land_on_ports(&graph, &drawn);
}

fn validator(entity: &str) -> jsonschema::Validator {
    let path = repo().join(format!(
        "crates/modelspec-diagram/schema/entities/modelspec.diagram.{entity}.schema.json"
    ));
    let text = std::fs::read_to_string(&path).unwrap();
    let schema: serde_json::Value = serde_json::from_str(&text).unwrap();
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

fn check(validator: &jsonschema::Validator, text: &str, file: &str) {
    let value: serde_json::Value = serde_json::from_str(text).unwrap();
    let errors: Vec<String> = validator
        .iter_errors(&value)
        .map(|e| format!("{} at {}", e, e.instance_path()))
        .collect();
    assert!(errors.is_empty(), "{file}: {errors:#?}");
}
