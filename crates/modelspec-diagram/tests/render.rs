//! The model-view renderer of DESIGN.md § 13.2 and § 13.5: graph from verified IR, layered
//! layout, SVG and JSON (story:ir-gallery-renderer, acceptance 1 to 6).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use modelspec_core::compile;
use modelspec_core::ir::print::print;
use modelspec_diagram::layout::{RenderRequest, render};
use modelspec_diagram::model::{DType, DiagramGraph, DiagramRender, NodeKind};
use modelspec_diagram::{json, model_graph, svg};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const TWO_LAYER: &str = "crates/modelspec-docs/models/two_layer_relu.mspec";

/// The files of `examples/` that `task check-examples` checks, and `two_layer_relu.mspec`.
fn layered_files() -> Vec<&'static str> {
    let taskfile = std::fs::read_to_string(repo().join("Taskfile.yml")).unwrap();
    let section = taskfile
        .split("check-examples:")
        .nth(1)
        .expect("a check-examples task")
        .split("\n\n")
        .next()
        .unwrap();
    let mut files: Vec<&'static str> = section
        .lines()
        .filter_map(|line| line.split(" check ").nth(1))
        .map(|file| &*Box::leak(file.trim().to_owned().into_boxed_str()))
        .collect();
    assert!(files.len() >= 3, "check-examples names {files:?}");
    files.push(TWO_LAYER);
    files
}

fn source(file: &str) -> String {
    std::fs::read_to_string(repo().join(file)).unwrap()
}

fn graph(file: &str) -> DiagramGraph {
    model_graph(file, &source(file)).unwrap_or_else(|d| panic!("{file}: {d:?}"))
}

fn ir_lines_of(file: &str) -> modelspec_diagram::graph::IrLines {
    modelspec_diagram::graph::ir_lines(&compile::compile_training(file, &source(file)).unwrap())
}

fn default_render(graph: &DiagramGraph) -> DiagramRender {
    render(graph, &RenderRequest::default())
}

/// One line of `modelspec ir --stage tensor`: `%2: Tensor<f32, [16, 8]> = MatMul %0, %1`.
#[derive(Debug)]
struct IrLine {
    id: u32,
    dtype: String,
    shape: Vec<u64>,
    op: String,
    operands: Vec<u32>,
    text: String,
}

fn ir_lines(file: &str) -> (BTreeMap<u32, IrLine>, u32) {
    let lowered = compile::compile_tensor(file, &source(file)).unwrap();
    let mut lines = BTreeMap::new();
    for text in print(&lowered.program).lines() {
        let Some(rest) = text.strip_prefix('%') else {
            continue;
        };
        let (id, rest) = rest.split_once(": Tensor<").unwrap();
        let (ty, rest) = rest.split_once("> = ").unwrap();
        let (dtype, shape) = ty.split_once(", [").unwrap();
        let shape = shape
            .trim_end_matches(']')
            .split(", ")
            .filter(|d| !d.is_empty())
            .map(|d| d.parse().unwrap())
            .collect();
        let op = rest.split([' ', '\n']).next().unwrap().to_owned();
        let after_attrs = match rest.find(']') {
            Some(close) if rest.find('[').is_some_and(|open| open < close) => &rest[close + 1..],
            _ => &rest[op.len()..],
        };
        let operands = after_attrs
            .split([' ', ','])
            .filter_map(|t| t.strip_prefix('%'))
            .map(|t| t.parse().unwrap())
            .collect();
        let id: u32 = id.parse().unwrap();
        lines.insert(
            id,
            IrLine {
                id,
                dtype: dtype.to_owned(),
                shape,
                op,
                operands,
                text: text.to_owned(),
            },
        );
    }
    (lines, lowered.prediction.0)
}

/// The definitions the prediction reaches through operands.
fn forward_slice(lines: &BTreeMap<u32, IrLine>, prediction: u32) -> BTreeSet<u32> {
    let mut slice = BTreeSet::new();
    let mut stack = vec![prediction];
    while let Some(id) = stack.pop() {
        if slice.insert(id) {
            stack.extend(lines[&id].operands.iter().copied());
        }
    }
    slice
}

fn dtype_text(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::I32 => "i32",
    }
}

/// diagram.model-matches-ir
#[test]
fn model_matches_ir() {
    let (lines, prediction) = ir_lines(TWO_LAYER);
    let slice = forward_slice(&lines, prediction);
    let graph = graph(TWO_LAYER);

    let operators: Vec<_> = graph
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Operator)
        .collect();
    assert_eq!(
        operators.len(),
        graph.nodes.len(),
        "a model graph of one scope"
    );
    let ids: BTreeSet<u32> = operators
        .iter()
        .map(|n| n.value_id.unwrap() as u32)
        .collect();
    assert_eq!(ids, slice, "one Operator node per forward-slice definition");
    assert_eq!(ids.len(), operators.len(), "no definition twice");
    for node in &operators {
        let line = &lines[&(node.value_id.unwrap() as u32)];
        assert_eq!(node.node_key, format!("%{}", line.id));
        assert_eq!(node.op.as_deref(), Some(line.op.as_str()), "{}", line.text);
        assert_eq!(dtype_text(node.dtype.unwrap()), line.dtype, "{}", line.text);
        assert_eq!(node.shape.as_ref().unwrap(), &line.shape, "{}", line.text);
        assert!(
            node.label.contains(&format!("%{}", line.id)),
            "{}",
            node.label
        );
    }
    assert!(operators.iter().all(|n| n.op.as_deref() != Some("Mse")));
    let target = lines
        .values()
        .find(|l| l.text.ends_with("= Input \"y\""))
        .expect("the target y is an Input of the Tensor IR");
    assert!(!ids.contains(&target.id), "no node of the target y");

    let mut expected = BTreeSet::new();
    for &id in &slice {
        for (i, operand) in lines[&id].operands.iter().enumerate() {
            let producer = &lines[operand];
            expected.insert((
                format!("%{operand}"),
                format!("%{id}"),
                i as i64,
                producer.dtype.clone(),
                producer.shape.clone(),
            ));
        }
    }
    let edges: BTreeSet<_> = graph
        .edges
        .iter()
        .map(|e| {
            (
                e.producer.clone(),
                e.consumer.clone(),
                e.operand,
                dtype_text(e.dtype).to_owned(),
                e.shape.clone(),
            )
        })
        .collect();
    assert_eq!(edges.len(), graph.edges.len(), "no edge twice");
    assert_eq!(edges, expected, "one edge per operand");
}

/// The value of every attribute named `name` in `svg`.
fn attributes<'a>(svg: &'a str, name: &str) -> Vec<&'a str> {
    let needle = format!(" {name}=\"");
    let mut found = Vec::new();
    let mut rest = svg;
    while let Some(at) = rest.find(&needle) {
        rest = &rest[at + needle.len()..];
        let end = rest.find('"').unwrap();
        found.push(&rest[..end]);
        rest = &rest[end..];
    }
    found
}

/// diagram.deterministic
#[test]
fn deterministic() {
    for file in layered_files() {
        let text = source(file);
        let compiled = compile::compile_training(file, &text).unwrap();
        let program = compile::compile_hir(file, &text).unwrap().train.name;
        let a = modelspec_diagram::graph::model(&compiled, &program, &text);
        let b = modelspec_diagram::graph::model(&compiled.clone(), &program, &text);
        let c = graph(file);
        let (ra, rb, rc) = (default_render(&a), default_render(&b), default_render(&c));
        let svg_a = svg::svg(&a, &ra, &modelspec_diagram::graph::ir_lines(&compiled));
        assert!(
            !a.nodes.is_empty() && !ra.boxes.is_empty(),
            "{file}: an empty diagram"
        );
        assert!(
            svg_a.contains("<g class=\"node\""),
            "{file}: an svg without nodes"
        );
        assert_eq!(
            svg_a,
            svg::svg(
                &b,
                &rb,
                &modelspec_diagram::graph::ir_lines(&compiled.clone())
            ),
            "{file}: svg of a clone"
        );
        assert_eq!(
            svg_a,
            svg::svg(&c, &rc, &ir_lines_of(file)),
            "{file}: svg of a second compile"
        );
        assert_eq!(json::graph_json(&a), json::graph_json(&b), "{file}");
        assert_eq!(json::graph_json(&a), json::graph_json(&c), "{file}");
        assert_eq!(json::render_json(&ra), json::render_json(&rb), "{file}");
        assert_eq!(json::render_json(&ra), json::render_json(&rc), "{file}");
        for name in ["x", "y", "width", "height", "d"] {
            for value in attributes(&svg_a, name) {
                assert!(!value.contains('.'), "{file}: {name}=\"{value}\"");
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    x: i64,
    y: i64,
    w: i64,
    h: i64,
}

impl Rect {
    fn overlaps(self, o: Rect) -> bool {
        self.x < o.x + o.w && o.x < self.x + self.w && self.y < o.y + o.h && o.y < self.y + self.h
    }

    /// Whether the segment from `a` to `b` meets the open interior of the rectangle
    /// (Liang–Barsky clipping in exact rationals).
    fn crossed_by(self, a: (i64, i64), b: (i64, i64)) -> bool {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        // t in [lo, hi] as fractions num/den with den > 0.
        let mut lo = (0i128, 1i128);
        let mut hi = (1i128, 1i128);
        let less = |p: (i128, i128), q: (i128, i128)| p.0 * q.1 < q.0 * p.1;
        for (p, q) in [
            (-dx, a.0 - self.x),
            (dx, self.x + self.w - a.0),
            (-dy, a.1 - self.y),
            (dy, self.y + self.h - a.1),
        ] {
            let (p, q) = (p as i128, q as i128);
            if p == 0 {
                if q <= 0 {
                    return false;
                }
                continue;
            }
            let t = if p < 0 { (-q, -p) } else { (q, p) };
            if p < 0 {
                if less(lo, t) {
                    lo = t;
                }
            } else if less(t, hi) {
                hi = t;
            }
        }
        less(lo, hi)
    }
}

/// diagram.layered
#[test]
fn layered() {
    for file in layered_files() {
        let graph = graph(file);
        let render = default_render(&graph);
        let boxes: BTreeMap<&str, (Rect, i64)> = render
            .boxes
            .iter()
            .map(|b| {
                (
                    b.node_key.as_str(),
                    (
                        Rect {
                            x: b.x,
                            y: b.y,
                            w: b.width,
                            h: b.height,
                        },
                        b.rank,
                    ),
                )
            })
            .collect();
        assert!(!boxes.is_empty(), "{file}");
        let all: Vec<_> = boxes.iter().collect();
        for (i, (ka, (ra, _))) in all.iter().enumerate() {
            assert!(ra.x + ra.w <= render.width && ra.y + ra.h <= render.height);
            for (kb, (rb, _)) in &all[i + 1..] {
                assert!(!ra.overlaps(*rb), "{file}: boxes {ka} and {kb} overlap");
            }
        }
        assert!(!render.routes.is_empty(), "{file}");
        for route in &render.routes {
            let (_, from) = boxes[route.producer.as_str()];
            let (_, to) = boxes[route.consumer.as_str()];
            assert!(
                from < to,
                "{file}: route {} -> {} goes from rank {from} to rank {to}",
                route.producer,
                route.consumer
            );
            assert!(route.points.len() >= 4 && route.points.len() % 2 == 0);
            let points: Vec<(i64, i64)> = route.points.chunks(2).map(|p| (p[0], p[1])).collect();
            for segment in points.windows(2) {
                for (key, (rect, _)) in &boxes {
                    if *key == route.producer || *key == route.consumer {
                        continue;
                    }
                    assert!(
                        !rect.crossed_by(segment[0], segment[1]),
                        "{file}: route {} -> {} passes through {key}",
                        route.producer,
                        route.consumer
                    );
                }
            }
        }
    }
}

fn validator(entity: &str) -> jsonschema::Validator {
    let path = repo().join(format!(
        "crates/modelspec-diagram/schema/entities/modelspec.diagram.{entity}.schema.json"
    ));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let schema: serde_json::Value = serde_json::from_str(&text).unwrap();
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

/// diagram.json-validates
#[test]
fn json_validates() {
    let graphs = validator("DiagramGraph");
    let renders = validator("DiagramRender");
    for file in layered_files() {
        let graph = graph(file);
        let render = default_render(&graph);
        for (validator, text) in [
            (&graphs, json::graph_json(&graph)),
            (&renders, json::render_json(&render)),
        ] {
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            let errors: Vec<String> = validator
                .iter_errors(&value)
                .map(|e| format!("{} at {}", e, e.instance_path()))
                .collect();
            assert!(errors.is_empty(), "{file}: {errors:#?}");
        }
        let back: DiagramGraph = serde_json::from_str(&json::graph_json(&graph)).unwrap();
        assert_eq!(back, graph, "{file}: the graph JSON reads back");
        let back: DiagramRender = serde_json::from_str(&json::render_json(&render)).unwrap();
        assert_eq!(back, render, "{file}: the render JSON reads back");
    }
}

/// The `<g class="node" …>` element of `key` in `svg`, up to its closing tag.
fn node_group<'a>(svg: &'a str, key: &str) -> &'a str {
    let open = format!("<g class=\"node\" data-key=\"{key}\"");
    let start = svg.find(&open).unwrap_or_else(|| panic!("no node {key}"));
    let body = &svg[start..];
    let mut depth = 0usize;
    let mut at = 0usize;
    loop {
        let next_open = body[at..].find("<g").map(|i| at + i);
        let next_close = body[at..].find("</g>").map(|i| at + i).unwrap();
        match next_open {
            Some(o) if o < next_close => {
                depth += 1;
                at = o + 2;
            }
            _ => {
                depth -= 1;
                at = next_close + 4;
                if depth == 0 {
                    return &body[..at];
                }
            }
        }
    }
}

/// diagram.chips
#[test]
fn chips() {
    let graph = graph(TWO_LAYER);
    let render = default_render(&graph);
    let svg = svg::svg(&graph, &render, &ir_lines_of(TWO_LAYER));
    let boxed: BTreeMap<&str, &modelspec_diagram::model::NodeBox> = render
        .boxes
        .iter()
        .map(|b| (b.node_key.as_str(), b))
        .collect();
    let op = |key: &str| {
        graph
            .nodes
            .iter()
            .find(|n| n.node_key == key)
            .and_then(|n| n.op.clone())
            .unwrap()
    };
    let reads: Vec<_> = graph
        .nodes
        .iter()
        .filter(|n| n.op.as_deref() == Some("ParamRead"))
        .collect();
    assert_eq!(reads.len(), 2);
    for read in reads {
        assert!(
            !boxed.contains_key(read.node_key.as_str()),
            "{} is a ranked node",
            read.node_key
        );
        let edges: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.producer == read.node_key)
            .collect();
        assert_eq!(edges.len(), 1);
        let reader = &edges[0].consumer;
        assert_eq!(op(reader), "MatMul");
        assert!(
            node_group(&svg, reader).contains(&format!(
                "<g class=\"chip\" data-key=\"{}\" data-operand=\"{}\"",
                read.node_key, edges[0].operand
            )),
            "{} is a chip on {reader}",
            read.node_key
        );
        assert!(
            render.routes.iter().all(|r| r.producer != read.node_key),
            "a chip has no route"
        );
    }
    let mut ranked: Vec<_> = render.boxes.iter().collect();
    ranked.sort_by_key(|b| (b.rank, b.x));
    let ops: Vec<String> = ranked.iter().map(|b| op(&b.node_key)).collect();
    assert_eq!(ops, ["Input", "MatMul", "Relu", "MatMul"]);
    let ranks: Vec<i64> = ranked.iter().map(|b| b.rank).collect();
    assert_eq!(ranks, [0, 1, 2, 3]);
}

/// diagram.standalone-svg
#[test]
fn standalone_svg() {
    for file in layered_files() {
        let graph = graph(file);
        let render = default_render(&graph);
        let svg = svg::svg(&graph, &render, &ir_lines_of(file));
        assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(svg.contains(" role=\"img\""), "{file}");
        assert!(svg.contains("<title") && svg.contains("<desc"), "{file}");
        let symbols: BTreeSet<&str> = attributes(&svg, "id")
            .into_iter()
            .filter(|id| id.starts_with("ms-"))
            .collect();
        let uses: BTreeSet<&str> = attributes(&svg, "href")
            .into_iter()
            .map(|h| h.strip_prefix('#').expect("a fragment reference"))
            .collect();
        assert!(!uses.is_empty(), "{file}");
        for used in &uses {
            assert!(
                svg.contains(&format!("<symbol id=\"{used}\"")),
                "{file}: <use href=\"#{used}\"> names no symbol of the file"
            );
        }
        assert!(
            symbols
                .iter()
                .filter(|s| svg.contains(&format!("<symbol id=\"{s}\"")))
                .all(|s| uses.contains(s)),
            "{file}: only the symbols it needs"
        );
        let (lines, _) = ir_lines(file);
        for b in &render.boxes {
            let group = node_group(&svg, &b.node_key);
            let id: u32 = b.node_key[1..].parse().unwrap();
            let title = format!(
                "<title>{}</title>",
                lines[&id]
                    .text
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
            );
            assert!(group.contains(&title), "{file}: {group}");
            for attribute in ["data-op", "data-family", "data-dtype", "data-shape"] {
                assert!(
                    group.contains(&format!(" {attribute}=\"")),
                    "{file}: {group}"
                );
            }
            assert!(group.contains("<use href=\"#ms-"), "{file}: {group}");
        }
        let edges = svg.matches("<path class=\"edge\" data-from=\"").count();
        assert_eq!(edges, render.routes.len(), "{file}: one path per route");
    }
}
