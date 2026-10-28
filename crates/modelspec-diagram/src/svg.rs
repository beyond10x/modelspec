//! The SVG of a render (DESIGN.md § 13.1, § 13.5): a standalone file that carries the glyph
//! symbols it uses, copied from [`sprite`], and the theme's light and dark defaults, so it shows
//! the same as a Markdown image, inlined in a page or opened alone.
//!
//! The root `<svg>` has `role="img"`, a `<title>` and a `<desc>` naming the program and view, and
//! the class `ms-glyphs`, on which the theme's custom properties sit (a page restyles a diagram
//! by setting them there). Each route is one `<path class="edge" data-from data-to
//! data-operand>`, curved between ranks, straight through each dummy's rank, with its arrowhead
//! as a second subpath; a route into a `StopGradient`, across which no gradient flows back, is
//! dashed and carries `data-stop-gradient="true"`. Each visible node is one
//! `<g class="node" data-key data-op data-family data-dtype data-shape>` holding a `<title>` with
//! the line `modelspec ir` prints for it, its box, its glyph `<use href="#ms-…">`, its two label
//! lines, and its chips, each a `<g class="chip" data-key data-operand>` with its own title, glyph
//! and name. Every coordinate is an integer.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::glyph::{Scheme, sprite, theme};
use crate::graph::IrLines;
use crate::layout::{
    CHIP_GLYPH, CHIP_HEIGHT, GLYPH, INNER, PAD, chip_rects, chip_text, chips, label_lines,
};
use crate::model::{DiagramGraph, DiagramNode, DiagramRender, NodeBox, OperatorFamily};

const FONT: &str = "'DejaVu Sans Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

/// The SVG of `render`, a level of `graph`; each node's `<title>` is its line of `ir_lines`
/// ([`crate::graph::ir_lines`]), or its label when the map has none.
pub fn svg(graph: &DiagramGraph, render: &DiagramRender, ir_lines: &IrLines) -> String {
    let index: BTreeMap<&str, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.node_key.as_str(), i))
        .collect();
    let mut chips_of: BTreeMap<usize, Vec<(i64, usize)>> = BTreeMap::new();
    for chip in chips(graph, &render.root) {
        chips_of
            .entry(chip.reader)
            .or_default()
            .push((chip.operand, chip.node));
    }
    for list in chips_of.values_mut() {
        list.sort();
    }
    let boxed: Vec<(&NodeBox, &DiagramNode, Vec<&DiagramNode>)> = render
        .boxes
        .iter()
        .filter_map(|b| {
            let i = *index.get(b.node_key.as_str())?;
            let chips = chips_of
                .get(&i)
                .map(|list| list.iter().map(|&(_, n)| &graph.nodes[n]).collect())
                .unwrap_or_default();
            Some((b, &graph.nodes[i], chips))
        })
        .collect();

    let mut used = BTreeSet::new();
    for (_, node, chips) in &boxed {
        used.insert(node.glyph.as_str());
        used.extend(chips.iter().map(|c| c.glyph.as_str()));
    }

    let id = format!(
        "ms-diagram-{}-{}",
        graph.view.name(),
        &graph.ir_hash[..graph.ir_hash.len().min(12)]
    );
    let (w, h) = (render.width, render.height);
    let mut out = String::new();
    let _ = write!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" class=\"ms-glyphs ms-diagram\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\" role=\"img\" aria-labelledby=\"{id}-title {id}-desc\" style=\"max-width:100%;height:auto\">"
    );
    let _ = write!(
        out,
        "<title id=\"{id}-title\">{}: {} view</title>",
        escape(&graph.program),
        graph.view.name()
    );
    let _ = write!(
        out,
        "<desc id=\"{id}-desc\">The {} view of train block {}, drawn from its verified Tensor IR: {} operators in {} ranks, parameters and dropout tokens with one reader drawn as chips on it.</desc>",
        graph.view.name(),
        escape(&graph.program),
        boxed.len() + boxed.iter().map(|(_, _, c)| c.len()).sum::<usize>(),
        render.boxes.iter().map(|b| b.rank + 1).max().unwrap_or(0),
    );
    out.push_str(&style());
    out.push_str("<defs>");
    out.push_str(&symbols(&used));
    out.push_str("</defs>");

    out.push_str("<g class=\"edges\" fill=\"none\" stroke=\"var(--ms-diagram-edge, var(--ms-diagram-muted, #606770))\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\">");
    for route in &render.routes {
        let stopped = index
            .get(route.consumer.as_str())
            .is_some_and(|&i| graph.nodes[i].op.as_deref() == Some("StopGradient"));
        let _ = write!(
            out,
            "<path class=\"edge\" data-from=\"{}\" data-to=\"{}\" data-operand=\"{}\"{} d=\"{}\"/>",
            attribute(&route.producer),
            attribute(&route.consumer),
            route.operand,
            if stopped {
                " data-stop-gradient=\"true\" stroke-dasharray=\"6 5\""
            } else {
                ""
            },
            route_path(&route.points)
        );
    }
    out.push_str("</g><g class=\"nodes\">");
    for (node_box, node, chips) in &boxed {
        out.push_str(&node_group(graph, ir_lines, node_box, node, chips));
    }
    out.push_str("</g></svg>");
    out
}

fn style() -> String {
    let rule = |scheme: Scheme| {
        let mut rule = String::from(":where(.ms-glyphs){");
        for (property, value) in theme(scheme) {
            let _ = write!(rule, "{property}:{value};");
        }
        rule.push('}');
        rule
    };
    format!(
        "<style>{}@media (prefers-color-scheme: dark){{{}}}.ms-diagram text{{font-family:{FONT}}}</style>",
        rule(Scheme::Light),
        rule(Scheme::Dark)
    )
}

/// The `<symbol>` elements of the sprite whose ids (without `ms-`) are in `used`, in id order.
fn symbols(used: &BTreeSet<&str>) -> String {
    let sprite = sprite();
    let mut out = String::new();
    let mut rest = sprite.as_str();
    while let Some(start) = rest.find("<symbol id=\"ms-") {
        rest = &rest[start..];
        let end = rest
            .find("</symbol>")
            .map_or(rest.len(), |e| e + "</symbol>".len());
        let symbol = &rest[..end];
        let id = &symbol["<symbol id=\"ms-".len()..];
        let id = &id[..id.find('"').unwrap_or(0)];
        if used.contains(id) {
            out.push_str(symbol);
        }
        rest = &rest[end..];
    }
    out
}

/// The path of a route's points: a vertical step is a line, a step between ranks a cubic curve
/// that leaves and arrives vertically; then the arrowhead at the last point.
fn route_path(points: &[i64]) -> String {
    let points: Vec<(i64, i64)> = points.chunks(2).map(|p| (p[0], p[1])).collect();
    let mut d = String::new();
    if let Some(&(x, y)) = points.first() {
        let _ = write!(d, "M{x} {y}");
    }
    for pair in points.windows(2) {
        let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
        if x0 == x1 {
            let _ = write!(d, "V{y1}");
        } else {
            let mid = (y0 + y1) / 2;
            let _ = write!(d, "C{x0} {mid} {x1} {mid} {x1} {y1}");
        }
    }
    if let Some(&(x, y)) = points.last() {
        let _ = write!(d, "M{} {}L{x} {y}L{} {}", x - 4, y - 6, x + 4, y - 6);
    }
    d
}

fn family_colour(family: Option<OperatorFamily>) -> String {
    match family {
        Some(family) => format!(
            "var(--ms-family-{}, var(--ms-diagram-muted, #606770))",
            family.css_name()
        ),
        None => "var(--ms-diagram-muted, #606770)".to_owned(),
    }
}

const INK: &str = "var(--ms-diagram-ink, var(--ifm-font-color-base, #1c1e21))";
const PAPER: &str = "var(--ms-diagram-paper, var(--ifm-background-color, #ffffff))";
const MUTED: &str = "var(--ms-diagram-muted, var(--ifm-color-emphasis-600, #606770))";

fn node_group(
    graph: &DiagramGraph,
    ir_lines: &IrLines,
    node_box: &NodeBox,
    node: &DiagramNode,
    chips: &[&DiagramNode],
) -> String {
    let NodeBox {
        x,
        y,
        width,
        height,
        ..
    } = *node_box;
    let mut out = String::new();
    let _ = write!(
        out,
        "<g class=\"node\" data-key=\"{}\" data-op=\"{}\" data-family=\"{}\" data-dtype=\"{}\" data-shape=\"{}\">",
        attribute(&node.node_key),
        attribute(node.op.as_deref().unwrap_or("")),
        node.family.map(|f| format!("{f:?}")).unwrap_or_default(),
        node.dtype.map(|d| format!("{d:?}")).unwrap_or_default(),
        node.shape
            .as_ref()
            .map(|s| s.iter().map(u64::to_string).collect::<Vec<_>>().join(","))
            .unwrap_or_default(),
    );
    let _ = write!(out, "<title>{}</title>", escape(&ir_line(ir_lines, node)));
    let accent = family_colour(node.family);
    let _ = write!(
        out,
        "<rect x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" rx=\"6\" fill=\"{PAPER}\" stroke=\"{accent}\" stroke-width=\"2\"/>"
    );
    let _ = write!(
        out,
        "<use href=\"#ms-{}\" x=\"{}\" y=\"{}\" width=\"{GLYPH}\" height=\"{GLYPH}\"/>",
        node.glyph,
        x + PAD,
        y + (height - GLYPH) / 2
    );
    let (first, second) = label_lines(node);
    let text_x = x + PAD + GLYPH + INNER;
    let _ = write!(
        out,
        "<text x=\"{text_x}\" y=\"{}\" font-size=\"11\" font-weight=\"600\" fill=\"{INK}\">{}</text>",
        y + 17,
        escape(first)
    );
    if !second.is_empty() {
        let _ = write!(
            out,
            "<text x=\"{text_x}\" y=\"{}\" font-size=\"10\" fill=\"{MUTED}\">{}</text>",
            y + 31,
            escape(second)
        );
    }
    let operands: BTreeMap<&str, i64> = graph
        .edges
        .iter()
        .filter(|e| e.consumer == node.node_key)
        .map(|e| (e.producer.as_str(), e.operand))
        .collect();
    for (chip, (cx, cy, cw, ch)) in chips.iter().zip(chip_rects(node_box, chips)) {
        let operand = operands.get(chip.node_key.as_str()).copied().unwrap_or(0);
        let colour = family_colour(chip.family);
        let _ = write!(
            out,
            "<g class=\"chip\" data-key=\"{}\" data-operand=\"{operand}\"><title>{}</title>",
            attribute(&chip.node_key),
            escape(&ir_line(ir_lines, chip))
        );
        let _ = write!(
            out,
            "<rect x=\"{cx}\" y=\"{cy}\" width=\"{cw}\" height=\"{ch}\" rx=\"8\" fill=\"{colour}\" fill-opacity=\"12%\" stroke=\"{colour}\" stroke-width=\"1\"/>"
        );
        let _ = write!(
            out,
            "<use href=\"#ms-{}\" x=\"{}\" y=\"{}\" width=\"{CHIP_GLYPH}\" height=\"{CHIP_GLYPH}\"/>",
            chip.glyph,
            cx + 4,
            cy + (CHIP_HEIGHT - CHIP_GLYPH) / 2
        );
        let _ = write!(
            out,
            "<text x=\"{}\" y=\"{}\" font-size=\"10\" fill=\"{INK}\">{}</text></g>",
            cx + 4 + CHIP_GLYPH + 4,
            cy + 12,
            escape(chip_text(chip))
        );
    }
    out.push_str("</g>");
    out
}

/// The line `modelspec ir` prints for a node, else its label on one line.
fn ir_line(ir_lines: &IrLines, node: &DiagramNode) -> String {
    match ir_lines.get(&node.node_key) {
        Some(line) => line.clone(),
        None => node.label.replace('\n', " "),
    }
}

/// Text content: `&`, `<` and `>` escaped.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// An attribute value: as [`escape`], and `"` too.
fn attribute(text: &str) -> String {
    escape(text).replace('"', "&quot;")
}
