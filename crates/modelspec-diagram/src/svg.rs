//! The SVG of a render (DESIGN.md § 13.1, § 13.3, § 13.5): a standalone file that carries the
//! glyph symbols it uses, copied from [`sprite`], and the theme's light and dark defaults, so it
//! shows the same as a Markdown image, inlined in a page or opened alone.
//!
//! The root `<svg>` has `role="img"`, a `<title>` and a `<desc>` naming the program and view, and
//! the class `ms-glyphs`, on which the theme's custom properties sit (a page restyles a diagram
//! by setting them there). Each route is one `<path class="edge" data-from data-to
//! data-operand>`, curved between ranks, straight through each dummy's rank, with its arrowhead
//! as a second subpath; a route into a `StopGradient`, across which no gradient flows back, is
//! dashed and carries `data-stop-gradient="true"`; a route standing for `n` edges is labelled
//! `×n` near its end. Each visible operator is one
//! `<g class="node" data-key data-op data-family data-dtype data-shape>` holding a `<title>` with
//! the line `modelspec ir` prints for it, its box, its glyph `<use href="#ms-…">`, its two label
//! lines, and its chips, each a `<g class="chip" data-key data-operand>` with its own title, glyph
//! and name. A collapsed scope is one `<g class="node" data-key data-kind="Composite">` with its
//! In ports on its top edge over their labels (the port row), its `scope-*` glyph, its name and
//! instance, and its Out ports on its bottom edge; with a link template it is wrapped in
//! `<a class="composite" href>` to its own level. A level whose root is not the Program scope
//! draws the root's ports as dashed boundary nodes and a breadcrumb of its ancestors above the
//! drawing, each ancestor an `<a class="crumb" href>` with a template. Every coordinate is an
//! integer.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::glyph::{Scheme, sprite, theme};
use crate::graph::IrLines;
use crate::layout::{
    ADVANCE, CHIP_GLYPH, CHIP_HEIGHT, GLYPH, HEIGHT, INNER, PAD, PORT_ROW, RenderRequest,
    chip_rects, chip_text, chips, label_lines, port_label, render as lay_out,
};
use crate::model::{
    DiagramGraph, DiagramNode, DiagramRender, NodeBox, OperatorFamily, PortDirection, ScopeKey,
};
use crate::visible::{Role, VisibleNode, visible};

const FONT: &str = "'DejaVu Sans Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

/// The height of the breadcrumb band above a level whose root has ancestors.
pub const CRUMB_BAND: i64 = 28;

/// The SVG of `render`, a level of `graph`; each node's `<title>` is its line of `ir_lines`
/// ([`crate::graph::ir_lines`]), or its label when the map has none. Composites and breadcrumb
/// entries are not links.
pub fn svg(graph: &DiagramGraph, render: &DiagramRender, ir_lines: &IrLines) -> String {
    draw(graph, render, ir_lines, None)
}

/// [`svg`] with links: each composite is an `<a class="composite">` and each breadcrumb ancestor
/// an `<a class="crumb">` whose `href` is `template` with `{scope}` replaced by the scope key.
pub fn svg_linked(
    graph: &DiagramGraph,
    render: &DiagramRender,
    ir_lines: &IrLines,
    template: &str,
) -> String {
    draw(graph, render, ir_lines, Some(template))
}

/// One SVG per scope of `graph`, in scope order: the scope as root at depth 0, every composite
/// and breadcrumb ancestor linking through `template` (`{scope}` replaced by the scope key).
/// Titles are the nodes' labels; [`levels_titled`] takes the `modelspec ir` lines instead.
pub fn levels(graph: &DiagramGraph, template: &str) -> Vec<(ScopeKey, String)> {
    levels_titled(graph, &IrLines::new(), template)
}

/// [`levels`] whose node titles are their lines of `ir_lines` ([`crate::graph::ir_lines`]).
pub fn levels_titled(
    graph: &DiagramGraph,
    ir_lines: &IrLines,
    template: &str,
) -> Vec<(ScopeKey, String)> {
    graph
        .scopes
        .iter()
        .map(|scope| {
            let request = RenderRequest {
                root: Some(scope.key.clone()),
                depth: Some(0),
                expanded: Vec::new(),
            };
            let render = lay_out(graph, &request);
            (
                scope.key.clone(),
                svg_linked(graph, &render, ir_lines, template),
            )
        })
        .collect()
}

/// `template` with `{scope}` replaced by `key`, as an attribute value.
fn href(template: &str, key: &ScopeKey) -> String {
    attribute(&template.replace("{scope}", &key.0))
}

/// The ancestors of `root`, outermost first.
fn ancestors(graph: &DiagramGraph, root: &ScopeKey) -> Vec<ScopeKey> {
    let parent: BTreeMap<&ScopeKey, Option<&ScopeKey>> = graph
        .scopes
        .iter()
        .map(|s| (&s.key, s.parent.as_ref()))
        .collect();
    let mut chain = Vec::new();
    let mut at = parent.get(root).copied().flatten();
    while let Some(key) = at {
        if chain.len() > graph.scopes.len() {
            break;
        }
        chain.push(key.clone());
        at = parent.get(key).copied().flatten();
    }
    chain.reverse();
    chain
}

/// The last segment of a scope key, as a breadcrumb shows it.
fn last_segment(key: &ScopeKey) -> &str {
    key.0.rsplit('/').next().unwrap_or(&key.0)
}

fn draw(
    graph: &DiagramGraph,
    render: &DiagramRender,
    ir_lines: &IrLines,
    template: Option<&str>,
) -> String {
    let shown = visible(
        graph,
        &RenderRequest {
            root: Some(render.root.clone()),
            depth: Some(u32::try_from(render.depth).unwrap_or(0)),
            expanded: render.expanded.clone(),
        },
    );
    let index: BTreeMap<&str, usize> = shown
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.node.node_key.as_str(), i))
        .collect();
    let mut chips_of: BTreeMap<usize, Vec<(i64, usize)>> = BTreeMap::new();
    for chip in chips(&shown) {
        chips_of
            .entry(chip.reader)
            .or_default()
            .push((chip.operand, chip.node));
    }
    for list in chips_of.values_mut() {
        list.sort();
    }
    let boxed: Vec<(&NodeBox, &VisibleNode, Vec<&DiagramNode>)> = render
        .boxes
        .iter()
        .filter_map(|b| {
            let i = *index.get(b.node_key.as_str())?;
            let chips = chips_of
                .get(&i)
                .map(|list| list.iter().map(|&(_, n)| &shown.nodes[n].node).collect())
                .unwrap_or_default();
            Some((b, &shown.nodes[i], chips))
        })
        .collect();
    let composites = boxed
        .iter()
        .filter(|(_, v, _)| v.role == Role::Composite)
        .count();

    let mut used = BTreeSet::new();
    for (_, node, chips) in &boxed {
        if !matches!(node.role, Role::Boundary(_)) {
            used.insert(node.node.glyph.as_str());
        }
        used.extend(chips.iter().map(|c| c.glyph.as_str()));
    }

    let crumbs = ancestors(graph, &render.root);
    let band = if crumbs.is_empty() { 0 } else { CRUMB_BAND };
    let crumb_width = {
        let text: usize = crumbs
            .iter()
            .map(|k| last_segment(k).chars().count() + 3)
            .sum::<usize>()
            + last_segment(&render.root).chars().count();
        2 * PAD + ADVANCE * text as i64
    };
    let mut id = format!(
        "ms-diagram-{}-{}",
        graph.view.name(),
        &graph.ir_hash[..graph.ir_hash.len().min(12)]
    );
    if band > 0 {
        id.push('-');
        id.extend(
            render
                .root
                .0
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }),
        );
    }
    let w = if band > 0 {
        render.width.max(crumb_width)
    } else {
        render.width
    };
    let h = render.height + band;
    let mut out = String::new();
    let _ = write!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" class=\"ms-glyphs ms-diagram\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\" role=\"img\" aria-labelledby=\"{id}-title {id}-desc\" style=\"max-width:100%;height:auto\">"
    );
    let _ = write!(
        out,
        "<title id=\"{id}-title\">{}: {} view{}</title>",
        escape(&graph.program),
        graph.view.name(),
        if band > 0 {
            format!(", scope {}", escape(&render.root.0))
        } else {
            String::new()
        }
    );
    let _ = write!(
        out,
        "<desc id=\"{id}-desc\">The {} view of train block {}, drawn from its verified Tensor IR: {} operators in {} ranks, parameters and dropout tokens with one reader drawn as chips on it.{}</desc>",
        graph.view.name(),
        escape(&graph.program),
        boxed.len() + boxed.iter().map(|(_, _, c)| c.len()).sum::<usize>(),
        render.boxes.iter().map(|b| b.rank + 1).max().unwrap_or(0),
        if composites > 0 {
            format!(
                " {composites} collapsed scopes are drawn as composite nodes with their ports{}.",
                if template.is_some() {
                    ", each linking to its own level"
                } else {
                    ""
                }
            )
        } else {
            String::new()
        }
    );
    out.push_str(&style());
    out.push_str("<defs>");
    out.push_str(&symbols(&used));
    out.push_str("</defs>");

    if band > 0 {
        out.push_str(&breadcrumb(&crumbs, &render.root, template));
        let _ = write!(out, "<g transform=\"translate(0 {band})\">");
    }
    let counts: BTreeMap<(&str, &str, i64), usize> = shown
        .edges
        .iter()
        .map(|e| {
            (
                (
                    shown.nodes[e.producer].node.node_key.as_str(),
                    shown.nodes[e.consumer].node.node_key.as_str(),
                    e.slot,
                ),
                e.count,
            )
        })
        .collect();
    out.push_str("<g class=\"edges\" fill=\"none\" stroke=\"var(--ms-diagram-edge, var(--ms-diagram-muted, #606770))\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\">");
    let mut labels = String::new();
    for route in &render.routes {
        let stopped = index.get(route.consumer.as_str()).is_some_and(|&i| {
            shown.nodes[i].role == Role::Operator
                && shown.nodes[i].node.op.as_deref() == Some("StopGradient")
        });
        let count = counts
            .get(&(
                route.producer.as_str(),
                route.consumer.as_str(),
                route.operand,
            ))
            .copied()
            .unwrap_or(1);
        let _ = write!(
            out,
            "<path class=\"edge\" data-from=\"{}\" data-to=\"{}\" data-operand=\"{}\"{}{} d=\"{}\"/>",
            attribute(&route.producer),
            attribute(&route.consumer),
            route.operand,
            if count > 1 {
                format!(" data-count=\"{count}\"")
            } else {
                String::new()
            },
            if stopped {
                " data-stop-gradient=\"true\" stroke-dasharray=\"6 5\""
            } else {
                ""
            },
            route_path(&route.points)
        );
        if count > 1
            && let [.., x0, y0, x1, y1] = route.points[..]
        {
            let _ = write!(
                labels,
                "<text class=\"edge-count\" x=\"{}\" y=\"{}\" font-size=\"10\" font-weight=\"600\" fill=\"{INK}\">×{count}</text>",
                (x0 + x1) / 2 + 6,
                (y0 + y1) / 2 + 4
            );
        }
    }
    out.push_str("</g>");
    if !labels.is_empty() {
        let _ = write!(out, "<g class=\"edge-counts\">{labels}</g>");
    }
    out.push_str("<g class=\"nodes\">");
    for (node_box, node, chips) in &boxed {
        match node.role {
            Role::Operator => {
                out.push_str(&node_group(graph, ir_lines, node_box, &node.node, chips));
            }
            Role::Composite => out.push_str(&composite_group(node_box, node, template)),
            Role::Boundary(direction) => {
                out.push_str(&boundary_group(ir_lines, node_box, node, direction));
            }
        }
    }
    out.push_str("</g>");
    if band > 0 {
        out.push_str("</g>");
    }
    out.push_str("</svg>");
    out
}

/// The breadcrumb of a level: each ancestor of `root`, outermost first, then `root` in bold,
/// separated by ` / `.
fn breadcrumb(crumbs: &[ScopeKey], root: &ScopeKey, template: Option<&str>) -> String {
    let mut out = String::from("<g class=\"breadcrumb\" font-size=\"11\">");
    let mut x = PAD;
    let y = 18;
    for key in crumbs {
        let text = last_segment(key);
        let item = format!(
            "<text x=\"{x}\" y=\"{y}\" fill=\"var(--ms-diagram-link, var(--ifm-link-color, #2e8555))\" text-decoration=\"underline\"><title>{}</title>{}</text>",
            escape(&key.0),
            escape(text)
        );
        match template {
            Some(template) => {
                let _ = write!(
                    out,
                    "<a class=\"crumb\" href=\"{}\">{item}</a>",
                    href(template, key)
                );
            }
            None => out.push_str(&item),
        }
        x += ADVANCE * text.chars().count() as i64;
        let _ = write!(
            out,
            "<text x=\"{x}\" y=\"{y}\" fill=\"{MUTED}\" xml:space=\"preserve\"> / </text>"
        );
        x += ADVANCE * 3;
    }
    let _ = write!(
        out,
        "<text x=\"{x}\" y=\"{y}\" font-weight=\"600\" fill=\"{INK}\"><title>{}</title>{}</text></g>",
        escape(&root.0),
        escape(last_segment(root))
    );
    out
}

/// A collapsed scope: its box, its In ports on the top edge over their labels, its glyph, name and
/// instance below them, its Out ports on the bottom edge with their labels beside them.
fn composite_group(node_box: &NodeBox, node: &VisibleNode, template: Option<&str>) -> String {
    let NodeBox {
        x,
        y,
        width,
        height,
        ..
    } = *node_box;
    let key = node
        .node
        .composite_of
        .as_ref()
        .map_or(ScopeKey(node.node.node_key.clone()), Clone::clone);
    let mut out = String::new();
    if let Some(template) = template {
        let _ = write!(
            out,
            "<a class=\"composite\" href=\"{}\">",
            href(template, &key)
        );
    }
    let list = |direction: PortDirection| {
        node.inputs
            .iter()
            .chain(&node.outputs)
            .filter(|p| p.direction == direction)
            .map(port_label)
            .collect::<Vec<_>>()
            .join(" ")
    };
    let _ = write!(
        out,
        "<g class=\"node\" data-key=\"{}\" data-kind=\"Composite\" data-glyph=\"{}\" data-in=\"{}\" data-out=\"{}\">",
        attribute(&node.node.node_key),
        attribute(node.node.glyph.as_str()),
        list(PortDirection::In),
        list(PortDirection::Out),
    );
    let (first, second) = label_lines(&node.node);
    let _ = write!(
        out,
        "<title>{}: {first}, {second}{}</title>",
        escape(&key.0),
        if template.is_some() {
            "; opens its level"
        } else {
            ""
        }
    );
    let _ = write!(
        out,
        "<rect x=\"{}\" y=\"{}\" width=\"{width}\" height=\"{height}\" rx=\"6\" fill=\"{MUTED}\" fill-opacity=\"10%\" stroke=\"{MUTED}\" stroke-width=\"1\"/>",
        x + 4,
        y + 4
    );
    let _ = write!(
        out,
        "<rect x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" rx=\"6\" fill=\"{PAPER}\" stroke=\"{INK}\" stroke-width=\"2\"/>"
    );
    let body = y + PORT_ROW;
    let _ = write!(
        out,
        "<use href=\"#ms-{}\" x=\"{}\" y=\"{}\" width=\"{GLYPH}\" height=\"{GLYPH}\"/>",
        node.node.glyph,
        x + PAD,
        body + (HEIGHT - GLYPH) / 2
    );
    let text_x = x + PAD + GLYPH + INNER;
    let _ = write!(
        out,
        "<text x=\"{text_x}\" y=\"{}\" font-size=\"11\" font-weight=\"600\" fill=\"{INK}\">{}</text>",
        body + 17,
        escape(first)
    );
    if !second.is_empty() {
        let _ = write!(
            out,
            "<text x=\"{text_x}\" y=\"{}\" font-size=\"10\" fill=\"{MUTED}\">{}</text>",
            body + 31,
            escape(second)
        );
    }
    let n = node.inputs.len() as i64;
    for (i, port) in node.inputs.iter().enumerate() {
        let px = x + width * (2 * i as i64 + 1) / (2 * n);
        let _ = write!(
            out,
            "<g class=\"port\" data-direction=\"In\" data-value=\"{}\"><title>In {}: {}[{}]</title><circle cx=\"{px}\" cy=\"{y}\" r=\"3\" fill=\"{PAPER}\" stroke=\"{INK}\" stroke-width=\"1.5\"/><text x=\"{px}\" y=\"{}\" font-size=\"9\" text-anchor=\"middle\" fill=\"{MUTED}\">{}</text></g>",
            port.value_id,
            port_label(port),
            port.dtype.name(),
            dims(&port.shape),
            y + 13,
            port_label(port)
        );
    }
    let m = node.outputs.len() as i64;
    for (j, port) in node.outputs.iter().enumerate() {
        let px = x + width * (2 * j as i64 + 1) / (2 * m);
        let bottom = y + height;
        let _ = write!(
            out,
            "<g class=\"port\" data-direction=\"Out\" data-value=\"{}\"><title>Out {}: {}[{}]</title><circle cx=\"{px}\" cy=\"{bottom}\" r=\"3\" fill=\"{PAPER}\" stroke=\"{INK}\" stroke-width=\"1.5\"/><text x=\"{}\" y=\"{}\" font-size=\"9\" fill=\"{MUTED}\">{}</text></g>",
            port.value_id,
            port_label(port),
            port.dtype.name(),
            dims(&port.shape),
            px + 6,
            bottom + 12,
            port_label(port)
        );
    }
    out.push_str("</g>");
    if template.is_some() {
        out.push_str("</a>");
    }
    out
}

/// A port of the level's root: a dashed box with its direction and value over its type.
fn boundary_group(
    ir_lines: &IrLines,
    node_box: &NodeBox,
    node: &VisibleNode,
    direction: PortDirection,
) -> String {
    let NodeBox {
        x,
        y,
        width,
        height,
        ..
    } = *node_box;
    let (first, second) = label_lines(&node.node);
    let value = format!("%{}", node.value);
    let title = match ir_lines.get(&value) {
        Some(line) => format!("{first}: {line}"),
        None => format!("{first} {second}"),
    };
    let mut out = String::new();
    let _ = write!(
        out,
        "<g class=\"node\" data-key=\"{}\" data-kind=\"Boundary\" data-direction=\"{direction:?}\" data-value=\"{}\"><title>{}</title>",
        attribute(&node.node.node_key),
        node.value,
        escape(&title)
    );
    let _ = write!(
        out,
        "<rect x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" rx=\"{}\" fill=\"{PAPER}\" stroke=\"{MUTED}\" stroke-width=\"1.5\" stroke-dasharray=\"4 3\"/>",
        height / 2
    );
    let _ = write!(
        out,
        "<text x=\"{}\" y=\"{}\" font-size=\"11\" font-weight=\"600\" fill=\"{INK}\">{}</text>",
        x + PAD,
        y + 17,
        escape(first)
    );
    if !second.is_empty() {
        let _ = write!(
            out,
            "<text x=\"{}\" y=\"{}\" font-size=\"10\" fill=\"{MUTED}\">{}</text>",
            x + PAD,
            y + 31,
            escape(second)
        );
    }
    out.push_str("</g>");
    out
}

fn dims(shape: &[u64]) -> String {
    shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
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
