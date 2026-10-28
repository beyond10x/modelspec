//! The layered layout of DESIGN.md § 13.5: a `DiagramRender` of a graph's visible level, every
//! coordinate an integer and every step deterministic (ordered maps, stable sorts, no float).
//!
//! 1. Visible: the nodes of the root scope (every node of a model graph; collapsed scopes and
//!    their ports arrive with story:diagram-zoom).
//! 2. Chips: a `ParamRead`, a `Scalar` or a dropout RNG token with exactly one visible reader is
//!    drawn as a chip inside its reader's box, on the right, and gets neither a box nor a route
//!    ([`chips`], [`chip_rects`]).
//! 3. Ranks (`ranks_of`): longest-path ranks from the sources, then tightened so the edges are
//!    short: a node with a consumer moves down to one rank above its first consumer when it has
//!    at least as many consumers as producers (an input read deep in the graph sits just above
//!    its first reader), and up to one rank below its last producer otherwise, until no node
//!    moves. An edge spanning more than one rank gets one dummy per rank it crosses.
//! 4. Order within a rank (`order_layers`): by the lowest ValueId a node holds (a dummy by its
//!    producer's), then alternating down and up barycenter sweeps over port positions (a
//!    consumer's operand slots, a producer's bottom points), each followed by a transpose step
//!    that keeps an exchange of neighbours only when it lowers the crossings; the ordering with
//!    the fewest crossings seen is kept. Barycenters are exact fractions compared by
//!    cross-multiplying `i64` sums, ties keeping the earlier order. Operand slots keep operand
//!    order; the producers are ordered to match.
//! 5. Coordinates: a box is 40 units high and as wide as its glyph (24), its label (7 units a
//!    character on its first line, 6 on its smaller second), its chips and 12 units of padding
//!    each side; 24 units between boxes and 48 between ranks. Each rank is first centred on the
//!    widest, then down and up passes place each rank where its links' ports best meet the ports
//!    at their other ends (weighted least squares, long-edge links weighing more), keeping the
//!    order and the gaps and staying inside the widest rank's extent (weighted isotonic
//!    regression in integers clamped to that extent), so the drawing is no wider than its widest
//!    rank; then each long edge is drawn as few vertical runs as its dummies' room allows
//!    (`straighten`).
//! 6. Routes: an edge leaves its producer's bottom (a producer's routes at evenly spaced points,
//!    ordered by where each goes next), passes each dummy's rank vertically and enters its
//!    consumer's top at the slot of its operand, `x + width·(2i + 1) / (2n)`.
//!
//! [`crossings`], [`route_length`] and [`detours`] measure a render.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::model::{
    DiagramGraph, DiagramNode, DiagramRender, DiagramRenderId, DiagramRenderState, DiagramView,
    EdgeRoute, NodeBox, NodeKind, ScopeKey,
};

/// The glyph's side.
pub const GLYPH: i64 = 24;
/// The advance of one label character.
pub const ADVANCE: i64 = 7;
/// The advance of one character of the label's second line, set smaller.
pub const ADVANCE_SECOND: i64 = 6;
/// The padding left of the glyph and right of the label or the chips.
pub const PAD: i64 = 12;
/// Between the glyph and the label, and between the label and each chip column.
pub const INNER: i64 = 8;
/// The height of an operator or endpoint box.
pub const HEIGHT: i64 = 40;
/// The height of a composite box, with its port row.
pub const COMPOSITE_HEIGHT: i64 = 56;
/// Between two boxes of one rank.
pub const GAP: i64 = 24;
/// Between two ranks.
pub const RANK_GAP: i64 = 48;
/// Around the drawing.
pub const MARGIN: i64 = 16;
/// The height of a chip.
pub const CHIP_HEIGHT: i64 = 16;
/// The side of a chip's glyph.
pub const CHIP_GLYPH: i64 = 12;
/// The advance of one chip character.
pub const CHIP_ADVANCE: i64 = 6;

/// The unit of an ordering position: a rank's k-th item spans `[k·SCALE, (k+1)·SCALE)`, and its ports
/// divide that span (exactly for up to ten ports, 5,040 being divisible by every `2n` up to 20).
const SCALE: i64 = 5040;
/// Down and up barycenter sweeps at most, each followed by the transpose step.
const SWEEPS: usize = 24;
/// Sweeps without a better ordering before the ordering stops.
const PATIENCE: usize = 4;
/// Passes of the transpose step over all ranks at most.
const TRANSPOSE_PASSES: usize = 16;
/// Down and up placement passes.
const PLACE_PASSES: usize = 12;

/// What to draw: the root scope (the graph's Program scope when absent), how many levels below it
/// are expanded (the view's default when absent) and the scopes expanded beyond that.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderRequest {
    pub root: Option<ScopeKey>,
    pub depth: Option<u32>,
    pub expanded: Vec<ScopeKey>,
}

/// The default depth of a view (DESIGN.md § 13.3): 1 in the model view, 0 in a pipeline view.
pub fn default_depth(view: DiagramView) -> u32 {
    match view {
        DiagramView::Model => 1,
        DiagramView::Train | DiagramView::Eval | DiagramView::Infer => 0,
    }
}

/// A node drawn as a chip on one operand slot of its only reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chip {
    /// The chip's node, an index into `graph.nodes`.
    pub node: usize,
    /// The reader's node, an index into `graph.nodes`.
    pub reader: usize,
    pub operand: i64,
}

/// The chips of the level `root` of `graph`, in node order.
pub fn chips(graph: &DiagramGraph, root: &ScopeKey) -> Vec<Chip> {
    let index = key_index(graph);
    let visible = |i: usize| graph.nodes[i].scope == *root;
    let mut readers: BTreeMap<usize, Vec<(usize, i64)>> = BTreeMap::new();
    for edge in &graph.edges {
        let (Some(&p), Some(&c)) = (
            index.get(edge.producer.as_str()),
            index.get(edge.consumer.as_str()),
        ) else {
            continue;
        };
        if visible(p) && visible(c) {
            readers.entry(p).or_default().push((c, edge.operand));
        }
    }
    let mut found = Vec::new();
    for (node, reads) in readers {
        if let [(reader, operand)] = reads[..]
            && chip_kind(&graph.nodes[node])
            && !chip_kind(&graph.nodes[reader])
        {
            found.push(Chip {
                node,
                reader,
                operand,
            });
        }
    }
    found
}

fn chip_kind(node: &DiagramNode) -> bool {
    node.kind == NodeKind::Operator
        && (matches!(node.op.as_deref(), Some("ParamRead" | "Scalar"))
            || node.glyph.as_str() == "op-input-token")
}

/// The two lines of a node's label: the operator and its value, then its name and type.
pub fn label_lines(node: &DiagramNode) -> (&str, &str) {
    node.label.split_once('\n').unwrap_or((&node.label, ""))
}

/// The text of a chip: the name on its label's second line after its last `.` (a dropout
/// token's `dropout#<k>`), else that whole line.
pub fn chip_text(node: &DiagramNode) -> &str {
    let (_, second) = label_lines(node);
    match second.rsplit_once(' ') {
        Some((name, _)) => name.rsplit('.').next().unwrap_or(name),
        None => second,
    }
}

fn chars(text: &str) -> i64 {
    text.chars().count() as i64
}

/// The width of a chip.
pub fn chip_width(node: &DiagramNode) -> i64 {
    4 + CHIP_GLYPH + 4 + CHIP_ADVANCE * chars(chip_text(node)) + 6
}

/// Chips stand in columns of two; the width of each column.
fn chip_columns(chips: &[&DiagramNode]) -> Vec<i64> {
    chips
        .chunks(2)
        .map(|column| column.iter().map(|c| chip_width(c)).max().unwrap_or(0))
        .collect()
}

/// The size of a node's box with `chips` (its chips, by operand).
pub fn box_size(node: &DiagramNode, chips: &[&DiagramNode]) -> (i64, i64) {
    let (first, second) = label_lines(node);
    let text = (ADVANCE * chars(first)).max(ADVANCE_SECOND * chars(second));
    let columns: i64 = chip_columns(chips).iter().map(|w| w + INNER).sum();
    let height = match node.kind {
        NodeKind::Composite => COMPOSITE_HEIGHT,
        NodeKind::Operator | NodeKind::Endpoint => HEIGHT,
    };
    (PAD + GLYPH + INNER + text + columns + PAD, height)
}

/// Where each of `chips` (by operand) is drawn inside `node_box`: `(x, y, width, height)`.
pub fn chip_rects(node_box: &NodeBox, chips: &[&DiagramNode]) -> Vec<(i64, i64, i64, i64)> {
    let columns = chip_columns(chips);
    let total: i64 = columns.iter().map(|w| w + INNER).sum();
    let mut x = node_box.x + node_box.width - PAD - total + INNER;
    let mut rects = Vec::new();
    for (column, width) in chips.chunks(2).zip(&columns) {
        let ys: &[i64] = match column.len() {
            1 => &[(HEIGHT - CHIP_HEIGHT) / 2],
            _ => &[3, HEIGHT - 3 - CHIP_HEIGHT],
        };
        for (chip, dy) in column.iter().zip(ys) {
            rects.push((x, node_box.y + dy, chip_width(chip), CHIP_HEIGHT));
        }
        x += width + INNER;
    }
    rects
}

fn key_index(graph: &DiagramGraph) -> BTreeMap<&str, usize> {
    graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.node_key.as_str(), i))
        .collect()
}

/// One entry of a rank: a visible node or a dummy of a long edge.
struct Item {
    rank: usize,
    order: (i64, i64, usize),
    width: i64,
    height: i64,
    dummy: bool,
    /// The links entering the item from the rank above.
    ins: Vec<usize>,
    /// The links leaving the item to the rank below.
    outs: Vec<usize>,
}

/// One step of an edge between two adjacent ranks, entering `lower` at operand slot `slot` of
/// `slots` (a dummy has one slot).
struct Link {
    upper: usize,
    lower: usize,
    slot: i64,
    slots: i64,
}

fn connect(
    items: &mut [Item],
    links: &mut Vec<Link>,
    upper: usize,
    lower: usize,
    slot: i64,
    slots: i64,
) {
    items[upper].outs.push(links.len());
    items[lower].ins.push(links.len());
    links.push(Link {
        upper,
        lower,
        slot,
        slots,
    });
}

/// Longest-path ranks from the sources, then tightened: every node that has a consumer moves to
/// the rank between its producers and its consumers that makes its edges shortest, down to one
/// rank above its first consumer when it has at least as many consumers as producers (so a
/// source sits just above its first reader) and up to one rank below its last producer
/// otherwise, until no node moves; the ranks are then shifted to start at 0.
fn ranks_of(nodes: &[usize], edges: &[(usize, usize, usize)]) -> BTreeMap<usize, usize> {
    let mut rank: BTreeMap<usize, i64> = nodes.iter().map(|&i| (i, 0)).collect();
    for _ in 0..=nodes.len() {
        let mut changed = false;
        for &(_, p, c) in edges {
            if rank[&c] < rank[&p] + 1 {
                rank.insert(c, rank[&p] + 1);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut preds: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut succs: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(_, p, c) in edges {
        preds.entry(c).or_default().push(p);
        succs.entry(p).or_default().push(c);
    }
    for _ in 0..4 * nodes.len() + 4 {
        let mut sequence = nodes.to_vec();
        sequence.sort_by_key(|&i| (std::cmp::Reverse(rank[&i]), i));
        let mut changed = false;
        for i in sequence {
            let Some(below) = succs.get(&i) else {
                continue;
            };
            let above = preds.get(&i).map_or(&[][..], Vec::as_slice);
            let high = below.iter().map(|c| rank[c]).min().unwrap_or(0) - 1;
            let low = above.iter().map(|p| rank[p] + 1).max().unwrap_or(0);
            let target = if below.len() >= above.len() {
                high
            } else {
                low
            };
            if target != rank[&i] {
                rank.insert(i, target);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let first = rank.values().copied().min().unwrap_or(0);
    rank.into_iter()
        .map(|(i, r)| (i, (r - first) as usize))
        .collect()
}

/// The render of `graph` that `request` asks for.
pub fn render(graph: &DiagramGraph, request: &RenderRequest) -> DiagramRender {
    let root = request.root.clone().unwrap_or_else(|| {
        graph
            .scopes
            .iter()
            .find(|s| s.parent.is_none())
            .map(|s| s.key.clone())
            .unwrap_or_else(|| ScopeKey(graph.program.clone()))
    });
    let depth = request.depth.unwrap_or_else(|| default_depth(graph.view));
    let index = key_index(graph);
    let chip_list = chips(graph, &root);
    let mut chips_of: BTreeMap<usize, Vec<(i64, usize)>> = BTreeMap::new();
    for chip in &chip_list {
        chips_of
            .entry(chip.reader)
            .or_default()
            .push((chip.operand, chip.node));
    }
    for list in chips_of.values_mut() {
        list.sort();
    }
    let is_chip = |i: usize| chip_list.iter().any(|c| c.node == i);
    let ranked: Vec<usize> = (0..graph.nodes.len())
        .filter(|&i| graph.nodes[i].scope == root && !is_chip(i))
        .collect();
    let ranked_set: std::collections::BTreeSet<usize> = ranked.iter().copied().collect();
    let edges: Vec<(usize, usize, usize)> = graph
        .edges
        .iter()
        .enumerate()
        .filter_map(|(e, edge)| {
            let p = *index.get(edge.producer.as_str())?;
            let c = *index.get(edge.consumer.as_str())?;
            (ranked_set.contains(&p) && ranked_set.contains(&c)).then_some((e, p, c))
        })
        .collect();

    let rank = ranks_of(&ranked, &edges);
    let mut operands: BTreeMap<usize, i64> = BTreeMap::new();
    for edge in &graph.edges {
        if let Some(&c) = index.get(edge.consumer.as_str()) {
            let n = operands.entry(c).or_insert(0);
            *n = (*n).max(edge.operand + 1);
        }
    }

    let chips_for = |i: usize| -> Vec<&DiagramNode> {
        chips_of
            .get(&i)
            .map(|list| list.iter().map(|&(_, n)| &graph.nodes[n]).collect())
            .unwrap_or_default()
    };
    let mut items: Vec<Item> = Vec::new();
    let mut item_of: BTreeMap<usize, usize> = BTreeMap::new();
    for &i in &ranked {
        let node = &graph.nodes[i];
        let (width, height) = box_size(node, &chips_for(i));
        item_of.insert(i, items.len());
        items.push(Item {
            rank: rank[&i],
            order: (node.value_id.unwrap_or(i64::MAX), 0, i),
            width,
            height,
            dummy: false,
            ins: Vec::new(),
            outs: Vec::new(),
        });
    }
    let mut links: Vec<Link> = Vec::new();
    let mut chains: Vec<Vec<usize>> = Vec::new();
    let mut first_link: Vec<usize> = Vec::new();
    for &(e, p, c) in &edges {
        let producer_value = graph.nodes[p].value_id.unwrap_or(i64::MAX);
        let mut previous = item_of[&p];
        let mut chain = Vec::new();
        first_link.push(links.len());
        for r in rank[&p] + 1..rank[&c] {
            let dummy = items.len();
            items.push(Item {
                rank: r,
                order: (producer_value, 1, e),
                width: 0,
                height: 0,
                dummy: true,
                ins: Vec::new(),
                outs: Vec::new(),
            });
            connect(&mut items, &mut links, previous, dummy, 0, 1);
            chain.push(dummy);
            previous = dummy;
        }
        let slots = operands.get(&c).copied().unwrap_or(1).max(1);
        connect(
            &mut items,
            &mut links,
            previous,
            item_of[&c],
            graph.edges[e].operand,
            slots,
        );
        chains.push(chain);
    }

    let ranks = items.iter().map(|it| it.rank + 1).max().unwrap_or(0);
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); ranks];
    for (i, item) in items.iter().enumerate() {
        layers[item.rank].push(i);
    }
    for layer in &mut layers {
        layer.sort_by_key(|&i| items[i].order);
    }
    order_layers(&items, &links, &mut layers);

    let row_height: Vec<i64> = layers
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|&i| items[i].height)
                .max()
                .unwrap_or(HEIGHT)
                .max(HEIGHT)
        })
        .collect();
    let mut row_y = Vec::with_capacity(ranks);
    let mut y = MARGIN;
    for height in &row_height {
        row_y.push(y);
        y += height + RANK_GAP;
    }
    let height = if ranks == 0 {
        2 * MARGIN + HEIGHT
    } else {
        y - RANK_GAP + MARGIN
    };

    let places = port_places(&items, &links, &positions(&items, &layers));
    let centre = place(&items, &links, &layers, &places, &chains);
    let left = |i: usize| centre[i] - items[i].width / 2;
    let min_left = (0..items.len()).map(left).min().unwrap_or(MARGIN);
    let shift = MARGIN - min_left;
    let x = |i: usize| left(i) + shift;
    let width = (0..items.len())
        .map(|i| x(i) + items[i].width)
        .max()
        .unwrap_or(MARGIN)
        + MARGIN;

    let boxes: Vec<NodeBox> = ranked
        .iter()
        .map(|&i| {
            let item = &items[item_of[&i]];
            NodeBox {
                node_key: graph.nodes[i].node_key.clone(),
                x: x(item_of[&i]),
                y: row_y[item.rank],
                width: item.width,
                height: item.height,
                rank: item.rank as i64,
            }
        })
        .collect();

    // A producer's routes leave its bottom at evenly spaced points, ordered by where each goes
    // next ([`port_places`]), so a fan-out does not start as one stroke.
    let points: Vec<Vec<i64>> = edges
        .iter()
        .zip(&chains)
        .zip(&first_link)
        .map(|((&(e, p, c), chain), &first)| {
            let pi = item_of[&p];
            let (j, m) = places[first];
            let mut points = vec![
                x(pi) + items[pi].width * (2 * j + 1) / (2 * m),
                row_y[items[pi].rank] + items[pi].height,
            ];
            for &d in chain {
                let r = items[d].rank;
                points.extend([x(d), row_y[r], x(d), row_y[r] + row_height[r]]);
            }
            let ci = item_of[&c];
            let n = operands.get(&c).copied().unwrap_or(1).max(1);
            points.extend([
                x(ci) + items[ci].width * (2 * graph.edges[e].operand + 1) / (2 * n),
                row_y[items[ci].rank],
            ]);
            points
        })
        .collect();
    let routes = edges
        .iter()
        .zip(points)
        .map(|(&(e, _, _), points)| {
            let edge = &graph.edges[e];
            EdgeRoute {
                producer: edge.producer.clone(),
                consumer: edge.consumer.clone(),
                operand: edge.operand,
                points,
            }
        })
        .collect();

    let mut render_id = format!("{}#{}@{depth}", graph.graph_id.0, root.0);
    for scope in &request.expanded {
        render_id.push('+');
        render_id.push_str(&scope.0);
    }
    DiagramRender {
        render_id: DiagramRenderId(render_id),
        graph_id: graph.graph_id.clone(),
        root,
        depth: i64::from(depth),
        expanded: request.expanded.clone(),
        width,
        height,
        boxes,
        routes,
        state: DiagramRenderState::LaidOut,
    }
}

/// A route segment between two adjacent ranks, `((x0, y0), (x1, y1))`.
type Segment = ((i64, i64), (i64, i64));

/// The segments of a route between two adjacent ranks, `((x0, y0), (x1, y1))`: a route's points
/// are its producer's bottom point, a top and a bottom point per dummy and its consumer's slot, so
/// the segments between ranks are the pairs starting at even points.
fn gap_segments(route: &EdgeRoute) -> impl Iterator<Item = Segment> + '_ {
    route
        .points
        .chunks(4)
        .filter(|c| c.len() == 4)
        .map(|c| ((c[0], c[1]), (c[2], c[3])))
}

/// How often two routes cross: pairs of their segments that end on the same rank's top and
/// whose ends are in opposite orders at their two ends. Both are drawn as the same vertical-tangent
/// curve over the same gap, so they meet exactly when their order flips; a shared end point is
/// not a crossing.
pub fn route_crossings(a: &EdgeRoute, b: &EdgeRoute) -> u64 {
    let mut count = 0;
    for (a0, a1) in gap_segments(a) {
        for (b0, b1) in gap_segments(b) {
            if a1.1 == b1.1 && (a0.0 - b0.0).signum() * (a1.0 - b1.0).signum() < 0 {
                count += 1;
            }
        }
    }
    count
}

/// The number of crossings of `render`: every pair of route segments between two adjacent ranks
/// that cross, at their real ports (a producer's bottom points, a consumer's operand slots).
pub fn crossings(render: &DiagramRender) -> u64 {
    let routes = &render.routes;
    let mut by_gap: BTreeMap<i64, Vec<Segment>> = BTreeMap::new();
    for route in routes {
        for segment in gap_segments(route) {
            by_gap.entry(segment.1.1).or_default().push(segment);
        }
    }
    by_gap
        .values()
        .map(|segments| {
            let mut count = 0;
            for (k, (a0, a1)) in segments.iter().enumerate() {
                for (b0, b1) in &segments[k + 1..] {
                    if (a0.0 - b0.0).signum() * (a1.0 - b1.0).signum() < 0 {
                        count += 1;
                    }
                }
            }
            count
        })
        .sum()
}

/// The total length of `render`'s routes, each step between two points counted as its
/// horizontal plus its vertical extent.
pub fn route_length(render: &DiagramRender) -> i64 {
    render
        .routes
        .iter()
        .map(|r| {
            let points: Vec<&[i64]> = r.points.chunks(2).collect();
            points
                .windows(2)
                .map(|w| (w[0][0] - w[1][0]).abs() + (w[0][1] - w[1][1]).abs())
                .sum::<i64>()
        })
        .sum()
}

/// The number of long edges (routes through at least one dummy) with a point outside the
/// horizontal span of their producer's and consumer's boxes together.
pub fn detours(render: &DiagramRender) -> u64 {
    let boxes: BTreeMap<&str, &NodeBox> = render
        .boxes
        .iter()
        .map(|b| (b.node_key.as_str(), b))
        .collect();
    render
        .routes
        .iter()
        .filter(|r| r.points.len() > 4)
        .filter(|r| {
            let (Some(p), Some(c)) = (
                boxes.get(r.producer.as_str()),
                boxes.get(r.consumer.as_str()),
            ) else {
                return false;
            };
            let low = p.x.min(c.x);
            let high = (p.x + p.width).max(c.x + c.width);
            r.points.chunks(2).any(|q| q[0] < low || q[0] > high)
        })
        .count() as u64
}

/// `sum / count`, compared by cross-multiplication.
#[derive(Clone, Copy)]
struct Fraction {
    sum: i64,
    count: i64,
}

impl Fraction {
    fn cmp(self, other: Fraction) -> Ordering {
        (self.sum * other.count).cmp(&(other.sum * self.count))
    }
}

/// Each item's index in its rank.
fn positions(items: &[Item], layers: &[Vec<usize>]) -> Vec<i64> {
    let mut position = vec![0i64; items.len()];
    for layer in layers {
        for (k, &i) in layer.iter().enumerate() {
            position[i] = k as i64;
        }
    }
    position
}

/// Where `link` enters its lower item, in ordering units: its operand slot of the item's span.
fn lower_port(link: &Link, position: &[i64]) -> i64 {
    position[link.lower] * SCALE + SCALE * (2 * link.slot + 1) / (2 * link.slots)
}

/// The place `(j, m)` of every link among the `m` links leaving its upper item: ordered by where
/// each enters the rank below, then by link.
fn port_places(items: &[Item], links: &[Link], position: &[i64]) -> Vec<(i64, i64)> {
    let mut places = vec![(0, 1); links.len()];
    for item in items {
        let mut outs: Vec<(i64, usize)> = item
            .outs
            .iter()
            .map(|&l| (lower_port(&links[l], position), l))
            .collect();
        outs.sort();
        let m = outs.len() as i64;
        for (j, &(_, l)) in outs.iter().enumerate() {
            places[l] = (j as i64, m);
        }
    }
    places
}

/// The `(upper, lower)` ports, in ordering units, of the links leaving `layer`.
fn gap_ports(
    items: &[Item],
    links: &[Link],
    layer: &[usize],
    position: &[i64],
) -> Vec<(usize, i64, i64)> {
    let mut ports = Vec::new();
    for &u in layer {
        let mut outs: Vec<(i64, usize)> = items[u]
            .outs
            .iter()
            .map(|&l| (lower_port(&links[l], position), l))
            .collect();
        outs.sort();
        let m = outs.len() as i64;
        for (j, &(lower, l)) in outs.iter().enumerate() {
            let upper = position[u] * SCALE + SCALE * (2 * j as i64 + 1) / (2 * m);
            ports.push((l, upper, lower));
        }
    }
    ports
}

/// The crossings between `layer` and the rank below it: pairs of links whose ports are in
/// opposite orders in the two ranks.
fn gap_crossings(items: &[Item], links: &[Link], layer: &[usize], position: &[i64]) -> u64 {
    let ports = gap_ports(items, links, layer, position);
    let mut count = 0;
    for (k, &(_, a0, a1)) in ports.iter().enumerate() {
        for &(_, b0, b1) in &ports[k + 1..] {
            if (a0 - b0).signum() * (a1 - b1).signum() < 0 {
                count += 1;
            }
        }
    }
    count
}

fn total_crossings(items: &[Item], links: &[Link], layers: &[Vec<usize>], position: &[i64]) -> u64 {
    layers
        .iter()
        .map(|layer| gap_crossings(items, links, layer, position))
        .sum()
}

/// The crossings on both sides of rank `r`.
fn crossings_around(
    items: &[Item],
    links: &[Link],
    layers: &[Vec<usize>],
    position: &[i64],
    r: usize,
) -> u64 {
    let above = if r > 0 {
        gap_crossings(items, links, &layers[r - 1], position)
    } else {
        0
    };
    above + gap_crossings(items, links, &layers[r], position)
}

/// Sorts rank `r` by the barycenter of its links to the rank above (`down`) or below: for each
/// link, where the item would have to stand for its port at that end to meet the port at the
/// other end; an item without such links stays where it is. Ties keep the earlier order.
fn sort_by_barycenter(
    items: &[Item],
    links: &[Link],
    layers: &mut [Vec<usize>],
    position: &mut [i64],
    r: usize,
    down: bool,
) {
    let other = if down { r - 1 } else { r };
    let ports: BTreeMap<usize, (i64, i64)> = gap_ports(items, links, &layers[other], position)
        .into_iter()
        .map(|(l, upper, lower)| (l, (upper, lower)))
        .collect();
    let bary: BTreeMap<usize, Fraction> = layers[r]
        .iter()
        .map(|&i| {
            let item = &items[i];
            let own = position[i] * SCALE;
            let wanted: Vec<i64> = if down {
                item.ins
                    .iter()
                    .map(|l| {
                        let (upper, lower) = ports[l];
                        upper - (lower - own)
                    })
                    .collect()
            } else {
                item.outs
                    .iter()
                    .map(|l| {
                        let (upper, lower) = ports[l];
                        lower - (upper - own)
                    })
                    .collect()
            };
            let fraction = if wanted.is_empty() {
                Fraction { sum: own, count: 1 }
            } else {
                Fraction {
                    sum: wanted.iter().sum(),
                    count: wanted.len() as i64,
                }
            };
            (i, fraction)
        })
        .collect();
    layers[r].sort_by(|a, b| bary[a].cmp(bary[b]));
    for (k, &i) in layers[r].iter().enumerate() {
        position[i] = k as i64;
    }
}

/// Exchanges neighbours within each rank while an exchange lowers the crossings on both sides
/// of that rank.
fn transpose(items: &[Item], links: &[Link], layers: &mut [Vec<usize>], position: &mut [i64]) {
    for _ in 0..TRANSPOSE_PASSES {
        let mut improved = false;
        for r in 0..layers.len() {
            for k in 0..layers[r].len().saturating_sub(1) {
                let before = crossings_around(items, links, layers, position, r);
                let (a, b) = (layers[r][k], layers[r][k + 1]);
                layers[r].swap(k, k + 1);
                position.swap(a, b);
                if crossings_around(items, links, layers, position, r) < before {
                    improved = true;
                } else {
                    layers[r].swap(k, k + 1);
                    position.swap(a, b);
                }
            }
        }
        if !improved {
            break;
        }
    }
}

/// Down and up barycenter sweeps over port positions, each followed by the transpose step,
/// keeping the ordering with the fewest crossings seen (the first of equals); the sweeps stop at
/// zero crossings, after [`PATIENCE`] sweeps without a better ordering or after [`SWEEPS`].
fn order_layers(items: &[Item], links: &[Link], layers: &mut Vec<Vec<usize>>) {
    let mut position = positions(items, layers);
    let mut best = layers.clone();
    let mut fewest = total_crossings(items, links, layers, &position);
    let mut stale = 0;
    for sweep in 0..SWEEPS {
        if fewest == 0 || stale >= PATIENCE {
            break;
        }
        if sweep % 2 == 0 {
            for r in 1..layers.len() {
                sort_by_barycenter(items, links, layers, &mut position, r, true);
            }
        } else {
            for r in (0..layers.len().saturating_sub(1)).rev() {
                sort_by_barycenter(items, links, layers, &mut position, r, false);
            }
        }
        transpose(items, links, layers, &mut position);
        let count = total_crossings(items, links, layers, &position);
        if count < fewest {
            fewest = count;
            best = layers.clone();
            stale = 0;
        } else {
            stale += 1;
        }
    }
    *layers = best;
}

/// How strongly a link pulls its ends into line: 8 between two dummies, 2 between a dummy and a
/// box, 1 between two boxes, so a long edge runs straight.
fn weight(items: &[Item], link: &Link) -> i64 {
    match (items[link.upper].dummy, items[link.lower].dummy) {
        (true, true) => 8,
        (false, false) => 1,
        _ => 2,
    }
}

/// The offset of a link's two ports from the centres of its items.
fn port_offsets(items: &[Item], link: &Link, place: (i64, i64)) -> (i64, i64) {
    let (j, m) = place;
    let upper = items[link.upper].width;
    let lower = items[link.lower].width;
    (
        upper * (2 * j + 1) / (2 * m) - upper / 2,
        lower * (2 * link.slot + 1) / (2 * link.slots) - lower / 2,
    )
}

/// The centre x of every item: each rank centred on the widest, then [`PLACE_PASSES`] down and up
/// passes that place each rank where its links' ports best meet the ports at their other ends,
/// then the long edges straightened ([`straighten`]).
fn place(
    items: &[Item],
    links: &[Link],
    layers: &[Vec<usize>],
    places: &[(i64, i64)],
    chains: &[Vec<usize>],
) -> Vec<i64> {
    let mut centre = vec![0i64; items.len()];
    let span = |layer: &[usize]| -> i64 {
        layer.iter().map(|&i| items[i].width).sum::<i64>() + GAP * (layer.len() as i64 - 1).max(0)
    };
    let widest = layers.iter().map(|l| span(l)).max().unwrap_or(0);
    for layer in layers {
        let mut x = (widest - span(layer)) / 2;
        for &i in layer {
            centre[i] = x + items[i].width / 2;
            x += items[i].width + GAP;
        }
    }
    let offsets: Vec<(i64, i64)> = links
        .iter()
        .zip(places)
        .map(|(link, &place)| port_offsets(items, link, place))
        .collect();
    for _ in 0..PLACE_PASSES {
        for layer in layers {
            align(items, links, &offsets, layer, &mut centre, widest);
        }
        for layer in layers.iter().rev() {
            align(items, links, &offsets, layer, &mut centre, widest);
        }
    }
    straighten(items, layers, chains, &mut centre, widest);
    centre
}

/// Where dummy `d` may stand with its rank's other items fixed: [`GAP`] from its neighbours and
/// inside `[0, widest]`.
fn room(
    items: &[Item],
    layers: &[Vec<usize>],
    position: &[i64],
    centre: &[i64],
    d: usize,
    widest: i64,
) -> (i64, i64) {
    let layer = &layers[items[d].rank];
    let k = position[d] as usize;
    let low = match k.checked_sub(1).map(|j| layer[j]) {
        Some(n) => centre[n] + (items[n].width + 1) / 2 + GAP,
        None => 0,
    };
    let high = match layer.get(k + 1) {
        Some(&n) => centre[n] - items[n].width / 2 - GAP,
        None => widest,
    };
    (low, high)
}

/// Draws each long edge (longest first, then in edge order) as few vertical runs as its
/// dummies' room allows: from its first dummy down, as many dummies as share some x in their
/// rooms form one run, placed at the lower median of their x clamped to that shared room.
fn straighten(
    items: &[Item],
    layers: &[Vec<usize>],
    chains: &[Vec<usize>],
    centre: &mut [i64],
    widest: i64,
) {
    let position = positions(items, layers);
    let mut order: Vec<&Vec<usize>> = chains.iter().filter(|c| c.len() > 1).collect();
    order.sort_by_key(|c| std::cmp::Reverse(c.len()));
    for chain in order {
        let mut start = 0;
        while start < chain.len() {
            let (mut low, mut high) = (i64::MIN, i64::MAX);
            let mut end = start;
            while end < chain.len() {
                let (l, h) = room(items, layers, &position, centre, chain[end], widest);
                if low.max(l) > high.min(h) {
                    break;
                }
                low = low.max(l);
                high = high.min(h);
                end += 1;
            }
            let end = end.max(start + 1);
            let mut xs: Vec<i64> = chain[start..end].iter().map(|&d| centre[d]).collect();
            xs.sort();
            let x = xs[(xs.len() - 1) / 2].clamp(low.min(high), high);
            for &d in &chain[start..end] {
                centre[d] = x;
            }
            start = end;
        }
    }
}

/// Places the items of `layer`, the other ranks fixed, to minimise the weighted squared distance
/// between the two ports of every link they have: weighted least squares under
/// `c[k+1] - c[k] >= (w[k] + w[k+1]) / 2 + GAP`, solved by pooling adjacent violators on
/// `c[k] - offset[k]` with exact comparisons, each pooled mean rounded down; an item without
/// links stays where it is.
fn align(
    items: &[Item],
    links: &[Link],
    offsets: &[(i64, i64)],
    layer: &[usize],
    centre: &mut [i64],
    widest: i64,
) {
    if layer.is_empty() {
        return;
    }
    let mut offset = Vec::with_capacity(layer.len());
    let mut acc = 0i64;
    for (k, &i) in layer.iter().enumerate() {
        if k > 0 {
            let prev = layer[k - 1];
            acc += (items[prev].width + 1) / 2 + items[i].width / 2 + GAP;
        }
        offset.push(acc);
    }
    // (weighted sum of wanted c - offset, total weight, items)
    let mut blocks: Vec<(i64, i64, usize)> = Vec::new();
    for (k, &i) in layer.iter().enumerate() {
        let (mut sum, mut total) = (0i64, 0i64);
        for &l in &items[i].ins {
            let link = &links[l];
            let w = weight(items, link);
            let (up, low) = offsets[l];
            sum += w * (centre[link.upper] + up - low);
            total += w;
        }
        for &l in &items[i].outs {
            let link = &links[l];
            let w = weight(items, link);
            let (up, low) = offsets[l];
            sum += w * (centre[link.lower] + low - up);
            total += w;
        }
        if total == 0 {
            sum = centre[i];
            total = 1;
        }
        blocks.push((sum - total * offset[k], total, 1));
        while blocks.len() > 1 {
            let (s1, n1, k1) = blocks[blocks.len() - 1];
            let (s0, n0, k0) = blocks[blocks.len() - 2];
            if s0 * n1 <= s1 * n0 {
                break;
            }
            blocks.pop();
            blocks.pop();
            blocks.push((s0 + s1, n0 + n1, k0 + k1));
        }
    }
    // Every item stays inside [0, widest]: in z = c - offset that is one constant interval for
    // all of them, and clamping an isotonic solution to it keeps it optimal and ordered.
    let (first, last) = (layer[0], layer[layer.len() - 1]);
    let low = items[first].width / 2;
    let high = (widest - offset[layer.len() - 1] - (items[last].width + 1) / 2).max(low);
    let mut k = 0;
    for (sum, count, len) in blocks {
        let value = sum.div_euclid(count).clamp(low, high);
        for _ in 0..len {
            centre[layer[k]] = value + offset[k];
            k += 1;
        }
    }
}
