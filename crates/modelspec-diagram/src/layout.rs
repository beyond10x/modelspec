//! The layered layout of DESIGN.md § 13.5: a `DiagramRender` of a graph's visible level, every
//! coordinate an integer and every step deterministic (ordered maps, stable sorts, no float).
//!
//! 1. Visible: the nodes of the root scope (every node of a model graph; collapsed scopes and
//!    their ports arrive with story:diagram-zoom).
//! 2. Chips: a `ParamRead`, a `Scalar` or a dropout RNG token with exactly one visible reader is
//!    drawn as a chip inside its reader's box, on the right, and gets neither a box nor a route
//!    ([`chips`], [`chip_rects`]).
//! 3. Ranks: 0 for sources, otherwise one more than the highest rank among a node's producers; an
//!    edge spanning more than one rank gets one dummy per rank it crosses.
//! 4. Order within a rank: by the lowest ValueId a node holds (a dummy by its producer's), then
//!    four down and up barycenter sweeps, each barycenter compared as an exact fraction by
//!    cross-multiplying `i64` sums, ties keeping the earlier order.
//! 5. Coordinates: a box is 40 units high and as wide as its glyph (24), its label (7 units a
//!    character on its first line, 6 on its smaller second), its chips and 12 units of padding
//!    each side; 24 units between boxes and 48 between ranks. Each rank is first centred on the
//!    widest, then four down and up passes move every item toward the mean centre of its
//!    neighbours in the rank before (or after), keeping the order and the gaps and staying inside
//!    the widest rank's extent (the closest such placement, by isotonic regression in integers
//!    clamped to that extent), so the drawing is no wider than its widest rank.
//! 6. Routes: an edge leaves its producer's bottom (a producer's routes at evenly spaced points,
//!    ordered by where each goes next), passes each dummy's rank vertically and enters its
//!    consumer's top at the slot of its operand, `x + width·(2i + 1) / (2n)`.

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

const SWEEPS: usize = 4;

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
    preds: Vec<usize>,
    succs: Vec<usize>,
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

    let mut rank: BTreeMap<usize, usize> = ranked.iter().map(|&i| (i, 0)).collect();
    for _ in 0..=ranked.len() {
        let mut changed = false;
        for &(_, p, c) in &edges {
            if rank[&c] < rank[&p] + 1 {
                rank.insert(c, rank[&p] + 1);
                changed = true;
            }
        }
        if !changed {
            break;
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
            preds: Vec::new(),
            succs: Vec::new(),
        });
    }
    let mut chains: Vec<Vec<usize>> = Vec::new();
    for &(e, p, c) in &edges {
        let producer_value = graph.nodes[p].value_id.unwrap_or(i64::MAX);
        let mut previous = item_of[&p];
        let mut chain = Vec::new();
        for r in rank[&p] + 1..rank[&c] {
            let dummy = items.len();
            items.push(Item {
                rank: r,
                order: (producer_value, 1, e),
                width: 0,
                height: 0,
                preds: vec![previous],
                succs: Vec::new(),
            });
            items[previous].succs.push(dummy);
            chain.push(dummy);
            previous = dummy;
        }
        let consumer = item_of[&c];
        items[previous].succs.push(consumer);
        items[consumer].preds.push(previous);
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
    order_layers(&items, &mut layers);

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

    let centre = place(&items, &layers);
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

    let mut operands: BTreeMap<usize, i64> = BTreeMap::new();
    for edge in &graph.edges {
        if let Some(&c) = index.get(edge.consumer.as_str()) {
            let n = operands.entry(c).or_insert(0);
            *n = (*n).max(edge.operand + 1);
        }
    }
    let mut points: Vec<Vec<i64>> = edges
        .iter()
        .zip(&chains)
        .map(|(&(e, _, c), chain)| {
            let mut points = Vec::new();
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
    // A producer's routes leave its bottom at evenly spaced points, ordered by where each goes
    // next, so a fan-out does not start as one stroke.
    let mut leaving: BTreeMap<usize, Vec<(i64, usize)>> = BTreeMap::new();
    for (k, &(_, p, _)) in edges.iter().enumerate() {
        leaving.entry(p).or_default().push((points[k][0], k));
    }
    for (p, mut out) in leaving {
        out.sort();
        let pi = item_of[&p];
        let m = out.len() as i64;
        for (j, &(_, k)) in out.iter().enumerate() {
            let start = [
                x(pi) + items[pi].width * (2 * j as i64 + 1) / (2 * m),
                row_y[items[pi].rank] + items[pi].height,
            ];
            points[k].splice(0..0, start);
        }
    }
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

/// Four down and up barycenter sweeps over the ranks.
fn order_layers(items: &[Item], layers: &mut [Vec<usize>]) {
    let mut position = vec![0i64; items.len()];
    let record = |layers: &[Vec<usize>], position: &mut [i64]| {
        for layer in layers {
            for (p, &i) in layer.iter().enumerate() {
                position[i] = p as i64;
            }
        }
    };
    record(layers, &mut position);
    for _ in 0..SWEEPS {
        for r in 1..layers.len() {
            sort_by_barycenter(items, &mut layers[r], &position, |it| &it.preds);
            record(layers, &mut position);
        }
        for r in (0..layers.len().saturating_sub(1)).rev() {
            sort_by_barycenter(items, &mut layers[r], &position, |it| &it.succs);
            record(layers, &mut position);
        }
    }
}

fn sort_by_barycenter(
    items: &[Item],
    layer: &mut [usize],
    position: &[i64],
    neighbours: impl Fn(&Item) -> &Vec<usize>,
) {
    let bary: BTreeMap<usize, Fraction> = layer
        .iter()
        .map(|&i| {
            let next = neighbours(&items[i]);
            let fraction = if next.is_empty() {
                Fraction {
                    sum: position[i],
                    count: 1,
                }
            } else {
                Fraction {
                    sum: next.iter().map(|&n| position[n]).sum(),
                    count: next.len() as i64,
                }
            };
            (i, fraction)
        })
        .collect();
    layer.sort_by(|a, b| bary[a].cmp(bary[b]));
}

/// The centre x of every item: each rank centred on the widest, then four down and up passes
/// toward the neighbours' mean centre.
fn place(items: &[Item], layers: &[Vec<usize>]) -> Vec<i64> {
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
    for _ in 0..SWEEPS {
        for layer in layers.iter().skip(1) {
            align(items, layer, &mut centre, widest, |it| &it.preds);
        }
        for layer in layers.iter().rev().skip(1) {
            align(items, layer, &mut centre, widest, |it| &it.succs);
        }
    }
    centre
}

/// Moves the items of `layer` as close as their order and gaps allow to the mean centre of their
/// neighbours: least squares under `c[k+1] - c[k] >= (w[k] + w[k+1]) / 2 + GAP`, solved by
/// pooling adjacent violators on `c[k] - offset[k]`, every mean rounded down.
fn align(
    items: &[Item],
    layer: &[usize],
    centre: &mut [i64],
    widest: i64,
    neighbours: impl Fn(&Item) -> &Vec<usize>,
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
    let target: Vec<i64> = layer
        .iter()
        .zip(&offset)
        .map(|(&i, o)| {
            let next = neighbours(&items[i]);
            let want = if next.is_empty() {
                centre[i]
            } else {
                next.iter()
                    .map(|&n| centre[n])
                    .sum::<i64>()
                    .div_euclid(next.len() as i64)
            };
            want - o
        })
        .collect();
    let mut blocks: Vec<(i64, i64, usize)> = Vec::new();
    for &t in &target {
        blocks.push((t, 1, 1));
        while blocks.len() > 1 {
            let (s1, n1, k1) = blocks[blocks.len() - 1];
            let (s0, n0, k0) = blocks[blocks.len() - 2];
            if s0.div_euclid(n0) <= s1.div_euclid(n1) {
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
