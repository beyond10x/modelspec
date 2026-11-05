//! The register-blocked key-side attention gradient kernels
//! (story:vulkan-attention-blocked-key-tiles): `AttentionGradK` and `AttentionGradV` sharing
//! their call's softmax statistics (`Kernel::softmax_stats`), under a key mask, over sequences
//! of `S` keys and `L` queries, both multiples of [`BLOCK_TILE`], and heads of `d` columns, `d`
//! a multiple of [`QUARTER`] of at most `modelspec_core::schedule::ATTENTION_BLOCK_MAX_HEAD`
//! (`modelspec_core::schedule::attention_block_tile`).
//!
//! Workgroup `(b · heads + a) · tiles + t` of `ATTENTION_WORKGROUP` (128) invocations computes
//! head `a` of the 32 key rows of tile `t` of sequence `b`, every sum in the order the
//! per-row kernels of `attention.rs` sum it, so the result's bits are theirs. It loops over the
//! `L` queries of the sequence in chunks of 32. For a chunk, in a first phase, it stages 16
//! columns at a time of the chunk's `q` rows and the tile's `k` rows (and for `AttentionGradK`
//! of the chunk's `g` rows and the tile's `v` rows) in workgroup memory between barriers, each
//! staged row padded by one value against bank conflicts, and invocation `i` computes the
//! scores of queries `4 (i / 16) ..+ 4` with keys `i % 16` and `i % 16 + 16`, each a dot
//! product over the head's columns in ascending order from `-0.0` with every product fused, as
//! `Shell::dot`; then the probability `exp(score · scale - max) / sum` of each from the
//! statistics of the query, and for `AttentionGradK` `dS = p · (dP - rd)` with `dP` the dot
//! product of the query's `g` row and the key's `v` row by the same instructions, and stores
//! the 8 weights into the chunk's `[32, 32]` weights, a row per query padded by one value. In a
//! second phase, it stages the whole chunk of `g` rows (`AttentionGradV`) or `q` rows
//! (`AttentionGradK`) and invocation `i` adds, for each query of the chunk in ascending order,
//! the weight of keys `4 (i / 16) ..+ 4` times the row's columns `i % 16 + 16 u` into its 4 ·
//! `d / 16` output sums, held in function variables across the chunks, from `-0.0` with every
//! product fused, as `Shell::row_sums` over all `L` queries. After the last chunk each sum is
//! written to its element of the result, times the scale for `AttentionGradK`, or `+0` for a
//! masked key, whose weights are computed and discarded; no value of a masked key's row reaches
//! another row's sum, because a key's row of the result sums only that key's weights.
//!
//! Workgroup memory, in `f32` values: a staging region of `max(4 · 32 · 17, 32 · (d + 1))`
//! values for `AttentionGradK` and `max(2 · 32 · 17, 32 · (d + 1))` for `AttentionGradV`, then
//! the `32 · 33` weights ([`block_shared_values`]), at most 3,232 of the 4,096 every device has.

use super::{EmitError, Shell, Word};
use modelspec_core::schedule::{ATTENTION_BLOCK_TILE, ATTENTION_WORKGROUP};

/// The key rows of one workgroup, `modelspec_core::schedule::ATTENTION_BLOCK_TILE`.
pub(super) const BLOCK_TILE: u32 = ATTENTION_BLOCK_TILE as u32;
/// The queries of one chunk.
const CHUNK: u32 = 32;
/// The columns of a head staged per step of the first phase.
pub(super) const QUARTER: u32 = 16;
/// The stride of a staged row of the first phase, [`QUARTER`] columns and one pad.
const STAGE_STRIDE: u32 = QUARTER + 1;
/// The values of one staged operand of the first phase.
const STAGE_SLOT: u32 = CHUNK * STAGE_STRIDE;
/// The stride of a row of the chunk's weights, [`BLOCK_TILE`] keys and one pad.
const WEIGHT_STRIDE: u32 = BLOCK_TILE + 1;
/// The invocations of a workgroup that share one row group: 16 key or column lanes.
const LANES: u32 = 16;
/// The queries an invocation scores per chunk, and the keys an invocation sums per chunk.
const ROWS_PER_INVOCATION: u32 = 4;
/// The keys an invocation scores per chunk, `BLOCK_TILE / LANES`.
const KEYS_PER_INVOCATION: u32 = BLOCK_TILE / LANES;

/// The shapes of a blocked kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Blocked {
    /// `AttentionGradK`; else `AttentionGradV`.
    pub(super) grad_k: bool,
    pub(super) s: u32,
    pub(super) l: u32,
    pub(super) h: u32,
    pub(super) d: u32,
    pub(super) heads: u32,
    pub(super) scale: f32,
    /// The workgroups that compute: sequences times heads times `s / 32`.
    pub(super) groups: u32,
}

/// The `f32` values of workgroup memory a blocked kernel of `grad_k` over heads of `d` columns
/// stages: the staging region, then the chunk's weights.
pub(super) fn block_shared_values(grad_k: bool, d: u32) -> u32 {
    block_weights_base(grad_k, d) + CHUNK * WEIGHT_STRIDE
}

/// Where the chunk's weights start: after the staging region.
fn block_weights_base(grad_k: bool, d: u32) -> u32 {
    let slots = if grad_k { 4 } else { 2 };
    (slots * STAGE_SLOT).max(CHUNK * (d + 1))
}

/// The `f32` function variables a blocked kernel keeps: one output sum per owned element.
pub(super) fn block_f32_vars(d: u32) -> usize {
    (ROWS_PER_INVOCATION * (d / LANES)) as usize
}

impl Shell {
    /// The module body of a blocked kernel: the key tile of the invocation's workgroup.
    pub(super) fn attention_blocked(&mut self, at: Blocked) -> Result<(), EmitError> {
        let local = self
            .local
            .ok_or_else(|| EmitError::Build("a blocked attention module stages rows".to_owned()))?;
        let index = self.index;
        let size = self.uint(ATTENTION_WORKGROUP);
        let group = self.div(index, size)?;
        let groups = self.uint(at.groups);
        let inside = self.less(group, groups)?;
        self.when(inside, |m| m.blocked_key_tile(&at, group, local))
    }

    fn blocked_key_tile(
        &mut self,
        at: &Blocked,
        group: Word,
        local: Word,
    ) -> Result<(), EmitError> {
        let (tiles, heads) = (self.uint(at.s / BLOCK_TILE), self.uint(at.heads));
        let (s, l, h, d) = (
            self.uint(at.s),
            self.uint(at.l),
            self.uint(at.h),
            self.uint(at.d),
        );
        let tile_c = self.uint(BLOCK_TILE);
        let tile = self.rem(group, tiles)?;
        let pair = self.div(group, tiles)?;
        let head = self.rem(pair, heads)?;
        let sequence = self.div(pair, heads)?;
        let key_first = self.mul(sequence, s)?;
        let tile_start = self.mul(tile, tile_c)?;
        let key_first = self.add(key_first, tile_start)?;
        let query_first = self.mul(sequence, l)?;
        let column = self.mul(head, d)?;
        let lanes = self.uint(LANES);
        let row_group = self.div(local, lanes)?;
        let lane = self.rem(local, lanes)?;
        let four = self.uint(ROWS_PER_INVOCATION);
        let own_row = self.mul(row_group, four)?;
        let columns = at.d / LANES;
        let sums: Vec<Word> = self.matmul_vars[..block_f32_vars(at.d)].to_vec();
        let start_sum = self.float(-0.0);
        for &sum in &sums {
            self.b.store(sum, start_sum, None, [])?;
        }
        let weights_base = block_weights_base(at.grad_k, at.d);
        let chunk_var = self.extra_u32_vars[0];
        let (zero, chunk_c) = (self.uint(0), self.uint(CHUNK));
        self.counted(chunk_var, zero, l, chunk_c, |m, chunk| {
            let query_chunk = m.add(query_first, chunk)?;
            m.blocked_weights(
                at,
                query_chunk,
                key_first,
                column,
                head,
                local,
                own_row,
                lane,
                weights_base,
            )?;
            m.blocked_sums(
                at,
                query_chunk,
                column,
                local,
                own_row,
                lane,
                weights_base,
                &sums,
            )
        })?;
        // Each owned element: the sum, times the scale for `AttentionGradK`, or `+0` for a
        // masked key.
        let f_zero = self.float(0.0);
        let scale = self.float(at.scale);
        for i in 0..ROWS_PER_INVOCATION {
            let i_c = self.uint(i);
            let key_in_tile = self.add(own_row, i_c)?;
            let key_row = self.add(key_first, key_in_tile)?;
            let entry = self.read(3, key_row)?;
            let keep = self
                .b
                .f_unord_not_equal(self.bool_ty, None, entry, f_zero)?;
            let row_at = self.mul(key_row, h)?;
            let row_at = self.add(row_at, column)?;
            for u in 0..columns {
                let sum = self.b.load(
                    self.f32_ty,
                    None,
                    sums[(i * columns + u) as usize],
                    None,
                    [],
                )?;
                let value = if at.grad_k {
                    self.f_mul(sum, scale)?
                } else {
                    sum
                };
                let value = self.b.select(self.f32_ty, None, keep, value, f_zero)?;
                let offset = self.uint(u * LANES);
                let c = self.add(lane, offset)?;
                let at_c = self.add(row_at, c)?;
                self.write(at_c, value)?;
            }
        }
        Ok(())
    }

    /// Stages rows `row_base ..+ 32` of binding `binding`, columns `column + first ..+ 16`, into
    /// staging slot `slot`, element `e = local + 128 u` as row `e / 16`, column `e % 16`.
    #[allow(clippy::too_many_arguments)]
    fn stage_quarter(
        &mut self,
        at: &Blocked,
        binding: usize,
        row_base: Word,
        column: Word,
        first: u32,
        slot: u32,
        local: Word,
    ) -> Result<(), EmitError> {
        let h = self.uint(at.h);
        let (quarter, stride) = (self.uint(QUARTER), self.uint(STAGE_STRIDE));
        let first_c = self.uint(first);
        let slot_c = self.uint(slot * STAGE_SLOT);
        for u in 0..(CHUNK * QUARTER) / ATTENTION_WORKGROUP {
            let offset = self.uint(u * ATTENTION_WORKGROUP);
            let e = self.add(local, offset)?;
            let row = self.div(e, quarter)?;
            let col = self.rem(e, quarter)?;
            let global_row = self.add(row_base, row)?;
            let at_row = self.mul(global_row, h)?;
            let at_col = self.add(column, first_c)?;
            let at_col = self.add(at_col, col)?;
            let place = self.add(at_row, at_col)?;
            let value = self.read(binding, place)?;
            let to_row = self.mul(row, stride)?;
            let to = self.add(to_row, col)?;
            let to = self.add(slot_c, to)?;
            self.store_shared(to, value)?;
        }
        Ok(())
    }

    /// Phase 1 of a chunk: the weights of the chunk's queries with the tile's keys, into the
    /// chunk's weights at `weights_base`, ending in a barrier.
    #[allow(clippy::too_many_arguments)]
    fn blocked_weights(
        &mut self,
        at: &Blocked,
        query_chunk: Word,
        key_first: Word,
        column: Word,
        head: Word,
        local: Word,
        own_row: Word,
        lane: Word,
        weights_base: u32,
    ) -> Result<(), EmitError> {
        let start = self.float(-0.0);
        let mut scores = vec![start; (ROWS_PER_INVOCATION * KEYS_PER_INVOCATION) as usize];
        let mut dots = if at.grad_k {
            Some(scores.clone())
        } else {
            None
        };
        let stride = self.uint(STAGE_STRIDE);
        // Where this invocation's query rows and key rows start in a staging slot.
        let mut query_rows = Vec::new();
        for i in 0..ROWS_PER_INVOCATION {
            let i_c = self.uint(i);
            let row = self.add(own_row, i_c)?;
            query_rows.push(self.mul(row, stride)?);
        }
        let mut key_rows = Vec::new();
        for j in 0..KEYS_PER_INVOCATION {
            let offset = self.uint(j * LANES);
            let row = self.add(lane, offset)?;
            key_rows.push(self.mul(row, stride)?);
        }
        for quarter in 0..at.d / QUARTER {
            let first = quarter * QUARTER;
            self.barrier()?;
            self.stage_quarter(at, 0, query_chunk, column, first, 0, local)?;
            self.stage_quarter(at, 1, key_first, column, first, 1, local)?;
            if at.grad_k {
                self.stage_quarter(at, 4, query_chunk, column, first, 2, local)?;
                self.stage_quarter(at, 2, key_first, column, first, 3, local)?;
            }
            self.barrier()?;
            for col in 0..QUARTER {
                let load =
                    |m: &mut Self, slot: u32, rows: &[Word]| -> Result<Vec<Word>, EmitError> {
                        let base = m.uint(slot * STAGE_SLOT + col);
                        rows.iter()
                            .map(|&row| {
                                let place = m.add(base, row)?;
                                m.load_shared(place)
                            })
                            .collect()
                    };
                let qs = load(self, 0, &query_rows)?;
                let ks = load(self, 1, &key_rows)?;
                debug_assert_eq!(qs.len(), ROWS_PER_INVOCATION as usize);
                debug_assert_eq!(ks.len(), KEYS_PER_INVOCATION as usize);
                for (i, &q) in qs.iter().enumerate() {
                    for (j, &k) in ks.iter().enumerate() {
                        let e = i * KEYS_PER_INVOCATION as usize + j;
                        scores[e] = self.f_fma(q, k, scores[e])?;
                    }
                }
                if let Some(dots) = dots.as_mut() {
                    let gs = load(self, 2, &query_rows)?;
                    let vs = load(self, 3, &key_rows)?;
                    for (i, &g) in gs.iter().enumerate() {
                        for (j, &v) in vs.iter().enumerate() {
                            let e = i * KEYS_PER_INVOCATION as usize + j;
                            dots[e] = self.f_fma(g, v, dots[e])?;
                        }
                    }
                }
            }
        }
        // The weights from the statistics of each query, into the chunk's weights.
        let (heads, three) = (self.uint(at.heads), self.uint(3));
        let scale = self.float(at.scale);
        let weight_stride = self.uint(WEIGHT_STRIDE);
        let weights_base_c = self.uint(weights_base);
        for i in 0..ROWS_PER_INVOCATION {
            let i_c = self.uint(i);
            let row_in_chunk = self.add(own_row, i_c)?;
            let row = self.add(query_chunk, row_in_chunk)?;
            let query = self.mul(row, heads)?;
            let query = self.add(query, head)?;
            let stats_at = self.mul(query, three)?;
            let max = self.read(5, stats_at)?;
            let one = self.uint(1);
            let sum_at = self.add(stats_at, one)?;
            let total = self.read(5, sum_at)?;
            let rd = if at.grad_k {
                let two = self.uint(2);
                let rd_at = self.add(stats_at, two)?;
                Some(self.read(5, rd_at)?)
            } else {
                None
            };
            let weight_row = self.mul(row_in_chunk, weight_stride)?;
            let weight_row = self.add(weights_base_c, weight_row)?;
            for j in 0..KEYS_PER_INVOCATION {
                let e = (i * KEYS_PER_INVOCATION + j) as usize;
                let x = self.f_mul(scores[e], scale)?;
                let diff = self.f_sub(x, max)?;
                let exp = self.exp(diff)?;
                let p = self.f_div(exp, total)?;
                let weight = match (rd, dots.as_ref()) {
                    (Some(rd), Some(dots)) => {
                        let diff = self.f_sub(dots[e], rd)?;
                        self.f_mul(p, diff)?
                    }
                    _ => p,
                };
                let offset = self.uint(j * LANES);
                let key = self.add(lane, offset)?;
                let place = self.add(weight_row, key)?;
                self.store_shared(place, weight)?;
            }
        }
        self.barrier()
    }

    /// Phase 2 of a chunk: the chunk's rows of `g` (`AttentionGradV`) or `q` (`AttentionGradK`)
    /// staged whole, then each owned sum over the chunk's queries in ascending order.
    #[allow(clippy::too_many_arguments)]
    fn blocked_sums(
        &mut self,
        at: &Blocked,
        query_chunk: Word,
        column: Word,
        local: Word,
        own_row: Word,
        lane: Word,
        weights_base: u32,
        sums: &[Word],
    ) -> Result<(), EmitError> {
        let binding = if at.grad_k { 0 } else { 4 };
        let (h, d, chunk_stride) = (self.uint(at.h), self.uint(at.d), self.uint(at.d + 1));
        let total = CHUNK * at.d;
        for u in 0..total.div_ceil(ATTENTION_WORKGROUP) {
            let offset = self.uint(u * ATTENTION_WORKGROUP);
            let e = self.add(local, offset)?;
            let row = self.div(e, d)?;
            let col = self.rem(e, d)?;
            let global_row = self.add(query_chunk, row)?;
            let at_row = self.mul(global_row, h)?;
            let at_col = self.add(column, col)?;
            let place = self.add(at_row, at_col)?;
            let value = self.read(binding, place)?;
            let to_row = self.mul(row, chunk_stride)?;
            let to = self.add(to_row, col)?;
            self.store_shared(to, value)?;
        }
        self.barrier()?;
        let columns = at.d / LANES;
        let mut acc: Vec<Word> = sums
            .iter()
            .map(|&sum| Ok(self.b.load(self.f32_ty, None, sum, None, [])?))
            .collect::<Result<_, EmitError>>()?;
        let weights_base_c = self.uint(weights_base);
        let own_weight = self.add(weights_base_c, own_row)?;
        let mut col_places = Vec::new();
        for u in 0..columns {
            let offset = self.uint(u * LANES);
            col_places.push(self.add(lane, offset)?);
        }
        for q in 0..CHUNK {
            let weight_row = self.uint(q * WEIGHT_STRIDE);
            let weight_at = self.add(own_weight, weight_row)?;
            let mut ws = Vec::new();
            for i in 0..ROWS_PER_INVOCATION {
                let i_c = self.uint(i);
                let place = self.add(weight_at, i_c)?;
                ws.push(self.load_shared(place)?);
            }
            let row_at = self.uint(q * (at.d + 1));
            let mut xs = Vec::new();
            for &c in &col_places {
                let place = self.add(row_at, c)?;
                xs.push(self.load_shared(place)?);
            }
            debug_assert_eq!(xs.len(), columns as usize);
            for (i, &w) in ws.iter().enumerate() {
                for (u, &x) in xs.iter().enumerate() {
                    let e = i * columns as usize + u;
                    acc[e] = self.f_fma(w, x, acc[e])?;
                }
            }
        }
        for (&sum, &value) in sums.iter().zip(&acc) {
            self.b.store(sum, value, None, [])?;
        }
        // The next chunk's staging overwrites the chunk's rows and weights.
        self.barrier()
    }
}
