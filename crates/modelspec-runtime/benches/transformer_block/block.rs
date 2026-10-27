//! The pure part of the transformer-block benchmark (story:transformer-block-benchmark): the
//! points, the source of each, the matmul-only FLOP count of the block, and the result line both
//! sides write. Nothing here opens a device; `tests/transformer_block.rs` includes this file and
//! checks it without one. The run discipline (warm-up, calibration, timed runs, median, build)
//! is the MLP sweep's (`benches/mlp_sweep/sweep.rs`), whose types this file reuses.
//!
//! The model is `examples/transformer_block.mspec`, read into the binary ([`EXAMPLE`]) and run
//! with its batch substituted ([`Point::source`]), never a second copy: the two post-layer-norm
//! encoder layers of BERT-tiny (hidden `H` = 128, 2 heads, feed-forward `F` = 512, sequences of
//! `S` = 128) on `N = B·S` rows of the runner's synthetic batch, an MSE loss and SGD.
//!
//! # The FLOP rule, `matmul-only`
//!
//! [`Point::flop_per_step`] counts the matrix products of one training step, forward and
//! backward, `2·m·k·n` for an `[m, k]` by `[k, n]` product, and nothing else: no bias, residual,
//! GELU, layer-norm, softmax, mask, scale, loss or update work. It is the MLP sweep's rule
//! (`3196·B·H`, its five matmuls) applied to every product of the block:
//!
//! - Dense. Each layer has six dense matmuls on `N` rows: the projections `Q`, `K`, `V` and the
//!   output projection, `[N, H]` by `[H, H]`, `2·N·H²` each, and the feed-forward pair, `[N, H]`
//!   by `[H, F]` and `[N, F]` by `[F, H]`, `2·N·H·F` each; `8·N·H² + 4·N·H·F` forward. Backward,
//!   as in the sweep, each takes the same count again for its weight gradient and again for the
//!   adjoint of its input, except the three projections of the first layer, whose input is the
//!   batch `x`, which nothing differentiates. So `3·L·(8·N·H² + 4·N·H·F) − 3·2·N·H²`, with
//!   `L = 2` layers `42·N·H² + 24·N·H·F` ([`Point::dense_flop`]).
//! - Attention. Per layer, head and sequence the two batched products `Q·Kᵀ` (`[S, d]` by
//!   `[d, S]`) and `P·V` (`[S, S]` by `[S, d]`), `2·S²·d` each, with `d = H / heads`: over the
//!   `heads` heads and `B` sequences `2·N·S·H` each. Backward, the adjoints of both operands of
//!   each, `dQ` and `dK` from the scores' gradient and `dP` and `dV` from the output's, four more
//!   of the same size: `6·2·N·S·H` per layer, `24·N·S·H` for two ([`Point::attention_flop`]).
//!   ModelSpec's attention kernels recompute the probabilities in the backward pass; the rule
//!   counts the products of the step's mathematics, the same for every side, not a side's
//!   recomputation.
//!
//! Together `N·(42·H² + 24·H·F + 24·S·H)`, at these sizes `2,654,208` FLOP per row and
//! `339,738,624·B` per step: 2,717,908,992 at `B = 8`, 10,871,635,968 at 32 and
//! 43,486,543,872 at 128. The opponent `opponents/transformer_block/transformer_block.py` counts
//! the same.

use serde::{Deserialize, Serialize};

use super::sweep::{Build, TimedRun, median};

/// `examples/transformer_block.mspec`, as the binary was built with it.
pub const EXAMPLE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/transformer_block.mspec"
));
/// The example's path in the repository, for messages.
pub const EXAMPLE_PATH: &str = "examples/transformer_block.mspec";
/// The batch the example declares, in sequences.
pub const FILE_BATCH: u64 = 8;
/// The batches of the benchmark, in sequences.
pub const BATCHES: [u64; 3] = [8, 32, 128];
/// The sequence length `S`.
pub const SEQ: u64 = 128;
/// The hidden width `H`.
pub const HIDDEN: u64 = 128;
/// The feed-forward width `F`.
pub const FFN: u64 = 512;
/// The encoder layers `L`.
pub const LAYERS: u64 = 2;
/// The declarations of the `[B·S, H]` input, target and model output in the example: the
/// forward's `x` and return type and the train block's `x` and `y`.
pub const ROW_DECLARATIONS: usize = 4;
/// The declarations of the `[B, S]` key mask in the example: the forward's and the train block's.
pub const MASK_DECLARATIONS: usize = 2;
/// What the FLOP count counts.
pub const FLOP_RULE: &str = "matmul-only";

/// One point of the benchmark: a batch of `batch` sequences.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    pub batch: u64,
}

impl Point {
    /// The rows `N = B·S` the layers run on, one token each.
    pub fn rows(&self) -> u64 {
        self.batch * SEQ
    }

    /// The tokens of one step, `B·S`.
    pub fn tokens_per_step(&self) -> u64 {
        self.rows()
    }

    /// The dense part of the FLOP rule, `3·L·(8·N·H² + 4·N·H·F) − 6·N·H²` (module docs).
    pub fn dense_flop(&self) -> u64 {
        let n = self.rows();
        let forward = 8 * n * HIDDEN * HIDDEN + 4 * n * HIDDEN * FFN;
        3 * LAYERS * forward - 3 * 2 * n * HIDDEN * HIDDEN
    }

    /// The attention part of the FLOP rule, `L·6·2·N·S·H` (module docs).
    pub fn attention_flop(&self) -> u64 {
        LAYERS * 6 * 2 * self.rows() * SEQ * HIDDEN
    }

    /// The matmul-only FLOP of one training step (module docs).
    pub fn flop_per_step(&self) -> u64 {
        self.dense_flop() + self.attention_flop()
    }

    /// The example's source at this batch: every `[8·S, H]` declaration becomes `[B·S, H]` and
    /// every `[8, S]` one `[B, S]`, nothing else changes. Refused, naming the file, when the
    /// example does not declare them [`ROW_DECLARATIONS`] and [`MASK_DECLARATIONS`] times.
    pub fn source(&self) -> Result<String, String> {
        let rows = format!("[{}, {HIDDEN}]", FILE_BATCH * SEQ);
        let mask = format!("[{FILE_BATCH}, {SEQ}]");
        let (found_rows, found_mask) = (
            EXAMPLE.matches(&rows).count(),
            EXAMPLE.matches(&mask).count(),
        );
        if (found_rows, found_mask) != (ROW_DECLARATIONS, MASK_DECLARATIONS) {
            return Err(format!(
                "{EXAMPLE_PATH} declares `{rows}` {found_rows} times and `{mask}` {found_mask} \
                 times; the bench substitutes the batch in {ROW_DECLARATIONS} and \
                 {MASK_DECLARATIONS}"
            ));
        }
        Ok(EXAMPLE
            .replace(&rows, &format!("[{}, {HIDDEN}]", self.rows()))
            .replace(&mask, &format!("[{}, {SEQ}]", self.batch)))
    }

    /// The name the point's source carries in diagnostics.
    pub fn file_name(&self) -> String {
        format!("transformer_block_b{}.mspec", self.batch)
    }
}

/// The three points, in [`BATCHES`] order.
pub fn points() -> Vec<Point> {
    BATCHES.iter().map(|&batch| Point { batch }).collect()
}

/// What a point came to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Outcome {
    /// Timed runs of `steps` steps each.
    Measured {
        steps: u64,
        runs: Vec<TimedRun>,
        median_us_per_step: f64,
        /// `tokens_per_step` over the median step time, `B·S·10⁶ / µs`.
        tokens_per_s: f64,
        /// `flop_per_step` over the median step time.
        gflops: f64,
    },
    /// The point did not compile or run; `refusal` is what was printed.
    Refused { refusal: String },
}

/// One result line: a point of one side.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockResult {
    /// `modelspec-vulkan`, `pytorch-eager` or `pytorch-compile`.
    pub side: String,
    pub batch: u64,
    pub seq: u64,
    pub tokens_per_step: u64,
    pub flop_per_step: u64,
    pub flop_rule: String,
    /// The device and software the side ran on, as it names them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// The source the side ran from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<Build>,
    #[serde(flatten)]
    pub outcome: Outcome,
}

impl BlockResult {
    fn of(side: &str, point: Point, device: Option<String>, outcome: Outcome) -> Self {
        BlockResult {
            side: side.to_owned(),
            batch: point.batch,
            seq: SEQ,
            tokens_per_step: point.tokens_per_step(),
            flop_per_step: point.flop_per_step(),
            flop_rule: FLOP_RULE.to_owned(),
            device,
            build: None,
            outcome,
        }
    }

    /// A measured point; the median, tokens per second and GFLOP/s are computed from `runs`.
    pub fn measured(
        side: &str,
        point: Point,
        steps: u64,
        runs: Vec<TimedRun>,
        device: Option<String>,
    ) -> Self {
        let times: Vec<f64> = runs.iter().map(|run| run.us_per_step).collect();
        let median_us_per_step = median(&times).unwrap_or(f64::NAN);
        let outcome = Outcome::Measured {
            steps,
            runs,
            median_us_per_step,
            tokens_per_s: point.tokens_per_step() as f64 * 1e6 / median_us_per_step,
            gflops: point.flop_per_step() as f64 / (median_us_per_step * 1e3),
        };
        Self::of(side, point, device, outcome)
    }

    /// A point that did not compile or run.
    pub fn refused(side: &str, point: Point, refusal: String, device: Option<String>) -> Self {
        Self::of(side, point, device, Outcome::Refused { refusal })
    }

    /// The result, naming the source it ran from.
    pub fn with_build(self, build: Build) -> Self {
        BlockResult {
            build: Some(build),
            ..self
        }
    }

    /// The result as one JSON line.
    pub fn to_line(&self) -> String {
        serde_json::to_string(self).expect("a result serializes")
    }

    /// A result from one JSON line.
    pub fn from_line(line: &str) -> Result<Self, String> {
        serde_json::from_str(line).map_err(|error| error.to_string())
    }
}
