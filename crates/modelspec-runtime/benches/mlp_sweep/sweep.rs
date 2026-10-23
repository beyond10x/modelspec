//! The pure part of the MLP size sweep (`story:mlp-size-sweep-vs-pytorch-cuda`): the points,
//! the source text of each, the FLOP count, the median, the result line both sides write, and
//! the launch-bound crossover rule. Nothing here opens a device; `tests/mlp_sweep.rs` includes
//! this file and checks it without one.
//!
//! The model is 784 -> H ReLU -> 10, no bias, `f32`, `mse` on synthetic teacher-student data
//! (the runner's generator, seed 42), SGD at [`LR`]. The FLOP count is the matmul count alone,
//! `2·B·H·(784 + 10 + 10 + 10 + 784) = 3196·B·H`: the two forward matmuls, the gradient of `w2`, the
//! adjoint of the hidden activation and the gradient of `w1`; no input adjoint, and no
//! elementwise, loss or update work ([`FLOP_RULE`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The input width.
pub const INPUTS: u64 = 784;
/// The output width.
pub const OUTPUTS: u64 = 10;
/// The hidden widths of the sweep, ascending.
pub const HIDDEN: [u64; 4] = [64, 256, 1024, 4096];
/// The batch sizes of the sweep.
pub const BATCHES: [u64; 3] = [100, 1000, 8192];
/// The SGD learning rate, as the source writes it. Small enough that the widest point does not
/// diverge over a run of several seconds; the step time does not depend on it.
pub const LR: &str = "0.001";
/// The seed of the initial parameters, the teacher and the batch.
pub const SEED: u64 = 42;
/// The steps of the warm-up run before a point's timed runs.
pub const WARMUP_STEPS: u64 = 20;
/// The timed runs per point; the figure is their median.
pub const RUNS: usize = 3;
/// What the FLOP count counts.
pub const FLOP_RULE: &str = "matmul-only";
/// A width is past the launch-bound regime when its step time is more than this times the
/// previous width's at the same batch.
pub const LAUNCH_BOUND_RATIO: f64 = 1.10;
/// The batch whose crossover the pitch slide shows.
pub const PITCH_BATCH: u64 = 1000;

/// One point of the sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    pub batch: u64,
    pub hidden: u64,
}

impl Point {
    /// The matmul FLOP of one training step, `3196·B·H`.
    pub fn flop_per_step(&self) -> u64 {
        2 * self.batch * self.hidden * (INPUTS + 3 * OUTPUTS + INPUTS)
    }

    /// The `.mspec` source of the point, training for `steps` steps.
    pub fn source(&self, steps: u64) -> String {
        let Point { batch, hidden } = *self;
        format!(
            "// The MLP size sweep at batch {batch}, hidden {hidden} (benches/mlp_sweep.rs).
model SweepMlp {{
    param w1: Tensor<f32, [{INPUTS}, {hidden}]> = xavier_uniform();
    param w2: Tensor<f32, [{hidden}, {OUTPUTS}]> = xavier_uniform();

    fn forward(x: Tensor<f32, [{batch}, {INPUTS}]>)
        -> Tensor<f32, [{batch}, {OUTPUTS}]> {{
        return matmul(relu(matmul(x, w1)), w2);
    }}
}}

train Fit {{
    model = SweepMlp;
    input x: Tensor<f32, [{batch}, {INPUTS}]>;
    target y: Tensor<f32, [{batch}, {OUTPUTS}]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = {LR});
    steps = {steps};
    seed = {SEED};
    device = vulkan(0);
}}
"
        )
    }

    /// The name the point's source carries in diagnostics.
    pub fn file_name(&self) -> String {
        format!("mlp_sweep_b{}_h{}.mspec", self.batch, self.hidden)
    }
}

/// The 12 points: each batch in [`BATCHES`] order, each with the widths of [`HIDDEN`] ascending.
pub fn points() -> Vec<Point> {
    BATCHES
        .iter()
        .flat_map(|&batch| HIDDEN.iter().map(move |&hidden| Point { batch, hidden }))
        .collect()
}

/// The median: the middle value, or the mean of the two middle values of an even count.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    Some(if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    })
}

/// The steps a run needs to last `target` seconds, from a run of `steps` steps that took
/// `seconds`; never fewer than [`WARMUP_STEPS`].
pub fn steps_for(seconds: f64, steps: u64, target: f64) -> u64 {
    let per_step = seconds / steps as f64;
    ((target / per_step).ceil() as u64).max(WARMUP_STEPS)
}

/// One timed run of a point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimedRun {
    /// The timed span over the steps.
    pub us_per_step: f64,
    /// The timed span.
    pub seconds: f64,
    /// The timed span's start and end, Unix milliseconds, to find the GPU clock samples taken
    /// during it.
    pub span_unix_ms: [u64; 2],
}

/// What a point came to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Outcome {
    /// [`RUNS`] timed runs of `steps` steps each.
    Measured {
        steps: u64,
        runs: Vec<TimedRun>,
        median_us_per_step: f64,
        /// `flop_per_step` over the median step time.
        gflops: f64,
    },
    /// The point did not compile or run; `refusal` is what was printed.
    Refused { refusal: String },
}

/// One result line: a point of one side.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointResult {
    /// `modelspec-vulkan`, `pytorch-eager` or `pytorch-compile`.
    pub side: String,
    pub batch: u64,
    pub hidden: u64,
    pub flop_per_step: u64,
    pub flop_rule: String,
    /// The device and software the side ran on, as it names them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(flatten)]
    pub outcome: Outcome,
}

impl PointResult {
    /// A measured point; the median and GFLOP/s are computed from `runs`.
    pub fn measured(
        side: &str,
        point: Point,
        steps: u64,
        runs: Vec<TimedRun>,
        device: Option<String>,
    ) -> Self {
        let times: Vec<f64> = runs.iter().map(|run| run.us_per_step).collect();
        let median_us_per_step = median(&times).unwrap_or(f64::NAN);
        let flop_per_step = point.flop_per_step();
        PointResult {
            side: side.to_owned(),
            batch: point.batch,
            hidden: point.hidden,
            flop_per_step,
            flop_rule: FLOP_RULE.to_owned(),
            device,
            outcome: Outcome::Measured {
                steps,
                runs,
                median_us_per_step,
                gflops: flop_per_step as f64 / (median_us_per_step * 1e3),
            },
        }
    }

    /// A point that did not compile or run.
    pub fn refused(side: &str, point: Point, refusal: String, device: Option<String>) -> Self {
        PointResult {
            side: side.to_owned(),
            batch: point.batch,
            hidden: point.hidden,
            flop_per_step: point.flop_per_step(),
            flop_rule: FLOP_RULE.to_owned(),
            device,
            outcome: Outcome::Refused { refusal },
        }
    }

    /// The median step time of a measured point.
    pub fn median_us_per_step(&self) -> Option<f64> {
        match self.outcome {
            Outcome::Measured {
                median_us_per_step, ..
            } => Some(median_us_per_step),
            Outcome::Refused { .. } => None,
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

/// Where a side stops being launch-bound at one batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crossover {
    /// The first width whose step time is more than [`LAUNCH_BOUND_RATIO`] times the previous
    /// width's.
    At(u64),
    /// Every width was compared and none is.
    NotReached,
    /// `hidden` has no figure and no earlier width crossed, so the first crossing is not known.
    Undetermined { hidden: u64 },
}

/// The crossover over `widths`, ascending `(hidden, median µs per step)`, `None` for a width
/// without a figure.
pub fn crossover(widths: &[(u64, Option<f64>)]) -> Crossover {
    if let Some(&(hidden, None)) = widths.first() {
        return Crossover::Undetermined { hidden };
    }
    for pair in widths.windows(2) {
        let [(_, previous), (hidden, current)] = [pair[0], pair[1]];
        let (Some(previous), Some(current)) = (previous, current) else {
            let missing = if previous.is_none() {
                pair[0].0
            } else {
                hidden
            };
            return Crossover::Undetermined { hidden: missing };
        };
        if current > previous * LAUNCH_BOUND_RATIO {
            return Crossover::At(hidden);
        }
    }
    Crossover::NotReached
}

/// The crossover of one side at one batch.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossoverLine {
    pub side: String,
    pub batch: u64,
    pub crossover: Crossover,
    /// Whether this is the batch the pitch slide shows ([`PITCH_BATCH`]).
    pub pitch_slide: bool,
}

#[derive(Serialize)]
struct CrossoverJson<'a> {
    side: &'a str,
    batch: u64,
    verdict: &'a str,
    hidden: Option<u64>,
    pitch_slide: bool,
}

impl CrossoverLine {
    /// The line as JSON: `verdict` is `reached`, `not reached` or `undetermined`; `hidden` is
    /// the crossover width, or the width without a figure when undetermined.
    pub fn to_line(&self) -> String {
        let (verdict, hidden) = match self.crossover {
            Crossover::At(hidden) => ("reached", Some(hidden)),
            Crossover::NotReached => ("not reached", None),
            Crossover::Undetermined { hidden } => ("undetermined", Some(hidden)),
        };
        serde_json::to_string(&CrossoverJson {
            side: &self.side,
            batch: self.batch,
            verdict,
            hidden,
            pitch_slide: self.pitch_slide,
        })
        .expect("a crossover line serializes")
    }
}

/// The crossover of every side at every batch of [`BATCHES`], over the widths of [`HIDDEN`], from
/// the result lines of any number of sides; sides in the order they first appear. A width a side
/// has no measured line for counts as without a figure; a later line for the same point replaces
/// an earlier one.
pub fn crossover_lines(results: &[PointResult]) -> Vec<CrossoverLine> {
    let mut sides: Vec<&str> = Vec::new();
    let mut figures: BTreeMap<(&str, u64, u64), Option<f64>> = BTreeMap::new();
    for result in results {
        if !sides.contains(&result.side.as_str()) {
            sides.push(&result.side);
        }
        figures.insert(
            (&result.side, result.batch, result.hidden),
            result.median_us_per_step(),
        );
    }
    let mut lines = Vec::new();
    for side in sides {
        for batch in BATCHES {
            let widths: Vec<(u64, Option<f64>)> = HIDDEN
                .iter()
                .map(|&hidden| {
                    let figure = figures.get(&(side, batch, hidden)).copied().flatten();
                    (hidden, figure)
                })
                .collect();
            lines.push(CrossoverLine {
                side: side.to_owned(),
                batch,
                crossover: crossover(&widths),
                pitch_slide: batch == PITCH_BATCH,
            });
        }
    }
    lines
}
