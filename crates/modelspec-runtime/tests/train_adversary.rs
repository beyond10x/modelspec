//! Adversary cases for the Philox draws of `train::init` (DESIGN.md § 2.3): the extremes of the
//! uniform mapping, which the draws of seed 42 never reach, and the shape refusals.
//!
//! The seeds were found by an exhaustive search over `block(seed, stream, id, 0)` outside the
//! repository; each case asserts the word it relies on before it uses it.

use modelspec_core::ir::tensor::{DType, Dim, TensorType};
use modelspec_runtime::train::init::{
    InitError, Stream, block, canonical_id, standard_normal, xavier_uniform,
};

/// `block(SEED_ZERO_WORD, Stream::Data, canonical_id("x"), 0)` has word 0 equal to 0.
const SEED_ZERO_WORD: u64 = 1_067_664_386;
/// `block(SEED_TOP_WORD, Stream::Init, canonical_id("LinearReg.w"), 0)` has word 0
/// `0xffffffed`.
const SEED_TOP_WORD: u64 = 72_369_776;

/// DESIGN.md § 2.3: `u = (w + 0.5) / 2^32` in `(0, 1)`, so word 0 = 0 gives `u0 = 2^-33` and a
/// finite `sqrt(-2 ln u0) * cos(2 pi u1)`. A mapping `w / 2^32` takes `ln 0`; the draws of the
/// suite differ from it by less than an `f32` rounding.
#[test]
fn a_zero_word_draws_a_finite_normal_from_u_2_to_the_minus_33() {
    let [w0, w1, _, _] = block(SEED_ZERO_WORD, Stream::Data, canonical_id("x"), 0);
    assert_eq!(w0, 0, "the seed no longer addresses a zero word");
    let t = standard_normal(SEED_ZERO_WORD, Stream::Data, "x", &TensorType::f32(&[1])).unwrap();
    let u0 = 0.5f64 / 4_294_967_296.0;
    let u1 = (f64::from(w1) + 0.5) / 4_294_967_296.0;
    let expected = ((-2.0 * u0.ln()).sqrt() * (std::f64::consts::TAU * u1).cos()) as f32;
    assert!(t.data()[0].is_finite(), "{:?}", t.data());
    assert_eq!(t.data()[0], expected);
}

/// `xavier_uniform` is documented as drawing each element from `U(-a, a)`, and the unit's own
/// cases assert `|v| < a`. At word 0 `0xffffffed` the `f64` value `a * (2u - 1)` is below `a`,
/// and its rounding to `f32` is `f32(a)`, which is above `a = sqrt(6 / 5)`.
#[test]
fn xavier_uniform_stays_inside_minus_a_to_a_after_rounding() {
    let id = canonical_id("LinearReg.w");
    assert_eq!(block(SEED_TOP_WORD, Stream::Init, id, 0)[0], 0xffff_ffed);
    let t = xavier_uniform(
        SEED_TOP_WORD,
        Stream::Init,
        "LinearReg.w",
        &TensorType::f32(&[4, 1]),
    )
    .unwrap();
    let a = (6.0f64 / 5.0).sqrt();
    let v = t.data()[0];
    assert!(
        v.abs() <= a as f32,
        "element 0 is {v:.10} = {:.17}; a is {a:.17}",
        f64::from(v)
    );
}

/// DESIGN.md § 2.2: empty shapes are hard errors. A zero or unresolved dimension is refused as
/// `InitError::Shape`, not drawn and not a panic.
#[test]
fn a_zero_or_unresolved_dimension_is_refused_not_drawn() {
    let zero = TensorType::f32(&[0, 4]);
    assert_eq!(
        xavier_uniform(42, Stream::Init, "M.z", &zero),
        Err(InitError::Shape { path: "M.z".into() })
    );
    assert_eq!(
        standard_normal(42, Stream::Data, "z", &TensorType::f32(&[8, 0])),
        Err(InitError::Shape { path: "z".into() })
    );
    let symbolic = TensorType {
        dtype: DType::F32,
        shape: vec![Dim::Symbol(0), Dim::Static(4)],
    };
    assert_eq!(
        xavier_uniform(42, Stream::Init, "M.s", &symbolic),
        Err(InitError::Shape { path: "M.s".into() })
    );
}
