//! The shared written-out `f32` `sin` and `cos` routine of `sigreg` (DESIGN.md § 4;
//! story:jepa-sigreg-cpu).
//!
//! `sigreg`'s `SigregColumns` and `SigregColumnsGrad` evaluate `sin` and `cos` in `f32` through
//! one routine, Cody–Waite reduction by `pi/2` with a three-part constant and fixed minimax
//! polynomials, not the platform `sinf`, so the Vulkan kernels can emit the identical sequence.
//! Its error is measured here against the `f64` standard library, in units of the `f32` spacing at
//! the `f64` value.

use modelspec_runtime::cpu::{Element, sin_cos};

/// The `f32` spacing at `|v|`: the distance from the `f32` of `|v|` to the next larger `f32`.
/// Below the smallest normal it is the subnormal spacing.
fn ulp_at(v: f64) -> f64 {
    let x = (v.abs() as f32).max(f32::MIN_POSITIVE);
    f64::from(f32::from_bits(x.to_bits() + 1)) - f64::from(x)
}

/// The error of `got` against `want` in `f32` ulps at `want`.
fn ulps(got: f32, want: f64) -> f64 {
    (f64::from(got) - want).abs() / ulp_at(want)
}

/// Done when: at every multiple of `1/256` in `[-64, 64]` the routine is within 2 `f32` ulp of
/// `f64` `sin` and `cos`.
#[test]
fn sin_cos_is_within_two_ulp_of_f64_on_a_1_256_grid() {
    let mut worst = (0.0f64, 0.0f32, "");
    for i in -64 * 256..=64 * 256 {
        let x = i as f32 / 256.0;
        let (s, c) = sin_cos(x);
        let wide = f64::from(x);
        for (name, got, want) in [("sin", s, wide.sin()), ("cos", c, wide.cos())] {
            let e = ulps(got, want);
            if e > worst.0 {
                worst = (e, x, name);
            }
            assert!(
                e <= 2.0,
                "{name}({x}) = {got:e}, f64 {want:e}: {e} ulp over 2"
            );
        }
    }
    eprintln!("worst: {} ulp, {}({})", worst.0, worst.2, worst.1);
}

/// Done when: the routine returns the same bits on every call, for every argument of the grid and
/// for repeated calls of one argument.
#[test]
fn sin_cos_returns_the_same_bits_on_every_call() {
    let grid: Vec<f32> = (-64 * 256..=64 * 256).map(|i| i as f32 / 256.0).collect();
    let first: Vec<(u32, u32)> = grid
        .iter()
        .map(|&x| {
            let (s, c) = sin_cos(x);
            (s.to_bits(), c.to_bits())
        })
        .collect();
    for _ in 0..3 {
        for (&x, &(s0, c0)) in grid.iter().zip(&first) {
            let (s, c) = sin_cos(x);
            assert_eq!((s.to_bits(), c.to_bits()), (s0, c0), "sin_cos({x})");
        }
    }
}

/// The `Element` route the interpreter takes: `f32` goes through the shared routine bit for bit,
/// `f64` through the `f64` standard library, the reference math of the gradient checks.
#[test]
fn element_sin_cos_is_the_routine_in_f32_and_the_standard_library_in_f64() {
    for i in -1024..=1024 {
        let x = i as f32 / 16.0;
        let (s, c) = <f32 as Element>::sin_cos(x);
        let (rs, rc) = sin_cos(x);
        assert_eq!(
            (s.to_bits(), c.to_bits()),
            (rs.to_bits(), rc.to_bits()),
            "{x}"
        );
        let wide = f64::from(x);
        let (s, c) = <f64 as Element>::sin_cos(wide);
        assert_eq!(
            (s.to_bits(), c.to_bits()),
            (wide.sin().to_bits(), wide.cos().to_bits()),
            "{wide}"
        );
    }
}
