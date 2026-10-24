//! Deterministic parameter initialization and random draws (DESIGN.md § 2.3, § 6.1).
//!
//! Every random value comes from the counter-based PRNG Philox4x32-10 (Salmon et al., SC 2011),
//! never from a generator advanced in compiler iteration order. A draw is addressed by its key
//! and counter alone:
//!
//! - key `[seed as u32, (seed >> 32) as u32]`, from the global seed of the `train` block;
//! - counter `[index, stream, id as u32, (id >> 32) as u32]`: the element index in row-major
//!   order, the [`Stream`] tag, and the 64-bit id of what is drawn — for a parameter its
//!   canonical parameter id ([`canonical_id`] of its canonical path), for a data input the
//!   [`canonical_id`] of its name.
//!
//! Element `i` takes block `philox4x32_10(counter(i), key)`. A uniform draw uses word 0 as
//! `u = (w + 0.5) / 2^32`, in `(0, 1)`; a standard normal draw is Box-Muller over words 0 and 1,
//! `sqrt(-2 ln u0) * cos(2 pi u1)`. Both are computed in `f64` and rounded to `f32`. A tensor
//! with more than `2^32` elements has no counter for its last ones and is refused.
//!
//! `modelspec train --init <dir>` replaces the draw of a parameter by the values of its file in
//! `<dir>` ([`read_params`]).

use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::Path;

use modelspec_core::diag::Diagnostic;
use modelspec_core::ir::tensor::{Dim, Initializer, ParamId, TensorProgram, TensorType};
use thiserror::Error;

use super::data::{DataError, E_DATA_007, non_finite, read_f32, unreadable};
use crate::cpu::HostTensor;

/// What a draw is for: the stream tag of the counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stream {
    /// The initial values of the trained parameters.
    Init,
    /// The teacher parameters of M0 synthetic data, the known solution of DESIGN.md § 9 gate 4.
    Teacher,
    /// M0 synthetic data inputs.
    Data,
    /// Masked-language-model masking, drawn per step (`dataset::mlm`; coordinator decision D26).
    Masking,
    /// The epoch order of packed `Train` sequences (`dataset::mlm`; coordinator decision D26).
    /// Tag 4, between this and [`Stream::Masking`], is reserved for dropout.
    DataOrder,
}

impl Stream {
    /// The counter word that names the stream.
    pub const fn tag(self) -> u32 {
        match self {
            Stream::Init => 0,
            Stream::Teacher => 1,
            Stream::Data => 2,
            Stream::Masking => 3,
            Stream::DataOrder => 5,
        }
    }
}

/// Why a value could not be drawn.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum InitError {
    #[error("`{path}` is {ty}; xavier_uniform needs rank 2 [In, Out]")]
    Rank { path: String, ty: TensorType },
    #[error("`{path}` has {count} elements; a draw addresses at most 2^32")]
    TooManyElements { path: String, count: u128 },
    #[error("`{path}` has an unresolved or zero dimension")]
    Shape { path: String },
}

/// Philox4x32-10 of `counter` under `key`: ten rounds, the key bumped between rounds.
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    const M0: u32 = 0xD251_1F53;
    const M1: u32 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;
    let mulhilo = |a: u32, b: u32| {
        let product = u64::from(a) * u64::from(b);
        ((product >> 32) as u32, product as u32)
    };
    let mut c = counter;
    let mut k = key;
    for round in 0..10 {
        if round > 0 {
            k = [k[0].wrapping_add(W0), k[1].wrapping_add(W1)];
        }
        let (hi0, lo0) = mulhilo(M0, c[0]);
        let (hi1, lo1) = mulhilo(M1, c[2]);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
    }
    c
}

/// The canonical id of a path: the first 8 bytes of its `blake3` hash, little-endian
/// (DESIGN.md § 2.3).
pub fn canonical_id(path: &str) -> u64 {
    let hash = blake3::hash(path.as_bytes());
    let mut first = [0u8; 8];
    first.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(first)
}

/// The Philox block of element `index` of the draw `id` on `stream` under `seed`.
pub fn block(seed: u64, stream: Stream, id: u64, index: u32) -> [u32; 4] {
    philox4x32_10(
        [index, stream.tag(), id as u32, (id >> 32) as u32],
        [seed as u32, (seed >> 32) as u32],
    )
}

/// A word as a uniform draw in `(0, 1)`.
fn uniform(word: u32) -> f64 {
    (f64::from(word) + 0.5) / 4_294_967_296.0
}

/// The static dimensions of `ty` and its element count, refused past `2^32` elements.
fn dims(path: &str, ty: &TensorType) -> Result<(Vec<u64>, u32), InitError> {
    let mut shape = Vec::with_capacity(ty.shape.len());
    let mut count: u128 = 1;
    for dim in &ty.shape {
        match *dim {
            Dim::Static(n) if n > 0 => {
                shape.push(n);
                count = count.saturating_mul(u128::from(n));
            }
            _ => return Err(InitError::Shape { path: path.into() }),
        }
    }
    if count > 1 << 32 {
        return Err(InitError::TooManyElements {
            path: path.into(),
            count,
        });
    }
    Ok((shape, (count - 1) as u32))
}

/// A tensor of type `ty` whose element `i` is `draw(block(i))`.
fn draw(
    seed: u64,
    stream: Stream,
    path: &str,
    ty: &TensorType,
    element: impl Fn([u32; 4]) -> f64,
) -> Result<HostTensor<f32>, InitError> {
    let (shape, last) = dims(path, ty)?;
    let id = canonical_id(path);
    let data = (0..=last)
        .map(|i| element(block(seed, stream, id, i)) as f32)
        .collect();
    Ok(HostTensor::from_parts(shape, data))
}

/// `xavier_uniform` of the parameter `path` of type `ty` `[In, Out]`: each element drawn in `f64`
/// from `U(-a, a)`, `a = sqrt(6 / (In + Out))`, then rounded to `f32`, so `|v| <= f32(a)`.
pub fn xavier_uniform(
    seed: u64,
    stream: Stream,
    path: &str,
    ty: &TensorType,
) -> Result<HostTensor<f32>, InitError> {
    let (fan_in, fan_out) = match ty.shape[..] {
        [Dim::Static(fan_in), Dim::Static(fan_out)] => (fan_in, fan_out),
        [_, _] => return Err(InitError::Shape { path: path.into() }),
        _ => {
            return Err(InitError::Rank {
                path: path.into(),
                ty: ty.clone(),
            });
        }
    };
    let a = (6.0 / (fan_in as f64 + fan_out as f64)).sqrt();
    draw(seed, stream, path, ty, |b| a * (2.0 * uniform(b[0]) - 1.0))
}

/// A standard normal tensor of type `ty`, drawn by Box-Muller, named `name` for its id.
pub fn standard_normal(
    seed: u64,
    stream: Stream,
    name: &str,
    ty: &TensorType,
) -> Result<HostTensor<f32>, InitError> {
    draw(seed, stream, name, ty, |b| {
        (-2.0 * uniform(b[0]).ln()).sqrt() * (std::f64::consts::TAU * uniform(b[1])).cos()
    })
}

/// The value of every declared parameter of `program`, by its initializer, on `stream`.
pub fn initialize(
    program: &TensorProgram,
    seed: u64,
    stream: Stream,
) -> Result<BTreeMap<ParamId, HostTensor<f32>>, InitError> {
    program
        .params
        .iter()
        .map(|(&param, decl)| {
            let value = match decl.init {
                Initializer::XavierUniform => xavier_uniform(seed, stream, &decl.path, &decl.ty)?,
            };
            Ok((param, value))
        })
        .collect()
}

/// The file name stem of the parameter at canonical path `path` under `--init`: the path without
/// its leading model name (`w1` for `Mnist.w1`).
pub fn init_file_stem(path: &str) -> &str {
    path.split_once('.').map_or(path, |(_, rest)| rest)
}

/// The values of every parameter of `program` that has a file `<dir>/<stem>.f32`
/// ([`init_file_stem`]), row-major little-endian `f32` (`spec/domains/data.yaml`,
/// `ParamInitDirectory`). A parameter with no file is absent from the map and keeps its
/// initializer, and a file that names no parameter is ignored. Refused with [`E_DATA_007`]: a
/// directory that cannot be read, a directory with no file of any parameter, and a file that
/// cannot be read, is not exactly the parameter's size, or holds a NaN or an infinity.
pub fn read_params(
    dir: &Path,
    program: &TensorProgram,
) -> Result<BTreeMap<ParamId, HostTensor<f32>>, DataError> {
    if let Err(error) = std::fs::read_dir(dir) {
        return Err(unreadable("the init directory", dir, &error));
    }
    let mut values = BTreeMap::new();
    for (&param, decl) in &program.params {
        let path = dir.join(format!("{}.f32", init_file_stem(&decl.path)));
        let what = format!("the init file of the parameter `{}`", decl.path);
        let data = match read_f32(&path) {
            Ok(data) => data,
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => return Err(unreadable(&what, &path, &error)),
        };
        let (shape, last) = dims(&decl.path, &decl.ty)?;
        let count = u64::from(last) + 1;
        if data.len() as u64 != count {
            return Err(DataError::Refused(
                Diagnostic::new(E_DATA_007, format!("{what} is not the parameter's size"))
                    .with("file", path.display().to_string())
                    .with("values", data.len().to_string())
                    .with(decl.path.clone(), format!("{} = {count} values", decl.ty)),
            ));
        }
        if let Some(refused) = non_finite(&what, &path, &data) {
            return Err(refused);
        }
        values.insert(param, HostTensor::from_parts(shape, data));
    }
    if values.is_empty() {
        let expected = program
            .params
            .values()
            .map(|decl| format!("{}.f32", init_file_stem(&decl.path)))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(DataError::Refused(
            Diagnostic::new(
                E_DATA_007,
                "the init directory holds no file of any parameter",
            )
            .with("directory", dir.display().to_string())
            .with("expected", expected),
        ));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelspec_core::ir::tensor::TensorProgramBuilder;

    /// The Philox4x32-10 known-answer vectors of Random123 (Salmon et al., SC 2011),
    /// `kat_vectors`: counter, key, result.
    #[test]
    fn philox_matches_the_published_known_answer_vectors() {
        let vectors: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
            (
                [0, 0, 0, 0],
                [0, 0],
                [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
            ),
            (
                [0xffffffff; 4],
                [0xffffffff; 2],
                [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
            ),
            (
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0xa4093822, 0x299f31d0],
                [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
            ),
        ];
        for (counter, key, expected) in vectors {
            assert_eq!(
                philox4x32_10(counter, key),
                expected,
                "counter {counter:08x?} key {key:08x?}"
            );
        }
    }

    /// `blake3("")` is `af1349b9f5f9a1a6...`: its first 8 bytes read little-endian.
    #[test]
    fn canonical_id_is_the_first_8_blake3_bytes_little_endian() {
        assert_eq!(canonical_id(""), 0xa6a1_f9f5_b949_13af);
        assert_ne!(canonical_id("LinearReg.w"), canonical_id("LinearReg.b"));
    }

    #[test]
    fn a_block_is_keyed_by_the_seed_and_counts_index_stream_and_id() {
        let seed = 0x0123_4567_89ab_cdef_u64;
        let id = 0xfedc_ba98_7654_3210_u64;
        assert_eq!(
            block(seed, Stream::Data, id, 7),
            philox4x32_10([7, 2, 0x7654_3210, 0xfedc_ba98], [0x89ab_cdef, 0x0123_4567])
        );
        assert_eq!(Stream::Init.tag(), 0);
        assert_eq!(Stream::Teacher.tag(), 1);
    }

    fn w() -> TensorType {
        TensorType::f32(&[4, 1])
    }

    #[test]
    fn xavier_uniform_draws_word_0_of_each_element_into_minus_a_to_a() {
        let t = xavier_uniform(42, Stream::Init, "LinearReg.w", &w()).unwrap();
        assert_eq!(t.shape(), &[4, 1]);
        let a = (6.0f64 / 5.0).sqrt();
        let id = canonical_id("LinearReg.w");
        for (i, &v) in t.data().iter().enumerate() {
            let word = block(42, Stream::Init, id, i as u32)[0];
            let u = (f64::from(word) + 0.5) / 4_294_967_296.0;
            assert_eq!(v, (a * (2.0 * u - 1.0)) as f32, "element {i}");
            assert!(v.abs() <= a as f32, "element {i}: {v}");
        }
    }

    #[test]
    fn xavier_uniform_covers_minus_a_to_a_evenly() {
        let ty = TensorType::f32(&[200, 100]);
        let t = xavier_uniform(7, Stream::Init, "M.big", &ty).unwrap();
        let a = (6.0f64 / 300.0).sqrt();
        let n = t.len() as f64;
        let mean = t.data().iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = t
            .data()
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.01 * a, "mean {mean}");
        assert!((var / (a * a / 3.0) - 1.0).abs() < 0.03, "variance {var}");
        assert!(t.data().iter().all(|&v| v.abs() <= a as f32));
    }

    #[test]
    fn xavier_uniform_refuses_another_rank() {
        let ty = TensorType::f32(&[4]);
        assert_eq!(
            xavier_uniform(42, Stream::Init, "M.v", &ty),
            Err(InitError::Rank {
                path: "M.v".into(),
                ty
            })
        );
    }

    #[test]
    fn a_draw_past_2_to_the_32_elements_is_refused() {
        let ty = TensorType::f32(&[1 << 17, 1 << 16]);
        assert_eq!(
            xavier_uniform(42, Stream::Init, "M.huge", &ty),
            Err(InitError::TooManyElements {
                path: "M.huge".into(),
                count: 1 << 33
            })
        );
    }

    #[test]
    fn the_same_seed_draws_the_same_values() {
        let a = xavier_uniform(42, Stream::Init, "LinearReg.w", &w()).unwrap();
        let b = xavier_uniform(42, Stream::Init, "LinearReg.w", &w()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn another_seed_draws_other_values() {
        let a = xavier_uniform(42, Stream::Init, "LinearReg.w", &w()).unwrap();
        let b = xavier_uniform(43, Stream::Init, "LinearReg.w", &w()).unwrap();
        assert!(
            a.data().iter().zip(b.data()).all(|(x, y)| x != y),
            "{a:?} {b:?}"
        );
    }

    #[test]
    fn another_stream_or_path_draws_other_values() {
        let a = xavier_uniform(42, Stream::Init, "LinearReg.w", &w()).unwrap();
        let teacher = xavier_uniform(42, Stream::Teacher, "LinearReg.w", &w()).unwrap();
        let other = xavier_uniform(42, Stream::Init, "LinearReg.v", &w()).unwrap();
        assert_ne!(a, teacher);
        assert_ne!(a, other);
    }

    /// Values depend on the canonical path, not on the order parameters are declared in.
    #[test]
    fn initialize_does_not_depend_on_declaration_order() {
        let program = |paths: [&str; 2]| {
            let mut b = TensorProgramBuilder::new();
            for p in paths {
                b.declare_param(p, w(), Initializer::XavierUniform).unwrap();
            }
            b.finish()
        };
        let ab = initialize(&program(["M.a", "M.b"]), 42, Stream::Init).unwrap();
        let ba = initialize(&program(["M.b", "M.a"]), 42, Stream::Init).unwrap();
        assert_eq!(ab[&ParamId(0)], ba[&ParamId(1)]);
        assert_eq!(ab[&ParamId(1)], ba[&ParamId(0)]);
        assert_eq!(
            ab[&ParamId(0)],
            xavier_uniform(42, Stream::Init, "M.a", &w()).unwrap()
        );
    }

    #[test]
    fn standard_normal_is_box_muller_over_words_0_and_1() {
        let ty = TensorType::f32(&[8, 4]);
        let t = standard_normal(42, Stream::Data, "x", &ty).unwrap();
        assert_eq!(t.shape(), &[8, 4]);
        let id = canonical_id("x");
        for (i, &v) in t.data().iter().enumerate() {
            let [w0, w1, _, _] = block(42, Stream::Data, id, i as u32);
            let u0 = (f64::from(w0) + 0.5) / 4_294_967_296.0;
            let u1 = (f64::from(w1) + 0.5) / 4_294_967_296.0;
            let z = (-2.0 * u0.ln()).sqrt() * (std::f64::consts::TAU * u1).cos();
            assert_eq!(v, z as f32, "element {i}");
        }
    }

    #[test]
    fn standard_normal_has_mean_0_and_variance_1() {
        let ty = TensorType::f32(&[400, 250]);
        let t = standard_normal(42, Stream::Data, "x", &ty).unwrap();
        let n = t.len() as f64;
        let mean = t.data().iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = t
            .data()
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
        assert!(t.data().iter().all(|v| v.is_finite()));
    }
}
