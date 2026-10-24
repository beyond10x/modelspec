//! Adversary case for `story:matmul-shape-aware-tiling`: an artifact compiled before the tile
//! rule changed carries matmul module ids derived from the old tile text, and the reader of
//! this tree refuses it (E-ARTIFACT-003, "is not the id its content gives") instead of
//! accepting SPIR-V written for the old tile.
//!
//! Ignored by default: it needs an artifact directory compiled by the base commit's
//! `modelspec compile <example> --backend vulkan --output <dir> --device-profile <profile>`,
//! named by `ADVERSARY_TILING_BASE_ARTIFACT`, and fails without it.

use std::path::PathBuf;

use modelspec_core::artifact;

const ENV: &str = "ADVERSARY_TILING_BASE_ARTIFACT";

#[test]
#[ignore = "needs an artifact compiled at the base commit, named by ADVERSARY_TILING_BASE_ARTIFACT"]
fn an_artifact_compiled_before_the_tile_change_is_refused() {
    let dir = PathBuf::from(
        std::env::var(ENV).unwrap_or_else(|_| panic!("{ENV} names the base artifact")),
    );
    assert!(dir.join("artifact.json").is_file(), "{}", dir.display());
    match artifact::read(&dir) {
        Ok(_) => panic!("the reader accepted an artifact compiled with the old tile rule"),
        Err(diagnostic) => {
            let rendered = diagnostic.to_string();
            eprintln!("{rendered}");
            assert_eq!(diagnostic.code.as_str(), "E-ARTIFACT-003", "{rendered}");
            assert!(
                rendered.contains("is not the id its content gives"),
                "{rendered}"
            );
        }
    }
}
