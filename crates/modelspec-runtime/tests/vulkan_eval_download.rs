//! The download buffers of a stepwise Vulkan run (story:vulkan-eval-download-per-batch,
//! coordinator decision D76, 2026-10-26), without a device: the loss of a step and one held-out
//! batch's prediction, whatever the number of `HeldOut` batches, so `examples/bert_tiny.mspec`
//! over the packed WikiText-103 corpus, `[4096, 8192]` `f32` logits per batch and 68 batches,
//! downloads `4 + 134,217,728` bytes, where a buffer of every batch's logits, 9,126,805,508
//! bytes, exceeded the RTX 3090's `maxStorageBufferRange` before step 0.
#![cfg(feature = "vulkan")]

use modelspec_core::ir::kernel::BufferView;
use modelspec_core::ir::tensor::DType;
use modelspec_runtime::vulkan::stepwise::{DownloadBytes, download_bytes};

/// The `maxStorageBufferRange` the RTX 3090 reports, the limit of the refusal D76 records.
const RTX_3090_MAX_STORAGE_BUFFER_RANGE: u64 = 4_294_967_295;
/// The `HeldOut` batches of 32 sequences of the packed corpus at seq-len 128.
const CORPUS_BATCHES: usize = 68;

/// An `f32` prediction view of `rows` by `classes`.
fn prediction(rows: u64, classes: u64) -> BufferView {
    BufferView {
        buffer: 0,
        dtype: DType::F32,
        element_count: rows * classes,
        offset: 0,
        size_bytes: 4 * rows * classes,
        alignment: 4,
    }
}

/// Scenario 1: the logits of `examples/bert_tiny.mspec` over the packed corpus, `[4096, 8192]`
/// `f32` and 68 batches, are downloaded through `4 + 134,217,728` bytes, below the RTX 3090's
/// `maxStorageBufferRange`.
#[test]
fn the_packed_corpus_downloads_one_prediction() {
    let bytes = download_bytes(&prediction(4096, 8192), CORPUS_BATCHES);
    assert_eq!(bytes.total(), 4 + 134_217_728, "{bytes:?}");
    assert_eq!(
        bytes,
        DownloadBytes {
            loss: 4,
            prediction: 134_217_728,
        }
    );
    assert!(bytes.prediction <= RTX_3090_MAX_STORAGE_BUFFER_RANGE);
}

/// The download is one prediction for any positive number of batches, and the loss alone
/// without held-out evaluation.
#[test]
fn the_download_does_not_grow_with_the_batches() {
    let view = prediction(4096, 8192);
    let one = download_bytes(&view, 1);
    for batches in [2, 64, CORPUS_BATCHES, 10_000] {
        assert_eq!(download_bytes(&view, batches), one, "{batches} batches");
    }
    assert_eq!(
        download_bytes(&view, 0),
        DownloadBytes {
            loss: 4,
            prediction: 0,
        }
    );
}
