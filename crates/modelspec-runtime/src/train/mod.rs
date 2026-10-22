//! Training: deterministic parameter initialization ([`init`]), M0 synthetic teacher data
//! ([`data`]), SGD steps on the CPU interpreter ([`optimizer`]), the training loop ([`runner`])
//! and `.msckpt` checkpoints ([`checkpoint`]).

pub mod checkpoint;
pub mod data;
pub mod init;
pub mod optimizer;
pub mod runner;
