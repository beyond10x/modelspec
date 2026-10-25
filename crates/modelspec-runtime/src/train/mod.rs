//! Training: deterministic parameter initialization ([`init`]), M0 synthetic teacher data
//! ([`data`]), SGD steps on the CPU interpreter ([`optimizer`]), the training loop ([`runner`]),
//! the held-out evaluation of `--eval-every` ([`eval`]) and `.msckpt` checkpoints ([`checkpoint`]).

pub mod checkpoint;
pub mod data;
pub mod eval;
pub mod init;
pub mod optimizer;
pub mod runner;
