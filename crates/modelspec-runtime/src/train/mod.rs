//! Training: deterministic parameter initialization ([`init`]), M0 synthetic teacher data
//! ([`data`]), SGD steps on the CPU interpreter ([`optimizer`]) and the training loop.

pub mod data;
pub mod init;
pub mod optimizer;
pub mod runner;
