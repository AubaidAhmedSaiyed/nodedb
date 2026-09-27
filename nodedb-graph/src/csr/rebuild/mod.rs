// SPDX-License-Identifier: Apache-2.0

//! Rebuild of a live `CsrIndex` off its owning thread, with no lost writes.
//!
//! - `begin_rebuild` snapshots the index and opens a write journal.
//! - `CsrRebuildSeed::build` compacts the snapshot on any thread.
//! - `finish_rebuild` restores the compacted copy, replays the journal onto
//!   it and returns it for the caller to install in one step.

pub mod install;
pub mod journal;
pub mod seed;

pub use journal::CsrJournal;
pub use seed::{CsrRebuildSeed, CsrRebuilt};
