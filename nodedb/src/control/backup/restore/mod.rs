// SPDX-License-Identifier: BUSL-1.1

//! RESTORE TENANT — module root.
//!
//! Submodule wiring only. All restore orchestrator logic lives in
//! [`orchestrate`]; each engine's re-issue lives in its own submodule; the
//! section decoding lives in `sections`; the destination databases are
//! resolved in `databases`; `target` maps a source collection name to its
//! destination database.

pub mod columnar_reissue;
pub(crate) mod crdt_reissue;
mod databases;
mod durable;
pub(crate) mod guard;
mod kv_reissue;
mod orchestrate;
mod quorum;
mod redo_reissue;
mod sections;
mod target;
pub mod timeseries_reissue;
pub mod vector_reissue;

pub use orchestrate::{RestoreStats, restore_tenant};
