// SPDX-License-Identifier: BUSL-1.1

//! Capture sites for inbound array sync.

use faultbox::{Capture, EventKind};

use crate::diag::context;

/// Report a poisoned snapshot-assembly lock and the assemblies discarded
/// with it. Called only from the lock site that detects the poison.
pub fn array_snapshot_buffer_poisoned(
    array: &str,
    tenant_id: u64,
    database_id: u64,
    discarded: usize,
) {
    let ctx = context::ArraySnapshotBufferPoisoned {
        array,
        tenant_id,
        database_id,
        discarded,
    };
    let _ = Capture::new(
        EventKind::InvariantViolation,
        "array sync: a panic poisoned the snapshot-assembly lock, so its buffers were discarded",
    )
    .domain(&ctx)
    .with_backtrace()
    .emit();
}
