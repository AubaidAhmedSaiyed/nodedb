// SPDX-License-Identifier: BUSL-1.1

//! Snapshot phase for `MOVE TENANT`.
//!
//! Snapshots the tenant on every core of the local node and returns the
//! merged snapshot bytes. The snapshot covers the local node only: the
//! offline drain window ensures no cross-node writes are in flight. The
//! cluster fan-out orchestrator (`backup::orchestrator::backup_tenant`) is
//! bypassed here because its caller path requires `Arc<SharedState>`, which
//! the DDL dispatch pipeline does not carry.

use std::time::Duration;

use bytes::Bytes;

use crate::control::server::exchange::snapshot_tenant_on_local_cores;
use crate::control::state::SharedState;
use crate::types::{DatabaseId, TenantId};
use nodedb_types::NodeDbError;

/// Run the snapshot phase: produce a snapshot of `tenant_id` from every core
/// of the local node.
///
/// `source_db_id` is the database the tenant is being moved FROM. The
/// snapshot captures its live data.
///
/// Returns the raw snapshot bytes on success.
pub async fn run(
    state: &SharedState,
    tenant_id: TenantId,
    source_db_id: DatabaseId,
    timeout: Duration,
) -> Result<Bytes, NodeDbError> {
    // The phase code stays the statement's verdict, and the typed dispatch
    // error rides as its cause with its own class.
    let raw = snapshot_tenant_on_local_cores(state, tenant_id, source_db_id, timeout)
        .await
        .map_err(|e| {
            NodeDbError::move_tenant_snapshot_failed(
                tenant_id.as_u64().to_string(),
                "snapshot dispatch failed",
            )
            .with_cause(crate::error_classify::classify(&e))
        })?;
    Ok(Bytes::from(raw))
}

/// Return the temporary in-cluster storage key for the tenant's snapshot.
///
/// This key is recorded in the journal so crash recovery can clean up any
/// partial snapshot artifact.
pub fn temp_key(tenant_id: TenantId) -> String {
    format!("_move_tenant_snapshot_{}", tenant_id.as_u64())
}

/// Delete the temporary snapshot (best-effort; called on cutover success or
/// failure compensation).
///
/// In this implementation the snapshot lives in memory and is not persisted
/// to a separate store — the `temp_key` recorded in the journal is used for
/// identification only.  This function is a no-op but is kept as an extension
/// point for future durable snapshot storage.
pub async fn delete_temp(_state: &SharedState, _key: &str) -> Result<(), NodeDbError> {
    Ok(())
}
