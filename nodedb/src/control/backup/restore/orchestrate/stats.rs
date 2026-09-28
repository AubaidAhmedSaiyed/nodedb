// SPDX-License-Identifier: BUSL-1.1

//! Aggregate stats returned to the client at the end of a RESTORE TENANT.

use serde::Serialize;

use crate::types::TenantDataSnapshot;

/// Aggregate stats returned to the client at the end of a restore.
#[derive(Debug, Default, Clone, Serialize)]
pub struct RestoreStats {
    pub tenant_id: u64,
    pub dry_run: bool,
    pub sections: u16,
    /// Number of databases the backup covers.
    pub databases: usize,
    /// Number of those databases the restore created on this cluster.
    pub databases_created: usize,
    pub source_vshard_count: u16,
    pub documents: usize,
    pub indexes: usize,
    pub edges: usize,
    pub vectors: usize,
    pub kv_tables: usize,
    pub crdt_state: usize,
    pub timeseries: usize,
    pub columnar_engines: usize,
    pub flushed_ts_segments: usize,
    /// Number of timeseries collections re-issued durably (Raft/WAL) on restore.
    pub timeseries_reissued: usize,
    /// Number of CRDT tenant-snapshot imports re-issued durably (Raft/WAL) on
    /// restore — one per distinct data group that owns any CRDT collection.
    pub crdt_reissued: usize,
    /// Number of individual vectors re-issued durably (Raft/WAL) on restore.
    pub vectors_reissued: usize,
    /// Number of individual KV rows re-issued durably (Raft/WAL) on restore.
    pub kv_reissued: usize,
    /// Number of (collection, field) vector-index HNSW/PQ/IVF configs
    /// re-issued durably (Raft/WAL) on restore.
    pub vector_params_reissued: usize,
    /// Number of PK→surrogate identity bindings rebound into the catalog.
    pub surrogate_pk: usize,
    /// Document sub-records re-issued: one per current row, one per version
    /// of a `bitemporal=true` row.
    pub documents_reissued: usize,
    /// Edge versions re-issued.
    pub edges_reissued: usize,
    /// Redo records the document and edge re-issue committed.
    pub redo_records: usize,
}

impl RestoreStats {
    /// Add the section sizes of one database's merged snapshot. The
    /// columnar count is the number re-issued, so a restore adds it as it
    /// re-issues and a dry run adds the section size.
    pub fn count_sections(&mut self, snap: &TenantDataSnapshot) {
        self.documents += snap.documents.len() + snap.documents_versioned.len();
        self.indexes += snap.indexes.len() + snap.indexes_versioned.len();
        self.edges += snap.edges.len();
        self.vectors += snap.vectors.len();
        self.kv_tables += snap.kv_tables.len();
        // CRDT state is one entry per (tenant, collection).
        self.crdt_state += snap.crdt_state.len();
        self.timeseries += snap.timeseries.len();
        self.flushed_ts_segments += snap.flushed_ts_segments.len();
        self.surrogate_pk += snap.surrogate_pk.len();
    }
}
