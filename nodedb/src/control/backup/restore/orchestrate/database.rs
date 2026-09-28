// SPDX-License-Identifier: BUSL-1.1

//! Re-issue one backed-up database's rows into its destination database.
//!
//! Every section re-issues as durable writes: Raft-replicated to every
//! replica of its group in cluster mode, WAL-appended then installed on a
//! single node. None is installed straight into a Data-Plane map, which
//! would hold it on one node only and lose it on restart. Every write names
//! the destination database, so each row lands on the vShard and core its
//! destination collection key homes to.

use std::sync::Arc;

use crate::Error;
use crate::control::state::SharedState;
use crate::types::TenantDataSnapshot;

use super::super::target::DatabaseTarget;
use super::rebind;
use super::reissue;
use super::stats::RestoreStats;

/// Re-issue every section of `snap`, the merged backup of the source
/// database `target.source`, into `target.dest`. Any failure is fatal — no
/// warn-and-continue.
pub(super) async fn reissue_database(
    state: &Arc<SharedState>,
    tenant_id: u64,
    target: DatabaseTarget,
    mut snap: TenantDataSnapshot,
    stats: &mut RestoreStats,
) -> Result<(), Error> {
    let columnar_snapshots = std::mem::take(&mut snap.columnar_engines);
    let timeseries_memtables = std::mem::take(&mut snap.timeseries);
    let flushed_ts_segments = std::mem::take(&mut snap.flushed_ts_segments);
    let crdt_state = std::mem::take(&mut snap.crdt_state);
    let kv_tables = std::mem::take(&mut snap.kv_tables);
    let vector_snapshots = std::mem::take(&mut snap.vectors);
    // Vector-index config re-issues as `VectorOp::SetParams` before the first
    // vector `Insert`: the Data Plane creates a (collection, field) HNSW index
    // on its first `Insert`, from whatever params it holds by then.
    let vector_params_snapshots = std::mem::take(&mut snap.vector_params);
    let index_config_snapshots = std::mem::take(&mut snap.index_configs);

    // The PK→surrogate identity map. It is bound on this node before any
    // re-issue, so a re-issued row keeps the surrogate the backup stored it
    // under unless this node already binds its key.
    let surrogate_binds = std::mem::take(&mut snap.surrogate_pk);
    rebind::rebind_surrogates(state, target, &surrogate_binds)?;

    // Document rows, their versions and graph edges re-issue as committed
    // redo records through each collection's apply log: every replica binds
    // the rows' identities, appends the record to its WAL, installs the rows
    // and derives their secondary index entries. The backup's own index
    // entries are therefore not installed.
    let redo = super::super::redo_reissue::reissue_rows_and_edges(
        state,
        tenant_id,
        target,
        super::super::redo_reissue::RestoredRows {
            documents: std::mem::take(&mut snap.documents),
            documents_versioned: std::mem::take(&mut snap.documents_versioned),
            edges: std::mem::take(&mut snap.edges),
            binds: &surrogate_binds,
        },
    )
    .await?;
    stats.documents_reissued += redo.documents;
    stats.edges_reissued += redo.edges;
    stats.redo_records += redo.records;

    // Plain-columnar rows: each collection's live rows replay as one durable
    // `ColumnarOp::Insert`. Collections with zero live rows are skipped.
    stats.columnar_engines +=
        reissue::reissue_columnar_snapshots(state, tenant_id, target, columnar_snapshots).await?;

    // Timeseries rows: each collection's memtable rows plus every flushed
    // partition's rows replay as one durable `TimeseriesOp::Ingest`.
    stats.timeseries_reissued += reissue::reissue_timeseries_snapshots(
        state,
        tenant_id,
        target,
        timeseries_memtables,
        flushed_ts_segments,
    )
    .await?;

    // CRDT state: each collection's Loro snapshot is proposed to the data
    // group owning that collection's vshard. Every replica applies the same
    // idempotent Loro merge and converges deterministically.
    stats.crdt_reissued +=
        super::super::crdt_reissue::reissue_crdt_snapshots(state, target, crdt_state).await?;

    // KV rows, one `KvOp::Put` per live row.
    stats.kv_reissued +=
        super::super::kv_reissue::reissue_kv_tables(state, tenant_id, target, kv_tables).await?;

    // Vector-index configuration, as `VectorOp::SetParams`. MUST run before
    // the vector-insert re-issue below — see the `vector_params_snapshots`
    // drain comment above.
    stats.vector_params_reissued += reissue::reissue_vector_params(
        state,
        tenant_id,
        target,
        vector_params_snapshots,
        index_config_snapshots,
    )
    .await?;

    // Vector rows, one `VectorOp::Insert` per restored vector.
    stats.vectors_reissued +=
        reissue::reissue_vector_snapshots(state, tenant_id, target, vector_snapshots).await?;

    Ok(())
}
