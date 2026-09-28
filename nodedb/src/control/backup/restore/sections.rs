// SPDX-License-Identifier: BUSL-1.1

//! Catalog-section and data-section helpers for RESTORE TENANT.

use std::collections::BTreeMap;
use std::sync::Arc;

use nodedb_types::backup_envelope::{
    DatabaseDataSection, Envelope, SECTION_ORIGIN_CATALOG_ROWS, SECTION_ORIGIN_DATABASES,
    SECTION_ORIGIN_SOURCE_TOMBSTONES, SECTION_ORIGIN_SURROGATE_PK, Section, SourceTombstoneEntry,
    StoredCollectionBlob, SurrogateBindBlob,
};

use crate::Error;
use crate::control::catalog_entry::CatalogEntry;
use crate::control::metadata_proposer::propose_catalog_entry;
use crate::control::security::catalog::StoredCollection;
use crate::control::state::SharedState;
use crate::types::{SurrogateBindEntry, TenantDataSnapshot};

use super::databases::DatabaseMap;

/// Merge every data section and the surrogate-pk section into one
/// `TenantDataSnapshot` per source database, keyed by source database id.
pub(super) fn merge_sections(
    sections: &[Section],
) -> Result<BTreeMap<u64, TenantDataSnapshot>, Error> {
    let mut merged: BTreeMap<u64, TenantDataSnapshot> = BTreeMap::new();
    for section in sections {
        // The surrogate-pk metadata section carries the PK→surrogate identity
        // map (not a tenant snapshot). Each bind goes to its database's
        // snapshot, so the restore rebinds it in that database.
        if section.origin_node_id == SECTION_ORIGIN_SURROGATE_PK {
            let binds: Vec<SurrogateBindBlob> =
                zerompk::from_msgpack(&section.body).map_err(|_| Error::Internal {
                    detail: "invalid backup format: surrogate-pk section payload is not decodable"
                        .into(),
                })?;
            for b in binds {
                merged
                    .entry(b.database_id)
                    .or_default()
                    .surrogate_pk
                    .push(SurrogateBindEntry {
                        database_id: b.database_id,
                        tenant_id: b.tenant_id,
                        collection: b.collection,
                        pk: b.pk,
                        surrogate: b.surrogate,
                    });
            }
            continue;
        }
        if is_metadata_section(section) {
            continue;
        }
        let data: DatabaseDataSection =
            zerompk::from_msgpack(&section.body).map_err(|_| Error::Internal {
                detail: "invalid backup format: section payload is not a database data section"
                    .into(),
            })?;
        let snap: TenantDataSnapshot =
            zerompk::from_msgpack(&data.snapshot).map_err(|_| Error::Internal {
                detail: "invalid backup format: section payload is not a tenant snapshot".into(),
            })?;
        append_snapshot(merged.entry(data.database_id).or_default(), snap);
    }
    Ok(merged)
}

/// Concatenate every section of `snap` onto `into`. Each source node holds
/// disjoint vShards, so the sections never overlap.
fn append_snapshot(into: &mut TenantDataSnapshot, snap: TenantDataSnapshot) {
    // Destructure exhaustively so a new field fails to compile here rather
    // than being dropped from the restore.
    let TenantDataSnapshot {
        documents,
        indexes,
        edges,
        vectors,
        kv_tables,
        crdt_state,
        crdt_constraints,
        timeseries,
        flushed_ts_segments,
        columnar_engines,
        vector_params,
        index_configs,
        surrogate_pk,
        tenant_edges,
        group_write_marks,
        documents_versioned,
        indexes_versioned,
    } = snap;
    into.documents.extend(documents);
    into.indexes.extend(indexes);
    into.documents_versioned.extend(documents_versioned);
    into.indexes_versioned.extend(indexes_versioned);
    into.edges.extend(edges);
    into.vectors.extend(vectors);
    into.vector_params.extend(vector_params);
    into.index_configs.extend(index_configs);
    into.kv_tables.extend(kv_tables);
    // CRDT state is per-collection and tenant-explicit. Loro import is a
    // monotonic merge so concatenating section contributions is safe.
    into.crdt_state.extend(crdt_state);
    into.crdt_constraints.extend(crdt_constraints);
    into.timeseries.extend(timeseries);
    into.flushed_ts_segments.extend(flushed_ts_segments);
    into.columnar_engines.extend(columnar_engines);
    into.surrogate_pk.extend(surrogate_pk);
    into.tenant_edges.extend(tenant_edges);
    // A backup carries no marks: the guard reads the destination's marks.
    into.group_write_marks.extend(group_write_marks);
}

pub(super) fn is_metadata_section(section: &Section) -> bool {
    matches!(
        section.origin_node_id,
        SECTION_ORIGIN_CATALOG_ROWS
            | SECTION_ORIGIN_SOURCE_TOMBSTONES
            | SECTION_ORIGIN_SURROGATE_PK
            | SECTION_ORIGIN_DATABASES
    )
}

/// Apply catalog-row and source-tombstone sections to the destination catalog.
/// Runs BEFORE the data-section restore, after `databases` maps every source
/// database to its destination.
///
/// Each catalog row moves to its destination database. Catalog rows are
/// proposed cluster-wide through the metadata Raft group (group 0) — exactly
/// like CREATE COLLECTION — so every node's catalog learns the restored
/// collection and can serve it. A catalog-propose failure on this path is
/// FATAL: returning the data restored but unqueryable on non-coordinator nodes
/// is the silent-partial-success anti-pattern this codebase forbids.
///
/// Returns every collection written to the catalog, in section order. The
/// caller registers each one with this node's Data Plane before any restored
/// row is installed or reissued: a catalog row alone leaves `doc_configs`
/// without the collection's declaration, and a reissued timeseries row would
/// then be ingested into an inferred shape.
pub(super) fn apply_metadata_sections(
    state: &Arc<SharedState>,
    tenant_id: u64,
    env: &Envelope,
    databases: &DatabaseMap,
) -> Result<Vec<StoredCollection>, Error> {
    let catalog = state.credentials.catalog();
    let mut restored: Vec<StoredCollection> = Vec::new();

    for section in &env.sections {
        match section.origin_node_id {
            SECTION_ORIGIN_CATALOG_ROWS => {
                let blobs = zerompk::from_msgpack::<Vec<StoredCollectionBlob>>(&section.body)
                    .map_err(|_| Error::Internal {
                        detail: "invalid backup format: catalog-rows section is not decodable"
                            .into(),
                    })?;
                for blob in blobs {
                    let mut coll =
                        zerompk::from_msgpack::<StoredCollection>(&blob.bytes).map_err(|_| {
                            Error::Internal {
                                detail: format!(
                                    "invalid backup format: catalog row of '{}' is not decodable",
                                    blob.name
                                ),
                            }
                        })?;
                    coll.database_id = databases.target(blob.database_id)?.dest;
                    // Propose the collection through the metadata Raft
                    // group so every node's applier (`catalog_entry::
                    // apply::collection::put`) writes the row — mirroring
                    // CREATE COLLECTION and DROP COLLECTION. The proposer
                    // blocks on its local applied-index watcher, so on the
                    // cluster path it has already applied the put via the
                    // same applier — we must NOT also put locally (double-put).
                    let entry = CatalogEntry::PutCollection(Box::new(coll.clone()));
                    if propose_catalog_entry(state, &entry)?.needs_local_apply() {
                        // Single-node / no-cluster fallback: apply the
                        // catalog mutation directly, matching what the
                        // applier would have done on a clustered deployment.
                        // A failure here is FATAL — the collection would be
                        // unqueryable otherwise.
                        catalog.put_collection(coll.database_id, &coll)?;
                    }
                    restored.push(coll);
                }
            }
            SECTION_ORIGIN_SOURCE_TOMBSTONES => {
                let tombs = zerompk::from_msgpack::<Vec<SourceTombstoneEntry>>(&section.body)
                    .map_err(|_| Error::Internal {
                        detail: "invalid backup format: source-tombstones section is not \
                                 decodable"
                            .into(),
                    })?;
                for t in tombs {
                    let database_id = databases.target(t.database_id)?.dest.as_u64();
                    // Replicate via the metadata Raft group so every node's boot WAL
                    // replay barrier matches — a coordinator-local tombstone lets purged
                    // writes resurrect on follower restart.
                    let entry = CatalogEntry::RecordWalTombstone {
                        database_id,
                        tenant_id,
                        collection: t.collection.clone(),
                        purge_lsn: t.purge_lsn,
                    };
                    if propose_catalog_entry(state, &entry)?.needs_local_apply() {
                        // Single-node / no-cluster fallback: apply directly,
                        // matching the applier. A failure here is FATAL — a
                        // silently-skipped tombstone means purged writes resurrect
                        // on restart.
                        catalog.record_wal_tombstone(
                            database_id,
                            tenant_id,
                            &t.collection,
                            t.purge_lsn,
                        )?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(restored)
}
