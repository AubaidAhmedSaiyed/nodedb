// SPDX-License-Identifier: BUSL-1.1

//! BACKUP TENANT orchestrator.
//!
//! Discovers every node that holds a vShard owning some of the
//! tenant's data, dispatches `MetaOp::CreateTenantSnapshot` to each
//! (local SPSC for self, `RaftRpc::ExecuteRequest` for remotes),
//! and packs the gathered per-node snapshots into a `BackupEnvelope`.
//!
//! The backup covers every database the tenant has a collection in. Each
//! source node snapshots each of those databases after one consistent cut,
//! and each snapshot becomes one data section that names its database.
//!
//! Single-node mode is the degenerate case: routing table absent
//! (or 1 node) → one source, origin = self.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use bytes::Bytes;
use nodedb_cluster::routing::VSHARD_COUNT;
use nodedb_types::backup_envelope::{DatabaseDataSection, EnvelopeMeta, EnvelopeWriter};

use crate::Error;
use crate::bridge::envelope::PhysicalPlan;
use crate::control::state::SharedState;
use crate::types::DatabaseId;
use nodedb_physical::physical_plan::MetaOp;

use super::metadata::{TenantDatabase, encode_section_part};
use super::node_snapshot::{is_self, snapshot_remote, snapshot_self};

/// Build a complete tenant backup envelope by fanning out across the
/// cluster, gathering each node's slice, and framing the result.
///
/// Single-node and cluster paths converge here — a single-node server
/// produces one data section per database with origin = self.
pub async fn backup_tenant(state: &Arc<SharedState>, tenant_id: u64) -> Result<Bytes, Error> {
    // Assign every vshard to exactly ONE source node (the leader of its Raft
    // group, or — when no leader is elected yet — the lowest-id member), and
    // gather only from those source nodes. Under RF>1 every replica holds the
    // full vshard data, so gathering from all members and merging would
    // MULTIPLY append-style engine rows (columnar / timeseries) by the
    // replication factor. Filtering each source node's snapshot to the vshards
    // it owns makes the union cover each vshard exactly once.
    let assignment = source_assignment(state);

    // The envelope watermark is the consistent cut: every user write
    // committed below it has applied before the snapshots below, and every
    // write committed at or above it refuses a restore of this envelope.
    let snapshot_watermark = super::cut::consistent_cut(state, tenant_id).await?;

    // Every database the tenant has a collection in. Read after the cut, so
    // a collection created before the cut is in the list.
    let databases = super::metadata::tenant_databases(state, tenant_id)?;

    // Each source node snapshots its databases in order. The nodes run
    // concurrently.
    let per_node =
        futures::future::join_all(assignment.into_iter().map(|(node_id, source_vshards)| {
            let databases = &databases;
            async move {
                snapshot_node(
                    state,
                    NodeSnapshot {
                        node_id,
                        tenant_id,
                        snapshot_watermark,
                        source_vshards: &source_vshards,
                        databases,
                    },
                )
                .await
            }
        }))
        .await;

    let meta = EnvelopeMeta {
        tenant_id,
        source_vshard_count: VSHARD_COUNT as u16,
        hash_seed: 0, // VSHARD_COUNT-derived hash; no seed today
        snapshot_watermark,
    };
    let mut writer = EnvelopeWriter::new(meta);

    for node_sections in per_node {
        for (node_id, body) in node_sections? {
            writer
                .push_section(node_id, body)
                .map_err(|e| Error::Internal {
                    detail: format!("backup envelope: {e}"),
                })?;
        }
    }

    // Metadata sections: databases, catalog rows, surrogate binds and
    // source-side tombstones. These live in dedicated sections with sentinel
    // origin_node_ids so the restore path can distinguish them from per-node
    // engine data. Without these, a backup taken during a collection's
    // retention window loses its soft-deleted row (UNDROP can't work after
    // restore), and a restore whose source has already purged a collection
    // can resurrect rows that were properly reaped.
    super::metadata::push_metadata_sections(state, tenant_id, &databases, &mut writer)?;

    // A backup KEK must be configured; plaintext backup envelopes are no
    // longer supported.
    let envelope_bytes = match &state.backup_kek {
        Some(kek) => writer
            .finalize_encrypted(kek)
            .map_err(|e| Error::Internal {
                detail: format!("backup envelope encryption: {e}"),
            })?,
        None => {
            return Err(Error::Internal {
                detail: "backup: no [backup_encryption] KEK configured; \
                         plaintext backup envelopes are no longer supported"
                    .into(),
            });
        }
    };
    Ok(Bytes::from(envelope_bytes))
}

/// One source node's share of a backup.
struct NodeSnapshot<'a> {
    node_id: u64,
    tenant_id: u64,
    snapshot_watermark: u64,
    /// The vShards this node is the assigned source for.
    source_vshards: &'a HashSet<u32>,
    databases: &'a [TenantDatabase],
}

/// Snapshot every database of the tenant on one source node. Returns one
/// `(origin_node_id, section_body)` per database.
///
/// This node took the cut already. A remote node takes the cut at the same
/// watermark before its first snapshot. Its later snapshots follow that cut.
async fn snapshot_node(
    state: &Arc<SharedState>,
    job: NodeSnapshot<'_>,
) -> Result<Vec<(u64, Vec<u8>)>, Error> {
    let local = is_self(state, job.node_id);
    let mut sections = Vec::with_capacity(job.databases.len());
    for (index, database) in job.databases.iter().enumerate() {
        let database_id = database.id();
        let body = if local {
            snapshot_self(state, job.tenant_id, database_id).await?
        } else {
            // A remote node takes the cut before its first snapshot.
            let plan = PhysicalPlan::Meta(MetaOp::CreateTenantSnapshot {
                tenant_id: job.tenant_id,
                cut_watermark: (index == 0).then_some(job.snapshot_watermark),
            });
            snapshot_remote(state, job.node_id, job.tenant_id, database_id, &plan).await?
        };
        // Single-node / single-replica: the node owns every vshard it leads,
        // so the filter retains everything (no-op). Under RF>1: keep only the
        // vshards this node is the assigned source for; the other replicas'
        // copies are dropped here so the restore merge sums disjoint sections.
        let snapshot = filter_node_snapshot(body, job.tenant_id, database_id, job.source_vshards)?;
        let section = DatabaseDataSection {
            database_id: database_id.as_u64(),
            snapshot,
        };
        sections.push((job.node_id, encode_section_part("data section", &section)?));
    }
    Ok(sections)
}

/// Assign every vshard to exactly ONE source node, returning the per-source
/// `(node_id, owned_vshard_set)` pairs to gather from.
///
/// Source of a vshard = the leader of its Raft group; when the group has no
/// elected leader yet (`leader == 0`), fall back deterministically to the
/// lowest-id member so the vshard is still captured exactly once (dropping it
/// would be data loss; assigning it to two nodes would duplicate it). The
/// metadata group (0) owns no vshards, so it never appears.
///
/// In single-node mode (no routing table) returns `[(self.node_id, ALL vshards)]`
/// — the filter is then a no-op and every section is captured once, matching
/// the previous single-section behavior.
fn source_assignment(state: &SharedState) -> Vec<(u64, HashSet<u32>)> {
    let Some(routing) = state.cluster_routing.as_ref() else {
        return vec![(state.node_id, (0..VSHARD_COUNT).collect())];
    };
    let table = routing.read().unwrap_or_else(|p| p.into_inner());

    let mut by_node: BTreeMap<u64, HashSet<u32>> = BTreeMap::new();
    for vshard in 0..VSHARD_COUNT {
        let Ok(group_id) = table.group_for_vshard(vshard) else {
            continue;
        };
        let Some(info) = table.group_info(group_id) else {
            continue;
        };
        let source = if info.leader != 0 {
            info.leader
        } else {
            // No elected leader: deterministic lowest-id member so the vshard
            // is captured by exactly one node.
            match info.members.iter().copied().min() {
                Some(m) => m,
                None => continue,
            }
        };
        by_node.entry(source).or_default().insert(vshard);
    }

    if by_node.is_empty() {
        return vec![(state.node_id, (0..VSHARD_COUNT).collect())];
    }
    by_node.into_iter().collect()
}

/// Decode a gathered per-node `TenantDataSnapshot` of `database_id`, filter
/// it in place to the vshards this node is the assigned source for, and
/// re-encode it.
///
/// The per-section vshard classification is shared with the Raft snapshot SEND
/// builder via `snapshot_keys::retain_tenant_data_for_vshards`. The vshard-of
/// closure routes each stored name in `database_id`, matching the snapshot
/// builder.
fn filter_node_snapshot(
    body: Vec<u8>,
    tenant_id: u64,
    database_id: DatabaseId,
    source_vshards: &HashSet<u32>,
) -> Result<Vec<u8>, Error> {
    let mut snap: crate::types::TenantDataSnapshot =
        zerompk::from_msgpack(&body).map_err(|e| Error::Internal {
            detail: format!("backup: decode per-node snapshot: {e}"),
        })?;
    crate::control::backup::snapshot_keys::retain_tenant_data_for_vshards(
        &mut snap,
        tenant_id,
        source_vshards,
        |collection| super::snapshot_keys::vshard_of_stored(database_id, collection),
    );
    zerompk::to_msgpack_vec(&snap).map_err(|e| Error::Internal {
        detail: format!("backup: re-encode filtered snapshot: {e}"),
    })
}
