// SPDX-License-Identifier: BUSL-1.1

//! Data-Plane-facing [`SnapshotBuilder`] implementation for the Raft snapshot
//! SEND path.
//!
//! `nodedb-cluster` defines the [`nodedb_cluster::SnapshotBuilder`] trait but
//! cannot depend on `nodedb` (circular), so the host crate supplies this
//! implementation. The Raft tick loop calls it on the LEADER before framing the
//! chunked `InstallSnapshot` RPC for a lagging/new follower.
//!
//! The build reuses the existing Data-Plane snapshot builder
//! (`MetaOp::CreateTenantSnapshot`) per tenant and per database the tenant has
//! collections in, then FILTERS every section down to the collections whose
//! vshard belongs to the target Raft group, and merges the slices into one
//! `TenantDataSnapshot` for the wire. Every section entry names its database,
//! so the follower installs each row in the database it came from.
//!
//! The vshard-partitioned engines are filtered and shipped, including graph
//! `edges` (the edge key already embeds the collection, so it is routed through
//! the same vshard filter as every other section). CRDT is one Loro doc per
//! (tenant, collection); each `crdt_state` entry carries its single collection
//! and is shipped to the group that owns that collection's vshard — the same
//! per-collection vshard filter as every other section.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;
use std::time::Duration;

use nodedb_types::id::DatabaseId;

use crate::Error;
use crate::control::backup::snapshot_keys::{
    extract_db_scoped_collection, extract_db_tenant_scoped_collection, vshard_of_stored,
};
use crate::control::security::catalog::SystemCatalog;
use crate::control::state::SharedState;
use crate::engine::graph::edge_store::parse_versioned_edge_key;
use crate::types::{SurrogateBindEntry, TenantDataSnapshot, TenantId};

/// Per-tenant snapshot dispatch timeout (mirrors the backup orchestrator).
const TENANT_SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(120);

/// Builds per-group snapshot payloads from the local Data Plane for the Raft
/// snapshot SEND path.
pub struct DataPlaneSnapshotBuilder {
    shared: Arc<SharedState>,
}

impl DataPlaneSnapshotBuilder {
    /// Construct a builder bound to the node's shared state.
    pub fn new(shared: Arc<SharedState>) -> Self {
        Self { shared }
    }

    /// Every tenant with an active collection, and the databases it has
    /// active collections in.
    fn tenant_databases(catalog: &SystemCatalog) -> Result<BTreeMap<u64, BTreeSet<u64>>, Error> {
        let mut tenants: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
        for coll in catalog
            .load_all_collections_across_databases()?
            .iter()
            .filter(|c| c.is_active)
        {
            tenants
                .entry(coll.tenant_id)
                .or_default()
                .insert(coll.database_id.as_u64());
        }
        Ok(tenants)
    }

    /// Capture PK→surrogate bindings for every active collection whose vshard
    /// belongs to the target group, for each enumerated tenant, in every
    /// database.
    ///
    /// Routes each collection by its `(database, bare name)` key, the same
    /// key every section's filter routes by, so only in-group collections'
    /// identities ship — never more, never less than the data sections carry.
    fn capture_surrogates(
        catalog: &SystemCatalog,
        tenants: &BTreeMap<u64, BTreeSet<u64>>,
        group_vshards: &HashSet<u32>,
        merged: &mut TenantDataSnapshot,
    ) -> Result<(), Error> {
        let collections = catalog.load_all_collections_across_databases()?;
        for coll in collections
            .iter()
            .filter(|c| c.is_active && tenants.contains_key(&c.tenant_id))
        {
            let key = nodedb_types::CollectionKey::from_bare(coll.database_id, &coll.name);
            if !group_vshards.contains(&nodedb_cluster::routing::vshard_for_collection(key)) {
                continue;
            }
            let bindings =
                catalog.scan_surrogates_for_collection(key, TenantId::new(coll.tenant_id))?;
            for (pk, surrogate) in bindings {
                merged.surrogate_pk.push(SurrogateBindEntry {
                    database_id: coll.database_id.as_u64(),
                    tenant_id: coll.tenant_id,
                    collection: coll.name.clone(),
                    pk,
                    surrogate: surrogate.as_u32(),
                });
            }
        }
        Ok(())
    }

    /// Build the group-filtered snapshot of `tenant_id` in `database_id` and
    /// merge it into `merged`.
    async fn build_tenant_filtered(
        &self,
        tenant_id: u64,
        database_id: DatabaseId,
        group_vshards: &HashSet<u32>,
        merged: &mut TenantDataSnapshot,
    ) -> Result<(), Error> {
        let bytes = crate::control::server::exchange::snapshot_tenant_on_local_cores(
            &self.shared,
            TenantId::new(tenant_id),
            database_id,
            TENANT_SNAPSHOT_TIMEOUT,
        )
        .await?;

        let snap: TenantDataSnapshot =
            zerompk::from_msgpack(&bytes).map_err(|e| Error::Internal {
                detail: format!(
                    "snapshot build: decode tenant {tenant_id} snapshot of database {}: {e}",
                    database_id.as_u64()
                ),
            })?;
        // Every section names its collection as the Data Plane stores it in
        // `database_id`.
        let in_group =
            |stored: &str| group_vshards.contains(&vshard_of_stored(database_id, stored));

        // db-tenant-scoped sections: key shape "{db}:{tid}:{collection}[:suffix]"
        let in_group_db_tenant_scoped =
            |key: &str| extract_db_tenant_scoped_collection(key, tenant_id).is_some_and(in_group);
        // db-scoped sections: key shape "{db}:{tid}:{collection}" (coll may contain ':')
        let in_group_db_scoped =
            |key: &str| extract_db_scoped_collection(key, tenant_id).is_some_and(in_group);

        for (k, v) in snap.documents {
            if in_group_db_tenant_scoped(&k) {
                merged.documents.push((k, v));
            }
        }
        for (k, v) in snap.indexes {
            if in_group_db_tenant_scoped(&k) {
                merged.indexes.push((k, v));
            }
        }
        for (k, v) in snap.documents_versioned {
            if in_group_db_tenant_scoped(&k) {
                merged.documents_versioned.push((k, v));
            }
        }
        for (k, v) in snap.indexes_versioned {
            if in_group_db_tenant_scoped(&k) {
                merged.indexes_versioned.push((k, v));
            }
        }
        for (k, v) in snap.vectors {
            if in_group_db_tenant_scoped(&k) {
                merged.vectors.push((k, v));
            }
        }
        for (k, v) in snap.timeseries {
            if in_group_db_tenant_scoped(&k) {
                merged.timeseries.push((k, v));
            }
        }
        // kv_tables / flushed_ts_segments / columnar_engines: db-scoped keys.
        for (k, v) in snap.kv_tables {
            if in_group_db_scoped(&k) {
                merged.kv_tables.push((k, v));
            }
        }
        for blob in snap.flushed_ts_segments {
            if in_group_db_scoped(&blob.collection_key) {
                merged.flushed_ts_segments.push(blob);
            }
        }
        for (k, v) in snap.columnar_engines {
            if in_group_db_scoped(&k) {
                merged.columnar_engines.push((k, v));
            }
        }
        for (k, v) in snap.vector_params {
            if in_group_db_tenant_scoped(&k) {
                merged.vector_params.push((k, v));
            }
        }
        for (k, v) in snap.index_configs {
            if in_group_db_tenant_scoped(&k) {
                merged.index_configs.push((k, v));
            }
        }

        // Graph edges: the versioned edge key embeds the collection as its
        // FIRST `\x00`-delimited component, and edge writes are homed at
        // the vshard of the collection's `(database, bare name)` key — the SAME
        // routing every other section's filter uses. The restore path parses
        // the key and rebuilds CSR, so no key transformation is needed here.
        //
        // Unlike every other section, the edge key carries neither the
        // database nor the tenant, so the merged snapshot (applied ONCE with no
        // per-database or per-tenant dispatch) carries edges via
        // `tenant_edges` — pushing to the plain `edges` field here would
        // install them under the wrong database and tenant on apply.
        for (key, value) in snap.edges {
            match parse_versioned_edge_key(&key) {
                Some((collection, ..)) => {
                    if in_group(collection) {
                        merged
                            .tenant_edges
                            .push((database_id.as_u64(), tenant_id, key, value));
                    }
                }
                None => {
                    // All edge keys are the versioned format; an unparseable
                    // key has no determinable group, and restore would reject
                    // it via `put_edge_raw`. Do NOT silently drop it — surface
                    // it. Log only a short prefix, never the full key.
                    let key_prefix: String = key.chars().take(32).collect();
                    tracing::warn!(key_prefix, "snapshot build: unparseable edge key, skipping");
                }
            }
        }

        // CRDT: one Loro doc per (tenant, collection). Each entry carries its
        // single collection; include it iff that collection's vshard belongs to
        // this group — the same per-collection vshard filter every other engine
        // uses.
        for (crdt_db, tid, collection, bytes) in snap.crdt_state {
            if in_group(collection.as_str()) {
                merged.crdt_state.push((crdt_db, tid, collection, bytes));
            }
        }

        // CRDT constraints: same per-collection vshard filter as `crdt_state`
        // — each entry is routed by its single collection's vshard.
        for entry in snap.crdt_constraints {
            if in_group(entry.collection.as_str()) {
                merged.crdt_constraints.push(entry);
            }
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl nodedb_cluster::SnapshotBuilder for DataPlaneSnapshotBuilder {
    async fn build_group_snapshot(
        &self,
        group_id: u64,
        _last_included_index: u64,
        _last_included_term: u64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        // Resolve the group's vshards. Single-node (no routing) or an
        // empty/ownerless group → nothing to ship; the sender falls back to the
        // stub chunk.
        let group_vshards: HashSet<u32> = match self.shared.cluster_routing.as_ref() {
            Some(routing) => {
                let table = routing.read().map_err(|_| {
                    Box::new(Error::Internal {
                        detail: "snapshot build: cluster_routing RwLock poisoned".into(),
                    }) as Box<dyn std::error::Error + Send + Sync>
                })?;
                table.vshards_for_group(group_id).into_iter().collect()
            }
            None => return Ok(Vec::new()),
        };
        if group_vshards.is_empty() {
            return Ok(Vec::new());
        }

        // Enumerate tenants and their databases from the system catalog — the
        // same source the backup orchestrator's catalog sections use. Every
        // active collection carries its `tenant_id` and `database_id`; each
        // distinct pair is one snapshot to take. With no collection there is
        // nothing durable to enumerate, so ship an empty (well-formed) snapshot.
        let tenants = Self::tenant_databases(self.shared.credentials.catalog())
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        let mut merged = TenantDataSnapshot::default();
        for (tenant_id, databases) in &tenants {
            for database_id in databases {
                self.build_tenant_filtered(
                    *tenant_id,
                    DatabaseId::new(*database_id),
                    &group_vshards,
                    &mut merged,
                )
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
            }
        }

        // Capture the PK→surrogate identity map for every in-group collection.
        // The surrogate map is DATA-derived and travels with the data-group
        // snapshot (not the metadata group): without it a snapshot-installed
        // follower has documents but cannot resolve PK point-lookups. The
        // catalog is Control-Plane state (the Data-Plane snapshot handler can't
        // see it), so it is captured here and rebound on the apply side.
        {
            let catalog = self.shared.credentials.catalog();
            Self::capture_surrogates(catalog, &tenants, &group_vshards, &mut merged)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
        }

        // The group's tenant write marks travel with its data: the follower
        // that installs the snapshot never applies the entries it covers.
        merged.group_write_marks = self.shared.tenant_marks.group_entries(group_id);

        // Always return a well-formed serialized struct (even when empty) so the
        // follower-apply unit receives a decodable payload rather than a stub.
        let out = zerompk::to_msgpack_vec(&merged).map_err(|e| {
            Box::new(Error::Internal {
                detail: format!("snapshot build: encode merged group {group_id} snapshot: {e}"),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;
        Ok(out)
    }
}
