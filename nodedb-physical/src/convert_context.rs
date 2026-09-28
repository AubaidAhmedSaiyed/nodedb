// SPDX-License-Identifier: Apache-2.0

//! Deployment-neutral context threaded through the shared `SqlPlan →
//! PhysicalPlan` converter helpers in `crate::convert`.
//!
//! Carries only fields both Origin and Lite can supply. Origin-only state
//! (WAL handle, array catalog, credential store, retention registries) lives
//! on Origin's wrapper context and is consumed by Origin-only converter
//! arms (array DDL/DML, timeseries-retention tier-down) that the shared
//! helpers never touch.

use std::sync::Arc;

use nodedb_types::DatabaseId;

use crate::SurrogateAssigner;

/// Inputs every shared converter helper needs.
///
/// Origin and Lite construct this with the same shape; their visitor
/// implementations wrap it (Origin adds catalog/WAL handles, Lite passes
/// it through unchanged).
pub struct SharedConvertContext {
    /// Database scope for vShard computation. Every `CollectionKey` the
    /// converter builds uses this value, so collections in different
    /// databases route to distinct shards.
    pub database_id: DatabaseId,

    /// Per-tenant maximum vector dimension (0 = unlimited). Checked during
    /// `VectorPrimaryInsert` lowering.
    pub max_vector_dim: u32,

    /// `true` when the node is running in cluster mode with a live
    /// topology. Origin's array DML/query converters emit `ClusterArray`
    /// variants when set; single-node Origin and Lite leave this `false`.
    pub cluster_enabled: bool,

    /// CP-side surrogate assigner. Threaded into INSERT/UPSERT/KV-INSERT
    /// helpers to bind `(collection, pk_bytes)` → `Surrogate` before the
    /// op crosses any plane boundary. `None` only for sub-planners that
    /// never lower to surrogate-bearing variants.
    pub surrogate_assigner: Option<Arc<dyn SurrogateAssigner>>,
}
