// SPDX-License-Identifier: BUSL-1.1

//! Inputs of a KV predicate `UPDATE` / `DELETE` resolve.

use nodedb_physical::physical_plan::ReturningSpec;
use nodedb_types::RlsWriteCheck;

/// Routing and policy inputs shared by both predicate resolves.
pub(in crate::data::executor) struct KvPredicateCtx<'a> {
    pub did: u64,
    pub tid: u64,
    pub collection: &'a str,
    pub filters: &'a [u8],
    pub rls_write_check: &'a RlsWriteCheck,
    /// When `Some`, project one STORED row per matched key per spec instead
    /// of reporting a bare count: the post-image for an update, the
    /// pre-image for a delete.
    pub returning: Option<&'a ReturningSpec>,
    /// Compiled read policy bounding which of those rows may be shown back.
    pub rls_filters: &'a [u8],
}
