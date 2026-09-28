// SPDX-License-Identifier: BUSL-1.1

//! `CoreLoop::execute_vector_multi_search` -- multi-vector-field search with
//! RRF fusion. Extracted from `vector_search_exec.rs` to keep file sizes
//! within the 500-line limit.
//!
//! Not in scope for the in-transaction read-your-own-writes overlay merge
//! (see `handlers::vector_search_exec::execute_vector_search` for that) --
//! `MultiSearch` staging/merge is an explicitly out-of-scope follow-up.

use nodedb_types::StorageKey;
use tracing::debug;

use super::hybrid_key::HybridFusionKey;
use super::vector_search::{
    VectorMultiSearchParams, build_search_hit, effective_ef, encode_hits_response,
};
use crate::bridge::envelope::{ErrorCode, Response};
use crate::data::executor::core_loop::CoreLoop;
use crate::data::executor::response_codec::VectorSearchHit;
use crate::engine::vector::collection::VectorCollection;
use crate::engine::vector::hnsw::SearchResult;
use crate::query::fusion::{RankedResult, reciprocal_rank_fusion};

impl CoreLoop {
    /// Multi-vector search: query all named vector fields in a collection,
    /// fuse results via RRF.
    pub(in crate::data::executor) fn execute_vector_multi_search(
        &self,
        params: VectorMultiSearchParams<'_>,
    ) -> Response {
        let VectorMultiSearchParams {
            task,
            tid,
            collection,
            query_vector,
            top_k,
            ef_search,
            filter_bitmap,
            rls_filters,
        } = params;
        debug!(core = self.core_id, %collection, top_k, "vector multi-search");

        let database_id = task.request.database_id.as_u64();
        let db = nodedb_types::DatabaseId::new(database_id);
        let tenant_id = crate::types::TenantId::new(tid);
        let plain_key = CoreLoop::vector_index_key(database_id, tid, collection, "");
        // A named-field key looks like `"{collection}:{field_name}"` in the String part.
        let field_prefix = format!("{collection}:");

        // Over-fetch when RLS is active so the CP-side post-filter has
        // headroom to still return `top_k` after rejecting candidates.
        let fetch_k = if rls_filters.is_empty() {
            top_k
        } else {
            top_k.saturating_mul(2).max(20)
        };

        // Each searched field's collection with its hits.
        let mut all_results: Vec<(&VectorCollection, Vec<SearchResult>)> = Vec::new();
        // The width of a field index the query could not be compared with.
        // Fields of other widths are skipped; when no field has the query's
        // width, the query is the caller's data error.
        let mut other_width: Option<usize> = None;
        let mut any_field_of_width = false;

        for (key, coll) in &self.vector_collections {
            if key.0 != db || key.1 != tenant_id {
                continue;
            }
            if key == &plain_key || key.2.starts_with(&field_prefix) {
                if coll.dim() != query_vector.len() {
                    other_width = Some(coll.dim());
                    continue;
                }
                any_field_of_width = true;
                if coll.is_empty() {
                    continue;
                }
                let ef = effective_ef(ef_search, fetch_k);
                match super::vector_search::search_vector_leg(
                    coll,
                    query_vector,
                    fetch_k,
                    ef,
                    filter_bitmap,
                ) {
                    Ok(results) => all_results.push((coll, results)),
                    Err(code) => return self.response_error(task, code),
                }
            }
        }

        if all_results.is_empty() {
            if !any_field_of_width && let Some(width) = other_width {
                return self.response_error(
                    task,
                    super::vector::dimension_mismatch(width, query_vector.len()),
                );
            }
            return self.response_error(task, ErrorCode::NotFound);
        }

        let attach = !rls_filters.is_empty();
        let hits: crate::Result<Vec<VectorSearchHit>> =
            if let [(coll, results)] = all_results.as_slice() {
                // Single field: its own ranking, resolved in its own collection.
                results
                    .iter()
                    .take(fetch_k)
                    .map(|r| build_search_hit(Some(*coll), r.id, r.distance))
                    .map(|hit| self.attach_body(database_id, tid, collection, attach, hit))
                    .collect()
            } else {
                // RRF across fields, fused on each row's surrogate. Every
                // field's collection numbers its nodes on its own, so a local
                // id names a row only within the collection that ranked it.
                let ranked_lists: Vec<Vec<RankedResult<FieldFusionKey>>> = all_results
                    .iter()
                    .enumerate()
                    .map(|(field, (coll, results))| {
                        results
                            .iter()
                            .enumerate()
                            .map(|(rank, r)| RankedResult {
                                document_id: FieldFusionKey::of(coll, field, r.id),
                                rank,
                                score: r.distance,
                                source: "vector",
                            })
                            .collect()
                    })
                    .collect();
                reciprocal_rank_fusion(&ranked_lists, None, top_k)
                    .into_iter()
                    .map(|f| {
                        let hit = VectorSearchHit {
                            id: f.document_id.hit_key(),
                            distance: f.rrf_score as f32,
                            doc_id: None,
                            body: None,
                        };
                        self.attach_body(database_id, tid, collection, attach, hit)
                    })
                    .collect()
            };
        let hits = match hits {
            Ok(hits) => hits,
            Err(e) => return self.response_error(task, e),
        };
        if let Some(ref m) = self.metrics {
            m.record_vector_search(0);
            m.record_query_by_engine("vector");
        }
        encode_hits_response(self, task, &hits)
    }
}

/// The key a multi-field search fuses on. A bound hit fuses under its row's
/// storage key, so the same row ranked by two fields fuses into one result.
/// A headless hit has no row identity and fuses with nothing: its key names
/// the field that ranked it and its local id there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum FieldFusionKey {
    Bound(StorageKey),
    Headless { field: usize, local_id: u32 },
}

impl FieldFusionKey {
    fn of(collection: &VectorCollection, field: usize, local_id: u32) -> Self {
        match collection.get_surrogate(local_id) {
            Some(surrogate) => Self::Bound(StorageKey::for_surrogate(surrogate)),
            None => Self::Headless { field, local_id },
        }
    }

    /// The hit id the response carries.
    fn hit_key(self) -> HybridFusionKey {
        match self {
            Self::Bound(key) => HybridFusionKey::Bound(key),
            Self::Headless { local_id, .. } => HybridFusionKey::Headless(local_id),
        }
    }
}
