// SPDX-License-Identifier: BUSL-1.1

//! `CoreLoop::execute_vector_multi_search` -- multi-vector-field search with
//! RRF fusion. Extracted from `vector_search_exec.rs` to keep file sizes
//! within the 500-line limit.
//!
//! Not in scope for the in-transaction read-your-own-writes overlay merge
//! (see `handlers::vector_search_exec::execute_vector_search` for that) --
//! `MultiSearch` staging/merge is an explicitly out-of-scope follow-up.

use tracing::debug;

use super::vector_search::{
    VectorMultiSearchParams, build_search_hit, effective_ef, encode_hits_response,
};
use crate::bridge::envelope::{ErrorCode, Response};
use crate::data::executor::core_loop::CoreLoop;

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

        let mut all_results: Vec<Vec<crate::engine::vector::hnsw::SearchResult>> = Vec::new();
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
                    Ok(results) => all_results.push(results),
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

        // Single field — return directly.
        if all_results.len() == 1 {
            let Some(results) = all_results.into_iter().next() else {
                return self.response_error(task, ErrorCode::NotFound);
            };
            let doc_source = self.vector_collections.get(&plain_key);
            let hits: Vec<_> = results
                .iter()
                .map(|r| build_search_hit(doc_source, r.id, r.distance))
                .map(|hit| {
                    self.attach_body(
                        task.request.database_id.as_u64(),
                        tid,
                        collection,
                        !rls_filters.is_empty(),
                        hit,
                    )
                })
                .take(fetch_k)
                .collect();
            if let Some(ref m) = self.metrics {
                m.record_vector_search(0);
                m.record_query_by_engine("vector");
            }
            return encode_hits_response(self, task, &hits);
        }

        // RRF fusion across fields using shared fusion module.
        use crate::query::fusion::{RankedResult, reciprocal_rank_fusion};

        let ranked_lists: Vec<Vec<RankedResult>> = all_results
            .iter()
            .map(|results| {
                results
                    .iter()
                    .enumerate()
                    .map(|(rank, r)| RankedResult {
                        document_id: r.id.to_string(),
                        rank,
                        score: r.distance,
                        source: "vector",
                    })
                    .collect()
            })
            .collect();

        let fused = reciprocal_rank_fusion(&ranked_lists, None, top_k);

        // Surface fused results with surrogate-as-id; CP fills doc_id and
        // applies the RLS predicate at the response boundary.
        let hits: Vec<_> = fused
            .iter()
            .filter_map(|f| {
                let local_id: u32 = f.document_id.parse().ok()?;
                let source = self.vector_collections.get(&plain_key).or_else(|| {
                    self.vector_collections
                        .iter()
                        .filter(|(k, _)| {
                            k.0 == db
                                && k.1 == tenant_id
                                && (k == &&plain_key || k.2.starts_with(&field_prefix))
                        })
                        .map(|(_, c)| c)
                        .next()
                });
                let hit = build_search_hit(source, local_id, f.rrf_score as f32);
                Some(self.attach_body(
                    task.request.database_id.as_u64(),
                    tid,
                    collection,
                    !rls_filters.is_empty(),
                    hit,
                ))
            })
            .collect();
        if let Some(ref m) = self.metrics {
            m.record_vector_search(0);
            m.record_query_by_engine("vector");
        }
        encode_hits_response(self, task, &hits)
    }
}
