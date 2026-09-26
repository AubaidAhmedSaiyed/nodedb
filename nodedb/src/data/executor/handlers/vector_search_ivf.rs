// SPDX-License-Identifier: BUSL-1.1

//! IVF-PQ search for `CoreLoop::execute_vector_search`.

use super::vector_search::{build_search_hit, encode_hits_response};
use crate::bridge::envelope::Response;
use crate::data::executor::core_loop::CoreLoop;
use crate::data::executor::task::ExecutionTask;

/// Parameters for [`CoreLoop::search_ivf`].
pub(super) struct SearchIvfParams<'a> {
    pub task: &'a ExecutionTask,
    pub tid: u64,
    pub collection: &'a str,
    pub index_key: &'a (nodedb_types::DatabaseId, crate::types::TenantId, String),
    pub ivf: &'a crate::engine::vector::ivf::IvfPqIndex,
    pub query_vector: &'a [f32],
    pub top_k: usize,
    pub filter_bitmap: Option<&'a nodedb_types::SurrogateBitmap>,
    pub rls_filters: &'a [u8],
}

impl CoreLoop {
    /// Search an IVF-PQ index with optional bitmap post-filtering.
    pub(super) fn search_ivf(&self, params: SearchIvfParams<'_>) -> Response {
        let SearchIvfParams {
            task,
            tid,
            collection,
            index_key,
            ivf,
            query_vector,
            top_k,
            filter_bitmap,
            rls_filters,
        } = params;
        if ivf.dim() != query_vector.len() {
            return self.response_error(
                task,
                super::vector::dimension_mismatch(ivf.dim(), query_vector.len()),
            );
        }
        if ivf.is_empty() {
            return super::vector_search::empty_hits_response(self, task);
        }
        let fetch_k = if filter_bitmap.is_some() || !rls_filters.is_empty() {
            top_k * self.query_tuning.bitmap_over_fetch_factor.max(2)
        } else {
            top_k
        };
        let results = match ivf.search(query_vector, fetch_k) {
            Ok(results) => results,
            Err(e) => return self.response_error(task, crate::Error::from(e)),
        };
        let surrogate_source = self.vector_collections.get(index_key);

        let mut hits: Vec<_> = results
            .iter()
            .map(|r| build_search_hit(surrogate_source, r.id, r.distance))
            .collect();

        if let Some(surrogate_bm) = filter_bitmap {
            // Bitmap is a set of surrogates: keep only bound hits whose
            // surrogate is in the bitmap. A headless hit has none, so it
            // never survives a surrogate-bitmap filter.
            hits.retain(|h| {
                h.id.storage_key()
                    .is_some_and(|key| surrogate_bm.contains(key.surrogate()))
            });
        }
        if !rls_filters.is_empty() {
            // CP-side translator runs the predicate; DP only attaches body.
            hits = hits
                .into_iter()
                .map(|h| {
                    self.attach_body(task.request.database_id.as_u64(), tid, collection, true, h)
                })
                .collect();
        } else {
            hits.truncate(top_k);
        }

        if let Some(ref m) = self.metrics {
            m.record_vector_search(0);
            m.record_query_by_engine("vector");
        }
        encode_hits_response(self, task, &hits)
    }
}
