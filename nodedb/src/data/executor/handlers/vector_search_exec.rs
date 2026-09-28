// SPDX-License-Identifier: BUSL-1.1

//! CoreLoop methods for vector search execution.

use roaring::RoaringBitmap;
use tracing::{debug, warn};

use super::vector_search::{
    VectorSearchParams, build_search_hit, effective_ef, encode_hits_response,
    surrogate_bitmap_to_global_ids,
};
use super::vector_search_ann::{ResolvedAnnOptions, apply_ann_options, quantization_matches};
use crate::bridge::envelope::{ErrorCode, Response};
use crate::data::executor::core_loop::CoreLoop;
use crate::data::executor::task::ExecutionTask;

impl CoreLoop {
    /// Fetch the document body via the sparse engine (keyed by
    /// surrogate-hex) and attach it to the hit. Used both by the RLS path
    /// (the Control Plane evaluates the predicate against `body`) and by
    /// the slow-path SELECT (the Control Plane response translator flattens
    /// the body's fields into the hit JSON so payload columns surface to
    /// the client). When `attach == false`, or the hit carries no surrogate
    /// binding, the hit is returned unchanged. A storage read error fails
    /// the call: a hit without its body would skip the RLS predicate check.
    ///
    /// The bytes are normalized to a standard msgpack map through the shared
    /// sparse-body normalizer, resolved from the collection's registered kind.
    /// A vector-primary collection's rows are `zerompk` TAGGED sidecars and a
    /// classic collection's are ordinary document bodies; the two are
    /// indistinguishable byte-wise, and both the RLS filter evaluator and the
    /// Control-Plane flatten read the attached body as one shape.
    #[inline]
    pub(in crate::data::executor) fn attach_body(
        &self,
        database_id: u64,
        tid: u64,
        collection: &str,
        attach: bool,
        mut hit: super::super::response_codec::VectorSearchHit,
    ) -> crate::Result<super::super::response_codec::VectorSearchHit> {
        if !attach {
            return Ok(hit);
        }
        let Some(key) = hit.id.storage_key() else {
            return Ok(hit);
        };
        if let Some(bytes) = self.sparse.get(database_id, tid, collection, &key)? {
            let format = self.sparse_body_format(
                crate::types::DatabaseId::new(database_id),
                crate::types::TenantId::new(tid),
                collection,
            );
            // Owned: the hit outlives `bytes`, which is the storage read's
            // local buffer.
            hit.body = Some(
                crate::data::executor::scan_normalize::sparse_body_to_msgpack(
                    &bytes,
                    format.as_format_ref(),
                )
                .into_owned(),
            );
        }
        Ok(hit)
    }

    pub(in crate::data::executor) fn execute_vector_search(
        &mut self,
        params: VectorSearchParams<'_>,
    ) -> Response {
        let VectorSearchParams {
            task,
            tid,
            collection,
            query_vector,
            top_k,
            ef_search,
            metric,
            filter_bitmap,
            field_name,
            rls_filters,
            inline_prefilter_plan,
            ann_options,
            skip_payload_fetch,
            payload_filters,
        } = params;
        // RLS requires body fetch regardless of projection. If RLS filters are
        // active, ignore the skip flag and record why at debug level.
        let skip_payload_fetch = if skip_payload_fetch && !rls_filters.is_empty() {
            debug!(
                core = self.core_id,
                %collection,
                reason = "rls",
                "skip_payload_fetch suppressed: RLS filters present"
            );
            false
        } else {
            skip_payload_fetch
        };

        let ResolvedAnnOptions {
            ef_search,
            oversample,
        } = apply_ann_options(self.core_id, collection, ef_search, ann_options);

        // Materialize cross-engine prefilter sub-plan (e.g. ARRAY_SLICE
        // → surrogate bitmap) and intersect with any pre-existing
        // `filter_bitmap`. The sub-plan emits document-shaped rows whose
        // `id` is the cell's surrogate as 8-char zero-padded lowercase
        // hex; `collect_surrogates` decodes that back into surrogate IDs.
        let inline_bitmap = inline_prefilter_plan.map(|sub_plan| {
            crate::data::executor::dispatch::bitmap::hashjoin_inline::run_bitmap_subplan(
                self, task, sub_plan,
            )
        });
        let effective_filter: Option<nodedb_types::SurrogateBitmap> =
            match (filter_bitmap.cloned(), inline_bitmap) {
                (Some(a), Some(b)) => Some(a.intersect(&b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) if !b.is_empty() => Some(b),
                _ => None,
            };
        let filter_bitmap = effective_filter.as_ref();
        debug!(core = self.core_id, %collection, top_k, ef_search, "vector search");

        // Scan-quiesce gate.
        let _scan_guard = match self.acquire_scan_guard(task, tid, collection) {
            Ok(g) => g,
            Err(resp) => return resp,
        };

        let database_id = task.request.database_id.as_u64();
        let index_key = CoreLoop::vector_index_key(database_id, tid, collection, field_name);

        // Every index type is one `VectorCollection`: an IVF-PQ collection
        // answers from its exact buffer or its trained IVF-PQ index.
        // If the specific field-named index does not exist, fall back to the
        // empty-field index. This handles data synced from NodeDB-Lite (which
        // uses collection-level storage, not named-field storage) being
        // searched via a field-specific SQL query (e.g. vector_distance(embedding, ...)).
        let effective_key =
            if !self.vector_collections.contains_key(&index_key) && !field_name.is_empty() {
                let fallback_key = CoreLoop::vector_index_key(database_id, tid, collection, "");
                if self.vector_collections.contains_key(&fallback_key) {
                    fallback_key
                } else {
                    index_key
                }
            } else {
                index_key
            };
        // An index with no base rows still answers a transaction's own staged
        // rows: the merge ranks them over an empty base result.
        let staged_only = |core: &Self, txn_id| {
            core.search_overlay_only(
                task,
                super::transaction::overlay::VectorMergeParams {
                    txn_id,
                    database_id: task.request.database_id,
                    tid: crate::types::TenantId::new(tid),
                    collection,
                    field_name,
                    query_vector,
                    metric,
                    top_k,
                    filter_bitmap,
                    payload_filters,
                },
            )
        };
        let Some(collection_ref) = self.vector_collections.get(&effective_key) else {
            if let Some(txn_id) = task.request.txn_id {
                return staged_only(self, txn_id);
            }
            return self.response_error(task, ErrorCode::NotFound);
        };
        // The index width is fixed even while it holds no vector, so a
        // query of another width fails here too.
        if collection_ref.dim() != query_vector.len() {
            return self.response_error(
                task,
                super::vector::dimension_mismatch(collection_ref.dim(), query_vector.len()),
            );
        }
        if collection_ref.is_empty() {
            if let Some(txn_id) = task.request.txn_id {
                return staged_only(self, txn_id);
            }
            return super::vector_search::empty_hits_response(self, task);
        }

        // Quantization mismatch: if the SQL caller requested a specific
        // quantization that differs from what the index actually uses, warn
        // once per query and proceed with the collection's actual quantization.
        // Per-collection codec dispatch will honor the hint when that lands.
        if let Some(requested_q) = ann_options.quantization {
            let index_q = collection_ref.stats().quantization;
            if !quantization_matches(requested_q, index_q) {
                warn!(
                    core = self.core_id,
                    %collection,
                    requested = ?requested_q,
                    actual = %index_q,
                    "ann_options: quantization hint does not match index; proceeding with index quantization"
                );
            }
        }

        // Over-fetch to accommodate both oversample breadth (for re-rank
        // headroom) and RLS post-filter headroom. The two factors are
        // multiplied so each can independently request more candidates.
        let fetch_k = if rls_filters.is_empty() {
            top_k.saturating_mul(oversample)
        } else {
            top_k.saturating_mul(2).saturating_mul(oversample).max(20)
        };
        let ef = effective_ef(ef_search, fetch_k);

        // Derive payload bitmap (node-id space) from `(field, value)`
        // equalities by intersecting per-field equality bitmaps. Returns
        // `None` when no payload filters were requested or any filter
        // references a field with no registered payload index.
        let payload_bm: Option<RoaringBitmap> = if payload_filters.is_empty() {
            None
        } else {
            let preds: Vec<nodedb_vector::collection::FilterPredicate> = payload_filters
                .iter()
                .map(|atom| match atom {
                    nodedb_types::PayloadAtom::Eq(f, v) => {
                        nodedb_vector::collection::FilterPredicate::Eq {
                            field: f.to_ascii_lowercase(),
                            value: v.clone(),
                        }
                    }
                    nodedb_types::PayloadAtom::In(f, vs) => {
                        nodedb_vector::collection::FilterPredicate::In {
                            field: f.to_ascii_lowercase(),
                            values: vs.clone(),
                        }
                    }
                    nodedb_types::PayloadAtom::Range {
                        field,
                        low,
                        low_inclusive,
                        high,
                        high_inclusive,
                    } => nodedb_vector::collection::FilterPredicate::Range {
                        field: field.to_ascii_lowercase(),
                        low: low.clone(),
                        low_inclusive: *low_inclusive,
                        high: high.clone(),
                        high_inclusive: *high_inclusive,
                    },
                    _ => nodedb_vector::collection::FilterPredicate::And(vec![]),
                })
                .collect();
            let conj = nodedb_vector::collection::FilterPredicate::And(preds);
            collection_ref.payload.pre_filter(&conj)
        };

        let combined_bm: Option<RoaringBitmap> = match (filter_bitmap, payload_bm) {
            (Some(surrogate_bm), Some(pbm)) => {
                let mut bm = surrogate_bitmap_to_global_ids(collection_ref, surrogate_bm);
                bm &= pbm;
                Some(bm)
            }
            (Some(surrogate_bm), None) => {
                Some(surrogate_bitmap_to_global_ids(collection_ref, surrogate_bm))
            }
            (None, Some(pbm)) => Some(pbm),
            (None, None) => None,
        };

        // A filter that cannot be serialized fails the search: searching
        // without it would return rows the filter excludes.
        let searched = match combined_bm {
            Some(local_bm) => {
                let mut buf = Vec::with_capacity(local_bm.serialized_size());
                if let Err(e) = local_bm.serialize_into(&mut buf) {
                    return self.response_error(
                        task,
                        ErrorCode::Internal {
                            detail: format!("vector search filter bitmap serialization: {e}"),
                        },
                    );
                }
                collection_ref.search_with_bitmap_bytes_and_metric(
                    query_vector,
                    fetch_k,
                    ef,
                    &buf,
                    metric,
                )
            }
            None => collection_ref.search_with_metric(query_vector, fetch_k, ef, metric),
        };
        // A query of the wrong dimension is the caller's data error (22000);
        // the core keeps serving.
        let results = match searched {
            Ok(results) => results,
            Err(e) => return self.response_error(task, crate::Error::from(e)),
        };

        // Pure-vector fast path: projection contains only id/distance.
        // Skip the sparse-store body fetch entirely.
        if skip_payload_fetch {
            let mut hits: Vec<_> = results
                .iter()
                .map(|r| build_search_hit(Some(collection_ref), r.id, r.distance))
                .collect();
            // Read-your-own-writes for vector search: fold this
            // transaction's staged vector inserts into the base HNSW/IVF
            // result before truncation, so a vector inserted earlier in the
            // same transaction is ranked in by true distance before COMMIT.
            if let Some(txn_id) = task.request.txn_id {
                if let Err(e) = self.merge_vector_overlay_into_search(
                    super::transaction::overlay::VectorMergeParams {
                        txn_id,
                        database_id: task.request.database_id,
                        tid: crate::types::TenantId::new(tid),
                        collection,
                        field_name,
                        query_vector,
                        metric,
                        top_k,
                        filter_bitmap,
                        payload_filters,
                    },
                    &mut hits,
                ) {
                    return self.response_error(task, e);
                }
            } else {
                hits.truncate(top_k);
            }
            if let Some(ref m) = self.metrics {
                m.record_vector_search(0);
                m.record_query_by_engine("vector");
            }
            return encode_hits_response(self, task, &hits);
        }

        // RLS evaluation lives at the Control-Plane response boundary
        // (`response_translate::vector`). DP attaches the document body
        // when filters are active so CP can run the predicate without
        // a follow-up round-trip; CP applies the filter and truncates to
        // `top_k`. Data Plane stays pure SIMD + sparse-fetch.
        // Attach body bytes whenever skip_payload_fetch is false (slow path)
        // OR when RLS filters need them; the CP response translator flattens
        // the bytes' fields into the hit JSON for client column projection.
        let attach = !skip_payload_fetch || !rls_filters.is_empty();
        let hits: crate::Result<Vec<_>> = results
            .iter()
            .map(|r| build_search_hit(Some(collection_ref), r.id, r.distance))
            .map(|hit| {
                self.attach_body(
                    task.request.database_id.as_u64(),
                    tid,
                    collection,
                    attach,
                    hit,
                )
            })
            .collect();
        let mut hits = match hits {
            Ok(hits) => hits,
            Err(e) => return self.response_error(task, e),
        };
        let truncate_to = if rls_filters.is_empty() {
            top_k
        } else {
            fetch_k
        };
        // Read-your-own-writes for vector search: fold this transaction's
        // staged vector inserts into the base HNSW/IVF result before
        // truncation, so a vector inserted earlier in the same transaction
        // is ranked in by true distance before COMMIT.
        if let Some(txn_id) = task.request.txn_id {
            if let Err(e) = self.merge_vector_overlay_into_search(
                super::transaction::overlay::VectorMergeParams {
                    txn_id,
                    database_id: task.request.database_id,
                    tid: crate::types::TenantId::new(tid),
                    collection,
                    field_name,
                    query_vector,
                    metric,
                    top_k: truncate_to,
                    filter_bitmap,
                    payload_filters,
                },
                &mut hits,
            ) {
                return self.response_error(task, e);
            }
        } else {
            hits.truncate(truncate_to);
        }
        if let Some(ref m) = self.metrics {
            m.record_vector_search(0);
            m.record_query_by_engine("vector");
        }
        encode_hits_response(self, task, &hits)
    }

    /// Answer a search from the transaction's staged rows alone, for a
    /// collection whose index holds no base row yet.
    fn search_overlay_only(
        &self,
        task: &ExecutionTask,
        params: super::transaction::overlay::VectorMergeParams<'_>,
    ) -> Response {
        let mut hits: Vec<super::super::response_codec::VectorSearchHit> = Vec::new();
        if let Err(e) = self.merge_vector_overlay_into_search(params, &mut hits) {
            return self.response_error(task, e);
        }
        if let Some(ref m) = self.metrics {
            m.record_vector_search(0);
            m.record_query_by_engine("vector");
        }
        encode_hits_response(self, task, &hits)
    }
}
