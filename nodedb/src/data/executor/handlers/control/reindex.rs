// SPDX-License-Identifier: BUSL-1.1

//! Concurrent index rebuild (REINDEX CONCURRENTLY) for HNSW, FTS LSM, and graph CSR.
//!
//! Design: the Data Plane dispatches a `RebuildIndex` op. HNSW segments are
//! rebuilt on the core's HNSW builder thread, the same path every graph build
//! takes (`handlers::vector_build`). For FTS and CSR a background OS thread
//! performs the heavy build work while the owning core continues to serve
//! reads from the live index.  On each subsequent tick
//! the core polls `pending_reindex` for completion via `try_recv`; when the
//! build succeeds the core performs an in-memory swap and returns the ACK.
//!
//! The non-concurrent path runs the rebuild inline, compacting tombstones and
//! write buffers without moving data off-core.
//!
//! Plane rules: the background thread is a plain OS thread (not a tokio task).
//! It receives plain `Send` data, builds in isolation, and sends serialized
//! bytes back.  The `!Send` engine state is touched only on the Data Plane
//! thread.
//!
//! Background-thread rebuild functions and Data-Plane cutover appliers live in
//! the sibling `reindex_apply` module.

use std::sync::mpsc;

use tracing::{error, info, warn};

use super::reindex_apply::{
    FtsRebuild, RebuildOutput, apply_csr, apply_fts, rebuild_csr_thread, rebuild_fts_thread,
};
use crate::bridge::envelope::{ErrorCode, Response};
use crate::data::executor::core_loop::CoreLoop;
use crate::data::executor::task::ExecutionTask;
use crate::types::TenantId;

// ── PendingReindex ────────────────────────────────────────────────────────────

/// An in-flight concurrent rebuild tracked on the `CoreLoop`.
pub struct PendingReindex {
    pub database_id: nodedb_types::DatabaseId,
    pub tenant_id: TenantId,
    pub collection_key: String,
    rx: mpsc::Receiver<crate::Result<RebuildOutput>>,
}

// ── CoreLoop integration ──────────────────────────────────────────────────────

impl CoreLoop {
    /// Handle a `MetaOp::RebuildIndex` dispatch.
    pub(in crate::data::executor) fn execute_rebuild_index(
        &mut self,
        task: &ExecutionTask,
        tid: u64,
        collection: &str,
        index_name: Option<&str>,
        concurrent: bool,
    ) -> Response {
        let tenant_id = TenantId::new(tid);
        let collection_key = collection.to_string();

        if !concurrent {
            return self.rebuild_index_inline(task, tenant_id, &collection_key);
        }

        // Reject duplicate concurrent rebuild for same collection.
        if self
            .maintenance
            .pending_reindex
            .iter()
            .any(|p| p.tenant_id == tenant_id && p.collection_key == collection_key)
        {
            return self.response_error(
                task,
                ErrorCode::Internal {
                    detail: format!("rebuild already in progress for collection \"{collection}\""),
                },
            );
        }

        let rebuild_hnsw = index_name
            .map(|n| n.eq_ignore_ascii_case("hnsw"))
            .unwrap_or(true);
        let rebuild_fts = index_name
            .map(|n| n.eq_ignore_ascii_case("fts"))
            .unwrap_or(true);
        let rebuild_csr = index_name
            .map(|n| n.eq_ignore_ascii_case("csr"))
            .unwrap_or(true);

        // Start the first applicable rebuild (priority: HNSW > FTS > CSR).
        let start_result = if rebuild_hnsw {
            self.start_hnsw_rebuild(task, tenant_id, &collection_key);
            Ok(())
        } else if rebuild_fts {
            self.start_fts_rebuild(task, tenant_id, &collection_key)
        } else if rebuild_csr {
            self.start_csr_rebuild(task, tenant_id, &collection_key)
        } else {
            Ok(())
        };
        if let Err(e) = start_result {
            return self.response_error(task, e);
        }

        self.response_ok(task)
    }

    /// Poll all in-flight concurrent rebuilds.  Called from `tick()`.
    pub fn poll_pending_reindex(&mut self) {
        // Collect completed and failed entries, leaving only still-running ones.
        // We must separate the poll loop from the apply loop to satisfy the borrow checker:
        // apply_* functions take &mut self, which conflicts with holding a reference into
        // self.maintenance.pending_reindex at the same time.
        enum Outcome {
            Done {
                database_id: nodedb_types::DatabaseId,
                tenant_id: nodedb_types::TenantId,
                collection_key: String,
                output: RebuildOutput,
            },
            Failed {
                collection_key: String,
                error: String,
            },
        }

        let mut outcomes: Vec<Outcome> = Vec::new();
        let mut still_running: Vec<PendingReindex> = Vec::new();

        for pending in self.maintenance.pending_reindex.drain(..) {
            match pending.rx.try_recv() {
                Ok(Ok(output)) => outcomes.push(Outcome::Done {
                    database_id: pending.database_id,
                    tenant_id: pending.tenant_id,
                    collection_key: pending.collection_key,
                    output,
                }),
                Ok(Err(e)) => outcomes.push(Outcome::Failed {
                    collection_key: pending.collection_key,
                    error: e.to_string(),
                }),
                Err(mpsc::TryRecvError::Disconnected) => outcomes.push(Outcome::Failed {
                    collection_key: pending.collection_key,
                    error: "rebuild thread disconnected".to_owned(),
                }),
                Err(mpsc::TryRecvError::Empty) => still_running.push(pending),
            }
        }
        self.maintenance.pending_reindex = still_running;

        for outcome in outcomes {
            match outcome {
                Outcome::Done {
                    database_id,
                    tenant_id,
                    collection_key,
                    output,
                } => {
                    match output {
                        RebuildOutput::Csr { bytes } => {
                            apply_csr(self, &database_id, &tenant_id, &collection_key, bytes);
                        }
                        RebuildOutput::Fts(rebuild) => {
                            apply_fts(self, &database_id, &tenant_id, &collection_key, rebuild);
                        }
                    }
                    info!(
                        core = self.core_id,
                        collection = %collection_key,
                        "concurrent index rebuild cutover complete"
                    );
                }
                Outcome::Failed {
                    collection_key,
                    error,
                } => {
                    error!(
                        core = self.core_id,
                        collection = %collection_key,
                        error = %error,
                        "concurrent index rebuild failed; live index unchanged"
                    );
                }
            }
        }
    }

    // ── Inline (non-concurrent) rebuild ──────────────────────────────────────

    fn rebuild_index_inline(
        &mut self,
        task: &ExecutionTask,
        tenant_id: TenantId,
        collection_key: &str,
    ) -> Response {
        // HNSW: compact tombstones from sealed segments.
        let db = task.request.database_id;
        if let Some(coll) =
            self.vector_collections
                .get_mut(&(db, tenant_id, collection_key.to_string()))
        {
            let removed = coll.compact_tombstones();
            info!(
                core = self.core_id,
                collection = %collection_key,
                removed,
                "inline HNSW tombstone compaction"
            );
        }

        // CSR: compact write buffers into dense arrays.
        if let Err(e) = self.csr.compact_all() {
            warn!(
                core = self.core_id,
                error = %e,
                "inline CSR compact failed (budget); continuing"
            );
        }

        self.response_ok(task)
    }

    // ── Background-thread starters ────────────────────────────────────────────

    /// Queue a rebuild of every sealed HNSW segment of the collection's
    /// vector indexes on this core's builder thread. Each segment is rebuilt
    /// from its own vectors with its node ids kept, quantized again under the
    /// collection's config, and swapped in on this core; search reads the old
    /// graph until then. The growing and building segments are left alone.
    fn start_hnsw_rebuild(
        &mut self,
        task: &ExecutionTask,
        tenant_id: TenantId,
        collection_key: &str,
    ) {
        // Vector collections are stored under two key forms depending on how
        // they were inserted:
        //   - Bare:             (db, tenant, "coll")             — BatchInsert / native
        //   - Field-qualified:  (db, tenant, "coll:field_name")  — SQL INSERT, DirectUpsert
        let db = task.request.database_id;
        let field_prefix = format!("{collection_key}:");
        let matching_keys: Vec<(nodedb_types::DatabaseId, TenantId, String)> = self
            .vector_collections
            .iter()
            .filter(|((d, t, k), coll)| {
                *d == db
                    && *t == tenant_id
                    && (k.as_str() == collection_key || k.starts_with(&field_prefix))
                    // An IVF-PQ collection keeps no HNSW segments to rebuild.
                    && !coll.is_ivf()
            })
            .map(|(key, _)| key.clone())
            .collect();
        for key in matching_keys {
            let queued = self.queue_vector_rebuild(&key);
            info!(
                core = self.core_id,
                collection = %key.2,
                queued,
                "HNSW rebuild queued"
            );
        }
    }

    fn start_fts_rebuild(
        &mut self,
        task: &ExecutionTask,
        tenant_id: TenantId,
        collection_key: &str,
    ) -> crate::Result<()> {
        use nodedb_fts::backend::FtsBackend;

        let database_id = task.request.database_id;
        let db_u64 = database_id.as_u64();
        let tid = tenant_id.as_u64();
        let backend = self.inverted.backend();

        let terms = backend
            .collection_terms(db_u64, tid, collection_key)
            .map_err(|e| crate::Error::Storage {
                engine: "fts".to_string(),
                detail: format!("FTS terms: {e}"),
            })?;

        let mut postings: Vec<(String, Vec<nodedb_fts::posting::Posting>)> =
            Vec::with_capacity(terms.len());
        for term in &terms {
            let ps = backend
                .read_postings(db_u64, tid, collection_key, term)
                .map_err(|e| crate::Error::Storage {
                    engine: "fts".to_string(),
                    detail: format!("FTS postings '{term}': {e}"),
                })?;
            postings.push((term.clone(), ps));
        }

        // Collect doc lengths from posting entries.
        let mut dl_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for (_, ps) in &postings {
            for p in ps {
                let k = p.doc_id.as_u32();
                if let std::collections::hash_map::Entry::Vacant(slot) = dl_map.entry(k)
                    && let Ok(Some(dl)) =
                        backend.read_doc_length(db_u64, tid, collection_key, p.doc_id)
                {
                    slot.insert(dl);
                }
            }
        }
        let doc_lengths: Vec<(nodedb_types::Surrogate, u32)> = dl_map
            .iter()
            .map(|(&k, &dl)| (nodedb_types::Surrogate::new(k), dl))
            .collect();

        let (doc_count, total_tokens) = backend
            .collection_stats(db_u64, tid, collection_key)
            .map_err(|e| crate::Error::Storage {
                engine: "fts".to_string(),
                detail: format!("FTS stats: {e}"),
            })?;

        let analyzer_meta = backend
            .read_meta(db_u64, tid, collection_key, "analyzer")
            .unwrap_or(None);

        let input = FtsRebuild {
            postings,
            doc_lengths,
            doc_count,
            total_tokens,
            analyzer_meta,
        };
        let (tx, rx) = mpsc::sync_channel::<crate::Result<RebuildOutput>>(1);
        std::thread::spawn(move || {
            let _ = tx.send(rebuild_fts_thread(input));
        });

        self.maintenance.pending_reindex.push(PendingReindex {
            database_id,
            tenant_id,
            collection_key: collection_key.to_string(),
            rx,
        });
        Ok(())
    }

    fn start_csr_rebuild(
        &mut self,
        task: &ExecutionTask,
        tenant_id: TenantId,
        collection_key: &str,
    ) -> crate::Result<()> {
        let database_id = task.request.database_id;
        let partition = match self.csr.partition(database_id, tenant_id) {
            Some(p) => p,
            None => return Ok(()), // nothing to rebuild
        };

        let snapshot_bytes =
            partition
                .checkpoint_to_bytes()
                .map_err(|e| crate::Error::Storage {
                    engine: "graph".to_string(),
                    detail: format!("CSR serialize: {e}"),
                })?;

        let memory = nodedb_mem::ScopedMemory::new(
            self.governor.clone(),
            database_id,
            tenant_id,
            nodedb_mem::EngineId::Graph,
        );
        let (tx, rx) = mpsc::sync_channel::<crate::Result<RebuildOutput>>(1);
        std::thread::spawn(move || {
            let _ = tx.send(rebuild_csr_thread(snapshot_bytes, memory));
        });

        self.maintenance.pending_reindex.push(PendingReindex {
            database_id,
            tenant_id,
            collection_key: collection_key.to_string(),
            rx,
        });
        Ok(())
    }
}
