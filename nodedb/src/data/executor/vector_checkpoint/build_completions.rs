// SPDX-License-Identifier: BUSL-1.1

//! Installing finished HNSW builds into the live collections.
//!
//! Lives beside the checkpoint because it shares the checkpoint's key
//! encoding: a `BuildComplete.key` is the same `"{db}:{tid}:{coll}"` string a
//! checkpoint filename carries, so both parse it through
//! [`parse_build_key`](super::paths::parse_build_key).

use super::paths::parse_build_key;
use crate::data::executor::core_loop::CoreLoop;
use crate::engine::vector::collection::{BuildComplete, BuildKind};

impl CoreLoop {
    /// Drain finished builds from this core's builder thread, install each
    /// one on its collection, then send the next backlog jobs.
    ///
    /// Called at the top of `tick()`. An install happens on this core, in one
    /// call, so a search sees either the brute-force segment or the built
    /// graph, never a mix.
    pub fn poll_build_completions(&mut self) {
        let mut drained = Vec::new();
        if let Some(rx) = &self.vector_builds.rx {
            while let Ok(complete) = rx.try_recv() {
                drained.push(complete);
            }
        }
        for complete in drained {
            self.install_build(complete);
        }
        if !self.vector_builds.backlog.is_empty() {
            self.dispatch_vector_builds();
        } else {
            self.sync_build_pending_metric();
        }
    }

    /// Install one finished build. A failed build, or one whose segment was
    /// truncated, dropped or renumbered meanwhile, leaves the collection as
    /// it is; the segment keeps answering by brute force or by its old graph.
    fn install_build(&mut self, complete: BuildComplete) {
        let BuildComplete {
            key,
            segment_id,
            kind,
            result,
        } = complete;
        let Some(tuple_key) = parse_build_key(&key) else {
            tracing::error!(
                core = self.core_id,
                key = %key,
                "HNSW build completion has unparseable key; dropping"
            );
            return;
        };
        self.vector_builds.note_finished(&tuple_key);
        let memory = nodedb_mem::ScopedMemory::new(
            self.governor.clone(),
            tuple_key.0,
            tuple_key.1,
            nodedb_mem::EngineId::Vector,
        );
        let Some(coll) = self.vector_collections.get_mut(&tuple_key) else {
            return;
        };
        let index = match result {
            Ok(index) => index,
            Err(e) => {
                coll.note_build_failed();
                let (kind_label, segment) = match kind {
                    BuildKind::Seal => ("seal", segment_id),
                    BuildKind::Rebuild { base_id, .. } => ("rebuild", base_id),
                };
                crate::diag::vector_build_failed(
                    &e,
                    &crate::diag::VectorBuildTarget {
                        kind: kind_label,
                        database_id: tuple_key.0.as_u64(),
                        tenant_id: tuple_key.1.as_u64(),
                        index: &tuple_key.2,
                        segment,
                    },
                );
                tracing::error!(
                    core = self.core_id,
                    key = %key,
                    segment_id,
                    error = %e,
                    "HNSW build failed; the segment stays as it is"
                );
                if let Some(m) = &self.metrics {
                    m.record_vector_build_failed();
                }
                return;
            }
        };
        let installed = match kind {
            BuildKind::Seal => coll.complete_build(segment_id, index, memory),
            BuildKind::Rebuild { base_id, len } => {
                coll.complete_rebuild(segment_id, base_id, len, index, memory)
            }
        };
        if installed {
            tracing::info!(
                core = self.core_id,
                key = %key,
                segment_id,
                ?kind,
                "HNSW build installed"
            );
            // A rebuilt segment replaces the old one in this single call on the
            // owning core, so search reads the old graph or the new one, never
            // a mix. REINDEX observers count this event, one per segment.
            if let BuildKind::Rebuild { base_id, len } = kind {
                tracing::info!(
                    target: "nodedb::reindex",
                    core = self.core_id,
                    index = "hnsw",
                    key = %key,
                    base_id,
                    len,
                    "atomic_cutover"
                );
            }
            self.checkpoint_coordinator.mark_dirty("vector", 1);
            if let Some(m) = &self.metrics {
                m.record_vector_build_completed();
            }
        } else {
            tracing::debug!(
                core = self.core_id,
                key = %key,
                segment_id,
                ?kind,
                "HNSW build no longer matches its segment; discarded"
            );
        }
    }
}
