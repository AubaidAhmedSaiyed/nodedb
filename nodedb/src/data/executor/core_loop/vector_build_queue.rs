// SPDX-License-Identifier: BUSL-1.1

//! Per-core HNSW build queue: the core's side of its builder thread.
//!
//! Each Data Plane core owns one builder thread (`nodedb_vector::builder`).
//! A build is CPU-heavy, so it never runs on the core's reactor, and the
//! core never blocks on the builder:
//!
//! - A job enters the `backlog` as a descriptor (index key plus segment),
//!   not as vectors. The request is read from the collection when the job
//!   is sent, so a job whose segment is gone is dropped then.
//! - Sending uses `try_send` on the bounded request queue. A full queue
//!   leaves the job at the head of the backlog for the next tick, and the
//!   segment stays searchable by brute force.
//! - Finished builds come back on a bounded completion queue the core
//!   drains every tick; the builder thread waits while it is full.
//!
//! The backlog holds one small descriptor per unbuilt segment, so its size
//! is bounded by the segments the core's collections hold.

use std::collections::{HashMap, VecDeque};

use crate::data::executor::handlers::vector_direct_row::VectorIndexKey;
use crate::engine::vector::builder::{BUILD_QUEUE_CAPACITY, BuildSender, CompleteReceiver};

/// What a queued build produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::data::executor) enum BuildJobKind {
    /// Build the graph of building segment `segment_id`.
    Seal { segment_id: u32 },
    /// Rebuild the sealed segment at `base_id` in place.
    Rebuild { base_id: u32 },
}

/// One queued build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::data::executor) struct BuildJob {
    pub key: VectorIndexKey,
    pub kind: BuildJobKind,
}

/// The core's builder channels, backlog and in-flight count.
pub(in crate::data::executor) struct VectorBuildQueue {
    /// `None` when the builder thread could not be spawned. Jobs then wait
    /// in the backlog and the next dispatch spawns it again.
    pub(in crate::data::executor) tx: Option<BuildSender>,
    pub(in crate::data::executor) rx: Option<CompleteReceiver>,
    /// Jobs waiting for room in the builder queue, oldest first.
    pub(in crate::data::executor) backlog: VecDeque<BuildJob>,
    /// Jobs sent and not yet finished, per index.
    pub(in crate::data::executor) in_flight: HashMap<VectorIndexKey, usize>,
    /// Pending jobs (backlog plus in flight) last added to the shared
    /// backlog gauge.
    pub(in crate::data::executor) reported_pending: u64,
}

impl VectorBuildQueue {
    /// Spawn the builder thread for core `core_id`.
    pub(in crate::data::executor) fn spawn(core_id: usize) -> Self {
        let mut queue = Self {
            tx: None,
            rx: None,
            backlog: VecDeque::new(),
            in_flight: HashMap::new(),
            reported_pending: 0,
        };
        queue.respawn(core_id);
        queue
    }

    /// Spawn a fresh builder thread, replacing a dead one. On failure the
    /// channels stay `None` and the next dispatch tries again.
    pub(in crate::data::executor) fn respawn(&mut self, core_id: usize) {
        match crate::engine::vector::builder::spawn_builder(core_id, BUILD_QUEUE_CAPACITY) {
            Ok((tx, rx, _handle)) => {
                // The thread stops on its own once `tx` drops.
                self.tx = Some(tx);
                self.rx = Some(rx);
                // Builds the dead thread held never come back.
                self.in_flight.clear();
            }
            Err(e) => {
                tracing::error!(core = core_id, error = %e, "HNSW builder thread spawn failed");
                crate::diag::vector_builder_spawn_failed(&e, core_id);
                self.tx = None;
                self.rx = None;
            }
        }
    }

    /// Queue `job` unless the same job already waits.
    pub(in crate::data::executor) fn push(&mut self, job: BuildJob) {
        if !self.backlog.contains(&job) {
            self.backlog.push_back(job);
        }
    }

    /// Builds of `key` waiting or running.
    pub(in crate::data::executor) fn pending_for(&self, key: &VectorIndexKey) -> usize {
        self.backlog.iter().filter(|j| &j.key == key).count()
            + self.in_flight.get(key).copied().unwrap_or(0)
    }

    /// Builds waiting or running on this core.
    pub(in crate::data::executor) fn pending_total(&self) -> u64 {
        (self.backlog.len() + self.in_flight.values().sum::<usize>()) as u64
    }

    /// Record a job sent for `key`.
    pub(in crate::data::executor) fn note_sent(&mut self, key: VectorIndexKey) {
        *self.in_flight.entry(key).or_insert(0) += 1;
    }

    /// Record a finished job for `key`.
    pub(in crate::data::executor) fn note_finished(&mut self, key: &VectorIndexKey) {
        if let Some(n) = self.in_flight.get_mut(key) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.in_flight.remove(key);
            }
        }
    }
}
