// SPDX-License-Identifier: Apache-2.0

//! Starting a rebuild on the owning thread, and the build that runs off it.

use std::sync::atomic::{AtomicU64, Ordering};

use nodedb_mem::ScopedMemory;

use super::journal::CsrJournal;
use crate::GraphError;
use crate::csr::index::CsrIndex;

static NEXT_REBUILD_TOKEN: AtomicU64 = AtomicU64::new(1);

/// Node labels as the index holds them. The checkpoint format omits them,
/// so a rebuild carries them beside it.
#[derive(Debug, Clone)]
pub(crate) struct NodeLabelState {
    /// Label names in id order.
    pub(crate) names: Vec<String>,
    /// Label bitset per node id.
    pub(crate) bits: Vec<u64>,
}

/// A rebuild's input, taken on the owning thread. Every field is `Send`.
#[derive(Debug)]
pub struct CsrRebuildSeed {
    token: u64,
    checkpoint: Vec<u8>,
    node_labels: NodeLabelState,
}

/// A compacted copy on its way back to the owning thread. Every field is
/// `Send`.
#[derive(Debug)]
pub struct CsrRebuilt {
    pub(crate) token: u64,
    pub(crate) checkpoint: Vec<u8>,
    pub(crate) node_labels: NodeLabelState,
}

impl CsrRebuildSeed {
    /// The token that ties this rebuild to the journal on the live index.
    pub fn token(&self) -> u64 {
        self.token
    }

    /// Restore the snapshot, compact it and serialize the result. Runs on
    /// any thread. `memory` bounds the copy's allocations.
    pub fn build(self, memory: ScopedMemory) -> Result<CsrRebuilt, GraphError> {
        let mut copy = restore_checkpoint(&self.checkpoint, memory)?;
        copy.compact()?;
        let checkpoint = copy.checkpoint_to_bytes()?;
        Ok(CsrRebuilt {
            token: self.token,
            checkpoint,
            node_labels: self.node_labels,
        })
    }
}

impl CsrRebuilt {
    /// The token of the rebuild that produced this copy.
    pub fn token(&self) -> u64 {
        self.token
    }
}

impl CsrIndex {
    /// Snapshot this index and open its write journal.
    ///
    /// Every later mutation records itself until `finish_rebuild` or
    /// `abort_rebuild` closes the journal. `max_journal_bytes` bounds the
    /// journal. Returns [`GraphError::RebuildInProgress`] when a journal is
    /// already open.
    pub fn begin_rebuild(
        &mut self,
        max_journal_bytes: usize,
    ) -> Result<CsrRebuildSeed, GraphError> {
        if self.rebuild_journal.is_some() {
            return Err(GraphError::RebuildInProgress);
        }
        let checkpoint = self.checkpoint_to_bytes()?;
        let token = NEXT_REBUILD_TOKEN.fetch_add(1, Ordering::Relaxed);
        self.rebuild_journal = Some(CsrJournal::new(token, max_journal_bytes));
        Ok(CsrRebuildSeed {
            token,
            checkpoint,
            node_labels: NodeLabelState {
                names: self.node_label_names.clone(),
                bits: self.node_label_bits.clone(),
            },
        })
    }

    /// Whether a rebuild journal is open on this index.
    pub fn rebuild_in_progress(&self) -> bool {
        self.rebuild_journal.is_some()
    }

    /// Close the journal of rebuild `token`. The index itself is unchanged.
    /// A journal of another rebuild stays open.
    pub fn abort_rebuild(&mut self, token: u64) {
        if self
            .rebuild_journal
            .as_ref()
            .is_some_and(|journal| journal.token == token)
        {
            self.rebuild_journal = None;
        }
    }
}

/// Decode a checkpoint a rebuild produced.
pub(crate) fn restore_checkpoint(
    bytes: &[u8],
    memory: ScopedMemory,
) -> Result<CsrIndex, GraphError> {
    CsrIndex::from_checkpoint(bytes, memory)
        .map_err(|e| GraphError::RebuildSnapshotInvalid {
            detail: e.to_string(),
        })?
        .ok_or_else(|| GraphError::RebuildSnapshotInvalid {
            detail: "checkpoint bytes carry no CSR header".to_string(),
        })
}
