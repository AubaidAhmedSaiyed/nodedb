// SPDX-License-Identifier: Apache-2.0

//! The write journal a `CsrIndex` keeps while a rebuild of it runs.
//!
//! Every public mutation records itself here with the outcome it had on
//! the live index. At cutover the journal replays onto the rebuilt copy,
//! so the copy holds every write the live index took after the snapshot.
//!
//! The journal is bounded in bytes. Past the bound it drops its entries,
//! frees their memory and marks itself overflowed. The cutover then
//! refuses the rebuilt copy with [`GraphError::RebuildJournalOverflow`].
//! The live index keeps every write either way.

use std::mem::size_of;

use crate::GraphError;
use crate::csr::index::CsrIndex;

/// One mutation of the live index, recorded for replay.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CsrWriteOp {
    AddEdge {
        src: String,
        label: String,
        dst: String,
        collection: String,
        weight: f64,
        force_weights: bool,
    },
    PutEdge {
        src: String,
        label: String,
        dst: String,
        collection: String,
        weight: f64,
    },
    RemoveEdge {
        src: String,
        label: String,
        dst: String,
        collection: String,
    },
    RemoveNodeEdges {
        node: String,
    },
    AddNode {
        name: String,
    },
    SetNodeSurrogate {
        node: String,
        surrogate: u32,
    },
    RestoreNodeSurrogate {
        node: String,
        prior: u32,
    },
    AddNodeLabel {
        node: String,
        label: String,
    },
    RemoveNodeLabel {
        node: String,
        label: String,
    },
    WithdrawNewestNode {
        node: String,
    },
    WithdrawNewestNodeLabel {
        label: String,
    },
}

impl CsrWriteOp {
    /// Name of the operation, for errors.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::AddEdge { .. } => "add_edge",
            Self::PutEdge { .. } => "put_edge",
            Self::RemoveEdge { .. } => "remove_edge",
            Self::RemoveNodeEdges { .. } => "remove_node_edges",
            Self::AddNode { .. } => "add_node",
            Self::SetNodeSurrogate { .. } => "set_node_surrogate",
            Self::RestoreNodeSurrogate { .. } => "restore_node_surrogate",
            Self::AddNodeLabel { .. } => "add_node_label",
            Self::RemoveNodeLabel { .. } => "remove_node_label",
            Self::WithdrawNewestNode { .. } => "withdraw_newest_node",
            Self::WithdrawNewestNodeLabel { .. } => "withdraw_newest_node_label",
        }
    }

    /// Bytes the entry holds: the enum itself plus its string contents.
    fn byte_cost(&self) -> usize {
        let strings = match self {
            Self::AddEdge {
                src,
                label,
                dst,
                collection,
                ..
            }
            | Self::PutEdge {
                src,
                label,
                dst,
                collection,
                ..
            }
            | Self::RemoveEdge {
                src,
                label,
                dst,
                collection,
            } => src.len() + label.len() + dst.len() + collection.len(),
            Self::RemoveNodeEdges { node }
            | Self::SetNodeSurrogate { node, .. }
            | Self::RestoreNodeSurrogate { node, .. }
            | Self::WithdrawNewestNode { node } => node.len(),
            Self::AddNode { name } => name.len(),
            Self::AddNodeLabel { node, label } | Self::RemoveNodeLabel { node, label } => {
                node.len() + label.len()
            }
            Self::WithdrawNewestNodeLabel { label } => label.len(),
        };
        size_of::<(Self, OpOutcome)>() + strings
    }
}

/// What a mutation returned on the index it ran against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpOutcome {
    /// The call returned `Ok`, or returns nothing.
    Applied,
    /// `add_node_label` returned `Ok(false)`: the label limit ignored it.
    Ignored,
    /// The call returned an error.
    Failed,
}

impl OpOutcome {
    pub(crate) fn of<T>(result: &Result<T, GraphError>) -> Self {
        if result.is_ok() {
            Self::Applied
        } else {
            Self::Failed
        }
    }

    pub(crate) fn of_label(result: &Result<bool, GraphError>) -> Self {
        match result {
            Ok(true) => Self::Applied,
            Ok(false) => Self::Ignored,
            Err(_) => Self::Failed,
        }
    }
}

/// Mutations recorded since a rebuild's snapshot, oldest first.
#[derive(Debug)]
pub struct CsrJournal {
    pub(crate) token: u64,
    cap_bytes: usize,
    used_bytes: usize,
    ops: Vec<(CsrWriteOp, OpOutcome)>,
    overflowed: bool,
}

impl CsrJournal {
    pub(crate) fn new(token: u64, cap_bytes: usize) -> Self {
        Self {
            token,
            cap_bytes,
            used_bytes: 0,
            ops: Vec::new(),
            overflowed: false,
        }
    }

    fn record(&mut self, op: CsrWriteOp, outcome: OpOutcome) {
        if self.overflowed {
            return;
        }
        let cost = op.byte_cost();
        if self.used_bytes.saturating_add(cost) > self.cap_bytes {
            self.overflowed = true;
            self.ops = Vec::new();
            self.used_bytes = 0;
            return;
        }
        self.used_bytes += cost;
        self.ops.push((op, outcome));
    }

    /// The recorded entries, or the overflow error when the bound was hit.
    pub(crate) fn into_ops(self) -> Result<Vec<(CsrWriteOp, OpOutcome)>, GraphError> {
        if self.overflowed {
            return Err(GraphError::RebuildJournalOverflow {
                cap_bytes: self.cap_bytes,
            });
        }
        Ok(self.ops)
    }
}

impl CsrIndex {
    /// Record a mutation when a rebuild journal is open. `op` runs only
    /// then, so an index with no rebuild allocates nothing here.
    pub(crate) fn journal_record(&mut self, op: impl FnOnce() -> CsrWriteOp, outcome: OpOutcome) {
        if let Some(journal) = self.rebuild_journal.as_mut() {
            journal.record(op(), outcome);
        }
    }

    /// Run one recorded mutation against this index and return its outcome.
    pub(crate) fn replay_op(&mut self, op: &CsrWriteOp) -> OpOutcome {
        match op {
            CsrWriteOp::AddEdge {
                src,
                label,
                dst,
                collection,
                weight,
                force_weights,
            } => OpOutcome::of(&self.apply_add_edge(
                src,
                label,
                dst,
                collection,
                *weight,
                *force_weights,
            )),
            CsrWriteOp::PutEdge {
                src,
                label,
                dst,
                collection,
                weight,
            } => OpOutcome::of(&self.apply_put_edge(src, label, dst, collection, *weight)),
            CsrWriteOp::RemoveEdge {
                src,
                label,
                dst,
                collection,
            } => {
                self.apply_remove_edge(src, label, dst, collection);
                OpOutcome::Applied
            }
            CsrWriteOp::RemoveNodeEdges { node } => {
                self.apply_remove_node_edges(node);
                OpOutcome::Applied
            }
            CsrWriteOp::AddNode { name } => OpOutcome::of(&self.ensure_node(name)),
            CsrWriteOp::SetNodeSurrogate { node, surrogate } => {
                self.apply_set_node_surrogate(node, *surrogate);
                OpOutcome::Applied
            }
            CsrWriteOp::RestoreNodeSurrogate { node, prior } => {
                self.apply_restore_node_surrogate(node, *prior);
                OpOutcome::Applied
            }
            CsrWriteOp::AddNodeLabel { node, label } => {
                OpOutcome::of_label(&self.apply_add_node_label(node, label))
            }
            CsrWriteOp::RemoveNodeLabel { node, label } => {
                self.apply_remove_node_label(node, label);
                OpOutcome::Applied
            }
            CsrWriteOp::WithdrawNewestNode { node } => {
                OpOutcome::of(&self.apply_withdraw_newest_node(node))
            }
            CsrWriteOp::WithdrawNewestNodeLabel { label } => {
                OpOutcome::of(&self.apply_withdraw_newest_node_label(label))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_drops_entries_and_reports_the_bound() {
        let op = CsrWriteOp::AddNode {
            name: "n".to_string(),
        };
        let cap = op.byte_cost() * 2;
        let mut journal = CsrJournal::new(1, cap);
        journal.record(op.clone(), OpOutcome::Applied);
        journal.record(op.clone(), OpOutcome::Applied);
        assert_eq!(journal.ops.len(), 2);
        journal.record(op, OpOutcome::Applied);
        assert!(
            journal.ops.is_empty(),
            "an overflowed journal frees its entries"
        );
        assert!(matches!(
            journal.into_ops(),
            Err(GraphError::RebuildJournalOverflow { cap_bytes }) if cap_bytes == cap
        ));
    }
}
