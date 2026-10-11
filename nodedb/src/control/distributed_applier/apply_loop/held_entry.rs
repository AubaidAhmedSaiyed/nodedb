// SPDX-License-Identifier: BUSL-1.1

//! A committed entry this replica cannot apply as its peers do holds its
//! group's lane (see [`super::lane`]). Later entries applied on state
//! without it would diverge from the other replicas, and reads would see
//! that state.
//!
//! While the lane holds, the group's replica here owes a snapshot install:
//! - The debt is durable, so a restart keeps it.
//! - The replica refuses log entries, steps down from a leadership it
//!   holds, and never campaigns. Its leader sends a snapshot.
//! - The install that covers the held entry releases it and settles the
//!   debt. Backlog entries at or below the install conclude as covered.
//!
//! The debt is kept once per install of the group. A debt write that
//! failed is written again on the apply loop's timed wake.

use std::collections::{BTreeSet, HashMap};

use crate::control::state::SharedState;

use super::context::ApplyContext;
use super::lane::Lane;
use super::proposal_gate::{HoldReason, ProposalGate};
use super::snapshot_gate::{EntryAdmission, admit_entry};

/// Where a group's lane stands against a hold.
pub(super) enum HoldCheck {
    /// No entry holds the lane: it can start entries.
    Free,
    /// An entry holds the lane and no install covers it yet.
    Held,
    /// An install holds or awaits the group's apply gate.
    Installing,
}

/// Note that the entry at `log_index` of `group_id` entered a hold. The
/// group's debt is recorded by [`keep_snapshot_owed`], under the group's
/// apply gate.
pub(super) fn note_hold(group_id: u64, log_index: u64, reason: HoldReason) {
    tracing::error!(
        group_id,
        log_index,
        reason = reason.as_str(),
        "apply loop: this replica cannot apply the committed entry as its peers do; the \
         group's apply holds at it until a snapshot of the group installs"
    );
}

/// The snapshot debt this loop kept for each held group.
#[derive(Default)]
pub(super) struct HeldDebts {
    groups: HashMap<u64, KeptDebt>,
}

#[derive(Clone, Copy)]
struct KeptDebt {
    /// The group's installed index when the debt was kept.
    installed_through: u64,
    /// Whether the debt write persisted.
    recorded: bool,
}

/// What a pass does for the debt of a held group.
#[derive(Debug, PartialEq, Eq)]
enum DebtStep {
    /// The debt persisted after the group's last install.
    Kept,
    /// Write the debt.
    Write {
        /// Raise the replica's snapshot requirement too: no install lifted it
        /// since this loop last raised it.
        raise: bool,
        /// The previous write failed: this write continues that failure run.
        failing: bool,
    },
}

impl HeldDebts {
    /// An install settles the debt and can lift the requirement, so a hold
    /// that outlives one keeps both again.
    fn step(&self, group_id: u64, installed_through: u64) -> DebtStep {
        match self.groups.get(&group_id) {
            Some(kept) if kept.installed_through == installed_through && kept.recorded => {
                DebtStep::Kept
            }
            Some(kept) => DebtStep::Write {
                raise: kept.installed_through != installed_through,
                failing: !kept.recorded,
            },
            None => DebtStep::Write {
                raise: true,
                failing: false,
            },
        }
    }

    fn note(&mut self, group_id: u64, installed_through: u64, recorded: bool) {
        self.groups.insert(
            group_id,
            KeptDebt {
                installed_through,
                recorded,
            },
        );
    }

    /// Drop the record of `group_id`: no entry holds its lane.
    fn forget(&mut self, group_id: u64) {
        self.groups.remove(&group_id);
    }

    /// Whether a held group's debt write failed. The apply loop wakes on a
    /// timer to write it again.
    pub fn retry_due(&self) -> bool {
        self.groups.values().any(|kept| !kept.recorded)
    }
}

/// Record the group's snapshot debt and raise its replica's snapshot
/// requirement, once per install of the group. `installed_through` is the
/// group's installed index as the caller read it.
///
/// The caller holds the group's apply gate or read it open below the held
/// entry, so no install that covers the entry settled the debt first.
pub(super) fn keep_snapshot_owed(
    state: &SharedState,
    debts: &mut HeldDebts,
    group_id: u64,
    log_index: u64,
    installed_through: u64,
) {
    let DebtStep::Write { raise, failing } = debts.step(group_id, installed_through) else {
        return;
    };
    let written = if state.redo_chunks.owes_snapshot(group_id) {
        Ok(())
    } else {
        state
            .redo_chunks
            .record_owed(state.credentials.catalog(), &BTreeSet::from([group_id]))
    };
    let recorded = match written {
        Ok(()) => {
            if failing {
                tracing::info!(
                    group_id,
                    log_index,
                    "apply loop: the snapshot debt of a held group persisted"
                );
            }
            true
        }
        // One report per failure run: the timed retries collapse into it.
        Err(error) if failing => {
            tracing::debug!(
                group_id,
                log_index,
                %error,
                "apply loop: the snapshot debt of a held group still did not persist"
            );
            false
        }
        Err(error) => {
            tracing::error!(
                group_id,
                log_index,
                %error,
                "apply loop: the snapshot debt of a held group did not persist; the loop \
                 writes it again on each Raft tick"
            );
            crate::diag::held_entry_debt_not_recorded(group_id, log_index, &error);
            false
        }
    };
    debts.note(group_id, installed_through, recorded);
    // Once the debt persists, the redo-stream membership pass holds the
    // requirement up. Until then only this loop does.
    if !raise && recorded {
        return;
    }
    let Some(multi_raft) = state.multi_raft.get() else {
        return;
    };
    // The data here diverged from the log: the replica must not lead.
    multi_raft
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .require_snapshot(group_id, nodedb_raft::ReplicaState::AwaitingSnapshot);
}

/// Release the entry that holds `lane` once an installed snapshot covers
/// it. The install's index is read under the group's apply gate, as an
/// entry's admission reads it. An entry still held keeps its group owing a
/// snapshot.
pub(super) fn release_covered_hold(
    ctx: ApplyContext<'_>,
    lane: &mut Lane,
    gate: &mut ProposalGate,
    debts: &mut HeldDebts,
    group_id: u64,
) -> HoldCheck {
    let Some(held) = lane.held_index() else {
        debts.forget(group_id);
        return HoldCheck::Free;
    };
    let gates = ctx.state.raft_apply_gates.get().map(|gates| &**gates);
    let covered_through = ctx.tracker.covered_through(group_id);
    let admission = admit_entry(gates, group_id, held, covered_through);
    let installed_through = admission.installed_through;
    // The permit lives to the end: no install starts before the debt is
    // kept.
    let (through, _permit) = match admission.decision {
        EntryAdmission::Installing => return HoldCheck::Installing,
        EntryAdmission::Start(permit) => (admission.installed_through, permit),
        EntryAdmission::Covered => (admission.installed_through, None),
        // Above the install's Raft index and at or below its cut: the
        // installed state holds the entry.
        EntryAdmission::CoveredByCut => (covered_through, None),
    };
    match lane.cover_through(through) {
        Some(proposal_key) => {
            gate.note_covered(proposal_key);
            debts.forget(group_id);
            tracing::info!(
                group_id,
                log_index = held,
                installed_through = through,
                "apply loop: an installed snapshot covers the held entry; the group's apply \
                 resumes"
            );
            HoldCheck::Free
        }
        None => {
            keep_snapshot_owed(ctx.state, debts, group_id, held, installed_through);
            HoldCheck::Held
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GROUP: u64 = 4;

    /// The debt is written once per install of the group: an install
    /// settles it and can lift the requirement.
    #[test]
    fn a_held_debt_is_kept_once_per_install() {
        let mut debts = HeldDebts::default();
        let first = DebtStep::Write {
            raise: true,
            failing: false,
        };
        assert_eq!(debts.step(GROUP, 10), first);
        debts.note(GROUP, 10, true);
        assert_eq!(debts.step(GROUP, 10), DebtStep::Kept);
        assert!(!debts.retry_due());
        assert_eq!(debts.step(GROUP, 12), first, "an install settled the debt");
    }

    /// A failed write is retried as one failure run, across installs, until
    /// a write persists or the hold ends.
    #[test]
    fn a_failed_debt_write_retries_as_one_failure_run() {
        let mut debts = HeldDebts::default();
        debts.note(GROUP, 10, false);
        assert!(debts.retry_due());
        assert_eq!(
            debts.step(GROUP, 10),
            DebtStep::Write {
                raise: false,
                failing: true,
            }
        );
        assert_eq!(
            debts.step(GROUP, 12),
            DebtStep::Write {
                raise: true,
                failing: true,
            }
        );

        debts.note(GROUP, 10, true);
        assert!(!debts.retry_due());

        debts.note(GROUP, 10, false);
        debts.forget(GROUP);
        assert!(!debts.retry_due(), "a released hold retries nothing");
    }
}
