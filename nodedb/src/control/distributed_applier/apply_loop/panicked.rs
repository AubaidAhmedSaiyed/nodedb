// SPDX-License-Identifier: BUSL-1.1

//! An entry whose enqueue or apply panicked.
//!
//! The loop polls every running entry in one task, so an unguarded panic
//! ends the loop for every group and leaves each waiter to its deadline.
//! A guarded panic concludes its own entry instead:
//! - Its effect on this replica is unknown, so the entry holds its group's
//!   lane like an entry this replica cannot apply as its peers do. No later
//!   entry of the group starts until a snapshot install covers it.
//! - Its waiter receives an error at once.
//! - The panic files a report.
//!
//! Other groups keep applying.

use std::panic::AssertUnwindSafe;

use futures::FutureExt;

use crate::data::panic_payload::panic_payload_to_string;

use super::context::{ApplyContext, ApplyFuture, LoopEvent, LoopFuture};
use super::lane::{Lane, SlotState};
use super::proposal_gate::{EntryOutcome, HoldReason, ProposalGate};

/// Run `future`, the enqueue or apply of the entry at `log_index` of
/// `group_id`. A panic inside it becomes [`LoopEvent::Panicked`].
pub(super) fn guarded(group_id: u64, log_index: u64, future: LoopFuture<'_>) -> LoopFuture<'_> {
    Box::pin(
        AssertUnwindSafe(future)
            .catch_unwind()
            .map(move |polled| match polled {
                Ok(event) => event,
                Err(payload) => LoopEvent::Panicked {
                    group_id,
                    log_index,
                    detail: panic_payload_to_string(payload.as_ref()),
                },
            }),
    )
}

/// The guarded event of an entry's running apply.
pub(super) fn finished_event(
    group_id: u64,
    log_index: u64,
    apply: ApplyFuture<'_>,
) -> LoopFuture<'_> {
    guarded(
        group_id,
        log_index,
        Box::pin(apply.map(LoopEvent::Finished)),
    )
}

/// Conclude the entry at `log_index` whose enqueue or apply panicked: report
/// it, answer its waiter with an error, and hold `lane`. Returns `false` when
/// no entry at that index starts or runs.
pub(super) fn conclude_panicked(
    ctx: ApplyContext<'_>,
    lane: &mut Lane,
    gate: &mut ProposalGate,
    log_index: u64,
    detail: &str,
) -> bool {
    let group_id = lane.group_id();
    let Some(proposal_key) = lane.running_key(log_index) else {
        return false;
    };
    tracing::error!(
        group_id,
        log_index,
        "apply loop: the committed entry's apply panicked; the group's apply holds at it \
         until a snapshot of the group installs"
    );
    crate::diag::apply_panicked(group_id, log_index, detail);
    ctx.tracker.complete(
        group_id,
        log_index,
        proposal_key,
        Err(crate::Error::Internal {
            detail: format!(
                "the apply of group {group_id} entry {log_index} panicked on this replica; \
                 the group's apply holds until a snapshot installs"
            ),
        }),
    );
    ctx.tracker.note_concluded(group_id, log_index);
    // A panicked enqueue leaves its entry starting. A panicked apply leaves
    // it running, and this is a no-op.
    lane.enqueued(log_index, SlotState::Running, None, false);
    let outcome = EntryOutcome::Held {
        reason: HoldReason::ApplyPanicked,
    };
    lane.conclude(log_index, false, |key| gate.conclude(key, true, outcome))
}
