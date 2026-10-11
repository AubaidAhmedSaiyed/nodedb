// SPDX-License-Identifier: BUSL-1.1

//! Settle every group's concluded entries and save the floors that moved.

use std::collections::HashMap;

use crate::control::server::dispatch_utils::publish_settled_changes;

use super::bookkeeping::record_durable_apply;
use super::context::ApplyContext;
use super::lane::Lane;

/// Settle every group's concluded entries in log order, release their
/// window, and save each durable floor that moved.
///
/// The tenant write marks of the settled entries are persisted before any
/// floor that covers them. An entry above the saved floor is delivered
/// again after a restart and records its mark again, so every committed
/// write keeps a durable mark.
pub(super) fn settle_lanes(ctx: ApplyContext<'_>, lanes: &mut HashMap<u64, Lane>) {
    let state = ctx.state;
    let tracker = ctx.tracker;
    for (group_id, lane) in lanes {
        let first = lane.front_index();
        let settled = lane.settle(tracker, &state.tenant_marks);
        if settled > 0 {
            tracker.window().release(*group_id, settled);
            // The entries' change events publish in log order, at their
            // log positions, on every replica.
            if let (Some(first), Some(last)) = (first, lane.last_settled()) {
                publish_settled_changes(state, *group_id, first, last);
            }
        }
        if !lane.floor_pending() {
            continue;
        }
        // One save per pass that moved the floor, never one per entry:
        // each save is an fsync, and the pass coalesces every apply that
        // finished before it.
        if let Err(error) = state.tenant_marks.persist(state.credentials.catalog()) {
            // The floor stays where it is, so every entry above it keeps
            // its place in the log. The next pass persists and saves again.
            tracing::error!(
                group_id = *group_id,
                %error,
                "apply loop: tenant write marks did not persist; the applied floor waits"
            );
            continue;
        }
        if let Some(floor) = lane.take_floor_to_save() {
            // Every entry the floor covers keeps its changes on disk.
            state.change_stream.journal_group(*group_id, floor);
            record_durable_apply(state, *group_id, floor);
        }
    }
}
