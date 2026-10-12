// SPDX-License-Identifier: BUSL-1.1

//! Per-Raft-group applied-index watchers.
//!
//! A single primitive — [`AppliedIndexWatcher`] — tracks the highest
//! log index applied on this node for one Raft group. The
//! [`GroupAppliedWatchers`] registry indexes one watcher per group so
//! callers (proposers, consistent reads, lease renewals, recovery
//! checks) can wait on the apply watermark of *their* group rather
//! than only the metadata group.
//!
//! Bump points:
//!
//! 1. The metadata lane, once a metadata entry applied.
//! 2. The tick loop, for the Calvin sequencer group, which the host applies
//!    inline.
//! 3. The host apply loop, for data groups, once an entry's apply concluded
//!    and every earlier entry of the group settled.
//! 4. A snapshot install, for every group whose state it restored.
//! 5. A group mount, seeded with the durable applied index.

pub mod registry;
pub mod watcher;

pub use registry::GroupAppliedWatchers;
pub use watcher::{AppliedIndexWatcher, WaitOutcome};
