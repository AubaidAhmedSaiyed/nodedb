// SPDX-License-Identifier: BUSL-1.1

//! Per-group registry of [`AppliedIndexWatcher`]s.
//!
//! A node hosts one or more Raft groups. Each gets its own watcher.
//! Lookup is keyed by `group_id` (`u64`) — the same key Raft uses
//! internally — so the metadata group (`METADATA_GROUP_ID = 0`) and
//! every data vshard group share one namespace.
//!
//! Lifecycle:
//!
//! - **Create**: lazy on first [`Self::get_or_create`]. The
//!   [`crate::raft_loop`] tick calls this once per locally-mounted
//!   group on every tick that produces apply work, so groups are
//!   registered as soon as Raft observes them.
//! - **Close**: explicit via [`Self::remove`] when a conf-change
//!   removes this node from the group, or when the group is dropped
//!   during decommission. Outstanding waiters wake with
//!   [`crate::applied_watcher::WaitOutcome::GroupGone`].

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use super::watcher::{AppliedIndexWatcher, WaitOutcome};

/// Registry of per-group apply watchers. Cheap to clone (one `Arc`).
#[derive(Debug, Default)]
pub struct GroupAppliedWatchers {
    inner: RwLock<HashMap<u64, Arc<AppliedIndexWatcher>>>,
}

impl GroupAppliedWatchers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up the watcher for `group_id`, creating it if absent.
    pub fn get_or_create(&self, group_id: u64) -> Arc<AppliedIndexWatcher> {
        // Fast path: read lock, return if present.
        {
            let r = self.inner.read().unwrap_or_else(|p| p.into_inner());
            if let Some(w) = r.get(&group_id) {
                return Arc::clone(w);
            }
        }
        // Slow path: upgrade to write and double-check.
        let mut w = self.inner.write().unwrap_or_else(|p| p.into_inner());
        Arc::clone(
            w.entry(group_id)
                .or_insert_with(|| Arc::new(AppliedIndexWatcher::new())),
        )
    }

    /// Look up the watcher for `group_id` without creating it.
    pub fn get(&self, group_id: u64) -> Option<Arc<AppliedIndexWatcher>> {
        let r = self.inner.read().unwrap_or_else(|p| p.into_inner());
        r.get(&group_id).map(Arc::clone)
    }

    /// Bump the watermark for `group_id`. Creates the watcher if
    /// absent so the first apply on a brand-new group always has a
    /// home. Idempotent on stale indices.
    pub fn bump(&self, group_id: u64, applied_index: u64) {
        self.get_or_create(group_id).bump(applied_index);
    }

    /// Remove the watcher for `group_id`. Wakes outstanding waiters
    /// with [`WaitOutcome::GroupGone`]. Idempotent.
    pub fn remove(&self, group_id: u64) {
        let removed = {
            let mut w = self.inner.write().unwrap_or_else(|p| p.into_inner());
            w.remove(&group_id)
        };
        if let Some(w) = removed {
            w.close();
        }
    }

    /// Wait until `group_id`'s applied watermark reaches `target`,
    /// `deadline` passes, or the group is removed. Creates the watcher
    /// when absent, so a wait that starts before the first apply here
    /// still wakes.
    pub async fn wait_for_async(
        &self,
        group_id: u64,
        target: u64,
        deadline: Instant,
    ) -> WaitOutcome {
        let watcher = self.get_or_create(group_id);
        watcher.wait_for_async(target, deadline).await
    }

    /// Number of groups currently registered. Useful for metrics.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap_or_else(|p| p.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot of `(group_id, applied_index)` for every registered
    /// group. Sorted by `group_id` for deterministic iteration.
    /// Used by test harnesses to drive cross-node apply-watermark
    /// convergence without poll-on-SQL.
    pub fn snapshot(&self) -> Vec<(u64, u64)> {
        let r = self.inner.read().unwrap_or_else(|p| p.into_inner());
        let mut v: Vec<(u64, u64)> = r.iter().map(|(gid, w)| (*gid, w.current())).collect();
        v.sort_by_key(|(gid, _)| *gid);
        v
    }

    /// List of currently-registered `group_id`s.
    pub fn group_ids(&self) -> Vec<u64> {
        let r = self.inner.read().unwrap_or_else(|p| p.into_inner());
        let mut v: Vec<u64> = r.keys().copied().collect();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn after(duration: Duration) -> Instant {
        Instant::now() + duration
    }

    #[test]
    fn separate_groups_have_independent_watermarks() {
        let r = GroupAppliedWatchers::new();
        r.bump(1, 5);
        r.bump(2, 10);
        assert_eq!(r.get(1).unwrap().current(), 5);
        assert_eq!(r.get(2).unwrap().current(), 10);
    }

    #[tokio::test]
    async fn wait_returns_reached_after_bump() {
        let r = Arc::new(GroupAppliedWatchers::new());
        let waiter = {
            let r = Arc::clone(&r);
            tokio::spawn(async move { r.wait_for_async(7, 5, after(Duration::from_secs(2))).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        r.bump(7, 5);
        assert_eq!(waiter.await.unwrap(), WaitOutcome::Reached);
    }

    #[tokio::test]
    async fn remove_wakes_waiter_with_group_gone() {
        let r = Arc::new(GroupAppliedWatchers::new());
        // Pre-create so the waiter watches the same Arc that remove() closes.
        r.get_or_create(3);
        let waiter = {
            let r = Arc::clone(&r);
            tokio::spawn(
                async move { r.wait_for_async(3, 99, after(Duration::from_secs(2))).await },
            )
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        r.remove(3);
        assert_eq!(waiter.await.unwrap(), WaitOutcome::GroupGone);
    }

    #[tokio::test]
    async fn missing_group_creates_watcher_lazily() {
        let r = GroupAppliedWatchers::new();
        let outcome = r
            .wait_for_async(42, 1, after(Duration::from_millis(20)))
            .await;
        assert_eq!(outcome, WaitOutcome::TimedOut);
        assert!(r.get(42).is_some());
    }
}
