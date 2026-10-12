// SPDX-License-Identifier: BUSL-1.1

//! Async wait on a Raft group's local apply watermark.
//!
//! A waiter parks as a task on a [`Notify`], so it holds no thread while it
//! waits. A `closed` flag tells "still waiting for apply" apart from "the
//! group is no longer hosted on this node": a removed group wakes its
//! waiters with a terminal outcome instead of timing them out.

use std::sync::Mutex;
use std::time::Instant;

use tokio::sync::Notify;

/// Outcome of [`AppliedIndexWatcher::wait_for_async`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitOutcome {
    /// The applied watermark reached the requested target.
    Reached,
    /// The deadline elapsed before the target was reached.
    TimedOut,
    /// The group was closed (removed from this node) while waiting.
    GroupGone,
}

impl WaitOutcome {
    pub fn is_reached(self) -> bool {
        matches!(self, Self::Reached)
    }
}

/// Tracks the highest Raft log index applied on this node for one
/// Raft group, and the last index of the batch being applied now.
#[derive(Debug, Default)]
pub struct AppliedIndexWatcher {
    state: Mutex<State>,
    /// Woken on every watermark advance and on close.
    changed: Notify,
}

#[derive(Debug, Default)]
struct State {
    applied: u64,
    /// The last index of the batch the applier is applying now. A batch
    /// makes its effects visible entry by entry before `applied` reaches
    /// its end.
    applying_through: u64,
    closed: bool,
}

impl AppliedIndexWatcher {
    pub fn new() -> Self {
        Self::default()
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Advance the watermark. Idempotent: smaller indices are ignored.
    /// Bumps on a closed watcher are also ignored (the group is gone).
    pub fn bump(&self, applied_index: u64) {
        let advanced = {
            let mut guard = self.state();
            let advanced = !guard.closed && applied_index > guard.applied;
            if advanced {
                guard.applied = applied_index;
            }
            advanced
        };
        if advanced {
            self.changed.notify_waiters();
        }
    }

    /// Read the current watermark without blocking.
    pub fn current(&self) -> u64 {
        self.state().applied
    }

    /// Declare that the applier starts a batch ending at `last_index`. The
    /// applier calls it before the batch mutates any state a reader can see.
    /// Idempotent: smaller indices are ignored.
    pub fn begin_batch(&self, last_index: u64) {
        let mut guard = self.state();
        guard.applying_through = guard.applying_through.max(last_index);
    }

    /// The floor a reader stamps on work planned against what it sees now:
    /// `max(applied, applying-through)`.
    ///
    /// A reader that sees an effect of the batch in progress gets a floor at
    /// or above the index of the entry that made it visible. A waiter for
    /// that floor waits on [`Self::wait_for_async`], which the finished batch
    /// satisfies.
    pub fn floor(&self) -> u64 {
        let guard = self.state();
        guard.applied.max(guard.applying_through)
    }

    /// True once [`Self::close`] has been called.
    pub fn is_closed(&self) -> bool {
        self.state().closed
    }

    /// Mark this watcher closed and wake every waiter with
    /// [`WaitOutcome::GroupGone`]. Idempotent.
    pub fn close(&self) {
        let closed_now = {
            let mut guard = self.state();
            let closed_now = !guard.closed;
            guard.closed = true;
            closed_now
        };
        if closed_now {
            self.changed.notify_waiters();
        }
    }

    /// The outcome of a wait for `target` as the state stands, `None` while
    /// it is still pending.
    fn settled(&self, target: u64) -> Option<WaitOutcome> {
        let guard = self.state();
        if guard.applied >= target {
            Some(WaitOutcome::Reached)
        } else if guard.closed {
            Some(WaitOutcome::GroupGone)
        } else {
            None
        }
    }

    /// Wait until the watermark reaches `target`, `deadline` passes, or the
    /// watcher is closed. The waiting task holds no thread.
    pub async fn wait_for_async(&self, target: u64, deadline: Instant) -> WaitOutcome {
        let deadline = tokio::time::Instant::from_std(deadline);
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            // Registered before the state is read, so a bump between the
            // read and the await still wakes this waiter.
            notified.as_mut().enable();
            if let Some(outcome) = self.settled(target) {
                return outcome;
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return self.settled(target).unwrap_or(WaitOutcome::TimedOut);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    fn after(duration: Duration) -> Instant {
        Instant::now() + duration
    }

    #[tokio::test]
    async fn a_bump_wakes_the_waiter() {
        let w = Arc::new(AppliedIndexWatcher::new());
        let waiter = {
            let w = Arc::clone(&w);
            tokio::spawn(async move { w.wait_for_async(5, after(Duration::from_secs(5))).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        w.bump(3);
        tokio::time::sleep(Duration::from_millis(20)).await;
        w.bump(5);
        let started = Instant::now();
        assert_eq!(waiter.await.expect("join"), WaitOutcome::Reached);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn the_wait_times_out_at_its_deadline() {
        let w = AppliedIndexWatcher::new();
        let started = Instant::now();
        let outcome = w.wait_for_async(1, after(Duration::from_millis(200))).await;
        assert_eq!(outcome, WaitOutcome::TimedOut);
        let waited = started.elapsed();
        assert!(waited >= Duration::from_millis(200), "{waited:?}");
        assert!(waited < Duration::from_secs(2), "{waited:?}");
    }

    #[tokio::test]
    async fn a_passed_deadline_still_reports_a_reached_target() {
        let w = AppliedIndexWatcher::new();
        w.bump(10);
        let outcome = w.wait_for_async(5, Instant::now()).await;
        assert_eq!(outcome, WaitOutcome::Reached);
    }

    #[tokio::test]
    async fn close_wakes_the_waiter_with_group_gone() {
        let w = Arc::new(AppliedIndexWatcher::new());
        let waiter = {
            let w = Arc::clone(&w);
            tokio::spawn(async move { w.wait_for_async(99, after(Duration::from_secs(5))).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        w.close();
        assert_eq!(waiter.await.expect("join"), WaitOutcome::GroupGone);
    }

    /// A bump that races the start of a wait is never lost: every waiter
    /// ends `Reached` well before its deadline.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_bump_racing_the_wait_start_is_not_lost() {
        for round in 1..=200u64 {
            let w = Arc::new(AppliedIndexWatcher::new());
            let waiter = {
                let w = Arc::clone(&w);
                tokio::spawn(async move {
                    w.wait_for_async(round, after(Duration::from_secs(10)))
                        .await
                })
            };
            let bumper = {
                let w = Arc::clone(&w);
                tokio::spawn(async move { w.bump(round) })
            };
            bumper.await.expect("join bumper");
            let outcome = tokio::time::timeout(Duration::from_secs(5), waiter)
                .await
                .expect("the waiter woke on the bump")
                .expect("join waiter");
            assert_eq!(outcome, WaitOutcome::Reached, "round {round}");
        }
    }

    #[test]
    fn bump_after_close_is_ignored() {
        let w = AppliedIndexWatcher::new();
        w.close();
        w.bump(5);
        assert_eq!(w.current(), 0);
        assert!(w.is_closed());
    }

    #[test]
    fn close_is_idempotent() {
        let w = AppliedIndexWatcher::new();
        w.close();
        w.close();
        assert!(w.is_closed());
    }

    #[test]
    fn the_floor_covers_the_batch_in_progress() {
        let w = AppliedIndexWatcher::new();
        w.bump(4);
        assert_eq!(w.floor(), 4);
        w.begin_batch(9);
        assert_eq!(w.current(), 4, "the applied watermark waits for the batch");
        assert_eq!(w.floor(), 9);
        w.begin_batch(7);
        assert_eq!(w.floor(), 9, "a smaller batch end is ignored");
        w.bump(9);
        assert_eq!(w.floor(), 9);
        w.bump(12);
        assert_eq!(w.floor(), 12);
    }
}
