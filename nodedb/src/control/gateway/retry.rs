// SPDX-License-Identifier: BUSL-1.1

//! Typed `NotLeader` retry that runs until the statement deadline.
//!
//! When a remote dispatch returns `Error::NotLeader`, the retry helper:
//! 1. Extracts the hinted new leader and its term from the error.
//! 2. Offers them to the routing table entry for the affected group. The
//!    hint moves only when the redirect's term is above the hint's term, so
//!    a stale redirect never undoes a newer leader.
//! 3. Sleeps for the next backoff of its [`RetryPacing`].
//! 4. Re-invokes the closure.
//!
//! If the hinted leader is unknown (no hint), we still retry after sleep
//! without updating the routing table — a subsequent routing lookup will
//! re-read the table from the current routing state.
//!
//! A re-election can outlast any fixed attempt count, so retries continue
//! while the statement has time left. Every attempt shares the statement's
//! deadline. When the next backoff would reach it, one last attempt starts
//! at once, and its verdict ends the statement. A lost answer to a read then
//! ends as the deadline the last send reports. No retry starts once the
//! deadline passed. The last refusal then ends the statement: it is
//! definite, so it never becomes `DeadlineExceeded`, which means a lost
//! answer.

use std::future::Future;
use std::sync::RwLock;

use tokio::time::{Duration, Instant, sleep};
use tracing::debug;

use nodedb_cluster::RoutingTable;
use nodedb_types::config::tuning::ClusterTransportTuning;

use crate::Error;
use crate::control::server::shared::session::deadline::hop_budget_ms;

/// One dispatch attempt of a statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attempt {
    /// The attempt's 0-based index.
    pub index: usize,
    /// The statement's deadline. Every attempt carries the same instant.
    pub deadline: Instant,
}

impl Attempt {
    /// The milliseconds left on the statement now, at least 1. See
    /// [`hop_budget_ms`].
    pub fn remaining_ms(&self) -> u64 {
        hop_budget_ms(self.deadline.into_std())
    }
}

/// The backoff between leader retries: doubles from `first` up to `cap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPacing {
    /// The pause before the first retry.
    pub first: Duration,
    /// The longest pause between two retries.
    pub cap: Duration,
}

impl RetryPacing {
    /// Pacing from the cluster's Raft timing.
    ///
    /// The first pause is one Raft tick. The cap is the minimum election
    /// timeout: no new leader can appear faster than that.
    pub fn from_tuning(tuning: &ClusterTransportTuning) -> Self {
        let first = Duration::from_millis(tuning.raft_tick_interval_ms.max(1));
        let cap = Duration::from_millis(tuning.effective_election_timeout_min_ms()).max(first);
        Self { first, cap }
    }

    /// The pause after `retries` retries already ran.
    fn backoff(&self, retries: u32) -> Duration {
        self.first
            .checked_mul(2u32.saturating_pow(retries))
            .map_or(self.cap, |d| d.min(self.cap))
    }
}

impl Default for RetryPacing {
    fn default() -> Self {
        Self::from_tuning(&ClusterTransportTuning::default())
    }
}

/// Execute `f`, retrying on `Error::NotLeader` and
/// `Error::RetryableLeaderChange` until `deadline`.
///
/// `f` receives the current [`Attempt`]: its index and the statement's
/// `deadline`, which every attempt shares.
///
/// On `NotLeader` with a hinted leader, the routing table is updated before
/// the next retry so the caller's routing decision changes. On non-`NotLeader`
/// errors the error is propagated immediately without retry.
pub async fn retry_not_leader<F, Fut, T>(
    routing: Option<&RwLock<RoutingTable>>,
    deadline: Instant,
    pacing: RetryPacing,
    f: F,
) -> Result<T, Error>
where
    F: Fn(Attempt) -> Fut,
    Fut: Future<Output = Result<T, Error>>,
{
    let mut attempt = 0;
    let mut last_call = false;
    loop {
        let refusal = match f(Attempt {
            index: attempt,
            deadline,
        })
        .await
        {
            Ok(v) => return Ok(v),
            Err(Error::NotLeader {
                vshard_id,
                leader_node,
                leader_term,
                ..
            }) => {
                debug!(
                    attempt,
                    vshard_id = vshard_id.as_u32(),
                    leader_node,
                    leader_term,
                    "gateway: NotLeader — will retry with new leader hint"
                );

                // Update routing table with the redirect:
                //   • `leader_node != 0` with a term → offer the leader at its
                //     term. It applies above the hint's term, and fills a hint
                //     cleared at that term.
                //   • `leader_node != 0` with term 0 → the source named an
                //     owner without a term. It fills only a hint that holds
                //     no term.
                //   • `leader_node == 0` → transport failure or no hint; clear
                //     the group leader and keep its term, so the next attempt
                //     falls back to local dispatch rather than retrying the
                //     same dead node.
                if let Some(rt) = routing
                    && let Ok(mut table) = rt.write()
                    && let Ok(group_id) = table.group_for_vshard(vshard_id.as_u32())
                {
                    if leader_node == 0 {
                        table.clear_leader(group_id);
                    } else if leader_term == 0 {
                        table.set_leader(group_id, leader_node);
                    } else {
                        table.confirm_leader(group_id, leader_node, leader_term);
                    }
                }

                Error::NotLeader {
                    vshard_id,
                    leader_node,
                    leader_addr: String::new(),
                    leader_term,
                }
            }
            Err(Error::RetryableLeaderChange {
                group_id,
                log_index,
            }) => {
                // A new leader's election no-op overwrote the entry the
                // proposer waited on. The next attempt re-proposes under the
                // new leader's term.
                debug!(
                    attempt,
                    group_id, log_index, "gateway: leader change overwrote entry — will re-propose"
                );

                Error::RetryableLeaderChange {
                    group_id,
                    log_index,
                }
            }
            Err(other) => return Err(other),
        };

        // Both refusals mean nothing applied. Once the deadline passed, or
        // the last call ran, that definite refusal ends the statement.
        let now = Instant::now();
        if last_call || now >= deadline {
            debug!(
                attempt,
                "gateway: statement deadline reached — no further retry"
            );
            return Err(refusal);
        }
        let pause = pacing.backoff(u32::try_from(attempt).unwrap_or(u32::MAX));
        if now + pause >= deadline {
            last_call = true;
        } else {
            sleep(pause).await;
        }
        attempt += 1;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, RwLock,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::types::VShardId;

    fn far_deadline() -> Instant {
        Instant::now() + Duration::from_secs(30)
    }

    fn pacing() -> RetryPacing {
        RetryPacing {
            first: Duration::from_millis(10),
            cap: Duration::from_millis(150),
        }
    }

    fn not_leader(leader_node: u64, leader_term: u64) -> Error {
        Error::NotLeader {
            vshard_id: VShardId::new(0),
            leader_node,
            leader_addr: String::new(),
            leader_term,
        }
    }

    #[tokio::test]
    async fn success_on_first_attempt() {
        let result = retry_not_leader(None, far_deadline(), pacing(), |_attempt| async {
            Ok::<u32, Error>(42)
        })
        .await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn success_on_second_attempt() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&call_count);
        let result = retry_not_leader(None, far_deadline(), pacing(), move |_attempt| {
            let c = Arc::clone(&count);
            async move {
                if c.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err(not_leader(2, 1))
                } else {
                    Ok::<u32, Error>(99)
                }
            }
        })
        .await;
        assert_eq!(result.unwrap(), 99);
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn non_not_leader_error_propagates_immediately() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&call_count);
        let result = retry_not_leader(None, far_deadline(), pacing(), move |_| {
            let c = Arc::clone(&count);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err::<u32, Error>(Error::BadRequest {
                    detail: "bad".into(),
                })
            }
        })
        .await;
        assert!(matches!(result, Err(Error::BadRequest { .. })));
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    /// A re-election longer than a few fixed attempts is retried until the
    /// new leader answers.
    #[tokio::test(start_paused = true)]
    async fn a_leader_change_past_350_ms_is_retried_until_it_resolves() {
        let started = Instant::now();
        let resolves_at = started + Duration::from_millis(1_000);
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let result = retry_not_leader(None, far_deadline(), pacing(), move |_| {
            let c = Arc::clone(&count);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                if Instant::now() < resolves_at {
                    Err(not_leader(0, 0))
                } else {
                    Ok::<u32, Error>(7)
                }
            }
        })
        .await;
        assert_eq!(result.unwrap(), 7);
        assert!(calls.load(Ordering::SeqCst) > 3, "more than three attempts");
        let waited = started.elapsed();
        assert!(
            waited < Duration::from_millis(1_000) + pacing().cap,
            "the retry follows the resolution within one capped backoff: {waited:?}"
        );
    }

    /// Retries stop when the next backoff reaches the deadline. The last
    /// refusal stays the answer, never `DeadlineExceeded`.
    #[tokio::test(start_paused = true)]
    async fn a_passing_deadline_stops_retries_with_the_last_refusal() {
        let started = Instant::now();
        let deadline = started + Duration::from_millis(600);
        let result = retry_not_leader(None, deadline, pacing(), |attempt| async move {
            Err::<(), Error>(not_leader(7, attempt.index as u64 + 1))
        })
        .await;
        match result {
            Err(Error::NotLeader {
                leader_node: 7,
                leader_term,
                ..
            }) => assert!(leader_term > 1, "the last refusal returns"),
            other => panic!("expected the last NotLeader, got {other:?}"),
        }
        assert!(
            Instant::now() <= deadline,
            "no retry runs past the deadline"
        );
    }

    #[tokio::test]
    async fn routing_table_updated_on_not_leader_hint() {
        let table = RoutingTable::uniform(1, &[1, 2], 2);
        let rt = Arc::new(RwLock::new(table));
        let rt_clone = Arc::clone(&rt);

        let call_count = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&call_count);

        let _ = retry_not_leader(Some(&*rt_clone), far_deadline(), pacing(), move |_| {
            let c = Arc::clone(&count);
            async move {
                if c.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err(not_leader(2, 4))
                } else {
                    Ok::<(), Error>(())
                }
            }
        })
        .await;

        let table = rt.read().unwrap();
        assert_eq!(table.leader_at_term_for_vshard(0).unwrap(), (2, 4));
    }

    /// A redirect at a lower term than the hint never moves it back.
    #[tokio::test]
    async fn a_stale_redirect_leaves_a_newer_hint() {
        let mut table = RoutingTable::uniform(1, &[1, 2, 3], 3);
        let group_id = table.group_for_vshard(0).unwrap();
        assert!(table.observe_leader(group_id, 3, 7));
        let rt = RwLock::new(table);

        let _ = retry_not_leader(Some(&rt), far_deadline(), pacing(), |attempt| async move {
            if attempt.index == 0 {
                Err(not_leader(2, 5))
            } else {
                Ok::<(), Error>(())
            }
        })
        .await;

        let table = rt.read().unwrap();
        assert_eq!(table.leader_at_term_for_vshard(0).unwrap(), (3, 7));
    }

    /// A redirect without a term fills a hint that holds no term, and never
    /// replaces a termed one.
    #[tokio::test]
    async fn a_term_less_redirect_never_replaces_a_termed_hint() {
        async fn redirect_to(rt: &RwLock<RoutingTable>, leader_node: u64) {
            let _ = retry_not_leader(Some(rt), far_deadline(), pacing(), |attempt| async move {
                if attempt.index == 0 {
                    Err(not_leader(leader_node, 0))
                } else {
                    Ok::<(), Error>(())
                }
            })
            .await;
        }

        let rt = RwLock::new(RoutingTable::uniform(1, &[1, 2, 3], 3));
        redirect_to(&rt, 2).await;
        assert_eq!(
            rt.read().unwrap().leader_at_term_for_vshard(0).unwrap(),
            (2, 0)
        );

        let group_id = rt.read().unwrap().group_for_vshard(0).unwrap();
        assert!(rt.write().unwrap().observe_leader(group_id, 3, 7));
        redirect_to(&rt, 2).await;
        assert_eq!(
            rt.read().unwrap().leader_at_term_for_vshard(0).unwrap(),
            (3, 7)
        );
    }

    /// Every attempt carries the one statement deadline, so the budget left
    /// shrinks across attempts and never restarts.
    #[tokio::test(start_paused = true)]
    async fn the_statement_deadline_is_shared_across_attempts() {
        let deadline = Instant::now() + Duration::from_millis(500);
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let record = Arc::clone(&seen);
        let result = retry_not_leader(None, deadline, pacing(), move |attempt| {
            let record = Arc::clone(&record);
            async move {
                record.lock().unwrap().push(attempt);
                Err::<(), Error>(not_leader(0, 0))
            }
        })
        .await;
        assert!(matches!(result, Err(Error::NotLeader { .. })));

        let seen = seen.lock().unwrap();
        assert!(seen.len() > 1);
        for (index, attempt) in seen.iter().enumerate() {
            assert_eq!(attempt.index, index);
            assert_eq!(attempt.deadline, deadline);
        }
    }

    /// A retry never starts once the statement's deadline has passed. The
    /// refusal of the attempt that ran stays the answer: nothing applied.
    #[tokio::test]
    async fn no_retry_starts_past_the_deadline() {
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let result = retry_not_leader(None, Instant::now(), pacing(), move |_| {
            let c = Arc::clone(&count);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err::<(), Error>(not_leader(7, 3))
            }
        })
        .await;
        assert!(
            matches!(
                result,
                Err(Error::NotLeader {
                    leader_node: 7,
                    leader_term: 3,
                    ..
                })
            ),
            "got {result:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// A lost answer to a read just before the deadline gets one last call
    /// at once. Its verdict, the deadline, ends the statement.
    #[tokio::test(start_paused = true)]
    async fn the_last_call_before_the_deadline_decides_the_verdict() {
        let deadline = Instant::now() + Duration::from_millis(5);
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let result = retry_not_leader(None, deadline, pacing(), move |_| {
            let c = Arc::clone(&count);
            async move {
                if c.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err::<(), Error>(not_leader(0, 0))
                } else {
                    Err(Error::DeadlineExceeded {
                        request_id: crate::types::RequestId::new(0),
                    })
                }
            }
        })
        .await;
        assert!(
            matches!(result, Err(Error::DeadlineExceeded { .. })),
            "got {result:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2, "one last call, no more");
    }

    /// A refusal from the last call ends the statement: no call follows it.
    #[tokio::test(start_paused = true)]
    async fn a_refused_last_call_ends_the_retries() {
        let deadline = Instant::now() + Duration::from_millis(5);
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let result = retry_not_leader(None, deadline, pacing(), move |_| {
            let c = Arc::clone(&count);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err::<(), Error>(not_leader(0, 0))
            }
        })
        .await;
        assert!(
            matches!(result, Err(Error::NotLeader { .. })),
            "got {result:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// An overwritten entry was never applied. Past the deadline it stays a
    /// definite refusal, never an unknown outcome.
    #[tokio::test]
    async fn an_overwritten_entry_past_the_deadline_stays_definite() {
        let result = retry_not_leader(None, Instant::now(), pacing(), |_| async {
            Err::<(), Error>(Error::RetryableLeaderChange {
                group_id: 4,
                log_index: 9,
            })
        })
        .await;
        assert!(
            matches!(
                result,
                Err(Error::RetryableLeaderChange {
                    group_id: 4,
                    log_index: 9
                })
            ),
            "got {result:?}"
        );
    }

    /// The backoff doubles from one Raft tick and stops at the minimum
    /// election timeout.
    #[test]
    fn pacing_follows_the_raft_timing() {
        let tuning = ClusterTransportTuning {
            raft_tick_interval_ms: 10,
            election_timeout_min_ms: 150,
            ..ClusterTransportTuning::default()
        };
        let pacing = RetryPacing::from_tuning(&tuning);
        assert_eq!(pacing.backoff(0), Duration::from_millis(10));
        assert_eq!(pacing.backoff(1), Duration::from_millis(20));
        assert_eq!(pacing.backoff(4), Duration::from_millis(150));
        assert_eq!(pacing.backoff(u32::MAX), Duration::from_millis(150));
    }

    /// A spent deadline still sends a 1 ms hop, so the receiver checks it.
    #[tokio::test]
    async fn remaining_ms_never_reaches_zero() {
        let attempt = Attempt {
            index: 0,
            deadline: Instant::now(),
        };
        assert_eq!(attempt.remaining_ms(), 1);
    }
}
