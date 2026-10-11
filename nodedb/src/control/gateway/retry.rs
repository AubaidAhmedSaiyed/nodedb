// SPDX-License-Identifier: BUSL-1.1

//! Typed `NotLeader` retry with 3-attempt budget + 50/100/200 ms backoff.
//!
//! When a remote dispatch returns `Error::NotLeader`, the retry helper:
//! 1. Extracts the hinted new leader and its term from the error.
//! 2. Offers them to the routing table entry for the affected group. The
//!    hint moves only when the redirect's term is above the hint's term, so
//!    a stale redirect never undoes a newer leader.
//! 3. Sleeps for the appropriate backoff duration.
//! 4. Re-invokes the closure.
//!
//! If the hinted leader is unknown (no hint), we still retry after sleep
//! without updating the routing table — a subsequent routing lookup will
//! re-read the table from the current routing state.
//!
//! Every attempt shares the statement's deadline, so the attempts together
//! stay within one statement budget. No retry starts once it has passed. The
//! last refusal then ends the statement: it is definite, so it never becomes
//! `DeadlineExceeded`, which means a lost answer.
//!
//! After `MAX_RETRIES` attempts the final `NotLeader` error is propagated.

use std::future::Future;
use std::sync::RwLock;

use tokio::time::{Duration, Instant, sleep};
use tracing::debug;

use nodedb_cluster::RoutingTable;

use crate::Error;
use crate::control::server::shared::session::deadline::hop_budget_ms;

/// Maximum number of dispatch attempts (initial + 2 retries = 3 total).
pub const MAX_RETRIES: usize = 3;

/// Backoff durations for each retry attempt.
const BACKOFF_MS: [u64; MAX_RETRIES] = [50, 100, 200];

/// One dispatch attempt of a statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attempt {
    /// The attempt's 0-based index.
    pub index: usize,
    /// The statement's deadline. Every attempt carries the same instant.
    pub deadline: Instant,
}

impl Attempt {
    /// The attempts left, this one included.
    pub fn attempts_left(&self) -> usize {
        MAX_RETRIES.saturating_sub(self.index)
    }

    /// The milliseconds left on the statement now, at least 1. See
    /// [`hop_budget_ms`].
    pub fn remaining_ms(&self) -> u64 {
        hop_budget_ms(self.deadline.into_std())
    }

    /// The one attempt of a send that no retry follows.
    pub fn sole(deadline: Instant) -> Self {
        Self {
            index: MAX_RETRIES - 1,
            deadline,
        }
    }
}

/// Execute `f` up to `MAX_RETRIES` times, retrying on `Error::NotLeader`.
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
    f: F,
) -> Result<T, Error>
where
    F: Fn(Attempt) -> Fut,
    Fut: Future<Output = Result<T, Error>>,
{
    let mut attempt = 0;
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
                // The Raft entry the proposer was waiting on was
                // overwritten by a new leader's election no-op. Re-
                // propose: the next attempt will land at a fresh log
                // index under the new leader's term. Same backoff
                // schedule as NotLeader — bounded retries so a
                // pathological cluster doesn't loop forever.
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

        // Both refusals mean nothing applied. When no retry follows, that
        // definite refusal ends the statement, never an unknown outcome.
        attempt += 1;
        if attempt >= MAX_RETRIES {
            return Err(refusal);
        }
        sleep(Duration::from_millis(BACKOFF_MS[attempt - 1])).await;
        // A retry starts only while the statement has time left. That keeps
        // every attempt's wait within the budget plus one reply margin.
        if Instant::now() >= deadline {
            debug!(
                attempt,
                "gateway: statement deadline passed — no further retry"
            );
            return Err(refusal);
        }
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

    #[tokio::test]
    async fn success_on_first_attempt() {
        let result = retry_not_leader(None, far_deadline(), |_attempt| async {
            Ok::<u32, Error>(42)
        })
        .await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn success_on_second_attempt() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&call_count);
        let result = retry_not_leader(None, far_deadline(), move |_attempt| {
            let c = Arc::clone(&count);
            async move {
                let n = c.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err(Error::NotLeader {
                        vshard_id: VShardId::new(0),
                        leader_node: 2,
                        leader_addr: "10.0.0.2:9400".into(),
                        leader_term: 1,
                    })
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
    async fn exhausts_retries_returns_not_leader() {
        let result = retry_not_leader(None, far_deadline(), |_| async {
            Err::<u32, Error>(Error::NotLeader {
                vshard_id: VShardId::new(1),
                leader_node: 0,
                leader_addr: String::new(),
                leader_term: 0,
            })
        })
        .await;
        assert!(matches!(result, Err(Error::NotLeader { .. })));
    }

    #[tokio::test]
    async fn non_not_leader_error_propagates_immediately() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&call_count);
        let result = retry_not_leader(None, far_deadline(), move |_| {
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

    #[tokio::test]
    async fn routing_table_updated_on_not_leader_hint() {
        let table = RoutingTable::uniform(1, &[1, 2], 2);
        let rt = Arc::new(RwLock::new(table));
        let rt_clone = Arc::clone(&rt);

        let call_count = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&call_count);

        let _ = retry_not_leader(Some(&*rt_clone), far_deadline(), move |_| {
            let c = Arc::clone(&count);
            async move {
                let n = c.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err(Error::NotLeader {
                        vshard_id: VShardId::new(0),
                        leader_node: 2,
                        leader_addr: "addr".into(),
                        leader_term: 4,
                    })
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

        let _ = retry_not_leader(Some(&rt), far_deadline(), |attempt| async move {
            if attempt.index == 0 {
                Err(Error::NotLeader {
                    vshard_id: VShardId::new(0),
                    leader_node: 2,
                    leader_addr: "addr".into(),
                    leader_term: 5,
                })
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
            let _ = retry_not_leader(Some(rt), far_deadline(), |attempt| async move {
                if attempt.index == 0 {
                    Err(Error::NotLeader {
                        vshard_id: VShardId::new(0),
                        leader_node,
                        leader_addr: String::new(),
                        leader_term: 0,
                    })
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
    #[tokio::test]
    async fn the_statement_deadline_is_shared_across_attempts() {
        let deadline = far_deadline();
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let record = Arc::clone(&seen);
        let result = retry_not_leader(None, deadline, move |attempt| {
            let record = Arc::clone(&record);
            async move {
                record.lock().unwrap().push((
                    attempt,
                    attempt.attempts_left(),
                    attempt.remaining_ms(),
                ));
                Err::<(), Error>(Error::NotLeader {
                    vshard_id: VShardId::new(0),
                    leader_node: 0,
                    leader_addr: String::new(),
                    leader_term: 0,
                })
            }
        })
        .await;
        assert!(matches!(result, Err(Error::NotLeader { .. })));

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), MAX_RETRIES);
        for (index, (attempt, attempts_left, _)) in seen.iter().enumerate() {
            assert_eq!(attempt.index, index);
            assert_eq!(attempt.deadline, deadline);
            assert_eq!(*attempts_left, MAX_RETRIES - index);
        }
        let first_remaining = seen[0].2;
        let last_remaining = seen[MAX_RETRIES - 1].2;
        let backoff: u64 = BACKOFF_MS[..MAX_RETRIES - 1].iter().sum();
        assert!(
            last_remaining <= first_remaining - backoff,
            "the last attempt has {last_remaining} ms, the first had {first_remaining} ms"
        );
    }

    /// A retry never starts once the statement's deadline has passed. The
    /// refusal of the attempt that ran stays the answer: nothing applied.
    #[tokio::test]
    async fn no_retry_starts_past_the_deadline() {
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let result = retry_not_leader(None, Instant::now(), move |_| {
            let c = Arc::clone(&count);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err::<(), Error>(Error::NotLeader {
                    vshard_id: VShardId::new(0),
                    leader_node: 7,
                    leader_addr: String::new(),
                    leader_term: 3,
                })
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

    /// An overwritten entry was never applied. Past the deadline it stays a
    /// definite refusal, never an unknown outcome.
    #[tokio::test]
    async fn an_overwritten_entry_past_the_deadline_stays_definite() {
        let result = retry_not_leader(None, Instant::now(), |_| async {
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

    /// A sole attempt leaves no retry behind it.
    #[test]
    fn a_sole_attempt_has_one_attempt_left() {
        let deadline = far_deadline();
        let attempt = Attempt::sole(deadline);
        assert_eq!(attempt.attempts_left(), 1);
        assert_eq!(attempt.deadline, deadline);
    }

    /// A spent deadline still sends a 1 ms hop, so the receiver checks it.
    #[tokio::test]
    async fn remaining_ms_never_reaches_zero() {
        let attempt = Attempt::sole(Instant::now());
        assert_eq!(attempt.remaining_ms(), 1);
    }
}
