// SPDX-License-Identifier: BUSL-1.1

//! The scheduled-job dispatcher observes shutdown, caps concurrency, and
//! refuses a job over its result budget.
//!
//! A refusal at the concurrency cap is `DispatchOutcome::OverBudget`, a typed
//! and transient outcome: nothing ran, and the same job spawns once a running
//! job finishes. The scheduler's next tick retries it.

use std::sync::Arc;
use std::time::Duration;

use nodedb::event::scheduler::dispatcher::{DispatchOutcome, JobDispatcher, JobDispatcherConfig};

/// How long the test waits for a finished job to free its slot.
const SLOT_FREE_WAIT: Duration = Duration::from_secs(5);

#[tokio::test]
async fn dispatcher_refuses_over_cap_then_admits_once_a_slot_frees() {
    // Cap concurrency at 2. Of five jobs that run until released, the first
    // two spawn and the other three are refused, neither queued nor spawned.
    let cfg = JobDispatcherConfig {
        max_concurrent_jobs: 2,
        max_result_bytes: 1 << 20,
    };
    let dispatcher = Arc::new(JobDispatcher::new(cfg));
    let (release_tx, release_rx) = tokio::sync::watch::channel(false);

    let outcomes: Vec<DispatchOutcome> = (0..5)
        .map(|_| {
            let mut release = release_rx.clone();
            dispatcher.try_spawn(|_shutdown| async move {
                let _ = release.wait_for(|released| *released).await;
                Ok(())
            })
        })
        .collect();

    let accepted = outcomes
        .iter()
        .filter(|o| matches!(o, DispatchOutcome::Spawned))
        .count();
    let refused = outcomes
        .iter()
        .filter(|o| matches!(o, DispatchOutcome::OverBudget))
        .count();

    assert_eq!(accepted, 2, "only two jobs fit under the concurrency cap");
    assert_eq!(refused, 3, "jobs over the cap are refused, not queued");

    // The refusal is transient: once the running jobs finish, a job spawns.
    release_tx
        .send(true)
        .expect("the running jobs hold the release receiver");
    let freed = tokio::time::timeout(SLOT_FREE_WAIT, async {
        while dispatcher.in_flight() > 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(freed.is_ok(), "the released jobs free their slots");
    let retried = dispatcher.try_spawn(|_shutdown| async move { Ok(()) });
    assert_eq!(
        retried,
        DispatchOutcome::Spawned,
        "a refused job spawns once a slot frees"
    );

    dispatcher.shutdown_and_drain().await;
}

#[tokio::test]
async fn dispatcher_aborts_inflight_jobs_on_shutdown() {
    // Jobs observe the shutdown receiver. On shutdown the dispatcher must
    // signal shutdown, abort stragglers within the drain deadline, and
    // release its Arc<SharedState>-equivalent resources.
    let cfg = JobDispatcherConfig {
        max_concurrent_jobs: 4,
        max_result_bytes: 1 << 20,
    };
    let dispatcher = Arc::new(JobDispatcher::new(cfg));

    for _ in 0..3 {
        let outcome = dispatcher.try_spawn(|mut shutdown| async move {
            let _ = shutdown.changed().await;
            Ok(())
        });
        assert!(matches!(outcome, DispatchOutcome::Spawned));
    }

    assert_eq!(dispatcher.in_flight(), 3);
    dispatcher.shutdown_and_drain().await;
    assert_eq!(
        dispatcher.in_flight(),
        0,
        "all in-flight jobs must drain after shutdown_and_drain"
    );
}

#[tokio::test]
async fn dispatcher_rejects_jobs_whose_result_set_exceeds_memory_ceiling() {
    // The third compounding hazard: `execute_job` may produce unbounded
    // result sets. The dispatcher is the place that enforces a byte ceiling
    // — jobs exceeding it must surface an `OverBudget` outcome back to the
    // history store rather than silently eating memory.
    let cfg = JobDispatcherConfig {
        max_concurrent_jobs: 8,
        max_result_bytes: 1024,
    };
    let dispatcher = Arc::new(JobDispatcher::new(cfg));

    let outcome = dispatcher.try_spawn_with_budget(1 << 20, |_shutdown| async move {
        // Reporting a 1 MiB result set when the ceiling is 1 KiB must fail.
        Ok(())
    });
    assert!(
        matches!(outcome, DispatchOutcome::OverBudget),
        "job whose declared result-set size exceeds the ceiling must be rejected"
    );

    dispatcher.shutdown_and_drain().await;
}
