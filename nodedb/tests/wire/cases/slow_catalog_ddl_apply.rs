// SPDX-License-Identifier: BUSL-1.1

//! Compiled only with `--features failpoints`.
//!
//! A catalog DDL whose apply runs longer than a few seconds still succeeds.
//! The proposer waits for its apply until the statement deadline, so a slow
//! apply is never reported as an error while the deadline holds.

#[cfg(feature = "failpoints")]
use crate::harness::TestServer;

/// How long the armed fail point holds the collection's apply.
#[cfg(feature = "failpoints")]
const APPLY_PAUSE_MS: u64 = 6_000;

#[cfg(feature = "failpoints")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_create_collection_with_a_six_second_apply_succeeds() {
    let server = TestServer::start_with_failpoints(&format!(
        "catalog_ddl::before_apply::slow_apply=sleep({APPLY_PAUSE_MS})"
    ))
    .await;

    let started = std::time::Instant::now();
    server
        .exec(
            "CREATE COLLECTION slow_apply (id TEXT PRIMARY KEY, val INT) \
             WITH (engine='document_strict')",
        )
        .await
        .expect("CREATE COLLECTION must succeed while its apply runs inside the deadline");
    assert!(
        started.elapsed() >= std::time::Duration::from_millis(APPLY_PAUSE_MS),
        "the fail point did not hold the apply, so this test proves nothing: {:?}",
        started.elapsed()
    );

    server
        .exec("INSERT INTO slow_apply (id, val) VALUES ('a', 1)")
        .await
        .expect("the created collection takes writes");
    let rows = server
        .query_rows("SELECT val FROM slow_apply WHERE id = 'a'")
        .await
        .expect("the created collection serves reads");
    assert_eq!(rows, vec![vec!["1".to_string()]]);
}
