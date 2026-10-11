// SPDX-License-Identifier: BUSL-1.1

//! A node that takes a forwarded request and never answers costs a statement
//! at most its one budget.
//!
//! The execute gate `exec::before_execute` parks every request that the
//! leader of the collection's group receives. A follower forwards its
//! statements on the collection to that leader.
//!
//! - A read waits only its first attempt's share of the budget, then
//!   retries. The test releases the gate once the retry starts, and the
//!   read returns its row inside the statement timeout.
//! - A write waits its whole budget, then reports an unknown outcome
//!   (`57014`). It is never sent again.
//!
//! Requires `--features failpoints`.

#![cfg(feature = "failpoints")]

use std::path::Path;
use std::time::{Duration, Instant};

use nodedb_cluster::rpc_codec::REPLY_MARGIN;
use nodedb_test_support::fail_point::{FailAction, FailGuard};
use nodedb_types::id::DatabaseId;

use crate::common::cluster_harness::shared_steps::db_detail;
use crate::common::cluster_harness::{GroupLeader, TestCluster, TestClusterNode, wait_for};

const COLLECTION: &str = "hung_exec";
const GATE: &str = "exec::before_execute";
/// Every node's fixed RPC timeout. A read reserves this much plus the reply
/// margin for each attempt after its first.
const RPC_TIMEOUT_SECS: u64 = 3;
/// The read's statement timeout. Its first attempt gets
/// max(12 - 2 × (3 + 2), 3) = 3 s and waits 5 s with the reply margin,
/// which leaves the retry time to answer.
const READ_TIMEOUT: Duration = Duration::from_secs(12);
/// The write's statement timeout. A write waits all of it.
const WRITE_TIMEOUT: Duration = Duration::from_secs(4);
/// How often the test checks the follower's retry count.
const RETRY_POLL: Duration = Duration::from_millis(10);

/// Release the gate at `release` once `node`'s gateway retries past
/// `retries_before`. Gives up after `give_up`, so a read that never retries
/// fails its own assertions instead of hanging the test.
async fn release_on_retry(
    node: &TestClusterNode,
    retries_before: u64,
    release: &Path,
    give_up: Duration,
) {
    let started = Instant::now();
    while node.not_leader_retry_count() <= retries_before && started.elapsed() < give_up {
        tokio::time::sleep(RETRY_POLL).await;
    }
    std::fs::write(release, b"release").expect("release the execute gate");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_silent_leader_costs_a_read_one_retry_and_a_write_its_budget() {
    let cluster = TestCluster::spawn_three_with_rpc_timeout(RPC_TIMEOUT_SECS)
        .await
        .expect("3-node cluster");
    cluster
        .exec_ddl_on_any_leader(&format!(
            "CREATE COLLECTION {COLLECTION} (id TEXT PRIMARY KEY, val INT) \
             WITH (engine='document_strict')"
        ))
        .await
        .unwrap_or_else(|e| panic!("CREATE COLLECTION {COLLECTION}: {e}"));
    wait_for(
        "all 3 nodes see the collection",
        Duration::from_secs(10),
        Duration::from_millis(50),
        || {
            cluster
                .nodes
                .iter()
                .all(|n| n.cached_collection_count() >= 1)
        },
    )
    .await;

    let vshard = nodedb_cluster::routing::vshard_for_collection(
        nodedb_types::CollectionKey::from_bare(DatabaseId::DEFAULT, COLLECTION),
    );
    let group = cluster.nodes[0]
        .shared
        .cluster_routing
        .as_ref()
        .expect("cluster_routing")
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .group_for_vshard(vshard)
        .expect("the collection's group");
    let leader_id = cluster
        .wait_for_group_leader(group, GroupLeader::Any, Duration::from_secs(20))
        .await;
    let leader = cluster
        .nodes
        .iter()
        .position(|n| n.node_id == leader_id)
        .expect("the group's leader is a cluster node");
    let follower = &cluster.nodes[(leader + 1) % 3];

    cluster.nodes[leader]
        .client
        .simple_query(&format!(
            "INSERT INTO {COLLECTION} (id, val) VALUES ('k1', 1)"
        ))
        .await
        .unwrap_or_else(|e| panic!("seed row: {}", db_detail(&e)));
    cluster
        .wait_for_full_apply_convergence(Duration::from_secs(20))
        .await;

    // A read: its first attempt times out, its retry answers.
    follower
        .client
        .simple_query(&format!(
            "SET statement_timeout = '{}s'",
            READ_TIMEOUT.as_secs()
        ))
        .await
        .expect("SET statement_timeout for the read");
    let read_dir = tempfile::tempdir().expect("read gate directory");
    let read_release = read_dir.path().join("release");
    let read_gate = FailGuard::for_node(
        leader_id,
        GATE,
        FailAction::WaitForFile(read_release.clone()),
    );
    let retries_before = follower.not_leader_retry_count();
    let read_sql = format!("SELECT id, val FROM {COLLECTION} WHERE id = 'k1'");
    let started = Instant::now();
    let (read, ()) = tokio::join!(
        follower.client.simple_query(&read_sql),
        release_on_retry(follower, retries_before, &read_release, READ_TIMEOUT * 2),
    );
    let read_elapsed = started.elapsed();
    drop(read_gate);

    let rows = read
        .unwrap_or_else(|e| panic!("the read retries past the silent leader: {}", db_detail(&e)))
        .into_iter()
        .filter(|m| matches!(m, tokio_postgres::SimpleQueryMessage::Row(_)))
        .count();
    assert_eq!(rows, 1, "the retried read returns the seeded row");
    assert!(
        read_dir.path().join("release.parked").exists(),
        "the leader took the read's first attempt and held it"
    );
    assert!(
        follower.not_leader_retry_count() > retries_before,
        "the read retried after its first attempt went unanswered"
    );
    assert!(
        read_elapsed < READ_TIMEOUT,
        "the read finished in {read_elapsed:?}, past its {READ_TIMEOUT:?} statement budget"
    );

    // A write: it waits its whole budget and is never sent again.
    follower
        .client
        .simple_query(&format!(
            "SET statement_timeout = '{}s'",
            WRITE_TIMEOUT.as_secs()
        ))
        .await
        .expect("SET statement_timeout for the write");
    let write_dir = tempfile::tempdir().expect("write gate directory");
    let write_release = write_dir.path().join("release");
    let write_gate = FailGuard::for_node(
        leader_id,
        GATE,
        FailAction::WaitForFile(write_release.clone()),
    );
    let retries_before = follower.not_leader_retry_count();
    let started = Instant::now();
    let write = follower
        .client
        .simple_query(&format!(
            "INSERT INTO {COLLECTION} (id, val) VALUES ('k2', 2)"
        ))
        .await;
    let write_elapsed = started.elapsed();
    std::fs::write(&write_release, b"release").expect("release the execute gate");
    drop(write_gate);

    let err = write.expect_err("a write whose answer never arrives is an unknown outcome");
    assert_eq!(
        err.as_db_error().map(|db| db.code().code()),
        Some("57014"),
        "the write reports its deadline: {}",
        db_detail(&err)
    );
    assert!(
        write_dir.path().join("release.parked").exists(),
        "the leader took the write and held it"
    );
    assert_eq!(
        follower.not_leader_retry_count(),
        retries_before,
        "a write with a lost answer is never sent again"
    );
    assert!(
        write_elapsed >= WRITE_TIMEOUT,
        "the write waited {write_elapsed:?}, less than its {WRITE_TIMEOUT:?} budget"
    );
    assert!(
        write_elapsed < (WRITE_TIMEOUT + REPLY_MARGIN) * 2,
        "the write waited {write_elapsed:?}, more than one attempt's wait"
    );

    cluster.shutdown().await;
}
