// SPDX-License-Identifier: BUSL-1.1

//! A live slow node gets a statement's whole budget. A dead node costs it
//! about one RPC timeout.
//!
//! A follower forwards its statements on the collection to the leader of the
//! collection's group. The leader's execute gate `exec::before_execute`
//! slows or parks the requests it receives:
//!
//! - A slow leader keeps its connection alive, so a read waits for it and
//!   returns its row with no retry.
//! - A parked leader never answers. A read and a write each wait their whole
//!   budget and end `57014`, with no retry.
//! - A leader cut off by `transport::blackhole` goes silent. Its connection
//!   hits the QUIC idle timeout, the read retries on the new leader, and it
//!   returns its row within two RPC timeouts.
//!
//! Requires `--features failpoints`.

#![cfg(feature = "failpoints")]

use std::time::{Duration, Instant};

use nodedb_cluster::rpc_codec::REPLY_MARGIN;
use nodedb_cluster::transport::endpoint::BLACKHOLE_FAIL_POINT;
use nodedb_test_support::fail_point::{FailAction, FailGuard};
use nodedb_types::id::DatabaseId;

use crate::common::cluster_harness::shared_steps::db_detail;
use crate::common::cluster_harness::{GroupLeader, TestCluster, TestClusterNode, wait_for};

const COLLECTION: &str = "hung_exec";
const GATE: &str = "exec::before_execute";
/// Every node's RPC timeout, and so its QUIC idle timeout.
const RPC_TIMEOUT: Duration = Duration::from_secs(3);
/// How long a slow leader holds each request it receives.
const SLOW_EXECUTE: Duration = Duration::from_secs(8);
/// The statement timeout of a read sent to a slow leader.
const SLOW_READ_TIMEOUT: Duration = Duration::from_secs(12);
/// The statement timeout of a request sent to a parked leader.
const PARKED_TIMEOUT: Duration = Duration::from_secs(4);
/// The statement timeout of a read sent to a silent leader.
const SILENT_READ_TIMEOUT: Duration = Duration::from_secs(20);
/// Scheduling slack on the upper bounds.
const SLACK: Duration = Duration::from_secs(2);

/// A 3-node cluster with one seeded row, the collection's group leader, and
/// a follower that forwards to it.
struct Setup {
    cluster: TestCluster,
    leader_id: u64,
    follower: usize,
}

async fn setup() -> Setup {
    let cluster = TestCluster::spawn_three_with_rpc_timeout(RPC_TIMEOUT.as_secs())
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
    Setup {
        cluster,
        leader_id,
        follower: (leader + 1) % 3,
    }
}

async fn set_statement_timeout(node: &TestClusterNode, timeout: Duration) {
    node.client
        .simple_query(&format!(
            "SET statement_timeout = '{}ms'",
            timeout.as_millis()
        ))
        .await
        .expect("SET statement_timeout");
}

fn read_sql() -> String {
    format!("SELECT id, val FROM {COLLECTION} WHERE id = 'k1'")
}

fn row_count(messages: Vec<tokio_postgres::SimpleQueryMessage>) -> usize {
    messages
        .into_iter()
        .filter(|m| matches!(m, tokio_postgres::SimpleQueryMessage::Row(_)))
        .count()
}

fn sqlstate(err: &tokio_postgres::Error) -> Option<&str> {
    err.as_db_error().map(|db| db.code().code())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_slow_leader_gets_a_read_its_whole_budget() {
    let Setup {
        cluster,
        leader_id,
        follower,
    } = setup().await;
    let follower = &cluster.nodes[follower];
    set_statement_timeout(follower, SLOW_READ_TIMEOUT).await;

    let gate = FailGuard::for_node(leader_id, GATE, FailAction::Sleep(SLOW_EXECUTE));
    let retries_before = follower.not_leader_retry_count();
    let started = Instant::now();
    let read = follower.client.simple_query(&read_sql()).await;
    let elapsed = started.elapsed();
    drop(gate);

    let rows = read.unwrap_or_else(|e| panic!("the slow leader answers: {}", db_detail(&e)));
    assert_eq!(row_count(rows), 1, "the read returns the seeded row");
    assert_eq!(
        follower.not_leader_retry_count(),
        retries_before,
        "a live slow leader is never retried"
    );
    assert!(
        elapsed >= SLOW_EXECUTE,
        "the read returned in {elapsed:?}, before the leader ran it"
    );
    assert!(
        elapsed < SLOW_READ_TIMEOUT,
        "the read finished in {elapsed:?}, past its {SLOW_READ_TIMEOUT:?} budget"
    );

    cluster.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_parked_leader_costs_a_read_its_budget() {
    let Setup {
        cluster,
        leader_id,
        follower,
    } = setup().await;
    let follower = &cluster.nodes[follower];
    set_statement_timeout(follower, PARKED_TIMEOUT).await;

    // A read: the live leader holds it, so it waits its whole budget.
    let read_dir = tempfile::tempdir().expect("read gate directory");
    let read_release = read_dir.path().join("release");
    let read_gate = FailGuard::for_node(
        leader_id,
        GATE,
        FailAction::WaitForFile(read_release.clone()),
    );
    let retries_before = follower.not_leader_retry_count();
    let started = Instant::now();
    let read = follower.client.simple_query(&read_sql()).await;
    let read_elapsed = started.elapsed();
    std::fs::write(&read_release, b"release").expect("release the execute gate");
    drop(read_gate);

    let err = read.expect_err("a read the leader never answers runs out of budget");
    assert_eq!(
        sqlstate(&err),
        Some("57014"),
        "the read reports its deadline: {}",
        db_detail(&err)
    );
    assert!(
        read_dir.path().join("release.parked").exists(),
        "the leader took the read and held it"
    );
    assert_eq!(
        follower.not_leader_retry_count(),
        retries_before,
        "a live leader that holds a read is never retried"
    );
    assert!(
        read_elapsed >= PARKED_TIMEOUT,
        "the read waited {read_elapsed:?}, less than its {PARKED_TIMEOUT:?} budget"
    );
    assert!(
        read_elapsed < PARKED_TIMEOUT + REPLY_MARGIN + SLACK,
        "the read waited {read_elapsed:?}, past its budget and reply margin"
    );

    // A write: it waits its whole budget and is never sent again.
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
        sqlstate(&err),
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
        write_elapsed >= PARKED_TIMEOUT,
        "the write waited {write_elapsed:?}, less than its {PARKED_TIMEOUT:?} budget"
    );
    assert!(
        write_elapsed < PARKED_TIMEOUT + REPLY_MARGIN + SLACK,
        "the write waited {write_elapsed:?}, past its budget and reply margin"
    );

    cluster.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_silent_leader_costs_a_read_one_idle_timeout() {
    let Setup {
        cluster,
        leader_id,
        follower,
    } = setup().await;
    let follower = &cluster.nodes[follower];
    set_statement_timeout(follower, SILENT_READ_TIMEOUT).await;

    // The leader drops every datagram, both ways, while its process runs.
    let blackhole = FailGuard::for_node(
        leader_id,
        BLACKHOLE_FAIL_POINT,
        FailAction::Fail("cut off".into()),
    );
    let retries_before = follower.not_leader_retry_count();
    let started = Instant::now();
    let read = follower.client.simple_query(&read_sql()).await;
    let elapsed = started.elapsed();
    drop(blackhole);

    let rows = read
        .unwrap_or_else(|e| panic!("the read retries past the silent leader: {}", db_detail(&e)));
    assert_eq!(
        row_count(rows),
        1,
        "the retried read returns the seeded row"
    );
    assert!(
        follower.not_leader_retry_count() > retries_before,
        "the read retried after the silent leader's connection timed out"
    );
    assert!(
        elapsed < RPC_TIMEOUT * 2,
        "the read took {elapsed:?}, more than two {RPC_TIMEOUT:?} RPC timeouts"
    );

    cluster.shutdown().await;
}
