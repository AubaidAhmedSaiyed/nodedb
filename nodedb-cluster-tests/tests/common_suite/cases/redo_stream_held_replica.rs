// SPDX-License-Identifier: BUSL-1.1

//! A replica that dropped a chunked redo stream holds its group at the
//! entry it cannot apply, never leads the group, and converges by snapshot.
//!
//! A 3-node cluster caps a redo entry at 64 KiB, so one large commit
//! travels as chunk entries and a final entry. Follower `C` of the
//! collection's group:
//! - applies the stream's first chunk and parks the second chunk on its
//!   gate `redo_chunk::before_apply::1`;
//! - receives the commit and one later write in its log while the other
//!   replicas apply them;
//! - unmounts the group, which drops the group's streams on `C`, and mounts
//!   it again from its log. The group then owes a snapshot on `C`;
//! - releases the chunk. Its stream is gone, so the entry holds the group's
//!   apply on `C`. Every snapshot build of the group parks on
//!   `snapshot::before_build::<group>`, so the hold lasts.
//!
//! While it holds, `C` stores no row of the collection, its applied index
//! stays below the held entry, a leadership transfer to it never makes it
//! lead, and a read at `C` returns the full count or waits. Once the build
//! gate opens, a snapshot installs and `C`'s content equals its peers'.
//!
//! Requires `--features failpoints`.

#![cfg(feature = "failpoints")]

use std::time::{Duration, Instant};

use nodedb_raft::ReplicaState;
use nodedb_test_support::fail_point::{FailAction, FailGuard};
use tokio_postgres::SimpleQueryMessage;

use super::calvin_replica_content::{local_content, vshard_of, wait_replicas_agree};
use crate::common::cluster_harness::shared_steps::group_members;
use crate::common::cluster_harness::{
    GroupLeader, TestCluster, TestClusterNode, is_no_serving_leader, wait_for,
};

/// The redo entry limit every node runs: the smallest the config takes.
const ENTRY_LIMIT: usize = 64 * 1024;

/// Rows one statement of the large transaction inserts.
const ROWS_PER_STATEMENT: usize = 100;

/// Statements the large transaction runs. With 400-byte values its redo
/// holds several times `ENTRY_LIMIT`.
const STATEMENTS: usize = 6;

/// Rows the collection holds once every write applied: the large
/// transaction's rows and one later write.
const EXPECTED_ROWS: usize = STATEMENTS * ROWS_PER_STATEMENT + 1;

/// How long each step gets.
const STEP_DEADLINE: Duration = Duration::from_secs(30);

/// Pause between two polls of a step.
const STEP_POLL: Duration = Duration::from_millis(50);

/// How long the hold is watched before a snapshot may build.
const HOLD_WINDOW: Duration = Duration::from_secs(4);

/// How long a read at the held replica gets.
const READ_DEADLINE: Duration = Duration::from_secs(5);

const COLL: &str = "held_redo_stream";

/// The index in `cluster.nodes` of node `node_id`.
fn index_of(cluster: &TestCluster, node_id: u64) -> usize {
    cluster
        .nodes
        .iter()
        .position(|node| node.node_id == node_id)
        .unwrap_or_else(|| panic!("node {node_id} is live"))
}

/// A fresh session on `node`.
async fn session(node: &TestClusterNode) -> tokio_postgres::Client {
    let (client, connection) = tokio_postgres::connect(
        &format!(
            "host=127.0.0.1 port={} user=nodedb dbname=default",
            node.pg_addr.port()
        ),
        tokio_postgres::NoTls,
    )
    .await
    .expect("connect a session");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
}

/// Commit one transaction whose redo passes `ENTRY_LIMIT` several times.
async fn commit_large_transaction(client: &tokio_postgres::Client) {
    let filler = "x".repeat(400);
    client.simple_query("BEGIN").await.expect("BEGIN");
    for statement in 0..STATEMENTS {
        let values: Vec<String> = (0..ROWS_PER_STATEMENT)
            .map(|row| {
                let id = statement * ROWS_PER_STATEMENT + row;
                format!("('r{id}', '{filler}')")
            })
            .collect();
        client
            .simple_query(&format!(
                "INSERT INTO {COLL} (id, value) VALUES {}",
                values.join(", ")
            ))
            .await
            .expect("INSERT inside the transaction");
    }
    client.simple_query("COMMIT").await.expect("COMMIT");
}

/// The commit index of `group_id` in node `node`'s Raft.
fn raft_committed(node: &TestClusterNode, group_id: u64) -> u64 {
    node.shared.multi_raft.get().map_or(0, |multi_raft| {
        multi_raft
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .commit_index_for(group_id)
            .unwrap_or(0)
    })
}

/// Entries of `group_id` node `node`'s Raft handed to its apply loop.
fn raft_delivered(node: &TestClusterNode, group_id: u64) -> u64 {
    node.shared.multi_raft.get().map_or(0, |multi_raft| {
        multi_raft
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .last_applied(group_id)
            .unwrap_or(0)
    })
}

/// What `node`'s replica of `group_id` needs before it takes log entries.
fn replica_state(node: &TestClusterNode, group_id: u64) -> Option<ReplicaState> {
    node.shared.multi_raft.get().and_then(|multi_raft| {
        multi_raft
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .replica_state(group_id)
    })
}

/// Documents of `COLL` `node` stores, read from its own cores.
async fn stored_rows(node: &TestClusterNode) -> usize {
    local_content(node, &[COLL])
        .await
        .get(&vshard_of(COLL))
        .map_or(0, |content| {
            content.keys().filter(|key| key.starts_with("doc:")).count()
        })
}

/// Read `COUNT(*)` at `node`. A count must be the full count. A read that
/// waits past `READ_DEADLINE`, or finds no serving leader, returns no count.
async fn assert_read_is_whole_or_waits(node: &TestClusterNode) {
    let client = session(node).await;
    let count_sql = format!("SELECT COUNT(*) FROM {COLL}");
    let read = client.simple_query(&count_sql);
    match tokio::time::timeout(READ_DEADLINE, read).await {
        Err(_) => {}
        Ok(Err(error)) if is_no_serving_leader(&error) => {}
        Ok(Err(error)) => panic!("node {}: COUNT(*) failed: {error}", node.node_id),
        Ok(Ok(messages)) => {
            let count: Vec<String> = messages
                .iter()
                .filter_map(|message| match message {
                    SimpleQueryMessage::Row(row) => row.get(0).map(str::to_owned),
                    _ => None,
                })
                .collect();
            assert_eq!(
                count,
                vec![EXPECTED_ROWS.to_string()],
                "node {}: a read served while the node's apply holds must count every \
                 committed row",
                node.node_id
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_replica_that_lost_a_redo_stream_holds_its_group_until_a_snapshot() {
    let cluster = TestCluster::spawn_three_with_redo_entry_limit(ENTRY_LIMIT)
        .await
        .expect("3-node cluster");
    cluster
        .exec_ddl_on_any_leader(&format!(
            "CREATE COLLECTION {COLL} (id TEXT PRIMARY KEY, value TEXT) \
             WITH (engine='document_strict')"
        ))
        .await
        .unwrap_or_else(|e| panic!("create {COLL}: {e}"));
    wait_for(
        "every node replicates the collection's group",
        STEP_DEADLINE,
        STEP_POLL,
        || {
            cluster.nodes.iter().all(|node| {
                node.group_id_for_collection(COLL)
                    .is_some_and(|group| node.replicates_data_group(group))
            })
        },
    )
    .await;
    cluster.wait_for_preferred_leaders(STEP_DEADLINE).await;

    let group = cluster.nodes[0]
        .group_id_for_collection(COLL)
        .expect("the collection's data group");
    let leader = cluster
        .wait_for_group_leader(group, GroupLeader::Any, STEP_DEADLINE)
        .await;
    let held_id = group_members(&cluster.nodes[0], group)
        .into_iter()
        .find(|member| *member != leader)
        .expect("the group has a follower");
    let held = &cluster.nodes[index_of(&cluster, held_id)];
    let writer = &cluster.nodes[index_of(&cluster, leader)];

    let gate_dir = tempfile::tempdir().expect("gate directory");
    let chunk_release = gate_dir.path().join("chunk");
    let chunk_parked = gate_dir.path().join("chunk.parked");
    let build_release = gate_dir.path().join("build");
    let chunk_gate = FailGuard::for_node(
        held_id,
        "redo_chunk::before_apply::1",
        FailAction::WaitForFile(chunk_release.clone()),
    );
    let build_gate = FailGuard::install(
        &format!("snapshot::before_build::{group}"),
        FailAction::WaitForFile(build_release.clone()),
    );

    let client = session(writer).await;
    commit_large_transaction(&client).await;
    client
        .simple_query(&format!(
            "INSERT INTO {COLL} (id, value) VALUES ('after', 'later write')"
        ))
        .await
        .expect("a write after the large transaction");
    let leader_applied = raft_committed(writer, group);
    wait_for(
        "the leader applies every committed entry",
        STEP_DEADLINE,
        STEP_POLL,
        || writer.shared.applied_index_watcher(group).current() >= leader_applied,
    )
    .await;

    wait_for(
        "the held node parks the stream's second chunk and its log holds every write",
        STEP_DEADLINE,
        STEP_POLL,
        || chunk_parked.exists() && raft_delivered(held, group) >= leader_applied,
    )
    .await;
    let tracker = held.shared.propose_tracker.get().expect("propose tracker");
    let held_index = tracker
        .applying(group)
        .expect("the held node's group waits behind the parked chunk")
        .log_index;
    assert!(
        held_index < leader_applied,
        "the parked chunk at {held_index} comes before the later write at {leader_applied}"
    );

    held.remount_data_group(group)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        held.shared.redo_chunks.owes_snapshot(group),
        "node {held_id} dropped the group's stream: it owes a snapshot"
    );
    assert_eq!(
        replica_state(held, group),
        Some(ReplicaState::AwaitingSnapshot),
        "node {held_id}'s data diverged: it must not lead"
    );

    std::fs::write(&chunk_release, b"release").expect("release the parked chunk");
    drop(chunk_gate);

    // A transfer to the held node starts its campaign. A refused transfer
    // starts nothing. Either way it must never lead.
    let _ = writer.transfer_group_leadership(group, held_id);
    let watch_until = Instant::now() + HOLD_WINDOW;
    while Instant::now() < watch_until {
        let applied = held.shared.applied_index_watcher(group).current();
        assert!(
            applied < held_index,
            "node {held_id} applied through {applied}, past the held entry at {held_index}"
        );
        assert_eq!(
            tracker.applying(group).map(|entry| entry.log_index),
            Some(held_index),
            "node {held_id}'s group waits behind the held entry"
        );
        assert!(
            !held.leads_group(group),
            "node {held_id} leads group {group} while it awaits a snapshot"
        );
        assert_eq!(
            replica_state(held, group),
            Some(ReplicaState::AwaitingSnapshot)
        );
        tokio::time::sleep(STEP_POLL).await;
    }
    assert_eq!(
        stored_rows(held).await,
        0,
        "node {held_id} stores rows the held stream's final entry or the later write carry"
    );
    assert_read_is_whole_or_waits(held).await;

    std::fs::write(&build_release, b"release").expect("release the snapshot build");
    drop(build_gate);

    wait_for(
        "the held node installs a snapshot and settles its debt",
        STEP_DEADLINE,
        STEP_POLL,
        || {
            !held.shared.redo_chunks.owes_snapshot(group)
                && replica_state(held, group) == Some(ReplicaState::Whole)
                && held.shared.applied_index_watcher(group).current() >= leader_applied
        },
    )
    .await;
    let content = wait_replicas_agree(&cluster, &[COLL]).await;
    let rows = content.get(&vshard_of(COLL)).map_or(0, |stored| {
        stored.keys().filter(|key| key.starts_with("doc:")).count()
    });
    assert_eq!(
        rows, EXPECTED_ROWS,
        "every replica stores every committed row"
    );
    assert_eq!(stored_rows(held).await, EXPECTED_ROWS);

    cluster.shutdown().await;
}
