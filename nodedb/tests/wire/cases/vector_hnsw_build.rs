// SPDX-License-Identifier: BUSL-1.1

//! Sealed vector segments get their HNSW graph from the core's builder
//! thread, and REINDEX rebuilds them through the same path.
//!
//! - Past the seal threshold, every sealed segment reaches built:
//!   `SHOW VECTOR INDEX` reports no building segment, no queued build, and
//!   completed builds. Search finds every vector once.
//! - A restart right after the inserts, with builds still queued or
//!   running, recovers: the segments build again and search is unchanged.
//! - REINDEX keeps every row's identity: each row is still found under its
//!   own id, deleted rows stay gone, no row appears twice, and the PQ codes
//!   are present afterwards.

use std::time::{Duration, Instant};

use crate::harness::TestServer;

const COLL: &str = "hnsw_build";
/// Vectors per sealed segment.
const SEAL: usize = 64;
/// 3 sealed segments of 64 plus 8 vectors in the growing segment.
const ROWS: usize = 200;
const SEALED_SEGMENTS: usize = ROWS / SEAL;

/// Distinct per `i`: the first component is `i + 1`.
fn vector(i: usize) -> String {
    let parts: Vec<String> = [1usize, 7, 11, 13]
        .iter()
        .map(|&m| format!("{:.1}", (if m == 1 { i + 1 } else { i % m + 1 }) as f32))
        .collect();
    format!("[{}]", parts.join(", "))
}

async fn create(srv: &TestServer) {
    srv.exec(&format!(
        "CREATE COLLECTION {COLL} WITH (engine='document_schemaless')"
    ))
    .await
    .unwrap();
    srv.exec(&format!(
        "CREATE VECTOR INDEX idx_{COLL} ON {COLL} (embedding) METRIC l2 DIM 4 \
         INDEX_TYPE hnsw_pq PQ_M 2"
    ))
    .await
    .unwrap();
}

async fn insert_all(srv: &TestServer) {
    for i in 0..ROWS {
        srv.exec(&format!(
            "INSERT INTO {COLL} {{ id: 'v{i}', embedding: {} }}",
            vector(i)
        ))
        .await
        .unwrap_or_else(|e| panic!("insert of v{i} must succeed: {e}"));
    }
}

async fn status(srv: &TestServer, property: &str) -> String {
    let rows = srv
        .query_rows(&format!("SHOW VECTOR INDEX status ON {COLL}.embedding"))
        .await
        .unwrap();
    rows.iter()
        .find(|r| r[0] == property)
        .map(|r| r[1].clone())
        .unwrap_or_else(|| panic!("SHOW VECTOR INDEX must report {property}: {rows:?}"))
}

async fn status_num(srv: &TestServer, property: &str) -> u64 {
    status(srv, property)
        .await
        .parse()
        .unwrap_or_else(|e| panic!("{property} must be a number: {e}"))
}

/// Wait until at least the expected sealed segments exist, none is still
/// building, no build is queued, and at least `completed` builds have been
/// installed.
async fn wait_built(srv: &TestServer, completed: u64) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let sealed = status_num(srv, "sealed_segments").await;
        let building = status_num(srv, "building_segments").await;
        let queued = status_num(srv, "builds_queued").await;
        let done = status_num(srv, "builds_completed").await;
        // A restart re-indexes every document from the store, so it can hold
        // more sealed segments than the first run did.
        if sealed >= SEALED_SEGMENTS as u64 && building == 0 && queued == 0 && done >= completed {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "builds did not finish: sealed={sealed} building={building} queued={queued} \
             completed={done}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Every id a wide search returns, sorted.
async fn all_ids(srv: &TestServer) -> Vec<String> {
    let mut ids: Vec<String> = srv
        .query_rows(&format!(
            "SELECT id FROM {COLL} ORDER BY vector_distance(embedding, ARRAY{}) LIMIT 1000",
            vector(0)
        ))
        .await
        .unwrap()
        .into_iter()
        .map(|r| r[0].clone())
        .collect();
    ids.sort();
    ids
}

fn expected_ids(skip: &[usize]) -> Vec<String> {
    let mut ids: Vec<String> = (0..ROWS)
        .filter(|i| !skip.contains(i))
        .map(|i| format!("v{i}"))
        .collect();
    ids.sort();
    ids
}

/// Assert that a search returned every expected row exactly once. On a
/// mismatch, name the missing, extra and repeated ids instead of printing
/// two long lists.
fn assert_same_ids(actual: Vec<String>, skip: &[usize], context: &str) {
    let expected = expected_ids(skip);
    if actual == expected {
        return;
    }
    let missing: Vec<&String> = expected.iter().filter(|id| !actual.contains(id)).collect();
    let extra: Vec<&String> = actual.iter().filter(|id| !expected.contains(id)).collect();
    let repeated: Vec<&String> = actual
        .iter()
        .enumerate()
        .filter(|(i, id)| actual[..*i].contains(id))
        .map(|(_, id)| id)
        .collect();
    panic!(
        "{context}: {} ids returned, {} expected; missing {missing:?}, \
         not expected {extra:?}, repeated {repeated:?}",
        actual.len(),
        expected.len()
    );
}

async fn nearest(srv: &TestServer, i: usize) -> String {
    let rows = srv
        .query_rows(&format!(
            "SELECT id FROM {COLL} ORDER BY vector_distance(embedding, ARRAY{}) LIMIT 1",
            vector(i)
        ))
        .await
        .unwrap();
    rows[0][0].clone()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sealed_segments_build_and_search_uses_them() {
    let srv = TestServer::start_with_vector_seal_threshold(SEAL).await;
    create(&srv).await;
    insert_all(&srv).await;

    wait_built(&srv, SEALED_SEGMENTS as u64).await;
    // One HNSW node per row: an insert indexed once by the document write
    // and again by a separate vector insert leaves a tombstoned twin per row.
    assert_eq!(status_num(&srv, "live_count").await, ROWS as u64);
    assert_eq!(status_num(&srv, "tombstone_count").await, 0);
    assert_eq!(
        status_num(&srv, "sealed_segments").await,
        SEALED_SEGMENTS as u64
    );
    assert_eq!(
        status_num(&srv, "growing_vectors").await,
        (ROWS % SEAL) as u64
    );
    assert_eq!(status(&srv, "quantization").await, "pq");
    assert_same_ids(all_ids(&srv).await, &[], "after the builds");
    for i in [0, 63, 64, 150, 199] {
        assert_eq!(nearest(&srv, i).await, format!("v{i}"));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restart_during_the_builds_recovers() {
    let srv = TestServer::start_with_vector_seal_threshold(SEAL).await;
    create(&srv).await;
    insert_all(&srv).await;

    // Stop at once: builds are still queued or running.
    let (srv, dir) = srv.take_dir();
    srv.graceful_shutdown().await;
    let (srv2, _dir) = TestServer::open_on_path_with_vector_seal_threshold(dir, SEAL).await;

    wait_built(&srv2, 0).await;
    assert_same_ids(all_ids(&srv2).await, &[], "after the restart");
    for i in [0, 64, 199] {
        assert_eq!(nearest(&srv2, i).await, format!("v{i}"));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reindex_keeps_identities_and_codes() {
    let srv = TestServer::start_with_vector_seal_threshold(SEAL).await;
    create(&srv).await;
    insert_all(&srv).await;
    wait_built(&srv, SEALED_SEGMENTS as u64).await;

    // Tombstones in two sealed segments.
    for i in [5, 70] {
        srv.exec(&format!("DELETE FROM {COLL} WHERE id = 'v{i}'"))
            .await
            .unwrap();
    }
    assert_same_ids(all_ids(&srv).await, &[5, 70], "before REINDEX");
    let done = status_num(&srv, "builds_completed").await;

    srv.exec(&format!("REINDEX CONCURRENTLY {COLL}"))
        .await
        .unwrap();
    wait_built(&srv, done + SEALED_SEGMENTS as u64).await;

    assert_same_ids(
        all_ids(&srv).await,
        &[5, 70],
        "after REINDEX: every row once, deleted rows still gone",
    );
    // Each row is still found at distance 0 under its own id: the rebuilt
    // graphs kept every node id, so every surrogate binding still holds.
    for i in [0, 4, 6, 63, 64, 69, 71, 127, 128, 191, 192, 199] {
        assert_eq!(nearest(&srv, i).await, format!("v{i}"));
    }
    assert_eq!(status(&srv, "quantization").await, "pq");
    assert_eq!(
        status_num(&srv, "growing_vectors").await,
        (ROWS % SEAL) as u64
    );
}
