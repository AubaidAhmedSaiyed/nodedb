// SPDX-License-Identifier: BUSL-1.1

//! An `ivf_pq` vector index buffers vectors until it holds its training
//! threshold, `max(IVF_CELLS, 256)`, then trains and serves through IVF-PQ.
//!
//! - Below the threshold every insert succeeds and search finds each vector
//!   exactly.
//! - Past the threshold search still finds every vector once.
//! - A WAL-only restart keeps every vector searchable and the index trained.
//! - `SHOW VECTOR INDEX` reports the threshold and the training.

use crate::harness::TestServer;

const COLL: &str = "ivf_docs";
/// Vectors held once the threshold is crossed.
const PAST_THRESHOLD: usize = 300;

/// Distinct per `i`: the first component is `i + 1`.
fn vector(i: usize) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for (slot, m) in v.iter_mut().zip([1usize, 7, 11, 13, 17, 19, 23, 29]) {
        *slot = (if m == 1 { i + 1 } else { i % m + 1 }) as f32;
    }
    v
}

fn array(v: &[f32; 8]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:.1}")).collect();
    format!("[{}]", parts.join(", "))
}

async fn insert_range(srv: &TestServer, range: std::ops::Range<usize>) {
    for i in range {
        srv.exec(&format!(
            "INSERT INTO {COLL} {{ id: 'v{i}', embedding: {} }}",
            array(&vector(i))
        ))
        .await
        .unwrap_or_else(|e| panic!("insert of v{i} must succeed: {e}"));
    }
}

async fn nearest(srv: &TestServer, i: usize) -> String {
    let rows = srv
        .query_rows(&format!(
            "SELECT id FROM {COLL} ORDER BY vector_distance(embedding, ARRAY{}) LIMIT 1",
            array(&vector(i))
        ))
        .await
        .unwrap();
    assert_eq!(rows.len(), 1, "search for v{i} must return a row");
    rows[0][0].clone()
}

/// Every id the index returns for one wide search, sorted.
async fn all_ids(srv: &TestServer) -> Vec<String> {
    let rows = srv
        .query_rows(&format!(
            "SELECT id FROM {COLL} ORDER BY vector_distance(embedding, ARRAY{}) LIMIT 1000",
            array(&vector(0))
        ))
        .await
        .unwrap();
    let mut ids: Vec<String> = rows.into_iter().map(|r| r[0].clone()).collect();
    ids.sort();
    ids
}

fn expected_ids(n: usize) -> Vec<String> {
    let mut ids: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    ids.sort();
    ids
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ivf_pq_buffers_trains_and_survives_a_restart() {
    let srv = TestServer::start().await;
    srv.exec(&format!(
        "CREATE COLLECTION {COLL} WITH (engine='document_schemaless')"
    ))
    .await
    .unwrap();
    srv.exec(&format!(
        "CREATE VECTOR INDEX idx_{COLL} ON {COLL} (embedding) METRIC l2 DIM 8 \
         INDEX_TYPE ivf_pq PQ_M 4 IVF_CELLS 4 IVF_NPROBE 4"
    ))
    .await
    .unwrap();

    // Below the threshold: the exact buffer answers.
    insert_range(&srv, 0..10).await;
    for i in 0..10 {
        assert_eq!(nearest(&srv, i).await, format!("v{i}"));
    }
    assert_eq!(status(&srv, "ivf_training_threshold").await, "256");
    assert_eq!(status(&srv, "ivf_trained").await, "false");

    // Past the threshold: trained, and every vector found once.
    insert_range(&srv, 10..PAST_THRESHOLD).await;
    assert_eq!(status(&srv, "ivf_trained").await, "true");
    assert_eq!(status(&srv, "ivf_cells").await, "4");
    assert_eq!(all_ids(&srv).await, expected_ids(PAST_THRESHOLD));
    for i in [0, 9, 255, 256, 299] {
        assert_eq!(nearest(&srv, i).await, format!("v{i}"));
    }

    // WAL-only restart: the buffer replays, trains again, and serves.
    let (srv, dir) = srv.take_dir();
    srv.graceful_shutdown().await;
    let (srv2, _dir) = TestServer::open_on_path(dir).await;
    assert_eq!(status(&srv2, "ivf_trained").await, "true");
    assert_eq!(all_ids(&srv2).await, expected_ids(PAST_THRESHOLD));
    for i in [0, 9, 255, 256, 299] {
        assert_eq!(nearest(&srv2, i).await, format!("v{i}"));
    }

    // Inserts after the restart land in the trained index.
    insert_range(&srv2, PAST_THRESHOLD..PAST_THRESHOLD + 5).await;
    assert_eq!(all_ids(&srv2).await, expected_ids(PAST_THRESHOLD + 5));
}
