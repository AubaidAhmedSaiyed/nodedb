// SPDX-License-Identifier: BUSL-1.1

//! REINDEX must keep every write, and must rebuild every index it names.
//!
//! - A full-text or CSR rebuild snapshots its index, builds off the core,
//!   then swaps the result in. Rows written, updated or deleted between the
//!   snapshot and the swap must show in the swapped-in index exactly as
//!   they do in the rows. A failpoint holds the rebuild thread so the
//!   writes land inside that window.
//! - REINDEX with no index name rebuilds every index kind the collection
//!   has: HNSW, full-text and CSR. Both the concurrent and the plain form
//!   are covered.
//! - Plain REINDEX rebuilds off the core like the concurrent form and
//!   answers only after the cutover. While its rebuild is held, the same
//!   core still answers queries on other collections.
//!
//! Each rebuild reports on the `nodedb::reindex` tracing target:
//! `rebuild_started` when it starts, then `atomic_cutover` when it is
//! swapped in or `rebuild_refused` when it is discarded. The tests count
//! those events.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use nodedb_test_support::pgwire_harness::TestServer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

// ── Event recorder ────────────────────────────────────────────────────────────

/// `(message, index, scope)` of every `nodedb::reindex` event. `scope` is
/// the event's `collection` field, or its `key` field for HNSW.
#[derive(Default)]
struct EventLog(Mutex<Vec<(String, String, String)>>);

impl EventLog {
    fn count(&self, message: &str, index: &str, collection: &str) -> usize {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|(m, i, s)| m == message && i == index && s.contains(collection))
            .count()
    }
}

struct Recorder(Arc<EventLog>);

#[derive(Default)]
struct Fields {
    message: String,
    index: String,
    scope: String,
}

impl Fields {
    fn record_text(&mut self, field: &tracing::field::Field, text: &str) {
        match field.name() {
            "message" => self.message = text.to_string(),
            "index" => self.index = text.to_string(),
            "collection" | "key" => self.scope = text.to_string(),
            _ => {}
        }
    }
}

impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let text = format!("{value:?}");
        self.record_text(field, text.trim_matches('"'));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.record_text(field, value);
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Recorder {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if event.metadata().target() != "nodedb::reindex" {
            return;
        }
        let mut fields = Fields::default();
        event.record(&mut fields);
        self.0
            .0
            .lock()
            .unwrap()
            .push((fields.message, fields.index, fields.scope));
    }
}

/// The process-wide event log. Installed before the first server starts.
fn events() -> Arc<EventLog> {
    static LOG: OnceLock<Arc<EventLog>> = OnceLock::new();
    Arc::clone(LOG.get_or_init(|| {
        let log = Arc::new(EventLog::default());
        tracing_subscriber::registry()
            .with(Recorder(Arc::clone(&log)))
            .try_init()
            .expect("the reindex event recorder must be the process's tracing subscriber");
        log
    }))
}

/// Wait until every started `index` rebuild of `collection` has been swapped
/// in, and fail on any rebuild that was discarded instead.
async fn wait_for_cutovers(log: &EventLog, index: &str, collection: &str) {
    let started = log.count("rebuild_started", index, collection);
    assert!(
        started > 0,
        "REINDEX must start a {index} rebuild of {collection}"
    );
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let refused = log.count("rebuild_refused", index, collection);
        assert_eq!(
            refused, 0,
            "a {index} rebuild of {collection} was discarded instead of swapped in"
        );
        let cutovers = log.count("atomic_cutover", index, collection);
        if cutovers >= started {
            assert_eq!(cutovers, started, "one cutover per started {index} rebuild");
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{index} rebuild of {collection} did not cut over: {cutovers} of {started}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Ids a full-text match on `body` returns, sorted.
async fn text_ids(server: &TestServer, collection: &str, term: &str) -> Vec<String> {
    let mut ids = server
        .query_text(&format!(
            "SELECT id FROM {collection} WHERE text_match(body, '{term}')"
        ))
        .await
        .unwrap();
    ids.sort();
    ids
}

/// Nodes one hop out of `root` along label `l`, as one text blob.
async fn out_of_root(server: &TestServer, collection: &str) -> String {
    server
        .query_text_joined(&format!(
            "GRAPH TRAVERSE IN '{collection}' FROM 'root' DEPTH 1 LABEL 'l' DIRECTION out"
        ))
        .await
        .unwrap()
        .join("\n")
}

async fn insert_edge(server: &TestServer, collection: &str, dst: &str) {
    server
        .exec(&format!(
            "GRAPH INSERT EDGE IN '{collection}' FROM 'root' TO '{dst}' TYPE 'l'"
        ))
        .await
        .unwrap();
}

// ── Writes during a held rebuild ──────────────────────────────────────────────

#[cfg(feature = "failpoints")]
mod held {
    use nodedb::fail_point::{FailAction, FailGuard};

    use super::*;

    /// Arm `failpoint` to hold its rebuild thread until the returned path
    /// exists.
    fn hold(failpoint: &str) -> (FailGuard, tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let release = dir.path().join("release");
        let guard = FailGuard::install(failpoint, FailAction::WaitForFile(release.clone()));
        (guard, dir, release)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fts_rebuild_keeps_writes_made_during_the_rebuild() {
        let log = events();
        let server = TestServer::start().await;
        server
            .exec("CREATE COLLECTION rx_fts WITH (engine='document_schemaless')")
            .await
            .unwrap();
        for i in 1..=4 {
            server
                .exec(&format!(
                    "INSERT INTO rx_fts {{ id: 'd{i}', body: 'alpha seed' }}"
                ))
                .await
                .unwrap();
        }

        let (_guard, _dir, release) = hold("reindex::fts_build_hold");
        server
            .exec("REINDEX INDEX fts CONCURRENTLY rx_fts")
            .await
            .unwrap();
        assert!(log.count("rebuild_started", "fts", "rx_fts") > 0);

        // The rebuild's snapshot is pinned; these writes land after it.
        for i in 5..=7 {
            server
                .exec(&format!(
                    "INSERT INTO rx_fts {{ id: 'd{i}', body: 'alpha fresh' }}"
                ))
                .await
                .unwrap();
        }
        server
            .exec("UPDATE rx_fts SET body = 'beta moved' WHERE id = 'd1'")
            .await
            .unwrap();
        server
            .exec("DELETE FROM rx_fts WHERE id = 'd2'")
            .await
            .unwrap();
        assert_eq!(
            log.count("atomic_cutover", "fts", "rx_fts"),
            0,
            "the held rebuild must not cut over before it is released"
        );

        std::fs::write(&release, b"").unwrap();
        wait_for_cutovers(&log, "fts", "rx_fts").await;

        assert_eq!(
            text_ids(&server, "rx_fts", "alpha").await,
            ["d3", "d4", "d5", "d6", "d7"],
            "rows inserted during the rebuild are found once, the updated row \
             lost the term, and the deleted row is gone"
        );
        assert_eq!(text_ids(&server, "rx_fts", "beta").await, ["d1"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn csr_rebuild_keeps_writes_made_during_the_rebuild() {
        let log = events();
        let server = TestServer::start().await;
        server.exec("CREATE COLLECTION rx_csr").await.unwrap();
        for dst in ["keep0", "gone1", "keep2"] {
            insert_edge(&server, "rx_csr", dst).await;
        }

        let (_guard, _dir, release) = hold("reindex::csr_build_hold");
        server
            .exec("REINDEX INDEX csr CONCURRENTLY rx_csr")
            .await
            .unwrap();
        assert!(log.count("rebuild_started", "csr", "rx_csr") > 0);

        // The partition snapshot is taken; these writes land after it.
        insert_edge(&server, "rx_csr", "new3").await;
        insert_edge(&server, "rx_csr", "new4").await;
        server
            .exec("GRAPH DELETE EDGE IN 'rx_csr' FROM 'root' TO 'gone1' TYPE 'l'")
            .await
            .unwrap();
        assert_eq!(
            log.count("atomic_cutover", "csr", "rx_csr"),
            0,
            "the held rebuild must not cut over before it is released"
        );

        std::fs::write(&release, b"").unwrap();
        wait_for_cutovers(&log, "csr", "rx_csr").await;

        let blob = out_of_root(&server, "rx_csr").await;
        for kept in ["keep0", "keep2", "new3", "new4"] {
            assert!(
                blob.contains(kept),
                "traversal after the cutover must reach {kept}; got: {blob}"
            );
        }
        assert!(
            !blob.contains("gone1"),
            "an edge deleted during the rebuild must stay deleted; got: {blob}"
        );
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn plain_reindex_leaves_the_core_serving_while_its_rebuild_is_held() {
        let log = events();
        let server = TestServer::start().await;
        server
            .exec("CREATE COLLECTION rx_held WITH (engine='document_schemaless')")
            .await
            .unwrap();
        server
            .exec("INSERT INTO rx_held { id: 'h1', body: 'alpha held' }")
            .await
            .unwrap();
        server
            .exec("CREATE COLLECTION rx_other WITH (engine='document_schemaless')")
            .await
            .unwrap();
        server
            .exec("INSERT INTO rx_other { id: 'o1', body: 'other row' }")
            .await
            .unwrap();

        let (_guard, _dir, release) = hold("reindex::fts_build_hold");

        // The plain REINDEX runs on its own connection: it answers only
        // after the cutover, and the cutover waits for the release.
        let conn_str = format!(
            "host=127.0.0.1 port={} user=nodedb dbname=default",
            server.pg_port
        );
        let (client, conn) = tokio_postgres::connect(&conn_str, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = conn.await;
        });
        let reindex = tokio::spawn(async move {
            client
                .simple_query("REINDEX INDEX fts rx_held")
                .await
                .map(|_| ())
                .map_err(|e| e.to_string())
        });

        let deadline = Instant::now() + Duration::from_secs(30);
        while log.count("rebuild_started", "fts", "rx_held") == 0 {
            assert!(Instant::now() < deadline, "the plain REINDEX never started");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // The core that holds the rebuild answers other requests.
        let rows = tokio::time::timeout(
            Duration::from_secs(10),
            server.query_text("SELECT id FROM rx_other"),
        )
        .await
        .expect("a query on another collection must answer while the rebuild is held")
        .unwrap();
        assert_eq!(rows, ["o1"]);
        assert!(
            !reindex.is_finished(),
            "plain REINDEX must not answer before its cutover"
        );
        assert_eq!(log.count("atomic_cutover", "fts", "rx_held"), 0);

        std::fs::write(&release, b"").unwrap();
        tokio::time::timeout(Duration::from_secs(60), reindex)
            .await
            .expect("plain REINDEX must answer after the release")
            .unwrap()
            .expect("plain REINDEX must succeed");
        assert_eq!(
            log.count("atomic_cutover", "fts", "rx_held"),
            log.count("rebuild_started", "fts", "rx_held"),
            "plain REINDEX answers only after its cutover"
        );
        assert_eq!(text_ids(&server, "rx_held", "alpha").await, ["h1"]);
    }
}

// ── REINDEX with no index name ────────────────────────────────────────────────

/// A collection with an HNSW index, full-text rows and graph edges.
async fn collection_with_every_index(server: &TestServer, collection: &str) {
    server
        .exec(&format!("CREATE COLLECTION {collection} TYPE document"))
        .await
        .unwrap();
    server
        .exec(&format!(
            "CREATE VECTOR INDEX idx_{collection} ON {collection} (embedding) METRIC cosine DIM 4"
        ))
        .await
        .unwrap();
    for (id, body, emb) in [
        ("r1", "alpha one", "1.0,0.0,0.0,0.0"),
        ("r2", "alpha two", "0.0,1.0,0.0,0.0"),
        ("r3", "alpha three", "0.0,0.0,1.0,0.0"),
    ] {
        server
            .exec(&format!(
                "INSERT INTO {collection} (id, body, embedding) VALUES ('{id}', '{body}', ARRAY[{emb}])"
            ))
            .await
            .unwrap();
    }
    insert_edge(server, collection, "r1").await;
    insert_edge(server, collection, "r2").await;
}

/// One numeric property of the collection's vector index status.
async fn vector_status(server: &TestServer, collection: &str, property: &str) -> u64 {
    let rows = server
        .query_rows(&format!(
            "SHOW VECTOR INDEX status ON {collection}.embedding"
        ))
        .await
        .unwrap();
    rows.iter()
        .find(|r| r[0] == property)
        .unwrap_or_else(|| panic!("SHOW VECTOR INDEX must report {property}: {rows:?}"))[1]
        .parse()
        .unwrap()
}

/// Seal the vector rows into one segment and wait for its first build, so
/// an HNSW rebuild has a sealed segment to work on.
async fn seal_vectors(server: &TestServer, collection: &str) {
    server
        .exec(&format!(
            "ALTER VECTOR INDEX ON {collection}.embedding SEAL"
        ))
        .await
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let sealed = vector_status(server, collection, "sealed_segments").await;
        let building = vector_status(server, collection, "building_segments").await;
        if sealed >= 1 && building == 0 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the sealed segment did not build: sealed={sealed} building={building}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_reindex_without_a_name_rebuilds_every_index_kind() {
    let log = events();
    let server = TestServer::start().await;
    collection_with_every_index(&server, "rx_all").await;
    seal_vectors(&server, "rx_all").await;

    server.exec("REINDEX CONCURRENTLY rx_all").await.unwrap();

    wait_for_cutovers(&log, "fts", "rx_all").await;
    wait_for_cutovers(&log, "csr", "rx_all").await;
    let deadline = Instant::now() + Duration::from_secs(60);
    while log.count("atomic_cutover", "hnsw", "rx_all") == 0 {
        assert!(
            Instant::now() < deadline,
            "REINDEX with no index name must rebuild the HNSW index too"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    assert_eq!(
        text_ids(&server, "rx_all", "alpha").await,
        ["r1", "r2", "r3"]
    );
    let blob = out_of_root(&server, "rx_all").await;
    assert!(blob.contains("r1") && blob.contains("r2"), "got: {blob}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn plain_reindex_without_a_name_rebuilds_every_index_kind() {
    let log = events();
    let server = TestServer::start().await;
    collection_with_every_index(&server, "rx_plain").await;
    seal_vectors(&server, "rx_plain").await;

    // The plain form answers after every cutover.
    server.exec("REINDEX rx_plain").await.unwrap();

    for index in ["fts", "csr"] {
        let started = log.count("rebuild_started", index, "rx_plain");
        assert!(started > 0, "REINDEX must rebuild the {index} index");
        assert_eq!(
            log.count("atomic_cutover", index, "rx_plain"),
            started,
            "the plain {index} rebuild must cut over before REINDEX returns"
        );
        assert_eq!(log.count("rebuild_refused", index, "rx_plain"), 0);
    }
    assert!(
        log.count("atomic_cutover", "hnsw", "rx_plain") > 0,
        "the plain HNSW rebuild must cut over before REINDEX returns"
    );

    assert_eq!(
        text_ids(&server, "rx_plain", "alpha").await,
        ["r1", "r2", "r3"]
    );
    let blob = out_of_root(&server, "rx_plain").await;
    assert!(blob.contains("r1") && blob.contains("r2"), "got: {blob}");
}
