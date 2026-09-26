// SPDX-License-Identifier: BUSL-1.1

//! Per-row SQL functions and window output on a document collection.
//!
//! - A window column is computed by the window pass only. ORDER BY and the
//!   SELECT list read it as a column, so ORDER BY can name the window alias.
//! - `doc_get`, `doc_exists`, `doc_array_contains` and `vector_distance`
//!   evaluate per row in a projection and in WHERE.
//! - A vector dimension mismatch, a non-vector operand and a malformed
//!   JSONPath fail with SQLSTATE `22000`, never a silent `NULL`.
//! - An index-owned search function in a row filter is refused at plan time.

use crate::harness::TestServer;

async fn rows(srv: &TestServer, sql: &str) -> Vec<Vec<String>> {
    srv.query_rows(sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

async fn collection(srv: &TestServer, name: &str) {
    srv.exec(&format!(
        "CREATE COLLECTION {name} WITH (engine='document_schemaless')"
    ))
    .await
    .unwrap();
}

#[tokio::test]
async fn order_by_a_window_alias_sorts_on_the_window_column() {
    let srv = TestServer::start().await;
    collection(&srv, "win_order").await;
    for id in ["w1", "w2", "w3"] {
        srv.exec(&format!("INSERT INTO win_order {{ id: '{id}', n: 1 }}"))
            .await
            .unwrap();
    }

    assert_eq!(
        rows(
            &srv,
            "SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) AS rn FROM win_order ORDER BY rn",
        )
        .await,
        vec![
            vec!["w3".to_string(), "1".to_string()],
            vec!["w2".to_string(), "2".to_string()],
            vec!["w1".to_string(), "3".to_string()],
        ]
    );

    assert_eq!(
        rows(
            &srv,
            "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM win_order \
             ORDER BY rn DESC LIMIT 2",
        )
        .await,
        vec![
            vec!["w3".to_string(), "3".to_string()],
            vec!["w2".to_string(), "2".to_string()],
        ]
    );
}

async fn seeded_events(name: &str) -> TestServer {
    let srv = TestServer::start().await;
    collection(&srv, name).await;
    srv.exec(&format!(
        "INSERT INTO {name} {{ id: 'e1', payload: {{ user: {{ name: 'ada', email: 'a@x' }}, \
         tags: ['important', 'ops'] }} }}"
    ))
    .await
    .unwrap();
    srv.exec(&format!(
        "INSERT INTO {name} {{ id: 'e2', payload: {{ user: {{ name: 'bob' }}, tags: ['ops'] }} }}"
    ))
    .await
    .unwrap();
    srv
}

#[tokio::test]
async fn doc_get_projects_a_nested_field() {
    let srv = seeded_events("ev_get").await;
    assert_eq!(
        rows(
            &srv,
            "SELECT id, doc_get(payload, '$.user.name') AS name, \
             doc_get(payload, '$.user.email', 'none') AS email FROM ev_get ORDER BY id",
        )
        .await,
        vec![
            vec!["e1".to_string(), "ada".to_string(), "a@x".to_string()],
            vec!["e2".to_string(), "bob".to_string(), "none".to_string()],
        ]
    );
}

#[tokio::test]
async fn doc_exists_and_doc_array_contains_filter_rows() {
    let srv = seeded_events("ev_where").await;
    let ids =
        |r: Vec<Vec<String>>| -> Vec<String> { r.into_iter().map(|row| row[0].clone()).collect() };

    assert_eq!(
        ids(rows(
            &srv,
            "SELECT id FROM ev_where WHERE doc_exists(payload, '$.user.email') ORDER BY id",
        )
        .await),
        vec!["e1"]
    );
    assert_eq!(
        ids(rows(
            &srv,
            "SELECT id FROM ev_where WHERE doc_array_contains(payload, '$.tags', 'ops') \
             ORDER BY id",
        )
        .await),
        vec!["e1", "e2"]
    );
    assert_eq!(
        ids(rows(
            &srv,
            "SELECT id FROM ev_where \
             WHERE doc_array_contains(payload, '$.tags', 'important') ORDER BY id",
        )
        .await),
        vec!["e1"]
    );
}

/// Without a vector-search ORDER BY, `vector_distance` evaluates per row as
/// the squared L2 distance a vector search reports.
#[tokio::test]
async fn vector_distance_in_a_projection_evaluates_per_row() {
    let srv = TestServer::start().await;
    collection(&srv, "vd_rows").await;
    srv.exec("INSERT INTO vd_rows { id: 'v1', emb: [3.0, 4.0] }")
        .await
        .unwrap();

    let r = rows(
        &srv,
        "SELECT id, vector_distance(emb, ARRAY[0.0, 0.0]) AS d FROM vd_rows",
    )
    .await;
    assert_eq!(r.len(), 1, "one row expected, got {r:?}");
    assert_eq!(r[0][0], "v1");
    let d: f64 = r[0][1]
        .parse()
        .unwrap_or_else(|_| panic!("distance must be numeric, got {r:?}"));
    assert!((d - 25.0).abs() < 1e-9, "expected 25, got {d}");
}

/// Operands of different dimensions fail the row with `22000`, naming both
/// dimensions, in the vector engine's message shape.
#[tokio::test]
async fn vector_dimension_mismatch_is_a_data_exception() {
    let srv = TestServer::start().await;
    collection(&srv, "vd_mismatch").await;
    srv.exec("INSERT INTO vd_mismatch { id: 'v1', emb: [3.0, 4.0] }")
        .await
        .unwrap();

    let error = srv
        .query_text("SELECT id, vector_distance(emb, ARRAY[0.0, 0.0, 0.0]) AS d FROM vd_mismatch")
        .await
        .expect_err("a dimension mismatch must fail");
    assert!(
        error.contains("22000") && error.contains("vector dimension mismatch: expected 2, got 3"),
        "expected 22000 naming both dimensions, got: {error}"
    );
}

/// A non-vector operand fails with `22000`, naming its position and type.
#[tokio::test]
async fn a_non_vector_operand_is_a_data_exception() {
    let srv = TestServer::start().await;
    collection(&srv, "vd_badarg").await;
    srv.exec("INSERT INTO vd_badarg { id: 'v1', emb: 7 }")
        .await
        .unwrap();

    let error = srv
        .query_text("SELECT id, vector_distance(emb, ARRAY[0.0]) AS d FROM vd_badarg")
        .await
        .expect_err("a non-vector operand must fail");
    assert!(
        error.contains("22000") && error.contains("argument 1") && error.contains("got int"),
        "expected 22000 naming argument 1 and its type, got: {error}"
    );
}

/// A malformed JSONPath fails with `22000`. A missing path returns the
/// default, as `doc_get_projects_a_nested_field` shows.
#[tokio::test]
async fn a_malformed_json_path_is_a_data_exception() {
    let srv = seeded_events("ev_badpath").await;
    let error = srv
        .query_text("SELECT id, doc_get(payload, '$..name') AS n FROM ev_badpath")
        .await
        .expect_err("a malformed JSONPath must fail");
    assert!(
        error.contains("22000") && error.contains("invalid JSONPath"),
        "expected 22000 for the malformed path, got: {error}"
    );
}

/// `bm25_score` reads the full-text index and has no per-row value, so a
/// comparison on it in WHERE is refused with `0A000` at plan time, even on
/// an empty collection.
#[tokio::test]
async fn a_search_score_in_a_row_filter_is_refused() {
    let srv = TestServer::start().await;
    collection(&srv, "fts_refuse").await;
    let error = srv
        .query_text("SELECT id FROM fts_refuse WHERE bm25_score(body, 'rust') > 1.0")
        .await
        .expect_err("bm25_score in a row filter must be refused");
    assert!(
        error.contains("0A000") && error.contains("bm25_score"),
        "expected 0A000 naming bm25_score, got: {error}"
    );
}
