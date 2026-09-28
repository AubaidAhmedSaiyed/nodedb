// SPDX-License-Identifier: BUSL-1.1

//! `LIKE` / `ILIKE` and `ARRAY[...]` inside expression predicates.
//!
//! A `LIKE` that sits inside a larger expression (an `OR` with another
//! comparison, a function call on its operand) is evaluated row by row as a
//! `like` / `ilike` call. An `ARRAY[...]` with a column element is built per
//! row. Both return the matching rows, never an empty result.

use crate::harness::TestServer;

async fn seeded(name: &str) -> TestServer {
    let srv = TestServer::start().await;
    srv.exec(&format!(
        "CREATE COLLECTION {name} WITH (engine='document_schemaless')"
    ))
    .await
    .unwrap();
    for (id, who, n) in [
        ("a1", "Alice", 1),
        ("a2", "amy", 3),
        ("b1", "bob", 9),
        ("c1", "Carl", 2),
    ] {
        srv.exec(&format!(
            "INSERT INTO {name} {{ id: '{id}', name: '{who}', n: {n} }}"
        ))
        .await
        .unwrap();
    }
    srv
}

async fn ids(srv: &TestServer, sql: &str) -> Vec<String> {
    srv.query_rows(sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .into_iter()
        .map(|row| row[0].clone())
        .collect()
}

#[tokio::test]
async fn like_inside_an_or_predicate_matches_rows() {
    let srv = seeded("like_or").await;
    assert_eq!(
        ids(
            &srv,
            "SELECT id FROM like_or WHERE lower(name) LIKE 'a%' OR n > 5 ORDER BY id"
        )
        .await,
        vec!["a1", "a2", "b1"]
    );
}

#[tokio::test]
async fn not_like_and_ilike_inside_expression_predicates() {
    let srv = seeded("like_not").await;
    assert_eq!(
        ids(
            &srv,
            "SELECT id FROM like_not WHERE lower(name) NOT LIKE 'a%' AND n < 5 ORDER BY id"
        )
        .await,
        vec!["c1"]
    );
    assert_eq!(
        ids(
            &srv,
            "SELECT id FROM like_not WHERE name ILIKE 'A%' OR n > 100 ORDER BY id"
        )
        .await,
        vec!["a1", "a2"]
    );
}

#[tokio::test]
async fn an_array_with_a_column_element_is_built_per_row() {
    let srv = seeded("array_col").await;
    assert_eq!(
        ids(
            &srv,
            "SELECT id FROM array_col WHERE array_contains(ARRAY[n, 100], 9) ORDER BY id"
        )
        .await,
        vec!["b1"]
    );
}
