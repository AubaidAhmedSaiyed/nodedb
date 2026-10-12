// SPDX-License-Identifier: BUSL-1.1

//! A sorted index over numbers orders integers and floats by value.
//!
//! `RANGE` bounds arrive as SQL literals: `11` is an integer, `10.5` a float.
//! A FLOAT column stores `11` as 11.0. The index must treat an integer bound
//! and a float row of equal value as one number, and must rank 11 above 10.5.
//!
//! The `WINDOW` clause tests pin the DDL rules the window relies on.

use nodedb_test_support::pgwire_harness::TestServer;

const BOARD: &str = "sidx_mix_board";
const DESC_INDEX: &str = "sidx_mix_desc";
const ASC_INDEX: &str = "sidx_mix_asc";

async fn exec(server: &TestServer, sql: &str) {
    server
        .exec(sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
}

/// A FLOAT leaderboard holding 10, 10.5 and 11, with a DESC and an ASC index.
async fn mixed_board() -> TestServer {
    let server = TestServer::start().await;
    exec(
        &server,
        &format!(
            "CREATE COLLECTION {BOARD} (id STRING PRIMARY KEY, score FLOAT) WITH (engine='kv')"
        ),
    )
    .await;
    for (id, score) in [("ten_half", "10.5"), ("eleven", "11"), ("ten", "10")] {
        exec(
            &server,
            &format!("INSERT INTO {BOARD} (id, score) VALUES ('{id}', {score})"),
        )
        .await;
    }
    exec(
        &server,
        &format!("CREATE SORTED INDEX {DESC_INDEX} ON {BOARD} (score DESC) KEY id"),
    )
    .await;
    exec(
        &server,
        &format!("CREATE SORTED INDEX {ASC_INDEX} ON {BOARD} (score ASC) KEY id"),
    )
    .await;
    server
}

/// The keys a table function delivers, in row order. Rows are `(rank, key)`.
async fn keys(server: &TestServer, sql: &str) -> Vec<String> {
    server
        .query_rows(sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .into_iter()
        .map(|row| row.get(1).cloned().unwrap_or_default())
        .collect()
}

async fn sorted_keys(server: &TestServer, sql: &str) -> Vec<String> {
    let mut out = keys(server, sql).await;
    out.sort();
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn eleven_ranks_above_ten_and_a_half_in_both_directions() {
    let server = mixed_board().await;

    assert_eq!(
        keys(&server, &format!("SELECT * FROM TOPK({DESC_INDEX}, 10)")).await,
        vec!["eleven", "ten_half", "ten"]
    );
    assert_eq!(
        keys(&server, &format!("SELECT * FROM TOPK({ASC_INDEX}, 10)")).await,
        vec!["ten", "ten_half", "eleven"]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn range_with_integer_and_float_bounds_selects_by_value() {
    let server = mixed_board().await;

    for index in [DESC_INDEX, ASC_INDEX] {
        assert_eq!(
            sorted_keys(&server, &format!("SELECT * FROM RANGE({index}, 10.5, 11)")).await,
            vec!["eleven", "ten_half"],
            "{index}: [10.5, 11] holds the float row and the row stored as 11.0"
        );
        assert_eq!(
            sorted_keys(&server, &format!("SELECT * FROM RANGE({index}, 11, 11)")).await,
            vec!["eleven"],
            "{index}: an integer bound matches a float row of equal value"
        );
        assert_eq!(
            sorted_keys(
                &server,
                &format!("SELECT * FROM RANGE({index}, 10.25, 10.75)")
            )
            .await,
            vec!["ten_half"],
            "{index}"
        );
    }
}

/// A custom window takes ISO 8601 bounds and reads a millisecond column.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn custom_window_with_iso_bounds_hides_rows_outside_it() {
    let server = TestServer::start().await;
    exec(
        &server,
        "CREATE COLLECTION sidx_mix_win (id STRING PRIMARY KEY, score FLOAT, ts BIGINT) \
         WITH (engine='kv')",
    )
    .await;
    exec(
        &server,
        "INSERT INTO sidx_mix_win (id, score, ts) VALUES ('old', 99, 100)",
    )
    .await;
    exec(
        &server,
        "INSERT INTO sidx_mix_win (id, score, ts) VALUES ('new', 1.5, 1714608000000)",
    )
    .await;
    exec(
        &server,
        "CREATE SORTED INDEX sidx_mix_win_idx ON sidx_mix_win (score DESC, ts ASC) KEY id \
         WINDOW CUSTOM START '2000-01-01' END '2999-01-01T00:00:00Z' ON ts",
    )
    .await;

    assert_eq!(
        keys(&server, "SELECT * FROM TOPK(sidx_mix_win_idx, 10)").await,
        vec!["new"],
        "a row from 1970 is outside a window that starts in 2000"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn invalid_window_clauses_are_refused_with_a_syntax_error() {
    let server = TestServer::start().await;
    exec(
        &server,
        "CREATE COLLECTION sidx_mix_bad (id STRING PRIMARY KEY, score FLOAT, ts BIGINT) \
         WITH (engine='kv')",
    )
    .await;

    for (sql, names) in [
        (
            "CREATE SORTED INDEX sidx_bad1 ON sidx_mix_bad (score DESC) KEY id \
             WINDOW DAILY ON ts",
            "ts",
        ),
        (
            "CREATE SORTED INDEX sidx_bad2 ON sidx_mix_bad (score DESC, ts ASC) KEY id \
             WINDOW CUSTOM START 'next week' END 5 ON ts",
            "next week",
        ),
        (
            "CREATE SORTED INDEX sidx_bad3 ON sidx_mix_bad (score DESC, ts ASC) KEY id \
             WINDOW CUSTOM START '2026-03-31' END '2026-01-01' ON ts",
            "END",
        ),
    ] {
        let err = server.exec(sql).await.expect_err(sql);
        assert!(err.contains("42601"), "{sql}: {err}");
        assert!(err.contains(names), "{sql}: {err}");
    }
}
