// SPDX-License-Identifier: BUSL-1.1

//! The native `KvSortedIndexRange` opcode takes typed score bounds.
//!
//! The server encodes them the same way it encodes SQL `RANGE` arguments.
//! An integer bound and a float row of equal value are one number, so the
//! opcode and the SQL function select the same rows.

use nodedb_test_support::native_harness::{NativeTestServer, do_handshake, send_request, send_sql};
use nodedb_types::protocol::opcodes::ResponseStatus;
use nodedb_types::protocol::text_fields::TextFields;
use nodedb_types::protocol::{HelloFrame, NativeResponse, OpCode};
use nodedb_types::value::Value;

fn require_ok(what: &str, response: NativeResponse) -> NativeResponse {
    assert_eq!(
        response.status,
        ResponseStatus::Ok,
        "{what} must succeed: {response:?}"
    );
    response
}

/// The `key` column of a range reply, sorted.
///
/// SQL `RANGE` and the opcode both answer one row per entry with a `rank` and
/// a `key` column. The key is the row's primary key as a string.
fn keys_in(what: &str, response: &NativeResponse) -> Vec<String> {
    let columns = response.columns.as_deref().unwrap_or_default();
    let key_at = columns
        .iter()
        .position(|column| column == "key")
        .unwrap_or_else(|| panic!("{what}: no `key` column in {columns:?}"));
    let mut keys: Vec<String> = response
        .rows
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|row| match row.get(key_at) {
            Some(Value::String(key)) => key.clone(),
            other => panic!("{what}: `key` cell must be a string, got {other:?}"),
        })
        .collect();
    keys.sort_unstable();
    keys
}

#[tokio::test]
async fn native_range_with_typed_bounds_matches_sql_range() {
    let server = NativeTestServer::start().await;
    let (mut stream, _ack) = do_handshake(server.addr, &HelloFrame::current())
        .await
        .expect("handshake");
    let mut seq = 0u64;

    let setup = [
        "CREATE COLLECTION nat_rng (id STRING PRIMARY KEY, score FLOAT) WITH (engine='kv')"
            .to_string(),
        "INSERT INTO nat_rng (id, score) VALUES ('alpha', 10)".to_string(),
        "INSERT INTO nat_rng (id, score) VALUES ('bravo', 10.5)".to_string(),
        "INSERT INTO nat_rng (id, score) VALUES ('charlie', 11)".to_string(),
        "INSERT INTO nat_rng (id, score) VALUES ('delta', 12)".to_string(),
        "CREATE SORTED INDEX nat_rng_idx ON nat_rng (score DESC) KEY id".to_string(),
    ];
    for statement in setup {
        seq += 1;
        require_ok(&statement, send_sql(&mut stream, seq, &statement).await);
    }

    for (min, max, sql_min, sql_max, expected) in [
        (
            Value::Float(10.5),
            Value::Integer(11),
            "10.5",
            "11",
            vec!["bravo", "charlie"],
        ),
        (
            Value::Integer(11),
            Value::Integer(11),
            "11",
            "11",
            vec!["charlie"],
        ),
        (
            Value::Integer(10),
            Value::Float(10.75),
            "10",
            "10.75",
            vec!["alpha", "bravo"],
        ),
        (
            Value::Null,
            Value::Float(10.25),
            "NULL",
            "10.25",
            vec!["alpha"],
        ),
    ] {
        let text = format!("SELECT * FROM RANGE(nat_rng_idx, {sql_min}, {sql_max})");
        seq += 1;
        let via_sql = require_ok(&text, send_sql(&mut stream, seq, &text).await);

        seq += 1;
        let fields = TextFields {
            index_name: Some("nat_rng_idx".into()),
            score_min: Some(min.clone()),
            score_max: Some(max.clone()),
            ..TextFields::default()
        };
        let via_op = require_ok(
            "KvSortedIndexRange",
            send_request(&mut stream, seq, OpCode::KvSortedIndexRange, fields).await,
        );

        let expected: Vec<String> = expected.into_iter().map(String::from).collect();
        assert_eq!(keys_in(&text, &via_sql), expected, "{text}");
        assert_eq!(
            keys_in("KvSortedIndexRange", &via_op),
            expected,
            "native range [{min:?}, {max:?}] must select what SQL RANGE selects"
        );
    }

    server.shutdown().await;
}
