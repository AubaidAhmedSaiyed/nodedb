// SPDX-License-Identifier: BUSL-1.1

//! BACKUP and RESTORE of a tenant whose collections span several databases.
//!
//! The tenant writes the same collection names in the default database and
//! in two named databases, with a different row count in each. A backup
//! must carry every database, and a restore must bring each row back into
//! the database it came from: a row restored into the wrong database changes
//! that database's count, and a lost database fails its reads.
//!
//! Three shapes run:
//! - a fresh server, which has none of the named databases, so the restore
//!   creates them;
//! - the same server after every collection in every database is purged;
//! - the purge shape on a multi-core server, so each database's collections
//!   home on the cores their `(database, collection)` keys route to.

use super::backup_support::{drain_backup, push_restore};
use crate::harness::TestServer;

const TENANT: u64 = 1;

/// Every database the tenant writes, with the rows each collection holds
/// there.
const DATABASES: [(&str, usize); 3] = [("default", 2), ("bk_sales", 3), ("bk_ops", 4)];

/// Every collection the tenant creates in each database.
const COLLECTIONS: [&str; 4] = ["bk_strict", "bk_loose", "bk_kv", "bk_cols"];

async fn use_database(server: &TestServer, database: &str) {
    server
        .exec(&format!("USE DATABASE {database}"))
        .await
        .unwrap_or_else(|e| panic!("USE DATABASE {database}: {e}"));
}

/// Create the named databases, then create every collection in every
/// database and fill it. Each row's text names its database.
async fn seed(server: &TestServer) {
    for (database, _) in DATABASES.iter().skip(1) {
        server
            .exec(&format!("CREATE DATABASE {database}"))
            .await
            .unwrap_or_else(|e| panic!("CREATE DATABASE {database}: {e}"));
    }
    for (database, rows) in DATABASES {
        use_database(server, database).await;
        for ddl in [
            "CREATE COLLECTION bk_strict (id TEXT PRIMARY KEY, content TEXT) \
             WITH (engine='document_strict')",
            "CREATE COLLECTION bk_loose (id STRING PRIMARY KEY, content STRING) \
             WITH (engine='document_schemaless')",
            "CREATE COLLECTION bk_kv (key STRING PRIMARY KEY, value STRING) WITH (engine='kv')",
            "CREATE COLLECTION bk_cols COLUMNS (id TEXT, region TEXT, ts BIGINT) \
             WITH (engine='columnar')",
        ] {
            server
                .exec(ddl)
                .await
                .unwrap_or_else(|e| panic!("{ddl} in {database}: {e}"));
        }
        for i in 0..rows {
            for insert in [
                format!("INSERT INTO bk_strict (id, content) VALUES ('k{i}', '{database}-{i}')"),
                format!("INSERT INTO bk_loose (id, content) VALUES ('k{i}', '{database}-{i}')"),
                format!("INSERT INTO bk_kv (key, value) VALUES ('k{i}', '{database}-{i}')"),
                format!(
                    "INSERT INTO bk_cols (id, region, ts) VALUES ('k{i}', '{database}-{i}', {i})"
                ),
            ] {
                server
                    .exec(&insert)
                    .await
                    .unwrap_or_else(|e| panic!("{insert} in {database}: {e}"));
            }
        }
    }
    use_database(server, "default").await;
}

/// Hard-purge every collection in every database.
async fn purge_all(server: &TestServer) {
    for (database, _) in DATABASES {
        use_database(server, database).await;
        for collection in COLLECTIONS {
            server
                .exec(&format!("DROP COLLECTION {collection} PURGE"))
                .await
                .unwrap_or_else(|e| {
                    panic!("DROP COLLECTION {collection} PURGE in {database}: {e}")
                });
        }
    }
    use_database(server, "default").await;
}

/// Every collection in every database reads back its own rows through SQL:
/// the right count, and a point lookup that returns the row's own text.
async fn assert_restored(server: &TestServer) {
    for (database, rows) in DATABASES {
        use_database(server, database).await;
        for collection in COLLECTIONS {
            let count = server
                .query_text(&format!("SELECT COUNT(*) FROM {collection}"))
                .await
                .unwrap_or_else(|e| panic!("COUNT(*) FROM {collection} in {database}: {e}"));
            assert_eq!(
                count,
                vec![rows.to_string()],
                "{database}.{collection} must hold exactly its own {rows} rows after the restore"
            );
        }
        let last = rows - 1;
        let expected = vec![format!("{database}-{last}")];
        for (collection, sql) in [
            (
                "bk_strict",
                format!("SELECT content FROM bk_strict WHERE id = 'k{last}'"),
            ),
            (
                "bk_loose",
                format!("SELECT content FROM bk_loose WHERE id = 'k{last}'"),
            ),
            (
                "bk_kv",
                format!("SELECT value FROM bk_kv WHERE key = 'k{last}'"),
            ),
            (
                "bk_cols",
                format!("SELECT region FROM bk_cols WHERE id = 'k{last}'"),
            ),
        ] {
            let got = server
                .query_text(&sql)
                .await
                .unwrap_or_else(|e| panic!("{sql} in {database}: {e}"));
            assert_eq!(
                got, expected,
                "the point lookup on {database}.{collection} must return the row of {database}"
            );
        }
    }
    use_database(server, "default").await;
}

/// A fresh server has neither named database. The restore creates both and
/// brings every row back into the database it came from.
#[tokio::test]
async fn restore_recreates_every_database_on_a_fresh_server() {
    let source = TestServer::start().await;
    seed(&source).await;
    let envelope = drain_backup(&source.client, TENANT)
        .await
        .expect("BACKUP TENANT");
    drop(source);

    let target = TestServer::start().await;
    push_restore(&target.client, TENANT, envelope)
        .await
        .expect("RESTORE into a fresh server");
    assert_restored(&target).await;
}

/// Every collection in every database is purged on the same server. The
/// restore brings every row back into the database it came from.
#[tokio::test]
async fn restore_after_a_purge_brings_back_every_database() {
    let server = TestServer::start().await;
    seed(&server).await;
    let envelope = drain_backup(&server.client, TENANT)
        .await
        .expect("BACKUP TENANT");
    purge_all(&server).await;

    push_restore(&server.client, TENANT, envelope)
        .await
        .expect("RESTORE after the purge");
    assert_restored(&server).await;
}

/// The purge shape on four cores: each database's rows re-issue to the core
/// their destination `(database, collection)` key homes to, and every read
/// fans out to find them there.
#[tokio::test]
async fn restore_after_a_purge_brings_back_every_database_on_many_cores() {
    let server = TestServer::start_multicores(4).await;
    seed(&server).await;
    let envelope = drain_backup(&server.client, TENANT)
        .await
        .expect("BACKUP TENANT");
    purge_all(&server).await;

    push_restore(&server.client, TENANT, envelope)
        .await
        .expect("RESTORE after the purge");
    assert_restored(&server).await;
}
