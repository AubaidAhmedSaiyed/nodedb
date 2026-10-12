// SPDX-License-Identifier: BUSL-1.1

//! Checkpoint export / restore for sorted indexes.
//!
//! A sorted index registration has no durable home outside the WAL: its only
//! record is the `kv_register_sorted_index` WAL record, which a checkpoint's
//! replay floor gates out of a WAL that truncation then deletes. So a checkpoint
//! that publishes rows must publish the registrations with them, in the same
//! generation, or a restart silently comes back with the rows and no index.
//!
//! Content is exported as raw `(sort_key, primary_key)` pairs rather than
//! rebuilt from the restored rows. Reinstalling the exported keys verbatim
//! reproduces the tree that was live, with no second pass over the rows.

use super::super::engine_helpers::table_key;
use super::index::SortedIndex;
use super::manager::{SortedIndexDef, SortedIndexManager, index_key};
use super::tree::OrderStatTree;

/// One sorted index as a checkpoint sees it: its definition plus its full tree
/// content in sort order.
pub struct SortedIndexSnapshot<'a> {
    /// The registration to reinstate.
    pub def: &'a SortedIndexDef,
    /// Every `(sort_key, primary_key)` pair in the tree.
    pub entries: Vec<(Vec<u8>, Vec<u8>)>,
}

impl SortedIndexManager {
    /// Every sorted index registered on `table_key`, with its content.
    ///
    /// Skips index keys with no live index — unreachable while `register` and
    /// `drop` maintain both maps together, and an entry naming an index that is
    /// not there carries nothing to export.
    pub fn export_for_table(&self, table_key: u64) -> Vec<SortedIndexSnapshot<'_>> {
        let Some(idx_keys) = self.collection_indexes.get(&table_key) else {
            return Vec::new();
        };
        idx_keys
            .iter()
            .filter_map(|idx_key| {
                let idx = self.indexes.get(idx_key)?;
                let mut entries = Vec::with_capacity(idx.tree.count() as usize);
                idx.tree.for_each_in_order(|sort_key, primary_key| {
                    entries.push((sort_key.to_vec(), primary_key.to_vec()));
                    true
                });
                Some(SortedIndexSnapshot {
                    def: &idx.def,
                    entries,
                })
            })
            .collect()
    }

    /// Reinstate a registration and its exported content verbatim.
    ///
    /// The checkpoint-restore counterpart of [`SortedIndexManager::register`],
    /// which differs only in where the tree comes from: `register` derives it by
    /// backfilling from the collection's rows, this one installs the exact pairs
    /// the checkpoint captured.
    ///
    /// Restoring over an existing registration REPLACES it, on the same terms
    /// and for the same reasons as `register` — a checkpoint load that lands on
    /// a manager already holding the name (a later generation over an earlier
    /// one) reinstates the generation being loaded, and leaves exactly one
    /// binding behind.
    pub fn restore(
        &mut self,
        database_id: u64,
        tenant_id: u64,
        def: SortedIndexDef,
        entries: &[(Vec<u8>, Vec<u8>)],
    ) {
        self.drop(database_id, tenant_id, &def.name);

        let idx_key = index_key(database_id, tenant_id, &def.name);
        let tbl_key = table_key(database_id, tenant_id, &def.collection);

        let mut tree = OrderStatTree::new();
        for (sort_key, primary_key) in entries {
            tree.insert(sort_key.clone(), primary_key.clone());
        }

        self.collection_indexes
            .entry(tbl_key)
            .or_default()
            .insert(idx_key.clone());
        self.indexes.insert(idx_key, SortedIndex { def, tree });
    }
}

#[cfg(test)]
mod tests {
    use super::super::key::{SortColumn, SortDirection, SortKeyEncoder};
    use super::super::window::WindowConfig;
    use super::*;

    fn scores_table_key() -> u64 {
        table_key(0, 1, "scores")
    }

    fn make_def(name: &str, collection: &str) -> SortedIndexDef {
        SortedIndexDef {
            name: name.into(),
            collection: collection.into(),
            key_column: "player_id".into(),
            encoder: SortKeyEncoder::new(vec![SortColumn {
                name: "score".into(),
                direction: SortDirection::Desc,
            }]),
            window: WindowConfig::none(),
        }
    }

    /// Export → restore must reproduce an index that answers the same rank /
    /// top-k queries, since those — not the struct's shape — are what a restored
    /// collection has to serve.
    #[test]
    fn export_restore_reproduces_ranking() {
        let mut mgr = SortedIndexManager::new();
        mgr.register(0, 1, make_def("lb", "scores"), std::iter::empty());
        let tbl_key = scores_table_key();
        for (pk, score) in [(&b"alice"[..], 100i64), (b"bob", 300), (b"carol", 200)] {
            mgr.on_put(tbl_key, pk, Some(&serde_json::json!({ "score": score })));
        }

        let exported = mgr.export_for_table(tbl_key);
        assert_eq!(exported.len(), 1, "the registration must export");
        assert_eq!(exported[0].entries.len(), 3, "content must export");
        let def = make_def("lb", "scores");
        let entries = exported[0].entries.clone();

        let mut restored = SortedIndexManager::new();
        restored.restore(0, 1, def, &entries);

        // DESC: bob(300) rank 1, carol(200) rank 2, alice(100) rank 3.
        assert_eq!(restored.rank(0, 1, "lb", b"bob", 0), Some(1));
        assert_eq!(restored.rank(0, 1, "lb", b"carol", 0), Some(2));
        assert_eq!(restored.rank(0, 1, "lb", b"alice", 0), Some(3));
        assert_eq!(restored.count(0, 1, "lb", 0), Some(3));
        assert!(
            restored.has_indexes(tbl_key),
            "the restored index must be wired to its collection, or later PUTs \
             would silently stop maintaining it"
        );
    }

    /// A restored index must keep tracking writes: the reverse map that routes
    /// `on_put` to it is part of the registration, not of the content.
    #[test]
    fn restored_index_still_maintained_on_put() {
        let mut mgr = SortedIndexManager::new();
        mgr.restore(0, 1, make_def("lb", "scores"), &[]);

        mgr.on_put(
            scores_table_key(),
            b"dave",
            Some(&serde_json::json!({ "score": 50 })),
        );
        assert_eq!(
            mgr.rank(0, 1, "lb", b"dave", 0),
            Some(1),
            "a restored registration must keep tracking later writes"
        );
    }

    /// A restored windowed index decodes timestamps from the exported keys.
    #[test]
    fn restored_windowed_index_ranks_inside_the_window() {
        let def = || {
            let mut def = make_def("lb", "scores");
            def.encoder = SortKeyEncoder::new(vec![
                SortColumn {
                    name: "score".into(),
                    direction: SortDirection::Desc,
                },
                SortColumn {
                    name: "ts".into(),
                    direction: SortDirection::Desc,
                },
            ]);
            def.window = WindowConfig::custom("ts", 1_000, 2_000);
            def
        };
        let mut mgr = SortedIndexManager::new();
        mgr.register(0, 1, def(), std::iter::empty());
        let rows = [
            (
                &b"before"[..],
                serde_json::json!({ "score": 99, "ts": 999 }),
            ),
            (b"int", serde_json::json!({ "score": 11, "ts": 1_000 })),
            (
                b"float",
                serde_json::json!({ "score": 10.5, "ts": 1_500.5 }),
            ),
            (
                b"iso",
                serde_json::json!({ "score": 5, "ts": "1970-01-01T00:00:01Z" }),
            ),
            (b"after", serde_json::json!({ "score": 50, "ts": 2_000 })),
        ];
        for (pk, doc) in &rows {
            mgr.on_put(scores_table_key(), pk, Some(doc));
        }
        let entries = mgr.export_for_table(scores_table_key())[0].entries.clone();

        let mut restored = SortedIndexManager::new();
        restored.restore(0, 1, def(), &entries);

        let now = 1_500;
        assert_eq!(restored.count(0, 1, "lb", now), Some(3));
        assert_eq!(restored.rank(0, 1, "lb", b"int", now), Some(1));
        assert_eq!(restored.rank(0, 1, "lb", b"float", now), Some(2));
        assert_eq!(restored.rank(0, 1, "lb", b"iso", now), Some(3));
        assert_eq!(restored.rank(0, 1, "lb", b"before", now), None);
        assert_eq!(restored.rank(0, 1, "lb", b"after", now), None);
    }
}
