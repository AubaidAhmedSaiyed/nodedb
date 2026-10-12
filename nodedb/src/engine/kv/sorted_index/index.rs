// SPDX-License-Identifier: BUSL-1.1

//! One sorted index: its definition, its order-statistic tree, and the reads
//! it answers.
//!
//! The manager holds one of these per registered index. A transaction builds
//! a private one from the collection's base rows with its staged writes
//! folded in. Both answer through the same methods, so a read inside a
//! transaction ranks, counts and windows exactly like an autocommit read.

use super::key::SortKeyEncoder;
use super::manager::SortedIndexDef;
use super::tree::OrderStatTree;
use super::windowed_query::{self, SortedIndexRef};

/// A sorted index: definition plus tree.
pub struct SortedIndex {
    pub(super) def: SortedIndexDef,
    pub(super) tree: OrderStatTree,
}

impl std::fmt::Debug for SortedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SortedIndex")
            .field("name", &self.def.name)
            .field("collection", &self.def.collection)
            .field("count", &self.tree.count())
            .finish()
    }
}

impl SortedIndex {
    /// Build an index over `rows`, `(primary_key_bytes, value_bytes)` pairs.
    ///
    /// A row that lacks a sort column is left out. Returns the index and the
    /// number of rows it holds.
    pub fn build(
        def: SortedIndexDef,
        rows: impl Iterator<Item = (Vec<u8>, Vec<u8>)>,
    ) -> (Self, u32) {
        let mut tree = OrderStatTree::new();
        let mut indexed = 0u32;
        for (pk_bytes, value_bytes) in rows {
            if let Some(sort_key) = extract_sort_key_from_value(&def, &value_bytes) {
                tree.insert(sort_key, pk_bytes);
                indexed += 1;
            }
        }
        (Self { def, tree }, indexed)
    }

    /// The definition this index was built from.
    pub fn def(&self) -> &SortedIndexDef {
        &self.def
    }

    fn index_ref(&self) -> SortedIndexRef<'_> {
        SortedIndexRef {
            def: &self.def,
            tree: &self.tree,
        }
    }

    /// The 1-based rank of `primary_key`.
    ///
    /// A windowed index counts only the entries inside the current window.
    pub fn rank(&self, primary_key: &[u8], now_ms: u64) -> Option<u32> {
        if self.def.window.is_unwindowed() {
            return self.tree.rank(primary_key);
        }
        windowed_query::windowed_rank(&self.index_ref(), primary_key, now_ms)
    }

    /// The top `k` entries as `(rank, primary_key)` pairs.
    pub fn top_k(&self, k: u32, now_ms: u64) -> Vec<(u32, Vec<u8>)> {
        if self.def.window.is_unwindowed() {
            return self
                .tree
                .top_k(k)
                .into_iter()
                .enumerate()
                .map(|(i, (_, pk))| (i as u32 + 1, pk.to_vec()))
                .collect();
        }
        windowed_query::windowed_top_k(&self.index_ref(), k, now_ms)
    }

    /// The entries in a score range as `(rank, primary_key)` pairs.
    ///
    /// `score_min` and `score_max` are the raw value bytes of the index's
    /// LEADING sort column (as [`extract_sort_key_from_value`] produces them),
    /// not encoded tree keys. The caller names a score. Only the index's own
    /// encoder knows the framing and direction that turn it into a bound the
    /// tree compares against.
    pub fn range(
        &self,
        score_min: Option<&[u8]>,
        score_max: Option<&[u8]>,
        now_ms: u64,
    ) -> Vec<(u32, Vec<u8>)> {
        let (lower, upper) = self
            .def
            .encoder
            .first_column_range_bounds(score_min, score_max);
        let entries = self.tree.range(lower.as_deref(), upper.as_deref());

        if self.def.window.is_unwindowed() {
            return entries
                .into_iter()
                .filter_map(|(_, pk)| {
                    let rank = self.tree.rank(pk)?;
                    Some((rank, pk.to_vec()))
                })
                .collect();
        }
        windowed_query::windowed_range(&self.index_ref(), &entries, now_ms)
    }

    /// The number of entries, inside the current window for a windowed index.
    pub fn count(&self, now_ms: u64) -> u32 {
        if self.def.window.is_unwindowed() {
            return self.tree.count();
        }
        windowed_query::windowed_count(&self.index_ref(), now_ms)
    }

    /// The sort key of `primary_key` (ZSCORE equivalent).
    pub fn score(&self, primary_key: &[u8]) -> Option<Vec<u8>> {
        self.tree.get_sort_key(primary_key).map(|s| s.to_vec())
    }
}

/// Extract the sort columns from a MessagePack-encoded KV value and build a
/// sort key. `None` when the value is not a map or lacks a sort column.
pub(super) fn extract_sort_key_from_value(
    def: &SortedIndexDef,
    value_bytes: &[u8],
) -> Option<Vec<u8>> {
    let doc: serde_json::Value = nodedb_types::json_from_msgpack(value_bytes).ok()?;
    let obj = doc.as_object()?;

    let mut values: Vec<Vec<u8>> = Vec::with_capacity(def.encoder.column_count());
    for col in def.encoder.columns() {
        let field_val = obj.get(&col.name)?;
        values.push(field_value_to_sort_bytes(field_val));
    }

    let refs: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();
    Some(def.encoder.encode(&refs))
}

/// Convert a JSON field value to sortable bytes.
///
/// The bytes match what the KV write path extracts for `on_put`. A row
/// backfilled here and the same row written later get the same sort key.
fn field_value_to_sort_bytes(val: &serde_json::Value) -> Vec<u8> {
    match val {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                SortKeyEncoder::encode_i64(i).to_vec()
            } else if let Some(f) = n.as_f64() {
                SortKeyEncoder::encode_f64(f).to_vec()
            } else {
                Vec::new()
            }
        }
        serde_json::Value::String(s) => s.as_bytes().to_vec(),
        serde_json::Value::Bool(b) => vec![u8::from(*b)],
        serde_json::Value::Null => Vec::new(),
        other @ (serde_json::Value::Array(_) | serde_json::Value::Object(_)) => {
            other.to_string().into_bytes()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::key::{SortColumn, SortDirection};
    use super::super::window::WindowConfig;
    use super::*;

    /// 2024-05-02 15:36:40 UTC, a Thursday.
    const NOW_MS: u64 = 1_714_664_200_000;
    /// 2024-05-02 00:00:00 UTC.
    const DAY_START_MS: u64 = 1_714_608_000_000;
    /// Monday 2024-04-29 00:00:00 UTC.
    const WEEK_START_MS: u64 = 1_714_348_800_000;
    /// 2024-05-01 00:00:00 UTC.
    const MONTH_START_MS: u64 = 1_714_521_600_000;

    fn def(window: WindowConfig, ts_direction: SortDirection) -> SortedIndexDef {
        SortedIndexDef {
            name: "lb".into(),
            collection: "scores".into(),
            key_column: "player".into(),
            encoder: SortKeyEncoder::new(vec![
                SortColumn {
                    name: "score".into(),
                    direction: SortDirection::Desc,
                },
                SortColumn {
                    name: "ts".into(),
                    direction: ts_direction,
                },
            ]),
            window,
        }
    }

    /// A row with `ts` set to `ts`, or with no `ts` field when `None`.
    fn row(player: &str, score: i64, ts: Option<serde_json::Value>) -> (Vec<u8>, Vec<u8>) {
        let mut doc = serde_json::json!({ "player": player, "score": score });
        if let Some(ts) = ts {
            doc["ts"] = ts;
        }
        let value = nodedb_types::json_to_msgpack(&doc).expect("encode row");
        (player.as_bytes().to_vec(), value)
    }

    fn ms(ts: u64) -> Option<serde_json::Value> {
        Some(serde_json::json!(ts))
    }

    fn build(def: SortedIndexDef, rows: Vec<(Vec<u8>, Vec<u8>)>) -> SortedIndex {
        SortedIndex::build(def, rows.into_iter()).0
    }

    fn pks(entries: &[(u32, Vec<u8>)]) -> Vec<(u32, &str)> {
        entries
            .iter()
            .map(|(rank, pk)| (*rank, std::str::from_utf8(pk).expect("utf-8 pk")))
            .collect()
    }

    /// `old` has the top score but a timestamp before the window start.
    /// `fresh` and `late` are inside the window. Every windowed read must
    /// hide `old` and rank the other two as if `old` did not exist.
    fn assert_hides_old_keeps_fresh(idx: &SortedIndex, now_ms: u64) {
        assert_eq!(idx.count(now_ms), 2);
        assert_eq!(idx.rank(b"old", now_ms), None);
        assert_eq!(idx.rank(b"fresh", now_ms), Some(1));
        assert_eq!(idx.rank(b"late", now_ms), Some(2));
        assert_eq!(pks(&idx.top_k(10, now_ms)), vec![(1, "fresh"), (2, "late")]);
        assert_eq!(pks(&idx.top_k(1, now_ms)), vec![(1, "fresh")]);
        assert_eq!(
            pks(&idx.range(None, None, now_ms)),
            vec![(1, "fresh"), (2, "late")]
        );
        let score_200 = SortKeyEncoder::encode_i64(200);
        let score_300 = SortKeyEncoder::encode_i64(300);
        assert_eq!(
            pks(&idx.range(Some(&score_200), Some(&score_300), now_ms)),
            vec![(1, "fresh")]
        );
    }

    fn window_rows(start_ms: u64) -> Vec<(Vec<u8>, Vec<u8>)> {
        vec![
            row("old", 300, ms(start_ms - 1)),
            row("fresh", 200, ms(start_ms)),
            row("late", 100, ms(NOW_MS)),
        ]
    }

    #[test]
    fn daily_window_hides_entries_before_today_utc_start() {
        let idx = build(
            def(WindowConfig::daily("ts"), SortDirection::Asc),
            window_rows(DAY_START_MS),
        );
        assert_hides_old_keeps_fresh(&idx, NOW_MS);
    }

    #[test]
    fn daily_window_hides_entries_from_days_ago() {
        let idx = build(
            def(WindowConfig::daily("ts"), SortDirection::Asc),
            vec![
                row("old", 300, ms(DAY_START_MS - 30 * 86_400_000)),
                row("fresh", 200, ms(DAY_START_MS + 1)),
                row("late", 100, ms(NOW_MS)),
            ],
        );
        assert_hides_old_keeps_fresh(&idx, NOW_MS);
    }

    #[test]
    fn weekly_window_starts_at_monday_utc() {
        let idx = build(
            def(WindowConfig::weekly("ts"), SortDirection::Asc),
            window_rows(WEEK_START_MS),
        );
        assert_hides_old_keeps_fresh(&idx, NOW_MS);
    }

    #[test]
    fn monthly_window_starts_at_the_first_of_the_month_utc() {
        let idx = build(
            def(WindowConfig::monthly("ts"), SortDirection::Asc),
            window_rows(MONTH_START_MS),
        );
        assert_hides_old_keeps_fresh(&idx, NOW_MS);
    }

    #[test]
    fn desc_timestamp_column_windows_the_same_as_asc() {
        let idx = build(
            def(WindowConfig::daily("ts"), SortDirection::Desc),
            window_rows(DAY_START_MS),
        );
        assert_hides_old_keeps_fresh(&idx, NOW_MS);
    }

    #[test]
    fn float_timestamp_column_windows_by_its_value() {
        let start = DAY_START_MS as f64;
        let idx = build(
            def(WindowConfig::daily("ts"), SortDirection::Asc),
            vec![
                row("old", 300, Some(serde_json::json!(start - 0.5))),
                row("fresh", 200, Some(serde_json::json!(start + 0.5))),
                row("late", 100, Some(serde_json::json!(NOW_MS as f64 + 0.25))),
            ],
        );
        assert_hides_old_keeps_fresh(&idx, NOW_MS);
    }

    #[test]
    fn float_timestamp_column_with_desc_direction_windows_by_its_value() {
        let start = DAY_START_MS as f64;
        let idx = build(
            def(WindowConfig::daily("ts"), SortDirection::Desc),
            vec![
                row("old", 300, Some(serde_json::json!(start - 0.5))),
                row("fresh", 200, Some(serde_json::json!(start + 0.5))),
                row("late", 100, Some(serde_json::json!(NOW_MS as f64 + 0.25))),
            ],
        );
        assert_hides_old_keeps_fresh(&idx, NOW_MS);
    }

    #[test]
    fn iso_8601_string_timestamp_column_windows_by_its_instant() {
        let idx = build(
            def(WindowConfig::daily("ts"), SortDirection::Asc),
            vec![
                row("old", 300, Some(serde_json::json!("2024-05-01T23:59:59Z"))),
                row(
                    "fresh",
                    200,
                    Some(serde_json::json!("2024-05-02T00:00:00Z")),
                ),
                row("late", 100, Some(serde_json::json!("2024-05-02T15:00:00Z"))),
            ],
        );
        assert_hides_old_keeps_fresh(&idx, NOW_MS);
    }

    #[test]
    fn custom_window_shows_nothing_when_now_is_outside_it() {
        let (start, end) = (DAY_START_MS, DAY_START_MS + 3_600_000);
        let idx = build(
            def(WindowConfig::custom("ts", start, end), SortDirection::Asc),
            vec![
                row("a", 300, ms(start)),
                row("b", 200, ms(start + 1)),
                row("c", 100, ms(end + 1)),
            ],
        );
        for now_ms in [start - 1, end, end + 86_400_000] {
            assert_eq!(idx.count(now_ms), 0);
            assert!(idx.top_k(10, now_ms).is_empty());
            assert!(idx.range(None, None, now_ms).is_empty());
            for pk in [&b"a"[..], &b"b"[..], &b"c"[..]] {
                assert_eq!(idx.rank(pk, now_ms), None);
            }
        }
    }

    #[test]
    fn custom_window_shows_only_entries_inside_its_range() {
        let (start, end) = (DAY_START_MS, DAY_START_MS + 3_600_000);
        let idx = build(
            def(WindowConfig::custom("ts", start, end), SortDirection::Asc),
            vec![
                row("before", 400, ms(start - 1)),
                row("first", 300, ms(start)),
                row("last", 200, ms(end - 1)),
                row("after", 100, ms(end)),
            ],
        );
        let now_ms = start + 60_000;
        assert_eq!(idx.count(now_ms), 2);
        assert_eq!(idx.rank(b"before", now_ms), None);
        assert_eq!(idx.rank(b"first", now_ms), Some(1));
        assert_eq!(idx.rank(b"last", now_ms), Some(2));
        assert_eq!(idx.rank(b"after", now_ms), None);
        assert_eq!(pks(&idx.top_k(10, now_ms)), vec![(1, "first"), (2, "last")]);
        assert_eq!(
            pks(&idx.range(None, None, now_ms)),
            vec![(1, "first"), (2, "last")]
        );
    }

    #[test]
    fn entry_with_a_null_timestamp_is_outside_every_window() {
        let idx = build(
            def(WindowConfig::daily("ts"), SortDirection::Asc),
            vec![
                row("null_ts", 300, Some(serde_json::Value::Null)),
                row("fresh", 200, ms(DAY_START_MS)),
            ],
        );
        assert_eq!(idx.count(NOW_MS), 1);
        assert_eq!(idx.rank(b"null_ts", NOW_MS), None);
        assert_eq!(idx.rank(b"fresh", NOW_MS), Some(1));
        assert_eq!(pks(&idx.top_k(10, NOW_MS)), vec![(1, "fresh")]);
        assert_eq!(pks(&idx.range(None, None, NOW_MS)), vec![(1, "fresh")]);
    }

    #[test]
    fn entry_without_a_timestamp_field_is_not_indexed() {
        let (idx, indexed) = SortedIndex::build(
            def(WindowConfig::daily("ts"), SortDirection::Asc),
            vec![row("no_ts", 300, None), row("fresh", 200, ms(DAY_START_MS))].into_iter(),
        );
        assert_eq!(indexed, 1);
        assert_eq!(idx.count(NOW_MS), 1);
        assert_eq!(idx.rank(b"no_ts", NOW_MS), None);
        assert_eq!(idx.rank(b"fresh", NOW_MS), Some(1));
    }

    #[test]
    fn top_zero_of_a_windowed_index_is_empty() {
        let idx = build(
            def(WindowConfig::daily("ts"), SortDirection::Asc),
            window_rows(DAY_START_MS),
        );
        assert!(idx.top_k(0, NOW_MS).is_empty());
    }

    #[test]
    fn unwindowed_index_shows_every_entry() {
        let idx = build(
            def(WindowConfig::none(), SortDirection::Asc),
            window_rows(DAY_START_MS),
        );
        assert_eq!(idx.count(NOW_MS), 3);
        assert_eq!(idx.rank(b"old", NOW_MS), Some(1));
    }

    #[test]
    fn backfilled_bool_sorts_like_a_written_bool() {
        assert_eq!(
            field_value_to_sort_bytes(&serde_json::json!(true)),
            vec![1u8]
        );
        assert_eq!(
            field_value_to_sort_bytes(&serde_json::json!(false)),
            vec![0u8]
        );
        assert!(field_value_to_sort_bytes(&serde_json::Value::Null).is_empty());
    }
}
