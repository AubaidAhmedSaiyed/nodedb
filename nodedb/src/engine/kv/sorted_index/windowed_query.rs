// SPDX-License-Identifier: BUSL-1.1

//! Windowed query helpers for sorted indexes.
//!
//! These functions handle time-windowed queries where only entries within
//! the current window are considered for rank/top_k/range/count. An entry is
//! visible when the timestamp column of its sort key decodes to a timestamp
//! inside the window bounds. An entry with no decodable timestamp is hidden.
//!
//! Performance: uses `for_each_in_order` with early termination instead of
//! fetching all entries. windowed_top_k(10) on a 10M-entry tree is O(N) in
//! the worst case (all entries outside window) but O(k) in the typical case
//! where windowed entries are near the front of sort order.

use std::collections::HashMap;

use super::manager::SortedIndexDef;
use super::timestamp::decode_column_timestamp;
use super::tree::OrderStatTree;
use super::window::WindowBounds;

/// Internal state for a sorted index (definition + tree).
pub(super) struct SortedIndexRef<'a> {
    pub def: &'a SortedIndexDef,
    pub tree: &'a OrderStatTree,
}

/// Windowed rank: count visible entries up to and including `primary_key`.
///
/// `None` when `primary_key` is absent or outside the window.
pub(super) fn windowed_rank(
    idx: &SortedIndexRef<'_>,
    primary_key: &[u8],
    now_ms: u64,
) -> Option<u32> {
    let bounds = idx.def.window.bounds(now_ms)?;
    let target_sort = idx.tree.get_sort_key(primary_key)?;
    if !entry_in_window(idx.def, target_sort, &bounds) {
        return None;
    }

    let mut rank = 0u32;
    let mut found = false;

    idx.tree.for_each_in_order(|sort_key, pk| {
        if !entry_in_window(idx.def, sort_key, &bounds) {
            return true; // Skip, continue.
        }
        rank += 1;
        if pk == primary_key && sort_key == target_sort {
            found = true;
            return false; // Stop.
        }
        true
    });

    if found { Some(rank) } else { None }
}

/// Windowed top-k: collect top K entries that are in the current window.
///
/// Early termination: stops as soon as k windowed entries are found.
pub(super) fn windowed_top_k(idx: &SortedIndexRef<'_>, k: u32, now_ms: u64) -> Vec<(u32, Vec<u8>)> {
    let Some(bounds) = idx.def.window.bounds(now_ms) else {
        return Vec::new();
    };
    if k == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(k as usize);
    let mut rank = 0u32;

    idx.tree.for_each_in_order(|sort_key, pk| {
        if !entry_in_window(idx.def, sort_key, &bounds) {
            return true; // Skip, continue.
        }
        rank += 1;
        result.push((rank, pk.to_vec()));
        rank < k // Stop when we have k entries.
    });

    result
}

/// Windowed range: keep the visible entries of `entries`, ranked among all
/// visible entries.
pub(super) fn windowed_range(
    idx: &SortedIndexRef<'_>,
    entries: &[(&[u8], &[u8])],
    now_ms: u64,
) -> Vec<(u32, Vec<u8>)> {
    let Some(bounds) = idx.def.window.bounds(now_ms) else {
        return Vec::new();
    };

    // Build windowed ranks via streaming traversal.
    let mut windowed_ranks: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut rank = 0u32;

    idx.tree.for_each_in_order(|sort_key, pk| {
        if entry_in_window(idx.def, sort_key, &bounds) {
            rank += 1;
            windowed_ranks.insert(pk.to_vec(), rank);
        }
        true // Must scan all to compute correct ranks.
    });

    entries
        .iter()
        .filter_map(|(_, pk)| {
            let r = windowed_ranks.get(*pk)?;
            Some((*r, pk.to_vec()))
        })
        .collect()
}

/// Windowed count: count entries in the current window.
pub(super) fn windowed_count(idx: &SortedIndexRef<'_>, now_ms: u64) -> u32 {
    let Some(bounds) = idx.def.window.bounds(now_ms) else {
        return 0;
    };

    let mut count = 0u32;
    idx.tree.for_each_in_order(|sort_key, _| {
        if entry_in_window(idx.def, sort_key, &bounds) {
            count += 1;
        }
        true
    });
    count
}

/// `true` when the timestamp column of `sort_key` lies inside `bounds`.
///
/// Registration rejects a windowed index whose timestamp column is not a sort
/// column. A key without a decodable timestamp is outside every window.
fn entry_in_window(def: &SortedIndexDef, sort_key: &[u8], bounds: &WindowBounds) -> bool {
    def.encoder
        .column_value(sort_key, &def.window.timestamp_column)
        .and_then(|value| decode_column_timestamp(&value))
        .is_some_and(|ts| bounds.contains(ts))
}
