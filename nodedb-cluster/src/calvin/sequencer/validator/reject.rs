// SPDX-License-Identifier: BUSL-1.1

//! The reported position and conflict context of a cycle-rejected
//! transaction.

use crate::calvin::sequencer::metrics::ConflictKey;

use super::graph::{ConflictIndex, KeyRef};
use super::keys::{FlatKey, KeyId, KeyScope, conflicts};

/// Build the `(position_admitted, conflict_context)` for a cycle-rejected txn.
///
/// A rejected txn conflicts with any admitted txn that writes a key it reads
/// or reads a key it writes, by [`conflicts`]. We select the conflict with
/// the smallest admitted position (ties broken by key identity) so both the
/// reported position and the [`ConflictKey`] are deterministic. If no
/// conflicting txn was admitted, the position defaults to `0` and the context
/// to `None`.
pub(super) fn reject_context(
    idx: usize,
    tenant: u64,
    index: &ConflictIndex,
    emitted_position: &[Option<u32>],
) -> (u32, Option<ConflictKey>) {
    // Best conflict so far, ordered by (admitted position, key identity).
    let mut best: Option<(u32, KeyId, &'static str)> = None;
    let mut consider = |fk: &FlatKey, other: usize| {
        if other == idx {
            return;
        }
        let Some(pos) = emitted_position[other] else {
            return;
        };
        let key = fk.key_id();
        let better = match &best {
            None => true,
            Some((bpos, bkey, _)) => (pos, &key) < (*bpos, bkey),
        };
        if better {
            best = Some((pos, key, fk.engine_name));
        }
    };

    for fk in &index.reads[idx] {
        // A writer of a key this txn reads.
        for &other in index.writers_of.get(&fk.key_id()).into_iter().flatten() {
            consider(fk, other);
        }
        // A writer whose key's scope covers this read, or the reverse.
        if let Some(coll) = index.by_collection.get(&fk.collection) {
            let row_writes: &[KeyRef] = match fk.scope {
                KeyScope::Row => &[],
                KeyScope::Collection => &coll.row_writes,
            };
            for &w in coll.scoped_writes.iter().chain(row_writes) {
                if conflicts(fk, index.write(w)) {
                    consider(fk, w.0);
                }
            }
        }
    }
    for fk in &index.writes[idx] {
        // A reader of a key this txn writes.
        for &other in index.readers_of.get(&fk.key_id()).into_iter().flatten() {
            consider(fk, other);
        }
        // A reader whose key's scope covers this write, or the reverse.
        if let Some(coll) = index.by_collection.get(&fk.collection) {
            let row_reads: &[KeyRef] = match fk.scope {
                KeyScope::Row => &[],
                KeyScope::Collection => &coll.row_reads,
            };
            for &r in coll.scoped_reads.iter().chain(row_reads) {
                if conflicts(index.read(r), fk) {
                    consider(fk, r.0);
                }
            }
        }
    }

    match best {
        Some((pos, key, engine)) => (
            pos,
            Some(ConflictKey {
                tenant,
                engine,
                collection: key.1,
            }),
        ),
        None => (0, None),
    }
}
