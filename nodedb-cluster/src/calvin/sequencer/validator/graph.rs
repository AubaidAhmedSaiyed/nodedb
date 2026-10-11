// SPDX-License-Identifier: BUSL-1.1

//! The reader/writer index of a candidate batch and the reorder edges it
//! implies.
//!
//! Every map is a `BTreeMap` and every list holds ascending transaction
//! indices, so every iteration that feeds output is deterministic.

use std::collections::{BTreeMap, BTreeSet};

use super::keys::{FlatKey, KeyId, KeyScope, conflicts};

/// One key of one transaction: `(transaction index, key index)`. The key
/// index points into that transaction's flattened read or write keys.
pub(super) type KeyRef = (usize, usize);

/// One collection's keys, split by scope.
#[derive(Debug, Default)]
pub(super) struct CollectionKeys {
    /// Every collection-scope read key.
    pub(super) scoped_reads: Vec<KeyRef>,
    /// Every collection-scope write key.
    pub(super) scoped_writes: Vec<KeyRef>,
    /// One row read key per transaction that reads rows of the collection.
    /// It stands for all of them: a row key conflicts with a collection-scope
    /// key of its collection whatever its bytes.
    pub(super) row_reads: Vec<KeyRef>,
    /// One row write key per transaction that writes rows of the collection.
    pub(super) row_writes: Vec<KeyRef>,
}

/// The flattened keys of every candidate and the maps that find conflicting
/// pairs without comparing every key against every other.
#[derive(Debug, Default)]
pub(super) struct ConflictIndex {
    /// Flattened read keys, per transaction.
    pub(super) reads: Vec<Vec<FlatKey>>,
    /// Flattened write keys, per transaction.
    pub(super) writes: Vec<Vec<FlatKey>>,
    /// Transactions that read each exact key.
    pub(super) readers_of: BTreeMap<KeyId, Vec<usize>>,
    /// Transactions that write each exact key.
    pub(super) writers_of: BTreeMap<KeyId, Vec<usize>>,
    /// Keys per collection, for the pairs a collection-scope key makes.
    pub(super) by_collection: BTreeMap<String, CollectionKeys>,
}

impl ConflictIndex {
    /// Add transaction `i`'s keys. Transactions are added in index order.
    pub(super) fn add(&mut self, i: usize, reads: Vec<FlatKey>, writes: Vec<FlatKey>) {
        for (k, fk) in reads.iter().enumerate() {
            push_txn(self.readers_of.entry(fk.key_id()).or_default(), i);
            let coll = self.by_collection.entry(fk.collection.clone()).or_default();
            match fk.scope {
                KeyScope::Row => push_key(&mut coll.row_reads, (i, k)),
                KeyScope::Collection => coll.scoped_reads.push((i, k)),
            }
        }
        for (k, fk) in writes.iter().enumerate() {
            push_txn(self.writers_of.entry(fk.key_id()).or_default(), i);
            let coll = self.by_collection.entry(fk.collection.clone()).or_default();
            match fk.scope {
                KeyScope::Row => push_key(&mut coll.row_writes, (i, k)),
                KeyScope::Collection => coll.scoped_writes.push((i, k)),
            }
        }
        self.reads.push(reads);
        self.writes.push(writes);
    }

    /// The read key `r` names.
    pub(super) fn read(&self, (txn, key): KeyRef) -> &FlatKey {
        &self.reads[txn][key]
    }

    /// The write key `w` names.
    pub(super) fn write(&self, (txn, key): KeyRef) -> &FlatKey {
        &self.writes[txn][key]
    }

    /// The reorder edges `reader -> writer` of every conflicting pair, as a
    /// successor set per transaction, and each transaction's in-degree.
    ///
    /// A `BTreeSet` of successors deduplicates edges that several keys imply
    /// and keeps successor iteration deterministic.
    pub(super) fn edges(&self, n: usize) -> (Vec<BTreeSet<usize>>, Vec<usize>) {
        let mut out: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        let mut in_degree: Vec<usize> = vec![0; n];
        let mut add = |r: usize, w: usize| {
            if r != w && out[r].insert(w) {
                in_degree[w] += 1;
            }
        };

        // A reader and a writer of one exact key.
        for (key, readers) in &self.readers_of {
            if let Some(writers) = self.writers_of.get(key) {
                for &r in readers {
                    for &w in writers {
                        add(r, w);
                    }
                }
            }
        }

        // A collection-scope key against the keys of its collection.
        for coll in self.by_collection.values() {
            for &r in &coll.scoped_reads {
                for &w in coll.scoped_writes.iter().chain(&coll.row_writes) {
                    if conflicts(self.read(r), self.write(w)) {
                        add(r.0, w.0);
                    }
                }
            }
            for &r in &coll.row_reads {
                for &w in &coll.scoped_writes {
                    if conflicts(self.read(r), self.write(w)) {
                        add(r.0, w.0);
                    }
                }
            }
        }
        (out, in_degree)
    }
}

/// Append transaction `i` unless it is already the last entry.
fn push_txn(list: &mut Vec<usize>, i: usize) {
    if list.last() != Some(&i) {
        list.push(i);
    }
}

/// Append key `key` unless its transaction already has an entry.
fn push_key(list: &mut Vec<KeyRef>, key: KeyRef) {
    if list.last().map(|(txn, _)| *txn) != Some(key.0) {
        list.push(key);
    }
}
