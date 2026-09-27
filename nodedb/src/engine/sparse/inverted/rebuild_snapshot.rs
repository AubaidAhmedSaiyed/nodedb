// SPDX-License-Identifier: BUSL-1.1

//! The off-core half of a collection rebuild: read the pinned snapshot and
//! derive the collection's canonical index rows from it.
//!
//! Every type here is `Send` and touches no `!Send` engine state. The redb
//! read transaction was opened on the owning core by
//! `InvertedIndex::begin_rebuild`, in the same step that opened the write
//! journal, so the snapshot and the journal meet with no gap.

use std::collections::{BTreeMap, BTreeSet};

use redb::{ReadTransaction, ReadableTable as _};

use nodedb_fts::posting::Posting;

use super::errors::inverted_err;
use crate::engine::sparse::fts_redb::tables::{DOC_LENGTHS, POSTINGS};

/// Upper bound for the `term` component of a posting range scan.
const MAX_TERM: &str = "\u{10ffff}";

/// A pinned snapshot of one collection's index, ready to read.
pub struct FtsRebuildTicket {
    token: u64,
    database_id: u64,
    tid: u64,
    collection: String,
    txn: ReadTransaction,
}

/// One collection's postings and document lengths as the snapshot holds them.
pub struct FtsSnapshot {
    token: u64,
    database_id: u64,
    tid: u64,
    collection: String,
    postings: Vec<(String, Vec<Posting>)>,
    doc_lengths: Vec<(u32, u32)>,
}

/// The canonical index rows of one collection, derived from a snapshot.
pub struct FtsRebuilt {
    pub(super) token: u64,
    pub(super) database_id: u64,
    pub(super) tid: u64,
    pub(super) collection: String,
    /// One list per term, one posting per document, ordered by surrogate.
    pub(super) postings: Vec<(String, Vec<Posting>)>,
    /// `(surrogate, token count)` per indexed document.
    pub(super) doc_lengths: Vec<(u32, u32)>,
    /// `(surrogate, distinct terms)` per document with a posting.
    pub(super) doc_terms: Vec<(u32, Vec<String>)>,
    /// Number of indexed documents.
    pub(super) doc_count: u32,
    /// Sum of the indexed documents' token counts.
    pub(super) total_tokens: u64,
}

impl FtsRebuildTicket {
    pub(super) fn new(
        token: u64,
        database_id: u64,
        tid: u64,
        collection: String,
        txn: ReadTransaction,
    ) -> Self {
        Self {
            token,
            database_id,
            tid,
            collection,
            txn,
        }
    }

    /// The token that ties this rebuild to its journal.
    pub fn token(&self) -> u64 {
        self.token
    }

    /// Read the collection's postings and document lengths from the pinned
    /// snapshot. Runs on any thread. The snapshot is released on return.
    pub fn read(self) -> crate::Result<FtsSnapshot> {
        let db = self.database_id;
        let t = self.tid;
        let coll = self.collection.as_str();

        let postings_table = self
            .txn
            .open_table(POSTINGS)
            .map_err(|e| inverted_err("rebuild open postings", e))?;
        let mut postings = Vec::new();
        for entry in postings_table
            .range((db, t, coll, "")..=(db, t, coll, MAX_TERM))
            .map_err(|e| inverted_err("rebuild postings range", e))?
        {
            let (key, value) = entry.map_err(|e| inverted_err("rebuild postings entry", e))?;
            let list: Vec<Posting> = zerompk::from_msgpack(value.value())
                .map_err(|e| inverted_err("rebuild decode postings", e))?;
            postings.push((key.value().3.to_string(), list));
        }

        let lengths_table = self
            .txn
            .open_table(DOC_LENGTHS)
            .map_err(|e| inverted_err("rebuild open doc_lengths", e))?;
        let mut doc_lengths = Vec::new();
        for entry in lengths_table
            .range((db, t, coll, 0u32)..=(db, t, coll, u32::MAX))
            .map_err(|e| inverted_err("rebuild doc_lengths range", e))?
        {
            let (key, value) = entry.map_err(|e| inverted_err("rebuild doc_length entry", e))?;
            let len: u32 = zerompk::from_msgpack(value.value())
                .map_err(|e| inverted_err("rebuild decode doc_length", e))?;
            doc_lengths.push((key.value().3, len));
        }

        Ok(FtsSnapshot {
            token: self.token,
            database_id: self.database_id,
            tid: self.tid,
            collection: self.collection,
            postings,
            doc_lengths,
        })
    }
}

impl FtsSnapshot {
    /// The token that ties this rebuild to its journal.
    pub fn token(&self) -> u64 {
        self.token
    }

    /// Derive the canonical rows: one posting per document per term, the
    /// term set of each document, and corpus stats counted from the
    /// document lengths. Runs on any thread.
    pub fn compact(self) -> FtsRebuilt {
        let mut postings = Vec::with_capacity(self.postings.len());
        let mut terms_by_doc: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
        for (term, mut list) in self.postings {
            list.sort_unstable_by(|a, b| {
                a.doc_id
                    .as_u32()
                    .cmp(&b.doc_id.as_u32())
                    .then(b.term_freq.cmp(&a.term_freq))
            });
            list.dedup_by_key(|p| p.doc_id.as_u32());
            if list.is_empty() {
                continue;
            }
            for posting in &list {
                terms_by_doc
                    .entry(posting.doc_id.as_u32())
                    .or_default()
                    .insert(term.clone());
            }
            postings.push((term, list));
        }
        let doc_terms = terms_by_doc
            .into_iter()
            .map(|(doc, terms)| (doc, terms.into_iter().collect()))
            .collect();
        let doc_count = u32::try_from(self.doc_lengths.len()).unwrap_or(u32::MAX);
        let total_tokens = self
            .doc_lengths
            .iter()
            .map(|&(_, len)| u64::from(len))
            .sum();
        FtsRebuilt {
            token: self.token,
            database_id: self.database_id,
            tid: self.tid,
            collection: self.collection,
            postings,
            doc_lengths: self.doc_lengths,
            doc_terms,
            doc_count,
            total_tokens,
        }
    }
}

impl FtsRebuilt {
    /// The token that ties this rebuild to its journal.
    pub fn token(&self) -> u64 {
        self.token
    }

    /// The collection this rebuild covers.
    pub fn collection(&self) -> &str {
        &self.collection
    }
}
