// SPDX-License-Identifier: BUSL-1.1

//! Per-key classification of a lock request, and the grant that installs a
//! requester on a key it can take now.

use std::collections::VecDeque;

use smallvec::smallvec;

use crate::control::cluster::calvin::scheduler::lock::lock_entry::{LockEntry, LockMode};
use crate::control::cluster::calvin::scheduler::lock::lock_key::{LockKey, TxnId};

use super::types::LockManager;

/// Where one requested key stands for one requester.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum KeyState {
    /// No entry: the requester can take the key.
    Free,
    /// The requester already holds the key in a mode that covers the request,
    /// or can raise its mode in place because every other holder is
    /// compatible with the raised mode.
    Held,
    /// Other holders hold the key in a compatible mode and nobody waits: the
    /// requester can join them.
    Join,
    /// The requester must wait: an incompatible holder, an earlier waiter, or
    /// its own earlier wait on this key.
    Conflict,
}

impl LockManager {
    /// Classify `key` requested in `mode` by `txn`.
    pub(super) fn classify(&self, txn: TxnId, key: &LockKey, mode: LockMode) -> KeyState {
        let Some(entry) = self.table.get(key) else {
            return KeyState::Free;
        };
        if let Some(own) = entry.mode_of(txn) {
            let raised = own.merge(mode);
            if raised == own || entry.others_compatible(txn, raised) {
                return KeyState::Held;
            }
            return KeyState::Conflict;
        }
        if entry.has_waiter(txn) {
            return KeyState::Conflict;
        }
        if entry.mode.compatible(mode) && entry.waiters.is_empty() {
            return KeyState::Join;
        }
        KeyState::Conflict
    }

    /// Install `txn` on `key` in `mode`. The caller classified the key as
    /// [`KeyState::Free`], [`KeyState::Held`], or [`KeyState::Join`].
    pub(super) fn grant(&mut self, txn: TxnId, key: &LockKey, mode: LockMode) {
        match self.table.get_mut(key) {
            None => {
                self.table.insert(
                    key.clone(),
                    LockEntry {
                        mode,
                        holders: smallvec![(txn, mode)],
                        waiters: VecDeque::new(),
                    },
                );
            }
            // A holder raises its own mode in place. Classification admitted
            // the raised mode against every other holder.
            Some(entry) => entry.hold(txn, mode),
        }
    }

    /// Enqueue `txn` as a waiter on a [`KeyState::Conflict`] key.
    ///
    /// A txn that holds the key and needs a stronger mode drops its own hold
    /// and waits for the merged mode. Its read degrades to plain OCC until the
    /// grant, and the key can drain so the grant can fire.
    pub(super) fn enqueue(&mut self, txn: TxnId, key: &LockKey, mode: LockMode) {
        let Some(entry) = self.table.get_mut(key) else {
            return;
        };
        if entry.has_waiter(txn) {
            return;
        }
        let wanted = match entry.mode_of(txn) {
            Some(own) => {
                entry.drop_holder(txn);
                own.merge(mode)
            }
            None => mode,
        };
        entry.waiters.push_back((txn, wanted));
    }
}
