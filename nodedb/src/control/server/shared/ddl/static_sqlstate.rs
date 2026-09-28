// SPDX-License-Identifier: BUSL-1.1

//! A DDL error's SQLSTATE as a `&'static str`.
//!
//! `DdlError` holds its SQLSTATE as a `String`. The error renderers take a
//! `&'static str`. This module interns each distinct SQLSTATE once, so a
//! DDL error keeps its exact SQLSTATE on every surface.
//!
//! The interned set is bounded. Every DDL SQLSTATE comes from a server
//! constant or literal, never from client input. A string that is not a
//! well-formed SQLSTATE is never interned.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use nodedb_types::error::sqlstate;

/// The interned form of `state`.
///
/// A well-formed SQLSTATE is five ASCII digits or uppercase letters. Any
/// other string has no class, so it renders `XX000`.
pub fn static_sqlstate(state: &str) -> &'static str {
    if !is_well_formed(state) {
        return sqlstate::INTERNAL_ERROR;
    }
    static INTERNED: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let interned = INTERNED.get_or_init(|| Mutex::new(HashSet::new()));
    // A panic while the lock was held leaves the set intact: every insert
    // is one complete `&'static str`.
    let mut set = match interned.lock() {
        Ok(set) => set,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Some(found) = set.get(state).copied() {
        return found;
    }
    let leaked: &'static str = Box::leak(state.to_owned().into_boxed_str());
    set.insert(leaked);
    leaked
}

/// Five characters, each an ASCII digit or uppercase letter.
fn is_well_formed(state: &str) -> bool {
    state.len() == 5
        && state
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sqlstate_keeps_its_value() {
        assert_eq!(static_sqlstate("42710"), "42710");
        assert_eq!(static_sqlstate("2BP01"), "2BP01");
    }

    #[test]
    fn one_sqlstate_interns_once() {
        let first = static_sqlstate("42P07");
        let second = static_sqlstate(&String::from("42P07"));
        assert!(std::ptr::eq(first, second));
    }

    #[test]
    fn a_malformed_sqlstate_is_internal() {
        assert_eq!(static_sqlstate("bad"), sqlstate::INTERNAL_ERROR);
        assert_eq!(static_sqlstate("42p01"), sqlstate::INTERNAL_ERROR);
        assert_eq!(static_sqlstate("423010"), sqlstate::INTERNAL_ERROR);
    }
}
