// SPDX-License-Identifier: BUSL-1.1

//! Internal [`crate::Error`] classification: the public mapping table and
//! the unclassified-failure predicate.

mod public;
mod unclassified;

pub(crate) use public::classify;
pub(crate) use unclassified::is_unclassified_failure;
