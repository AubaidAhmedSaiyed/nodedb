// SPDX-License-Identifier: BUSL-1.1

//! KV predicate DML: `UPDATE` / `DELETE` whose `WHERE` names no primary key.
//! The Control Plane resolves both to a mutation list before they are
//! logged, so the Data Plane reads the predicate only to resolve or stage it.

mod ctx;
mod matches;

pub(in crate::data::executor) use ctx::KvPredicateCtx;
