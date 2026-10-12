// SPDX-License-Identifier: Apache-2.0

//! Decoders that turn `QueryResult` rows back into typed domain structs.
//!
//! Each row shape has one parser, shared by the `NodeDb` trait default impls
//! and the feature-gated `remote` / `native` clients. A system-catalog column
//! layout change has one place to fix.
//!
//! A decoder compiles under the features of its callers. `search_hit` serves
//! only the client search methods, so it compiles with `native` or `remote`.

pub(crate) mod dropped_collection;
#[cfg(any(feature = "native", feature = "remote"))]
pub(crate) mod search_hit;
pub(crate) mod value;

pub(crate) use dropped_collection::parse_dropped_collection_rows;
#[cfg(any(feature = "native", feature = "remote"))]
pub(crate) use search_hit::{HitSource, decode_search_hits};
