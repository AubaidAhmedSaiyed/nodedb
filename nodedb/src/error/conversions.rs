// SPDX-License-Identifier: BUSL-1.1

//! `From` impls that build a [`super::Error`] from `nodedb-physical` error
//! types (wire decoding, physical-plan conversion) and from the
//! `nodedb-types` collection-key error. Kept apart from the enum
//! definition in `types.rs` so a new physical-layer error source has one
//! obvious home instead of growing the enum file further.

use super::Error;

impl From<nodedb_physical::physical_plan::wire::WireError> for Error {
    fn from(e: nodedb_physical::physical_plan::wire::WireError) -> Self {
        Error::Internal {
            detail: e.to_string(),
        }
    }
}

impl From<nodedb_physical::ConvertError> for Error {
    fn from(e: nodedb_physical::ConvertError) -> Self {
        use nodedb_physical::ConvertError;
        match e {
            ConvertError::PlanError(detail) => Error::PlanError { detail },
            ConvertError::BadRequest(detail) => Error::BadRequest { detail },
            ConvertError::LimitExceeded {
                limit_name,
                value,
                max,
            } => Error::LimitExceeded {
                limit_name,
                value,
                max,
            },
            ConvertError::Surrogate(s) => Error::Internal {
                detail: s.to_string(),
            },
            ConvertError::Serialization(detail) => Error::Serialization {
                format: "msgpack".into(),
                detail,
            },
            ConvertError::Other(detail) => Error::Internal { detail },
        }
    }
}

/// A qualified collection name that does not carry its database's qualifier
/// reached a placement or surrogate path. Every qualified name is built by
/// `QualifiedCollection::new`, so this is an internal invariant break.
impl From<nodedb_types::CollectionKeyError> for Error {
    fn from(e: nodedb_types::CollectionKeyError) -> Self {
        Error::Internal {
            detail: e.to_string(),
        }
    }
}
