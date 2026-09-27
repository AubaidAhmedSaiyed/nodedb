// SPDX-License-Identifier: BUSL-1.1

//! The owner-bearing catalog object kinds `DROP USER` reassigns or deletes.

use crate::control::security::catalog::auth_types::object_type;

/// Every catalog object kind that carries an owner reference. The nine
/// parent-replicated kinds each own a primary `Stored*` record with an
/// in-band `owner` field; `Index` is the standalone path (a bare
/// `StoredOwner` row with no parent record).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum OwnerKind {
    Collection,
    Function,
    Procedure,
    Trigger,
    MaterializedView,
    StreamingMaterializedView,
    Sequence,
    Schedule,
    ChangeStream,
    ContinuousAggregate,
    Index,
}

impl OwnerKind {
    /// Map a persisted `StoredOwner.object_type` to its kind. `None`
    /// means the writer of that owner row introduced a kind this module
    /// does not yet handle — the caller turns that into a hard error so
    /// the drop fails closed instead of leaking a dangling reference.
    pub(super) fn from_object_type(s: &str) -> Option<Self> {
        Some(match s {
            object_type::COLLECTION => Self::Collection,
            object_type::FUNCTION => Self::Function,
            object_type::PROCEDURE => Self::Procedure,
            object_type::TRIGGER => Self::Trigger,
            object_type::MATERIALIZED_VIEW => Self::MaterializedView,
            object_type::STREAMING_MATERIALIZED_VIEW => Self::StreamingMaterializedView,
            object_type::SEQUENCE => Self::Sequence,
            object_type::SCHEDULE => Self::Schedule,
            object_type::CHANGE_STREAM => Self::ChangeStream,
            object_type::CONTINUOUS_AGGREGATE => Self::ContinuousAggregate,
            object_type::INDEX => Self::Index,
            _ => return None,
        })
    }

    pub(super) fn as_object_type(&self) -> &'static str {
        match self {
            Self::Collection => object_type::COLLECTION,
            Self::Function => object_type::FUNCTION,
            Self::Procedure => object_type::PROCEDURE,
            Self::Trigger => object_type::TRIGGER,
            Self::MaterializedView => object_type::MATERIALIZED_VIEW,
            Self::StreamingMaterializedView => object_type::STREAMING_MATERIALIZED_VIEW,
            Self::Sequence => object_type::SEQUENCE,
            Self::Schedule => object_type::SCHEDULE,
            Self::ChangeStream => object_type::CHANGE_STREAM,
            Self::ContinuousAggregate => object_type::CONTINUOUS_AGGREGATE,
            Self::Index => object_type::INDEX,
        }
    }
}
