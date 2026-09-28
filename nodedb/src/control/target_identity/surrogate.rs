// SPDX-License-Identifier: BUSL-1.1

//! Assign a fresh, catalog-registered surrogate for a row written into a
//! target collection on behalf of another operation.

use nodedb_types::{CollectionKey, Surrogate, TenantId, extract_pk_value};

use super::pk::TargetPk;
use crate::control::state::SharedState;

/// Assign a fresh, registered surrogate for one written row on the TARGET's
/// primary key. `target` is the target collection's canonical key.
pub(crate) fn assign_target_surrogate(
    state: &SharedState,
    target: CollectionKey<'_>,
    tenant_id: TenantId,
    target_pk: &TargetPk,
    body: &[u8],
) -> crate::Result<Surrogate> {
    match target_pk {
        TargetPk::AutoRowId => {
            let (surrogate, _) = state.surrogate_assigner.assign_fresh(target, tenant_id)?;
            Ok(surrogate)
        }
        TargetPk::Field { name, declared } => match extract_pk_value(body, name) {
            // The empty string is a key like any other. Minting a fresh
            // surrogate for it would let two rows share it.
            Some(pk) => state
                .surrogate_assigner
                .assign(target, tenant_id, pk.as_bytes()),
            // No usable key value on a DDL-declared PRIMARY KEY: NOT NULL is
            // implied, so refuse rather than mint a surrogate for a row that
            // plain INSERT would already reject.
            None if *declared => Err(crate::Error::RejectedConstraint {
                collection: target.name().to_string(),
                constraint: "not_null".to_string(),
                detail: format!("primary key '{name}' cannot be NULL or omitted"),
            }),
            // Undeclared `id`-by-convention field: mint a fresh unique
            // surrogate rather than collapsing every keyless row onto one
            // binding. The row's identity is its document storage key, so
            // the allocator binds the hex form.
            _ => {
                let (surrogate, _) = state.surrogate_assigner.assign_fresh(target, tenant_id)?;
                Ok(surrogate)
            }
        },
    }
}
