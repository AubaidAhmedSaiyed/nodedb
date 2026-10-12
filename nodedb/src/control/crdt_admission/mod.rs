// SPDX-License-Identifier: BUSL-1.1

//! Serialized preview, policy, and fenced CRDT apply admission.

pub(crate) mod apply;
mod classify;
mod fence;
mod legs;
mod policy;
mod request;
mod restore;
#[cfg(test)]
mod test_support;
mod workflow;

pub use apply::{
    dispatch_authorized_crdt_apply_admitted, dispatch_authorized_crdt_apply_admitted_outcome,
};
pub use classify::changes_crdt_frontier;
pub use policy::{CrdtPostImagePolicy, TrustedInternalCrdtPolicy};
pub use request::{
    AuthorizedCrdtApplyAdmissionRequest, CrdtAdmissionOutcome, CrdtApplyAdmissionRequest,
    CrdtRestoreAdmissionRequest,
};
pub(crate) use restore::dispatch_crdt_restore_admitted;
