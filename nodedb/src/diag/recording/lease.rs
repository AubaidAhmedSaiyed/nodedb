// SPDX-License-Identifier: BUSL-1.1

//! Capture sites for descriptor leases the renewal loop could not maintain.

use faultbox::{Capture, EventKind, error_chain_of};

use super::shared::error_class;
use crate::diag::context;

/// Report a descriptor lease the renewal loop could not refresh or release.
/// Called only from the renewal loop's error arms. `step` is `"renew"` or
/// `"release"`.
pub fn descriptor_lease_not_renewed(
    err: &crate::Error,
    step: &'static str,
    descriptor: &nodedb_cluster::DescriptorId,
    version: u64,
    node_id: u64,
) {
    let class = error_class(err);
    let descriptor = format!("{descriptor:?}");
    let ctx = context::DescriptorLeaseNotRenewed {
        step,
        descriptor: &descriptor,
        version,
        node_id,
        error_class: &class,
    };
    let _ = Capture::new(
        EventKind::Error,
        "descriptor lease renewal: the lease was neither refreshed nor released",
    )
    .error_chain(error_chain_of(err))
    .domain(&ctx)
    .with_backtrace()
    .emit();
}
