// SPDX-License-Identifier: BUSL-1.1

//! The post-image policy an admission evaluates before it proposes.

use nodedb_types::CrdtPreviewResult;

pub trait CrdtPostImagePolicy: Send + Sync {
    fn evaluate(&self, preview: &CrdtPreviewResult) -> crate::Result<()>;
}

/// Explicit policy for trusted internal callers.
pub struct TrustedInternalCrdtPolicy;

impl CrdtPostImagePolicy for TrustedInternalCrdtPolicy {
    fn evaluate(&self, _preview: &CrdtPreviewResult) -> crate::Result<()> {
        Ok(())
    }
}
