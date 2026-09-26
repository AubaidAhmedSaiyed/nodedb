// SPDX-License-Identifier: Apache-2.0

//! Methods on `VectorCollection` for building the collection-level
//! codec-dispatch index (RaBitQ, BBQ).
//!
//! All `impl VectorCollection` blocks in `collection/` extend the same type.

use super::codec_dispatch::{CollectionCodec, build_collection_codec};
use super::lifecycle::VectorCollection;

use crate::error::VectorError;

impl VectorCollection {
    /// Every live FP32 vector of the sealed segments, keyed by its global
    /// vector id. The collection-level codec index covers the sealed
    /// segments only: search reads the growing and building segments by
    /// brute force beside it.
    pub(crate) fn gather_sealed_vectors_fp32(&self) -> Vec<(u32, Vec<f32>)> {
        let mut out = Vec::new();
        for seg in &self.sealed {
            let n = seg.index.len();
            for i in 0..n as u32 {
                if !seg.index.is_deleted(i)
                    && let Some(v) = seg.index.get_vector(i)
                {
                    out.push((seg.base_id + i, v.to_vec()));
                }
            }
        }
        out
    }

    /// Build a codec-dispatched index over the sealed vectors using the
    /// requested quantization. Replaces any existing dispatch index for
    /// this collection. Idempotent.
    ///
    /// Returns a reference to the new index, or `Ok(None)` if the
    /// quantization tag is not supported (falls back to per-segment Sq8/PQ
    /// paths) or there are no vectors to train on.
    pub fn build_codec_dispatch(
        &mut self,
        quantization: &str,
    ) -> Result<Option<&CollectionCodec>, VectorError> {
        let vectors = self.gather_sealed_vectors_fp32();
        let dim = self.dim;
        let m = self.params.m;
        let ef_construction = self.params.ef_construction;
        let seed = 42_u64;
        self.codec_dispatch =
            build_collection_codec(quantization, &vectors, dim, m, ef_construction, seed)?;
        Ok(self.codec_dispatch.as_ref())
    }
}
