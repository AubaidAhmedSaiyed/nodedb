// SPDX-License-Identifier: Apache-2.0

//! `RerankCodec` wrapper for SQ8 scalar quantization.
//!
//! Bridges `Sq8Codec` (which implements `VectorCodec` with associated types)
//! into the object-safe `RerankCodec` trait used by the rerank sidecar.

use nodedb_codec::vector_quant::layout::UnifiedQuantizedVectorRef;

use crate::{
    quantize::sq8::Sq8Codec,
    rerank::codec::{CodecName, PreparedQuery, RerankCodec},
    rerank::types::RerankError,
};

// ── packed_bits_len helper ────────────────────────────────────────────────────

/// SQ8 is 8 bpw, so packed_bits_len == dim bytes.
#[inline]
fn sq8_packed_bits_len(dim: usize) -> usize {
    dim
}

// ── Sq8Rerank ─────────────────────────────────────────────────────────────────

/// Object-safe `RerankCodec` wrapper around `Sq8Codec`.
///
/// `train()` calls `Sq8Codec::calibrate` to fit per-dimension min/max from a
/// sample of vectors. Subsequent `encode` / `distance_prepared` calls use the
/// calibrated codec.
pub struct Sq8Rerank {
    codec: Sq8Codec,
    dim: usize,
}

impl Sq8Rerank {
    /// Create an untrained wrapper with a default-calibrated codec.
    ///
    /// The default codec treats every dimension's min as 0.0 and max as 1.0,
    /// which is suitable for normalized embeddings. For best accuracy call
    /// `train()` with representative samples before encoding.
    pub fn new(dim: usize) -> Self {
        // Calibrated over the unit range so encoding is functional before
        // train() is called.
        Self {
            codec: Sq8Codec::unit_range(dim),
            dim,
        }
    }

    /// Wrap an already-trained `Sq8Codec`.
    pub fn from_codec(codec: Sq8Codec) -> Self {
        let dim = codec.dim;
        Self { codec, dim }
    }
}

impl RerankCodec for Sq8Rerank {
    /// Encode a full-precision vector to SQ8 bytes.
    ///
    /// The serialized form is the raw `UnifiedQuantizedVector` buffer
    /// (`as_bytes()`), which embeds a 32-byte `QuantHeader` followed by
    /// `dim` packed INT8 codes.
    fn encode(&self, v: &[f32]) -> Result<Vec<u8>, RerankError> {
        if v.len() != self.dim {
            return Err(RerankError::BadInput(format!(
                "sq8 encode: vector len {} != codec dim {}",
                v.len(),
                self.dim
            )));
        }
        use nodedb_codec::vector_quant::codec::VectorCodec as _;
        let quantized = self.codec.encode(v);
        Ok(quantized.as_ref().as_bytes().to_vec())
    }

    /// Prepare the query for repeated distance calls.
    ///
    /// SQ8 is asymmetric: the query is kept in full FP32 precision while
    /// candidates are INT8. The prepared form is therefore `PreparedQuery::Raw`.
    fn prepare_query(&self, q: &[f32]) -> Result<PreparedQuery, RerankError> {
        if q.len() != self.dim {
            return Err(RerankError::BadInput(format!(
                "sq8 prepare_query: query len {} != codec dim {}",
                q.len(),
                self.dim
            )));
        }
        Ok(PreparedQuery::Raw(q.to_vec()))
    }

    /// Compute asymmetric L2 distance from a prepared FP32 query to an
    /// SQ8-encoded candidate.
    fn distance_prepared(
        &self,
        prepared: &PreparedQuery,
        encoded: &[u8],
    ) -> Result<f32, RerankError> {
        let q = match prepared {
            PreparedQuery::Raw(q) => q,
            _ => {
                return Err(RerankError::BadInput(
                    "sq8 distance: expected PreparedQuery::Raw".to_string(),
                ));
            }
        };

        let packed_len = sq8_packed_bits_len(self.dim);
        let uqv_ref = UnifiedQuantizedVectorRef::from_bytes(encoded, packed_len).map_err(|e| {
            RerankError::BadInput(format!("sq8 distance: failed to parse encoded bytes: {e}"))
        })?;

        let packed = uqv_ref.packed_bits();
        let dist = self.codec.asymmetric_l2(q, packed);
        Ok(dist)
    }

    fn name(&self) -> CodecName {
        CodecName::Sq8
    }

    fn to_bytes(&self) -> Result<Vec<u8>, RerankError> {
        Ok(self.codec.to_bytes())
    }

    /// Calibrate from a sample of vectors.
    ///
    /// Replaces the current codec state. Requires at least one sample.
    fn train(&mut self, samples: &[&[f32]]) -> Result<(), RerankError> {
        if samples.is_empty() {
            return Err(RerankError::BadInput(
                "sq8 train: empty sample set".to_string(),
            ));
        }
        self.codec = Sq8Codec::calibrate(samples, self.dim)
            .map_err(|e| RerankError::BadInput(format!("sq8 train: {e}")))?;
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 16;
    const EPS: f32 = 1e-2;

    fn make_vec(base: f32) -> Vec<f32> {
        (0..DIM).map(|i| base + i as f32 * 0.01).collect()
    }

    fn trained_codec() -> Sq8Rerank {
        let samples: Vec<Vec<f32>> = (0..50).map(|i| make_vec(i as f32 * 0.1)).collect();
        let refs: Vec<&[f32]> = samples.iter().map(|v| v.as_slice()).collect();
        let mut codec = Sq8Rerank::new(DIM);
        codec.train(&refs).expect("train must succeed");
        codec
    }

    #[test]
    fn round_trip_returns_finite_distance() {
        let codec = trained_codec();
        let v1 = make_vec(0.5);
        let v2 = make_vec(1.0);

        let enc = codec.encode(&v1).expect("encode v1");
        let prepared = codec.prepare_query(&v2).expect("prepare_query v2");
        let dist = codec
            .distance_prepared(&prepared, &enc)
            .expect("distance_prepared");
        assert!(dist.is_finite(), "expected finite distance, got {dist}");
        assert!(dist >= 0.0, "expected non-negative distance, got {dist}");
    }

    #[test]
    fn identical_vectors_small_distance() {
        let codec = trained_codec();
        let v = make_vec(0.5);

        let enc = codec.encode(&v).expect("encode");
        let prepared = codec.prepare_query(&v).expect("prepare_query");
        let dist = codec
            .distance_prepared(&prepared, &enc)
            .expect("distance_prepared");
        assert!(dist.is_finite());
        assert!(
            dist < EPS,
            "identical vectors should have near-zero distance, got {dist}"
        );
    }

    #[test]
    fn wrong_prepared_query_variant_returns_bad_input() {
        let codec = trained_codec();
        let v = make_vec(0.5);
        let enc = codec.encode(&v).expect("encode");
        let bad_prepared = PreparedQuery::Bytes(vec![0u8; 8]);

        let result = codec.distance_prepared(&bad_prepared, &enc);
        assert!(result.is_err(), "expected BadInput error");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("Raw"),
            "error message should mention Raw, got: {msg}"
        );
    }

    #[test]
    fn name_returns_sq8() {
        let codec = Sq8Rerank::new(DIM);
        assert_eq!(codec.name(), CodecName::Sq8);
    }

    #[test]
    fn train_calibrates_without_error() {
        let mut codec = Sq8Rerank::new(DIM);
        let samples: Vec<Vec<f32>> = (0..20).map(|i| make_vec(i as f32 * 0.05)).collect();
        let refs: Vec<&[f32]> = samples.iter().map(|v| v.as_slice()).collect();
        codec.train(&refs).expect("train must succeed");

        // After training, encode + distance must still work.
        let v = make_vec(0.5);
        let enc = codec.encode(&v).expect("encode after train");
        let prep = codec.prepare_query(&v).expect("prepare after train");
        let dist = codec
            .distance_prepared(&prep, &enc)
            .expect("distance after train");
        assert!(dist.is_finite());
    }

    #[test]
    fn wrong_dim_encode_returns_error() {
        let codec = Sq8Rerank::new(DIM);
        let bad = vec![0.0f32; DIM + 1];
        assert!(codec.encode(&bad).is_err());
    }
}
