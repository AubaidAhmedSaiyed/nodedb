// SPDX-License-Identifier: Apache-2.0

//! Per-collection codec selection. Wraps the generic `HnswCodecIndex<C>`
//! for codecs other than Sq8 (which retains its specialised fast path
//! in `quantize.rs` / `search.rs`).

use nodedb_codec::vector_quant::bbq::BbqCodec;
use nodedb_codec::vector_quant::rabitq::RaBitQCodec;

use crate::codec_index::HnswCodecIndex;
use crate::error::VectorError;

/// One built codec-index per collection (other than Sq8). Variants match
/// the publicly-selectable quantization choices that route through
/// `HnswCodecIndex`.
#[non_exhaustive]
pub enum CollectionCodec {
    RaBitQ(HnswCodecIndex<RaBitQCodec>),
    Bbq(HnswCodecIndex<BbqCodec>),
}

impl CollectionCodec {
    /// Forwarding `search` so the collection layer doesn't have to match
    /// on the variant for the common case.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<crate::codec_index::CodecSearchResult>, VectorError> {
        match self {
            Self::RaBitQ(idx) => idx.search(query, k, ef_search),
            Self::Bbq(idx) => idx.search(query, k, ef_search),
        }
    }

    /// Forwarding `insert`.
    pub fn insert(&mut self, id: u32, v: &[f32]) -> Result<(), VectorError> {
        match self {
            Self::RaBitQ(idx) => idx.insert(id, v),
            Self::Bbq(idx) => idx.insert(id, v),
        }
    }

    /// Total nodes (including deleted).
    pub fn len(&self) -> usize {
        match self {
            Self::RaBitQ(idx) => idx.len(),
            Self::Bbq(idx) => idx.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Quantization tag for stats reporting.
    pub fn quantization(&self) -> &'static str {
        match self {
            Self::RaBitQ(_) => "rabitq",
            Self::Bbq(_) => "bbq",
        }
    }
}

/// Build a `CollectionCodec` from a quantization tag and training vectors.
///
/// Returns `Ok(None)` for unsupported or unrecognised tags (e.g. "sq8",
/// "pq", "none") — those variants use separate per-segment code paths — and
/// for an empty vector set. A vector without `dim` components fails with
/// [`VectorError::DimensionMismatch`].
///
/// Each entry is `(id, vector)`; the index reports `id` in its results, so
/// it must be the collection's global vector id.
pub fn build_collection_codec(
    quantization: &str,
    vectors: &[(u32, Vec<f32>)],
    dim: usize,
    m: usize,
    ef_construction: usize,
    seed: u64,
) -> Result<Option<CollectionCodec>, VectorError> {
    if vectors.is_empty() {
        return Ok(None);
    }
    // Calibration reads every vector as `dim` components; check before it.
    for (_, v) in vectors {
        crate::error::check_dim(dim, v.len())?;
    }
    let refs: Vec<&[f32]> = vectors.iter().map(|(_, v)| v.as_slice()).collect();
    match quantization {
        "rabitq" => {
            let codec = RaBitQCodec::calibrate(&refs, dim, seed);
            let mut idx = HnswCodecIndex::new(dim, m, ef_construction, codec, seed);
            for (id, v) in vectors {
                idx.insert(*id, v)?;
            }
            Ok(Some(CollectionCodec::RaBitQ(idx)))
        }
        "bbq" => {
            let codec = BbqCodec::calibrate(&refs, dim, 3);
            let mut idx = HnswCodecIndex::new(dim, m, ef_construction, codec, seed);
            for (id, v) in vectors {
                idx.insert(*id, v)?;
            }
            Ok(Some(CollectionCodec::Bbq(idx)))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<(u32, Vec<f32>)> {
        (0..n)
            .map(|i| {
                let v = (0..dim).map(|d| (i * dim + d) as f32 * 0.01).collect();
                (i as u32, v)
            })
            .collect()
    }

    #[test]
    fn build_rabitq_returns_some() {
        let vecs = make_vectors(50, 8);
        let result = build_collection_codec("rabitq", &vecs, 8, 16, 100, 42).unwrap();
        assert!(
            matches!(result, Some(CollectionCodec::RaBitQ(_))),
            "expected RaBitQ variant"
        );
    }

    #[test]
    fn build_bbq_returns_some() {
        let vecs = make_vectors(50, 8);
        let result = build_collection_codec("bbq", &vecs, 8, 16, 100, 42).unwrap();
        assert!(
            matches!(result, Some(CollectionCodec::Bbq(_))),
            "expected Bbq variant"
        );
    }

    #[test]
    fn unknown_codec_returns_none() {
        let vecs = make_vectors(50, 8);
        let result = build_collection_codec("unknown_codec", &vecs, 8, 16, 100, 42).unwrap();
        assert!(result.is_none(), "unknown codec should return None");
    }

    #[test]
    fn sq8_tag_returns_none() {
        let vecs = make_vectors(50, 8);
        let result = build_collection_codec("sq8", &vecs, 8, 16, 100, 42).unwrap();
        assert!(
            result.is_none(),
            "sq8 tag should fall through to per-segment path"
        );
    }

    #[test]
    fn empty_vectors_returns_none() {
        let result = build_collection_codec("rabitq", &[], 8, 16, 100, 42).unwrap();
        assert!(result.is_none(), "empty vectors should return None");
    }

    #[test]
    fn len_and_is_empty() {
        let vecs = make_vectors(20, 4);
        let codec = build_collection_codec("bbq", &vecs, 4, 8, 50, 1)
            .unwrap()
            .unwrap();
        assert_eq!(codec.len(), 20);
        assert!(!codec.is_empty());
    }

    #[test]
    fn quantization_tag() {
        let vecs = make_vectors(10, 4);
        let rabitq = build_collection_codec("rabitq", &vecs, 4, 8, 50, 1)
            .unwrap()
            .unwrap();
        assert_eq!(rabitq.quantization(), "rabitq");
        let bbq = build_collection_codec("bbq", &vecs, 4, 8, 50, 1)
            .unwrap()
            .unwrap();
        assert_eq!(bbq.quantization(), "bbq");
    }

    #[test]
    fn wrong_dimension_query_is_a_typed_error() {
        let vecs = make_vectors(20, 4);
        for tag in ["rabitq", "bbq"] {
            let mut codec = build_collection_codec(tag, &vecs, 4, 8, 50, 1)
                .unwrap()
                .unwrap();
            assert!(matches!(
                codec.search(&[0.0; 3], 5, 20),
                Err(VectorError::DimensionMismatch {
                    expected: 4,
                    got: 3
                })
            ));
            assert!(matches!(
                codec.insert(99, &[0.0; 5]),
                Err(VectorError::DimensionMismatch {
                    expected: 4,
                    got: 5
                })
            ));
        }
        let short = vec![(0_u32, vec![0.0_f32; 3])];
        assert!(matches!(
            build_collection_codec("rabitq", &short, 4, 8, 50, 1),
            Err(VectorError::DimensionMismatch {
                expected: 4,
                got: 3
            })
        ));
    }
}
