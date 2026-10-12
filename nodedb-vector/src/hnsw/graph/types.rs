// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;

use nodedb_types::vector_dtype::VectorStorageDtype;

/// Result of a k-NN search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Internal node identifier (insertion order).
    pub id: u32,
    /// Distance from the query vector.
    pub distance: f32,
}

/// Per-node vector storage, discriminated by dtype.
///
/// `F32` keeps `Vec<f32>` directly for zero-copy `&[f32]` access via
/// `get_vector`. `Bytes` stores a typed byte buffer for F16/BF16 — the
/// dtype tag mirrors `HnswParams::dtype` and is stored here for convenience
/// so callers that only have a `Node` reference do not need to thread the
/// index params.
pub enum NodeStorage {
    /// F32 storage — 4 bytes per dim; direct `&[f32]` access, no conversion.
    F32(Vec<f32>),
    /// Reduced-precision storage — 2 bytes per dim; dtype identifies encoding.
    Bytes {
        dtype: VectorStorageDtype,
        bytes: Vec<u8>,
    },
}

impl NodeStorage {
    /// Returns a byte-level view of the stored vector regardless of dtype.
    ///
    /// For `F32`, the cast is alignment-safe in the `f32 → u8` direction
    /// (any-alignment requirement for `u8` is satisfied). For `Bytes`, the
    /// slice is returned directly.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            NodeStorage::F32(v) => bytemuck::cast_slice::<f32, u8>(v.as_slice()),
            NodeStorage::Bytes { bytes, .. } => bytes.as_slice(),
        }
    }

    /// Returns `Some(&[f32])` only for `F32` storage.
    ///
    /// For non-F32 storage this returns `None`. Callers that require an f32
    /// view of a reduced-precision node must decode via
    /// [`crate::dtype::cast_to_f32`] or use [`Self::as_bytes`] and
    /// `distance_typed`.
    #[inline]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            NodeStorage::F32(v) => Some(v.as_slice()),
            NodeStorage::Bytes { .. } => None,
        }
    }
}

/// A vector scored against index nodes, in the form the index dtype compares.
///
/// Build one per operation with `HnswIndex::probe` or `HnswIndex::node_probe`.
/// Scoring it against many nodes then copies nothing.
pub(crate) enum Probe<'a> {
    /// f32 values, compared directly against f32 node vectors.
    F32(&'a [f32]),
    /// Bytes in the index dtype, compared with the typed kernels.
    Encoded(Cow<'a, [u8]>),
}

/// A node in the HNSW graph.
pub struct Node {
    /// Vector data in the index's configured storage dtype.
    pub storage: NodeStorage,
    /// Neighbors at each layer this node participates in.
    pub neighbors: Vec<Vec<u32>>,
    /// Tombstone flag for soft-deletion.
    pub deleted: bool,
}

/// Lightweight xorshift64 PRNG for layer assignment.
pub struct Xorshift64(pub u64);

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f64) / (u64::MAX as f64)
    }
}

/// Ordered candidate for priority queues during search and construction.
#[derive(Clone, Copy, PartialEq)]
pub struct Candidate {
    pub dist: f32,
    pub id: u32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.id.cmp(&other.id))
    }
}
