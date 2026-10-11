// SPDX-License-Identifier: BUSL-1.1

//! Cell-to-tile partitioning for distributed array writes.
//!
//! Both `coord_put` and `coord_delete` receive a flat list of cells (or
//! coord tuples) paired with their pre-computed tile keys. This module
//! groups them by owning vShard, the vShard of the key's routing bucket
//! from `array_vshard_for_tile`.
//!
//! Callers pre-attach tile keys because the query planner already
//! holds the schema and can compute them at planning time; doing the same
//! computation again inside the coordinator would require an additional
//! schema lookup that `nodedb-cluster` intentionally avoids (it carries no
//! dependency on `nodedb-array`).
//!
//! ## Output shape
//!
//! `partition_put_cells` returns one `PutBucket` per vShard: its
//! `representative_tile_key` and `cells_msgpack`, a zerompk-encoded
//! `Vec<raw-cell-bytes>` (the same bytes the shard handler passes to
//! `exec_put`).
//!
//! `partition_delete_coords` returns one `DeleteBucket` per vShard: its
//! `representative_tile_key` and `coords_msgpack`, a zerompk-encoded
//! `Vec<raw-coord-bytes>`.

use std::collections::HashMap;

use crate::error::{ClusterError, Result};

use super::routing::array_vshard_for_tile;

/// One bucket produced by `partition_put_cells`.
pub struct PutBucket {
    pub vshard_id: u32,
    /// Tile key of the first cell added to this bucket. The shard handler
    /// routes it to check ownership (`validate_put_routing`).
    pub representative_tile_key: u64,
    /// zerompk-encoded `Vec<raw-cell-bytes>` for all cells in this bucket.
    pub cells_msgpack: Vec<u8>,
}

/// One bucket produced by `partition_delete_coords`.
pub struct DeleteBucket {
    pub vshard_id: u32,
    /// Tile key of the first coord added to this bucket. The shard handler
    /// routes it to check ownership (`validate_delete_routing`).
    pub representative_tile_key: u64,
    /// zerompk-encoded `Vec<raw-coord-bytes>` for all coords in this bucket.
    pub coords_msgpack: Vec<u8>,
}

/// Group `(tile_key, cell_msgpack)` pairs by owning vShard.
///
/// `cells` — each element is `(tile_key, zerompk-encoded single cell)`.
/// `prefix_bits` — routing granularity (1–16).
///
/// Returns one `PutBucket` per unique owning vShard. The order of buckets
/// in the output is unspecified but deterministic within a call.
pub fn partition_put_cells(cells: &[(u64, Vec<u8>)], prefix_bits: u8) -> Result<Vec<PutBucket>> {
    if cells.is_empty() {
        return Ok(Vec::new());
    }

    // vshard_id → (representative tile key, per-cell byte blobs)
    let mut buckets: HashMap<u32, (u64, Vec<Vec<u8>>)> = HashMap::new();

    for (tile_key, cell_bytes) in cells {
        let vshard_id = array_vshard_for_tile(*tile_key, prefix_bits)?;
        let entry = buckets.entry(vshard_id).or_insert((*tile_key, Vec::new()));
        entry.1.push(cell_bytes.clone());
    }

    buckets
        .into_iter()
        .map(|(vshard_id, (representative, cell_blobs))| {
            let cells_msgpack = encode_blob_vec(&cell_blobs)?;
            Ok(PutBucket {
                vshard_id,
                representative_tile_key: representative,
                cells_msgpack,
            })
        })
        .collect()
}

/// Group `(tile_key, coord_msgpack)` pairs by owning vShard.
///
/// `coords` — each element is `(tile_key, zerompk-encoded single coord)`.
/// `prefix_bits` — routing granularity (1–16).
///
/// Returns one `DeleteBucket` per unique owning vShard.
pub fn partition_delete_coords(
    coords: &[(u64, Vec<u8>)],
    prefix_bits: u8,
) -> Result<Vec<DeleteBucket>> {
    if coords.is_empty() {
        return Ok(Vec::new());
    }

    // vshard_id → (representative tile key, coord byte blobs)
    let mut buckets: HashMap<u32, (u64, Vec<Vec<u8>>)> = HashMap::new();

    for (tile_key, coord_bytes) in coords {
        let vshard_id = array_vshard_for_tile(*tile_key, prefix_bits)?;
        let entry = buckets.entry(vshard_id).or_insert((*tile_key, Vec::new()));
        entry.1.push(coord_bytes.clone());
    }

    buckets
        .into_iter()
        .map(|(vshard_id, (representative, coord_blobs))| {
            let coords_msgpack = encode_blob_vec(&coord_blobs)?;
            Ok(DeleteBucket {
                vshard_id,
                representative_tile_key: representative,
                coords_msgpack,
            })
        })
        .collect()
}

/// Encode a `Vec<Vec<u8>>` (collection of opaque blobs) as a zerompk array.
///
/// Each blob is emitted as raw msgpack bytes (already encoded by the caller)
/// wrapped in a msgpack bin or str value. We re-encode the entire collection
/// as a msgpack array of byte-arrays so the shard handler can deserialise it
/// as `Vec<raw-bytes>` and forward to the engine.
fn encode_blob_vec(blobs: &[Vec<u8>]) -> Result<Vec<u8>> {
    // zerompk::to_msgpack_vec requires Sized, so pass as a Vec reference.
    let owned: Vec<Vec<u8>> = blobs.to_vec();
    zerompk::to_msgpack_vec(&owned).map_err(|e| ClusterError::Codec {
        detail: format!("partition encode blob vec: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a fake (tile_key, cell_bytes) pair. The cell_bytes are
    // opaque for the partitioner — any non-empty Vec<u8> works.
    fn cell(prefix: u64, tag: u8) -> (u64, Vec<u8>) {
        (prefix, vec![tag])
    }

    // Build a fake coord pair.
    fn coord(prefix: u64, tag: u8) -> (u64, Vec<u8>) {
        (prefix, vec![tag])
    }

    // Helper: compute expected vshard for a given tile key and prefix_bits.
    fn expected_vshard(prefix: u64, prefix_bits: u8) -> u32 {
        array_vshard_for_tile(prefix, prefix_bits).unwrap()
    }

    #[test]
    fn partition_put_empty_returns_empty() {
        let buckets = partition_put_cells(&[], 8).unwrap();
        assert!(buckets.is_empty());
    }

    #[test]
    fn partition_delete_empty_returns_empty() {
        let buckets = partition_delete_coords(&[], 8).unwrap();
        assert!(buckets.is_empty());
    }

    #[test]
    fn partition_put_single_cell_one_bucket() {
        // Key 0 mixes to 0 → bucket 0 → vShard 0.
        let cells = vec![cell(0, 0xAA)];
        let buckets = partition_put_cells(&cells, 10).unwrap();
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].vshard_id, 0);
        assert_eq!(buckets[0].representative_tile_key, 0);
    }

    #[test]
    fn partition_put_cells_by_tile_three_buckets() {
        // With prefix_bits=10 (stride=1), each mixed bucket is its own
        // vShard. The three keys mix to three distinct buckets.
        let p0 = 0x0000_0000_0000_0000u64;
        let p1 = 0x0040_0000_0000_0000u64;
        let p2 = 0x0080_0000_0000_0000u64;

        let cells = vec![
            cell(p0, 0x01),
            cell(p1, 0x02),
            cell(p0, 0x03), // same bucket as first
            cell(p2, 0x04),
            cell(p1, 0x05), // same bucket as second
        ];

        let buckets = partition_put_cells(&cells, 10).unwrap();
        assert_eq!(buckets.len(), 3);

        // p0 and p1 each own 2 cells, p2 owns 1.
        for (key, want) in [(p0, 2), (p1, 2), (p2, 1)] {
            let vshard = expected_vshard(key, 10);
            let bucket = buckets
                .iter()
                .find(|b| b.vshard_id == vshard)
                .expect("bucket for key");
            let blobs: Vec<Vec<u8>> = zerompk::from_msgpack(&bucket.cells_msgpack).unwrap();
            assert_eq!(blobs.len(), want, "key={key:#x}");
        }
    }

    #[test]
    fn partition_delete_coords_by_tile() {
        let p0 = 0x0000_0000_0000_0000u64;
        let p1 = 0x0040_0000_0000_0000u64;

        let coords = vec![coord(p0, 0xAA), coord(p1, 0xBB), coord(p0, 0xCC)];

        let buckets = partition_delete_coords(&coords, 10).unwrap();
        assert_eq!(buckets.len(), 2);
        for (key, want) in [(p0, 2), (p1, 1)] {
            let vshard = expected_vshard(key, 10);
            let bucket = buckets
                .iter()
                .find(|b| b.vshard_id == vshard)
                .expect("bucket for key");
            let blobs: Vec<Vec<u8>> = zerompk::from_msgpack(&bucket.coords_msgpack).unwrap();
            assert_eq!(blobs.len(), want, "key={key:#x}");
        }
    }

    #[test]
    fn partition_put_invalid_prefix_bits_errors() {
        let cells = vec![cell(0, 0x01)];
        assert!(partition_put_cells(&cells, 0).is_err());
        assert!(partition_put_cells(&cells, 17).is_err());
    }
}
