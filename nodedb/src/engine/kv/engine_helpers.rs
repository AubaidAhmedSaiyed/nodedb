// SPDX-License-Identifier: BUSL-1.1

//! Key encoding and field extraction helpers for the KV engine.

use super::sorted_index::number;

/// Compute a numeric key for per-collection HashMap lookups.
///
/// Uses FxHash on `(database_id, tenant_id, collection)` to produce a `u64` —
/// zero allocation, O(1). The `database_id` scopes the key so collections of
/// the same name in different databases never collide.
pub(super) fn table_key(database_id: u64, tenant_id: u64, collection: &str) -> u64 {
    super::hash_helpers::fxhash_multi(&[
        &database_id.to_le_bytes(),
        b":",
        &tenant_id.to_le_bytes(),
        b":",
        collection.as_bytes(),
    ])
}

/// The `{database_id}:{tenant_id}:{collection}\0` prefix shared by the expiry
/// wheel's composite keys and by collection-purge prefix matching. Centralised
/// so the encode shape stays in one place and the two callers can't drift.
/// The null byte separator is safe because collection names can't contain null.
pub(super) fn expiry_prefix(database_id: u64, tenant_id: u64, collection: &str) -> String {
    format!("{database_id}:{tenant_id}:{collection}\0")
}

/// Construct a composite key for the expiry wheel:
/// "{database_id}:{tenant_id}:{collection}\0{key_bytes}".
pub(super) fn expiry_key(
    database_id: u64,
    tenant_id: u64,
    collection: &str,
    key: &[u8],
) -> Vec<u8> {
    let mut composite = expiry_prefix(database_id, tenant_id, collection).into_bytes();
    composite.extend_from_slice(key);
    composite
}

/// Parse a composite expiry key back into (database_id, tenant_id, collection, key_bytes).
pub(super) fn parse_expiry_key(composite: &[u8]) -> Option<(u64, u64, String, Vec<u8>)> {
    let null_pos = composite.iter().position(|&b| b == 0)?;
    let prefix = std::str::from_utf8(&composite[..null_pos]).ok()?;
    let db_colon = prefix.find(':')?;
    let database_id: u64 = prefix[..db_colon].parse().ok()?;
    let rest = &prefix[db_colon + 1..];
    let tenant_colon = rest.find(':')?;
    let tenant_id: u64 = rest[..tenant_colon].parse().ok()?;
    let collection = rest[tenant_colon + 1..].to_string();
    let key = composite[null_pos + 1..].to_vec();
    Some((database_id, tenant_id, collection, key))
}

// ---------------------------------------------------------------------------
// Field extraction from MessagePack values (for secondary index maintenance)
// ---------------------------------------------------------------------------

/// Decode a MessagePack-encoded KV value. `None` when it does not decode.
pub(super) fn decode_document(data: &[u8]) -> Option<serde_json::Value> {
    nodedb_types::json_from_msgpack(data).ok()
}

/// Extract all field name → value bytes pairs from a MessagePack-encoded document.
///
/// The value is assumed to be a MessagePack map (as produced by SQL INSERT).
/// Returns `(field_name, value_bytes)` where `value_bytes` is the secondary
/// index encoding of the scalar value.
pub(super) fn extract_all_field_values_from_msgpack(data: &[u8]) -> Vec<(String, Vec<u8>)> {
    decode_document(data)
        .map(|doc| document_field_values(&doc))
        .unwrap_or_default()
}

/// Every top-level field of `document` with its secondary index encoding.
/// Empty when `document` is not an object.
pub(super) fn document_field_values(document: &serde_json::Value) -> Vec<(String, Vec<u8>)> {
    let Some(obj) = document.as_object() else {
        return Vec::new();
    };
    obj.iter()
        .map(|(k, v)| (k.clone(), json_value_to_index_bytes(v)))
        .collect()
}

/// Extract a single field's value from a MessagePack-encoded document.
pub(super) fn extract_field_values_from_msgpack(data: &[u8], field: &str) -> Vec<Vec<u8>> {
    let Ok(value) = nodedb_types::json_from_msgpack(data) else {
        return Vec::new();
    };

    let Some(obj) = value.as_object() else {
        return Vec::new();
    };

    match obj.get(field) {
        Some(v) => vec![json_value_to_index_bytes(v)],
        None => Vec::new(),
    }
}

/// Convert a JSON value to bytes suitable for B-Tree ordering.
///
/// Strings are stored as UTF-8 bytes. Numbers use the sorted index number
/// encoding, so 11 and 11.0 are one key. Booleans as 0/1. Null as empty.
fn json_value_to_index_bytes(v: &serde_json::Value) -> Vec<u8> {
    match v {
        serde_json::Value::String(s) => s.as_bytes().to_vec(),
        serde_json::Value::Number(n) => number::encode_json_number(n).to_vec(),
        serde_json::Value::Bool(b) => vec![*b as u8],
        serde_json::Value::Null => Vec::new(),
        // Arrays and objects: serialize as JSON string for comparison.
        other => other.to_string().into_bytes(),
    }
}
