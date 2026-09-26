// SPDX-License-Identifier: Apache-2.0

//! Implementation of the 9 PostgreSQL JSON operators, lowered to function calls.
//!
//! `->` / `->>` / `#>` / `#>>` / `@>` / `<@` / `?` / `?&` / `?|`
//!
//! Document columns in the schemaless engine are stored as string literals
//! (the raw JSON text) rather than pre-parsed objects. Every operator that
//! accepts a JSON target therefore calls `coerce_json_string` first so that
//! a `Value::String(s)` that contains valid JSON text is transparently parsed
//! and the operator proceeds against the resulting structured value.

use nodedb_types::Value;

/// If `v` is a `Value::String` containing valid JSON, parse and return the
/// structured `Value`. Otherwise return a clone of `v` unchanged.
///
/// This is a cheap path: non-string values pass through a single `matches!`
/// check; strings that are not valid JSON also return quickly from the parser.
pub(super) fn coerce_json_string(v: &Value) -> std::borrow::Cow<'_, Value> {
    if let Value::String(s) = v
        && let Ok(parsed) = sonic_rs::from_str::<serde_json::Value>(s)
    {
        let ndb = nodedb_types::conversion::json_to_value(parsed);
        return std::borrow::Cow::Owned(ndb);
    }
    std::borrow::Cow::Borrowed(v)
}

/// `->`: get JSON field by key (object) or by integer index (array). Returns Json.
pub(super) fn pg_json_get(target: &Value, key: &Value) -> Value {
    let target = coerce_json_string(target);
    let target = target.as_ref();
    match (target, key) {
        (Value::Object(map), Value::String(k)) => {
            map.get(k.as_str()).cloned().unwrap_or(Value::Null)
        }
        (Value::Array(arr), Value::Integer(i)) => {
            let idx = if *i < 0 {
                arr.len().checked_sub(i.unsigned_abs() as usize)
            } else {
                Some(*i as usize)
            };
            idx.and_then(|i| arr.get(i)).cloned().unwrap_or(Value::Null)
        }
        (Value::Array(arr), Value::String(s)) => {
            // PG allows string index into arrays if it parses as integer.
            s.parse::<usize>()
                .ok()
                .and_then(|i| arr.get(i))
                .cloned()
                .unwrap_or(Value::Null)
        }
        _ => Value::Null,
    }
}

/// `->>`': like `->` but returns the result as a String (text). Returns String.
pub(super) fn pg_json_get_text(target: &Value, key: &Value) -> Value {
    let val = pg_json_get(target, key);
    value_to_text(val)
}

/// `#>`: get JSON sub-object at path `'{a,b,c}'`. Returns Json.
pub(super) fn pg_json_path_get(target: &Value, path_lit: &Value) -> Value {
    let coerced = coerce_json_string(target);
    let target = coerced.as_ref();
    let keys = match parse_pg_path_literal(path_lit) {
        Some(k) => k,
        None => return Value::Null,
    };
    let mut current = target;
    for key in &keys {
        current = match current {
            Value::Object(map) => match map.get(key.as_str()) {
                Some(v) => v,
                None => return Value::Null,
            },
            Value::Array(arr) => match key.parse::<usize>().ok().and_then(|i| arr.get(i)) {
                Some(v) => v,
                None => return Value::Null,
            },
            _ => return Value::Null,
        };
    }
    current.clone()
}

/// `#>>`: like `#>` but returns result as String. Returns String.
pub(super) fn pg_json_path_get_text(target: &Value, path_lit: &Value) -> Value {
    let val = pg_json_path_get(target, path_lit);
    value_to_text(val)
}

/// `@>`: true iff every (key,value) pair in `b` is present in `a` (recursive).
pub(super) fn pg_json_contains(a: &Value, b: &Value) -> Value {
    let ca = coerce_json_string(a);
    let cb = coerce_json_string(b);
    Value::Bool(json_contains(ca.as_ref(), cb.as_ref()))
}

/// `<@`: reverse of `@>` — true iff `a` is contained by `b`.
pub(super) fn pg_json_contained_by(a: &Value, b: &Value) -> Value {
    let ca = coerce_json_string(a);
    let cb = coerce_json_string(b);
    Value::Bool(json_contains(cb.as_ref(), ca.as_ref()))
}

/// `?`: true iff `key` exists as a top-level key in object, or as element in array.
pub(super) fn pg_json_has_key(target: &Value, key: &Value) -> Value {
    let k = match key.as_str() {
        Some(s) => s,
        None => return Value::Null,
    };
    let coerced = coerce_json_string(target);
    let target = coerced.as_ref();
    let result = match target {
        Value::Object(map) => map.contains_key(k),
        Value::Array(arr) => arr.iter().any(|v| v.as_str() == Some(k)),
        _ => false,
    };
    Value::Bool(result)
}

/// `?&`: true iff ALL keys in `keys` array exist in target.
pub(super) fn pg_json_has_all_keys(target: &Value, keys: &Value) -> Value {
    let key_list = match collect_string_array(keys) {
        Some(k) => k,
        None => return Value::Null,
    };
    let coerced = coerce_json_string(target);
    let target = coerced.as_ref();
    let result = key_list.iter().all(|k| has_key(target, k));
    Value::Bool(result)
}

/// `?|`: true iff ANY key in `keys` array exists in target.
pub(super) fn pg_json_has_any_key(target: &Value, keys: &Value) -> Value {
    let key_list = match collect_string_array(keys) {
        Some(k) => k,
        None => return Value::Null,
    };
    let coerced = coerce_json_string(target);
    let target = coerced.as_ref();
    let result = key_list.iter().any(|k| has_key(target, k));
    Value::Bool(result)
}

// ── helpers ────────────────────────────────────────────────────────────────

/// Parse PG path literal `'{a,b,c}'` into a vec of keys.
/// Returns `None` for malformed input (missing braces, etc.).
fn parse_pg_path_literal(v: &Value) -> Option<Vec<String>> {
    let s = v.as_str()?;
    let s = s.trim();
    if !s.starts_with('{') || !s.ends_with('}') {
        return None;
    }
    let inner = &s[1..s.len() - 1];
    if inner.is_empty() {
        return Some(vec![]);
    }
    Some(inner.split(',').map(|k| k.trim().to_string()).collect())
}

pub(super) fn value_to_text(val: Value) -> Value {
    match val {
        Value::Null => Value::Null,
        Value::String(s) => Value::String(s),
        Value::Bool(b) => Value::String(if b { "true".into() } else { "false".into() }),
        Value::Integer(n) => Value::String(n.to_string()),
        Value::Float(f) => Value::String(f.to_string()),
        other => {
            // For nested objects/arrays: serialize as JSON-like text.
            Value::String(format!("{other:?}"))
        }
    }
}

fn has_key(target: &Value, k: &str) -> bool {
    match target {
        Value::Object(map) => map.contains_key(k),
        Value::Array(arr) => arr.iter().any(|v| v.as_str() == Some(k)),
        _ => false,
    }
}

fn collect_string_array(v: &Value) -> Option<Vec<String>> {
    match v {
        Value::Array(arr) => {
            let strings: Option<Vec<String>> = arr
                .iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            strings
        }
        Value::String(s) => Some(vec![s.clone()]),
        _ => None,
    }
}

/// Recursive containment check for `@>`.
///
/// For objects: every (key, value) in `b` must exist in `a` recursively.
/// For arrays: every element of `b` must appear as an element of `a`
///   (equality check, not recursive containment for non-object elements).
/// For scalars: `a == b`.
fn json_contains(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(a_map), Value::Object(b_map)) => b_map.iter().all(|(k, bv)| {
            a_map
                .get(k.as_str())
                .is_some_and(|av| json_contains(av, bv))
        }),
        (Value::Array(a_arr), Value::Array(b_arr)) => b_arr.iter().all(|bv| {
            a_arr.iter().any(|av| match (av, bv) {
                (Value::Object(_), Value::Object(_)) => json_contains(av, bv),
                _ => av == bv,
            })
        }),
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn obj(pairs: &[(&str, Value)]) -> Value {
        Value::Object(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect::<HashMap<_, _>>(),
        )
    }

    fn arr(items: Vec<Value>) -> Value {
        Value::Array(items)
    }

    // ── pg_json_get ──────────────────────────────────────────────────────

    #[test]
    fn get_object_field() {
        let o = obj(&[("key", Value::Integer(42))]);
        assert_eq!(
            pg_json_get(&o, &Value::String("key".into())),
            Value::Integer(42)
        );
    }

    #[test]
    fn get_missing_field_is_null() {
        let o = obj(&[]);
        assert_eq!(
            pg_json_get(&o, &Value::String("missing".into())),
            Value::Null
        );
    }

    #[test]
    fn get_array_by_index() {
        let a = arr(vec![Value::Integer(10), Value::Integer(20)]);
        assert_eq!(pg_json_get(&a, &Value::Integer(1)), Value::Integer(20));
    }

    #[test]
    fn get_array_out_of_bounds_is_null() {
        let a = arr(vec![Value::Integer(10)]);
        assert_eq!(pg_json_get(&a, &Value::Integer(5)), Value::Null);
    }

    // ── pg_json_get_text ─────────────────────────────────────────────────

    #[test]
    fn get_text_returns_string() {
        let o = obj(&[("k", Value::Integer(7))]);
        assert_eq!(
            pg_json_get_text(&o, &Value::String("k".into())),
            Value::String("7".into())
        );
    }

    #[test]
    fn get_text_missing_is_null() {
        let o = obj(&[]);
        assert_eq!(
            pg_json_get_text(&o, &Value::String("no".into())),
            Value::Null
        );
    }

    // ── pg_json_path_get ─────────────────────────────────────────────────

    #[test]
    fn path_get_nested() {
        let inner = obj(&[("c", Value::String("deep".into()))]);
        let mid = obj(&[("b", inner)]);
        let root = obj(&[("a", mid)]);
        let path = Value::String("{a,b,c}".into());
        assert_eq!(pg_json_path_get(&root, &path), Value::String("deep".into()));
    }

    #[test]
    fn path_get_missing_returns_null() {
        let root = obj(&[("a", Value::Integer(1))]);
        let path = Value::String("{a,b}".into());
        assert_eq!(pg_json_path_get(&root, &path), Value::Null);
    }

    #[test]
    fn path_get_malformed_path_returns_null() {
        let root = obj(&[]);
        // Missing closing brace.
        assert_eq!(
            pg_json_path_get(&root, &Value::String("{a,b".into())),
            Value::Null
        );
    }

    // ── pg_json_path_get_text ────────────────────────────────────────────

    #[test]
    fn path_get_text_returns_string() {
        let inner = obj(&[("v", Value::Integer(99))]);
        let root = obj(&[("x", inner)]);
        assert_eq!(
            pg_json_path_get_text(&root, &Value::String("{x,v}".into())),
            Value::String("99".into())
        );
    }

    // ── pg_json_contains ─────────────────────────────────────────────────

    #[test]
    fn contains_exact_match() {
        let a = obj(&[("a", Value::Integer(1))]);
        let b = obj(&[("a", Value::Integer(1))]);
        assert_eq!(pg_json_contains(&a, &b), Value::Bool(true));
    }

    #[test]
    fn contains_superset() {
        let a = obj(&[("a", Value::Integer(1)), ("b", Value::Integer(2))]);
        let b = obj(&[("a", Value::Integer(1))]);
        assert_eq!(pg_json_contains(&a, &b), Value::Bool(true));
    }

    #[test]
    fn contains_missing_key_false() {
        let a = obj(&[("a", Value::Integer(1))]);
        let b = obj(&[("a", Value::Integer(1)), ("b", Value::Integer(2))]);
        assert_eq!(pg_json_contains(&a, &b), Value::Bool(false));
    }

    #[test]
    fn contains_nested_object() {
        let inner_a = obj(&[("x", Value::Integer(1)), ("y", Value::Integer(2))]);
        let inner_b = obj(&[("x", Value::Integer(1))]);
        let a = obj(&[("n", inner_a)]);
        let b = obj(&[("n", inner_b)]);
        assert_eq!(pg_json_contains(&a, &b), Value::Bool(true));
    }

    #[test]
    fn contains_array_subset() {
        let a = arr(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        let b = arr(vec![Value::Integer(1), Value::Integer(2)]);
        assert_eq!(pg_json_contains(&a, &b), Value::Bool(true));
    }

    #[test]
    fn contains_array_missing_elem() {
        let a = arr(vec![Value::Integer(1), Value::Integer(2)]);
        let b = arr(vec![Value::Integer(3)]);
        assert_eq!(pg_json_contains(&a, &b), Value::Bool(false));
    }

    // ── pg_json_contained_by ─────────────────────────────────────────────

    #[test]
    fn contained_by_reverse() {
        let small = obj(&[("a", Value::Integer(1))]);
        let big = obj(&[("a", Value::Integer(1)), ("b", Value::Integer(2))]);
        assert_eq!(pg_json_contained_by(&small, &big), Value::Bool(true));
        assert_eq!(pg_json_contained_by(&big, &small), Value::Bool(false));
    }

    // ── pg_json_has_key ───────────────────────────────────────────────────

    #[test]
    fn has_key_present() {
        let o = obj(&[("k", Value::Null)]);
        assert_eq!(
            pg_json_has_key(&o, &Value::String("k".into())),
            Value::Bool(true)
        );
    }

    #[test]
    fn has_key_missing() {
        let o = obj(&[]);
        assert_eq!(
            pg_json_has_key(&o, &Value::String("k".into())),
            Value::Bool(false)
        );
    }

    #[test]
    fn has_key_array_element() {
        let a = arr(vec![Value::String("x".into()), Value::String("y".into())]);
        assert_eq!(
            pg_json_has_key(&a, &Value::String("x".into())),
            Value::Bool(true)
        );
        assert_eq!(
            pg_json_has_key(&a, &Value::String("z".into())),
            Value::Bool(false)
        );
    }

    // ── pg_json_has_all_keys ──────────────────────────────────────────────

    #[test]
    fn has_all_keys_success() {
        let o = obj(&[("a", Value::Null), ("b", Value::Null), ("c", Value::Null)]);
        let keys = arr(vec![Value::String("a".into()), Value::String("b".into())]);
        assert_eq!(pg_json_has_all_keys(&o, &keys), Value::Bool(true));
    }

    #[test]
    fn has_all_keys_partial_fail() {
        let o = obj(&[("a", Value::Null)]);
        let keys = arr(vec![Value::String("a".into()), Value::String("b".into())]);
        assert_eq!(pg_json_has_all_keys(&o, &keys), Value::Bool(false));
    }

    // ── pg_json_has_any_key ───────────────────────────────────────────────

    #[test]
    fn has_any_key_one_present() {
        let o = obj(&[("a", Value::Null)]);
        let keys = arr(vec![Value::String("a".into()), Value::String("b".into())]);
        assert_eq!(pg_json_has_any_key(&o, &keys), Value::Bool(true));
    }

    #[test]
    fn has_any_key_none_present() {
        let o = obj(&[("c", Value::Null)]);
        let keys = arr(vec![Value::String("a".into()), Value::String("b".into())]);
        assert_eq!(pg_json_has_any_key(&o, &keys), Value::Bool(false));
    }
}
