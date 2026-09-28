// SPDX-License-Identifier: Apache-2.0

//! Array scalar functions.

use nodedb_types::Value;

pub(super) fn try_eval(name: &str, args: &[Value]) -> Option<Value> {
    let v = match name {
        "array_length" | "cardinality" => match args.first() {
            Some(Value::Array(arr)) => Value::Integer(arr.len() as i64),
            _ => Value::Null,
        },
        "array_append" => {
            let mut arr = match args.first() {
                Some(Value::Array(a)) => a.clone(),
                _ => return Some(Value::Null),
            };
            if let Some(val) = args.get(1) {
                arr.push(val.clone());
            }
            Value::Array(arr)
        }
        "array_prepend" => {
            let val = args.first().cloned().unwrap_or(Value::Null);
            let mut arr = match args.get(1) {
                Some(Value::Array(a)) => a.clone(),
                _ => return Some(Value::Null),
            };
            arr.insert(0, val);
            Value::Array(arr)
        }
        "array_remove" => {
            let arr = match args.first() {
                Some(Value::Array(a)) => a,
                _ => return Some(Value::Null),
            };
            let needle = args.get(1).unwrap_or(&Value::Null);
            Value::Array(arr.iter().filter(|v| *v != needle).cloned().collect())
        }
        "array_concat" | "array_cat" => {
            let mut result = match args.first() {
                Some(Value::Array(a)) => a.clone(),
                _ => return Some(Value::Null),
            };
            if let Some(Value::Array(b)) = args.get(1) {
                result.extend(b.iter().cloned());
            }
            Value::Array(result)
        }
        "array_distinct" => {
            let arr = match args.first() {
                Some(Value::Array(a)) => a,
                _ => return Some(Value::Null),
            };
            let mut unique = Vec::new();
            for v in arr {
                if !unique.contains(v) {
                    unique.push(v.clone());
                }
            }
            Value::Array(unique)
        }
        "array_contains" => {
            let arr = match args.first() {
                Some(Value::Array(a)) => a,
                _ => return Some(Value::Bool(false)),
            };
            let needle = args.get(1).unwrap_or(&Value::Null);
            Value::Bool(arr.contains(needle))
        }
        "array_position" => {
            let arr = match args.first() {
                Some(Value::Array(a)) => a,
                _ => return Some(Value::Null),
            };
            let needle = args.get(1).unwrap_or(&Value::Null);
            match arr.iter().position(|v| v == needle) {
                Some(pos) => Value::Integer((pos + 1) as i64),
                None => Value::Null,
            }
        }
        "array_reverse" => {
            let arr = match args.first() {
                Some(Value::Array(a)) => a,
                _ => return Some(Value::Null),
            };
            let mut reversed = arr.clone();
            reversed.reverse();
            Value::Array(reversed)
        }
        // `ARRAY[a, b, ...]` with a non-literal element lowers to this call,
        // so the array is built per row from the evaluated elements. A NULL
        // element stays a NULL element.
        "make_array" => Value::Array(args.to_vec()),
        _ => return None,
    };
    Some(v)
}
