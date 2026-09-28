// SPDX-License-Identifier: Apache-2.0

//! `ndb_chunk_text(text, chunk_size [, overlap])` as a per-row scalar.
//!
//! The scalar form returns the chunk texts as an array, split on character
//! boundaries, the default strategy of the `SELECT * FROM NDB_CHUNK_TEXT(...)`
//! table function. `overlap` is 0 when omitted. A `NULL` or non-text `text`,
//! a `chunk_size` of 0 or less, or an `overlap` not below `chunk_size` gives
//! `NULL`.

use nodedb_types::Value;

use crate::chunk_text::{ChunkStrategy, chunk_text};

pub(super) fn try_eval(name: &str, args: &[Value]) -> Option<Value> {
    if name != "ndb_chunk_text" {
        return None;
    }
    Some(eval_chunk_text(args).unwrap_or(Value::Null))
}

fn eval_chunk_text(args: &[Value]) -> Option<Value> {
    let text = args.first()?.as_str()?;
    let chunk_size = count_arg(args.get(1)?)?;
    let overlap = match args.get(2) {
        Some(value) => count_arg(value)?,
        None => 0,
    };
    let chunks = chunk_text(text, chunk_size, overlap, ChunkStrategy::Character).ok()?;
    Some(Value::Array(
        chunks
            .into_iter()
            .map(|chunk| Value::String(chunk.text))
            .collect(),
    ))
}

fn count_arg(value: &Value) -> Option<usize> {
    match value {
        Value::Integer(n) => usize::try_from(*n).ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> Value {
        Value::String(s.into())
    }

    #[test]
    fn splits_into_character_chunks() {
        let out = try_eval("ndb_chunk_text", &[text("abcdef"), Value::Integer(4)]);
        assert_eq!(out, Some(Value::Array(vec![text("abcd"), text("ef")])));
    }

    #[test]
    fn overlap_repeats_the_tail() {
        let out = try_eval(
            "ndb_chunk_text",
            &[text("abcdef"), Value::Integer(4), Value::Integer(2)],
        );
        assert_eq!(out, Some(Value::Array(vec![text("abcd"), text("cdef")])));
    }

    #[test]
    fn invalid_sizes_are_null() {
        assert_eq!(
            try_eval("ndb_chunk_text", &[text("abc"), Value::Integer(0)]),
            Some(Value::Null)
        );
        assert_eq!(
            try_eval(
                "ndb_chunk_text",
                &[text("abc"), Value::Integer(2), Value::Integer(2)]
            ),
            Some(Value::Null)
        );
    }

    #[test]
    fn null_text_is_null() {
        assert_eq!(
            try_eval("ndb_chunk_text", &[Value::Null, Value::Integer(2)]),
            Some(Value::Null)
        );
    }
}
