// SPDX-License-Identifier: BUSL-1.1

//! Statement and argument parsing for the sorted-index family.

use nodedb_types::Value;

use super::super::super::result::DdlError;
use crate::engine::kv::sorted_index::value::encode_json;

pub(super) fn ddl_err(sqlstate: &str, message: impl Into<String>) -> DdlError {
    DdlError::new(sqlstate, message)
}

pub(super) fn parse_sort_columns(cols_str: &str) -> Result<Vec<(String, String)>, DdlError> {
    let mut columns = Vec::new();
    for part in cols_str.split(',') {
        let tokens: Vec<&str> = part.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        let name = crate::control::server::shared::ddl::sql_parse::parse_ident_token(tokens[0])?;
        let dir = if tokens.len() > 1 {
            tokens[1].to_uppercase()
        } else {
            "ASC".into()
        };
        if dir != "ASC" && dir != "DESC" {
            return Err(ddl_err(
                "42601",
                format!("invalid sort direction '{dir}', expected ASC or DESC"),
            ));
        }
        columns.push((name, dir));
    }
    if columns.is_empty() {
        return Err(ddl_err("42601", "at least one sort column required"));
    }
    Ok(columns)
}

pub(super) fn parse_function_args(sql: &str) -> Result<Vec<String>, DdlError> {
    let start = sql
        .find('(')
        .ok_or_else(|| ddl_err("42601", "expected '(' in function call"))?;
    let end = sql
        .rfind(')')
        .ok_or_else(|| ddl_err("42601", "expected ')' in function call"))?;
    if start >= end {
        return Ok(Vec::new());
    }
    let inner = &sql[start + 1..end];
    Ok(super::super::kv_atomic::split_args(inner))
}

pub(super) fn unquote(s: &str) -> String {
    let t = s.trim();
    if t.starts_with('\'') && t.ends_with('\'') && t.len() >= 2 {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

/// The sorted index value encoding of a `RANGE` score bound. `None` for a
/// null bound, which leaves that end of the range open.
///
/// SQL `RANGE(...)` and the native `KvSortedIndexRange` both encode here. The
/// bound converts to JSON the way a stored row does, then encodes as a stored
/// field value, so a bound and a row of equal value get equal bytes.
pub(crate) fn encode_score_bound(bound: Value) -> Option<Vec<u8>> {
    match serde_json::Value::from(bound) {
        serde_json::Value::Null => None,
        json => Some(encode_json(&json)),
    }
}

/// A SQL `RANGE(...)` score argument, as the value encoding of the index's
/// leading sort column.
///
/// An integer literal is a number, past i64 too. A decimal literal is a
/// float. `NULL`, `NONE` and `*` leave the end open. Anything else is a
/// string. A quoted literal unquotes first, so a bound parameter `'100'` is
/// the number 100.
pub(super) fn parse_score_arg(s: &str) -> Option<Vec<u8>> {
    let t = unquote(s);
    if t.eq_ignore_ascii_case("NULL") || t.eq_ignore_ascii_case("NONE") || t == "*" {
        return None;
    }
    let bound = if let Ok(v) = t.parse::<i64>() {
        Value::Integer(v)
    } else if let Ok(v) = t.parse::<u64>() {
        Value::Decimal(rust_decimal::Decimal::from(v))
    } else if let Ok(v) = t.parse::<f64>() {
        Value::Float(v)
    } else {
        Value::String(t)
    };
    encode_score_bound(bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::kv::sorted_index::value::{encode_float, encode_int, encode_text};

    #[test]
    fn integers_past_i64_stay_exact() {
        let past = u64::MAX;
        assert_eq!(
            parse_score_arg(&past.to_string()),
            Some(encode_json(&serde_json::json!(past)))
        );
    }

    #[test]
    fn score_args_encode_like_stored_values() {
        assert_eq!(parse_score_arg("11"), Some(encode_int(11)));
        assert_eq!(parse_score_arg("'11'"), Some(encode_int(11)));
        assert_eq!(parse_score_arg("10.5"), Some(encode_float(10.5)));
        assert_eq!(parse_score_arg("11.0"), Some(encode_int(11)));
        assert_eq!(parse_score_arg("'gold'"), Some(encode_text("gold")));
        assert_eq!(parse_score_arg("NULL"), None);
        assert_eq!(parse_score_arg("*"), None);
    }
}
