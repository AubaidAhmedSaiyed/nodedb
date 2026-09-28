// SPDX-License-Identifier: Apache-2.0

//! Document navigation functions: `doc_get`, `doc_exists`,
//! `doc_array_contains`, and `nav`.
//!
//! Each takes a document value and a path. The path is JSONPath (`$.a.b`,
//! `$.arr[0]`). A path without the leading `$` is read as `$.` plus the
//! path, so `'user.name'` and `'$.user.name'` name the same field. A document
//! held as JSON text is parsed before the walk.
//!
//! A malformed path, or a path argument that is not text, fails the
//! statement with a typed [`EvalError`]. A `NULL` path gives `NULL`. A path
//! that resolves to nothing gives the default, `NULL`, or `false`.

use nodedb_types::Value;

use super::path::{PathStep, parse_jsonpath, walk_path};
use super::pg_ops::coerce_json_string;
use crate::expr::EvalError;
use crate::value_ops::coerced_eq;

/// Parse the path argument at 1-based `position`. `Ok(None)` for a `NULL`
/// path.
fn path_steps(
    function: &'static str,
    position: usize,
    path: &Value,
) -> Result<Option<Vec<PathStep>>, EvalError> {
    let text = match path {
        Value::Null => return Ok(None),
        Value::String(text) => text,
        other => {
            return Err(EvalError::ArgumentType {
                function,
                position,
                expected: "a JSONPath text",
                got: other.type_name(),
            });
        }
    };
    let parsed = if text.starts_with('$') {
        parse_jsonpath(text)
    } else {
        parse_jsonpath(&format!("$.{text}"))
    };
    parsed.map(Some).map_err(|e| EvalError::InvalidJsonPath {
        function,
        path: text.clone(),
        reason: e.to_string(),
    })
}

/// The non-null value at the path in `args[1]` inside the document in
/// `args[0]`, cloned. `Ok(None)` when the path is `NULL` or resolves to
/// nothing.
fn resolve(function: &'static str, args: &[Value]) -> Result<Option<Value>, EvalError> {
    let doc = args.first().unwrap_or(&Value::Null);
    let path = args.get(1).unwrap_or(&Value::Null);
    let Some(steps) = path_steps(function, 2, path)? else {
        return Ok(None);
    };
    let doc = coerce_json_string(doc);
    Ok(match walk_path(&doc, &steps) {
        Some(Value::Null) | None => None,
        Some(found) => Some(found.clone()),
    })
}

/// `doc_get(doc, path [, default])`: the value at `path`, or `default` when
/// the path is missing or null. `default` is `NULL` when omitted.
pub(super) fn doc_get(args: &[Value]) -> Result<Value, EvalError> {
    Ok(resolve("doc_get", args)?.unwrap_or_else(|| args.get(2).cloned().unwrap_or(Value::Null)))
}

/// `nav(doc, path)`: the value at `path`, or `NULL`.
pub(super) fn nav(args: &[Value]) -> Result<Value, EvalError> {
    Ok(resolve("nav", args)?.unwrap_or(Value::Null))
}

/// `doc_exists(doc, path)`: whether `path` holds a non-null value.
pub(super) fn doc_exists(args: &[Value]) -> Result<Value, EvalError> {
    Ok(Value::Bool(resolve("doc_exists", args)?.is_some()))
}

/// `doc_array_contains(doc, path, value)`: whether the array at `path` holds
/// an element equal to `value`. Equality coerces a numeric string to a
/// number and an ISO-8601 string to an instant. A missing path or a
/// non-array value at the path gives `false`.
pub(super) fn doc_array_contains(args: &[Value]) -> Result<Value, EvalError> {
    let needle = args.get(2).unwrap_or(&Value::Null);
    let contains = match resolve("doc_array_contains", args)? {
        Some(Value::Array(items) | Value::Set(items)) => {
            items.iter().any(|item| coerced_eq(item, needle))
        }
        Some(_) | None => false,
    };
    Ok(Value::Bool(contains))
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

    fn text(s: &str) -> Value {
        Value::String(s.into())
    }

    fn event() -> Value {
        obj(&[
            (
                "user",
                obj(&[("name", text("ada")), ("email", Value::Null)]),
            ),
            (
                "tags",
                Value::Array(vec![text("important"), text("ops"), Value::Integer(7)]),
            ),
        ])
    }

    #[test]
    fn doc_get_reads_a_nested_field() {
        assert_eq!(doc_get(&[event(), text("$.user.name")]), Ok(text("ada")));
    }

    #[test]
    fn doc_get_reads_a_path_without_the_dollar() {
        assert_eq!(doc_get(&[event(), text("user.name")]), Ok(text("ada")));
    }

    #[test]
    fn doc_get_reads_an_array_element() {
        assert_eq!(doc_get(&[event(), text("$.tags[1]")]), Ok(text("ops")));
    }

    #[test]
    fn doc_get_missing_path_gives_the_default() {
        assert_eq!(
            doc_get(&[event(), text("$.user.age"), Value::Integer(0)]),
            Ok(Value::Integer(0))
        );
        assert_eq!(doc_get(&[event(), text("$.user.age")]), Ok(Value::Null));
    }

    #[test]
    fn doc_get_null_field_gives_the_default() {
        assert_eq!(
            doc_get(&[event(), text("$.user.email"), text("none")]),
            Ok(text("none"))
        );
    }

    #[test]
    fn doc_get_parses_a_json_text_document() {
        let doc = text(r#"{"user":{"name":"ada"}}"#);
        assert_eq!(doc_get(&[doc, text("$.user.name")]), Ok(text("ada")));
    }

    #[test]
    fn doc_get_malformed_path_is_an_error() {
        let err = doc_get(&[event(), text("$..name"), text("d")]).unwrap_err();
        assert!(
            matches!(
                &err,
                EvalError::InvalidJsonPath { function: "doc_get", path, .. } if path == "$..name"
            ),
            "{err:?}"
        );
        let err = doc_get(&[event(), text("$.tags[x]")]).unwrap_err();
        assert!(matches!(err, EvalError::InvalidJsonPath { .. }), "{err:?}");
    }

    #[test]
    fn doc_array_contains_malformed_path_is_an_error() {
        let err = doc_array_contains(&[event(), text("$.tags["), text("ops")]).unwrap_err();
        assert!(
            matches!(
                err,
                EvalError::InvalidJsonPath {
                    function: "doc_array_contains",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_non_text_path_is_an_argument_error() {
        let err = doc_exists(&[event(), Value::Integer(3)]).unwrap_err();
        assert_eq!(
            err,
            EvalError::ArgumentType {
                function: "doc_exists",
                position: 2,
                expected: "a JSONPath text",
                got: "int",
            }
        );
    }

    #[test]
    fn a_null_path_resolves_to_nothing() {
        assert_eq!(doc_get(&[event(), Value::Null, text("d")]), Ok(text("d")));
        assert_eq!(doc_exists(&[event(), Value::Null]), Ok(Value::Bool(false)));
    }

    #[test]
    fn nav_matches_doc_get_without_a_default() {
        assert_eq!(nav(&[event(), text("$.user.name")]), Ok(text("ada")));
        assert_eq!(nav(&[event(), text("$.missing")]), Ok(Value::Null));
    }

    #[test]
    fn doc_exists_is_true_only_for_a_non_null_value() {
        assert_eq!(
            doc_exists(&[event(), text("$.user.name")]),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            doc_exists(&[event(), text("$.user.email")]),
            Ok(Value::Bool(false))
        );
        assert_eq!(
            doc_exists(&[event(), text("$.nope")]),
            Ok(Value::Bool(false))
        );
        assert_eq!(
            doc_exists(&[Value::Null, text("$.a")]),
            Ok(Value::Bool(false))
        );
    }

    #[test]
    fn doc_array_contains_finds_an_element() {
        assert_eq!(
            doc_array_contains(&[event(), text("$.tags"), text("important")]),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            doc_array_contains(&[event(), text("$.tags"), text("absent")]),
            Ok(Value::Bool(false))
        );
    }

    #[test]
    fn doc_array_contains_coerces_a_numeric_string() {
        assert_eq!(
            doc_array_contains(&[event(), text("$.tags"), text("7")]),
            Ok(Value::Bool(true))
        );
    }

    #[test]
    fn doc_array_contains_on_a_non_array_is_false() {
        assert_eq!(
            doc_array_contains(&[event(), text("$.user.name"), text("ada")]),
            Ok(Value::Bool(false))
        );
    }
}
