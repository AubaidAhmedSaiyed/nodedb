// SPDX-License-Identifier: Apache-2.0

//! Per-row vector distance functions.
//!
//! `vector_distance` (the `<->` operator), `vector_cosine_distance` (`<=>`)
//! and `vector_neg_inner_product` (`<#>`) give the same numbers a vector
//! search reports for the metric each name selects:
//! - `vector_distance`: squared Euclidean (L2) distance;
//! - `vector_cosine_distance`: `1 - cosine similarity`;
//! - `vector_neg_inner_product`: the negated dot product.
//!
//! A vector search plan serves these calls in `ORDER BY`. Every other
//! position evaluates them here, once per row. A `NULL` operand gives `NULL`.
//! Operands of different dimensions, or an operand that is not a numeric
//! vector, fail the statement with a typed [`EvalError`].

use nodedb_types::Value;
use nodedb_types::vector_distance::{cosine_distance, l2_squared, neg_inner_product};

use crate::expr::EvalError;

/// The expected-type text an argument error names.
const NUMERIC_VECTOR: &str = "a numeric vector";

/// A distance between two vectors of equal dimension.
type Metric = fn(&[f32], &[f32]) -> f32;

pub(super) fn try_eval(name: &str, args: &[Value]) -> Option<Result<Value, EvalError>> {
    let (function, metric): (&'static str, Metric) = match name {
        "vector_distance" => ("vector_distance", l2_squared),
        "vector_cosine_distance" => ("vector_cosine_distance", cosine_distance),
        "vector_neg_inner_product" => ("vector_neg_inner_product", neg_inner_product),
        _ => return None,
    };
    Some(eval_distance(function, metric, args))
}

fn eval_distance(
    function: &'static str,
    metric: Metric,
    args: &[Value],
) -> Result<Value, EvalError> {
    let left = args.first().unwrap_or(&Value::Null);
    let right = args.get(1).unwrap_or(&Value::Null);
    if matches!(left, Value::Null) || matches!(right, Value::Null) {
        return Ok(Value::Null);
    }
    let a = as_vector(function, 1, left)?;
    let b = as_vector(function, 2, right)?;
    if a.len() != b.len() {
        return Err(EvalError::VectorDimensionMismatch {
            function,
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(Value::Float(f64::from(metric(&a, &b))))
}

/// Read a vector argument: a vector value, an array of numbers, or JSON text
/// holding an array of numbers. `position` is 1-based.
fn as_vector(
    function: &'static str,
    position: usize,
    value: &Value,
) -> Result<Vec<f32>, EvalError> {
    let wrong_type = |got: &'static str| EvalError::ArgumentType {
        function,
        position,
        expected: NUMERIC_VECTOR,
        got,
    };
    match value {
        Value::Vector(floats) => Ok(floats.to_vec()),
        Value::Array(items) => items
            .iter()
            .map(|item| number(item).ok_or_else(|| wrong_type(item.type_name())))
            .collect(),
        Value::String(text) => sonic_rs::from_str::<Vec<f64>>(text)
            .map(|parsed| parsed.into_iter().map(|x| x as f32).collect())
            .map_err(|_| wrong_type(value.type_name())),
        other => Err(wrong_type(other.type_name())),
    }
}

fn number(value: &Value) -> Option<f32> {
    match value {
        Value::Float(f) => Some(*f as f32),
        Value::Integer(i) => Some(*i as f32),
        Value::Decimal(d) => {
            use rust_decimal::prelude::ToPrimitive;
            d.to_f32()
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(items: &[f64]) -> Value {
        Value::Array(items.iter().map(|x| Value::Float(*x)).collect())
    }

    fn eval(name: &str, a: Value, b: Value) -> Result<Value, EvalError> {
        try_eval(name, &[a, b]).expect("vector function")
    }

    #[test]
    fn l2_is_squared_euclidean() {
        assert_eq!(
            eval("vector_distance", vector(&[0.0, 0.0]), vector(&[3.0, 4.0])),
            Ok(Value::Float(25.0))
        );
    }

    #[test]
    fn decimal_literal_elements_are_numeric() {
        let decimals = Value::Array(vec![
            Value::Decimal(rust_decimal::Decimal::new(30, 1)),
            Value::Decimal(rust_decimal::Decimal::new(40, 1)),
        ]);
        assert_eq!(
            eval("vector_distance", vector(&[0.0, 0.0]), decimals),
            Ok(Value::Float(25.0))
        );
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_one() {
        let Ok(Value::Float(d)) = eval(
            "vector_cosine_distance",
            vector(&[1.0, 0.0]),
            vector(&[0.0, 1.0]),
        ) else {
            panic!("expected a float");
        };
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn neg_inner_product_negates_the_dot_product() {
        assert_eq!(
            eval(
                "vector_neg_inner_product",
                vector(&[1.0, 2.0]),
                vector(&[3.0, 4.0])
            ),
            Ok(Value::Float(-11.0))
        );
    }

    #[test]
    fn vector_values_integer_elements_and_json_text_are_vectors() {
        let ints = Value::Array(vec![Value::Integer(0), Value::Integer(0)]);
        assert_eq!(
            eval("vector_distance", ints, Value::String("[3, 4]".into())),
            Ok(Value::Float(25.0))
        );
        let packed = Value::Vector(vec![3.0_f32, 4.0].into());
        assert_eq!(
            eval("vector_distance", packed, vector(&[0.0, 0.0])),
            Ok(Value::Float(25.0))
        );
    }

    #[test]
    fn dimension_mismatch_names_both_dimensions() {
        let err = eval("vector_distance", vector(&[1.0]), vector(&[1.0, 2.0])).unwrap_err();
        assert_eq!(
            err,
            EvalError::VectorDimensionMismatch {
                function: "vector_distance",
                expected: 1,
                got: 2,
            }
        );
        assert_eq!(
            err.to_string(),
            "vector_distance(): vector dimension mismatch: expected 1, got 2"
        );
    }

    #[test]
    fn non_vector_argument_names_its_position_and_type() {
        let err = eval("vector_cosine_distance", vector(&[1.0]), Value::Integer(1)).unwrap_err();
        assert_eq!(
            err,
            EvalError::ArgumentType {
                function: "vector_cosine_distance",
                position: 2,
                expected: NUMERIC_VECTOR,
                got: "int",
            }
        );
    }

    #[test]
    fn non_numeric_element_names_the_element_type() {
        let mixed = Value::Array(vec![Value::Float(1.0), Value::String("x".into())]);
        let err = eval("vector_distance", mixed, vector(&[1.0, 2.0])).unwrap_err();
        assert!(
            matches!(
                err,
                EvalError::ArgumentType {
                    position: 1,
                    got: "string",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn text_that_is_not_a_vector_is_an_argument_error() {
        let err = eval(
            "vector_distance",
            Value::String("not a vector".into()),
            vector(&[1.0]),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                EvalError::ArgumentType {
                    position: 1,
                    got: "string",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn null_operand_is_null() {
        assert_eq!(
            eval("vector_distance", Value::Null, vector(&[1.0])),
            Ok(Value::Null)
        );
        assert_eq!(
            eval("vector_distance", vector(&[1.0]), Value::Null),
            Ok(Value::Null)
        );
    }

    #[test]
    fn other_names_are_not_claimed() {
        assert!(try_eval("bm25_score", &[]).is_none());
    }
}
