// SPDX-License-Identifier: BUSL-1.1

//! What a vector-primary direct write may put in a log record.
//!
//! The sidecar store is written through, while the vector index restores
//! from its checkpoint. Replay therefore runs against sidecars newer than
//! the index. A record names its rows and their values, never a row
//! predicate or a patch expression replay evaluates again. Such a write
//! resolves to rows first (`control::write_resolve`).

use nodedb_physical::physical_plan::{UpdateValue, VectorWriteTargets};
use nodedb_types::Surrogate;

/// The rows `targets` names. Refuses a row predicate.
pub(crate) fn logged_rows<'a>(
    collection: &str,
    targets: &'a VectorWriteTargets,
) -> crate::Result<&'a [Surrogate]> {
    match targets {
        VectorWriteTargets::Surrogates(rows) => Ok(rows),
        VectorWriteTargets::Predicate(_) => Err(crate::Error::Internal {
            detail: format!(
                "internal invariant break: a vector-primary write on '{collection}' reached \
                 the log with a row predicate; write_resolve resolves it to rows first"
            ),
        }),
    }
}

/// Refuse a `patch` that holds an expression.
pub(crate) fn check_logged_patch(
    collection: &str,
    patch: &[(String, UpdateValue)],
) -> crate::Result<()> {
    for (column, value) in patch {
        match value {
            UpdateValue::Literal(_) => {}
            UpdateValue::Expr(_) => {
                return Err(crate::Error::Internal {
                    detail: format!(
                        "internal invariant break: a vector-primary write on '{collection}' \
                         reached the log with an expression for '{column}'; write_resolve \
                         resolves it to a value first"
                    ),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_predicate_is_refused_and_rows_pass() {
        let rows = VectorWriteTargets::Surrogates(vec![Surrogate::new(3)]);
        assert_eq!(logged_rows("v", &rows).expect("rows"), &[Surrogate::new(3)]);
        let predicate = VectorWriteTargets::Predicate(Vec::new());
        assert!(logged_rows("v", &predicate).is_err());
    }

    #[test]
    fn an_expression_is_refused_and_a_literal_passes() {
        let literal = [("n".to_string(), UpdateValue::Literal(vec![0x01]))];
        assert!(check_logged_patch("v", &literal).is_ok());
        let expression = [(
            "n".to_string(),
            UpdateValue::Expr(nodedb_query::expr::SqlExpr::Column("n".into())),
        )];
        assert!(check_logged_patch("v", &expression).is_err());
    }
}
