// SPDX-License-Identifier: Apache-2.0

//! Plan-time refusal of index-owned search functions in row-evaluated
//! positions.
//!
//! `bm25_score`, `search_score`, `text_match`, `search`, `rrf_score`,
//! `sparse_score`, `graph_score`, `multi_vector_score` and
//! `multi_vector_search` read a search index. The planner lowers each call it
//! recognises into its search plan, which serves the score as a column. A
//! call the planner could not lower stays in a filter, projection, sort key,
//! assignment or aggregate argument. The row evaluator has no index and no
//! value for it, so this pass refuses the statement at plan time. The
//! refusal does not depend on whether the collection holds rows.
//!
//! A wrapper plan (a subquery tail, aggregate, join or lateral join) over a
//! search plan is not checked for its own expressions: its projection names
//! the score column the search plan serves.

use crate::error::{Result, SqlError};
use crate::functions::registry::{FunctionRegistry, SearchTrigger};
use crate::types::query::{AggregateExpr, Projection, SortKey, WindowSpec};
use crate::types::{Filter, FilterExpr, MergePlanAction, SqlPlan};
use crate::types_expr::SqlExpr;

/// Refuse `plan` when an index-owned search function sits where the row
/// evaluator runs it.
pub fn refuse_row_scoped_search_functions(
    plan: &SqlPlan,
    functions: &FunctionRegistry,
) -> Result<()> {
    Scope { functions }.plan(plan)
}

struct Scope<'a> {
    functions: &'a FunctionRegistry,
}

impl Scope<'_> {
    fn plan(&self, plan: &SqlPlan) -> Result<()> {
        match plan {
            SqlPlan::Scan {
                filters,
                projection,
                sort_keys,
                window_functions,
                ..
            }
            | SqlPlan::DocumentIndexLookup {
                filters,
                projection,
                sort_keys,
                window_functions,
                ..
            } => {
                self.filters(filters)?;
                self.projection(projection)?;
                self.sort_keys(sort_keys)?;
                self.windows(window_functions)
            }
            SqlPlan::PointGet { projection, .. } | SqlPlan::RangeScan { projection, .. } => {
                self.projection(projection)
            }
            SqlPlan::KvInsert {
                on_conflict_updates,
                ..
            }
            | SqlPlan::Upsert {
                on_conflict_updates,
                ..
            }
            | SqlPlan::VectorPrimaryInsert {
                on_conflict_updates,
                ..
            } => self.assignments(on_conflict_updates),
            SqlPlan::InsertSelect {
                source, column_map, ..
            } => {
                self.plan(source)?;
                self.assignments(column_map)
            }
            SqlPlan::Update {
                assignments,
                filters,
                ..
            }
            | SqlPlan::VectorPrimaryUpdate {
                assignments,
                filters,
                ..
            } => {
                self.assignments(assignments)?;
                self.filters(filters)
            }
            SqlPlan::UpdateFrom {
                source,
                assignments,
                target_filters,
                ..
            } => {
                self.plan(source)?;
                self.assignments(assignments)?;
                self.filters(target_filters)
            }
            SqlPlan::Delete { filters, .. } | SqlPlan::VectorPrimaryDelete { filters, .. } => {
                self.filters(filters)
            }
            SqlPlan::Join {
                left,
                right,
                condition,
                projection,
                filters,
                ..
            } => {
                self.plan(left)?;
                self.plan(right)?;
                if has_search_plan(left) || has_search_plan(right) {
                    return Ok(());
                }
                if let Some(condition) = condition {
                    self.expr(condition)?;
                }
                self.projection(projection)?;
                self.filters(filters)
            }
            SqlPlan::Aggregate {
                input,
                group_by,
                aggregates,
                having,
                sort_keys,
                ..
            } => {
                self.plan(input)?;
                if has_search_plan(input) {
                    return Ok(());
                }
                self.exprs(group_by)?;
                self.aggregates(aggregates)?;
                self.filters(having)?;
                self.sort_keys(sort_keys)
            }
            SqlPlan::TimeseriesScan {
                aggregates,
                filters,
                projection,
                sort_keys,
                ..
            } => {
                self.aggregates(aggregates)?;
                self.filters(filters)?;
                self.projection(projection)?;
                self.sort_keys(sort_keys)
            }
            // A search plan serves its own score call as a column; only its
            // residual filters run on the row evaluator.
            SqlPlan::VectorSearch { filters, .. } | SqlPlan::TextSearch { filters, .. } => {
                self.filters(filters)
            }
            SqlPlan::SpatialScan {
                attribute_filters, ..
            } => self.filters(attribute_filters),
            SqlPlan::RecursiveScan {
                base_filters,
                recursive_filters,
                ..
            } => {
                self.filters(base_filters)?;
                self.filters(recursive_filters)
            }
            SqlPlan::Union { inputs, .. } => inputs.iter().try_for_each(|input| self.plan(input)),
            SqlPlan::Intersect { left, right, .. } | SqlPlan::Except { left, right, .. } => {
                self.plan(left)?;
                self.plan(right)
            }
            SqlPlan::Cte { definitions, outer } => {
                for (_, definition) in definitions {
                    self.plan(definition)?;
                }
                self.plan(outer)
            }
            SqlPlan::Subquery {
                input,
                filters,
                projection,
                window_functions,
                sort_keys,
                ..
            } => {
                self.plan(input)?;
                if has_search_plan(input) {
                    return Ok(());
                }
                self.filters(filters)?;
                self.projection(projection)?;
                self.windows(window_functions)?;
                self.sort_keys(sort_keys)
            }
            SqlPlan::Merge {
                source, clauses, ..
            } => {
                self.plan(source)?;
                for clause in clauses {
                    self.filters(&clause.extra_predicate)?;
                    match &clause.action {
                        MergePlanAction::Update { assignments } => self.assignments(assignments)?,
                        MergePlanAction::Insert { values, .. } => self.exprs(values)?,
                        MergePlanAction::Delete | MergePlanAction::DoNothing => {}
                    }
                }
                Ok(())
            }
            SqlPlan::LateralTopK {
                outer,
                inner_filters,
                inner_order_by,
                projection,
                ..
            } => {
                self.plan(outer)?;
                self.filters(inner_filters)?;
                self.sort_keys(inner_order_by)?;
                if has_search_plan(outer) {
                    return Ok(());
                }
                self.projection(projection)
            }
            SqlPlan::LateralLoop {
                outer,
                inner,
                projection,
                ..
            } => {
                self.plan(outer)?;
                self.plan(inner)?;
                if has_search_plan(outer) || has_search_plan(inner) {
                    return Ok(());
                }
                self.projection(projection)
            }
            // No row-evaluated expression: constants, literal-row writes,
            // search plans that carry no residual filter, array statements,
            // and DDL.
            SqlPlan::ConstantResult { .. }
            | SqlPlan::Insert { .. }
            | SqlPlan::Truncate { .. }
            | SqlPlan::TimeseriesIngest { .. }
            | SqlPlan::MultiVectorSearch { .. }
            | SqlPlan::SparseSearch { .. }
            | SqlPlan::HybridSearch { .. }
            | SqlPlan::HybridSearchTriple { .. }
            | SqlPlan::RecursiveValue { .. }
            | SqlPlan::CreateArray { .. }
            | SqlPlan::DropArray { .. }
            | SqlPlan::AlterArray { .. }
            | SqlPlan::InsertArray { .. }
            | SqlPlan::DeleteArray { .. }
            | SqlPlan::ArraySlice { .. }
            | SqlPlan::ArrayProject { .. }
            | SqlPlan::ArrayAgg { .. }
            | SqlPlan::ArrayElementwise { .. }
            | SqlPlan::ArrayFlush { .. }
            | SqlPlan::ArrayCompact { .. }
            | SqlPlan::VectorPrimaryTruncate { .. }
            | SqlPlan::CreateIndex { .. }
            | SqlPlan::DropIndex { .. } => Ok(()),
        }
    }

    fn filters(&self, filters: &[Filter]) -> Result<()> {
        filters
            .iter()
            .try_for_each(|filter| self.filter(&filter.expr))
    }

    fn filter(&self, expr: &FilterExpr) -> Result<()> {
        match expr {
            FilterExpr::Expr(expr) => self.expr(expr),
            FilterExpr::And(children) | FilterExpr::Or(children) => self.filters(children),
            FilterExpr::Not(child) => self.filter(&child.expr),
            FilterExpr::Comparison { .. }
            | FilterExpr::InList { .. }
            | FilterExpr::Between { .. }
            | FilterExpr::IsNull { .. }
            | FilterExpr::IsNotNull { .. } => Ok(()),
        }
    }

    fn projection(&self, projection: &[Projection]) -> Result<()> {
        for item in projection {
            match item {
                Projection::Computed { expr, .. } | Projection::CpComputed { expr, .. } => {
                    self.expr(expr)?
                }
                Projection::Column(_) | Projection::Star | Projection::QualifiedStar(_) => {}
            }
        }
        Ok(())
    }

    fn sort_keys(&self, sort_keys: &[SortKey]) -> Result<()> {
        sort_keys.iter().try_for_each(|key| self.expr(&key.expr))
    }

    fn windows(&self, windows: &[WindowSpec]) -> Result<()> {
        for window in windows {
            self.exprs(&window.args)?;
            self.exprs(&window.partition_by)?;
            self.sort_keys(&window.order_by)?;
        }
        Ok(())
    }

    fn aggregates(&self, aggregates: &[AggregateExpr]) -> Result<()> {
        aggregates
            .iter()
            .try_for_each(|aggregate| self.exprs(&aggregate.args))
    }

    fn assignments(&self, assignments: &[(String, SqlExpr)]) -> Result<()> {
        assignments.iter().try_for_each(|(_, expr)| self.expr(expr))
    }

    fn exprs(&self, exprs: &[SqlExpr]) -> Result<()> {
        exprs.iter().try_for_each(|expr| self.expr(expr))
    }

    fn expr(&self, expr: &SqlExpr) -> Result<()> {
        match first_search_function(expr, self.functions) {
            Some(name) => Err(SqlError::SearchFunctionOutsideSearch {
                name: name.to_owned(),
            }),
            None => Ok(()),
        }
    }
}

/// Whether the search trigger `trigger` names a function that reads an
/// index and has no per-row value.
fn is_index_owned(trigger: SearchTrigger) -> bool {
    match trigger {
        SearchTrigger::MultiVectorSearch
        | SearchTrigger::SparseSearch
        | SearchTrigger::TextSearch
        | SearchTrigger::HybridSearch
        | SearchTrigger::TextMatch
        | SearchTrigger::GraphSearch => true,
        // The vector distances and the spatial predicates evaluate per row;
        // the time bucket is a scalar; the array functions are table-valued
        // and planned from FROM.
        SearchTrigger::None
        | SearchTrigger::VectorSearch
        | SearchTrigger::SpatialDWithin
        | SearchTrigger::SpatialContains
        | SearchTrigger::SpatialIntersects
        | SearchTrigger::SpatialWithin
        | SearchTrigger::TimeBucket
        | SearchTrigger::ArraySlice
        | SearchTrigger::ArrayProject
        | SearchTrigger::ArrayAgg
        | SearchTrigger::ArrayElementwise
        | SearchTrigger::ArrayFlush
        | SearchTrigger::ArrayCompact => false,
    }
}

/// The first index-owned search function `expr` calls outside a subquery.
fn first_search_function<'e>(expr: &'e SqlExpr, functions: &FunctionRegistry) -> Option<&'e str> {
    let find = |e: &'e SqlExpr| first_search_function(e, functions);
    match expr {
        SqlExpr::Function { name, args, .. } => {
            if is_index_owned(functions.search_trigger(name)) {
                return Some(name.as_str());
            }
            args.iter().find_map(find)
        }
        SqlExpr::BinaryOp { left, right, .. } => find(left).or_else(|| find(right)),
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Cast { expr, .. }
        | SqlExpr::IsNull { expr, .. } => find(expr),
        SqlExpr::Case {
            operand,
            when_then,
            else_expr,
        } => operand
            .as_deref()
            .and_then(find)
            .or_else(|| {
                when_then
                    .iter()
                    .find_map(|(when, then)| find(when).or_else(|| find(then)))
            })
            .or_else(|| else_expr.as_deref().and_then(find)),
        SqlExpr::InList { expr, list, .. } => find(expr).or_else(|| list.iter().find_map(find)),
        SqlExpr::Between {
            expr, low, high, ..
        } => find(expr).or_else(|| find(low)).or_else(|| find(high)),
        SqlExpr::Like { expr, pattern, .. } => find(expr).or_else(|| find(pattern)),
        SqlExpr::ArrayLiteral(items) => items.iter().find_map(find),
        SqlExpr::Column { .. } | SqlExpr::Literal(_) | SqlExpr::Subquery(_) | SqlExpr::Wildcard => {
            None
        }
    }
}

/// Whether `plan` is, or wraps, a search plan that serves a score column.
fn has_search_plan(plan: &SqlPlan) -> bool {
    match plan {
        SqlPlan::VectorSearch { .. }
        | SqlPlan::MultiVectorSearch { .. }
        | SqlPlan::SparseSearch { .. }
        | SqlPlan::TextSearch { .. }
        | SqlPlan::HybridSearch { .. }
        | SqlPlan::HybridSearchTriple { .. } => true,
        SqlPlan::Subquery { input, .. } | SqlPlan::Aggregate { input, .. } => {
            has_search_plan(input)
        }
        SqlPlan::Join { left, right, .. } => has_search_plan(left) || has_search_plan(right),
        SqlPlan::LateralTopK { outer, .. } => has_search_plan(outer),
        SqlPlan::LateralLoop { outer, inner, .. } => {
            has_search_plan(outer) || has_search_plan(inner)
        }
        SqlPlan::Union { inputs, .. } => inputs.iter().any(has_search_plan),
        SqlPlan::Intersect { left, right, .. } | SqlPlan::Except { left, right, .. } => {
            has_search_plan(left) || has_search_plan(right)
        }
        SqlPlan::Cte { outer, .. } => has_search_plan(outer),
        SqlPlan::ConstantResult { .. }
        | SqlPlan::Scan { .. }
        | SqlPlan::PointGet { .. }
        | SqlPlan::DocumentIndexLookup { .. }
        | SqlPlan::RangeScan { .. }
        | SqlPlan::Insert { .. }
        | SqlPlan::KvInsert { .. }
        | SqlPlan::Upsert { .. }
        | SqlPlan::InsertSelect { .. }
        | SqlPlan::Update { .. }
        | SqlPlan::UpdateFrom { .. }
        | SqlPlan::Delete { .. }
        | SqlPlan::Truncate { .. }
        | SqlPlan::TimeseriesScan { .. }
        | SqlPlan::TimeseriesIngest { .. }
        | SqlPlan::SpatialScan { .. }
        | SqlPlan::RecursiveScan { .. }
        | SqlPlan::RecursiveValue { .. }
        | SqlPlan::CreateArray { .. }
        | SqlPlan::DropArray { .. }
        | SqlPlan::AlterArray { .. }
        | SqlPlan::InsertArray { .. }
        | SqlPlan::DeleteArray { .. }
        | SqlPlan::ArraySlice { .. }
        | SqlPlan::ArrayProject { .. }
        | SqlPlan::ArrayAgg { .. }
        | SqlPlan::ArrayElementwise { .. }
        | SqlPlan::ArrayFlush { .. }
        | SqlPlan::ArrayCompact { .. }
        | SqlPlan::Merge { .. }
        | SqlPlan::VectorPrimaryInsert { .. }
        | SqlPlan::VectorPrimaryDelete { .. }
        | SqlPlan::VectorPrimaryTruncate { .. }
        | SqlPlan::VectorPrimaryUpdate { .. }
        | SqlPlan::CreateIndex { .. }
        | SqlPlan::DropIndex { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::query::EngineType;
    use crate::types_expr::SqlValue;

    fn call(name: &str) -> SqlExpr {
        SqlExpr::Function {
            name: name.into(),
            args: vec![
                SqlExpr::Column {
                    table: None,
                    name: "body".into(),
                },
                SqlExpr::Literal(SqlValue::String("rust".into())),
            ],
            distinct: false,
        }
    }

    fn scan(projection: Vec<Projection>, filters: Vec<Filter>) -> SqlPlan {
        SqlPlan::Scan {
            collection: "docs".into(),
            alias: None,
            engine: EngineType::DocumentSchemaless,
            filters,
            projection,
            sort_keys: Vec::new(),
            limit: None,
            offset: 0,
            distinct: false,
            window_functions: Vec::new(),
            temporal: Default::default(),
        }
    }

    fn check(plan: &SqlPlan) -> Result<()> {
        refuse_row_scoped_search_functions(plan, &FunctionRegistry::new())
    }

    #[test]
    fn a_score_in_a_scan_projection_is_refused() {
        let plan = scan(
            vec![Projection::Computed {
                expr: call("bm25_score"),
                alias: "s".into(),
            }],
            Vec::new(),
        );
        assert_eq!(
            check(&plan),
            Err(SqlError::SearchFunctionOutsideSearch {
                name: "bm25_score".into()
            })
        );
    }

    #[test]
    fn a_match_nested_in_a_scan_filter_is_refused() {
        let nested = SqlExpr::BinaryOp {
            left: Box::new(call("text_match")),
            op: crate::types_expr::BinaryOp::Or,
            right: Box::new(SqlExpr::Literal(SqlValue::Bool(false))),
        };
        let plan = scan(
            Vec::new(),
            vec![Filter {
                expr: FilterExpr::Expr(nested),
            }],
        );
        assert!(matches!(
            check(&plan),
            Err(SqlError::SearchFunctionOutsideSearch { .. })
        ));
    }

    #[test]
    fn row_scalars_in_a_scan_pass() {
        let plan = scan(
            vec![
                Projection::Computed {
                    expr: call("vector_distance"),
                    alias: "d".into(),
                },
                Projection::Computed {
                    expr: call("doc_get"),
                    alias: "g".into(),
                },
            ],
            Vec::new(),
        );
        assert_eq!(check(&plan), Ok(()));
    }

    #[test]
    fn a_search_plan_projection_serves_its_score() {
        let plan = SqlPlan::TextSearch {
            collection: "docs".into(),
            query: crate::fts_types::FtsQuery::Plain {
                text: "rust".into(),
                fuzzy: true,
            },
            top_k: 10,
            filters: Vec::new(),
            score_alias: Some("s".into()),
            projection: vec![Projection::Computed {
                expr: call("bm25_score"),
                alias: "s".into(),
            }],
        };
        assert_eq!(check(&plan), Ok(()));
    }

    #[test]
    fn a_subquery_tail_over_a_search_plan_is_not_checked() {
        let search = SqlPlan::TextSearch {
            collection: "docs".into(),
            query: crate::fts_types::FtsQuery::Plain {
                text: "rust".into(),
                fuzzy: true,
            },
            top_k: 10,
            filters: Vec::new(),
            score_alias: Some("s".into()),
            projection: Vec::new(),
        };
        let plan = SqlPlan::Subquery {
            input: Box::new(search),
            filters: Vec::new(),
            projection: vec![Projection::Computed {
                expr: call("bm25_score"),
                alias: "s".into(),
            }],
            window_functions: Vec::new(),
            sort_keys: Vec::new(),
            offset: 0,
            distinct: false,
            limit: None,
        };
        assert_eq!(check(&plan), Ok(()));
    }
}
