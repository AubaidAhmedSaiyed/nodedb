// SPDX-License-Identifier: BUSL-1.1

//! The clauses of `CREATE SORTED INDEX` after the sort column list:
//!
//! ```text
//! KEY key_col
//!   [WINDOW {DAILY | WEEKLY | MONTHLY} [ON ts_col]]
//!   [WINDOW CUSTOM START bound END bound [ON ts_col]]
//! ```
//!
//! Identifiers keep their quoting, so a quoted mixed-case column resolves
//! the same way the sort column list resolves it. A bound is integer
//! milliseconds since the Unix epoch or an ISO 8601 instant.

use nodedb_types::NdbDateTime;

use super::super::super::result::DdlError;
use super::parse::ddl_err;
use crate::control::server::shared::ddl::sql_parse::parse_ident_token;

/// The timestamp column a window reads when the clause names none.
const DEFAULT_TIMESTAMP_COLUMN: &str = "updated_at";

/// The parsed clauses after the sort column list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct IndexTail {
    pub key_column: String,
    pub window: WindowClause,
}

/// A parsed `WINDOW` clause, in the shape `RegisterSortedIndex` carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WindowClause {
    /// `none`, `daily`, `weekly`, `monthly` or `custom`.
    pub window_type: String,
    /// Empty when `window_type` is `none`.
    pub timestamp_column: String,
    /// Custom windows only: the half-open range `[start_ms, end_ms)`.
    pub start_ms: u64,
    pub end_ms: u64,
}

impl WindowClause {
    fn none() -> Self {
        Self {
            window_type: "none".into(),
            timestamp_column: String::new(),
            start_ms: 0,
            end_ms: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token<'a> {
    /// A bare word or a double-quoted identifier, quotes kept.
    Word(&'a str),
    /// A single-quoted string literal, unescaped.
    Literal(String),
}

impl Token<'_> {
    fn is_keyword(&self, keyword: &str) -> bool {
        matches!(self, Token::Word(word) if word.eq_ignore_ascii_case(keyword))
    }

    fn describe(&self) -> String {
        match self {
            Token::Word(word) => (*word).to_string(),
            Token::Literal(text) => format!("'{text}'"),
        }
    }
}

/// Parse the text after the sort column list's closing parenthesis.
///
/// `sort_columns` are the parsed sort columns. A window timestamp column must
/// be one of them.
pub(super) fn parse_index_tail(
    text: &str,
    sort_columns: &[(String, String)],
) -> Result<IndexTail, DdlError> {
    let tokens = tokenize(text)?;
    let (key_column, rest) = match tokens.as_slice() {
        [key, Token::Word(name), tail @ ..] if key.is_keyword("KEY") => {
            (parse_ident_token(name)?, tail)
        }
        _ => {
            return Err(ddl_err(
                "42601",
                "missing KEY clause: expected KEY <column>",
            ));
        }
    };

    let window = match rest {
        [] => WindowClause::none(),
        [window, tail @ ..] if window.is_keyword("WINDOW") => parse_window(tail)?,
        [other, ..] => {
            return Err(ddl_err(
                "42601",
                format!("unexpected '{}' after KEY {key_column}", other.describe()),
            ));
        }
    };

    if window.window_type != "none"
        && !sort_columns
            .iter()
            .any(|(name, _)| *name == window.timestamp_column)
    {
        return Err(ddl_err(
            "42601",
            format!(
                "WINDOW timestamp column '{}' must be one of the sort columns",
                window.timestamp_column
            ),
        ));
    }

    Ok(IndexTail { key_column, window })
}

fn parse_window(tokens: &[Token<'_>]) -> Result<WindowClause, DdlError> {
    let (kind, rest) = match tokens {
        [Token::Word(kind), rest @ ..] => (kind.to_ascii_lowercase(), rest),
        [other, ..] => return Err(unknown_window(&other.describe())),
        [] => return Err(ddl_err("42601", "WINDOW requires a window type")),
    };
    match kind.as_str() {
        "daily" | "weekly" | "monthly" => {
            let timestamp_column = match rest {
                [] => DEFAULT_TIMESTAMP_COLUMN.to_string(),
                [on, Token::Word(column)] if on.is_keyword("ON") => parse_ident_token(column)?,
                [other, ..] => return Err(unexpected_in_window(other)),
            };
            Ok(WindowClause {
                window_type: kind,
                timestamp_column,
                start_ms: 0,
                end_ms: 0,
            })
        }
        "custom" => parse_custom_window(rest),
        _ => Err(unknown_window(&kind)),
    }
}

/// `START bound END bound [ON ts_col]`, each clause at most once, any order.
fn parse_custom_window(mut rest: &[Token<'_>]) -> Result<WindowClause, DdlError> {
    let mut start_ms = None;
    let mut end_ms = None;
    let mut timestamp_column = None;
    while let [keyword, value, tail @ ..] = rest {
        let duplicate = if keyword.is_keyword("START") {
            start_ms.replace(parse_bound(value)?).is_some()
        } else if keyword.is_keyword("END") {
            end_ms.replace(parse_bound(value)?).is_some()
        } else if keyword.is_keyword("ON") {
            let Token::Word(column) = value else {
                return Err(unexpected_in_window(value));
            };
            timestamp_column
                .replace(parse_ident_token(column)?)
                .is_some()
        } else {
            return Err(unexpected_in_window(keyword));
        };
        if duplicate {
            return Err(ddl_err(
                "42601",
                format!("WINDOW CUSTOM repeats '{}'", keyword.describe()),
            ));
        }
        rest = tail;
    }
    if let [dangling] = rest {
        return Err(ddl_err(
            "42601",
            format!("WINDOW CUSTOM '{}' has no value", dangling.describe()),
        ));
    }

    let (Some(start_ms), Some(end_ms)) = (start_ms, end_ms) else {
        return Err(ddl_err("42601", "WINDOW CUSTOM requires START and END"));
    };
    if end_ms <= start_ms {
        return Err(ddl_err(
            "42601",
            format!("WINDOW CUSTOM END ({end_ms} ms) must be after START ({start_ms} ms)"),
        ));
    }
    Ok(WindowClause {
        window_type: "custom".into(),
        timestamp_column: timestamp_column.unwrap_or_else(|| DEFAULT_TIMESTAMP_COLUMN.into()),
        start_ms,
        end_ms,
    })
}

/// Integer milliseconds since the Unix epoch, or an ISO 8601 instant at or
/// after the epoch.
fn parse_bound(token: &Token<'_>) -> Result<u64, DdlError> {
    let text = match token {
        Token::Word(word) => *word,
        Token::Literal(text) => text.as_str(),
    };
    if let Ok(ms) = text.trim().parse::<u64>() {
        return Ok(ms);
    }
    NdbDateTime::parse(text)
        .and_then(|instant| u64::try_from(instant.unix_millis()).ok())
        .ok_or_else(|| {
            ddl_err(
                "42601",
                format!(
                    "invalid WINDOW CUSTOM bound {}: expected milliseconds since the Unix \
                     epoch or an ISO 8601 instant at or after 1970-01-01",
                    token.describe()
                ),
            )
        })
}

fn unknown_window(kind: &str) -> DdlError {
    ddl_err(
        "42601",
        format!("unknown WINDOW type '{kind}', expected DAILY, WEEKLY, MONTHLY or CUSTOM"),
    )
}

fn unexpected_in_window(token: &Token<'_>) -> DdlError {
    ddl_err(
        "42601",
        format!("unexpected '{}' in WINDOW clause", token.describe()),
    )
}

/// Split `text` into words, double-quoted identifiers and single-quoted
/// literals. A trailing `;` ends the statement.
fn tokenize(text: &str) -> Result<Vec<Token<'_>>, DdlError> {
    let mut tokens = Vec::new();
    let mut rest = text.trim_start();
    while let Some(first) = rest.chars().next() {
        let consumed = match first {
            ';' => {
                if !rest[1..].trim().is_empty() {
                    return Err(ddl_err(
                        "42601",
                        format!("unexpected text after ';': {}", rest[1..].trim()),
                    ));
                }
                break;
            }
            '\'' => {
                let (literal, len) = quoted(rest, '\'')?;
                tokens.push(Token::Literal(literal));
                len
            }
            '"' => {
                let (_, len) = quoted(rest, '"')?;
                tokens.push(Token::Word(&rest[..len]));
                len
            }
            _ => {
                let len = rest
                    .find(|c: char| c.is_whitespace() || c == ';')
                    .unwrap_or(rest.len());
                tokens.push(Token::Word(&rest[..len]));
                len
            }
        };
        rest = rest[consumed..].trim_start();
    }
    Ok(tokens)
}

/// The unescaped body of the quoted token at the start of `text`, and the
/// token's byte length. A doubled quote inside is one quote.
fn quoted(text: &str, quote: char) -> Result<(String, usize), DdlError> {
    let mut body = String::new();
    let mut chars = text.char_indices().skip(1).peekable();
    while let Some((at, ch)) = chars.next() {
        if ch != quote {
            body.push(ch);
            continue;
        }
        if chars.peek().is_some_and(|&(_, next)| next == quote) {
            body.push(quote);
            chars.next();
            continue;
        }
        return Ok((body, at + ch.len_utf8()));
    }
    Err(ddl_err(
        "42601",
        format!("unterminated quoted text: {text}"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cols(names: &[&str]) -> Vec<(String, String)> {
        names
            .iter()
            .map(|name| (name.to_string(), "ASC".to_string()))
            .collect()
    }

    fn sqlstate_and_message(err: DdlError) -> (String, String) {
        (err.sqlstate, err.message)
    }

    #[test]
    fn key_only() {
        let tail = parse_index_tail(" KEY player_id;", &cols(&["score"])).expect("parse");
        assert_eq!(tail.key_column, "player_id");
        assert_eq!(tail.window, WindowClause::none());
    }

    #[test]
    fn daily_window_on_a_sort_column() {
        let tail = parse_index_tail(
            "key id window daily on updated_at",
            &cols(&["score", "updated_at"]),
        )
        .expect("parse");
        assert_eq!(tail.window.window_type, "daily");
        assert_eq!(tail.window.timestamp_column, "updated_at");
    }

    #[test]
    fn quoted_mixed_case_timestamp_column_matches_its_sort_column() {
        let quoted = parse_ident_token("\"UpdatedAt\"").expect("ident");
        let tail = parse_index_tail(
            "KEY \"PlayerId\" WINDOW WEEKLY ON \"UpdatedAt\"",
            &cols(&["score", quoted.as_str()]),
        )
        .expect("parse");
        assert_eq!(
            tail.key_column,
            parse_ident_token("\"PlayerId\"").expect("ident")
        );
        assert_eq!(tail.window.timestamp_column, quoted);
    }

    #[test]
    fn custom_window_accepts_millis_and_iso_8601() {
        let columns = cols(&["score", "ts"]);
        let tail = parse_index_tail(
            "KEY id WINDOW CUSTOM START '2026-01-01' END 1767312000000 ON ts",
            &columns,
        )
        .expect("parse");
        assert_eq!(tail.window.window_type, "custom");
        assert_eq!(tail.window.start_ms, 1_767_225_600_000);
        assert_eq!(tail.window.end_ms, 1_767_312_000_000);
        assert_eq!(tail.window.timestamp_column, "ts");

        let tail = parse_index_tail(
            "KEY id WINDOW CUSTOM ON ts START '2026-01-01T00:00:00Z' END '2026-03-31 12:00:00'",
            &columns,
        )
        .expect("parse");
        assert_eq!(tail.window.start_ms, 1_767_225_600_000);
        assert_eq!(tail.window.end_ms, 1_774_958_400_000);
    }

    #[test]
    fn custom_window_refuses_a_bad_bound_and_names_it() {
        let err = parse_index_tail(
            "KEY id WINDOW CUSTOM START 'last tuesday' END 5 ON ts",
            &cols(&["ts"]),
        )
        .expect_err("bad bound");
        let (state, message) = sqlstate_and_message(err);
        assert_eq!(state, "42601");
        assert!(message.contains("'last tuesday'"), "{message}");
    }

    #[test]
    fn custom_window_refuses_an_empty_or_inverted_range() {
        for clause in [
            "KEY id WINDOW CUSTOM START 10 END 10 ON ts",
            "KEY id WINDOW CUSTOM START '2026-03-31' END '2026-01-01' ON ts",
            "KEY id WINDOW CUSTOM START 10 ON ts",
            "KEY id WINDOW CUSTOM START 1 START 2 END 3 ON ts",
            "KEY id WINDOW CUSTOM START 1 END",
        ] {
            let err = parse_index_tail(clause, &cols(&["ts"])).expect_err(clause);
            assert_eq!(err.sqlstate, "42601", "{clause}");
        }
    }

    #[test]
    fn window_timestamp_column_must_be_a_sort_column() {
        let err = parse_index_tail(
            "KEY player_id WINDOW DAILY ON updated_at",
            &cols(&["score"]),
        )
        .expect_err("not a sort column");
        let (state, message) = sqlstate_and_message(err);
        assert_eq!(state, "42601");
        assert!(message.contains("updated_at"), "{message}");
    }

    #[test]
    fn unknown_window_type_or_trailing_text_is_refused() {
        for clause in [
            "KEY id WINDOW HOURLY ON ts",
            "KEY id WINDOW DAILY ON ts EXTRA",
            "KEY id EXTRA",
            "id",
            "KEY id; DROP",
        ] {
            let err = parse_index_tail(clause, &cols(&["ts"])).expect_err(clause);
            assert_eq!(err.sqlstate, "42601", "{clause}");
        }
    }
}
