// SPDX-License-Identifier: Apache-2.0

//! SQL `LIKE` / `ILIKE` pattern matching: the one matcher every LIKE path
//! uses, the scan filters and the `like` / `ilike` scalar functions alike.
//!
//! - `%` matches zero or more characters.
//! - `_` matches exactly one character (a Unicode scalar value, not a byte).
//! - The escape character makes the next pattern character literal. The
//!   default escape is `\`, as in PostgreSQL. A trailing escape character
//!   matches itself.
//! - `ILIKE` lowercases input and pattern by Unicode rules before matching.

/// The escape character a pattern uses when none is given.
pub const DEFAULT_LIKE_ESCAPE: char = '\\';

/// Match `input` against the SQL LIKE `pattern`, with `\` as the escape.
pub fn sql_like_match(input: &str, pattern: &str, case_insensitive: bool) -> bool {
    sql_like_match_escaped(input, pattern, case_insensitive, Some(DEFAULT_LIKE_ESCAPE))
}

/// Match `input` against the SQL LIKE `pattern` with the escape character
/// `escape`. `None` means the pattern has no escape character.
pub fn sql_like_match_escaped(
    input: &str,
    pattern: &str,
    case_insensitive: bool,
    escape: Option<char>,
) -> bool {
    let (input, pattern) = if case_insensitive {
        (input.to_lowercase(), pattern.to_lowercase())
    } else {
        (input.to_owned(), pattern.to_owned())
    };
    let input: Vec<char> = input.chars().collect();
    let tokens = tokenize(&pattern, escape);
    matches(&input, &tokens)
}

/// One element of a compiled pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    /// A character that must match itself.
    Char(char),
    /// `_`: any one character.
    AnyOne,
    /// `%`: any run of characters, the empty run included.
    AnyRun,
}

fn tokenize(pattern: &str, escape: Option<char>) -> Vec<Token> {
    let mut tokens = Vec::with_capacity(pattern.len());
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        let token = if Some(c) == escape {
            // The escaped character is literal. A trailing escape is itself.
            Token::Char(chars.next().unwrap_or(c))
        } else {
            match c {
                '%' => Token::AnyRun,
                '_' => Token::AnyOne,
                other => Token::Char(other),
            }
        };
        tokens.push(token);
    }
    tokens
}

/// Greedy match with single-point backtracking to the last `%`.
fn matches(input: &[char], tokens: &[Token]) -> bool {
    let (mut i, mut t) = (0usize, 0usize);
    let mut backtrack: Option<(usize, usize)> = None;
    while i < input.len() {
        match tokens.get(t) {
            Some(Token::AnyRun) => {
                backtrack = Some((t, i));
                t += 1;
                continue;
            }
            Some(Token::AnyOne) => {
                i += 1;
                t += 1;
                continue;
            }
            Some(Token::Char(c)) if *c == input[i] => {
                i += 1;
                t += 1;
                continue;
            }
            Some(Token::Char(_)) | None => {}
        }
        match backtrack {
            Some((run_t, run_i)) => {
                backtrack = Some((run_t, run_i + 1));
                i = run_i + 1;
                t = run_t + 1;
            }
            None => return false,
        }
    }
    tokens[t..].iter().all(|token| *token == Token::AnyRun)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn like_basic() {
        assert!(sql_like_match("hello world", "%world", false));
        assert!(sql_like_match("hello world", "hello%", false));
        assert!(!sql_like_match("hello world", "xyz%", false));
        assert!(sql_like_match("", "%", false));
        assert!(!sql_like_match("", "_", false));
    }

    #[test]
    fn underscore_matches_one_character_not_one_byte() {
        assert!(sql_like_match("é", "_", false));
        assert!(sql_like_match("naïve", "na_ve", false));
        assert!(!sql_like_match("naïve", "na__ve", false));
    }

    #[test]
    fn escape_makes_wildcards_literal() {
        assert!(sql_like_match("100%", "100\\%", false));
        assert!(!sql_like_match("1000", "100\\%", false));
        assert!(sql_like_match("a_b", "a\\_b", false));
        assert!(!sql_like_match("axb", "a\\_b", false));
        assert!(sql_like_match("a\\", "a\\", false));
        assert!(sql_like_match_escaped("50!%", "50!!!%", false, Some('!')));
        assert!(sql_like_match_escaped("a\\b", "a\\b", false, None));
    }

    #[test]
    fn ilike_case_insensitive() {
        assert!(sql_like_match("Hello", "hello", true));
        assert!(sql_like_match("WORLD", "%world%", true));
        assert!(!sql_like_match("WORLD", "%world%", false));
    }

    #[test]
    fn ilike_folds_unicode_case() {
        assert!(sql_like_match("ÉCOLE", "école", true));
        assert!(sql_like_match("ΣΟΦΙΑ", "σοφια", true));
        assert!(!sql_like_match("ÉCOLE", "école", false));
    }

    #[test]
    fn backtracking_finds_a_later_match() {
        assert!(sql_like_match("abcabd", "%abd", false));
        assert!(sql_like_match("aaa", "%a%a", false));
        assert!(!sql_like_match("abc", "%d%", false));
    }
}
