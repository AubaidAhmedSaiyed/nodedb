// SPDX-License-Identifier: BUSL-1.1

//! Read a window timestamp back from the value bytes of one sort column.
//!
//! The value encoding carries a type tag, so the reader needs no guess:
//!
//! - a number is milliseconds since the Unix epoch, whole or fractional;
//! - a string is an ISO 8601 instant;
//! - anything else, and a string that is not a date, has no timestamp.

use nodedb_types::NdbDateTime;

use super::number::SortNumber;
use super::value::{SortValue, decode};

/// A timestamp read from a sort column, in milliseconds since the Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnTimestamp {
    /// Whole milliseconds, from a whole number or an ISO 8601 string.
    Millis(i128),
    /// Fractional milliseconds, or a float past the whole-number range.
    FractionalMillis(f64),
}

/// Decode the value bytes of a timestamp column.
///
/// Returns `None` for null, bool, array, object, and a string that is not
/// ISO 8601.
pub fn decode_column_timestamp(value: &[u8]) -> Option<ColumnTimestamp> {
    match decode(value)? {
        SortValue::Number(SortNumber::Whole(ms)) => Some(ColumnTimestamp::Millis(ms)),
        SortValue::Number(SortNumber::Float(ms)) => Some(ColumnTimestamp::FractionalMillis(ms)),
        SortValue::Text(text) => NdbDateTime::parse(&text)
            .map(|instant| ColumnTimestamp::Millis(i128::from(instant.unix_millis()))),
        SortValue::Null | SortValue::Bool(_) | SortValue::Json(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::value::{encode_float, encode_int, encode_json, encode_text};
    use super::*;

    #[test]
    fn integer_timestamp_decodes_to_its_millis() {
        for ms in [0i64, 1, 1_714_608_000_000, (1 << 53) + 1, i64::MAX] {
            assert_eq!(
                decode_column_timestamp(&encode_int(ms)),
                Some(ColumnTimestamp::Millis(i128::from(ms)))
            );
        }
    }

    #[test]
    fn negative_integer_timestamp_decodes_negative() {
        assert_eq!(
            decode_column_timestamp(&encode_int(-86_400_000)),
            Some(ColumnTimestamp::Millis(-86_400_000))
        );
    }

    #[test]
    fn float_timestamp_decodes_to_its_fractional_millis() {
        for ms in [
            1_714_608_000_000.5,
            1_714_607_999_999.25,
            -1_714_608_000_000.5,
        ] {
            assert_eq!(
                decode_column_timestamp(&encode_float(ms)),
                Some(ColumnTimestamp::FractionalMillis(ms))
            );
        }
    }

    #[test]
    fn whole_float_timestamp_decodes_to_whole_millis() {
        assert_eq!(
            decode_column_timestamp(&encode_float(1_714_608_000_000.0)),
            Some(ColumnTimestamp::Millis(1_714_608_000_000))
        );
    }

    /// Ten bytes is the length of an ISO date. The type tag keeps it a string.
    #[test]
    fn iso_8601_string_timestamp_decodes_to_millis() {
        assert_eq!(
            decode_column_timestamp(&encode_text("2024-05-02T00:00:00Z")),
            Some(ColumnTimestamp::Millis(1_714_608_000_000))
        );
        assert_eq!(
            decode_column_timestamp(&encode_text("2024-05-02")),
            Some(ColumnTimestamp::Millis(1_714_608_000_000))
        );
    }

    #[test]
    fn null_bool_or_non_date_value_has_no_timestamp() {
        assert_eq!(
            decode_column_timestamp(&encode_json(&serde_json::Value::Null)),
            None
        );
        assert_eq!(
            decode_column_timestamp(&encode_json(&serde_json::json!(true))),
            None
        );
        assert_eq!(decode_column_timestamp(&encode_text("yesterday")), None);
        assert_eq!(decode_column_timestamp(b""), None);
    }
}
