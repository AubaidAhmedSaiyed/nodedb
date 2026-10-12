// SPDX-License-Identifier: BUSL-1.1

//! Read a window timestamp back from the value bytes of one sort column.
//!
//! A sort column stores the bytes the KV value extractor wrote for it:
//!
//! - an integer: [`SortKeyEncoder::encode_i64`], 8 bytes;
//! - a float: [`SortKeyEncoder::encode_f64`], 8 bytes;
//! - a string: its UTF-8 bytes;
//! - null: no bytes.
//!
//! The bytes carry no type tag. A non-negative integer and a non-negative
//! float both encode with the top bit set, so the payload under that bit
//! picks the decoder. An integer timestamp of 2^53 ms is past the year
//! 287,000. A non-negative float whose bit pattern is below 2^53 is below
//! 2^-1021 ms. Payloads below 2^53 decode as integers, the rest as floats.
//! The two ranges overlap only on values no timestamp column holds.

use nodedb_types::NdbDateTime;

use super::key::SortKeyEncoder;

/// Top bit of an encoded 8-byte number. Set for every non-negative value.
const SIGN_BIT: u64 = 1 << 63;

/// Payloads below this bound decode as integer milliseconds.
const INTEGER_PAYLOAD_LIMIT: u64 = 1 << 53;

/// A timestamp read from a sort column, in milliseconds since the Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnTimestamp {
    /// Whole milliseconds, from an integer or an ISO 8601 string column.
    Millis(i64),
    /// Fractional milliseconds, from a float column.
    FractionalMillis(f64),
}

/// Decode the value bytes of a timestamp column.
///
/// Returns `None` for null (no bytes) and for a string that is not ISO 8601.
pub fn decode_column_timestamp(value: &[u8]) -> Option<ColumnTimestamp> {
    if let Ok(bytes) = <&[u8; 8]>::try_from(value) {
        return Some(decode_number(bytes));
    }
    let text = std::str::from_utf8(value).ok()?;
    NdbDateTime::parse(text).map(|instant| ColumnTimestamp::Millis(instant.unix_millis()))
}

/// Decode an 8-byte number written by `encode_i64` or `encode_f64`.
fn decode_number(bytes: &[u8; 8]) -> ColumnTimestamp {
    let encoded = u64::from_be_bytes(*bytes);
    // A clear top bit is a negative value in both encodings. Every window
    // starts at or after the epoch, so the integer reading keeps it outside.
    if (encoded & SIGN_BIT) == 0 || (encoded ^ SIGN_BIT) < INTEGER_PAYLOAD_LIMIT {
        ColumnTimestamp::Millis(SortKeyEncoder::decode_i64(bytes))
    } else {
        ColumnTimestamp::FractionalMillis(SortKeyEncoder::decode_f64(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_timestamp_decodes_to_its_millis() {
        for ms in [0i64, 1, 1_714_608_000_000, (1 << 53) - 1] {
            let bytes = SortKeyEncoder::encode_i64(ms);
            assert_eq!(
                decode_column_timestamp(&bytes),
                Some(ColumnTimestamp::Millis(ms))
            );
        }
    }

    #[test]
    fn negative_integer_timestamp_decodes_negative() {
        let bytes = SortKeyEncoder::encode_i64(-86_400_000);
        assert_eq!(
            decode_column_timestamp(&bytes),
            Some(ColumnTimestamp::Millis(-86_400_000))
        );
    }

    #[test]
    fn float_timestamp_decodes_to_its_fractional_millis() {
        for ms in [1.0f64, 1_714_608_000_000.5, 1_714_607_999_999.25] {
            let bytes = SortKeyEncoder::encode_f64(ms);
            assert_eq!(
                decode_column_timestamp(&bytes),
                Some(ColumnTimestamp::FractionalMillis(ms))
            );
        }
    }

    #[test]
    fn negative_float_timestamp_decodes_negative() {
        let bytes = SortKeyEncoder::encode_f64(-1_714_608_000_000.5);
        match decode_column_timestamp(&bytes) {
            Some(ColumnTimestamp::Millis(ms)) => assert!(ms < 0),
            Some(ColumnTimestamp::FractionalMillis(ms)) => assert!(ms < 0.0),
            None => panic!("a negative float must decode"),
        }
    }

    #[test]
    fn iso_8601_string_timestamp_decodes_to_millis() {
        assert_eq!(
            decode_column_timestamp(b"2024-05-02T00:00:00Z"),
            Some(ColumnTimestamp::Millis(1_714_608_000_000))
        );
        assert_eq!(
            decode_column_timestamp(b"2024-05-02"),
            Some(ColumnTimestamp::Millis(1_714_608_000_000))
        );
    }

    #[test]
    fn null_or_non_date_value_has_no_timestamp() {
        assert_eq!(decode_column_timestamp(b""), None);
        assert_eq!(decode_column_timestamp(b"yesterday"), None);
    }
}
