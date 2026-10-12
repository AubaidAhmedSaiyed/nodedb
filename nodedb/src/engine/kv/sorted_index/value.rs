// SPDX-License-Identifier: BUSL-1.1

//! Order-preserving encoding of one sort column value.
//!
//! Every value starts with a type tag, so types order as
//! null < bool < number < string < array/object, whatever their payload.
//!
//! - null: the tag alone;
//! - bool: the tag, then 0 or 1;
//! - number: the tag, then [`number`](super::number) bytes, one order for
//!   integers and floats;
//! - string, array/object: the tag, then the UTF-8 (array/object as JSON
//!   text) with each 0x00 written as 0x00 0xFF, then the terminator 0x00 0x00.
//!
//! No encoded value is a prefix of another. So a composite key can concatenate
//! column values, and complementing one value reverses its order.

use std::borrow::Cow;

use super::number::{self, NUMBER_LEN, SortNumber};

const TAG_NULL: u8 = 0x01;
const TAG_BOOL: u8 = 0x02;
const TAG_NUMBER: u8 = 0x03;
const TAG_TEXT: u8 = 0x04;
const TAG_JSON: u8 = 0x05;

/// The escape that follows a literal 0x00 inside text.
const ESCAPE: u8 = 0xFF;

/// The highest first byte of any encoded value, plain or complemented.
///
/// A range bound byte above it sorts after every continuation of a key.
pub const MAX_LEADING_BYTE: u8 = !TAG_NULL;

/// A sort column value read back from its encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum SortValue<'a> {
    Null,
    Bool(bool),
    Number(SortNumber),
    Text(Cow<'a, str>),
    /// An array or object, as JSON text.
    Json(Cow<'a, str>),
}

/// Encode a document field value.
pub fn encode_json(value: &serde_json::Value) -> Vec<u8> {
    match value {
        serde_json::Value::Null => vec![TAG_NULL],
        serde_json::Value::Bool(b) => vec![TAG_BOOL, u8::from(*b)],
        serde_json::Value::Number(n) => encode_number(number::encode_json_number(n)),
        serde_json::Value::String(s) => encode_text(s),
        other @ (serde_json::Value::Array(_) | serde_json::Value::Object(_)) => {
            terminated(TAG_JSON, other.to_string().as_bytes())
        }
    }
}

/// Encode an integer.
pub fn encode_int(value: i64) -> Vec<u8> {
    encode_number(number::encode_i64(value))
}

/// Encode a float.
pub fn encode_float(value: f64) -> Vec<u8> {
    encode_number(number::encode_f64(value))
}

/// Encode a string.
pub fn encode_text(value: &str) -> Vec<u8> {
    terminated(TAG_TEXT, value.as_bytes())
}

/// The length of the encoded value at the start of `bytes`.
///
/// `flip` is XORed into every byte read: 0x00 for a plain value, 0xFF for a
/// complemented one. `None` when the bytes do not start with a whole value.
pub fn encoded_len(bytes: &[u8], flip: u8) -> Option<usize> {
    let len = match *bytes.first()? ^ flip {
        TAG_NULL => 1,
        TAG_BOOL => 2,
        TAG_NUMBER => 1 + NUMBER_LEN,
        TAG_TEXT | TAG_JSON => 1 + terminated_len(bytes.get(1..)?, flip)?,
        _ => return None,
    };
    (len <= bytes.len()).then_some(len)
}

/// Decode one whole encoded value. `None` for bytes no value encodes to.
pub fn decode(bytes: &[u8]) -> Option<SortValue<'_>> {
    if encoded_len(bytes, 0)? != bytes.len() {
        return None;
    }
    let (&tag, payload) = bytes.split_first()?;
    match tag {
        TAG_NULL => Some(SortValue::Null),
        TAG_BOOL => match payload {
            [0] => Some(SortValue::Bool(false)),
            [1] => Some(SortValue::Bool(true)),
            _ => None,
        },
        TAG_NUMBER => number::decode(payload.try_into().ok()?).map(SortValue::Number),
        TAG_TEXT => unescape(payload).map(SortValue::Text),
        TAG_JSON => unescape(payload).map(SortValue::Json),
        _ => None,
    }
}

fn encode_number(bytes: [u8; NUMBER_LEN]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + NUMBER_LEN);
    out.push(TAG_NUMBER);
    out.extend_from_slice(&bytes);
    out
}

fn terminated(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 3);
    out.push(tag);
    for &byte in payload {
        out.push(byte);
        if byte == 0 {
            out.push(ESCAPE);
        }
    }
    out.extend_from_slice(&[0, 0]);
    out
}

/// The length of an escaped payload, terminator included.
fn terminated_len(bytes: &[u8], flip: u8) -> Option<usize> {
    let mut at = 0;
    loop {
        if *bytes.get(at)? ^ flip != 0 {
            at += 1;
            continue;
        }
        match *bytes.get(at + 1)? ^ flip {
            0 => return Some(at + 2),
            ESCAPE => at += 2,
            _ => return None,
        }
    }
}

/// The UTF-8 text of an escaped, terminated payload.
fn unescape(payload: &[u8]) -> Option<Cow<'_, str>> {
    let body = payload.strip_suffix(&[0u8, 0])?;
    if !body.contains(&0) {
        return std::str::from_utf8(body).ok().map(Cow::Borrowed);
    }
    let mut out = Vec::with_capacity(body.len());
    let mut bytes = body.iter();
    while let Some(&byte) = bytes.next() {
        out.push(byte);
        if byte == 0 && bytes.next() != Some(&ESCAPE) {
            return None;
        }
    }
    String::from_utf8(out).ok().map(Cow::Owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn complement(bytes: &[u8]) -> Vec<u8> {
        bytes.iter().map(|b| !b).collect()
    }

    /// Values in ascending sort order.
    fn ascending() -> Vec<serde_json::Value> {
        vec![
            json!(null),
            json!(false),
            json!(true),
            json!(i64::MIN),
            json!(-10.5),
            json!(-1),
            json!(0),
            json!(10),
            json!(10.5),
            json!(11),
            json!(11.5),
            json!(i64::MAX),
            json!(u64::MAX),
            json!(1e300),
            json!(""),
            json!("\u{0}"),
            json!("\u{0}a"),
            json!("a"),
            json!("a\u{0}"),
            json!("a\u{0}\u{0}"),
            json!("a\u{1}"),
            json!("aa"),
            json!("b"),
            json!([1]),
            json!({"k": 1}),
        ]
    }

    #[test]
    fn byte_order_matches_sort_order() {
        let encoded: Vec<Vec<u8>> = ascending().iter().map(encode_json).collect();
        for pair in encoded.windows(2) {
            assert!(
                pair[0] < pair[1],
                "{:?} must sort below {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn complemented_byte_order_is_reversed() {
        let encoded: Vec<Vec<u8>> = ascending().iter().map(encode_json).collect();
        for pair in encoded.windows(2) {
            assert!(complement(&pair[0]) > complement(&pair[1]));
        }
    }

    #[test]
    fn no_value_is_a_prefix_of_another() {
        let encoded: Vec<Vec<u8>> = ascending().iter().map(encode_json).collect();
        for a in &encoded {
            for b in &encoded {
                if a != b {
                    assert!(!b.starts_with(a), "{a:?} is a prefix of {b:?}");
                }
            }
        }
    }

    #[test]
    fn encoded_len_finds_the_end_of_each_value_plain_and_complemented() {
        for value in ascending() {
            let encoded = encode_json(&value);
            let mut followed = encoded.clone();
            followed.extend_from_slice(&encode_text("next"));
            assert_eq!(encoded_len(&followed, 0), Some(encoded.len()), "{value}");

            let mut flipped = complement(&encoded);
            flipped.extend_from_slice(&encode_int(7));
            assert_eq!(encoded_len(&flipped, 0xFF), Some(encoded.len()), "{value}");
        }
    }

    #[test]
    fn leading_bytes_stay_at_or_below_the_bound() {
        for value in ascending() {
            let encoded = encode_json(&value);
            assert!(encoded[0] <= MAX_LEADING_BYTE);
            assert!(!encoded[0] <= MAX_LEADING_BYTE);
        }
    }

    #[test]
    fn decode_reads_back_each_type() {
        assert_eq!(decode(&encode_json(&json!(null))), Some(SortValue::Null));
        assert_eq!(
            decode(&encode_json(&json!(true))),
            Some(SortValue::Bool(true))
        );
        assert_eq!(
            decode(&encode_json(&json!(11))),
            Some(SortValue::Number(SortNumber::Whole(11)))
        );
        assert_eq!(
            decode(&encode_json(&json!(10.5))),
            Some(SortValue::Number(SortNumber::Float(10.5)))
        );
        assert_eq!(
            decode(&encode_text("a\u{0}b")),
            Some(SortValue::Text(Cow::Owned("a\u{0}b".into())))
        );
        assert_eq!(
            decode(&encode_json(&json!([1]))),
            Some(SortValue::Json(Cow::Borrowed("[1]")))
        );
    }

    #[test]
    fn decode_refuses_truncated_or_trailing_bytes() {
        let text = encode_text("abc");
        assert_eq!(decode(&text[..text.len() - 1]), None);
        let mut trailing = encode_int(1);
        trailing.push(0);
        assert_eq!(decode(&trailing), None);
        assert_eq!(decode(b""), None);
        assert_eq!(decode(&[0x7F]), None);
    }
}
