// SPDX-License-Identifier: BUSL-1.1

//! Exact order-preserving encoding for the numbers of a sort column.
//!
//! Integers and floats share one encoding, ordered by numeric value. A number
//! encodes as two parts, compared in order:
//!
//! - the greatest f64 at or below the value, as 8 order-preserving bytes;
//! - the integer residual above that f64, as 2 big-endian bytes.
//!
//! A float is its own floor, so its residual is 0. An integer whose magnitude
//! exceeds 2^53 can fall between two f64 values. Its floor is the f64 below it
//! and the residual holds the exact remainder. No cast rounds a value.
//!
//! Two values with different floors order by floor: the value with the lower
//! floor is below the next f64, which is at or below the other value. Two
//! values with one floor order by residual. So byte order is numeric order.
//!
//! Equal numbers encode equal: 11 and 11.0 give the same bytes, and so do
//! -0.0 and +0.0. Every NaN encodes as one value above +infinity.

/// Bytes in an encoded number.
pub const NUMBER_LEN: usize = 10;

/// Top bit of an f64 bit pattern.
const SIGN_BIT: u64 = 1 << 63;

/// 2^64. Every integer this module encodes has a magnitude at or below it.
const WHOLE_LIMIT: f64 = 18_446_744_073_709_551_616.0;

/// A number read back from its encoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SortNumber {
    /// A whole number with a magnitude at or below 2^64, from an integer or a float.
    Whole(i128),
    /// Any other float: a fraction, a whole number past 2^64, an infinity, or NaN.
    Float(f64),
}

/// Encode a signed integer.
pub fn encode_i64(value: i64) -> [u8; NUMBER_LEN] {
    encode_integer(i128::from(value))
}

/// Encode an unsigned integer.
pub fn encode_u64(value: u64) -> [u8; NUMBER_LEN] {
    encode_integer(i128::from(value))
}

/// Encode a float.
pub fn encode_f64(value: f64) -> [u8; NUMBER_LEN] {
    pack(value, 0)
}

/// Encode a JSON number by its exact value.
pub fn encode_json_number(value: &serde_json::Number) -> [u8; NUMBER_LEN] {
    match (value.as_i64(), value.as_u64()) {
        (Some(i), _) => encode_i64(i),
        (None, Some(u)) => encode_u64(u),
        // A number that is neither i64 nor u64 is an f64, so `as_f64` is `Some`.
        (None, None) => encode_f64(value.as_f64().unwrap_or(f64::NAN)),
    }
}

/// Decode bytes written by this module. `None` for bytes no value encodes to.
pub fn decode(bytes: &[u8; NUMBER_LEN]) -> Option<SortNumber> {
    let (floor_bytes, residual_bytes) = bytes.split_first_chunk::<8>()?;
    let residual_bytes: &[u8; 2] = residual_bytes.try_into().ok()?;
    let floor = f64_from_order_bits(u64::from_be_bytes(*floor_bytes));
    let residual = u16::from_be_bytes(*residual_bytes);

    if floor.fract() == 0.0 && floor.abs() <= WHOLE_LIMIT {
        // The guard bounds `floor` to an exact i128, so the cast is exact.
        return Some(SortNumber::Whole(floor as i128 + i128::from(residual)));
    }
    (residual == 0).then_some(SortNumber::Float(floor))
}

/// Encode an integer with a magnitude at or below 2^64.
fn encode_integer(value: i128) -> [u8; NUMBER_LEN] {
    let floor = floor_f64(value);
    // `floor` is whole and at most 2^64 in magnitude, so the cast is exact.
    let residual = value - floor as i128;
    // The residual is below the spacing of f64 values at `floor`. That spacing
    // is at most 2^11 for magnitudes up to 2^64, so it fits in a u16.
    pack(floor, residual as u16)
}

/// The greatest f64 at or below `value`.
fn floor_f64(value: i128) -> f64 {
    // The conversion rounds to nearest, so it can land one f64 above `value`.
    let nearest = value as f64;
    if nearest as i128 > value {
        nearest.next_down()
    } else {
        nearest
    }
}

fn pack(floor: f64, residual: u16) -> [u8; NUMBER_LEN] {
    let mut out = [0u8; NUMBER_LEN];
    out[..8].copy_from_slice(&order_bits(canonical(floor)).to_be_bytes());
    out[8..].copy_from_slice(&residual.to_be_bytes());
    out
}

/// One bit pattern per numeric value: a single NaN, and +0.0 for both zeros.
fn canonical(value: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else if value == 0.0 {
        0.0
    } else {
        value
    }
}

/// Map f64 bits to u64 so unsigned order matches numeric order.
fn order_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & SIGN_BIT == 0 {
        bits ^ SIGN_BIT
    } else {
        !bits
    }
}

fn f64_from_order_bits(encoded: u64) -> f64 {
    if encoded & SIGN_BIT != 0 {
        f64::from_bits(encoded ^ SIGN_BIT)
    } else {
        f64::from_bits(!encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A number of either kind, compared by exact numeric value.
    #[derive(Debug, Clone, Copy)]
    enum Num {
        I(i64),
        U(u64),
        F(f64),
    }

    impl Num {
        fn encode(self) -> [u8; NUMBER_LEN] {
            match self {
                Num::I(v) => encode_i64(v),
                Num::U(v) => encode_u64(v),
                Num::F(v) => encode_f64(v),
            }
        }
    }

    /// Exact numeric comparison of an integer with a finite float.
    fn cmp_int_float(int: i128, float: f64) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        if float >= 2f64.powi(100) {
            return Ordering::Less;
        }
        if float <= -2f64.powi(100) {
            return Ordering::Greater;
        }
        let whole = float.floor();
        let whole_int = whole as i128;
        match int.cmp(&whole_int) {
            Ordering::Equal if float > whole => Ordering::Less,
            other => other,
        }
    }

    fn numeric_cmp(a: Num, b: Num) -> std::cmp::Ordering {
        let as_int = |n: Num| match n {
            Num::I(v) => Some(i128::from(v)),
            Num::U(v) => Some(i128::from(v)),
            Num::F(_) => None,
        };
        match (a, b) {
            (Num::F(x), Num::F(y)) => x.partial_cmp(&y).unwrap_or_else(|| {
                // NaN sorts above every number and equals itself.
                x.is_nan().cmp(&y.is_nan())
            }),
            (Num::F(x), other) => {
                let int = as_int(other).unwrap_or_default();
                if x.is_nan() {
                    std::cmp::Ordering::Greater
                } else if x.is_infinite() {
                    if x > 0.0 {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Less
                    }
                } else {
                    cmp_int_float(int, x).reverse()
                }
            }
            (other, Num::F(_)) => numeric_cmp(b, other).reverse(),
            (x, y) => as_int(x).cmp(&as_int(y)),
        }
    }

    /// Edge values: around ±2^53, ±2^63, 2^64, fractions between large
    /// consecutive integers, zeros, infinities and NaN.
    fn edge_values() -> Vec<Num> {
        let p53 = 1i64 << 53;
        let mut out = vec![
            Num::I(i64::MIN),
            Num::I(i64::MIN + 1),
            Num::I(i64::MAX),
            Num::I(i64::MAX - 1),
            Num::U(u64::MAX),
            Num::U(u64::MAX - 1),
            Num::U(1 << 63),
            Num::U((1 << 63) + 1),
            Num::I(0),
            Num::I(1),
            Num::I(-1),
            Num::I(10),
            Num::I(11),
            Num::F(10.5),
            Num::F(11.0),
            Num::F(-10.5),
            Num::F(0.0),
            Num::F(-0.0),
            Num::F(0.5),
            Num::F(-0.5),
            Num::F(f64::MIN_POSITIVE),
            Num::F(-f64::MIN_POSITIVE),
            Num::F(f64::MAX),
            Num::F(f64::MIN),
            Num::F(f64::INFINITY),
            Num::F(f64::NEG_INFINITY),
            Num::F(f64::NAN),
            Num::F(-f64::NAN),
            Num::F(9_223_372_036_854_775_808.0),
            Num::F(-9_223_372_036_854_775_808.0),
            Num::F(18_446_744_073_709_551_616.0),
            Num::F(9_223_372_036_854_774_784.0),
        ];
        for base in [p53, -p53, 1i64 << 62, -(1i64 << 62)] {
            for delta in -3i64..=3 {
                out.push(Num::I(base + delta));
                out.push(Num::F((base + delta) as f64));
            }
            // A float strictly between two consecutive integers.
            out.push(Num::F(base as f64 / 4.0 + 0.5));
        }
        for v in [p53 - 1, p53 + 1, p53 + 3] {
            out.push(Num::F(v as f64 - 0.5));
        }
        out
    }

    #[test]
    fn byte_order_matches_numeric_order_for_every_pair_of_edge_values() {
        let values = edge_values();
        for &a in &values {
            for &b in &values {
                assert_eq!(
                    a.encode().cmp(&b.encode()),
                    numeric_cmp(a, b),
                    "{a:?} vs {b:?}"
                );
            }
        }
    }

    #[test]
    fn eleven_sorts_above_ten_and_a_half() {
        assert!(encode_f64(10.5) < encode_i64(11));
        assert!(encode_i64(10) < encode_f64(10.5));
    }

    #[test]
    fn equal_numbers_encode_equal() {
        assert_eq!(encode_i64(11), encode_f64(11.0));
        assert_eq!(encode_u64(11), encode_i64(11));
        assert_eq!(encode_f64(-0.0), encode_f64(0.0));
        assert_eq!(encode_f64(-0.0), encode_i64(0));
        assert_eq!(encode_f64(f64::NAN), encode_f64(-f64::NAN));
    }

    #[test]
    fn integers_past_two_pow_53_stay_distinct() {
        let p53 = 1i64 << 53;
        assert!(encode_i64(p53) < encode_i64(p53 + 1));
        assert!(encode_i64(p53 + 1) < encode_f64((p53 + 2) as f64));
        assert!(encode_i64(i64::MAX - 1) < encode_i64(i64::MAX));
        assert!(encode_i64(i64::MAX) < encode_f64(9_223_372_036_854_775_808.0));
        assert!(encode_f64(-9_223_372_036_854_775_808.0) == encode_i64(i64::MIN));
    }

    #[test]
    fn nan_sorts_above_infinity() {
        assert!(encode_f64(f64::INFINITY) < encode_f64(f64::NAN));
        assert!(encode_u64(u64::MAX) < encode_f64(f64::INFINITY));
    }

    #[test]
    fn decode_returns_the_encoded_value() {
        for v in [i64::MIN, -(1 << 53) - 1, -1, 0, 1, (1 << 53) + 1, i64::MAX] {
            assert_eq!(
                decode(&encode_i64(v)),
                Some(SortNumber::Whole(i128::from(v)))
            );
        }
        assert_eq!(
            decode(&encode_u64(u64::MAX)),
            Some(SortNumber::Whole(i128::from(u64::MAX)))
        );
        assert_eq!(decode(&encode_f64(11.0)), Some(SortNumber::Whole(11)));
        assert_eq!(decode(&encode_f64(-0.0)), Some(SortNumber::Whole(0)));
        for v in [10.5, -10.5, f64::INFINITY, f64::NEG_INFINITY, 1e30] {
            assert_eq!(decode(&encode_f64(v)), Some(SortNumber::Float(v)));
        }
        assert!(matches!(
            decode(&encode_f64(f64::NAN)),
            Some(SortNumber::Float(v)) if v.is_nan()
        ));
    }

    #[test]
    fn a_residual_on_a_fraction_is_not_a_value() {
        let mut bytes = encode_f64(10.5);
        bytes[NUMBER_LEN - 1] = 1;
        assert_eq!(decode(&bytes), None);
    }
}
