//! Nasdaq TotalView-ITCH 5.0 decoder.
//!
//! Files published at <https://emi.nasdaq.com/ITCH/> are in BinaryFILE
//! framing: each message is preceded by a big-endian `u16` length, and the
//! first byte of the payload is the message type.
//!
//! Decoding is a pure function from a byte slice to a [`Body`]. No I/O and no
//! allocation happen here, which keeps the hot path clean and makes every
//! message type testable from a literal byte array.
//!
//! All prices are fixed-point with four implied decimals, kept as raw `u32`
//! ticks of $0.0001. Converting to floating point in the decoder would throw
//! away exactness for no benefit; the analysis layer converts once, at the end.
//!
//! Timestamps are nanoseconds since midnight Eastern, packed into six bytes.

pub mod reader;

/// A symbol, right-padded with spaces in the wire format.
pub type Symbol = [u8; 8];

/// Trim the wire format's trailing spaces and render as UTF-8.
pub fn symbol_str(s: &Symbol) -> &str {
    let end = s.iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);
    std::str::from_utf8(&s[..end]).unwrap_or("")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// Payload shorter than the message type requires.
    Truncated { kind: u8, need: usize, got: usize },
    /// A side byte that was neither `B` nor `S`.
    BadSide(u8),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Truncated { kind, need, got } => write!(
                f,
                "truncated {} message: need {need} bytes, got {got}",
                *kind as char
            ),
            ParseError::BadSide(b) => write!(f, "invalid side byte {:?}", *b as char),
        }
    }
}

impl std::error::Error for ParseError {}

/// Header fields shared by every ITCH message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Venue-assigned symbol id. Order messages carry only this, not the
    /// symbol, so a locate-to-symbol map must be built from `StockDirectory`.
    pub stock_locate: u16,
    pub tracking_number: u16,
    /// Nanoseconds since midnight.
    pub timestamp: u64,
}

/// The decoded body of a message.
///
/// Message types that exist in the protocol but never affect the book or the
/// analysis are collapsed into [`Body::Other`] rather than given variants; the
/// framing still validates their length, so nothing is silently skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Body {
    SystemEvent {
        event_code: u8,
    },
    StockDirectory {
        stock: Symbol,
        round_lot_size: u32,
    },
    TradingAction {
        stock: Symbol,
        trading_state: u8,
    },
    AddOrder {
        order_ref: u64,
        side: Side,
        shares: u32,
        stock: Symbol,
        price: u32,
        /// `true` when the message carried an MPID attribution (type `F`).
        attributed: bool,
    },
    /// Execution against a resting order at that order's price.
    OrderExecuted {
        order_ref: u64,
        executed_shares: u32,
        match_number: u64,
    },
    /// Execution against a resting order at a price other than its own.
    ///
    /// `printable == false` means the execution must not be counted as a
    /// trade print, though it still removes shares from the book.
    OrderExecutedWithPrice {
        order_ref: u64,
        executed_shares: u32,
        match_number: u64,
        printable: bool,
        execution_price: u32,
    },
    /// Partial cancel: `cancelled_shares` leave the order, which remains.
    OrderCancel {
        order_ref: u64,
        cancelled_shares: u32,
    },
    /// Full removal of the remaining quantity.
    OrderDelete {
        order_ref: u64,
    },
    /// Delete plus add under a new id. Queue priority is lost, which is why
    /// this cannot be modelled as an in-place amend.
    OrderReplace {
        original_order_ref: u64,
        new_order_ref: u64,
        shares: u32,
        price: u32,
    },
    /// A trade against non-displayed liquidity.
    ///
    /// Critically, this must **not** be applied to the visible book: the
    /// shares were never in it. Applying `P` messages to the book is a
    /// classic reconstruction bug that inflates depth consumption.
    TradeNonCross {
        order_ref: u64,
        side: Side,
        shares: u32,
        stock: Symbol,
        price: u32,
        match_number: u64,
    },
    /// Opening, closing, or halt cross. Also not applied to the visible book.
    CrossTrade {
        shares: u64,
        stock: Symbol,
        cross_price: u32,
        match_number: u64,
        cross_type: u8,
    },
    /// A previously reported trade was busted. Trades are cumulative, so
    /// downstream consumers must be able to retract by `match_number`.
    BrokenTrade {
        match_number: u64,
    },
    /// A known message type with no effect on the book or the study.
    Other {
        kind: u8,
    },
}

/// Coarse classification of a message, for counting and reporting.
///
/// Lives here rather than in the consumer so that adding a [`Body`] variant
/// forces the classification to be updated in the same place, and so that a
/// caller can tally messages with an array index instead of a second match
/// over the same value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum MessageKind {
    Add = 0,
    Executed,
    Cancel,
    Delete,
    Replace,
    HiddenTrade,
    Cross,
    Administrative,
}

impl MessageKind {
    pub const COUNT: usize = 8;

    pub const ALL: [MessageKind; Self::COUNT] = [
        MessageKind::Add,
        MessageKind::Executed,
        MessageKind::Cancel,
        MessageKind::Delete,
        MessageKind::Replace,
        MessageKind::HiddenTrade,
        MessageKind::Cross,
        MessageKind::Administrative,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            MessageKind::Add => "add",
            MessageKind::Executed => "executed",
            MessageKind::Cancel => "cancel",
            MessageKind::Delete => "delete",
            MessageKind::Replace => "replace",
            MessageKind::HiddenTrade => "hidden trade",
            MessageKind::Cross => "cross",
            MessageKind::Administrative => "administrative",
        }
    }
}

impl Body {
    pub const fn kind(&self) -> MessageKind {
        match self {
            Body::AddOrder { .. } => MessageKind::Add,
            Body::OrderExecuted { .. } | Body::OrderExecutedWithPrice { .. } => {
                MessageKind::Executed
            }
            Body::OrderCancel { .. } => MessageKind::Cancel,
            Body::OrderDelete { .. } => MessageKind::Delete,
            Body::OrderReplace { .. } => MessageKind::Replace,
            Body::TradeNonCross { .. } => MessageKind::HiddenTrade,
            Body::CrossTrade { .. } => MessageKind::Cross,
            _ => MessageKind::Administrative,
        }
    }
}

/// Wire length of each message type, including the type byte.
///
/// Returns `None` for an unrecognised type, which is treated as a hard error
/// by the reader: an unknown type means the stream is desynchronised, and
/// guessing a length would silently corrupt everything after it.
pub const fn message_len(kind: u8) -> Option<usize> {
    Some(match kind {
        b'S' => 12,
        b'R' => 39,
        b'H' => 25,
        b'Y' => 20,
        b'L' => 26,
        b'V' => 35,
        b'W' => 12,
        b'K' => 28,
        b'J' => 35,
        b'h' => 21,
        b'A' => 36,
        b'F' => 40,
        b'E' => 31,
        b'C' => 36,
        b'X' => 23,
        b'D' => 19,
        b'U' => 35,
        b'P' => 44,
        b'Q' => 40,
        b'B' => 19,
        b'I' => 50,
        // Retail Price Improvement Indicator.
        b'N' => 20,
        // Direct Listing with Capital Raise price discovery.
        b'O' => 48,
        _ => return None,
    })
}

#[inline(always)]
fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_be_bytes([b[o], b[o + 1]])
}

#[inline(always)]
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[inline(always)]
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_be_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}

/// Six-byte big-endian timestamp.
#[inline(always)]
fn u48_at(b: &[u8], o: usize) -> u64 {
    u64::from_be_bytes([0, 0, b[o], b[o + 1], b[o + 2], b[o + 3], b[o + 4], b[o + 5]])
}

#[inline(always)]
fn sym_at(b: &[u8], o: usize) -> Symbol {
    let mut s = [0u8; 8];
    s.copy_from_slice(&b[o..o + 8]);
    s
}

#[inline(always)]
fn side_at(b: &[u8], o: usize) -> Result<Side, ParseError> {
    match b[o] {
        b'B' => Ok(Side::Buy),
        b'S' => Ok(Side::Sell),
        other => Err(ParseError::BadSide(other)),
    }
}

/// Decode one message payload, including its leading type byte.
///
/// `buf` must be exactly the message: the reader is responsible for framing.
pub fn decode(buf: &[u8]) -> Result<(Header, Body), ParseError> {
    if buf.is_empty() {
        return Err(ParseError::Truncated {
            kind: 0,
            need: 1,
            got: 0,
        });
    }
    let kind = buf[0];
    let need = message_len(kind).unwrap_or(0);
    if need == 0 || buf.len() < need {
        return Err(ParseError::Truncated {
            kind,
            need,
            got: buf.len(),
        });
    }

    let header = Header {
        stock_locate: u16_at(buf, 1),
        tracking_number: u16_at(buf, 3),
        timestamp: u48_at(buf, 5),
    };

    // Every body begins at offset 11, after the shared header.
    let body = match kind {
        b'S' => Body::SystemEvent {
            event_code: buf[11],
        },
        b'R' => Body::StockDirectory {
            stock: sym_at(buf, 11),
            round_lot_size: u32_at(buf, 21),
        },
        b'H' => Body::TradingAction {
            stock: sym_at(buf, 11),
            trading_state: buf[19],
        },
        b'A' => Body::AddOrder {
            order_ref: u64_at(buf, 11),
            side: side_at(buf, 19)?,
            shares: u32_at(buf, 20),
            stock: sym_at(buf, 24),
            price: u32_at(buf, 32),
            attributed: false,
        },
        b'F' => Body::AddOrder {
            order_ref: u64_at(buf, 11),
            side: side_at(buf, 19)?,
            shares: u32_at(buf, 20),
            stock: sym_at(buf, 24),
            price: u32_at(buf, 32),
            attributed: true,
        },
        b'E' => Body::OrderExecuted {
            order_ref: u64_at(buf, 11),
            executed_shares: u32_at(buf, 19),
            match_number: u64_at(buf, 23),
        },
        b'C' => Body::OrderExecutedWithPrice {
            order_ref: u64_at(buf, 11),
            executed_shares: u32_at(buf, 19),
            match_number: u64_at(buf, 23),
            printable: buf[31] == b'Y',
            execution_price: u32_at(buf, 32),
        },
        b'X' => Body::OrderCancel {
            order_ref: u64_at(buf, 11),
            cancelled_shares: u32_at(buf, 19),
        },
        b'D' => Body::OrderDelete {
            order_ref: u64_at(buf, 11),
        },
        b'U' => Body::OrderReplace {
            original_order_ref: u64_at(buf, 11),
            new_order_ref: u64_at(buf, 19),
            shares: u32_at(buf, 27),
            price: u32_at(buf, 31),
        },
        b'P' => Body::TradeNonCross {
            order_ref: u64_at(buf, 11),
            side: side_at(buf, 19)?,
            shares: u32_at(buf, 20),
            stock: sym_at(buf, 24),
            price: u32_at(buf, 32),
            match_number: u64_at(buf, 36),
        },
        b'Q' => Body::CrossTrade {
            shares: u64_at(buf, 11),
            stock: sym_at(buf, 19),
            cross_price: u32_at(buf, 27),
            match_number: u64_at(buf, 31),
            cross_type: buf[39],
        },
        b'B' => Body::BrokenTrade {
            match_number: u64_at(buf, 11),
        },
        other => Body::Other { kind: other },
    };

    Ok((header, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a message with the shared header pre-filled.
    fn framed(kind: u8, body: &[u8]) -> Vec<u8> {
        let mut v = vec![kind];
        v.extend_from_slice(&1u16.to_be_bytes()); // stock_locate
        v.extend_from_slice(&2u16.to_be_bytes()); // tracking_number
        v.extend_from_slice(&[0, 0, 0, 0, 0x12, 0x34]); // 6-byte timestamp
        v.extend_from_slice(body);
        v
    }

    #[test]
    fn header_is_parsed_from_every_type() {
        let m = framed(b'D', &7u64.to_be_bytes());
        let (h, b) = decode(&m).unwrap();
        assert_eq!(h.stock_locate, 1);
        assert_eq!(h.tracking_number, 2);
        assert_eq!(h.timestamp, 0x1234);
        assert_eq!(b, Body::OrderDelete { order_ref: 7 });
    }

    #[test]
    fn add_order_fields_land_at_the_right_offsets() {
        let mut body = Vec::new();
        body.extend_from_slice(&42u64.to_be_bytes()); // order_ref
        body.push(b'B'); // side
        body.extend_from_slice(&100u32.to_be_bytes()); // shares
        body.extend_from_slice(b"AAPL    "); // stock
        body.extend_from_slice(&1_234_500u32.to_be_bytes()); // price = $123.45
        let m = framed(b'A', &body);
        assert_eq!(m.len(), message_len(b'A').unwrap());

        let (_, b) = decode(&m).unwrap();
        match b {
            Body::AddOrder {
                order_ref,
                side,
                shares,
                stock,
                price,
                attributed,
            } => {
                assert_eq!(order_ref, 42);
                assert_eq!(side, Side::Buy);
                assert_eq!(shares, 100);
                assert_eq!(symbol_str(&stock), "AAPL");
                assert_eq!(price, 1_234_500);
                assert!(!attributed);
            }
            other => panic!("expected AddOrder, got {other:?}"),
        }
    }

    /// `F` is `A` plus a four-byte attribution, and must decode identically
    /// apart from the flag. Getting this wrong shifts every field.
    #[test]
    fn attributed_add_matches_plain_add() {
        let mut body = Vec::new();
        body.extend_from_slice(&42u64.to_be_bytes());
        body.push(b'S');
        body.extend_from_slice(&300u32.to_be_bytes());
        body.extend_from_slice(b"MSFT    ");
        body.extend_from_slice(&9_999u32.to_be_bytes());

        let plain = framed(b'A', &body);
        let mut attr_body = body.clone();
        attr_body.extend_from_slice(b"MPID");
        let attributed = framed(b'F', &attr_body);
        assert_eq!(attributed.len(), message_len(b'F').unwrap());

        let (_, a) = decode(&plain).unwrap();
        let (_, f) = decode(&attributed).unwrap();
        match (a, f) {
            (
                Body::AddOrder {
                    order_ref: r1,
                    side: s1,
                    shares: q1,
                    stock: k1,
                    price: p1,
                    ..
                },
                Body::AddOrder {
                    order_ref: r2,
                    side: s2,
                    shares: q2,
                    stock: k2,
                    price: p2,
                    attributed,
                },
            ) => {
                assert_eq!((r1, s1, q1, k1, p1), (r2, s2, q2, k2, p2));
                assert!(attributed);
            }
            other => panic!("expected two AddOrders, got {other:?}"),
        }
    }

    #[test]
    fn executed_with_price_reads_the_printable_flag() {
        let mut body = Vec::new();
        body.extend_from_slice(&5u64.to_be_bytes());
        body.extend_from_slice(&50u32.to_be_bytes());
        body.extend_from_slice(&77u64.to_be_bytes());
        body.push(b'N'); // non-printable
        body.extend_from_slice(&1_000_000u32.to_be_bytes());
        let m = framed(b'C', &body);
        assert_eq!(m.len(), message_len(b'C').unwrap());

        let (_, b) = decode(&m).unwrap();
        match b {
            Body::OrderExecutedWithPrice {
                printable,
                execution_price,
                ..
            } => {
                assert!(!printable);
                assert_eq!(execution_price, 1_000_000);
            }
            other => panic!("expected OrderExecutedWithPrice, got {other:?}"),
        }
    }

    #[test]
    fn replace_carries_both_order_refs() {
        let mut body = Vec::new();
        body.extend_from_slice(&11u64.to_be_bytes());
        body.extend_from_slice(&22u64.to_be_bytes());
        body.extend_from_slice(&500u32.to_be_bytes());
        body.extend_from_slice(&2_000_000u32.to_be_bytes());
        let m = framed(b'U', &body);
        assert_eq!(m.len(), message_len(b'U').unwrap());

        let (_, b) = decode(&m).unwrap();
        assert_eq!(
            b,
            Body::OrderReplace {
                original_order_ref: 11,
                new_order_ref: 22,
                shares: 500,
                price: 2_000_000
            }
        );
    }

    #[test]
    fn truncated_input_is_an_error_not_a_panic() {
        let m = framed(b'A', &[0u8; 4]);
        assert!(matches!(
            decode(&m),
            Err(ParseError::Truncated { kind: b'A', .. })
        ));
    }

    #[test]
    fn unknown_type_is_rejected() {
        assert_eq!(message_len(b'z'), None);
        assert!(decode(&[b'z', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn symbols_drop_padding_only_on_the_right() {
        assert_eq!(symbol_str(b"AAPL    "), "AAPL");
        assert_eq!(symbol_str(b"BRK.A   "), "BRK.A");
        assert_eq!(symbol_str(b"        "), "");
    }
}
