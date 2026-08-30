//! Streaming reader over BinaryFILE-framed ITCH, gzipped or plain.
//!
//! A trading day is hundreds of millions of messages, so the whole file is
//! never held in memory. The reader keeps one growable buffer and refills it
//! as messages are consumed.

use crate::{Body, Header, ParseError, decode, message_len};
use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

#[derive(Debug)]
pub enum ReadError {
    Io(io::Error),
    Parse(ParseError),
    /// A length prefix that does not match the message type's known length.
    ///
    /// This is fatal rather than skippable: the framing and the type byte
    /// disagreeing means the stream position is wrong, so every subsequent
    /// message would be garbage.
    FramingMismatch { kind: u8, declared: usize, expected: usize },
    /// The type byte is not one this decoder knows.
    UnknownType { kind: u8, offset: u64 },
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::Io(e) => write!(f, "io error: {e}"),
            ReadError::Parse(e) => write!(f, "parse error: {e:?}"),
            ReadError::FramingMismatch { kind, declared, expected } => write!(
                f,
                "framing mismatch for type {}: length prefix says {declared}, spec says {expected}",
                *kind as char
            ),
            ReadError::UnknownType { kind, offset } => {
                write!(f, "unknown message type {:?} (0x{kind:02x}) at byte {offset}", *kind as char)
            }
        }
    }
}

impl std::error::Error for ReadError {}

impl From<io::Error> for ReadError {
    fn from(e: io::Error) -> Self {
        ReadError::Io(e)
    }
}

pub struct Reader<R: Read> {
    inner: R,
    buf: Vec<u8>,
    /// Bytes of `buf` that hold unconsumed data.
    filled: usize,
    /// Read cursor into the filled region.
    pos: usize,
    /// Absolute byte offset of `pos` in the decompressed stream, for errors.
    offset: u64,
    eof: bool,
}

/// Open an ITCH file, transparently decompressing if it ends in `.gz`.
pub fn open(path: impl AsRef<Path>) -> io::Result<Reader<Box<dyn Read>>> {
    let path = path.as_ref();
    let file = File::open(path)?;
    // 4 MiB read buffer: the file is large and sequential, so bigger reads
    // measurably beat the default 8 KiB.
    let buffered = BufReader::with_capacity(4 << 20, file);
    let stream: Box<dyn Read> = if path.extension().is_some_and(|e| e == "gz") {
        // MultiGz rather than Gz: these files are concatenated members.
        Box::new(MultiGzDecoder::new(buffered))
    } else {
        Box::new(buffered)
    };
    Ok(Reader::new(stream))
}

impl<R: Read> Reader<R> {
    pub fn new(inner: R) -> Self {
        Reader {
            inner,
            buf: vec![0; 1 << 20],
            filled: 0,
            pos: 0,
            offset: 0,
            eof: false,
        }
    }

    /// Ensure at least `need` bytes are available from `pos`, refilling and
    /// compacting as required. Returns false at a clean end of stream.
    fn ensure(&mut self, need: usize) -> io::Result<bool> {
        while self.filled - self.pos < need {
            if self.eof {
                return Ok(false);
            }
            // Compact: move the unconsumed tail to the front.
            if self.pos > 0 {
                self.buf.copy_within(self.pos..self.filled, 0);
                self.filled -= self.pos;
                self.pos = 0;
            }
            if self.filled + need > self.buf.len() {
                self.buf.resize((self.filled + need).next_power_of_two(), 0);
            }
            let n = self.inner.read(&mut self.buf[self.filled..])?;
            if n == 0 {
                self.eof = true;
                return Ok(self.filled - self.pos >= need);
            }
            self.filled += n;
        }
        Ok(true)
    }

    /// Decode the next message, or `None` at end of stream.
    pub fn next_message(&mut self) -> Result<Option<(Header, Body)>, ReadError> {
        // Some files pad between members with zero bytes; a zero-length
        // prefix is the documented end-of-session marker.
        if !self.ensure(2)? {
            return Ok(None);
        }
        let len = u16::from_be_bytes([self.buf[self.pos], self.buf[self.pos + 1]]) as usize;
        if len == 0 {
            return Ok(None);
        }

        if !self.ensure(2 + len)? {
            return Ok(None);
        }
        let start = self.pos + 2;
        let kind = self.buf[start];

        let expected = message_len(kind)
            .ok_or(ReadError::UnknownType { kind, offset: self.offset })?;
        if expected != len {
            return Err(ReadError::FramingMismatch { kind, declared: len, expected });
        }

        let msg = decode(&self.buf[start..start + len]).map_err(ReadError::Parse)?;
        self.pos += 2 + len;
        self.offset += (2 + len) as u64;
        Ok(Some(msg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Side;

    /// Frame a payload the way BinaryFILE does: big-endian u16 length, then
    /// the message.
    fn frame(msg: &[u8]) -> Vec<u8> {
        let mut v = (msg.len() as u16).to_be_bytes().to_vec();
        v.extend_from_slice(msg);
        v
    }

    fn add_order(order_ref: u64, side: u8, shares: u32, price: u32) -> Vec<u8> {
        let mut m = vec![b'A'];
        m.extend_from_slice(&1u16.to_be_bytes());
        m.extend_from_slice(&0u16.to_be_bytes());
        m.extend_from_slice(&[0, 0, 0, 0, 0, 1]);
        m.extend_from_slice(&order_ref.to_be_bytes());
        m.push(side);
        m.extend_from_slice(&shares.to_be_bytes());
        m.extend_from_slice(b"TEST    ");
        m.extend_from_slice(&price.to_be_bytes());
        m
    }

    #[test]
    fn reads_a_sequence_of_messages() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&frame(&add_order(1, b'B', 100, 1000)));
        stream.extend_from_slice(&frame(&add_order(2, b'S', 200, 2000)));

        let mut r = Reader::new(&stream[..]);
        let (_, b1) = r.next_message().unwrap().unwrap();
        let (_, b2) = r.next_message().unwrap().unwrap();
        assert!(r.next_message().unwrap().is_none());

        match (b1, b2) {
            (
                Body::AddOrder { order_ref: 1, side: Side::Buy, shares: 100, .. },
                Body::AddOrder { order_ref: 2, side: Side::Sell, shares: 200, .. },
            ) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    /// The refill path is where a streaming reader usually breaks. Force it
    /// by starting with a buffer far smaller than one message.
    #[test]
    fn survives_refill_and_compaction() {
        let mut stream = Vec::new();
        for i in 0..500u64 {
            stream.extend_from_slice(&frame(&add_order(i, b'B', 100, 1000)));
        }
        let mut r = Reader::new(&stream[..]);
        r.buf = vec![0; 8]; // smaller than a single 36-byte message
        r.filled = 0;
        r.pos = 0;

        let mut seen = 0u64;
        while let Some((_, body)) = r.next_message().unwrap() {
            match body {
                Body::AddOrder { order_ref, .. } => assert_eq!(order_ref, seen),
                other => panic!("unexpected: {other:?}"),
            }
            seen += 1;
        }
        assert_eq!(seen, 500);
    }

    #[test]
    fn zero_length_prefix_ends_the_stream() {
        let mut stream = frame(&add_order(1, b'B', 100, 1000));
        stream.extend_from_slice(&0u16.to_be_bytes());
        stream.extend_from_slice(&frame(&add_order(2, b'B', 100, 1000)));

        let mut r = Reader::new(&stream[..]);
        assert!(r.next_message().unwrap().is_some());
        assert!(r.next_message().unwrap().is_none());
    }

    /// A length prefix disagreeing with the spec means we have lost sync.
    /// Failing loudly here is the whole point.
    #[test]
    fn framing_mismatch_is_fatal() {
        let msg = add_order(1, b'B', 100, 1000);
        let mut bad = 35u16.to_be_bytes().to_vec(); // 'A' is 36
        bad.extend_from_slice(&msg[..35]);

        let mut r = Reader::new(&bad[..]);
        assert!(matches!(
            r.next_message(),
            Err(ReadError::FramingMismatch { kind: b'A', declared: 35, expected: 36 })
        ));
    }

    #[test]
    fn unknown_type_is_fatal() {
        let mut msg = add_order(1, b'B', 100, 1000);
        msg[0] = b'z';
        let stream = frame(&msg);
        let mut r = Reader::new(&stream[..]);
        assert!(matches!(r.next_message(), Err(ReadError::UnknownType { kind: b'z', .. })));
    }
}
