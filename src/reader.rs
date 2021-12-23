use std::io;

use byteorder::ReadBytesExt;

pub struct BitReader<'a> {
    inner: &'a mut dyn std::io::Read,
    bits_cache: u8,
    bits_remaining: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(inner: &'a mut dyn std::io::Read) -> Self {
        Self {
            inner,
            bits_cache: 0,
            bits_remaining: 0,
        }
    }

    pub fn read_bit(&mut self) -> Result<u8, std::io::Error> {
        if std::intrinsics::likely(self.bits_remaining != 0) {
            let result: u8 = self.bits_cache >> 7;
            self.bits_cache <<= 1;
            self.bits_remaining <<= 1;
            return Ok(result);
        }

        self.bits_cache = self.inner.read_u8()?;
        if std::intrinsics::unlikely(self.bits_cache == 0xff) {
            self.bits_cache = self.inner.read_u8()?;
            if std::intrinsics::unlikely(self.bits_cache != 0) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Expected non-zero following marker",
                ));
            }
            self.bits_cache = 0xff;
        }
        self.bits_remaining = 0xff;

        let result: u8 = self.bits_cache >> 7;
        self.bits_cache <<= 1;
        self.bits_remaining <<= 1;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self};

    #[test]
    fn test_read_bits() {
        let mut cursor = io::Cursor::new(vec![0xAA, 0xAF, 0x01, 0x40]);
        let mut reader = BitReader::new(&mut cursor);

        // read 12 bits
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
    }

    #[test]
    fn test_escape_marker() {
        let mut cursor = io::Cursor::new(vec![0xFF, 0x00]);
        let mut reader = BitReader::new(&mut cursor);

        for _ in 0..8 {
            assert_eq!(reader.read_bit().unwrap(), 1);
        }
        assert_eq!(reader.read_bit().unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn test_eof() {
        let mut cursor = io::Cursor::new(vec![]);
        let mut reader = BitReader::new(&mut cursor);

        assert_eq!(reader.read_bit().unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[bench]
    fn read_bits_bench(b: &mut test::Bencher) {
        b.iter(|| {
            let mut cursor = io::Cursor::new(vec![
                0xAA, 0xAF, 0xFF, 0x00, 0x01, 0x40, 0xAA, 0xAF, 0xFF, 0x00, 0x01, 0x4, 0xAA, 0xAF, 0xFF, 0x00, 0x01,
                0x4, 0xAA, 0xAF, 0xFF, 0x00, 0x01, 0x4,
            ]);
            let mut reader = BitReader::new(&mut cursor);

            for _ in 0..160 {
                reader.read_bit().expect("Unexpected error reading from cursor");
            }
        });
    }
}
