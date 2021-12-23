use byteorder::{NetworkEndian, ReadBytesExt};
use std::collections::HashMap;

pub struct HuffmanTable {
    min_code: [i16; 16],
    max_code: [i16; 16],
    value_ptrs: [i16; 16],
    values: Vec<u8>,
}

pub struct HuffmanTables {
    dc_tables: HashMap<u8, HuffmanTable>,
    ac_tables: HashMap<u8, HuffmanTable>,
}

impl HuffmanTables {
    pub fn new() -> Self {
        Self {
            dc_tables: HashMap::new(),
            ac_tables: HashMap::new(),
        }
    }

    pub fn find_dc_table(&self, id: u8) -> Option<&HuffmanTable> {
        self.dc_tables.get(&id)
    }

    pub fn find_ac_table(&self, id: u8) -> Option<&HuffmanTable> {
        self.ac_tables.get(&id)
    }

    pub fn parse(&mut self, reader: &mut dyn std::io::Read) -> Result<(), std::io::Error> {
        let mut bytes_left = reader.read_u16::<NetworkEndian>()? as usize;
        bytes_left -= 2;

        while bytes_left != 0 {
            let id_class = reader.read_u8()?;
            let id = id_class & 0x0f;
            let is_dc = id_class & 0xf0 == 0; // if the high 4 bits of id_class are 0, it's a dc table

            // read the count of codes for each bit length
            let mut bits: [u8; 16] = [0; 16];
            reader.read_exact(&mut bits)?;

            // sum the counts for each bit length and read that many values
            let value_count = bits.iter().map(|&x| x as usize).sum();
            let mut values: Vec<u8> = vec![0; value_count];
            reader.read_exact(&mut values)?;

            // generate the table and store it in the proper hashmap
            let table = HuffmanTable::generate_table(bits, values);
            if is_dc {
                self.dc_tables.insert(id, table);
            } else {
                self.ac_tables.insert(id, table);
            };

            // decrement bytes left by the number of bytes read
            bytes_left -= value_count + 17; // this is counting on a panic if we underflow
        }

        Ok(())
    }
}

impl HuffmanTable {
    pub fn decode(&self, bits: &mut crate::reader::BitReader) -> Result<u8, std::io::Error> {
        let mut i = 0;

        let mut code: i16 = bits.read_bit()? as i16;
        while code > self.max_code[i] {
            code = (code << 1) + bits.read_bit()? as i16;
            i += 1
        }

        let value_index: usize = (self.value_ptrs[i] + (code - self.min_code[i])) as usize;
        return Ok(self.values[value_index]);
    }

    fn generate_table(bits: [u8; 16], values: Vec<u8>) -> Self {
        let mut min_code: [i16; 16] = [-1; 16];
        let mut max_code: [i16; 16] = [-1; 16];
        let mut value_ptrs: [i16; 16] = [0; 16];

        let mut code_sizes: Vec<u16> = Vec::with_capacity(values.len());
        let mut index: u16 = 1;
        for bit in bits {
            for _ in 0..bit {
                code_sizes.push(index);
            }
            index += 1
        }

        let mut codes: Vec<i16> = Vec::with_capacity(values.len());

        let mut k = 0;
        let mut code = 0;
        let mut si = code_sizes[0];
        while k < code_sizes.len() {
            codes.push(code);
            code += 1;
            k += 1;

            if k < code_sizes.len() {
                if code_sizes[k] != si {
                    // in other words, it changed
                    code <<= 1;
                    si += 1;
                    while code_sizes[k] != si {
                        code <<= 1;
                        si += 1;
                    }
                }
            }
        }

        let mut val_index: i16 = 0;
        for code_size in 0..16 {
            if bits[code_size] > 0 {
                value_ptrs[code_size] = val_index;
                min_code[code_size] = codes[val_index as usize];
                val_index += bits[code_size] as i16 - 1;
                max_code[code_size] = codes[val_index as usize];
                val_index += 1;
            }
        }

        Self {
            min_code,
            max_code,
            value_ptrs,
            values,
        }
    }
}

pub struct Decoder<'a> {
    _dc_pred: i16,
    dc_table: &'a HuffmanTable,
    ac_table: &'a HuffmanTable,
}

impl<'a> Decoder<'a> {
    pub fn new(dc_table: &'a HuffmanTable, ac_table: &'a HuffmanTable) -> Self {
        Decoder {
            _dc_pred: 0,
            dc_table,
            ac_table,
        }
    }

    pub fn decode_block(&mut self, reader: &mut crate::reader::BitReader) -> Result<[i16; 64], std::io::Error> {
        fn extend(v: i16, t: u8) -> i16 {
            let vt: i16 = (1 << t) >> 1;
            if v < vt {
                return v + (-1 << t) + 1;
            } else {
                return v;
            }
        }

        fn receive(ssss: u8, reader: &mut crate::reader::BitReader) -> Result<i16, std::io::Error> {
            let mut v: i16 = 0;
            for _ in 0..ssss {
                v <<= 1;
                v += reader.read_bit().unwrap() as i16;
            }

            Ok(v)
        }

        let mut result: [i16; 64] = [0; 64];
        let dc_coef = self.dc_table.decode(reader).unwrap();
        let v = receive(dc_coef, reader).unwrap();
        let dc_diff = extend(v, dc_coef);
        result[0] = self._dc_pred + dc_diff;
        self._dc_pred = result[0];

        let mut k: usize = 1;
        loop {
            let rs = self.ac_table.decode(reader).unwrap();
            let ssss = rs % 16;
            let rrrr = rs >> 4;
            let r = rrrr as usize;

            if ssss == 0 {
                if r != 15 {
                    break;
                } else {
                    k += 16;
                }
            } else {
                k += r;
                let zz_k_1 = receive(ssss, reader).unwrap();
                let zz_k_2 = extend(zz_k_1, ssss);
                result[crate::util::map_zigzag_to_normal(k)] = zz_k_2;

                if k == 63 {
                    break;
                } else {
                    k += 1;
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::BitReader;
    use std::io;

    // dc table with id 1
    const SINGLE_DC_TABLE: [u8; 0x1F] = [
        0x00, 0x1F, 0x01, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
    ];

    // dc table with id 2, ac table with id 1
    const MULTI_AC_DC_TABLE: [u8; 0xd2] = [
        0x00, 0xD2, 0x02, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x11, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11,
        0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
        0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67,
        0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9,
        0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
        0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8,
        0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
    ];

    #[test]
    fn test_parse() {
        let mut mut_tables = HuffmanTables::new();

        let mut single_reader = io::Cursor::new(SINGLE_DC_TABLE);
        mut_tables
            .parse(&mut single_reader)
            .expect("Error parsing multiple tables");

        let mut multi_reader = io::Cursor::new(MULTI_AC_DC_TABLE);
        mut_tables
            .parse(&mut multi_reader)
            .expect("Error parsing multiple tables");

        // verify the tables are all found and those that are missing aren't found
        let tables = &mut_tables;
        assert!(tables.find_dc_table(1).is_some());
        assert!(tables.find_dc_table(2).is_some());
        assert!(tables.find_dc_table(3).is_none());
        assert!(tables.find_ac_table(1).is_some());
        assert!(tables.find_ac_table(2).is_none());

        let table = tables.find_dc_table(1).unwrap();
        assert_eq!(table.value_ptrs, [0, 0, 1, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            table.max_code,
            [-1, 0, 6, 14, 30, 62, 126, 254, 510, -1, -1, -1, -1, -1, -1, -1]
        );
        assert_eq!(
            table.min_code,
            [-1, 0, 2, 14, 30, 62, 126, 254, 510, -1, -1, -1, -1, -1, -1, -1]
        );

        assert_eq!(table.values, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn test_dc_table() {
        let mut mut_tables = HuffmanTables::new();
        let mut single_reader = io::Cursor::new(SINGLE_DC_TABLE);
        mut_tables
            .parse(&mut single_reader)
            .expect("Error parsing single tables");

        let tables = &mut_tables;
        assert!(tables.find_dc_table(1).is_some());

        let table = tables.find_dc_table(1).unwrap();

        let mut cursor = io::Cursor::new([
            0b00010011u8,
            0b10010111u8,
            0b01110111u8,
            0b10111110u8,
            0b11111101u8,
            0b11111101u8,
            0b11111110u8,
        ]);
        let mut reader = BitReader::new(&mut cursor);

        assert_eq!(table.decode(&mut reader).unwrap(), 0);
        assert_eq!(table.decode(&mut reader).unwrap(), 1);
        assert_eq!(table.decode(&mut reader).unwrap(), 2);
        assert_eq!(table.decode(&mut reader).unwrap(), 3);
        assert_eq!(table.decode(&mut reader).unwrap(), 4);
        assert_eq!(table.decode(&mut reader).unwrap(), 5);
        assert_eq!(table.decode(&mut reader).unwrap(), 6);
        assert_eq!(table.decode(&mut reader).unwrap(), 7);
        assert_eq!(table.decode(&mut reader).unwrap(), 8);
        assert_eq!(table.decode(&mut reader).unwrap(), 9);
        assert_eq!(table.decode(&mut reader).unwrap(), 10);
        assert_eq!(table.decode(&mut reader).unwrap(), 11);
    }

    #[test]
    fn test_ac_decode() {
        let mut mut_tables = HuffmanTables::new();
        let mut multi_reader = io::Cursor::new(MULTI_AC_DC_TABLE);
        mut_tables
            .parse(&mut multi_reader)
            .expect("Error parsing multiple tables");

        let tables = &mut_tables;
        assert!(tables.find_ac_table(1).is_some());
        let table = tables.find_ac_table(1).unwrap();

        let mut bitstream = io::Cursor::new([0b00011110u8, 0b01111111u8, 0b11000111u8, 0b11111111u8, 0b00000000u8]);
        let mut reader = BitReader::new(&mut bitstream);

        assert_eq!(table.decode(&mut reader).unwrap(), 1);
        assert_eq!(table.decode(&mut reader).unwrap(), 2);
        assert_eq!(table.decode(&mut reader).unwrap(), 33);
        assert_eq!(table.decode(&mut reader).unwrap(), 0x34);
    }

    #[test]
    fn test_block_decode() {
        let mut tables = HuffmanTables::new();
        let mut multi_reader = io::Cursor::new(MULTI_AC_DC_TABLE);
        tables.parse(&mut multi_reader).expect("Error parsing multiple tables");
        assert!(tables.find_dc_table(2).is_some());
        assert!(tables.find_ac_table(1).is_some());

        let mut decoder = Decoder::new(tables.find_dc_table(2).unwrap(), tables.find_ac_table(1).unwrap());
        let data: [u8; 9] = [0xC9, 0x72, 0xB6, 0x82, 0xCD, 0xE2, 0xED, 0x11, 0x15];
        let mut bitstream = io::Cursor::new(data);
        let mut reader = BitReader::new(&mut bitstream);
        let res = decoder.decode_block(&mut reader).expect("Error parsing block");

        assert_eq!(
            res,
            [
                -22, 3, -1, -12, -5, 1, 0, 0, 1, -11, 3, -1, -1, 0, 0, 0, -2, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
    }
}
