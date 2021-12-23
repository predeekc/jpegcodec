use byteorder::{NetworkEndian, ReadBytesExt};

// maps the decode zigzag index to the coef un-zigzagged index
const ZIGZAG_TO_NORMAL: &'static [usize] = &[
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54,
    47, 55, 62, 63,
];

// maps the coef index to the index of the zig zagged data
const NORMAL_TO_ZIGZAG: &'static [usize] = &[
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44,
    53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49,
    57, 58, 62, 63,
];

#[inline(always)]
pub fn map_normal_to_zigzag(index: usize) -> usize {
    NORMAL_TO_ZIGZAG[index]
}

#[inline(always)]
pub fn map_zigzag_to_normal(index: usize) -> usize {
    ZIGZAG_TO_NORMAL[index]
}

pub struct QuantizationTable {
    pub id: u8,
    matrix: [i16; 64],
}

pub struct QuantizationTables {
    tables: Vec<QuantizationTable>,
}

impl QuantizationTable {
    #[inline(always)]
    pub fn dequantize(&self, coef: &mut [i16; 64]) {
        for i in 0..64 {
            coef[i] = coef[i] * self.matrix[i];
        }
    }
}

impl QuantizationTables {
    pub fn new() -> Self {
        QuantizationTables { tables: vec![] }
    }

    pub fn parse(&mut self, reader: &mut dyn std::io::Read) -> Result<(), std::io::Error> {
        let size = reader.read_u16::<NetworkEndian>()?;

        let table_count = (size - 2) / 65;
        for _ in 0..table_count {
            let id = reader.read_u8()? & 0x0f;

            let mut zz_values: [u8; 64] = [0; 64];
            reader.read_exact(&mut zz_values)?;

            let mut matrix: [i16; 64] = [0; 64];
            for i in 0..64 {
                matrix[i] = zz_values[map_normal_to_zigzag(i)] as i16;
            }

            self.tables.push(QuantizationTable { id, matrix });
        }

        Ok(())
    }

    pub fn find_table(&self, id: u8) -> Option<&QuantizationTable> {
        return self.tables.iter().find(|f| f.id == id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    const SINGLE_TABLE: [u8; 67] = [
        0x00, 0x43, 0x00, 0x08, 0x04, 0x04, 0x04, 0x04, 0x04, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x06, 0x06, 0x06,
        0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x07, 0x07, 0x07, 0x08, 0x08, 0x08, 0x07, 0x07,
        0x07, 0x06, 0x06, 0x07, 0x07, 0x08, 0x08, 0x08, 0x08, 0x09, 0x09, 0x09, 0x08, 0x08, 0x08, 0x08, 0x09, 0x09,
        0x0A, 0x0A, 0x0A, 0x0C, 0x0C, 0x0B, 0x0B, 0x0E, 0x0E, 0x0E, 0x11, 0x11, 0x14,
    ];

    const MULTI_TABLE: [u8; 132] = [
        0x00, 0x84, 0x01, 0x08, 0x04, 0x04, 0x04, 0x04, 0x04, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x06, 0x06, 0x06,
        0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x07, 0x07, 0x07, 0x08, 0x08, 0x08, 0x07, 0x07,
        0x07, 0x06, 0x06, 0x07, 0x07, 0x08, 0x08, 0x08, 0x08, 0x09, 0x09, 0x09, 0x08, 0x08, 0x08, 0x08, 0x09, 0x09,
        0x0A, 0x0A, 0x0A, 0x0C, 0x0C, 0x0B, 0x0B, 0x0E, 0x0E, 0x0E, 0x11, 0x11, 0x14, 0x02, 0x08, 0x04, 0x04, 0x04,
        0x04, 0x04, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
        0x06, 0x06, 0x06, 0x07, 0x07, 0x07, 0x08, 0x08, 0x08, 0x07, 0x07, 0x07, 0x06, 0x06, 0x07, 0x07, 0x08, 0x08,
        0x08, 0x08, 0x09, 0x09, 0x09, 0x08, 0x08, 0x08, 0x08, 0x09, 0x09, 0x0A, 0x0A, 0x0A, 0x0C, 0x0C, 0x0B, 0x0B,
        0x0E, 0x0E, 0x0E, 0x11, 0x11, 0x14,
    ];

    // TODO: support the case where a newer quant table replaces an older one (use a map, not a vector)

    #[test]
    fn test_parse_quant_table() {
        let mut mut_tables = QuantizationTables::new();

        let mut single_reader = io::Cursor::new(SINGLE_TABLE);
        mut_tables
            .parse(&mut single_reader)
            .expect("Error parsing single table");

        let mut multi_reader = io::Cursor::new(MULTI_TABLE);
        mut_tables
            .parse(&mut multi_reader)
            .expect("Error parsing multiple table");

        // verify the tables are all found and those that are missing aren't found
        let tables = &mut_tables;
        assert!(tables.find_table(0).is_some());
        assert!(tables.find_table(1).is_some());
        assert!(tables.find_table(2).is_some());
        assert!(tables.find_table(3).is_none());

        // setup an array of 1 coefficients and ensure the output of dequantization
        // matches the un-zigzagged values
        let mut coef: [i16; 64] = [1; 64];
        tables.find_table(0).unwrap().dequantize(&mut coef);
        assert_eq!(
            coef,
            [
                8, 4, 4, 5, 6, 6, 7, 8, 4, 4, 5, 6, 6, 7, 8, 9, 4, 5, 6, 6, 7, 8, 8, 9, 5, 5, 6, 6, 7, 8, 9, 10, 5, 6,
                6, 7, 8, 8, 10, 12, 6, 6, 7, 8, 8, 10, 12, 14, 6, 6, 7, 8, 9, 11, 14, 17, 6, 7, 8, 9, 11, 14, 17, 20
            ]
        );
    }
}
