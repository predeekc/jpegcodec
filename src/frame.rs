use byteorder::{NetworkEndian, ReadBytesExt};

use crate::coding;
use crate::reader;
use crate::util;

struct Component {
    id: u8,
    quant_table_id: u8,
    h_sampling: u8,
    v_sampling: u8,
}

#[derive(Default)]
pub struct Frame {
    #[allow(dead_code)]
    precision: u8,
    height: u16,
    width: u16,
    components: Vec<Component>,
    image: crate::image::RgbData,
}
struct ComponentScan<'a> {
    #[allow(dead_code)]
    id: u8,
    h_sampling: u8,
    v_sampling: u8,
    quant_table: &'a util::QuantizationTable,
    decoder: coding::Decoder<'a>,
}

impl Frame {
    pub fn parse(reader: &mut dyn std::io::Read) -> Result<Self, std::io::Error> {
        let _ = reader.read_u16::<NetworkEndian>()?;
        let precision = reader.read_u8()?;
        let height = reader.read_u16::<NetworkEndian>()?;
        let width = reader.read_u16::<NetworkEndian>()?;
        let component_count = reader.read_u8()? as usize;

        let components = (0..component_count)
            .map(|_| Component::parse(reader))
            .collect::<Result<Vec<Component>, std::io::Error>>()?;

        Ok(Frame {
            precision,
            height,
            width,
            components,
            image: crate::image::RgbData::new(width, height),
        })
    }

    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let mut bmp_file = std::fs::File::create(path).unwrap();
        self.image.save_bmp(&mut bmp_file)?;
        Ok(())
    }

    pub fn process_scan(
        &mut self,
        reader: &mut dyn std::io::Read,
        dct: &crate::dct::Dct,
        quant_tables: &util::QuantizationTables,
        coding_tables: &coding::HuffmanTables,
    ) -> Result<(), std::io::Error> {
        let _ = reader.read_u16::<NetworkEndian>()?;
        let component_count = reader.read_u8()? as usize;
        let mut components = Vec::with_capacity(component_count);

        for _ in 0..component_count {
            let id = reader.read_u8()?;
            let tables = reader.read_u8()?;
            let dc_table_id = tables >> 4;
            let ac_table_id = tables & 0x0F;
            let component = self.components.iter().find(|f| f.id == id).expect("Can't find table");

            components.push(ComponentScan {
                id,
                h_sampling: component.h_sampling,
                v_sampling: component.v_sampling,
                quant_table: quant_tables
                    .find_table(component.quant_table_id)
                    .expect("Can't find quant table"),
                decoder: coding::Decoder::new(
                    coding_tables.find_dc_table(dc_table_id).expect("Can't find DC table"),
                    coding_tables.find_ac_table(ac_table_id).expect("Can't find AC table"),
                ),
            })
        }

        reader.read_u8()?;
        reader.read_u8()?;
        reader.read_u8()?;

        // TODO: handle 444 an 422 files
        // process the MCUs until the bitream runs out
        let x_blocks = self.width / 16;
        let y_blocks = self.height / 16;

        let mut bit_reader = reader::BitReader::new(reader);
        for y in 0..y_blocks {
            for x in 0..x_blocks {
                let y_p = components[0].process(dct, &mut bit_reader).unwrap();
                let u_p = components[1].process(dct, &mut bit_reader).unwrap();
                let v_p = components[2].process(dct, &mut bit_reader).unwrap();

                self.image
                    .write_yuv420_mcu_fast(x * 16, y * 16, &y_p[0..256], &u_p[0..64], &v_p[0..64]);
            }
        }

        Ok(())
    }
}

impl Component {
    pub fn parse(reader: &mut dyn std::io::Read) -> Result<Self, std::io::Error> {
        let id = reader.read_u8()?;
        let sampling = reader.read_u8()?;
        let quant_table_id = reader.read_u8()?;

        Ok(Component {
            id,
            quant_table_id,
            h_sampling: sampling >> 4,
            v_sampling: sampling & 0x0F,
        })
    }
}

impl<'a> ComponentScan<'a> {
    pub fn process(
        &mut self,
        dct: &crate::dct::Dct,
        reader: &mut reader::BitReader,
    ) -> Result<Vec<i16>, std::io::Error> {
        let mut result: Vec<i16> = vec![0; 64 * self.v_sampling as usize * self.h_sampling as usize];

        // perf improvements?
        // - add 128 using simd in the dct
        // - do the clamp in the dct
        // - figure out how to do the conversions

        for i in 0..self.v_sampling * self.h_sampling {
            let mut coef = self.decoder.decode_block(reader).unwrap();
            self.quant_table.dequantize(&mut coef);
            let pixels: [i16; 64] = dct.idct_rows_cols_faster(&coef);

            let block_start: usize = i as usize * 64;
            let block = &mut result[block_start..block_start + 64];
            for i in 0..64 {
                block[i] = (pixels[i] + 128).clamp(0, 255);
            }
        }

        Ok(result)
    }
}

// TODO: add tests around frame composition and processing
