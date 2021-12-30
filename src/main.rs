#![feature(test)]
#![feature(stdsimd)]
#![feature(core_intrinsics)]
extern crate test;

mod coding;
mod frame;
mod image;
mod reader;
mod transform;
mod util;

use byteorder::ReadBytesExt;
use std::io::BufReader;

#[inline(never)]
fn process(reader: &mut dyn std::io::Read) -> Result<bool, std::io::Error> {
    let mut done = false;

    let mut frame = frame::Frame::default();
    let mut quant_tables = util::QuantizationTables::new();
    let mut coding_tables = coding::HuffmanTables::new();
    while !done {
        let value = reader.read_u8().unwrap();
        if value == 0xFF {
            match reader.read_u8().unwrap() {
                0xC0 => {
                    println!("Start of Frame (Baseline DCT)");
                    frame = frame::Frame::parse(reader).unwrap();
                }
                0xD8 => {
                    println!("Start of image");
                }
                0xD9 => {
                    println!("End of image");
                    frame.save("./data/out2.bmp")?;
                    done = true;
                }
                0xDA => {
                    println!("Start of Scan");
                    frame.process_scan(reader, &quant_tables, &coding_tables)?;
                }
                0xFE => println!("Comment"),
                0xC4 => {
                    println!("Define huffman table");
                    coding_tables.parse(reader)?;
                }
                0xDB => {
                    println!("Define quantization table");
                    quant_tables.parse(reader)?;
                }
                marker => println!("Marker 0x{:X}", marker),
            }
        }
    }

    Ok(true)
}

fn main() {
    let jpeg_file = std::fs::File::open("./data/sample_1920x1280.jpg").expect("Error loading Jpeg file");
    let mut reader = BufReader::new(jpeg_file);

    process(&mut reader).expect("Error with file");
}
#[cfg(test)]
mod tests {
    use std::io::Read;

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    #[bench]
    fn full_bench(b: &mut test::Bencher) {
        fn get_file_as_byte_vec(filename: &str) -> Vec<u8> {
            let mut f = std::fs::File::open(&filename).expect("no file found");
            let metadata = std::fs::metadata(&filename).expect("unable to read metadata");
            let mut buffer = vec![0; metadata.len() as usize];
            f.read(&mut buffer).expect("buffer overflow");

            buffer
        }

        let data = get_file_as_byte_vec("./data/sample_1920x1280.jpg");

        b.iter(|| {
            let mut reader = std::io::Cursor::new(data.clone());
            process(&mut reader).expect("Error with file")
        });
    }
}
