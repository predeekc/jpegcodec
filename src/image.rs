#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use byteorder::{LittleEndian, WriteBytesExt};

#[derive(Default)]
pub struct RgbData {
    width: u16,
    height: u16,
    image: Vec<u8>,
}

fn to_rgb(y: i16, u: i16, v: i16) -> (i16, i16, i16) {
    let ay = y; // - 16;
    let au = u - 128;
    let av = v - 128;

    let r8 = ay + av + (av >> 2) + (av >> 3) + (av >> 5);
    let g8 = ay - (au >> 2) - (au >> 4) - (au >> 5) - (av >> 1) - (av >> 3) - (av >> 4) - (av >> 5);
    let b8 = ay + au + (au >> 1) + (au >> 2) + (au >> 6);

    // let c = y as i32 - 16;
    // let d = u as i32 - 128;
    // let e = v as i32 - 128;
    // let r = 298 * c + 409 * e + 128;
    // let g = 298 * c - 100 * d - 208 * e + 128;
    // let b = 298 * c + 516 * d + 128;
    // let r8 = (r >> 8).clamp(0, 255) as u8;
    // let g8 = (g >> 8).clamp(0, 255) as u8;
    // let b8 = (b >> 8).clamp(0, 255) as u8;

    (r8.clamp(0, 255), g8.clamp(0, 255), b8.clamp(0, 255))
}

impl RgbData {
    pub fn new(width: u16, height: u16) -> Self {
        let image: Vec<u8> = vec![0; width as usize * height as usize * 3];
        RgbData { width, height, image }
    }

    pub fn rgb_data(&self) -> &[u8] {
        &self.image[..]
    }

    // write a 32x32 block of image data
    //
    pub fn write_yuv420_mcu(&mut self, x: u16, y: u16, c_y: &[i16], u: &[i16], v: &[i16]) {
        let image_offset = ((y as usize) * (self.width as usize) + (x as usize)) as usize * 3;

        for line in 0..8 {
            let mut line_start = image_offset + (line * self.width as usize * 3) as usize;
            let block_start = (line * 8) as usize;
            let small_block_start = ((line >> 1) * 8) as usize;
            for x in 0..8 {
                let yv = c_y[block_start + x];
                let uv = u[small_block_start + (x >> 1) as usize];
                let vv = v[small_block_start + (x >> 1) as usize];

                let (r, g, b) = to_rgb(yv, uv, vv);
                self.image[line_start + 2] = r as u8;
                self.image[line_start + 1] = g as u8;
                self.image[line_start] = b as u8;
                line_start += 3
            }
            let block_start = (line * 8) as usize + 64;
            for x in 0..8 {
                let yv = c_y[block_start + x];
                let uv = u[small_block_start + ((x + 8) >> 1) as usize];
                let vv = v[small_block_start + ((x + 8) >> 1) as usize];

                let (r, g, b) = to_rgb(yv, uv, vv);
                self.image[line_start + 2] = r as u8;
                self.image[line_start + 1] = g as u8;
                self.image[line_start] = b as u8;
                line_start += 3
            }
        }
        for line in 8..16 {
            let mut line_start = image_offset + (line * self.width as usize * 3) as usize;
            let block_start = ((line - 8) * 8) as usize + 128;
            let small_block_start = ((line >> 1) * 8) as usize;
            for x in 0..8 {
                let yv = c_y[block_start + x];
                let uv = u[small_block_start + (x >> 1) as usize];
                let vv = v[small_block_start + (x >> 1) as usize];

                let (r, g, b) = to_rgb(yv, uv, vv);
                self.image[line_start + 2] = r as u8;
                self.image[line_start + 1] = g as u8;
                self.image[line_start] = b as u8;
                line_start += 3
            }
            let block_start = ((line - 8) * 8) as usize + 192;
            for x in 0..8 {
                let yv = c_y[block_start + x];
                let uv = u[small_block_start + ((x + 8) >> 1) as usize];
                let vv = v[small_block_start + ((x + 8) >> 1) as usize];

                let (r, g, b) = to_rgb(yv, uv, vv);
                self.image[line_start + 2] = r as u8;
                self.image[line_start + 1] = g as u8;
                self.image[line_start] = b as u8;
                line_start += 3
            }
        }
    }

    //#[inline(never)]
    pub fn write_yuv420_mcu_fast(&mut self, x: u16, y: u16, c_y: &[i16], u: &[i16], v: &[i16]) {
        #[target_feature(enable = "sse4.1")]
        pub unsafe fn convert(image: *mut u8, stride: isize, y: &[i16], u: &[i16], v: &[i16]) {
            let f_128: [i16; 8] = [128; 8];

            #[target_feature(enable = "sse4.1")]
            unsafe fn convert_band(y1_reg: __m128i, u1_reg: __m128i, v1_reg: __m128i, output: *mut u8) {
                // set the initial r, g, b values
                let r_reg = _mm_add_epi16(y1_reg, v1_reg);
                let g_reg = y1_reg;
                let b_reg = _mm_add_epi16(y1_reg, u1_reg);

                // add/sub the various shifts to the U values to the B and G results
                let u1_reg = _mm_srai_epi16(u1_reg, 1);
                let b_reg = _mm_add_epi16(b_reg, u1_reg);
                let u1_reg = _mm_srai_epi16(u1_reg, 1);
                let b_reg = _mm_add_epi16(b_reg, u1_reg);
                let g_reg = _mm_sub_epi16(g_reg, u1_reg);
                let u1_reg = _mm_srai_epi16(u1_reg, 2);
                let g_reg = _mm_sub_epi16(g_reg, u1_reg);
                let u1_reg = _mm_srai_epi16(u1_reg, 1);
                let g_reg = _mm_sub_epi16(g_reg, u1_reg);
                let u1_reg = _mm_srai_epi16(u1_reg, 1);
                let b_reg = _mm_add_epi16(b_reg, u1_reg);

                // add/sub the various shifts to the V values to the G and R results
                let v1_reg = _mm_srai_epi16(v1_reg, 1);
                let g_reg = _mm_sub_epi16(g_reg, v1_reg);
                let v1_reg = _mm_srai_epi16(v1_reg, 1);
                let r_reg = _mm_add_epi16(r_reg, v1_reg);
                let v1_reg = _mm_srai_epi16(v1_reg, 1);
                let r_reg = _mm_add_epi16(r_reg, v1_reg);
                let g_reg = _mm_sub_epi16(g_reg, v1_reg);
                let v1_reg = _mm_srai_epi16(v1_reg, 1);
                let g_reg = _mm_sub_epi16(g_reg, v1_reg);
                let v1_reg = _mm_srai_epi16(v1_reg, 1);
                let g_reg = _mm_sub_epi16(g_reg, v1_reg);
                let r_reg = _mm_add_epi16(r_reg, v1_reg);

                // convert from planar results to the interlaced rgb data
                let r_b_lo = _mm_unpacklo_epi16(b_reg, r_reg);
                let r_g_hi = _mm_unpackhi_epi16(b_reg, g_reg);
                let g_b_hi = _mm_unpackhi_epi16(g_reg, r_reg);

                let t2 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(r_b_lo, 0xb4), 0xc9);
                let t1 = _mm_unpacklo_epi16(r_b_lo, g_reg);
                let t3 = _mm_shufflehi_epi16(t1, 0x78);

                let out_1_a = _mm_blend_epi16(t2, t3, 0xa7);
                let g_1_single = _mm_shufflehi_epi16(_mm_shuffle_epi32(g_reg, 0x00), 0x55);
                let out_1 = _mm_blend_epi16(out_1_a, g_1_single, 0x10);

                let out_2 = _mm_blend_epi16(t2, t3, 0x40);

                let t6 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(r_g_hi, 0xb4), 0xd8);
                let t7 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(g_b_hi, 0x1e), 0xe1);
                let out_2_a = _mm_blend_epi16(t6, t7, 0x04);
                let out_2 = _mm_shuffle_epi32(_mm_blend_epi16(out_2, out_2_a, 0x0f), 0x4e);

                let out_3_a = _mm_blend_epi16(t6, t7, 0xd3);
                let t8 = _mm_shuffle_epi32(r_g_hi, 0x08);
                let out_3 = _mm_blend_epi16(out_3_a, t8, 0x0c);

                let zero_mask = _mm_setzero_si128();
                let out_1 = _mm_max_epi16(out_1, zero_mask);
                let out_2 = _mm_max_epi16(out_2, zero_mask);
                let out_3 = _mm_max_epi16(out_3, zero_mask);

                let max: [i16; 8] = [255; 8];
                let max_mask = _mm_loadu_si128(max.as_ptr().cast::<__m128i>());
                let out_1 = _mm_min_epi16(out_1, max_mask);
                let out_2 = _mm_min_epi16(out_2, max_mask);
                let out_3 = _mm_min_epi16(out_3, max_mask);

                let vperm = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
                let u8_1 = _mm_shuffle_epi8(out_1, vperm);
                let u8_2 = _mm_shuffle_epi8(out_2, vperm);
                let u8_3 = _mm_shuffle_epi8(out_3, vperm);

                _mm_storel_epi64(output.offset(0).cast::<__m128i>(), u8_1);
                _mm_storel_epi64(output.offset(8).cast::<__m128i>(), u8_2);
                _mm_storel_epi64(output.offset(16).cast::<__m128i>(), u8_3);
            }

            for line in 0..8 {
                let y1_reg = _mm_loadu_si128(y.as_ptr().offset(line * 8).cast::<__m128i>());
                let y2_reg = _mm_loadu_si128(y.as_ptr().offset(line * 8 + 64).cast::<__m128i>());

                // load and subtract 128 from the U & V values
                let f_128_reg = _mm_loadu_si128(f_128.as_ptr().cast::<__m128i>());
                let u_reg = _mm_sub_epi16(
                    _mm_loadu_si128(u.as_ptr().offset((line >> 1) * 8).cast::<__m128i>()),
                    f_128_reg,
                );
                let v_reg = _mm_sub_epi16(
                    _mm_loadu_si128(v.as_ptr().offset((line >> 1) * 8).cast::<__m128i>()),
                    f_128_reg,
                );

                // duplicate the U and V values
                let u1_reg = _mm_unpacklo_epi16(u_reg, u_reg);
                let u2_reg = _mm_unpackhi_epi16(u_reg, u_reg);
                let v1_reg = _mm_unpacklo_epi16(v_reg, v_reg);
                let v2_reg = _mm_unpackhi_epi16(v_reg, v_reg);

                let line_start = image.offset(stride * line);
                convert_band(y1_reg, u1_reg, v1_reg, line_start.offset(0));
                convert_band(y2_reg, u2_reg, v2_reg, line_start.offset(24));
            }
            for line in 0..8 {
                let y1_reg = _mm_loadu_si128(y.as_ptr().offset(line * 8 + 128).cast::<__m128i>());
                let y2_reg = _mm_loadu_si128(y.as_ptr().offset(line * 8 + 192).cast::<__m128i>());

                // load and subtract 128 from the U & V values
                let f_128_reg = _mm_loadu_si128(f_128.as_ptr().cast::<__m128i>());
                let u_reg = _mm_sub_epi16(
                    _mm_loadu_si128(u.as_ptr().offset((line >> 1) * 8 + 32).cast::<__m128i>()),
                    f_128_reg,
                );
                let v_reg = _mm_sub_epi16(
                    _mm_loadu_si128(v.as_ptr().offset((line >> 1) * 8 + 32).cast::<__m128i>()),
                    f_128_reg,
                );

                // duplicate the U and V values
                let u1_reg = _mm_unpacklo_epi16(u_reg, u_reg);
                let u2_reg = _mm_unpackhi_epi16(u_reg, u_reg);
                let v1_reg = _mm_unpacklo_epi16(v_reg, v_reg);
                let v2_reg = _mm_unpackhi_epi16(v_reg, v_reg);

                let line_start = image.offset(stride * (line + 8));
                convert_band(y1_reg, u1_reg, v1_reg, line_start.offset(0));
                convert_band(y2_reg, u2_reg, v2_reg, line_start.offset(24));
            }

            // do the yuv calculation and get the r/g/b in multiple registers
            //_mm_storeu_si128(res.as_mut_ptr().offset(24).cast::<__m128i>(), r_reg);
            //_mm_storeu_si128(res.as_mut_ptr().offset(32).cast::<__m128i>(), g_reg);
            //_mm_storeu_si128(res.as_mut_ptr().offset(40).cast::<__m128i>(), b_reg);
        }

        let stride = self.width as isize * 3;
        unsafe {
            convert(
                self.image.as_mut_ptr().offset((y as isize * stride) + (x as isize * 3)),
                stride,
                &c_y,
                &u,
                &v,
            );
        }

        // let mut r_buf: [i16; 8] = [0; 8];
        // let mut g_buf: [i16; 8] = [0; 8];
        // let mut b_buf: [i16; 8] = [0; 8];

        // unsafe fn write_interleaved(image: &mut [u8], offset: isize, r: &[i16; 8], g: &[i16; 8], b: &[i16; 8]) {
        //     let r_reg = _mm_loadu_si128(b.as_ptr().cast::<__m128i>());
        //     let g_reg = _mm_loadu_si128(g.as_ptr().cast::<__m128i>());
        //     let b_reg = _mm_loadu_si128(r.as_ptr().cast::<__m128i>());

        //     let r_b_lo = _mm_unpacklo_epi16(r_reg, b_reg);
        //     let r_g_hi = _mm_unpackhi_epi16(r_reg, g_reg);
        //     let g_b_hi = _mm_unpackhi_epi16(g_reg, b_reg);

        //     let t2 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(r_b_lo, 0xb4), 0xc9);
        //     let t1 = _mm_unpacklo_epi16(r_b_lo, g_reg);
        //     let t3 = _mm_shufflehi_epi16(t1, 0x78);

        //     let out_1_a = _mm_blend_epi16(t2, t3, 0xa7);
        //     let g_1_single = _mm_shufflehi_epi16(_mm_shuffle_epi32(g_reg, 0x00), 0x55);
        //     let out_1 = _mm_blend_epi16(out_1_a, g_1_single, 0x10);

        //     let out_2 = _mm_blend_epi16(t2, t3, 0x40);

        //     let t6 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(r_g_hi, 0xb4), 0xd8);
        //     let t7 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(g_b_hi, 0x1e), 0xe1);
        //     let out_2_a = _mm_blend_epi16(t6, t7, 0x04);
        //     let out_2 = _mm_shuffle_epi32(_mm_blend_epi16(out_2, out_2_a, 0x0f), 0x4e);

        //     let out_3_a = _mm_blend_epi16(t6, t7, 0xd3);
        //     let t8 = _mm_shuffle_epi32(r_g_hi, 0x08);
        //     let out_3 = _mm_blend_epi16(out_3_a, t8, 0x0c);

        //     let vperm = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
        //     let u8_1 = _mm_shuffle_epi8(out_1, vperm);
        //     let u8_2 = _mm_shuffle_epi8(out_2, vperm);
        //     let u8_3 = _mm_shuffle_epi8(out_3, vperm);

        //     _mm_storel_epi64(image.as_mut_ptr().offset(offset).cast::<__m128i>(), u8_1);
        //     _mm_storel_epi64(image.as_mut_ptr().offset(offset + 8).cast::<__m128i>(), u8_2);
        //     _mm_storel_epi64(image.as_mut_ptr().offset(offset + 16).cast::<__m128i>(), u8_3);
        // }

        // let image_offset = ((y as isize) * (self.width as isize) + (x as isize)) as isize * 3;

        // for line in 0..8 {
        //     let mut line_start = image_offset + (line * self.width as isize * 3) as isize;
        //     let block_start = (line * 8) as usize;
        //     let small_block_start = ((line >> 1) * 8) as usize;
        //     for x in 0..8 {
        //         let yv = c_y[block_start + x];
        //         let uv = u[small_block_start + (x >> 1) as usize];
        //         let vv = v[small_block_start + (x >> 1) as usize];

        //         let (r, g, b) = to_rgb(yv, uv, vv);
        //         r_buf[x] = r;
        //         g_buf[x] = g;
        //         b_buf[x] = b;
        //         // self.image[line_start + 2] = r;
        //         // self.image[line_start + 1] = g;
        //         // self.image[line_start] = b;
        //         // line_start += 3
        //     }
        //     unsafe {
        //         write_interleaved(&mut self.image, line_start, &r_buf, &g_buf, &b_buf);
        //     }
        //     line_start += 24;
        //     let block_start = (line * 8) as usize + 64;
        //     for x in 0..8 {
        //         let yv = c_y[block_start + x];
        //         let uv = u[small_block_start + ((x + 8) >> 1) as usize];
        //         let vv = v[small_block_start + ((x + 8) >> 1) as usize];

        //         let (r, g, b) = to_rgb(yv, uv, vv);
        //         r_buf[x] = r;
        //         g_buf[x] = g;
        //         b_buf[x] = b;
        //         // self.image[line_start + 2] = r;
        //         // self.image[line_start + 1] = g;
        //         // self.image[line_start] = b;
        //         // line_start += 3
        //     }

        //     unsafe {
        //         write_interleaved(&mut self.image, line_start, &r_buf, &g_buf, &b_buf);
        //     }
        //     line_start += 24;
        // }
        // for line in 8..16 {
        //     let mut line_start = image_offset + (line * self.width as isize * 3) as isize;
        //     let block_start = ((line - 8) * 8) as usize + 128;
        //     let small_block_start = ((line >> 1) * 8) as usize;
        //     for x in 0..8 {
        //         let yv = c_y[block_start + x];
        //         let uv = u[small_block_start + (x >> 1) as usize];
        //         let vv = v[small_block_start + (x >> 1) as usize];

        //         let (r, g, b) = to_rgb(yv, uv, vv);
        //         r_buf[x] = r;
        //         g_buf[x] = g;
        //         b_buf[x] = b;
        //         // self.image[line_start + 2] = r;
        //         // self.image[line_start + 1] = g;
        //         // self.image[line_start] = b;
        //         // line_start += 3
        //     }
        //     unsafe {
        //         write_interleaved(&mut self.image, line_start, &r_buf, &g_buf, &b_buf);
        //     }
        //     line_start += 24;
        //     let block_start = ((line - 8) * 8) as usize + 192;
        //     for x in 0..8 {
        //         let yv = c_y[block_start + x];
        //         let uv = u[small_block_start + ((x + 8) >> 1) as usize];
        //         let vv = v[small_block_start + ((x + 8) >> 1) as usize];

        //         let (r, g, b) = to_rgb(yv, uv, vv);
        //         r_buf[x] = r;
        //         g_buf[x] = g;
        //         b_buf[x] = b;
        //         // self.image[line_start + 2] = r;
        //         // self.image[line_start + 1] = g;
        //         // self.image[line_start] = b;
        //         // line_start += 3
        //     }
        //     unsafe {
        //         write_interleaved(&mut self.image, line_start, &r_buf, &g_buf, &b_buf);
        //     }
        //     line_start += 24;
        // }
    }

    pub fn save_bmp(&self, file: &mut dyn std::io::Write) -> Result<(), std::io::Error> {
        let header_size: u32 = 54;
        let data_size: u32 = self.width as u32 * self.height as u32 * 3;

        file.write_u8(0x42)?;
        file.write_u8(0x4D)?;
        file.write_u32::<LittleEndian>(header_size + data_size)?;
        file.write_u32::<LittleEndian>(0)?; // reserved
        file.write_u32::<LittleEndian>(header_size)?; // offset to image data

        file.write_u32::<LittleEndian>(40)?; // this header size
        file.write_i32::<LittleEndian>(self.width as i32)?; // image width
        file.write_i32::<LittleEndian>(-(self.height as i32))?; // image height (-1 to handle data direction)

        file.write_u16::<LittleEndian>(1)?; // color planes
        file.write_u16::<LittleEndian>(24)?; // bits per pixel
        file.write_u32::<LittleEndian>(0)?; // compression method
        file.write_u32::<LittleEndian>(data_size)?; // image data size
        file.write_u32::<LittleEndian>(0)?; // horizontal resolution
        file.write_u32::<LittleEndian>(0)?; // vertical resolution
        file.write_u32::<LittleEndian>(0)?; // palette size
        file.write_u32::<LittleEndian>(0)?; // important color count

        file.write_all(&self.image)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_block() {
        let mut image = RgbData::new(16, 16);
        let mut y: [i16; 256] = [0; 256];
        let block_1: [i16; 64] = [
            0xA4, 0xA5, 0x82, 0x6C, 0x58, 0x45, 0x44, 0x2E, 0x65, 0x83, 0x72, 0x78, 0x6F, 0x53, 0x4D, 0x27, 0x2F, 0x35,
            0x32, 0x5A, 0x6E, 0x5B, 0x5B, 0x4E, 0x32, 0x0F, 0x08, 0x24, 0x36, 0x31, 0x4A, 0x80, 0x42, 0x24, 0x0E, 0x1A,
            0x2C, 0x22, 0x2E, 0x61, 0x76, 0x72, 0x42, 0x44, 0x53, 0x3B, 0x3D, 0x44, 0x77, 0x92, 0x77, 0x6E, 0x59, 0x41,
            0x5D, 0x5E, 0x42, 0x64, 0x6F, 0x5F, 0x32, 0x2C, 0x58, 0x5B,
        ];
        let block_2: [i16; 64] = [
            0x28, 0x44, 0x5E, 0x74, 0x4C, 0x40, 0x35, 0x1E, 0x1E, 0x2F, 0x2C, 0x30, 0x2D, 0x37, 0x20, 0x1D, 0x3A, 0x2C,
            0x16, 0x0C, 0x1C, 0x2C, 0x10, 0x22, 0x70, 0x5D, 0x59, 0x38, 0x20, 0x1D, 0x16, 0x2D, 0x61, 0x63, 0x8D, 0x78,
            0x3C, 0x15, 0x22, 0x41, 0x3C, 0x3F, 0x70, 0x80, 0x55, 0x2C, 0x33, 0x47, 0x60, 0x54, 0x5D, 0x6B, 0x4F, 0x52,
            0x47, 0x31, 0x90, 0x79, 0x6A, 0x6F, 0x46, 0x65, 0x52, 0x22,
        ];
        let block_3: [i16; 64] = [
            0x26, 0x39, 0x54, 0x55, 0x2D, 0x2C, 0x3D, 0x4D, 0x12, 0x32, 0x48, 0x48, 0x34, 0x39, 0x4E, 0x5B, 0x01, 0x0E,
            0x2B, 0x45, 0x50, 0x54, 0x5C, 0x52, 0x20, 0x21, 0x3F, 0x6D, 0x82, 0x71, 0x71, 0x5F, 0x64, 0x68, 0x57, 0x66,
            0x7D, 0x73, 0x98, 0x9A, 0x7A, 0x80, 0x48, 0x34, 0x40, 0x4A, 0x8F, 0x8C, 0x48, 0x54, 0x3E, 0x36, 0x38, 0x39,
            0x5F, 0x55, 0x36, 0x3B, 0x45, 0x51, 0x5C, 0x4A, 0x3E, 0x44,
        ];
        let block_4: [i16; 64] = [
            0x86, 0x89, 0x86, 0x5E, 0x58, 0x7E, 0x56, 0x21, 0x6E, 0x67, 0x4F, 0x45, 0x63, 0x7D, 0x5B, 0x30, 0x4C, 0x57,
            0x45, 0x45, 0x5F, 0x65, 0x5F, 0x4D, 0x49, 0x65, 0x5F, 0x48, 0x48, 0x49, 0x5E, 0x68, 0x56, 0x64, 0x60, 0x45,
            0x3D, 0x2D, 0x3D, 0x6D, 0x3E, 0x4A, 0x53, 0x4B, 0x50, 0x3A, 0x2D, 0x5C, 0x2C, 0x34, 0x43, 0x4A, 0x50, 0x52,
            0x41, 0x3F, 0x24, 0x0C, 0x1C, 0x3C, 0x33, 0x46, 0x5A, 0x4E,
        ];
        y[0..64].copy_from_slice(&block_1);
        y[64..128].copy_from_slice(&block_2);
        y[128..192].copy_from_slice(&block_3);
        y[192..256].copy_from_slice(&block_4);

        let u: [i16; 64] = [
            0x7E, 0x79, 0x77, 0x78, 0x77, 0x79, 0x7E, 0x80, 0x80, 0x79, 0x77, 0x78, 0x79, 0x7B, 0x7F, 0x7F, 0x80, 0x79,
            0x76, 0x78, 0x79, 0x7B, 0x7E, 0x7D, 0x7D, 0x78, 0x76, 0x77, 0x77, 0x78, 0x7B, 0x7B, 0x7B, 0x78, 0x78, 0x78,
            0x75, 0x75, 0x79, 0x7B, 0x7B, 0x79, 0x7A, 0x7A, 0x75, 0x74, 0x78, 0x7B, 0x7B, 0x78, 0x79, 0x79, 0x75, 0x74,
            0x77, 0x79, 0x7B, 0x77, 0x76, 0x77, 0x74, 0x73, 0x76, 0x76,
        ];

        let v: [i16; 64] = [
            0x99, 0x88, 0x7F, 0x82, 0x8F, 0x95, 0x89, 0x83, 0x9A, 0x8A, 0x7E, 0x82, 0x91, 0x98, 0x8E, 0x84, 0xA2, 0x93,
            0x83, 0x87, 0x98, 0xA0, 0x99, 0x8A, 0xA1, 0x96, 0x87, 0x8B, 0x97, 0x9E, 0x9B, 0x8B, 0x8F, 0x8E, 0x8A, 0x8C,
            0x8D, 0x8E, 0x8E, 0x83, 0x80, 0x88, 0x91, 0x92, 0x87, 0x81, 0x84, 0x81, 0x81, 0x83, 0x90, 0x90, 0x84, 0x80,
            0x82, 0x83, 0x85, 0x7D, 0x85, 0x85, 0x7E, 0x7F, 0x80, 0x81,
        ];

        image.write_yuv420_mcu(0, 0, &y, &u, &v);

        assert_eq!(
            image.image,
            [
                159, 151, 198, 160, 152, 199, 116, 129, 141, 94, 107, 119, 70, 97, 84, 51, 78, 65, 53, 71, 70, 31, 49,
                48, 22, 37, 59, 50, 65, 87, 80, 85, 122, 102, 107, 144, 71, 74, 88, 59, 62, 76, 53, 52, 56, 30, 29, 33,
                96, 88, 135, 126, 118, 165, 100, 113, 125, 106, 119, 131, 93, 120, 107, 65, 92, 79, 62, 80, 79, 24, 42,
                41, 12, 27, 49, 29, 44, 66, 30, 35, 72, 34, 39, 76, 40, 43, 57, 50, 53, 67, 32, 31, 35, 29, 28, 32, 47,
                30, 82, 53, 36, 88, 36, 48, 63, 76, 88, 103, 92, 119, 105, 73, 100, 86, 76, 94, 93, 63, 81, 80, 44, 51,
                81, 30, 37, 67, 11, 10, 55, 1, 0, 45, 24, 23, 46, 40, 39, 62, 12, 17, 21, 30, 35, 39, 50, 33, 85, 15,
                0, 50, 0, 6, 21, 22, 34, 49, 36, 63, 49, 31, 58, 44, 59, 77, 76, 113, 131, 130, 98, 105, 135, 79, 86,
                116, 78, 77, 122, 45, 44, 89, 28, 27, 50, 25, 24, 47, 18, 23, 27, 41, 46, 50, 66, 42, 113, 36, 12, 83,
                0, 6, 39, 12, 18, 51, 25, 48, 47, 15, 38, 37, 31, 47, 54, 82, 98, 105, 83, 85, 130, 85, 87, 132, 130,
                122, 186, 109, 101, 165, 55, 47, 94, 16, 8, 55, 27, 31, 47, 58, 62, 78, 118, 94, 165, 114, 90, 161, 52,
                58, 91, 54, 60, 93, 64, 87, 86, 40, 63, 62, 46, 62, 69, 53, 69, 76, 46, 48, 93, 49, 51, 96, 101, 93,
                157, 117, 109, 173, 80, 72, 119, 39, 31, 78, 44, 48, 64, 64, 68, 84, 112, 99, 165, 139, 126, 192, 104,
                109, 148, 95, 100, 139, 70, 91, 97, 46, 67, 73, 75, 92, 107, 76, 93, 108, 78, 87, 126, 66, 75, 114, 78,
                78, 133, 92, 92, 147, 68, 66, 115, 71, 69, 118, 60, 69, 85, 38, 47, 63, 59, 46, 112, 93, 80, 146, 96,
                101, 140, 80, 85, 124, 31, 52, 58, 25, 46, 52, 70, 87, 102, 73, 90, 105, 126, 135, 174, 103, 112, 151,
                91, 91, 146, 96, 96, 151, 59, 57, 106, 90, 88, 137, 71, 80, 96, 23, 32, 48, 27, 34, 57, 46, 53, 76, 69,
                80, 102, 70, 81, 103, 30, 43, 58, 29, 42, 57, 46, 58, 77, 62, 74, 93, 113, 132, 151, 116, 135, 154,
                113, 131, 152, 73, 91, 112, 74, 84, 106, 112, 122, 144, 75, 89, 89, 22, 36, 36, 7, 14, 37, 39, 46, 69,
                57, 68, 90, 57, 68, 90, 37, 50, 65, 42, 55, 70, 63, 75, 94, 76, 88, 107, 89, 108, 127, 82, 101, 120,
                58, 76, 97, 48, 66, 87, 85, 95, 117, 111, 121, 143, 80, 94, 94, 37, 51, 51, 0, 5, 1, 3, 18, 14, 29, 42,
                54, 55, 68, 80, 68, 73, 103, 72, 77, 107, 80, 84, 116, 70, 74, 106, 55, 78, 84, 66, 89, 95, 47, 74, 70,
                47, 74, 70, 80, 97, 100, 86, 103, 106, 84, 99, 96, 66, 81, 78, 21, 36, 32, 22, 37, 33, 49, 62, 74, 95,
                108, 120, 118, 123, 153, 101, 106, 136, 101, 105, 137, 83, 87, 119, 52, 75, 81, 80, 103, 109, 73, 100,
                96, 50, 77, 73, 57, 74, 77, 58, 75, 78, 83, 98, 95, 93, 108, 105, 89, 104, 101, 93, 108, 105, 72, 90,
                90, 87, 105, 105, 111, 118, 147, 101, 108, 137, 138, 145, 174, 140, 147, 176, 65, 89, 91, 79, 103, 105,
                74, 101, 96, 47, 74, 69, 43, 65, 63, 27, 49, 47, 47, 64, 64, 95, 112, 112, 111, 126, 123, 117, 132,
                129, 57, 75, 75, 37, 55, 55, 50, 57, 86, 60, 67, 96, 129, 136, 165, 126, 133, 162, 41, 65, 67, 53, 77,
                79, 61, 88, 83, 53, 80, 75, 62, 84, 82, 40, 62, 60, 31, 48, 48, 78, 95, 95, 61, 74, 78, 73, 86, 90, 44,
                72, 56, 36, 64, 48, 37, 59, 62, 38, 60, 63, 77, 98, 101, 67, 88, 91, 22, 53, 39, 30, 61, 47, 42, 77,
                63, 49, 84, 70, 61, 85, 80, 63, 87, 82, 46, 70, 66, 44, 68, 64, 43, 56, 60, 48, 61, 65, 51, 79, 63, 63,
                91, 75, 73, 95, 98, 55, 77, 80, 44, 65, 68, 50, 71, 74, 14, 45, 31, 0, 21, 7, 3, 38, 24, 35, 70, 56,
                32, 56, 51, 51, 75, 70, 71, 95, 91, 59, 83, 79
            ]
        );

        let mut bmp_file = std::fs::File::create("./data/out.bmp").expect("Error loading Jpeg file");
        image.save_bmp(&mut bmp_file).expect("Error writing");
    }
    #[test]
    fn test_block_fast() {
        let mut image = RgbData::new(16, 16);
        let mut y: [i16; 256] = [0; 256];
        let block_1: [i16; 64] = [
            0xA4, 0xA5, 0x82, 0x6C, 0x58, 0x45, 0x44, 0x2E, 0x65, 0x83, 0x72, 0x78, 0x6F, 0x53, 0x4D, 0x27, 0x2F, 0x35,
            0x32, 0x5A, 0x6E, 0x5B, 0x5B, 0x4E, 0x32, 0x0F, 0x08, 0x24, 0x36, 0x31, 0x4A, 0x80, 0x42, 0x24, 0x0E, 0x1A,
            0x2C, 0x22, 0x2E, 0x61, 0x76, 0x72, 0x42, 0x44, 0x53, 0x3B, 0x3D, 0x44, 0x77, 0x92, 0x77, 0x6E, 0x59, 0x41,
            0x5D, 0x5E, 0x42, 0x64, 0x6F, 0x5F, 0x32, 0x2C, 0x58, 0x5B,
        ];
        let block_2: [i16; 64] = [
            0x28, 0x44, 0x5E, 0x74, 0x4C, 0x40, 0x35, 0x1E, 0x1E, 0x2F, 0x2C, 0x30, 0x2D, 0x37, 0x20, 0x1D, 0x3A, 0x2C,
            0x16, 0x0C, 0x1C, 0x2C, 0x10, 0x22, 0x70, 0x5D, 0x59, 0x38, 0x20, 0x1D, 0x16, 0x2D, 0x61, 0x63, 0x8D, 0x78,
            0x3C, 0x15, 0x22, 0x41, 0x3C, 0x3F, 0x70, 0x80, 0x55, 0x2C, 0x33, 0x47, 0x60, 0x54, 0x5D, 0x6B, 0x4F, 0x52,
            0x47, 0x31, 0x90, 0x79, 0x6A, 0x6F, 0x46, 0x65, 0x52, 0x22,
        ];
        let block_3: [i16; 64] = [
            0x26, 0x39, 0x54, 0x55, 0x2D, 0x2C, 0x3D, 0x4D, 0x12, 0x32, 0x48, 0x48, 0x34, 0x39, 0x4E, 0x5B, 0x01, 0x0E,
            0x2B, 0x45, 0x50, 0x54, 0x5C, 0x52, 0x20, 0x21, 0x3F, 0x6D, 0x82, 0x71, 0x71, 0x5F, 0x64, 0x68, 0x57, 0x66,
            0x7D, 0x73, 0x98, 0x9A, 0x7A, 0x80, 0x48, 0x34, 0x40, 0x4A, 0x8F, 0x8C, 0x48, 0x54, 0x3E, 0x36, 0x38, 0x39,
            0x5F, 0x55, 0x36, 0x3B, 0x45, 0x51, 0x5C, 0x4A, 0x3E, 0x44,
        ];
        let block_4: [i16; 64] = [
            0x86, 0x89, 0x86, 0x5E, 0x58, 0x7E, 0x56, 0x21, 0x6E, 0x67, 0x4F, 0x45, 0x63, 0x7D, 0x5B, 0x30, 0x4C, 0x57,
            0x45, 0x45, 0x5F, 0x65, 0x5F, 0x4D, 0x49, 0x65, 0x5F, 0x48, 0x48, 0x49, 0x5E, 0x68, 0x56, 0x64, 0x60, 0x45,
            0x3D, 0x2D, 0x3D, 0x6D, 0x3E, 0x4A, 0x53, 0x4B, 0x50, 0x3A, 0x2D, 0x5C, 0x2C, 0x34, 0x43, 0x4A, 0x50, 0x52,
            0x41, 0x3F, 0x24, 0x0C, 0x1C, 0x3C, 0x33, 0x46, 0x5A, 0x4E,
        ];
        y[0..64].copy_from_slice(&block_1);
        y[64..128].copy_from_slice(&block_2);
        y[128..192].copy_from_slice(&block_3);
        y[192..256].copy_from_slice(&block_4);

        let u: [i16; 64] = [
            0x7E, 0x79, 0x77, 0x78, 0x77, 0x79, 0x7E, 0x80, 0x80, 0x79, 0x77, 0x78, 0x79, 0x7B, 0x7F, 0x7F, 0x80, 0x79,
            0x76, 0x78, 0x79, 0x7B, 0x7E, 0x7D, 0x7D, 0x78, 0x76, 0x77, 0x77, 0x78, 0x7B, 0x7B, 0x7B, 0x78, 0x78, 0x78,
            0x75, 0x75, 0x79, 0x7B, 0x7B, 0x79, 0x7A, 0x7A, 0x75, 0x74, 0x78, 0x7B, 0x7B, 0x78, 0x79, 0x79, 0x75, 0x74,
            0x77, 0x79, 0x7B, 0x77, 0x76, 0x77, 0x74, 0x73, 0x76, 0x76,
        ];

        let v: [i16; 64] = [
            0x99, 0x88, 0x7F, 0x82, 0x8F, 0x95, 0x89, 0x83, 0x9A, 0x8A, 0x7E, 0x82, 0x91, 0x98, 0x8E, 0x84, 0xA2, 0x93,
            0x83, 0x87, 0x98, 0xA0, 0x99, 0x8A, 0xA1, 0x96, 0x87, 0x8B, 0x97, 0x9E, 0x9B, 0x8B, 0x8F, 0x8E, 0x8A, 0x8C,
            0x8D, 0x8E, 0x8E, 0x83, 0x80, 0x88, 0x91, 0x92, 0x87, 0x81, 0x84, 0x81, 0x81, 0x83, 0x90, 0x90, 0x84, 0x80,
            0x82, 0x83, 0x85, 0x7D, 0x85, 0x85, 0x7E, 0x7F, 0x80, 0x81,
        ];

        image.write_yuv420_mcu_fast(0, 0, &y, &u, &v);

        assert_eq!(
            image.image,
            [
                159, 151, 198, 160, 152, 199, 116, 129, 141, 94, 107, 119, 70, 97, 84, 51, 78, 65, 53, 71, 70, 31, 49,
                48, 22, 37, 59, 50, 65, 87, 80, 85, 122, 102, 107, 144, 71, 74, 88, 59, 62, 76, 53, 52, 56, 30, 29, 33,
                96, 88, 135, 126, 118, 165, 100, 113, 125, 106, 119, 131, 93, 120, 107, 65, 92, 79, 62, 80, 79, 24, 42,
                41, 12, 27, 49, 29, 44, 66, 30, 35, 72, 34, 39, 76, 40, 43, 57, 50, 53, 67, 32, 31, 35, 29, 28, 32, 47,
                30, 82, 53, 36, 88, 36, 48, 63, 76, 88, 103, 92, 119, 105, 73, 100, 86, 76, 94, 93, 63, 81, 80, 44, 51,
                81, 30, 37, 67, 11, 10, 55, 1, 0, 45, 24, 23, 46, 40, 39, 62, 12, 17, 21, 30, 35, 39, 50, 33, 85, 15,
                0, 50, 0, 6, 21, 22, 34, 49, 36, 63, 49, 31, 58, 44, 59, 77, 76, 113, 131, 130, 98, 105, 135, 79, 86,
                116, 78, 77, 122, 45, 44, 89, 28, 27, 50, 25, 24, 47, 18, 23, 27, 41, 46, 50, 66, 42, 113, 36, 12, 83,
                0, 6, 39, 12, 18, 51, 25, 48, 47, 15, 38, 37, 31, 47, 54, 82, 98, 105, 83, 85, 130, 85, 87, 132, 130,
                122, 186, 109, 101, 165, 55, 47, 94, 16, 8, 55, 27, 31, 47, 58, 62, 78, 118, 94, 165, 114, 90, 161, 52,
                58, 91, 54, 60, 93, 64, 87, 86, 40, 63, 62, 46, 62, 69, 53, 69, 76, 46, 48, 93, 49, 51, 96, 101, 93,
                157, 117, 109, 173, 80, 72, 119, 39, 31, 78, 44, 48, 64, 64, 68, 84, 112, 99, 165, 139, 126, 192, 104,
                109, 148, 95, 100, 139, 70, 91, 97, 46, 67, 73, 75, 92, 107, 76, 93, 108, 78, 87, 126, 66, 75, 114, 78,
                78, 133, 92, 92, 147, 68, 66, 115, 71, 69, 118, 60, 69, 85, 38, 47, 63, 59, 46, 112, 93, 80, 146, 96,
                101, 140, 80, 85, 124, 31, 52, 58, 25, 46, 52, 70, 87, 102, 73, 90, 105, 126, 135, 174, 103, 112, 151,
                91, 91, 146, 96, 96, 151, 59, 57, 106, 90, 88, 137, 71, 80, 96, 23, 32, 48, 27, 34, 57, 46, 53, 76, 69,
                80, 102, 70, 81, 103, 30, 43, 58, 29, 42, 57, 46, 58, 77, 62, 74, 93, 113, 132, 151, 116, 135, 154,
                113, 131, 152, 73, 91, 112, 74, 84, 106, 112, 122, 144, 75, 89, 89, 22, 36, 36, 7, 14, 37, 39, 46, 69,
                57, 68, 90, 57, 68, 90, 37, 50, 65, 42, 55, 70, 63, 75, 94, 76, 88, 107, 89, 108, 127, 82, 101, 120,
                58, 76, 97, 48, 66, 87, 85, 95, 117, 111, 121, 143, 80, 94, 94, 37, 51, 51, 0, 5, 1, 3, 18, 14, 29, 42,
                54, 55, 68, 80, 68, 73, 103, 72, 77, 107, 80, 84, 116, 70, 74, 106, 55, 78, 84, 66, 89, 95, 47, 74, 70,
                47, 74, 70, 80, 97, 100, 86, 103, 106, 84, 99, 96, 66, 81, 78, 21, 36, 32, 22, 37, 33, 49, 62, 74, 95,
                108, 120, 118, 123, 153, 101, 106, 136, 101, 105, 137, 83, 87, 119, 52, 75, 81, 80, 103, 109, 73, 100,
                96, 50, 77, 73, 57, 74, 77, 58, 75, 78, 83, 98, 95, 93, 108, 105, 89, 104, 101, 93, 108, 105, 72, 90,
                90, 87, 105, 105, 111, 118, 147, 101, 108, 137, 138, 145, 174, 140, 147, 176, 65, 89, 91, 79, 103, 105,
                74, 101, 96, 47, 74, 69, 43, 65, 63, 27, 49, 47, 47, 64, 64, 95, 112, 112, 111, 126, 123, 117, 132,
                129, 57, 75, 75, 37, 55, 55, 50, 57, 86, 60, 67, 96, 129, 136, 165, 126, 133, 162, 41, 65, 67, 53, 77,
                79, 61, 88, 83, 53, 80, 75, 62, 84, 82, 40, 62, 60, 31, 48, 48, 78, 95, 95, 61, 74, 78, 73, 86, 90, 44,
                72, 56, 36, 64, 48, 37, 59, 62, 38, 60, 63, 77, 98, 101, 67, 88, 91, 22, 53, 39, 30, 61, 47, 42, 77,
                63, 49, 84, 70, 61, 85, 80, 63, 87, 82, 46, 70, 66, 44, 68, 64, 43, 56, 60, 48, 61, 65, 51, 79, 63, 63,
                91, 75, 73, 95, 98, 55, 77, 80, 44, 65, 68, 50, 71, 74, 14, 45, 31, 0, 21, 7, 3, 38, 24, 35, 70, 56,
                32, 56, 51, 51, 75, 70, 71, 95, 91, 59, 83, 79
            ]
        );

        let mut bmp_file = std::fs::File::create("./data/out.bmp").expect("Error loading Jpeg file");
        image.save_bmp(&mut bmp_file).expect("Error writing");
    }

    #[bench]
    fn iterative_yuv420_rgb(b: &mut test::Bencher) {
        let mut y: [i16; 256] = [0; 256];
        let block_1: [i16; 64] = [
            0xA4, 0xA5, 0x82, 0x6C, 0x58, 0x45, 0x44, 0x2E, 0x65, 0x83, 0x72, 0x78, 0x6F, 0x53, 0x4D, 0x27, 0x2F, 0x35,
            0x32, 0x5A, 0x6E, 0x5B, 0x5B, 0x4E, 0x32, 0x0F, 0x08, 0x24, 0x36, 0x31, 0x4A, 0x80, 0x42, 0x24, 0x0E, 0x1A,
            0x2C, 0x22, 0x2E, 0x61, 0x76, 0x72, 0x42, 0x44, 0x53, 0x3B, 0x3D, 0x44, 0x77, 0x92, 0x77, 0x6E, 0x59, 0x41,
            0x5D, 0x5E, 0x42, 0x64, 0x6F, 0x5F, 0x32, 0x2C, 0x58, 0x5B,
        ];
        let block_2: [i16; 64] = [
            0x28, 0x44, 0x5E, 0x74, 0x4C, 0x40, 0x35, 0x1E, 0x1E, 0x2F, 0x2C, 0x30, 0x2D, 0x37, 0x20, 0x1D, 0x3A, 0x2C,
            0x16, 0x0C, 0x1C, 0x2C, 0x10, 0x22, 0x70, 0x5D, 0x59, 0x38, 0x20, 0x1D, 0x16, 0x2D, 0x61, 0x63, 0x8D, 0x78,
            0x3C, 0x15, 0x22, 0x41, 0x3C, 0x3F, 0x70, 0x80, 0x55, 0x2C, 0x33, 0x47, 0x60, 0x54, 0x5D, 0x6B, 0x4F, 0x52,
            0x47, 0x31, 0x90, 0x79, 0x6A, 0x6F, 0x46, 0x65, 0x52, 0x22,
        ];
        let block_3: [i16; 64] = [
            0x26, 0x39, 0x54, 0x55, 0x2D, 0x2C, 0x3D, 0x4D, 0x12, 0x32, 0x48, 0x48, 0x34, 0x39, 0x4E, 0x5B, 0x01, 0x0E,
            0x2B, 0x45, 0x50, 0x54, 0x5C, 0x52, 0x20, 0x21, 0x3F, 0x6D, 0x82, 0x71, 0x71, 0x5F, 0x64, 0x68, 0x57, 0x66,
            0x7D, 0x73, 0x98, 0x9A, 0x7A, 0x80, 0x48, 0x34, 0x40, 0x4A, 0x8F, 0x8C, 0x48, 0x54, 0x3E, 0x36, 0x38, 0x39,
            0x5F, 0x55, 0x36, 0x3B, 0x45, 0x51, 0x5C, 0x4A, 0x3E, 0x44,
        ];
        let block_4: [i16; 64] = [
            0x86, 0x89, 0x86, 0x5E, 0x58, 0x7E, 0x56, 0x21, 0x6E, 0x67, 0x4F, 0x45, 0x63, 0x7D, 0x5B, 0x30, 0x4C, 0x57,
            0x45, 0x45, 0x5F, 0x65, 0x5F, 0x4D, 0x49, 0x65, 0x5F, 0x48, 0x48, 0x49, 0x5E, 0x68, 0x56, 0x64, 0x60, 0x45,
            0x3D, 0x2D, 0x3D, 0x6D, 0x3E, 0x4A, 0x53, 0x4B, 0x50, 0x3A, 0x2D, 0x5C, 0x2C, 0x34, 0x43, 0x4A, 0x50, 0x52,
            0x41, 0x3F, 0x24, 0x0C, 0x1C, 0x3C, 0x33, 0x46, 0x5A, 0x4E,
        ];
        y[0..64].copy_from_slice(&block_1);
        y[64..128].copy_from_slice(&block_2);
        y[128..192].copy_from_slice(&block_3);
        y[192..256].copy_from_slice(&block_4);

        let u: [i16; 64] = [
            0x7E, 0x79, 0x77, 0x78, 0x77, 0x79, 0x7E, 0x80, 0x80, 0x79, 0x77, 0x78, 0x79, 0x7B, 0x7F, 0x7F, 0x80, 0x79,
            0x76, 0x78, 0x79, 0x7B, 0x7E, 0x7D, 0x7D, 0x78, 0x76, 0x77, 0x77, 0x78, 0x7B, 0x7B, 0x7B, 0x78, 0x78, 0x78,
            0x75, 0x75, 0x79, 0x7B, 0x7B, 0x79, 0x7A, 0x7A, 0x75, 0x74, 0x78, 0x7B, 0x7B, 0x78, 0x79, 0x79, 0x75, 0x74,
            0x77, 0x79, 0x7B, 0x77, 0x76, 0x77, 0x74, 0x73, 0x76, 0x76,
        ];

        let v: [i16; 64] = [
            0x99, 0x88, 0x7F, 0x82, 0x8F, 0x95, 0x89, 0x83, 0x9A, 0x8A, 0x7E, 0x82, 0x91, 0x98, 0x8E, 0x84, 0xA2, 0x93,
            0x83, 0x87, 0x98, 0xA0, 0x99, 0x8A, 0xA1, 0x96, 0x87, 0x8B, 0x97, 0x9E, 0x9B, 0x8B, 0x8F, 0x8E, 0x8A, 0x8C,
            0x8D, 0x8E, 0x8E, 0x83, 0x80, 0x88, 0x91, 0x92, 0x87, 0x81, 0x84, 0x81, 0x81, 0x83, 0x90, 0x90, 0x84, 0x80,
            0x82, 0x83, 0x85, 0x7D, 0x85, 0x85, 0x7E, 0x7F, 0x80, 0x81,
        ];

        let mut image = RgbData::new(16, 16);
        b.iter(|| image.write_yuv420_mcu(0, 0, &y, &u, &v));
    }

    #[bench]
    fn iterative_yuv420_rgb_fast(b: &mut test::Bencher) {
        let mut y: [i16; 256] = [0; 256];
        let block_1: [i16; 64] = [
            0xA4, 0xA5, 0x82, 0x6C, 0x58, 0x45, 0x44, 0x2E, 0x65, 0x83, 0x72, 0x78, 0x6F, 0x53, 0x4D, 0x27, 0x2F, 0x35,
            0x32, 0x5A, 0x6E, 0x5B, 0x5B, 0x4E, 0x32, 0x0F, 0x08, 0x24, 0x36, 0x31, 0x4A, 0x80, 0x42, 0x24, 0x0E, 0x1A,
            0x2C, 0x22, 0x2E, 0x61, 0x76, 0x72, 0x42, 0x44, 0x53, 0x3B, 0x3D, 0x44, 0x77, 0x92, 0x77, 0x6E, 0x59, 0x41,
            0x5D, 0x5E, 0x42, 0x64, 0x6F, 0x5F, 0x32, 0x2C, 0x58, 0x5B,
        ];
        let block_2: [i16; 64] = [
            0x28, 0x44, 0x5E, 0x74, 0x4C, 0x40, 0x35, 0x1E, 0x1E, 0x2F, 0x2C, 0x30, 0x2D, 0x37, 0x20, 0x1D, 0x3A, 0x2C,
            0x16, 0x0C, 0x1C, 0x2C, 0x10, 0x22, 0x70, 0x5D, 0x59, 0x38, 0x20, 0x1D, 0x16, 0x2D, 0x61, 0x63, 0x8D, 0x78,
            0x3C, 0x15, 0x22, 0x41, 0x3C, 0x3F, 0x70, 0x80, 0x55, 0x2C, 0x33, 0x47, 0x60, 0x54, 0x5D, 0x6B, 0x4F, 0x52,
            0x47, 0x31, 0x90, 0x79, 0x6A, 0x6F, 0x46, 0x65, 0x52, 0x22,
        ];
        let block_3: [i16; 64] = [
            0x26, 0x39, 0x54, 0x55, 0x2D, 0x2C, 0x3D, 0x4D, 0x12, 0x32, 0x48, 0x48, 0x34, 0x39, 0x4E, 0x5B, 0x01, 0x0E,
            0x2B, 0x45, 0x50, 0x54, 0x5C, 0x52, 0x20, 0x21, 0x3F, 0x6D, 0x82, 0x71, 0x71, 0x5F, 0x64, 0x68, 0x57, 0x66,
            0x7D, 0x73, 0x98, 0x9A, 0x7A, 0x80, 0x48, 0x34, 0x40, 0x4A, 0x8F, 0x8C, 0x48, 0x54, 0x3E, 0x36, 0x38, 0x39,
            0x5F, 0x55, 0x36, 0x3B, 0x45, 0x51, 0x5C, 0x4A, 0x3E, 0x44,
        ];
        let block_4: [i16; 64] = [
            0x86, 0x89, 0x86, 0x5E, 0x58, 0x7E, 0x56, 0x21, 0x6E, 0x67, 0x4F, 0x45, 0x63, 0x7D, 0x5B, 0x30, 0x4C, 0x57,
            0x45, 0x45, 0x5F, 0x65, 0x5F, 0x4D, 0x49, 0x65, 0x5F, 0x48, 0x48, 0x49, 0x5E, 0x68, 0x56, 0x64, 0x60, 0x45,
            0x3D, 0x2D, 0x3D, 0x6D, 0x3E, 0x4A, 0x53, 0x4B, 0x50, 0x3A, 0x2D, 0x5C, 0x2C, 0x34, 0x43, 0x4A, 0x50, 0x52,
            0x41, 0x3F, 0x24, 0x0C, 0x1C, 0x3C, 0x33, 0x46, 0x5A, 0x4E,
        ];
        y[0..64].copy_from_slice(&block_1);
        y[64..128].copy_from_slice(&block_2);
        y[128..192].copy_from_slice(&block_3);
        y[192..256].copy_from_slice(&block_4);

        let u: [i16; 64] = [
            0x7E, 0x79, 0x77, 0x78, 0x77, 0x79, 0x7E, 0x80, 0x80, 0x79, 0x77, 0x78, 0x79, 0x7B, 0x7F, 0x7F, 0x80, 0x79,
            0x76, 0x78, 0x79, 0x7B, 0x7E, 0x7D, 0x7D, 0x78, 0x76, 0x77, 0x77, 0x78, 0x7B, 0x7B, 0x7B, 0x78, 0x78, 0x78,
            0x75, 0x75, 0x79, 0x7B, 0x7B, 0x79, 0x7A, 0x7A, 0x75, 0x74, 0x78, 0x7B, 0x7B, 0x78, 0x79, 0x79, 0x75, 0x74,
            0x77, 0x79, 0x7B, 0x77, 0x76, 0x77, 0x74, 0x73, 0x76, 0x76,
        ];

        let v: [i16; 64] = [
            0x99, 0x88, 0x7F, 0x82, 0x8F, 0x95, 0x89, 0x83, 0x9A, 0x8A, 0x7E, 0x82, 0x91, 0x98, 0x8E, 0x84, 0xA2, 0x93,
            0x83, 0x87, 0x98, 0xA0, 0x99, 0x8A, 0xA1, 0x96, 0x87, 0x8B, 0x97, 0x9E, 0x9B, 0x8B, 0x8F, 0x8E, 0x8A, 0x8C,
            0x8D, 0x8E, 0x8E, 0x83, 0x80, 0x88, 0x91, 0x92, 0x87, 0x81, 0x84, 0x81, 0x81, 0x83, 0x90, 0x90, 0x84, 0x80,
            0x82, 0x83, 0x85, 0x7D, 0x85, 0x85, 0x7E, 0x7F, 0x80, 0x81,
        ];

        let mut image = RgbData::new(16, 16);
        b.iter(|| image.write_yuv420_mcu_fast(0, 0, &y, &u, &v));
    }
}
