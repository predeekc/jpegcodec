#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// default assumes SSE2 support (It was around in 2001 and Windows 8 requires SSE2 so I think this goes back far enough)
pub fn transform_to_pixels(coef: &[i16; 64], pixels: &mut [i16]) {
    assert_eq!(pixels.len(), 64);
    if is_x86_feature_detected!("avx2") {
        unsafe {
            idct_avx2(coef, pixels);
        }
    } else if is_x86_feature_detected!("sse2") {
        unsafe {
            idct_sse2(coef, pixels);
        }
    } else {
        idct_fast(coef, pixels);
    }
}

const INV_SQRT_8: f32 = 0.353553390593274;
const SQRT_2: f32 = 1.414213562373095;
const SQRT_2_COS_2: f32 = SQRT_2 * 0.923879532511287;
const SQRT_2_SIN_2: f32 = SQRT_2 * 0.38268343236509;
const COS_1: f32 = 0.98078528040323; // cos(PI/16)
const COS_3: f32 = 0.831469612302545; // cos(3PI/16)
const SIN_1: f32 = 0.195090322016128; // sin(PI/16)
const SIN_3: f32 = 0.555570233019602; // sin(3PI/16)

fn idct_fast(coef: &[i16; 64], result: &mut [i16]) {
    fn dct_1d(coef: &[f32; 8]) -> [f32; 8] {
        let mut a: [f32; 8] = [0.0; 8];
        a[0] = coef[0] * INV_SQRT_8;
        a[1] = coef[4] * INV_SQRT_8;
        a[2] = coef[2] * INV_SQRT_8;
        a[3] = coef[6] * INV_SQRT_8;
        a[4] = (coef[1] - coef[7]) * INV_SQRT_8;
        a[5] = coef[3] * SQRT_2 * INV_SQRT_8;
        a[6] = coef[5] * SQRT_2 * INV_SQRT_8;
        a[7] = (coef[7] + coef[1]) * INV_SQRT_8;

        let mut b: [f32; 8] = [0.0; 8];
        b[0] = a[0] + a[1];
        b[1] = a[0] - a[1];
        b[2] = a[2] * SQRT_2_SIN_2 - a[3] * SQRT_2_COS_2;
        b[3] = a[2] * SQRT_2_COS_2 + a[3] * SQRT_2_SIN_2;
        b[4] = a[4] + a[6];
        b[5] = a[7] - a[5];
        b[6] = a[4] - a[6];
        b[7] = a[7] + a[5];

        let mut c: [f32; 8] = [0.0; 8];
        c[0] = b[0] + b[3];
        c[1] = b[1] + b[2];
        c[2] = b[1] - b[2];
        c[3] = b[0] - b[3];
        c[4] = b[4] * COS_3 - b[7] * SIN_3;
        c[5] = b[5] * COS_1 - b[6] * SIN_1;
        c[6] = b[5] * SIN_1 + b[6] * COS_1;
        c[7] = b[4] * SIN_3 + b[7] * COS_3;

        let mut res: [f32; 8] = [0.0; 8];
        res[0] = c[0] + c[7];
        res[1] = c[1] + c[6];
        res[2] = c[2] + c[5];
        res[3] = c[3] + c[4];
        res[4] = c[3] - c[4];
        res[5] = c[2] - c[5];
        res[6] = c[1] - c[6];
        res[7] = c[0] - c[7];

        res
    }

    let mut temp1: [f32; 64] = [0.0; 64];
    let mut temp2: [f32; 64] = [0.0; 64];
    let mut row: [f32; 8] = [0.0; 8];
    let mut col: [f32; 8] = [0.0; 8];

    for x in 0..8 {
        for i in 0..8 {
            col[i] = coef[i * 8 + x] as f32;
        }
        let pixels = dct_1d(&col);
        for i in 0..8 {
            temp1[i * 8 + x] = pixels[i];
        }
    }

    for y in 0..8 {
        for i in 0..8 {
            row[i] = temp1[y * 8 + i];
        }
        let pixels = dct_1d(&row);
        for i in 0..8 {
            temp2[y * 8 + i] = pixels[i];
        }
    }

    for i in 0..64 {
        result[i] = (temp2[i].round() + 128.0).clamp(0.0, 255.0) as i16;
    }
}

#[allow(dead_code)]
fn idct_baseline(coef: &[i16; 64], result: &mut [i16]) {
    let mut temp: [f32; 64] = [0.0; 64];

    fn c(i: usize) -> f32 {
        if i != 0 {
            1.0
        } else {
            0.70710678118654752440084436210485
        } // 1/sqrt(2)
    }

    fn cs_m_c(a: usize, b: usize) -> f32 {
        (c(b) as f32) * (((2.0 * (a as f32) + 1.0) * (b as f32) * std::f32::consts::PI) / 16.0).cos()
    }

    for x in 0..8 {
        let mut index = x;
        for i in 0..8 {
            let mut val: f32 = 0.0;
            for u in 0..8 {
                val += cs_m_c(i, u) * coef[(u * 8 + x) as usize] as f32;
            }
            temp[index] = val * 0.5;
            index += 8;
        }
    }

    for y in 0..8 {
        let mut index = y * 8;
        for i in 0..8 {
            let mut val: f32 = 0.0;
            for u in 0..8 {
                val += cs_m_c(i, u) * temp[(y * 8 + u) as usize];
            }
            result[index] = (val * 0.5).round() as i16;
            index += 1;
        }
    }

    for i in 0..64 {
        result[i] = (result[i] + 128).clamp(0, 255);
    }
}

#[target_feature(enable = "sse,sse3,sse4.1,avx,avx2")]
unsafe fn idct_avx2(coef: &[i16; 64], pixels: &mut [i16]) {
    #[inline(always)]
    unsafe fn _mm256_set_ps1(v: f32) -> __m256 {
        _mm256_set_ps(v, v, v, v, v, v, v, v)
    }
    #[inline(always)]
    unsafe fn load_ps_from_i16(src: *const i16, index: isize) -> __m256 {
        let line = _mm_loadu_si128(src.offset(index).cast::<__m128i>());
        let result = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(line));
        result
    }

    #[inline(always)]
    unsafe fn calc_dct_1d(f: [__m256; 8]) -> [__m256; 8] {
        let inv_s8 = _mm256_set_ps1(0.353553390593274);
        let s2 = _mm256_set_ps1(1.414213562373095);
        let cos2 = _mm256_set_ps1(1.414213562373095 * 0.923879532511287);
        let sin2 = _mm256_set_ps1(1.414213562373095 * 0.38268343236509);
        let cos1 = _mm256_set_ps1(0.98078528040323);
        let sin1 = _mm256_set_ps1(0.195090322016128);
        let cos3 = _mm256_set_ps1(0.831469612302545);
        let sin3 = _mm256_set_ps1(0.555570233019602);

        let f0 = _mm256_mul_ps(inv_s8, f[0]);
        let f1 = _mm256_mul_ps(inv_s8, f[1]);
        let f2 = _mm256_mul_ps(inv_s8, f[2]);
        let f3 = _mm256_mul_ps(inv_s8, f[3]);
        let f4 = _mm256_mul_ps(inv_s8, f[4]);
        let f5 = _mm256_mul_ps(inv_s8, f[5]);
        let f6 = _mm256_mul_ps(inv_s8, f[6]);
        let f7 = _mm256_mul_ps(inv_s8, f[7]);

        let a0 = f0;
        let a1 = f4;
        let a2 = f2;
        let a3 = f6;
        let a4 = _mm256_sub_ps(f1, f7);
        let a5 = _mm256_mul_ps(s2, f3);
        let a6 = _mm256_mul_ps(s2, f5);
        let a7 = _mm256_add_ps(f1, f7);

        let b0 = _mm256_add_ps(a0, a1);
        let b1 = _mm256_sub_ps(a0, a1);
        let b2 = _mm256_sub_ps(_mm256_mul_ps(sin2, a2), _mm256_mul_ps(cos2, a3));
        let b3 = _mm256_add_ps(_mm256_mul_ps(cos2, a2), _mm256_mul_ps(sin2, a3));
        let b4 = _mm256_add_ps(a4, a6);
        let b5 = _mm256_sub_ps(a7, a5);
        let b6 = _mm256_sub_ps(a4, a6);
        let b7 = _mm256_add_ps(a5, a7);

        let c0 = _mm256_add_ps(b0, b3);
        let c1 = _mm256_add_ps(b1, b2);
        let c2 = _mm256_sub_ps(b1, b2);
        let c3 = _mm256_sub_ps(b0, b3);
        let c4 = _mm256_sub_ps(_mm256_mul_ps(cos3, b4), _mm256_mul_ps(sin3, b7));
        let c5 = _mm256_sub_ps(_mm256_mul_ps(cos1, b5), _mm256_mul_ps(sin1, b6));
        let c6 = _mm256_add_ps(_mm256_mul_ps(sin1, b5), _mm256_mul_ps(cos1, b6));
        let c7 = _mm256_add_ps(_mm256_mul_ps(sin3, b4), _mm256_mul_ps(cos3, b7));

        return [
            _mm256_add_ps(c0, c7),
            _mm256_add_ps(c1, c6),
            _mm256_add_ps(c2, c5),
            _mm256_add_ps(c3, c4),
            _mm256_sub_ps(c3, c4),
            _mm256_sub_ps(c2, c5),
            _mm256_sub_ps(c1, c6),
            _mm256_sub_ps(c0, c7),
        ];
    }

    #[inline(always)]
    unsafe fn convert_to_pixel(data: __m256) -> __m256i {
        let pixel_shift = _mm256_set_ps1(128.0);
        let pixel_min = _mm256_set_ps1(0.0);
        let pixel_max = _mm256_set_ps1(255.0);

        let pixel = _mm256_cvtps_epi32(_mm256_min_ps(
            pixel_max,
            _mm256_max_ps(
                pixel_min,
                _mm256_add_ps(pixel_shift, _mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(data)),
            ),
        ));

        let vperm = _mm256_setr_epi8(
            0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1,
            -1, -1,
        );
        return _mm256_permute4x64_epi64(_mm256_shuffle_epi8(pixel, vperm), 0b00001000);
    }

    #[inline(always)]
    unsafe fn transpose(f: [__m256; 8]) -> [__m256; 8] {
        let t0 = _mm256_unpacklo_ps(f[0], f[1]);
        let t1 = _mm256_unpackhi_ps(f[0], f[1]);
        let t2 = _mm256_unpacklo_ps(f[2], f[3]);
        let t3 = _mm256_unpackhi_ps(f[2], f[3]);
        let t4 = _mm256_unpacklo_ps(f[4], f[5]);
        let t5 = _mm256_unpackhi_ps(f[4], f[5]);
        let t6 = _mm256_unpacklo_ps(f[6], f[7]);
        let t7 = _mm256_unpackhi_ps(f[6], f[7]);

        let r0 = _mm256_shuffle_ps::<0b01_00_01_00>(t0, t2);
        let r1 = _mm256_shuffle_ps::<0b11_10_11_10>(t0, t2);
        let r2 = _mm256_shuffle_ps::<0b01_00_01_00>(t1, t3);
        let r3 = _mm256_shuffle_ps::<0b11_10_11_10>(t1, t3);
        let r4 = _mm256_shuffle_ps::<0b01_00_01_00>(t4, t6);
        let r5 = _mm256_shuffle_ps::<0b11_10_11_10>(t4, t6);
        let r6 = _mm256_shuffle_ps::<0b01_00_01_00>(t5, t7);
        let r7 = _mm256_shuffle_ps::<0b11_10_11_10>(t5, t7);

        return [
            _mm256_permute2f128_ps(r0, r4, 0x20),
            _mm256_permute2f128_ps(r1, r5, 0x20),
            _mm256_permute2f128_ps(r2, r6, 0x20),
            _mm256_permute2f128_ps(r3, r7, 0x20),
            _mm256_permute2f128_ps(r0, r4, 0x31),
            _mm256_permute2f128_ps(r1, r5, 0x31),
            _mm256_permute2f128_ps(r2, r6, 0x31),
            _mm256_permute2f128_ps(r3, r7, 0x31),
        ];
    }

    let coef_ptr = coef.as_ptr();
    let rows = transpose(calc_dct_1d([
        load_ps_from_i16(coef_ptr, 0),
        load_ps_from_i16(coef_ptr, 8),
        load_ps_from_i16(coef_ptr, 16),
        load_ps_from_i16(coef_ptr, 24),
        load_ps_from_i16(coef_ptr, 32),
        load_ps_from_i16(coef_ptr, 40),
        load_ps_from_i16(coef_ptr, 48),
        load_ps_from_i16(coef_ptr, 56),
    ]));

    let results = transpose(calc_dct_1d(rows));
    for j in 0..8 {
        let result = convert_to_pixel(results[j]);
        let cast_result = _mm256_castsi256_si128(result);
        _mm_storeu_si128(
            pixels.as_mut_ptr().offset((j * 8) as isize).cast::<__m128i>(),
            cast_result,
        );
    }
}

#[target_feature(enable = "sse,sse2")]
unsafe fn idct_sse2(coef: &[i16; 64], pixels: &mut [i16]) {
    #[inline(always)]
    unsafe fn calc_dct_1d(f: [__m128; 8]) -> [__m128; 8] {
        let inv_s8 = _mm_set_ps1(0.353553390593274);
        let s2 = _mm_set_ps1(1.414213562373095);
        let cos2 = _mm_set_ps1(1.414213562373095 * 0.923879532511287);
        let sin2 = _mm_set_ps1(1.414213562373095 * 0.38268343236509);
        let cos1 = _mm_set_ps1(0.98078528040323);
        let sin1 = _mm_set_ps1(0.195090322016128);
        let cos3 = _mm_set_ps1(0.831469612302545);
        let sin3 = _mm_set_ps1(0.555570233019602);

        let f0 = _mm_mul_ps(inv_s8, f[0]);
        let f1 = _mm_mul_ps(inv_s8, f[1]);
        let f2 = _mm_mul_ps(inv_s8, f[2]);
        let f3 = _mm_mul_ps(inv_s8, f[3]);
        let f4 = _mm_mul_ps(inv_s8, f[4]);
        let f5 = _mm_mul_ps(inv_s8, f[5]);
        let f6 = _mm_mul_ps(inv_s8, f[6]);
        let f7 = _mm_mul_ps(inv_s8, f[7]);

        let a0 = f0;
        let a1 = f4;
        let a2 = f2;
        let a3 = f6;
        let a4 = _mm_sub_ps(f1, f7);
        let a5 = _mm_mul_ps(s2, f3);
        let a6 = _mm_mul_ps(s2, f5);
        let a7 = _mm_add_ps(f1, f7);

        let b0 = _mm_add_ps(a0, a1);
        let b1 = _mm_sub_ps(a0, a1);
        let b2 = _mm_sub_ps(_mm_mul_ps(sin2, a2), _mm_mul_ps(cos2, a3));
        let b3 = _mm_add_ps(_mm_mul_ps(cos2, a2), _mm_mul_ps(sin2, a3));
        let b4 = _mm_add_ps(a4, a6);
        let b5 = _mm_sub_ps(a7, a5);
        let b6 = _mm_sub_ps(a4, a6);
        let b7 = _mm_add_ps(a5, a7);

        let c0 = _mm_add_ps(b0, b3);
        let c1 = _mm_add_ps(b1, b2);
        let c2 = _mm_sub_ps(b1, b2);
        let c3 = _mm_sub_ps(b0, b3);
        let c4 = _mm_sub_ps(_mm_mul_ps(cos3, b4), _mm_mul_ps(sin3, b7));
        let c5 = _mm_sub_ps(_mm_mul_ps(cos1, b5), _mm_mul_ps(sin1, b6));
        let c6 = _mm_add_ps(_mm_mul_ps(sin1, b5), _mm_mul_ps(cos1, b6));
        let c7 = _mm_add_ps(_mm_mul_ps(sin3, b4), _mm_mul_ps(cos3, b7));

        return [
            _mm_add_ps(c0, c7),
            _mm_add_ps(c1, c6),
            _mm_add_ps(c2, c5),
            _mm_add_ps(c3, c4),
            _mm_sub_ps(c3, c4),
            _mm_sub_ps(c2, c5),
            _mm_sub_ps(c1, c6),
            _mm_sub_ps(c0, c7),
        ];
    }

    #[inline(always)]
    unsafe fn transpose(left: [__m128; 8], right: [__m128; 8]) -> [[__m128; 8]; 2] {
        // upper left
        let r0 = left[0];
        let r1 = left[1];
        let r2 = left[2];
        let r3 = left[3];
        let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
        let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
        let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
        let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
        let out_lo_0 = _mm_shuffle_ps(tmp0, tmp1, 0x88);
        let out_lo_1 = _mm_shuffle_ps(tmp0, tmp1, 0xdd);
        let out_lo_2 = _mm_shuffle_ps(tmp2, tmp3, 0x88);
        let out_lo_3 = _mm_shuffle_ps(tmp2, tmp3, 0xdd);

        // upper right quadrant
        let r0 = left[4];
        let r1 = left[5];
        let r2 = left[6];
        let r3 = left[7];
        let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
        let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
        let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
        let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
        let out_hi_0 = _mm_shuffle_ps(tmp0, tmp1, 0x88);
        let out_hi_1 = _mm_shuffle_ps(tmp0, tmp1, 0xdd);
        let out_hi_2 = _mm_shuffle_ps(tmp2, tmp3, 0x88);
        let out_hi_3 = _mm_shuffle_ps(tmp2, tmp3, 0xdd);

        // lower left quadrant
        let r0 = right[0];
        let r1 = right[1];
        let r2 = right[2];
        let r3 = right[3];
        let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
        let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
        let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
        let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
        let out_lo_4 = _mm_shuffle_ps(tmp0, tmp1, 0x88);
        let out_lo_5 = _mm_shuffle_ps(tmp0, tmp1, 0xdd);
        let out_lo_6 = _mm_shuffle_ps(tmp2, tmp3, 0x88);
        let out_lo_7 = _mm_shuffle_ps(tmp2, tmp3, 0xdd);

        // lower right quadrant
        let r0 = right[4];
        let r1 = right[5];
        let r2 = right[6];
        let r3 = right[7];
        let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
        let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
        let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
        let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
        let out_hi_4 = _mm_shuffle_ps(tmp0, tmp1, 0x88);
        let out_hi_5 = _mm_shuffle_ps(tmp0, tmp1, 0xdd);
        let out_hi_6 = _mm_shuffle_ps(tmp2, tmp3, 0x88);
        let out_hi_7 = _mm_shuffle_ps(tmp2, tmp3, 0xdd);

        return [
            [
                out_lo_0, out_lo_1, out_lo_2, out_lo_3, out_lo_4, out_lo_5, out_lo_6, out_lo_7,
            ],
            [
                out_hi_0, out_hi_1, out_hi_2, out_hi_3, out_hi_4, out_hi_5, out_hi_6, out_hi_7,
            ],
        ];
    }

    #[inline(always)]
    unsafe fn load_ps_from_i16(src: *const i16, index: isize) -> __m128 {
        return _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si64(src.offset(index).cast::<u8>())));
    }

    #[inline(always)]
    unsafe fn convert_to_pixel(data: __m128) -> __m128i {
        let pixel_shift = _mm_set_ps1(128.0);
        let pixel_min = _mm_set_ps1(0.0);
        let pixel_max = _mm_set_ps1(255.0);

        let d = _mm_cvtps_epi32(_mm_min_ps(
            pixel_max,
            _mm_max_ps(
                pixel_min,
                _mm_add_ps(pixel_shift, _mm_round_ps::<_MM_FROUND_TO_NEAREST_INT>(data)),
            ),
        ));

        let vperm = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
        _mm_shuffle_epi8(d, vperm)
    }

    let coef_ptr = coef.as_ptr();
    let rows = transpose(
        calc_dct_1d([
            load_ps_from_i16(coef_ptr, 0),
            load_ps_from_i16(coef_ptr, 8),
            load_ps_from_i16(coef_ptr, 16),
            load_ps_from_i16(coef_ptr, 24),
            load_ps_from_i16(coef_ptr, 32),
            load_ps_from_i16(coef_ptr, 40),
            load_ps_from_i16(coef_ptr, 48),
            load_ps_from_i16(coef_ptr, 56),
        ]),
        calc_dct_1d([
            load_ps_from_i16(coef_ptr, 0 + 4),
            load_ps_from_i16(coef_ptr, 8 + 4),
            load_ps_from_i16(coef_ptr, 16 + 4),
            load_ps_from_i16(coef_ptr, 24 + 4),
            load_ps_from_i16(coef_ptr, 32 + 4),
            load_ps_from_i16(coef_ptr, 40 + 4),
            load_ps_from_i16(coef_ptr, 48 + 4),
            load_ps_from_i16(coef_ptr, 56 + 4),
        ]),
    );

    let results = transpose(calc_dct_1d(rows[0]), calc_dct_1d(rows[1]));

    for i in 0..2 {
        let shift = i * 4;
        for j in 0..8 {
            _mm_storel_epi64(
                pixels.as_mut_ptr().offset((j * 8 + shift) as isize).cast::<__m128i>(),
                convert_to_pixel(results[i][j]),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    const COEF: [i16; 64] = [
        -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10, 10,
        6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,
    ];

    const EXPECTED: [i16; 64] = [
        48, 83, 79, 27, 32, 33, 31, 34, 39, 86, 70, 23, 24, 31, 33, 25, 45, 88, 52, 18, 21, 35, 36, 23, 74, 83, 32, 20,
        26, 38, 34, 26, 98, 60, 21, 31, 29, 32, 28, 28, 86, 28, 21, 39, 25, 26, 27, 31, 53, 15, 31, 33, 21, 25, 32, 33,
        32, 21, 43, 23, 20, 28, 33, 31,
    ];

    #[test]
    fn test_idct_baseline() {
        let mut pixels: [i16; 64] = [0; 64];
        idct_baseline(&COEF, &mut pixels);
        assert_eq!(pixels, EXPECTED);
    }

    #[test]
    fn test_idct_sse2() {
        let mut pixels: [i16; 64] = [0; 64];
        unsafe {
            idct_sse2(&COEF, &mut pixels);
        }

        assert_eq!(pixels, EXPECTED);
    }

    #[test]
    fn test_idct_avx2() {
        let mut pixels: [i16; 64] = [0; 64];
        unsafe {
            idct_avx2(&COEF, &mut pixels);
        }

        assert_eq!(pixels, EXPECTED);
    }

    #[bench]
    fn bench_idct_baseline(b: &mut test::Bencher) {
        let mut pixels: [i16; 64] = [0; 64];
        b.iter(|| idct_baseline(&COEF, &mut pixels));
    }

    #[bench]
    fn bench_idct_fast(b: &mut test::Bencher) {
        let mut pixels: [i16; 64] = [0; 64];
        b.iter(|| idct_fast(&COEF, &mut pixels));
    }

    #[bench]
    fn bench_idct_sse2(b: &mut test::Bencher) {
        let mut pixels: [i16; 64] = [0; 64];
        b.iter(|| unsafe {
            idct_sse2(&COEF, &mut pixels);
        });
    }

    #[bench]
    fn bench_idct_avx2(b: &mut test::Bencher) {
        let mut pixels: [i16; 64] = [0; 64];
        b.iter(|| unsafe {
            idct_avx2(&COEF, &mut pixels);
        });
    }
}
