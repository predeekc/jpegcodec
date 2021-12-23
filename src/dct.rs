#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::f64;

pub struct Dct {
    dct_matrix: [[f64; 8]; 8],
    dct_t_matrix: [[f64; 8]; 8],

    cs_table: [[f64; 8]; 8],
    cs_table_32: [[f32; 8]; 8],

    f_a: f32,
    f_b: f32,
    f_c: f32,
    f_d: f32,
    f_e: f32,
    f_f: f32,
    f_g: f32,
}

impl Dct {
    pub fn new() -> Self {
        fn c(i: usize) -> f64 {
            if i != 0 {
                1.0
            } else {
                0.70710678118654752440084436210485
            } // 1/sqrt(2)
        }

        fn cs_m_c(a: usize, b: usize) -> f64 {
            (c(b) as f64) * (((2.0 * (a as f64) + 1.0) * (b as f64) * std::f64::consts::PI) / 16.0).cos()
        }

        let mut cs_table: [[f64; 8]; 8] = [[0.0; 8]; 8];
        let mut cs_table_32: [[f32; 8]; 8] = [[0.0; 8]; 8];
        for x in 0..8 {
            for u in 0..8 {
                cs_table[x][u] = cs_m_c(x, u);
                cs_table_32[x][u] = cs_table[x][u] as f32;
            }
        }

        let mut dct_matrix = [[0.0; 8]; 8];
        let mut dct_t_matrix = [[0.0; 8]; 8];

        let val = 0.35355339059327376220042218105242; // 1.0 / (8.0).sqrt();
        for j in 0..8 {
            dct_matrix[0][j] = val;
            dct_t_matrix[j][0] = val;
        }

        let factor = 0.5; // (0.25).sqrt();
        for i in 1..8 {
            for j in 0..8 {
                let val = factor * (std::f64::consts::PI * i as f64 * (2.0 * j as f64 + 1.0) / 16.0).cos();
                dct_matrix[i][j] = val;
                dct_t_matrix[j][i] = val;
            }
        }

        Self {
            dct_matrix,
            dct_t_matrix,
            cs_table,
            cs_table_32,
            f_a: val as f32,
            f_b: 0.490392, //(val * (std::f64::consts::PI / 16.0).cos()) as f32,
            f_c: 0.415734, //(val * (std::f64::consts::PI * 3.0 / 16.0).cos()) as f32,
            f_d: 0.277785, //(val * (std::f64::consts::PI * 5.0 / 16.0).cos()) as f32,
            f_e: 0.097545, //(val * (std::f64::consts::PI * 7.0 / 16.0).cos()) as f32,
            f_f: 0.461939, // (val * (std::f64::consts::PI * 2.0 / 16.0).cos()) as f32,
            f_g: 0.191341, //(val * (std::f64::consts::PI * 6.0 / 16.0).cos()) as f32,
        }
    }

    pub fn idct_manual(&self, coef: &[i16; 64]) -> [i16; 64] {
        let mut result: [i16; 64] = [0; 64];

        let mut index = 0;
        for y in 0..8 {
            for x in 0..8 {
                let mut val = 0.0;
                for v in 0..8 {
                    let cs_yv = self.cs_table[y][v];
                    for u in 0..8 {
                        val += cs_yv * self.cs_table[x][u] * coef[(v * 8 + u) as usize] as f64;
                    }
                }
                result[index] = (val * 0.25).round() as i16;
                index += 1;
            }
        }

        result
    }

    pub fn idct_rows_cols(&self, coef: &[i16; 64]) -> [i16; 64] {
        let mut result: [i16; 64] = [0; 64];
        let mut temp: [f32; 64] = [0.0; 64];

        for x in 0..8 {
            let mut index = x;
            for i in 0..8 {
                let mut val = 0.0;
                for u in 0..8 {
                    val += self.cs_table_32[i][u] * coef[(u * 8 + x) as usize] as f32;
                }
                temp[index] = val * 0.5;
                index += 8;
            }
        }

        for y in 0..8 {
            let mut index = y * 8;
            for i in 0..8 {
                let mut val = 0.0;
                for u in 0..8 {
                    val += self.cs_table_32[i][u] * temp[(y * 8 + u) as usize];
                }
                result[index] = (val * 0.5).round() as i16;
                index += 1;
            }
        }

        result
    }

    // Fast DCT
    /*
                let cf0 = coef[x] as f32;
            let cf1 = coef[8 + x] as f32;
            let cf2 = coef[16 + x] as f32;
            let cf3 = coef[24 + x] as f32;
            let cf4 = coef[32 + x] as f32;
            let cf5 = coef[40 + x] as f32;
            let cf6 = coef[48 + x] as f32;
            let cf7 = coef[56 + x] as f32;

            let t0 = cf0 - cf7;
            let t1 = cf0 + cf7;
            let t2 = cf1 - cf6;
            let t3 = cf1 + cf6;
            let t4 = -cf1 + cf6;
            let t5 = -cf1 - cf6;
            let t6 = cf2 + cf5;
            let t7 = cf2 - cf5;
            let t8 = -cf2 + cf5;
            let t9 = cf2 - cf5;
            let t10 = cf3 + cf4;
            let t11 = cf3 - cf4;
            let t12 = -cf3 + cf4;
            let t13 = -cf3 - cf4;

            temp[x] = self.f_a * (t1 + t3 + t6 + t10);
            temp[x + 8] = self.f_b * t0 + self.f_c * t2 + self.f_d * t7 + self.f_e * t11;
            temp[x + 16] = self.f_f * (t0 + t13) + self.f_g * (t3 + t9);
            temp[x + 24] = self.f_c * t0 + self.f_e * t5 + self.f_b * t8 + self.f_d * t11;
            temp[x + 32] = self.f_a * (t1 + t5 + t9 + t10);
            temp[x + 40] = self.f_d * t0 + self.f_e * t4 + self.f_b * t8 + self.f_d * t11;
            temp[x + 48] = self.f_g * (t1 + t13) + self.f_f * (t5 + t6);
            temp[x + 54] = self.f_e * t0 + self.f_d * t4 + self.f_c * t7 + self.f_b * t12;
    */
    //#[inline(never)]
    pub fn idct_rows_cols_fast(&self, coef: &[i16; 64]) -> [i16; 64] {
        let mut result: [i16; 64] = [0; 64];
        let mut temp: [f32; 64] = [0.0; 64];

        for x in 0..8 {
            let cf0 = coef[x] as f32;
            let cf1 = coef[8 + x] as f32;
            let cf2 = coef[16 + x] as f32;
            let cf3 = coef[24 + x] as f32;
            let cf4 = coef[32 + x] as f32;
            let cf5 = coef[40 + x] as f32;
            let cf6 = coef[48 + x] as f32;
            let cf7 = coef[56 + x] as f32;

            let a_p = self.f_a * (cf0 + cf4);
            let a_n = self.f_a * (cf0 - cf4);

            let cf1_b = self.f_b * cf1;
            let cf1_c = self.f_c * cf1;
            let cf1_d = self.f_d * cf1;
            let cf1_e = self.f_e * cf1;

            let cf2_f = self.f_f * cf2;
            let cf2_g = self.f_g * cf2;

            let cf3_b = self.f_b * cf3;
            let cf3_c = self.f_c * cf3;
            let cf3_d = self.f_d * cf3;
            let cf3_e = self.f_e * cf3;

            let cf5_b = self.f_b * cf5;
            let cf5_c = self.f_c * cf5;
            let cf5_d = self.f_d * cf5;
            let cf5_e = self.f_e * cf5;

            let cf6_f = self.f_f * cf6;
            let cf6_g = self.f_g * cf6;

            let cf7_b = self.f_b * cf7;
            let cf7_c = self.f_c * cf7;
            let cf7_d = self.f_d * cf7;
            let cf7_e = self.f_e * cf7;

            temp[x] = a_p + cf1_b + cf2_f + cf3_c + cf5_d + cf6_g + cf7_e;
            temp[x + 8] = a_n + cf1_c + cf2_g - cf3_e - cf5_b - cf6_f - cf7_d;
            temp[x + 16] = a_n + cf1_d - cf2_g - cf3_b + cf5_e + cf6_f + cf7_c;
            temp[x + 24] = a_p + cf1_e - cf2_f - cf3_d + cf5_c - cf6_g - cf7_b;
            temp[x + 32] = a_p - cf1_e - cf2_f + cf3_d - cf5_c - cf6_g + cf7_b;
            temp[x + 40] = a_n - cf1_d - cf2_g + cf3_b - cf5_e + cf6_f - cf7_c;
            temp[x + 48] = a_n - cf1_c + cf2_g + cf3_e + cf5_b - cf6_f + cf7_d;
            temp[x + 56] = a_p - cf1_b + cf2_f - cf3_c - cf5_d + cf6_g - cf7_e;
        }

        for y in 0..8 {
            let offset = y * 8;
            let cf0 = temp[offset + 0];
            let cf1 = temp[offset + 1];
            let cf2 = temp[offset + 2];
            let cf3 = temp[offset + 3];
            let cf4 = temp[offset + 4];
            let cf5 = temp[offset + 5];
            let cf6 = temp[offset + 6];
            let cf7 = temp[offset + 7];

            let a_p = self.f_a * (cf0 + cf4);
            let a_n = self.f_a * (cf0 - cf4);

            let cf1_b = self.f_b * cf1;
            let cf1_c = self.f_c * cf1;
            let cf1_d = self.f_d * cf1;
            let cf1_e = self.f_e * cf1;

            let cf2_f = self.f_f * cf2;
            let cf2_g = self.f_g * cf2;

            let cf3_b = self.f_b * cf3;
            let cf3_c = self.f_c * cf3;
            let cf3_d = self.f_d * cf3;
            let cf3_e = self.f_e * cf3;

            let cf5_b = self.f_b * cf5;
            let cf5_c = self.f_c * cf5;
            let cf5_d = self.f_d * cf5;
            let cf5_e = self.f_e * cf5;

            let cf6_f = self.f_f * cf6;
            let cf6_g = self.f_g * cf6;

            let cf7_b = self.f_b * cf7;
            let cf7_c = self.f_c * cf7;
            let cf7_d = self.f_d * cf7;
            let cf7_e = self.f_e * cf7;

            result[offset + 0] = (a_p + cf1_b + cf2_f + cf3_c + cf5_d + cf6_g + cf7_e).round() as i16;
            result[offset + 1] = (a_n + cf1_c + cf2_g - cf3_e - cf5_b - cf6_f - cf7_d).round() as i16;
            result[offset + 2] = (a_n + cf1_d - cf2_g - cf3_b + cf5_e + cf6_f + cf7_c).round() as i16;
            result[offset + 3] = (a_p + cf1_e - cf2_f - cf3_d + cf5_c - cf6_g - cf7_b).round() as i16;
            result[offset + 4] = (a_p - cf1_e - cf2_f + cf3_d - cf5_c - cf6_g + cf7_b).round() as i16;
            result[offset + 5] = (a_n - cf1_d - cf2_g + cf3_b - cf5_e + cf6_f - cf7_c).round() as i16;
            result[offset + 6] = (a_n - cf1_c + cf2_g + cf3_e + cf5_b - cf6_f + cf7_d).round() as i16;
            result[offset + 7] = (a_p - cf1_b + cf2_f - cf3_c - cf5_d + cf6_g - cf7_e).round() as i16;
        }

        result
    }

    #[inline(never)]
    pub fn idct_rows_cols_fast2(&self, coef: &[i16; 64]) -> [i16; 64] {
        let mut temp: [[f32; 8]; 8] = [[0.0; 8]; 8];

        let mut cf: [[f32; 8]; 8] = [[0.0; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                cf[i][j] = coef[i * 8 + j] as f32;
            }
        }

        for x in 0..8 {
            let a_p = self.f_a * (cf[0][x] + cf[4][x]);
            let a_n = self.f_a * (cf[0][x] - cf[4][x]);

            let cf1_b = self.f_b * cf[1][x];
            let cf1_c = self.f_c * cf[1][x];
            let cf1_d = self.f_d * cf[1][x];
            let cf1_e = self.f_e * cf[1][x];

            let cf2_f = self.f_f * cf[2][x];
            let cf2_g = self.f_g * cf[2][x];

            let cf3_b = self.f_b * cf[3][x];
            let cf3_c = self.f_c * cf[3][x];
            let cf3_d = self.f_d * cf[3][x];
            let cf3_e = self.f_e * cf[3][x];

            let cf5_b = self.f_b * cf[5][x];
            let cf5_c = self.f_c * cf[5][x];
            let cf5_d = self.f_d * cf[5][x];
            let cf5_e = self.f_e * cf[5][x];

            let cf6_f = self.f_f * cf[6][x];
            let cf6_g = self.f_g * cf[6][x];

            let cf7_b = self.f_b * cf[7][x];
            let cf7_c = self.f_c * cf[7][x];
            let cf7_d = self.f_d * cf[7][x];
            let cf7_e = self.f_e * cf[7][x];

            // tranpose the results
            temp[x][0] = a_p + cf1_b + cf2_f + cf3_c + cf5_d + cf6_g + cf7_e;
            temp[x][1] = a_n + cf1_c + cf2_g - cf3_e - cf5_b - cf6_f - cf7_d;
            temp[x][2] = a_n + cf1_d - cf2_g - cf3_b + cf5_e + cf6_f + cf7_c;
            temp[x][3] = a_p + cf1_e - cf2_f - cf3_d + cf5_c - cf6_g - cf7_b;
            temp[x][4] = a_p - cf1_e - cf2_f + cf3_d - cf5_c - cf6_g + cf7_b;
            temp[x][5] = a_n - cf1_d - cf2_g + cf3_b - cf5_e + cf6_f - cf7_c;
            temp[x][6] = a_n - cf1_c + cf2_g + cf3_e + cf5_b - cf6_f + cf7_d;
            temp[x][7] = a_p - cf1_b + cf2_f - cf3_c - cf5_d + cf6_g - cf7_e;
        }

        for x in 0..8 {
            let a_p = self.f_a * (temp[0][x] + temp[4][x]);
            let a_n = self.f_a * (temp[0][x] - temp[4][x]);

            let cf1_b = self.f_b * temp[1][x];
            let cf1_c = self.f_c * temp[1][x];
            let cf1_d = self.f_d * temp[1][x];
            let cf1_e = self.f_e * temp[1][x];

            let cf2_f = self.f_f * temp[2][x];
            let cf2_g = self.f_g * temp[2][x];

            let cf3_b = self.f_b * temp[3][x];
            let cf3_c = self.f_c * temp[3][x];
            let cf3_d = self.f_d * temp[3][x];
            let cf3_e = self.f_e * temp[3][x];

            let cf5_b = self.f_b * temp[5][x];
            let cf5_c = self.f_c * temp[5][x];
            let cf5_d = self.f_d * temp[5][x];
            let cf5_e = self.f_e * temp[5][x];

            let cf6_f = self.f_f * temp[6][x];
            let cf6_g = self.f_g * temp[6][x];

            let cf7_b = self.f_b * temp[7][x];
            let cf7_c = self.f_c * temp[7][x];
            let cf7_d = self.f_d * temp[7][x];
            let cf7_e = self.f_e * temp[7][x];

            // transpose the results
            cf[x][0] = a_p + cf1_b + cf2_f + cf3_c + cf5_d + cf6_g + cf7_e;
            cf[x][1] = a_n + cf1_c + cf2_g - cf3_e - cf5_b - cf6_f - cf7_d;
            cf[x][2] = a_n + cf1_d - cf2_g - cf3_b + cf5_e + cf6_f + cf7_c;
            cf[x][3] = a_p + cf1_e - cf2_f - cf3_d + cf5_c - cf6_g - cf7_b;
            cf[x][4] = a_p - cf1_e - cf2_f + cf3_d - cf5_c - cf6_g + cf7_b;
            cf[x][5] = a_n - cf1_d - cf2_g + cf3_b - cf5_e + cf6_f - cf7_c;
            cf[x][6] = a_n - cf1_c + cf2_g + cf3_e + cf5_b - cf6_f + cf7_d;
            cf[x][7] = a_p - cf1_b + cf2_f - cf3_c - cf5_d + cf6_g - cf7_e;
        }

        let mut result: [i16; 64] = [0; 64];
        for j in 0..8 {
            for i in 0..8 {
                result[j * 8 + i] = cf[j][i].round() as i16;
            }
        }

        result
    }

    pub fn idct_rows_cols_faster(&self, coef: &[i16; 64]) -> [i16; 64] {
        #[target_feature(enable = "sse,sse2,sse4.1")]
        unsafe fn dct_cols(coef: *const i16, output: &mut [f32; 64]) {
            let mut temp_data: [f32; 64] = [0.0; 64];
            //let mut result_data: [f32; 64] = [0.0; 64];

            let factor_a: [f32; 4] = [0.35355339059327376220042218105242; 4];
            let factor_b: [f32; 4] = [0.490392; 4];
            let factor_c: [f32; 4] = [0.415734; 4];
            let factor_d: [f32; 4] = [0.277785; 4];
            let factor_e: [f32; 4] = [0.097545; 4];
            let factor_f: [f32; 4] = [0.461939; 4];
            let factor_g: [f32; 4] = [0.191341; 4];

            let a_ref = _mm_loadu_ps(factor_a.as_ptr());
            let b_ref = _mm_loadu_ps(factor_b.as_ptr());
            let c_ref = _mm_loadu_ps(factor_c.as_ptr());
            let d_ref = _mm_loadu_ps(factor_d.as_ptr());
            let e_ref = _mm_loadu_ps(factor_e.as_ptr());
            let f_ref = _mm_loadu_ps(factor_f.as_ptr());
            let g_ref = _mm_loadu_ps(factor_g.as_ptr());

            let temp = temp_data.as_mut_ptr().cast::<f32>();
            let result = output.as_mut_ptr().cast::<f32>();

            for i in 0..2 {
                let shift = i * 4;

                // load the first half of each column
                let cf_0a = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si64(coef.offset(shift).cast::<u8>())));
                let cf_1a = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si64(coef.offset(8 + shift).cast::<u8>())));
                let cf_2a = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si64(coef.offset(16 + shift).cast::<u8>())));
                let cf_3a = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si64(coef.offset(24 + shift).cast::<u8>())));
                let cf_4a = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si64(coef.offset(32 + shift).cast::<u8>())));
                let cf_5a = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si64(coef.offset(40 + shift).cast::<u8>())));
                let cf_6a = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si64(coef.offset(48 + shift).cast::<u8>())));
                let cf_7a = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si64(coef.offset(56 + shift).cast::<u8>())));

                // start with the col 0/4 term
                let cf_sum_04a = _mm_mul_ps(_mm_add_ps(cf_0a, cf_4a), a_ref);
                let cf_diff_04a = _mm_mul_ps(_mm_sub_ps(cf_0a, cf_4a), a_ref);

                // calculate the first and 8th row of each column DCT
                let t0 = _mm_mul_ps(cf_1a, b_ref);
                let t1 = _mm_mul_ps(cf_2a, f_ref);
                let t2 = _mm_mul_ps(cf_3a, c_ref);
                let t3 = _mm_mul_ps(cf_5a, d_ref);
                let t4 = _mm_mul_ps(cf_6a, g_ref);
                let t5 = _mm_mul_ps(cf_7a, e_ref);
                let x_0a = _mm_add_ps(cf_sum_04a, t0);
                let x_0a = _mm_add_ps(x_0a, t1);
                let x_0a = _mm_add_ps(x_0a, t2);
                let x_0a = _mm_add_ps(x_0a, t3);
                let x_0a = _mm_add_ps(x_0a, t4);
                let x_0a = _mm_add_ps(x_0a, t5);
                _mm_storeu_ps(temp.offset(0 + shift), x_0a);
                let x_7a = _mm_sub_ps(cf_sum_04a, t0);
                let x_7a = _mm_add_ps(x_7a, t1);
                let x_7a = _mm_sub_ps(x_7a, t2);
                let x_7a = _mm_sub_ps(x_7a, t3);
                let x_7a = _mm_add_ps(x_7a, t4);
                let x_7a = _mm_sub_ps(x_7a, t5);
                _mm_storeu_ps(temp.offset(56 + shift), x_7a);

                // calculate the second and seventh row of each column DCT
                let t0 = _mm_mul_ps(cf_1a, c_ref);
                let t1 = _mm_mul_ps(cf_2a, g_ref);
                let t2 = _mm_mul_ps(cf_3a, e_ref);
                let t3 = _mm_mul_ps(cf_5a, b_ref);
                let t4 = _mm_mul_ps(cf_6a, f_ref);
                let t5 = _mm_mul_ps(cf_7a, d_ref);
                let x_1a = _mm_add_ps(cf_diff_04a, t0);
                let x_1a = _mm_add_ps(x_1a, t1);
                let x_1a = _mm_sub_ps(x_1a, t2);
                let x_1a = _mm_sub_ps(x_1a, t3);
                let x_1a = _mm_sub_ps(x_1a, t4);
                let x_1a = _mm_sub_ps(x_1a, t5);
                _mm_storeu_ps(temp.offset(8 + shift), x_1a);
                let x_6a = _mm_sub_ps(cf_diff_04a, t0);
                let x_6a = _mm_add_ps(x_6a, t1);
                let x_6a = _mm_add_ps(x_6a, t2);
                let x_6a = _mm_add_ps(x_6a, t3);
                let x_6a = _mm_sub_ps(x_6a, t4);
                let x_6a = _mm_add_ps(x_6a, t5);
                _mm_storeu_ps(temp.offset(48 + shift), x_6a);

                // calculate the third and sixth row of each column DCT
                let t0 = _mm_mul_ps(cf_1a, d_ref);
                let t1 = _mm_mul_ps(cf_2a, g_ref);
                let t2 = _mm_mul_ps(cf_3a, b_ref);
                let t3 = _mm_mul_ps(cf_5a, e_ref);
                let t4 = _mm_mul_ps(cf_6a, f_ref);
                let t5 = _mm_mul_ps(cf_7a, c_ref);
                let x_2a = _mm_add_ps(cf_diff_04a, t0);
                let x_2a = _mm_sub_ps(x_2a, t1);
                let x_2a = _mm_sub_ps(x_2a, t2);
                let x_2a = _mm_add_ps(x_2a, t3);
                let x_2a = _mm_add_ps(x_2a, t4);
                let x_2a = _mm_add_ps(x_2a, t5);
                _mm_storeu_ps(temp.offset(16 + shift), x_2a);
                let x_5a = _mm_sub_ps(cf_diff_04a, _mm_mul_ps(cf_1a, d_ref));
                let x_5a = _mm_sub_ps(x_5a, _mm_mul_ps(cf_2a, g_ref));
                let x_5a = _mm_add_ps(x_5a, _mm_mul_ps(cf_3a, b_ref));
                let x_5a = _mm_sub_ps(x_5a, _mm_mul_ps(cf_5a, e_ref));
                let x_5a = _mm_add_ps(x_5a, _mm_mul_ps(cf_6a, f_ref));
                let x_5a = _mm_sub_ps(x_5a, _mm_mul_ps(cf_7a, c_ref));
                _mm_storeu_ps(temp.offset(40 + shift), x_5a);

                // calculate the fourth and fifth row of each column DCT
                let t0 = _mm_mul_ps(cf_1a, e_ref);
                let t1 = _mm_mul_ps(cf_2a, f_ref);
                let t2 = _mm_mul_ps(cf_3a, d_ref);
                let t3 = _mm_mul_ps(cf_5a, c_ref);
                let t4 = _mm_mul_ps(cf_6a, g_ref);
                let t5 = _mm_mul_ps(cf_7a, b_ref);
                let x_3a = _mm_add_ps(cf_sum_04a, t0);
                let x_3a = _mm_sub_ps(x_3a, t1);
                let x_3a = _mm_sub_ps(x_3a, t2);
                let x_3a = _mm_add_ps(x_3a, t3);
                let x_3a = _mm_sub_ps(x_3a, t4);
                let x_3a = _mm_sub_ps(x_3a, t5);
                _mm_storeu_ps(temp.offset(24 + shift), x_3a);
                let x_4a = _mm_sub_ps(cf_sum_04a, t0);
                let x_4a = _mm_sub_ps(x_4a, t1);
                let x_4a = _mm_add_ps(x_4a, t2);
                let x_4a = _mm_sub_ps(x_4a, t3);
                let x_4a = _mm_sub_ps(x_4a, t4);
                let x_4a = _mm_add_ps(x_4a, t5);
                _mm_storeu_ps(temp.offset(32 + shift), x_4a);
            }

            // upper left quadrant
            let r0 = _mm_loadu_ps(temp.offset(0)); // TODO: get this aligned
            let r1 = _mm_loadu_ps(temp.offset(8));
            let r2 = _mm_loadu_ps(temp.offset(16));
            let r3 = _mm_loadu_ps(temp.offset(24));
            let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
            let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
            let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
            let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
            _mm_storeu_ps(result.offset(0), _mm_shuffle_ps(tmp0, tmp1, 0x88));
            _mm_storeu_ps(result.offset(8), _mm_shuffle_ps(tmp0, tmp1, 0xdd));
            _mm_storeu_ps(result.offset(16), _mm_shuffle_ps(tmp2, tmp3, 0x88));
            _mm_storeu_ps(result.offset(24), _mm_shuffle_ps(tmp2, tmp3, 0xdd));

            // upper right quadrant
            let r0 = _mm_loadu_ps(temp.offset(32)); // TODO: get this aligned
            let r1 = _mm_loadu_ps(temp.offset(40));
            let r2 = _mm_loadu_ps(temp.offset(48));
            let r3 = _mm_loadu_ps(temp.offset(56));
            let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
            let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
            let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
            let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
            _mm_storeu_ps(result.offset(4), _mm_shuffle_ps(tmp0, tmp1, 0x88));
            _mm_storeu_ps(result.offset(12), _mm_shuffle_ps(tmp0, tmp1, 0xdd));
            _mm_storeu_ps(result.offset(20), _mm_shuffle_ps(tmp2, tmp3, 0x88));
            _mm_storeu_ps(result.offset(28), _mm_shuffle_ps(tmp2, tmp3, 0xdd));

            // lower left quadrant
            let r0 = _mm_loadu_ps(temp.offset(4)); // TODO: get this aligned
            let r1 = _mm_loadu_ps(temp.offset(12));
            let r2 = _mm_loadu_ps(temp.offset(20));
            let r3 = _mm_loadu_ps(temp.offset(28));
            let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
            let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
            let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
            let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
            _mm_storeu_ps(result.offset(32), _mm_shuffle_ps(tmp0, tmp1, 0x88));
            _mm_storeu_ps(result.offset(40), _mm_shuffle_ps(tmp0, tmp1, 0xdd));
            _mm_storeu_ps(result.offset(48), _mm_shuffle_ps(tmp2, tmp3, 0x88));
            _mm_storeu_ps(result.offset(56), _mm_shuffle_ps(tmp2, tmp3, 0xdd));

            // lower right quadrant
            let r0 = _mm_loadu_ps(temp.offset(36)); // TODO: get this aligned
            let r1 = _mm_loadu_ps(temp.offset(44));
            let r2 = _mm_loadu_ps(temp.offset(52));
            let r3 = _mm_loadu_ps(temp.offset(60));
            let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
            let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
            let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
            let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
            _mm_storeu_ps(result.offset(36), _mm_shuffle_ps(tmp0, tmp1, 0x88));
            _mm_storeu_ps(result.offset(44), _mm_shuffle_ps(tmp0, tmp1, 0xdd));
            _mm_storeu_ps(result.offset(52), _mm_shuffle_ps(tmp2, tmp3, 0x88));
            _mm_storeu_ps(result.offset(60), _mm_shuffle_ps(tmp2, tmp3, 0xdd));

            for i in 0..2 {
                let shift = i * 4;

                // load the first half of each row
                let cf_0a = _mm_loadu_ps(result.offset(shift));
                let cf_1a = _mm_loadu_ps(result.offset(8 + shift));
                let cf_2a = _mm_loadu_ps(result.offset(16 + shift));
                let cf_3a = _mm_loadu_ps(result.offset(24 + shift));
                let cf_4a = _mm_loadu_ps(result.offset(32 + shift));
                let cf_5a = _mm_loadu_ps(result.offset(40 + shift));
                let cf_6a = _mm_loadu_ps(result.offset(48 + shift));
                let cf_7a = _mm_loadu_ps(result.offset(56 + shift));

                // start with the col 0/4 term
                let cf_sum_04a = _mm_mul_ps(_mm_add_ps(cf_0a, cf_4a), a_ref);
                let cf_diff_04a = _mm_mul_ps(_mm_sub_ps(cf_0a, cf_4a), a_ref);

                // calculate the first and 8th row of each column DCT
                let t0 = _mm_mul_ps(cf_1a, b_ref);
                let t1 = _mm_mul_ps(cf_2a, f_ref);
                let t2 = _mm_mul_ps(cf_3a, c_ref);
                let t3 = _mm_mul_ps(cf_5a, d_ref);
                let t4 = _mm_mul_ps(cf_6a, g_ref);
                let t5 = _mm_mul_ps(cf_7a, e_ref);
                let x_0a = _mm_add_ps(cf_sum_04a, t0);
                let x_0a = _mm_add_ps(x_0a, t1);
                let x_0a = _mm_add_ps(x_0a, t2);
                let x_0a = _mm_add_ps(x_0a, t3);
                let x_0a = _mm_add_ps(x_0a, t4);
                let x_0a = _mm_add_ps(x_0a, t5);
                _mm_storeu_ps(temp.offset(0 + shift), _mm_round_ps::<_MM_FROUND_TO_NEAREST_INT>(x_0a));
                let x_7a = _mm_sub_ps(cf_sum_04a, t0);
                let x_7a = _mm_add_ps(x_7a, t1);
                let x_7a = _mm_sub_ps(x_7a, t2);
                let x_7a = _mm_sub_ps(x_7a, t3);
                let x_7a = _mm_add_ps(x_7a, t4);
                let x_7a = _mm_sub_ps(x_7a, t5);
                _mm_storeu_ps(temp.offset(56 + shift), _mm_round_ps::<_MM_FROUND_TO_NEAREST_INT>(x_7a));

                // calculate the second and seventh row of each column DCT
                let t0 = _mm_mul_ps(cf_1a, c_ref);
                let t1 = _mm_mul_ps(cf_2a, g_ref);
                let t2 = _mm_mul_ps(cf_3a, e_ref);
                let t3 = _mm_mul_ps(cf_5a, b_ref);
                let t4 = _mm_mul_ps(cf_6a, f_ref);
                let t5 = _mm_mul_ps(cf_7a, d_ref);
                let x_1a = _mm_add_ps(cf_diff_04a, t0);
                let x_1a = _mm_add_ps(x_1a, t1);
                let x_1a = _mm_sub_ps(x_1a, t2);
                let x_1a = _mm_sub_ps(x_1a, t3);
                let x_1a = _mm_sub_ps(x_1a, t4);
                let x_1a = _mm_sub_ps(x_1a, t5);
                _mm_storeu_ps(temp.offset(8 + shift), _mm_round_ps::<_MM_FROUND_TO_NEAREST_INT>(x_1a));
                let x_6a = _mm_sub_ps(cf_diff_04a, t0);
                let x_6a = _mm_add_ps(x_6a, t1);
                let x_6a = _mm_add_ps(x_6a, t2);
                let x_6a = _mm_add_ps(x_6a, t3);
                let x_6a = _mm_sub_ps(x_6a, t4);
                let x_6a = _mm_add_ps(x_6a, t5);
                _mm_storeu_ps(temp.offset(48 + shift), _mm_round_ps::<_MM_FROUND_TO_NEAREST_INT>(x_6a));

                // calculate the third and sixth row of each column DCT
                let t0 = _mm_mul_ps(cf_1a, d_ref);
                let t1 = _mm_mul_ps(cf_2a, g_ref);
                let t2 = _mm_mul_ps(cf_3a, b_ref);
                let t3 = _mm_mul_ps(cf_5a, e_ref);
                let t4 = _mm_mul_ps(cf_6a, f_ref);
                let t5 = _mm_mul_ps(cf_7a, c_ref);
                let x_2a = _mm_add_ps(cf_diff_04a, t0);
                let x_2a = _mm_sub_ps(x_2a, t1);
                let x_2a = _mm_sub_ps(x_2a, t2);
                let x_2a = _mm_add_ps(x_2a, t3);
                let x_2a = _mm_add_ps(x_2a, t4);
                let x_2a = _mm_add_ps(x_2a, t5);
                _mm_storeu_ps(temp.offset(16 + shift), _mm_round_ps::<_MM_FROUND_TO_NEAREST_INT>(x_2a));
                let x_5a = _mm_sub_ps(cf_diff_04a, _mm_mul_ps(cf_1a, d_ref));
                let x_5a = _mm_sub_ps(x_5a, _mm_mul_ps(cf_2a, g_ref));
                let x_5a = _mm_add_ps(x_5a, _mm_mul_ps(cf_3a, b_ref));
                let x_5a = _mm_sub_ps(x_5a, _mm_mul_ps(cf_5a, e_ref));
                let x_5a = _mm_add_ps(x_5a, _mm_mul_ps(cf_6a, f_ref));
                let x_5a = _mm_sub_ps(x_5a, _mm_mul_ps(cf_7a, c_ref));
                _mm_storeu_ps(temp.offset(40 + shift), _mm_round_ps::<_MM_FROUND_TO_NEAREST_INT>(x_5a));

                // calculate the fourth and fifth row of each column DCT
                let t0 = _mm_mul_ps(cf_1a, e_ref);
                let t1 = _mm_mul_ps(cf_2a, f_ref);
                let t2 = _mm_mul_ps(cf_3a, d_ref);
                let t3 = _mm_mul_ps(cf_5a, c_ref);
                let t4 = _mm_mul_ps(cf_6a, g_ref);
                let t5 = _mm_mul_ps(cf_7a, b_ref);
                let x_3a = _mm_add_ps(cf_sum_04a, t0);
                let x_3a = _mm_sub_ps(x_3a, t1);
                let x_3a = _mm_sub_ps(x_3a, t2);
                let x_3a = _mm_add_ps(x_3a, t3);
                let x_3a = _mm_sub_ps(x_3a, t4);
                let x_3a = _mm_sub_ps(x_3a, t5);
                _mm_storeu_ps(temp.offset(24 + shift), _mm_round_ps::<_MM_FROUND_TO_NEAREST_INT>(x_3a));
                let x_4a = _mm_sub_ps(cf_sum_04a, t0);
                let x_4a = _mm_sub_ps(x_4a, t1);
                let x_4a = _mm_add_ps(x_4a, t2);
                let x_4a = _mm_sub_ps(x_4a, t3);
                let x_4a = _mm_sub_ps(x_4a, t4);
                let x_4a = _mm_add_ps(x_4a, t5);
                _mm_storeu_ps(temp.offset(32 + shift), _mm_round_ps::<_MM_FROUND_TO_NEAREST_INT>(x_4a));
            }

            // upper left quadrant
            let r0 = _mm_loadu_ps(temp.offset(0)); // TODO: get this aligned
            let r1 = _mm_loadu_ps(temp.offset(8));
            let r2 = _mm_loadu_ps(temp.offset(16));
            let r3 = _mm_loadu_ps(temp.offset(24));
            let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
            let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
            let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
            let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
            _mm_storeu_ps(result.offset(0), _mm_shuffle_ps(tmp0, tmp1, 0x88));
            _mm_storeu_ps(result.offset(8), _mm_shuffle_ps(tmp0, tmp1, 0xdd));
            _mm_storeu_ps(result.offset(16), _mm_shuffle_ps(tmp2, tmp3, 0x88));
            _mm_storeu_ps(result.offset(24), _mm_shuffle_ps(tmp2, tmp3, 0xdd));

            // upper right quadrant
            let r0 = _mm_loadu_ps(temp.offset(32)); // TODO: get this aligned
            let r1 = _mm_loadu_ps(temp.offset(40));
            let r2 = _mm_loadu_ps(temp.offset(48));
            let r3 = _mm_loadu_ps(temp.offset(56));
            let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
            let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
            let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
            let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
            _mm_storeu_ps(result.offset(4), _mm_shuffle_ps(tmp0, tmp1, 0x88));
            _mm_storeu_ps(result.offset(12), _mm_shuffle_ps(tmp0, tmp1, 0xdd));
            _mm_storeu_ps(result.offset(20), _mm_shuffle_ps(tmp2, tmp3, 0x88));
            _mm_storeu_ps(result.offset(28), _mm_shuffle_ps(tmp2, tmp3, 0xdd));

            // lower left quadrant
            let r0 = _mm_loadu_ps(temp.offset(4)); // TODO: get this aligned
            let r1 = _mm_loadu_ps(temp.offset(12));
            let r2 = _mm_loadu_ps(temp.offset(20));
            let r3 = _mm_loadu_ps(temp.offset(28));
            let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
            let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
            let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
            let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
            _mm_storeu_ps(result.offset(32), _mm_shuffle_ps(tmp0, tmp1, 0x88));
            _mm_storeu_ps(result.offset(40), _mm_shuffle_ps(tmp0, tmp1, 0xdd));
            _mm_storeu_ps(result.offset(48), _mm_shuffle_ps(tmp2, tmp3, 0x88));
            _mm_storeu_ps(result.offset(56), _mm_shuffle_ps(tmp2, tmp3, 0xdd));

            // lower right quadrant
            let r0 = _mm_loadu_ps(temp.offset(36)); // TODO: get this aligned
            let r1 = _mm_loadu_ps(temp.offset(44));
            let r2 = _mm_loadu_ps(temp.offset(52));
            let r3 = _mm_loadu_ps(temp.offset(60));
            let tmp0 = _mm_shuffle_ps(r0, r1, 0x44);
            let tmp2 = _mm_shuffle_ps(r0, r1, 0xEE);
            let tmp1 = _mm_shuffle_ps(r2, r3, 0x44);
            let tmp3 = _mm_shuffle_ps(r2, r3, 0xEE);
            _mm_storeu_ps(result.offset(36), _mm_shuffle_ps(tmp0, tmp1, 0x88));
            _mm_storeu_ps(result.offset(44), _mm_shuffle_ps(tmp0, tmp1, 0xdd));
            _mm_storeu_ps(result.offset(52), _mm_shuffle_ps(tmp2, tmp3, 0x88));
            _mm_storeu_ps(result.offset(60), _mm_shuffle_ps(tmp2, tmp3, 0xdd));

            // do the rounding and shift to result data

            // for i in 0..8 {
            //     for j in 0..8 {
            //         res[i * 8 + j] = result_data[i][j] as i16;
            //     }
            // }
        }

        let mut res: [i16; 64] = [0; 64];
        let mut output: [f32; 64] = [0.0; 64];
        unsafe {
            dct_cols(coef.as_ptr(), &mut output);
        }

        for i in 0..64 {
            res[i] = output[i] as i16;
        }

        res
    }

    pub fn idct_matrix(&self, coef: &[i16; 64]) -> [i16; 64] {
        let mut temp: [[f64; 8]; 8] = [[0.0; 8]; 8];
        let mut result: [i16; 64] = [0; 64];

        for i in 0..8 {
            for j in 0..8 {
                let mut val = 0.0;
                for a in 0..8 {
                    val += self.dct_t_matrix[i][a] * coef[a * 8 + j] as f64;
                }
                temp[i][j] = val;
            }
        }

        let mut offset = 0;
        for i in 0..8 {
            for j in 0..8 {
                let mut val = 0.0;
                for a in 0..8 {
                    val += temp[i][a] * self.dct_matrix[a][j];
                }
                result[offset] = val.round() as i16;
                offset += 1;
            }
        }

        result
    }

    // pub fn idct_matrix_fixed(&self, coef: &[i16; 64]) -> [i16; 64] {
    //     let mut temp: [[f64; 8]; 8] = [[0; 8]; 8];
    //     let mut result: [i16; 64] = [0; 64];

    //     for i in 0..8 {
    //         for j in 0..8 {
    //             let mut val = 0.0;
    //             for a in 0..8 {
    //                 val += self.dct_t_matrix[i][a] * (coef[a * 8 + j] as i32 << 16);
    //             }
    //             temp[i][j] = val;
    //         }
    //     }

    //     let mut offset = 0;
    //     for i in 0..8 {
    //         for j in 0..8 {
    //             let mut val = 0.0;
    //             for a in 0..8 {
    //                 val += temp[i][a] * self.dct_matrix[a][j];
    //             }
    //             result[offset] = val.round() as i16;
    //             offset += 1;
    //         }
    //     }

    //     result
    // }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn dct_basic() {
        let coef: [i16; 64] = [
            -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10,
            10, 6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];

        let dct = Dct::new();
        let pixels = dct.idct_manual(&coef);

        let expected: [i16; 64] = [
            -80, -45, -49, -101, -96, -95, -97, -94, -89, -42, -58, -105, -104, -97, -95, -103, -83, -40, -76, -110,
            -107, -93, -92, -105, -54, -45, -96, -108, -102, -90, -94, -102, -30, -68, -107, -97, -99, -96, -100, -100,
            -42, -100, -107, -89, -103, -102, -101, -97, -75, -113, -97, -95, -107, -103, -96, -95, -96, -107, -85,
            -105, -108, -100, -95, -97,
        ];
        assert_eq!(pixels, expected);
    }

    #[test]
    fn dct_rows_cols() {
        let coef: [i16; 64] = [
            -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10,
            10, 6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];

        let dct = Dct::new();
        let pixels = dct.idct_rows_cols(&coef);

        let expected: [i16; 64] = [
            -80, -45, -49, -101, -96, -95, -97, -94, -89, -42, -58, -105, -104, -97, -95, -103, -83, -40, -76, -110,
            -107, -93, -92, -105, -54, -45, -96, -108, -102, -90, -94, -102, -30, -68, -107, -97, -99, -96, -100, -100,
            -42, -100, -107, -89, -103, -102, -101, -97, -75, -113, -97, -95, -107, -103, -96, -95, -96, -107, -85,
            -105, -108, -100, -95, -97,
        ];
        assert_eq!(pixels, expected);
    }

    #[test]
    fn dct_rows_cols_fast() {
        let coef: [i16; 64] = [
            -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10,
            10, 6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];

        let dct = Dct::new();
        let pixels = dct.idct_rows_cols_fast2(&coef);

        let expected: [i16; 64] = [
            -80, -45, -49, -101, -96, -95, -97, -94, -89, -42, -58, -105, -104, -97, -95, -103, -83, -40, -76, -110,
            -107, -93, -92, -105, -54, -45, -96, -108, -102, -90, -94, -102, -30, -68, -107, -97, -99, -96, -100, -100,
            -42, -100, -107, -89, -103, -102, -101, -97, -75, -113, -97, -95, -107, -103, -96, -95, -96, -107, -85,
            -105, -108, -100, -95, -97,
        ];
        assert_eq!(pixels, expected);
    }

    #[test]
    fn dct_rows_cols_faster() {
        let coef: [i16; 64] = [
            -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10,
            10, 6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];

        let dct = Dct::new();
        let pixels = dct.idct_rows_cols_faster(&coef);

        let expected: [i16; 64] = [
            -80, -45, -49, -101, -96, -95, -97, -94, -89, -42, -58, -105, -104, -97, -95, -103, -83, -40, -76, -110,
            -107, -93, -92, -105, -54, -45, -96, -108, -102, -90, -94, -102, -30, -68, -107, -97, -99, -96, -100, -100,
            -42, -100, -107, -89, -103, -102, -101, -97, -75, -113, -97, -95, -107, -103, -96, -95, -96, -107, -85,
            -105, -108, -100, -95, -97,
        ];
        assert_eq!(pixels, expected);
    }

    #[test]
    fn dct_matrix() {
        let coef: [i16; 64] = [
            -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10,
            10, 6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];

        let dct = Dct::new();
        let pixels = dct.idct_matrix(&coef);

        let expected: [i16; 64] = [
            -80, -45, -49, -101, -96, -95, -97, -94, -89, -42, -58, -105, -104, -97, -95, -103, -83, -40, -76, -110,
            -107, -93, -92, -105, -54, -45, -96, -108, -102, -90, -94, -102, -30, -68, -107, -97, -99, -96, -100, -100,
            -42, -100, -107, -89, -103, -102, -101, -97, -75, -113, -97, -95, -107, -103, -96, -95, -96, -107, -85,
            -105, -108, -100, -95, -97,
        ];
        assert_eq!(pixels, expected);
    }

    #[bench]
    fn dct_manual_bench(b: &mut test::Bencher) {
        let coef: [i16; 64] = [
            -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10,
            10, 6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];

        let dct = Dct::new();
        b.iter(|| dct.idct_manual(&coef))
    }

    #[bench]
    fn dct_rows_cols_bench(b: &mut test::Bencher) {
        let coef: [i16; 64] = [
            -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10,
            10, 6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];

        let dct = Dct::new();
        b.iter(|| dct.idct_rows_cols(&coef))
    }

    #[bench]
    fn dct_rows_cols_bench_fast2(b: &mut test::Bencher) {
        let coef: [i16; 64] = [
            -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10,
            10, 6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];

        let dct = Dct::new();
        b.iter(|| dct.idct_rows_cols_fast2(&coef))
    }
    #[bench]
    fn dct_rows_cols_bench_fast(b: &mut test::Bencher) {
        let coef: [i16; 64] = [
            -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10,
            10, 6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];

        let dct = Dct::new();
        b.iter(|| dct.idct_rows_cols_fast(&coef))
    }

    #[bench]
    fn dct_rows_cols_bench_faster(b: &mut test::Bencher) {
        let coef: [i16; 64] = [
            -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10,
            10, 6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];

        let dct = Dct::new();
        b.iter(|| dct.idct_rows_cols_faster(&coef))
    }

    #[bench]
    fn dct_matrix_bench(b: &mut test::Bencher) {
        let coef: [i16; 64] = [
            -720, 84, 56, 10, -24, -6, 0, 0, 40, 36, 0, -24, -54, -42, -16, 0, -12, -25, -30, -42, -21, 0, 16, 18, 10,
            10, 6, 12, 35, 16, 9, 0, 10, 6, 6, 7, 0, -8, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];

        let dct = Dct::new();
        b.iter(|| dct.idct_matrix(&coef))
    }
}
