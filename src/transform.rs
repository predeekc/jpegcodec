#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// transforms from YUV pixels to coefficients
//  - includes the +128/-128 to the values
//  - includes the dct/idct
//  - possibly includes the quantization in the future

// currently intel only, will create new implementations of each method if I go further than that

const FACTOR_A: [f32; 4] = [0.35355339059327376220042218105242; 4];
const FACTOR_B: [f32; 4] = [0.490392; 4];
const FACTOR_C: [f32; 4] = [0.415734; 4];
const FACTOR_D: [f32; 4] = [0.277785; 4];
const FACTOR_E: [f32; 4] = [0.097545; 4];
const FACTOR_F: [f32; 4] = [0.461939; 4];
const FACTOR_G: [f32; 4] = [0.191341; 4];

// default assumes SSE2 support (It was around in 2001 and Windows 8 requires SSE2 so I think this goes back far enough)
pub fn transform_to_pixels(coef: &[i16; 64], pixels: &mut [i16]) {
    assert_eq!(pixels.len(), 64);
    unsafe {
        idct_sse2(coef, pixels);
    }
}

#[target_feature(enable = "sse,sse2")]
unsafe fn idct_sse2(coef: &[i16; 64], pixels: &mut [i16]) {
    let mut result_data: [f32; 64] = [0.0; 64];

    let mut temp_data: [f32; 64] = [0.0; 64];

    let a_ref = _mm_loadu_ps(FACTOR_A.as_ptr());
    let b_ref = _mm_loadu_ps(FACTOR_B.as_ptr());
    let c_ref = _mm_loadu_ps(FACTOR_C.as_ptr());
    let d_ref = _mm_loadu_ps(FACTOR_D.as_ptr());
    let e_ref = _mm_loadu_ps(FACTOR_E.as_ptr());
    let f_ref = _mm_loadu_ps(FACTOR_F.as_ptr());
    let g_ref = _mm_loadu_ps(FACTOR_G.as_ptr());

    let coef = coef.as_ptr().cast::<i16>();
    let result = result_data.as_mut_ptr().cast::<f32>();
    let temp = temp_data.as_mut_ptr().cast::<f32>();

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

    for i in 0..64 {
        pixels[i] = (result_data[i].round() as i16 + 128).clamp(0, 255);
    }
}

#[target_feature(enable = "sse,sse2,sse3,sse4.1")]
unsafe fn idct_sse4_1(coef: &[i16; 64], pixels: &mut [i16]) {
    let mut result_data: [f32; 64] = [0.0; 64];

    let mut temp_data: [f32; 64] = [0.0; 64];

    let a_ref = _mm_loadu_ps(FACTOR_A.as_ptr());
    let b_ref = _mm_loadu_ps(FACTOR_B.as_ptr());
    let c_ref = _mm_loadu_ps(FACTOR_C.as_ptr());
    let d_ref = _mm_loadu_ps(FACTOR_D.as_ptr());
    let e_ref = _mm_loadu_ps(FACTOR_E.as_ptr());
    let f_ref = _mm_loadu_ps(FACTOR_F.as_ptr());
    let g_ref = _mm_loadu_ps(FACTOR_G.as_ptr());

    let coef = coef.as_ptr().cast::<i16>();
    let result = result_data.as_mut_ptr().cast::<f32>();
    let temp = temp_data.as_mut_ptr().cast::<f32>();

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

    for i in 0..64 {
        pixels[i] = (result_data[i].round() as i16 + 128).clamp(0, 255);
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
    fn test_caps() {
        if is_x86_feature_detected!("sse") {
            println!("sse detected");
        } else {
            println!("sse not detected");
        }
        if is_x86_feature_detected!("sse2") {
            println!("sse2 detected");
        } else {
            println!("sse2 not detected");
        }
        if is_x86_feature_detected!("sse3") {
            println!("sse3 detected");
        } else {
            println!("sse3 not detected");
        }
        if is_x86_feature_detected!("sse4.1") {
            println!("sse4.1 detected");
        } else {
            println!("sse4.1 not detected");
        }
        if is_x86_feature_detected!("sse4.2") {
            println!("sse4.2 detected");
        } else {
            println!("sse4.2 not detected");
        }
        if is_x86_feature_detected!("avx") {
            println!("avx detected");
        } else {
            println!("avx not detected");
        }
        if is_x86_feature_detected!("avx2") {
            println!("avx2 detected");
        } else {
            println!("avx2 not detected");
        }
        if is_x86_feature_detected!("avx512bw") {
            println!("avx512bw detected");
        } else {
            println!("avx512bw not detected");
        }
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
    fn test_idct_sse4_1() {
        let mut pixels: [i16; 64] = [0; 64];
        unsafe {
            idct_sse4_1(&COEF, &mut pixels);
        }

        assert_eq!(pixels, EXPECTED);
    }

    #[bench]
    fn bench_idct_sse2(b: &mut test::Bencher) {
        let mut pixels: [i16; 64] = [0; 64];
        b.iter(|| unsafe {
            idct_sse2(&COEF, &mut pixels);
        });
    }

    #[bench]
    fn bench_idct_sse4_1(b: &mut test::Bencher) {
        let mut pixels: [i16; 64] = [0; 64];
        b.iter(|| unsafe {
            idct_sse4_1(&COEF, &mut pixels);
        });
    }
}
