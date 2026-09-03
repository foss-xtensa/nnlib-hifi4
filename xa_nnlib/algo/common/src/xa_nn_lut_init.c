/*******************************************************************************
* Copyright (c) 2018-2026 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
/* ------------------------------------------------------------------------ */
/* Copyright (c) 2018 by Cadence Design Systems, Inc. ALL RIGHTS RESERVED.  */
/* These coded instructions, statements, and computer programs ("Cadence    */
/* Libraries") are the copyrighted works of Cadence Design Systems Inc.     */
/* Cadence IP is licensed for use with Cadence processor cores only and     */
/* must not be used for any other processors and platforms. Your use of the */
/* Cadence Libraries is subject to the terms of the license agreement you   */
/* have entered into with Cadence Design Systems, or a sublicense granted   */
/* to you by a direct Cadence licensee.                                     */
/* ------------------------------------------------------------------------ */
/*
 * xa_nn_lut_init.c
 *
 * Common (core-independent) LUT initialisation functions for quantised
 * activation functions.  These are pure scalar C implementations that run on
 * every supported core (HiFi4, HiFi5, HiFi IQ) and produce the look-up
 * tables consumed by xa_nn_vec_apply_lut_asym8s_asym8s and related kernels.
 *
 * Exported API:
*   xa_nn_init_lut_asym8s_sigmoid  -- build 256-entry sigmoid LUT
 *   xa_nn_init_lut_asym8s_tanh     -- build 256-entry tanh LUT
 */

#include "xa_type_def.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common.h"

/* ========================================================================= */
/* Internal fixed-point math helpers (file-scope only)                       */
/* ========================================================================= */

static const int CONSTANT_TERM           = (0x70f5a894);
static const int CONSTANT_1_OVER_3       = (0x2aaaaaab);
static const int CONSTANT_1_OVER_8       = (0x10000000);
static const int ONE_QUATER_Q27          = (0x2000000);
static const int ONE_QUATER_Q26              = (0x1000000); // 0.25 Q6.26
static const int MASK                        = (0xffffff);
static const int LUT_MASK7F                  = (0x1ffffff);
static const int LUT_Q31                     = 0x7fffffff;
static const int LUT_constant_48_over_17     = 1515870810;
static const int LUT_constant_neg_32_over_17 = -1010580540;
static const int LUT_F2_ONE                  = 0x20000000;
static const int Q31                         = 0x7fffffff;

static WORD32 mul_fract_q31_asym(WORD32 x1, WORD32 x2)
{
    WORD32 out;
    WORD64 temp64 = (1 << 30);
    temp64 += (WORD64)x1 * (WORD64)x2;
    out = (WORD32)(temp64 >> 31);
    return out;
}

static WORD32 mul_fract_q31_sym(WORD32 x1, WORD32 x2)
{
    WORD32 out;
    WORD64 temp64 = (1 << 30);
    WORD64 tempmul64 = (WORD64)x1 * (WORD64)x2;

    if (tempmul64 > 0) {
        temp64 += tempmul64;
        out = (WORD32)(temp64 >> 31);
    } else {
        temp64 += -tempmul64;
        out = -(WORD32)(temp64 >> 31);
    }
    return out;
}

static WORD32 shift_right_round(WORD32 x, WORD32 shift)
{
    WORD64 out = 1 << (shift - 1);
    out += (WORD64)x;
    return (WORD32)(out >> shift);
}

static WORD32 exp_on_interval(WORD32 inp)
{
    WORD32 x1_in, x2, x3, x4, x4_by_4, y1, y2, y3, y4, y5, y6, y_out;

    x1_in   = inp + CONSTANT_1_OVER_8;
    x2      = mul_fract_q31_sym(x1_in, x1_in);
    x3      = mul_fract_q31_asym(x2, x1_in);
    x4      = mul_fract_q31_asym(x2, x2);
    x4_by_4 = shift_right_round(x4, 2);
    y1      = x4_by_4 + x3;
    y2      = mul_fract_q31_sym(y1, CONSTANT_1_OVER_3);
    y3      = y2 + x2;
    y4      = shift_right_round(y3, 1);
    y5      = x1_in + y4;
    y6      = mul_fract_q31_sym(y5, CONSTANT_TERM);
    y_out   = y6 + CONSTANT_TERM;
    return y_out;
}

static WORD32 barrel_shifter(WORD32 in, WORD32 exponent,
                                 WORD32 multiplier, WORD32 remainder)
{
    int shift_amount = 27 + exponent;
    int targetbit    = (1 << shift_amount) & remainder;
    if (targetbit == 0)
        return in;
    return mul_fract_q31_asym(in, multiplier);
}

static WORD32 multiply_by_quantized_multiplier(WORD32 x,
                                                    WORD32 quantized_multiplier,
                                                    WORD32 shift)
{
    const WORD64 total_shift = 31 - shift;
    WORD64 round  = ((WORD64)1) << (total_shift - 1);
    if (total_shift == 0)
        round = 0;
    WORD64 result = (WORD64)x * (WORD64)quantized_multiplier + round;
    result >>= total_shift;
    result = result >  2147483647LL ?  2147483647LL : result;
    result = result < -2147483648LL ? -2147483648LL : result;
    return (WORD32)result;
}

static WORD64 rounding_half_sum(WORD64 a)
{
    WORD64 s, r = -1;
    s = LUT_Q31 + a;
    if (0 <= s) r = 1;
    s = ((s + r) >> 1);
    return s;
}

static WORD32 exp_q26(WORD32 inp)
{
    WORD32 y, x_in, x2, remainder, a_mod;

    x2    = inp & LUT_MASK7F;
    a_mod = x2 - ONE_QUATER_Q27;
    x_in  = a_mod << 4;

    y = exp_on_interval(x_in);
    remainder = a_mod - inp;

    y = barrel_shifter(y, -2, 1672461947, remainder);
    y = barrel_shifter(y, -1, 1302514674, remainder);
    y = barrel_shifter(y,  0,  790015084, remainder);
    y = barrel_shifter(y,  1,  290630308, remainder);
    y = barrel_shifter(y,  2,   39332535, remainder);
    y = barrel_shifter(y,  3,     720401, remainder);
    y = barrel_shifter(y,  4,        242, remainder);

    if (inp == 0) y = LUT_Q31;
    return y;
}

static WORD32 one_over_one_plus_x(WORD32 a1)
{
    WORD32 y1, half_den1, m1, x1, hdt_x1, one_minus_hdt_x1;
    WORD64 t1 = rounding_half_sum((WORD64)a1);
    int j;

    half_den1 = (WORD32)t1;
    m1 = mul_fract_q31_sym(half_den1, LUT_constant_neg_32_over_17);
    x1 = m1 + LUT_constant_48_over_17;

    for (j = 0; j < 3; j++) {
        hdt_x1           = mul_fract_q31_sym(x1, half_den1);
        one_minus_hdt_x1 = LUT_F2_ONE - hdt_x1;
        m1 = mul_fract_q31_sym(x1, one_minus_hdt_x1);
        m1 = m1 << 2;
        x1 = x1 + m1;
    }

    y1 = x1 << 1;
    return y1;
}

/* Computes (1-x)/(1+x) for x in [0,1) in Q30, used by tanh LUT init. */
static WORD32 one_minus_x_over_one_plus_x(WORD32 a1)
{
    WORD32 y1, half_den1, m1, x1, hdt_x1, one_minus_hdt_x1;
    WORD64 t1 = rounding_half_sum((WORD64)a1);
    int j;

    half_den1 = (WORD32)t1;
    m1 = mul_fract_q31_asym(half_den1, LUT_constant_neg_32_over_17);
    x1 = m1 + LUT_constant_48_over_17;

    for (j = 0; j < 3; j++) {
        hdt_x1           = mul_fract_q31_asym(x1, half_den1);
        one_minus_hdt_x1 = LUT_F2_ONE - hdt_x1;
        m1 = mul_fract_q31_asym(x1, one_minus_hdt_x1);
        m1 = m1 << 2;
        x1 = x1 + m1;
    }

    x1 = x1 - LUT_F2_ONE;
    y1 = x1 << 2;
    return y1;
}

static inline WORD32 EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_REF(WORD32 inp)
{
    WORD32 x1_in, x2, x3, x4, x4_by_4, y1, y2, y3, y4, y5, y6;
    WORD32 y_out;
    WORD32 CT = (CONSTANT_TERM);
    WORD32 CT_1_BY_3 = (CONSTANT_1_OVER_3);
    WORD32 CT_1_BY_8 = (CONSTANT_1_OVER_8);

    x1_in = inp + CT_1_BY_8;
    x2 = mul_fract_q31_sym(x1_in, x1_in);
    x3 = mul_fract_q31_asym(x2, x1_in);
    x4 = mul_fract_q31_asym(x2, x2);
    x4_by_4 = shift_right_round(x4, 2);
    y1 = x4_by_4 + x3;
    y2 = mul_fract_q31_sym(y1, CT_1_BY_3);
    y3 = y2 + x2;
    y4 = shift_right_round(y3, 1);

    y5 = (x1_in + y4);
    y6 = mul_fract_q31_sym(y5, CT);
    y_out = (y6 + CT);

    return y_out;
}

static inline WORD32 GEMMLOWP_EXP_BARREL_SHIFTER_REF(WORD32 in, WORD32 exponent, WORD32 multiplier, WORD32 remainder)
{
	int shift_amount = 26 + exponent;
	int targetbit = (1<<shift_amount)&remainder;
	int out;

	if (targetbit == 0){
		out = in;
	} else {
		out = mul_fract_q31_asym(in, multiplier);
	}

	return out;
}

WORD32 EXP_Q26_ref(WORD32 inp)
{
    WORD32 y;
    WORD32 x_in, x2, remainder;
    WORD32 a_mod_quater_minus_q_1_by_4;

    WORD32 mask_6fs = (MASK);
    WORD32 q_1_by_4 = (ONE_QUATER_Q26);

    x2 = (inp & mask_6fs);
    a_mod_quater_minus_q_1_by_4 = (x2 - q_1_by_4);
    x_in = (a_mod_quater_minus_q_1_by_4<<5);

    y = EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_REF(x_in);

    remainder = (a_mod_quater_minus_q_1_by_4 - inp);

    y = GEMMLOWP_EXP_BARREL_SHIFTER_REF(y,-2, 1672461947, remainder);
    y = GEMMLOWP_EXP_BARREL_SHIFTER_REF(y,-1, 1302514674, remainder);
    y = GEMMLOWP_EXP_BARREL_SHIFTER_REF(y,0, 790015084,   remainder);
    y = GEMMLOWP_EXP_BARREL_SHIFTER_REF(y,1, 290630308,   remainder);
    y = GEMMLOWP_EXP_BARREL_SHIFTER_REF(y,2, 39332535,    remainder);
    y = GEMMLOWP_EXP_BARREL_SHIFTER_REF(y,3, 720401,      remainder);
    y = GEMMLOWP_EXP_BARREL_SHIFTER_REF(y,4, 242,         remainder);

    if(inp == 0) {
    	y = Q31;
    }
    return y;
}
/* ========================================================================= */
/* xa_nn_init_lut_asym8s_sigmoid                                             */
/*                                                                            */
/* Fills a 256-entry LUT (WORD8, offset-binary) for the sigmoid activation   */
/* with the given quantisation parameters.  The table is then passed directly*/
/* to xa_nn_vec_apply_lut_asym8s_asym8s to evaluate sigmoid on a vector.    */
/*                                                                            */
/* Parameters:                                                                */
/*   p_lut              : pointer to output LUT buffer (256 bytes, must be   */
/*                         16-byte aligned)                                   */
/*   zero_point         : input zero-point (asym8s quantisation)             */
/*   input_range_radius : half-range of the linear region (int16 saturated)  */
/*   input_multiplier   : quantised multiplier for beta * input              */
/*   input_left_shift   : left shift for the multiplier                      */
/*                                                                            */
/* Returns 0 on success, -1 on invalid arguments.                            */
/* ========================================================================= */

WORD32 xa_nn_init_lut_asym8s_sigmoid(WORD8  * __restrict__ p_lut,
                                     WORD32  zero_point,
                                     WORD32  input_range_radius,
                                     WORD32  input_multiplier,
                                     WORD32  input_left_shift)
{
    int i;
    WORD8 pinput_lut[256];

    XA_NNLIB_ARG_CHK_PTR(p_lut, -1);
    XA_NNLIB_ARG_CHK_COND((input_multiplier < 0), -1);
    XA_NNLIB_ARG_CHK_COND(((input_left_shift < -31) || (input_left_shift > 31)), -1);

    /* Saturate radius to int16 range (hardware comparison uses 16-bit) */
    if (input_range_radius > 32767) input_range_radius = 32767;

    for (i = 0; i < 256; i++)
        pinput_lut[i] = (WORD8)(i - 128);

    for (i = 0; i < 256; i++) {
        WORD16 m0  = (WORD16)pinput_lut[i];
        WORD16 z10 = m0 - (WORD16)zero_point;

        int bflag = (z10 <= -(WORD16)input_range_radius) ? 1 : 0;
        int dflag = (z10 <   (WORD16)input_range_radius) ? 1 : 0;
        int fflag = (z10 < 0) ? 1 : 0;

        WORD32 x10 = multiply_by_quantized_multiplier(
                         (WORD32)z10, input_multiplier, input_left_shift);

        x10 = (x10 < 0) ? -x10 : x10;
        x10 = -x10;

        WORD32 exp_x = exp_q26(x10);
        x10 = one_over_one_plus_x(exp_x);

        WORD16 m10 = (WORD16)shift_right_round(x10, 23);
        z10 = m10;

        z10 = (fflag == 1) ? (256 - z10) : z10;
        z10 = (dflag == 0) ?  255        : z10;
        z10 = (bflag == 1) ?    0        : z10;

        m0 = z10;
        m0 = (m0 > 255) ? 255 : m0;

        p_lut[i] = (WORD8)(m0 ^ 128);
    }
    // len is fixed at 256. Permuting table entries to remove +128 in lut function
    WORD8 temp[256];
    for(i = 0; i < 256; i++){
      int in_id = (UWORD8)(i-128);
      int out_id = i;
      temp[out_id] = p_lut[in_id];
    }
    for(i = 0; i < 256; i++) {
      p_lut[i] = temp[i];
    }
    return 0;
}

/* ========================================================================= */
/* xa_nn_init_lut_asym8s_tanh                                                */
/*                                                                            */
/* Fills a 256-entry LUT (WORD8, signed) for the tanh activation with the   */
/* given quantisation parameters.  The table is then passed directly to      */
/* xa_nn_vec_apply_lut_asym8s_asym8s to evaluate tanh on a vector.          */
/*                                                                            */
/* Parameters: same as xa_nn_init_lut_asym8s_sigmoid.                        */
/* Returns 0 on success, -1 on invalid arguments.                            */
/* ========================================================================= */

WORD32 xa_nn_init_lut_asym8s_tanh(WORD8  * __restrict__ p_lut,
                                   WORD32  zero_point,
                                   WORD32  input_range_radius,
                                   WORD32  input_multiplier,
                                   WORD32  input_left_shift)
{
    int i;
    WORD8 pinput_lut[256];

    XA_NNLIB_ARG_CHK_PTR(p_lut, -1);
    XA_NNLIB_ARG_CHK_COND((input_multiplier < 0), -1);
    XA_NNLIB_ARG_CHK_COND(((input_left_shift < -31) || (input_left_shift > 31)), -1);

    /* Saturate radius to int16 range */
    if (input_range_radius > 32767) input_range_radius = 32767;

    for (i = 0; i < 256; i++)
        pinput_lut[i] = (WORD8)(i - 128);

    for (i = 0; i < 256; i++) {
        WORD16 m0  = (WORD16)pinput_lut[i];
        WORD16 z10 = m0 - (WORD16)zero_point;

        int bflag = (z10 <= -(WORD16)input_range_radius) ? 1 : 0;
        int dflag = (z10 <   (WORD16)input_range_radius) ? 1 : 0;
        int fflag = (z10 < 0) ? 1 : 0;

        WORD32 x10 = multiply_by_quantized_multiplier(
                         (WORD32)z10, input_multiplier, input_left_shift);

        /* tanh uses 2x scaled input: abs then negate then left-shift by 1 */
        x10 = (x10 < 0) ? -x10 : x10;
        x10 = -x10;
        /* Clamp to prevent int32 overflow on the x2 left-shift */
        if (x10 < -(1 << 30)) x10 = -(1 << 30);
        x10 = x10 << 1;
        WORD32 exp_x = exp_q26(x10);
        x10 = one_minus_x_over_one_plus_x(exp_x);

        /* Downscale to 8-bit signed: shift 24 (vs sigmoid's 23) */
        WORD16 m10 = (WORD16)shift_right_round(x10, 24);
        z10 = m10;

        z10 = (fflag == 1) ? (-z10) : z10;
        z10 = (dflag == 0) ?  127   : z10;
        z10 = (bflag == 1) ? -128   : z10;

        m0 = z10;
        m0 = (m0 >  127) ?  127 : m0;
        m0 = (m0 < -128) ? -128 : m0;

        p_lut[i] = (WORD8)m0;
    }
    // len is fixed at 256. Permuting table entries to remove +128 in lut function
    WORD8 temp[256];
    for(i = 0; i < 256; i++){
      int in_id = (UWORD8)(i-128);
      int out_id = i;
      temp[out_id] = p_lut[in_id];
    }
    for(i = 0; i < 256; i++) {
      p_lut[i] = temp[i];
    }
    return 0;
}

#if ETIE_LUT || !(XCHAL_HAVE_HIFI_IQ)  // For ETIE, we do not need to split LUT. For hifi4/hifi5 we dont use split LUT.

WORD32 xa_nn_init_lut_asym8s_softmax(WORD32 *plut, 
                       WORD32 diffmin, 
                       WORD32 input_beta_multiplier, 
                       WORD32 input_beta_left_shift) 
{
  int i;
  XA_NNLIB_ARG_CHK_PTR(plut, -1);
  for(i = 0; i < 256; i++)
  {
    int y32 = i-255;
    int f32 = (diffmin <= y32) ? 1 : 0;

    WORD32 dequantized_y32 = multiply_by_quantized_multiplier(y32, input_beta_multiplier, input_beta_left_shift);

    WORD32 exp_y32 = EXP_Q26_ref(dequantized_y32);
    exp_y32 = (f32 == 0) ? 0 : exp_y32;
    plut[i] = exp_y32;
  }
 
  return 0;
}
#else
/* softmax 8b implementation with split table. This improves non-etie performance.
 * xa_nn_init_lut_asym8s_softmax below generates the table and stores in 4 split parts
*/
WORD32 xa_nn_init_lut_asym8s_softmax(WORD32 *plut, 
                       WORD32 diffmin, 
                       WORD32 input_beta_multiplier, 
                       WORD32 input_beta_left_shift) 
{
  int i;
  XA_NNLIB_ARG_CHK_PTR(plut, -1);
  WORD32 tmp[256];
  (void)plut; /* keep signature use */
  #pragma no_unroll
  for(i = 0; i < 256; i++)
  {
    int y32 = i-255;
    int f32 = (diffmin <= y32) ? 1 : 0;

    WORD32 dequantized_y32 = multiply_by_quantized_multiplier(y32, input_beta_multiplier, input_beta_left_shift);

    WORD32 exp_y32 = EXP_Q26_ref(dequantized_y32);
    exp_y32 = (f32 == 0) ? 0 : exp_y32;
    tmp[i] = exp_y32;
  }

  /* Re-order table bytes: store LSBs of all entries first, then next bytes, etc.
     This produces 4 blocks of 256 bytes each preserving bit-exact values. */
  unsigned char *p = (unsigned char *)plut;
  for(int byte = 0; byte < 4; byte++) {
    for(i = 0; i < 256; i++) {
      p[byte * 256 + i] = (unsigned char)((tmp[i] >> (8 * byte)) & 0xFF);
    }
  }
  return 0;
}
#endif
