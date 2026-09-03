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
#include "xa_nnlib_common.h"

WORD32 xa_nn_instance_norm_3D_8_8_nhwc(
    WORD8 *p_out,
    const WORD8 *p_inp,
    const WORD16 *p_alpha,
    const WORD32 *p_beta,
    const WORD32 *p_rsqrt,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 output_shift,   /* Shift value to bring the final value to 8b */
    WORD32 mean_shift,  /* set to a S to do the division */
    WORD32 mean_scale,  /*Scale = (1<<S) /H*W */
    WORD32 sq_acc_shift, /*  set to a shift value of accumulation of squares to 32 bits*/
    WORD32 min_val,     /* minimum Value for clamping if reluFlag is set to 1 */
    WORD32 max_val)
{
  /* NULL pointer check */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_alpha, -1);
  XA_NNLIB_ARG_CHK_PTR(p_beta, -1);
  XA_NNLIB_ARG_CHK_PTR(p_rsqrt, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0 || input_channels <= 0),-1);
  XA_NNLIB_ARG_CHK_COND((output_shift <= -24 || output_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((mean_shift < -32 || mean_shift >= -2), -1);
  XA_NNLIB_ARG_CHK_COND((mean_shift >= sq_acc_shift), -1);
  XA_NNLIB_ARG_CHK_COND((mean_scale <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((sq_acc_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((min_val < -128 || min_val > 127), -1);
  XA_NNLIB_ARG_CHK_COND((max_val < min_val || max_val > 127), -1);

  WORD32 itr_c, itr_h, itr_w;
  WORD32 index;
  WORD32 index0, index1, index2, index3;
  ae_int16x4 d_input_val, d16_mean;
  ae_int32x2 d_mean, d_tmp, d_tmp0, d_tmp1;
  ae_int64 d_var, d64_tmp0, d64_tmp1, d64_tmp2, d64_tmp3;

  ae_int16x4 *pae_alpha = (ae_int16x4 *)p_alpha;
  ae_int32x2 *pae_beta = (ae_int32x2 *)p_beta;

  ae_valign alpha_a, beta_a;

  alpha_a = AE_LA64_PP(pae_alpha);
  beta_a = AE_LA64_PP(pae_beta);

  for(itr_c = 0; itr_c < (input_channels >> 2); itr_c++)
  {
    ae_int16x4 d_alpha_val;
    ae_int32x2 d_beta_val0, d_beta_val1;

    AE_LA16X4_IP(d_alpha_val, alpha_a, pae_alpha);
    AE_LA32X2_IP(d_beta_val0, beta_a, pae_beta);
    AE_LA32X2_IP(d_beta_val1, beta_a, pae_beta);

    ae_int32x2 d_mean0 = AE_ZERO32();
    ae_int32x2 d_mean1 = AE_ZERO32();
    ae_int64 d_var0 = AE_ZERO64();
    ae_int64 d_var1 = AE_ZERO64();
    ae_int64 d_var2 = AE_ZERO64();
    ae_int64 d_var3 = AE_ZERO64();

    const UWORD8 *ptu_inp = (const UWORD8 *)&p_inp[itr_c << 2];

    WORD32 loop_count = input_height * input_width;
    itr_h = 0;
    while(itr_h < input_height * input_width)
    {
      loop_count = input_height * input_width - itr_h < 1 << 17 ? input_height * input_width - itr_h : 1 << 17;
      d_tmp0 = AE_ZERO32();
      d_tmp1 = AE_ZERO32();
      for(itr_w = 0; itr_w < loop_count; itr_w++)
      {
        ae_int16x4 d0, d1;
        d0 = AE_MOVDA16X2(ptu_inp[0], ptu_inp[1]);
        d1 = AE_MOVDA16X2(ptu_inp[2], ptu_inp[3]);
        d_input_val = AE_SEL16_5410(d0, d1);
        d_input_val = AE_SRAI16(AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(AE_MOVINT32X2_FROMINT16X4(d_input_val), 8)), 8);
        AE_MULA16X4(d_mean0, d_mean1, d_input_val, AE_MOVDA16(1));
        AE_MULA16X4(d_tmp0, d_tmp1, d_input_val, d_input_val);
        ptu_inp += input_channels;
      }
      AE_MULA32_HH(d_var0, d_tmp0, AE_MOVDA32(1));
      AE_MULA32_LL(d_var1, d_tmp0, AE_MOVDA32(1));
      AE_MULA32_HH(d_var2, d_tmp1, AE_MOVDA32(1));
      AE_MULA32_LL(d_var3, d_tmp1, AE_MOVDA32(1));
      itr_h += itr_w;
    };
    // mean_4_times = ROUNDASYM(mean * mean_scale, mean_shift - 2);
    // mean_s8 = MINMAX32(ROUNDASYM(mean * mean_scale, mean_shift), -128, 127);
    d64_tmp0 = AE_MUL32_HH(d_mean0, AE_MOVDA32(mean_scale));
    d64_tmp1 = AE_SLAA64S(d64_tmp0, 32 + mean_shift + 2);
    d64_tmp2 = AE_MUL32_LL(d_mean0, AE_MOVDA32(mean_scale));
    d64_tmp3 = AE_SLAA64S(d64_tmp2, 32 + mean_shift + 2);
    ae_int32x2 d_mean_4_times0 = AE_ROUND32X2F64SASYM(d64_tmp1, d64_tmp3);

    d64_tmp1 = AE_SLAA64S(d64_tmp0, 32 + mean_shift);
    d64_tmp3 = AE_SLAA64S(d64_tmp2, 32 + mean_shift);

    d_mean0 = AE_ROUND32X2F64SASYM(d64_tmp1, d64_tmp3);

    d64_tmp0 = AE_MUL32_HH(d_mean1, AE_MOVDA32(mean_scale));
    d64_tmp1 = AE_SLAA64S(d64_tmp0, 32 + mean_shift + 2);
    d64_tmp2 = AE_MUL32_LL(d_mean1, AE_MOVDA32(mean_scale));
    d64_tmp3 = AE_SLAA64S(d64_tmp2, 32 + mean_shift + 2);
    ae_int32x2 d_mean_4_times1 = AE_ROUND32X2F64SASYM(d64_tmp1, d64_tmp3);

    d64_tmp1 = AE_SLAA64S(d64_tmp0, 32 + mean_shift);
    d64_tmp3 = AE_SLAA64S(d64_tmp2, 32 + mean_shift);

    d_mean1 = AE_ROUND32X2F64SASYM(d64_tmp1, d64_tmp3);

    d16_mean = AE_SAT16X4(d_mean0, d_mean1);
    d16_mean = AE_SRAI16(AE_SLAI16S(d16_mean, 8), 8);

    // var_shift = var >> sq_acc_shift;
    // UWORD32 upper_limit = ((1 << 14) - (1 << 4));
    // var = MINMAX32(ROUNDASYM(var_shift * mean_scale, mean_shift - sq_acc_shift));
    // UWORD32 index = MINMAX(((var << 4) - (mean_4_times * mean_4_times)) >> 4, 0, upper_limit);
    d_var0 = AE_SRAA64(d_var0, -sq_acc_shift);
    d64_tmp0 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var0), AE_MOVDA32(mean_scale));
    d64_tmp1 = AE_SLAA64S(d64_tmp0, 32 + (mean_shift - sq_acc_shift));

    d_var1 = AE_SRAA64(d_var1, -sq_acc_shift);
    d64_tmp2 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var1), AE_MOVDA32(mean_scale));
    d64_tmp3 = AE_SLAA64S(d64_tmp2, 32 + (mean_shift - sq_acc_shift));

    ae_int32x2 d_var_scaled0 = AE_ROUND32X2F64SASYM(d64_tmp1, d64_tmp3);

    d_var2 = AE_SRAA64(d_var2, -sq_acc_shift);
    d64_tmp0 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var2), AE_MOVDA32(mean_scale));
    d64_tmp1 = AE_SLAA64S(d64_tmp0, 32 + (mean_shift - sq_acc_shift));

    d_var3 = AE_SRAA64(d_var3, -sq_acc_shift);
    d64_tmp2 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var3), AE_MOVDA32(mean_scale));
    d64_tmp3 = AE_SLAA64S(d64_tmp2, 32 + (mean_shift - sq_acc_shift));

    ae_int32x2 d_var_scaled1 = AE_ROUND32X2F64SASYM(d64_tmp1, d64_tmp3);

    d64_tmp0 = AE_MUL32_HH(d_var_scaled0, AE_MOVDA32(1 << 4));
    d64_tmp1 = AE_MUL32_HH(d_mean_4_times0, d_mean_4_times0);
    d64_tmp0 = AE_SUB64(d64_tmp0, d64_tmp1);

    d64_tmp2 = AE_MUL32_LL(d_var_scaled0, AE_MOVDA32(1 << 4));
    d64_tmp3 = AE_MUL32_LL(d_mean_4_times0, d_mean_4_times0);
    d64_tmp2 = AE_SUB64(d64_tmp2, d64_tmp3);

    ae_int32x2 d_index0 = AE_TRUNCA32X2F64S(d64_tmp0, d64_tmp2, 28);
    d_index0 = AE_MIN32(AE_MAX32(d_index0, AE_ZERO32()), AE_MOVDA32(((1 << 14) - (1 << 4))));
    index0 = AE_MOVAD32_H(d_index0);
    index1 = AE_MOVAD32_L(d_index0);

    d64_tmp0 = AE_MUL32_HH(d_var_scaled1, AE_MOVDA32(1 << 4));
    d64_tmp1 = AE_MUL32_HH(d_mean_4_times1, d_mean_4_times1);
    d64_tmp0 = AE_SUB64(d64_tmp0, d64_tmp1);

    d64_tmp2 = AE_MUL32_LL(d_var_scaled1, AE_MOVDA32(1 << 4));
    d64_tmp3 = AE_MUL32_LL(d_mean_4_times1, d_mean_4_times1);
    d64_tmp2 = AE_SUB64(d64_tmp2, d64_tmp3);

    ae_int32x2 d_index1 = AE_TRUNCA32X2F64S(d64_tmp0, d64_tmp2, 28);
    d_index1 = AE_MIN32(AE_MAX32(d_index1, AE_ZERO32()), AE_MOVDA32(((1 << 14) - (1 << 4))));
    index2 = AE_MOVAD32_H(d_index1);
    index3 = AE_MOVAD32_L(d_index1);

    WORD32 lut_key_shift0 = 0;
    lut_key_shift0 = (21 - AE_NSAZ32_L(AE_SEL32_HH(d_index0, d_index0)) + 1) & (~1);
    lut_key_shift0 = lut_key_shift0 < 2 ? 2 : lut_key_shift0 > 20 ? 20 : lut_key_shift0;
    lut_key_shift0 = index0 < 1 << 10 ? 0 : lut_key_shift0;

    WORD32 lut_key_shift1 = 0;
    lut_key_shift1 = (21 - AE_NSAZ32_L(d_index0) + 1) & (~1);
    lut_key_shift1 = lut_key_shift1 < 2 ? 2 : lut_key_shift1 > 20 ? 20 : lut_key_shift1;
    lut_key_shift1 = index1 < 1 << 10 ? 0 : lut_key_shift1;

    WORD32 lut_key_shift2 = 0;
    lut_key_shift2 = (21 - AE_NSAZ32_L(AE_SEL32_HH(d_index1, d_index1)) + 1) & (~1);
    lut_key_shift2 = lut_key_shift2 < 2 ? 2 : lut_key_shift2 > 20 ? 20 : lut_key_shift2;
    lut_key_shift2 = index2 < 1 << 10 ? 0 : lut_key_shift2;

    WORD32 lut_key_shift3 = 0;
    lut_key_shift3 = (21 - AE_NSAZ32_L(d_index1) + 1) & (~1);
    lut_key_shift3 = lut_key_shift3 < 2 ? 2 : lut_key_shift3 > 20 ? 20 : lut_key_shift3;
    lut_key_shift3 = index3 < 1 << 10 ? 0 : lut_key_shift3;

    d_tmp0 = AE_SRAA32((*(ae_int32 *)(&p_rsqrt[index0 >> lut_key_shift0])), (lut_key_shift0 >> 1));
    d_tmp1 = AE_SRAA32((*(ae_int32 *)(&p_rsqrt[index1 >> lut_key_shift1])), (lut_key_shift1 >> 1));
    ae_int32x2 d_var_scale0 = AE_SEL32_LL(d_tmp0, d_tmp1);

    d_tmp0 = AE_SRAA32((*(ae_int32 *)(&p_rsqrt[index2 >> lut_key_shift2])), (lut_key_shift2 >> 1));
    d_tmp1 = AE_SRAA32((*(ae_int32 *)(&p_rsqrt[index3 >> lut_key_shift3])), (lut_key_shift3 >> 1));
    ae_int32x2 d_var_scale1 = AE_SEL32_LL(d_tmp0, d_tmp1);

    ptu_inp = (const UWORD8 *)&p_inp[itr_c << 2];

#pragma concurrent
    for(itr_h = 0; itr_h < input_height * input_width; itr_h++)
    {
        // input_val = alpha_val * (input_val - mean_s8);
        // input_val = input_val * var_scale;
        // input_val += betaVal << 24;
        // PRIME_8X4F(pt_inp, inp_a);
        // AE_LA8X4F_IP(d_input_val, inp_a, pt_inp);
        ae_int16x4 d0, d1;
        d0 = AE_MOVDA16X2(ptu_inp[0], ptu_inp[1]);
        d1 = AE_MOVDA16X2(ptu_inp[2], ptu_inp[3]);
        d_input_val = AE_SEL16_5410(d0, d1);
        d_input_val = AE_SRAI16(AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(AE_MOVINT32X2_FROMINT16X4(d_input_val), 8)), 8);
        d_input_val = AE_SUB16S(d_input_val, d16_mean);
        AE_MUL16X4(d_tmp0, d_tmp1, d_input_val, d_alpha_val);
        d64_tmp0 = AE_MUL32_HH(d_tmp0, d_var_scale0);
        AE_MULA32_HH(d64_tmp0, d_beta_val0, AE_MOVDA32(1<<24));

        d64_tmp1 = AE_MUL32_LL(d_tmp0, d_var_scale0);
        AE_MULA32_LL(d64_tmp1, d_beta_val0, AE_MOVDA32(1<<24));

        d64_tmp2 = AE_MUL32_HH(d_tmp1, d_var_scale1);
        AE_MULA32_HH(d64_tmp2, d_beta_val1, AE_MOVDA32(1<<24));

        d64_tmp3 = AE_MUL32_LL(d_tmp1, d_var_scale1);
        AE_MULA32_LL(d64_tmp3, d_beta_val1, AE_MOVDA32(1<<24));

        // input_val = ROUNDASYM(input_val, output_shift+24);
        // input_val = MINMAX(input_val, min_val, max_val);
        // p_out[(itr_h * input_width + itr_w) * input_channels + itr_c] = input_val;
        d64_tmp0 = AE_SLAA64S(d64_tmp0, 32 + (output_shift - 24));
        d64_tmp1 = AE_SLAA64S(d64_tmp1, 32 + (output_shift - 24));
        d64_tmp2 = AE_SLAA64S(d64_tmp2, 32 + (output_shift - 24));
        d64_tmp3 = AE_SLAA64S(d64_tmp3, 32 + (output_shift - 24));
        d_tmp0 = AE_ROUND32X2F64SASYM(d64_tmp0, d64_tmp1);
        d_tmp1 = AE_ROUND32X2F64SASYM(d64_tmp2, d64_tmp3);
        d_tmp0 = AE_MIN32(AE_MAX32(d_tmp0, AE_MOVDA32(min_val)), AE_MOVDA32(max_val));
        d_tmp1 = AE_MIN32(AE_MAX32(d_tmp1, AE_MOVDA32(min_val)), AE_MOVDA32(max_val));
        ae_int16x4 d16_tmp = AE_SAT16X4(d_tmp0, d_tmp1);
        p_out[itr_h * input_channels + (itr_c << 2) + 0] = (WORD8)AE_MOVAD16_3(d16_tmp);
        p_out[itr_h * input_channels + (itr_c << 2) + 1] = (WORD8)AE_MOVAD16_2(d16_tmp);
        p_out[itr_h * input_channels + (itr_c << 2) + 2] = (WORD8)AE_MOVAD16_1(d16_tmp);
        p_out[itr_h * input_channels + (itr_c << 2) + 3] = (WORD8)AE_MOVAD16_0(d16_tmp);
        ptu_inp += input_channels;
    }
  }

  for(itr_c = (input_channels & (~3)); itr_c < input_channels; itr_c++)
  {
    ae_int16x4 d_alpha_val = AE_MOVDA16(p_alpha[itr_c]);
    ae_int32x2 d_beta_val = AE_MOVDA32(p_beta[itr_c]);
    d_mean = AE_ZERO32();
    d_var = AE_ZERO64();

    for(itr_h = 0; itr_h < input_height * input_width; itr_h++)
    {
      // for(itr_w = 0; itr_w < input_width; itr_w++)
      {
        d_input_val = AE_MOVDA16(p_inp[(itr_h) * input_channels + itr_c]);
        AE_MULA16X4(d_tmp, d_mean, d_input_val, AE_MOVDA16(1));
        AE_MULA16_00(d_var, d_input_val, d_input_val);
      }
    }
    // mean_4_times = ROUNDASYM(mean * mean_scale, mean_shift - 2);
    // mean_s8 = MINMAX32(ROUNDASYM(mean * mean_scale, mean_shift), -128, 127);
    d64_tmp0 = AE_MUL32_LL(d_mean, AE_MOVDA32(mean_scale));
    d64_tmp1 = AE_SLAA64S(d64_tmp0, 32 + mean_shift + 2);
    ae_int32x2 d_mean_4_times = AE_ROUND32F64SASYM(d64_tmp1);

    d64_tmp1 = AE_SLAA64S(d64_tmp0, 32 + mean_shift);
    d_mean = AE_ROUND32F64SASYM(d64_tmp1);
    d16_mean = AE_SAT16X4(d_mean, d_mean);
    d16_mean = AE_SRAI16(AE_SLAI16S(d16_mean, 8), 8);

    // var_shift = var >> sq_acc_shift;
    // UWORD32 upper_limit = ((1 << 14) - (1 << 4));
    // var = MINMAX32(ROUNDASYM(var_shift * mean_scale, mean_shift - sq_acc_shift));
    // UWORD32 index = MINMAX(((var << 4) - (mean_4_times * mean_4_times)) >> 4, 0, upper_limit);
    d_var = AE_SRAA64(d_var, -sq_acc_shift);
    d64_tmp0 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var), AE_MOVDA32(mean_scale));
    d64_tmp1 = AE_SLAA64S(d64_tmp0, 32 + (mean_shift - sq_acc_shift));
    ae_int32x2 d_var_scaled = AE_ROUND32F64SASYM(d64_tmp1);
    d64_tmp0 = AE_MUL32_LL(d_var_scaled, AE_MOVDA32(1 << 4));
    d64_tmp1 = AE_MUL32_LL(d_mean_4_times, d_mean_4_times);
    d64_tmp0 = AE_SUB64(d64_tmp0, d64_tmp1);
    ae_int32x2 d_index = AE_TRUNCA32F64S(d64_tmp0, 28);
    d_index = AE_MIN32(AE_MAX32(d_index, AE_ZERO32()), AE_MOVDA32(((1 << 14) - (1 << 4))));
    index = AE_MOVAD32_L(d_index);

    WORD32 lut_key_shift = 0;
    lut_key_shift = (21 - AE_NSAZ32_L(d_index) + 1) & (~1);
    lut_key_shift = lut_key_shift < 2 ? 2 : lut_key_shift > 20 ? 20 : lut_key_shift;
    lut_key_shift = index < 1 << 10 ? 0 : lut_key_shift;

    ae_int32x2 d_var_scale = AE_SRAA32((*(ae_int32 *)(&p_rsqrt[index >> lut_key_shift])), (lut_key_shift >> 1));

    for(itr_h = 0; itr_h < input_height; itr_h++)
    {
      for(itr_w = 0; itr_w < input_width; itr_w++)
      {
        // input_val = alpha_val * (input_val - mean_s8);
        // input_val = input_val * var_scale;
        // input_val += betaVal << 24;
        d_input_val = AE_MOVDA16(p_inp[(itr_h * input_width + itr_w) * input_channels + itr_c]);
        d_input_val = AE_SUB16S(d_input_val, d16_mean);
        AE_MUL16X4(d_mean, d_tmp, d_input_val, d_alpha_val);
        d64_tmp0 = AE_MUL32_LL(d_tmp, d_var_scale);
        AE_MULA32_LL(d64_tmp0, d_beta_val, AE_MOVDA32(1<<24));

        // input_val = ROUNDASYM(input_val, output_shift+24);
        // input_val = MINMAX(input_val, min_val, max_val);
        // p_out[(itr_h * input_width + itr_w) * input_channels + itr_c] = input_val;
        d64_tmp0 = AE_SLAA64S(d64_tmp0, 32 + (output_shift - 24));
        d_tmp = AE_ROUND32F64SASYM(d64_tmp0);
        d_tmp = AE_MIN32(AE_MAX32(d_tmp, AE_MOVDA32(min_val)), AE_MOVDA32(max_val));
        ae_int16x4 d16_tmp = AE_SAT16X4(d_tmp, d_tmp);
        p_out[(itr_h * input_width + itr_w) * input_channels + itr_c] = (WORD8)AE_MOVAD16_0(d16_tmp);
      }
    }
  }
  return 0;
}
