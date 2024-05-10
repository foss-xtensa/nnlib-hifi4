/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
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
#include "xa_nn_basic_state.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"

#if XCHAL_HAVE_HIFI1

#define MAX_16X4(id1, id0) \
        id1 = AE_MAX16(id1, id0);
#define MIN_16X4(id1, id0) \
        id1 = AE_MIN16(id1, id0);
#define LIMIT16X4(out, inp, min, max){\
        out = min;\
        MAX_16X4(out, inp);\
        MIN_16X4(out, max);\
}

#else

#define MAX_16X4(id1, id0) { \
        xtbool4 bool4 = AE_LT16(id1, id0); \
        AE_MOVT16X4(id1, id0, bool4);\
}
#define MIN_16X4(id1, id0) { \
        xtbool4 bool4 = AE_LT16(id1, id0); \
        AE_MOVF16X4(id1, id0, bool4);\
}
#define LIMIT16X4(out, inp, min, max){\
        out = min;\
        MAX_16X4(out, inp);\
        MIN_16X4(out, max);\
}
#endif

#define MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(out, inp1, multiplier, right_shift) \
{\
  inp1 = AE_MULFP32X2RAS(inp1, ((multiplier))); \
  out = AE_MULFP32X2RS(inp1, right_shift); \
}

static void internal_elm_squared_diff_broadcast_2D_sym16sxsym16s_sym16s(WORD16 * __restrict__ p_out,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   WORD16 * __restrict__ p_inp1,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const   WORD16 * __restrict__ p_inp2,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  out_lc,
                            WORD32  in_lc)
{
  int i, j;
  WORD16 * __restrict__ p_a = (WORD16 *)p_inp1;
  WORD16 * __restrict__ p_b = (WORD16 *)p_inp2;
  WORD16 *__restrict__ p_c;

#if TFLITE_SINGLE_ROUNDING
  WORD32 out_ls, out_rs;
  out_ls = out_left_shift;
  out_rs = out_left_shift;
#if XCHAL_HAVE_HIFI1S
  out_ls = 31 - out_ls;
  out_ls = out_ls << 16 | out_ls;
#endif   
  (void)out_rs;
#else
  WORD32 out_ls, out_rs;
  out_ls = out_left_shift > 0 ? out_left_shift : 0;
  out_rs = out_left_shift < 0 ? -out_left_shift : 0;
#endif

//  const ae_int32x2 za = AE_MOVDA32(-inp1_zero_bias);
//  const ae_int32x2 zb = AE_MOVDA32(-inp2_zero_bias);

  /* intermediate results and scratch registers */
  ae_int16x4 a0_3, a4_7, b0_3, b4_7;

  ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
  ae_int32x2 shifted_b0_1, shifted_b2_3, shifted_b4_5, shifted_b6_7;

  ae_int32x2 raw_diff0_1, raw_diff2_3, raw_diff4_5, raw_diff6_7;

//  ae_int16x4 out0, out1;
//  ae_int16x4 out2, out3;

  int num_simd8_ops;
  int num_scalar_ops;

  if(out_lc == 1)
  {
    num_simd8_ops = in_lc >> 3;
    num_scalar_ops = in_lc & 7;
  }
  else
  {
    num_simd8_ops = (in_lc >> 4) << 1;
    num_scalar_ops = in_lc & 15;
  }

  for(i = 0; i < out_lc; i++)
  {
    p_a = (WORD16 *)&p_inp1[i * in_lc];
    p_b = (WORD16 *)p_inp2;
    p_c = (WORD16 *)&p_out[i * in_lc];

    ae_valign va_a, va_b, va_c = AE_ZALIGN64();
    va_a = AE_LA64_PP(p_a);
    va_b = AE_LA64_PP(p_b);

    for(j = 0; j < num_simd8_ops; j++)
    {
      ae_int32x2 out0_32, out1_32, out2_32, out3_32;
      AE_LA16X4_IP(a0_3, va_a, (ae_int16x4 *)p_a);
      AE_LA16X4_IP(a4_7, va_a, (ae_int16x4 *)p_a);
      AE_LA16X4_IP(b0_3, va_b, (ae_int16x4 *)p_b);
      AE_LA16X4_IP(b4_7, va_b, (ae_int16x4 *)p_b);

      /* LSH (and promote to 32-bit)*/
      AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
//      shifted_a0_1 = AE_SUB32(shifted_a0_1, za);
//      shifted_a2_3 = AE_SUB32(shifted_a2_3, za);
      shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
      shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

      AE_MUL16X4(shifted_a4_5, shifted_a6_7, a4_7, AE_MOVDA16(1));
//      shifted_a4_5 = AE_SUB32(shifted_a4_5, za);
//      shifted_a6_7 = AE_SUB32(shifted_a6_7, za);
      shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
      shifted_a6_7 = AE_SLAA32S(shifted_a6_7, left_shift);

      AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, AE_MOVDA16(1));
//      shifted_b0_1 = AE_SUB32(shifted_b0_1, zb);
//      shifted_b2_3 = AE_SUB32(shifted_b2_3, zb);
      shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
      shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);

      AE_MUL16X4(shifted_b4_5, shifted_b6_7, b4_7, AE_MOVDA16(1));
//      shifted_b4_5 = AE_SUB32(shifted_b4_5, zb);
//      shifted_b6_7 = AE_SUB32(shifted_b6_7, zb);
      shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
      shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);

#if 0
      raw_sum0_1 = raw_sum2_3 = raw_sum4_5 = raw_sum6_7 = AE_ZERO32();
      // Scaled input
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_b0_1, shifted_b2_3, inp2_multiplier, inp2_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_b4_5, shifted_b6_7, inp2_multiplier, inp2_left_shift);
      // Raw Output
      out0_32 = out1_32 = out2_32 = out3_32 = AE_MOVDA32(out_zero_bias);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(out0_32, out1_32, raw_sum0_1, raw_sum2_3, out_multiplier, out_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(out2_32, out3_32, raw_sum4_5, raw_sum6_7, out_multiplier, out_left_shift);
      out0 = AE_SAT16X4(out0_32, out1_32);
      out1 = AE_SAT16X4(out2_32, out3_32);
      // Clamp output
      LIMIT16X4(out2, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      LIMIT16X4(out3, out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_SA16X4_IP(out2, va_c, (ae_int16x4 *)p_c);
      AE_SA16X4_IP(out3, va_c, (ae_int16x4 *)p_c);
#else
      raw_diff0_1 = raw_diff2_3 = raw_diff4_5 = raw_diff6_7 = AE_ZERO32();
      // Calculate diff
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_b0_1, shifted_b2_3, inp2_multiplier, inp2_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_diff4_5, raw_diff6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff4_5, raw_diff6_7, shifted_b4_5, shifted_b6_7, inp2_multiplier, inp2_left_shift);
      // Square of diff
        raw_diff0_1 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32)));
        raw_diff2_3 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff2_3, raw_diff2_3), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff2_3, raw_diff2_3), 32)));
        raw_diff4_5 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff4_5, raw_diff4_5), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff4_5, raw_diff4_5), 32)));
        raw_diff6_7 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff6_7, raw_diff6_7), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff6_7, raw_diff6_7), 32)));

      // Raw Output
      out0_32 = out1_32 = out2_32 = out3_32 = 0;
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out0_32, raw_diff0_1, out_multiplier, out_ls, out_rs);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out1_32, raw_diff2_3, out_multiplier, out_ls, out_rs);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out2_32, raw_diff4_5, out_multiplier, out_ls, out_rs);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out3_32, raw_diff6_7, out_multiplier, out_ls, out_rs);
#else      
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out0_32, raw_diff0_1, out_multiplier, out_ls, out_rs);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out1_32, raw_diff2_3, out_multiplier, out_ls, out_rs);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out2_32, raw_diff4_5, out_multiplier, out_ls, out_rs);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out3_32, raw_diff6_7, out_multiplier, out_ls, out_rs);
#endif

      CLAMP_VAL(out0_32, out0_32, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
      CLAMP_VAL(out1_32, out1_32, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
      CLAMP_VAL(out2_32, out2_32, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
      CLAMP_VAL(out3_32, out3_32, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));

      ae_int16x4 outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out0_32), AE_MOVINT16X4_FROMINT32X2(out1_32));
      AE_SA16X4_IP(outval, va_c, (ae_int16x4 *)p_c);
      outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out2_32), AE_MOVINT16X4_FROMINT32X2(out3_32));
      AE_SA16X4_IP(outval, va_c, (ae_int16x4 *)p_c);
#endif
    }
    AE_SA64POS_FP(va_c, (ae_int16x4 *)p_c);
  }

  if(num_scalar_ops!=0){
    for(i = 0; i < out_lc; i++)
    {
      ae_int32x2 out0_32;
      p_a = (WORD16 *)&p_inp1[i * in_lc + (num_simd8_ops << 3)];
      p_c = (WORD16 *)&p_out[i * in_lc + (num_simd8_ops << 3)];
      p_b = (WORD16 *)&p_inp2[num_simd8_ops << 3];
      for(j = 0; j< num_scalar_ops; j++)
      {
        b0_3 = AE_MOVDA16(p_b[j]);
        a0_3 = AE_MOVDA16(p_a[j]);
        AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
        AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, AE_MOVDA16(1));

        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        raw_diff0_1 = AE_ZERO32();
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2_OUT32(raw_diff0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2_OUT32(raw_diff0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift);
        raw_diff0_1 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32)));
        out0_32 = 0;
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out0_32, raw_diff0_1, out_multiplier, out_ls, out_rs);
#else
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(out0_32, raw_diff0_1, out_multiplier, out_ls, out_rs);
#endif
        CLAMP_VAL(out0_32, out0_32, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
        ae_int16x4 outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out0_32), AE_MOVINT16X4_FROMINT32X2(out0_32));
        *p_c++ = (WORD16)(AE_MOVAD16_0(outval));
      }
    }
  }

  return;
}

static void internal_elm_squared_diff_broadcast_sym16sxsym16s_sym16s(WORD16 * __restrict__ p_out,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   WORD16 * __restrict__ p_inp1,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const   WORD16 * __restrict__ p_inp2,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
#if TFLITE_SINGLE_ROUNDING
  WORD32 out_ls, out_rs;
  out_ls = out_left_shift;
  out_rs = out_left_shift;
#if XCHAL_HAVE_HIFI1S
  out_ls = 31 - out_ls;
  out_ls = out_ls << 16 | out_ls;
#endif
  (void)out_rs;
#else
  WORD32 out_ls, out_rs;
  out_ls = out_left_shift > 0 ? out_left_shift : 0;
  out_rs = out_left_shift < 0 ? -out_left_shift : 0;
#endif

  int i;
  WORD16 * __restrict__ p_a = (WORD16 *)p_inp1;
  WORD16 * __restrict__ p_b = (WORD16 *)p_inp2;
  WORD16 *__restrict__ p_c =          p_out;

  ae_int16x4 b;

  ae_int16x4 a0_3;
  ae_int32x2 shifted_a0_1, shifted_a2_3;
  ae_int32x2 shifted_b0, shifted_b1;
  ae_int32x2 scaled_b0;

  ae_int32x2 raw_diff0_1, raw_diff2_3;

  const int num_simd4_ops = num_elm >> 2;
  const int num_scalar_ops = num_elm & 3;

  {
    ae_valign va_a;
    va_a = AE_LA64_PP(p_a);
    ae_valign va_c = AE_ZALIGN64();

    b = AE_MOVDA16(p_b[0]);
    AE_MUL16X4(shifted_b0, shifted_b1, b, AE_MOVDA16(1));

    shifted_b0 = AE_SLAA32S(shifted_b0, left_shift);
    shifted_b1 = AE_SLAA32S(shifted_b1, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0, shifted_b0, inp2_multiplier, inp2_left_shift);

    for(i=0; i<num_simd4_ops; i++)
    {
      ae_int32x2 out0_32, out1_32;
      AE_LA16X4_IP(a0_3, va_a, (ae_int16x4 *)p_a);

      AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
      shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
      shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

      /* Calculate squared diff */
      raw_diff0_1 = raw_diff2_3 = scaled_b0;
      MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
      raw_diff0_1 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32)));
      raw_diff2_3 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff2_3, raw_diff2_3), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff2_3, raw_diff2_3), 32)));

      out0_32 = out1_32 = 0;
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out0_32, raw_diff0_1, out_multiplier, out_ls, out_rs);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out1_32, raw_diff2_3, out_multiplier, out_ls, out_rs);
#else
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out0_32, raw_diff0_1, out_multiplier, out_ls, out_rs);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out1_32, raw_diff2_3, out_multiplier, out_ls, out_rs);
#endif      
      CLAMP_VAL(out0_32, out0_32, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
      CLAMP_VAL(out1_32, out1_32, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
      ae_int16x4 outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out0_32), AE_MOVINT16X4_FROMINT32X2(out1_32));
      AE_SA16X4_IP(outval, va_c, (ae_int16x4 *)p_c);
    }
    AE_SA64POS_FP(va_c, p_c);
  }

  b = AE_MOVDA16(p_b[0]);
  AE_MUL16X4(shifted_b0, shifted_b1, b, AE_MOVDA16(1));

  shifted_b0 = AE_SLAA32S(shifted_b0, left_shift);
  shifted_b1 = AE_SLAA32S(shifted_b1, left_shift);

  MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0, shifted_b0, inp2_multiplier, inp2_left_shift);

  for(i=0; i<num_scalar_ops; i++)
  {
    ae_int32x2 out0_32;
    a0_3 = AE_MOVDA16(p_a[i]);

    AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
    shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
    shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

    raw_diff0_1 = raw_diff2_3 = scaled_b0;
    MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2_OUT32(raw_diff0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
    raw_diff0_1 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32)));
    out0_32 = 0;
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out0_32, raw_diff0_1, out_multiplier, out_ls, out_rs);
#else
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out0_32, raw_diff0_1, out_multiplier, out_ls, out_rs);
#endif    
    CLAMP_VAL(out0_32, out0_32, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
    ae_int16x4 outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out0_32), AE_MOVINT16X4_FROMINT32X2(out0_32));
    *p_c = (WORD16)(AE_MOVAD16_3(outval));
    p_c++;
  }

  return;
}

WORD32 xa_nn_elm_squared_diff_broadcast_4D_sym16sxsym16s_sym16s(WORD16 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                      const WORD16 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                      const WORD16 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND((left_shift != 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_min < -32768) || (out_activation_min > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_max < -32768) || (out_activation_max > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

  /* Check shapes */
  int i;
  for(i = 0; i < 4; i++)
  {
    if((p_inp1_shape[i] != p_inp2_shape[i] && p_inp1_shape[i] != 1 && p_inp2_shape[i] != 1) ||
       (p_out_shape[i] != (p_inp1_shape[i] > p_inp2_shape[i] ? p_inp1_shape[i] : p_inp2_shape[i])))
    {
      return -1;
    }
  }

  WORD32 inp1_strides[4], inp2_strides[4];
  inp1_strides[3] = 1;
  inp2_strides[3] = 1;
  for(i = 2; i >= 0; i--)
  {
    ae_int32x2 d_str, d_shape;
    d_str = AE_MOVDA32X2(inp1_strides[i + 1], inp2_strides[i + 1]);
    d_shape = AE_MOVDA32X2(p_inp1_shape[i + 1], p_inp2_shape[i + 1]);
    d_str = AE_MULP32X2(d_str, d_shape);
    inp1_strides[i] = AE_MOVAD32_H(d_str);
    inp2_strides[i] = AE_MOVAD32_L(d_str);
  }

  int need_broadcast = 0;
  int inp1_const = 1, inp2_const = 1;
  for(i = 0; i < 4; i++)
  {
    if(p_inp1_shape[i] != p_inp2_shape[i])
    {
      if(p_inp1_shape[i] == 1)
        inp1_strides[i] = 0;
      else
        inp2_strides[i] = 0;

      need_broadcast = 1;
    }
    if(p_inp1_shape[i] != 1)
      inp1_const &= 0;
    if(p_inp2_shape[i] != 1)
      inp2_const &= 0;
  }
  int itr0, itr1, itr2;

  WORD16 *p_out_tmp = p_out;
  const WORD16 *__restrict__ p_inp1_tmp = p_inp1;
  const WORD16 *__restrict__ p_inp2_tmp = p_inp2;

  if(need_broadcast == 0)
  {
    internal_elm_squared_diff_broadcast_2D_sym16sxsym16s_sym16s(
                p_out,
                out_left_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1,
                inp1_left_shift,
                inp1_multiplier,
                p_inp2,
                inp2_left_shift,
                inp2_multiplier,
                left_shift,
                1,
                p_out_shape[0] * inp1_strides[0]);
  }
  else if(inp1_strides[3] == inp2_strides[3])
  {
    WORD32 in_lc, out_lc;
    WORD32 inp1_ls, inp1_mult;
    WORD32 inp2_ls, inp2_mult;

    inp1_ls = inp1_left_shift;
    inp1_mult = inp1_multiplier;
    inp2_ls = inp2_left_shift;
    inp2_mult = inp2_multiplier;

    in_lc = p_out_shape[2] * p_out_shape[3];
    out_lc = 1;
    if(inp1_strides[2] == 0)
    {
      inp2_ls = inp1_left_shift;
      inp2_mult = inp1_multiplier;
      inp1_ls = inp2_left_shift;
      inp1_mult = inp2_multiplier;
      const WORD16 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;

      int tmp_strides[2];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];

      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];

      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      in_lc = p_out_shape[3];
      out_lc = p_out_shape[2];
    }
    else if(inp2_strides[2] == 0)
    {
      in_lc = p_out_shape[3];
      out_lc = p_out_shape[2];
    }

   for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const WORD16 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const WORD16 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        internal_elm_squared_diff_broadcast_2D_sym16sxsym16s_sym16s(
            p_out_tmp,
            out_left_shift,
            out_multiplier,
            out_activation_min,
            out_activation_max,
            p_inp1_tmp0,
            inp1_ls,
            inp1_mult,
            p_inp2_tmp0,
            inp2_ls,
            inp2_mult,
            left_shift,
            out_lc,
            in_lc);
        p_out_tmp += in_lc * out_lc;
        p_inp1_tmp0 += inp1_strides[1];
        p_inp2_tmp0 += inp2_strides[1];
      }
      p_inp1_tmp += inp1_strides[0];
      p_inp2_tmp += inp2_strides[0];
    }
  }
  else if(inp1_const == 1 || inp2_const == 1)
  {
    WORD32 inp1_ls, inp1_mult;
    WORD32 inp2_ls, inp2_mult;
    inp1_ls = inp1_left_shift;
    inp1_mult = inp1_multiplier;
    inp2_ls = inp2_left_shift;
    inp2_mult = inp2_multiplier;
    if(inp1_strides[3] == 0)
    {
      inp2_ls = inp1_left_shift;
      inp2_mult = inp1_multiplier;
      inp1_ls = inp2_left_shift;
      inp1_mult = inp2_multiplier;
      const WORD16 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
    }

    internal_elm_squared_diff_broadcast_sym16sxsym16s_sym16s(
        p_out_tmp,
        out_left_shift,
        out_multiplier,
        out_activation_min,
        out_activation_max,
        p_inp1_tmp,
        inp1_ls,
        inp1_mult,
        p_inp2_tmp,
        inp2_ls,
        inp2_mult,
        left_shift,
        p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3]);
  }
  else
  {
    WORD32 inp1_ls, inp1_mult;
    WORD32 inp2_ls, inp2_mult;
    inp1_ls = inp1_left_shift;
    inp1_mult = inp1_multiplier;
    inp2_ls = inp2_left_shift;
    inp2_mult = inp2_multiplier;
    if(inp1_strides[3] == 0)
    {
      inp2_ls = inp1_left_shift;
      inp2_mult = inp1_multiplier;
      inp1_ls = inp2_left_shift;
      inp1_mult = inp2_multiplier;
      const WORD16 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;

      int tmp_strides[3];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];
      tmp_strides[2] = inp1_strides[2];

      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];
      inp1_strides[2] = inp2_strides[2];

      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      inp2_strides[2] = tmp_strides[2];
    }
    for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const WORD16 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const WORD16 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        const WORD16 *__restrict__ p_inp1_tmp1 = p_inp1_tmp0;
        const WORD16 *__restrict__ p_inp2_tmp1 = p_inp2_tmp0;
        for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
        {
          {
            internal_elm_squared_diff_broadcast_sym16sxsym16s_sym16s(
                p_out_tmp,
                out_left_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1_tmp1,
                inp1_ls,
                inp1_mult,
                p_inp2_tmp1,
                inp2_ls,
                inp2_mult,
                left_shift,
                p_out_shape[3]);
          }
          p_out_tmp += p_out_shape[3];
          p_inp1_tmp1 += inp1_strides[2];
          p_inp2_tmp1 += inp2_strides[2];
        }
        p_inp1_tmp0 += inp1_strides[1];
        p_inp2_tmp0 += inp2_strides[1];
      }
      p_inp1_tmp += inp1_strides[0];
      p_inp2_tmp += inp2_strides[0];
    }
  }
  return 0;
}
