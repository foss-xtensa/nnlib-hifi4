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
#include "xa_nnlib_common_macros.h"

#define STORE_8X4_FROM_16X4(out_ptr, val){\
    int o1, o2, o3, o4;\
    o1 = AE_MOVAD16_3(val);\
    o2 = AE_MOVAD16_2(val);\
    o3 = AE_MOVAD16_1(val);\
    o4 = AE_MOVAD16_0(val);\
    *out_ptr++ = (UWORD8)o1;\
    *out_ptr++ = (UWORD8)o2;\
    *out_ptr++ = (UWORD8)o3;\
    *out_ptr++ = (UWORD8)o4;\
}

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

static void internal_elm_squared_diff_broadcast_2D_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  out_lc,
                            WORD32  in_lc)
{
  int i, j;
  WORD8 * __restrict__ p_a = (WORD8 *)p_inp1;
  WORD8 * __restrict__ p_b = (WORD8 *)p_inp2;
  WORD8 *__restrict__ p_c;

  const ae_int16x4 za = AE_MOVDA16(-inp1_zero_bias);
  const ae_int16x4 zb = AE_MOVDA16(-inp2_zero_bias);

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b0_3, b4_7;

  ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
  ae_int32x2 shifted_b0_1, shifted_b2_3, shifted_b4_5, shifted_b6_7;

  ae_int32x2 raw_diff0_1, raw_diff2_3, raw_diff4_5, raw_diff6_7;

  ae_int16x4 out0, out1;
  ae_int16x4 out2, out3;

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

//#pragma loop_count min=1
  for(i = 0; i < out_lc; i++)
  {
    p_a = (WORD8 *)&p_inp1[i * in_lc];
    p_b = (WORD8 *)p_inp2;
    p_c = (WORD8 *)&p_out[i * in_lc];
    xtbool io_pointers_aligned = ((uintptr_t)p_a%4 == 0) && ((uintptr_t)p_b%4==0) && ((uintptr_t)p_c%4==0);
    if (io_pointers_aligned)
    {
      for(j = 0; j < num_simd8_ops; j++)
      {
#if XCHAL_HAVE_HIFI1
        AE_L8X4S_IP(a0_3, p_a, 4);
        AE_L8X4S_IP(a4_7, p_a, 4);
        AE_L8X4S_IP(b0_3, p_b, 4);
        AE_L8X4S_IP(b4_7, p_b, 4);
#else
        AE_L8X4F_IP(a0_3, p_a, 4);
        AE_L8X4F_IP(a4_7, p_a, 4);
        AE_L8X4F_IP(b0_3, p_b, 4);
        AE_L8X4F_IP(b4_7, p_b, 4);

        a0_3 = AE_SRAI16(a0_3, 8);
        a4_7 = AE_SRAI16(a4_7, 8);
        b0_3 = AE_SRAI16(b0_3, 8);
        b4_7 = AE_SRAI16(b4_7, 8);
#endif
        // Add input zero bias
        a0_3 = AE_SUB16S(a0_3, za);
        a4_7 = AE_SUB16S(a4_7, za);
        b0_3 = AE_SUB16S(b0_3, zb);
        b4_7 = AE_SUB16S(b4_7, zb);

        // LSH (and promote to 32-bit)
#if XCHAL_HAVE_HIFI1
      shifted_a0_1 = AE_SEXT32X2D16_32(a0_3);
      shifted_a2_3 = AE_SEXT32X2D16_10(a0_3);
#else
        AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
#endif
        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);
#if XCHAL_HAVE_HIFI1
        shifted_a4_5 = AE_SEXT32X2D16_32(a4_7);
        shifted_a6_7 = AE_SEXT32X2D16_10(a4_7);
#else
        AE_MUL16X4(shifted_a4_5, shifted_a6_7, a4_7, AE_MOVDA16(1));
#endif
        shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
        shifted_a6_7 = AE_SLAA32S(shifted_a6_7, left_shift);
#if XCHAL_HAVE_HIFI1
        shifted_b0_1 = AE_SEXT32X2D16_32(b0_3);
        shifted_b2_3 = AE_SEXT32X2D16_10(b0_3);
#else
        AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, AE_MOVDA16(1));
#endif
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);
#if XCHAL_HAVE_HIFI1
        shifted_b4_5 = AE_SEXT32X2D16_32(b4_7);
        shifted_b6_7 = AE_SEXT32X2D16_10(b4_7);
#else
        AE_MUL16X4(shifted_b4_5, shifted_b6_7, b4_7, AE_MOVDA16(1));
#endif
        shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
        shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);

        raw_diff0_1 = raw_diff2_3 = raw_diff4_5 = raw_diff6_7 = AE_ZERO32();
        // Scaled input
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_b0_1, shifted_b2_3, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_diff4_5, raw_diff6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff4_5, raw_diff6_7, shifted_b4_5, shifted_b6_7, inp2_multiplier, inp2_left_shift);
        // Squared Diff
#if XCHAL_HAVE_HIFI1
        raw_diff0_1 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32);
        raw_diff2_3 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff2_3, raw_diff2_3), AE_MUL32_LL(raw_diff2_3, raw_diff2_3), 32);
        raw_diff4_5 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff4_5, raw_diff4_5), AE_MUL32_LL(raw_diff4_5, raw_diff4_5), 32);
        raw_diff6_7 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff6_7, raw_diff6_7), AE_MUL32_LL(raw_diff6_7, raw_diff6_7), 32);
#else
        raw_diff0_1 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32)));
        raw_diff2_3 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff2_3, raw_diff2_3), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff2_3, raw_diff2_3), 32)));
        raw_diff4_5 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff4_5, raw_diff4_5), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff4_5, raw_diff4_5), 32)));
        raw_diff6_7 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff6_7, raw_diff6_7), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff6_7, raw_diff6_7), 32)));
#endif
        // Raw Output
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_diff0_1, raw_diff2_3, out_multiplier, out_left_shift, out_zero_bias);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out1, raw_diff4_5, raw_diff6_7, out_multiplier, out_left_shift, out_zero_bias);
        // Clamp output
        LIMIT16X4(out2, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        LIMIT16X4(out3, out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

        STORE_8X4_FROM_16X4(p_c, out2);
        STORE_8X4_FROM_16X4(p_c, out3);
      }
    }
    else
    {
      ALIGN_REGISTER_TYPE va_a, va_b;
      PRIME_8X4F(p_a, va_a);
      PRIME_8X4F(p_b, va_b);

      for(j = 0; j < num_simd8_ops; j++)
      {
#if XCHAL_HAVE_HIFI1
        AE_LA8X4S_IP(a0_3, va_a, p_a);
        AE_LA8X4S_IP(a4_7, va_a, p_a);
        AE_LA8X4S_IP(b0_3, va_b, p_b);
        AE_LA8X4S_IP(b4_7, va_b, p_b);
#else
        AE_LA8X4F_IP(a0_3, va_a, p_a);
        AE_LA8X4F_IP(a4_7, va_a, p_a);
        AE_LA8X4F_IP(b0_3, va_b, p_b);
        AE_LA8X4F_IP(b4_7, va_b, p_b);

        a0_3 = AE_SRAI16(a0_3, 8);
        a4_7 = AE_SRAI16(a4_7, 8);
        b0_3 = AE_SRAI16(b0_3, 8);
        b4_7 = AE_SRAI16(b4_7, 8);
#endif
        // Add input zero bias
        a0_3 = AE_SUB16S(a0_3, za);
        a4_7 = AE_SUB16S(a4_7, za);
        b0_3 = AE_SUB16S(b0_3, zb);
        b4_7 = AE_SUB16S(b4_7, zb);

        // LSH (and promote to 32-bit)
#if XCHAL_HAVE_HIFI1
        shifted_a0_1 = AE_SEXT32X2D16_32(a0_3);
        shifted_a2_3 = AE_SEXT32X2D16_10(a0_3);
#else
        AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
#endif
        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);
#if XCHAL_HAVE_HIFI1
        shifted_a4_5 = AE_SEXT32X2D16_32(a4_7);
        shifted_a6_7 = AE_SEXT32X2D16_10(a4_7);
#else
        AE_MUL16X4(shifted_a4_5, shifted_a6_7, a4_7, AE_MOVDA16(1));
#endif
        shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
        shifted_a6_7 = AE_SLAA32S(shifted_a6_7, left_shift);
#if XCHAL_HAVE_HIFI1
        shifted_b0_1 = AE_SEXT32X2D16_32(b0_3);
        shifted_b2_3 = AE_SEXT32X2D16_10(b0_3);
#else
        AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, AE_MOVDA16(1));
#endif
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);
#if XCHAL_HAVE_HIFI1
        shifted_b4_5 = AE_SEXT32X2D16_32(b4_7);
        shifted_b6_7 = AE_SEXT32X2D16_10(b4_7);
#else
        AE_MUL16X4(shifted_b4_5, shifted_b6_7, b4_7, AE_MOVDA16(1));
#endif
        shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
        shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);

        raw_diff0_1 = raw_diff2_3 = raw_diff4_5 = raw_diff6_7 = AE_ZERO32();
        // Scaled input
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_b0_1, shifted_b2_3, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_diff4_5, raw_diff6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff4_5, raw_diff6_7, shifted_b4_5, shifted_b6_7, inp2_multiplier, inp2_left_shift);
    // Squared Diff
#if XCHAL_HAVE_HIFI1
        raw_diff0_1 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32);
        raw_diff2_3 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff2_3, raw_diff2_3), AE_MUL32_LL(raw_diff2_3, raw_diff2_3), 32);
        raw_diff4_5 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff4_5, raw_diff4_5), AE_MUL32_LL(raw_diff4_5, raw_diff4_5), 32);
        raw_diff6_7 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff6_7, raw_diff6_7), AE_MUL32_LL(raw_diff6_7, raw_diff6_7), 32);
#else
        raw_diff0_1 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32)));
        raw_diff2_3 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff2_3, raw_diff2_3), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff2_3, raw_diff2_3), 32)));
        raw_diff4_5 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff4_5, raw_diff4_5), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff4_5, raw_diff4_5), 32)));
        raw_diff6_7 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff6_7, raw_diff6_7), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff6_7, raw_diff6_7), 32)));
#endif
        // Raw Output
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_diff0_1, raw_diff2_3, out_multiplier, out_left_shift, out_zero_bias);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out1, raw_diff4_5, raw_diff6_7, out_multiplier, out_left_shift, out_zero_bias);
        // Clamp output
        LIMIT16X4(out2, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        LIMIT16X4(out3, out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

        STORE_8X4_FROM_16X4(p_c, out2);
        STORE_8X4_FROM_16X4(p_c, out3);
      }
    }
  }

  if(num_scalar_ops!=0){
    for(i = 0; i < out_lc; i++)
    {
      p_a = (WORD8 *)&p_inp1[i * in_lc + (num_simd8_ops << 3)];
      p_c = (WORD8 *)&p_out[i * in_lc + (num_simd8_ops << 3)];
      p_b = (WORD8 *)&p_inp2[num_simd8_ops << 3];
      for(j = 0; j< num_scalar_ops; j++)
      {
        b0_3 = AE_MOVDA16(p_b[j]);
        a0_3 = AE_MOVDA16(p_a[j]);
        a0_3 = AE_SUB16S(a0_3, za);
        b0_3 = AE_SUB16S(b0_3, zb);
#if XCHAL_HAVE_HIFI1
        shifted_a0_1 = AE_SEXT32X2D16_32(a0_3);
        shifted_a2_3 = AE_SEXT32X2D16_10(a0_3);
        shifted_b0_1 = AE_SEXT32X2D16_32(b0_3);
        shifted_b2_3 = AE_SEXT32X2D16_10(b0_3);
#else
        AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
        AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, AE_MOVDA16(1));
#endif
        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        raw_diff0_1 = AE_ZERO32();
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2_OUT32(raw_diff0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2_OUT32(raw_diff0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift);
#if XCHAL_HAVE_HIFI1
        raw_diff0_1 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32);
#else
        raw_diff0_1 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32)));
#endif
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT16_ZB(out0, raw_diff0_1, out_multiplier, out_left_shift, out_zero_bias);
        LIMIT16X4(out1, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        *(WORD8 *)p_c = (WORD8)AE_MOVAD16_0(out1);
        p_c++;
     }
   }
  }
}


static void internal_elm_squared_diff_broadcast_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  int i;
  WORD8 * __restrict__ p_a = (WORD8 *)p_inp1;
  WORD8 * __restrict__ p_b = (WORD8 *)p_inp2;
  WORD8 *__restrict__ p_c =          p_out;

  ae_int16x4 a0_7, b;

  ae_int16x4  za = AE_MOVDA16(-inp1_zero_bias);
  ae_int16x4  zb = AE_MOVDA16(-inp2_zero_bias);
  WORD32 a_ls, a_mult, b_ls, b_mult;
  a_ls = inp1_left_shift;
  a_mult = inp1_multiplier;
  b_ls = inp2_left_shift;
  b_mult = inp2_multiplier;

  ae_int16x4 a0_3, b0;
  ae_int32x2 shifted_a0_1, shifted_a2_3;
  ae_int32x2 shifted_b0, shifted_b1;
  ae_int32x2 scaled_b0;

  ae_int32x2 raw_diff0_1, raw_diff2_3;
  ae_int16x4 out0, out1;

  const int num_simd4_ops = num_elm >> 2;
  const int num_scalar_ops = num_elm & 3;

  xtbool io_pointers_aligned = ((uintptr_t)p_inp1%4 == 0) && ((uintptr_t)p_inp2%4==0) && ((uintptr_t)p_out%4==0);
  if(io_pointers_aligned)
  {
    b = AE_MOVDA16(p_b[0]);
    b0 = AE_SUB16(b, zb);

    AE_MUL16X4(shifted_b0, shifted_b1, b0, AE_MOVDA16(1));
    shifted_b0 = AE_SLAA32S(shifted_b0, left_shift);
    shifted_b1 = AE_SLAA32S(shifted_b1, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0, shifted_b0, b_mult, b_ls);

    for(i=0; i<num_simd4_ops; i++)
    {
#if XCHAL_HAVE_HIFI1
      AE_L8X4S_IP(a0_7, p_a, 4);
#else
      AE_L8X4F_IP(a0_7, p_a, 4);
      a0_7 = AE_SRAI16(a0_7, 8);
#endif
      // Add input zero bias
      a0_3 = AE_SUB16(a0_7, za);

      // LSH (and promote to 32-bit)
#if XCHAL_HAVE_HIFI1
      shifted_a0_1 = AE_SEXT32X2D16_32(a0_3);
      shifted_a2_3 = AE_SEXT32X2D16_10(a0_3);
#else
      AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
#endif
      shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
      shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

      raw_diff0_1 = raw_diff2_3 = scaled_b0;
      MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_a0_1, shifted_a2_3, a_mult, a_ls);
#if XCHAL_HAVE_HIFI1
      raw_diff0_1 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32);
      raw_diff2_3 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff2_3, raw_diff2_3), AE_MUL32_LL(raw_diff2_3, raw_diff2_3), 32);
#else
      raw_diff0_1 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32)));
      raw_diff2_3 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff2_3, raw_diff2_3), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff2_3, raw_diff2_3), 32)));
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_diff0_1, raw_diff2_3, out_multiplier, out_left_shift, out_zero_bias);
      LIMIT16X4(out1, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      STORE_8X4_FROM_16X4(p_c, out1);
    }
  }
  else
  {
    ALIGN_REGISTER_TYPE va_a;
    PRIME_8X4F(p_a, va_a);

    b = AE_MOVDA16(p_b[0]);
    b0 = AE_SUB16(b, zb);

#if XCHAL_HAVE_HIFI1
    shifted_b0 = AE_SEXT32X2D16_32(b0);
    shifted_b1 = AE_SEXT32X2D16_10(b0);
#else
    AE_MUL16X4(shifted_b0, shifted_b1, b0, AE_MOVDA16(1));
#endif
    shifted_b0 = AE_SLAA32S(shifted_b0, left_shift);
    shifted_b1 = AE_SLAA32S(shifted_b1, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0, shifted_b0, b_mult, b_ls);

    for(i=0; i<num_simd4_ops; i++)
    {
#if XCHAL_HAVE_HIFI1
      AE_LA8X4S_IP(a0_7, va_a, p_a);
#else
      AE_LA8X4F_IP(a0_7, va_a, p_a);
      a0_7 = AE_SRAI16(a0_7, 8);
#endif
      // Add input zero bias
      a0_3 = AE_SUB16(a0_7, za);

      // LSH (and promote to 32-bit)
#if XCHAL_HAVE_HIFI1
        shifted_a0_1 = AE_SEXT32X2D16_32(a0_3);
        shifted_a2_3 = AE_SEXT32X2D16_10(a0_3);
#else
        AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
#endif
      shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
      shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

      raw_diff0_1 = raw_diff2_3 = scaled_b0;
      MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_a0_1, shifted_a2_3, a_mult, a_ls);
#if XCHAL_HAVE_HIFI1
      raw_diff0_1 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32);
      raw_diff2_3 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff2_3, raw_diff2_3), AE_MUL32_LL(raw_diff2_3, raw_diff2_3), 32);
#else
      raw_diff0_1 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32)));
      raw_diff2_3 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff2_3, raw_diff2_3), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff2_3, raw_diff2_3), 32))); 
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_diff0_1, raw_diff2_3, out_multiplier, out_left_shift, out_zero_bias);
      LIMIT16X4(out1, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      STORE_8X4_FROM_16X4(p_c, out1);
    }
  }
  for(i=0; i<num_scalar_ops; i++)
  {
    a0_7 = AE_MOVDA16(p_a[i]);
    a0_3 = AE_SUB16(a0_7, za);

#if XCHAL_HAVE_HIFI1
        shifted_a0_1 = AE_SEXT32X2D16_32(a0_3);
        shifted_a2_3 = AE_SEXT32X2D16_10(a0_3);
#else
    AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
#endif
    shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
    shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

    raw_diff0_1 = raw_diff2_3 = scaled_b0;
    MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2_OUT32(raw_diff0_1, shifted_a0_1, a_mult, a_ls);
#if XCHAL_HAVE_HIFI1
    raw_diff0_1 = AE_TRUNCA32X2F64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32);
#else
    raw_diff0_1 = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_HH(raw_diff0_1, raw_diff0_1), 32)),AE_MOVINT32X2_FROMINT64(AE_SLAI64S(AE_MUL32_LL(raw_diff0_1, raw_diff0_1), 32)));
#endif
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT16_ZB(out0, raw_diff0_1, out_multiplier, out_left_shift, out_zero_bias);
    LIMIT16X4(out1, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
    *p_c =  (WORD8)AE_MOVAD16_3(out1);
    p_c++;
  }
}


WORD32 xa_nn_elm_squared_diff_broadcast_4D_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                      const WORD8 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                      const WORD8 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape,
                            WORD32  inp2_zero_bias,
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
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
    // inp1_strides[i] = inp1_strides[i + 1] * p_inp1_shape[i + 1];
    // inp2_strides[i] = inp2_strides[i + 1] * p_inp2_shape[i + 1];
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

  WORD8 *p_out_tmp = p_out;
  const WORD8 *__restrict__ p_inp1_tmp = p_inp1;
  const WORD8 *__restrict__ p_inp2_tmp = p_inp2;
  if(need_broadcast == 0)
  {
    internal_elm_squared_diff_broadcast_2D_asym8sxasym8s_asym8s(
                p_out,
                out_zero_bias,
                out_left_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1,
                inp1_zero_bias,
                inp1_left_shift,
                inp1_multiplier,
                p_inp2,
                inp2_zero_bias,
                inp2_left_shift,
                inp2_multiplier,
                left_shift,
                1,
                p_out_shape[0] * inp1_strides[0]);
  }
  else if(inp1_strides[3] == inp2_strides[3])
  {
    WORD32 in_lc, out_lc;
    WORD32 inp1_zb, inp1_ls, inp1_mult;
    WORD32 inp2_zb, inp2_ls, inp2_mult;

    inp1_zb = inp1_zero_bias;
    inp1_ls = inp1_left_shift;
    inp1_mult = inp1_multiplier;
    inp2_zb = inp2_zero_bias;
    inp2_ls = inp2_left_shift;
    inp2_mult = inp2_multiplier;

    in_lc = p_out_shape[2] * p_out_shape[3];
    out_lc = 1;
    if(inp1_strides[2] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp2_ls = inp1_left_shift;
      inp2_mult = inp1_multiplier;
      inp1_zb = inp2_zero_bias;
      inp1_ls = inp2_left_shift;
      inp1_mult = inp2_multiplier;
      const WORD8 *tmp;
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
      const WORD8 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const WORD8 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        internal_elm_squared_diff_broadcast_2D_asym8sxasym8s_asym8s(
            p_out_tmp,
            out_zero_bias,
            out_left_shift,
            out_multiplier,
            out_activation_min,
            out_activation_max,
            p_inp1_tmp0,
            inp1_zb,
            inp1_ls,
            inp1_mult,
            p_inp2_tmp0,
            inp2_zb,
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
    WORD32 inp1_zb, inp1_ls, inp1_mult;
    WORD32 inp2_zb, inp2_ls, inp2_mult;
    inp1_zb = inp1_zero_bias;
    inp1_ls = inp1_left_shift;
    inp1_mult = inp1_multiplier;
    inp2_zb = inp2_zero_bias;
    inp2_ls = inp2_left_shift;
    inp2_mult = inp2_multiplier;
    /* Reversing the inputs is okay because difference is squared */
    if(inp1_strides[3] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp2_ls = inp1_left_shift;
      inp2_mult = inp1_multiplier;
      inp1_zb = inp2_zero_bias;
      inp1_ls = inp2_left_shift;
      inp1_mult = inp2_multiplier;
      const WORD8 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
    }
    internal_elm_squared_diff_broadcast_asym8sxasym8s_asym8s(
        p_out_tmp,
        out_zero_bias,
        out_left_shift,
        out_multiplier,
        out_activation_min,
        out_activation_max,
        p_inp1_tmp,
        inp1_zb,
        inp1_ls,
        inp1_mult,
        p_inp2_tmp,
        inp2_zb,
        inp2_ls,
        inp2_mult,
        left_shift,
        p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3]);
  }
  else
  {
    WORD32 inp1_zb, inp1_ls, inp1_mult;
    WORD32 inp2_zb, inp2_ls, inp2_mult;
    inp1_zb = inp1_zero_bias;
    inp1_ls = inp1_left_shift;
    inp1_mult = inp1_multiplier;
    inp2_zb = inp2_zero_bias;
    inp2_ls = inp2_left_shift;
    inp2_mult = inp2_multiplier;
    if(inp1_strides[3] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp2_ls = inp1_left_shift;
      inp2_mult = inp1_multiplier;
      inp1_zb = inp2_zero_bias;
      inp1_ls = inp2_left_shift;
      inp1_mult = inp2_multiplier;
      const WORD8 *tmp;
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
      const WORD8 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const WORD8 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        const WORD8 *__restrict__ p_inp1_tmp1 = p_inp1_tmp0;
        const WORD8 *__restrict__ p_inp2_tmp1 = p_inp2_tmp0;
        for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
        {
          {
            internal_elm_squared_diff_broadcast_asym8sxasym8s_asym8s(
                p_out_tmp,
                out_zero_bias,
                out_left_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1_tmp1,
                inp1_zb,
                inp1_ls,
                inp1_mult,
                p_inp2_tmp1,
                inp2_zb,
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

