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

static void internal_elm_add_broadcast_2D_asym16sxasym16s_asym16s(WORD16 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   WORD16 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const   WORD16 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
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


  const ae_int32x2 za = AE_MOVDA32(-inp1_zero_bias);
  const ae_int32x2 zb = AE_MOVDA32(-inp2_zero_bias);

  /* intermediate results and scratch registers */
  ae_int16x4 a0_3, a4_7, b0_3, b4_7;

  ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
  ae_int32x2 shifted_b0_1, shifted_b2_3, shifted_b4_5, shifted_b6_7;

  ae_int32x2 raw_sum0_1, raw_sum2_3, raw_sum4_5, raw_sum6_7;

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
  
  //SEANET SPECIAL CASE OPTIMIZATION
  if((out_left_shift <= 0) && (inp1_left_shift <= 0) && (inp2_left_shift <=0 ) && (left_shift <= 15))
  {
    ae_f32x2 multiplier1, multiplier2, op_multiplier, op_zero_bias, activation_min, activation_max;
    int inp1_right_shift = (0XFFFFFFFF << (31 + inp1_left_shift));
    int inp2_right_shift = (0XFFFFFFFF << (31 + inp2_left_shift));
    int out_right_shift  = (0XFFFFFFFF << (31 + out_left_shift)); 
    
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    int inp1_left_shift_hifi1s = 31 - inp1_left_shift; 
    int inp2_left_shift_hifi1s = 31 - inp2_left_shift; 
    int out_left_shift_hifi1s = 31 - out_left_shift; 
    inp1_left_shift_hifi1s = inp1_left_shift_hifi1s << 16 | inp1_left_shift_hifi1s; 
    inp2_left_shift_hifi1s = inp2_left_shift_hifi1s << 16 | inp2_left_shift_hifi1s; 
    out_left_shift_hifi1s = out_left_shift_hifi1s << 16 | out_left_shift_hifi1s; 
#endif

    WORD32 const1 = 1 << left_shift;
    ae_int32x2 const1_32x2 =  AE_MOVDA32X2(const1,const1);
            
    WORD32 const2_inp1 = (WORD32)((UWORD32)inp1_zero_bias << left_shift);
    WORD32 const2_inp2 = (WORD32)((UWORD32)inp2_zero_bias << left_shift);

    ae_int32x2 const2_32x2_LO_1 =  AE_MOVDA32X2(const2_inp1,const2_inp1);
    ae_int32x2 const2_32x2_LO_2 =  AE_MOVDA32X2(const2_inp2,const2_inp2);

    multiplier1 = AE_NEG32S(AE_MOVDA32(inp1_multiplier));
    multiplier2 = AE_NEG32S(AE_MOVDA32(inp2_multiplier));
    op_multiplier = AE_NEG32S(AE_MOVDA32(out_multiplier));
    op_zero_bias = AE_MOVDA32(out_zero_bias);

    activation_min = AE_MOVDA32(out_activation_min);
    activation_max = AE_MOVDA32(out_activation_max);

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
        ae_f32x2 scaled_v1,scaled_v2,scaled_v3,scaled_v4;      
        ae_f32x2 scaled_v5,scaled_v6,scaled_v7,scaled_v8;      
        ae_f32x2 raw_sum12, raw_sum34, raw_sum56, raw_sum78;
        ae_f32x2 raw_out12, raw_out34, raw_out56, raw_out78;

        ae_f32x2 d_0,d_1,d_2,d_3,d_4,d_5,d_6,d_7;

        AE_LA16X4_IP(a0_3, va_a, (ae_int16x4 *)p_a);
        AE_LA16X4_IP(a4_7, va_a, (ae_int16x4 *)p_a);
        AE_LA16X4_IP(b0_3, va_b, (ae_int16x4 *)p_b);
        AE_LA16X4_IP(b4_7, va_b, (ae_int16x4 *)p_b);

        d_0 = d_1 = d_2 = d_3 = const2_32x2_LO_1;
        d_4 = d_5 = d_6 = d_7 = const2_32x2_LO_2;

        AE_MULAP32X16X2_H(d_0,const1_32x2,(a0_3));
        AE_MULAP32X16X2_L(d_1,const1_32x2,(a0_3));
        AE_MULAP32X16X2_H(d_2,const1_32x2,(a4_7));
        AE_MULAP32X16X2_L(d_3,const1_32x2,(a4_7));

        AE_MULAP32X16X2_H(d_4,const1_32x2,(b0_3));
        AE_MULAP32X16X2_L(d_5,const1_32x2,(b0_3));
        AE_MULAP32X16X2_H(d_6,const1_32x2,(b4_7));
        AE_MULAP32X16X2_L(d_7,const1_32x2,(b4_7));

#if !TFLITE_SINGLE_ROUNDING
        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v1, d_0, multiplier1, inp1_right_shift)
        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v2, d_1, multiplier1, inp1_right_shift)
        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v3, d_2, multiplier1, inp1_right_shift)
        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v4, d_3, multiplier1, inp1_right_shift)

        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v5, d_4, multiplier2, inp2_right_shift)
        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v6, d_5, multiplier2, inp2_right_shift)
        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v7, d_6, multiplier2, inp2_right_shift)
        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v8, d_7, multiplier2, inp2_right_shift)
        // Raw Sum
        raw_sum12   = AE_ADD32S(scaled_v1, scaled_v5);
        raw_sum34   = AE_ADD32S(scaled_v2, scaled_v6);
        raw_sum56   = AE_ADD32S(scaled_v3, scaled_v7);
        raw_sum78   = AE_ADD32S(scaled_v4, scaled_v8);
        // Raw Output
        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(raw_out12, raw_sum12, op_multiplier, out_right_shift)
        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(raw_out34, raw_sum34, op_multiplier, out_right_shift)
        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(raw_out56, raw_sum56, op_multiplier, out_right_shift)
        MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(raw_out78, raw_sum78, op_multiplier, out_right_shift)
#else
       (void)inp1_right_shift; (void)inp2_right_shift; (void)out_right_shift;
       (void)multiplier1; (void)multiplier2; (void)op_multiplier;
#if XCHAL_HAVE_HIFI1S
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_v1, d_0, inp1_multiplier, inp1_left_shift_hifi1s, inp1_right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_v2, d_1, inp1_multiplier, inp1_left_shift_hifi1s, inp1_right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_v3, d_2, inp1_multiplier, inp1_left_shift_hifi1s, inp1_right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_v4, d_3, inp1_multiplier, inp1_left_shift_hifi1s, inp1_right_shift);

        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_v5, d_4, inp2_multiplier, inp2_left_shift_hifi1s, inp2_right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_v6, d_5, inp2_multiplier, inp2_left_shift_hifi1s, inp2_right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_v7, d_6, inp2_multiplier, inp2_left_shift_hifi1s, inp2_right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_v8, d_7, inp2_multiplier, inp2_left_shift_hifi1s, inp2_right_shift);
#else
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v1, d_0, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v2, d_1, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v3, d_2, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v4, d_3, inp1_multiplier, inp1_left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v5, d_4, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v6, d_5, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v7, d_6, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v8, d_7, inp2_multiplier, inp2_left_shift);
#endif        
        // Raw Sum
        raw_sum12   = AE_ADD32S(scaled_v1, scaled_v5);
        raw_sum34   = AE_ADD32S(scaled_v2, scaled_v6);
        raw_sum56   = AE_ADD32S(scaled_v3, scaled_v7);
        raw_sum78   = AE_ADD32S(scaled_v4, scaled_v8);
        // Raw Output
#if XCHAL_HAVE_HIFI1S        
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(raw_out12, raw_sum12, out_multiplier, out_left_shift_hifi1s, out_right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(raw_out34, raw_sum34, out_multiplier, out_left_shift_hifi1s, out_right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(raw_out56, raw_sum56, out_multiplier, out_left_shift_hifi1s, out_right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(raw_out78, raw_sum78, out_multiplier, out_left_shift_hifi1s, out_right_shift);
#else
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out12, raw_sum12, out_multiplier, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out34, raw_sum34, out_multiplier, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out56, raw_sum56, out_multiplier, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out78, raw_sum78, out_multiplier, out_left_shift);
#endif        
#endif

        raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);
        raw_out34 = AE_ADD32S(raw_out34, op_zero_bias);
        raw_out56 = AE_ADD32S(raw_out56, op_zero_bias);
        raw_out78 = AE_ADD32S(raw_out78, op_zero_bias);

        // Clamp output
        AE_MINMAX32(raw_out12,activation_min,activation_max);
        AE_MINMAX32(raw_out34,activation_min,activation_max);
        AE_MINMAX32(raw_out56,activation_min,activation_max);
        AE_MINMAX32(raw_out78,activation_min,activation_max);

        out0 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(raw_out12), AE_MOVINT16X4_FROMINT32X2(raw_out34));
        out1 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(raw_out56), AE_MOVINT16X4_FROMINT32X2(raw_out78));
        AE_SA16X4_IP(out0, va_c, (ae_int16x4 *)p_c);
        AE_SA16X4_IP(out1, va_c, (ae_int16x4 *)p_c);

      }
      AE_SA64POS_FP(va_c, (ae_int16x4 *)p_c);
    }
  }

  else
  {
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
        shifted_a0_1 = AE_SUB32(shifted_a0_1, za);
        shifted_a2_3 = AE_SUB32(shifted_a2_3, za);
        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

        AE_MUL16X4(shifted_a4_5, shifted_a6_7, a4_7, AE_MOVDA16(1));
        shifted_a4_5 = AE_SUB32(shifted_a4_5, za);
        shifted_a6_7 = AE_SUB32(shifted_a6_7, za);
        shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
        shifted_a6_7 = AE_SLAA32S(shifted_a6_7, left_shift);

        AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, AE_MOVDA16(1));
        shifted_b0_1 = AE_SUB32(shifted_b0_1, zb);
        shifted_b2_3 = AE_SUB32(shifted_b2_3, zb);
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);

        AE_MUL16X4(shifted_b4_5, shifted_b6_7, b4_7, AE_MOVDA16(1));
        shifted_b4_5 = AE_SUB32(shifted_b4_5, zb);
        shifted_b6_7 = AE_SUB32(shifted_b6_7, zb);
        shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
        shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);

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
      }
      AE_SA64POS_FP(va_c, (ae_int16x4 *)p_c);
    }
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
        shifted_a0_1 = AE_SUB32(shifted_a0_1, za);
        shifted_a2_3 = AE_SUB32(shifted_a2_3, za);
        shifted_b0_1 = AE_SUB32(shifted_b0_1, zb);
        shifted_b2_3 = AE_SUB32(shifted_b2_3, zb);

        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        raw_sum0_1 = AE_ZERO32();
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2_OUT32(raw_sum0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2_OUT32(raw_sum0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift);
        out0_32 = AE_MOVDA32(out_zero_bias);
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2_OUT32(out0_32, raw_sum0_1, out_multiplier, out_left_shift);
        out0 = AE_SAT16X4(out0_32, out0_32);
        LIMIT16X4(out1, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        *p_c++ = (WORD16)(AE_MOVAD16_0(out1));
      }
    }
  }

  return;
}

static void internal_elm_add_broadcast_asym16sxasym16s_asym16s(WORD16 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD16 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const    WORD16 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  int i;
  WORD16 * __restrict__ p_a = (WORD16 *)p_inp1;
  WORD16 * __restrict__ p_b = (WORD16 *)p_inp2;
  WORD16 *__restrict__ p_c =          p_out;

  ae_int16x4 b;

  ae_int32x2  za = AE_MOVDA32(-inp1_zero_bias);
  ae_int32x2  zb = AE_MOVDA32(-inp2_zero_bias);

  ae_int16x4 a0_3;
  ae_int32x2 shifted_a0_1, shifted_a2_3;
  ae_int32x2 shifted_b0, shifted_b1;
  ae_int32x2 scaled_b0;

  ae_int32x2 raw_sum0_1, raw_sum2_3;
  ae_int16x4 out0, out1;

  const int num_simd4_ops = num_elm >> 2;
  const int num_scalar_ops = num_elm & 3;

  if((out_left_shift <= 0) && (inp1_left_shift <= 0) && (inp2_left_shift <=0 ) && (left_shift <= 15))
  {
    ae_f32x2 scaled_v1, scaled_v2;
    ae_f32x2 scaled_v3;
    ae_f32x2 raw_sum12, raw_sum34;
    ae_f32x2 raw_out12, raw_out34;
    ae_f32x2 d_0, d_1, d_2;
    ae_int16x4 clamped_out;
    ae_f32x2 multiplier1, multiplier2, op_multiplier, op_zero_bias, activation_min, activation_max;

    int inp1_right_shift = (0XFFFFFFFF << (31 + inp1_left_shift));
    int inp2_right_shift = (0XFFFFFFFF << (31 + inp2_left_shift));
    int out_right_shift  = (0XFFFFFFFF << (31 + out_left_shift));

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    int inp1_left_shift_hifi1s = 31 - inp1_left_shift; 
    int inp2_left_shift_hifi1s = 31 - inp2_left_shift; 
    int out_left_shift_hifi1s = 31 - out_left_shift; 
    inp1_left_shift_hifi1s = inp1_left_shift_hifi1s << 16 | inp1_left_shift_hifi1s; 
    inp2_left_shift_hifi1s = inp2_left_shift_hifi1s << 16 | inp2_left_shift_hifi1s; 
    out_left_shift_hifi1s = out_left_shift_hifi1s << 16 | out_left_shift_hifi1s; 
#endif

    WORD32 const1 = 1 << left_shift;
    ae_int32x2 const1_32x2 =  AE_MOVDA32X2(const1,const1);
            
    WORD32 const2_inp1 = inp1_zero_bias << left_shift;
    WORD32 const2_inp2 = inp2_zero_bias << left_shift;

    ae_int32x2 const2_32x2_LO_1 =  AE_MOVDA32X2(const2_inp1,const2_inp1);
    ae_int32x2 const2_32x2_LO_2 =  AE_MOVDA32X2(const2_inp2,const2_inp2);

    multiplier1   = AE_NEG32S(AE_MOVDA32(inp1_multiplier));
    multiplier2   = AE_NEG32S(AE_MOVDA32(inp2_multiplier));
    op_multiplier = AE_NEG32S(AE_MOVDA32(out_multiplier));
 
    activation_min = AE_MOVDA32(out_activation_min);
    activation_max = AE_MOVDA32(out_activation_max);

    op_zero_bias = AE_MOVDA32(out_zero_bias);

    b = AE_MOVDA16(p_b[0]);
    d_2 = const2_32x2_LO_2;
    AE_MULAP32X16X2_H(d_2,const1_32x2,b);
#if !TFLITE_SINGLE_ROUNDING
    MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v3, d_2, multiplier2, inp2_right_shift)
#else
#if XCHAL_HAVE_HIFI1S
    (void)inp2_right_shift;
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_v3, d_2, inp2_multiplier, inp2_left_shift_hifi1s, inp2_right_shift);
#else
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v3, d_2, inp2_multiplier, inp2_left_shift);
#endif    
#endif
     
    ae_valign va_a;
    va_a = AE_LA64_PP(p_a);
    ae_valign va_c = AE_ZALIGN64();

    for(i=0; i<num_simd4_ops; i++)
    {
      AE_LA16X4_IP(a0_3, va_a, (ae_int16x4 *)p_a);

      d_0 = d_1 = const2_32x2_LO_1;
      AE_MULAP32X16X2_H(d_0, const1_32x2, a0_3);
      AE_MULAP32X16X2_L(d_1, const1_32x2, a0_3);

#if !TFLITE_SINGLE_ROUNDING
      MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v1, d_0, multiplier1, inp1_right_shift)
      MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v2, d_1, multiplier1, inp1_right_shift)
      // Raw Sum
      raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);
      raw_sum34 = AE_ADD32S(scaled_v2, scaled_v3);
      // Raw Output
      MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(raw_out12, raw_sum12, op_multiplier, out_right_shift)
      MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(raw_out34, raw_sum34, op_multiplier, out_right_shift)
#else
      (void)inp1_right_shift; (void)inp2_right_shift; (void)out_right_shift;
      (void)multiplier1; (void)multiplier2; (void)op_multiplier;

#if XCHAL_HAVE_HIFI1S
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_v1, d_0, inp1_multiplier, inp1_left_shift_hifi1s, inp1_left_shift_hifi1s);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_v2, d_1, inp1_multiplier, inp1_left_shift_hifi1s, inp1_left_shift_hifi1s);
#else
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v1, d_0, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v2, d_1, inp1_multiplier, inp1_left_shift);
#endif      
      // Raw Sum
      raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);
      raw_sum34 = AE_ADD32S(scaled_v2, scaled_v3);
      // Raw Output
#if XCHAL_HAVE_HIFI1S
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(raw_out12, raw_sum12, out_multiplier, out_left_shift_hifi1s, out_left_shift_hifi1s);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(raw_out34, raw_sum34, out_multiplier, out_left_shift_hifi1s, out_left_shift_hifi1s);
#else      
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out12, raw_sum12, out_multiplier, out_left_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out34, raw_sum34, out_multiplier, out_left_shift);
#endif      
#endif     
      raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);
      raw_out34 = AE_ADD32S(raw_out34, op_zero_bias);

      AE_MINMAX32(raw_out12, activation_min, activation_max);
      AE_MINMAX32(raw_out34, activation_min, activation_max);

      clamped_out = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(raw_out12), AE_MOVINT16X4_FROMINT32X2(raw_out34));	

      AE_SA16X4_IP(clamped_out, va_c, (ae_int16x4 *)p_c);
    }
    AE_SA64POS_FP(va_c, p_c); 
  }

  else
  {
    ae_valign va_a;
    va_a = AE_LA64_PP(p_a);
    ae_valign va_c = AE_ZALIGN64();

    b = AE_MOVDA16(p_b[0]);
    AE_MUL16X4(shifted_b0, shifted_b1, b, AE_MOVDA16(1));
    shifted_b0 = AE_SUB32(shifted_b0, zb);
    shifted_b1 = AE_SUB32(shifted_b1, zb);

    shifted_b0 = AE_SLAA32S(shifted_b0, left_shift);
    shifted_b1 = AE_SLAA32S(shifted_b1, left_shift);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    int inp2_left_shift_hifi1s = 31 - inp2_left_shift; 
    inp2_left_shift_hifi1s = inp2_left_shift_hifi1s << 16 | inp2_left_shift_hifi1s; 
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_b0, shifted_b0, inp2_multiplier, inp2_left_shift_hifi1s, inp2_right_shift);
#else
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0, shifted_b0, inp2_multiplier, inp2_left_shift);
#endif
    for(i=0; i<num_simd4_ops; i++)
    {
      ae_int32x2 out0_32, out1_32;
      AE_LA16X4_IP(a0_3, va_a, (ae_int16x4 *)p_a);

      AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
      shifted_a0_1 = AE_SUB32(shifted_a0_1, za);
      shifted_a2_3 = AE_SUB32(shifted_a2_3, za);
      shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
      shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

      raw_sum0_1 = raw_sum2_3 = scaled_b0;
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
      out0_32 = out1_32 = AE_MOVDA32(out_zero_bias);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(out0_32, out1_32, raw_sum0_1, raw_sum2_3, out_multiplier, out_left_shift);
      out0 = AE_SAT16X4(out0_32, out1_32);
      LIMIT16X4(out1, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      AE_SA16X4_IP(out1, va_c, (ae_int16x4 *)p_c);
    }
    AE_SA64POS_FP(va_c, p_c);
  }

  b = AE_MOVDA16(p_b[0]);
  AE_MUL16X4(shifted_b0, shifted_b1, b, AE_MOVDA16(1));
  shifted_b0 = AE_SUB32(shifted_b0, zb);
  shifted_b1 = AE_SUB32(shifted_b1, zb);

  shifted_b0 = AE_SLAA32S(shifted_b0, left_shift);
  shifted_b1 = AE_SLAA32S(shifted_b1, left_shift);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
  int inp2_left_shift_hifi1s = 31 - inp2_left_shift; 
  inp2_left_shift_hifi1s = inp2_left_shift_hifi1s << 16 | inp2_left_shift_hifi1s; 
  MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_b0, shifted_b0, inp2_multiplier, inp2_left_shift_hifi1s, inp2_right_shift);
#else
  MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0, shifted_b0, inp2_multiplier, inp2_left_shift);
#endif
  for(i=0; i<num_scalar_ops; i++)
  {
    ae_int32x2 out0_32;
    a0_3 = AE_MOVDA16(p_a[i]);

    AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
    shifted_a0_1 = AE_SUB32(shifted_a0_1, za);
    shifted_a2_3 = AE_SUB32(shifted_a2_3, za);
    shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
    shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

    raw_sum0_1 = raw_sum2_3 = scaled_b0;
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2_OUT32(raw_sum0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
    out0_32 = AE_MOVDA32(out_zero_bias);
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2_OUT32(out0_32, raw_sum0_1, out_multiplier, out_left_shift);
    out0 = AE_SAT16X4(out0_32, out0_32);
    LIMIT16X4(out1, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
    *p_c = (WORD16)(AE_MOVAD16_3(out1));
    p_c++;
  }

  return;
}

WORD32 xa_nn_elm_add_broadcast_4D_asym16sxasym16s_asym16s(WORD16 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                      const WORD16 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                      const WORD16 * __restrict__ p_inp2,
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -32768) || (out_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -32767) || (inp1_zero_bias > 32768)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -32767) || (inp2_zero_bias > 32768)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
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
    internal_elm_add_broadcast_2D_asym16sxasym16s_asym16s(
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
        internal_elm_add_broadcast_2D_asym16sxasym16s_asym16s(
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
    if(inp1_strides[3] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp2_ls = inp1_left_shift;
      inp2_mult = inp1_multiplier;
      inp1_zb = inp2_zero_bias;
      inp1_ls = inp2_left_shift;
      inp1_mult = inp2_multiplier;
      const WORD16 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
    }

    internal_elm_add_broadcast_asym16sxasym16s_asym16s(
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
            internal_elm_add_broadcast_asym16sxasym16s_asym16s(
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

