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

#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_equal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  vbool2 b32, b10;
  vbool4 flag;

  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);
  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 inp1_mul = AE_MOVDA32(inp1_multiplier);
  ae_int32x2 inp2_mul = AE_MOVDA32(inp2_multiplier);

  ae_int16x4 out = AE_ZERO16();

  ae_valign in1_a, in2_a, out_a;
  in1_a = AE_LA64_PP(p_in1);
  in2_a = AE_LA64_PP(p_in2);
  out_a = AE_ZALIGN64();

  for(i=0; i<(num_elm >> 2); i++)
  {
    AE_LA8X4S_IP(m1, in1_a, p_in1);
    AE_LA8X4S_IP(m2, in2_a, p_in2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    x32 = AE_SEXT32X2D16_32(x1);
    x10 = AE_SEXT32X2D16_10(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    y10 = AE_SEXT32X2D16_10(y1);

    x32 = AE_SLAA32S(x32, left_shift);
    x10 = AE_SLAA32S(x10, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    y10 = AE_SLAA32S(y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

    b32 = AE_EQ32(dequantized_x32, dequantized_y32);
    b10 = AE_EQ32(dequantized_x10, dequantized_y10);

    flag = vbool2_join_vbool4(b10, b32);
    AE_MOVT16X4(out, ONE_16X4, flag);
    AE_SA8X4U_IP(out, out_a, (ae_int32 *)p_o);
    out = AE_ZERO16();
  }
  
    //Remainder Loop
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    if(rem_length)
    {
        AE_LAV8X4S_XP(m1, in1_a, (ae_int8x4 *)p_in1, rem_length);
        AE_LAV8X4S_XP(m2, in2_a, (ae_int8x4 *)p_in2, rem_length);

        x1 = AE_ADD16(m1, inp1_z_b);
        y1 = AE_ADD16(m2, inp2_z_b);

        x32 = AE_SEXT32X2D16_32(x1);
        x10 = AE_SEXT32X2D16_10(x1);
        y32 = AE_SEXT32X2D16_32(y1);
        y10 = AE_SEXT32X2D16_10(y1);

        x32 = AE_SLAA32S(x32, left_shift);
        x10 = AE_SLAA32S(x10, left_shift);
        y32 = AE_SLAA32S(y32, left_shift);
        y10 = AE_SLAA32S(y10, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

        b32 = AE_EQ32(dequantized_x32, dequantized_y32);
        b10 = AE_EQ32(dequantized_x10, dequantized_y10);

        flag = vbool2_join_vbool4(b10, b32);
        AE_MOVT16X4(out, ONE_16X4, flag);
        AE_SAV8X4U_XP(out, out_a, (ae_int8x4u *)p_o, rem_length);
    }
  AE_SA64POS_FP(out_a, p_o);
#else
  AE_SA64POS_FP(out_a, p_o);

  for(i = 0; i < rem_length; i++)
  {
    AE_L8S_IP(m1, p_in1, 1);
    AE_L8S_IP(m2, p_in2, 1);
    
    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);
    
    x32 = AE_SEXT32X2D16_32(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    
    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    
    dequantized_x32 = AE_MULFP32X2RAS_L(x32, inp1_mul);
    dequantized_x32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_x32), inp1_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_x32), inp1_shift));

    dequantized_y32 = AE_MULFP32X2RAS_L(y32, inp2_mul);
    dequantized_y32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_y32), inp2_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_y32), inp2_shift));

    b32 = AE_EQ32(dequantized_x32, dequantized_y32);

    flag = vbool2_join_vbool4(b32, b32);
    AE_MOVT16X4(out, ONE_16X4, flag);
    
    AE_S8_0_IP_HIFI1(out, p_o, 1);
    
    out = AE_ZERO16();
  }
#endif
  return 0;
}
#else
WORD32 xa_nn_elm_equal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  xtbool2 b32, b10;
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);
  ae_int32x2 ONE_32x2 = AE_MOVDA32(1);

  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 out_32 = AE_ZERO32();
  ae_int32x2 out_10 = AE_ZERO32();
#if XCHAL_HAVE_HIFI4
  int sar_reg_val = AE_MOVASAR();
  int sar_reg_low_half = sar_reg_val & 0x7F;
  sar_reg_val = sar_reg_val >> 7;
  int sar_reg_up_half = sar_reg_val & 0x7F;
  WUR_AE_SAR(left_shift);
#endif
  if(((((unsigned)p_in1)&3) == 0) && ((((unsigned)p_in2)&3) == 0))
  {
    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_L8X4F_IP(m1, p_in1, 4*sizeof(WORD8));
      AE_L8X4F_IP(m2, p_in2, 4*sizeof(WORD8));

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);

#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_EQ32(dequantized_x32, dequantized_y32);
      b10 = AE_EQ32(dequantized_x10, dequantized_y10);

      AE_MOVT32X2(out_32, ONE_32x2, b32);
      AE_MOVT32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  else
  {
    ALIGN_REGISTER_TYPE in1_a, in2_a;

    PRIME_8X4F(p_in1, in1_a);
    PRIME_8X4F(p_in2, in2_a);

    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_LA8X4F_IP(m1, in1_a, p_in1);
      AE_LA8X4F_IP(m2, in2_a, p_in2);

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);

#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_EQ32(dequantized_x32, dequantized_y32);
      b10 = AE_EQ32(dequantized_x10, dequantized_y10);

      AE_MOVT32X2(out_32, ONE_32x2, b32);
      AE_MOVT32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  //Remainder Loop
  for(i = 0; i < rem_length; i++)
  {
    WORD16 i1, i2;

    i1 = (WORD16) *(p_in1 + i);
    i2 = (WORD16) *(p_in2 + i);

    m1 = AE_MOVDA16(i1);
    m2 = AE_MOVDA16(i2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    AE_MUL16X4(x32, x10, x1, ONE_16X4);
    AE_MUL16X4(y32, y10, y1, ONE_16X4);

    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)

    b32 = AE_EQ32(dequantized_x32, dequantized_y32);

    AE_MOVT32X2(out_32, ONE_32x2, b32);

    i1 = (WORD16)(AE_MOVAD32_H(out_32));
    *p_o++ = (WORD8) i1;

    out_32 = AE_ZERO32();
  }
#if XCHAL_HAVE_HIFI4
  AE_MOVSARA7X2(sar_reg_up_half, sar_reg_low_half);
#endif
  return 0;
}
#endif


#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_notequal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  vbool2 b32, b10;
  vbool4 flag;
  
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);

  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 inp1_mul = AE_MOVDA32(inp1_multiplier);
  ae_int32x2 inp2_mul = AE_MOVDA32(inp2_multiplier);

  ae_int16x4 out = AE_ZERO16();
 
  ae_valign in1_a, in2_a, out_a;
  in1_a = AE_LA64_PP(p_in1);
  in2_a = AE_LA64_PP(p_in2);
  out_a = AE_ZALIGN64();

  for(i=0; i<(num_elm >> 2); i++)
  {
    AE_LA8X4S_IP(m1, in1_a, p_in1);
    AE_LA8X4S_IP(m2, in2_a, p_in2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    x32 = AE_SEXT32X2D16_32(x1);
    x10 = AE_SEXT32X2D16_10(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    y10 = AE_SEXT32X2D16_10(y1);

    x32 = AE_SLAA32S(x32, left_shift);
    x10 = AE_SLAA32S(x10, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    y10 = AE_SLAA32S(y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

    b32 = AE_EQ32(dequantized_x32, dequantized_y32);
    b10 = AE_EQ32(dequantized_x10, dequantized_y10);

    flag = vbool2_join_vbool4(b10, b32);
    AE_MOVF16X4(out, ONE_16X4, flag);
    AE_SA8X4U_IP(out, out_a, (ae_int32 *)p_o);
    
    out = AE_ZERO16();
  }
    //Remainder Loop
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    if(rem_length)
    {
        AE_LAV8X4S_XP(m1, in1_a, (ae_int8x4 *)p_in1, rem_length);
        AE_LAV8X4S_XP(m2, in2_a, (ae_int8x4 *)p_in2, rem_length);

        x1 = AE_ADD16(m1, inp1_z_b);
        y1 = AE_ADD16(m2, inp2_z_b);

        x32 = AE_SEXT32X2D16_32(x1);
        x10 = AE_SEXT32X2D16_10(x1);
        y32 = AE_SEXT32X2D16_32(y1);
        y10 = AE_SEXT32X2D16_10(y1);

        x32 = AE_SLAA32S(x32, left_shift);
        x10 = AE_SLAA32S(x10, left_shift);
        y32 = AE_SLAA32S(y32, left_shift);
        y10 = AE_SLAA32S(y10, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

        b32 = AE_EQ32(dequantized_x32, dequantized_y32);
        b10 = AE_EQ32(dequantized_x10, dequantized_y10);

        flag = vbool2_join_vbool4(b10, b32);
        AE_MOVF16X4(out, ONE_16X4, flag);
        AE_SAV8X4U_XP(out, out_a, (ae_int8x4u *)p_o, rem_length);
    }
    AE_SA64POS_FP(out_a, p_o);
#else
  AE_SA64POS_FP(out_a, p_o);

  for(i = 0; i < rem_length; i++)
  {
    AE_L8S_IP(m1, p_in1, 1);
    AE_L8S_IP(m2, p_in2, 1);
    
    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);
    
    x32 = AE_SEXT32X2D16_32(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    
    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    
    dequantized_x32 = AE_MULFP32X2RAS_L(x32, inp1_mul);
    dequantized_x32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_x32), inp1_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_x32), inp1_shift));
    
    dequantized_y32 = AE_MULFP32X2RAS_L(y32, inp2_mul);
    dequantized_y32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_y32), inp2_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_y32), inp2_shift));
    
    b32 = AE_EQ32(dequantized_x32, dequantized_y32);
    
    flag = vbool2_join_vbool4(b32, b32);
    AE_MOVF16X4(out, ONE_16X4, flag);
    
    AE_S8_0_IP_HIFI1(out, p_o, 1);
    
    out=AE_ZERO16();
  }
#endif
  return 0;
}
#else
WORD32 xa_nn_elm_notequal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  xtbool2 b32, b10;
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);
  ae_int32x2 ONE_32x2 = AE_MOVDA32(1);

  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 out_32 = AE_ZERO32();
  ae_int32x2 out_10 = AE_ZERO32();

#if XCHAL_HAVE_HIFI4
  int sar_reg_val = AE_MOVASAR();
  int sar_reg_low_half = sar_reg_val & 0x7F;
  sar_reg_val = sar_reg_val >> 7;
  int sar_reg_up_half = sar_reg_val & 0x7F;
  WUR_AE_SAR(left_shift);
#endif
  if(((((unsigned)p_in1)&3) == 0) && ((((unsigned)p_in2)&3) == 0))
  {
    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_L8X4F_IP(m1, p_in1, 4*sizeof(WORD8));
      AE_L8X4F_IP(m2, p_in2, 4*sizeof(WORD8));

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);

#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_EQ32(dequantized_x32, dequantized_y32);
      b10 = AE_EQ32(dequantized_x10, dequantized_y10);

      AE_MOVF32X2(out_32, ONE_32x2, b32);
      AE_MOVF32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  else
  {
    ALIGN_REGISTER_TYPE in1_a, in2_a;

    PRIME_8X4F(p_in1, in1_a);
    PRIME_8X4F(p_in2, in2_a);

    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_LA8X4F_IP(m1, in1_a, p_in1);
      AE_LA8X4F_IP(m2, in2_a, p_in2);

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);

#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_EQ32(dequantized_x32, dequantized_y32);
      b10 = AE_EQ32(dequantized_x10, dequantized_y10);

      AE_MOVF32X2(out_32, ONE_32x2, b32);
      AE_MOVF32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  //Remainder Loop
  for(i = 0; i < rem_length; i++)
  {
    WORD16 i1, i2;

    i1 = (WORD16) *(p_in1 + i);
    i2 = (WORD16) *(p_in2 + i);

    m1 = AE_MOVDA16(i1);
    m2 = AE_MOVDA16(i2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    AE_MUL16X4(x32, x10, x1, ONE_16X4);
    AE_MUL16X4(y32, y10, y1, ONE_16X4);

    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)

    b32 = AE_EQ32(dequantized_x32, dequantized_y32);

    AE_MOVF32X2(out_32, ONE_32x2, b32);

    i1 = (WORD16)(AE_MOVAD32_H(out_32));
    *p_o++ = (WORD8) i1;

    out_32 = AE_ZERO32();
  }
#if XCHAL_HAVE_HIFI4
  AE_MOVSARA7X2(sar_reg_up_half, sar_reg_low_half);
#endif
  return 0;
}
#endif

#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_greater_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  vbool2 b32, b10;
  vbool4 flag;
  
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);

  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 inp1_mul = AE_MOVDA32(inp1_multiplier);
  ae_int32x2 inp2_mul = AE_MOVDA32(inp2_multiplier);

  ae_int16x4 out = AE_ZERO16();

  ae_valign in1_a, in2_a, out_a;
  in1_a = AE_LA64_PP(p_in1);
  in2_a = AE_LA64_PP(p_in2);
  out_a = AE_ZALIGN64();

  for(i=0; i<(num_elm >> 2); i++)
  {
    AE_LA8X4S_IP(m1, in1_a, p_in1);
    AE_LA8X4S_IP(m2, in2_a, p_in2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    x32 = AE_SEXT32X2D16_32(x1);
    x10 = AE_SEXT32X2D16_10(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    y10 = AE_SEXT32X2D16_10(y1);

    x32 = AE_SLAA32S(x32, left_shift);
    x10 = AE_SLAA32S(x10, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    y10 = AE_SLAA32S(y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

    b32 = AE_LE32(dequantized_x32, dequantized_y32);
    b10 = AE_LE32(dequantized_x10, dequantized_y10);
 
    flag = vbool2_join_vbool4(b10, b32);
    AE_MOVF16X4(out, ONE_16X4, flag);
    AE_SA8X4U_IP(out, out_a, (ae_int32 *)p_o);
  
    out = AE_ZERO16();
  }
    //Remainder Loop
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    if(rem_length)
    {
        AE_LAV8X4S_XP(m1, in1_a, (ae_int8x4 *)p_in1, rem_length);
        AE_LAV8X4S_XP(m2, in2_a, (ae_int8x4 *)p_in2, rem_length);

        x1 = AE_ADD16(m1, inp1_z_b);
        y1 = AE_ADD16(m2, inp2_z_b);

        x32 = AE_SEXT32X2D16_32(x1);
        x10 = AE_SEXT32X2D16_10(x1);
        y32 = AE_SEXT32X2D16_32(y1);
        y10 = AE_SEXT32X2D16_10(y1);

        x32 = AE_SLAA32S(x32, left_shift);
        x10 = AE_SLAA32S(x10, left_shift);
        y32 = AE_SLAA32S(y32, left_shift);
        y10 = AE_SLAA32S(y10, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

        b32 = AE_LE32(dequantized_x32, dequantized_y32);
        b10 = AE_LE32(dequantized_x10, dequantized_y10);
 
        flag = vbool2_join_vbool4(b10, b32);
        AE_MOVF16X4(out, ONE_16X4, flag);
        AE_SAV8X4U_XP(out, out_a, (ae_int8x4u *)p_o, rem_length);
    }
    AE_SA64POS_FP(out_a, p_o);
#else

  AE_SA64POS_FP(out_a, p_o);

  for(i = 0; i < rem_length; i++)
  {
    AE_L8S_IP(m1, p_in1, 1);
    AE_L8S_IP(m2, p_in2, 1);
    
    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);
    
    x32 = AE_SEXT32X2D16_32(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    
    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    
    dequantized_x32 = AE_MULFP32X2RAS_L(x32, inp1_mul);
    dequantized_x32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_x32), inp1_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_x32), inp1_shift));
    
    dequantized_y32 = AE_MULFP32X2RAS_L(y32, inp2_mul);
    dequantized_y32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_y32), inp2_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_y32), inp2_shift));
    
    b32 = AE_LE32(dequantized_x32, dequantized_y32);
    
    flag = vbool2_join_vbool4(b32, b32);
    AE_MOVF16X4(out, ONE_16X4, flag);
    
    AE_S8_0_IP_HIFI1(out, p_o, 1);
    
    out=AE_ZERO16();
  }
#endif
  return 0;
}
#else
WORD32 xa_nn_elm_greater_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  xtbool2 b32, b10;
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);
  ae_int32x2 ONE_32x2 = AE_MOVDA32(1);

  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 out_32 = AE_ZERO32();
  ae_int32x2 out_10 = AE_ZERO32();

#if XCHAL_HAVE_HIFI4
  int sar_reg_val = AE_MOVASAR();
  int sar_reg_low_half = sar_reg_val & 0x7F;
  sar_reg_val = sar_reg_val >> 7;
  int sar_reg_up_half = sar_reg_val & 0x7F;
  WUR_AE_SAR(left_shift);
#endif
  if(((((unsigned)p_in1)&3) == 0) && ((((unsigned)p_in2)&3) == 0))
  {
    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_L8X4F_IP(m1, p_in1, 4*sizeof(WORD8));
      AE_L8X4F_IP(m2, p_in2, 4*sizeof(WORD8));

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);

#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_LE32(dequantized_x32, dequantized_y32);
      b10 = AE_LE32(dequantized_x10, dequantized_y10);

      AE_MOVF32X2(out_32, ONE_32x2, b32);
      AE_MOVF32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  else
  {
    ALIGN_REGISTER_TYPE in1_a, in2_a;

    PRIME_8X4F(p_in1, in1_a);
    PRIME_8X4F(p_in2, in2_a);

    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_LA8X4F_IP(m1, in1_a, p_in1);
      AE_LA8X4F_IP(m2, in2_a, p_in2);

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);

#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_LE32(dequantized_x32, dequantized_y32);
      b10 = AE_LE32(dequantized_x10, dequantized_y10);

      AE_MOVF32X2(out_32, ONE_32x2, b32);
      AE_MOVF32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  //Remainder Loop
  for(i = 0; i < rem_length; i++)
  {
    WORD16 i1, i2;

    i1 = (WORD16) *(p_in1 + i);
    i2 = (WORD16) *(p_in2 + i);

    m1 = AE_MOVDA16(i1);
    m2 = AE_MOVDA16(i2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    AE_MUL16X4(x32, x10, x1, ONE_16X4);
    AE_MUL16X4(y32, y10, y1, ONE_16X4);

    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)

    b32 = AE_LE32(dequantized_x32, dequantized_y32);

    AE_MOVF32X2(out_32, ONE_32x2, b32);

    i1 = (WORD16)(AE_MOVAD32_H(out_32));
    *p_o++ = (WORD8) i1;

    out_32 = AE_ZERO32();
  }
#if XCHAL_HAVE_HIFI4
  AE_MOVSARA7X2(sar_reg_up_half, sar_reg_low_half);
#endif
  return 0;
}
#endif


#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_greaterequal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  vbool2 b32, b10;
  vbool4 flag;
  
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);
  
  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 inp1_mul = AE_MOVDA32(inp1_multiplier);
  ae_int32x2 inp2_mul = AE_MOVDA32(inp2_multiplier);

  ae_int16x4 out = AE_ZERO16();

  ae_valign in1_a, in2_a, out_a;
  in1_a = AE_LA64_PP(p_in1);
  in2_a = AE_LA64_PP(p_in2);
  out_a = AE_ZALIGN64();

  for(i=0; i<(num_elm >> 2); i++)
  {
    AE_LA8X4S_IP(m1, in1_a, p_in1);
    AE_LA8X4S_IP(m2, in2_a, p_in2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    x32 = AE_SEXT32X2D16_32(x1);
    x10 = AE_SEXT32X2D16_10(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    y10 = AE_SEXT32X2D16_10(y1);

    x32 = AE_SLAA32S(x32, left_shift);
    x10 = AE_SLAA32S(x10, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    y10 = AE_SLAA32S(y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

    b32 = AE_LT32(dequantized_x32, dequantized_y32);
    b10 = AE_LT32(dequantized_x10, dequantized_y10);

    flag = vbool2_join_vbool4(b10, b32);
    AE_MOVF16X4(out, ONE_16X4, flag);
    AE_SA8X4U_IP(out, out_a, (ae_int32 *)p_o);
  
    out = AE_ZERO16();

  }
    //Remainder Loop
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    if(rem_length)
    {
        AE_LAV8X4S_XP(m1, in1_a, (ae_int8x4 *)p_in1, rem_length);
        AE_LAV8X4S_XP(m2, in2_a, (ae_int8x4 *)p_in2, rem_length);

        x1 = AE_ADD16(m1, inp1_z_b);
        y1 = AE_ADD16(m2, inp2_z_b);

        x32 = AE_SEXT32X2D16_32(x1);
        x10 = AE_SEXT32X2D16_10(x1);
        y32 = AE_SEXT32X2D16_32(y1);
        y10 = AE_SEXT32X2D16_10(y1);

        x32 = AE_SLAA32S(x32, left_shift);
        x10 = AE_SLAA32S(x10, left_shift);
        y32 = AE_SLAA32S(y32, left_shift);
        y10 = AE_SLAA32S(y10, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

        b32 = AE_LT32(dequantized_x32, dequantized_y32);
        b10 = AE_LT32(dequantized_x10, dequantized_y10);

        flag = vbool2_join_vbool4(b10, b32);
        AE_MOVF16X4(out, ONE_16X4, flag);
        AE_SAV8X4U_XP(out, out_a, (ae_int8x4u *)p_o, rem_length);
    }
    AE_SA64POS_FP(out_a, p_o);
#else
  AE_SA64POS_FP(out_a, p_o);

  for(i = 0; i < rem_length; i++)
  {
    AE_L8S_IP(m1, p_in1, 1);
    AE_L8S_IP(m2, p_in2, 1);
    
    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);
    
    x32 = AE_SEXT32X2D16_32(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    
    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    
    dequantized_x32 = AE_MULFP32X2RAS_L(x32, inp1_mul);
    dequantized_x32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_x32), inp1_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_x32), inp1_shift));
    
    dequantized_y32 = AE_MULFP32X2RAS_L(y32, inp2_mul);
    dequantized_y32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_y32), inp2_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_y32), inp2_shift));
    
    b32 = AE_LT32(dequantized_x32, dequantized_y32);
    
    flag = vbool2_join_vbool4(b32, b32);
    AE_MOVF16X4(out, ONE_16X4, flag);
    
    AE_S8_0_IP_HIFI1(out, p_o, 1);
    
    out = AE_ZERO16();
  }
#endif
  return 0;
}
#else
WORD32 xa_nn_elm_greaterequal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  xtbool2 b32, b10;
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);
  ae_int32x2 ONE_32x2 = AE_MOVDA32(1);

  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 out_32 = AE_ZERO32();
  ae_int32x2 out_10 = AE_ZERO32();

#if XCHAL_HAVE_HIFI4
  int sar_reg_val = AE_MOVASAR();
  int sar_reg_low_half = sar_reg_val & 0x7F;
  sar_reg_val = sar_reg_val >> 7;
  int sar_reg_up_half = sar_reg_val & 0x7F;
  WUR_AE_SAR(left_shift);
#endif
  if(((((unsigned)p_in1)&3) == 0) && ((((unsigned)p_in2)&3) == 0))
  {
    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_L8X4F_IP(m1, p_in1, 4*sizeof(WORD8));
      AE_L8X4F_IP(m2, p_in2, 4*sizeof(WORD8));

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);

#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_LT32(dequantized_x32, dequantized_y32);
      b10 = AE_LT32(dequantized_x10, dequantized_y10);

      AE_MOVF32X2(out_32, ONE_32x2, b32);
      AE_MOVF32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  else
  {
    ALIGN_REGISTER_TYPE in1_a, in2_a;

    PRIME_8X4F(p_in1, in1_a);
    PRIME_8X4F(p_in2, in2_a);

    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_LA8X4F_IP(m1, in1_a, p_in1);
      AE_LA8X4F_IP(m2, in2_a, p_in2);

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);

#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_LT32(dequantized_x32, dequantized_y32);
      b10 = AE_LT32(dequantized_x10, dequantized_y10);

      AE_MOVF32X2(out_32, ONE_32x2, b32);
      AE_MOVF32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  //Remainder Loop
  for(i = 0; i < rem_length; i++)
  {
    WORD16 i1, i2;

    i1 = (WORD16) *(p_in1 + i);
    i2 = (WORD16) *(p_in2 + i);

    m1 = AE_MOVDA16(i1);
    m2 = AE_MOVDA16(i2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    AE_MUL16X4(x32, x10, x1, ONE_16X4);
    AE_MUL16X4(y32, y10, y1, ONE_16X4);

    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)

    b32 = AE_LT32(dequantized_x32, dequantized_y32);

    AE_MOVF32X2(out_32, ONE_32x2, b32);

    i1 = (WORD16)(AE_MOVAD32_H(out_32));
    *p_o++ = (WORD8) i1;

    out_32 = AE_ZERO32();
  }
#if XCHAL_HAVE_HIFI4
  AE_MOVSARA7X2(sar_reg_up_half, sar_reg_low_half);
#endif
  return 0;
}
#endif


#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_less_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  vbool2 b32, b10;
  vbool4 flag;
  
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);
  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 inp1_mul = AE_MOVDA32(inp1_multiplier);
  ae_int32x2 inp2_mul = AE_MOVDA32(inp2_multiplier);

  ae_int16x4 out = AE_ZERO16();
  
  ae_valign in1_a, in2_a, out_a;
  in1_a = AE_LA64_PP(p_in1);
  in2_a = AE_LA64_PP(p_in2);
  out_a = AE_ZALIGN64();

  for(i=0; i<(num_elm >> 2); i++)
  {
    AE_LA8X4S_IP(m1, in1_a, p_in1);
    AE_LA8X4S_IP(m2, in2_a, p_in2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    x32 = AE_SEXT32X2D16_32(x1);
    x10 = AE_SEXT32X2D16_10(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    y10 = AE_SEXT32X2D16_10(y1);

    x32 = AE_SLAA32S(x32, left_shift);
    x10 = AE_SLAA32S(x10, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    y10 = AE_SLAA32S(y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

    b32 = AE_LT32(dequantized_x32, dequantized_y32);
    b10 = AE_LT32(dequantized_x10, dequantized_y10);

    flag = vbool2_join_vbool4(b10, b32);
    AE_MOVT16X4(out, ONE_16X4, flag);
    AE_SA8X4U_IP(out, out_a, (ae_int32 *)p_o);
  
    out = AE_ZERO16();
  }
    //Remainder Loop
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    if(rem_length)
    {
        AE_LAV8X4S_XP(m1, in1_a, (ae_int8x4 *)p_in1, rem_length);
        AE_LAV8X4S_XP(m2, in2_a, (ae_int8x4 *)p_in2, rem_length);

        x1 = AE_ADD16(m1, inp1_z_b);
        y1 = AE_ADD16(m2, inp2_z_b);

        x32 = AE_SEXT32X2D16_32(x1);
        x10 = AE_SEXT32X2D16_10(x1);
        y32 = AE_SEXT32X2D16_32(y1);
        y10 = AE_SEXT32X2D16_10(y1);

        x32 = AE_SLAA32S(x32, left_shift);
        x10 = AE_SLAA32S(x10, left_shift);
        y32 = AE_SLAA32S(y32, left_shift);
        y10 = AE_SLAA32S(y10, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

        b32 = AE_LT32(dequantized_x32, dequantized_y32);
        b10 = AE_LT32(dequantized_x10, dequantized_y10);

        flag = vbool2_join_vbool4(b10, b32);
        AE_MOVT16X4(out, ONE_16X4, flag);
        AE_SAV8X4U_XP(out, out_a, (ae_int8x4u *)p_o, rem_length);
    }
    AE_SA64POS_FP(out_a, p_o);
#else
  AE_SA64POS_FP(out_a, p_o);
  
  for(i = 0; i < rem_length; i++)
  {
    AE_L8S_IP(m1, p_in1, 1);
    AE_L8S_IP(m2, p_in2, 1);
    
    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);
    
    x32 = AE_SEXT32X2D16_32(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    
    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    
    dequantized_x32 = AE_MULFP32X2RAS_L(x32, inp1_mul);
    dequantized_x32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_x32), inp1_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_x32), inp1_shift));
    
    dequantized_y32 = AE_MULFP32X2RAS_L(y32, inp2_mul);
    dequantized_y32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_y32), inp2_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_y32), inp2_shift));
    
    b32 = AE_LT32(dequantized_x32, dequantized_y32);
    
    flag = vbool2_join_vbool4(b32, b32);
    AE_MOVT16X4(out, ONE_16X4, flag);
    
    AE_S8_0_IP_HIFI1(out, p_o, 1);
    
    out=AE_ZERO16();
  }
#endif
  return 0;
}
#else
WORD32 xa_nn_elm_less_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  xtbool2 b32, b10;
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);
  ae_int32x2 ONE_32x2 = AE_MOVDA32(1);

  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 out_32 = AE_ZERO32();
  ae_int32x2 out_10 = AE_ZERO32();

#if XCHAL_HAVE_HIFI4
  int sar_reg_val = AE_MOVASAR();
  int sar_reg_low_half = sar_reg_val & 0x7F;
  sar_reg_val = sar_reg_val >> 7;
  int sar_reg_up_half = sar_reg_val & 0x7F;
  WUR_AE_SAR(left_shift);
#endif
  if(((((unsigned)p_in1)&3) == 0) && ((((unsigned)p_in2)&3) == 0))
  {
    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_L8X4F_IP(m1, p_in1, 4*sizeof(WORD8));
      AE_L8X4F_IP(m2, p_in2, 4*sizeof(WORD8));

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);

#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_LT32(dequantized_x32, dequantized_y32);
      b10 = AE_LT32(dequantized_x10, dequantized_y10);

      AE_MOVT32X2(out_32, ONE_32x2, b32);
      AE_MOVT32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  else
  {
    ALIGN_REGISTER_TYPE in1_a, in2_a;

    PRIME_8X4F(p_in1, in1_a);
    PRIME_8X4F(p_in2, in2_a);

    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_LA8X4F_IP(m1, in1_a, p_in1);
      AE_LA8X4F_IP(m2, in2_a, p_in2);

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);

#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_LT32(dequantized_x32, dequantized_y32);
      b10 = AE_LT32(dequantized_x10, dequantized_y10);

      AE_MOVT32X2(out_32, ONE_32x2, b32);
      AE_MOVT32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  //Remainder Loop
  for(i = 0; i < rem_length; i++)
  {
    WORD16 i1, i2;

    i1 = (WORD16) *(p_in1 + i);
    i2 = (WORD16) *(p_in2 + i);

    m1 = AE_MOVDA16(i1);
    m2 = AE_MOVDA16(i2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    AE_MUL16X4(x32, x10, x1, ONE_16X4);
    AE_MUL16X4(y32, y10, y1, ONE_16X4);

    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)

    b32 = AE_LT32(dequantized_x32, dequantized_y32);

    AE_MOVT32X2(out_32, ONE_32x2, b32);

    i1 =(WORD16)( AE_MOVAD32_H(out_32));
    *p_o++ = (WORD8) i1;

    out_32 = AE_ZERO32();
  }
#if XCHAL_HAVE_HIFI4
  AE_MOVSARA7X2(sar_reg_up_half, sar_reg_low_half);
#endif
  return 0;
}
#endif

#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_lessequal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  vbool2 b32, b10;
  vbool4 flag;
  
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);
  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 inp1_mul = AE_MOVDA32(inp1_multiplier);
  ae_int32x2 inp2_mul = AE_MOVDA32(inp2_multiplier);

  ae_int16x4 out = AE_ZERO16();

  ae_valign in1_a, in2_a, out_a;
  in1_a = AE_LA64_PP(p_in1);
  in2_a = AE_LA64_PP(p_in2);
  out_a = AE_ZALIGN64();

  for(i=0; i<(num_elm >> 2); i++)
  {
    AE_LA8X4S_IP(m1, in1_a, p_in1);
    AE_LA8X4S_IP(m2, in2_a, p_in2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    x32 = AE_SEXT32X2D16_32(x1);
    x10 = AE_SEXT32X2D16_10(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    y10 = AE_SEXT32X2D16_10(y1);

    x32 = AE_SLAA32S(x32, left_shift);
    x10 = AE_SLAA32S(x10, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    y10 = AE_SLAA32S(y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

    b32 = AE_LE32(dequantized_x32, dequantized_y32);
    b10 = AE_LE32(dequantized_x10, dequantized_y10);

    flag = vbool2_join_vbool4(b10, b32);
    AE_MOVT16X4(out, ONE_16X4, flag);
    AE_SA8X4U_IP(out, out_a, (ae_int32 *)p_o);
  
    out = AE_ZERO16();
  }
    //Remainder Loop
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    if(rem_length)
    {
        AE_LAV8X4S_XP(m1, in1_a, (ae_int8x4 *)p_in1, rem_length);
        AE_LAV8X4S_XP(m2, in2_a, (ae_int8x4 *)p_in2, rem_length);

        x1 = AE_ADD16(m1, inp1_z_b);
        y1 = AE_ADD16(m2, inp2_z_b);

        x32 = AE_SEXT32X2D16_32(x1);
        x10 = AE_SEXT32X2D16_10(x1);
        y32 = AE_SEXT32X2D16_32(y1);
        y10 = AE_SEXT32X2D16_10(y1);

        x32 = AE_SLAA32S(x32, left_shift);
        x10 = AE_SLAA32S(x10, left_shift);
        y32 = AE_SLAA32S(y32, left_shift);
        y10 = AE_SLAA32S(y10, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_mul, inp1_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_mul, inp2_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_mul, inp2_shift)

        b32 = AE_LE32(dequantized_x32, dequantized_y32);
        b10 = AE_LE32(dequantized_x10, dequantized_y10);

        flag = vbool2_join_vbool4(b10, b32);
        AE_MOVT16X4(out, ONE_16X4, flag);
        AE_SAV8X4U_XP(out, out_a, (ae_int8x4u *)p_o, rem_length);
    }
    AE_SA64POS_FP(out_a, p_o);
#else
  AE_SA64POS_FP(out_a, p_o);

  for(i = 0; i < rem_length; i++)
  {
    AE_L8S_IP(m1, p_in1, 1);
    AE_L8S_IP(m2, p_in2, 1);
    
    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);
    
    x32 = AE_SEXT32X2D16_32(x1);
    y32 = AE_SEXT32X2D16_32(y1);
    
    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);
    
    dequantized_x32 = AE_MULFP32X2RAS_L(x32, inp1_mul);
    dequantized_x32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_x32), inp1_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_x32), inp1_shift));
    
    dequantized_y32 = AE_MULFP32X2RAS_L(y32, inp2_mul);
    dequantized_y32 = AE_ROUND32X2F64SSYM(AE_SLAA64S(AE_CVT64F32_H(dequantized_y32), inp2_shift),AE_SLAA64S(AE_CVT64F32_L(dequantized_y32), inp2_shift));
    
    b32 = AE_LE32(dequantized_x32, dequantized_y32);
    
    flag = vbool2_join_vbool4(b32, b32);
    AE_MOVT16X4(out, ONE_16X4, flag);
    
    AE_S8_0_IP_HIFI1(out, p_o, 1);
    
    out=AE_ZERO16();
  }
#endif
  return 0;
}
#else
WORD32 xa_nn_elm_lessequal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_int16x4 m1, m2, x1, y1;
  ae_int32x2 x32, x10, y32, y10, dequantized_x32, dequantized_x10, dequantized_y32, dequantized_y10;
  xtbool2 b32, b10;
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);
  ae_int32x2 ONE_32x2 = AE_MOVDA32(1);

  ae_int16x4 inp1_z_b = AE_MOVDA16(inp1_zero_bias);
  ae_int16x4 inp2_z_b = AE_MOVDA16(inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_int32x2 out_32 = AE_ZERO32();
  ae_int32x2 out_10 = AE_ZERO32();

#if XCHAL_HAVE_HIFI4
  int sar_reg_val = AE_MOVASAR();
  int sar_reg_low_half = sar_reg_val & 0x7F;
  sar_reg_val = sar_reg_val >> 7;
  int sar_reg_up_half = sar_reg_val & 0x7F;
  WUR_AE_SAR(left_shift);
#endif
  if(((((unsigned)p_in1)&3) == 0) && ((((unsigned)p_in2)&3) == 0))
  {
    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_L8X4F_IP(m1, p_in1, 4*sizeof(WORD8));
      AE_L8X4F_IP(m2, p_in2, 4*sizeof(WORD8));

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);
#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_LE32(dequantized_x32, dequantized_y32);
      b10 = AE_LE32(dequantized_x10, dequantized_y10);

      AE_MOVT32X2(out_32, ONE_32x2, b32);
      AE_MOVT32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  else
  {
    ALIGN_REGISTER_TYPE in1_a, in2_a;

    PRIME_8X4F(p_in1, in1_a);
    PRIME_8X4F(p_in2, in2_a);

    for(i=0; i<(num_elm >> 2); i++)
    {
      AE_LA8X4F_IP(m1, in1_a, p_in1);
      AE_LA8X4F_IP(m2, in2_a, p_in2);

      m1 = AE_SRAI16(m1, 8);
      m2 = AE_SRAI16(m2, 8);

      x1 = AE_ADD16(m1, inp1_z_b);
      y1 = AE_ADD16(m2, inp2_z_b);

      AE_MUL16X4(x32, x10, x1, ONE_16X4);
      AE_MUL16X4(y32, y10, y1, ONE_16X4);
#if XCHAL_HAVE_HIFI4
      x32 = AE_SLAS32S(x32);
      x10 = AE_SLAS32S(x10);
      y32 = AE_SLAS32S(y32);
      y10 = AE_SLAS32S(y10);
#else
      x32 = AE_SLAA32S(x32, left_shift);
      x10 = AE_SLAA32S(x10, left_shift);
      y32 = AE_SLAA32S(y32, left_shift);
      y10 = AE_SLAA32S(y10, left_shift);
#endif
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x10, x10, inp1_multiplier, inp1_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y10, y10, inp2_multiplier, inp2_shift)

      b32 = AE_LE32(dequantized_x32, dequantized_y32);
      b10 = AE_LE32(dequantized_x10, dequantized_y10);

      AE_MOVT32X2(out_32, ONE_32x2, b32);
      AE_MOVT32X2(out_10, ONE_32x2, b10);

      STORE_8X4_FROM_32X4(p_o, out_32, out_10)

      out_32 = AE_ZERO32();
      out_10 = AE_ZERO32();
    }
  }

  //Remainder Loop
  for(i = 0; i < rem_length; i++)
  {
    WORD16 i1, i2;

    i1 = (WORD16) *(p_in1 + i);
    i2 = (WORD16) *(p_in2 + i);

    m1 = AE_MOVDA16(i1);
    m2 = AE_MOVDA16(i2);

    x1 = AE_ADD16(m1, inp1_z_b);
    y1 = AE_ADD16(m2, inp2_z_b);

    AE_MUL16X4(x32, x10, x1, ONE_16X4);
    AE_MUL16X4(y32, y10, y1, ONE_16X4);

    x32 = AE_SLAA32S(x32, left_shift);
    y32 = AE_SLAA32S(y32, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_x32, x32, inp1_multiplier, inp1_shift)
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(dequantized_y32, y32, inp2_multiplier, inp2_shift)

    b32 = AE_LE32(dequantized_x32, dequantized_y32);

    AE_MOVT32X2(out_32, ONE_32x2, b32);

    i1 = (WORD16)(AE_MOVAD32_H(out_32));
    *p_o++ = (WORD8) i1;

    out_32 = AE_ZERO32();
  }
#if XCHAL_HAVE_HIFI4
  AE_MOVSARA7X2(sar_reg_up_half, sar_reg_low_half);
#endif
  return 0;
}
#endif
