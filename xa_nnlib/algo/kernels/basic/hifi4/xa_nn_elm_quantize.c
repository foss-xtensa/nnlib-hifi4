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
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"
#include "xa_nnlib_common_macros.h"
#include <math.h>

WORD32 xa_nn_elm_requantize_asym16s_asym8s(WORD8 * __restrict__ p_out,
                                    const WORD16 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  WORD8 *out = p_out;
  WORD16 *p_i = (WORD16 *)p_inp;

  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_valign align_inp = AE_LA64_PP(p_inp);

#if XCHAL_HAVE_HIFI1
  ae_valign align_out = AE_ZALIGN64();
#endif
  
  ae_int32x2 inp_z_b = AE_MOVDA32(inp_zero_bias);
  ae_int32x2 out_mult = AE_MOVDA32(out_multiplier);
  ae_int32x2 quant_min = AE_MOVDA32(-128);
  ae_int32x2 quant_max = AE_MOVDA32(127);
  
  for(i = 0; i < (num_elm >> 2); i++)
  {
    ae_int16x4 inp0;
    ae_int32x2 inp32, inp10;
    ae_int32x2 unclamped_out32, unclamped_out10;
#if !(XCHAL_HAVE_HIFI1 & (XCHAL_HW_VERSION >= RI9_HWVERSION))
    ae_int32x2 clamped_out32, clamped_out10;
#endif
    AE_LA16X4_IP(inp0, align_inp, (ae_int16x4 *)p_i);

    inp32 = AE_SEXT32X2D16_32(inp0);
    inp10 = AE_SEXT32X2D16_10(inp0);

    inp32 = AE_SUB32S(inp32, inp_z_b);
    inp10 = AE_SUB32S(inp10, inp_z_b);

    // unclamped result
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out32, inp32, out_mult, left_shift, right_shift)
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out10, inp10, out_mult, left_shift, right_shift)
    unclamped_out32 = AE_ADD32(unclamped_out32, out_zero_bias);
    unclamped_out10 = AE_ADD32(unclamped_out10, out_zero_bias);

#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    // clamped_out
    ae_int8x8 clamped = AE_SAT8X4X32_H(unclamped_out32, unclamped_out10);
    // Store Output
    AE_SAV8X8_XP(clamped, align_out, (ae_int8x8 *)out, 4);
#else
    // clamped_out
    CLAMP_VAL(clamped_out32, unclamped_out32, quant_min, quant_max)
    CLAMP_VAL(clamped_out10, unclamped_out10, quant_min, quant_max)
    ae_int16x4 temp = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(clamped_out32),AE_MOVINT16X4_FROMINT32X2(clamped_out10));
    AE_SA8X4U_IP(temp, align_out, (ae_int32 *)out);
#endif
#else
    // clamped_out
    CLAMP_VAL(clamped_out32, unclamped_out32, quant_min, quant_max)
    CLAMP_VAL(clamped_out10, unclamped_out10, quant_min, quant_max)
    // Store Output
    STORE_8X4_FROM_32X4(out, clamped_out32, clamped_out10)
#endif
  }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, out);
#endif

  // Remainder Loop
  for(i = 0; i < (num_elm & 3); i++)
  {
    int inp;
    ae_int32x2 inp_HL;
    ae_int32x2 unclamped_out_HL;
    ae_int32x2 clamped_out_HL;

    inp = (int)p_i[i];
    inp_HL = AE_MOVDA32(inp);
    inp_HL = AE_SUB32S(inp_HL, inp_z_b);

    MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out_HL, inp_HL, out_mult, left_shift, right_shift)
    unclamped_out_HL = AE_ADD32(unclamped_out_HL, out_zero_bias);
    
    // clamped_out
    CLAMP_VAL(clamped_out_HL, unclamped_out_HL, quant_min, quant_max)

    *out++ = (WORD8)(AE_MOVAD32_H(clamped_out_HL));
  }
  return 0;  
}


WORD32 xa_nn_elm_requantize_asym16s_asym16s(WORD16 * __restrict__ p_out,
                                    const WORD16 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -32768) || (out_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  ae_int16x4 * __restrict__  p_o = (ae_int16x4 *)p_out;
  const ae_int16x4 * __restrict__  p_i = (const ae_int16x4 *)p_inp;

  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_int32x2 d_inp_zero_bias = AE_MOVDA32(inp_zero_bias);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);
  
  ae_int16x4 out_0/*, out_1*/;

  ae_valign align_inp = AE_LA64_PP(p_inp);
  ae_valign align_dst = AE_ZALIGN64();

  for(i = 0; i < num_elm>>2 ; i++)
  {
    ae_int16x4 d_inp0;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_LA16X4_IP(d_inp0, align_inp, p_i);
    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp0, ONE_16X4);
    d_inp32_0 = AE_SUB32(d_inp32_0, d_inp_zero_bias);
    d_inp32_1 = AE_SUB32(d_inp32_1, d_inp_zero_bias);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_1, d_inp32_1, d_out_multiplier, left_shift, right_shift);
    d_inp32_0 = AE_ADD32S(d_inp32_0, AE_MOVDA32(out_zero_bias));
    d_inp32_1 = AE_ADD32S(d_inp32_1, AE_MOVDA32(out_zero_bias));
    out_0 = AE_SAT16X4(d_inp32_0, d_inp32_1);
    
    AE_SA16X4_IP(out_0, align_dst, p_o);
  }
#if (XCHAL_HAVE_HIFI1 &( XCHAL_HW_VERSION >= RI9_HWVERSION ))
  {
    ae_int16x4 d_inp0;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_LAV16X4_XP(d_inp0, align_inp, (ae_int16x4 *)p_i, ((num_elm & 0x3)<<1));
    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp0, ONE_16X4);
    d_inp32_0 = AE_SUB32(d_inp32_0, d_inp_zero_bias);
    d_inp32_1 = AE_SUB32(d_inp32_1, d_inp_zero_bias);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_1, d_inp32_1, d_out_multiplier, left_shift, right_shift);
    d_inp32_0 = AE_ADD32S(d_inp32_0, AE_MOVDA32(out_zero_bias));
    d_inp32_1 = AE_ADD32S(d_inp32_1, AE_MOVDA32(out_zero_bias));
    out_0 = AE_SAT16X4(d_inp32_0, d_inp32_1);
    
    AE_SAV16X4_XP(out_0, align_dst, (ae_int16x4 *)p_o, ((num_elm & 0x3)<<1));
  }
  AE_SA64POS_FP(align_dst, p_o);
#else
  AE_SA64POS_FP(align_dst, p_o);

  /*Remainder */
  for(i = 0; i < (num_elm & 0x3); i++)
  {
    ae_int16x4 d_inp0;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_L16_IP(d_inp0, (ae_int16 *)p_i, 2);
    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp0, ONE_16X4);
    d_inp32_0 = AE_SUB32(d_inp32_0, d_inp_zero_bias);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    d_inp32_0 = AE_ADD32S(d_inp32_0, AE_MOVDA32(out_zero_bias));
    out_0 = AE_SAT16X4(d_inp32_0, d_inp32_0);
    
    AE_S16_0_IP(out_0, (ae_int16 *)p_o, 2);
  }
#endif
  return 0;
}



/* derived from HiFi5 implementation and modified for HiFi4 */
WORD32 xa_nn_elm_requantize_asym16s_asym32s(WORD32 * __restrict__ p_out,
                                           const WORD16 * __restrict__ p_inp,
                                           WORD32  inp_zero_bias,
                                           WORD32  out_zero_bias,
                                           WORD32  out_shift,
                                           WORD32  out_multiplier,
                                           WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  //XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  WORD32 *p_o = p_out;
  WORD16 *p_i = (WORD16 *)p_inp;

  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_valign align_inp = AE_LA64_PP(p_inp);
  ae_valign align_dst = AE_ZALIGN64();

#if !XCHAL_HAVE_HIFI1
  ae_int16x4 ONE = AE_MOVDA16(1);
#endif
  ae_int32x2 d_inp_zero_bias  = AE_MOVDA32((WORD16)inp_zero_bias);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);
  ae_int32x2 d_out_zero_bias  = AE_MOVDA32(out_zero_bias);

  ae_int16x4 d_inp0;
  ae_int32x2 d_inp32_0, d_inp32_1;
  ae_int32x2 d_out0, d_out1;

  for(i = 0; i < (num_elm >> 2); i++)
  {
    AE_LA16X4_IP(d_inp0, align_inp, (ae_int16x4 *)p_i);

#if XCHAL_HAVE_HIFI1
    d_inp32_0 = AE_SEXT32X2D16_32(d_inp0);
    d_inp32_1 = AE_SEXT32X2D16_10(d_inp0);
#else
    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp0, ONE);
#endif

    d_inp32_0 = AE_SUB32S(d_inp32_0, d_inp_zero_bias);
    d_inp32_1 = AE_SUB32S(d_inp32_1, d_inp_zero_bias);

    MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_inp32_1, d_inp32_1, d_out_multiplier, left_shift, right_shift);

    d_out0 = AE_ADD32S(d_inp32_0, d_out_zero_bias);
    d_out1 = AE_ADD32S(d_inp32_1, d_out_zero_bias);

    AE_SA32X2_IP(d_out0, align_dst, (ae_int32x2*)p_o);
    AE_SA32X2_IP(d_out1, align_dst, (ae_int32x2*)p_o);
  }
  AE_SA64POS_FP(align_dst, p_o);

  /*Remainder loop*/
  for(i = 0; i < (num_elm & 3); i++)
  {
    AE_L16_IP(d_inp0, (ae_int16 *)p_i, 2);

#if XCHAL_HAVE_HIFI1
    d_inp32_0 = AE_SEXT32X2D16_32(d_inp0);
#else
    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp0, ONE);
#endif
    d_inp32_0 = AE_SUB32S(d_inp32_0, d_inp_zero_bias);

    MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    d_out0 = AE_ADD32S(d_inp32_0, d_out_zero_bias);
    AE_S32_L_IP(d_out0, (ae_int32*)p_o, 4);
  }
  return 0;
}

/* derived from HiFi5 implementation and modified for HiFi4 */
WORD32 xa_nn_elm_requantize_asym8s_asym32s(WORD32 * __restrict__ p_out,
                                           const WORD8 * __restrict__ p_inp,
                                           WORD32  inp_zero_bias,
                                           WORD32  out_zero_bias,
                                           WORD32  out_shift,
                                           WORD32  out_multiplier,
                                           WORD32  num_elm)
{
/* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  if(out_shift < -31)
  {
    out_multiplier = 0;
    out_shift = 0;
  }

  int i;
  ae_int32x2 *p_o = (ae_int32x2 *)p_out;
  WORD8 *p_i = (WORD8 *)p_inp;

  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

#if XCHAL_HAVE_HIFI1
  ae_valign align_inp;
  align_inp = AE_LA64_PP(p_i);
#else
  ALIGN_REGISTER_TYPE align_inp;
  PRIME_8X4F(p_i, align_inp);
#endif

  ae_valign align_dst = AE_ZALIGN64();

  ae_int16x4 d_inp_zero_bias = AE_MOVDA16(inp_zero_bias);

#if !XCHAL_HAVE_HIFI1
  ae_int16x4 ONE = AE_MOVDA16(1);
#endif

  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);
  ae_int32x2 d_out_zero_bias = AE_MOVDA32(out_zero_bias);

  ae_int32x2 d_out0, d_out1, d_out2, d_out3;

  for(i = 0; i < (num_elm >> 3); i++)
  {
    ae_int16x4 d_inp0, d_inp1;
    ae_int32x2 d_inp32_0, d_inp32_1, d_inp32_2, d_inp32_3;

#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(d_inp0, align_inp, p_i);
    AE_LA8X4S_IP(d_inp1, align_inp, p_i);
#else
    //AE_LA8X8X2_IP(d_inp0, d_inp1, align_inp, p_i);
    AE_LA8X4F_IP(d_inp0, align_inp, p_i);
    AE_LA8X4F_IP(d_inp1, align_inp, p_i);
    d_inp0 = AE_SRAI16(d_inp0, 8);
    d_inp1 = AE_SRAI16(d_inp1, 8);
#endif

    d_inp0 = AE_SUB16(d_inp0, d_inp_zero_bias);
    d_inp1 = AE_SUB16(d_inp1, d_inp_zero_bias);

#if XCHAL_HAVE_HIFI1
    d_inp32_0 = AE_SEXT32X2D16_32(d_inp0);
    d_inp32_1 = AE_SEXT32X2D16_10(d_inp0);
    d_inp32_2 = AE_SEXT32X2D16_32(d_inp1);
    d_inp32_3 = AE_SEXT32X2D16_10(d_inp1);
#else
    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp0, ONE);
    AE_MUL16X4(d_inp32_2, d_inp32_3, d_inp1, ONE);
#endif

    MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_inp32_1, d_inp32_1, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_inp32_2, d_inp32_2, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_inp32_3, d_inp32_3, d_out_multiplier, left_shift, right_shift);

    d_out0 = AE_ADD32S(d_inp32_0, d_out_zero_bias);
    d_out1 = AE_ADD32S(d_inp32_1, d_out_zero_bias);
    d_out2 = AE_ADD32S(d_inp32_2, d_out_zero_bias);
    d_out3 = AE_ADD32S(d_inp32_3, d_out_zero_bias);

    AE_SA32X2_IP(d_out0, align_dst, p_o);
    AE_SA32X2_IP(d_out1, align_dst, p_o);
    AE_SA32X2_IP(d_out2, align_dst, p_o);
    AE_SA32X2_IP(d_out3, align_dst, p_o);
  }

  AE_SA64POS_FP(align_dst, p_o);

  /*Remainder loop*/
#pragma no_unroll
  for(i = 0; i < (num_elm & 7); i++)
  {
    ae_int16x4 d_inp0 = (WORD16)*p_i;
    ae_int32x2 d_inp32_0 = 0;

    d_inp0 = AE_SUB16(d_inp0, d_inp_zero_bias);

#if XCHAL_HAVE_HIFI1
    d_inp32_0 = AE_SEXT32X2D16_32(d_inp0);
#else
    AE_MUL16X4(d_inp32_0, d_inp32_0, d_inp0, ONE); 
#endif

    MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);

    d_out0 = AE_ADD32S(d_inp32_0, d_out_zero_bias);
    AE_S32_L_IP(d_out0, (ae_int32*)p_o, 4);

    p_i++;
  }
  return 0;
}


WORD32 xa_nn_elm_requantize_asym8s_asym8s(WORD8 * __restrict__ p_out,
                                    const WORD8 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  WORD8 *p_i = (WORD8 *)p_inp;
  WORD8 *p_o = (WORD8 *)p_out;

  ae_int16x4 d_inp_zero_bias  = AE_MOVDA16(inp_zero_bias);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);
  ae_int16x4 d_inp0, d_inp1;
  ae_int16x4 ONE = AE_MOVDA16(1);
  ae_int32x2 d_inp32_0, d_inp32_1;
  ae_int32x2 d_inp32_2, d_inp32_3;
  ae_int32x2 d_out0_32, d_out1_32;
  ae_int32x2 d_out2_32, d_out3_32;
  ae_int32x2 quant_min = AE_MOVDA32(-128);
  ae_int32x2 quant_max = AE_MOVDA32(127);

  xtbool io_pointers_aligned = ((uintptr_t)p_inp%4 == 0);
#if XCHAL_HAVE_HIFI1
  ALIGN_REGISTER_TYPE align_out = AE_ZALIGN64();
#endif
  if(io_pointers_aligned){
    for(i = 0; i < num_elm >> 3; i++)
    {
#if XCHAL_HAVE_HIFI1
      AE_L8X4S_IP(d_inp0, p_i, 4);
      AE_L8X4S_IP(d_inp1, p_i, 4);
#else
      AE_L8X4F_IP(d_inp0, p_i, 4);
      AE_L8X4F_IP(d_inp1, p_i, 4);
      d_inp0 = AE_SRAI16(d_inp0, 8);
      d_inp1 = AE_SRAI16(d_inp1, 8);
#endif
      d_inp0 = AE_SUB16S(d_inp0, d_inp_zero_bias);
      d_inp1 = AE_SUB16S(d_inp1, d_inp_zero_bias);
      AE_MUL16X4(d_inp32_0 , d_inp32_1 , d_inp0 , ONE);
      AE_MUL16X4(d_inp32_2 , d_inp32_3 , d_inp1 , ONE);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out0_32, d_inp32_0, d_out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out1_32, d_inp32_1, d_out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out2_32, d_inp32_2, d_out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out3_32, d_inp32_3, d_out_multiplier, left_shift, right_shift);
      d_out0_32 = AE_ADD32S(d_out0_32, AE_MOVDA32(out_zero_bias));
      d_out1_32 = AE_ADD32S(d_out1_32, AE_MOVDA32(out_zero_bias));
      d_out2_32 = AE_ADD32S(d_out2_32, AE_MOVDA32(out_zero_bias));
      d_out3_32 = AE_ADD32S(d_out3_32, AE_MOVDA32(out_zero_bias));
#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
      // clamped_out
      ae_int8x8 clamped_01 = AE_SAT8X4X32_H(d_out0_32, d_out1_32);
      ae_int8x8 clamped_23 = AE_SAT8X4X32_H(d_out2_32, d_out3_32);
      // Store Output
      AE_SAV8X8_XP(clamped_01, align_out, (ae_int8x8 *)p_o, 4);
      AE_SAV8X8_XP(clamped_23, align_out, (ae_int8x8 *)p_o, 4);
#else
      CLAMP_VAL(d_out0_32, d_out0_32, quant_min, quant_max)
      CLAMP_VAL(d_out1_32, d_out1_32, quant_min, quant_max)
      CLAMP_VAL(d_out2_32, d_out2_32, quant_min, quant_max)
      CLAMP_VAL(d_out3_32, d_out3_32, quant_min, quant_max)
      ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d_out0_32), AE_MOVF16X4_FROMF32X2(d_out1_32));
      AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_o);
      out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d_out2_32), AE_MOVF16X4_FROMF32X2(d_out3_32));
      AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_o);
#endif
#else
      CLAMP_VAL(d_out0_32, d_out0_32, quant_min, quant_max)
      CLAMP_VAL(d_out1_32, d_out1_32, quant_min, quant_max)
      CLAMP_VAL(d_out2_32, d_out2_32, quant_min, quant_max)
      CLAMP_VAL(d_out3_32, d_out3_32, quant_min, quant_max)
      STORE_8X4_FROM_32X4(p_o, d_out0_32, d_out1_32);
      STORE_8X4_FROM_32X4(p_o, d_out2_32, d_out3_32);
#endif
    }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_o);
#endif
  }
  else{
    ALIGN_REGISTER_TYPE align_inp;
    PRIME_8X4F(p_i, align_inp);
    for(i = 0; i < num_elm >> 3; i++)
    {
#if XCHAL_HAVE_HIFI1
      AE_LA8X4S_IP(d_inp0, align_inp, p_i);
      AE_LA8X4S_IP(d_inp1, align_inp, p_i);
#else
      AE_LA8X4F_IP(d_inp0, align_inp, p_i);
      AE_LA8X4F_IP(d_inp1, align_inp, p_i);
      d_inp0 = AE_SRAI16(d_inp0, 8);
      d_inp1 = AE_SRAI16(d_inp1, 8);
#endif
      d_inp0 = AE_SUB16S(d_inp0, d_inp_zero_bias);
      d_inp1 = AE_SUB16S(d_inp1, d_inp_zero_bias);
      AE_MUL16X4(d_inp32_0 , d_inp32_1 , d_inp0 , ONE);
      AE_MUL16X4(d_inp32_2 , d_inp32_3 , d_inp1 , ONE);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out0_32, d_inp32_0, d_out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out1_32, d_inp32_1, d_out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out2_32, d_inp32_2, d_out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out3_32, d_inp32_3, d_out_multiplier, left_shift, right_shift);
      d_out0_32 = AE_ADD32S(d_out0_32, AE_MOVDA32(out_zero_bias));
      d_out1_32 = AE_ADD32S(d_out1_32, AE_MOVDA32(out_zero_bias));
      d_out2_32 = AE_ADD32S(d_out2_32, AE_MOVDA32(out_zero_bias));
      d_out3_32 = AE_ADD32S(d_out3_32, AE_MOVDA32(out_zero_bias));

#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
      // clamped_out
      ae_int8x8 clamped_01 = AE_SAT8X4X32_H(d_out0_32, d_out1_32);
      ae_int8x8 clamped_23 = AE_SAT8X4X32_H(d_out2_32, d_out3_32);
      // Store Output
      AE_SAV8X8_XP(clamped_01, align_out, (ae_int8x8 *)p_o, 4);
      AE_SAV8X8_XP(clamped_23, align_out, (ae_int8x8 *)p_o, 4);
#else
      CLAMP_VAL(d_out0_32, d_out0_32, quant_min, quant_max)
      CLAMP_VAL(d_out1_32, d_out1_32, quant_min, quant_max)
      CLAMP_VAL(d_out2_32, d_out2_32, quant_min, quant_max)
      CLAMP_VAL(d_out3_32, d_out3_32, quant_min, quant_max)
      ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d_out0_32), AE_MOVF16X4_FROMF32X2(d_out1_32));
      AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_o);
      out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d_out2_32), AE_MOVF16X4_FROMF32X2(d_out3_32));
      AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_o);
#endif
#else
      CLAMP_VAL(d_out0_32, d_out0_32, quant_min, quant_max)
      CLAMP_VAL(d_out1_32, d_out1_32, quant_min, quant_max)
      CLAMP_VAL(d_out2_32, d_out2_32, quant_min, quant_max)
      CLAMP_VAL(d_out3_32, d_out3_32, quant_min, quant_max)
      STORE_8X4_FROM_32X4(p_o, d_out0_32, d_out1_32);
      STORE_8X4_FROM_32X4(p_o, d_out2_32, d_out3_32);
#endif
    }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_o);
#endif
  }
  for(i = 0; i < num_elm % 8 ; i++)
  {
    ae_int32 tmp = (p_i[i] - inp_zero_bias);
    ae_int32x2 res = tmp;
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(res, res, d_out_multiplier, left_shift, right_shift);
    res = AE_ADD32S(res, AE_MOVDA32(out_zero_bias));
    CLAMP_VAL(res, res, quant_min, quant_max)
    *p_o = (WORD8)AE_MOVAD32_L(res);
    p_o++;
  }
  return 0;
}

WORD32 xa_nn_elm_requantize_asym8u_asym8s(WORD8 * __restrict__ p_out,
                                    const UWORD8 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < 0) || (inp_zero_bias > 255)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  WORD8 *p_i = (WORD8 *)p_inp;
  WORD8 *p_o = (WORD8 *)p_out;

  ae_int16x4 ONE = AE_MOVDA16(1);
  ae_int16x4 d_inp_zero_bias  = AE_MOVDA16(inp_zero_bias);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);
  ae_int16x4 d_inp0;
  ae_int32x2 d_inp32_0, d_inp32_1;
  ae_int32x2 d_out0_32, d_out1_32;
  ae_int32x2 quant_min = AE_MOVDA32(-128);
  ae_int32x2 quant_max = AE_MOVDA32(127);

  xtbool io_pointers_aligned = ((((uintptr_t)p_inp) & 3) == 0);
#if XCHAL_HAVE_HIFI1
  ALIGN_REGISTER_TYPE align_out = AE_ZALIGN64();
#endif
  if(io_pointers_aligned){
    for(i = 0; i < num_elm >> 2; i++)
    {
#if XCHAL_HAVE_HIFI1
      AE_L8X4U_IP(d_inp0, p_i, 4);
#else
      AE_L8X4F_IP(d_inp0, p_i, 4);
      d_inp0 = AE_SRAI16(d_inp0, 8);
      d_inp0 = AE_AND16(d_inp0, AE_MOVDA16(0x00ff));
#endif
      d_inp0 = AE_SUB16S(d_inp0, d_inp_zero_bias);

      AE_MUL16X4(d_inp32_0 , d_inp32_1 , d_inp0 , ONE);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out0_32, d_inp32_0, d_out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out1_32, d_inp32_1, d_out_multiplier, left_shift, right_shift);
      d_out0_32 = AE_ADD32S(d_out0_32, AE_MOVDA32(out_zero_bias));
      d_out1_32 = AE_ADD32S(d_out1_32, AE_MOVDA32(out_zero_bias));
#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
      // clamped_out
      ae_int8x8 clamped_01 = AE_SAT8X4X32_H(d_out0_32, d_out1_32);
      // Store Output
      AE_SAV8X8_XP(clamped_01, align_out, (ae_int8x8 *)p_o, 4);
#else
      CLAMP_VAL(d_out0_32, d_out0_32, quant_min, quant_max)
      CLAMP_VAL(d_out1_32, d_out1_32, quant_min, quant_max)
      ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d_out0_32), AE_MOVF16X4_FROMF32X2(d_out1_32));
      AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_o);
#endif
#else
      CLAMP_VAL(d_out0_32, d_out0_32, quant_min, quant_max)
      CLAMP_VAL(d_out1_32, d_out1_32, quant_min, quant_max)
      STORE_8X4_FROM_32X4(p_o, d_out0_32, d_out1_32);
#endif
    }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_o);
#endif
  }
  else{
    ALIGN_REGISTER_TYPE align_inp;
    PRIME_8X4F(p_i, align_inp);
    for(i = 0; i < num_elm >> 2; i++)
    {
#if XCHAL_HAVE_HIFI1
      AE_LA8X4U_IP(d_inp0, align_inp, p_i);
#else
      AE_LA8X4F_IP(d_inp0, align_inp, p_i);
      d_inp0 = AE_SRAI16(d_inp0, 8);
      d_inp0 = AE_AND16(d_inp0, AE_MOVDA16(0x00ff));
#endif
      d_inp0 = AE_SUB16S(d_inp0, d_inp_zero_bias);

      AE_MUL16X4(d_inp32_0 , d_inp32_1 , d_inp0 , ONE);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out0_32, d_inp32_0, d_out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out1_32, d_inp32_1, d_out_multiplier, left_shift, right_shift);
      d_out0_32 = AE_ADD32S(d_out0_32, AE_MOVDA32(out_zero_bias));
      d_out1_32 = AE_ADD32S(d_out1_32, AE_MOVDA32(out_zero_bias));
#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
      // clamped_out
      ae_int8x8 clamped_01 = AE_SAT8X4X32_H(d_out0_32, d_out1_32);
      // Store Output
      AE_SAV8X8_XP(clamped_01, align_out, (ae_int8x8 *)p_o, 4);
#else
      CLAMP_VAL(d_out0_32, d_out0_32, quant_min, quant_max)
      CLAMP_VAL(d_out1_32, d_out1_32, quant_min, quant_max)
      ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d_out0_32), AE_MOVF16X4_FROMF32X2(d_out1_32));
      AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_o);
#endif
#else
      CLAMP_VAL(d_out0_32, d_out0_32, quant_min, quant_max)
      CLAMP_VAL(d_out1_32, d_out1_32, quant_min, quant_max)
      STORE_8X4_FROM_32X4(p_o, d_out0_32, d_out1_32);
#endif
    }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_o);
#endif
  }
#pragma loop_count max=3
  for(i = 0; i < (num_elm & 3) ; i++)
  {
    ae_int16x4 d_inp0 = AE_MOVDA16((WORD32)((UWORD8 *)p_i)[i] - inp_zero_bias);

    d_inp32_0 = AE_SEXT32X2D16_10(d_inp0);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(d_out0_32, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    d_out0_32 = AE_ADD32S(d_out0_32, AE_MOVDA32(out_zero_bias));

    CLAMP_VAL(d_out0_32, d_out0_32, quant_min, quant_max)
    *p_o = (WORD8)AE_MOVAD32_L(d_out0_32);
    p_o++;
  }
  return 0;
}

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_elm_dequantize_asym8s_f32,
                               (FLOAT32 * __restrict__ p_out,
                               const WORD8 * __restrict__ p_inp,
                               WORD32  inp_zero_bias,
                               FLOAT32  inp_scale,
                               WORD32  num_elm))
#elif XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_dequantize_asym8s_f32(FLOAT32 * __restrict__ p_out,
                                       const WORD8 * __restrict__ p_inp,
                                       WORD32  inp_zero_bias,
                                       FLOAT32  inp_scale,
                                       WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);

  int i;
  xtfloatx2 *p_o = (xtfloatx2 *)p_out;
  WORD8 *p_i = (WORD8 *)p_inp;

  ae_valign align_inp = AE_LA64_PP(p_inp);
  ae_valign align_dst = AE_ZALIGN64();

  ae_int16x4 d_inp_zero_bias = AE_MOVDA16(inp_zero_bias);
  ae_int16x4 ONE = AE_MOVDA16(1);
  xtfloat *inp_scale_ptr = &inp_scale;
  xtfloat d_inp_scale;
  AE_LSIP(d_inp_scale, inp_scale_ptr, sizeof(FLOAT32));;
  xtfloatx2 d_out0, d_out1;

  for(i = 0; i < (num_elm >> 2); i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_inp16_0;
    ae_int32x2 d_inp32_0, d_inp32_1;

    AE_LA8X4S_IP(d_inp0, align_inp, p_i);
    d_inp16_0 =  AE_SUB16(d_inp0, d_inp_zero_bias);

    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp16_0, ONE);

    d_out0 = MUL_SX2(d_inp32_0, d_inp_scale);
    d_out1 = MUL_SX2(d_inp32_1, d_inp_scale);

    AE_SASX2IP(d_out0, align_dst, p_o);
    AE_SASX2IP(d_out1, align_dst, p_o);
  }
  AE_SA64POS_FP(align_dst, p_o);

  /*Remainder loop*/
  for(i = 0; i < (num_elm & 3); i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_inp16_0;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_L8S_IP(d_inp0, p_i, 1);
    d_inp16_0 = AE_SUB16(d_inp0, d_inp_zero_bias);
    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp16_0, ONE);
    d_out0 = MUL_SX2(d_inp32_0, d_inp_scale);
    AE_SSIP(d_out0, (xtfloat *)p_o, sizeof(FLOAT32));
  }
  return 0;
}
#else
WORD32 xa_nn_elm_dequantize_asym8s_f32(FLOAT32 * __restrict__ p_out,
                                       const WORD8 * __restrict__ p_inp,
                                       WORD32  inp_zero_bias,
                                       FLOAT32 inp_scale,
                                       WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);

  int i;
  xtfloat *p_o = (xtfloat *)p_out;
  WORD8 *p_i = (WORD8 *)p_inp;

  ALIGN_REGISTER_TYPE align_inp;
  PRIME_8X4U(p_i, align_inp);

  ae_valign align_dst = AE_ZALIGN64();

  ae_int16x4 d_inp_zero_bias = AE_MOVDA16(inp_zero_bias);
  ae_int16x4 ONE = AE_MOVDA16(1);

  xtfloatx2 d_inp_scale = inp_scale;

  xtfloatx2 d_out0, d_out1;

  for(i = 0; i < (num_elm >> 2); i++)
  {
    ae_int16x4 d_inp;
    ae_int16x4 d_inp16;
    ae_int32x2 d_inp32_0, d_inp32_1;

    AE_LA8X4U_IP(d_inp, align_inp, p_i);

    /* sign extend the 8-bit LSB in the 16-bit containers */
    d_inp = AE_MOVINT16X4_FROMINT64(
    AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_inp), 8) );
    d_inp = AE_SRAI16(d_inp, 8);

    d_inp16 =  AE_SUB16(d_inp, d_inp_zero_bias);

    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp16, ONE);

    d_out0 = XT_MUL_SX2(d_inp32_0, d_inp_scale);
    d_out1 = XT_MUL_SX2(d_inp32_1, d_inp_scale);

    AE_SA32X2_IP(XT_AE_MOVINT32X2_FROMXTFLOATX2(d_out0), align_dst, (ae_int32x2 *)p_o);
    AE_SA32X2_IP(XT_AE_MOVINT32X2_FROMXTFLOATX2(d_out1), align_dst, (ae_int32x2 *)p_o);
  }
  AE_SA64POS_FP(align_dst, p_o);

  /*Remainder loop*/
  for(i = 0; i < (num_elm & 3); i++)
  {
    WORD8 d_inp = *p_i;
    p_i++;

    WORD16 d_inp16 = d_inp - (WORD16)inp_zero_bias;

    FLOAT32 d_float = d_inp16 * inp_scale;

    *p_o = d_float;
    p_o++;
  }

  return 0;
}
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_elm_quantize_f32_asym8s,
                               (WORD8 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp,
                               FLOAT32 out_scale,
                               WORD32  out_zero_bias,
                               WORD32  num_elm))
#else /* #if !HAVE_VFPU */
WORD32 xa_nn_elm_quantize_f32_asym8s(WORD8 * __restrict__ p_out,
                                     const FLOAT32 * __restrict__ p_inp,
                                     FLOAT32 out_scale,
                                     WORD32  out_zero_bias,
                                     WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_scale == 0.0f || !isfinite(out_scale)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);

  int i;
  xtfloatx2 *p_i = (xtfloatx2 *)p_inp;
  WORD8 *p_o = (WORD8 *)p_out;
  ae_int32x2 d_out_zero_bias = AE_MOVDA32(out_zero_bias);
  xtfloat *out_scale_ptr = &out_scale;
  ae_valign align_inp = AE_LA64_PP(p_inp);
  ae_int32x2 quant_max = AE_MOVDA32(127);
  ae_int32x2 quant_min = AE_MOVDA32(-128);
  xtfloatx2 d_out_scale = (xtfloatx2)*out_scale_ptr;
  xtfloatx2 d_one = XT_FLOAT_SX2(AE_MOVDA32(1), 0);
  xtfloatx2 d_one_over_out_scale = XT_DIV_SX2(d_one, d_out_scale);

#if (XCHAL_HAVE_HIFI1 &( XCHAL_HW_VERSION >= RI9_HWVERSION ))
  ALIGN_REGISTER_TYPE align_out = AE_ZALIGN64();
#endif
  for(i = 0; i < (num_elm >> 2); i++)
  {
    xtfloatx2 d_inp0, d_inp1;
    xtfloatx2 d_inp0_t, d_inp1_t;
    ae_int32x2 d_out32_0, d_out32_1;

    XT_LASX2IP(d_inp0, align_inp, p_i);
    XT_LASX2IP(d_inp1, align_inp, p_i);
    d_inp0_t = XT_MUL_SX2(d_inp0, d_one_over_out_scale);
    d_inp1_t = XT_MUL_SX2(d_inp1, d_one_over_out_scale);
    d_inp0_t = XT_FIROUND_SX2(d_inp0_t);
    d_inp1_t = XT_FIROUND_SX2(d_inp1_t);    
    d_out32_0 = XT_TRUNC_SX2(d_inp0_t, 0);
    d_out32_1 = XT_TRUNC_SX2(d_inp1_t, 0);
    d_out32_0 = AE_ADD32S(d_out32_0, d_out_zero_bias);
    d_out32_1 = AE_ADD32S(d_out32_1, d_out_zero_bias);
#if (XCHAL_HAVE_HIFI1 &( XCHAL_HW_VERSION >= RI9_HWVERSION ))
    // clamped_out
    ae_int8x8 clamped = AE_SAT8X4X32_H(d_out32_0, d_out32_1);
    // Store Output
    AE_SAV8X8_XP(clamped, align_out, (ae_int8x8 *)p_o, 4);
#else
    CLAMP_VAL(d_out32_0, d_out32_0, quant_min, quant_max);
    CLAMP_VAL(d_out32_1, d_out32_1, quant_min, quant_max);
    STORE_8X4_FROM_32X4(p_o, d_out32_0, d_out32_1)
#endif
  }
#if (XCHAL_HAVE_HIFI1 &( XCHAL_HW_VERSION >= RI9_HWVERSION ))
    AE_SA64POS_FP(align_out, p_o);
#endif
  for(i = 0; i < (num_elm & 3) ; i++)
  {
    xtfloat d_inp0;
    xtfloat d_inp0_t;
    ae_int32x2 d_out32_0;
    XT_LSIP(d_inp0, (xtfloat *)p_i, sizeof(FLOAT32));
    d_inp0_t = XT_MUL_S(d_inp0, d_one_over_out_scale);
    d_inp0_t = XT_FIROUND_S(d_inp0_t);
    d_out32_0 = XT_TRUNC_S(d_inp0_t, 0);    
    d_out32_0 = AE_ADD32S(d_out32_0, d_out_zero_bias);
    CLAMP_VAL(d_out32_0, d_out32_0, quant_min, quant_max);
    *p_o = (WORD8)AE_MOVAD32_L(d_out32_0);
    p_o++;
  }

  return 0;
}
#endif /* #if !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_elm_quantize_f32_asym16s,
                               (WORD16 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp,
                               FLOAT32 out_scale,
                               WORD32  out_zero_bias,
                               WORD32  num_elm))
#else /* #if !HAVE_VFPU */
WORD32 xa_nn_elm_quantize_f32_asym16s(WORD16 * __restrict__ p_out,
                                     const FLOAT32 * __restrict__ p_inp,
                                     FLOAT32 out_scale,
                                     WORD32  out_zero_bias,
                                     WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_scale == 0.0f || !isfinite(out_scale)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -32768) || (out_zero_bias > 32767)), -1);

  int i;
  xtfloatx2 *p_i = (xtfloatx2 *)p_inp;
  WORD16 *p_o = (WORD16 *)p_out;
  ae_int32x2 d_out_zero_bias = AE_MOVDA32(out_zero_bias);
  xtfloat *out_scale_ptr = &out_scale;
  ae_valign align_inp = AE_LA64_PP(p_inp);
  ae_valign align_out = AE_ZALIGN64();
  ae_int32x2 quant_max = AE_MOVDA32(32767);
  ae_int32x2 quant_min = AE_MOVDA32(-32768);
  xtfloatx2 d_out_scale = (xtfloatx2)*out_scale_ptr;
  xtfloatx2 d_one = XT_FLOAT_SX2(AE_MOVDA32(1), 0);
  xtfloatx2 d_one_over_out_scale = XT_DIV_SX2(d_one, d_out_scale);

  for(i = 0; i < (num_elm >> 2); i++)
  {
    xtfloatx2 d_inp0, d_inp1;
    xtfloatx2 d_inp0_t, d_inp1_t;
    ae_int32x2 d_out32_0, d_out32_1;

    XT_LASX2IP(d_inp0, align_inp, p_i);
    XT_LASX2IP(d_inp1, align_inp, p_i);
    d_inp0_t = XT_MUL_SX2(d_inp0, d_one_over_out_scale);
    d_inp1_t = XT_MUL_SX2(d_inp1, d_one_over_out_scale);
    d_inp0_t = XT_FIROUND_SX2(d_inp0_t);
    d_inp1_t = XT_FIROUND_SX2(d_inp1_t);    
    d_out32_0 = XT_TRUNC_SX2(d_inp0_t, 0);
    d_out32_1 = XT_TRUNC_SX2(d_inp1_t, 0);
    d_out32_0 = AE_ADD32S(d_out32_0, d_out_zero_bias);
    d_out32_1 = AE_ADD32S(d_out32_1, d_out_zero_bias);

    CLAMP_VAL(d_out32_0, d_out32_0, quant_min, quant_max);
    CLAMP_VAL(d_out32_1, d_out32_1, quant_min, quant_max);
    ae_int16x4 outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(d_out32_0), AE_MOVINT16X4_FROMINT32X2(d_out32_1));
    AE_SA16X4_IP(outval, align_out, (ae_int16x4 *)p_o);
  }
  AE_SA64POS_FP(align_out, (ae_int16x4 *)p_o);

  for(i = 0; i < (num_elm & 3) ; i++)
  {
    xtfloat d_inp0;
    xtfloat d_inp0_t;
    ae_int32x2 d_out32_0;
    XT_LSIP(d_inp0, (xtfloat *)p_i, sizeof(FLOAT32));
    d_inp0_t = XT_MUL_S(d_inp0, d_one_over_out_scale);
    d_inp0_t = XT_FIROUND_S(d_inp0_t);
    d_out32_0 = XT_TRUNC_S(d_inp0_t, 0);    
    d_out32_0 = AE_ADD32S(d_out32_0, d_out_zero_bias);
    CLAMP_VAL(d_out32_0, d_out32_0, quant_min, quant_max);
    *p_o = (WORD16)AE_MOVAD32_L(d_out32_0);
    p_o++;
  }

  return 0;
}
#endif /* #if !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_elm_dequantize_asym16s_f32,
                               (FLOAT32 * __restrict__ p_out,
                               const WORD16 * __restrict__ p_inp,
                               WORD32  inp_zero_bias,
                               FLOAT32  inp_scale,
                               WORD32  num_elm))
#else
WORD32 xa_nn_elm_dequantize_asym16s_f32(FLOAT32 * __restrict__ p_out,
                                       const WORD16 * __restrict__ p_inp,
                                       WORD32  inp_zero_bias,
                                       FLOAT32 inp_scale,
                                       WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);

  int i;
  xtfloat *p_o = (xtfloat *)p_out;
  WORD16 *p_i = (WORD16 *)p_inp;

  ae_valign align_inp;
  align_inp = AE_LA64_PP((ae_int16x4*)p_i);

  ae_valign align_dst = AE_ZALIGN64();

  ae_int32x2 d_inp_zero_bias = AE_MOVDA32(inp_zero_bias);
  ae_int16x4 ONE = AE_MOVDA16(1);

  xtfloatx2 d_inp_scale = inp_scale;

  xtfloatx2 d_out0, d_out1;

  for(i = 0; i < (num_elm >> 2); i++)
  {
    ae_int16x4 d_inp;
    ae_int32x2 d_inp32_0, d_inp32_1;

    AE_LA16X4_IP(d_inp, align_inp, (ae_int16x4 *)p_i);

    /* Extend data to 32 bits */
    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp, ONE);
    d_inp32_0 = AE_SUB32(d_inp32_0, d_inp_zero_bias);
    d_inp32_1 = AE_SUB32(d_inp32_1, d_inp_zero_bias);

    d_out0 = XT_MUL_SX2(d_inp32_0, d_inp_scale);
    d_out1 = XT_MUL_SX2(d_inp32_1, d_inp_scale);

    AE_SA32X2_IP(XT_AE_MOVINT32X2_FROMXTFLOATX2(d_out0), align_dst, (ae_int32x2 *)p_o);
    AE_SA32X2_IP(XT_AE_MOVINT32X2_FROMXTFLOATX2(d_out1), align_dst, (ae_int32x2 *)p_o);
  }
  AE_SA64POS_FP(align_dst, p_o);

  /*Remainder loop*/
  for(i = 0; i < (num_elm & 3); i++)
  {
    WORD16 d_inp = *p_i;
    p_i++;

    WORD32 d_inp32 = (WORD32)d_inp - (WORD32)inp_zero_bias;
    FLOAT32 d_float = d_inp32 * inp_scale;

    *p_o = d_float;
    p_o++;
  }

  return 0;
}
#endif
