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

#define STORE_8X4_FROM_32X4(out_ptr, val12, val34){\
    int o1, o2, o3, o4;\
    o1 = AE_MOVAD32_H(val12);\
    o2 = AE_MOVAD32_L(val12);\
    o3 = AE_MOVAD32_H(val34);\
    o4 = AE_MOVAD32_L(val34);\
    *out_ptr++ = (UWORD8)o1;\
    *out_ptr++ = (UWORD8)o2;\
    *out_ptr++ = (UWORD8)o3;\
    *out_ptr++ = (UWORD8)o4;\
}
 
WORD32 xa_nn_renorm_asym8s_asym8s(WORD8 * __restrict__ p_out,
                              const WORD8 * __restrict__ p_inp,
                              WORD32 num_elm,
                              WORD32 renorm_scale,
                              WORD32 renorm_shift,
                              WORD32 input_zero_bias,
                              WORD32 output_zero_bias)
{
  /* NULL pointer check */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Basic Parameter checks */  
  XA_NNLIB_ARG_CHK_COND((renorm_shift<0 || renorm_shift >= 24),-1);
  XA_NNLIB_ARG_CHK_COND((renorm_scale<0 || renorm_scale > 65535),-1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -128 || input_zero_bias > 127),-1);
  XA_NNLIB_ARG_CHK_COND((output_zero_bias < -128 || output_zero_bias > 127),-1);
 
  WORD32 zero_in_out = (input_zero_bias * renorm_scale) - ((WORD32)output_zero_bias << renorm_shift);
  ae_int32x2 renorm_scale32 = AE_MOVDA32(renorm_scale);
  ae_int32x2 zero_point32 = AE_MOVDA32(zero_in_out);
 
  ae_int16x4 d_inp1;
  ae_int32x2 zero_point32_00, zero_point32_01;
  ae_int32x2 out32_00, out32_01;
  ae_int16x4 out16_0;
 
  const WORD8 * __restrict__ inp_ptr = p_inp;
  WORD8 * __restrict__ out_ptr = (WORD8 *)p_out;
  ae_int32x2 shift_one = AE_SRAA32(AE_MOVDA32(0x80000000), renorm_shift);
  int remainder;

#if !XCHAL_HAVE_HIFI1 || ( XCHAL_HW_VERSION < RI9_HWVERSION )
  ae_int16x4 CONST_127_16x4 = AE_MOVDA16(127);
  ae_int16x4 CONST_MINUS_128_16x4 = AE_MOVDA16(-128);
#else
  ae_valign align_out = AE_ZALIGN64();
  ae_valign align_in = AE_LA64_PP(inp_ptr);
#endif
 
  int i;
  int preloop_count = ((uintptr_t)inp_ptr & 0x3)? (4-((uintptr_t)inp_ptr & 0x3)):0;
  preloop_count = (preloop_count<num_elm)?preloop_count:num_elm;

#if !(XCHAL_HAVE_HIFI1 && ( XCHAL_HW_VERSION >= RI9_HWVERSION ))
  for(i=0; i<preloop_count; i++)
  {
    d_inp1 = (WORD16)*inp_ptr++;
    zero_point32_00 = AE_MOVDA32(zero_point32);
    AE_MULSP32X16X2_H(zero_point32_00, renorm_scale32, d_inp1);
    out32_00 = AE_MULFP32X2RAS(zero_point32_00, shift_one);
    
    out16_0 = AE_SAT16X4(out32_00, out32_00);
    xtbool4 bsat4 = AE_LT16(CONST_127_16x4, out16_0);
    AE_MOVT16X4(out16_0, CONST_127_16x4 , bsat4);
    bsat4 = AE_LT16(out16_0, CONST_MINUS_128_16x4);
    AE_MOVT16X4(out16_0, CONST_MINUS_128_16x4 , bsat4);
    *out_ptr++ = (WORD8)AE_MOVAD16_0(out16_0);
  }
  num_elm -= preloop_count;
#endif

  for(i=0; i<(num_elm & ~(4-1)); i+=4)
  {
#if XCHAL_HAVE_HIFI1 && ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    AE_LA8X4S_IP(d_inp1, align_in, inp_ptr);
#else
    AE_L8X4F_IP(d_inp1, inp_ptr, 4);
    d_inp1 = AE_SRAI16(d_inp1, 8);
#endif
   
    zero_point32_00 = AE_MOVDA32(zero_point32);
    zero_point32_01 = AE_MOVDA32(zero_point32);
 
    AE_MULSP32X16X2_H(zero_point32_00, renorm_scale32, d_inp1);
    AE_MULSP32X16X2_L(zero_point32_01, renorm_scale32, d_inp1);
   
    out32_00 = AE_MULFP32X2RAS(zero_point32_00, shift_one);
    out32_01 = AE_MULFP32X2RAS(zero_point32_01, shift_one);
 
#if XCHAL_HAVE_HIFI1 && ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    out16_0 = AE_SAT16X4(out32_00, out32_01);
    out16_0 = AE_SAT8S(out16_0);
    AE_SA8X4U_IP(out16_0, align_out, (ae_int32*)out_ptr);
#else
    out32_00 = AE_MIN32(out32_00, AE_MOVDA32(127));
    out32_01 = AE_MIN32(out32_01, AE_MOVDA32(127));
    out32_00 = AE_MAX32(out32_00, AE_MOVDA32(-128));
    out32_01 = AE_MAX32(out32_01, AE_MOVDA32(-128));
    STORE_8X4_FROM_32X4(out_ptr, out32_00, out32_01);
#endif
  }

  remainder = num_elm&3;

#if XCHAL_HAVE_HIFI1 && ( XCHAL_HW_VERSION >= RI9_HWVERSION )
  if(remainder)
  {
    AE_LAV8X4S_XP(d_inp1, align_in, (ae_int8x4 *)inp_ptr, remainder);
    zero_point32_00 = AE_MOVDA32(zero_point32);
    zero_point32_01 = AE_MOVDA32(zero_point32);
    AE_MULSP32X16X2_H(zero_point32_00, renorm_scale32, d_inp1);
    AE_MULSP32X16X2_L(zero_point32_01, renorm_scale32, d_inp1);
    out32_00 = AE_MULFP32X2RAS(zero_point32_00, shift_one);
    out32_01 = AE_MULFP32X2RAS(zero_point32_01, shift_one);
   
    out16_0 = AE_SAT16X4(out32_00, out32_01);
    out16_0 = AE_SAT8S(out16_0);
    AE_SAV8X4U_XP(out16_0, align_out, (ae_int8x4u *)out_ptr, remainder);
  }
  AE_SA64POS_FP(align_out, out_ptr);
#else
  for(i=0;i<remainder;i++)
  {
    d_inp1 = (WORD16)*inp_ptr++;
    zero_point32_00 = AE_MOVDA32(zero_point32);
    AE_MULSP32X16X2_H(zero_point32_00, renorm_scale32, d_inp1);
    out32_00 = AE_MULFP32X2RAS(zero_point32_00, shift_one);
   
    out16_0 = AE_SAT16X4(out32_00, out32_00);
    xtbool4 bsat4 = AE_LT16(CONST_127_16x4, out16_0);
    AE_MOVT16X4(out16_0, CONST_127_16x4 , bsat4);
    bsat4 = AE_LT16(out16_0, CONST_MINUS_128_16x4);
    AE_MOVT16X4(out16_0, CONST_MINUS_128_16x4 , bsat4);
    *out_ptr++ = (WORD8)AE_MOVAD16_0(out16_0);
  }
#endif
  return 0;
}
