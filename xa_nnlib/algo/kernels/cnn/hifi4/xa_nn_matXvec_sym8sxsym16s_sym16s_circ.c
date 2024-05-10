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
#include "xa_nn_conv2d_std_state.h"
#include "xa_nnlib_common_macros.h"

#if XCHAL_HAVE_HIFI1S
static inline ae_int32x2 MultiplyByQuantizedMultiplier_opt(ae_int64 d_x,
                                             int32_t quantized_multiplier,
                                             int shift){
  ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
  ae_int64 q = AE_MUL48X16_0(d_x, d_red_mul16);
  ae_int32x2 result = AE_ROUNDAV32X2F64SASYM (q, q, shift);
  return result;
}

static inline ae_int32x2 MultiplyByQuantizedMultiplier_x2_opt(ae_int64 d_x1, ae_int64 d_x2,
                                             int32_t quantized_multiplier,
                                             int shift) {
  ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
  ae_int64 q1 = AE_MUL48X16_0(d_x1, d_red_mul16);
  ae_int64 q2 = AE_MUL48X16_0(d_x2, d_red_mul16);
  ae_int32x2 result = AE_ROUNDAV32X2F64SASYM (q1, q2, shift);
  return result;
}
#else
static inline ae_int32x2 MultiplyByQuantizedMultiplier_opt(ae_int64 d_x,
                                             int32_t quantized_multiplier,
                                             int shift){
  // Inputs:
  // - quantized_multiplier has fixed point at bit 31
  // - shift is -31 to +7 (negative for right shift)
  //
  // Assumptions: The following input ranges are assumed
  // - quantize_scale>=0  (the usual range is (1<<30) to (1>>31)-1)
  // - scaling is chosen so final scaled result fits in int32_t
  // - input x is in the range -(1<<47) <= x < (1<<47)
/* shift_val  = -31 to 7
 * total_shift = 46 to 8
 * new_shift(-shift-17) = 46-32 to 8-32
 * */
  ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
  ae_int32x2 d_red_mul32 = AE_SEXT32X2D16_32(d_red_mul16);
  ae_int64 qL = AE_MUL32U_LL(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x));
  ae_int64 qH = AE_SLAI64(AE_MUL32_LH(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x)), 32);
  ae_int64 q = AE_ADD64(qL, qH);
  q = AE_SRAA64(q, (-shift-17));
  ae_int32x2 result = AE_ROUND32F64SASYM(q);
  return result;
}

static inline ae_int32x2 MultiplyByQuantizedMultiplier_x2_opt(ae_int64 d_x1, ae_int64 d_x2,
                                             int32_t quantized_multiplier,
                                             int shift) {
  ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
  ae_int32x2 d_red_mul32 = AE_SEXT32X2D16_32(d_red_mul16);
  ae_int64 qL1 = AE_MUL32U_LL(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x1));
  ae_int64 qL2 = AE_MUL32U_LL(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x2));
  ae_int64 qH1 = AE_SLAI64(AE_MUL32_LH(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x1)), 32);
  ae_int64 qH2 = AE_SLAI64(AE_MUL32_LH(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x2)), 32);
  ae_int64 q1 = AE_ADD64(qL1, qH1);
  ae_int64 q2 = AE_ADD64(qL2, qH2);
  q1 = AE_SRAA64(q1, (-shift-17));
  q2 = AE_SRAA64(q2, (-shift-17));
  ae_int32x2 result = AE_ROUND32X2F64SASYM(q1, q2);
  return result;
}
#endif

static inline void _xa_nn_dot_product_1row_1vec_mat_vecs_4bytes_aligned
(ae_int64 *out_0_0
 ,WORD16*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD32      cols1)
{
  int c_itr = 0;
  ae_int16x4 d_vec0;
  ae_int16x4 d_mat0;
  ae_int64 d_out0;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  d_out0 = *out_0_0;
#if !XCHAL_HAVE_HIFI1
  d_out0 = AE_SLAI64(d_out0, 8); 
#endif
  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L16X4_XC(d_mat0, pmat0, 8);
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_vec0, p_vec_0, 4);
#else
    AE_L8X4F_IP(d_vec0, p_vec_0, 4);
#endif
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
  }
#if XCHAL_HAVE_HIFI1
  *out_0_0 = d_out0;
#else
  *out_0_0 = AE_SRAI64(d_out0, 8);
#endif  
}

static inline void _xa_nn_dot_product_2row_1vec_mat_vecs_4bytes_aligned
(ae_int64 *out_0_0
 ,ae_int64 *out_1_1
 ,WORD16*      p_mat_0
 ,WORD16*      p_mat_1
 ,WORD8*      p_vec_0
 ,WORD32      cols1)
{
  int c_itr = 0;
  ae_int16x4 d_vec0;
  ae_int16x4 d_mat0, d_mat1;
  ae_int64 d_out0, d_out1;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  ae_int16x4 *pmat1 = (ae_int16x4*)p_mat_1;
  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
#if !XCHAL_HAVE_HIFI1
  d_out0 = AE_SLAI64(d_out0, 8);
  d_out1 = AE_SLAI64(d_out1, 8);
#endif
  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L16X4_XC(d_mat0, pmat0, 8);
    AE_L16X4_XC(d_mat1, pmat1, 8);
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_vec0, p_vec_0, 4);
#else
    AE_L8X4F_IP(d_vec0, p_vec_0, 4);
#endif
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
  }
#if XCHAL_HAVE_HIFI1
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
#else
  *out_0_0 = AE_SRAI64(d_out0, 8);
  *out_1_1 = AE_SRAI64(d_out1, 8);
#endif
}

static inline void _xa_nn_dot_product_1row_2vec_mat_vecs_4bytes_aligned
(ae_int64 *out_0_0
 ,ae_int64 *out_2_2
 ,WORD16*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD8*      p_vec_1
 ,WORD32      cols1)
{
  int c_itr = 0;
  ae_int16x4 d_vec0, d_vec1;
  ae_int16x4 d_mat0;
  ae_int64 d_out0, d_out2;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  d_out0 = *out_0_0;
  d_out2 = *out_2_2;
#if !XCHAL_HAVE_HIFI1
  d_out0 = AE_SLAI64(d_out0, 8);
  d_out2 = AE_SLAI64(d_out2, 8);
#endif
  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L16X4_XC(d_mat0, pmat0, 8);
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_vec0, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1, p_vec_1, 4);
#else
    AE_L8X4F_IP(d_vec0, p_vec_0, 4);
    AE_L8X4F_IP(d_vec1, p_vec_1, 4);
#endif
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec1);
  }
#if XCHAL_HAVE_HIFI1
  *out_0_0 = d_out0;
  *out_2_2 = d_out2;
#else
  *out_0_0 = AE_SRAI64(d_out0, 8);
  *out_2_2 = AE_SRAI64(d_out2, 8);
#endif
}

static inline void _xa_nn_dot_product_2row_2vec_mat_vecs_4bytes_aligned
(ae_int64 *out_0_0
 ,ae_int64 *out_1_1
 ,ae_int64 *out_2_2
 ,ae_int64 *out_3_3
 ,WORD16*      p_mat_0
 ,WORD16*      p_mat_1
 ,WORD8*      p_vec_0
 ,WORD8*      p_vec_1
 ,WORD32      cols1)
{
#if XCHAL_HAVE_HIFI1S
  int c_itr = 0;
  ae_int8x8 d_vec0, d_vec1;
  ae_int16x4 d_vec0_16, d_vec1_16;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  ae_int16x4 *pmat1 = (ae_int16x4*)p_mat_1;
  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;

  ae_valign valign_vec_0 = AE_LA64_PP(p_vec_0);
  ae_valign valign_vec_1 = AE_LA64_PP(p_vec_1);
  for(;c_itr<(cols1>>3); c_itr++)
  {
    AE_L16X4_XC(d_mat0, pmat0, 8);
    AE_L16X4_XC(d_mat1, pmat1, 8);
    AE_L16X4_XC(d_mat2, pmat0, 8);
    AE_L16X4_XC(d_mat3, pmat1, 8);
    AE_LA8X8_IP(d_vec0, valign_vec_0, (ae_int8x8*)p_vec_0);
    AE_LA8X8_IP(d_vec1, valign_vec_1, (ae_int8x8*)p_vec_1);
    AE_MULAO8X16(d_out0, d_mat0, d_mat2, d_vec0);
    AE_MULAO8X16(d_out1, d_mat1, d_mat3, d_vec0);
    AE_MULAO8X16(d_out2, d_mat0, d_mat2, d_vec1);
    AE_MULAO8X16(d_out3, d_mat1, d_mat3, d_vec1);
  }
  if(cols1%8)
  {
    AE_L16X4_XC(d_mat0, pmat0, 8);
    AE_L16X4_XC(d_mat1, pmat1, 8);
    AE_L8X4S_IP(d_vec0_16, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1_16, p_vec_1, 4);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0_16);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0_16);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec1_16);
    AE_MULAAAAQ16(d_out3, d_mat1, d_vec1_16);
  }
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;
#else  
  int c_itr = 0;
  ae_int16x4 d_vec0, d_vec1;
  ae_int16x4 d_mat0, d_mat1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  ae_int16x4 *pmat1 = (ae_int16x4*)p_mat_1;
  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;
#if !XCHAL_HAVE_HIFI1
  d_out0 = AE_SLAI64(d_out0, 8);
  d_out1 = AE_SLAI64(d_out1, 8);
  d_out2 = AE_SLAI64(d_out2, 8);
  d_out3 = AE_SLAI64(d_out3, 8);
#endif
  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L16X4_XC(d_mat0, pmat0, 8);
    AE_L16X4_XC(d_mat1, pmat1, 8);
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_vec0, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1, p_vec_1, 4);
#else
    AE_L8X4F_IP(d_vec0, p_vec_0, 4);
    AE_L8X4F_IP(d_vec1, p_vec_1, 4);
#endif
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out3, d_mat1, d_vec1);
  }
#if XCHAL_HAVE_HIFI1
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;
#else
  *out_0_0 = AE_SRAI64(d_out0, 8);
  *out_1_1 = AE_SRAI64(d_out1, 8);
  *out_2_2 = AE_SRAI64(d_out2, 8);
  *out_3_3 = AE_SRAI64(d_out3, 8);
#endif
#endif
}

static inline void _xa_nn_dot_product_1row_4vec_mat_vecs_4bytes_aligned
(ae_int64 *out_0_0
 ,ae_int64 *out_1_1
 ,ae_int64 *out_2_2
 ,ae_int64 *out_3_3
 ,WORD16*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD8*      p_vec_1
 ,WORD8*      p_vec_2
 ,WORD8*      p_vec_3
 ,WORD32      cols1)
{
#if XCHAL_HAVE_HIFI4
  int c_itr = 0;
  ae_int16x4 d_vec0, d_vec1, d_vec2, d_vec3;
  ae_int16x4 d_vec0_I, d_vec1_I, d_vec2_I, d_vec3_I;
  ae_int16x4 d_mat0_0;
  ae_int16x4 d_mat0_1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;
  d_out0 = AE_SLAI64(d_out0, 8);
  d_out1 = AE_SLAI64(d_out1, 8);
  d_out2 = AE_SLAI64(d_out2, 8);
  d_out3 = AE_SLAI64(d_out3, 8);

  for(;c_itr<(cols1>>3); c_itr++)
  {
    AE_L16X4_XC(d_mat0_0, pmat0, 8);
    AE_L16X4_XC(d_mat0_1, pmat0, 8);

    d_vec0_I = AE_L8X4F_I(p_vec_0, 4);
    AE_L8X4F_IP(d_vec0, p_vec_0, 8);
    d_vec1_I = AE_L8X4F_I(p_vec_1, 4);
    AE_L8X4F_IP(d_vec1, p_vec_1, 8);
    d_vec2_I = AE_L8X4F_I(p_vec_2, 4);
    AE_L8X4F_IP(d_vec2, p_vec_2, 8);
    d_vec3_I = AE_L8X4F_I(p_vec_3, 4);
    AE_L8X4F_IP(d_vec3, p_vec_3, 8);

    AE_MULAAAAQ16(d_out0, d_mat0_0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat0_0, d_vec1);
    AE_MULAAAAQ16(d_out2, d_mat0_0, d_vec2);
    AE_MULAAAAQ16(d_out3, d_mat0_0, d_vec3);

    AE_MULAAAAQ16(d_out0, d_mat0_1, d_vec0_I);
    AE_MULAAAAQ16(d_out1, d_mat0_1, d_vec1_I);
    AE_MULAAAAQ16(d_out2, d_mat0_1, d_vec2_I);
    AE_MULAAAAQ16(d_out3, d_mat0_1, d_vec3_I);
  }
  if((cols1&7)!=0)
  {
    AE_L16X4_XC(d_mat0_0, pmat0, 8);
    AE_L8X4F_IP(d_vec0, p_vec_0, 4);
    AE_L8X4F_IP(d_vec1, p_vec_1, 4);
    AE_L8X4F_IP(d_vec2, p_vec_2, 4);
    AE_L8X4F_IP(d_vec3, p_vec_3, 4);
    AE_MULAAAAQ16(d_out0, d_mat0_0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat0_0, d_vec1);
    AE_MULAAAAQ16(d_out2, d_mat0_0, d_vec2);
    AE_MULAAAAQ16(d_out3, d_mat0_0, d_vec3);

  }
  *out_0_0 = AE_SRAI64(d_out0, 8);
  *out_1_1 = AE_SRAI64(d_out1, 8);
  *out_2_2 = AE_SRAI64(d_out2, 8);
  *out_3_3 = AE_SRAI64(d_out3, 8);
#elif XCHAL_HAVE_HIFI1S
  int c_itr = 0;
  ae_int8x8 d_vec0, d_vec1, d_vec2, d_vec3;
  ae_int16x4 d_mat0, d_mat1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;

  ae_valign valign_vec_0 = AE_LA64_PP(p_vec_0);
  ae_valign valign_vec_1 = AE_LA64_PP(p_vec_1);
  ae_valign valign_vec_2 = AE_LA64_PP(p_vec_2);
  ae_valign valign_vec_3 = AE_LA64_PP(p_vec_3);

  for(;c_itr<(cols1>>3); c_itr++)
  {
    AE_L16X4_XC(d_mat0, pmat0, 8);
    AE_L16X4_XC(d_mat1, pmat0, 8);
    AE_LA8X8_IP(d_vec0, valign_vec_0, (ae_int8x8*)p_vec_0);
    AE_LA8X8_IP(d_vec1, valign_vec_1, (ae_int8x8*)p_vec_1);
    AE_LA8X8_IP(d_vec2, valign_vec_2, (ae_int8x8*)p_vec_2);
    AE_LA8X8_IP(d_vec3, valign_vec_3, (ae_int8x8*)p_vec_3);
    AE_MULAO8X16(d_out0, d_mat0, d_mat1, d_vec0);
    AE_MULAO8X16(d_out1, d_mat0, d_mat1, d_vec1);
    AE_MULAO8X16(d_out2, d_mat0, d_mat1, d_vec2);
    AE_MULAO8X16(d_out3, d_mat0, d_mat1, d_vec3);
  }
  if(cols1%8)
  {
    ae_int16x4 d_vec0_16, d_vec1_16, d_vec2_16, d_vec3_16;
    AE_L16X4_XC(d_mat0, pmat0, 8);
    AE_L8X4S_IP(d_vec0_16, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1_16, p_vec_1, 4);
    AE_L8X4S_IP(d_vec2_16, p_vec_2, 4);
    AE_L8X4S_IP(d_vec3_16, p_vec_3, 4);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0_16);  
    AE_MULAAAAQ16(d_out1, d_mat0, d_vec1_16);  
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec2_16);  
    AE_MULAAAAQ16(d_out3, d_mat0, d_vec3_16);  
  }
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;  
#else /*XCHAL_HAVE_HIFI4*/
  int c_itr = 0;
  ae_int16x4 d_vec0, d_vec1, d_vec2, d_vec3;
  ae_int16x4 d_mat0;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;
#if !XCHAL_HAVE_HIFI1
  d_out0 = AE_SLAI64(d_out0, 8);
  d_out1 = AE_SLAI64(d_out1, 8);
  d_out2 = AE_SLAI64(d_out2, 8);
  d_out3 = AE_SLAI64(d_out3, 8);
#endif
  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L16X4_XC(d_mat0, pmat0, 8);
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_vec0, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1, p_vec_1, 4);
    AE_L8X4S_IP(d_vec2, p_vec_2, 4);
    AE_L8X4S_IP(d_vec3, p_vec_3, 4);
#else
    AE_L8X4F_IP(d_vec0, p_vec_0, 4);
    AE_L8X4F_IP(d_vec1, p_vec_1, 4);
    AE_L8X4F_IP(d_vec2, p_vec_2, 4);
    AE_L8X4F_IP(d_vec3, p_vec_3, 4);
#endif
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec2);
    AE_MULAAAAQ16(d_out3, d_mat0, d_vec3);
  }
#if XCHAL_HAVE_HIFI1
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;
#else
  *out_0_0 = AE_SRAI64(d_out0, 8);
  *out_1_1 = AE_SRAI64(d_out1, 8);
  *out_2_2 = AE_SRAI64(d_out2, 8);
  *out_3_3 = AE_SRAI64(d_out3, 8);
#endif

#endif /*XCHAL_HAVE_HIFI4*/
}

static inline void _xa_nn_dot_product_2row_4vec_mat_vecs_4bytes_aligned
(ae_int64 *out_0_0
 ,ae_int64 *out_1_1
 ,ae_int64 *out_2_2
 ,ae_int64 *out_3_3
 ,ae_int64 *out_4_4
 ,ae_int64 *out_5_5
 ,ae_int64 *out_6_6
 ,ae_int64 *out_7_7
 ,WORD16*      p_mat_0
 ,WORD16*      p_mat_1
 ,WORD8*      p_vec_0
 ,WORD8*      p_vec_1
 ,WORD8*      p_vec_2
 ,WORD8*      p_vec_3
 ,WORD32      cols1)
{
#if XCHAL_HAVE_HIFI4
  int c_itr = 0;
  ae_int16x4 d_vec0_0, d_vec1_0, d_vec2_0, d_vec3_0;
  ae_int16x4 d_vec0_1, d_vec1_1, d_vec2_1, d_vec3_1;
  ae_int16x4 d_mat0_0, d_mat1_0;
  ae_int16x4 d_mat0_1, d_mat1_1;
  ae_int64 d_out0, d_out1, d_out2, d_out3, d_out4, d_out5, d_out6, d_out7;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  ae_int16x4 *pmat1 = (ae_int16x4*)p_mat_1;

  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;
  d_out4 = *out_4_4;
  d_out5 = *out_5_5;
  d_out6 = *out_6_6;
  d_out7 = *out_7_7;

  d_out0 = AE_SLAI64(d_out0, 8);
  d_out1 = AE_SLAI64(d_out1, 8);
  d_out2 = AE_SLAI64(d_out2, 8);
  d_out3 = AE_SLAI64(d_out3, 8);
  d_out4 = AE_SLAI64(d_out4, 8);
  d_out5 = AE_SLAI64(d_out5, 8);
  d_out6 = AE_SLAI64(d_out6, 8);
  d_out7 = AE_SLAI64(d_out7, 8);

  for(;c_itr<(cols1>>3); c_itr++)
  {
    /*
    * AE_L16X4_XC is split into AE_L16X4_I and AE_ADDCIRC16X4_XC to improve slotting of loads to 
    * avoid memory banking conflics.
    */ 
    d_mat0_0 = AE_L16X4_I(pmat0, 0);
    AE_ADDCIRC16X4_XC(pmat0, 8);
    d_mat0_1 = AE_L16X4_I(pmat0, 0);
    AE_ADDCIRC16X4_XC(pmat0, 8);
    d_mat1_0 = AE_L16X4_I(pmat1, 0);
    AE_ADDCIRC16X4_XC(pmat1, 8);
    d_mat1_1 = AE_L16X4_I(pmat1, 0);
    AE_ADDCIRC16X4_XC(pmat1, 8);

    d_vec0_1 = AE_L8X4F_I(p_vec_0, 4);
    AE_L8X4F_IP(d_vec0_0, p_vec_0, 8);
    AE_MULAAAAQ16(d_out0, d_mat0_0, d_vec0_0);
    AE_MULAAAAQ16(d_out1, d_mat1_0, d_vec0_0);
    AE_MULAAAAQ16(d_out0, d_mat0_1, d_vec0_1);
    AE_MULAAAAQ16(d_out1, d_mat1_1, d_vec0_1);

    d_vec1_1 = AE_L8X4F_I(p_vec_1, 4);
    AE_L8X4F_IP(d_vec1_0, p_vec_1, 8);
    AE_MULAAAAQ16(d_out2, d_mat0_0, d_vec1_0);
    AE_MULAAAAQ16(d_out3, d_mat1_0, d_vec1_0);
    AE_MULAAAAQ16(d_out2, d_mat0_1, d_vec1_1);
    AE_MULAAAAQ16(d_out3, d_mat1_1, d_vec1_1);

    d_vec2_1 = AE_L8X4F_I(p_vec_2, 4);
    AE_L8X4F_IP(d_vec2_0, p_vec_2, 8);
    AE_MULAAAAQ16(d_out4, d_mat0_0, d_vec2_0);
    AE_MULAAAAQ16(d_out5, d_mat1_0, d_vec2_0);
    AE_MULAAAAQ16(d_out4, d_mat0_1, d_vec2_1);
    AE_MULAAAAQ16(d_out5, d_mat1_1, d_vec2_1);

    d_vec3_1 = AE_L8X4F_I(p_vec_3, 4);
    AE_L8X4F_IP(d_vec3_0, p_vec_3, 8);
    AE_MULAAAAQ16(d_out6, d_mat0_0, d_vec3_0);
    AE_MULAAAAQ16(d_out7, d_mat1_0, d_vec3_0);
    AE_MULAAAAQ16(d_out6, d_mat0_1, d_vec3_1);
    AE_MULAAAAQ16(d_out7, d_mat1_1, d_vec3_1);
  }
  if((cols1 & 7) != 0)
  {
    AE_L16X4_XC(d_mat0_0, pmat0, 8);
    AE_L16X4_XC(d_mat1_0, pmat1, 8);
    AE_L8X4F_IP(d_vec0_0, p_vec_0, 4);
    AE_L8X4F_IP(d_vec1_0, p_vec_1, 4);
    AE_L8X4F_IP(d_vec2_0, p_vec_2, 4);
    AE_L8X4F_IP(d_vec3_0, p_vec_3, 4);
    AE_MULAAAAQ16(d_out0, d_mat0_0, d_vec0_0);
    AE_MULAAAAQ16(d_out1, d_mat1_0, d_vec0_0);
    AE_MULAAAAQ16(d_out2, d_mat0_0, d_vec1_0);
    AE_MULAAAAQ16(d_out3, d_mat1_0, d_vec1_0);
    AE_MULAAAAQ16(d_out4, d_mat0_0, d_vec2_0);
    AE_MULAAAAQ16(d_out5, d_mat1_0, d_vec2_0);
    AE_MULAAAAQ16(d_out6, d_mat0_0, d_vec3_0);
    AE_MULAAAAQ16(d_out7, d_mat1_0, d_vec3_0);
  }
  *out_0_0 = AE_SRAI64(d_out0, 8);
  *out_1_1 = AE_SRAI64(d_out1, 8);
  *out_2_2 = AE_SRAI64(d_out2, 8);
  *out_3_3 = AE_SRAI64(d_out3, 8);
  *out_4_4 = AE_SRAI64(d_out4, 8);
  *out_5_5 = AE_SRAI64(d_out5, 8);
  *out_6_6 = AE_SRAI64(d_out6, 8);
  *out_7_7 = AE_SRAI64(d_out7, 8);
#else /* XCHAL_HAVE_HIFI4 */
#if XCHAL_HAVE_HIFI3Z
  int c_itr = 0;
  ae_int16x4 d_vec0_0, d_vec1_0, d_vec2_0, d_vec3_0;
  ae_int16x4 d_vec0_1, d_vec1_1, d_vec2_1, d_vec3_1;
  ae_int16x4 d_mat0_0, d_mat1_0;
  ae_int16x4 d_mat0_1, d_mat1_1;
  ae_int64 d_out0, d_out1, d_out2, d_out3, d_out4, d_out5, d_out6, d_out7;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  ae_int16x4 *pmat1 = (ae_int16x4*)p_mat_1;

  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;
  d_out4 = *out_4_4;
  d_out5 = *out_5_5;
  d_out6 = *out_6_6;
  d_out7 = *out_7_7;

  d_out0 = AE_SLAI64(d_out0, 8);
  d_out1 = AE_SLAI64(d_out1, 8);
  d_out2 = AE_SLAI64(d_out2, 8);
  d_out3 = AE_SLAI64(d_out3, 8);
  d_out4 = AE_SLAI64(d_out4, 8);
  d_out5 = AE_SLAI64(d_out5, 8);
  d_out6 = AE_SLAI64(d_out6, 8);
  d_out7 = AE_SLAI64(d_out7, 8);

  for(;c_itr<(cols1>>3); c_itr++)
  {
    AE_L16X4_XC(d_mat0_0, pmat0, 8);
    AE_L16X4_XC(d_mat0_1, pmat0, 8);
    AE_L16X4_XC(d_mat1_0, pmat1, 8);
    AE_L16X4_XC(d_mat1_1, pmat1, 8);

    d_vec0_1 = AE_L8X4F_I(p_vec_0, 4);
    d_vec0_0 = AE_L8X4F_I(p_vec_0, 0);
    p_vec_0+=8;
    AE_MULAAAAQ16(d_out0, d_mat0_0, d_vec0_0);
    AE_MULAAAAQ16(d_out1, d_mat1_0, d_vec0_0);
    AE_MULAAAAQ16(d_out0, d_mat0_1, d_vec0_1);
    AE_MULAAAAQ16(d_out1, d_mat1_1, d_vec0_1);

    d_vec1_1 = AE_L8X4F_I(p_vec_1, 4);
    AE_L8X4F_IP(d_vec1_0, p_vec_1, 8);
    AE_MULAAAAQ16(d_out2, d_mat0_0, d_vec1_0);
    AE_MULAAAAQ16(d_out3, d_mat1_0, d_vec1_0);
    AE_MULAAAAQ16(d_out2, d_mat0_1, d_vec1_1);
    AE_MULAAAAQ16(d_out3, d_mat1_1, d_vec1_1);

    d_vec2_1 = AE_L8X4F_I(p_vec_2, 4);
    AE_L8X4F_IP(d_vec2_0, p_vec_2, 8);
    AE_MULAAAAQ16(d_out4, d_mat0_0, d_vec2_0);
    AE_MULAAAAQ16(d_out5, d_mat1_0, d_vec2_0);
    AE_MULAAAAQ16(d_out4, d_mat0_1, d_vec2_1);
    AE_MULAAAAQ16(d_out5, d_mat1_1, d_vec2_1);

    d_vec3_1 = AE_L8X4F_I(p_vec_3, 4);
    AE_L8X4F_IP(d_vec3_0, p_vec_3, 8);
    AE_MULAAAAQ16(d_out6, d_mat0_0, d_vec3_0);
    AE_MULAAAAQ16(d_out7, d_mat1_0, d_vec3_0);
    AE_MULAAAAQ16(d_out6, d_mat0_1, d_vec3_1);
    AE_MULAAAAQ16(d_out7, d_mat1_1, d_vec3_1);
  }
  if((cols1 & 7) != 0)
  {
    AE_L16X4_XC(d_mat0_0, pmat0, 8);
    AE_L16X4_XC(d_mat1_0, pmat1, 8);
    AE_L8X4F_IP(d_vec0_0, p_vec_0, 4);
    AE_L8X4F_IP(d_vec1_0, p_vec_1, 4);
    AE_L8X4F_IP(d_vec2_0, p_vec_2, 4);
    AE_L8X4F_IP(d_vec3_0, p_vec_3, 4);
    AE_MULAAAAQ16(d_out0, d_mat0_0, d_vec0_0);
    AE_MULAAAAQ16(d_out1, d_mat1_0, d_vec0_0);
    AE_MULAAAAQ16(d_out2, d_mat0_0, d_vec1_0);
    AE_MULAAAAQ16(d_out3, d_mat1_0, d_vec1_0);
    AE_MULAAAAQ16(d_out4, d_mat0_0, d_vec2_0);
    AE_MULAAAAQ16(d_out5, d_mat1_0, d_vec2_0);
    AE_MULAAAAQ16(d_out6, d_mat0_0, d_vec3_0);
    AE_MULAAAAQ16(d_out7, d_mat1_0, d_vec3_0);
  }
  *out_0_0 = AE_SRAI64(d_out0, 8);
  *out_1_1 = AE_SRAI64(d_out1, 8);
  *out_2_2 = AE_SRAI64(d_out2, 8);
  *out_3_3 = AE_SRAI64(d_out3, 8);
  *out_4_4 = AE_SRAI64(d_out4, 8);
  *out_5_5 = AE_SRAI64(d_out5, 8);
  *out_6_6 = AE_SRAI64(d_out6, 8);
  *out_7_7 = AE_SRAI64(d_out7, 8);
#else /* XCHAL_HAVE_HIFI3Z */
#if XCHAL_HAVE_HIFI1S
  int c_itr = 0;
  ae_int8x8 d_vec0, d_vec1, d_vec2, d_vec3;
  ae_int16x4 d_vec0_16, d_vec1_16, d_vec2_16, d_vec3_16;
  ae_int16x4 d_mat0, d_mat1;
  ae_int16x4 d_mat0n, d_mat1n;
  ae_int64 d_out0, d_out1, d_out2, d_out3, d_out4, d_out5, d_out6, d_out7;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  ae_int16x4 *pmat1 = (ae_int16x4*)p_mat_1;
  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;
  d_out4 = *out_4_4;
  d_out5 = *out_5_5;
  d_out6 = *out_6_6;
  d_out7 = *out_7_7;

  ae_valign valign_vec_0 = AE_LA64_PP(p_vec_0);
  ae_valign valign_vec_1 = AE_LA64_PP(p_vec_1);
  ae_valign valign_vec_2 = AE_LA64_PP(p_vec_2);
  ae_valign valign_vec_3 = AE_LA64_PP(p_vec_3);

  for(;c_itr<(cols1>>3); c_itr++)
  {
    AE_L16X4_XC(d_mat0, pmat0, 8);
    AE_L16X4_XC(d_mat1, pmat1, 8);
    AE_L16X4_XC(d_mat0n, pmat0, 8);
    AE_L16X4_XC(d_mat1n, pmat1, 8);

    AE_LA8X8_IP(d_vec0, valign_vec_0, (ae_int8x8*)p_vec_0);
    AE_LA8X8_IP(d_vec1, valign_vec_1, (ae_int8x8*)p_vec_1);
    AE_LA8X8_IP(d_vec2, valign_vec_2, (ae_int8x8*)p_vec_2);
    AE_LA8X8_IP(d_vec3, valign_vec_3, (ae_int8x8*)p_vec_3);

    AE_MULAO8X16(d_out0, d_mat0, d_mat0n, d_vec0);
    AE_MULAO8X16(d_out1, d_mat1, d_mat1n, d_vec0);
    AE_MULAO8X16(d_out2, d_mat0, d_mat0n, d_vec1);
    AE_MULAO8X16(d_out3, d_mat1, d_mat1n, d_vec1);
    AE_MULAO8X16(d_out4, d_mat0, d_mat0n, d_vec2);
    AE_MULAO8X16(d_out5, d_mat1, d_mat1n, d_vec2);
    AE_MULAO8X16(d_out6, d_mat0, d_mat0n, d_vec3);
    AE_MULAO8X16(d_out7, d_mat1, d_mat1n, d_vec3);
  }
  if(cols1 % 8)
  {
    AE_L16X4_XC(d_mat0, pmat0, 8);
    AE_L16X4_XC(d_mat1, pmat1, 8);

    AE_L8X4S_IP(d_vec0_16, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1_16, p_vec_1, 4);
    AE_L8X4S_IP(d_vec2_16, p_vec_2, 4);
    AE_L8X4S_IP(d_vec3_16, p_vec_3, 4);

    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0_16);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0_16);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec1_16);
    AE_MULAAAAQ16(d_out3, d_mat1, d_vec1_16);
    AE_MULAAAAQ16(d_out4, d_mat0, d_vec2_16);
    AE_MULAAAAQ16(d_out5, d_mat1, d_vec2_16);
    AE_MULAAAAQ16(d_out6, d_mat0, d_vec3_16);
    AE_MULAAAAQ16(d_out7, d_mat1, d_vec3_16);
  }
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;
  *out_4_4 = d_out4;
  *out_5_5 = d_out5;
  *out_6_6 = d_out6;
  *out_7_7 = d_out7;
#else  
  int c_itr = 0;
  ae_int16x4 d_vec0, d_vec1, d_vec2, d_vec3;
  ae_int16x4 d_mat0, d_mat1;
  ae_int64 d_out0, d_out1, d_out2, d_out3, d_out4, d_out5, d_out6, d_out7;
  ae_int16x4 *pmat0 = (ae_int16x4*)p_mat_0;
  ae_int16x4 *pmat1 = (ae_int16x4*)p_mat_1;
  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;
  d_out4 = *out_4_4;
  d_out5 = *out_5_5;
  d_out6 = *out_6_6;
  d_out7 = *out_7_7;
#if !XCHAL_HAVE_HIFI1 
  d_out0 = AE_SLAI64(d_out0, 8);
  d_out1 = AE_SLAI64(d_out1, 8);
  d_out2 = AE_SLAI64(d_out2, 8);
  d_out3 = AE_SLAI64(d_out3, 8);
  d_out4 = AE_SLAI64(d_out4, 8);
  d_out5 = AE_SLAI64(d_out5, 8);
  d_out6 = AE_SLAI64(d_out6, 8);
  d_out7 = AE_SLAI64(d_out7, 8);
#endif
  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L16X4_XC(d_mat0, pmat0, 8);
    AE_L16X4_XC(d_mat1, pmat1, 8);
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_vec0, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1, p_vec_1, 4);
    AE_L8X4S_IP(d_vec2, p_vec_2, 4);
    AE_L8X4S_IP(d_vec3, p_vec_3, 4);
#else
    AE_L8X4F_IP(d_vec0, p_vec_0, 4);
    AE_L8X4F_IP(d_vec1, p_vec_1, 4);
    AE_L8X4F_IP(d_vec2, p_vec_2, 4);
    AE_L8X4F_IP(d_vec3, p_vec_3, 4);
#endif
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out3, d_mat1, d_vec1);
    AE_MULAAAAQ16(d_out4, d_mat0, d_vec2);
    AE_MULAAAAQ16(d_out5, d_mat1, d_vec2);
    AE_MULAAAAQ16(d_out6, d_mat0, d_vec3);
    AE_MULAAAAQ16(d_out7, d_mat1, d_vec3);
  }
#if XCHAL_HAVE_HIFI1
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;
  *out_4_4 = d_out4;
  *out_5_5 = d_out5;
  *out_6_6 = d_out6;
  *out_7_7 = d_out7;
#else
  *out_0_0 = AE_SRAI64(d_out0, 8);
  *out_1_1 = AE_SRAI64(d_out1, 8);
  *out_2_2 = AE_SRAI64(d_out2, 8);
  *out_3_3 = AE_SRAI64(d_out3, 8);
  *out_4_4 = AE_SRAI64(d_out4, 8);
  *out_5_5 = AE_SRAI64(d_out5, 8);
  *out_6_6 = AE_SRAI64(d_out6, 8);
  *out_7_7 = AE_SRAI64(d_out7, 8);
#endif
#endif
#endif /* XCHAL_HAVE_HIFI3Z */
#endif /* XCHAL_HAVE_HIFI4  */
}

WORD32 xa_nn_matXvec_sym8sxsym16s_sym16s_circ(
    WORD16 * __restrict__ p_out,
    WORD16 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD64 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_stride,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift)
{
  int out_stride = out_row_offset<<1;
  int out_offset = out_col_offset;
  int m_itr, vec_itr;
  row_stride1 = row_stride1<<1;

  ae_int64 zeros[4] = {0, 0, 0, 0};
  if(p_mat1 && p_vec1 && (((unsigned int)p_mat1&0x7)==0) && (((unsigned int)p_vec1&0x3)==0) && (((unsigned int)p_bias&0x7) == 0) &&
      ((cols1&0x3)==0) && ((vec_stride&0x3)==0) && ((row_stride1&0x7)==0))
  {
    ae_int64 *pbias = (ae_int64*)p_bias;
    if(p_bias == NULL){
      pbias = zeros;
    }
    for(vec_itr = 0; vec_itr<(vec_count&(~0x3)); vec_itr+=4)
    {
      ae_int32x2 acc_row0_vec0, acc_row0_vec1, acc_row0_vec2, acc_row0_vec3;
      ae_int64 acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8;
      ae_int16x4 d1, d2;

      WORD16* p_dst0 = (WORD16*)p_out + (vec_itr+0) * out_offset;
      WORD16* p_dst1 = (WORD16*)p_out + (vec_itr+1) * out_offset;
      WORD16* p_dst2 = (WORD16*)p_out + (vec_itr+2) * out_offset;
      WORD16* p_dst3 = (WORD16*)p_out + (vec_itr+3) * out_offset;
      WORD8* p_vec_0 = (WORD8*)(p_vec1 + (vec_itr+0) * vec_stride);
      WORD8* p_vec_1 = (WORD8*)(p_vec1 + (vec_itr+1) * vec_stride);
      WORD8* p_vec_2 = (WORD8*)(p_vec1 + (vec_itr+2) * vec_stride);
      WORD8* p_vec_3 = (WORD8*)(p_vec1 + (vec_itr+3) * vec_stride);

#if XCHAL_HAVE_HIFI1S
      WORD32 out_shift_0 = 15 - p_out_shift[vec_itr+0];
      WORD32 out_shift_1 = 15 - p_out_shift[vec_itr+1];
      WORD32 out_shift_2 = 15 - p_out_shift[vec_itr+2];
      WORD32 out_shift_3 = 15 - p_out_shift[vec_itr+3];

      out_shift_0 = ( out_shift_0  << 16 ) | out_shift_0;
      out_shift_1 = ( out_shift_1  << 16 ) | out_shift_1;
      out_shift_2 = ( out_shift_2  << 16 ) | out_shift_2;
      out_shift_3 = ( out_shift_3  << 16 ) | out_shift_3;
#else
      WORD32 out_shift_0 = p_out_shift[vec_itr+0];
      WORD32 out_shift_1 = p_out_shift[vec_itr+1];
      WORD32 out_shift_2 = p_out_shift[vec_itr+2];
      WORD32 out_shift_3 = p_out_shift[vec_itr+3];
#endif
      for (m_itr=0; m_itr < (rows&(~0x1)); m_itr+=2)
      {
        acc1 = acc2 = AE_L64_I(pbias, 0);
        acc3 = acc4 = AE_L64_I(pbias, 8);
        acc5 = acc6 = AE_L64_I(pbias, 16);
        acc7 = acc8 = AE_L64_I(pbias, 24);
        WORD16 *p_mat1_0 = (WORD16*)p_mat1;
        WORD16 *p_mat1_1 = (WORD16*)p_mat1;

        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (m_itr+0) * row_stride1);
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1);
        _xa_nn_dot_product_2row_4vec_mat_vecs_4bytes_aligned
          (&acc1
           ,&acc2
           ,&acc3
           ,&acc4
           ,&acc5
           ,&acc6
           ,&acc7
           ,&acc8
           ,p_mat1_0
           ,p_mat1_1
           ,p_vec_0
           ,p_vec_1
           ,p_vec_2
           ,p_vec_3
           ,cols1
          );
        acc_row0_vec0 = MultiplyByQuantizedMultiplier_x2_opt(acc1, acc2, p_out_multiplier[vec_itr+0], out_shift_0);
        acc_row0_vec1 = MultiplyByQuantizedMultiplier_x2_opt(acc3, acc4, p_out_multiplier[vec_itr+1], out_shift_1);
        acc_row0_vec2 = MultiplyByQuantizedMultiplier_x2_opt(acc5, acc6, p_out_multiplier[vec_itr+2], out_shift_2);
        acc_row0_vec3 = MultiplyByQuantizedMultiplier_x2_opt(acc7, acc8, p_out_multiplier[vec_itr+3], out_shift_3);
        d1 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec1);
        d2 = AE_SAT16X4(acc_row0_vec2, acc_row0_vec3);
        AE_S16_0_XP(AE_SEL16_6543(d1, d1), (ae_int16*)p_dst0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d1, d1), (ae_int16*)p_dst0, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d1, d1), (ae_int16*)p_dst1, out_stride);
        AE_S16_0_XP(                   d1, (ae_int16*)p_dst1, out_stride);
        AE_S16_0_XP(AE_SEL16_6543(d2, d2), (ae_int16*)p_dst2, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d2, d2), (ae_int16*)p_dst2, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d2, d2), (ae_int16*)p_dst3, out_stride);
        AE_S16_0_XP(                   d2, (ae_int16*)p_dst3, out_stride);
      }
      if(rows&0x1)
      {
        acc1 = AE_L64_I(pbias, 0);
        acc3 = AE_L64_I(pbias, 8);
        acc5 = AE_L64_I(pbias, 16);
        acc7 = AE_L64_I(pbias, 24);

        WORD16 *p_mat1_0 = (WORD16*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (m_itr+0) * row_stride1);
        _xa_nn_dot_product_1row_4vec_mat_vecs_4bytes_aligned
          (&acc1
           ,&acc3
           ,&acc5
           ,&acc7
           ,p_mat1_0
           ,p_vec_0
           ,p_vec_1
           ,p_vec_2
           ,p_vec_3
           ,cols1
          );
        acc_row0_vec0 = AE_SEL32_LL(MultiplyByQuantizedMultiplier_opt(acc1, p_out_multiplier[vec_itr+0], out_shift_0), MultiplyByQuantizedMultiplier_opt(acc3, p_out_multiplier[vec_itr+1], out_shift_1));
        acc_row0_vec1 = AE_SEL32_LL(MultiplyByQuantizedMultiplier_opt(acc5, p_out_multiplier[vec_itr+2], out_shift_2), MultiplyByQuantizedMultiplier_opt(acc7, p_out_multiplier[vec_itr+3], out_shift_3));
        d1 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec1);
        AE_S16_0_XP(AE_SEL16_6543(d1, d1), (ae_int16*)p_dst0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d1, d1), (ae_int16*)p_dst1, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d1, d1), (ae_int16*)p_dst2, out_stride);
        AE_S16_0_XP(                   d1, (ae_int16*)p_dst3, out_stride);
      }
      /* dummy load, just to increment the pointer */
      if(p_bias != NULL){
        AE_L64_IP(acc1, pbias, 32);
      }
    }
    if(vec_count&0x2)
    {
      ae_int32x2 acc_row0_vec0, acc_row0_vec1;
      ae_int64 acc1, acc2, acc3, acc4;
      ae_int16x4 d1;
      WORD16* p_dst0 = (WORD16*)p_out + (vec_itr+0) * out_offset;
      WORD16* p_dst1 = (WORD16*)p_out + (vec_itr+1) * out_offset;
      WORD8* p_vec_0 = (WORD8*)(p_vec1 + (vec_itr+0) * vec_stride);
      WORD8* p_vec_1 = (WORD8*)(p_vec1 + (vec_itr+1) * vec_stride);

#if XCHAL_HAVE_HIFI1S
      WORD32 out_shift_0 = 15 - p_out_shift[vec_itr+0];
      WORD32 out_shift_1 = 15 - p_out_shift[vec_itr+1];
      out_shift_0 = ( out_shift_0  << 16 ) | out_shift_0;
      out_shift_1 = ( out_shift_1  << 16 ) | out_shift_1;
#else      
      WORD32 out_shift_0 = p_out_shift[vec_itr+0];
      WORD32 out_shift_1 = p_out_shift[vec_itr+1];
#endif      

      for (m_itr=0; m_itr < (rows&(~0x1)); m_itr+=2)
      {
        acc1 = acc2 = AE_L64_I(pbias, 0);
        acc3 = acc4 = AE_L64_I(pbias, 8);
        WORD16 *p_mat1_0 = (WORD16*)p_mat1;
        WORD16 *p_mat1_1 = (WORD16*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (m_itr+0) * row_stride1);
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1);
        _xa_nn_dot_product_2row_2vec_mat_vecs_4bytes_aligned
          (&acc1
           ,&acc2
           ,&acc3
           ,&acc4
           ,p_mat1_0
           ,p_mat1_1
           ,p_vec_0
           ,p_vec_1
           ,cols1
          );
        acc_row0_vec0 = MultiplyByQuantizedMultiplier_x2_opt(acc1, acc2, p_out_multiplier[vec_itr+0], out_shift_0);
        acc_row0_vec1 = MultiplyByQuantizedMultiplier_x2_opt(acc3, acc4, p_out_multiplier[vec_itr+1], out_shift_1);
        d1 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec1);
        AE_S16_0_XP(AE_SEL16_6543(d1, d1), (ae_int16*)p_dst0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d1, d1), (ae_int16*)p_dst0, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d1, d1), (ae_int16*)p_dst1, out_stride);
        AE_S16_0_XP(                   d1, (ae_int16*)p_dst1, out_stride);
      }
      if(rows&0x1)
      {
        acc1 = AE_L64_I(pbias, 0);
        acc3 = AE_L64_I(pbias, 8);
        WORD16 *p_mat1_0 = (WORD16*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1);
        _xa_nn_dot_product_1row_2vec_mat_vecs_4bytes_aligned
          (&acc1
           ,&acc3
           ,p_mat1_0
           ,p_vec_0
           ,p_vec_1
           ,cols1
          );
        acc_row0_vec0 = MultiplyByQuantizedMultiplier_opt(acc1, p_out_multiplier[vec_itr+0], out_shift_0);
        acc_row0_vec1 = MultiplyByQuantizedMultiplier_opt(acc3, p_out_multiplier[vec_itr+1], out_shift_1);
        d1 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec1);
        AE_S16_0_XP(AE_SEL16_5432(d1, d1), (ae_int16*)p_dst0, out_stride);
        AE_S16_0_XP(                   d1, (ae_int16*)p_dst1, out_stride);
      }
      /* dummy load, just to increment the pointer */
      if(p_bias != NULL){
        AE_L64_IP(acc1, pbias, 16);
      }
      vec_itr += 2;
    }
    if(vec_count&0x1)
    {
      WORD16* p_dst0 = (WORD16*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_vec_0  = (WORD8*)(p_vec1 + vec_itr * vec_stride);
      ae_int32x2 acc_row0_vec0;
      ae_int64 acc1, acc2;
      ae_int16x4 d1;

#if XCHAL_HAVE_HIFI1S
      WORD32 out_shift_0 = 15 - p_out_shift[vec_itr+0];
      out_shift_0 = ( out_shift_0  << 16 ) | out_shift_0;
#else      
      WORD32 out_shift_0 = p_out_shift[vec_itr+0];
#endif      

      for (m_itr=0; m_itr < (rows&(~0x1)); m_itr+=2)
      {
        if(p_bias != NULL){
          acc1 = acc2 = AE_L64_I(pbias, 0);
        }
        else{
          acc1 = acc2 = 0;
        }
        WORD16 *p_mat1_0 = (WORD16*)p_mat1;
        WORD16 *p_mat1_1 = (WORD16*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (m_itr+0) * row_stride1);
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1);
        _xa_nn_dot_product_2row_1vec_mat_vecs_4bytes_aligned
          (&acc1
           ,&acc2
           ,p_mat1_0
           ,p_mat1_1
           ,p_vec_0
           ,cols1
          );
        acc_row0_vec0 = MultiplyByQuantizedMultiplier_x2_opt(acc1, acc2, p_out_multiplier[vec_itr], out_shift_0);
        d1 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec0);
        AE_S16_0_XP(AE_SEL16_4321(d1, d1), (ae_int16*)p_dst0, out_stride);
        AE_S16_0_XP(                   d1, (ae_int16*)p_dst0, out_stride);
      }
      if(rows&0x1)
      {
        acc1 = AE_L64_I(pbias, 0);
        WORD16 *p_mat1_0 = (WORD16*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1);
        _xa_nn_dot_product_1row_1vec_mat_vecs_4bytes_aligned
          (&acc1
           ,p_mat1_0
           ,p_vec_0
           ,cols1
          );
        acc_row0_vec0 = MultiplyByQuantizedMultiplier_opt(acc1, p_out_multiplier[vec_itr], out_shift_0);
        d1 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec0);
        AE_S16_0_XP(d1, (ae_int16*)p_dst0, out_stride);
      }
      /* dummy load, just to increment the pointer */
      if(p_bias != NULL){
        AE_L64_IP(acc1, pbias, 8);
      }
    }
  }
  else
    return -1;

  return 0;
}
