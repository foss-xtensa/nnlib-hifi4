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
#include "xa_nnlib_common_macros.h"

#if XCHAL_HAVE_HIFI1S
static inline ae_int32x2 MultiplyByQuantizedMultiplier_ref(ae_int64 d_x,
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
#else // XCHAL_HAVE_HIFI1S
static inline ae_int32x2 MultiplyByQuantizedMultiplier_ref(ae_int64 d_x,
                                             int32_t quantized_multiplier,
                                             int shift){
  ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
  ae_int32x2 d_red_mul32 = AE_SEXT32X2D16_32(d_red_mul16); //upper 32
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
#endif // XCHAL_HAVE_HIFI1S

#if XCHAL_HAVE_HIFI1
#if XCHAL_HAVE_HIFI1S
static inline void _xa_nn_dot_product_4_rows_1_vecs_unaligned
( ae_int64* out_0_0
 ,ae_int64* out_1_1
 ,ae_int64* out_2_2
 ,ae_int64* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_mat_2
 ,WORD8*      p_mat_3
 ,WORD16*      p_vec_0
 ,WORD32      cols1)
{
  ae_int16x4 *pvec0 = (ae_int16x4*)p_vec_0;
  ae_int8x8 d_mat0_0, d_mat1_0, d_mat2_0, d_mat3_0;
  ae_int16x4 d_vec0, d_vec1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;

  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;

  ae_valign align_m0, align_m1, align_m2, align_m3;
  align_m0 = AE_LA64_PP((ae_int8x8 *)p_mat_0);
  align_m1 = AE_LA64_PP((ae_int8x8 *)p_mat_1);
  align_m2 = AE_LA64_PP((ae_int8x8 *)p_mat_2);
  align_m3 = AE_LA64_PP((ae_int8x8 *)p_mat_3);
  ae_valign align_v0;
  align_v0 = AE_LA64_PP((ae_int16x4 *)pvec0);

  int c_itr = 0;
    
  for(;c_itr<(cols1>>3); c_itr++)
  {
    AE_LA16X4_IP(d_vec0, align_v0, (ae_int16x4 *)pvec0);
    AE_LA16X4_IP(d_vec1, align_v0, (ae_int16x4 *)pvec0);
	
    AE_LA8X8_IP(d_mat0_0, align_m0, (ae_int8x8 *)p_mat_0);
    AE_LA8X8_IP(d_mat1_0, align_m1, (ae_int8x8 *)p_mat_1);
    AE_LA8X8_IP(d_mat2_0, align_m2, (ae_int8x8 *)p_mat_2);
    AE_LA8X8_IP(d_mat3_0, align_m3, (ae_int8x8 *)p_mat_3);

    AE_MULAO8X16(d_out0, d_vec0, d_vec1, d_mat0_0);
    AE_MULAO8X16(d_out1, d_vec0, d_vec1, d_mat1_0);
    AE_MULAO8X16(d_out2, d_vec0, d_vec1, d_mat2_0);
    AE_MULAO8X16(d_out3, d_vec0, d_vec1, d_mat3_0);
  }
  int reminder = cols1&0x07;
	
  if(reminder)
  {
    AE_LA16X4_IP(d_vec0, align_v0, pvec0);	
    AE_LA16X4_IP(d_vec1, align_v0, pvec0);
	
    AE_LAV8X8_XP(d_mat0_0, align_m0, (ae_int8x8 *)p_mat_0, reminder );
    AE_LAV8X8_XP(d_mat1_0, align_m1, (ae_int8x8 *)p_mat_1, reminder );
    AE_LAV8X8_XP(d_mat2_0, align_m2, (ae_int8x8 *)p_mat_2, reminder );
    AE_LAV8X8_XP(d_mat3_0, align_m3, (ae_int8x8 *)p_mat_3, reminder );

    AE_MULAO8X16(d_out0, d_vec0, d_vec1, d_mat0_0);
    AE_MULAO8X16(d_out1, d_vec0, d_vec1, d_mat1_0);
    AE_MULAO8X16(d_out2, d_vec0, d_vec1, d_mat2_0);
    AE_MULAO8X16(d_out3, d_vec0, d_vec1, d_mat3_0);
  }
  
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;
}
#else // XCHAL_HAVE_HIFI1S
static inline void _xa_nn_dot_product_4_rows_1_vecs_unaligned
( ae_int64* out_0_0
 ,ae_int64* out_1_1
 ,ae_int64* out_2_2
 ,ae_int64* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_mat_2
 ,WORD8*      p_mat_3
 ,WORD16*      p_vec_0
 ,WORD32      cols1)
{
  ae_int16x4 *pvec0 = (ae_int16x4*)p_vec_0;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3, d_vec0;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  
  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;

  ALIGN_REGISTER_TYPE align_d0, align_d1, align_d2, align_d3;
  PRIME_8X4F(p_mat_0, align_d0);
  PRIME_8X4F(p_mat_1, align_d1);
  PRIME_8X4F(p_mat_2, align_d2);
  PRIME_8X4F(p_mat_3, align_d3);

  ae_valign align_vec;
  align_vec = AE_LA64_PP(pvec0);

  int c_itr = 0;
  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_LA16X4_IP(d_vec0, align_vec, pvec0);
    AE_LA8X4S_IP(d_mat0, align_d0, p_mat_0);
    AE_LA8X4S_IP(d_mat1, align_d1, p_mat_1);
    AE_LA8X4S_IP(d_mat2, align_d2, p_mat_2);
    AE_LA8X4S_IP(d_mat3, align_d3, p_mat_3);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat2, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat3, d_vec0);
  }
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
  {
    AE_LAV16X4_XP(d_vec0, align_vec, (ae_int16x4 *)pvec0, ((cols1&0x03)<<1));
    AE_LAV8X4S_XP(d_mat0, align_d0, (ae_int8x4 *)p_mat_0, (cols1&0x03));
    AE_LAV8X4S_XP(d_mat1, align_d1, (ae_int8x4 *)p_mat_1, (cols1&0x03));
    AE_LAV8X4S_XP(d_mat2, align_d2, (ae_int8x4 *)p_mat_2, (cols1&0x03));
    AE_LAV8X4S_XP(d_mat3, align_d3, (ae_int8x4 *)p_mat_3, (cols1&0x03));
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat2, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat3, d_vec0);
  }

  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;
#else //( XCHAL_HW_VERSION >= RI9_HWVERSION )
  int64_t out0 = d_out0;
  int64_t out1 = d_out1;
  int64_t out2 = d_out2;
  int64_t out3 = d_out3;
  p_vec_0 = (WORD16 *)pvec0;

  for(c_itr=0; c_itr<(cols1&0x03); c_itr++)
  {
    out0 += (*p_mat_0)*(*p_vec_0);
    out1 += (*p_mat_1)*(*p_vec_0);
    out2 += (*p_mat_2)*(*p_vec_0);
    out3 += (*p_mat_3)*(*p_vec_0);
    p_mat_0++;
    p_mat_1++;
    p_mat_2++;
    p_mat_3++;
    p_vec_0++;
  }
  *out_0_0 = out0;
  *out_1_1 = out1;
  *out_2_2 = out2;
  *out_3_3 = out3;
#endif //( XCHAL_HW_VERSION >= RI9_HWVERSION )
}
#endif // XCHAL_HAVE_HIFI1S
#else//XCHAL_HAVE_HIFI1
static inline void _xa_nn_dot_product_4_rows_1_vecs_unaligned
( ae_int64* out_0_0
 ,ae_int64* out_1_1
 ,ae_int64* out_2_2
 ,ae_int64* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_mat_2
 ,WORD8*      p_mat_3
 ,WORD16*      p_vec_0
 ,WORD32      cols1)
{
  ae_int16x4 *pvec0 = (ae_int16x4*)p_vec_0;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3, d_vec0;
  ae_int64 d_out0, d_out1, d_out2, d_out3;

  d_out0 = AE_SLAI64(*out_0_0, 8);
  d_out1 = AE_SLAI64(*out_1_1, 8);
  d_out2 = AE_SLAI64(*out_2_2, 8);
  d_out3 = AE_SLAI64(*out_3_3, 8);

  ALIGN_REGISTER_TYPE align_d0, align_d1, align_d2, align_d3;
  PRIME_8X4F(p_mat_0, align_d0);
  PRIME_8X4F(p_mat_1, align_d1);
  PRIME_8X4F(p_mat_2, align_d2);
  PRIME_8X4F(p_mat_3, align_d3);

  ae_valign align_vec;
  align_vec = AE_LA64_PP(pvec0);

  int c_itr = 0;
  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_LA16X4_IP(d_vec0, align_vec, pvec0);
    AE_LA8X4F_IP(d_mat0, align_d0, p_mat_0);
    AE_LA8X4F_IP(d_mat1, align_d1, p_mat_1);
    AE_LA8X4F_IP(d_mat2, align_d2, p_mat_2);
    AE_LA8X4F_IP(d_mat3, align_d3, p_mat_3);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat2, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat3, d_vec0);
  }

  int64_t out0 = AE_SRAI64(d_out0, 8);
  int64_t out1 = AE_SRAI64(d_out1, 8);
  int64_t out2 = AE_SRAI64(d_out2, 8);
  int64_t out3 = AE_SRAI64(d_out3, 8);
  p_vec_0 = (WORD16 *)pvec0;

  for(c_itr=0; c_itr<(cols1&0x03); c_itr++)
  {
    out0 += (*p_mat_0)*(*p_vec_0);
    out1 += (*p_mat_1)*(*p_vec_0);
    out2 += (*p_mat_2)*(*p_vec_0);
    out3 += (*p_mat_3)*(*p_vec_0);
    p_mat_0++;
    p_mat_1++;
    p_mat_2++;
    p_mat_3++;
    p_vec_0++;
  }
  *out_0_0 = out0;
  *out_1_1 = out1;
  *out_2_2 = out2;
  *out_3_3 = out3;
}
#endif //XCHAL_HAVE_HIFI1

#if !XCHAL_HAVE_HIFI1
static inline void _xa_nn_dot_product_4_rows_1_vecs_aligned
( ae_int64* out_0_0
 ,ae_int64* out_1_1
 ,ae_int64* out_2_2
 ,ae_int64* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_mat_2
 ,WORD8*      p_mat_3
 ,WORD16*      p_vec_0
 ,WORD32      cols1)
{
  ae_int16x4 *pvec0 = (ae_int16x4*)p_vec_0;
  ae_int16x4 d_mat0_0, d_mat1_0, d_mat2_0, d_mat3_0, d_vec0_0;
  ae_int16x4 d_mat0_1, d_mat1_1, d_mat2_1, d_mat3_1, d_vec0_1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;

  d_out0 = AE_SLAI64(*out_0_0, 8);
  d_out1 = AE_SLAI64(*out_1_1, 8);
  d_out2 = AE_SLAI64(*out_2_2, 8);
  d_out3 = AE_SLAI64(*out_3_3, 8);

  ae_valign align_vec;
  align_vec = AE_LA64_PP(pvec0);

  int c_itr = 0;
  for(;c_itr<(cols1>>3); c_itr++)
  {
    AE_LA16X4_IP(d_vec0_0, align_vec, pvec0);
    AE_LA16X4_IP(d_vec0_1, align_vec, pvec0);
    d_mat0_1 = AE_L8X4F_I(p_mat_0, 4);
    d_mat1_1 = AE_L8X4F_I(p_mat_1, 4);
    d_mat2_1 = AE_L8X4F_I(p_mat_2, 4);
    d_mat3_1 = AE_L8X4F_I(p_mat_3, 4);
    AE_L8X4F_IP(d_mat0_0, p_mat_0, 8);
    AE_L8X4F_IP(d_mat1_0, p_mat_1, 8);
    AE_L8X4F_IP(d_mat2_0, p_mat_2, 8);
    AE_L8X4F_IP(d_mat3_0, p_mat_3, 8);

    AE_MULAAAAQ16(d_out0, d_mat0_0, d_vec0_0);
    AE_MULAAAAQ16(d_out1, d_mat1_0, d_vec0_0);
    AE_MULAAAAQ16(d_out2, d_mat2_0, d_vec0_0);
    AE_MULAAAAQ16(d_out3, d_mat3_0, d_vec0_0);

    AE_MULAAAAQ16(d_out0, d_mat0_1, d_vec0_1);
    AE_MULAAAAQ16(d_out1, d_mat1_1, d_vec0_1);
    AE_MULAAAAQ16(d_out2, d_mat2_1, d_vec0_1);
    AE_MULAAAAQ16(d_out3, d_mat3_1, d_vec0_1);
  }

  int64_t out0 = AE_SRAI64(d_out0, 8);
  int64_t out1 = AE_SRAI64(d_out1, 8);
  int64_t out2 = AE_SRAI64(d_out2, 8);
  int64_t out3 = AE_SRAI64(d_out3, 8);
  p_vec_0 = (WORD16 *)pvec0;

  for(c_itr=0; c_itr<(cols1&0x07); c_itr++)
  {
    out0 += (*p_mat_0)*(*p_vec_0);
    out1 += (*p_mat_1)*(*p_vec_0);
    out2 += (*p_mat_2)*(*p_vec_0);
    out3 += (*p_mat_3)*(*p_vec_0);
    p_mat_0++;
    p_mat_1++;
    p_mat_2++;
    p_mat_3++;
    p_vec_0++;
  }
  *out_0_0 = out0;
  *out_1_1 = out1;
  *out_2_2 = out2;
  *out_3_3 = out3;
}
#else
/* vec aligned, mat not required to be aligned */
#if XCHAL_HAVE_HIFI1S
static inline void _xa_nn_dot_product_4_rows_1_vecs_aligned
( ae_int64* out_0_0
 ,ae_int64* out_1_1
 ,ae_int64* out_2_2
 ,ae_int64* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_mat_2
 ,WORD8*      p_mat_3
 ,WORD16*      p_vec_0
 ,WORD32      cols1)
{
  ae_int16x4 *pvec0 = (ae_int16x4*)p_vec_0;
  ae_int8x8 d_mat0_0, d_mat1_0, d_mat2_0, d_mat3_0;
  ae_int16x4 d_vec0, d_vec1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;

  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;

  ae_valign align_m0, align_m1, align_m2, align_m3;
  align_m0 = AE_LA64_PP((ae_int8x8 *)p_mat_0);
  align_m1 = AE_LA64_PP((ae_int8x8 *)p_mat_1);
  align_m2 = AE_LA64_PP((ae_int8x8 *)p_mat_2);
  align_m3 = AE_LA64_PP((ae_int8x8 *)p_mat_3);

  int c_itr = 0;
    
  for(;c_itr<(cols1>>3); c_itr++)
  {
    AE_L16X4_IP(d_vec0, pvec0, 8);	
    AE_L16X4_IP(d_vec1, pvec0, 8);
	
    AE_LA8X8_IP(d_mat0_0, align_m0, (ae_int8x8 *)p_mat_0);
    AE_LA8X8_IP(d_mat1_0, align_m1, (ae_int8x8 *)p_mat_1);
    AE_LA8X8_IP(d_mat2_0, align_m2, (ae_int8x8 *)p_mat_2);
    AE_LA8X8_IP(d_mat3_0, align_m3, (ae_int8x8 *)p_mat_3);

    AE_MULAO8X16(d_out0, d_vec0, d_vec1, d_mat0_0);
    AE_MULAO8X16(d_out1, d_vec0, d_vec1, d_mat1_0);
    AE_MULAO8X16(d_out2, d_vec0, d_vec1, d_mat2_0);
    AE_MULAO8X16(d_out3, d_vec0, d_vec1, d_mat3_0);
  }
  int reminder = cols1&0x07;
	
  if(reminder)
  {
    AE_L16X4_IP(d_vec0, pvec0, 8);	
    AE_L16X4_IP(d_vec1, pvec0, 8);
	
    AE_LAV8X8_XP(d_mat0_0, align_m0, (ae_int8x8 *)p_mat_0, reminder );
    AE_LAV8X8_XP(d_mat1_0, align_m1, (ae_int8x8 *)p_mat_1, reminder );
    AE_LAV8X8_XP(d_mat2_0, align_m2, (ae_int8x8 *)p_mat_2, reminder );
    AE_LAV8X8_XP(d_mat3_0, align_m3, (ae_int8x8 *)p_mat_3, reminder );

    AE_MULAO8X16(d_out0, d_vec0, d_vec1, d_mat0_0);
    AE_MULAO8X16(d_out1, d_vec0, d_vec1, d_mat1_0);
    AE_MULAO8X16(d_out2, d_vec0, d_vec1, d_mat2_0);
    AE_MULAO8X16(d_out3, d_vec0, d_vec1, d_mat3_0);
  }
  
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;
}
#else // XCHAL_HAVE_HIFI1S
static inline void _xa_nn_dot_product_4_rows_1_vecs_aligned
( ae_int64* out_0_0
 ,ae_int64* out_1_1
 ,ae_int64* out_2_2
 ,ae_int64* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_mat_2
 ,WORD8*      p_mat_3
 ,WORD16*      p_vec_0
 ,WORD32      cols1)
{
  ae_int16x4 *pvec0 = (ae_int16x4*)p_vec_0;
  ae_int16x4 d_mat0_0, d_mat1_0, d_mat2_0, d_mat3_0, d_vec0;
  ae_int64 d_out0, d_out1, d_out2, d_out3;

  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;

  ae_valign align_m0, align_m1, align_m2, align_m3;
  align_m0 = AE_LA64_PP(p_mat_0);
  align_m1 = AE_LA64_PP(p_mat_1);
  align_m2 = AE_LA64_PP(p_mat_2);
  align_m3 = AE_LA64_PP(p_mat_3);

  int c_itr = 0;
  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L16X4_IP(d_vec0, pvec0, 8);
    AE_LA8X4S_IP(d_mat0_0, align_m0, p_mat_0);
    AE_LA8X4S_IP(d_mat1_0, align_m1, p_mat_1);
    AE_LA8X4S_IP(d_mat2_0, align_m2, p_mat_2);
    AE_LA8X4S_IP(d_mat3_0, align_m3, p_mat_3);

    AE_MULAAAAQ16(d_out0, d_mat0_0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1_0, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat2_0, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat3_0, d_vec0);
  }

  int64_t out0 = d_out0;
  int64_t out1 = d_out1;
  int64_t out2 = d_out2;
  int64_t out3 = d_out3;
  p_vec_0 = (WORD16 *)pvec0;

  for(c_itr=0; c_itr<(cols1&0x03); c_itr++)
  {
    out0 += (*p_mat_0)*(*p_vec_0);
    out1 += (*p_mat_1)*(*p_vec_0);
    out2 += (*p_mat_2)*(*p_vec_0);
    out3 += (*p_mat_3)*(*p_vec_0);
    p_mat_0++;
    p_mat_1++;
    p_mat_2++;
    p_mat_3++;
    p_vec_0++;
  }
  *out_0_0 = out0;
  *out_1_1 = out1;
  *out_2_2 = out2;
  *out_3_3 = out3;
}
#endif // XCHAL_HAVE_HIFI1S
#endif

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
    (ae_int64* out_0_0
    ,const WORD8*      p_mat_0
    ,const WORD16*      p_vec_0
    ,WORD32      cols1)
{
  int c_itr = 0;
  int64_t d_out;
  d_out = *out_0_0;
  for(;c_itr<(cols1); c_itr++)
  {
    d_out+=(*p_mat_0)*(*p_vec_0);
    p_mat_0++; p_vec_0++;
  }
  *out_0_0 = d_out;
}

/*  This kernel performs the following dual mat*vec op
 *    p_out = mat1*vec1 +
 *            mat2*vec2 +
 *            p_bias
 *
 *  If p_mat2 is NULL, then the second matrix-vec multiply op is not executed, and 
 *    p_out = mat1*vec1+
 *            p_bias
 */
WORD32 xa_nn_matXvec_sym8sxsym16s_sym16s(
    WORD16 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_mat2,
    const WORD16 * __restrict__ p_vec1,
    const WORD16 * __restrict__ p_vec2,
    const WORD64 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    WORD32 out_multiplier,
    WORD32 out_shift)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, 2*sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 15), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
  }

  int m_itr=0;

#if XCHAL_HAVE_HIFI1S  
  out_shift = 15 - out_shift;
  out_shift = (out_shift << 16) | (out_shift); 
#endif
  int bias_flag = 0;
  if(p_bias != NULL){
    bias_flag = 1;
  }
  
  int align_flag1=0, align_flag2=0;
#if XCHAL_HAVE_HIFI1
  /* HiFi1 core has unaligned support for 8-bit load, hence only vec needs to be aligned */
  if( (unsigned)p_vec1 % 4 == 0) {
    align_flag1 = 1;
  }
  if( (unsigned)p_vec2 % 4 == 0) {
    align_flag2 = 1;
  }
#else
  /* No 8-bit unaligned support, require each matrix row to be aligned */
  if( ((unsigned)p_mat1 % 4 == 0) && (row_stride1%4 == 0) ){
    align_flag1 = 1;
  }
  if( ((unsigned)p_mat2 % 4 == 0) && (row_stride2%4 == 0) ){
    align_flag2 = 1;
  }
#endif
 
  ae_valign align_out;
  align_out = AE_ZALIGN64();

  ae_int16x4* pout_16x4 = (ae_int16x4*)p_out;

  m_itr = 0;
  for (; m_itr < (rows&~0x03); m_itr+=4) {
    ae_int64 acc_row0_vec0 = 0;
    ae_int64 acc_row1_vec0 = 0;
    ae_int64 acc_row2_vec0 = 0;
    ae_int64 acc_row3_vec0 = 0;

    WORD8 *p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
    WORD8 *p_mat1_1 = (WORD8 *)(p_mat1+((m_itr+1) * row_stride1));
    WORD8 *p_mat1_2 = (WORD8 *)(p_mat1+((m_itr+2) * row_stride1));
    WORD8 *p_mat1_3 = (WORD8 *)(p_mat1+((m_itr+3) * row_stride1));
    WORD16 *p_vec1_0 = (WORD16 *)(p_vec1);

    if(bias_flag){
      acc_row0_vec0 = p_bias[m_itr+0];
      acc_row1_vec0 = p_bias[m_itr+1];
      acc_row2_vec0 = p_bias[m_itr+2];
      acc_row3_vec0 = p_bias[m_itr+3];
    }

    if(align_flag1) {
      _xa_nn_dot_product_4_rows_1_vecs_aligned
       (&acc_row0_vec0, &acc_row1_vec0, &acc_row2_vec0, &acc_row3_vec0
       ,p_mat1_0, p_mat1_1, p_mat1_2, p_mat1_3
       ,p_vec1_0, cols1
      );
    } else {
      _xa_nn_dot_product_4_rows_1_vecs_unaligned
       (&acc_row0_vec0, &acc_row1_vec0, &acc_row2_vec0, &acc_row3_vec0
       ,p_mat1_0, p_mat1_1, p_mat1_2, p_mat1_3
       ,p_vec1_0, cols1
       );
    }

    if(p_mat2 != NULL){
      WORD8 *p_mat2_0 = (WORD8 *)(p_mat2+(m_itr * row_stride2));
      WORD8 *p_mat2_1 = (WORD8 *)(p_mat2+((m_itr+1) * row_stride2));
      WORD8 *p_mat2_2 = (WORD8 *)(p_mat2+((m_itr+2) * row_stride2));
      WORD8 *p_mat2_3 = (WORD8 *)(p_mat2+((m_itr+3) * row_stride2));
      WORD16 *p_vec2_0 = (WORD16 *)(p_vec2);

      if(align_flag2) {
        _xa_nn_dot_product_4_rows_1_vecs_aligned
         (&acc_row0_vec0, &acc_row1_vec0, &acc_row2_vec0, &acc_row3_vec0
         ,p_mat2_0, p_mat2_1, p_mat2_2, p_mat2_3
         ,p_vec2_0, cols2
         );
      } else {
        _xa_nn_dot_product_4_rows_1_vecs_unaligned
         (&acc_row0_vec0, &acc_row1_vec0, &acc_row2_vec0, &acc_row3_vec0
         ,p_mat2_0, p_mat2_1, p_mat2_2, p_mat2_3
         ,p_vec2_0, cols2
         );
      }
    }
    ae_int32x2 res0 = MultiplyByQuantizedMultiplier_x2_opt(acc_row0_vec0, acc_row1_vec0, out_multiplier, out_shift);
    ae_int32x2 res2 = MultiplyByQuantizedMultiplier_x2_opt(acc_row2_vec0, acc_row3_vec0, out_multiplier, out_shift);
    ae_int16x4 d1 = AE_SAT16X4(res0, res2);
    AE_SA16X4_IP(d1, align_out, pout_16x4);
  }
  AE_SA64POS_FP(align_out, pout_16x4);
  p_out = (WORD16*)pout_16x4;

  for (; m_itr < rows; m_itr++) {
    ae_int64 acc_row0_vec0 = 0;

    WORD8 *p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
    WORD16 *p_vec1_0 = (WORD16 *)(p_vec1);


    if(bias_flag){
      acc_row0_vec0 = p_bias[m_itr];
    }
    _xa_nn_dot_product_1_rows_1_vecs_unaligned
      (&acc_row0_vec0
       ,p_mat1_0
       ,p_vec1_0
       ,cols1
      );

    if(p_mat2 != NULL){
      WORD8 *p_mat2_0 = (WORD8 *)(p_mat2+(m_itr * row_stride2));
      WORD16 *p_vec2_0 = (WORD16 *)(p_vec2);

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,p_mat2_0
         ,p_vec2_0
         ,cols2
        );
    }

    ae_int32x2 res = MultiplyByQuantizedMultiplier_ref(acc_row0_vec0, out_multiplier, out_shift);
    ae_int16x4 d1 = AE_SAT16X4(res, res);
    
    *p_out++ = (WORD16)AE_MOVAD16_3(d1);
  }

  return 0;
}
