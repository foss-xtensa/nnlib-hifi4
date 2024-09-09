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

static inline void _xa_nn_dot_product_2_rows_2_vecs_unaligned
(ae_int64* out_0_0, ae_int64 *out_0_1, ae_int64* out_1_0, ae_int64* out_1_1
 ,WORD16*   p_mat_0, WORD16*   p_mat_1
 ,WORD16*  p_vec_0, WORD16*  p_vec_1 
 ,WORD32      cols1)
{
  WORD16 *pvec0 = p_vec_0;
  WORD16 *pvec1 = p_vec_1;
  ae_int16x4 d_vec0, d_vec1;
  ae_int16x4 d_mat0, d_mat1;
  int c_itr = 0;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  d_out0 = *out_0_0;
  d_out1 = *out_0_1;
  d_out2 = *out_1_0;
  d_out3 = *out_1_1;

  ae_valign align_v0 = AE_LA64_PP(pvec0);
  ae_valign align_v1 = AE_LA64_PP(pvec1);
  ae_valign align_m0 = AE_LA64_PP(p_mat_0);
  ae_valign align_m1 = AE_LA64_PP(p_mat_1);

  for(;c_itr<(cols1&~0x03); c_itr+=4)
  {
    AE_LA16X4_IP(d_vec0, align_v0, (ae_int16x4*)pvec0);	
    AE_LA16X4_IP(d_vec1, align_v1, (ae_int16x4*)pvec1);
    AE_LA16X4_IP(d_mat0, align_m0, (ae_int16x4 *)p_mat_0);
    AE_LA16X4_IP(d_mat1, align_m1, (ae_int16x4 *)p_mat_1);
	
    AE_MULAAAAQ16(d_out0, d_vec0, d_mat0);
    AE_MULAAAAQ16(d_out1, d_vec0, d_mat1);
    AE_MULAAAAQ16(d_out2, d_vec1, d_mat0);
    AE_MULAAAAQ16(d_out3, d_vec1, d_mat1);
  } 

  int off = cols1&0x03;
#if XCHAL_HAVE_HIFI1 && ( XCHAL_HW_VERSION >= RI9_HWVERSION )
  if(off!=0)
  {
    AE_LA16X4_IP(d_vec0, align_v0, (ae_int16x4 *)pvec0);	
    AE_LA16X4_IP(d_vec1, align_v1, (ae_int16x4 *)pvec1);
    AE_LAV16X4_XP(d_mat0, align_m0, (ae_int16x4 *)p_mat_0, off<<1);
    AE_LAV16X4_XP(d_mat1, align_m1, (ae_int16x4 *)p_mat_1, off<<1);
	
    AE_MULAAAAQ16(d_out0, d_vec0, d_mat0);
    AE_MULAAAAQ16(d_out1, d_vec0, d_mat1);
    AE_MULAAAAQ16(d_out2, d_vec1, d_mat0);
    AE_MULAAAAQ16(d_out3, d_vec1, d_mat1);
  }
#else
  for(c_itr=0;c_itr<off;c_itr++){
    AE_L16_IP(d_vec0, (ae_int16 *)pvec0, 2);
    AE_L16_IP(d_vec1, (ae_int16 *)pvec1, 2);
    AE_L16_IP(d_mat0, (ae_int16 *)p_mat_0, 2);
    AE_L16_IP(d_mat1, (ae_int16 *)p_mat_1, 2);
    AE_MULA16_00(d_out0, d_mat0, d_vec0);
    AE_MULA16_00(d_out1, d_mat1, d_vec0);
    AE_MULA16_00(d_out2, d_mat0, d_vec1);
    AE_MULA16_00(d_out3, d_mat1, d_vec1);
  }
#endif
  *out_0_0 = d_out0;
  *out_0_1 = d_out1;
  *out_1_0 = d_out2;
  *out_1_1 = d_out3;
}

static inline void _xa_nn_dot_product_2_rows_1_vecs_unaligned
(ae_int64* out_0_0, ae_int64 *out_0_1
 ,WORD16*   p_mat_0, WORD16*   p_mat_1
 ,WORD16*  p_vec_0 
 ,WORD32      cols1)
{
  WORD16 *pvec0 = p_vec_0;
  ae_int16x4 d_vec0;
  ae_int16x4 d_mat0, d_mat1;
  int c_itr = 0;
  ae_int64 d_out0, d_out1;
  d_out0 = *out_0_0;
  d_out1 = *out_0_1;

  ae_valign align_v0 = AE_LA64_PP(pvec0);
  ae_valign align_m0 = AE_LA64_PP(p_mat_0);
  ae_valign align_m1 = AE_LA64_PP(p_mat_1);

  for(;c_itr<(cols1&~0x03); c_itr+=4)
  {
    AE_LA16X4_IP(d_vec0, align_v0, (ae_int16x4 *)pvec0);	
    AE_LA16X4_IP(d_mat0, align_m0, (ae_int16x4 *)p_mat_0);
    AE_LA16X4_IP(d_mat1, align_m1, (ae_int16x4 *)p_mat_1);
	
    AE_MULAAAAQ16(d_out0, d_vec0, d_mat0);
    AE_MULAAAAQ16(d_out1, d_vec0, d_mat1);
  } 

  int off = cols1&0x03;
#if XCHAL_HAVE_HIFI1 && ( XCHAL_HW_VERSION >= RI9_HWVERSION )
  if(off!=0)
  {
    AE_LA16X4_IP(d_vec0, align_v0, (ae_int16x4 *)pvec0);	
    AE_LAV16X4_XP(d_mat0, align_m0, (ae_int16x4 *)p_mat_0, off<<1);
    AE_LAV16X4_XP(d_mat1, align_m1, (ae_int16x4 *)p_mat_1, off<<1);
	
    AE_MULAAAAQ16(d_out0, d_vec0, d_mat0);
    AE_MULAAAAQ16(d_out1, d_vec0, d_mat1);
  }
#else
  for(c_itr=0;c_itr<off;c_itr++){
    AE_L16_IP(d_vec0, (ae_int16 *)pvec0, 2);
    AE_L16_IP(d_mat0, (ae_int16 *)p_mat_0, 2);
    AE_L16_IP(d_mat1, (ae_int16 *)p_mat_1, 2);
    AE_MULA16_00(d_out0, d_mat0, d_vec0);
    AE_MULA16_00(d_out1, d_mat1, d_vec0);
  }
#endif
  *out_0_0 = d_out0;
  *out_0_1 = d_out1;
}

static inline void _xa_nn_dot_product_4_rows_2_vecs_aligned
(ae_int64* out_0_0
 ,ae_int64* out_1_1
 ,ae_int64* out_2_2
 ,ae_int64* out_3_3
 ,ae_int64* out_4_4
 ,ae_int64* out_5_5
 ,ae_int64* out_6_6
 ,ae_int64* out_7_7
 ,WORD16*      p_mat_0
 ,WORD16*      p_mat_1
 ,WORD16*      p_mat_2
 ,WORD16*      p_mat_3
 ,WORD16*      p_vec_0
 ,WORD16*      p_vec_1
 ,WORD32      cols1)
{
  WORD16 *pvec0 = p_vec_0;
  WORD16 *pvec1 = p_vec_1;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3, d_vec0, d_vec1;
  int c_itr = 0;
  ae_int64 d_out0, d_out1, d_out2, d_out3, d_out4, d_out5, d_out6, d_out7;
  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;
  d_out4 = *out_4_4;
  d_out5 = *out_5_5;
  d_out6 = *out_6_6;
  d_out7 = *out_7_7;

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L16X4_IP(d_vec0, (ae_int16x4*)pvec0, 8);
    AE_L16X4_IP(d_vec1, (ae_int16x4*)pvec1, 8);
    AE_L16X4_IP(d_mat0, (ae_int16x4*)p_mat_0, 8);
    AE_L16X4_IP(d_mat1, (ae_int16x4*)p_mat_1, 8);
    AE_L16X4_IP(d_mat2, (ae_int16x4*)p_mat_2, 8);
    AE_L16X4_IP(d_mat3, (ae_int16x4*)p_mat_3, 8);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out2, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat1, d_vec1);
    AE_MULAAAAQ16(d_out4, d_mat2, d_vec0);
    AE_MULAAAAQ16(d_out5, d_mat2, d_vec1);
    AE_MULAAAAQ16(d_out6, d_mat3, d_vec0);
    AE_MULAAAAQ16(d_out7, d_mat3, d_vec1);
  }
  int off = cols1&0x03;
  for(c_itr=0;c_itr<off;c_itr++){
    AE_L16_IP(d_vec0, (ae_int16 *)pvec0, 2);
    AE_L16_IP(d_vec1, (ae_int16 *)pvec1, 2);
    AE_L16_IP(d_mat0, (ae_int16 *)p_mat_0, 2);
    AE_L16_IP(d_mat1, (ae_int16 *)p_mat_1, 2);
    AE_L16_IP(d_mat2, (ae_int16 *)p_mat_2, 2);
    AE_L16_IP(d_mat3, (ae_int16 *)p_mat_3, 2);
    AE_MULA16_00(d_out0, d_mat0, d_vec0);
    AE_MULA16_00(d_out1, d_mat0, d_vec1);
    AE_MULA16_00(d_out2, d_mat1, d_vec0);
    AE_MULA16_00(d_out3, d_mat1, d_vec1);
    AE_MULA16_00(d_out4, d_mat2, d_vec0);
    AE_MULA16_00(d_out5, d_mat2, d_vec1);
    AE_MULA16_00(d_out6, d_mat3, d_vec0);
    AE_MULA16_00(d_out7, d_mat3, d_vec1);
  }
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;
  *out_4_4 = d_out4;
  *out_5_5 = d_out5;
  *out_6_6 = d_out6;
  *out_7_7 = d_out7;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_aligned
(ae_int64* out_0_0
 ,ae_int64* out_1_1
 ,ae_int64* out_2_2
 ,ae_int64* out_3_3
 ,WORD16*      p_mat_0
 ,WORD16*      p_mat_1
 ,WORD16*      p_mat_2
 ,WORD16*      p_mat_3
 ,WORD16*      p_vec_0
 ,WORD32      cols1)
{
  WORD16 *pvec0 = p_vec_0;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3, d_vec0;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  int c_itr = 0;

  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L16X4_IP(d_vec0, (ae_int16x4*)pvec0, 8);
    AE_L16X4_IP(d_mat0, (ae_int16x4*)p_mat_0, 8);
    AE_L16X4_IP(d_mat1, (ae_int16x4*)p_mat_1, 8);
    AE_L16X4_IP(d_mat2, (ae_int16x4*)p_mat_2, 8);
    AE_L16X4_IP(d_mat3, (ae_int16x4*)p_mat_3, 8);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat2, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat3, d_vec0);
  }
  int off = cols1&0x03;
  for(c_itr=0;c_itr<off;c_itr++){
    AE_L16_IP(d_mat0, (ae_int16 *)p_mat_0, 2);
    AE_L16_IP(d_mat1, (ae_int16 *)p_mat_1, 2);
    AE_L16_IP(d_mat2, (ae_int16 *)p_mat_2, 2);
    AE_L16_IP(d_mat3, (ae_int16 *)p_mat_3, 2);
    AE_L16_IP(d_vec0, (ae_int16 *)pvec0, 2);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat2, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat3, d_vec0);
  }
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
(ae_int64* out_0_0
 ,WORD16*      p_mat_0
 ,WORD16*      p_vec_0
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

#define MPY_BY_QUANT_MULT_ACC64_OUT32(out0, inp0, mult, l_shift) \
{ \
  ae_int32x2 d_red_mult = AE_SEXT32X2D16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult))); \
  ae_int32x2 d_red_mult_l16 = AE_CVT32X2F16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult)));  \
  ae_int32x2 d_inp0_h = AE_ROUND32F64SASYM(inp0); \
  ae_int64 q0_l; \
  q0_l = AE_MUL32_LL(d_red_mult, AE_MOVINT32X2_FROMINT64(inp0)); \
  AE_MULAF32S_LL(q0_l, d_red_mult_l16, AE_SLAI32(d_inp0_h, 15)); \
  q0_l = AE_SLAA64S(q0_l, (l_shift + 17)); \
  out0 = AE_ROUND32F64SASYM(q0_l); \
}

#define MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out0, inp0, inp1, mult, l_shift) \
{ \
  ae_int32x2 d_red_mult = AE_SEXT32X2D16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult))); \
  ae_int32x2 d_red_mult_l16 = AE_CVT32X2F16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult)));  \
  ae_int32x2 d_inp01_h = AE_ROUND32X2F64SASYM(inp0, inp1); \
  ae_int64 q0_l, q1_l; \
  q0_l = AE_MUL32_LL(d_red_mult, AE_MOVINT32X2_FROMINT64(inp0)); \
  AE_MULAF32S_HH(q0_l, d_red_mult_l16, AE_SLAI32(d_inp01_h, 15)); \
  q1_l = AE_MUL32_LL(d_red_mult, AE_MOVINT32X2_FROMINT64(inp1)); \
  AE_MULAF32S_LL(q1_l, d_red_mult_l16, AE_SLAI32(d_inp01_h, 15)); \
  q0_l = AE_SLAA64S(q0_l, (l_shift + 17)); \
  q1_l = AE_SLAA64S(q1_l, (l_shift + 17)); \
  out0 = AE_ROUND32X2F64SASYM(q0_l, q1_l); \
}

WORD32 xa_nn_matmul_sym16sxsym16s_sym16s(
    WORD16 * __restrict__ p_out,
    const WORD16 * __restrict__ p_mat1,
    const WORD16 * __restrict__ p_vec1,
    const WORD64 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride,                      
    WORD32 mat1_zero_bias,
    WORD32 vec1_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias)
{
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((mat1_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);

  if(((rows&0x3) == 0) && ((row_stride1&0x3) == 0) && (((unsigned int)p_vec1 & 0x7) == 0) 
      && (((unsigned int)p_mat1 & 0x7) == 0) && ((vec_offset & 0x3) ==0))
  {
    ae_int64 *pbias = (ae_int64 *)p_bias;
    ae_int64 acc0;
    int m_itr, v_itr;
    for(m_itr = 0; m_itr < rows; m_itr+=4)
    {
      WORD16 *p_mat1_0 = (WORD16*)&p_mat1[(m_itr+0)*row_stride1];
      WORD16 *p_mat1_1 = (WORD16*)&p_mat1[(m_itr+1)*row_stride1];
      WORD16 *p_mat1_2 = (WORD16*)&p_mat1[(m_itr+2)*row_stride1];
      WORD16 *p_mat1_3 = (WORD16*)&p_mat1[(m_itr+3)*row_stride1];
      ae_int16 *p_dst0   = (ae_int16*)p_out + ((m_itr+0) * out_stride);
      ae_int16 *p_dst1   = (ae_int16*)p_out + ((m_itr+1) * out_stride);
      ae_int16 *p_dst2   = (ae_int16*)p_out + ((m_itr+2) * out_stride);
      ae_int16 *p_dst3   = (ae_int16*)p_out + ((m_itr+3) * out_stride);
      for(v_itr = 0; v_itr < (vec_count&~0x1); v_itr+=2)
      {
        ae_int64 acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8;
        WORD16* p_vec_0 = (WORD16*)(p_vec1 + ((v_itr+0) * vec_offset));
        WORD16* p_vec_1 = (WORD16*)(p_vec1 + ((v_itr+1) * vec_offset));
        acc1 = acc2 = AE_ZERO64();
        acc3 = acc4 = AE_ZERO64();
        acc5 = acc6 = AE_ZERO64();
        acc7 = acc8 = AE_ZERO64();
        if(p_bias != NULL){
          acc1 = acc2 = AE_L64_I(pbias, 0);
          acc3 = acc4 = AE_L64_I(pbias, 8);
          acc5 = acc6 = AE_L64_I(pbias, 16);
          acc7 = acc8 = AE_L64_I(pbias, 24);
        }
        _xa_nn_dot_product_4_rows_2_vecs_aligned
          ( &acc1, &acc2, &acc3, &acc4
           ,&acc5, &acc6, &acc7, &acc8
           ,p_mat1_0, p_mat1_1, p_mat1_2, p_mat1_3
           ,p_vec_0, p_vec_1
           ,cols1
          );
        ae_int32x2 result1, result2, result3, result4;
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(result1, acc1, acc2, out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(result2, acc3, acc4, out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(result3, acc5, acc6, out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(result4, acc7, acc8, out_multiplier, out_shift);
        ae_int16x4 d1 = AE_SAT16X4(result1, result2);
        ae_int16x4 d2 = AE_SAT16X4(result3, result4);
        AE_S16_0_XP(AE_SEL16_6543(d1, d1), p_dst0, out_offset*sizeof(WORD16));
        AE_S16_0_XP(AE_SEL16_5432(d1, d1), p_dst0, out_offset*sizeof(WORD16));
        AE_S16_0_XP(AE_SEL16_4321(d1, d1), p_dst1, out_offset*sizeof(WORD16));
        AE_S16_0_XP(	               d1, p_dst1, out_offset*sizeof(WORD16));
        AE_S16_0_XP(AE_SEL16_6543(d2, d2), p_dst2, out_offset*sizeof(WORD16));
        AE_S16_0_XP(AE_SEL16_5432(d2, d2), p_dst2, out_offset*sizeof(WORD16));
        AE_S16_0_XP(AE_SEL16_4321(d2, d2), p_dst3, out_offset*sizeof(WORD16));
        AE_S16_0_XP(	               d2, p_dst3, out_offset*sizeof(WORD16));
      }
      if(vec_count&0x1)
      {
        ae_int64 acc1, acc2, acc3, acc4;
        WORD16* p_vec_0 = (WORD16*)(p_vec1 + (v_itr * vec_offset));
        acc1 = AE_ZERO64();
        acc2 = AE_ZERO64();
        acc3 = AE_ZERO64();
        acc4 = AE_ZERO64();
        if(p_bias != NULL) {
          acc1 = AE_L64_I(pbias, 0);
          acc2 = AE_L64_I(pbias, 8);
          acc3 = AE_L64_I(pbias, 16);
          acc4 = AE_L64_I(pbias, 24);
        }
        _xa_nn_dot_product_4_rows_1_vecs_aligned
          ( &acc1, &acc2, &acc3, &acc4
           ,p_mat1_0, p_mat1_1, p_mat1_2, p_mat1_3
           ,p_vec_0, cols1
          );
        ae_int32x2 result1, result2, result3, result4;
        MPY_BY_QUANT_MULT_ACC64_OUT32(result1, acc1, out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_OUT32(result2, acc2, out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_OUT32(result3, acc3, out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_OUT32(result4, acc4, out_multiplier, out_shift);
        ae_int16x4 d1 = AE_SAT16X4(result1, result2);
        ae_int16x4 d2 = AE_SAT16X4(result3, result4);
        AE_S16_0_XP(AE_SEL16_6543(d1, d1), p_dst0, out_offset*sizeof(WORD16));
        AE_S16_0_XP(                   d1, p_dst1, out_offset*sizeof(WORD16));
        AE_S16_0_XP(AE_SEL16_6543(d2, d2), p_dst2, out_offset*sizeof(WORD16));
        AE_S16_0_XP(                   d2, p_dst3, out_offset*sizeof(WORD16));
      }
      if(p_bias != NULL) {
        AE_L64_IP(acc0, pbias, 32);
      }
    }
  }else
  if(p_mat1 && p_vec1)
  {
    ae_int64 *pbias = (ae_int64 *)p_bias;
    ae_int64 acc0;
    int m_itr=0, v_itr;
    for(; m_itr < (rows&~0x01); m_itr+=2)
    {
      WORD16 *p_mat1_0 = (WORD16*)&p_mat1[m_itr*row_stride1];
      WORD16 *p_mat1_1 = (WORD16*)&p_mat1[(m_itr+1)*row_stride1];
      ae_int16 *p_dst0   = (ae_int16*)p_out + (m_itr * out_stride);
      ae_int16 *p_dst1   = (ae_int16*)p_out + ((m_itr+1) * out_stride);

      v_itr = 0;

      for(; v_itr < (vec_count&~0x01); v_itr+=2)
      {
        ae_int64 acc1 = AE_ZERO64();
        ae_int64 acc2 = AE_ZERO64();
        ae_int64 acc3 = AE_ZERO64();
        ae_int64 acc4 = AE_ZERO64();
        if(p_bias != NULL) {
          acc1 = AE_L64_I(pbias, 0);
          acc2 = AE_L64_I(pbias, 8);
          acc3 = AE_L64_I(pbias, 0);
          acc4 = AE_L64_I(pbias, 8);
        }

        WORD16* p_vec_0 = (WORD16*)(p_vec1 + (v_itr * vec_offset));
        WORD16* p_vec_1 = (WORD16*)(p_vec1 + ((v_itr+1) * vec_offset));

        _xa_nn_dot_product_2_rows_2_vecs_unaligned(
                 &acc1, &acc2, &acc3, &acc4, p_mat1_0, p_mat1_1, p_vec_0, p_vec_1, cols1);

        ae_int32x2 result;
        MPY_BY_QUANT_MULT_ACC64_OUT32(result, acc1, out_multiplier, out_shift);
        ae_int16x4 d1 = AE_SAT16X4(result, result);
        AE_S16_0_XP(d1, p_dst0, out_offset*sizeof(WORD16));

        MPY_BY_QUANT_MULT_ACC64_OUT32(result, acc2, out_multiplier, out_shift);
        d1 = AE_SAT16X4(result, result);
        AE_S16_0_XP(d1, p_dst1, out_offset*sizeof(WORD16));

        MPY_BY_QUANT_MULT_ACC64_OUT32(result, acc3, out_multiplier, out_shift);
        d1 = AE_SAT16X4(result, result);
        AE_S16_0_XP(d1, p_dst0, out_offset*sizeof(WORD16));

        MPY_BY_QUANT_MULT_ACC64_OUT32(result, acc4, out_multiplier, out_shift);
        d1 = AE_SAT16X4(result, result);
        AE_S16_0_XP(d1, p_dst1, out_offset*sizeof(WORD16));
      }

      for(; v_itr < vec_count; v_itr++)
      {
        ae_int64 acc1 = AE_ZERO64();
        ae_int64 acc2 = AE_ZERO64();
        if(p_bias != NULL) {
          acc1 = AE_L64_I(pbias, 0);
          acc2 = AE_L64_I(pbias, 8);
        }

        WORD16* p_vec_0 = (WORD16*)(p_vec1 + (v_itr * vec_offset));
        _xa_nn_dot_product_2_rows_1_vecs_unaligned(&acc1, &acc2, p_mat1_0, p_mat1_1, p_vec_0, cols1);

        ae_int32x2 result;
        MPY_BY_QUANT_MULT_ACC64_OUT32(result, acc1, out_multiplier, out_shift);
        ae_int16x4 d1 = AE_SAT16X4(result, result);
        AE_S16_0_XP(d1, p_dst0, out_offset*sizeof(WORD16));

        MPY_BY_QUANT_MULT_ACC64_OUT32(result, acc2, out_multiplier, out_shift);
        d1 = AE_SAT16X4(result, result);
        AE_S16_0_XP(d1, p_dst1, out_offset*sizeof(WORD16));
      }
      if(p_bias != NULL) {
        AE_L64_IP(acc0, pbias, 16);
      }
    }
    for(; m_itr < rows; m_itr++)
    {
      WORD16 *p_mat1_0 = (WORD16*)&p_mat1[m_itr*row_stride1];
      ae_int16 *p_dst0   = (ae_int16*)p_out + (m_itr * out_stride);
      for(v_itr = 0; v_itr < vec_count; v_itr++)
      {
        ae_int64 acc1 = AE_ZERO64();
        if(p_bias != NULL) {
          acc1 = AE_L64_I(pbias, 0);
        }
        WORD16* p_vec_0 = (WORD16*)(p_vec1 + (v_itr * vec_offset));
        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc1
           ,p_mat1_0
           ,p_vec_0
           ,cols1
          );
        ae_int32x2 result;
        MPY_BY_QUANT_MULT_ACC64_OUT32(result, acc1, out_multiplier, out_shift);
        ae_int16x4 d1 = AE_SAT16X4(result, result);
        AE_S16_0_XP(d1, p_dst0, out_offset*sizeof(WORD16));
      }
      if(p_bias != NULL) {
        AE_L64_IP(acc0, pbias, 8);
      }
    }
  }
  else
    return -1;

  return 0;
}
