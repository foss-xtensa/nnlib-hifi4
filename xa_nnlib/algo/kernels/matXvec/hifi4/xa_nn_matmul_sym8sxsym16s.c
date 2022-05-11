/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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

static inline ae_int32x2 MultiplyByQuantizedMultiplier_ref(ae_int64 d_x,
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
 * new_shift = 46-32 to 8-32
 * */
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

static inline void _xa_nn_dot_product_4_rows_2_vecs_aligned
(ae_int64* out_0_0
 ,ae_int64* out_1_1
 ,ae_int64* out_2_2
 ,ae_int64* out_3_3
 ,ae_int64* out_4_4
 ,ae_int64* out_5_5
 ,ae_int64* out_6_6
 ,ae_int64* out_7_7
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_mat_2
 ,WORD8*      p_mat_3
 ,WORD16*      p_vec_0
 ,WORD16*      p_vec_1
 ,WORD32      cols1)
{
  ae_int16x4 *pvec0 = (ae_int16x4*)p_vec_0;
  ae_int16x4 *pvec1 = (ae_int16x4*)p_vec_1;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3, d_vec0, d_vec1;
  int c_itr = 0;
  ae_int64 d_out0, d_out1, d_out2, d_out3, d_out4, d_out5, d_out6, d_out7;
  d_out0 = AE_SLAI64(*out_0_0, 8);
  d_out1 = AE_SLAI64(*out_1_1, 8);
  d_out2 = AE_SLAI64(*out_2_2, 8);
  d_out3 = AE_SLAI64(*out_3_3, 8);
  d_out4 = AE_SLAI64(*out_4_4, 8);
  d_out5 = AE_SLAI64(*out_5_5, 8);
  d_out6 = AE_SLAI64(*out_6_6, 8);
  d_out7 = AE_SLAI64(*out_7_7, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L16X4_IP(d_vec0, pvec0, 8);
    AE_L16X4_IP(d_vec1, pvec1, 8);
    AE_L8X4F_IP(d_mat0, p_mat_0, 4);
    AE_L8X4F_IP(d_mat1, p_mat_1, 4);
    AE_L8X4F_IP(d_mat2, p_mat_2, 4);
    AE_L8X4F_IP(d_mat3, p_mat_3, 4);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out2, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat1, d_vec1);
    AE_MULAAAAQ16(d_out4, d_mat2, d_vec0);
    AE_MULAAAAQ16(d_out5, d_mat2, d_vec1);
    AE_MULAAAAQ16(d_out6, d_mat3, d_vec0);
    AE_MULAAAAQ16(d_out7, d_mat3, d_vec1);
  }
  *out_0_0 = AE_SRAI64(d_out0, 8);
  *out_1_1 = AE_SRAI64(d_out1, 8);
  *out_2_2 = AE_SRAI64(d_out2, 8);
  *out_3_3 = AE_SRAI64(d_out3, 8);
  *out_4_4 = AE_SRAI64(d_out4, 8);
  *out_5_5 = AE_SRAI64(d_out5, 8);
  *out_6_6 = AE_SRAI64(d_out6, 8);
  *out_7_7 = AE_SRAI64(d_out7, 8);
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_aligned
(ae_int64* out_0_0
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
  int c_itr = 0;

  d_out0 = AE_SLAI64(*out_0_0, 8);
  d_out1 = AE_SLAI64(*out_1_1, 8);
  d_out2 = AE_SLAI64(*out_2_2, 8);
  d_out3 = AE_SLAI64(*out_3_3, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L16X4_IP(d_vec0, pvec0, 8);
    AE_L8X4F_IP(d_mat0, p_mat_0, 4);
    AE_L8X4F_IP(d_mat1, p_mat_1, 4);
    AE_L8X4F_IP(d_mat2, p_mat_2, 4);
    AE_L8X4F_IP(d_mat3, p_mat_3, 4);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat2, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat3, d_vec0);
  }
  *out_0_0 = AE_SRAI64(d_out0, 8);
  *out_1_1 = AE_SRAI64(d_out1, 8);
  *out_2_2 = AE_SRAI64(d_out2, 8);
  *out_3_3 = AE_SRAI64(d_out3, 8);
}

/* tbd : un-optimized case, will optimize later */
static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
(ae_int64* out_0_0
 ,WORD8*      p_mat_0
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

WORD32 xa_nn_matmul_per_chan_sym8sxsym16s_sym16s(
    WORD16 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD16 * __restrict__ p_vec1,
    const WORD64 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride,                      
    WORD32 vec1_zero_bias,
    const WORD32* __restrict__ p_out_multiplier,
    const WORD32* __restrict__ p_out_shift,
    WORD32 out_zero_bias)
{
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shift, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shift, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  int itr = 0;
  for(itr=0; itr<rows; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  if(((rows&0x3) == 0) && ((cols1&0x3) == 0) && ((row_stride1&0x3) == 0) && (((unsigned int)p_vec1 & 0x7) == 0) 
      && (((unsigned int)p_mat1 & 0x3) == 0) && ((vec_offset & 0x3) ==0))
  {
    ae_int64 *pbias = (ae_int64 *)p_bias;
    ae_int64 acc1;
    int m_itr, v_itr;
    for(m_itr = 0; m_itr < rows; m_itr+=4)
    {
      WORD8 *p_mat1_0 = (WORD8*)&p_mat1[(m_itr+0)*row_stride1];
      WORD8 *p_mat1_1 = (WORD8*)&p_mat1[(m_itr+1)*row_stride1];
      WORD8 *p_mat1_2 = (WORD8*)&p_mat1[(m_itr+2)*row_stride1];
      WORD8 *p_mat1_3 = (WORD8*)&p_mat1[(m_itr+3)*row_stride1];
      ae_int16 *p_dst0   = (ae_int16*)p_out + ((m_itr+0) * out_stride);
      ae_int16 *p_dst1   = (ae_int16*)p_out + ((m_itr+1) * out_stride);
      ae_int16 *p_dst2   = (ae_int16*)p_out + ((m_itr+2) * out_stride);
      ae_int16 *p_dst3   = (ae_int16*)p_out + ((m_itr+3) * out_stride);
      for(v_itr = 0; v_itr < (vec_count&~0x1); v_itr+=2)
      {
        ae_int64 acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8;
        WORD16* p_vec_0 = (WORD16*)(p_vec1 + ((v_itr+0) * vec_offset));
        WORD16* p_vec_1 = (WORD16*)(p_vec1 + ((v_itr+1) * vec_offset));
	acc1 = acc2 = AE_L64_I(pbias, 0);
	acc3 = acc4 = AE_L64_I(pbias, 8);
	acc5 = acc6 = AE_L64_I(pbias, 16);
	acc7 = acc8 = AE_L64_I(pbias, 24);
        _xa_nn_dot_product_4_rows_2_vecs_aligned
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
           ,p_mat1_2
           ,p_mat1_3
           ,p_vec_0
           ,p_vec_1
           ,cols1
          );
        ae_int32x2 result1 = MultiplyByQuantizedMultiplier_x2_opt(acc1, acc2, p_out_multiplier[m_itr+0], p_out_shift[m_itr+0]);
        ae_int32x2 result2 = MultiplyByQuantizedMultiplier_x2_opt(acc3, acc4, p_out_multiplier[m_itr+1], p_out_shift[m_itr+1]);
        ae_int32x2 result3 = MultiplyByQuantizedMultiplier_x2_opt(acc5, acc6, p_out_multiplier[m_itr+2], p_out_shift[m_itr+2]);
        ae_int32x2 result4 = MultiplyByQuantizedMultiplier_x2_opt(acc7, acc8, p_out_multiplier[m_itr+3], p_out_shift[m_itr+3]);
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
      if(vec_count&0x1)//for(; v_itr < vec_count; v_itr++)
      {
        ae_int64 acc1, acc2, acc3, acc4;
        WORD16* p_vec_0 = (WORD16*)(p_vec1 + (v_itr * vec_offset));
        acc1 = AE_L64_I(pbias, 0);
        acc2 = AE_L64_I(pbias, 8);
        acc3 = AE_L64_I(pbias, 16);
        acc4 = AE_L64_I(pbias, 24);
        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc1
           ,&acc2
           ,&acc3
           ,&acc4
           ,p_mat1_0
           ,p_mat1_1
           ,p_mat1_2
           ,p_mat1_3
           ,p_vec_0
           ,cols1
          );
        ae_int32x2 result1 = MultiplyByQuantizedMultiplier_ref(acc1, p_out_multiplier[m_itr+0], p_out_shift[m_itr+0]);
        ae_int32x2 result2 = MultiplyByQuantizedMultiplier_ref(acc2, p_out_multiplier[m_itr+1], p_out_shift[m_itr+1]);
        ae_int32x2 result3 = MultiplyByQuantizedMultiplier_ref(acc3, p_out_multiplier[m_itr+2], p_out_shift[m_itr+2]);
        ae_int32x2 result4 = MultiplyByQuantizedMultiplier_ref(acc4, p_out_multiplier[m_itr+3], p_out_shift[m_itr+3]);
        ae_int16x4 d1 = AE_SAT16X4(result1, result2);
        ae_int16x4 d2 = AE_SAT16X4(result3, result4);
        AE_S16_0_XP(AE_SEL16_6543(d1, d1), p_dst0, out_offset*sizeof(WORD16));
        AE_S16_0_XP(                   d1, p_dst1, out_offset*sizeof(WORD16));
        AE_S16_0_XP(AE_SEL16_6543(d2, d2), p_dst2, out_offset*sizeof(WORD16));
        AE_S16_0_XP(                   d2, p_dst3, out_offset*sizeof(WORD16));
      }
      AE_L64_IP(acc1, pbias, 32);
    }
  }else
  if(p_mat1 && p_vec1)
  {
    /* tbd: bias check will be added later */
    ae_int64 *pbias = (ae_int64 *)p_bias;
    ae_int64 acc1;
    int m_itr, v_itr;
    for(m_itr = 0; m_itr < rows; m_itr++)
    {
      WORD8 *p_mat1_0 = (WORD8*)&p_mat1[m_itr*row_stride1];
      ae_int16 *p_dst0   = (ae_int16*)p_out + (m_itr * out_stride);
      for(v_itr = 0; v_itr < vec_count; v_itr++)
      {
        ae_int64 acc1 = AE_L64_I(pbias, 0);
        WORD16* p_vec_0 = (WORD16*)(p_vec1 + (v_itr * vec_offset));
        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc1
           ,p_mat1_0
           ,p_vec_0
           ,cols1
          );
        ae_int32x2 result = MultiplyByQuantizedMultiplier_ref(acc1, p_out_multiplier[m_itr], p_out_shift[m_itr]);
        ae_int16x4 d1 = AE_SAT16X4(result, result);
        AE_S16_0_XP(d1, p_dst0, out_offset*sizeof(WORD16));
      }
      AE_L64_IP(acc1, pbias, 8);
    }
  }
  else
    return -1;

  return 0;
}
