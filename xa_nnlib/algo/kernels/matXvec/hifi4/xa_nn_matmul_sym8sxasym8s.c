/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2(inp, multiplier, left_shift, right_shift) \
    inp = AE_SLAA32S(inp, left_shift); \
  inp = AE_MULFP32X2RAS(inp, AE_MOVDA32(multiplier)); \
  inp = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(inp), right_shift), AE_SRAA64(AE_CVT64F32_L(inp), right_shift));

#define AE_L8X4S_I_HIFI4(d, ptr, inc) \
    d = AE_L8X4F_I(ptr, inc); \
  d = AE_SRAI16(d, 8);

#define AE_MINMAX32_HF4(acc, min, max) \
    acc = AE_MAX32(acc, min); \
  acc = AE_MIN32(acc, max);

#define AE_S8_FROM32_WITHSTRIDE(val32, dst, stride) \
    *dst = (WORD8)val32; \
  dst += stride;

#define AE_S8_FROM32(val32, dst, index) \
    dst[index] = (WORD8)val32;


static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
(ae_int32x2* out_0_0
 ,WORD8*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD32      cols1
 ,WORD32      vec_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat, d_vec;
  ae_int32x2 d_tmp;
  ae_int32x2 d_out;
  d_out = *out_0_0;

  for(;c_itr<(cols1); c_itr++)
  {
    d_mat = AE_MOVDA16(*((WORD8 *)p_mat_0));
    d_vec = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_mat_0++;
    p_vec_0++;
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));
    AE_MULA16X4(d_out, d_tmp, d_mat, d_vec);
  }
  *out_0_0 = d_out;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,WORD8*      p_mat_0
 ,WORD32      matstride
 ,WORD8*      p_vec_0
 ,WORD32      cols1
 ,WORD32      vec_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3;
  ae_int16x4 d_vec0;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  WORD8 *p_mat_1, *p_mat_2, *p_mat_3;
  ae_int16x4 d_vzb;

  d_vzb = AE_MOVDA16(vec_zero_bias);
  p_mat_1 = p_mat_0 + matstride;
  p_mat_2 = p_mat_1 + matstride;
  p_mat_3 = p_mat_2 + matstride;

  d_out0 = AE_SRAI64(AE_CVT64F32_H(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_H(*out_1_1), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);

  for(c_itr = 0;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4F_IP(d_vec0, p_vec_0, 4);
    d_vec0 = AE_SRAI16(d_vec0, 8);
    d_vec0 = AE_ADD16(d_vec0, d_vzb);
    AE_L8X4F_IP(d_mat0, p_mat_0, 4);
    AE_L8X4F_IP(d_mat1, p_mat_1, 4);
    AE_L8X4F_IP(d_mat2, p_mat_2, 4);
    AE_L8X4F_IP(d_mat3, p_mat_3, 4);

    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat2, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat3, d_vec0);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  d_out2 = AE_SRAI64(d_out2, 8);
  d_out3 = AE_SRAI64(d_out3, 8);

  *out_0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0), AE_MOVINT32X2_FROMINT64(d_out1));
  *out_1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out2), AE_MOVINT32X2_FROMINT64(d_out3));
}

WORD32 xa_nn_matmul_per_chan_sym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
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
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
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

  int m_itr, v_itr;
  int l_shift[4], r_shift[4];
  ae_int32x2 min_int8, max_int8;
  ae_int32x2 acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0;
  max_int8 = AE_MOVDA32(127);
  min_int8 = AE_MOVDA32(-128);
  acc_row0_vec0 = AE_ZERO32();

  /*tbd : try with vec_count = 4 */
  if(((rows&0x3) == 0) && ((cols1&0x3) == 0) && ((row_stride1&0x3) == 0) && (((unsigned int)p_mat1 & 0x3) == 0) 
      && (((unsigned int)p_vec1 & 0x3) == 0) && ((vec_offset & 0x3) ==0))
  {
    for(m_itr = 0; m_itr < rows; m_itr+=4)
    {
      WORD8 *p_mat1_0 = (WORD8*)&p_mat1[m_itr*row_stride1];
      WORD8 *p_dst0   = (WORD8*)p_out + (m_itr * out_stride);
      WORD8 *p_dst1   = p_dst0 + out_stride;
      WORD8 *p_dst2   = p_dst1 + out_stride;
      WORD8 *p_dst3   = p_dst2 + out_stride;

      l_shift[0] = p_out_shift[m_itr+0] < 0 ? 0 :  p_out_shift[m_itr+0];
      r_shift[0] = p_out_shift[m_itr+0] > 0 ? 0 : -p_out_shift[m_itr+0];
      l_shift[1] = p_out_shift[m_itr+1] < 0 ? 0 :  p_out_shift[m_itr+1];
      r_shift[1] = p_out_shift[m_itr+1] > 0 ? 0 : -p_out_shift[m_itr+1];
      l_shift[2] = p_out_shift[m_itr+2] < 0 ? 0 :  p_out_shift[m_itr+2];
      r_shift[2] = p_out_shift[m_itr+2] > 0 ? 0 : -p_out_shift[m_itr+2];
      l_shift[3] = p_out_shift[m_itr+3] < 0 ? 0 :  p_out_shift[m_itr+3];
      r_shift[3] = p_out_shift[m_itr+3] > 0 ? 0 : -p_out_shift[m_itr+3];
      
      for(v_itr = 0; v_itr < vec_count; v_itr++)
      {
        acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = AE_ZERO32();
        if(p_bias)
        { 
          acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr+0], p_bias[m_itr+1]);
          acc_row3_vec0 = AE_MOVDA32X2(p_bias[m_itr+2], p_bias[m_itr+3]);
        }

        WORD8* p_vec_0 = (WORD8*)(p_vec1 + (v_itr * vec_offset));
        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row1_vec0
           ,&acc_row3_vec0
           ,p_mat1_0
           ,row_stride1
           ,p_vec_0
           ,cols1
           ,vec1_zero_bias
          );
        acc_row0_vec0 = AE_SEL32_HH(acc_row1_vec0, acc_row1_vec0);
        acc_row1_vec0 = AE_SEL32_LL(acc_row1_vec0, acc_row1_vec0);
        acc_row2_vec0 = AE_SEL32_HH(acc_row3_vec0, acc_row3_vec0);
        acc_row3_vec0 = AE_SEL32_LL(acc_row3_vec0, acc_row3_vec0);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[m_itr+0], l_shift[0], r_shift[0]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, p_out_multiplier[m_itr+1], l_shift[1], r_shift[1]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row2_vec0, p_out_multiplier[m_itr+2], l_shift[2], r_shift[2]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row3_vec0, p_out_multiplier[m_itr+3], l_shift[3], r_shift[3]);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        acc_row2_vec0 = AE_ADD32S(acc_row2_vec0, out_zero_bias);
        acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row1_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row2_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row3_vec0, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row1_vec0), p_dst1, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row2_vec0), p_dst2, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row3_vec0), p_dst3, out_offset);
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
    for(m_itr = 0; m_itr < rows; m_itr++)
    {
      WORD8 *p_mat1_0 = (WORD8*)&p_mat1[m_itr*row_stride1];
      WORD8 *p_dst0   = (WORD8*)p_out + (m_itr * out_stride);

      l_shift[0] = p_out_shift[m_itr] < 0 ? 0 : p_out_shift[m_itr];
      r_shift[0] = p_out_shift[m_itr] > 0 ? 0 : -p_out_shift[m_itr];

      for(v_itr = 0; v_itr < vec_count; v_itr++)
      {
        acc_row0_vec0 = AE_ZERO32();
        if(p_bias)
          acc_row0_vec0 = AE_MOVDA32(p_bias[m_itr]);

        WORD8* p_vec_0 = (WORD8*)(p_vec1 + (v_itr * vec_offset));

        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,vec1_zero_bias
          );
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[m_itr], l_shift[0], r_shift[0]);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_offset);
      }
    }
  }
  else
    return -1;

  return 0;
}
