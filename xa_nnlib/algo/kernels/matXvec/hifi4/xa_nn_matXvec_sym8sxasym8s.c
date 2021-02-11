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
#include "xa_nnlib_common_macros.h"

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2(inp, multiplier, left_shift, right_shift) \
    inp = AE_SLAA32S(inp, left_shift); \
    inp = AE_MULFP32X2RAS(inp, AE_MOVDA32(multiplier)); \
    inp = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(inp), right_shift), AE_SRAA64(AE_CVT64F32_L(inp), right_shift));

static inline void _xa_nn_dot_product_4_rows_1_vec_mat_unaligned_vec_aligned
    (ae_int32x2*  out_0_0
    ,ae_int32x2*  out_1_0
    ,const WORD8* p_mat_0
    ,const WORD8* p_vec_0
    ,WORD32       cols1
    ,WORD32       row_stride1
    ,WORD32       vec_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3, d_vec;
  ae_int16x4 d_mat0_la, d_mat1_la, d_mat2_la, d_mat3_la;
  ae_int64 out_0, out_1, out_2, out_3;

  WORD8 *p_mat_1 = ((WORD8 *)p_mat_0 + row_stride1);
  WORD8 *p_mat_2 = ((WORD8 *)p_mat_1 + row_stride1);
  WORD8 *p_mat_3 = ((WORD8 *)p_mat_2 + row_stride1);

  WORD8 *p_mat_0_tmp = (WORD8*)p_mat_0;
  WORD8 *p_mat_1_tmp = (WORD8*)p_mat_1;
  WORD8 *p_mat_2_tmp = (WORD8*)p_mat_2;
  WORD8 *p_mat_3_tmp = (WORD8*)p_mat_3;
  WORD8 *p_vec = (WORD8*)p_vec_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  out_0 = AE_CVT64F32_H(acc_row0_vec0);
  out_1 = AE_CVT64F32_L(acc_row0_vec0);
  out_2 = AE_CVT64F32_H(acc_row1_vec0);
  out_3 = AE_CVT64F32_L(acc_row1_vec0);
  out_0 = AE_SRAI64(out_0, 32-8);
  out_1 = AE_SRAI64(out_1, 32-8);
  out_2 = AE_SRAI64(out_2, 32-8);
  out_3 = AE_SRAI64(out_3, 32-8);

  PRIME_8X4F(p_mat_0_tmp, d_mat0_la);
  PRIME_8X4F(p_mat_1_tmp, d_mat1_la);
  PRIME_8X4F(p_mat_2_tmp, d_mat2_la);
  PRIME_8X4F(p_mat_3_tmp, d_mat3_la);

  for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
  {
    AE_LA8X4F_IP(d_mat0, d_mat0_la, p_mat_0_tmp);
    AE_LA8X4F_IP(d_mat1, d_mat1_la, p_mat_1_tmp);
    AE_LA8X4F_IP(d_mat2, d_mat2_la, p_mat_2_tmp);
    AE_LA8X4F_IP(d_mat3, d_mat3_la, p_mat_3_tmp);
    AE_L8X4F_IP(d_vec, p_vec, 4);
    d_vec = AE_SRAI16(d_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
  }

  out_0 = AE_SRAI64(out_0, 8);
  out_1 = AE_SRAI64(out_1, 8);
  out_2 = AE_SRAI64(out_2, 8);
  out_3 = AE_SRAI64(out_3, 8);
  acc_row0_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_0), AE_MOVINT32X2_FROMINT64(out_1));
  acc_row1_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_2), AE_MOVINT32X2_FROMINT64(out_3));

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,const WORD8*      p_mat_0
    ,const WORD8*      p_vec_0
    ,WORD32      cols1
    ,WORD32      row_stride1
    ,WORD32      vec_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3, d_vec;
  ae_int16x4 d_mat0_la, d_mat1_la, d_mat2_la, d_mat3_la, d_vec_la;
  ae_int64 out_0, out_1, out_2, out_3;

  WORD8 *p_vec = (WORD8*)p_vec_0;
  WORD8 *p_mat_1 = ((WORD8 *)p_mat_0 + row_stride1);
  WORD8 *p_mat_2 = ((WORD8 *)p_mat_1 + row_stride1);
  WORD8 *p_mat_3 = ((WORD8 *)p_mat_2 + row_stride1);

  WORD8 *p_mat_0_tmp = (WORD8 *)p_mat_0;
  WORD8 *p_mat_1_tmp = p_mat_1;
  WORD8 *p_mat_2_tmp = p_mat_2;
  WORD8 *p_mat_3_tmp = p_mat_3;
  WORD8 *p_vec_tmp = p_vec;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  out_0 = AE_CVT64F32_H(acc_row0_vec0);
  out_1 = AE_CVT64F32_L(acc_row0_vec0);
  out_2 = AE_CVT64F32_H(acc_row1_vec0);
  out_3 = AE_CVT64F32_L(acc_row1_vec0);
  out_0 = AE_SRAI64(out_0, 32-8);
  out_1 = AE_SRAI64(out_1, 32-8);
  out_2 = AE_SRAI64(out_2, 32-8);
  out_3 = AE_SRAI64(out_3, 32-8);

  PRIME_8X4F(p_mat_0_tmp, d_mat0_la);
  PRIME_8X4F(p_mat_1_tmp, d_mat1_la);
  PRIME_8X4F(p_mat_2_tmp, d_mat2_la);
  PRIME_8X4F(p_mat_3_tmp, d_mat3_la);
  PRIME_8X4F(p_vec_tmp, d_vec_la);

  for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
  {
    AE_LA8X4F_IP(d_mat0, d_mat0_la, p_mat_0_tmp);
    AE_LA8X4F_IP(d_mat1, d_mat1_la, p_mat_1_tmp);
    AE_LA8X4F_IP(d_mat2, d_mat2_la, p_mat_2_tmp);
    AE_LA8X4F_IP(d_mat3, d_mat3_la, p_mat_3_tmp);
    AE_LA8X4F_IP(d_vec, d_vec_la, p_vec_tmp);

    d_vec = AE_SRAI16(d_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
  }
  out_0 = AE_SRAI64(out_0, 8);
  out_1 = AE_SRAI64(out_1, 8);
  out_2 = AE_SRAI64(out_2, 8);
  out_3 = AE_SRAI64(out_3, 8);
  acc_row0_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_0), AE_MOVINT32X2_FROMINT64(out_1));
  acc_row1_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_2), AE_MOVINT32X2_FROMINT64(out_3));

  {
    int index = cols1&(~0x3);
    int rem;
    ae_int16x4 d_mat;
    WORD8 *p_vec_rem = (WORD8*)&p_vec_0[index];
    WORD8 *p_mat_0_rem = (WORD8*)&p_mat_0[index];
    WORD8 *p_mat_1_rem = (p_mat_0_rem + row_stride1);
    WORD8 *p_mat_2_rem = (p_mat_1_rem + row_stride1);
    WORD8 *p_mat_3_rem = (p_mat_2_rem + row_stride1);

    for(rem=0; rem<(cols1&0x3); rem++)
    {
      d_mat0 = AE_MOVDA16(*(p_mat_0_rem+rem));
      d_mat1 = AE_MOVDA16(*(p_mat_1_rem+rem));
      d_mat2 = AE_MOVDA16(*(p_mat_2_rem+rem));
      d_mat3 = AE_MOVDA16(*(p_mat_3_rem+rem));
      d_vec = AE_MOVDA16(*(p_vec_rem+rem));
      d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));
      d_mat = AE_SEL16_7531(AE_SEL16_7531(d_mat0, d_mat1), AE_SEL16_7531(d_mat2, d_mat3));
      AE_MULA16X4(acc_row0_vec0, acc_row1_vec0, d_mat, d_vec);
    }
  }
  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_8_rows_1_vec_mat_aligned_vec_aligned
    (ae_int32x2*  out_0_0
    ,ae_int32x2*  out_1_0
    ,ae_int32x2*  out_2_0
    ,ae_int32x2*  out_3_0
    ,const WORD8* p_mat_0
    ,const WORD8* p_vec_0
    ,WORD32       cols1
    ,WORD32       row_stride1
    ,WORD32       vec_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3, d_vec;
  ae_int16x4 d_mat4, d_mat5, d_mat6, d_mat7;
  ae_int64 out_0, out_1, out_2, out_3;
  ae_int64 out_4, out_5, out_6, out_7;

  WORD8 *p_mat_1 = ((WORD8 *)p_mat_0 + row_stride1);
  WORD8 *p_mat_2 = ((WORD8 *)p_mat_1 + row_stride1);
  WORD8 *p_mat_3 = ((WORD8 *)p_mat_2 + row_stride1);
  WORD8 *p_mat_4 = ((WORD8 *)p_mat_3 + row_stride1);
  WORD8 *p_mat_5 = ((WORD8 *)p_mat_4 + row_stride1);
  WORD8 *p_mat_6 = ((WORD8 *)p_mat_5 + row_stride1);
  WORD8 *p_mat_7 = ((WORD8 *)p_mat_6 + row_stride1);

  WORD8 *p_vec = (WORD8*)p_vec_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row2_vec0 = *out_2_0;
  ae_int32x2 acc_row3_vec0 = *out_3_0;

  out_0 = AE_CVT64F32_H(acc_row0_vec0);
  out_1 = AE_CVT64F32_L(acc_row0_vec0);
  out_2 = AE_CVT64F32_H(acc_row1_vec0);
  out_3 = AE_CVT64F32_L(acc_row1_vec0);
  out_4 = AE_CVT64F32_H(acc_row2_vec0);
  out_5 = AE_CVT64F32_L(acc_row2_vec0);
  out_6 = AE_CVT64F32_H(acc_row3_vec0);
  out_7 = AE_CVT64F32_L(acc_row3_vec0);
  out_0 = AE_SRAI64(out_0, 32-8);
  out_1 = AE_SRAI64(out_1, 32-8);
  out_2 = AE_SRAI64(out_2, 32-8);
  out_3 = AE_SRAI64(out_3, 32-8);
  out_4 = AE_SRAI64(out_4, 32-8);
  out_5 = AE_SRAI64(out_5, 32-8);
  out_6 = AE_SRAI64(out_6, 32-8);
  out_7 = AE_SRAI64(out_7, 32-8);

#if (XCHAL_HAVE_HIFI4) || (XCHAL_HAVE_HIFI3Z) || (XCHAL_HAVE_FUSION)
  for(c_itr = 0; c_itr < cols1 >> 3; c_itr++)
  {
    ae_int16x4 d_mat01, d_mat11, d_mat21, d_mat31, d_vec1;
    ae_int16x4 d_mat41, d_mat51, d_mat61, d_mat71;

    d_vec = AE_L8X4F_I(p_vec, 4);
    AE_L8X4F_IP(d_vec1, p_vec, 8);

    d_vec = AE_SRAI16(d_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    d_vec1 = AE_SRAI16(d_vec1, 8);
    d_vec1 = AE_ADD16(d_vec1, AE_MOVDA16(vec_zero_bias));

    d_mat0 =  AE_L8X4F_I(p_mat_0, 4);
    AE_L8X4F_IP(d_mat01,  p_mat_0, 8);
    d_mat1 =  AE_L8X4F_I(p_mat_1, 4);
    AE_L8X4F_IP(d_mat11,  p_mat_1, 8);
    d_mat2 =  AE_L8X4F_I(p_mat_2, 4);
    AE_L8X4F_IP(d_mat21,  p_mat_2, 8);
    d_mat3 =  AE_L8X4F_I(p_mat_3, 4);
    AE_L8X4F_IP(d_mat31,  p_mat_3, 8);
    d_mat4 =  AE_L8X4F_I(p_mat_4, 4);
    AE_L8X4F_IP(d_mat41,  p_mat_4, 8);
    d_mat5 =  AE_L8X4F_I(p_mat_5, 4);
    AE_L8X4F_IP(d_mat51,  p_mat_5, 8);
    d_mat6 =  AE_L8X4F_I(p_mat_6, 4);
    AE_L8X4F_IP(d_mat61,  p_mat_6, 8);
    d_mat7 =  AE_L8X4F_I(p_mat_7, 4);
    AE_L8X4F_IP(d_mat71,  p_mat_7, 8);



    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
    AE_MULAAAAQ16(out_4, d_mat4, d_vec);
    AE_MULAAAAQ16(out_5, d_mat5, d_vec);
    AE_MULAAAAQ16(out_6, d_mat6, d_vec);
    AE_MULAAAAQ16(out_7, d_mat7, d_vec);

    AE_MULAAAAQ16(out_0, d_mat01, d_vec1);
    AE_MULAAAAQ16(out_1, d_mat11, d_vec1);
    AE_MULAAAAQ16(out_2, d_mat21, d_vec1);
    AE_MULAAAAQ16(out_3, d_mat31, d_vec1);
    AE_MULAAAAQ16(out_4, d_mat41, d_vec1);
    AE_MULAAAAQ16(out_5, d_mat51, d_vec1);
    AE_MULAAAAQ16(out_6, d_mat61, d_vec1);
    AE_MULAAAAQ16(out_7, d_mat71, d_vec1);
  }

  /* Remaining 4 elements of multiple of 4 length */
  if((c_itr << 3) < cols1)
  {
    AE_L8X4F_IP(d_mat0,  p_mat_0, 4);
    AE_L8X4F_IP(d_mat1,  p_mat_1, 4);
    AE_L8X4F_IP(d_mat2,  p_mat_2, 4);
    AE_L8X4F_IP(d_mat3,  p_mat_3, 4);
    AE_L8X4F_IP(d_mat4,  p_mat_4, 4);
    AE_L8X4F_IP(d_mat5,  p_mat_5, 4);
    AE_L8X4F_IP(d_mat6,  p_mat_6, 4);
    AE_L8X4F_IP(d_mat7,  p_mat_7, 4);

    AE_L8X4F_IP(d_vec, p_vec, 4);
    d_vec = AE_SRAI16(d_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
    AE_MULAAAAQ16(out_4, d_mat4, d_vec);
    AE_MULAAAAQ16(out_5, d_mat5, d_vec);
    AE_MULAAAAQ16(out_6, d_mat6, d_vec);
    AE_MULAAAAQ16(out_7, d_mat7, d_vec);
  }
#else
  for(c_itr = 0; c_itr < cols1 >> 2; c_itr++)
  {
    AE_L8X4F_IP(d_mat0,  p_mat_0, 4);
    AE_L8X4F_IP(d_mat1,  p_mat_1, 4);
    AE_L8X4F_IP(d_mat2,  p_mat_2, 4);
    AE_L8X4F_IP(d_mat3,  p_mat_3, 4);
    AE_L8X4F_IP(d_mat4,  p_mat_4, 4);
    AE_L8X4F_IP(d_mat5,  p_mat_5, 4);
    AE_L8X4F_IP(d_mat6,  p_mat_6, 4);
    AE_L8X4F_IP(d_mat7,  p_mat_7, 4);
    AE_L8X4F_IP(d_vec, p_vec, 4);
    d_vec = AE_SRAI16(d_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
    AE_MULAAAAQ16(out_4, d_mat4, d_vec);
    AE_MULAAAAQ16(out_5, d_mat5, d_vec);
    AE_MULAAAAQ16(out_6, d_mat6, d_vec);
    AE_MULAAAAQ16(out_7, d_mat7, d_vec);
  }
#endif

  out_0 = AE_SRAI64(out_0, 8);
  out_1 = AE_SRAI64(out_1, 8);
  out_2 = AE_SRAI64(out_2, 8);
  out_3 = AE_SRAI64(out_3, 8);
  out_4 = AE_SRAI64(out_4, 8);
  out_5 = AE_SRAI64(out_5, 8);
  out_6 = AE_SRAI64(out_6, 8);
  out_7 = AE_SRAI64(out_7, 8);
  acc_row0_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_0), AE_MOVINT32X2_FROMINT64(out_1));
  acc_row1_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_2), AE_MOVINT32X2_FROMINT64(out_3));
  acc_row2_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_4), AE_MOVINT32X2_FROMINT64(out_5));
  acc_row3_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_6), AE_MOVINT32X2_FROMINT64(out_7));
  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
  *out_2_0 = acc_row2_vec0;
  *out_3_0 = acc_row3_vec0;
}

static inline void _xa_nn_dot_product_4_rows_1_vec_mat_aligned_vec_aligned
    (ae_int32x2*  out_0_0
    ,ae_int32x2*  out_1_0
    ,const WORD8* p_mat_0
    ,const WORD8* p_vec_0
    ,WORD32       cols1
    ,WORD32       row_stride1
    ,WORD32       vec_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3, d_vec;
  ae_int64 out_0, out_1, out_2, out_3;

  WORD8 *p_mat_1 = ((WORD8 *)p_mat_0 + row_stride1);
  WORD8 *p_mat_2 = ((WORD8 *)p_mat_1 + row_stride1);
  WORD8 *p_mat_3 = ((WORD8 *)p_mat_2 + row_stride1);

  WORD8 *p_vec = (WORD8*)p_vec_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  out_0 = AE_CVT64F32_H(acc_row0_vec0);
  out_1 = AE_CVT64F32_L(acc_row0_vec0);
  out_2 = AE_CVT64F32_H(acc_row1_vec0);
  out_3 = AE_CVT64F32_L(acc_row1_vec0);
  out_0 = AE_SRAI64(out_0, 32-8);
  out_1 = AE_SRAI64(out_1, 32-8);
  out_2 = AE_SRAI64(out_2, 32-8);
  out_3 = AE_SRAI64(out_3, 32-8);

#if (XCHAL_HAVE_HIFI4) || (XCHAL_HAVE_HIFI3Z) || (XCHAL_HAVE_FUSION)
  for(c_itr = 0; c_itr < cols1 >> 3; c_itr++)
  {
    d_mat0 =  AE_L8X4F_I(p_mat_0, 4);
    d_mat1 =  AE_L8X4F_I(p_mat_1, 4);
    d_mat2 =  AE_L8X4F_I(p_mat_2, 4);
    d_mat3 =  AE_L8X4F_I(p_mat_3, 4);

    d_vec = AE_L8X4F_I(p_vec, 4);
    d_vec = AE_SRAI16(d_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);

    AE_L8X4F_IP(d_mat0,  p_mat_0, 8);
    AE_L8X4F_IP(d_mat1,  p_mat_1, 8);
    AE_L8X4F_IP(d_mat2,  p_mat_2, 8);
    AE_L8X4F_IP(d_mat3,  p_mat_3, 8);

    AE_L8X4F_IP(d_vec, p_vec, 8);
    d_vec = AE_SRAI16(d_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
  }

  /* Remaining 4 elements of multiple of 4 length */
  if((c_itr << 3) < cols1)
  {
    AE_L8X4F_IP(d_mat0,  p_mat_0, 4);
    AE_L8X4F_IP(d_mat1,  p_mat_1, 4);
    AE_L8X4F_IP(d_mat2,  p_mat_2, 4);
    AE_L8X4F_IP(d_mat3,  p_mat_3, 4);
    AE_L8X4F_IP(d_vec, p_vec, 4);
    d_vec = AE_SRAI16(d_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
  }
#else
  for(c_itr = 0; c_itr < cols1 >> 2; c_itr++)
  {
    AE_L8X4F_IP(d_mat0,  p_mat_0, 4);
    AE_L8X4F_IP(d_mat1,  p_mat_1, 4);
    AE_L8X4F_IP(d_mat2,  p_mat_2, 4);
    AE_L8X4F_IP(d_mat3,  p_mat_3, 4);
    AE_L8X4F_IP(d_vec, p_vec, 4);
    d_vec = AE_SRAI16(d_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
  }
#endif

  out_0 = AE_SRAI64(out_0, 8);
  out_1 = AE_SRAI64(out_1, 8);
  out_2 = AE_SRAI64(out_2, 8);
  out_3 = AE_SRAI64(out_3, 8);
  acc_row0_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_0), AE_MOVINT32X2_FROMINT64(out_1));
  acc_row1_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_2), AE_MOVINT32X2_FROMINT64(out_3));
  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
    (ae_int32x2* out_0_0
    ,const WORD8*      p_mat_0
    ,const WORD8*      p_vec_0
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
    d_mat = AE_MOVDA16(*(((WORD8 *)p_mat_0)+c_itr));
    d_vec = AE_MOVDA16(*(((WORD8 *)p_vec_0)+c_itr));
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));
    AE_MULA16X4(d_out, d_tmp, d_mat, d_vec);
  }
  *out_0_0 = d_out;
}

WORD32 xa_nn_matXvec_sym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_mat2,
    const WORD8 * __restrict__ p_vec1,
    const WORD8 * __restrict__ p_vec2,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    WORD32 vec1_zero_bias,
    WORD32 vec2_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
    XA_NNLIB_ARG_CHK_COND((vec2_zero_bias < -127 || vec2_zero_bias > 128), -1);  
  }

  int m_itr=0;
  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);

  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;
  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;

  int bias_flag = 0;
  if(p_bias != NULL)
  {
    bias_flag = 1;
  }

  /* TBD:
   * dual matXvec mul handled in a very basic way: 1 row, 1 vec and unalined 
   * This can be done effectively but currently it is a basic implementaion
   * */
  if((p_mat1 != NULL) && (p_vec1 != NULL) && (p_mat2 != NULL) && (p_vec2 != NULL))
  {
    for (m_itr = 0; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;

      WORD8 *p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
      WORD8 *p_vec1_0 = (WORD8 *)(p_vec1);

      WORD8 *p_mat2_0 = (WORD8 *)(p_mat2+(m_itr * row_stride2));
      WORD8 *p_vec2_0 = (WORD8 *)(p_vec2);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32(p_bias[m_itr]);
      }

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,vec1_zero_bias
        );

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,p_mat2_0
         ,p_vec2_0
         ,cols2
         ,vec2_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
      acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
      acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  }
  else if(((((unsigned)p_mat1) & 3) == 0) && ((((unsigned)p_vec1) & 3) == 0) && ((row_stride1 & 3) == 0) && ((cols1 & 3) == 0))
  {
    const WORD8 *p_mat1_0;
    const WORD8 *p_vec1_0;

    for(m_itr = 0; m_itr < (rows & ~7); m_itr += 8)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      ae_int32x2 acc_row2_vec0 = ZERO32;
      ae_int32x2 acc_row3_vec0 = ZERO32;

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
        acc_row2_vec0 = AE_MOVDA32X2(p_bias[m_itr + 4], p_bias[m_itr + 5]);
        acc_row3_vec0 = AE_MOVDA32X2(p_bias[m_itr + 6], p_bias[m_itr + 7]);
      }

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      _xa_nn_dot_product_8_rows_1_vec_mat_aligned_vec_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,&acc_row2_vec0
         ,&acc_row3_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,row_stride1
         ,vec1_zero_bias
        );
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row2_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row3_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
      acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, AE_MOVDA32(out_zero_bias));
      acc_row2_vec0 = AE_ADD32S(acc_row2_vec0, AE_MOVDA32(out_zero_bias));
      acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, AE_MOVDA32(out_zero_bias));
      acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
      acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
      acc_row1_vec0 = AE_MAX32(acc_row1_vec0, min_int8);
      acc_row1_vec0 = AE_MIN32(acc_row1_vec0, max_int8);
      acc_row2_vec0 = AE_MAX32(acc_row2_vec0, min_int8);
      acc_row2_vec0 = AE_MIN32(acc_row2_vec0, max_int8);
      acc_row3_vec0 = AE_MAX32(acc_row3_vec0, min_int8);
      acc_row3_vec0 = AE_MIN32(acc_row3_vec0, max_int8);

      *p_out++ = (WORD8)AE_MOVAD32_H(acc_row0_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_H(acc_row1_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row1_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_H(acc_row2_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row2_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_H(acc_row3_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row3_vec0);
    }

    for(; m_itr < (rows & ~3); m_itr += 4)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
      }

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      _xa_nn_dot_product_4_rows_1_vec_mat_aligned_vec_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,row_stride1
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
      acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, AE_MOVDA32(out_zero_bias));
      acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
      acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
      acc_row1_vec0 = AE_MAX32(acc_row1_vec0, min_int8);
      acc_row1_vec0 = AE_MIN32(acc_row1_vec0, max_int8);

      *p_out++ = (WORD8)AE_MOVAD32_H(acc_row0_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_H(acc_row1_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row1_vec0);
    }

    /* Compute last (rows % 4) output element */
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;

      p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (WORD8 *)(p_vec1);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32(p_bias[m_itr]);
      }

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
      acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
      acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  }
  else if(((((unsigned)p_vec1) & 3) == 0) && ((row_stride1 & 3) == 0) && ((rows&3) == 0) && ((cols1 & 3) == 0))
  {
    const WORD8 *p_mat1_0;
    const WORD8 *p_vec1_0;

    for(m_itr = 0; m_itr < (rows); m_itr += 4)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
      }

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      _xa_nn_dot_product_4_rows_1_vec_mat_unaligned_vec_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,row_stride1
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
      acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, AE_MOVDA32(out_zero_bias));
      acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
      acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
      acc_row1_vec0 = AE_MAX32(acc_row1_vec0, min_int8);
      acc_row1_vec0 = AE_MIN32(acc_row1_vec0, max_int8);

      *p_out++ = (WORD8)AE_MOVAD32_H(acc_row0_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_H(acc_row1_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row1_vec0);
    }
  }
  else if((p_mat1 != NULL) && (p_vec1 != NULL))
  {
    const WORD8 *p_mat1_0;
    const WORD8 *p_vec1_0;

    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;

      p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (WORD8 *)(p_vec1);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
      }

      _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(WORD8*)p_mat1_0
         ,(WORD8*)p_vec1_0
         ,cols1
         ,row_stride1
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
      acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, AE_MOVDA32(out_zero_bias));
      acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
      acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
      acc_row1_vec0 = AE_MAX32(acc_row1_vec0, min_int8);
      acc_row1_vec0 = AE_MIN32(acc_row1_vec0, max_int8);

      *p_out++ = (WORD8)AE_MOVAD32_H(acc_row0_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_H(acc_row1_vec0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row1_vec0);
    }

    /* Compute last (rows % 4) output element */
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;

      p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (WORD8 *)(p_vec1);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32(p_bias[m_itr]);
      }

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
      acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
      acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  }
  else
  {
    return -1;
  }

  return 0;
}

WORD32 xa_nn_matXvec_out_stride_sym8sxasym8s_16(
    WORD16 * __restrict__ p_out,
    const WORD8  * __restrict__ p_mat1,
    const WORD8  * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 out_stride,
    WORD32 vec1_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);

  /* Iterators used in for loops */
  int m_itr;;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;
  int out_stride_by_2;

  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
  out_stride_by_2 = (out_stride<<1);

  int bias_flag = 0;
  if(p_bias != NULL)
  {
    bias_flag = 1;
  }
 
  if(((((unsigned)p_mat1) & 3) == 0) && ((((unsigned)p_vec1) & 3) == 0) && ((row_stride1 & 7) == 0) && ((cols1 & 3) == 0))
  {
    const WORD8 *p_mat1_0;
    const WORD8 *p_vec1_0;
    ae_int16x4 out16_0;
    ae_int16x4 out16_1;
    
    for(m_itr = 0; m_itr < (rows & ~7); m_itr += 8)
    { 
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      ae_int32x2 acc_row2_vec0 = ZERO32;
      ae_int32x2 acc_row3_vec0 = ZERO32;
      
      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
        acc_row2_vec0 = AE_MOVDA32X2(p_bias[m_itr + 4], p_bias[m_itr + 5]);
        acc_row3_vec0 = AE_MOVDA32X2(p_bias[m_itr + 6], p_bias[m_itr + 7]);
      }

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      _xa_nn_dot_product_8_rows_1_vec_mat_aligned_vec_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,&acc_row2_vec0
         ,&acc_row3_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,row_stride1
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row2_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row3_vec0, out_multiplier, left_shift, right_shift);

      out16_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
      out16_1 = AE_SAT16X4(acc_row2_vec0, acc_row3_vec0);

      AE_S16_0_XP(AE_SEL16_6543(out16_0, out16_0), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_5432(out16_0, out16_0), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_4321(out16_0, out16_0), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(out16_0, (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_6543(out16_1, out16_1), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_5432(out16_1, out16_1), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_4321(out16_1, out16_1), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(out16_1, (ae_int16 *) p_out, out_stride_by_2);
    }

    for(; m_itr < (rows & ~3); m_itr += 4)
    { 
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      
      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
      }

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      _xa_nn_dot_product_4_rows_1_vec_mat_aligned_vec_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,row_stride1
         ,vec1_zero_bias
        );
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);

      out16_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);

      AE_S16_0_XP(AE_SEL16_6543(out16_0, out16_0), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_5432(out16_0, out16_0), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_4321(out16_0, out16_0), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(out16_0, (ae_int16 *) p_out, out_stride_by_2);
    }

    /* Compute last (rows % 4) output element */
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;

      p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (WORD8 *)(p_vec1);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32(p_bias[m_itr]);
      }

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      out16_0 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec0);
      AE_S16_0_XP(out16_0, (ae_int16 *) p_out, out_stride_by_2);
    }
  }
  else if(((((unsigned)p_vec1) & 3) == 0) && ((row_stride1 & 3) == 0) && ((rows&3) == 0) && ((cols1 & 3) == 0))
  {
    const WORD8 *p_mat1_0;
    const WORD8 *p_vec1_0;
    ae_int16x4 out16_0;

    for(m_itr = 0; m_itr < (rows); m_itr += 4)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
      }

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      _xa_nn_dot_product_4_rows_1_vec_mat_unaligned_vec_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,row_stride1
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);

      out16_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);

      AE_S16_0_XP(AE_SEL16_6543(out16_0, out16_0), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_5432(out16_0, out16_0), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_4321(out16_0, out16_0), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(out16_0, (ae_int16 *) p_out, out_stride_by_2);
    }
  }
  else if(p_mat1 && p_vec1)
  {
    WORD8 *p_mat1_0;
    WORD8 *p_vec1_0;
    ae_int16x4 out16_0;
    WORD16* p_dst_0 = (WORD16*)(p_out);
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;

      p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (WORD8 *)(p_vec1);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
      }

      _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(WORD8*)p_mat1_0
         ,(WORD8*)p_vec1_0
         ,cols1
         ,row_stride1
         ,vec1_zero_bias
        );
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);

      out16_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
      AE_S16_0_XP(AE_SEL16_6543(out16_0, out16_0), (ae_int16 *) p_dst_0, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_5432(out16_0, out16_0), (ae_int16 *) p_dst_0, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_4321(out16_0, out16_0), (ae_int16 *) p_dst_0, out_stride_by_2);
      AE_S16_0_XP(out16_0, (ae_int16 *) p_dst_0, out_stride_by_2);
    }
    p_dst_0 = (WORD16*)(p_out + m_itr*out_stride);

    /* Compute last (rows % 4) output element */
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;

      p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (WORD8 *)(p_vec1);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32(p_bias[m_itr]);
      }

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      out16_0 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec0);
      AE_S16_0_XP(out16_0, (ae_int16 *) p_dst_0, out_stride_by_2);
    }
  }
  else
  {
    return -1;
  }
  return 0;
}
