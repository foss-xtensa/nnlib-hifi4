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
#include "xa_nnlib_common_macros.h"

#if XCHAL_HAVE_HIFI1
#define MULTIPLYBYQUANTIZEDMULTIPLIER(inp, multiplier, left_shift, right_shift) \
  inp = AE_SLAA32S(inp, left_shift); \
  inp = AE_MULFP32X2RAS_L(inp, AE_MOVDA32(multiplier)); \
  inp = AE_ROUND32F64SSYM(AE_SRAA64(AE_CVT64F32_L(inp), right_shift));
#endif

#define AE_MINMAX32_HF4(acc, min, max) \
    acc = AE_MAX32(acc, min); \
  acc = AE_MIN32(acc, max);

#if XCHAL_HAVE_HIFI1

#define AE_L8X4S_I_HIFI4(d, ptr, inc) \
    d = AE_L8X4S_I(ptr, inc);

#define AE_S8_FROM32_WITHSTRIDE(val32, dst, stride) \
    *dst = (WORD8)val32; \
  dst += stride;

#else

#define AE_S8_FROM32_WITHSTRIDE(val32, dst, stride) \
    *dst = (WORD8)val32; \
  dst += stride;

#endif

#if XCHAL_HAVE_HIFI1

#if XCHAL_HAVE_HIFI1S
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

  ae_int32x2 outx2_0, outx2_1, outx2_2, outx2_3;
  ae_int32x2 outx2_4, outx2_5, outx2_6, outx2_7;

  outx2_0 = AE_SEL32_HH(0, acc_row0_vec0);
  outx2_1 = AE_SEL32_HL(0, acc_row0_vec0);
  outx2_2 = AE_SEL32_HH(0, acc_row1_vec0);
  outx2_3 = AE_SEL32_HL(0, acc_row1_vec0);
  outx2_4 = AE_SEL32_HH(0, acc_row2_vec0);
  outx2_5 = AE_SEL32_HL(0, acc_row2_vec0);
  outx2_6 = AE_SEL32_HH(0, acc_row3_vec0);
  outx2_7 = AE_SEL32_HL(0, acc_row3_vec0);
  /* 11 cycles. 64 MACs*/
  for(c_itr = 0; c_itr < cols1 >> 3; c_itr++)
  {
    ae_int8x8  d_mat0, d_mat1, d_mat2, d_mat3, d_mat4, d_mat5, d_mat6, d_mat7;
    ae_int16x4 d_vec0, d_vec1;

    AE_L8X4S_IP(d_vec0, p_vec, 4);
    AE_L8X4S_IP(d_vec1, p_vec, 4);
    d_vec0 = AE_ADD16(d_vec0, AE_MOVDA16(vec_zero_bias));
    d_vec1 = AE_ADD16(d_vec1, AE_MOVDA16(vec_zero_bias));

    AE_L8X8_IP(d_mat0,  (ae_int8x8 *)p_mat_0, 8);
    AE_L8X8_IP(d_mat1,  (ae_int8x8 *)p_mat_1, 8);
    AE_L8X8_IP(d_mat2,  (ae_int8x8 *)p_mat_2, 8);
    AE_L8X8_IP(d_mat3,  (ae_int8x8 *)p_mat_3, 8);
    AE_L8X8_IP(d_mat4,  (ae_int8x8 *)p_mat_4, 8);
    AE_L8X8_IP(d_mat5,  (ae_int8x8 *)p_mat_5, 8);
    AE_L8X8_IP(d_mat6,  (ae_int8x8 *)p_mat_6, 8);
    AE_L8X8_IP(d_mat7,  (ae_int8x8 *)p_mat_7, 8);

    AE_MULAAAA16Q8(outx2_0, d_vec0, d_vec1, d_mat0);
    AE_MULAAAA16Q8(outx2_1, d_vec0, d_vec1, d_mat1);
    AE_MULAAAA16Q8(outx2_2, d_vec0, d_vec1, d_mat2);
    AE_MULAAAA16Q8(outx2_3, d_vec0, d_vec1, d_mat3);
    AE_MULAAAA16Q8(outx2_4, d_vec0, d_vec1, d_mat4);
    AE_MULAAAA16Q8(outx2_5, d_vec0, d_vec1, d_mat5);
    AE_MULAAAA16Q8(outx2_6, d_vec0, d_vec1, d_mat6);
    AE_MULAAAA16Q8(outx2_7, d_vec0, d_vec1, d_mat7);
  }
  outx2_0 = AE_ADD32_HL_LH(outx2_0, outx2_0);
  outx2_1 = AE_ADD32_HL_LH(outx2_1, outx2_1);
  outx2_2 = AE_ADD32_HL_LH(outx2_2, outx2_2);
  outx2_3 = AE_ADD32_HL_LH(outx2_3, outx2_3);
  outx2_4 = AE_ADD32_HL_LH(outx2_4, outx2_4);
  outx2_5 = AE_ADD32_HL_LH(outx2_5, outx2_5);
  outx2_6 = AE_ADD32_HL_LH(outx2_6, outx2_6);
  outx2_7 = AE_ADD32_HL_LH(outx2_7, outx2_7);

  out_0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, outx2_0));
  out_1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, outx2_1));
  out_2 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, outx2_2));
  out_3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, outx2_3));
  out_4 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, outx2_4));
  out_5 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, outx2_5));
  out_6 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, outx2_6));
  out_7 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, outx2_7));

  /* Remaining 4 elements of multiple of 4 length */
  if((c_itr << 3) < cols1)
  {
    AE_L8X4S_IP(d_mat0,  p_mat_0, 4);
    AE_L8X4S_IP(d_mat1,  p_mat_1, 4);
    AE_L8X4S_IP(d_mat2,  p_mat_2, 4);
    AE_L8X4S_IP(d_mat3,  p_mat_3, 4);
    AE_L8X4S_IP(d_mat4,  p_mat_4, 4);
    AE_L8X4S_IP(d_mat5,  p_mat_5, 4);
    AE_L8X4S_IP(d_mat6,  p_mat_6, 4);
    AE_L8X4S_IP(d_mat7,  p_mat_7, 4);

    AE_L8X4S_IP(d_vec, p_vec, 4);
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

  acc_row0_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_0), AE_MOVINT32X2_FROMINT64(out_1));
  acc_row1_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_2), AE_MOVINT32X2_FROMINT64(out_3));
  acc_row2_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_4), AE_MOVINT32X2_FROMINT64(out_5));
  acc_row3_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_6), AE_MOVINT32X2_FROMINT64(out_7));
  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
  *out_2_0 = acc_row2_vec0;
  *out_3_0 = acc_row3_vec0;
}
#else /* XCHAL_HAVE_HIFI1S */
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

  out_1 = AE_CVT64F32_L(acc_row0_vec0);
  out_3 = AE_CVT64F32_L(acc_row1_vec0);
  out_5 = AE_CVT64F32_L(acc_row2_vec0);
  out_7 = AE_CVT64F32_L(acc_row3_vec0);

  out_0 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(acc_row0_vec0), 32);
  out_1 = AE_SRAI64(out_1, 32);
  out_2 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(acc_row1_vec0), 32);
  out_3 = AE_SRAI64(out_3, 32);
  out_4 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(acc_row2_vec0), 32);
  out_5 = AE_SRAI64(out_5, 32);
  out_6 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(acc_row3_vec0), 32);
  out_7 = AE_SRAI64(out_7, 32);

  for(c_itr = 0; c_itr < cols1 >> 3; c_itr++)
  {
    ae_int16x4 d_mat01, d_mat11, d_mat21, d_mat31, d_vec1;
    ae_int16x4 d_mat41, d_mat51, d_mat61, d_mat71;

    d_vec = AE_L8X4S_I(p_vec, 4);
    AE_L8X4S_IP(d_vec1, p_vec, 8);

    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));
    d_vec1 = AE_ADD16(d_vec1, AE_MOVDA16(vec_zero_bias));

    d_mat0 =  AE_L8X4S_I(p_mat_0, 4);
    AE_L8X4S_IP(d_mat01,  p_mat_0, 8);
    d_mat1 =  AE_L8X4S_I(p_mat_1, 4);
    AE_L8X4S_IP(d_mat11,  p_mat_1, 8);
    d_mat2 =  AE_L8X4S_I(p_mat_2, 4);
    AE_L8X4S_IP(d_mat21,  p_mat_2, 8);
    d_mat3 =  AE_L8X4S_I(p_mat_3, 4);
    AE_L8X4S_IP(d_mat31,  p_mat_3, 8);
    d_mat4 =  AE_L8X4S_I(p_mat_4, 4);
    AE_L8X4S_IP(d_mat41,  p_mat_4, 8);
    d_mat5 =  AE_L8X4S_I(p_mat_5, 4);
    AE_L8X4S_IP(d_mat51,  p_mat_5, 8);
    d_mat6 =  AE_L8X4S_I(p_mat_6, 4);
    AE_L8X4S_IP(d_mat61,  p_mat_6, 8);
    d_mat7 =  AE_L8X4S_I(p_mat_7, 4);
    AE_L8X4S_IP(d_mat71,  p_mat_7, 8);

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
    AE_L8X4S_IP(d_mat0,  p_mat_0, 4);
    AE_L8X4S_IP(d_mat1,  p_mat_1, 4);
    AE_L8X4S_IP(d_mat2,  p_mat_2, 4);
    AE_L8X4S_IP(d_mat3,  p_mat_3, 4);
    AE_L8X4S_IP(d_mat4,  p_mat_4, 4);
    AE_L8X4S_IP(d_mat5,  p_mat_5, 4);
    AE_L8X4S_IP(d_mat6,  p_mat_6, 4);
    AE_L8X4S_IP(d_mat7,  p_mat_7, 4);

    AE_L8X4S_IP(d_vec, p_vec, 4);
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

  acc_row0_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_0), AE_MOVINT32X2_FROMINT64(out_1));
  acc_row1_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_2), AE_MOVINT32X2_FROMINT64(out_3));
  acc_row2_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_4), AE_MOVINT32X2_FROMINT64(out_5));
  acc_row3_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_6), AE_MOVINT32X2_FROMINT64(out_7));
  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
  *out_2_0 = acc_row2_vec0;
  *out_3_0 = acc_row3_vec0;
}
#endif

#else /* #if XCHAL_HAVE_HIFI1 */
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

  out_1 = AE_CVT64F32_L(acc_row0_vec0);
  out_3 = AE_CVT64F32_L(acc_row1_vec0);
  out_5 = AE_CVT64F32_L(acc_row2_vec0);
  out_7 = AE_CVT64F32_L(acc_row3_vec0);

  out_0 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(acc_row0_vec0), 24);
  out_1 = AE_SRAI64(out_1, 24);
  out_2 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(acc_row1_vec0), 24);
  out_3 = AE_SRAI64(out_3, 24);
  out_4 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(acc_row2_vec0), 24);
  out_5 = AE_SRAI64(out_5, 24);
  out_6 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(acc_row3_vec0), 24);
  out_7 = AE_SRAI64(out_7, 24);

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

#endif /* XCHAL_HAVE_HIFI1 */

#if XCHAL_HAVE_HIFI1S
static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
(ae_int32x2* out_0_0
 ,WORD8*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD32      cols1
 ,WORD32      vec_zero_bias)
{
  int c_itr = 0;
  ae_int8x8 d_mat8, d_vec8;
  ae_int16x4 d_vecn, d_vec, d_mat;

  ae_int32x2 d_acc0 = AE_ZERO32(), d_tmp;
  ae_int32x2 d_out = *out_0_0;

  ae_valign valign_mat_0 = AE_LA64_PP(p_mat_0);
  ae_valign valign_vec_0 = AE_LA64_PP(p_vec_0);
  ae_int8x8 d_vzb8 = AE_MOVDA8(-vec_zero_bias);

  for(;c_itr<(cols1>>3); c_itr++)
  {
    AE_LA8X8_IP(d_mat8, valign_mat_0, (ae_int8x8*)p_mat_0);
    AE_LA8X8_IP(d_vec8, valign_vec_0, (ae_int8x8*)p_vec_0);
    AE_SUBW8(d_vec, d_vecn, d_vec8, d_vzb8);
    AE_MULAAAA16Q8(d_acc0, d_vec, d_vecn, d_mat8);
  }
  d_acc0 = AE_ADD32_HL_LH(d_acc0, d_acc0);

  for(c_itr=0;c_itr<(cols1&7); c_itr++)
  {
    AE_L8S_IP(d_mat, p_mat_0, 1);
    AE_L8S_IP(d_vec, p_vec_0, 1);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));
    AE_MULA16X4(d_out, d_tmp, d_mat, d_vec);
  }
  *out_0_0 = AE_ADD32S(d_out, d_acc0);
}
#else
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
#if XCHAL_HAVE_HIFI1
    AE_L8S_IP(d_mat, p_mat_0, 1);
    AE_L8S_IP(d_vec, p_vec_0, 1);
#else
    d_mat = AE_MOVDA16(*((WORD8 *)p_mat_0));
    d_vec = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_mat_0++;
    p_vec_0++;
#endif
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));
    AE_MULA16X4(d_out, d_tmp, d_mat, d_vec);
  }
  *out_0_0 = d_out;
}
#endif

static inline void __attribute__((always_inline)) _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
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
  ALIGN_REGISTER_TYPE d_vec_la;
  ae_int64 out_0, out_1, out_2, out_3;

  WORD8 *p_vec = (WORD8*)p_vec_0;
  WORD8 *p_mat_1 = ((WORD8 *)p_mat_0 + 4 * row_stride1);
  WORD8 *p_mat_2 = ((WORD8 *)p_mat_1 + 4 * row_stride1);
  WORD8 *p_mat_3 = ((WORD8 *)p_mat_2 + 4 * row_stride1);

  WORD8 *p_mat_0_tmp = (WORD8 *)p_mat_0;
  WORD8 *p_mat_1_tmp = p_mat_1;
  WORD8 *p_mat_2_tmp = p_mat_2;
  WORD8 *p_mat_3_tmp = p_mat_3;
  WORD8 *p_vec_tmp = p_vec;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  WORD32 pre_loop_count=0;
  
  {
    int rem;
    ae_int16x4 d_mat;

    pre_loop_count = 4 - ((uintptr_t)p_mat_0_tmp & 0x3);
    pre_loop_count = (pre_loop_count==4)? 0: pre_loop_count;
    pre_loop_count = (pre_loop_count > cols1) ? cols1 : pre_loop_count;
    cols1 -= pre_loop_count;
    for(rem=0; rem<pre_loop_count; rem++)
    {
      d_mat0 = AE_MOVDA16(*p_mat_0_tmp++);
      d_mat1 = AE_MOVDA16(*p_mat_1_tmp++);
      d_mat2 = AE_MOVDA16(*p_mat_2_tmp++);
      d_mat3 = AE_MOVDA16(*p_mat_3_tmp++);
      d_vec = AE_MOVDA16(*p_vec_tmp++);
      d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));
      d_mat = AE_SEL16_7531(AE_SEL16_7531(d_mat0, d_mat1), AE_SEL16_7531(d_mat2, d_mat3));
      AE_MULA16X4(acc_row0_vec0, acc_row1_vec0, d_mat, d_vec);
    }
  }
  out_0 = AE_CVT64F32_H(acc_row0_vec0);
  out_1 = AE_CVT64F32_L(acc_row0_vec0);
  out_2 = AE_CVT64F32_H(acc_row1_vec0);
  out_3 = AE_CVT64F32_L(acc_row1_vec0);
#if XCHAL_HAVE_HIFI1
  out_0 = AE_SRAI64(out_0, 32);
  out_1 = AE_SRAI64(out_1, 32);
  out_2 = AE_SRAI64(out_2, 32);
  out_3 = AE_SRAI64(out_3, 32);
#else
  out_0 = AE_SRAI64(out_0, 32-8);
  out_1 = AE_SRAI64(out_1, 32-8);
  out_2 = AE_SRAI64(out_2, 32-8);
  out_3 = AE_SRAI64(out_3, 32-8);
#endif

  PRIME_8X4F(p_vec_tmp, d_vec_la);

  for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
  {
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_mat0, p_mat_0_tmp, 4);
    AE_L8X4S_IP(d_mat1, p_mat_1_tmp, 4);
    AE_L8X4S_IP(d_mat2, p_mat_2_tmp, 4);
    AE_L8X4S_IP(d_mat3, p_mat_3_tmp, 4);
    AE_LA8X4S_IP(d_vec, d_vec_la, p_vec_tmp);
#else
    AE_L8X4F_IP(d_mat0, p_mat_0_tmp, 4);
    AE_L8X4F_IP(d_mat1, p_mat_1_tmp, 4);
    AE_L8X4F_IP(d_mat2, p_mat_2_tmp, 4);
    AE_L8X4F_IP(d_mat3, p_mat_3_tmp, 4);
    AE_LA8X4F_IP(d_vec, d_vec_la, p_vec_tmp);
    d_vec = AE_SRAI16(d_vec, 8);
#endif

    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
  }

#if !XCHAL_HAVE_HIFI1
  out_0 = AE_SRAI64(out_0, 8);
  out_1 = AE_SRAI64(out_1, 8);
  out_2 = AE_SRAI64(out_2, 8);
  out_3 = AE_SRAI64(out_3, 8);
#endif
  acc_row0_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_0), AE_MOVINT32X2_FROMINT64(out_1));
  acc_row1_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_2), AE_MOVINT32X2_FROMINT64(out_3));

  {
    int index = cols1&(~0x3);
    int rem;
    ae_int16x4 d_mat;
    WORD8 *p_vec_rem = (WORD8*)&p_vec_0[index+pre_loop_count];
    WORD8 *p_mat_0_rem = (WORD8*)&p_mat_0[index+pre_loop_count];
    WORD8 *p_mat_1_rem = (p_mat_0_rem + 4 * row_stride1);
    WORD8 *p_mat_2_rem = (p_mat_1_rem + 4 * row_stride1);
    WORD8 *p_mat_3_rem = (p_mat_2_rem + 4 * row_stride1);

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

#if XCHAL_HAVE_HIFI1
  d_out0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(AE_ZERO32(), *out_0_0));//AE_SRAI64(AE_CVT64F32_H(*out_0_0), 32);
  d_out1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(AE_ZERO32(), *out_0_0));// AE_SRAI64(AE_CVT64F32_L(*out_0_0), 32);
  d_out2 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(AE_ZERO32(), *out_1_1));//AE_SRAI64(AE_CVT64F32_H(*out_1_1), 32);
  d_out3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(AE_ZERO32(), *out_1_1));//AE_SRAI64(AE_CVT64F32_L(*out_1_1), 32);
#else
  d_out0 = AE_SRAI64(AE_CVT64F32_H(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_H(*out_1_1), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
#endif
#pragma no_unroll
  for(c_itr = 0;c_itr<(cols1>>2); c_itr++)
  {
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_vec0, p_vec_0, 4);
    d_vec0 = AE_ADD16(d_vec0, d_vzb);
    AE_L8X4S_IP(d_mat0, p_mat_0, 4);
    AE_L8X4S_IP(d_mat1, p_mat_1, 4);
    AE_L8X4S_IP(d_mat2, p_mat_2, 4);
    AE_L8X4S_IP(d_mat3, p_mat_3, 4);
#else
    AE_L8X4F_IP(d_vec0, p_vec_0, 4);
    d_vec0 = AE_SRAI16(d_vec0, 8);
    d_vec0 = AE_ADD16(d_vec0, d_vzb);
    AE_L8X4F_IP(d_mat0, p_mat_0, 4);
    AE_L8X4F_IP(d_mat1, p_mat_1, 4);
    AE_L8X4F_IP(d_mat2, p_mat_2, 4);
    AE_L8X4F_IP(d_mat3, p_mat_3, 4);
#endif
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat2, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat3, d_vec0);
  }
#if !XCHAL_HAVE_HIFI1
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  d_out2 = AE_SRAI64(d_out2, 8);
  d_out3 = AE_SRAI64(d_out3, 8);
#endif
  *out_0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0), AE_MOVINT32X2_FROMINT64(d_out1));
  *out_1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out2), AE_MOVINT32X2_FROMINT64(d_out3));
}

#if XCHAL_HAVE_HIFI1S
static inline void _xa_nn_dot_product_4_rows_2_vecs_aligned_person_detect_spc
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_0
 ,ae_int32x2* out_2_1
 ,WORD8*      p_mat_0
 ,WORD32      matstride
 ,WORD8*      p_vec_0
 ,WORD32      vec_offset
 ,WORD32      cols1
 ,WORD32      vec_zero_bias)
{
  int c_itr = 0;
  ae_int8x8 d_mat0, d_mat1, d_mat2, d_mat3;
  ae_int16x4 d_vec0, d_vec0n, d_vec1, d_vec1n;
  ae_int32x2 d_out0, d_out1, d_out2, d_out3;
  ae_int32x2 d_out4, d_out5, d_out6, d_out7;
  
  WORD8 *p_mat_1, *p_mat_2, *p_mat_3;
  WORD8 *p_vec_1;

  p_mat_1 = p_mat_0 + matstride;
  p_mat_2 = p_mat_1 + matstride;
  p_mat_3 = p_mat_2 + matstride;

  p_vec_1 = p_vec_0 + vec_offset;
  
  d_out0 = (AE_SEL32_HH(AE_ZERO32(), *out_0_0));
  d_out1 = (AE_SEL32_LL(AE_ZERO32(), *out_0_0));
  d_out2 = (AE_SEL32_HH(AE_ZERO32(), *out_1_1));
  d_out3 = (AE_SEL32_LL(AE_ZERO32(), *out_1_1));
  d_out4 = (AE_SEL32_HH(AE_ZERO32(), *out_2_0));
  d_out5 = (AE_SEL32_LL(AE_ZERO32(), *out_2_0));
  d_out6 = (AE_SEL32_HH(AE_ZERO32(), *out_2_1));
  d_out7 = (AE_SEL32_LL(AE_ZERO32(), *out_2_1));

  ae_int8x8 d_vec0w, d_vec1w, d_vzb8;
  d_vzb8 = AE_MOVDA8(-vec_zero_bias);

  ae_valign valign_mat_0 = AE_LA64_PP(p_mat_0);
  ae_valign valign_mat_1 = AE_LA64_PP(p_mat_1);
  ae_valign valign_mat_2 = AE_LA64_PP(p_mat_2);
  ae_valign valign_mat_3 = AE_LA64_PP(p_mat_3);
  
  for(c_itr = 0;c_itr<(cols1>>3); c_itr++)
  {
    AE_L8X8_IP(d_vec0w, (ae_int8x8*)p_vec_0, 8);
    AE_SUBW8(d_vec0, d_vec0n, d_vec0w, d_vzb8);

    AE_L8X8_IP(d_vec1w, (ae_int8x8*)p_vec_1, 8);
    AE_SUBW8(d_vec1, d_vec1n, d_vec1w, d_vzb8);
	
    AE_LA8X8_IP(d_mat0, valign_mat_0, (ae_int8x8*)p_mat_0);
    AE_LA8X8_IP(d_mat1, valign_mat_1, (ae_int8x8*)p_mat_1);
    AE_LA8X8_IP(d_mat2, valign_mat_2, (ae_int8x8*)p_mat_2);
    AE_LA8X8_IP(d_mat3, valign_mat_3, (ae_int8x8*)p_mat_3);

    AE_MULAAAA16Q8(d_out0, d_vec0, d_vec0n, d_mat0);
    AE_MULAAAA16Q8(d_out1, d_vec0, d_vec0n, d_mat1);
    AE_MULAAAA16Q8(d_out2, d_vec0, d_vec0n, d_mat2);
    AE_MULAAAA16Q8(d_out3, d_vec0, d_vec0n, d_mat3);
	
    AE_MULAAAA16Q8(d_out4, d_vec1, d_vec1n, d_mat0);
    AE_MULAAAA16Q8(d_out5, d_vec1, d_vec1n, d_mat1);
    AE_MULAAAA16Q8(d_out6, d_vec1, d_vec1n, d_mat2);
    AE_MULAAAA16Q8(d_out7, d_vec1, d_vec1n, d_mat3);
  }

  *out_0_0 = AE_SEL32_HH(AE_ADD32_HL_LH(d_out0, d_out0), AE_ADD32_HL_LH(d_out1, d_out1));
  *out_1_1 = AE_SEL32_HH(AE_ADD32_HL_LH(d_out2, d_out2), AE_ADD32_HL_LH(d_out3, d_out3));
  *out_2_0 = AE_SEL32_HH(AE_ADD32_HL_LH(d_out4, d_out4), AE_ADD32_HL_LH(d_out5, d_out5));
  *out_2_1 = AE_SEL32_HH(AE_ADD32_HL_LH(d_out6, d_out6), AE_ADD32_HL_LH(d_out7, d_out7));
}

static inline void _xa_nn_dot_product_2_rows_2_vecs_unaligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,WORD8*      p_mat_0
 ,WORD32      matstride
 ,WORD8*      p_vec_0
 ,WORD32      vec_offset
 ,WORD32      cols1
 ,WORD32      vec_zero_bias)
{
  int c_itr = 0;
  ae_int8x8 d_mat0, d_mat1;
  ae_int8x8 d_vec0, d_vec1;
  ae_int16x4 d_vec00, d_vec01, d_vec10, d_vec11;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  WORD8 *p_mat_1;
  WORD8 *p_vec_1;
  ae_int8x8 d_vzb;

  d_vzb = AE_MOVDA8(-vec_zero_bias);
  p_mat_1 = p_mat_0 + matstride;
  p_vec_1 = p_vec_0 + vec_offset;

  d_out0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(AE_ZERO32(), *out_0_0));
  d_out1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(AE_ZERO32(), *out_0_0));
  d_out2 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(AE_ZERO32(), *out_1_1));
  d_out3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(AE_ZERO32(), *out_1_1));

  ae_valign vec0_align = AE_LA64_PP(p_vec_0);
  ae_valign vec1_align = AE_LA64_PP(p_vec_1);
  ae_valign mat0_align = AE_LA64_PP(p_mat_0);
  ae_valign mat1_align = AE_LA64_PP(p_mat_1);

  for(c_itr = 0;c_itr<(cols1>>3); c_itr++)
  {
    AE_LA8X8_IP(d_vec0, vec0_align, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(d_vec1, vec1_align, (ae_int8x8 *)p_vec_1);
    AE_SUBW8(d_vec00, d_vec01, d_vec0, d_vzb);
    AE_SUBW8(d_vec10, d_vec11, d_vec1, d_vzb);

    AE_LA8X8_IP(d_mat0, mat0_align, (ae_int8x8 *)p_mat_0);
    AE_LA8X8_IP(d_mat1, mat1_align, (ae_int8x8 *)p_mat_1);

    AE_MULAO8X16(d_out0, d_vec00, d_vec01, d_mat0) ;
    AE_MULAO8X16(d_out1, d_vec00, d_vec01, d_mat1) ;
    AE_MULAO8X16(d_out2, d_vec10, d_vec11, d_mat0) ;
    AE_MULAO8X16(d_out3, d_vec10, d_vec11, d_mat1) ;
  }
  int remcols = cols1%8;
  if(remcols)
  {
    AE_LAV8X8_XP(d_vec0, vec0_align, (ae_int8x8 *)p_vec_0, remcols);
    AE_LAV8X8_XP(d_vec1, vec1_align, (ae_int8x8 *)p_vec_1, remcols);
    AE_SUBW8(d_vec00, d_vec01, d_vec0, d_vzb);
    AE_SUBW8(d_vec10, d_vec11, d_vec1, d_vzb);

    AE_LAV8X8_XP(d_mat0, mat0_align, (ae_int8x8 *)p_mat_0, remcols);
    AE_LAV8X8_XP(d_mat1, mat1_align, (ae_int8x8 *)p_mat_1, remcols);

    AE_MULAO8X16(d_out0, d_vec00, d_vec01, d_mat0) ;
    AE_MULAO8X16(d_out1, d_vec00, d_vec01, d_mat1) ;
    AE_MULAO8X16(d_out2, d_vec10, d_vec11, d_mat0) ;
    AE_MULAO8X16(d_out3, d_vec10, d_vec11, d_mat1) ;
  }

  *out_0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0), AE_MOVINT32X2_FROMINT64(d_out1));
  *out_1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out2), AE_MOVINT32X2_FROMINT64(d_out3));
}

static inline void _xa_nn_dot_product_4_rows_2_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_0
 ,ae_int32x2* out_2_1
 ,WORD8*      p_mat_0
 ,WORD32      matstride
 ,WORD8*      p_vec_0
 ,WORD32      vec_offset
 ,WORD32      cols1
 ,WORD32      vec_zero_bias)
{
  _xa_nn_dot_product_2_rows_2_vecs_unaligned(out_0_0, out_2_0, p_mat_0, matstride, p_vec_0, vec_offset, cols1, vec_zero_bias);
  _xa_nn_dot_product_2_rows_2_vecs_unaligned(out_1_1, out_2_1, (p_mat_0+(2*matstride)), matstride, p_vec_0, vec_offset, cols1, vec_zero_bias);
}
#else
static inline void _xa_nn_dot_product_4_rows_2_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_0
 ,ae_int32x2* out_2_1
 ,WORD8*      p_mat_0
 ,WORD32      matstride
 ,WORD8*      p_vec_0
 ,WORD32      vec_offset
 ,WORD32      cols1
 ,WORD32      vec_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3;
  ae_int16x4 d_vec0, d_vec1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int64 d_out4, d_out5, d_out6, d_out7;
  WORD8 *p_mat_1, *p_mat_2, *p_mat_3;
  WORD8 *p_vec_1;
  ae_int16x4 d_vzb;

  d_vzb = AE_MOVDA16(vec_zero_bias);
  p_mat_1 = p_mat_0 + matstride;
  p_mat_2 = p_mat_1 + matstride;
  p_mat_3 = p_mat_2 + matstride;

  p_vec_1 = p_vec_0 + vec_offset;

#if XCHAL_HAVE_HIFI1
  d_out0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(AE_ZERO32(), *out_0_0));//AE_SRAI64(AE_CVT64F32_H(*out_0_0), 32);
  d_out1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(AE_ZERO32(), *out_0_0));// AE_SRAI64(AE_CVT64F32_L(*out_0_0), 32);
  d_out2 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(AE_ZERO32(), *out_1_1));//AE_SRAI64(AE_CVT64F32_H(*out_1_1), 32);
  d_out3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(AE_ZERO32(), *out_1_1));//AE_SRAI64(AE_CVT64F32_L(*out_1_1), 32);
  d_out4 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(AE_ZERO32(), *out_2_0));//AE_SRAI64(AE_CVT64F32_H(*out_0_0), 32);
  d_out5 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(AE_ZERO32(), *out_2_0));// AE_SRAI64(AE_CVT64F32_L(*out_0_0), 32);
  d_out6 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(AE_ZERO32(), *out_2_1));//AE_SRAI64(AE_CVT64F32_H(*out_1_1), 32);
  d_out7 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(AE_ZERO32(), *out_2_1));//AE_SRAI64(AE_CVT64F32_L(*out_1_1), 32);
#else
  d_out0 = AE_SRAI64(AE_CVT64F32_H(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_H(*out_1_1), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
  d_out4 = AE_SRAI64(AE_CVT64F32_H(*out_2_0), 24);
  d_out5 = AE_SRAI64(AE_CVT64F32_L(*out_2_0), 24);
  d_out6 = AE_SRAI64(AE_CVT64F32_H(*out_2_1), 24);
  d_out7 = AE_SRAI64(AE_CVT64F32_L(*out_2_1), 24);
#endif
#pragma no_unroll
  for(c_itr = 0;c_itr<(cols1>>2); c_itr++)
  {
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_vec0, p_vec_0, 4);
    d_vec0 = AE_ADD16(d_vec0, d_vzb);
    AE_L8X4S_IP(d_vec1, p_vec_1, 4);
    d_vec1 = AE_ADD16(d_vec1, d_vzb);
    AE_L8X4S_IP(d_mat0, p_mat_0, 4);
    AE_L8X4S_IP(d_mat1, p_mat_1, 4);
    AE_L8X4S_IP(d_mat2, p_mat_2, 4);
    AE_L8X4S_IP(d_mat3, p_mat_3, 4);
#else
    AE_L8X4F_IP(d_vec0, p_vec_0, 4);
    d_vec0 = AE_SRAI16(d_vec0, 8);
    d_vec0 = AE_ADD16(d_vec0, d_vzb);
    AE_L8X4F_IP(d_vec1, p_vec_1, 4);
    d_vec1 = AE_SRAI16(d_vec1, 8);
    d_vec1 = AE_ADD16(d_vec1, d_vzb);
    AE_L8X4F_IP(d_mat0, p_mat_0, 4);
    AE_L8X4F_IP(d_mat1, p_mat_1, 4);
    AE_L8X4F_IP(d_mat2, p_mat_2, 4);
    AE_L8X4F_IP(d_mat3, p_mat_3, 4);
#endif
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec0);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec0);
    AE_MULAAAAQ16(d_out2, d_mat2, d_vec0);
    AE_MULAAAAQ16(d_out3, d_mat3, d_vec0);
    AE_MULAAAAQ16(d_out4, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out5, d_mat1, d_vec1);
    AE_MULAAAAQ16(d_out6, d_mat2, d_vec1);
    AE_MULAAAAQ16(d_out7, d_mat3, d_vec1);
  }
#if !XCHAL_HAVE_HIFI1
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  d_out2 = AE_SRAI64(d_out2, 8);
  d_out3 = AE_SRAI64(d_out3, 8);
  d_out4 = AE_SRAI64(d_out4, 8);
  d_out5 = AE_SRAI64(d_out5, 8);
  d_out6 = AE_SRAI64(d_out6, 8);
  d_out7 = AE_SRAI64(d_out7, 8);
#endif
  *out_0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0), AE_MOVINT32X2_FROMINT64(d_out1));
  *out_1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out2), AE_MOVINT32X2_FROMINT64(d_out3));
  *out_2_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out4), AE_MOVINT32X2_FROMINT64(d_out5));
  *out_2_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out6), AE_MOVINT32X2_FROMINT64(d_out7));
}
#endif

WORD32 xa_nn_matmul_v2_per_chan_sym8sxasym8s_asym8s(
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
    WORD32 out_zero_bias,
    WORD32 out_activation_min,
    WORD32 out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
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
  /* MinMax activation range check */
  XA_NNLIB_ARG_CHK_COND((out_activation_min < -128 || out_activation_min > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < -128 || out_activation_max > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);
      
  int itr = 0;
  for(itr=0; itr<rows; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  int m_itr, v_itr;
  int l_shift[4], r_shift[4];
  ae_int32x2 min_int8, max_int8;
  ae_int32x2 acc_row0_vec0, acc_row1_vec0, acc_row3_vec0, acc_row1_vec1, acc_row3_vec1;
  max_int8 = AE_MOVDA32(out_activation_max);
  min_int8 = AE_MOVDA32(out_activation_min);
  acc_row0_vec0 = AE_ZERO32();

  /* Special case for cols == 8 */
  if(
      (cols1 == 8) &&
      (row_stride1 == 8) &&
      (vec_offset == 8) &&
      (((unsigned int)p_mat1 & 0x3) == 0) &&
      (((unsigned int)p_vec1 & 0x3) == 0) &&
      ((rows & 0x1) == 0) &&
      ((vec_count & 0x1) == 0)
    )
#if XCHAL_HAVE_HIFI1S
  {
    (void)min_int8; (void)max_int8;
    WORD8* __restrict__ p_mat1_0 = (WORD8*)&p_mat1[0];

    /* Negating the vec1_zero_bias as it is in the inverse range. i.e. [-127 128] */
    ae_int16x4 d_vec_z_b = AE_MOVDA16(-vec1_zero_bias);

    for(m_itr = 0; m_itr < rows; m_itr+=2)
    {
      WORD8* __restrict__ p_vec_0 = (WORD8*)&p_vec1[0];
      ae_int16x4 d_mat0_0, d_mat1_0, d_mat0_1, d_mat1_1;
      AE_L8X4S_IP(d_mat0_0, p_mat1_0, 4);
      AE_L8X4S_IP(d_mat0_1, p_mat1_0, 4);
      AE_L8X4S_IP(d_mat1_0, p_mat1_0, 4);
      AE_L8X4S_IP(d_mat1_1, p_mat1_0, 4);

      ae_int64 acc_row0, acc_row1;
      acc_row0 = acc_row1 = AE_ZERO64();

      AE_MULAAAAQ16(acc_row0, d_mat0_0, d_vec_z_b);
      AE_MULAAAAQ16(acc_row0, d_mat0_1, d_vec_z_b);
      AE_MULAAAAQ16(acc_row1, d_mat1_0, d_vec_z_b);
      AE_MULAAAAQ16(acc_row1, d_mat1_1, d_vec_z_b);

      acc_row0 = AE_NEG64(acc_row0);
      acc_row1 = AE_NEG64(acc_row1);

      WORD8 * __restrict__ p_dst0   = (WORD8*)p_out + (m_itr * out_stride);
      WORD8 * __restrict__ p_dst1   = p_dst0 + out_stride;

#if TFLITE_SINGLE_ROUNDING
      l_shift[0] = p_out_shift[m_itr+0];
      l_shift[1] = p_out_shift[m_itr+1];
      l_shift[0] = 31 - l_shift[0];
      l_shift[1] = 31 - l_shift[1];
      l_shift[0] = (l_shift[0] << 16) | l_shift[1];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)r_shift[0];
      (void)r_shift[1];
#else /* #if TFLITE_SINGLE_ROUNDING */
      l_shift[0] = p_out_shift[m_itr+0] < 0 ? 0 :  p_out_shift[m_itr+0];
      r_shift[0] = p_out_shift[m_itr+0] > 0 ? 0 : -p_out_shift[m_itr+0];
      l_shift[1] = p_out_shift[m_itr+1] < 0 ? 0 :  p_out_shift[m_itr+1];
      r_shift[1] = p_out_shift[m_itr+1] > 0 ? 0 : -p_out_shift[m_itr+1];
        l_shift[0] = (l_shift[0]<<16) | l_shift[1];
#endif /* #if TFLITE_SINGLE_ROUNDING */
     
      if(p_bias)
      {
        acc_row0 = AE_ADD64S(AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0,AE_MOVDA32(p_bias[m_itr+0]))), acc_row0);
        acc_row1 = AE_ADD64S(AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0,AE_MOVDA32(p_bias[m_itr+1]))), acc_row1);
      }

      ae_valign align_vec = AE_LA64_PP(p_vec_0);
      for(v_itr = 0; v_itr < vec_count; v_itr += 2)
      {
        ae_int64 d_out0_0 = acc_row0;
        ae_int64 d_out1_0 = acc_row1;
        ae_int64 d_out0_1 = acc_row0;
        ae_int64 d_out1_1 = acc_row1;

        ae_int8x8 d_vec0_0, d_vec1_0;

        AE_LA8X8_IP( d_vec0_0, align_vec, (ae_int8x8 *)p_vec_0);
        AE_LA8X8_IP( d_vec1_0, align_vec, (ae_int8x8 *)p_vec_0);		

        AE_MULAO8X16( d_out0_0, d_mat0_0, d_mat0_1, d_vec0_0 );		
        AE_MULAO8X16( d_out0_1, d_mat0_0, d_mat0_1, d_vec1_0 );	
        AE_MULAO8X16( d_out1_0, d_mat1_0, d_mat1_1, d_vec0_0 );
        AE_MULAO8X16( d_out1_1, d_mat1_0, d_mat1_1, d_vec1_0 );

        acc_row1_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0_0), AE_MOVINT32X2_FROMINT64(d_out1_0));
        acc_row3_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0_1), AE_MOVINT32X2_FROMINT64(d_out1_1));

#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1S(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1S(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
#else
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
#endif
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
        ae_int8x8 temp_h_8, temp0_8 = AE_SAT8X4X32_L(acc_row3_vec0, acc_row1_vec0);
        temp0_8 = AE_MIN8(AE_MAX8(temp0_8, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
        temp_h_8 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_4321(AE_MOVINT16X4_FROMINT8X8 (temp0_8), AE_MOVINT16X4_FROMINT8X8 (temp0_8)));
        AE_S8_0_XP(temp0_8, (ae_int8 *)p_dst1, out_offset);
        AE_S8_0_XP(temp_h_8, (ae_int8 *)p_dst1, out_offset);    
        temp0_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp0_8),8));
        temp_h_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp_h_8),8));
        AE_S8_0_XP(temp0_8, (ae_int8 *)p_dst0, out_offset);
        AE_S8_0_XP(temp_h_8, (ae_int8 *)p_dst0, out_offset);    
      }
    }
    return 0;
  }
#else // XCHAL_HAVE_HIFI1S
  {
    WORD8* __restrict__ p_mat1_0 = (WORD8*)&p_mat1[0];

    /* Negating the vec1_zero_bias as it is in the inverse range. i.e. [-127 128] */
#if XCHAL_HAVE_HIFI1
    ae_int16x4 d_vec_z_b = AE_MOVDA16(-vec1_zero_bias);
#else
    ae_int16x4 d_vec_z_b = AE_MOVDA16((-vec1_zero_bias)<<8);
#endif

    for(m_itr = 0; m_itr < rows; m_itr+=2)
    {
      WORD8* __restrict__ p_vec_0 = (WORD8*)&p_vec1[0];
      ae_int16x4 d_mat0_0, d_mat1_0, d_mat0_1, d_mat1_1;
#if XCHAL_HAVE_HIFI1
      AE_L8X4S_IP(d_mat0_0, p_mat1_0, 4);
      AE_L8X4S_IP(d_mat0_1, p_mat1_0, 4);
      AE_L8X4S_IP(d_mat1_0, p_mat1_0, 4);
      AE_L8X4S_IP(d_mat1_1, p_mat1_0, 4);
#else
      AE_L8X4F_IP(d_mat0_0, p_mat1_0, 4);
      AE_L8X4F_IP(d_mat0_1, p_mat1_0, 4);
      AE_L8X4F_IP(d_mat1_0, p_mat1_0, 4);
      AE_L8X4F_IP(d_mat1_1, p_mat1_0, 4);
#endif

      ae_int64 acc_row0, acc_row1;
      acc_row0 = acc_row1 = AE_ZERO64();

      AE_MULAAAAQ16(acc_row0, d_mat0_0, d_vec_z_b);
      AE_MULAAAAQ16(acc_row1, d_mat1_0, d_vec_z_b);
      AE_MULAAAAQ16(acc_row0, d_mat0_1, d_vec_z_b);
      AE_MULAAAAQ16(acc_row1, d_mat1_1, d_vec_z_b);

      WORD8 * __restrict__ p_dst0   = (WORD8*)p_out + (m_itr * out_stride);
      WORD8 * __restrict__ p_dst1   = p_dst0 + out_stride;

#if TFLITE_SINGLE_ROUNDING
      l_shift[0] = p_out_shift[m_itr+0];
      l_shift[1] = p_out_shift[m_itr+1];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)r_shift[0];
      (void)r_shift[1];
#else /* #if TFLITE_SINGLE_ROUNDING */
      l_shift[0] = p_out_shift[m_itr+0] < 0 ? 0 :  p_out_shift[m_itr+0];
      r_shift[0] = p_out_shift[m_itr+0] > 0 ? 0 : -p_out_shift[m_itr+0];
      l_shift[1] = p_out_shift[m_itr+1] < 0 ? 0 :  p_out_shift[m_itr+1];
      r_shift[1] = p_out_shift[m_itr+1] > 0 ? 0 : -p_out_shift[m_itr+1];
#if XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= RI9_HWVERSION)
        l_shift[0] = (l_shift[0]<<16) | l_shift[1];
#endif /* XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= RI9_HWVERSION) */
#endif /* #if TFLITE_SINGLE_ROUNDING */
     
      acc_row0 = AE_NEG64(acc_row0);
      acc_row1 = AE_NEG64(acc_row1);
      if(p_bias)
      {
#if XCHAL_HAVE_HIFI1
        acc_row0 = AE_ADD64S(AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0,AE_MOVDA32(p_bias[m_itr+0]))), acc_row0);
        acc_row1 = AE_ADD64S(AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0,AE_MOVDA32(p_bias[m_itr+1]))), acc_row1);
#else
        acc_row0 = AE_ADD64S((AE_SRAI64(AE_CVT64F32_H(AE_MOVDA32(p_bias[m_itr+0])), 16)), acc_row0);
        acc_row1 = AE_ADD64S((AE_SRAI64(AE_CVT64F32_H(AE_MOVDA32(p_bias[m_itr+1])), 16)), acc_row1);
#endif
      }

#pragma no_unroll
      for(v_itr = 0; v_itr < vec_count; v_itr += 2)
      {
        ae_int64 d_out0_0 = acc_row0;
        ae_int64 d_out1_0 = acc_row1;
        ae_int64 d_out0_1 = acc_row0;
        ae_int64 d_out1_1 = acc_row1;

        ae_int16x4 d_vec0_0, d_vec0_1, d_vec1_0, d_vec1_1;
#if XCHAL_HAVE_HIFI1
        AE_L8X4S_IP(d_vec0_0, p_vec_0, 4);
        AE_L8X4S_IP(d_vec0_1, p_vec_0, 4);
        AE_L8X4S_IP(d_vec1_0, p_vec_0, 4);
        AE_L8X4S_IP(d_vec1_1, p_vec_0, 4);
#else
        AE_L8X4F_IP(d_vec0_0, p_vec_0, 4);
        AE_L8X4F_IP(d_vec0_1, p_vec_0, 4);
        AE_L8X4F_IP(d_vec1_0, p_vec_0, 4);
        AE_L8X4F_IP(d_vec1_1, p_vec_0, 4);
#endif

        AE_MULAAAAQ16(d_out0_0, d_mat0_0, d_vec0_0);
        AE_MULAAAAQ16(d_out1_0, d_mat1_0, d_vec0_0);
        AE_MULAAAAQ16(d_out0_0, d_mat0_1, d_vec0_1);
        AE_MULAAAAQ16(d_out1_0, d_mat1_1, d_vec0_1);
        AE_MULAAAAQ16(d_out0_1, d_mat0_0, d_vec1_0);
        AE_MULAAAAQ16(d_out1_1, d_mat1_0, d_vec1_0);
        AE_MULAAAAQ16(d_out0_1, d_mat0_1, d_vec1_1);
        AE_MULAAAAQ16(d_out1_1, d_mat1_1, d_vec1_1);

#if !XCHAL_HAVE_HIFI1
        d_out0_0 = AE_SRAI64(d_out0_0, 16);
        d_out1_0 = AE_SRAI64(d_out1_0, 16);
        d_out0_1 = AE_SRAI64(d_out0_1, 16);
        d_out1_1 = AE_SRAI64(d_out1_1, 16);
#endif

        acc_row1_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0_0), AE_MOVINT32X2_FROMINT64(d_out1_0));
        acc_row3_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0_1), AE_MOVINT32X2_FROMINT64(d_out1_1));

#if XCHAL_HAVE_HIFI1
        (void)max_int8;(void)min_int8;
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);

        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
        ae_int32x2 acc_row2_vec0;
        acc_row0_vec0 = AE_SEL32_HH(acc_row1_vec0, acc_row1_vec0);
        acc_row1_vec0 = AE_SEL32_LL(acc_row1_vec0, acc_row1_vec0);
        acc_row2_vec0 = AE_SEL32_HH(acc_row3_vec0, acc_row3_vec0);
        acc_row3_vec0 = AE_SEL32_LL(acc_row3_vec0, acc_row3_vec0);

        ae_int16x4 temp01 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        temp01 = AE_MAX16(temp01, AE_MOVDA16(out_activation_min));
        temp01 = AE_MIN16(temp01, AE_MOVDA16(out_activation_max));
        ae_int16x4 temp23 = AE_SAT16X4(acc_row2_vec0, acc_row3_vec0);
        temp23 = AE_MAX16(temp23, AE_MOVDA16(out_activation_min));
        temp23 = AE_MIN16(temp23, AE_MOVDA16(out_activation_max));

        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp01, temp01), (WORD8 *)p_dst0, out_offset);
        AE_S8_0_XP_HIFI1(temp01, (WORD8 *)p_dst1, out_offset);
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp23, temp23), (WORD8 *)p_dst0, out_offset);
        AE_S8_0_XP_HIFI1(temp23, (WORD8 *)p_dst1, out_offset);
#else
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);

        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
        
        AE_MINMAX32_HF4(acc_row1_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row3_vec0, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row1_vec0), p_dst0, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row1_vec0), p_dst1, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row3_vec0), p_dst0, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row3_vec0), p_dst1, out_offset);
#endif
      }
    }
    return 0;
  }
#endif // XCHAL_HAVE_HIFI1S

#if XCHAL_HAVE_HIFI1S
  /*Special case for PD when vector is 8 byte aligned and matrix is 4 byte aligned*/
  else if(((rows&0x7) == 0) && ((cols1&0x7) == 0) && ((row_stride1&0x7) == 0) && (((unsigned int)p_mat1 & 0x3) == 0) 
      && (((unsigned int)p_vec1 & 0x7) == 0) && ((vec_offset & 0x3) ==0))
  {
    for(m_itr = 0; m_itr < rows; m_itr+=4)
    {
      WORD8 * __restrict__ p_mat1_0 = (WORD8*)&p_mat1[m_itr*row_stride1];
      WORD8 * __restrict__ p_dst0   = (WORD8*)p_out + (m_itr * out_stride);
      WORD8 * __restrict__ p_dst1   = p_dst0 + out_stride;
      WORD8 * __restrict__ p_dst2   = p_dst1 + out_stride;
      WORD8 * __restrict__ p_dst3   = p_dst2 + out_stride;

#if TFLITE_SINGLE_ROUNDING
      l_shift[0] = 31 - p_out_shift[m_itr+0];
      l_shift[1] = 31 - p_out_shift[m_itr+1];
      l_shift[2] = 31 - p_out_shift[m_itr+2];
      l_shift[3] = 31 - p_out_shift[m_itr+3];
      l_shift[0] = l_shift[0] << 16 | l_shift[1];
      l_shift[2] = l_shift[2] << 16 | l_shift[3];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)r_shift[0];
      (void)r_shift[1];
      (void)r_shift[2];
      (void)r_shift[3];
#else /* #if TFLITE_SINGLE_ROUNDING */
      l_shift[0] = p_out_shift[m_itr+0] < 0 ? 0 :  p_out_shift[m_itr+0];
      r_shift[0] = p_out_shift[m_itr+0] > 0 ? 0 : -p_out_shift[m_itr+0];
      l_shift[1] = p_out_shift[m_itr+1] < 0 ? 0 :  p_out_shift[m_itr+1];
      r_shift[1] = p_out_shift[m_itr+1] > 0 ? 0 : -p_out_shift[m_itr+1];
      l_shift[2] = p_out_shift[m_itr+2] < 0 ? 0 :  p_out_shift[m_itr+2];
      r_shift[2] = p_out_shift[m_itr+2] > 0 ? 0 : -p_out_shift[m_itr+2];
      l_shift[3] = p_out_shift[m_itr+3] < 0 ? 0 :  p_out_shift[m_itr+3];
      r_shift[3] = p_out_shift[m_itr+3] > 0 ? 0 : -p_out_shift[m_itr+3];
      l_shift[0] = (l_shift[0]<<16) | l_shift[1];
      l_shift[2] = (l_shift[2]<<16) | l_shift[3];
#endif /* #if TFLITE_SINGLE_ROUNDING */
      
      ae_valign bias_valign;
      bias_valign = AE_LA64_PP(p_bias); 

      ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();
      if(p_bias)
      {
        AE_LA32X2_IP(bias_01, bias_valign,(ae_int32x2 *)p_bias);
        AE_LA32X2_IP(bias_23, bias_valign,(ae_int32x2 *)p_bias);
      }
      for(v_itr = 0; v_itr < (vec_count & ~1); v_itr += 2)
      {
        acc_row1_vec0 = bias_01;
        acc_row3_vec0 = bias_23;
        acc_row1_vec1 = bias_01;
        acc_row3_vec1 = bias_23;

        WORD8* __restrict__ p_vec_0 = (WORD8*)(p_vec1 + (v_itr * vec_offset));
        _xa_nn_dot_product_4_rows_2_vecs_aligned_person_detect_spc
          (&acc_row1_vec0
           ,&acc_row3_vec0
           ,&acc_row1_vec1
           ,&acc_row3_vec1
           ,p_mat1_0
           ,row_stride1
           ,p_vec_0
           ,vec_offset
           ,cols1
           ,vec1_zero_bias
          );

#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1S(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1S(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1S(acc_row1_vec1, acc_row1_vec1, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1S(acc_row3_vec1, acc_row3_vec1, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
#else
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(acc_row1_vec1, acc_row1_vec1, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(acc_row3_vec1, acc_row3_vec1, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
#endif        
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
        acc_row1_vec1 = AE_ADD32S(acc_row1_vec1, out_zero_bias);
        acc_row3_vec1 = AE_ADD32S(acc_row3_vec1, out_zero_bias);

        ae_int8x8 temp_h_1_8, temp_h_0_8, temp1_8, temp0_8;
    
        temp0_8 = AE_SAT8X4X32_L(acc_row3_vec0, acc_row1_vec0);
        temp0_8 = AE_MIN8(AE_MAX8(temp0_8, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
        temp1_8 = AE_SAT8X4X32_L(acc_row3_vec1, acc_row1_vec1);
        temp1_8 = AE_MIN8(AE_MAX8(temp1_8, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
        
        temp_h_0_8 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_4321(AE_MOVINT16X4_FROMINT8X8 (temp0_8), AE_MOVINT16X4_FROMINT8X8 (temp0_8)));
        temp_h_1_8 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_4321(AE_MOVINT16X4_FROMINT8X8 (temp1_8), AE_MOVINT16X4_FROMINT8X8 (temp1_8)));
        
        AE_S8_0_XP(temp0_8, (ae_int8 *)p_dst1, out_offset);
        AE_S8_0_XP(temp1_8, (ae_int8 *)p_dst1, out_offset);
        AE_S8_0_XP(temp_h_0_8, (ae_int8 *)p_dst3, out_offset);
        AE_S8_0_XP(temp_h_1_8, (ae_int8 *)p_dst3, out_offset);
        
        temp0_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp0_8),8));
        temp1_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp1_8),8));
        temp_h_0_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp_h_0_8),8));
        temp_h_1_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp_h_1_8),8));
        
        AE_S8_0_XP(temp0_8, (ae_int8 *)p_dst0, out_offset);
        AE_S8_0_XP(temp1_8, (ae_int8 *)p_dst0, out_offset);
        AE_S8_0_XP(temp_h_0_8, (ae_int8 *)p_dst2, out_offset);
        AE_S8_0_XP(temp_h_1_8, (ae_int8 *)p_dst2, out_offset);      
      }
      if(vec_count & 1)
      {
          acc_row1_vec0 = bias_01;
          acc_row3_vec0 = bias_23;

        WORD8* __restrict__ p_vec_0 = (WORD8*)(p_vec1 + (v_itr * vec_offset));
        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row1_vec0
           ,&acc_row3_vec0
           ,p_mat1_0
           ,row_stride1
           ,p_vec_0
           ,cols1
           ,vec1_zero_bias
          );
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1S(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1S(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
#else
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
#endif
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);

        ae_int8x8 temp_h_8, temp0_8 = AE_SAT8X4X32_L(acc_row3_vec0, acc_row1_vec0);
        temp0_8 = AE_MIN8(AE_MAX8(temp0_8, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
        temp_h_8 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_4321(AE_MOVINT16X4_FROMINT8X8 (temp0_8), AE_MOVINT16X4_FROMINT8X8 (temp0_8)));
        AE_S8_0_XP(temp0_8, (ae_int8 *)p_dst1, out_offset);
        AE_S8_0_XP(temp_h_8, (ae_int8 *)p_dst3, out_offset);    
        temp0_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp0_8),8));
        temp_h_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp_h_8),8));
        AE_S8_0_XP(temp0_8, (ae_int8 *)p_dst0, out_offset);
        AE_S8_0_XP(temp_h_8, (ae_int8 *)p_dst2, out_offset);    
      }
    }
  }
#endif  
  else if(((rows&0x3) == 0) && ((cols1&0x3) == 0) && ((row_stride1&0x3) == 0) && (((unsigned int)p_mat1 & 0x3) == 0) 
      && (((unsigned int)p_vec1 & 0x3) == 0) && ((vec_offset & 0x3) ==0))
  {
    for(m_itr = 0; m_itr < rows; m_itr+=4)
    {
      WORD8 * __restrict__ p_mat1_0 = (WORD8*)&p_mat1[m_itr*row_stride1];
      WORD8 * __restrict__ p_dst0   = (WORD8*)p_out + (m_itr * out_stride);
      WORD8 * __restrict__ p_dst1   = p_dst0 + out_stride;
      WORD8 * __restrict__ p_dst2   = p_dst1 + out_stride;
      WORD8 * __restrict__ p_dst3   = p_dst2 + out_stride;

#if TFLITE_SINGLE_ROUNDING
      l_shift[0] = p_out_shift[m_itr+0];
      l_shift[1] = p_out_shift[m_itr+1];
      l_shift[2] = p_out_shift[m_itr+2];
      l_shift[3] = p_out_shift[m_itr+3];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)r_shift[0];
      (void)r_shift[1];
      (void)r_shift[2];
      (void)r_shift[3];
#else /* #if TFLITE_SINGLE_ROUNDING */
      l_shift[0] = p_out_shift[m_itr+0] < 0 ? 0 :  p_out_shift[m_itr+0];
      r_shift[0] = p_out_shift[m_itr+0] > 0 ? 0 : -p_out_shift[m_itr+0];
      l_shift[1] = p_out_shift[m_itr+1] < 0 ? 0 :  p_out_shift[m_itr+1];
      r_shift[1] = p_out_shift[m_itr+1] > 0 ? 0 : -p_out_shift[m_itr+1];
      l_shift[2] = p_out_shift[m_itr+2] < 0 ? 0 :  p_out_shift[m_itr+2];
      r_shift[2] = p_out_shift[m_itr+2] > 0 ? 0 : -p_out_shift[m_itr+2];
      l_shift[3] = p_out_shift[m_itr+3] < 0 ? 0 :  p_out_shift[m_itr+3];
      r_shift[3] = p_out_shift[m_itr+3] > 0 ? 0 : -p_out_shift[m_itr+3];
#if XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= RI9_HWVERSION)
        l_shift[0] = (l_shift[0]<<16) | l_shift[1];
        l_shift[2] = (l_shift[2]<<16) | l_shift[3];
#endif /* XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= RI9_HWVERSION) */
#endif /* #if TFLITE_SINGLE_ROUNDING */
     
      
      ae_valign bias_valign;
      bias_valign = AE_LA64_PP(p_bias); 

      ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();
      if(p_bias)
      {
        AE_LA32X2_IP(bias_01, bias_valign,(ae_int32x2 *)p_bias);
        AE_LA32X2_IP(bias_23, bias_valign,(ae_int32x2 *)p_bias);
      }
      for(v_itr = 0; v_itr < (vec_count & ~1); v_itr += 2)
      {
        acc_row1_vec0 = bias_01;
        acc_row3_vec0 = bias_23;
        acc_row1_vec1 = bias_01;
        acc_row3_vec1 = bias_23;

        WORD8* __restrict__ p_vec_0 = (WORD8*)(p_vec1 + (v_itr * vec_offset));
        _xa_nn_dot_product_4_rows_2_vecs_4bytes_aligned
          (&acc_row1_vec0
           ,&acc_row3_vec0
           ,&acc_row1_vec1
           ,&acc_row3_vec1
           ,p_mat1_0
           ,row_stride1
           ,p_vec_0
           ,vec_offset
           ,cols1
           ,vec1_zero_bias
          );

#if XCHAL_HAVE_HIFI1
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row1_vec1, acc_row1_vec1, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row3_vec1, acc_row3_vec1, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
        
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
        acc_row1_vec1 = AE_ADD32S(acc_row1_vec1, out_zero_bias);
        acc_row3_vec1 = AE_ADD32S(acc_row3_vec1, out_zero_bias);

        ae_int32x2 acc_row2_vec0, acc_row0_vec1, acc_row2_vec1;
        acc_row0_vec0 = AE_SEL32_HH(acc_row1_vec0, acc_row1_vec0);
        acc_row1_vec0 = AE_SEL32_LL(acc_row1_vec0, acc_row1_vec0);
        acc_row2_vec0 = AE_SEL32_HH(acc_row3_vec0, acc_row3_vec0);
        acc_row3_vec0 = AE_SEL32_LL(acc_row3_vec0, acc_row3_vec0);
        acc_row0_vec1 = AE_SEL32_HH(acc_row1_vec1, acc_row1_vec1);
        acc_row1_vec1 = AE_SEL32_LL(acc_row1_vec1, acc_row1_vec1);
        acc_row2_vec1 = AE_SEL32_HH(acc_row3_vec1, acc_row3_vec1);
        acc_row3_vec1 = AE_SEL32_LL(acc_row3_vec1, acc_row3_vec1);
        ae_int16x4 temp01 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        temp01 = AE_MAX16(temp01, AE_MOVDA16(out_activation_min));
        temp01 = AE_MIN16(temp01, AE_MOVDA16(out_activation_max));
        ae_int16x4 temp23 = AE_SAT16X4(acc_row2_vec0, acc_row3_vec0);
        temp23 = AE_MAX16(temp23, AE_MOVDA16(out_activation_min));
        temp23 = AE_MIN16(temp23, AE_MOVDA16(out_activation_max));
        ae_int16x4 temp45 = AE_SAT16X4(acc_row0_vec1, acc_row1_vec1);
        temp45 = AE_MAX16(temp45, AE_MOVDA16(out_activation_min));
        temp45 = AE_MIN16(temp45, AE_MOVDA16(out_activation_max));
        ae_int16x4 temp67 = AE_SAT16X4(acc_row2_vec1, acc_row3_vec1);
        temp67 = AE_MAX16(temp67, AE_MOVDA16(out_activation_min));
        temp67 = AE_MIN16(temp67, AE_MOVDA16(out_activation_max));
        
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp01, temp01), (WORD8 *)p_dst0, out_offset);
        AE_S8_0_XP_HIFI1(temp01, (WORD8 *)p_dst1, out_offset);
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp23, temp23), (WORD8 *)p_dst2, out_offset);
        AE_S8_0_XP_HIFI1(temp23, (WORD8 *)p_dst3, out_offset);
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp45, temp45), (WORD8 *)p_dst0, out_offset);
        AE_S8_0_XP_HIFI1(temp45, (WORD8 *)p_dst1, out_offset);
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp67, temp67), (WORD8 *)p_dst2, out_offset);
        AE_S8_0_XP_HIFI1(temp67, (WORD8 *)p_dst3, out_offset);

#else
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row1_vec1, acc_row1_vec1, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row3_vec1, acc_row3_vec1, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);

        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
        acc_row1_vec1 = AE_ADD32S(acc_row1_vec1, out_zero_bias);
        acc_row3_vec1 = AE_ADD32S(acc_row3_vec1, out_zero_bias);

        AE_MINMAX32_HF4(acc_row1_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row3_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row1_vec1, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row3_vec1, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row1_vec0), p_dst0, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row1_vec0), p_dst1, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row3_vec0), p_dst2, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row3_vec0), p_dst3, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row1_vec1), p_dst0, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row1_vec1), p_dst1, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row3_vec1), p_dst2, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row3_vec1), p_dst3, out_offset);
#endif

      }
      if(vec_count & 1)
      {
        acc_row1_vec0 = bias_01;
        acc_row3_vec0 = bias_23;

        WORD8* __restrict__ p_vec_0 = (WORD8*)(p_vec1 + (v_itr * vec_offset));
        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row1_vec0
           ,&acc_row3_vec0
           ,p_mat1_0
           ,row_stride1
           ,p_vec_0
           ,cols1
           ,vec1_zero_bias
          );

#if XCHAL_HAVE_HIFI1
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);

        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
        ae_int32x2 acc_row2_vec0;
        acc_row0_vec0 = AE_SEL32_HH(acc_row1_vec0, acc_row1_vec0);
        acc_row1_vec0 = AE_SEL32_LL(acc_row1_vec0, acc_row1_vec0);
        acc_row2_vec0 = AE_SEL32_HH(acc_row3_vec0, acc_row3_vec0);
        acc_row3_vec0 = AE_SEL32_LL(acc_row3_vec0, acc_row3_vec0);
        ae_int16x4 temp01 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        temp01 = AE_MAX16(temp01, AE_MOVDA16(out_activation_min));
        temp01 = AE_MIN16(temp01, AE_MOVDA16(out_activation_max));
        ae_int16x4 temp23 = AE_SAT16X4(acc_row2_vec0, acc_row3_vec0);
        temp23 = AE_MAX16(temp23, AE_MOVDA16(out_activation_min));
        temp23 = AE_MIN16(temp23, AE_MOVDA16(out_activation_max));
        
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp01, temp01), (WORD8 *)p_dst0, out_offset);
        AE_S8_0_XP_HIFI1(temp01, (WORD8 *)p_dst1, out_offset);
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp23, temp23), (WORD8 *)p_dst2, out_offset);
        AE_S8_0_XP_HIFI1(temp23, (WORD8 *)p_dst3, out_offset);
#else
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+0], p_out_multiplier[m_itr+1]), l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(p_out_multiplier[m_itr+2], p_out_multiplier[m_itr+3]), l_shift[2], l_shift[3], r_shift[2], r_shift[3]);

        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
        
        AE_MINMAX32_HF4(acc_row1_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row3_vec0, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row1_vec0), p_dst0, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row1_vec0), p_dst1, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row3_vec0), p_dst2, out_offset);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row3_vec0), p_dst3, out_offset);
#endif
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
    m_itr = 0;
    for(; m_itr < rows; m_itr++)
    {
      WORD8 *p_mat1_0 = (WORD8*)&p_mat1[m_itr*row_stride1];
      WORD8 *p_dst0   = (WORD8*)p_out + (m_itr * out_stride);

#if TFLITE_SINGLE_ROUNDING
      l_shift[0] = p_out_shift[m_itr+0];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)r_shift[0];
#else /* #if TFLITE_SINGLE_ROUNDING */
      l_shift[0] = p_out_shift[m_itr+0] < 0 ? 0 :  p_out_shift[m_itr+0];
      r_shift[0] = p_out_shift[m_itr+0] > 0 ? 0 : -p_out_shift[m_itr+0];
#endif /* #if TFLITE_SINGLE_ROUNDING */
     
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

        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[m_itr], l_shift[0], r_shift[0]);

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

WORD32 xa_nn_matmul_sym8sxasym8s_sym16s(
    WORD16 * __restrict__ p_out,
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
    WORD32 out_multiplier,
    WORD32 out_shift)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec_offset < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((out_offset <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);

  int m_itr, v_itr;
  ae_int32x2 acc_row0_vec0, acc_row1_vec0, acc_row3_vec0, acc_row1_vec1, acc_row3_vec1;
  // ae_int32x2 acc_row0_vec0;
  acc_row0_vec0 = AE_ZERO32();
  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  (void)right_shift;
#else
  left_shift = out_shift < 0 ? 0 : out_shift;
  right_shift = out_shift > 0 ? 0 : -out_shift;
#endif

  /* Special case for cols == 8 */
  if(
      (cols1 == 8) &&
      (row_stride1 == 8) &&
      (vec_offset == 8) &&
      (((unsigned int)p_mat1 & 0x3) == 0) &&
      (((unsigned int)p_vec1 & 0x3) == 0) &&
      ((rows & 0x1) == 0) &&
      ((vec_count & 0x1) == 0)
    )
  {
    WORD8* __restrict__ p_mat1_0 = (WORD8*)&p_mat1[0];

#if XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= 281090) && !TFLITE_SINGLE_ROUNDING
    left_shift = (left_shift<<16) | left_shift;
#endif
    /* Negating the vec1_zero_bias as it is in the inverse range. i.e. [-127 128] */
#if XCHAL_HAVE_HIFI1
    ae_int16x4 d_vec_z_b = AE_MOVDA16(-vec1_zero_bias);
#else
    ae_int16x4 d_vec_z_b = AE_MOVDA16((-vec1_zero_bias)<<8);
#endif

    for(m_itr = 0; m_itr < rows; m_itr+=2)
    {
      WORD8* __restrict__ p_vec_0 = (WORD8*)&p_vec1[0];
      ae_int16x4 d_mat0_0, d_mat1_0, d_mat0_1, d_mat1_1;
#if XCHAL_HAVE_HIFI1
      AE_L8X4S_IP(d_mat0_0, p_mat1_0, 4);
      AE_L8X4S_IP(d_mat0_1, p_mat1_0, 4);
      AE_L8X4S_IP(d_mat1_0, p_mat1_0, 4);
      AE_L8X4S_IP(d_mat1_1, p_mat1_0, 4);
#else
      AE_L8X4F_IP(d_mat0_0, p_mat1_0, 4);
      AE_L8X4F_IP(d_mat0_1, p_mat1_0, 4);
      AE_L8X4F_IP(d_mat1_0, p_mat1_0, 4);
      AE_L8X4F_IP(d_mat1_1, p_mat1_0, 4);
#endif

      ae_int64 acc_row0, acc_row1;
      acc_row0 = acc_row1 = AE_ZERO64();

      AE_MULAAAAQ16(acc_row0, d_mat0_0, d_vec_z_b);
      AE_MULAAAAQ16(acc_row1, d_mat1_0, d_vec_z_b);
      AE_MULAAAAQ16(acc_row0, d_mat0_1, d_vec_z_b);
      AE_MULAAAAQ16(acc_row1, d_mat1_1, d_vec_z_b);

      WORD16 * __restrict__ p_dst0   = (WORD16*)p_out + (m_itr * out_stride);
      WORD16 * __restrict__ p_dst1   = p_dst0 + out_stride;
     
      acc_row0 = AE_NEG64(acc_row0);
      acc_row1 = AE_NEG64(acc_row1);
      if(p_bias)
      {
#if XCHAL_HAVE_HIFI1
        acc_row0 = AE_ADD64S(AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0,AE_MOVDA32(p_bias[m_itr+0]))), acc_row0);
        acc_row1 = AE_ADD64S(AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0,AE_MOVDA32(p_bias[m_itr+1]))), acc_row1);
#else
        acc_row0 = AE_ADD64S((AE_SRAI64(AE_CVT64F32_H(AE_MOVDA32(p_bias[m_itr+0])), 16)), acc_row0);
        acc_row1 = AE_ADD64S((AE_SRAI64(AE_CVT64F32_H(AE_MOVDA32(p_bias[m_itr+1])), 16)), acc_row1);
#endif
      }

#pragma no_unroll
      for(v_itr = 0; v_itr < vec_count; v_itr += 2)
      {
        ae_int64 d_out0_0 = acc_row0;
        ae_int64 d_out1_0 = acc_row1;
        ae_int64 d_out0_1 = acc_row0;
        ae_int64 d_out1_1 = acc_row1;

        ae_int16x4 d_vec0_0, d_vec0_1, d_vec1_0, d_vec1_1;
#if XCHAL_HAVE_HIFI1
        AE_L8X4S_IP(d_vec0_0, p_vec_0, 4);
        AE_L8X4S_IP(d_vec0_1, p_vec_0, 4);
        AE_L8X4S_IP(d_vec1_0, p_vec_0, 4);
        AE_L8X4S_IP(d_vec1_1, p_vec_0, 4);
#else
        AE_L8X4F_IP(d_vec0_0, p_vec_0, 4);
        AE_L8X4F_IP(d_vec0_1, p_vec_0, 4);
        AE_L8X4F_IP(d_vec1_0, p_vec_0, 4);
        AE_L8X4F_IP(d_vec1_1, p_vec_0, 4);
#endif

        AE_MULAAAAQ16(d_out0_0, d_mat0_0, d_vec0_0);
        AE_MULAAAAQ16(d_out1_0, d_mat1_0, d_vec0_0);
        AE_MULAAAAQ16(d_out0_0, d_mat0_1, d_vec0_1);
        AE_MULAAAAQ16(d_out1_0, d_mat1_1, d_vec0_1);
        AE_MULAAAAQ16(d_out0_1, d_mat0_0, d_vec1_0);
        AE_MULAAAAQ16(d_out1_1, d_mat1_0, d_vec1_0);
        AE_MULAAAAQ16(d_out0_1, d_mat0_1, d_vec1_1);
        AE_MULAAAAQ16(d_out1_1, d_mat1_1, d_vec1_1);

#if !XCHAL_HAVE_HIFI1
        d_out0_0 = AE_SRAI64(d_out0_0, 16);
        d_out1_0 = AE_SRAI64(d_out1_0, 16);
        d_out0_1 = AE_SRAI64(d_out0_1, 16);
        d_out1_1 = AE_SRAI64(d_out1_1, 16);
#endif

        acc_row1_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0_0), AE_MOVINT32X2_FROMINT64(d_out1_0));
        acc_row3_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0_1), AE_MOVINT32X2_FROMINT64(d_out1_1));

#if XCHAL_HAVE_HIFI1
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
#endif
        ae_int16x4 out_16_0 = AE_SAT16X4(acc_row1_vec0, acc_row3_vec0);
        AE_S16_0_XP(AE_SEL16_6543(out_16_0, out_16_0), (ae_int16*)p_dst0, out_offset<<1);
        AE_S16_0_XP(AE_SEL16_5432(out_16_0, out_16_0), (ae_int16*)p_dst1, out_offset<<1);
        AE_S16_0_XP(AE_SEL16_4321(out_16_0, out_16_0), (ae_int16*)p_dst0, out_offset<<1);
        AE_S16_0_XP(out_16_0, (ae_int16*)p_dst1, out_offset<<1);
      }
    }
    return 0;
  }
#if XCHAL_HAVE_HIFI1S 
  if(((((unsigned)p_mat1) & 7) == 0) && ((((unsigned)p_vec1) & 3) == 0) && ((row_stride1 & 7) == 0) && ((cols1 & 3) == 0) && (vec_count == 1))
#else
  if(((((unsigned)p_mat1) & 3) == 0) && ((((unsigned)p_vec1) & 3) == 0) && ((row_stride1 & 7) == 0) && ((cols1 & 3) == 0) && (vec_count == 1))
#endif  
  {
    /* special case for vec_count=1, as it allows more row unroll */   
    int out_stride_by_2 = (out_stride<<1);

    int bias_flag = 0;
    ae_valign bias_valign;
    if(p_bias != NULL)
    {
      bias_valign = AE_LA64_PP(p_bias);
      bias_flag = 1;
    }

    WORD8 *p_mat1_0;
    WORD8 *p_vec1_0;
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
          AE_LA32X2_IP(acc_row0_vec0, bias_valign, (ae_int32x2 *)p_bias);
          AE_LA32X2_IP(acc_row1_vec0, bias_valign, (ae_int32x2 *)p_bias);
          AE_LA32X2_IP(acc_row2_vec0, bias_valign, (ae_int32x2 *)p_bias);
          AE_LA32X2_IP(acc_row3_vec0, bias_valign, (ae_int32x2 *)p_bias);
      }

      p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (WORD8 *)(p_vec1);

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

      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row2_vec0, acc_row2_vec0, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row3_vec0, acc_row3_vec0, out_multiplier, left_shift, right_shift);

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
          AE_LA32X2_IP(acc_row0_vec0, bias_valign, (ae_int32x2 *)p_bias);
          AE_LA32X2_IP(acc_row1_vec0, bias_valign, (ae_int32x2 *)p_bias);  
      }

      p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (WORD8 *)(p_vec1);

       _xa_nn_dot_product_4_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,p_mat1_0
         ,row_stride1
         ,p_vec1_0
         ,cols1
         ,vec1_zero_bias
        );
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);

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
        AE_L32_IP(acc_row0_vec0, (ae_int32 *) p_bias, 4);
      }

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,vec1_zero_bias
        );

      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      out16_0 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec0);
      AE_S16_0_XP(out16_0, (ae_int16 *) p_out, out_stride_by_2);
    }

    return 0;
  }
  else if(((rows&0x3) == 0) && ((cols1&0x3) == 0) && ((row_stride1&0x3) == 0) && (((unsigned int)p_mat1 & 0x3) == 0) 
      && (((unsigned int)p_vec1 & 0x3) == 0) && ((vec_offset & 0x3) ==0))
  {

#if XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= 281090) && !TFLITE_SINGLE_ROUNDING
    left_shift = (left_shift<<16) | left_shift;
#endif
    for(m_itr = 0; m_itr < rows; m_itr+=4)
    {
      WORD8 * __restrict__ p_mat1_0 = (WORD8*)&p_mat1[m_itr*row_stride1];
      WORD16 * __restrict__ p_dst0   = (WORD16*)p_out + (m_itr * out_stride);
      WORD16 * __restrict__ p_dst1   = p_dst0 + out_stride;
      WORD16 * __restrict__ p_dst2   = p_dst1 + out_stride;
      WORD16 * __restrict__ p_dst3   = p_dst2 + out_stride;

      ae_valign bias_valign;
      bias_valign = AE_LA64_PP(p_bias); 

      ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();
      if(p_bias)
      {
        AE_LA32X2_IP(bias_01, bias_valign,(ae_int32x2 *)p_bias);
        AE_LA32X2_IP(bias_23, bias_valign,(ae_int32x2 *)p_bias);
      }
      
      for(v_itr = 0; v_itr < (vec_count & ~1); v_itr += 2)
      {
        acc_row1_vec0 = bias_01;
        acc_row3_vec0 = bias_23;
        acc_row1_vec1 = bias_01;
        acc_row3_vec1 = bias_23;
        WORD8* __restrict__ p_vec_0 = (WORD8*)(p_vec1 + (v_itr * vec_offset));
        _xa_nn_dot_product_4_rows_2_vecs_4bytes_aligned
          (&acc_row1_vec0
           ,&acc_row3_vec0
           ,&acc_row1_vec1
           ,&acc_row3_vec1
           ,p_mat1_0
           ,row_stride1
           ,p_vec_0
           ,vec_offset
           ,cols1
           ,vec1_zero_bias
          );
          
#if XCHAL_HAVE_HIFI1
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row1_vec1, acc_row1_vec1, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row3_vec1, acc_row3_vec1, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row1_vec1, acc_row1_vec1, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row3_vec1, acc_row3_vec1, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
#endif
        ae_int16x4 out_16_0 = AE_SAT16X4(acc_row1_vec0, acc_row3_vec0);
        AE_S16_0_XP(AE_SEL16_6543(out_16_0, out_16_0), (ae_int16*)p_dst0, out_offset<<1);
        AE_S16_0_XP(AE_SEL16_5432(out_16_0, out_16_0), (ae_int16*)p_dst1, out_offset<<1);
        AE_S16_0_XP(AE_SEL16_4321(out_16_0, out_16_0), (ae_int16*)p_dst2, out_offset<<1);
        AE_S16_0_XP(out_16_0, (ae_int16*)p_dst3, out_offset<<1);
        out_16_0 = AE_SAT16X4(acc_row1_vec1, acc_row3_vec1);
        AE_S16_0_XP(AE_SEL16_6543(out_16_0, out_16_0), (ae_int16*)p_dst0, out_offset<<1);
        AE_S16_0_XP(AE_SEL16_5432(out_16_0, out_16_0), (ae_int16*)p_dst1, out_offset<<1);
        AE_S16_0_XP(AE_SEL16_4321(out_16_0, out_16_0), (ae_int16*)p_dst2, out_offset<<1);
        AE_S16_0_XP(out_16_0, (ae_int16*)p_dst3, out_offset<<1);
      }
      if(vec_count & 1)
      {
        acc_row1_vec0 = bias_01;
        acc_row3_vec0 = bias_23;

        WORD8* __restrict__ p_vec_0 = (WORD8*)(p_vec1 + (v_itr * vec_offset));
        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row1_vec0
           ,&acc_row3_vec0
           ,p_mat1_0
           ,row_stride1
           ,p_vec_0
           ,cols1
           ,vec1_zero_bias
          );

#if XCHAL_HAVE_HIFI1
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_HIFI1(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row1_vec0, acc_row1_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
        MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_SHIFT(acc_row3_vec0, acc_row3_vec0, AE_MOVDA32X2(out_multiplier, out_multiplier), left_shift, left_shift, right_shift, right_shift);
#endif
        ae_int16x4 out_16_0 = AE_SAT16X4(acc_row1_vec0, acc_row3_vec0);
        AE_S16_0_XP(AE_SEL16_6543(out_16_0, out_16_0), (ae_int16*)p_dst0, out_offset<<1);
        AE_S16_0_XP(AE_SEL16_5432(out_16_0, out_16_0), (ae_int16*)p_dst1, out_offset<<1);
        AE_S16_0_XP(AE_SEL16_4321(out_16_0, out_16_0), (ae_int16*)p_dst2, out_offset<<1);
        AE_S16_0_XP(out_16_0, (ae_int16*)p_dst3, out_offset<<1);
      }
    }
  }
  else 
  if(p_mat1 && p_vec1)
  {
    WORD8 * __restrict__ p_mat1_0;
    WORD16 * __restrict__ p_dst0;
    WORD16 * __restrict__ p_dst1;
    WORD16 * __restrict__ p_dst2;
    WORD16 * __restrict__ p_dst3;

    ae_int32x2 bias_01, bias_23;
    int ii;
    for(m_itr = 0; m_itr < (rows & ~(16 - 1)) ; m_itr += 16)
    {
      for(ii = 0; ii < 4; ii++)
      {
        p_mat1_0 = (WORD8 *)(p_mat1+((m_itr + ii) * row_stride1));
        p_dst0   = (WORD16*)p_out + ((m_itr + ii) * out_stride);
        p_dst1   = p_dst0 + 4 * out_stride;
        p_dst2   = p_dst1 + 4 * out_stride;
        p_dst3   = p_dst2 + 4 * out_stride;

        bias_01 = AE_ZERO32();
        bias_23 = AE_ZERO32();
        if(p_bias)
        {
          bias_01 = AE_MOVDA32X2(p_bias[m_itr + ii + 0], p_bias[m_itr + ii + 4]);
          bias_23 = AE_MOVDA32X2(p_bias[m_itr + ii + 8], p_bias[m_itr + ii + 12]);
        }

        for(v_itr = 0; v_itr < vec_count; v_itr++)
        {
          acc_row0_vec0 = bias_01;
          acc_row1_vec0 = bias_23;
          WORD8* __restrict__ p_vec1_0 = (WORD8*)(p_vec1 + (v_itr * vec_offset));

          _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
            (&acc_row0_vec0
            ,&acc_row1_vec0
            ,(WORD8*)p_mat1_0
            ,(WORD8*)p_vec1_0
            ,cols1
            ,row_stride1
            ,vec1_zero_bias
            );
          MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
          MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);

          ae_int16x4 out_16_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
          AE_S16_0_XP(AE_SEL16_6543(out_16_0, out_16_0), (ae_int16*)p_dst0, out_offset<<1);
          AE_S16_0_XP(AE_SEL16_5432(out_16_0, out_16_0), (ae_int16*)p_dst1, out_offset<<1);
          AE_S16_0_XP(AE_SEL16_4321(out_16_0, out_16_0), (ae_int16*)p_dst2, out_offset<<1);
          AE_S16_0_XP(out_16_0, (ae_int16*)p_dst3, out_offset<<1);
        }
      }
    }

    /* Compute last (rows % 16) output element */
    for(; m_itr < rows; m_itr++)
    {
      p_mat1_0 = (WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_dst0   = (WORD16*)p_out + (m_itr * out_stride);
     
      for(v_itr = 0; v_itr < vec_count; v_itr++)
      {
        acc_row0_vec0 = AE_ZERO32();
        if(p_bias)
          acc_row0_vec0 = AE_MOVDA32(p_bias[m_itr]);

        WORD8* __restrict__ p_vec1_0 = (WORD8*)(p_vec1 + (v_itr * vec_offset));

        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,p_mat1_0
           ,p_vec1_0
           ,cols1
           ,vec1_zero_bias
          );

        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);

        AE_S16_0_XP(AE_SAT16X4(acc_row0_vec0, acc_row0_vec0), (ae_int16*)p_dst0, out_offset<<1);
      }
    }
  }
  else
    return -1;
  return 0;
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
  WORD32 ret;
  ret = xa_nn_matmul_v2_per_chan_sym8sxasym8s_asym8s(
            p_out, p_mat1, p_vec1, p_bias, rows, cols1, row_stride1, vec_count, vec_offset,
            out_offset, out_stride, vec1_zero_bias, p_out_multiplier, p_out_shift, out_zero_bias, -128, 127, NULL);
  return ret;
}
