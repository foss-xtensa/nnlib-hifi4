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
#include "xa_nnlib_common_macros.h"

typedef void (*_dot_prod_1_rows_1_vecs_kernel)(
      ae_int32x2* out,
      const WORD8* p_mat,
      const WORD8* p_vec,
      WORD32 cols1,
      WORD32 vec_zero_bias);

typedef void (*_dot_prod_4_rows_1_vecs_kernel)(
    ae_int32x2*  out_0_0,
    ae_int32x2*  out_1_0,
    const WORD8* p_mat_0,
    const WORD8* p_vec_0,
    WORD32       cols1,
    WORD32       row_stride1,
    WORD32       vec_zero_bias);


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
  ALIGN_REGISTER_TYPE d_mat0_la, d_mat1_la, d_mat2_la, d_mat3_la;
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

  PRIME_8X4F(p_mat_0_tmp, d_mat0_la);
  PRIME_8X4F(p_mat_1_tmp, d_mat1_la);
  PRIME_8X4F(p_mat_2_tmp, d_mat2_la);
  PRIME_8X4F(p_mat_3_tmp, d_mat3_la);

  /* 4 cols at a time */
  for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
  {
#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(d_mat0, d_mat0_la, p_mat_0_tmp);
    AE_LA8X4S_IP(d_mat1, d_mat1_la, p_mat_1_tmp);
    AE_LA8X4S_IP(d_mat2, d_mat2_la, p_mat_2_tmp);
    AE_LA8X4S_IP(d_mat3, d_mat3_la, p_mat_3_tmp);
    AE_L8X4S_IP(d_vec, p_vec, 4);
#else
    AE_LA8X4F_IP(d_mat0, d_mat0_la, p_mat_0_tmp);
    AE_LA8X4F_IP(d_mat1, d_mat1_la, p_mat_1_tmp);
    AE_LA8X4F_IP(d_mat2, d_mat2_la, p_mat_2_tmp);
    AE_LA8X4F_IP(d_mat3, d_mat3_la, p_mat_3_tmp);
    AE_L8X4F_IP(d_vec, p_vec, 4);
    d_vec = AE_SRAI16(d_vec, 8);
#endif

    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
  }

  /* remaining columns */
  if(c_itr*4 < cols1) {

    int rem_cols = cols1 - c_itr*4;

    xtbool4 rem_cols_sel = AE_LE16(AE_MOVINT16X4_FROMINT64(0x0001000200030004),rem_cols);

#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(d_mat0, d_mat0_la, p_mat_0_tmp);
    AE_LA8X4S_IP(d_mat1, d_mat1_la, p_mat_1_tmp);
    AE_LA8X4S_IP(d_mat2, d_mat2_la, p_mat_2_tmp);
    AE_LA8X4S_IP(d_mat3, d_mat3_la, p_mat_3_tmp);
    AE_L8X4S_IP(d_vec, p_vec, 4);
#else
    AE_LA8X4F_IP(d_mat0, d_mat0_la, p_mat_0_tmp);
    AE_LA8X4F_IP(d_mat1, d_mat1_la, p_mat_1_tmp);
    AE_LA8X4F_IP(d_mat2, d_mat2_la, p_mat_2_tmp);
    AE_LA8X4F_IP(d_mat3, d_mat3_la, p_mat_3_tmp);
    AE_L8X4F_IP(d_vec, p_vec, 4);
    d_vec = AE_SRAI16(d_vec, 8);
#endif
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    // nullify the trailing elements in d_vec to ignore trailing columns
    AE_MOVF16X4(d_vec, AE_MOVDA16(0), rem_cols_sel);

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

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_4_rows_1_vec_mat_aligned_vec_unaligned
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
  ALIGN_REGISTER_TYPE d_vec_la;
  ae_int64 out_0, out_1, out_2, out_3;

  WORD8 *p_mat_1 = ((WORD8 *)p_mat_0 + 1*row_stride1);
  WORD8 *p_mat_2 = ((WORD8 *)p_mat_0 + 2*row_stride1);
  WORD8 *p_mat_3 = ((WORD8 *)p_mat_0 + 3*row_stride1);

  WORD8 *p_vec_tmp = (WORD8*)p_vec_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

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

  /* 4 columns at a time */
  for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
  {
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_mat0, p_mat_0, 4);
    AE_L8X4S_IP(d_mat1, p_mat_1, 4);
    AE_L8X4S_IP(d_mat2, p_mat_2, 4);
    AE_L8X4S_IP(d_mat3, p_mat_3, 4);
    AE_LA8X4S_IP(d_vec, d_vec_la, p_vec_tmp);
#else
    AE_L8X4F_IP(d_mat0, p_mat_0, 4);
    AE_L8X4F_IP(d_mat1, p_mat_1, 4);
    AE_L8X4F_IP(d_mat2, p_mat_2, 4);
    AE_L8X4F_IP(d_mat3, p_mat_3, 4);
    AE_LA8X4F_IP(d_vec, d_vec_la, p_vec_tmp);
    d_vec = AE_SRAI16(d_vec, 8);
#endif
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
  }

  /* remaining columns */
  if(c_itr*4 < cols1) {

    int rem_cols = cols1 - c_itr*4;

    xtbool4 rem_cols_sel = AE_LE16(AE_MOVINT16X4_FROMINT64(0x0001000200030004),rem_cols);

#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_mat0, p_mat_0, 4);
    AE_L8X4S_IP(d_mat1, p_mat_1, 4);
    AE_L8X4S_IP(d_mat2, p_mat_2, 4);
    AE_L8X4S_IP(d_mat3, p_mat_3, 4);
    AE_LA8X4S_IP(d_vec, d_vec_la, p_vec_tmp);
#else
    AE_L8X4F_IP(d_mat0, p_mat_0, 4);
    AE_L8X4F_IP(d_mat1, p_mat_1, 4);
    AE_L8X4F_IP(d_mat2, p_mat_2, 4);
    AE_L8X4F_IP(d_mat3, p_mat_3, 4);
    AE_LA8X4F_IP(d_vec, d_vec_la, p_vec_tmp);
    d_vec = AE_SRAI16(d_vec, 8);
#endif
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    // nullify the trailing elements in d_vec to ignore trailing columns
    AE_MOVF16X4(d_vec, AE_MOVDA16(0), rem_cols_sel);

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

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

// TBD. Regression doesn't cover this case and testsdon't fail.
static inline void _xa_nn_dot_product_4_rows_1_vec_mat_unaligned_vec_unaligned
    (ae_int32x2*  out_0_0
    ,ae_int32x2*  out_1_0
    ,const WORD8* p_mat_0
    ,const WORD8* p_vec_0
    ,WORD32       cols1
    ,WORD32       row_stride1
    ,WORD32       vec_zero_bias)
{

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
  ALIGN_REGISTER_TYPE d_mat0_la, d_mat1_la, d_mat2_la, d_mat3_la, d_vec_la;
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

  PRIME_8X4F(p_mat_0_tmp, d_mat0_la);
  PRIME_8X4F(p_mat_1_tmp, d_mat1_la);
  PRIME_8X4F(p_mat_2_tmp, d_mat2_la);
  PRIME_8X4F(p_mat_3_tmp, d_mat3_la);
  PRIME_8X4F(p_vec_tmp, d_vec_la);

  for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
  {
#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(d_mat0, d_mat0_la, p_mat_0_tmp);
    AE_LA8X4S_IP(d_mat1, d_mat1_la, p_mat_1_tmp);
    AE_LA8X4S_IP(d_mat2, d_mat2_la, p_mat_2_tmp);
    AE_LA8X4S_IP(d_mat3, d_mat3_la, p_mat_3_tmp);
    AE_LA8X4S_IP(d_vec, d_vec_la, p_vec_tmp);
#else
    AE_LA8X4F_IP(d_mat0, d_mat0_la, p_mat_0_tmp);
    AE_LA8X4F_IP(d_mat1, d_mat1_la, p_mat_1_tmp);
    AE_LA8X4F_IP(d_mat2, d_mat2_la, p_mat_2_tmp);
    AE_LA8X4F_IP(d_mat3, d_mat3_la, p_mat_3_tmp);
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

#if XCHAL_HAVE_HIFI1
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

  out_0 = AE_SRAI64(out_0, 32);
  out_1 = AE_SRAI64(out_1, 32);
  out_2 = AE_SRAI64(out_2, 32);
  out_3 = AE_SRAI64(out_3, 32);
  out_4 = AE_SRAI64(out_4, 32);
  out_5 = AE_SRAI64(out_5, 32);
  out_6 = AE_SRAI64(out_6, 32);
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
  out_0 = AE_SRAI64(out_0, 32);
  out_1 = AE_SRAI64(out_1, 32);
  out_2 = AE_SRAI64(out_2, 32);
  out_3 = AE_SRAI64(out_3, 32);


  /* 8 columns at a time */
  for(c_itr = 0; c_itr < cols1 >> 3; c_itr++)
  {
    d_mat0 =  AE_L8X4S_I(p_mat_0, 4);
    d_mat1 =  AE_L8X4S_I(p_mat_1, 4);
    d_mat2 =  AE_L8X4S_I(p_mat_2, 4);
    d_mat3 =  AE_L8X4S_I(p_mat_3, 4);

    d_vec = AE_L8X4S_I(p_vec, 4);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);

    AE_L8X4S_IP(d_mat0,  p_mat_0, 8);
    AE_L8X4S_IP(d_mat1,  p_mat_1, 8);
    AE_L8X4S_IP(d_mat2,  p_mat_2, 8);
    AE_L8X4S_IP(d_mat3,  p_mat_3, 8);

    AE_L8X4S_IP(d_vec, p_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
  }

  /* Remaining 4 columns of multiple of 4 length */
  c_itr *= 8;
  if( (c_itr + 4) <= cols1)
  {
    AE_L8X4S_IP(d_mat0,  p_mat_0, 4);
    AE_L8X4S_IP(d_mat1,  p_mat_1, 4);
    AE_L8X4S_IP(d_mat2,  p_mat_2, 4);
    AE_L8X4S_IP(d_mat3,  p_mat_3, 4);
    AE_L8X4S_IP(d_vec, p_vec, 4);

    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);

    c_itr += 4;
  }

  /* Remaining columns (less than 4)*/
  if(c_itr < cols1){

    int rem_cols = cols1 - c_itr;
    
    xtbool4 rem_col_sel = AE_LE16(AE_MOVINT16X4_FROMINT64(0x0001000200030004),rem_cols);

    AE_L8X4S_IP(d_mat0,  p_mat_0, 4);
    AE_L8X4S_IP(d_mat1,  p_mat_1, 4);
    AE_L8X4S_IP(d_mat2,  p_mat_2, 4);
    AE_L8X4S_IP(d_mat3,  p_mat_3, 4);
    AE_L8X4S_IP(d_vec, p_vec, 4);

    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    // nullify the trailing elements in d_vec to ignore trailing columns
    AE_MOVF16X4(d_vec, AE_MOVDA16(0), rem_col_sel);

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);
  }
  acc_row0_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_0), AE_MOVINT32X2_FROMINT64(out_1));
  acc_row1_vec0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(out_2), AE_MOVINT32X2_FROMINT64(out_3));
  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

#else
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

  /* 8 columns at a time */
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

  /* Remaining 4 columns of multiple of 4 length */
  c_itr *= 8;
  if( (c_itr + 4) <= cols1)
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

    c_itr += 4;
  }

  /* Remaining columns (less than 4)*/
  if(c_itr < cols1){

    int rem_cols = cols1 - c_itr;

    xtbool4 rem_col_sel = AE_LE16(AE_MOVINT16X4_FROMINT64(0x0001000200030004),rem_cols);

    AE_L8X4F_IP(d_mat0,  p_mat_0, 4);
    AE_L8X4F_IP(d_mat1,  p_mat_1, 4);
    AE_L8X4F_IP(d_mat2,  p_mat_2, 4);
    AE_L8X4F_IP(d_mat3,  p_mat_3, 4);
    AE_L8X4F_IP(d_vec, p_vec, 4);

    d_vec = AE_SRAI16(d_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    // nullify the trailing elements in d_vec to ignore trailing columns
    AE_MOVF16X4(d_vec, AE_MOVDA16(0), rem_col_sel);

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

#endif

static inline void _xa_nn_dot_product_1_rows_1_vecs_aligned
    (ae_int32x2*    out_0_0,
     const WORD8*   p_mat_0,
     const WORD8*   p_vec_0,
     WORD32         cols1,
     WORD32         vec_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat, d_vec;

  const WORD8 *p_mat = p_mat_0;
  const WORD8 *p_vec = p_vec_0;

  /* acc_part is interleaved acc,
   * i.e. acc = sum(part_01.H + part_01.L + part_23.H + part_23.L)
   */
  ae_int64 acc = 0;
  ae_int32x2 acc_part_01 = 0, acc_part_23 = 0;

  ae_int32 d_out = *out_0_0;

  WORD32 rem_elm_acc = 0;
  ae_int16x4 vec_zb = AE_MOVDA16(vec_zero_bias);

  // 4 columns per iteration
  for( ; c_itr<cols1/4; c_itr++)
  {
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_mat, p_mat, 4);
    AE_L8X4S_IP(d_vec, p_vec, 4);
#else
    AE_L8X4F_IP(d_mat, p_mat, 4);
    AE_L8X4F_IP(d_vec, p_vec, 4);
    d_mat = AE_SRAI16(d_mat, 8);
    d_vec = AE_SRAI16(d_vec, 8);
#endif

    d_vec = AE_ADD16(d_vec, vec_zb);

    AE_MULA16X4(acc_part_01, acc_part_23, d_mat, d_vec);
  }

  // add all parts of acc_part_XX together
#if !(XCHAL_HAVE_HIFI4)
  acc = AE_MULZAAD32X16_H1_L0(acc_part_01, AE_MOVDA16(1));
  AE_MULAAD32X16_H1_L0(acc, acc_part_23, AE_MOVDA16(1));
#else
  acc = AE_MULZAAAAQ32X16(acc_part_01, acc_part_23, AE_MOVDA16(1));
#endif

  // remaining columns
  for(c_itr *= 4; c_itr<cols1; c_itr++){
    rem_elm_acc += (*p_mat) * (*p_vec + vec_zero_bias);
    p_mat++, p_vec++;
  }

  acc += (ae_int64)rem_elm_acc;
  acc += (ae_int64)d_out;

  *out_0_0 = AE_MOVINT32_FROMINT64(acc);
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
#if XCHAL_HAVE_HIFI1
    AE_L8S_IP(d_mat, p_mat_0, 1);
    AE_L8S_IP(d_vec, p_vec_0, 1);
#else
    d_mat = AE_MOVDA16(*(((WORD8 *)p_mat_0)+c_itr));
    d_vec = AE_MOVDA16(*(((WORD8 *)p_vec_0)+c_itr));
#endif
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));
    AE_MULA16X4(d_out, d_tmp, d_mat, d_vec);
  }
  *out_0_0 = d_out;
}

/*  This kernel performs the following dual mat*vec op
 *    p_out = mat1*(vec1+vec1_zero_bias) +
 *            mat2*(vec2+vec2_zero_bias) +
 *            p_bias
 *
 *  If p_mat2 is NULL, then the second matrix-vec multiply op is not executed, and 
 *    p_out = mat1*(vec1+vec1_zero_bias) +
 *            p_bias
 */
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
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
   /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  int bias_flag = 0;
  if(p_bias != NULL)
  {
    bias_flag = 1;
  }

  /* If p_mat2 is not NULL, then we are doing a dual mat*vec operation
   *   p_out = mat1*(vec1+vec1_zero_bias) +
   *           mat2*(vec2+vec2_zero_bias) +
   *           p_bias
   */
  if(p_mat2 != NULL)
  {
    // Initialize function pointers for dot product of 1R and 1C
    _dot_prod_1_rows_1_vecs_kernel
      mat1vec1_1R1C_dotprod_func = _xa_nn_dot_product_1_rows_1_vecs_unaligned,
      mat2vec2_1R1C_dotprod_func = _xa_nn_dot_product_1_rows_1_vecs_unaligned;

    if( ((uintptr_t)p_mat1%4 == 0) && ((uintptr_t)p_vec1%4 == 0) && (row_stride1%4 == 0) ){
      mat1vec1_1R1C_dotprod_func = _xa_nn_dot_product_1_rows_1_vecs_aligned;
    }

    if( ((uintptr_t)p_mat2%4 == 0) && ((uintptr_t)p_vec2%4 == 0) && (row_stride2%4 == 0) ){
      mat2vec2_1R1C_dotprod_func = _xa_nn_dot_product_1_rows_1_vecs_aligned;
    }


    // Initialize function pointers for dot product of 4R and 1C
    _dot_prod_4_rows_1_vecs_kernel
      mat1vec1_4R1C_dotprod_func = _xa_nn_dot_product_4_rows_1_vec_mat_unaligned_vec_unaligned,
      mat2vec2_4R1C_dotprod_func = _xa_nn_dot_product_4_rows_1_vec_mat_unaligned_vec_unaligned;

    if( (uintptr_t)p_mat1%4 == 0 && row_stride1%4 == 0){
      if( (uintptr_t)p_vec1%4 == 0 ){
        mat1vec1_4R1C_dotprod_func = _xa_nn_dot_product_4_rows_1_vec_mat_aligned_vec_aligned;
      }else{
        mat1vec1_4R1C_dotprod_func = _xa_nn_dot_product_4_rows_1_vec_mat_aligned_vec_unaligned;
      }
    }else if((uintptr_t)p_vec1%4 == 0){
      mat1vec1_4R1C_dotprod_func = _xa_nn_dot_product_4_rows_1_vec_mat_unaligned_vec_aligned;
    }

    if( (uintptr_t)p_mat2%4 == 0 && row_stride2%4 == 0){
      if( (uintptr_t)p_vec2%4 == 0 ){
        mat2vec2_4R1C_dotprod_func = _xa_nn_dot_product_4_rows_1_vec_mat_aligned_vec_aligned;
      }else{
        mat2vec2_4R1C_dotprod_func = _xa_nn_dot_product_4_rows_1_vec_mat_aligned_vec_unaligned;
      }
    }else if((uintptr_t)p_vec2%4 == 0){
      mat2vec2_4R1C_dotprod_func = _xa_nn_dot_product_4_rows_1_vec_mat_unaligned_vec_aligned;
    }

    ae_int32x2 out_01 = 0, out_23 = 0;
#if XCHAL_HAVE_HIFI1
    ae_valign align_out = AE_ZALIGN64();
    ae_int16x4 out_0123;
#endif

    // 4 rows per iteration
    for(m_itr = 0; m_itr < (rows/4)*4 ; m_itr+=4) {
        
      /* Load bias in the accumulator */
      if(bias_flag) {
          out_01 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
          out_23 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
      }

      mat1vec1_4R1C_dotprod_func(
        &out_01, &out_23,
        p_mat1 + m_itr*row_stride1,
        p_vec1,
        cols1,
        row_stride1,
        vec1_zero_bias);

      mat2vec2_4R1C_dotprod_func(
        &out_01, &out_23,
        p_mat2 + m_itr*row_stride2,
        p_vec2,
        cols2,
        row_stride2,
        vec2_zero_bias);

      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_01, out_01, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_23, out_23, out_multiplier, left_shift, right_shift);

      out_01 = AE_ADD32S(out_01, AE_MOVDA32(out_zero_bias));
      out_23 = AE_ADD32S(out_23, AE_MOVDA32(out_zero_bias));

#if XCHAL_HAVE_HIFI1
      out_0123 = AE_SAT16X4(out_01, out_23);
      out_0123 = AE_SAT8S(out_0123);
      AE_SA8X4U_IP(out_0123, align_out, (ae_int32 *)p_out);
#else
      out_01 = AE_MAX32(out_01, min_int8);  out_01 = AE_MIN32(out_01, max_int8);
      out_23 = AE_MAX32(out_23, min_int8);  out_23 = AE_MIN32(out_23, max_int8);

      *p_out     = (WORD8)AE_MOVAD32_H(out_01);
      *(p_out+1) = (WORD8)AE_MOVAD32_L(out_01);
      *(p_out+2) = (WORD8)AE_MOVAD32_H(out_23);
      *(p_out+3) = (WORD8)AE_MOVAD32_L(out_23);
      p_out += 4;
#endif
    }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_out);
#endif

    // remaining rows
    for (; m_itr < rows; m_itr++) {
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

      mat1vec1_1R1C_dotprod_func
        (&acc_row0_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols1
         ,vec1_zero_bias
        );

      mat2vec2_1R1C_dotprod_func
        (&acc_row0_vec0
         ,p_mat2_0
         ,p_vec2_0
         ,cols2
         ,vec2_zero_bias
        );

      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
      acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
      acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  }
  /* Else, this is a single mat*vec operation */
  else
  {
    /* Matrix and vector (including other factors like dimensions/strides) are both aligned properly */
    if(((((unsigned)p_mat1) & 3) == 0) && ((((unsigned)p_vec1) & 3) == 0) && ((row_stride1 & 3) == 0) && ((cols1 & 3) == 0))
    {
      const WORD8 *p_mat1_0;
      const WORD8 *p_vec1_0;
#if XCHAL_HAVE_HIFI1
    ae_valign align_out = AE_ZALIGN64();
    ae_int16x4 rowvec01, rowvec23;
#endif
      
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
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row2_vec0, acc_row2_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row3_vec0, acc_row3_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, AE_MOVDA32(out_zero_bias));
        acc_row2_vec0 = AE_ADD32S(acc_row2_vec0, AE_MOVDA32(out_zero_bias));
        acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, AE_MOVDA32(out_zero_bias));
#if XCHAL_HAVE_HIFI1
        rowvec01 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        rowvec23 = AE_SAT16X4(acc_row2_vec0, acc_row3_vec0);
        rowvec01 = AE_SAT8S(rowvec01);
        rowvec23 = AE_SAT8S(rowvec23);
        AE_SA8X4U_IP(rowvec01, align_out, (ae_int32 *)p_out);
        AE_SA8X4U_IP(rowvec23, align_out, (ae_int32 *)p_out);
#else
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
#endif
      }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_out);
#endif
      
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
      
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, AE_MOVDA32(out_zero_bias));
#if XCHAL_HAVE_HIFI1
        rowvec01 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        rowvec01 = AE_SAT8S(rowvec01);

        AE_SA8X4U_IP(rowvec01, align_out, (ae_int32 *)p_out);
#else
        acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
        acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
        acc_row1_vec0 = AE_MAX32(acc_row1_vec0, min_int8);
        acc_row1_vec0 = AE_MIN32(acc_row1_vec0, max_int8);
      
        *p_out++ = (WORD8)AE_MOVAD32_H(acc_row0_vec0);
        *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
        *p_out++ = (WORD8)AE_MOVAD32_H(acc_row1_vec0);
        *p_out++ = (WORD8)AE_MOVAD32_L(acc_row1_vec0);
#endif
      }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_out);
#endif
      
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
    
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
        acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
        acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
        *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
      }
    }
    /* Matrix is not aligned, vector is */
    else if(((((unsigned)p_vec1) & 3) == 0) && ((row_stride1 & 3) == 0) && ((rows&3) == 0) && ((cols1 & 3) == 0))
    {
      const WORD8 *p_mat1_0;
      const WORD8 *p_vec1_0;
#if XCHAL_HAVE_HIFI1
    ae_valign align_out = AE_ZALIGN64();
    ae_int16x4 rowvec01;
#endif
    
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
    
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, AE_MOVDA32(out_zero_bias));

#if XCHAL_HAVE_HIFI1
        rowvec01 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        rowvec01 = AE_SAT8S(rowvec01);

        AE_SA8X4U_IP(rowvec01, align_out, (ae_int32 *)p_out);
#else
        acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
        acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
        acc_row1_vec0 = AE_MAX32(acc_row1_vec0, min_int8);
        acc_row1_vec0 = AE_MIN32(acc_row1_vec0, max_int8);
    
        *p_out++ = (WORD8)AE_MOVAD32_H(acc_row0_vec0);
        *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
        *p_out++ = (WORD8)AE_MOVAD32_H(acc_row1_vec0);
        *p_out++ = (WORD8)AE_MOVAD32_L(acc_row1_vec0);
#endif
      }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_out);
#endif
    }
    /* Generic case, no pre-conditions */
    else if((p_mat1 != NULL) && (p_vec1 != NULL))
    {
      const WORD8 *p_mat1_0;
      const WORD8 *p_vec1_0;
#if XCHAL_HAVE_HIFI1
    ae_valign align_out = AE_ZALIGN64();
    ae_int16x4 rowvec01;
#endif
    
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
    
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, AE_MOVDA32(out_zero_bias));
#if XCHAL_HAVE_HIFI1
        rowvec01 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        rowvec01 = AE_SAT8S(rowvec01);

        AE_SA8X4U_IP(rowvec01, align_out, (ae_int32 *)p_out);
#else
        acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
        acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
        acc_row1_vec0 = AE_MAX32(acc_row1_vec0, min_int8);
        acc_row1_vec0 = AE_MIN32(acc_row1_vec0, max_int8);
    
        *p_out++ = (WORD8)AE_MOVAD32_H(acc_row0_vec0);
        *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
        *p_out++ = (WORD8)AE_MOVAD32_H(acc_row1_vec0);
        *p_out++ = (WORD8)AE_MOVAD32_L(acc_row1_vec0);
#endif
      }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_out);
#endif
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
    
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
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

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
   /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

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
        acc_row0_vec0 = AE_MOVDA32(p_bias[m_itr]);
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

      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);

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
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);

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

      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
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
