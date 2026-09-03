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

#if XCHAL_HAVE_HIFI1S
static inline void _xa_nn_dot_product_4_rows_1_vec_mat_unaligned_vec_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,const WORD8*      p_mat_0
    ,const WORD8*      p_vec_0
    ,WORD32      cols1
    ,WORD32      row_stride1
    ,WORD32      vec_zero_bias)
{
  int c_itr = 0;
  ae_int8x8 d_mat0, d_mat1, d_mat2, d_mat3;
  
  ae_int16x4 d_vec16_1, d_vec16_2;

  WORD8 *p_mat_1 = ((WORD8 *)p_mat_0 + row_stride1);
  WORD8 *p_mat_2 = ((WORD8 *)p_mat_1 + row_stride1);
  WORD8 *p_mat_3 = ((WORD8 *)p_mat_2 + row_stride1);

  WORD8 *p_vec = (WORD8*)p_vec_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;


  ae_int32x2 d_acc0 = AE_SEL32_HH(AE_ZERO32(), acc_row0_vec0);
  ae_int32x2 d_acc1 = AE_SEL32_LL(AE_ZERO32(), acc_row0_vec0);
  ae_int32x2 d_acc2 = AE_SEL32_HH(AE_ZERO32(), acc_row1_vec0);
  ae_int32x2 d_acc3 = AE_SEL32_LL(AE_ZERO32(), acc_row1_vec0);

  ae_valign align_mat0 = AE_LA64_PP( p_mat_0 );
  ae_valign align_mat1 = AE_LA64_PP( p_mat_1 );
  ae_valign align_mat2 = AE_LA64_PP( p_mat_2 );
  ae_valign align_mat3 = AE_LA64_PP( p_mat_3 );
 
  /* 8 columns at a time */
  for(c_itr = 0; c_itr < cols1 >> 3; c_itr++)
  {
    AE_LA8X8_IP(d_mat0, align_mat0, (ae_int8x8 *)p_mat_0);
    AE_LA8X8_IP(d_mat1, align_mat1, (ae_int8x8 *)p_mat_1);
    AE_LA8X8_IP(d_mat2, align_mat2, (ae_int8x8 *)p_mat_2);
    AE_LA8X8_IP(d_mat3, align_mat3, (ae_int8x8 *)p_mat_3);
    
    AE_L8X4S_IP(d_vec16_1, p_vec, 4);
    AE_L8X4S_IP(d_vec16_2, p_vec, 4);
    d_vec16_1 = AE_ADD16(d_vec16_1, AE_MOVDA16(vec_zero_bias));
    d_vec16_2 = AE_ADD16(d_vec16_2, AE_MOVDA16(vec_zero_bias));
    
    AE_MULAAAA16Q8(d_acc0, d_vec16_1, d_vec16_2, d_mat0);  
    AE_MULAAAA16Q8(d_acc1, d_vec16_1, d_vec16_2, d_mat1);  
    AE_MULAAAA16Q8(d_acc2, d_vec16_1, d_vec16_2, d_mat2);  
    AE_MULAAAA16Q8(d_acc3, d_vec16_1, d_vec16_2, d_mat3);      
  } 
  int rem = cols1 & 0x7;
  if (rem)
  {
    AE_LAV8X8_XP(d_mat0, align_mat0, (ae_int8x8 *)p_mat_0, rem );
    AE_LAV8X8_XP(d_mat1, align_mat1, (ae_int8x8 *)p_mat_1, rem );
    AE_LAV8X8_XP(d_mat2, align_mat2, (ae_int8x8 *)p_mat_2, rem );
    AE_LAV8X8_XP(d_mat3, align_mat3, (ae_int8x8 *)p_mat_3, rem );
    
    AE_L8X4S_IP(d_vec16_1, p_vec, 4);
    AE_L8X4S_IP(d_vec16_2, p_vec, 4);
    d_vec16_1 = AE_ADD16(d_vec16_1, AE_MOVDA16(vec_zero_bias));
    d_vec16_2 = AE_ADD16(d_vec16_2, AE_MOVDA16(vec_zero_bias));
    
    AE_MULAAAA16Q8(d_acc0, d_vec16_1, d_vec16_2, d_mat0);  
    AE_MULAAAA16Q8(d_acc1, d_vec16_1, d_vec16_2, d_mat1);  
    AE_MULAAAA16Q8(d_acc2, d_vec16_1, d_vec16_2, d_mat2);  
    AE_MULAAAA16Q8(d_acc3, d_vec16_1, d_vec16_2, d_mat3); 
  }
  
  d_acc0 = AE_ADD32_HL_LH(d_acc0, d_acc0);
  d_acc1 = AE_ADD32_HL_LH(d_acc1, d_acc1);
  d_acc2 = AE_ADD32_HL_LH(d_acc2, d_acc2);
  d_acc3 = AE_ADD32_HL_LH(d_acc3, d_acc3);
  
  acc_row0_vec0 = AE_SEL32_LL(d_acc0, d_acc1);
  acc_row1_vec0 = AE_SEL32_LL(d_acc2, d_acc3);
  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}
#else
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
#endif

#if XCHAL_HAVE_HIFI1
#if XCHAL_HAVE_HIFI1S
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
  ae_int8x8 d_mat0, d_mat1, d_mat2, d_mat3;
  
  ae_int16x4 d_vec16_1, d_vec16_2;

  WORD8 *p_mat_1 = ((WORD8 *)p_mat_0 + row_stride1);
  WORD8 *p_mat_2 = ((WORD8 *)p_mat_1 + row_stride1);
  WORD8 *p_mat_3 = ((WORD8 *)p_mat_2 + row_stride1);

  WORD8 *p_vec = (WORD8*)p_vec_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;


  ae_int32x2 d_acc0 = AE_SEL32_HH(AE_ZERO32(), acc_row0_vec0);
  ae_int32x2 d_acc1 = AE_SEL32_LL(AE_ZERO32(), acc_row0_vec0);
  ae_int32x2 d_acc2 = AE_SEL32_HH(AE_ZERO32(), acc_row1_vec0);
  ae_int32x2 d_acc3 = AE_SEL32_LL(AE_ZERO32(), acc_row1_vec0);

  ae_valign align_mat0 = AE_LA64_PP( p_mat_0 );
  ae_valign align_mat1 = AE_LA64_PP( p_mat_1 );
  ae_valign align_mat2 = AE_LA64_PP( p_mat_2 );
  ae_valign align_mat3 = AE_LA64_PP( p_mat_3 );
 
  /* 8 columns at a time */
  for(c_itr = 0; c_itr < cols1 >> 3; c_itr++)
  {
    AE_LA8X8_IP(d_mat0, align_mat0, (ae_int8x8 *)p_mat_0);
    AE_LA8X8_IP(d_mat1, align_mat1, (ae_int8x8 *)p_mat_1);
    AE_LA8X8_IP(d_mat2, align_mat2, (ae_int8x8 *)p_mat_2);
    AE_LA8X8_IP(d_mat3, align_mat3, (ae_int8x8 *)p_mat_3);
    
    AE_L8X4S_IP(d_vec16_1, p_vec, 4);
    AE_L8X4S_IP(d_vec16_2, p_vec, 4);
    d_vec16_1 = AE_ADD16(d_vec16_1, AE_MOVDA16(vec_zero_bias));
    d_vec16_2 = AE_ADD16(d_vec16_2, AE_MOVDA16(vec_zero_bias));
    
    AE_MULAAAA16Q8(d_acc0, d_vec16_1, d_vec16_2, d_mat0);  
    AE_MULAAAA16Q8(d_acc1, d_vec16_1, d_vec16_2, d_mat1);  
    AE_MULAAAA16Q8(d_acc2, d_vec16_1, d_vec16_2, d_mat2);  
    AE_MULAAAA16Q8(d_acc3, d_vec16_1, d_vec16_2, d_mat3);      
  } 
  
  int rem = cols1 & 0x7;
  if (rem)
  {
    AE_LAV8X8_XP(d_mat0, align_mat0, (ae_int8x8 *)p_mat_0, rem );
    AE_LAV8X8_XP(d_mat1, align_mat1, (ae_int8x8 *)p_mat_1, rem );
    AE_LAV8X8_XP(d_mat2, align_mat2, (ae_int8x8 *)p_mat_2, rem );
    AE_LAV8X8_XP(d_mat3, align_mat3, (ae_int8x8 *)p_mat_3, rem );
    
    AE_L8X4S_IP(d_vec16_1, p_vec, 4);
    AE_L8X4S_IP(d_vec16_2, p_vec, 4);
    d_vec16_1 = AE_ADD16(d_vec16_1, AE_MOVDA16(vec_zero_bias));
    d_vec16_2 = AE_ADD16(d_vec16_2, AE_MOVDA16(vec_zero_bias));
    
    AE_MULAAAA16Q8(d_acc0, d_vec16_1, d_vec16_2, d_mat0);  
    AE_MULAAAA16Q8(d_acc1, d_vec16_1, d_vec16_2, d_mat1);  
    AE_MULAAAA16Q8(d_acc2, d_vec16_1, d_vec16_2, d_mat2);  
    AE_MULAAAA16Q8(d_acc3, d_vec16_1, d_vec16_2, d_mat3); 
  }
  
  d_acc0 = AE_ADD32_HL_LH(d_acc0, d_acc0);
  d_acc1 = AE_ADD32_HL_LH(d_acc1, d_acc1);
  d_acc2 = AE_ADD32_HL_LH(d_acc2, d_acc2);
  d_acc3 = AE_ADD32_HL_LH(d_acc3, d_acc3);
  
  acc_row0_vec0 = AE_SEL32_LL(d_acc0, d_acc1);
  acc_row1_vec0 = AE_SEL32_LL(d_acc2, d_acc3);
  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}
#else /* XCHAL_HAVE_HIFI1S */
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
#endif



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
#endif /* XCHAL_HAVE_HIFI1S */

#else

#if XA_HAVE_HIFI3_CORE
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

    AE_MULAAAAQ16(out_0, d_mat0, d_vec);
    AE_MULAAAAQ16(out_1, d_mat1, d_vec);
    AE_MULAAAAQ16(out_2, d_mat2, d_vec);
    AE_MULAAAAQ16(out_3, d_mat3, d_vec);

    AE_MULAAAAQ16(out_0, d_mat01, d_vec1);
    AE_MULAAAAQ16(out_1, d_mat11, d_vec1);
    AE_MULAAAAQ16(out_2, d_mat21, d_vec1);
    AE_MULAAAAQ16(out_3, d_mat31, d_vec1);
  }

  p_vec = (WORD8*)p_vec_0;

  for(c_itr = 0; c_itr < cols1 >> 3; c_itr++)
  {
    ae_int16x4 d_mat41, d_mat51, d_mat61, d_mat71, d_vec1;

    d_vec = AE_L8X4F_I(p_vec, 4);
    AE_L8X4F_IP(d_vec1, p_vec, 8);

    d_vec = AE_SRAI16(d_vec, 8);
    d_vec = AE_ADD16(d_vec, AE_MOVDA16(vec_zero_bias));

    d_vec1 = AE_SRAI16(d_vec1, 8);
    d_vec1 = AE_ADD16(d_vec1, AE_MOVDA16(vec_zero_bias));

    d_mat4 =  AE_L8X4F_I(p_mat_4, 4);
    AE_L8X4F_IP(d_mat41,  p_mat_4, 8);
    d_mat5 =  AE_L8X4F_I(p_mat_5, 4);
    AE_L8X4F_IP(d_mat51,  p_mat_5, 8);
    d_mat6 =  AE_L8X4F_I(p_mat_6, 4);
    AE_L8X4F_IP(d_mat61,  p_mat_6, 8);
    d_mat7 =  AE_L8X4F_I(p_mat_7, 4);
    AE_L8X4F_IP(d_mat71,  p_mat_7, 8);

    AE_MULAAAAQ16(out_4, d_mat4, d_vec);
    AE_MULAAAAQ16(out_5, d_mat5, d_vec);
    AE_MULAAAAQ16(out_6, d_mat6, d_vec);
    AE_MULAAAAQ16(out_7, d_mat7, d_vec);

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
#else //XA_HAVE_HIFI3_CORE
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
  if(cols1%8)
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

#endif //XA_HAVE_HIFI3_CORE


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

  out_1 = AE_CVT64F32_L(acc_row0_vec0);
  out_3 = AE_CVT64F32_L(acc_row1_vec0);
  out_0 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(acc_row0_vec0), 24);
  out_1 = AE_SRAI64(out_1, 24);
  out_2 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(acc_row1_vec0), 24);
  out_3 = AE_SRAI64(out_3, 24);
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

/* Following kernel calculates contribution of matrix zero-biases towards final sum */
static WORD32 internal_calc_mzbsum(WORD32 mat_zero_bias, WORD32 vec_zero_bias, const WORD8 * __restrict__ p_vec, int cols1)
{
  if(mat_zero_bias == 0){
    return 0;
  }

  WORD32 sum_mzb32 = 0, c_itr;
  WORD32 preloop_cnt = (4 - ((unsigned)p_vec-(((unsigned)p_vec)&~0x3))) & 0x03;
  if(preloop_cnt > cols1) { preloop_cnt = 0;}
  cols1 = cols1 - preloop_cnt;

  for(c_itr = 0; c_itr < preloop_cnt; c_itr++){
    int vecval = *p_vec++;
    sum_mzb32 += (vecval+vec_zero_bias)*mat_zero_bias;
  }
  
  ae_int64 sum_mzb = (ae_int64)sum_mzb32;
  ae_int16x4 mzb_16x4 = AE_MOVDA16(mat_zero_bias);
  ae_int16x4 d_vec0;

  for(c_itr = 0; c_itr < cols1>>2; c_itr++){            
    AE_L8X4F_IP(d_vec0, p_vec, 4);                       
    d_vec0 = AE_SRAI16(d_vec0, 8);
    d_vec0 = AE_ADD16(d_vec0, AE_MOVDA16(vec_zero_bias));
    AE_MULAAAAQ16(sum_mzb, mzb_16x4, d_vec0);
  }
  sum_mzb32 = AE_MOVINT32X2_FROMINT64(sum_mzb);

  for(c_itr = 0; c_itr < (cols1&0x3); c_itr++){
    int vecval = *p_vec++;
    sum_mzb32 += (vecval+vec_zero_bias)*mat_zero_bias;
  }
  return sum_mzb32;
}

WORD32 xa_nn_matXvec_v2_asym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat,
    const WORD8 * __restrict__ p_vec,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_stride,
    WORD32 mat_zero_bias,
    WORD32 vec_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias,
    WORD32 out_activation_min,
    WORD32 out_activation_max,
    xa_dma_cfg_t *p_dma_cfg) 
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
#if XCHAL_HAVE_HIFI1S
  XA_NNLIB_ARG_CHK_ALIGN(p_mat, 8, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, 8, -1);
#else
  XA_NNLIB_ARG_CHK_ALIGN(p_mat, 4, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, 4, -1);
#endif
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride < cols), -1);
  XA_NNLIB_ARG_CHK_COND((vec_zero_bias < -127 || vec_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((mat_zero_bias < -127 || mat_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_min < -128 || out_activation_min > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < -128 || out_activation_max > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

  int m_itr=0;
  ae_int32x2 max_int8 = AE_MOVDA32(out_activation_max);
  ae_int32x2 min_int8 = AE_MOVDA32(out_activation_min);

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
  ae_valign bias_valign;
  if(p_bias != NULL)
  {
    bias_valign = AE_LA64_PP(p_bias);
    bias_flag = 1;
  }

    WORD32 mat1_zb_sum =  internal_calc_mzbsum(mat_zero_bias, vec_zero_bias, p_vec, cols);
    ae_int32x2 mat1_zb_sumx2 = mat1_zb_sum;

#if XCHAL_HAVE_HIFI1S
    /* Matrix and vector (including other factors like dimensions/strides) are both aligned properly */
    if( ((row_stride & 7) == 0) && ((cols & 3) == 0))
#else
    if( ((row_stride & 3) == 0) && ((cols & 3) == 0))
#endif /* XCHAL_HAVE_HIFI1S */
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
          AE_LA32X2_IP(acc_row0_vec0, bias_valign, (ae_int32x2 *)p_bias);
          AE_LA32X2_IP(acc_row1_vec0, bias_valign, (ae_int32x2 *)p_bias);
          AE_LA32X2_IP(acc_row2_vec0, bias_valign, (ae_int32x2 *)p_bias);
          AE_LA32X2_IP(acc_row3_vec0, bias_valign, (ae_int32x2 *)p_bias);
        }
      
        p_mat1_0 = (const WORD8 *)(p_mat+(m_itr * row_stride));
        p_vec1_0 = (const WORD8 *)(p_vec);
      
        _xa_nn_dot_product_8_rows_1_vec_mat_aligned_vec_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,&acc_row2_vec0
           ,&acc_row3_vec0
           ,p_mat1_0
           ,p_vec1_0
           ,cols
           ,row_stride
           ,vec_zero_bias
          );
        acc_row0_vec0 = AE_ADD32(acc_row0_vec0, mat1_zb_sumx2);
        acc_row1_vec0 = AE_ADD32(acc_row1_vec0, mat1_zb_sumx2);
        acc_row2_vec0 = AE_ADD32(acc_row2_vec0, mat1_zb_sumx2);
        acc_row3_vec0 = AE_ADD32(acc_row3_vec0, mat1_zb_sumx2);

        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row2_vec0, acc_row2_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row3_vec0, acc_row3_vec0, out_multiplier, left_shift, right_shift);
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
#if XCHAL_HAVE_HIFI1
    ae_valign align_out = AE_ZALIGN64();
    ae_int16x4 rowvec01;
#endif
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
      
        p_mat1_0 = (const WORD8 *)(p_mat+(m_itr * row_stride));
        p_vec1_0 = (const WORD8 *)(p_vec);
      
        _xa_nn_dot_product_4_rows_1_vec_mat_aligned_vec_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec1_0
           ,cols
           ,row_stride
           ,vec_zero_bias
          );
        acc_row0_vec0 = AE_ADD32(acc_row0_vec0, mat1_zb_sumx2);
        acc_row1_vec0 = AE_ADD32(acc_row1_vec0, mat1_zb_sumx2);
      
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, AE_MOVDA32(out_zero_bias));
#if XCHAL_HAVE_HIFI1
        rowvec01 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        rowvec01 = AE_MIN16(rowvec01, AE_MOVDA16(out_activation_max));
        rowvec01 = AE_MAX16(rowvec01, AE_MOVDA16(out_activation_min));
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
    /* Matrix is not aligned, vector is */
    else /* if(((((unsigned)p_vec) & 3) == 0)) This condition is always met, as p_vec is aligned. */
    {
      const WORD8 *p_mat1_0;
      const WORD8 *p_vec1_0;
#if XCHAL_HAVE_HIFI1
    ae_valign align_out = AE_ZALIGN64();
    ae_int16x4 rowvec01;
#endif
    
      for(m_itr = 0; m_itr < (rows&~0x03); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0 = ZERO32;
        ae_int32x2 acc_row1_vec0 = ZERO32;
    
        if(bias_flag)
        {
          /* Load bias in the accumulator */
          AE_LA32X2_IP(acc_row0_vec0, bias_valign, (ae_int32x2 *)p_bias);
          AE_LA32X2_IP(acc_row1_vec0, bias_valign, (ae_int32x2 *)p_bias);          
        }
    
        p_mat1_0 = (const WORD8 *)(p_mat+(m_itr * row_stride));
        p_vec1_0 = (const WORD8 *)(p_vec);
    
        _xa_nn_dot_product_4_rows_1_vec_mat_unaligned_vec_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec1_0
           ,cols
           ,row_stride
           ,vec_zero_bias
          );
        acc_row0_vec0 = AE_ADD32(acc_row0_vec0, mat1_zb_sumx2);
        acc_row1_vec0 = AE_ADD32(acc_row1_vec0, mat1_zb_sumx2);
    
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, AE_MOVDA32(out_zero_bias));

#if XCHAL_HAVE_HIFI1
        rowvec01 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        rowvec01 = AE_MIN16(rowvec01, AE_MOVDA16(out_activation_max));
        rowvec01 = AE_MAX16(rowvec01, AE_MOVDA16(out_activation_min));
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

    /* Compute last (rows % 4) output element */
    const WORD8 *p_mat1_0;
    const WORD8 *p_vec1_0;
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
    
      p_mat1_0 = (WORD8 *)(p_mat+(m_itr * row_stride));
      p_vec1_0 = (WORD8 *)(p_vec);
    
      if(bias_flag)
      {
        AE_L32_IP(acc_row0_vec0, (ae_int32 *) p_bias, 4);
      }
    
      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,p_mat1_0
         ,p_vec1_0
         ,cols
         ,vec_zero_bias
        );
      acc_row0_vec0 = AE_ADD32(acc_row0_vec0, mat1_zb_sumx2);
    
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, AE_MOVDA32(out_zero_bias));
      acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
      acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  
  return 0;
}
