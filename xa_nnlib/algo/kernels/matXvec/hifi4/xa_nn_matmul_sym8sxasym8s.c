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
  ae_int32x2 acc_row0_vec0, acc_row1_vec0, acc_row3_vec0, acc_row1_vec1, acc_row3_vec1;
  max_int8 = AE_MOVDA32(127);
  min_int8 = AE_MOVDA32(-128);
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
#if (XCHAL_HW_VERSION >= RI9_HWVERSION )
        ae_int8x8 temp_h_8, temp0_8 = AE_SAT8X4X32_L(acc_row3_vec0, acc_row1_vec0);
        temp_h_8 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_4321(AE_MOVINT16X4_FROMINT8X8 (temp0_8), AE_MOVINT16X4_FROMINT8X8 (temp0_8)));
        AE_S8_0_XP(temp0_8, (ae_int8 *)p_dst1, out_offset);
        AE_S8_0_XP(temp_h_8, (ae_int8 *)p_dst1, out_offset);    
        temp0_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp0_8),8));
        temp_h_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp_h_8),8));
        AE_S8_0_XP(temp0_8, (ae_int8 *)p_dst0, out_offset);
        AE_S8_0_XP(temp_h_8, (ae_int8 *)p_dst0, out_offset);    
#else
        ae_int32x2 acc_row2_vec0;
        acc_row0_vec0 = AE_SEL32_HH(acc_row1_vec0, acc_row1_vec0);
        acc_row1_vec0 = AE_SEL32_LL(acc_row1_vec0, acc_row1_vec0);
        acc_row2_vec0 = AE_SEL32_HH(acc_row3_vec0, acc_row3_vec0);
        acc_row3_vec0 = AE_SEL32_LL(acc_row3_vec0, acc_row3_vec0);

        ae_int16x4 temp01 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        temp01 = AE_SAT8S(temp01);
        ae_int16x4 temp23 = AE_SAT16X4(acc_row2_vec0, acc_row3_vec0);
        temp23 = AE_SAT8S(temp23);

        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp01, temp01), (WORD8 *)p_dst0, out_offset);
        AE_S8_0_XP_HIFI1(temp01, (WORD8 *)p_dst1, out_offset);
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp23, temp23), (WORD8 *)p_dst0, out_offset);
        AE_S8_0_XP_HIFI1(temp23, (WORD8 *)p_dst1, out_offset);
#endif
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
        temp1_8 = AE_SAT8X4X32_L(acc_row3_vec1, acc_row1_vec1);
        
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

#if (XCHAL_HW_VERSION >= RI9_HWVERSION )
        ae_int8x8 temp_h_1_8, temp_h_0_8, temp1_8, temp0_8;
    
        temp0_8 = AE_SAT8X4X32_L(acc_row3_vec0, acc_row1_vec0);
        temp1_8 = AE_SAT8X4X32_L(acc_row3_vec1, acc_row1_vec1);
        
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
#else
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
        temp01 = AE_SAT8S(temp01);
        ae_int16x4 temp23 = AE_SAT16X4(acc_row2_vec0, acc_row3_vec0);
        temp23 = AE_SAT8S(temp23);
        ae_int16x4 temp45 = AE_SAT16X4(acc_row0_vec1, acc_row1_vec1);
        temp45 = AE_SAT8S(temp45);
        ae_int16x4 temp67 = AE_SAT16X4(acc_row2_vec1, acc_row3_vec1);
        temp67 = AE_SAT8S(temp67);
        
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp01, temp01), (WORD8 *)p_dst0, out_offset);
        AE_S8_0_XP_HIFI1(temp01, (WORD8 *)p_dst1, out_offset);
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp23, temp23), (WORD8 *)p_dst2, out_offset);
        AE_S8_0_XP_HIFI1(temp23, (WORD8 *)p_dst3, out_offset);
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp45, temp45), (WORD8 *)p_dst0, out_offset);
        AE_S8_0_XP_HIFI1(temp45, (WORD8 *)p_dst1, out_offset);
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp67, temp67), (WORD8 *)p_dst2, out_offset);
        AE_S8_0_XP_HIFI1(temp67, (WORD8 *)p_dst3, out_offset);
#endif

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
#if (XCHAL_HW_VERSION >= RI9_HWVERSION )
        ae_int8x8 temp_h_8, temp0_8 = AE_SAT8X4X32_L(acc_row3_vec0, acc_row1_vec0);
        temp_h_8 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_4321(AE_MOVINT16X4_FROMINT8X8 (temp0_8), AE_MOVINT16X4_FROMINT8X8 (temp0_8)));
        AE_S8_0_XP(temp0_8, (ae_int8 *)p_dst1, out_offset);
        AE_S8_0_XP(temp_h_8, (ae_int8 *)p_dst3, out_offset);    
        temp0_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp0_8),8));
        temp_h_8 = AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVINT16X4_FROMINT8X8(temp_h_8),8));
        AE_S8_0_XP(temp0_8, (ae_int8 *)p_dst0, out_offset);
        AE_S8_0_XP(temp_h_8, (ae_int8 *)p_dst2, out_offset);    
#else
        ae_int32x2 acc_row2_vec0;
        acc_row0_vec0 = AE_SEL32_HH(acc_row1_vec0, acc_row1_vec0);
        acc_row1_vec0 = AE_SEL32_LL(acc_row1_vec0, acc_row1_vec0);
        acc_row2_vec0 = AE_SEL32_HH(acc_row3_vec0, acc_row3_vec0);
        acc_row3_vec0 = AE_SEL32_LL(acc_row3_vec0, acc_row3_vec0);
        ae_int16x4 temp01 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        temp01 = AE_SAT8S(temp01);
        ae_int16x4 temp23 = AE_SAT16X4(acc_row2_vec0, acc_row3_vec0);
        temp23 = AE_SAT8S(temp23);
        
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp01, temp01), (WORD8 *)p_dst0, out_offset);
        AE_S8_0_XP_HIFI1(temp01, (WORD8 *)p_dst1, out_offset);
        AE_S8_0_XP_HIFI1(AE_SEL16_5432(temp23, temp23), (WORD8 *)p_dst2, out_offset);
        AE_S8_0_XP_HIFI1(temp23, (WORD8 *)p_dst3, out_offset);
#endif
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
#if ((XCHAL_HW_VERSION >= RI9_HWVERSION )& XCHAL_HAVE_HIFI1)
        ae_int8x8 temp0_8 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);
        AE_S8_0_XP(temp0_8, (ae_int8 *)p_dst0, out_offset);
#else
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_offset);
#endif
      }
    }
  }
  else
    return -1;

  return 0;
}
