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

#if XCHAL_HAVE_HIFI1
#define AE_L8X4S_I_HIFI4(d, ptr, inc) \
  d = AE_L8X4S_I(ptr, inc);
#else
#define AE_L8X4S_I_HIFI4(d, ptr, inc) \
  d = AE_L8X4F_I(ptr, inc); \
d = AE_SRAI16(d, 8);
#endif

#define AE_MINMAX32_HF4(acc, min, max) \
  acc = AE_MAX32(acc, min); \
acc = AE_MIN32(acc, max);

#if XCHAL_HAVE_HIFI1
#define AE_S8_FROM32X2_WITHSTRIDE(val32, dst, stride) \
  AE_S8_0_XP_HIFI1(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LH(val32, val32)), dst, stride);\
AE_S8_0_XP_HIFI1(AE_MOVINT16X4_FROMINT32X2(val32), dst, stride);\

#endif

#define AE_S8_FROM32_WITHSTRIDE(val32, dst, stride) \
  *dst = (WORD8)val32; \
dst += stride;

#if XCHAL_HAVE_HIFI1
static inline void _xa_nn_dot_product_1row_4vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD32      vecstride
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0;
  ae_int16x4 d_vec, d_vec1, d_vec2, d_vec3;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  WORD8 *p_vec_1, *p_vec_2, *p_vec_3;
  ae_int16x4 d_mzb;

  d_mzb = AE_MOVDA16(mat_zero_bias);
  p_vec_1 = p_vec_0 + vecstride;
  p_vec_2 = p_vec_1 + vecstride;
  p_vec_3 = p_vec_2 + vecstride;

  d_out0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_0_0));
  d_out1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_1_1));
  d_out2 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_2_2));
  d_out3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_3_3));

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4S_IP(d_vec, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1, p_vec_1, 4);
    AE_L8X4S_IP(d_vec2, p_vec_2, 4);
    AE_L8X4S_IP(d_vec3, p_vec_3, 4);

    d_mat0 = AE_ADD16(d_mat0, d_mzb);

    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
    AE_MULAAAAQ16(d_out1, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec2);
    AE_MULAAAAQ16(d_out3, d_mat0, d_vec3);
  }

  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
  *out_1_1 = AE_MOVINT32X2_FROMINT64(d_out1);
  *out_2_2 = AE_MOVINT32X2_FROMINT64(d_out2);
  *out_3_3 = AE_MOVINT32X2_FROMINT64(d_out3);
}

static inline void _xa_nn_dot_product_2row_2vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_vec_0
 ,WORD32      vecstride
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1;
  ae_int16x4 d_vec, d_vec1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  WORD8 *p_vec_1;
  ae_int16x4 d_mzb;

  d_mzb = AE_MOVDA16(mat_zero_bias);
  p_vec_1 = p_vec_0 + vecstride;

  d_out0 = d_out1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_0_0));
  d_out2 = d_out3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_1_1));

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_L8X4S_I_HIFI4(d_mat1, p_mat_1, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 4*sizeof(WORD8));
    AE_L8X4S_IP(d_vec, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1, p_vec_1, 4);

    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    d_mat1 = AE_ADD16(d_mat1, d_mzb);

    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out3, d_mat1, d_vec1);
  }
  *out_0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0), AE_MOVINT32X2_FROMINT64(d_out1));
  *out_1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out2), AE_MOVINT32X2_FROMINT64(d_out3));
}

static inline void _xa_nn_dot_product_1row_2vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,WORD8*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD32      vecstride
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0;
  ae_int16x4 d_vec, d_vec1;
  ae_int64 d_out0, d_out1;
  WORD8 *p_vec_1;
  ae_int16x4 d_mzb;

  d_mzb = AE_MOVDA16(mat_zero_bias);
  p_vec_1 = p_vec_0 + vecstride;

  d_out0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_0_0));
  d_out1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_1_1));

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4S_IP(d_vec, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1, p_vec_1, 4);

    d_mat0 = AE_ADD16(d_mat0, d_mzb);

    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
    AE_MULAAAAQ16(d_out1, d_mat0, d_vec1);
  }
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
  *out_1_1 = AE_MOVINT32X2_FROMINT64(d_out1);
}

#if XCHAL_HAVE_HIFI1S
static inline void _xa_nn_dot_product_2row_4vec_mat_vecs_8bytes_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_vec_0
 ,WORD32      vecstride
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1;
  ae_int16x4 d_vec, d_vec1, d_vec2, d_vec3;
  ae_int64 d_out0, d_out1, d_out2, d_out3, d_out4, d_out5, d_out6, d_out7;
  WORD8 *p_vec_1, *p_vec_2, *p_vec_3;
  ae_int16x4 d_mzb;

  d_mzb = AE_MOVDA16(mat_zero_bias);
  p_vec_1 = p_vec_0 + vecstride;
  p_vec_2 = p_vec_1 + vecstride;
  p_vec_3 = p_vec_2 + vecstride;

  ae_int32x2 d_out0x2, d_out1x2, d_out2x2, d_out3x2, d_out4x2, d_out5x2, d_out6x2, d_out7x2;
  d_out0x2 = AE_SEL32_HH(0, *out_0_0);
  d_out1x2 = AE_SEL32_LL(0, *out_0_0);
  d_out2x2 = AE_SEL32_HH(0, *out_1_1);
  d_out3x2 = AE_SEL32_LL(0, *out_1_1);
  d_out4x2 = AE_SEL32_HH(0, *out_2_2);
  d_out5x2 = AE_SEL32_LL(0, *out_2_2);
  d_out6x2 = AE_SEL32_HH(0, *out_3_3);
  d_out7x2 = AE_SEL32_LL(0, *out_3_3);
  
  ae_valign align0 = AE_LA64_PP(p_vec_0);
  ae_valign align1 = AE_LA64_PP(p_vec_1);
  ae_valign align2 = AE_LA64_PP(p_vec_2);
  ae_valign align3 = AE_LA64_PP(p_vec_3);

  ae_int8x8 d_mzb8x8 = AE_MOVDA8(-mat_zero_bias);

  for(; c_itr<(cols1>>3); c_itr++)
  {
    ae_int8x8 d_vec8_0, d_vec8_1, d_vec8_2, d_vec8_3;
    ae_int16x4 d_mat0, d_mat1, d_mat2, d_mat3;

    ae_int8x8 d_mat0_8x8, d_mat1_8x8;
    AE_L8X8_XC(d_mat0_8x8, (ae_int8x8 *)p_mat_0, 8);
    AE_L8X8_XC(d_mat1_8x8, (ae_int8x8 *)p_mat_1, 8);
    AE_SUBW8(d_mat0, d_mat1, d_mat0_8x8, d_mzb8x8);
    AE_SUBW8(d_mat2, d_mat3, d_mat1_8x8, d_mzb8x8);

    AE_LA8X8_IP(d_vec8_0, align0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(d_vec8_1, align1, (ae_int8x8 *)p_vec_1);
    AE_LA8X8_IP(d_vec8_2, align2, (ae_int8x8 *)p_vec_2);
    AE_LA8X8_IP(d_vec8_3, align3, (ae_int8x8 *)p_vec_3);

    AE_MULAAAA16Q8(d_out0x2, d_mat0, d_mat1, d_vec8_0);
    AE_MULAAAA16Q8(d_out1x2, d_mat2, d_mat3, d_vec8_0);
    AE_MULAAAA16Q8(d_out2x2, d_mat0, d_mat1, d_vec8_1);
    AE_MULAAAA16Q8(d_out3x2, d_mat2, d_mat3, d_vec8_1);
    AE_MULAAAA16Q8(d_out4x2, d_mat0, d_mat1, d_vec8_2);
    AE_MULAAAA16Q8(d_out5x2, d_mat2, d_mat3, d_vec8_2);
    AE_MULAAAA16Q8(d_out6x2, d_mat0, d_mat1, d_vec8_3);
    AE_MULAAAA16Q8(d_out7x2, d_mat2, d_mat3, d_vec8_3);
  }

  d_out0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, AE_ADD32_HL_LH(d_out0x2, d_out0x2)));
  d_out1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, AE_ADD32_HL_LH(d_out1x2, d_out1x2)));
  d_out2 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, AE_ADD32_HL_LH(d_out2x2, d_out2x2)));
  d_out3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, AE_ADD32_HL_LH(d_out3x2, d_out3x2)));
  d_out4 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, AE_ADD32_HL_LH(d_out4x2, d_out4x2)));
  d_out5 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, AE_ADD32_HL_LH(d_out5x2, d_out5x2)));
  d_out6 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, AE_ADD32_HL_LH(d_out6x2, d_out6x2)));
  d_out7 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, AE_ADD32_HL_LH(d_out7x2, d_out7x2)));
  cols1 = cols1%8;

  for(c_itr=0; c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_L8X4S_I_HIFI4(d_mat1, p_mat_1, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 4*sizeof(WORD8));
    AE_L8X4S_IP(d_vec, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1, p_vec_1, 4);
    AE_L8X4S_IP(d_vec2, p_vec_2, 4);
    AE_L8X4S_IP(d_vec3, p_vec_3, 4);

    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    d_mat1 = AE_ADD16(d_mat1, d_mzb);

    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out3, d_mat1, d_vec1);
    AE_MULAAAAQ16(d_out4, d_mat0, d_vec2);
    AE_MULAAAAQ16(d_out5, d_mat1, d_vec2);
    AE_MULAAAAQ16(d_out6, d_mat0, d_vec3);
    AE_MULAAAAQ16(d_out7, d_mat1, d_vec3);
  }

  *out_0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0), AE_MOVINT32X2_FROMINT64(d_out1));
  *out_1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out2), AE_MOVINT32X2_FROMINT64(d_out3));
  *out_2_2 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out4), AE_MOVINT32X2_FROMINT64(d_out5));
  *out_3_3 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out6), AE_MOVINT32X2_FROMINT64(d_out7));
}
#endif

static inline void _xa_nn_dot_product_2row_4vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_vec_0
 ,WORD32      vecstride
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1;
  ae_int16x4 d_vec, d_vec1, d_vec2, d_vec3;
  ae_int64 d_out0, d_out1, d_out2, d_out3, d_out4, d_out5, d_out6, d_out7;
  WORD8 *p_vec_1, *p_vec_2, *p_vec_3;
  ae_int16x4 d_mzb;

  d_mzb = AE_MOVDA16(mat_zero_bias);
  p_vec_1 = p_vec_0 + vecstride;
  p_vec_2 = p_vec_1 + vecstride;
  p_vec_3 = p_vec_2 + vecstride;

  d_out0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, *out_0_0));
  d_out1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_0_0));
  d_out2 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, *out_1_1));
  d_out3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_1_1));
  d_out4 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, *out_2_2));
  d_out5 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_2_2));
  d_out6 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(0, *out_3_3));
  d_out7 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_3_3));

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_L8X4S_I_HIFI4(d_mat1, p_mat_1, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 4*sizeof(WORD8));
    AE_L8X4S_IP(d_vec, p_vec_0, 4);
    AE_L8X4S_IP(d_vec1, p_vec_1, 4);
    AE_L8X4S_IP(d_vec2, p_vec_2, 4);
    AE_L8X4S_IP(d_vec3, p_vec_3, 4);

    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    d_mat1 = AE_ADD16(d_mat1, d_mzb);

    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out3, d_mat1, d_vec1);
    AE_MULAAAAQ16(d_out4, d_mat0, d_vec2);
    AE_MULAAAAQ16(d_out5, d_mat1, d_vec2);
    AE_MULAAAAQ16(d_out6, d_mat0, d_vec3);
    AE_MULAAAAQ16(d_out7, d_mat1, d_vec3);
  }

  *out_0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0), AE_MOVINT32X2_FROMINT64(d_out1));
  *out_1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out2), AE_MOVINT32X2_FROMINT64(d_out3));
  *out_2_2 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out4), AE_MOVINT32X2_FROMINT64(d_out5));
  *out_3_3 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out6), AE_MOVINT32X2_FROMINT64(d_out7));
}

static inline void _xa_nn_dot_product_2row_1vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_vec_0
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_vec;
  ae_int16x4 d_mat0, d_mat1;
  ae_int64 d_out0, d_out1;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  d_out0 = d_out1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_0_0)); //AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_L8X4S_I_HIFI4(d_mat1, p_mat_1, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 4*sizeof(WORD8));
    AE_L8X4S_IP(d_vec, p_vec_0, 4);
    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    d_mat1 = AE_ADD16(d_mat1, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec);
  }
  /*d_out0 = AE_SRAI64(d_out0, 8);
    d_out1 = AE_SRAI64(d_out1, 8);*/
  *out_0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0), AE_MOVINT32X2_FROMINT64(d_out1));
}

static inline void _xa_nn_dot_product_1row_1vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,WORD8*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_vec;
  ae_int16x4 d_mat0;
  ae_int64 d_out0;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  d_out0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(0, *out_0_0));

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4S_IP(d_vec, p_vec_0, 4);
    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
  }
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
}


#else //XCHAL_HAVE_HIFI1
static inline void _xa_nn_dot_product_1row_4vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD32      vecstride
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0;
  ae_int16x4 d_vec, d_vec1, d_vec2, d_vec3;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  WORD8 *p_vec_1, *p_vec_2, *p_vec_3;
  ae_int16x4 d_mzb;

  d_mzb = AE_MOVDA16(mat_zero_bias);
  p_vec_1 = p_vec_0 + vecstride;
  p_vec_2 = p_vec_1 + vecstride;
  p_vec_3 = p_vec_2 + vecstride;

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_L(*out_2_2), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_3_3), 24);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vec, p_vec_0, 4);
    AE_L8X4F_IP(d_vec1, p_vec_1, 4);
    AE_L8X4F_IP(d_vec2, p_vec_2, 4);
    AE_L8X4F_IP(d_vec3, p_vec_3, 4);

    d_mat0 = AE_ADD16(d_mat0, d_mzb);

    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
    AE_MULAAAAQ16(d_out1, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec2);
    AE_MULAAAAQ16(d_out3, d_mat0, d_vec3);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  d_out2 = AE_SRAI64(d_out2, 8);
  d_out3 = AE_SRAI64(d_out3, 8);

  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
  *out_1_1 = AE_MOVINT32X2_FROMINT64(d_out1);
  *out_2_2 = AE_MOVINT32X2_FROMINT64(d_out2);
  *out_3_3 = AE_MOVINT32X2_FROMINT64(d_out3);
}

static inline void _xa_nn_dot_product_2row_2vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_vec_0
 ,WORD32      vecstride
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1;
  ae_int16x4 d_vec, d_vec1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  WORD8 *p_vec_1;
  ae_int16x4 d_mzb;

  d_mzb = AE_MOVDA16(mat_zero_bias);
  p_vec_1 = p_vec_0 + vecstride;

  d_out0 = d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out2 = d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_L8X4S_I_HIFI4(d_mat1, p_mat_1, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vec, p_vec_0, 4);
    AE_L8X4F_IP(d_vec1, p_vec_1, 4);

    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    d_mat1 = AE_ADD16(d_mat1, d_mzb);

    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out3, d_mat1, d_vec1);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  d_out2 = AE_SRAI64(d_out2, 8);
  d_out3 = AE_SRAI64(d_out3, 8);
  *out_0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0), AE_MOVINT32X2_FROMINT64(d_out1));
  *out_1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out2), AE_MOVINT32X2_FROMINT64(d_out3));
}

static inline void _xa_nn_dot_product_1row_2vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,WORD8*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD32      vecstride
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0;
  ae_int16x4 d_vec, d_vec1;
  ae_int64 d_out0, d_out1;
  WORD8 *p_vec_1;
  ae_int16x4 d_mzb;

  d_mzb = AE_MOVDA16(mat_zero_bias);
  p_vec_1 = p_vec_0 + vecstride;

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vec, p_vec_0, 4);
    AE_L8X4F_IP(d_vec1, p_vec_1, 4);

    d_mat0 = AE_ADD16(d_mat0, d_mzb);

    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
    AE_MULAAAAQ16(d_out1, d_mat0, d_vec1);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
  *out_1_1 = AE_MOVINT32X2_FROMINT64(d_out1);
}

static inline void _xa_nn_dot_product_2row_4vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_vec_0
 ,WORD32      vecstride
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1;
  ae_int16x4 d_vec, d_vec1, d_vec2, d_vec3;
  ae_int64 d_out0, d_out1, d_out2, d_out3, d_out4, d_out5, d_out6, d_out7;
  WORD8 *p_vec_1, *p_vec_2, *p_vec_3;
  ae_int16x4 d_mzb;

  d_mzb = AE_MOVDA16(mat_zero_bias);
  p_vec_1 = p_vec_0 + vecstride;
  p_vec_2 = p_vec_1 + vecstride;
  p_vec_3 = p_vec_2 + vecstride;

  d_out0 = AE_SRAI64(AE_CVT64F32_H(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_H(*out_1_1), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
  d_out4 = AE_SRAI64(AE_CVT64F32_H(*out_2_2), 24);
  d_out5 = AE_SRAI64(AE_CVT64F32_L(*out_2_2), 24);
  d_out6 = AE_SRAI64(AE_CVT64F32_H(*out_3_3), 24);
  d_out7 = AE_SRAI64(AE_CVT64F32_L(*out_3_3), 24);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_L8X4S_I_HIFI4(d_mat1, p_mat_1, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vec, p_vec_0, 4);
    AE_L8X4F_IP(d_vec1, p_vec_1, 4);
    AE_L8X4F_IP(d_vec2, p_vec_2, 4);
    AE_L8X4F_IP(d_vec3, p_vec_3, 4);

    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    d_mat1 = AE_ADD16(d_mat1, d_mzb);

    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vec1);
    AE_MULAAAAQ16(d_out3, d_mat1, d_vec1);
    AE_MULAAAAQ16(d_out4, d_mat0, d_vec2);
    AE_MULAAAAQ16(d_out5, d_mat1, d_vec2);
    AE_MULAAAAQ16(d_out6, d_mat0, d_vec3);
    AE_MULAAAAQ16(d_out7, d_mat1, d_vec3);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  d_out2 = AE_SRAI64(d_out2, 8);
  d_out3 = AE_SRAI64(d_out3, 8);
  d_out4 = AE_SRAI64(d_out4, 8);
  d_out5 = AE_SRAI64(d_out5, 8);
  d_out6 = AE_SRAI64(d_out6, 8);
  d_out7 = AE_SRAI64(d_out7, 8);
  *out_0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0), AE_MOVINT32X2_FROMINT64(d_out1));
  *out_1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out2), AE_MOVINT32X2_FROMINT64(d_out3));
  *out_2_2 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out4), AE_MOVINT32X2_FROMINT64(d_out5));
  *out_3_3 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out6), AE_MOVINT32X2_FROMINT64(d_out7));
}

static inline void _xa_nn_dot_product_2row_1vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_vec_0
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_vec;
  ae_int16x4 d_mat0, d_mat1;
  ae_int64 d_out0, d_out1;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  d_out0 = d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_L8X4S_I_HIFI4(d_mat1, p_mat_1, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vec, p_vec_0, 4);
    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    d_mat1 = AE_ADD16(d_mat1, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
    AE_MULAAAAQ16(d_out1, d_mat1, d_vec);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  *out_0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(d_out0), AE_MOVINT32X2_FROMINT64(d_out1));
}

static inline void _xa_nn_dot_product_1row_1vec_mat_vecs_4bytes_aligned
(ae_int32x2* out_0_0
 ,WORD8*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_vec;
  ae_int16x4 d_mat0;
  ae_int64 d_out0;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vec, p_vec_0, 4);
    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vec);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
}

#endif /* XCHAL_HAVE_HIFI1 */

WORD32 xa_nn_matXvec_sym8sxasym8s_asym8s_circ(
    WORD8 * __restrict__ p_out,
    WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_stride,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 mat1_offset,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);
  int out_stride = out_row_offset;
  int out_offset = out_col_offset;

  int left_shift, right_shift;
  int m_itr, vec_itr;
  ae_int32 zeros[4] = {0, 0, 0, 0};
#if XCHAL_HAVE_HIFI1S 
  if( p_mat1 && p_vec1 && (((unsigned int)p_mat1&0x7)==0) && (((unsigned int)p_vec1&0x3)==0) && (((unsigned int)p_bias&0x3) == 0) &&
      ((cols1&0x3)==0) && ((vec_stride&0x3)==0) && ((row_stride1&0x7)==0))
  {
    ae_int32 *bias_ptr = (ae_int32*)p_bias;
    if(p_bias == NULL){
      bias_ptr = zeros;
    }    
    for(vec_itr = 0; vec_itr < ((vec_count>>2)<<2); vec_itr+=4)
    {
      ae_int32x2 acc_row0_vec0, acc_row0_vec1, acc_row0_vec2, acc_row0_vec3;
      WORD32 l_shift[4], r_shift[4];
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;

#if TFLITE_SINGLE_ROUNDING
      l_shift[0] = 31 - p_out_shift[vec_itr+0];
      l_shift[1] = 31 - p_out_shift[vec_itr+1];
      l_shift[2] = 31 - p_out_shift[vec_itr+2];
      l_shift[3] = 31 - p_out_shift[vec_itr+3];
      l_shift[0] = l_shift[0] << 16 | l_shift[0];
      l_shift[1] = l_shift[1] << 16 | l_shift[1];
      l_shift[2] = l_shift[2] << 16 | l_shift[2];
      l_shift[3] = l_shift[3] << 16 | l_shift[3];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)r_shift[0];
      (void)r_shift[1];
      (void)r_shift[2];
      (void)r_shift[3];
#else /* #if TFLITE_SINGLE_ROUNDING */
      WORD32 o_shift[4];
      o_shift[0] = p_out_shift[vec_itr+0];
      l_shift[0] = o_shift[0]<0?0:o_shift[0];
      r_shift[0] = o_shift[0]>0?0:-o_shift[0];
      o_shift[1] = p_out_shift[vec_itr+1];
      l_shift[1] = o_shift[1]<0?0:o_shift[1];
      r_shift[1] = o_shift[1]>0?0:-o_shift[1];
      o_shift[2] = p_out_shift[vec_itr+2];
      l_shift[2] = o_shift[2]<0?0:o_shift[2];
      r_shift[2] = o_shift[2]>0?0:-o_shift[2];
      o_shift[3] = p_out_shift[vec_itr+3];
      l_shift[3] = o_shift[3]<0?0:o_shift[3];
      r_shift[3] = o_shift[3]>0?0:-o_shift[3];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      WORD8* p_vec_0  = (WORD8*)(p_vec1 + (vec_itr+0) * vec_stride);
      for (m_itr = 0; m_itr < ((rows>>1)<<1); m_itr+=2)
      {
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        WORD8 *p_mat1_1 = (WORD8*)p_mat1;
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        acc_row0_vec1 = AE_L32_I(bias_ptr, 4);
        acc_row0_vec2 = AE_L32_I(bias_ptr, 8);
        acc_row0_vec3 = AE_L32_I(bias_ptr, 12);

        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (m_itr+0) * row_stride1 * sizeof(WORD8));
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_2row_4vec_mat_vecs_8bytes_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,&acc_row0_vec2
           ,&acc_row0_vec3
           ,p_mat1_0
           ,p_mat1_1
           ,p_vec_0
           ,vec_stride
           ,cols1
           ,mat1_offset
          );

#if TFLITE_SINGLE_ROUNDING          
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr+0], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec2, acc_row0_vec2, p_out_multiplier[vec_itr+2], l_shift[2], r_shift[2]);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec3, acc_row0_vec3, p_out_multiplier[vec_itr+3], l_shift[3], r_shift[3]);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr+0], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec2, acc_row0_vec2, p_out_multiplier[vec_itr+2], l_shift[2], r_shift[2]);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec3, acc_row0_vec3, p_out_multiplier[vec_itr+3], l_shift[3], r_shift[3]);
#endif        
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
        acc_row0_vec2 = AE_ADD32S(acc_row0_vec2, out_zero_bias);
        acc_row0_vec3 = AE_ADD32S(acc_row0_vec3, out_zero_bias);

        // clamped_out
        ae_int8x8 clamped_1 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec1);
        ae_int8x8 clamped_3 = AE_SAT8X4X32_L(acc_row0_vec2, acc_row0_vec3);
        ae_int8x8 clamped_0 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_1), AE_MOVF16X4_FROMINT8X8(clamped_1)));
        ae_int8x8 clamped_2 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_3), AE_MOVF16X4_FROMINT8X8(clamped_3)));
        // Store Output
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_0),8)), (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_1),8)), (ae_int8 *)p_dst1, out_stride);
        AE_S8_0_XP(clamped_1, (ae_int8 *)p_dst1, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_2),8)), (ae_int8 *)p_dst2, out_stride);
        AE_S8_0_XP(clamped_2, (ae_int8 *)p_dst2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_3),8)), (ae_int8 *)p_dst3, out_stride);
        AE_S8_0_XP(clamped_3, (ae_int8 *)p_dst3, out_stride);

      }
      /* rows reminder loop */
      for (; m_itr < (rows); m_itr++)
      {
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        acc_row0_vec1 = AE_L32_I(bias_ptr, 4);
        acc_row0_vec2 = AE_L32_I(bias_ptr, 8);
        acc_row0_vec3 = AE_L32_I(bias_ptr, 12);

        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));
        _xa_nn_dot_product_1row_4vec_mat_vecs_4bytes_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,&acc_row0_vec2
           ,&acc_row0_vec3
           ,p_mat1_0
           ,p_vec_0
           ,vec_stride
           ,cols1
           ,mat1_offset
          );

#if TFLITE_SINGLE_ROUNDING   
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr+0], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec2, acc_row0_vec2, p_out_multiplier[vec_itr+2], l_shift[2], r_shift[2]);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec3, acc_row0_vec3, p_out_multiplier[vec_itr+3], l_shift[3], r_shift[3]);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr+0], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec2, acc_row0_vec2, p_out_multiplier[vec_itr+2], l_shift[2], r_shift[2]);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec3, acc_row0_vec3, p_out_multiplier[vec_itr+3], l_shift[3], r_shift[3]);
#endif        
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
        acc_row0_vec2 = AE_ADD32S(acc_row0_vec2, out_zero_bias);
        acc_row0_vec3 = AE_ADD32S(acc_row0_vec3, out_zero_bias);

        // clamped_out
        ae_int8x8 clamped_1 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec1);
        ae_int8x8 clamped_3 = AE_SAT8X4X32_L(acc_row0_vec2, acc_row0_vec3);
        ae_int8x8 clamped_0 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_1), AE_MOVF16X4_FROMINT8X8(clamped_1)));
        ae_int8x8 clamped_2 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_3), AE_MOVF16X4_FROMINT8X8(clamped_3)));
        // Store Output
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(clamped_1, (ae_int8 *)p_dst1, out_stride);
        AE_S8_0_XP(clamped_2, (ae_int8 *)p_dst2, out_stride);
        AE_S8_0_XP(clamped_3, (ae_int8 *)p_dst3, out_stride);

      }
      /* dummy load, just to increment the pointer */
      if(p_bias != NULL){
        AE_L32_IP(acc_row0_vec0, bias_ptr, 16);
      }
    }

    /* for vec_count=2 */
    if(vec_count&0x2)
    {
      WORD32 l_shift[2], r_shift[2];
      ae_int32x2 acc_row0_vec0;
      ae_int32x2 acc_row0_vec1;
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
#if TFLITE_SINGLE_ROUNDING
      l_shift[0] = 31 - p_out_shift[vec_itr];
      l_shift[1] = 31 - p_out_shift[vec_itr+1];
      l_shift[0] = l_shift[0] << 16 | l_shift[0];
      l_shift[1] = l_shift[1] << 16 | l_shift[1];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)r_shift[0];
      (void)r_shift[1];
#else /* #if TFLITE_SINGLE_ROUNDING */
      WORD32 o_shift[2];
      o_shift[0] = p_out_shift[vec_itr];
      l_shift[0] = o_shift[0]<0?0:o_shift[0];
      r_shift[0] = o_shift[0]>0?0:-o_shift[0];
      o_shift[1] = p_out_shift[vec_itr+1];
      l_shift[1] = o_shift[1]<0?0:o_shift[1];
      r_shift[1] = o_shift[1]>0?0:-o_shift[1];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      WORD8* p_vec_0  = (WORD8*)(p_vec1 + vec_itr * vec_stride);
      for (m_itr = 0; m_itr < ((rows>>1)<<1); m_itr+=2)
      {
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        WORD8 *p_mat1_1 = (WORD8*)p_mat1;
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        acc_row0_vec1 = AE_L32_I(bias_ptr, 4);
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (m_itr+0) * row_stride1 * sizeof(WORD8));
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_2row_2vec_mat_vecs_4bytes_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_mat1_0
           ,p_mat1_1
           ,p_vec_0
           ,vec_stride
           ,cols1
           ,mat1_offset
          );

#if TFLITE_SINGLE_ROUNDING           
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
#endif        
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);

        // clamped_out
        ae_int8x8 clamped_1 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec1);
        ae_int8x8 clamped_0 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_1), AE_MOVF16X4_FROMINT8X8(clamped_1)));
        // Store Output
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_0),8)), (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_1),8)), (ae_int8 *)p_dst1, out_stride);
        AE_S8_0_XP(clamped_1, (ae_int8 *)p_dst1, out_stride);

      }
      /* rows reminder loop */
      for (; m_itr < (rows); m_itr++)
      {
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        acc_row0_vec1 = AE_L32_I(bias_ptr, 4);
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));
        _xa_nn_dot_product_1row_2vec_mat_vecs_4bytes_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_mat1_0
           ,p_vec_0
           ,vec_stride
           ,cols1
           ,mat1_offset
          );

#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
#endif        
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);

        // clamped_out
        ae_int8x8 clamped_1 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec1);
        ae_int8x8 clamped_0 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_1), AE_MOVF16X4_FROMINT8X8(clamped_1)));
        // Store Output
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(clamped_1, (ae_int8 *)p_dst1, out_stride);
      }
      /* dummy load, just to increment the pointer */
      vec_itr+=2;
      if(p_bias != NULL){
        AE_L32_IP(acc_row0_vec0, bias_ptr, 8);
      }
    }

    /* for vec_count=1 */
    if(vec_count&0x1)
    {
      ae_int32x2 acc_row0_vec0;
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
#if TFLITE_SINGLE_ROUNDING
      left_shift = 31 - p_out_shift[vec_itr];
      right_shift = p_out_shift[vec_itr];
      left_shift = left_shift << 16 | left_shift;
     /* Single rounding macro doesn't need two shifts so this is not used */
      (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int out_shift;
      out_shift = p_out_shift[vec_itr];
      left_shift = out_shift<0?0:out_shift;
      right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

      WORD8* p_vec_0  = (WORD8*)(p_vec1 + vec_itr * vec_stride);
      for (m_itr = 0; m_itr < ((rows>>1)<<1); m_itr+=2)
      {
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        WORD8 *p_mat1_1 = (WORD8*)p_mat1;
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (m_itr+0) * row_stride1 * sizeof(WORD8));
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_2row_1vec_mat_vecs_4bytes_aligned
          (&acc_row0_vec0
           ,p_mat1_0
           ,p_mat1_1
           ,p_vec_0
           ,cols1
           ,mat1_offset
          );
#if TFLITE_SINGLE_ROUNDING          
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
#endif        
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);

        // clamped_out
        ae_int8x8 clamped_0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);
        // Store Output
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_0),8)), (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
      }
      /* rows reminder loop */
      for (; m_itr < (rows); m_itr++)
      {
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));
        _xa_nn_dot_product_1row_1vec_mat_vecs_4bytes_aligned
          (&acc_row0_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,mat1_offset
          );
#if TFLITE_SINGLE_ROUNDING  
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
#endif
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);

        ae_int8x8 clamped_0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
      }
      /* dummy load, just to increment the pointer */
      if(p_bias != NULL){
        AE_L32_IP(acc_row0_vec0, bias_ptr, 4);
      }
    }
  }
  else
#endif // XCHAL_HAVE_HIFI1S  
  if(p_mat1 && p_vec1 && (((unsigned int)p_mat1&0x3)==0) && (((unsigned int)p_vec1&0x3)==0) && (((unsigned int)p_bias&0x3) == 0) &&
      ((cols1&0x3)==0) && ((vec_stride&0x3)==0) && ((row_stride1&0x3)==0))
  {
    ae_int32 *bias_ptr = (ae_int32*)p_bias;
    if(p_bias == NULL){
      bias_ptr = zeros;
    }
    for(vec_itr = 0; vec_itr < ((vec_count>>2)<<2); vec_itr+=4)
    {
      ae_int32x2 acc_row0_vec0, acc_row0_vec1, acc_row0_vec2, acc_row0_vec3;
      WORD32 l_shift[4], r_shift[4];
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;

#if TFLITE_SINGLE_ROUNDING
      l_shift[0] = p_out_shift[vec_itr+0];
      l_shift[1] = p_out_shift[vec_itr+1];
      l_shift[2] = p_out_shift[vec_itr+2];
      l_shift[3] = p_out_shift[vec_itr+3];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)r_shift[0];
      (void)r_shift[1];
      (void)r_shift[2];
      (void)r_shift[3];
#else /* #if TFLITE_SINGLE_ROUNDING */
      WORD32 o_shift[4];
      o_shift[0] = p_out_shift[vec_itr+0];
      l_shift[0] = o_shift[0]<0?0:o_shift[0];
      r_shift[0] = o_shift[0]>0?0:-o_shift[0];
      o_shift[1] = p_out_shift[vec_itr+1];
      l_shift[1] = o_shift[1]<0?0:o_shift[1];
      r_shift[1] = o_shift[1]>0?0:-o_shift[1];
      o_shift[2] = p_out_shift[vec_itr+2];
      l_shift[2] = o_shift[2]<0?0:o_shift[2];
      r_shift[2] = o_shift[2]>0?0:-o_shift[2];
      o_shift[3] = p_out_shift[vec_itr+3];
      l_shift[3] = o_shift[3]<0?0:o_shift[3];
      r_shift[3] = o_shift[3]>0?0:-o_shift[3];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      WORD8* p_vec_0  = (WORD8*)(p_vec1 + (vec_itr+0) * vec_stride);
      for (m_itr = 0; m_itr < ((rows>>1)<<1); m_itr+=2)
      {
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        WORD8 *p_mat1_1 = (WORD8*)p_mat1;
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        acc_row0_vec1 = AE_L32_I(bias_ptr, 4);
        acc_row0_vec2 = AE_L32_I(bias_ptr, 8);
        acc_row0_vec3 = AE_L32_I(bias_ptr, 12);

        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (m_itr+0) * row_stride1 * sizeof(WORD8));
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_2row_4vec_mat_vecs_4bytes_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,&acc_row0_vec2
           ,&acc_row0_vec3
           ,p_mat1_0
           ,p_mat1_1
           ,p_vec_0
           ,vec_stride
           ,cols1
           ,mat1_offset
          );
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr+0], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec2, acc_row0_vec2, p_out_multiplier[vec_itr+2], l_shift[2], r_shift[2]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec3, acc_row0_vec3, p_out_multiplier[vec_itr+3], l_shift[3], r_shift[3]);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
        acc_row0_vec2 = AE_ADD32S(acc_row0_vec2, out_zero_bias);
        acc_row0_vec3 = AE_ADD32S(acc_row0_vec3, out_zero_bias);

#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
        // clamped_out
        ae_int8x8 clamped_1 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec1);
        ae_int8x8 clamped_3 = AE_SAT8X4X32_L(acc_row0_vec2, acc_row0_vec3);
        ae_int8x8 clamped_0 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_1), AE_MOVF16X4_FROMINT8X8(clamped_1)));
        ae_int8x8 clamped_2 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_3), AE_MOVF16X4_FROMINT8X8(clamped_3)));
        // Store Output
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_0),8)), (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_1),8)), (ae_int8 *)p_dst1, out_stride);
        AE_S8_0_XP(clamped_1, (ae_int8 *)p_dst1, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_2),8)), (ae_int8 *)p_dst2, out_stride);
        AE_S8_0_XP(clamped_2, (ae_int8 *)p_dst2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_3),8)), (ae_int8 *)p_dst3, out_stride);
        AE_S8_0_XP(clamped_3, (ae_int8 *)p_dst3, out_stride);
#else
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec1, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec2, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec3, min_int8, max_int8);
        AE_S8_FROM32X2_WITHSTRIDE(acc_row0_vec0, p_dst0, out_stride);
        AE_S8_FROM32X2_WITHSTRIDE(acc_row0_vec1, p_dst1, out_stride);
        AE_S8_FROM32X2_WITHSTRIDE(acc_row0_vec2, p_dst2, out_stride);
        AE_S8_FROM32X2_WITHSTRIDE(acc_row0_vec3, p_dst3, out_stride);
#endif
#else
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec1, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec2, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec3, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row0_vec1), p_dst1, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec1), p_dst1, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row0_vec2), p_dst2, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec2), p_dst2, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row0_vec3), p_dst3, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec3), p_dst3, out_stride);
#endif
      }
      /* rows reminder loop */
      for (; m_itr < (rows); m_itr++)
      {
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        acc_row0_vec1 = AE_L32_I(bias_ptr, 4);
        acc_row0_vec2 = AE_L32_I(bias_ptr, 8);
        acc_row0_vec3 = AE_L32_I(bias_ptr, 12);

        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));
        _xa_nn_dot_product_1row_4vec_mat_vecs_4bytes_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,&acc_row0_vec2
           ,&acc_row0_vec3
           ,p_mat1_0
           ,p_vec_0
           ,vec_stride
           ,cols1
           ,mat1_offset
          );

#if XCHAL_HAVE_HIFI1

        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr+0], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec2, acc_row0_vec2, p_out_multiplier[vec_itr+2], l_shift[2], r_shift[2]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec3, acc_row0_vec3, p_out_multiplier[vec_itr+3], l_shift[3], r_shift[3]);
        
#else
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr+0], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec2, acc_row0_vec2, p_out_multiplier[vec_itr+2], l_shift[2], r_shift[2]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec3, acc_row0_vec3, p_out_multiplier[vec_itr+3], l_shift[3], r_shift[3]);
#endif
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
        acc_row0_vec2 = AE_ADD32S(acc_row0_vec2, out_zero_bias);
        acc_row0_vec3 = AE_ADD32S(acc_row0_vec3, out_zero_bias);
#if (XCHAL_HAVE_HIFI1 & ( XCHAL_HW_VERSION >= RI9_HWVERSION ))
        // clamped_out
        ae_int8x8 clamped_1 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec1);
        ae_int8x8 clamped_3 = AE_SAT8X4X32_L(acc_row0_vec2, acc_row0_vec3);
        ae_int8x8 clamped_0 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_1), AE_MOVF16X4_FROMINT8X8(clamped_1)));
        ae_int8x8 clamped_2 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_3), AE_MOVF16X4_FROMINT8X8(clamped_3)));
        // Store Output
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(clamped_1, (ae_int8 *)p_dst1, out_stride);
        AE_S8_0_XP(clamped_2, (ae_int8 *)p_dst2, out_stride);
        AE_S8_0_XP(clamped_3, (ae_int8 *)p_dst3, out_stride);
#else
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec1, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec2, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec3, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec1), p_dst1, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec2), p_dst2, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec3), p_dst3, out_stride);
#endif
      }
      /* dummy load, just to increment the pointer */
      if(p_bias != NULL){
        AE_L32_IP(acc_row0_vec0, bias_ptr, 16);
      }
    }

    /* for vec_count=2 */
    if(vec_count&0x2)
    {
      WORD32 l_shift[2], r_shift[2];
      ae_int32x2 acc_row0_vec0;
      ae_int32x2 acc_row0_vec1;
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
#if TFLITE_SINGLE_ROUNDING
      l_shift[0] = p_out_shift[vec_itr];
      l_shift[1] = p_out_shift[vec_itr+1];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)r_shift[0];
      (void)r_shift[1];
#else /* #if TFLITE_SINGLE_ROUNDING */
      WORD32 o_shift[2];
      o_shift[0] = p_out_shift[vec_itr];
      l_shift[0] = o_shift[0]<0?0:o_shift[0];
      r_shift[0] = o_shift[0]>0?0:-o_shift[0];
      o_shift[1] = p_out_shift[vec_itr+1];
      l_shift[1] = o_shift[1]<0?0:o_shift[1];
      r_shift[1] = o_shift[1]>0?0:-o_shift[1];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      WORD8* p_vec_0  = (WORD8*)(p_vec1 + vec_itr * vec_stride);
      for (m_itr = 0; m_itr < ((rows>>1)<<1); m_itr+=2)
      {
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        WORD8 *p_mat1_1 = (WORD8*)p_mat1;
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        acc_row0_vec1 = AE_L32_I(bias_ptr, 4);
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (m_itr+0) * row_stride1 * sizeof(WORD8));
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_2row_2vec_mat_vecs_4bytes_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_mat1_0
           ,p_mat1_1
           ,p_vec_0
           ,vec_stride
           ,cols1
           ,mat1_offset
          );
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
        // clamped_out
        ae_int8x8 clamped_1 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec1);
        ae_int8x8 clamped_0 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_1), AE_MOVF16X4_FROMINT8X8(clamped_1)));
        // Store Output
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_0),8)), (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_1),8)), (ae_int8 *)p_dst1, out_stride);
        AE_S8_0_XP(clamped_1, (ae_int8 *)p_dst1, out_stride);
#else
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec1, min_int8, max_int8);
        AE_S8_FROM32X2_WITHSTRIDE(acc_row0_vec0, p_dst0, out_stride);
        AE_S8_FROM32X2_WITHSTRIDE(acc_row0_vec1, p_dst1, out_stride);
#endif
#else
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec1, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row0_vec1), p_dst1, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec1), p_dst1, out_stride);
#endif
      }
      /* rows reminder loop */
      for (; m_itr < (rows); m_itr++)
      {
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        acc_row0_vec1 = AE_L32_I(bias_ptr, 4);
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));
        _xa_nn_dot_product_1row_2vec_mat_vecs_4bytes_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_mat1_0
           ,p_vec_0
           ,vec_stride
           ,cols1
           ,mat1_offset
          );
#if XCHAL_HAVE_HIFI1
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
#else
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], l_shift[0], r_shift[0]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
#endif
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
#if (XCHAL_HAVE_HIFI1 & ( XCHAL_HW_VERSION >= RI9_HWVERSION ))
        // clamped_out
        ae_int8x8 clamped_1 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec1);
        ae_int8x8 clamped_0 = AE_MOVINT8X8_FROMINT16X4(AE_SEL16_2301(AE_MOVF16X4_FROMINT8X8(clamped_1), AE_MOVF16X4_FROMINT8X8(clamped_1)));
        // Store Output
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(clamped_1, (ae_int8 *)p_dst1, out_stride);
#else
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec1, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec1), p_dst1, out_stride);
#endif
      }
      /* dummy load, just to increment the pointer */
      vec_itr+=2;
      if(p_bias != NULL){
        AE_L32_IP(acc_row0_vec0, bias_ptr, 8);
      }
    }

    /* for vec_count=1 */
    if(vec_count&0x1)
    {
      ae_int32x2 acc_row0_vec0;
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
#if TFLITE_SINGLE_ROUNDING
      left_shift = p_out_shift[vec_itr];
      right_shift = p_out_shift[vec_itr];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int out_shift;
      out_shift = p_out_shift[vec_itr];
      left_shift = out_shift<0?0:out_shift;
      right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

      WORD8* p_vec_0  = (WORD8*)(p_vec1 + vec_itr * vec_stride);
      for (m_itr = 0; m_itr < ((rows>>1)<<1); m_itr+=2)
      {
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        WORD8 *p_mat1_1 = (WORD8*)p_mat1;
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (m_itr+0) * row_stride1 * sizeof(WORD8));
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_2row_1vec_mat_vecs_4bytes_aligned
          (&acc_row0_vec0
           ,p_mat1_0
           ,p_mat1_1
           ,p_vec_0
           ,cols1
           ,mat1_offset
          );
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
        // clamped_out
        ae_int8x8 clamped_0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);
        // Store Output
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(AE_SRAI16(AE_MOVF16X4_FROMINT8X8(clamped_0),8)), (ae_int8 *)p_dst0, out_stride);
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
#else
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_S8_FROM32X2_WITHSTRIDE(acc_row0_vec0, p_dst0, out_stride);
#endif
#else
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
#endif
      }
      /* rows reminder loop */
      for (; m_itr < (rows); m_itr++)
      {
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));
        _xa_nn_dot_product_1row_1vec_mat_vecs_4bytes_aligned
          (&acc_row0_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,mat1_offset
          );
#if XCHAL_HAVE_HIFI1
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
#endif
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
#if (XCHAL_HAVE_HIFI1 & ( XCHAL_HW_VERSION >= RI9_HWVERSION ))
        ae_int8x8 clamped_0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);
        AE_S8_0_XP(clamped_0, (ae_int8 *)p_dst0, out_stride);
#else
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
#endif
      }
      /* dummy load, just to increment the pointer */
      if(p_bias != NULL){
        AE_L32_IP(acc_row0_vec0, bias_ptr, 4);
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
    ae_int32 *bias_ptr = (ae_int32*)p_bias;
    if(p_bias == NULL){
      bias_ptr = zeros;
    }     
//#if !ENABLE_PADDING_CONV2D_STD
#if HW_AE_ADDCIRC16X4_XC
    vec_itr = 0; 
#if XCHAL_HAVE_HIFI1S
    for(; vec_itr < (vec_count&~1); vec_itr+=2)
    {
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
#if TFLITE_SINGLE_ROUNDING
      int left_shift0 = 31 - p_out_shift[vec_itr];
      int left_shift1 = 31 - p_out_shift[vec_itr+1];
      left_shift0 = left_shift0 << 16 | left_shift0;
      left_shift1 = left_shift1 << 16 | left_shift1;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int left_shift0 = p_out_shift[vec_itr]<0?0:p_out_shift[vec_itr];
      int right_shift0 = p_out_shift[vec_itr]>0?0:-p_out_shift[vec_itr];
      int left_shift1 = p_out_shift[vec_itr+1]<0?0:p_out_shift[vec_itr+1];
      int right_shift1 = p_out_shift[vec_itr+1]>0?0:-p_out_shift[vec_itr+1];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      WORD8* p_dst1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;

      m_itr = 0;
      for (; m_itr < (rows&~1); m_itr+=2)
      {
        int c_itr = 0;

        WORD8* p_vec_0  = (WORD8*)(p_vec1 + (vec_itr+0) * vec_stride);
        WORD8* p_vec_1  = (WORD8*)(p_vec1 + (vec_itr+1) * vec_stride);

        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));
        WORD8 *p_mat1_1 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1 * sizeof(WORD8));

        ae_int32x2 acc00, acc01, acc10, acc11;
        acc00 = acc01 = acc10 = acc11 = 0;

        ae_valign align_mat0, align_mat1, align_vec0, align_vec1;
        align_vec0 = AE_LA64_PP(p_vec_0);
        align_vec1 = AE_LA64_PP(p_vec_1);
        AE_LA8X8POS_PC(align_mat0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8POS_PC(align_mat1, (ae_int8x8 *)p_mat1_1);

        ae_int16x4 d_mzb = AE_MOVDA16(mat1_offset);
        ae_int8x8 d_mzb8 = AE_MOVDA8(-mat1_offset);

        for(; c_itr<(cols1&~7); c_itr+=8)
        {
          ae_int8x8 d_vec0, d_vec1;
          ae_int8x8 d_mat0, d_mat1;
          ae_int16x4 d_mat0_0, d_mat0_1, d_mat1_0, d_mat1_1;

          AE_LA8X8_IC(d_mat0, align_mat0, (ae_int8x8 *)p_mat1_0);
          AE_LA8X8_IC(d_mat1, align_mat1, (ae_int8x8 *)p_mat1_1);
          AE_SUBW8(d_mat0_0, d_mat0_1, d_mat0, d_mzb8);
          AE_SUBW8(d_mat1_0, d_mat1_1, d_mat1, d_mzb8);
          AE_LA8X8_IP(d_vec0, align_vec0, (ae_int8x8 *)p_vec_0);
          AE_LA8X8_IP(d_vec1, align_vec1, (ae_int8x8 *)p_vec_1);
          
          AE_MULAAAA16Q8(acc00, d_mat0_0, d_mat0_1, d_vec0);
          AE_MULAAAA16Q8(acc01, d_mat0_0, d_mat0_1, d_vec1);
          AE_MULAAAA16Q8(acc10, d_mat1_0, d_mat1_1, d_vec0);
          AE_MULAAAA16Q8(acc11, d_mat1_0, d_mat1_1, d_vec1);
        }

        if(cols1%8) /* Following code could use LAV8X8 if available */
        {
          int rembits = 64 - (cols1%8)*8;
          ae_int8x8 d_vec0, d_vec1;
          ae_int8x8 d_mat0, d_mat1;
          ae_int16x4 d_mat0_0, d_mat0_1, d_mat1_0, d_mat1_1;

          AE_LA8X8_IC(d_mat0, align_mat0, (ae_int8x8 *)p_mat1_0);
          AE_LA8X8_IC(d_mat1, align_mat1, (ae_int8x8 *)p_mat1_1);
          d_mat0_0 =  AE_SEXT16X4Q8_H(d_mat0);
          d_mat0_1 =  AE_SEXT16X4Q8_L(d_mat0);
          d_mat1_0 =  AE_SEXT16X4Q8_H(d_mat1);
          d_mat1_1 =  AE_SEXT16X4Q8_L(d_mat1);

          AE_LA8X8_IP(d_vec0, align_vec0, (ae_int8x8 *)p_vec_0);
          AE_LA8X8_IP(d_vec1, align_vec1, (ae_int8x8 *)p_vec_1);
          d_vec0 = AE_MOVINT8X8_FROMINT64( AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(d_vec0), rembits), rembits) );
          d_vec1 = AE_MOVINT8X8_FROMINT64( AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(d_vec1), rembits), rembits) );

          d_mat0_0 = AE_ADD16(d_mat0_0, d_mzb);
          d_mat0_1 = AE_ADD16(d_mat0_1, d_mzb);
          d_mat1_0 = AE_ADD16(d_mat1_0, d_mzb);
          d_mat1_1 = AE_ADD16(d_mat1_1, d_mzb);
          
          AE_MULAAAA16Q8(acc00, d_mat0_0, d_mat0_1, d_vec0);
          AE_MULAAAA16Q8(acc01, d_mat0_0, d_mat0_1, d_vec1);
          AE_MULAAAA16Q8(acc10, d_mat1_0, d_mat1_1, d_vec0);
          AE_MULAAAA16Q8(acc11, d_mat1_0, d_mat1_1, d_vec1);
        }
        acc00 = AE_ADD32_HL_LH(acc00, acc00);
        acc01 = AE_ADD32_HL_LH(acc01, acc01);
        acc10 = AE_ADD32_HL_LH(acc10, acc10);
        acc11 = AE_ADD32_HL_LH(acc11, acc11);

        acc00 = AE_SEL32_HH(acc00, acc10);
        acc10 = AE_SEL32_HH(acc01, acc11);

        ae_int32x2 acc_row01_vec0 = AE_L32_I(bias_ptr, 0);
        ae_int32x2 acc_row01_vec1 = AE_L32_I(bias_ptr, 4);

        acc_row01_vec0 = AE_ADD32(acc_row01_vec0, acc00);
        acc_row01_vec1 = AE_ADD32(acc_row01_vec1, acc10);

#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row01_vec0, acc_row01_vec0, p_out_multiplier[vec_itr], left_shift0, right_shift0);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row01_vec0, acc_row01_vec0, p_out_multiplier[vec_itr], left_shift0, right_shift0);
#endif        
        acc_row01_vec0 = AE_ADD32S(acc_row01_vec0, out_zero_bias);
        acc_row01_vec0 = AE_MAX32(acc_row01_vec0, min_int8);
        acc_row01_vec0 = AE_MIN32(acc_row01_vec0, max_int8);

#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row01_vec1, acc_row01_vec1, p_out_multiplier[vec_itr+1], left_shift1, right_shift1);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row01_vec1, acc_row01_vec1, p_out_multiplier[vec_itr+1], left_shift1, right_shift1);
#endif        
        acc_row01_vec1 = AE_ADD32S(acc_row01_vec1, out_zero_bias);
        acc_row01_vec1 = AE_MAX32(acc_row01_vec1, min_int8);
        acc_row01_vec1 = AE_MIN32(acc_row01_vec1, max_int8);

        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row01_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row01_vec0), p_dst0, out_stride);

        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row01_vec1), p_dst1, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row01_vec1), p_dst1, out_stride);
      }

      /* Remainder Loop */
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        ae_int32x2 acc_row0_vec1 = AE_L32_I(bias_ptr, 4);

        WORD8* p_vec_0  = (WORD8*)(p_vec1 + (vec_itr+0) * vec_stride);
        WORD8* p_vec_1  = (WORD8*)(p_vec1 + (vec_itr+1) * vec_stride);

        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int32x2 acc00, acc01;
        acc00 = acc01 = 0;

        ae_valign align_mat0, align_vec0, align_vec1;
        align_vec0 = AE_LA64_PP(p_vec_0);
        align_vec1 = AE_LA64_PP(p_vec_1);
        AE_LA8X8POS_PC(align_mat0, (ae_int8x8 *)p_mat1_0);

        ae_int16x4 d_mzb = AE_MOVDA16(mat1_offset);

        int c_itr;
        for(c_itr = 0; c_itr<(cols1&~7); c_itr+=8)
        {
          ae_int8x8 d_vec0, d_vec1;
          ae_int8x8 d_mat0;
          ae_int16x4 d_mat0_0, d_mat0_1;

          AE_LA8X8_IC(d_mat0, align_mat0, (ae_int8x8 *)p_mat1_0);
          d_mat0_0 =  AE_SEXT16X4Q8_H(d_mat0);
          d_mat0_1 =  AE_SEXT16X4Q8_L(d_mat0);

          AE_LA8X8_IP(d_vec0, align_vec0, (ae_int8x8 *)p_vec_0);
          AE_LA8X8_IP(d_vec1, align_vec1, (ae_int8x8 *)p_vec_1);

          d_mat0_0 = AE_ADD16(d_mat0_0, d_mzb);
          d_mat0_1 = AE_ADD16(d_mat0_1, d_mzb);
          
          AE_MULAAAA16Q8(acc00, d_mat0_0, d_mat0_1, d_vec0);
          AE_MULAAAA16Q8(acc01, d_mat0_0, d_mat0_1, d_vec1);
        }

        if(cols1%8)
        {
          ae_int8x8 d_vec0, d_vec1;
          ae_int8x8 d_mat0;
          ae_int16x4 d_mat0_0, d_mat0_1;

          AE_LA8X8_IC(d_mat0, align_mat0, (ae_int8x8 *)p_mat1_0);
          d_mat0_0 =  AE_SEXT16X4Q8_H(d_mat0);
          d_mat0_1 =  AE_SEXT16X4Q8_L(d_mat0);

          AE_LA8X8_IP(d_vec0, align_vec0, (ae_int8x8 *)p_vec_0);
          AE_LA8X8_IP(d_vec1, align_vec1, (ae_int8x8 *)p_vec_1);

          int rembits = 64 - (cols1%8)*8;
          d_vec0 = AE_MOVINT8X8_FROMINT64( AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(d_vec0), rembits), rembits) );
          d_vec1 = AE_MOVINT8X8_FROMINT64( AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(d_vec1), rembits), rembits) );

          d_mat0_0 = AE_ADD16(d_mat0_0, d_mzb);
          d_mat0_1 = AE_ADD16(d_mat0_1, d_mzb);
          
          AE_MULAAAA16Q8(acc00, d_mat0_0, d_mat0_1, d_vec0);
          AE_MULAAAA16Q8(acc01, d_mat0_0, d_mat0_1, d_vec1);
        }

        acc00 = AE_ADD32_HL_LH(acc00, acc00);
        acc01 = AE_ADD32_HL_LH(acc01, acc01);

        acc_row0_vec0 = AE_ADD32(acc_row0_vec0, acc00);
        acc_row0_vec1 = AE_ADD32(acc_row0_vec1, acc01);

#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift0, right_shift0);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift0, right_shift0);
#endif        
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
        acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);

#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], left_shift1, right_shift1);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], left_shift1, right_shift1);
#endif        
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
        acc_row0_vec1 = AE_MAX32(acc_row0_vec1, min_int8);
        acc_row0_vec1 = AE_MIN32(acc_row0_vec1, max_int8);

        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec1), p_dst1, out_stride);
      }
      if(p_bias != NULL){
        AE_L32_IP(acc_row0_vec0, bias_ptr, 8);
      }
    }
#endif
    for(; vec_itr < (vec_count&~3); vec_itr+=4)
    {
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
#if TFLITE_SINGLE_ROUNDING
      int left_shift0 = p_out_shift[vec_itr];
      int left_shift1 = p_out_shift[vec_itr+1];
      int left_shift2 = p_out_shift[vec_itr+2];
      int left_shift3 = p_out_shift[vec_itr+3];
#else /* #if TFLITE_SINGLE_ROUNDING */
      int left_shift0 = p_out_shift[vec_itr]<0?0:p_out_shift[vec_itr];
      int right_shift0 = p_out_shift[vec_itr]>0?0:-p_out_shift[vec_itr];
      int left_shift1 = p_out_shift[vec_itr+1]<0?0:p_out_shift[vec_itr+1];
      int right_shift1 = p_out_shift[vec_itr+1]>0?0:-p_out_shift[vec_itr+1];
      int left_shift2 = p_out_shift[vec_itr+2]<0?0:p_out_shift[vec_itr+2];
      int right_shift2 = p_out_shift[vec_itr+2]>0?0:-p_out_shift[vec_itr+2];
      int left_shift3 = p_out_shift[vec_itr+3]<0?0:p_out_shift[vec_itr+3];
      int right_shift3 = p_out_shift[vec_itr+3]>0?0:-p_out_shift[vec_itr+3];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      WORD8* p_dst1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;

      ae_int16x4 mat1_offset_dr = mat1_offset;
      m_itr = 0;
      for (; m_itr < (rows&~3); m_itr+=4)
      {
        int c_itr;
        ae_int32x2 acc_row01_vec0 = AE_L32_I(bias_ptr, 0);
        ae_int32x2 acc_row23_vec0 = AE_L32_I(bias_ptr, 0);
        ae_int32x2 acc_row01_vec1 = AE_L32_I(bias_ptr, 4);
        ae_int32x2 acc_row23_vec1 = AE_L32_I(bias_ptr, 4);
        ae_int32x2 acc_row01_vec2 = AE_L32_I(bias_ptr, 8);
        ae_int32x2 acc_row23_vec2 = AE_L32_I(bias_ptr, 8);
        ae_int32x2 acc_row01_vec3 = AE_L32_I(bias_ptr, 12);
        ae_int32x2 acc_row23_vec3 = AE_L32_I(bias_ptr, 12);

        WORD8* p_vec_0  = (WORD8*)(p_vec1 + (vec_itr+0) * vec_stride);
        WORD8* p_vec_1  = (WORD8*)(p_vec1 + (vec_itr+1) * vec_stride);
        WORD8* p_vec_2  = (WORD8*)(p_vec1 + (vec_itr+2) * vec_stride);
        WORD8* p_vec_3  = (WORD8*)(p_vec1 + (vec_itr+3) * vec_stride);

        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));
        WORD8 *p_mat1_1 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1 * sizeof(WORD8));
        WORD8 *p_mat1_2 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_2, (m_itr+2) * row_stride1 * sizeof(WORD8));
        WORD8 *p_mat1_3 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_3, (m_itr+3) * row_stride1 * sizeof(WORD8));

        for(c_itr=0; c_itr<(cols1); c_itr++)
        {
          ae_int16x4 d_mat;
          ae_int32x2 d_vec0, d_vec1, d_vec2, d_vec3;
          ae_int32x2 mat_tmp1, mat_tmp2;
          UWORD32 mat0, mat1, mat2, mat3;
          UWORD32 vec0, vec1, vec2, vec3;

          vec0 = (*((UWORD8 *)p_vec_0)); p_vec_0++;
          vec1 = (*((UWORD8 *)p_vec_1)); p_vec_1++;
          vec2 = (*((UWORD8 *)p_vec_2)); p_vec_2++;
          vec3 = (*((UWORD8 *)p_vec_3)); p_vec_3++;
          ae_int32x2 vectmp_01 = AE_MOVDA32X2(vec0, vec1);
          ae_int32x2 vectmp_23 = AE_MOVDA32X2(vec2, vec3);
          vectmp_01 = AE_SEXT32(vectmp_01, 7);
          vectmp_23 = AE_SEXT32(vectmp_23, 7);
          d_vec0 = AE_SEL32_HH(vectmp_01, vectmp_01);
          d_vec1 = AE_SEL32_LL(vectmp_01, vectmp_01);
          d_vec2 = AE_SEL32_HH(vectmp_23, vectmp_23);
          d_vec3 = AE_SEL32_LL(vectmp_23, vectmp_23);

          mat0 = *((UWORD8 *)p_mat1_0);
          mat1 = *((UWORD8 *)p_mat1_1);
          mat2 = *((UWORD8 *)p_mat1_2);
          mat3 = *((UWORD8 *)p_mat1_3);
          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, sizeof(WORD8));
          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, sizeof(WORD8));
          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_2, sizeof(WORD8));
          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_3, sizeof(WORD8));

          mat_tmp1 = AE_SEXT32(AE_MOVDA32X2(mat0, mat1), 7);
          mat_tmp2 = AE_SEXT32(AE_MOVDA32X2(mat2, mat3), 7);

          d_mat = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(mat_tmp1), AE_MOVINT16X4_FROMINT32X2(mat_tmp2));

          d_mat = AE_ADD16(d_mat, mat1_offset_dr);
          AE_MULAP32X16X2_H(acc_row01_vec0, d_vec0, d_mat);
          AE_MULAP32X16X2_L(acc_row23_vec0, d_vec0, d_mat);
          AE_MULAP32X16X2_H(acc_row01_vec1, d_vec1, d_mat);
          AE_MULAP32X16X2_L(acc_row23_vec1, d_vec1, d_mat);
          AE_MULAP32X16X2_H(acc_row01_vec2, d_vec2, d_mat);
          AE_MULAP32X16X2_L(acc_row23_vec2, d_vec2, d_mat);
          AE_MULAP32X16X2_H(acc_row01_vec3, d_vec3, d_mat);
          AE_MULAP32X16X2_L(acc_row23_vec3, d_vec3, d_mat);
        }

        MPY_BY_QUANT_MULT_X2_OUT32(acc_row01_vec0, acc_row01_vec0, p_out_multiplier[vec_itr], left_shift0, right_shift0);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row23_vec0, acc_row23_vec0, p_out_multiplier[vec_itr], left_shift0, right_shift0);
        acc_row01_vec0 = AE_ADD32S(acc_row01_vec0, out_zero_bias);
        acc_row01_vec0 = AE_MAX32(acc_row01_vec0, min_int8);
        acc_row01_vec0 = AE_MIN32(acc_row01_vec0, max_int8);
        acc_row23_vec0 = AE_ADD32S(acc_row23_vec0, out_zero_bias);
        acc_row23_vec0 = AE_MAX32(acc_row23_vec0, min_int8);
        acc_row23_vec0 = AE_MIN32(acc_row23_vec0, max_int8);

        MPY_BY_QUANT_MULT_X2_OUT32(acc_row01_vec1, acc_row01_vec1, p_out_multiplier[vec_itr+1], left_shift1, right_shift1);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row23_vec1, acc_row23_vec1, p_out_multiplier[vec_itr+1], left_shift1, right_shift1);
        acc_row01_vec1 = AE_ADD32S(acc_row01_vec1, out_zero_bias);
        acc_row01_vec1 = AE_MAX32(acc_row01_vec1, min_int8);
        acc_row01_vec1 = AE_MIN32(acc_row01_vec1, max_int8);
        acc_row23_vec1 = AE_ADD32S(acc_row23_vec1, out_zero_bias);
        acc_row23_vec1 = AE_MAX32(acc_row23_vec1, min_int8);
        acc_row23_vec1 = AE_MIN32(acc_row23_vec1, max_int8);

        MPY_BY_QUANT_MULT_X2_OUT32(acc_row01_vec2, acc_row01_vec2, p_out_multiplier[vec_itr+2], left_shift2, right_shift2);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row23_vec2, acc_row23_vec2, p_out_multiplier[vec_itr+2], left_shift2, right_shift2);
        acc_row01_vec2 = AE_ADD32S(acc_row01_vec2, out_zero_bias);
        acc_row01_vec2 = AE_MAX32(acc_row01_vec2, min_int8);
        acc_row01_vec2 = AE_MIN32(acc_row01_vec2, max_int8);
        acc_row23_vec2 = AE_ADD32S(acc_row23_vec2, out_zero_bias);
        acc_row23_vec2 = AE_MAX32(acc_row23_vec2, min_int8);
        acc_row23_vec2 = AE_MIN32(acc_row23_vec2, max_int8);

        MPY_BY_QUANT_MULT_X2_OUT32(acc_row01_vec3, acc_row01_vec3, p_out_multiplier[vec_itr+3], left_shift3, right_shift3);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row23_vec3, acc_row23_vec3, p_out_multiplier[vec_itr+3], left_shift3, right_shift3);
        acc_row01_vec3 = AE_ADD32S(acc_row01_vec3, out_zero_bias);
        acc_row01_vec3 = AE_MAX32(acc_row01_vec3, min_int8);
        acc_row01_vec3 = AE_MIN32(acc_row01_vec3, max_int8);
        acc_row23_vec3 = AE_ADD32S(acc_row23_vec3, out_zero_bias);
        acc_row23_vec3 = AE_MAX32(acc_row23_vec3, min_int8);
        acc_row23_vec3 = AE_MIN32(acc_row23_vec3, max_int8);

        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row01_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row01_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row23_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row23_vec0), p_dst0, out_stride);

        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row01_vec1), p_dst1, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row01_vec1), p_dst1, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row23_vec1), p_dst1, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row23_vec1), p_dst1, out_stride);

        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row01_vec2), p_dst2, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row01_vec2), p_dst2, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row23_vec2), p_dst2, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row23_vec2), p_dst2, out_stride);

        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row01_vec3), p_dst3, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row01_vec3), p_dst3, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row23_vec3), p_dst3, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row23_vec3), p_dst3, out_stride);
      }

      /* Remainder Loop */
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        ae_int32x2 acc_row0_vec1 = AE_L32_I(bias_ptr, 4);
        ae_int32x2 acc_row0_vec2 = AE_L32_I(bias_ptr, 8);
        ae_int32x2 acc_row0_vec3 = AE_L32_I(bias_ptr, 12);

        WORD8* p_vec_0  = (WORD8*)(p_vec1 + (vec_itr+0) * vec_stride);
        WORD8* p_vec_1  = (WORD8*)(p_vec1 + (vec_itr+1) * vec_stride);
        WORD8* p_vec_2  = (WORD8*)(p_vec1 + (vec_itr+2) * vec_stride);
        WORD8* p_vec_3  = (WORD8*)(p_vec1 + (vec_itr+3) * vec_stride);

        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        int c_itr;

        for(c_itr=0; c_itr<(cols1); c_itr++)
        {
          ae_int16x4 d_mat, d_vec0, d_vec1, d_vec2, d_vec3;
          ae_int32x2 d_tmp;
          d_mat = AE_MOVDA16(*((WORD8 *)p_mat1_0));
          d_vec0 = AE_MOVDA16(*((WORD8 *)p_vec_0)); p_vec_0++;
          d_vec1 = AE_MOVDA16(*((WORD8 *)p_vec_1)); p_vec_1++;
          d_vec2 = AE_MOVDA16(*((WORD8 *)p_vec_2)); p_vec_2++;
          d_vec3 = AE_MOVDA16(*((WORD8 *)p_vec_3)); p_vec_3++;

          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, sizeof(WORD8));//p_mat_0++;

          d_mat = AE_ADD16(d_mat, AE_MOVDA16(mat1_offset));
          AE_MULA16X4(acc_row0_vec0, d_tmp, d_mat, d_vec0);
          AE_MULA16X4(acc_row0_vec1, d_tmp, d_mat, d_vec1);
          AE_MULA16X4(acc_row0_vec2, d_tmp, d_mat, d_vec2);
          AE_MULA16X4(acc_row0_vec3, d_tmp, d_mat, d_vec3);
        }

        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift0, right_shift0);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
        acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);

        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec1, acc_row0_vec1, p_out_multiplier[vec_itr+1], left_shift1, right_shift1);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
        acc_row0_vec1 = AE_MAX32(acc_row0_vec1, min_int8);
        acc_row0_vec1 = AE_MIN32(acc_row0_vec1, max_int8);

        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec2, acc_row0_vec2, p_out_multiplier[vec_itr+2], left_shift2, right_shift2);
        acc_row0_vec2 = AE_ADD32S(acc_row0_vec2, out_zero_bias);
        acc_row0_vec2 = AE_MAX32(acc_row0_vec2, min_int8);
        acc_row0_vec2 = AE_MIN32(acc_row0_vec2, max_int8);

        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec3, acc_row0_vec3, p_out_multiplier[vec_itr+3], left_shift3, right_shift3);
        acc_row0_vec3 = AE_ADD32S(acc_row0_vec3, out_zero_bias);
        acc_row0_vec3 = AE_MAX32(acc_row0_vec3, min_int8);
        acc_row0_vec3 = AE_MIN32(acc_row0_vec3, max_int8);

        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec1), p_dst1, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec2), p_dst2, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec3), p_dst3, out_stride);
      }
      if(p_bias != NULL){
        ae_int32x2 dummy;
        AE_L32_IP(dummy, bias_ptr, 16);
      }
    }
    for(; vec_itr < vec_count; vec_itr++)
    {
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
#if TFLITE_SINGLE_ROUNDING
      left_shift = p_out_shift[vec_itr];
      right_shift = p_out_shift[vec_itr];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
      left_shift = p_out_shift[vec_itr]<0?0:p_out_shift[vec_itr];
      right_shift = p_out_shift[vec_itr]>0?0:-p_out_shift[vec_itr];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      ae_int16x4 mat1_offset_dr = mat1_offset;
      for (m_itr = 0; m_itr < (rows&~3); m_itr+=4)
      {
        int c_itr;
        ae_int32x2 acc_row01_vec0 = AE_L32_I(bias_ptr, 0);
        ae_int32x2 acc_row23_vec0 = AE_L32_I(bias_ptr, 0);

        WORD8* p_vec_0  = (WORD8*)(p_vec1 + vec_itr * vec_stride);
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));
        WORD8 *p_mat1_1 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+1) * row_stride1 * sizeof(WORD8));
        WORD8 *p_mat1_2 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_2, (m_itr+2) * row_stride1 * sizeof(WORD8));
        WORD8 *p_mat1_3 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_3, (m_itr+3) * row_stride1 * sizeof(WORD8));

        for(c_itr=0; c_itr<(cols1); c_itr++)
        {
          ae_int16x4 d_mat, d_vec0;
          ae_int32x2 mat_tmp1, mat_tmp2;
          UWORD32 mat0, mat1, mat2, mat3;
          d_vec0 = AE_MOVDA16(*((WORD8 *)p_vec_0)); p_vec_0++;
          mat0 = *((UWORD8 *)p_mat1_0);
          mat1 = *((UWORD8 *)p_mat1_1);
          mat2 = *((UWORD8 *)p_mat1_2);
          mat3 = *((UWORD8 *)p_mat1_3);
          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, sizeof(WORD8));
          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, sizeof(WORD8));
          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_2, sizeof(WORD8));
          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_3, sizeof(WORD8));

          mat_tmp1 = AE_SEXT32(AE_MOVDA32X2(mat0, mat1), 7);
          mat_tmp2 = AE_SEXT32(AE_MOVDA32X2(mat2, mat3), 7);

          d_mat = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(mat_tmp1), AE_MOVINT16X4_FROMINT32X2(mat_tmp2));

          d_mat = AE_ADD16(d_mat, mat1_offset_dr);
          AE_MULA16X4(acc_row01_vec0, acc_row23_vec0, d_mat, d_vec0);
        }

        MPY_BY_QUANT_MULT_X2_OUT32(acc_row01_vec0, acc_row01_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row23_vec0, acc_row23_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
        acc_row01_vec0 = AE_ADD32S(acc_row01_vec0, out_zero_bias);
        acc_row01_vec0 = AE_MAX32(acc_row01_vec0, min_int8);
        acc_row01_vec0 = AE_MIN32(acc_row01_vec0, max_int8);

        acc_row23_vec0 = AE_ADD32S(acc_row23_vec0, out_zero_bias);
        acc_row23_vec0 = AE_MAX32(acc_row23_vec0, min_int8);
        acc_row23_vec0 = AE_MIN32(acc_row23_vec0, max_int8);

        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row01_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row01_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row23_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row23_vec0), p_dst0, out_stride);
      }

      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = AE_L32_I(bias_ptr, 0);

        WORD8* p_vec_0  = (WORD8*)(p_vec1 + vec_itr * vec_stride);
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));
        int c_itr;

        for(c_itr=0; c_itr<(cols1); c_itr++)
        {
          ae_int16x4 d_mat, d_vec;
          ae_int32x2 d_tmp;
          d_mat = AE_MOVDA16(*((WORD8 *)p_mat1_0));
          d_vec = AE_MOVDA16(*((WORD8 *)p_vec_0));
          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, sizeof(WORD8));//p_mat_0++;
          p_vec_0++;
          d_mat = AE_ADD16(d_mat, AE_MOVDA16(mat1_offset));
          AE_MULA16X4(acc_row0_vec0, d_tmp, d_mat, d_vec);
        }

        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
        acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);

        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
      }
      if(p_bias != NULL){
        ae_int32x2 dummy;
        AE_L32_IP(dummy, bias_ptr, 4);
      }
    }
#else
/* Under normal mode of operation, this code is not expected to be executed. It is added only as a compliance code in case the aligend code is not executed */
    WORD8* pEND = (WORD8 *)AE_GETCEND0();
    WORD8* pBEGIN = (WORD8 *)AE_GETCBEGIN0();
    unsigned int CIRC_WIDTH_BEGINEND = (pEND - pBEGIN);
 
    for(vec_itr = 0; vec_itr < vec_count; vec_itr++)
    {
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
#if TFLITE_SINGLE_ROUNDING
      left_shift = p_out_shift[vec_itr];
      right_shift = p_out_shift[vec_itr];
      /* Single rounding macro doesn't need two shifts so this is not used */
      (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
      left_shift = p_out_shift[vec_itr]<0?0:p_out_shift[vec_itr];
      right_shift = p_out_shift[vec_itr]>0?0:-p_out_shift[vec_itr];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      m_itr = 0;
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = AE_L32_I(bias_ptr, 0);

        WORD8* p_vec_0  = (WORD8*)(p_vec1 + vec_itr * vec_stride);
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        /*AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));*/
        p_mat1_0 += (m_itr * row_stride1 * sizeof(WORD8));
        p_mat1_0 = (p_mat1_0 >= pEND) ? (p_mat1_0 - CIRC_WIDTH_BEGINEND) : p_mat1_0;
        
        int c_itr;
#pragma no_unroll
        for(c_itr=0; c_itr<(cols1); c_itr++)
        {
          ae_int16x4 d_mat, d_vec;
          ae_int32x2 d_tmp;
          d_mat = AE_MOVDA16(*((WORD8 *)p_mat1_0));
          d_vec = AE_MOVDA16(*((WORD8 *)p_vec_0));
          /*AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, sizeof(WORD8));//p_mat_0++; */
          p_mat1_0++;
          p_mat1_0 = (p_mat1_0 >= pEND) ? (p_mat1_0 - CIRC_WIDTH_BEGINEND) : p_mat1_0;
          p_vec_0++;
          d_mat = AE_ADD16(d_mat, AE_MOVDA16(mat1_offset));
          AE_MULA16X4(acc_row0_vec0, d_tmp, d_mat, d_vec);
        }

        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_row0_vec0, acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
        acc_row0_vec0 = AE_MIN32(acc_row0_vec0, max_int8);

        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
      }
      if(p_bias != NULL){
        ae_int32x2 dummy;
        AE_L32_IP(dummy, bias_ptr, 4);
      }
    }
#endif /* ENABLE_PADDING_CONV2D_STD */
  }
  else {
    return -1;
  }

  return 0;
}
