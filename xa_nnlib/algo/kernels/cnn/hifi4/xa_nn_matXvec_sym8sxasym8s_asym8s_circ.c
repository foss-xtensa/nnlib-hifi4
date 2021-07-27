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
#include "xa_nn_conv2d_std_state.h"
#include "xa_nnlib_common_macros.h"

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

static inline void _xa_nn_dot_product_2row_4vec_mat_1byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,ae_int32x2* out_4_4
 ,ae_int32x2* out_5_5
 ,ae_int32x2* out_6_6
 ,ae_int32x2* out_7_7
 ,WORD8* p_mat_0
 ,WORD8* p_mat_1
 ,WORD8* p_vec_0
 ,WORD32 vec_stride
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vprv0, d_vprv1, d_vprv2, d_vprv3, d_tmp1, d_tmp2;
  ae_int16x4 d_vcur0, d_vtmp0, d_vout0;
  ae_int16x4 d_vcur1, d_vtmp1, d_vout1;
  ae_int16x4 d_vcur2, d_vtmp2, d_vout2;
  ae_int16x4 d_vcur3, d_vtmp3, d_vout3;
  ae_int16x4 d_mcur0, d_mprv0, d_mout0;
  ae_int16x4 d_mcur1, d_mprv1, d_mout1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int64 d_out4, d_out5, d_out6, d_out7;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  WORD8* p_vec_1 = p_vec_0 + vec_stride;
  WORD8* p_vec_2 = p_vec_1 + vec_stride;
  WORD8* p_vec_3 = p_vec_2 + vec_stride;

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_L(*out_2_2), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_3_3), 24);
  d_out4 = AE_SRAI64(AE_CVT64F32_L(*out_4_4), 24);
  d_out5 = AE_SRAI64(AE_CVT64F32_L(*out_5_5), 24);
  d_out6 = AE_SRAI64(AE_CVT64F32_L(*out_6_6), 24);
  d_out7 = AE_SRAI64(AE_CVT64F32_L(*out_7_7), 24);
  d_vprv0 = d_vprv1 = d_vprv2 = d_vprv3 = AE_ZERO16();

  d_mprv0 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_tmp1 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_tmp2 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_mprv0 = AE_SEL16_6543(d_mprv0, d_tmp1);
  d_mprv0 = AE_SEL16_6543(d_mprv0, d_tmp2); //0, 0, 1, 2

  d_mprv1 = AE_MOVDA16(*((WORD8 *)p_mat_1));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 1*sizeof(WORD8));
  d_tmp1 = AE_MOVDA16(*((WORD8 *)p_mat_1));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 1*sizeof(WORD8));
  d_tmp2 = AE_MOVDA16(*((WORD8 *)p_mat_1));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 1*sizeof(WORD8));
  d_mprv1 = AE_SEL16_6543(d_mprv1, d_tmp1);
  d_mprv1 = AE_SEL16_6543(d_mprv1, d_tmp2); //0, 0, 1, 2

  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    p_vec_1 += 1;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    p_vec_2 += 1;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    p_vec_3 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_vprv1 = AE_SEL16_7362(d_vprv1, d_tmp1); //0, 1, 0, 1

    p_vec_1 += 2;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_vprv2 = AE_SEL16_7362(d_vprv2, d_tmp1); //0, 1, 0, 1

    p_vec_2 += 2;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_vprv3 = AE_SEL16_7362(d_vprv3, d_tmp1);
    p_vec_3 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_1+2));
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp1);
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp2); //0, 0, 1, 2
    p_vec_1 += 3;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_2+2));
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp1);
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp2); //0, 0, 1, 2
    p_vec_2 += 3;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_3+2));
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp1);
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp2); //0, 0, 1, 2
    p_vec_3 += 3;
    vec_align_val = 1;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);
  d_vprv1 = AE_SLAI16S(d_vprv1, 8);
  d_vprv2 = AE_SLAI16S(d_vprv2, 8);
  d_vprv3 = AE_SLAI16S(d_vprv3, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mcur0, p_mat_0, 0);
    AE_L8X4S_I_HIFI4(d_mcur1, p_mat_1, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    AE_L8X4F_IP(d_vcur1, p_vec_1, 4);
    AE_L8X4F_IP(d_vcur2, p_vec_2, 4);
    AE_L8X4F_IP(d_vcur3, p_vec_3, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vprv1 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv1), (4-vec_align_val)*16));
    d_vprv2 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv2), (4-vec_align_val)*16));
    d_vprv3 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv3), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_vtmp1 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur1), vec_align_val*16));
    d_vtmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur2), vec_align_val*16));
    d_vtmp3 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur3), vec_align_val*16));
    d_mout0 = AE_SEL16_6543(d_mprv0, d_mcur0);
    d_mout1 = AE_SEL16_6543(d_mprv1, d_mcur1);
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vout1 = AE_OR16(d_vprv1, d_vtmp1);
    d_vout2 = AE_OR16(d_vprv2, d_vtmp2);
    d_vout3 = AE_OR16(d_vprv3, d_vtmp3);
    d_vprv0 = d_vcur0;
    d_vprv1 = d_vcur1;
    d_vprv2 = d_vcur2;
    d_vprv3 = d_vcur3;
    d_mprv0 = d_mcur0;
    d_mprv1 = d_mcur1;
    d_mout0 = AE_ADD16(d_mout0, d_mzb);
    d_mout1 = AE_ADD16(d_mout1, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mout0, d_vout0);
    AE_MULAAAAQ16(d_out1, d_mout0, d_vout1);
    AE_MULAAAAQ16(d_out2, d_mout0, d_vout2);
    AE_MULAAAAQ16(d_out3, d_mout0, d_vout3);
    AE_MULAAAAQ16(d_out4, d_mout1, d_vout0);
    AE_MULAAAAQ16(d_out5, d_mout1, d_vout1);
    AE_MULAAAAQ16(d_out6, d_mout1, d_vout2);
    AE_MULAAAAQ16(d_out7, d_mout1, d_vout3);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  d_out2 = AE_SRAI64(d_out2, 8);
  d_out3 = AE_SRAI64(d_out3, 8);
  d_out4 = AE_SRAI64(d_out4, 8);
  d_out5 = AE_SRAI64(d_out5, 8);
  d_out6 = AE_SRAI64(d_out6, 8);
  d_out7 = AE_SRAI64(d_out7, 8);
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
  *out_1_1 = AE_MOVINT32X2_FROMINT64(d_out1);
  *out_2_2 = AE_MOVINT32X2_FROMINT64(d_out2);
  *out_3_3 = AE_MOVINT32X2_FROMINT64(d_out3);
  *out_4_4 = AE_MOVINT32X2_FROMINT64(d_out4);
  *out_5_5 = AE_MOVINT32X2_FROMINT64(d_out5);
  *out_6_6 = AE_MOVINT32X2_FROMINT64(d_out6);
  *out_7_7 = AE_MOVINT32X2_FROMINT64(d_out7);
}

static inline void _xa_nn_dot_product_1row_4vec_mat_1byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,WORD8* p_mat_0
 ,WORD8* p_vec_0
 ,WORD32 vec_stride
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vprv0, d_vprv1, d_vprv2, d_vprv3, d_tmp1, d_tmp2;
  ae_int16x4 d_vcur0, d_vtmp0, d_vout0;
  ae_int16x4 d_vcur1, d_vtmp1, d_vout1;
  ae_int16x4 d_vcur2, d_vtmp2, d_vout2;
  ae_int16x4 d_vcur3, d_vtmp3, d_vout3;
  ae_int16x4 d_mcur0, d_mprv0, d_mout0;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  WORD8* p_vec_1 = p_vec_0 + vec_stride;
  WORD8* p_vec_2 = p_vec_1 + vec_stride;
  WORD8* p_vec_3 = p_vec_2 + vec_stride;

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_L(*out_2_2), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_3_3), 24);
  d_vprv0 = d_vprv1 = d_vprv2 = d_vprv3 = AE_ZERO16();

  d_mprv0 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_tmp1 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_tmp2 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_mprv0 = AE_SEL16_6543(d_mprv0, d_tmp1);
  d_mprv0 = AE_SEL16_6543(d_mprv0, d_tmp2); //0, 0, 1, 2

  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    p_vec_1 += 1;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    p_vec_2 += 1;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    p_vec_3 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_vprv1 = AE_SEL16_7362(d_vprv1, d_tmp1); //0, 1, 0, 1
    p_vec_1 += 2;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_vprv2 = AE_SEL16_7362(d_vprv2, d_tmp1); //0, 1, 0, 1
    p_vec_2 += 2;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_vprv3 = AE_SEL16_7362(d_vprv3, d_tmp1); //0, 1, 0, 1
    p_vec_3 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_1+2));
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp1);
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp2); //0, 0, 1, 2
    p_vec_1 += 3;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_2+2));
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp1);
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp2); //0, 0, 1, 2
    p_vec_2 += 3;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_3+2));
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp1);
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp2); //0, 0, 1, 2
    p_vec_3 += 3;
    vec_align_val = 3;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);
  d_vprv1 = AE_SLAI16S(d_vprv1, 8);
  d_vprv2 = AE_SLAI16S(d_vprv2, 8);
  d_vprv3 = AE_SLAI16S(d_vprv3, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mcur0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    AE_L8X4F_IP(d_vcur1, p_vec_1, 4);
    AE_L8X4F_IP(d_vcur2, p_vec_2, 4);
    AE_L8X4F_IP(d_vcur3, p_vec_3, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vprv1 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv1), (4-vec_align_val)*16));
    d_vprv2 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv2), (4-vec_align_val)*16));
    d_vprv3 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv3), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_vtmp1 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur1), vec_align_val*16));
    d_vtmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur2), vec_align_val*16));
    d_vtmp3 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur3), vec_align_val*16));
    d_mout0 = AE_SEL16_6543(d_mprv0, d_mcur0);
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vout1 = AE_OR16(d_vprv1, d_vtmp1);
    d_vout2 = AE_OR16(d_vprv2, d_vtmp2);
    d_vout3 = AE_OR16(d_vprv3, d_vtmp3);
    d_vprv0 = d_vcur0;
    d_vprv1 = d_vcur1;
    d_vprv2 = d_vcur2;
    d_vprv3 = d_vcur3;
    d_mprv0 = d_mcur0;
    d_mout0 = AE_ADD16(d_mout0, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mout0, d_vout0);
    AE_MULAAAAQ16(d_out1, d_mout0, d_vout1);
    AE_MULAAAAQ16(d_out2, d_mout0, d_vout2);
    AE_MULAAAAQ16(d_out3, d_mout0, d_vout3);
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

static inline void _xa_nn_dot_product_1row_1vec_mat_1byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,WORD8* p_mat_0
 ,WORD8* p_vec_0
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vprv0, d_tmp1, d_tmp2;
  ae_int16x4 d_vcur0, d_vtmp0, d_vout0;
  ae_int16x4 d_mcur0, d_mprv0, d_mout0;
  ae_int64 d_out0;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_vprv0 = AE_ZERO16();

  d_mprv0 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_tmp1 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_tmp2 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_mprv0 = AE_SEL16_6543(d_mprv0, d_tmp1);
  d_mprv0 = AE_SEL16_6543(d_mprv0, d_tmp2); //0, 0, 1, 2

  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    vec_align_val = 3;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mcur0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_mout0 = AE_SEL16_6543(d_mprv0, d_mcur0);
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vprv0 = d_vcur0;
    d_mprv0 = d_mcur0;
    d_mout0 = AE_ADD16(d_mout0, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mout0, d_vout0);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
}

static inline void _xa_nn_dot_product_2row_4vec_mat_2byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,ae_int32x2* out_4_4
 ,ae_int32x2* out_5_5
 ,ae_int32x2* out_6_6
 ,ae_int32x2* out_7_7
 ,WORD8* p_mat_0
 ,WORD8* p_mat_1
 ,WORD8* p_vec_0
 ,WORD32 vec_stride
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vprv0, d_vprv1, d_vprv2, d_vprv3, d_tmp1, d_tmp2;
  ae_int16x4 d_vcur0, d_vtmp0, d_vout0;
  ae_int16x4 d_vcur1, d_vtmp1, d_vout1;
  ae_int16x4 d_vcur2, d_vtmp2, d_vout2;
  ae_int16x4 d_vcur3, d_vtmp3, d_vout3;
  ae_int16x4 d_mcur0, d_mprv0, d_mout0;
  ae_int16x4 d_mcur1, d_mprv1, d_mout1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int64 d_out4, d_out5, d_out6, d_out7;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  WORD8* p_vec_1 = p_vec_0 + vec_stride;
  WORD8* p_vec_2 = p_vec_1 + vec_stride;
  WORD8* p_vec_3 = p_vec_2 + vec_stride;

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_L(*out_2_2), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_3_3), 24);
  d_out4 = AE_SRAI64(AE_CVT64F32_L(*out_4_4), 24);
  d_out5 = AE_SRAI64(AE_CVT64F32_L(*out_5_5), 24);
  d_out6 = AE_SRAI64(AE_CVT64F32_L(*out_6_6), 24);
  d_out7 = AE_SRAI64(AE_CVT64F32_L(*out_7_7), 24);
  d_vprv0 = d_vprv1 = d_vprv2 = d_vprv3 = AE_ZERO16();

  d_mprv0 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_tmp1 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_mprv0 = AE_SEL16_7362(d_mprv0, d_tmp1);

  d_mprv1 = AE_MOVDA16(*((WORD8 *)p_mat_1));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 1*sizeof(WORD8));
  d_tmp1 = AE_MOVDA16(*((WORD8 *)p_mat_1));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 1*sizeof(WORD8));
  d_mprv1 = AE_SEL16_7362(d_mprv1, d_tmp1);

  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    p_vec_1 += 1;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    p_vec_2 += 1;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    p_vec_3 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_vprv1 = AE_SEL16_7362(d_vprv1, d_tmp1); //0, 1, 0, 1
    p_vec_1 += 2;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_vprv2 = AE_SEL16_7362(d_vprv2, d_tmp1); //0, 1, 0, 1
    p_vec_2 += 2;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_vprv3 = AE_SEL16_7362(d_vprv3, d_tmp1); //0, 1, 0, 1
    p_vec_3 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_1+2));
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp1);
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp2); //0, 0, 1, 2
    p_vec_1 += 3;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_2+2));
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp1);
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp2); //0, 0, 1, 2
    p_vec_2 += 3;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_3+2));
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp1);
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp2); //0, 0, 1, 2
    p_vec_3 += 3;
    vec_align_val = 3;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);
  d_vprv1 = AE_SLAI16S(d_vprv1, 8);
  d_vprv2 = AE_SLAI16S(d_vprv2, 8);
  d_vprv3 = AE_SLAI16S(d_vprv3, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mcur0, p_mat_0, 0);
    AE_L8X4S_I_HIFI4(d_mcur1, p_mat_1, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    AE_L8X4F_IP(d_vcur1, p_vec_1, 4);
    AE_L8X4F_IP(d_vcur2, p_vec_2, 4);
    AE_L8X4F_IP(d_vcur3, p_vec_3, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vprv1 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv1), (4-vec_align_val)*16));
    d_vprv2 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv2), (4-vec_align_val)*16));
    d_vprv3 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv3), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_vtmp1 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur1), vec_align_val*16));
    d_vtmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur2), vec_align_val*16));
    d_vtmp3 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur3), vec_align_val*16));
    d_mout0 = AE_SEL16_5432(d_mprv0, d_mcur0);
    d_mout1 = AE_SEL16_5432(d_mprv1, d_mcur1);
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vout1 = AE_OR16(d_vprv1, d_vtmp1);
    d_vout2 = AE_OR16(d_vprv2, d_vtmp2);
    d_vout3 = AE_OR16(d_vprv3, d_vtmp3);
    d_vprv0 = d_vcur0;
    d_vprv1 = d_vcur1;
    d_vprv2 = d_vcur2;
    d_vprv3 = d_vcur3;
    d_mprv0 = d_mcur0;
    d_mprv1 = d_mcur1;
    d_mout0 = AE_ADD16(d_mout0, d_mzb);
    d_mout1 = AE_ADD16(d_mout1, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mout0, d_vout0);
    AE_MULAAAAQ16(d_out1, d_mout0, d_vout1);
    AE_MULAAAAQ16(d_out2, d_mout0, d_vout2);
    AE_MULAAAAQ16(d_out3, d_mout0, d_vout3);
    AE_MULAAAAQ16(d_out4, d_mout1, d_vout0);
    AE_MULAAAAQ16(d_out5, d_mout1, d_vout1);
    AE_MULAAAAQ16(d_out6, d_mout1, d_vout2);
    AE_MULAAAAQ16(d_out7, d_mout1, d_vout3);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  d_out2 = AE_SRAI64(d_out2, 8);
  d_out3 = AE_SRAI64(d_out3, 8);
  d_out4 = AE_SRAI64(d_out4, 8);
  d_out5 = AE_SRAI64(d_out5, 8);
  d_out6 = AE_SRAI64(d_out6, 8);
  d_out7 = AE_SRAI64(d_out7, 8);
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
  *out_1_1 = AE_MOVINT32X2_FROMINT64(d_out1);
  *out_2_2 = AE_MOVINT32X2_FROMINT64(d_out2);
  *out_3_3 = AE_MOVINT32X2_FROMINT64(d_out3);
  *out_4_4 = AE_MOVINT32X2_FROMINT64(d_out4);
  *out_5_5 = AE_MOVINT32X2_FROMINT64(d_out5);
  *out_6_6 = AE_MOVINT32X2_FROMINT64(d_out6);
  *out_7_7 = AE_MOVINT32X2_FROMINT64(d_out7);
}

static inline void _xa_nn_dot_product_1row_4vec_mat_2byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,WORD8* p_mat_0
 ,WORD8* p_vec_0
 ,WORD32 vec_stride
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vprv0, d_vprv1, d_vprv2, d_vprv3, d_tmp1, d_tmp2;
  ae_int16x4 d_vcur0, d_vtmp0, d_vout0;
  ae_int16x4 d_vcur1, d_vtmp1, d_vout1;
  ae_int16x4 d_vcur2, d_vtmp2, d_vout2;
  ae_int16x4 d_vcur3, d_vtmp3, d_vout3;
  ae_int16x4 d_mcur0, d_mprv0, d_mout0;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  WORD8* p_vec_1 = p_vec_0 + vec_stride;
  WORD8* p_vec_2 = p_vec_1 + vec_stride;
  WORD8* p_vec_3 = p_vec_2 + vec_stride;

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_L(*out_2_2), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_3_3), 24);
  d_vprv0 = d_vprv1 = d_vprv2 = d_vprv3 = AE_ZERO16();

  d_mprv0 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_tmp1 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_mprv0 = AE_SEL16_7362(d_mprv0, d_tmp1);

  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    p_vec_1 += 1;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    p_vec_2 += 1;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    p_vec_3 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_vprv1 = AE_SEL16_7362(d_vprv1, d_tmp1); //0, 1, 0, 1
    p_vec_1 += 2;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_vprv2 = AE_SEL16_7362(d_vprv2, d_tmp1); //0, 1, 0, 1
    p_vec_2 += 2;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_vprv3 = AE_SEL16_7362(d_vprv3, d_tmp1); //0, 1, 0, 1
    p_vec_3 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_1+2));
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp1);
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp2); //0, 0, 1, 2
    p_vec_1 += 3;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_2+2));
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp1);
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp2); //0, 0, 1, 2
    p_vec_2 += 3;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_3+2));
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp1);
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp2); //0, 0, 1, 2
    p_vec_3 += 3;
    vec_align_val = 3;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);
  d_vprv1 = AE_SLAI16S(d_vprv1, 8);
  d_vprv2 = AE_SLAI16S(d_vprv2, 8);
  d_vprv3 = AE_SLAI16S(d_vprv3, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mcur0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    AE_L8X4F_IP(d_vcur1, p_vec_1, 4);
    AE_L8X4F_IP(d_vcur2, p_vec_2, 4);
    AE_L8X4F_IP(d_vcur3, p_vec_3, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vprv1 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv1), (4-vec_align_val)*16));
    d_vprv2 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv2), (4-vec_align_val)*16));
    d_vprv3 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv3), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_vtmp1 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur1), vec_align_val*16));
    d_vtmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur2), vec_align_val*16));
    d_vtmp3 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur3), vec_align_val*16));
    d_mout0 = AE_SEL16_5432(d_mprv0, d_mcur0);
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vout1 = AE_OR16(d_vprv1, d_vtmp1);
    d_vout2 = AE_OR16(d_vprv2, d_vtmp2);
    d_vout3 = AE_OR16(d_vprv3, d_vtmp3);
    d_vprv0 = d_vcur0;
    d_vprv1 = d_vcur1;
    d_vprv2 = d_vcur2;
    d_vprv3 = d_vcur3;
    d_mprv0 = d_mcur0;
    d_mout0 = AE_ADD16(d_mout0, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mout0, d_vout0);
    AE_MULAAAAQ16(d_out1, d_mout0, d_vout1);
    AE_MULAAAAQ16(d_out2, d_mout0, d_vout2);
    AE_MULAAAAQ16(d_out3, d_mout0, d_vout3);
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

static inline void _xa_nn_dot_product_1row_1vec_mat_2byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,WORD8* p_mat_0
 ,WORD8* p_vec_0
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vprv0, d_tmp1, d_tmp2;
  ae_int16x4 d_vcur0, d_vtmp0, d_vout0;
  ae_int16x4 d_mcur0, d_mprv0, d_mout0;
  ae_int64 d_out0;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_vprv0 = AE_ZERO16();

  d_mprv0 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_tmp1 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_mprv0 = AE_SEL16_7362(d_mprv0, d_tmp1);

  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    vec_align_val = 3;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mcur0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_mout0 = AE_SEL16_5432(d_mprv0, d_mcur0);
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vprv0 = d_vcur0;
    d_mprv0 = d_mcur0;
    d_mout0 = AE_ADD16(d_mout0, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mout0, d_vout0);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
}

static inline void _xa_nn_dot_product_2row_4vec_mat_3byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,ae_int32x2* out_4_4
 ,ae_int32x2* out_5_5
 ,ae_int32x2* out_6_6
 ,ae_int32x2* out_7_7
 ,WORD8* p_mat_0
 ,WORD8* p_mat_1
 ,WORD8* p_vec_0
 ,WORD32 vec_stride
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vprv0, d_vprv1, d_vprv2, d_vprv3, d_tmp1, d_tmp2;
  ae_int16x4 d_vcur0, d_vtmp0, d_vout0;
  ae_int16x4 d_vcur1, d_vtmp1, d_vout1;
  ae_int16x4 d_vcur2, d_vtmp2, d_vout2;
  ae_int16x4 d_vcur3, d_vtmp3, d_vout3;
  ae_int16x4 d_mcur0, d_mprv0, d_mout0;
  ae_int16x4 d_mcur1, d_mprv1, d_mout1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int64 d_out4, d_out5, d_out6, d_out7;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  WORD8* p_vec_1 = p_vec_0 + vec_stride;
  WORD8* p_vec_2 = p_vec_1 + vec_stride;
  WORD8* p_vec_3 = p_vec_2 + vec_stride;

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_L(*out_2_2), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_3_3), 24);
  d_out4 = AE_SRAI64(AE_CVT64F32_L(*out_4_4), 24);
  d_out5 = AE_SRAI64(AE_CVT64F32_L(*out_5_5), 24);
  d_out6 = AE_SRAI64(AE_CVT64F32_L(*out_6_6), 24);
  d_out7 = AE_SRAI64(AE_CVT64F32_L(*out_7_7), 24);

  d_mprv0 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));
  d_mprv1 = AE_MOVDA16(*((WORD8 *)p_mat_1));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 1*sizeof(WORD8));
  
  d_vprv0 = d_vprv1 = d_vprv2 = d_vprv3 = AE_ZERO16();

  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    p_vec_1 += 1;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    p_vec_2 += 1;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    p_vec_3 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_vprv1 = AE_SEL16_7362(d_vprv1, d_tmp1); //0, 1, 0, 1
    p_vec_1 += 2;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_vprv2 = AE_SEL16_7362(d_vprv2, d_tmp1); //0, 1, 0, 1
    p_vec_2 += 2;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_vprv3 = AE_SEL16_7362(d_vprv3, d_tmp1); //0, 1, 0, 1
    p_vec_3 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_1+2));
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp1);
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp2); //0, 0, 1, 2
    p_vec_1 += 3;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_2+2));
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp1);
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp2); //0, 0, 1, 2
    p_vec_2 += 3;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_3+2));
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp1);
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp2); //0, 0, 1, 2
    p_vec_3 += 3;
    vec_align_val = 3;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);
  d_vprv1 = AE_SLAI16S(d_vprv1, 8);
  d_vprv2 = AE_SLAI16S(d_vprv2, 8);
  d_vprv3 = AE_SLAI16S(d_vprv3, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mcur0, p_mat_0, 0);
    AE_L8X4S_I_HIFI4(d_mcur1, p_mat_1, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    AE_L8X4F_IP(d_vcur1, p_vec_1, 4);
    AE_L8X4F_IP(d_vcur2, p_vec_2, 4);
    AE_L8X4F_IP(d_vcur3, p_vec_3, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vprv1 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv1), (4-vec_align_val)*16));
    d_vprv2 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv2), (4-vec_align_val)*16));
    d_vprv3 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv3), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_vtmp1 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur1), vec_align_val*16));
    d_vtmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur2), vec_align_val*16));
    d_vtmp3 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur3), vec_align_val*16));
    d_mout0 = AE_SEL16_4321(d_mprv0, d_mcur0);
    d_mout1 = AE_SEL16_4321(d_mprv1, d_mcur1);
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vout1 = AE_OR16(d_vprv1, d_vtmp1);
    d_vout2 = AE_OR16(d_vprv2, d_vtmp2);
    d_vout3 = AE_OR16(d_vprv3, d_vtmp3);
    d_vprv0 = d_vcur0;
    d_vprv1 = d_vcur1;
    d_vprv2 = d_vcur2;
    d_vprv3 = d_vcur3;
    d_mprv0 = d_mcur0;
    d_mprv1 = d_mcur1;
    d_mout0 = AE_ADD16(d_mout0, d_mzb);
    d_mout1 = AE_ADD16(d_mout1, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mout0, d_vout0);
    AE_MULAAAAQ16(d_out1, d_mout0, d_vout1);
    AE_MULAAAAQ16(d_out2, d_mout0, d_vout2);
    AE_MULAAAAQ16(d_out3, d_mout0, d_vout3);
    AE_MULAAAAQ16(d_out4, d_mout1, d_vout0);
    AE_MULAAAAQ16(d_out5, d_mout1, d_vout1);
    AE_MULAAAAQ16(d_out6, d_mout1, d_vout2);
    AE_MULAAAAQ16(d_out7, d_mout1, d_vout3);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  d_out2 = AE_SRAI64(d_out2, 8);
  d_out3 = AE_SRAI64(d_out3, 8);
  d_out4 = AE_SRAI64(d_out4, 8);
  d_out5 = AE_SRAI64(d_out5, 8);
  d_out6 = AE_SRAI64(d_out6, 8);
  d_out7 = AE_SRAI64(d_out7, 8);
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
  *out_1_1 = AE_MOVINT32X2_FROMINT64(d_out1);
  *out_2_2 = AE_MOVINT32X2_FROMINT64(d_out2);
  *out_3_3 = AE_MOVINT32X2_FROMINT64(d_out3);
  *out_4_4 = AE_MOVINT32X2_FROMINT64(d_out4);
  *out_5_5 = AE_MOVINT32X2_FROMINT64(d_out5);
  *out_6_6 = AE_MOVINT32X2_FROMINT64(d_out6);
  *out_7_7 = AE_MOVINT32X2_FROMINT64(d_out7);
}

static inline void _xa_nn_dot_product_1row_4vec_mat_3byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,WORD8* p_mat_0
 ,WORD8* p_vec_0
 ,WORD32 vec_stride
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vprv0, d_vprv1, d_vprv2, d_vprv3, d_tmp1, d_tmp2;
  ae_int16x4 d_vcur0, d_vtmp0, d_vout0;
  ae_int16x4 d_vcur1, d_vtmp1, d_vout1;
  ae_int16x4 d_vcur2, d_vtmp2, d_vout2;
  ae_int16x4 d_vcur3, d_vtmp3, d_vout3;
  ae_int16x4 d_mcur0, d_mprv0, d_mout0;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  WORD8* p_vec_1 = p_vec_0 + vec_stride;
  WORD8* p_vec_2 = p_vec_1 + vec_stride;
  WORD8* p_vec_3 = p_vec_2 + vec_stride;

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_L(*out_2_2), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_3_3), 24);

  d_mprv0 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));

  d_vprv0 = d_vprv1 = d_vprv2 = d_vprv3 = AE_ZERO16();

  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    p_vec_1 += 1;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    p_vec_2 += 1;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    p_vec_3 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_vprv1 = AE_SEL16_7362(d_vprv1, d_tmp1); //0, 1, 0, 1
    p_vec_1 += 2;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_vprv2 = AE_SEL16_7362(d_vprv2, d_tmp1); //0, 1, 0, 1
    p_vec_2 += 2;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_vprv3 = AE_SEL16_7362(d_vprv3, d_tmp1); //0, 1, 0, 1
    p_vec_3 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_1+2));
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp1);
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp2); //0, 0, 1, 2
    p_vec_1 += 3;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_2+2));
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp1);
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp2); //0, 0, 1, 2
    p_vec_2 += 3;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_3+2));
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp1);
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp2); //0, 0, 1, 2
    p_vec_3 += 3;
    vec_align_val = 3;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);
  d_vprv1 = AE_SLAI16S(d_vprv1, 8);
  d_vprv2 = AE_SLAI16S(d_vprv2, 8);
  d_vprv3 = AE_SLAI16S(d_vprv3, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mcur0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    AE_L8X4F_IP(d_vcur1, p_vec_1, 4);
    AE_L8X4F_IP(d_vcur2, p_vec_2, 4);
    AE_L8X4F_IP(d_vcur3, p_vec_3, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vprv1 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv1), (4-vec_align_val)*16));
    d_vprv2 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv2), (4-vec_align_val)*16));
    d_vprv3 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv3), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_vtmp1 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur1), vec_align_val*16));
    d_vtmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur2), vec_align_val*16));
    d_vtmp3 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur3), vec_align_val*16));
    d_mout0 = AE_SEL16_4321(d_mprv0, d_mcur0);
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vout1 = AE_OR16(d_vprv1, d_vtmp1);
    d_vout2 = AE_OR16(d_vprv2, d_vtmp2);
    d_vout3 = AE_OR16(d_vprv3, d_vtmp3);
    d_vprv0 = d_vcur0;
    d_vprv1 = d_vcur1;
    d_vprv2 = d_vcur2;
    d_vprv3 = d_vcur3;
    d_mprv0 = d_mcur0;
    d_mout0 = AE_ADD16(d_mout0, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mout0, d_vout0);
    AE_MULAAAAQ16(d_out1, d_mout0, d_vout1);
    AE_MULAAAAQ16(d_out2, d_mout0, d_vout2);
    AE_MULAAAAQ16(d_out3, d_mout0, d_vout3);
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

static inline void _xa_nn_dot_product_1row_1vec_mat_3byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,WORD8* p_mat_0
 ,WORD8* p_vec_0
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vprv0, d_tmp1, d_tmp2;
  ae_int16x4 d_vcur0, d_vtmp0, d_vout0;
  ae_int16x4 d_mcur0, d_mprv0, d_mout0;
  ae_int64 d_out0;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);

  d_mprv0 = AE_MOVDA16(*((WORD8 *)p_mat_0));
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 1*sizeof(WORD8));

  d_vprv0 = AE_ZERO16();

  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    vec_align_val = 3;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mcur0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_mout0 = AE_SEL16_4321(d_mprv0, d_mcur0);
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vprv0 = d_vcur0;
    d_mprv0 = d_mcur0;
    d_mout0 = AE_ADD16(d_mout0, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mout0, d_vout0);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
}

static inline void _xa_nn_dot_product_2row_4vec_mat_4byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,ae_int32x2* out_4_4
 ,ae_int32x2* out_5_5
 ,ae_int32x2* out_6_6
 ,ae_int32x2* out_7_7
 ,WORD8* p_mat_0
 ,WORD8* p_mat_1
 ,WORD8* p_vec_0
 ,WORD32 vec_stride
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vcur0, d_vprv0, d_vout0;
  ae_int16x4 d_vcur1, d_vprv1, d_vout1;
  ae_int16x4 d_vcur2, d_vprv2, d_vout2;
  ae_int16x4 d_vcur3, d_vprv3, d_vout3;
  ae_int16x4 d_tmp1, d_tmp2, d_vtmp0, d_vtmp1, d_vtmp2, d_vtmp3;
  ae_int16x4 d_mat0, d_mat1;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int64 d_out4, d_out5, d_out6, d_out7;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  WORD8* p_vec_1 = p_vec_0 + vec_stride;
  WORD8* p_vec_2 = p_vec_1 + vec_stride;
  WORD8* p_vec_3 = p_vec_2 + vec_stride;

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_L(*out_2_2), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_3_3), 24);
  d_out4 = AE_SRAI64(AE_CVT64F32_L(*out_4_4), 24);
  d_out5 = AE_SRAI64(AE_CVT64F32_L(*out_5_5), 24);
  d_out6 = AE_SRAI64(AE_CVT64F32_L(*out_6_6), 24);
  d_out7 = AE_SRAI64(AE_CVT64F32_L(*out_7_7), 24);

  d_vprv0 = d_vprv1 = d_vprv2 = d_vprv3 = AE_ZERO16();

  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    p_vec_1 += 1;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    p_vec_2 += 1;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    p_vec_3 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_vprv1 = AE_SEL16_7362(d_vprv1, d_tmp1); //0, 1, 0, 1
    p_vec_1 += 2;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_vprv2 = AE_SEL16_7362(d_vprv2, d_tmp1); //0, 1, 0, 1
    p_vec_2 += 2;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_vprv3 = AE_SEL16_7362(d_vprv3, d_tmp1); //0, 1, 0, 1
    p_vec_3 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_1+2));
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp1);
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp2); //0, 0, 1, 2
    p_vec_1 += 3;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_2+2));
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp1);
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp2); //0, 0, 1, 2
    p_vec_2 += 3;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_3+2));
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp1);
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp2); //0, 0, 1, 2
    p_vec_3 += 3;
    vec_align_val = 3;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);
  d_vprv1 = AE_SLAI16S(d_vprv1, 8);
  d_vprv2 = AE_SLAI16S(d_vprv2, 8);
  d_vprv3 = AE_SLAI16S(d_vprv3, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_L8X4S_I_HIFI4(d_mat1, p_mat_1, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    AE_L8X4F_IP(d_vcur1, p_vec_1, 4);
    AE_L8X4F_IP(d_vcur2, p_vec_2, 4);
    AE_L8X4F_IP(d_vcur3, p_vec_3, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vprv0 = d_vcur0;
    d_vprv1 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv1), (4-vec_align_val)*16));
    d_vtmp1 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur1), vec_align_val*16));
    d_vout1 = AE_OR16(d_vprv1, d_vtmp1);
    d_vprv1 = d_vcur1;
    d_vprv2 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv2), (4-vec_align_val)*16));
    d_vtmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur2), vec_align_val*16));
    d_vout2 = AE_OR16(d_vprv2, d_vtmp2);
    d_vprv2 = d_vcur2;
    d_vprv3 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv3), (4-vec_align_val)*16));
    d_vtmp3 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur3), vec_align_val*16));
    d_vout3 = AE_OR16(d_vprv3, d_vtmp3);
    d_vprv3 = d_vcur3;
    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    d_mat1 = AE_ADD16(d_mat1, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vout0);
    AE_MULAAAAQ16(d_out1, d_mat0, d_vout1);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vout2);
    AE_MULAAAAQ16(d_out3, d_mat0, d_vout3);
    AE_MULAAAAQ16(d_out4, d_mat1, d_vout0);
    AE_MULAAAAQ16(d_out5, d_mat1, d_vout1);
    AE_MULAAAAQ16(d_out6, d_mat1, d_vout2);
    AE_MULAAAAQ16(d_out7, d_mat1, d_vout3);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  d_out1 = AE_SRAI64(d_out1, 8);
  d_out2 = AE_SRAI64(d_out2, 8);
  d_out3 = AE_SRAI64(d_out3, 8);
  d_out4 = AE_SRAI64(d_out4, 8);
  d_out5 = AE_SRAI64(d_out5, 8);
  d_out6 = AE_SRAI64(d_out6, 8);
  d_out7 = AE_SRAI64(d_out7, 8);
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
  *out_1_1 = AE_MOVINT32X2_FROMINT64(d_out1);
  *out_2_2 = AE_MOVINT32X2_FROMINT64(d_out2);
  *out_3_3 = AE_MOVINT32X2_FROMINT64(d_out3);
  *out_4_4 = AE_MOVINT32X2_FROMINT64(d_out4);
  *out_5_5 = AE_MOVINT32X2_FROMINT64(d_out5);
  *out_6_6 = AE_MOVINT32X2_FROMINT64(d_out6);
  *out_7_7 = AE_MOVINT32X2_FROMINT64(d_out7);
}

static inline void _xa_nn_dot_product_1row_4vec_mat_4byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,WORD8* p_mat_0
 ,WORD8* p_vec_0
 ,WORD32 vec_stride
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vcur0, d_vprv0, d_vout0;
  ae_int16x4 d_vcur1, d_vprv1, d_vout1;
  ae_int16x4 d_vcur2, d_vprv2, d_vout2;
  ae_int16x4 d_vcur3, d_vprv3, d_vout3;
  ae_int16x4 d_tmp1, d_tmp2, d_vtmp0, d_vtmp1, d_vtmp2, d_vtmp3;
  ae_int16x4 d_mat0;
  ae_int64 d_out0, d_out1, d_out2, d_out3;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  WORD8* p_vec_1 = p_vec_0 + vec_stride;
  WORD8* p_vec_2 = p_vec_1 + vec_stride;
  WORD8* p_vec_3 = p_vec_2 + vec_stride;

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_out1 = AE_SRAI64(AE_CVT64F32_L(*out_1_1), 24);
  d_out2 = AE_SRAI64(AE_CVT64F32_L(*out_2_2), 24);
  d_out3 = AE_SRAI64(AE_CVT64F32_L(*out_3_3), 24);

  d_vprv0 = d_vprv1 = d_vprv2 = d_vprv3 = AE_ZERO16();

  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    p_vec_1 += 1;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    p_vec_2 += 1;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    p_vec_3 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_vprv1 = AE_SEL16_7362(d_vprv1, d_tmp1); //0, 1, 0, 1
    p_vec_1 += 2;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_vprv2 = AE_SEL16_7362(d_vprv2, d_tmp1); //0, 1, 0, 1
    p_vec_2 += 2;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_vprv3 = AE_SEL16_7362(d_vprv3, d_tmp1); //0, 1, 0, 1
    p_vec_3 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    d_vprv1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_1+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_1+2));
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp1);
    d_vprv1 = AE_SEL16_6543(d_vprv1, d_tmp2); //0, 0, 1, 2
    p_vec_1 += 3;
    d_vprv2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_2+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_2+2));
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp1);
    d_vprv2 = AE_SEL16_6543(d_vprv2, d_tmp2); //0, 0, 1, 2
    p_vec_2 += 3;
    d_vprv3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_3+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_3+2));
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp1);
    d_vprv3 = AE_SEL16_6543(d_vprv3, d_tmp2); //0, 0, 1, 2
    p_vec_3 += 3;
    vec_align_val = 3;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);
  d_vprv1 = AE_SLAI16S(d_vprv1, 8);
  d_vprv2 = AE_SLAI16S(d_vprv2, 8);
  d_vprv3 = AE_SLAI16S(d_vprv3, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    AE_L8X4F_IP(d_vcur1, p_vec_1, 4);
    AE_L8X4F_IP(d_vcur2, p_vec_2, 4);
    AE_L8X4F_IP(d_vcur3, p_vec_3, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vprv0 = d_vcur0;
    d_vprv1 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv1), (4-vec_align_val)*16));
    d_vtmp1 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur1), vec_align_val*16));
    d_vout1 = AE_OR16(d_vprv1, d_vtmp1);
    d_vprv1 = d_vcur1;
    d_vprv2 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv2), (4-vec_align_val)*16));
    d_vtmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur2), vec_align_val*16));
    d_vout2 = AE_OR16(d_vprv2, d_vtmp2);
    d_vprv2 = d_vcur2;
    d_vprv3 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv3), (4-vec_align_val)*16));
    d_vtmp3 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur3), vec_align_val*16));
    d_vout3 = AE_OR16(d_vprv3, d_vtmp3);
    d_vprv3 = d_vcur3;
    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vout0);
    AE_MULAAAAQ16(d_out1, d_mat0, d_vout1);
    AE_MULAAAAQ16(d_out2, d_mat0, d_vout2);
    AE_MULAAAAQ16(d_out3, d_mat0, d_vout3);
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

static inline void _xa_nn_dot_product_1row_1vec_mat_4byte_aligned_vec_unaligned
(ae_int32x2* out_0_0
 ,WORD8* p_mat_0
 ,WORD8* p_vec_0
 ,WORD32 cols1
 ,WORD32 mat_zero_bias
 ,WORD32 vec_align_val
 )
{
  int c_itr = 0;
  ae_int16x4 d_vcur0, d_vprv0, d_vout0;
  ae_int16x4 d_tmp1, d_tmp2, d_vtmp0;
  ae_int16x4 d_mat0;
  ae_int64 d_out0;
  ae_int16x4 d_mzb = AE_MOVDA16(mat_zero_bias);

  d_out0 = AE_SRAI64(AE_CVT64F32_L(*out_0_0), 24);
  d_vprv0 = AE_ZERO16();
  if(vec_align_val == 3)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    p_vec_0 += 1;
    vec_align_val = 1;
  }
  else if(vec_align_val == 2)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_vprv0 = AE_SEL16_7362(d_vprv0, d_tmp1); //0, 1, 0, 1
    p_vec_0 += 2;
  }
  else if(vec_align_val == 1)
  {
    d_vprv0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_tmp1 = AE_MOVDA16(*((WORD8 *)p_vec_0+1));
    d_tmp2 = AE_MOVDA16(*((WORD8 *)p_vec_0+2));
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp1);
    d_vprv0 = AE_SEL16_6543(d_vprv0, d_tmp2); //0, 0, 1, 2
    p_vec_0 += 3;
    vec_align_val = 3;
  }
  d_vprv0 = AE_SLAI16S(d_vprv0, 8);

  for(;c_itr<(cols1>>2); c_itr++)
  {
    AE_L8X4S_I_HIFI4(d_mat0, p_mat_0, 0);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, 4*sizeof(WORD8));
    AE_L8X4F_IP(d_vcur0, p_vec_0, 4);
    /* vector: shift the values and select the required using OR */
    d_vprv0 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_vprv0), (4-vec_align_val)*16));
    d_vtmp0 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_vcur0), vec_align_val*16));
    d_vout0 = AE_OR16(d_vprv0, d_vtmp0);
    d_vprv0 = d_vcur0;
    d_mat0 = AE_ADD16(d_mat0, d_mzb);
    AE_MULAAAAQ16(d_out0, d_mat0, d_vout0);
  }
  d_out0 = AE_SRAI64(d_out0, 8);
  *out_0_0 = AE_MOVINT32X2_FROMINT64(d_out0);
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

/* vec and rows unaligned generic implementation can be optimized */
static inline void _xa_nn_dot_product_2_rows_4_vecs_unaligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,ae_int32x2* out_4_4
 ,ae_int32x2* out_5_5
 ,ae_int32x2* out_6_6
 ,ae_int32x2* out_7_7
 ,WORD8*      p_mat_0
 ,WORD8*      p_mat_1
 ,WORD8*      p_vec_0
 ,WORD32      vec_stride
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat0, d_mat1, d_vec0, d_vec1, d_vec2, d_vec3;
  ae_int32x2 d_tmp;
  ae_int32x2 d_out0, d_out1, d_out2, d_out3;
  ae_int32x2 d_out4, d_out5, d_out6, d_out7;
  WORD8* p_vec_1 = p_vec_0 + vec_stride;
  WORD8* p_vec_2 = p_vec_1 + vec_stride;
  WORD8* p_vec_3 = p_vec_2 + vec_stride;

  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;
  d_out4 = *out_4_4;
  d_out5 = *out_5_5;
  d_out6 = *out_6_6;
  d_out7 = *out_7_7;

  for(;c_itr<(cols1); c_itr++)
  {
    d_mat0 = AE_MOVDA16(*((WORD8 *)p_mat_0));
    d_mat1 = AE_MOVDA16(*((WORD8 *)p_mat_1));
    d_vec0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_vec1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_vec2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_vec3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, sizeof(WORD8));//p_mat_0++;
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_1, sizeof(WORD8));//p_mat_0++;
    p_vec_0++;
    p_vec_1++;
    p_vec_2++;
    p_vec_3++;
    d_mat0 = AE_ADD16(d_mat0, AE_MOVDA16(mat_zero_bias));
    d_mat1 = AE_ADD16(d_mat1, AE_MOVDA16(mat_zero_bias));
    AE_MULA16X4(d_out0, d_tmp, d_mat0, d_vec0);
    AE_MULA16X4(d_out1, d_tmp, d_mat0, d_vec1);
    AE_MULA16X4(d_out2, d_tmp, d_mat0, d_vec2);
    AE_MULA16X4(d_out3, d_tmp, d_mat0, d_vec3);
    AE_MULA16X4(d_out4, d_tmp, d_mat1, d_vec0);
    AE_MULA16X4(d_out5, d_tmp, d_mat1, d_vec1);
    AE_MULA16X4(d_out6, d_tmp, d_mat1, d_vec2);
    AE_MULA16X4(d_out7, d_tmp, d_mat1, d_vec3);
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

static inline void _xa_nn_dot_product_1_rows_4_vecs_unaligned
(ae_int32x2* out_0_0
 ,ae_int32x2* out_1_1
 ,ae_int32x2* out_2_2
 ,ae_int32x2* out_3_3
 ,WORD8*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD32      vec_stride
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
{
  int c_itr = 0;
  ae_int16x4 d_mat, d_vec0, d_vec1, d_vec2, d_vec3;
  ae_int32x2 d_tmp;
  ae_int32x2 d_out0, d_out1, d_out2, d_out3;
  WORD8* p_vec_1 = p_vec_0 + vec_stride;
  WORD8* p_vec_2 = p_vec_1 + vec_stride;
  WORD8* p_vec_3 = p_vec_2 + vec_stride;

  d_out0 = *out_0_0;
  d_out1 = *out_1_1;
  d_out2 = *out_2_2;
  d_out3 = *out_3_3;

  for(;c_itr<(cols1); c_itr++)
  {
    d_mat = AE_MOVDA16(*((WORD8 *)p_mat_0));
    d_vec0 = AE_MOVDA16(*((WORD8 *)p_vec_0));
    d_vec1 = AE_MOVDA16(*((WORD8 *)p_vec_1));
    d_vec2 = AE_MOVDA16(*((WORD8 *)p_vec_2));
    d_vec3 = AE_MOVDA16(*((WORD8 *)p_vec_3));
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, sizeof(WORD8));//p_mat_0++;
    p_vec_0++;
    p_vec_1++;
    p_vec_2++;
    p_vec_3++;
    d_mat = AE_ADD16(d_mat, AE_MOVDA16(mat_zero_bias));
    AE_MULA16X4(d_out0, d_tmp, d_mat, d_vec0);
    AE_MULA16X4(d_out1, d_tmp, d_mat, d_vec1);
    AE_MULA16X4(d_out2, d_tmp, d_mat, d_vec2);
    AE_MULA16X4(d_out3, d_tmp, d_mat, d_vec3);
  }
  *out_0_0 = d_out0;
  *out_1_1 = d_out1;
  *out_2_2 = d_out2;
  *out_3_3 = d_out3;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
(ae_int32x2* out_0_0
 ,WORD8*      p_mat_0
 ,WORD8*      p_vec_0
 ,WORD32      cols1
 ,WORD32      mat_zero_bias)
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
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat_0, sizeof(WORD8));//p_mat_0++;
    p_vec_0++;
    d_mat = AE_ADD16(d_mat, AE_MOVDA16(mat_zero_bias));
    AE_MULA16X4(d_out, d_tmp, d_mat, d_vec);
  }
  *out_0_0 = d_out;
}

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
  int out_shift;

  int left_shift, right_shift;
  int m_itr, vec_itr;

  /* vec, mat and bias 4-byte aigned */
  if(p_mat1 && p_vec1 && p_bias &&
      (((unsigned int)p_mat1&0x3)==0) && (((unsigned int)p_vec1&0x3)==0) && (((unsigned int)p_bias&0x3) == 0) &&
      ((cols1&0x3)==0) && ((vec_stride&0x3)==0) && ((row_stride1&0x3)==0))
  {
    ae_int32 *bias_ptr = (ae_int32*)p_bias;
    for(vec_itr = 0; vec_itr < ((vec_count>>2)<<2); vec_itr+=4)
    {
      ae_int32x2 acc_row0_vec0, acc_row0_vec1, acc_row0_vec2, acc_row0_vec3;
      WORD32 o_shift[4], l_shift[4], r_shift[4];
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;

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
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[vec_itr+0], l_shift[0], r_shift[0]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec2, p_out_multiplier[vec_itr+2], l_shift[2], r_shift[2]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec3, p_out_multiplier[vec_itr+3], l_shift[3], r_shift[3]);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
        acc_row0_vec2 = AE_ADD32S(acc_row0_vec2, out_zero_bias);
        acc_row0_vec3 = AE_ADD32S(acc_row0_vec3, out_zero_bias);
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

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[vec_itr+0], l_shift[0], r_shift[0]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec2, p_out_multiplier[vec_itr+2], l_shift[2], r_shift[2]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec3, p_out_multiplier[vec_itr+3], l_shift[3], r_shift[3]);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
        acc_row0_vec2 = AE_ADD32S(acc_row0_vec2, out_zero_bias);
        acc_row0_vec3 = AE_ADD32S(acc_row0_vec3, out_zero_bias);
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec1, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec2, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec3, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec1), p_dst1, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec2), p_dst2, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec3), p_dst3, out_stride);
      }
      /* dummy load, just to increment the pointer */
      AE_L32_IP(acc_row0_vec0, bias_ptr, 16);
    }

    /* for vec_count=2 */
    if(vec_count&0x2)
    {
      WORD32 o_shift[2], l_shift[2], r_shift[2];
      ae_int32x2 acc_row0_vec0;
      ae_int32x2 acc_row0_vec1;
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      o_shift[0] = p_out_shift[vec_itr];
      l_shift[0] = o_shift[0]<0?0:o_shift[0];
      r_shift[0] = o_shift[0]>0?0:-o_shift[0];
      o_shift[1] = p_out_shift[vec_itr+1];
      l_shift[1] = o_shift[1]<0?0:o_shift[1];
      r_shift[1] = o_shift[1]>0?0:-o_shift[1];

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
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[vec_itr], l_shift[0], r_shift[0]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec1, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row0_vec1), p_dst1, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec1), p_dst1, out_stride);
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
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[vec_itr], l_shift[0], r_shift[0]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec1, p_out_multiplier[vec_itr+1], l_shift[1], r_shift[1]);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32_HF4(acc_row0_vec1, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec1), p_dst1, out_stride);
      }
      /* dummy load, just to increment the pointer */
      vec_itr+=2;
      AE_L32_IP(acc_row0_vec0, bias_ptr, 8);
    }

    /* for vec_count=1 */
    if(vec_count&0x1)
    {
      ae_int32x2 acc_row0_vec0;
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      out_shift = p_out_shift[vec_itr];
      left_shift = out_shift<0?0:out_shift;
      right_shift = out_shift>0?0:-out_shift;

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
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_H(acc_row0_vec0), p_dst0, out_stride);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
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
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
      }
      /* dummy load, just to increment the pointer */
      AE_L32_IP(acc_row0_vec0, bias_ptr, 4);
    }
  }
  else if(p_mat1 && p_vec1 && p_bias)
  {
    ae_int32* bias_ptr = (ae_int32*)p_bias;
    for(vec_itr = 0; vec_itr < (vec_count & ~(16 - 1)); vec_itr += 16)
    {
      int l_shift[4], r_shift[4], o_shift[4];
      ae_int32x2 acc_row0_vec0, acc_row0_vec1, acc_row0_vec2, acc_row0_vec3;
      int ii, jj;
      for(ii = 0; ii < 4; ii++)
      {
        WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0 + ii) * out_offset;
        WORD8* p_dst1 = (WORD8*)p_out + (vec_itr + 4 + ii) * out_offset;
        WORD8* p_dst2 = (WORD8*)p_out + (vec_itr + 8 + ii) * out_offset;
        WORD8* p_dst3 = (WORD8*)p_out + (vec_itr + 12+ ii) * out_offset;

        o_shift[0] = p_out_shift[vec_itr + 0 + ii];
        o_shift[1] = p_out_shift[vec_itr + 4 + ii];
        o_shift[2] = p_out_shift[vec_itr + 8 + ii];
        o_shift[3] = p_out_shift[vec_itr + 12+ ii];

        l_shift[0] = o_shift[0]<0?0:o_shift[0];
        r_shift[0] = o_shift[0]>0?0:-o_shift[0];
        l_shift[1] = o_shift[1]<0?0:o_shift[1];
        r_shift[1] = o_shift[1]>0?0:-o_shift[1];
        l_shift[2] = o_shift[2]<0?0:o_shift[2];
        r_shift[2] = o_shift[2]>0?0:-o_shift[2];
        l_shift[3] = o_shift[3]<0?0:o_shift[3];
        r_shift[3] = o_shift[3]>0?0:-o_shift[3];

        WORD8* p_vec_0  = (WORD8*)(p_vec1 + (vec_itr + 0 + ii) * vec_stride);
        int vec_align_val = (((unsigned int)p_vec_0) & 0x3);
				for(m_itr = 0; m_itr< (rows&~(8-1)); m_itr += 8)
				{
					for(jj = 0; jj < 4; jj++)
					{
						ae_int32x2 acc0, acc1, acc2, acc3;
						
						acc0 = acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
						acc1 = acc_row0_vec1 = AE_L32_I(bias_ptr, 16);
						acc2 = acc_row0_vec2 = AE_L32_X(bias_ptr, 32);
						acc3 = acc_row0_vec3 = AE_L32_X(bias_ptr, 48);
						
						WORD8 *p_mat1_0 = (WORD8*)p_mat1;
						WORD8 *p_mat1_1 = (WORD8*)p_mat1;
						AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (m_itr+0+jj) * row_stride1 * sizeof(WORD8));
						AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (m_itr+4+jj) * row_stride1 * sizeof(WORD8));						

						int mat_align_val = (((unsigned int)p_mat1_0)&0x3);
						/* mat and vec are same byte aligned */
						if((mat_align_val == vec_align_val) && (cols1>=(4-mat_align_val)))
						{
							_xa_nn_dot_product_2_rows_4_vecs_unaligned
								(&acc_row0_vec0
								 ,&acc_row0_vec1
								 ,&acc_row0_vec2
								 ,&acc_row0_vec3
								 ,&acc0
								 ,&acc1
								 ,&acc2
								 ,&acc3
								 ,p_mat1_0
                 ,p_mat1_1
								 ,p_vec_0
								 ,4*vec_stride
								 ,(4-mat_align_val)
								 ,mat1_offset
                );
							AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (4-mat_align_val)*sizeof(WORD8));
							AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (4-mat_align_val)*sizeof(WORD8));

              acc_row0_vec0 = AE_SEL32_LL(acc_row0_vec0, acc0);
              acc_row0_vec1 = AE_SEL32_LL(acc_row0_vec1, acc1);
              acc_row0_vec2 = AE_SEL32_LL(acc_row0_vec2, acc2);
              acc_row0_vec3 = AE_SEL32_LL(acc_row0_vec3, acc3);
              _xa_nn_dot_product_2row_4vec_mat_vecs_4bytes_aligned
								(&acc_row0_vec0
								 ,&acc_row0_vec1
								 ,&acc_row0_vec2
								 ,&acc_row0_vec3
								 ,p_mat1_0
								 ,p_mat1_1
								 ,p_vec_0+(4-vec_align_val)
								 ,4*vec_stride
								 ,(((cols1-(4-mat_align_val))>>2)<<2)
								 ,mat1_offset
								);
              acc0 = AE_SEL32_LL(acc_row0_vec0, acc_row0_vec0);
              acc1 = AE_SEL32_LL(acc_row0_vec1, acc_row0_vec1);
              acc2 = AE_SEL32_LL(acc_row0_vec2, acc_row0_vec2);
              acc3 = AE_SEL32_LL(acc_row0_vec3, acc_row0_vec3);
              acc_row0_vec0 = AE_SEL32_HH(acc_row0_vec0, acc_row0_vec0);
              acc_row0_vec1 = AE_SEL32_HH(acc_row0_vec1, acc_row0_vec1);
              acc_row0_vec2 = AE_SEL32_HH(acc_row0_vec2, acc_row0_vec2);
              acc_row0_vec3 = AE_SEL32_HH(acc_row0_vec3, acc_row0_vec3);

              AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (((cols1-(4-mat_align_val))>>2)<<2)*sizeof(WORD8));
							AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (((cols1-(4-mat_align_val))>>2)<<2)*sizeof(WORD8));							

              _xa_nn_dot_product_2_rows_4_vecs_unaligned
								(&acc_row0_vec0
								 ,&acc_row0_vec1
								 ,&acc_row0_vec2
								 ,&acc_row0_vec3
								 ,&acc0
								 ,&acc1
								 ,&acc2
								 ,&acc3
								 ,p_mat1_0
                 ,p_mat1_1
								 ,p_vec_0+(4-vec_align_val)+(((cols1-(4-mat_align_val))>>2)<<2)
								 ,4*vec_stride
								 ,((cols1-(4-mat_align_val))&0x3)
								 ,mat1_offset
                );
            }
						else if(mat_align_val == 0)
						{
							_xa_nn_dot_product_2row_4vec_mat_4byte_aligned_vec_unaligned
								(&acc_row0_vec0
								 ,&acc_row0_vec1
								 ,&acc_row0_vec2
								 ,&acc_row0_vec3
                 ,&acc0
                 ,&acc1
                 ,&acc2
                 ,&acc3
								 ,p_mat1_0
								 ,p_mat1_1
								 ,p_vec_0
								 ,4*vec_stride
								 ,((cols1>>2)<<2)
								 ,mat1_offset
								 ,vec_align_val
                );
							AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
							AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, ((cols1>>2)<<2)*sizeof(WORD8));							
							_xa_nn_dot_product_2_rows_4_vecs_unaligned
								(&acc_row0_vec0
								 ,&acc_row0_vec1
								 ,&acc_row0_vec2
								 ,&acc_row0_vec3
								 ,&acc0
								 ,&acc1
								 ,&acc2
								 ,&acc3
								 ,p_mat1_0
                 ,p_mat1_1
								 ,p_vec_0+((cols1>>2)<<2)
								 ,4*vec_stride
								 ,(cols1&0x3)
								 ,mat1_offset
                );
						}
						else if(mat_align_val == 3)
						{
							_xa_nn_dot_product_2row_4vec_mat_3byte_aligned_vec_unaligned
								(&acc_row0_vec0
								 ,&acc_row0_vec1
								 ,&acc_row0_vec2
								 ,&acc_row0_vec3
                 ,&acc0
                 ,&acc1
                 ,&acc2
                 ,&acc3
								 ,p_mat1_0
                 ,p_mat1_1
								 ,p_vec_0
								 ,4*vec_stride
								 ,((cols1>>2)<<2)
								 ,mat1_offset
								 ,vec_align_val
								);
								
							AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
							AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, ((cols1>>2)<<2)*sizeof(WORD8));							
							_xa_nn_dot_product_2_rows_4_vecs_unaligned
								(&acc_row0_vec0
								 ,&acc_row0_vec1
								 ,&acc_row0_vec2
								 ,&acc_row0_vec3
								 ,&acc0
								 ,&acc1
								 ,&acc2
								 ,&acc3
								 ,p_mat1_0
                 ,p_mat1_1
								 ,p_vec_0+((cols1>>2)<<2)
								 ,4*vec_stride
								 ,(cols1&0x3)
								 ,mat1_offset
                );
						}
						else if(mat_align_val == 2)
						{
							_xa_nn_dot_product_2row_4vec_mat_2byte_aligned_vec_unaligned
								(&acc_row0_vec0
								 ,&acc_row0_vec1
								 ,&acc_row0_vec2
								 ,&acc_row0_vec3
                 ,&acc0
                 ,&acc1
                 ,&acc2
                 ,&acc3
								 ,p_mat1_0
                 ,p_mat1_1
								 ,p_vec_0
								 ,4*vec_stride
								 ,((cols1>>2)<<2)
								 ,mat1_offset
								 ,vec_align_val
								);
							AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
							AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, ((cols1>>2)<<2)*sizeof(WORD8));							
							_xa_nn_dot_product_2_rows_4_vecs_unaligned
								(&acc_row0_vec0
								 ,&acc_row0_vec1
								 ,&acc_row0_vec2
								 ,&acc_row0_vec3
								 ,&acc0
								 ,&acc1
								 ,&acc2
								 ,&acc3
								 ,p_mat1_0
                 ,p_mat1_1
								 ,p_vec_0+((cols1>>2)<<2)
								 ,4*vec_stride
								 ,(cols1&0x3)
								 ,mat1_offset
                );
            }
						else if(mat_align_val == 1)        
						{
							_xa_nn_dot_product_2row_4vec_mat_1byte_aligned_vec_unaligned
								(&acc_row0_vec0
								 ,&acc_row0_vec1
								 ,&acc_row0_vec2
								 ,&acc_row0_vec3
								 ,&acc0
								 ,&acc1
								 ,&acc2
								 ,&acc3
								 ,p_mat1_0
								 ,p_mat1_1
								 ,p_vec_0
								 ,4*vec_stride
								 ,((cols1>>2)<<2)
								 ,mat1_offset
								 ,vec_align_val
								);

							AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
							AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, ((cols1>>2)<<2)*sizeof(WORD8));							
							_xa_nn_dot_product_2_rows_4_vecs_unaligned
								(&acc_row0_vec0
								 ,&acc_row0_vec1
								 ,&acc_row0_vec2
								 ,&acc_row0_vec3
								 ,&acc0
								 ,&acc1
								 ,&acc2
								 ,&acc3
								 ,p_mat1_0
                 ,p_mat1_1
								 ,p_vec_0+((cols1>>2)<<2)
								 ,4*vec_stride
								 ,(cols1&0x3)
								 ,mat1_offset
                );
            }
            acc_row0_vec0 = AE_SEL32_LL(acc_row0_vec0, acc0);
            acc_row0_vec1 = AE_SEL32_LL(acc_row0_vec1, acc1);
            acc_row0_vec2 = AE_SEL32_LL(acc_row0_vec2, acc2);
            acc_row0_vec3 = AE_SEL32_LL(acc_row0_vec3, acc3);
						MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[vec_itr + 0 + ii], l_shift[0], r_shift[0]);
						MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec1, p_out_multiplier[vec_itr + 4 + ii], l_shift[1], r_shift[1]);
						MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec2, p_out_multiplier[vec_itr + 8 + ii], l_shift[2], r_shift[2]);
						MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec3, p_out_multiplier[vec_itr + 12+ ii], l_shift[3], r_shift[3]);
            acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
						acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
						acc_row0_vec2 = AE_ADD32S(acc_row0_vec2, out_zero_bias);
						acc_row0_vec3 = AE_ADD32S(acc_row0_vec3, out_zero_bias);
            AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
						AE_MINMAX32_HF4(acc_row0_vec1, min_int8, max_int8);
						AE_MINMAX32_HF4(acc_row0_vec2, min_int8, max_int8);
						AE_MINMAX32_HF4(acc_row0_vec3, min_int8, max_int8);
            AE_S8_FROM32(AE_MOVAD32_H(acc_row0_vec0), p_dst0, jj*out_stride);
						AE_S8_FROM32(AE_MOVAD32_H(acc_row0_vec1), p_dst1, jj*out_stride);
						AE_S8_FROM32(AE_MOVAD32_H(acc_row0_vec2), p_dst2, jj*out_stride);
						AE_S8_FROM32(AE_MOVAD32_H(acc_row0_vec3), p_dst3, jj*out_stride);				
						AE_S8_FROM32(AE_MOVAD32_L(acc_row0_vec0), p_dst0, (4+jj)*out_stride);
						AE_S8_FROM32(AE_MOVAD32_L(acc_row0_vec1), p_dst1, (4+jj)*out_stride);
						AE_S8_FROM32(AE_MOVAD32_L(acc_row0_vec2), p_dst2, (4+jj)*out_stride);
						AE_S8_FROM32(AE_MOVAD32_L(acc_row0_vec3), p_dst3, (4+jj)*out_stride);
					}
					p_dst0+=8*out_stride;
					p_dst1+=8*out_stride;
					p_dst2+=8*out_stride;
					p_dst3+=8*out_stride;
				}
        for (; m_itr < rows; m_itr++)
			  {
          acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
          acc_row0_vec1 = AE_L32_I(bias_ptr, 16);
          acc_row0_vec2 = AE_L32_X(bias_ptr, 32);
          acc_row0_vec3 = AE_L32_X(bias_ptr, 48);

          WORD8 *p_mat1_0 = (WORD8*)p_mat1;
          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));
          int mat_align_val = (((unsigned int)p_mat1_0)&0x3);

          /* mat and vec are same byte aligned */
          if((mat_align_val == vec_align_val) && (cols1>=(4-mat_align_val)))
          {
            _xa_nn_dot_product_1_rows_4_vecs_unaligned
              (&acc_row0_vec0
               ,&acc_row0_vec1
               ,&acc_row0_vec2
               ,&acc_row0_vec3
               ,p_mat1_0
               ,p_vec_0
               ,4*vec_stride
               ,(4-mat_align_val)
               ,mat1_offset
              );

            AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (4-mat_align_val)*sizeof(WORD8));
            _xa_nn_dot_product_1row_4vec_mat_vecs_4bytes_aligned
              (&acc_row0_vec0
               ,&acc_row0_vec1
               ,&acc_row0_vec2
               ,&acc_row0_vec3
               ,p_mat1_0
               ,p_vec_0+(4-vec_align_val)
               ,4*vec_stride
               ,(((cols1-(4-mat_align_val))>>2)<<2)
               ,mat1_offset
              );

            AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (((cols1-(4-mat_align_val))>>2)<<2)*sizeof(WORD8));
            _xa_nn_dot_product_1_rows_4_vecs_unaligned
              (&acc_row0_vec0
               ,&acc_row0_vec1
               ,&acc_row0_vec2
               ,&acc_row0_vec3
               ,p_mat1_0
               ,p_vec_0+(4-vec_align_val)+(((cols1-(4-mat_align_val))>>2)<<2)
               ,4*vec_stride
               ,((cols1-(4-mat_align_val))&0x3)
               ,mat1_offset
              );
          }
          else if(mat_align_val == 0)
          {
            _xa_nn_dot_product_1row_4vec_mat_4byte_aligned_vec_unaligned
              (&acc_row0_vec0
               ,&acc_row0_vec1
               ,&acc_row0_vec2
               ,&acc_row0_vec3
               ,p_mat1_0
               ,p_vec_0
               ,4*vec_stride
               ,((cols1>>2)<<2)
               ,mat1_offset
               ,vec_align_val
              );

            AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
            _xa_nn_dot_product_1_rows_4_vecs_unaligned
              (&acc_row0_vec0
               ,&acc_row0_vec1
               ,&acc_row0_vec2
               ,&acc_row0_vec3
               ,p_mat1_0
               ,p_vec_0+((cols1>>2)<<2)
               ,4*vec_stride
               ,(cols1&0x3)
               ,mat1_offset
              );
          }
          else if(mat_align_val == 3)
          {
            _xa_nn_dot_product_1row_4vec_mat_3byte_aligned_vec_unaligned
              (&acc_row0_vec0
               ,&acc_row0_vec1
               ,&acc_row0_vec2
               ,&acc_row0_vec3
               ,p_mat1_0
               ,p_vec_0
               ,4*vec_stride
               ,((cols1>>2)<<2)
               ,mat1_offset
               ,vec_align_val
              );

            AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
            _xa_nn_dot_product_1_rows_4_vecs_unaligned
              (&acc_row0_vec0
               ,&acc_row0_vec1
               ,&acc_row0_vec2
               ,&acc_row0_vec3
               ,p_mat1_0
               ,p_vec_0+((cols1>>2)<<2)
               ,4*vec_stride
               ,(cols1&0x3)
               ,mat1_offset
              );
          }
          else if(mat_align_val == 2)
          {
            _xa_nn_dot_product_1row_4vec_mat_2byte_aligned_vec_unaligned
              (&acc_row0_vec0
               ,&acc_row0_vec1
               ,&acc_row0_vec2
               ,&acc_row0_vec3
               ,p_mat1_0
               ,p_vec_0
               ,4*vec_stride
               ,((cols1>>2)<<2)
               ,mat1_offset
               ,vec_align_val
              );

            AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
            _xa_nn_dot_product_1_rows_4_vecs_unaligned
              (&acc_row0_vec0
               ,&acc_row0_vec1
               ,&acc_row0_vec2
               ,&acc_row0_vec3
               ,p_mat1_0
               ,p_vec_0+((cols1>>2)<<2)
               ,4*vec_stride
               ,(cols1&0x3)
               ,mat1_offset
              );
          }
          else if(mat_align_val == 1)        
          {
            _xa_nn_dot_product_1row_4vec_mat_1byte_aligned_vec_unaligned
              (&acc_row0_vec0
               ,&acc_row0_vec1
               ,&acc_row0_vec2
               ,&acc_row0_vec3
               ,p_mat1_0
               ,p_vec_0
               ,4*vec_stride
               ,((cols1>>2)<<2)
               ,mat1_offset
               ,vec_align_val
              );

            AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
            _xa_nn_dot_product_1_rows_4_vecs_unaligned
              (&acc_row0_vec0
               ,&acc_row0_vec1
               ,&acc_row0_vec2
               ,&acc_row0_vec3
               ,p_mat1_0
               ,p_vec_0+((cols1>>2)<<2)
               ,4*vec_stride
               ,(cols1&0x3)
               ,mat1_offset
              );
          }

          MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[vec_itr + 0 + ii], l_shift[0], r_shift[0]);
          MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec1, p_out_multiplier[vec_itr + 4 + ii], l_shift[1], r_shift[1]);
          MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec2, p_out_multiplier[vec_itr + 8 + ii], l_shift[2], r_shift[2]);
          MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec3, p_out_multiplier[vec_itr + 12+ ii], l_shift[3], r_shift[3]);					
          acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
          acc_row0_vec1 = AE_ADD32S(acc_row0_vec1, out_zero_bias);
          acc_row0_vec2 = AE_ADD32S(acc_row0_vec2, out_zero_bias);
          acc_row0_vec3 = AE_ADD32S(acc_row0_vec3, out_zero_bias);					
          AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
          AE_MINMAX32_HF4(acc_row0_vec1, min_int8, max_int8);
          AE_MINMAX32_HF4(acc_row0_vec2, min_int8, max_int8);
          AE_MINMAX32_HF4(acc_row0_vec3, min_int8, max_int8);					
          AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
          AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec1), p_dst1, out_stride);
          AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec2), p_dst2, out_stride);
          AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec3), p_dst3, out_stride);					
        }
        AE_L32_IP(acc_row0_vec0, bias_ptr, 4); //dummy load for pointer update
      }
        AE_L32_XP(acc_row0_vec0, bias_ptr, 48); //dummy load for pointer update
    }			
    for(; vec_itr < vec_count; vec_itr++)
    {
      ae_int32x2 acc_row0_vec0;
      WORD8* p_dst0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      out_shift = p_out_shift[vec_itr];
      left_shift = out_shift<0?0:out_shift;
      right_shift = out_shift>0?0:-out_shift;

      WORD8* p_vec_0  = (WORD8*)(p_vec1 + vec_itr * vec_stride);
      int vec_align_val = (((unsigned int)p_vec_0) & 0x3);
      for (m_itr = 0; m_itr < rows; m_itr++)
      {
        acc_row0_vec0 = AE_L32_I(bias_ptr, 0);
        WORD8 *p_mat1_0 = (WORD8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        int mat_align_val = (((unsigned int)p_mat1_0)&0x3);
        /* mat and vec are same byte aligned */
        if((mat_align_val == vec_align_val) && (cols1>=(4-mat_align_val)))
        {
          _xa_nn_dot_product_1_rows_1_vecs_unaligned
            (&acc_row0_vec0
             ,p_mat1_0
             ,p_vec_0
             ,(4-mat_align_val)
             ,mat1_offset
            );

          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (4-mat_align_val)*sizeof(WORD8));
          _xa_nn_dot_product_1row_1vec_mat_vecs_4bytes_aligned
            (&acc_row0_vec0
             ,p_mat1_0
             ,p_vec_0+(4-vec_align_val)
             ,(((cols1-(4-mat_align_val))>>2)<<2)
             ,mat1_offset
            );

          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (((cols1-(4-mat_align_val))>>2)<<2)*sizeof(WORD8));
          _xa_nn_dot_product_1_rows_1_vecs_unaligned
            (&acc_row0_vec0
             ,p_mat1_0
             ,p_vec_0+(4-vec_align_val)+(((cols1-(4-mat_align_val))>>2)<<2)
             ,((cols1-(4-mat_align_val))&0x3)
             ,mat1_offset
            );
        }
        else if(mat_align_val == 0)
        {
          _xa_nn_dot_product_1row_1vec_mat_4byte_aligned_vec_unaligned
            (&acc_row0_vec0
             ,p_mat1_0
             ,p_vec_0
             ,((cols1>>2)<<2)
             ,mat1_offset
             ,vec_align_val
            );

          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
          _xa_nn_dot_product_1_rows_1_vecs_unaligned
            (&acc_row0_vec0
             ,p_mat1_0
             ,p_vec_0+((cols1>>2)<<2)
             ,(cols1&0x3)
             ,mat1_offset
            );
        }
        else if(mat_align_val == 3)
        {
          _xa_nn_dot_product_1row_1vec_mat_3byte_aligned_vec_unaligned
            (&acc_row0_vec0
             ,p_mat1_0
             ,p_vec_0
             ,((cols1>>2)<<2)
             ,mat1_offset
             ,vec_align_val
            );

          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
          _xa_nn_dot_product_1_rows_1_vecs_unaligned
            (&acc_row0_vec0
             ,p_mat1_0
             ,p_vec_0+((cols1>>2)<<2)
             ,(cols1&0x3)
             ,mat1_offset
            );
        }
        else if(mat_align_val == 2)
        {
          _xa_nn_dot_product_1row_1vec_mat_2byte_aligned_vec_unaligned
            (&acc_row0_vec0
             ,p_mat1_0
             ,p_vec_0
             ,((cols1>>2)<<2)
             ,mat1_offset
             ,vec_align_val
            );

          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
          _xa_nn_dot_product_1_rows_1_vecs_unaligned
            (&acc_row0_vec0
             ,p_mat1_0
             ,p_vec_0+((cols1>>2)<<2)
             ,(cols1&0x3)
             ,mat1_offset
            );
        }
        else if(mat_align_val == 1)        
        {
          _xa_nn_dot_product_1row_1vec_mat_1byte_aligned_vec_unaligned
            (&acc_row0_vec0
             ,p_mat1_0
             ,p_vec_0
             ,((cols1>>2)<<2)
             ,mat1_offset
             ,vec_align_val
            );

          AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, ((cols1>>2)<<2)*sizeof(WORD8));
          _xa_nn_dot_product_1_rows_1_vecs_unaligned
            (&acc_row0_vec0
             ,p_mat1_0
             ,p_vec_0+((cols1>>2)<<2)
             ,(cols1&0x3)
             ,mat1_offset
            );
        }

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[vec_itr], left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        AE_MINMAX32_HF4(acc_row0_vec0, min_int8, max_int8);
        AE_S8_FROM32_WITHSTRIDE(AE_MOVAD32_L(acc_row0_vec0), p_dst0, out_stride);
      }
        AE_L32_IP(acc_row0_vec0, bias_ptr, 4); //dummy load
    }
  }
  else
    return -1;

  return 0;
}
