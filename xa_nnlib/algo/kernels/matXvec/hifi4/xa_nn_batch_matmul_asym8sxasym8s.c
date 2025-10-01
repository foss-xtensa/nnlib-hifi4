/*******************************************************************************
* Copyright (c) 2018-2025 Cadence Design Systems, Inc.
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

WORD32 xa_nn_batch_matmul_getsize(
    const WORD32 *const p_mat1_shape,
    const WORD32 *const p_mat2_shape,
    WORD32 mat1_transpose,
    WORD32 mat2_transpose,
    WORD32 mat1_precision,
    WORD32 mat2_precision)
{
#ifndef ENABLE_SCRATCH_SIZE_API_ONLY
   /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat2_shape, sizeof(WORD32), -1);
#endif
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((mat1_transpose != 0 && mat1_transpose != 1), -1);
  XA_NNLIB_ARG_CHK_COND((mat2_transpose != 0 && mat2_transpose != 1), -1);

  WORD32 itr;
  for(itr = 0; itr < 5; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_mat1_shape[itr] <= 0 || p_mat2_shape[itr] <= 0), -1);
  }
  WORD32 size = 0;
  WORD32 mat1_elm_size = 1, mat2_elm_size = 1;
  if(mat1_transpose == 1)
  {
    WORD32 mat1_size;
    switch(mat1_precision)
    {
      case 8:
      case -4:
      case -5:
        mat1_elm_size = sizeof(WORD8);
        break;
      case -8:
      case 16:
        mat1_elm_size = sizeof(WORD16);
        break;
      default:
        return -1;
        break;
    }
    mat1_size = p_mat1_shape[0] * p_mat1_shape[1] * p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
    size = mat1_size * mat1_elm_size;
  }
  if(mat2_transpose == 1)
  {
    WORD32 mat2_size;
    switch(mat2_precision)
    {
      case 8:
      case -4:
      case -5:
        mat2_elm_size = sizeof(WORD8);
        break;
      case -8:
      case 16:
        mat2_elm_size = sizeof(WORD16);
        break;
      default:
        return -1;
        break;
    }
    mat2_size = p_mat2_shape[0] * p_mat2_shape[1] * p_mat2_shape[2] * p_mat2_shape[3] * p_mat2_shape[4];
    size += mat2_size * mat2_elm_size;
  }
  return size;
}

#ifndef ENABLE_SCRATCH_SIZE_API_ONLY
/* Following kernel calculates contribution of matrix zero-biases towards final sum */
static WORD32 internal_calc_mzbsum(WORD16 * __restrict__ p_out, const WORD8 * __restrict__ p_vec, WORD32 mat_zero_bias, WORD32 vec_zero_bias, int cols1)
{
/*  if(mat_zero_bias == 0){
    return 0;
  }*/

  WORD32 sum_mzb32 = 0, c_itr;
  WORD32 preloop_cnt = (4 - ((unsigned)p_vec-(((unsigned)p_vec)&~0x3))) & 0x03;
  if(preloop_cnt > cols1) { preloop_cnt = 0;}
  cols1 = cols1 - preloop_cnt;

  for(c_itr = 0; c_itr < preloop_cnt; c_itr++){
    int vecval = *p_vec++;
    sum_mzb32 += (vecval+vec_zero_bias)*mat_zero_bias;
    *p_out++ = (WORD16)(vecval+vec_zero_bias);
  }
  
  ae_int64 sum_mzb = (ae_int64)sum_mzb32;
  ae_int16x4 mzb_16x4 = AE_MOVDA16(mat_zero_bias);
  ae_int16x4 d_vec0;

  ae_valign out_align = AE_ZALIGN64();

  for(c_itr = 0; c_itr < cols1>>2; c_itr++){            
    AE_L8X4F_IP(d_vec0, p_vec, 4);                       
    d_vec0 = AE_SRAI16(d_vec0, 8);
    d_vec0 = AE_ADD16(d_vec0, AE_MOVDA16(vec_zero_bias));
    AE_MULAAAAQ16(sum_mzb, mzb_16x4, d_vec0);
    AE_SA16X4_IP(d_vec0, out_align, (ae_int16x4 *)p_out);
  }
  AE_SA64POS_FP(out_align, p_out);
  sum_mzb32 = AE_MOVINT32X2_FROMINT64(sum_mzb);

  for(c_itr = 0; c_itr < (cols1&0x3); c_itr++){
    int vecval = *p_vec++;
    sum_mzb32 += (vecval+vec_zero_bias)*mat_zero_bias;
    *p_out++ = (WORD16)(vecval+vec_zero_bias);
  }
  return sum_mzb32;
}

/* This function implements transpose(A)*v operation, where v is single vector 
*  p_mat is pointer to (rows x cols) matrix A
*  p_vec is pointer to (rows x 1)    vector v
*  p_out is pointer to output vector (cols x 1)
*/

#ifndef ALIGN_PTR
#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))
#endif

static WORD32 xa_nn_matTXvec_asym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat,
    const WORD8 * __restrict__ p_vec,
    WORD32 rows,
    WORD32 cols,
    WORD32 mat_zero_bias,
    WORD32 vec_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias,
    VOID  *p_scratch)
{
  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;
  ae_int32x2 min_int8 = AE_MOVDA32(-128);
  ae_int32x2 max_int8 = AE_MOVDA32(127);

  WORD16 *p_vec16 = (WORD16 *)ALIGN_PTR(p_scratch, 2);
  WORD32 mzbsum32;
  mzbsum32 = internal_calc_mzbsum(p_vec16, p_vec, mat_zero_bias, vec_zero_bias, rows);

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
   /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  int c_itr = 0;

  /* Compute last (rows % 4) output element */
  const WORD8 *p_mat_0;
  const WORD16 *p_vec_0;
  for (; c_itr < (cols&~0x7); c_itr+=8)
  {
    p_mat_0 = (WORD8 *)(p_mat+c_itr);
    p_vec_0 = (WORD16 *)(p_vec16);
    /* core loop */

    ae_int32x2 acc01 = mzbsum32;
    ae_int32x2 acc23 = mzbsum32;
    ae_int32x2 acc45 = mzbsum32;
    ae_int32x2 acc67 = mzbsum32;

    int m_itr;
    for(m_itr=0; m_itr<rows; m_itr++){
      ae_int16x4 mat_0 = AE_L8X4F_I(p_mat_0, 0);
      ae_int16x4 mat_1 = AE_L8X4F_I(p_mat_0, 4);
      ae_int16x4 vec_0;
      AE_L16_IP(vec_0, (ae_int16*)p_vec_0, 2);
      mat_0 = AE_SRAI16(mat_0, 8);
      mat_1 = AE_SRAI16(mat_1, 8);

      AE_MULA16X4(acc01, acc23, mat_0, vec_0);
      AE_MULA16X4(acc45, acc67, mat_1, vec_0);
      p_mat_0 += cols;
    }

    MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc01, acc01, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc23, acc23, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc45, acc45, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc67, acc67, out_multiplier, left_shift, right_shift);
    acc01 = AE_ADD32S(acc01, AE_MOVDA32(out_zero_bias));
    acc23 = AE_ADD32S(acc23, AE_MOVDA32(out_zero_bias));
    acc45 = AE_ADD32S(acc45, AE_MOVDA32(out_zero_bias));
    acc67 = AE_ADD32S(acc67, AE_MOVDA32(out_zero_bias));
    AE_MINMAX32(acc01, min_int8, max_int8);
    AE_MINMAX32(acc23, min_int8, max_int8);
    AE_MINMAX32(acc45, min_int8, max_int8);
    AE_MINMAX32(acc67, min_int8, max_int8);
    *p_out++ = (WORD8)AE_MOVAD32_H(acc01);
    *p_out++ = (WORD8)AE_MOVAD32_L(acc01);
    *p_out++ = (WORD8)AE_MOVAD32_H(acc23);
    *p_out++ = (WORD8)AE_MOVAD32_L(acc23);
    *p_out++ = (WORD8)AE_MOVAD32_H(acc45);
    *p_out++ = (WORD8)AE_MOVAD32_L(acc45);
    *p_out++ = (WORD8)AE_MOVAD32_H(acc67);
    *p_out++ = (WORD8)AE_MOVAD32_L(acc67);
  }

  if (cols&0x7)
  {
    p_mat_0 = (WORD8 *)(p_mat+c_itr);
    p_vec_0 = (WORD16 *)(p_vec16);
    /* core loop */

    ae_int32x2 acc01 = mzbsum32;
    ae_int32x2 acc23 = mzbsum32;

    int m_itr;
    for(m_itr=0; m_itr<rows; m_itr++){
      ae_int16x4 mat_0 = AE_L8X4F_I(p_mat_0, 0);
      ae_int16x4 vec_0;
      AE_L16_IP(vec_0, (ae_int16*)p_vec_0, 2);
      mat_0 = AE_SRAI16(mat_0, 8);

      AE_MULA16X4(acc01, acc23, mat_0, vec_0);
      p_mat_0 += cols;
    }

    MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc01, acc01, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc23, acc23, out_multiplier, left_shift, right_shift);
    acc01 = AE_ADD32S(acc01, AE_MOVDA32(out_zero_bias));
    acc23 = AE_ADD32S(acc23, AE_MOVDA32(out_zero_bias));
    AE_MINMAX32(acc01, min_int8, max_int8);
    AE_MINMAX32(acc23, min_int8, max_int8);
    *p_out++ = (WORD8)AE_MOVAD32_H(acc01);
    *p_out++ = (WORD8)AE_MOVAD32_L(acc01);
    *p_out++ = (WORD8)AE_MOVAD32_H(acc23);
    *p_out++ = (WORD8)AE_MOVAD32_L(acc23);
  }
  return 0;
}

WORD32 xa_nn_batch_matmul_asym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD32 *const p_out_shape,
    const WORD8 * __restrict__ p_mat1,
    const WORD32 *const p_mat1_shape,
    const WORD8 * __restrict__ p_mat2,
    const WORD32 *const p_mat2_shape,
    WORD32 mat1_transpose,
    WORD32 mat2_transpose,
    WORD32 mat1_zero_bias,
    WORD32 mat2_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias,
    VOID   *p_scratch)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat2_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((mat1_transpose != 0 && mat1_transpose != 1), -1);
  XA_NNLIB_ARG_CHK_COND((mat2_transpose != 0 && mat2_transpose != 1), -1);
  XA_NNLIB_ARG_CHK_COND((mat1_zero_bias < -127 || mat1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((mat2_zero_bias < -127 || mat2_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  WORD32 itr;
  const WORD8 *p_mat1_final, *p_mat2_final;
  WORD32 p_mat1_final_shape[5], p_mat2_final_shape[5];
  for(itr = 0; itr < 5; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_mat1_shape[itr] <= 0 || p_mat2_shape[itr] <= 0 || p_out_shape[itr] <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((itr < 3 && (p_mat1_shape[itr] != p_mat2_shape[itr] && p_mat1_shape[itr] != 1 && p_mat2_shape[itr] != 1)), -1);
    XA_NNLIB_ARG_CHK_COND((itr < 3 && (p_out_shape[itr] != p_mat1_shape[itr] && p_out_shape[itr] != p_mat2_shape[itr])), -1);
    p_mat1_final_shape[itr] = p_mat1_shape[itr];
    p_mat2_final_shape[itr] = p_mat2_shape[itr];
  }

  p_mat1_final = p_mat1;
  p_mat2_final = p_mat2;

  int ret = 0;
  WORD32 mat1_ext0, mat1_ext1, mat1_ext2;
  mat1_ext0 = p_mat1_final_shape[0] == 1 ? 0 : p_mat1_final_shape[1] * p_mat1_final_shape[2] * p_mat1_final_shape[3] * p_mat1_final_shape[4];
  mat1_ext1 = p_mat1_final_shape[1] == 1 ? 0 : p_mat1_final_shape[2] * p_mat1_final_shape[3] * p_mat1_final_shape[4];
  mat1_ext2 = p_mat1_final_shape[2] == 1 ? 0 : p_mat1_final_shape[3] * p_mat1_final_shape[4];

  WORD32 mat2_ext0, mat2_ext1, mat2_ext2;
  mat2_ext0 = p_mat2_final_shape[0] == 1 ? 0 : p_mat2_final_shape[1] * p_mat2_final_shape[2] * p_mat2_final_shape[3] * p_mat2_final_shape[4];
  mat2_ext1 = p_mat2_final_shape[1] == 1 ? 0 : p_mat2_final_shape[2] * p_mat2_final_shape[3] * p_mat2_final_shape[4];
  mat2_ext2 = p_mat2_final_shape[2] == 1 ? 0 : p_mat2_final_shape[3] * p_mat2_final_shape[4];

  if ( (((unsigned)p_mat1)%4 == 0)  &&    (mat1_transpose) && (p_mat2_final_shape[3] == 1) && (p_mat1_final_shape[4]%4 == 0)  )
  { 
    /* Special case of transpose(mat1)*mat2, where mat2 is a vector, Implemented as xa_nn_matTXvec_asym8sxasym8s_asym8s */
    WORD32 b0, b1, b2;
    for (b0 = 0; b0 < p_out_shape[0]; b0++)
    { 
      const WORD8 *ptr0_mat1 = p_mat1_final + b0 * mat1_ext0;
      const WORD8 *ptr0_mat2 = p_mat2_final + b0 * mat2_ext0;
      for (b1 = 0; b1 < p_out_shape[1]; b1++)
      { 
        const WORD8 *ptr1_mat1 = ptr0_mat1 + b1 * mat1_ext1;
        const WORD8 *ptr1_mat2 = ptr0_mat2 + b1 * mat2_ext1;
        for (b2 = 0; b2 < p_out_shape[2]; b2++)
        { 
          WORD32 ret = 0;
          const WORD8 *ptr2_mat1 = ptr1_mat1 + b2 * mat1_ext2;
          const WORD8 *ptr2_mat2 = ptr1_mat2 + b2 * mat2_ext2;
          WORD8 *ptr_out = p_out + ((b0 * p_out_shape[1] + b1) * p_out_shape[2] + b2) * p_mat1_final_shape[4] * p_mat2_final_shape[3];
          ret = xa_nn_matTXvec_asym8sxasym8s_asym8s(ptr_out, ptr2_mat1, ptr2_mat2,
                                                    p_mat1_final_shape[3], p_mat1_final_shape[4],
                                                    mat1_zero_bias, mat2_zero_bias,
                                                    out_multiplier, out_shift, out_zero_bias, p_scratch);
          if (ret != 0)
            return -1;
        }
      }
    }
    return ret;
  }

  if ( (((unsigned)p_mat2)%4 == 0)  &&  (mat2_transpose)  &&  (p_mat1_final_shape[3] == 1) &&  (p_mat2_final_shape[4]%4 == 0) )
  {
    /* Special case of mat*transpose(mat2), where mat1 is a vector, can reuse xa_nn_matTXvec_asym8sxasym8s_asym8s as mat2 is stored column-major format */
    WORD32 b0, b1, b2;
    for (b0 = 0; b0 < p_out_shape[0]; b0++)
    {
      const WORD8 *ptr0_mat1 = p_mat1_final + b0 * mat1_ext0;
      const WORD8 *ptr0_mat2 = p_mat2_final + b0 * mat2_ext0;
      for (b1 = 0; b1 < p_out_shape[1]; b1++)
      {
        const WORD8 *ptr1_mat1 = ptr0_mat1 + b1 * mat1_ext1;
        const WORD8 *ptr1_mat2 = ptr0_mat2 + b1 * mat2_ext1;
        for (b2 = 0; b2 < p_out_shape[2]; b2++)
        {
          WORD32 ret = 0;
          const WORD8 *ptr2_mat1 = ptr1_mat1 + b2 * mat1_ext2;
          const WORD8 *ptr2_mat2 = ptr1_mat2 + b2 * mat2_ext2;
          WORD8 *ptr_out = p_out + ((b0 * p_out_shape[1] + b1) * p_out_shape[2] + b2) * p_mat2_final_shape[4] * p_mat1_final_shape[3];
          ret = xa_nn_matTXvec_asym8sxasym8s_asym8s(ptr_out, ptr2_mat2, ptr2_mat1,
                                                    p_mat2_final_shape[3], p_mat2_final_shape[4],
                                                    mat2_zero_bias, mat1_zero_bias,
                                                    out_multiplier, out_shift, out_zero_bias, p_scratch);
          if (ret != 0)
            return -1;
        }
      }
    }
    return ret;
  }

  WORD32 mat1_accum_dim = mat1_transpose == 0 ? p_mat1_shape[4] : p_mat1_shape[3];
  WORD32 mat2_accum_dim = mat2_transpose == 0 ? p_mat2_shape[4] : p_mat2_shape[3];
  XA_NNLIB_ARG_CHK_COND((mat1_accum_dim != mat2_accum_dim), -1);

  WORD32 mat1_rows = mat1_transpose == 0 ? p_mat1_shape[3] : p_mat1_shape[4];
  WORD32 mat2_cols = mat2_transpose == 0 ? p_mat2_shape[3] : p_mat2_shape[4];
  XA_NNLIB_ARG_CHK_COND((p_out_shape[3] != mat2_cols), -1);
  XA_NNLIB_ARG_CHK_COND((p_out_shape[4] != mat1_rows), -1);

  if(mat1_transpose)
  {
    WORD32 mat1_size, ret;
    WORD32 permute_vec[5] = {0, 1, 2, 4, 3};
    p_mat1_final_shape[3] = p_mat1_shape[4];
    p_mat1_final_shape[4] = p_mat1_shape[3];
    ret = xa_nn_transpose_8_8((WORD8 *)p_scratch,
                              p_mat1_final_shape,
                              p_mat1,
                              p_mat1_shape,
                              permute_vec,
                              5,
                              5);
    if(ret != 0)
      return -1;
    p_mat1_final = (const WORD8 *)p_scratch;
    mat1_size = p_mat1_shape[0] * p_mat1_shape[1] * p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
    p_scratch = (VOID *)(p_mat1_final + mat1_size);
  }

  if(mat2_transpose)
  {
    WORD32 ret;
    WORD32 permute_vec[5] = {0, 1, 2, 4, 3};
    p_mat2_final_shape[3] = p_mat2_shape[4];
    p_mat2_final_shape[4] = p_mat2_shape[3];
    ret = xa_nn_transpose_8_8((WORD8 *)p_scratch,
                              p_mat2_final_shape,
                              p_mat2,
                              p_mat2_shape,
                              permute_vec,
                              5,
                              5);
    if(ret != 0)
      return -1;
    p_mat2_final = (const WORD8 *)p_scratch;
  }

  mat1_ext0 = p_mat1_shape[0] == 1 ? 0 : p_mat1_shape[1] * p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
  mat1_ext1 = p_mat1_shape[1] == 1 ? 0 : p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
  mat1_ext2 = p_mat1_shape[2] == 1 ? 0 : p_mat1_shape[3] * p_mat1_shape[4];

  mat2_ext0 = p_mat2_shape[0] == 1 ? 0 : p_mat2_shape[1] * p_mat2_shape[2] * p_mat2_shape[3] * p_mat2_shape[4];
  mat2_ext1 = p_mat2_shape[1] == 1 ? 0 : p_mat2_shape[2] * p_mat2_shape[3] * p_mat2_shape[4];
  mat2_ext2 = p_mat2_shape[2] == 1 ? 0 : p_mat2_shape[3] * p_mat2_shape[4];

  WORD32 b0, b1, b2;
  for(b0 = 0; b0 < p_out_shape[0]; b0++)
  {
    const WORD8 *ptr0_mat1 = p_mat1_final + b0 * mat1_ext0;
    const WORD8 *ptr0_mat2 = p_mat2_final + b0 * mat2_ext0;
    for(b1 = 0; b1 < p_out_shape[1]; b1++)
    {
      const WORD8 *ptr1_mat1 = ptr0_mat1 + b1 * mat1_ext1;
      const WORD8 *ptr1_mat2 = ptr0_mat2 + b1 * mat2_ext1;
      for(b2 = 0; b2 < p_out_shape[2]; b2++)
      {
        WORD32 ret = 0;
        const WORD8 *ptr2_mat1 = ptr1_mat1 + b2 * mat1_ext2;
        const WORD8 *ptr2_mat2 = ptr1_mat2 + b2 * mat2_ext2;
        WORD8 *ptr_out = p_out + ((b0 * p_out_shape[1] + b1) * p_out_shape[2] + b2) * mat1_rows * mat2_cols;
        ret = xa_nn_matmul_asym8sxasym8s_asym8s(ptr_out,
                                                ptr2_mat1,
                                                ptr2_mat2,
                                                NULL,
                                                mat1_rows,
                                                mat1_accum_dim,
                                                mat1_accum_dim,
                                                mat2_cols,
                                                mat1_accum_dim,
                                                mat1_rows,
                                                1,
                                                mat1_zero_bias,
                                                mat2_zero_bias,
                                                out_multiplier,
                                                out_shift,
                                                out_zero_bias);
        if(ret != 0)
          return -1;
      }
    }
  }

  return 0;
}
#endif /* #ifndef ENABLE_SCRATCH_SIZE_API_ONLY */
