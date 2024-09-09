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

extern WORD32 xa_nn_matmul_sym16sxsym16s_sym16s(
  WORD16 * __restrict__ p_out,
  const WORD16 * __restrict__ p_mat1,
  const WORD16 * __restrict__ p_vec1,
  const WORD64 * __restrict__ p_bias,
  WORD32 rows,
  WORD32 cols1,
  WORD32 row_stride1,
  WORD32 vec_count,
  WORD32 vec_offset,
  WORD32 out_offset,
  WORD32 out_stride,
  WORD32 mat1_zero_bias,
  WORD32 vec1_zero_bias,
  WORD32 out_multiplier,
  WORD32 out_shift,
  WORD32 out_zero_bias);

WORD32 xa_nn_batch_matmul_sym16sxsym16s_sym16s(
    WORD16 * __restrict__ p_out,
    const WORD32 *const p_out_shape,
    const WORD16 * __restrict__ p_mat1,
    const WORD32 *const p_mat1_shape,
    const WORD16 * __restrict__ p_mat2,
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat2, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((mat1_transpose != 0 && mat1_transpose != 1), -1);
  XA_NNLIB_ARG_CHK_COND((mat2_transpose != 0 && mat2_transpose != 1), -1);
  XA_NNLIB_ARG_CHK_COND((mat1_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((mat2_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias != 0), -1);

  WORD32 itr;
  WORD32 p_mat1_final_shape[5], p_mat2_final_shape[5];
  for(itr = 0; itr < 5; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_mat1_shape[itr] <= 0 || p_mat2_shape[itr] <= 0 || p_out_shape[itr] <= 0), -1);
    p_mat1_final_shape[itr] = p_mat1_shape[itr];
    p_mat2_final_shape[itr] = p_mat2_shape[itr];
  }

  const WORD16 *p_mat1_final, *p_mat2_final;
  p_mat1_final = p_mat1;
  p_mat2_final = p_mat2;
  if(mat1_transpose)
  {
    WORD32 mat1_size, ret;
    WORD32 permute_vec[5] = {0, 1, 2, 4, 3};
    p_mat1_final_shape[3] = p_mat1_shape[4];
    p_mat1_final_shape[4] = p_mat1_shape[3];
    ret = xa_nn_transpose_16_16((WORD16 *)p_scratch,
                                p_mat1_final_shape,
                                p_mat1,
                                p_mat1_shape,
                                permute_vec,
                                5,
                                5);
    if(ret != 0)
      return -1;
    p_mat1_final = (const WORD16 *)p_scratch;
    mat1_size = p_mat1_shape[0] * p_mat1_shape[1] * p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
    p_scratch = (VOID *)(p_mat1_final + mat1_size);
  }

  if(mat2_transpose)
  {
    WORD32 ret;
    WORD32 permute_vec[5] = {0, 1, 2, 4, 3};
    p_mat2_final_shape[3] = p_mat2_shape[4];
    p_mat2_final_shape[4] = p_mat2_shape[3];
    ret = xa_nn_transpose_16_16((WORD16 *)p_scratch,
                                p_mat2_final_shape,
                                p_mat2,
                                p_mat2_shape,
                                permute_vec,
                                5,
                                5);
    if(ret != 0)
      return -1;
    p_mat2_final = (const WORD16 *)p_scratch;
  }

  WORD32 accum_depth, mat1_rows, mat2_cols;
  accum_depth = p_mat1_final_shape[4];
  mat1_rows = p_mat1_final_shape[3];
  mat2_cols = p_mat2_final_shape[3];

  WORD32 mat1_ext0, mat1_ext1, mat1_ext2;
  mat1_ext0 = p_mat1_shape[0] == 1 ? 0 : p_mat1_shape[1] * p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
  mat1_ext1 = p_mat1_shape[1] == 1 ? 0 : p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
  mat1_ext2 = p_mat1_shape[2] == 1 ? 0 : p_mat1_shape[3] * p_mat1_shape[4];

  WORD32 mat2_ext0, mat2_ext1, mat2_ext2;
  mat2_ext0 = p_mat2_shape[0] == 1 ? 0 : p_mat2_shape[1] * p_mat2_shape[2] * p_mat2_shape[3] * p_mat2_shape[4];
  mat2_ext1 = p_mat2_shape[1] == 1 ? 0 : p_mat2_shape[2] * p_mat2_shape[3] * p_mat2_shape[4];
  mat2_ext2 = p_mat2_shape[2] == 1 ? 0 : p_mat2_shape[3] * p_mat2_shape[4];

  WORD32 b0, b1, b2;
  for(b0 = 0; b0 < p_out_shape[0]; b0++)
  {
    const WORD16 *ptr0_mat1 = p_mat1_final + b0 * mat1_ext0;
    const WORD16 *ptr0_mat2 = p_mat2_final + b0 * mat2_ext0;
    for(b1 = 0; b1 < p_out_shape[1]; b1++)
    {
      const WORD16 *ptr1_mat1 = ptr0_mat1 + b1 * mat1_ext1;
      const WORD16 *ptr1_mat2 = ptr0_mat2 + b1 * mat2_ext1;
      for(b2 = 0; b2 < p_out_shape[2]; b2++)
      {
        WORD32 ret = 0;
        const WORD16 *ptr2_mat1 = ptr1_mat1 + b2 * mat1_ext2;
        const WORD16 *ptr2_mat2 = ptr1_mat2 + b2 * mat2_ext2;
        WORD16 *ptr_out = (WORD16*)(p_out + ((b0 * p_out_shape[1] + b1) * p_out_shape[2] + b2) * mat1_rows * mat2_cols);
        ret = xa_nn_matmul_sym16sxsym16s_sym16s((WORD16*)ptr_out,
                                                ptr2_mat1,
                                                ptr2_mat2,
                                                NULL,
                                                mat1_rows,
                                                accum_depth,
                                                accum_depth,
                                                mat2_cols,
                                                accum_depth,
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
