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
#include "xa_type_def.h"
#include "xa_nn_common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common.h"

WORD32 xa_nn_split_v_8_8(WORD8 ** __restrict__ pp_outs
                         ,const WORD32 *const *pp_outs_shape
                         ,const WORD8 * __restrict__ p_inp
                         ,const WORD32 *const p_inp_shape
                         ,WORD32 num_out
                         ,WORD32 num_out_dims
                         ,WORD32 num_inp_dims
                         ,WORD32 axis)
{
  XA_NNLIB_ARG_CHK_PTR(pp_outs, -1);
  XA_NNLIB_ARG_CHK_PTR(pp_outs_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(pp_outs, sizeof(WORD8 *), -1);
  XA_NNLIB_ARG_CHK_ALIGN(pp_outs_shape, sizeof(WORD32 *), -1);
  //Validate Arguments
  XA_NNLIB_ARG_CHK_COND((num_out <= 0 || num_out > 10), -1);
  XA_NNLIB_ARG_CHK_COND((num_out_dims <= 0 || num_out_dims > 6), -1);
  XA_NNLIB_ARG_CHK_COND((num_inp_dims != num_out_dims), -1);
  XA_NNLIB_ARG_CHK_COND((axis < -num_inp_dims || axis >= num_inp_dims), -1);

  int i = 0, j = 0;
  for(i = 0; i < num_inp_dims; i++)
  {
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[i] <= 0), -1);
  }

  if(axis < 0)
    axis = num_inp_dims + axis;

  WORD32 concat_size = 0;
  for (i = 0; i < num_out; i++)
  {
    XA_NNLIB_ARG_CHK_PTR(pp_outs[i], -1);
    XA_NNLIB_ARG_CHK_PTR(pp_outs_shape[i], -1);
    XA_NNLIB_ARG_CHK_ALIGN(pp_outs_shape[i], sizeof(WORD32), -1);
#pragma loop_count min=1
    for(j = 0; j < num_inp_dims; j++)
    {
      XA_NNLIB_ARG_CHK_COND((pp_outs_shape[i][j] != p_inp_shape[j] && j != axis), -1);
    }
    XA_NNLIB_ARG_CHK_COND((pp_outs_shape[i][axis] <= 0), -1);
    concat_size += pp_outs_shape[i][axis];
  }

  XA_NNLIB_ARG_CHK_COND((p_inp_shape[axis] != concat_size), -1);

  //Calculate outer and inner size for axis
  WORD32 outer_size = 1;
#pragma no_simd
  for(int i = 0; i < axis; i++)
  {
    outer_size *= p_inp_shape[i];
  }

  WORD32 base_inner_size = 1;
#pragma no_simd
  for(int i = axis + 1; i < num_inp_dims; i++)
  {
    base_inner_size *= p_inp_shape[i];
  }

  const WORD8 *ptmp_inp = p_inp;
  for(int i = 0; i < num_out; i++)
  {
    const WORD32 copy_size = pp_outs_shape[i][axis] * base_inner_size;
    const WORD8 *input_ptr = ptmp_inp;
    WORD8* output_ptr = pp_outs[i];

    if(((copy_size & 1) == 0) && (((concat_size * base_inner_size) & 1) == 0)
      && (((unsigned)input_ptr & 1) == 0) && (((unsigned)output_ptr & 1) == 0))
    {
      if(copy_size <= 8)
      {
        ae_int16 *pae_out = (ae_int16 *)output_ptr;
        for(int k = 0; k < outer_size; k++)
        {
          const ae_int16 *pae_inp = (const ae_int16 *)input_ptr;
#pragma concurrent
#pragma no_simd
          for(int ic = 0; ic < (copy_size >> 1); ic++)
          {
            *pae_out++ = *pae_inp++;
          }
          input_ptr += concat_size * base_inner_size;
        }
      }
      else
      {
        for(int k = 0; k < outer_size; k++)
        {
          const ae_int16x4 *pae_inp = (const ae_int16x4 *)input_ptr;
          ae_int16x4 *pae_out = (ae_int16x4 *)output_ptr;
          ae_valign inp_a, out_a;
          inp_a = AE_LA64_PP(pae_inp);
          out_a = AE_ZALIGN64();
          for(int ic = 0; ic < (copy_size >> 3); ic++)
          {
            ae_int16x4 d0;
            AE_LA16X4_IP(d0, inp_a, pae_inp);
            AE_SA16X4_IP(d0, out_a, pae_out);
          }
          AE_SA64POS_FP(out_a, pae_out);
          const ae_int16 *puae_inp = (const ae_int16 *)pae_inp;
          ae_int16 *puae_out = (ae_int16 *)pae_out;
#pragma concurrent
          for(int ic = 0; ic < ((copy_size >> 1) & 3); ic++)
          {
            puae_out[ic] = puae_inp[ic];
          }
          output_ptr += copy_size;
          input_ptr += concat_size * base_inner_size;
        }
      }
    }
    else
    {
      if(copy_size <= 6)
      {
        for(int k = 0; k < outer_size; k++)
        {
#pragma concurrent
#pragma no_unroll
          for(int ic = 0; ic < copy_size; ic++)
          {
            *output_ptr++ = input_ptr[ic];
          }
          input_ptr += concat_size * base_inner_size;
        }
      }
      else
      {
        for(int k = 0; k < outer_size; k++)
        {
          const ae_int24x2 *pae_inp = (const ae_int24x2 *)input_ptr;
          ae_int24x2 *pae_out = (ae_int24x2 *)output_ptr;
          ae_valign inp_a, out_a;
          inp_a = AE_LA64_PP(pae_inp);
          out_a = AE_ZALIGN64();

          int copy_size_by6 = AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(copy_size, 0x2AAAAAAB)));
          int copy_size_rem_start = 6*copy_size_by6;
#pragma concurrent
          for(int ic = 0; ic < copy_size_by6; ic++)
          {
            ae_int24x2 d0;
            AE_LA24X2_IP(d0, inp_a, pae_inp);
            AE_SA24X2_IP(d0, out_a, pae_out);
          }
          AE_SA64POS_FP(out_a, pae_out);
          for(int ic = copy_size_rem_start; ic < copy_size; ic++)
          {
            output_ptr[ic] = input_ptr[ic];
          }
          output_ptr += copy_size;
          input_ptr += concat_size * base_inner_size;
        }
      }
    }
    ptmp_inp += copy_size;
  }
  return 0;
}
