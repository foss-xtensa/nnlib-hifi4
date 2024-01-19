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

WORD32 xa_nn_space_to_depth_8_8
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_inp
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  block_size
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  out_channels
  ,WORD32  inp_data_format
  ,WORD32  out_data_format
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((block_size <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_height != out_height * block_size), -1);
  XA_NNLIB_ARG_CHK_COND((input_width != out_width * block_size), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels * block_size * block_size != out_channels), -1);
  XA_NNLIB_ARG_CHK_COND((inp_data_format != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

  int itr_h, itr_b0;

  WORD8 *ptmp_inp1, *ptmp_out1;
  WORD8 *ptmp_inp, *ptmp_out;
  if(block_size == 1)
  {
    MEMCPY_8b(p_out, p_inp, input_height * input_width * input_channels);
  }
  else
  {
    for(itr_b0 = 0; itr_b0 < (block_size-1); itr_b0+=2)
    {
      ptmp_out = (WORD8 *)(&p_out[itr_b0 * input_channels * block_size]);
      ptmp_out1 = (WORD8 *)(&p_out[(itr_b0 + 1) * input_channels * block_size]);
      for(itr_h = 0; itr_h < out_height; itr_h++)
      {
        ptmp_inp = (WORD8 *)(&p_inp[(itr_h * block_size + itr_b0)*input_width*input_channels]);
        ptmp_inp1 = (WORD8 *)(&p_inp[(itr_h * block_size + itr_b0 + 1)*input_width*input_channels]);
        DUAL_MEMCPY_2D_8b_CONT_INP(ptmp_out, ptmp_out1, ptmp_inp, ptmp_inp1, out_width, input_channels * block_size, out_channels);
        ptmp_out += out_width * out_channels;
        ptmp_out1 += out_width * out_channels;
      }
    }
    if((block_size&1) != 0)
    {
      itr_b0 = block_size&(~1);
      ptmp_out = (WORD8 *)(&p_out[itr_b0 * input_channels * block_size]);
      for(itr_h = 0; itr_h < out_height; itr_h++)
      {
        ptmp_inp = (WORD8 *)(&p_inp[(itr_h * block_size + itr_b0)*input_width*input_channels]);
        MEMCPY_2D_8b_CONT_INP(ptmp_out, ptmp_inp, out_width, input_channels * block_size, out_channels);
        ptmp_out += out_width * out_channels;
      }
    }
  }

  return 0;
}
