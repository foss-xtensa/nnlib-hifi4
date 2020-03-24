/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
#include "common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"

WORD32 xa_nn_conv2d_pointwise_8x8
  (pWORD8 __restrict__ p_out
   ,pWORD8  __restrict__ p_kernel
   ,pWORD8 __restrict__ p_inp
   ,pWORD8 __restrict__ p_bias
   ,WORD32  input_height   /* Compared to earlier it is out_height */
   ,WORD32  input_width    /* Compared to earlier it is out_width */
   ,WORD32  input_channels /* Compared to earlier it is input_channels * channels_multiplier */
   ,WORD32  out_channels   /* Number of 1D pointwise kernels */
   ,WORD32  acc_shift
   ,WORD32  bias_shift
   ,WORD32  out_data_format
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_kernel, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 1), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND(((input_channels&3) != 0), -1);

  int itr_oc = 0;
  int itr_rppi = 0;
  int itr_row = 0;
  int total_rows = 0;
  int trailing_loop_rows = 0;

#define ROWS_PROCESSED_PER_ITR 32

  /* A local array to copy a bias value */
  WORD8 bias[ROWS_PROCESSED_PER_ITR];

  total_rows = (input_height * input_width);
  trailing_loop_rows = (total_rows % ROWS_PROCESSED_PER_ITR);

  for (itr_oc = 0; itr_oc < out_channels; itr_oc++)
  {
    for (itr_rppi = 0; itr_rppi < ROWS_PROCESSED_PER_ITR; itr_rppi++)
    {
      bias[itr_rppi] = p_bias[itr_oc];
    }

    for (itr_row = 0; itr_row < (total_rows - trailing_loop_rows); itr_row += ROWS_PROCESSED_PER_ITR)
    {
      xa_nn_matXvec_8x8_8
        (&p_out[(itr_oc * input_height * input_width) + itr_row]
         ,&p_inp[itr_row * input_channels]
         ,NULL
         ,&p_kernel[itr_oc * input_channels]
         ,NULL
         ,bias
         ,ROWS_PROCESSED_PER_ITR
         ,input_channels
         ,0
         ,input_channels
         ,0
         ,acc_shift
         ,bias_shift
        );
    }
    /* Traling loop */
    if (trailing_loop_rows)
    {
      xa_nn_matXvec_8x8_8
        (&p_out[(itr_oc * input_height * input_width) + (total_rows - trailing_loop_rows)]
         ,&p_inp[(total_rows - trailing_loop_rows) * input_channels]
         ,NULL
         ,&p_kernel[itr_oc * input_channels]
         ,NULL
         ,bias
         ,trailing_loop_rows
         ,input_channels
         ,0
         ,input_channels
         ,0
         ,acc_shift
         ,bias_shift
        );
    }
  }

#undef ROWS_PROCESSED_PER_ITR

  return 0;
}
