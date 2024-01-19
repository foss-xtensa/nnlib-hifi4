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
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"

static WORD32 xa_nn_conv2d_pointwise_nhwc_8x8
  (pWORD8 __restrict__ p_out
   ,pWORD8  __restrict__ p_kernel
   ,pWORD8 __restrict__ p_inp
   ,pWORD8 __restrict__ p_bias
   ,WORD32  input_height   /* Compared to earlier it is out_height */
   ,WORD32  input_width    /* Compared to earlier it is out_width */
   ,WORD32  input_channels /* Compared to earlier it is input_channels * channels_multiplier */
   ,WORD32  out_channels   /* Number of 1D pointwise kernels */
   ,WORD32  acc_shift
   ,WORD32  bias_shift)
{
    int ret, out_plane_size;
    out_plane_size = input_height*input_width;
    int vec_offset, out_offset;

    vec_offset = input_channels;
    out_offset = out_channels;

    ret = xa_nn_matmul_8x8_8(p_out,
                             p_kernel,
                             p_inp,
                             p_bias,
                             out_channels,
                             input_channels,
                             input_channels,
                             acc_shift,
                             bias_shift,
                             out_plane_size,
                             vec_offset,
                             out_offset,
                             1
                            ); 
    if(ret<0)
        return ret;
    return 0;

}

static WORD32 xa_nn_conv2d_pointwise_nchw_8x8
  (pWORD8 __restrict__ p_out
   ,pWORD8  __restrict__ p_kernel
   ,pWORD8 __restrict__ p_inp
   ,pWORD8 __restrict__ p_bias
   ,WORD32  input_height   /* Compared to earlier it is out_height */
   ,WORD32  input_width    /* Compared to earlier it is out_width */
   ,WORD32  input_channels /* Compared to earlier it is input_channels * channels_multiplier */
   ,WORD32  out_channels   /* Number of 1D pointwise kernels */
   ,WORD32  acc_shift
   ,WORD32  bias_shift)
{
    int ret, out_plane_size;
    out_plane_size = input_height*input_width;
    int vec_offset, out_offset;

    vec_offset = input_channels;
    out_offset = 1;

    ret = xa_nn_matmul_8x8_8(p_out,
                             p_kernel,
                             p_inp,
                             p_bias,
                             out_channels,
                             input_channels,
                             input_channels,
                             acc_shift,
                             bias_shift,
                             out_plane_size,
                             vec_offset,
                             out_offset,
                             out_plane_size
                             ); 
    if(ret<0)
        return ret;
    return 0;
}

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
  //XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(WORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);

  int ret=0;

  if(out_data_format == 0){
        ret = xa_nn_conv2d_pointwise_nhwc_8x8(
                p_out,
                p_kernel,
                p_inp,
                p_bias,
                input_height, 
                input_width, 
                input_channels, 
                out_channels,
                acc_shift,
                bias_shift);
  }
  else if(out_data_format == 1){
        ret = xa_nn_conv2d_pointwise_nchw_8x8(
                p_out,
                p_kernel,
                p_inp,
                p_bias,
                input_height, 
                input_width, 
                input_channels, 
                out_channels,
                acc_shift,
                bias_shift);
  }
  return ret;
}

