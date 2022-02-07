/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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
#include "common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros.h"

static WORD32 xa_nn_conv2d_pointwise_nhwc_asym8xasym8(
    UWORD8* __restrict__ p_out,
    UWORD8* __restrict__ p_kernel,
    UWORD8* __restrict__ p_inp,
    WORD32* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  input_zero_bias,
    WORD32  kernel_zero_bias,
    WORD32  out_multiplier,
    WORD32  out_shift,
    WORD32  out_zero_bias)
{
    int ret, out_plane_size;
    out_plane_size = input_height*input_width;
    int vec_offset, out_offset;

    vec_offset = input_channels;
    out_offset = out_channels;


    ret = xa_nn_matmul_asym8xasym8_asym8(p_out,
                                         p_kernel,
                                         p_inp,
                                         p_bias,
                                         out_channels,
                                         input_channels,
                                         input_channels,
                                         out_plane_size,
                                         vec_offset,
                                         out_offset,
                                         1,
                                         kernel_zero_bias,
                                         input_zero_bias,
                                         out_multiplier,
                                         out_shift,
                                         out_zero_bias
                                         );
    if(ret<0)
        return ret;
    return 0;
}


static WORD32 xa_nn_conv2d_pointwise_nchw_asym8xasym8(
    UWORD8* __restrict__ p_out,
    UWORD8* __restrict__ p_kernel,
    UWORD8* __restrict__ p_inp,
    WORD32* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  input_zero_bias,
    WORD32  kernel_zero_bias,
    WORD32  out_multiplier,
    WORD32  out_shift,
    WORD32  out_zero_bias)
{
    int ret, out_plane_size;
    out_plane_size = input_height*input_width;
    int vec_offset, out_offset;

    vec_offset = input_channels;
    out_offset = 1;

    ret = xa_nn_matmul_asym8xasym8_asym8(p_out,
                                         p_kernel,
                                         p_inp,
                                         p_bias,
                                         out_channels,
                                         input_channels,
                                         input_channels,
                                         out_plane_size,
                                         vec_offset,
                                         out_offset,
                                         out_plane_size,
                                         kernel_zero_bias,
                                         input_zero_bias,
                                         out_multiplier,
                                         out_shift,
                                         out_zero_bias
                                        );
    if(ret<0)
        return ret;
    return 0;
}

WORD32 xa_nn_conv2d_pointwise_asym8xasym8(
    UWORD8* __restrict__ p_out,
    UWORD8* __restrict__ p_kernel,
    UWORD8* __restrict__ p_inp,
    WORD32* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  input_zero_bias,
    WORD32  kernel_zero_bias,
    WORD32  out_multiplier,
    WORD32  out_shift,
    WORD32  out_zero_bias,
    WORD32  out_data_format)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_zero_bias < -255 || input_zero_bias > 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_zero_bias < -255 || kernel_zero_bias > 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
    XA_NNLIB_ARG_CHK_COND((out_zero_bias > 255 || out_zero_bias < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);

    int ret=0;

    if(out_data_format == 0){
        ret = xa_nn_conv2d_pointwise_nhwc_asym8xasym8(
                p_out,
                p_kernel,
                p_inp,
                p_bias,
                input_height,
                input_width,
                input_channels,
                out_channels,
                input_zero_bias,
                kernel_zero_bias,
                out_multiplier,
                out_shift,
                out_zero_bias);
    }
    else if(out_data_format == 1){
        ret = xa_nn_conv2d_pointwise_nchw_asym8xasym8(
                p_out,
                p_kernel,
                p_inp,
                p_bias,
                input_height,
                input_width,
                input_channels,
                out_channels,
                input_zero_bias,
                kernel_zero_bias,
                out_multiplier,
                out_shift,
                out_zero_bias);
    }
    return ret;
}
