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
#include "common_fpu.h"
#include "xa_type_def.h"
#include <xtensa/tie/xt_hifi2.h>
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_conv2d_pointwise_f32,(
    FLOAT32* __restrict__ p_out,
    FLOAT32* __restrict__ p_kernel,
    FLOAT32* __restrict__ p_inp,
    FLOAT32* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  out_data_format))
#else /* #if !HAVE_VFPU */
WORD32 xa_nn_conv2d_pointwise_f32(
    FLOAT32* __restrict__ p_out,
    FLOAT32* __restrict__ p_kernel,
    FLOAT32* __restrict__ p_inp,
    FLOAT32* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  out_data_format)
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
    XA_NNLIB_ARG_CHK_COND((out_data_format != 1), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND(((input_channels&3) != 0), -1);

#define ROWS_PROCESSED_PER_ITR    32
    FLOAT32 *pt_out[1], *pt_vec[1], *pt_inp;
    FLOAT32 pt_bias[ROWS_PROCESSED_PER_ITR];
    int i, j, ret, out_plane_size;
    out_plane_size = input_height*input_width;
    for(i = 0; i < out_channels; i++)
    {
        xtfloatx2 *ptx2_bias = (xtfloatx2 *)pt_bias;
        xtfloatx2 bias_val = p_bias[i];
        for(j = 0; j < (ROWS_PROCESSED_PER_ITR>>1); j++)
        {
            ptx2_bias[j] = bias_val;
        }
        pt_vec[0] = &p_kernel[i*input_channels];
        for(j = 0; j < (out_plane_size&(~(ROWS_PROCESSED_PER_ITR-1))); j+= ROWS_PROCESSED_PER_ITR)
        {
            pt_inp = &p_inp[j*input_channels];
            pt_out[0] = &p_out[i*input_height*input_width+j];
            ret = xa_nn_matXvec_batch_f32xf32_f32(pt_out
                                                  ,pt_inp
                                                  ,pt_vec
                                                  ,pt_bias
                                                  ,ROWS_PROCESSED_PER_ITR
                                                  ,input_channels
                                                  ,input_channels
                                                  ,1
                                                  );
            if(ret<0)
                return ret;
        }
        if(j < out_plane_size)
        {
            pt_inp = &p_inp[j*input_channels];
            pt_out[0] = &p_out[i*input_height*input_width+j];
            ret = xa_nn_matXvec_batch_f32xf32_f32(pt_out
                                                ,pt_inp
                                                ,pt_vec
                                                ,pt_bias
                                                ,(input_height*input_width-j)
                                                ,input_channels
                                                ,input_channels
                                                ,1
                                                );
            if(ret<0)
                return ret;
        }
    }
    return 0;
}
#endif /* #if !HAVE_VFPU */
