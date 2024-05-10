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
#include "xa_nn_maxpool_state.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common_macros.h"

static WORD32 xa_nn_maxpool_getsize_nchw(
    WORD32 inp_precision,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 out_width)
{
    int total_size, state_size, scratch_size;
    int full_buf_width, full_out_width;
    int /*inp_bytewidth,*/ acc_bytewidth;

    XA_NNLIB_CHK_COND((input_width <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_width > input_width), -1);
    XA_NNLIB_CHK_COND((x_stride <= 0), -1);
    XA_NNLIB_CHK_COND((y_stride <= 0), -1);
    XA_NNLIB_CHK_COND((x_padding < 0), -1);
    XA_NNLIB_CHK_COND((out_width <= 0), -1);


    switch(inp_precision)
    {
        case 8:
            //inp_bytewidth = sizeof(WORD8);
            acc_bytewidth = sizeof(WORD16);
            break;
        case 16:
            //inp_bytewidth = sizeof(WORD16);
            acc_bytewidth = sizeof(WORD16);
            break;
        case -1:
            //inp_bytewidth = sizeof(WORD32);
            acc_bytewidth = sizeof(WORD32);
            break;
        case -3:
            //inp_bytewidth = sizeof(UWORD8);
            acc_bytewidth = sizeof(WORD32);
            break;
        default:
            return -1;
            break;
    }

    /* State size */
    state_size = ALIGNED_SIZE(sizeof(xa_nn_maxpool_state_t), ALIGNMENT);
    /* Output scratch buffer size */
    full_buf_width = kernel_width + (out_width - 1)*x_stride;
    full_buf_width = MAX(full_buf_width, x_padding + input_width);
    full_buf_width = ALIGNED_SIZE(full_buf_width, ALIGNMENT/2);
    /* maxpool: Need 2 rows of padded input width as acratch for temp output */
    full_out_width = ALIGNED_SIZE(full_buf_width + kernel_width, 4);
    scratch_size = 2 * full_out_width*acc_bytewidth;

    /* Total size */
    total_size = state_size + scratch_size;
    return total_size;
}

static WORD32 xa_nn_maxpool_getsize_nhwc(WORD32  inp_precision,
                                         WORD32  input_width,
                                         WORD32  input_channels,
                                         WORD32 kernel_height,
                                         WORD32 kernel_width,
                                         WORD32 x_stride,
                                         WORD32 y_stride,
                                         WORD32 x_padding,
                                         WORD32 out_width)
{
    int scratch_bytewidth, scratch_size;

    if(input_channels == 1)
    {
          scratch_size = xa_nn_maxpool_getsize_nchw(inp_precision
                  ,input_width
                  ,kernel_height
                  ,kernel_width
                  ,x_stride
                  ,y_stride
                  ,x_padding
                  ,out_width);

          return scratch_size;
    }

    if(inp_precision == -1)
    {
        scratch_bytewidth = sizeof(FLOAT32);
        return ALIGNED_SIZE(input_channels*input_width*scratch_bytewidth, ALIGNMENT);
    }
    else if(inp_precision == -3)
    {
        scratch_bytewidth = sizeof(WORD16);
        scratch_size = ALIGNED_SIZE((input_channels*(input_width)*scratch_bytewidth), ALIGNMENT);
        scratch_size += ALIGNED_SIZE((input_channels*scratch_bytewidth), ALIGNMENT);
        return scratch_size;
    }
    else if(inp_precision == 8)
    {
        scratch_bytewidth = sizeof(WORD16);
        scratch_size = ALIGNED_SIZE((input_channels*(input_width)*scratch_bytewidth), ALIGNMENT);
        scratch_size += ALIGNED_SIZE((input_channels*scratch_bytewidth), ALIGNMENT);
        return scratch_size;
    }
    else if(inp_precision == 16)
    {
        scratch_bytewidth = sizeof(WORD16);
        return ALIGNED_SIZE((input_channels*(input_width + 1)*scratch_bytewidth), ALIGNMENT);
    }

    return 0;

}

#ifndef NNLIB_V2
WORD32 xa_nn_maxpool_getsize(
    WORD32 inp_precision,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 out_width)
{
    int total_size, state_size, scratch_size;
    int full_buf_width, full_out_width;
    int inp_bytewidth, acc_bytewidth;

    XA_NNLIB_CHK_COND((input_width <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_width > input_width), -1);
    XA_NNLIB_CHK_COND((x_stride <= 0), -1);
    XA_NNLIB_CHK_COND((y_stride <= 0), -1);
    XA_NNLIB_CHK_COND((x_padding < 0), -1);
    XA_NNLIB_CHK_COND((out_width <= 0), -1);


    switch(inp_precision)
    {
        case 8:
            inp_bytewidth = sizeof(WORD8);
            acc_bytewidth = sizeof(WORD16);
            break;
        case 16:
            inp_bytewidth = sizeof(WORD16);
            acc_bytewidth = sizeof(WORD16);
            break;
        case -1:
            inp_bytewidth = sizeof(WORD32);
            acc_bytewidth = sizeof(WORD32);
            break;
        case -3:
            inp_bytewidth = sizeof(UWORD8);
            acc_bytewidth = sizeof(WORD32);
            break;
        default:
            return -1;
            break;
    }

    /* State size */
    state_size = ALIGNED_SIZE(sizeof(xa_nn_maxpool_state_t), ALIGNMENT);
    /* Output scratch buffer size */
    full_buf_width = kernel_width + (out_width - 1)*x_stride;
    full_buf_width = XT_MAX(full_buf_width, x_padding + input_width);
    full_buf_width = ALIGNED_SIZE(full_buf_width, ALIGNMENT/2);
    /* maxpool: Need 2 rows of padded input width as acratch for temp output */
    full_out_width = ALIGNED_SIZE(full_buf_width + kernel_width, 4);
    scratch_size = 2 * full_out_width*acc_bytewidth;

    /* Total size */
    total_size = state_size + scratch_size;
    return total_size;
}
#else
WORD32 xa_nn_maxpool_getsize(
        WORD32 input_channels,
        WORD32 inp_precision,
        WORD32 out_precision,
        WORD32 input_height,
        WORD32 input_width,
        WORD32 kernel_height,
        WORD32 kernel_width,
        WORD32 x_stride,
        WORD32 y_stride,
        WORD32 x_padding,
        WORD32 y_padding,
        WORD32 out_height,
        WORD32 out_width,
        WORD32 inp_data_format,
        WORD32 out_data_format)
{
    int scratch_size;

    (void)out_precision;
    (void)input_height;
    (void)y_padding;
    (void)inp_data_format;
    (void)out_height;

    if(out_data_format == 0)
    {
        scratch_size = xa_nn_maxpool_getsize_nhwc(
                inp_precision,
                input_width,
                input_channels,
                kernel_height,
                kernel_width,
                x_stride,
                y_stride,
                x_padding,
                out_width);
    }
    else if(out_data_format == 1)
    {
        scratch_size = xa_nn_maxpool_getsize_nchw(
                inp_precision,
                input_width,
                kernel_height,
                kernel_width,
                x_stride,
                y_stride,
                x_padding,
                out_width);
    }
    else
    {
        scratch_size = -1;
    }

    return scratch_size;
}
#endif

#ifndef ENABLE_SCRATCH_SIZE_API_ONLY
WORD32 xa_nn_maxpool_init(
    WORD32 inp_precision,
    pVOID  p_scratch)
{
    pWORD8 p_mem = (pVOID)p_scratch;
    xa_nn_maxpool_state_t *p_state = (xa_nn_maxpool_state_t *)p_mem;
    int state_size;
#if 0
    int inp_bytewidth;

    switch(inp_precision)
    {
        case 8:
            inp_bytewidth = sizeof(WORD8);
            break;
        case 16:
            inp_bytewidth = sizeof(WORD16);
            break;
        case -1:
            inp_bytewidth = sizeof(WORD32);
            break;
        case -3:
            inp_bytewidth = sizeof(UWORD8);
            break;
        default:
            break;
    }
#endif

    state_size = ALIGNED_SIZE(sizeof(xa_nn_maxpool_state_t), ALIGNMENT);

    p_mem = (p_mem + state_size);
    /* Initialize output scratch pointer */
    p_state->p_scratch = (pVOID)p_mem;
    return 0;
}
#endif /* #ifndef ENABLE_SCRATCH_SIZE_API_ONLY */
