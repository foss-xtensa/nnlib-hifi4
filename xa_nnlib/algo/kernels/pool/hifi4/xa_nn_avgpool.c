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
#include "xa_nn_avgpool_state.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common_macros.h"

static WORD32 xa_nn_avgpool_getsize_nchw(
    WORD32 inp_precision,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 out_height,
    WORD32 out_width)
{
    XA_NNLIB_CHK_COND((input_width <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_width > input_width), -1);
    /* For 8 and 16 bit variants kernel_height and kernel_width should be less than or equal to 1024 */
    XA_NNLIB_CHK_COND((inp_precision != -1 && kernel_height > 1024), -1);
    XA_NNLIB_CHK_COND((inp_precision != -1 && kernel_width > 1024), -1);
    XA_NNLIB_CHK_COND((x_stride <= 0), -1);
    XA_NNLIB_CHK_COND((y_stride <= 0), -1);
    XA_NNLIB_CHK_COND((x_padding < 0), -1);
    XA_NNLIB_CHK_COND((out_height <= 0), -1);
    XA_NNLIB_CHK_COND((out_width <= 0), -1);

    int total_size, state_size, tmp_out_size;
    int den_array_size;     /* Array to store 1/den for out_height and out_width */
    int full_buf_width, full_out_width;
    int /*inp_bytewidth,*/ acc_bytewidth;

    /* Precision check is taken care here */
    switch(inp_precision)
    {
        case 8:
            //inp_bytewidth = sizeof(WORD8);
            acc_bytewidth = sizeof(WORD32);
            break;
        case 16:
            //inp_bytewidth = sizeof(WORD16);
            acc_bytewidth = sizeof(WORD32);
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
    state_size = ALIGNED_SIZE(sizeof(xa_nn_avgpool_state_t), ALIGNMENT);
   /* Array for storing 1/den values */
    if(inp_precision == 16 || inp_precision == 8 || inp_precision == -3)
        den_array_size = ALIGNED_SIZE((out_height+out_width)*sizeof(WORD32), ALIGNMENT);
    else
        den_array_size = 0;
    /* Output scratch buffer size */
    full_buf_width = kernel_width + (out_width - 1)*x_stride;
    full_buf_width = MAX(full_buf_width, ALIGNED_SIZE(x_padding, 2)+input_width);
    full_buf_width = ALIGNED_SIZE(full_buf_width, ALIGNMENT/2);
    /* Need 2 rows of padded input width as acratch for temp output */
    full_out_width = ALIGNED_SIZE(full_buf_width + kernel_width, 4);
    tmp_out_size = 2 * full_out_width*acc_bytewidth;

    /* Total size */
    total_size = state_size + den_array_size + tmp_out_size;
    return total_size;
}

static WORD32 xa_nn_avgpool_getsize_nhwc(
    WORD32 inp_precision,
    WORD32 input_channels,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 out_height,
    WORD32 out_width)
{
    //XA_NNLIB_CHK_COND((kernel_width > input_width), -1);
    /* For 8 and 16 bit variants kernel_height and kernel_width should be less than or equal to 1024 */
    XA_NNLIB_CHK_COND((inp_precision != -1 && kernel_height > 1024), -1);
    XA_NNLIB_CHK_COND((inp_precision != -1 && kernel_width > 1024), -1);

    int total_size;
    int den_array_size;     /* Array to store 1/den for out_height and out_width */

    if(input_channels == 1)
    {
        total_size = xa_nn_avgpool_getsize_nchw(
                inp_precision,
                input_width,
                kernel_height,
                kernel_width,
                x_stride,
                y_stride,
                x_padding,
                out_height,
                out_width);

        return total_size;
    }

    if(inp_precision == -1)
    {
        den_array_size = out_width*out_height;

        total_size = 2*ALIGNED_SIZE((sizeof(FLOAT32) * input_width * input_channels), ALIGNMENT) +
                     ALIGNED_SIZE((sizeof(FLOAT32) * den_array_size), ALIGNMENT);

        total_size = ALIGNED_SIZE(total_size, ALIGNMENT);
    }
    else if((inp_precision == -3) || (inp_precision == 8))
    {
        int cw_plane_size;
        int zero_mem_bytes;
        cw_plane_size = input_width*input_channels;

        if(kernel_height <= (int)MAX_HEIGHT_16_BIT_ACC) // Accumulation in 16 bit container
        {
            zero_mem_bytes = MAX((int)(sizeof(UWORD8)*cw_plane_size), (int)(sizeof(WORD16)*input_channels));

            total_size = ALIGNED_SIZE(sizeof(WORD32)* out_height, ALIGNMENT) +
                         ALIGNED_SIZE(sizeof(WORD32)* out_width, ALIGNMENT) +
                         ALIGNED_SIZE((sizeof(WORD16)*cw_plane_size), ALIGNMENT) +
                         ALIGNED_SIZE((sizeof(WORD32)*input_channels), ALIGNMENT) +
                         zero_mem_bytes;

            total_size = ALIGNED_SIZE(total_size, ALIGNMENT);
        }
        else  // Accumulation in 32 bit container
        {
            zero_mem_bytes = MAX((int)(sizeof(UWORD8)*cw_plane_size), (int)(sizeof(WORD32)*input_channels));

            total_size = ALIGNED_SIZE(sizeof(WORD32)*out_height, ALIGNMENT) +
                         ALIGNED_SIZE(sizeof(WORD32)*out_width, ALIGNMENT) +
                         ALIGNED_SIZE(sizeof(WORD32)*cw_plane_size, ALIGNMENT) +
                         ALIGNED_SIZE(sizeof(WORD32)*input_channels, ALIGNMENT) +
                         zero_mem_bytes;

            total_size = ALIGNED_SIZE(total_size, ALIGNMENT);
        }
    }
    else if(inp_precision == 16)
    {
        int cw_plane_size;
        int zero_mem_bytes;

        cw_plane_size = input_width*input_channels;
        zero_mem_bytes = MAX((int)(sizeof(WORD16)*cw_plane_size), (int)(sizeof(WORD32)*input_channels));

        total_size = ALIGNED_SIZE(sizeof(WORD32)*out_height, ALIGNMENT) +
            ALIGNED_SIZE(sizeof(WORD32)*out_width, ALIGNMENT) +
            ALIGNED_SIZE(sizeof(WORD32)*cw_plane_size, ALIGNMENT) +
            ALIGNED_SIZE(sizeof(WORD32)*input_channels, ALIGNMENT) +
            zero_mem_bytes;

            total_size = ALIGNED_SIZE(total_size, ALIGNMENT);
    }
    else
    {
        total_size = -1;
    }

    return total_size;
}

#ifndef NNLIB_V2
WORD32 xa_nn_avgpool_getsize(
    WORD32 inp_precision,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 out_height,
    WORD32 out_width)
{
    XA_NNLIB_CHK_COND((input_width <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_width > input_width), -1);
    /* For 8 and 16 bit variants kernel_height and kernel_width should be less than or equal to 1024 */
    XA_NNLIB_CHK_COND((inp_precision != -1 && kernel_height > 1024), -1);
    XA_NNLIB_CHK_COND((inp_precision != -1 && kernel_width > 1024), -1);
    XA_NNLIB_CHK_COND((x_stride <= 0), -1);
    XA_NNLIB_CHK_COND((y_stride <= 0), -1);
    XA_NNLIB_CHK_COND((x_padding < 0), -1);
    XA_NNLIB_CHK_COND((out_height <= 0), -1);
    XA_NNLIB_CHK_COND((out_width <= 0), -1);

    int total_size, state_size, tmp_out_size;
    int den_array_size;     /* Array to store 1/den for out_height and out_width */
    int full_buf_width, full_out_width;
    int inp_bytewidth, acc_bytewidth;

    /* Precision check is taken care here */
    switch(inp_precision)
    {
        case 8:
            inp_bytewidth = sizeof(WORD8);
            acc_bytewidth = sizeof(WORD32);
            break;
        case 16:
            inp_bytewidth = sizeof(WORD16);
            acc_bytewidth = sizeof(WORD32);
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
    state_size = ALIGNED_SIZE(sizeof(xa_nn_avgpool_state_t), ALIGNMENT);
   /* Array for storing 1/den values */
    if(inp_precision == 16 || inp_precision == 8 || inp_precision == -3)
        den_array_size = ALIGNED_SIZE((out_height+out_width)*sizeof(WORD32), ALIGNMENT);
    else
        den_array_size = 0;
    /* Output scratch buffer size */
    full_buf_width = kernel_width + (out_width - 1)*x_stride;
    full_buf_width = MAX(full_buf_width, ALIGNED_SIZE(x_padding, 2)+input_width);
    full_buf_width = ALIGNED_SIZE(full_buf_width, ALIGNMENT/2);
    /* Need 2 rows of padded input width as acratch for temp output */
    full_out_width = ALIGNED_SIZE(full_buf_width + kernel_width, 4);
    tmp_out_size = 2 * full_out_width*acc_bytewidth;

    /* Total size */
    total_size = state_size + den_array_size + tmp_out_size;
    return total_size;
}
#else
WORD32 xa_nn_avgpool_getsize(
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

    if(out_data_format == 0)
    {
        scratch_size = xa_nn_avgpool_getsize_nhwc(
                inp_precision,
                input_channels,
                input_width,
                kernel_height,
                kernel_width,
                x_stride,
                y_stride,
                x_padding,
                out_height,
                out_width);
    }
    else if(out_data_format == 1)
    {
        scratch_size = xa_nn_avgpool_getsize_nchw(
                inp_precision,
                input_width,
                kernel_height,
                kernel_width,
                x_stride,
                y_stride,
                x_padding,
                out_height,
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
VOID xa_nn_avgpool_init(
    WORD32 inp_precision,
    pVOID  p_scratch,
    WORD32 out_height,
    WORD32 out_width)
{
    pWORD8 p_mem = (pVOID)p_scratch;
    xa_nn_avgpool_state_t *p_state = (xa_nn_avgpool_state_t *)p_mem;
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
    state_size = ALIGNED_SIZE(sizeof(xa_nn_avgpool_state_t), ALIGNMENT);

    p_mem = (p_mem + state_size);
    /* Initialize 1/den array pointers */
    if(inp_precision == 16 || inp_precision == 8 || inp_precision == -3)
    {
        p_state->p_den_height = (WORD32 *)p_mem;
        p_mem = (p_mem + out_height*sizeof(WORD32));
        p_state->p_den_width = (WORD32 *)p_mem;
        p_mem = (p_mem + out_width*sizeof(WORD32));
    }
    else
    {
        p_state->p_den_height = NULL;
        p_state->p_den_width = NULL;
    }
    /* Initialize output scratch pointer */
    p_state->p_tmp_out = (pVOID)ALIGN_PTR(p_mem, ALIGNMENT);
}
#endif // #ifndef ENABLE_SCRATCH_SIZE_API_ONLY
