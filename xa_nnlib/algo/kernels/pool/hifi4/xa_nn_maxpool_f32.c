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
#include "xa_nnlib_common_fpu.h"
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_maxpool_state.h"
#include "xa_nnlib_err_chk.h"
#include <math.h>

#include "xa_nnlib_common.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_maxpool_f32,(
    FLOAT32* __restrict__ p_out,
const FLOAT32* __restrict__ p_inp,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
#ifdef NNLIB_V2
    WORD32  inp_data_format,
#endif
    WORD32  out_data_format,
    VOID *p_scratch))
#else /* #if !HAVE_VFPU */

#define INCR_N_ROW(ptr, n) \
    ptr = (xtfloatx2 *)((FLOAT32 *)(ptr) + (n) * (input_width));

#define INCR_ROW_IF_HEIGHT(ptr, height) \
        if(height) \
        { \
            INCR_N_ROW(ptr, 1); \
            height--; \
        }

#define INC_1_IF_WIDTH(ptr, width) \
        if(width) \
        { \
            ptr = (xtfloatx2 *)((FLOAT32 *)ptr + 1); \
            width--; \
        }

/* Max pooling without using extra copy of input data
 * Works with unaligned input, output.
 */

static void maxpool_f32_hw(
      FLOAT32* __restrict__ p_out,
const FLOAT32* __restrict__ p_inp,
      WORD32  input_height,
      WORD32   input_width,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32  x_padding,
      WORD32  y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in)
{
    FLOAT32 *p_scratch = (FLOAT32 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int left_pad_aligned, right_pad, total_out_width, scratch_width;
    xtfloatx2 * p_src1, * p_src2, * p_src3;
    xtfloatx2 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    xtfloatx2 *p_dst, *p_dst_temp;
    ae_valign align_src1, align_src2, align_src3;
    int i;
    FLOAT32 *p_dst_pad;


    left_pad_aligned = ALIGNED_SIZE(x_padding, ALIGNMENT/sizeof(FLOAT32));

    /* Left padding of temporary output with min_value */
    p_dst_pad = p_scratch;
    for(i = 0; i < left_pad_aligned; i++)
    {
        p_dst_pad[i] = -INFINITY;
    }

    total_out_width = XT_MAX(input_width + x_padding, (out_width - 1) * x_stride + kernel_width);
    right_pad = total_out_width - (x_padding + input_width);

    /* Right padding of temporary output with min_value,
     * add kernel_width values more for the aligning load operations */
    p_dst_pad = p_scratch + left_pad_aligned + input_width;
    for(i = 0; i < right_pad + kernel_width; i++)
    {
        p_dst_pad[i] = -INFINITY;
    }

    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row, end_row;

        /* Pool height processing */

        /* Compare the input rows for the required pooling height and store on scratch */
        start_row  = itr_oh * y_stride - y_padding;
        end_row = start_row + kernel_height;
        LIMIT(start_row , 0, input_height);
        LIMIT(end_row , 0, input_height);

        pool_height = end_row - start_row;

        p_dst = (xtfloatx2 *)((FLOAT32 *)p_scratch + left_pad_aligned);

        if(pool_height)
        {
            p_src1 = (xtfloatx2 *)p_inp;
            INCR_N_ROW(p_src1, start_row);
            pool_height--;

            p_src2 = p_src1;
            INCR_ROW_IF_HEIGHT(p_src2, pool_height);

            p_src3 = p_src2;
            INCR_ROW_IF_HEIGHT(p_src3, pool_height);

            /* Compare three rows per iteration */
            do
            {
                p_dst_temp = p_dst;
                p_src1_temp = p_src1;
                p_src2_temp = p_src2;
                p_src3_temp = p_src3;

                /* prime */
                align_src1 = XT_LASX2PP(p_src1_temp);
                align_src2 = XT_LASX2PP(p_src2_temp);
                align_src3 = XT_LASX2PP(p_src3_temp);

                for(i = 0; i < (input_width >> 1); i++)
                {
                    xtfloatx2 temp, i1, i2, i3, out;

                    XT_LASX2IP(i1, align_src1, p_src1_temp);
                    XT_LASX2IP(i2, align_src2, p_src2_temp);
                    XT_LASX2IP(i3, align_src3, p_src3_temp);

                    temp = XT_MAX_SX2(i1, i2);
                    out = XT_MAX_SX2(temp, i3);
                    XT_SSX2IP(out, p_dst_temp, 8);
                }

                /* reminder loop for input_width */
                if(input_width & 1)
                {
                    xtfloatx2 temp, i1, i2, i3, out;

                    i1 = ((FLOAT32 *)p_src1_temp)[0];
                    i2 = ((FLOAT32 *)p_src2_temp)[0];
                    i3 = ((FLOAT32 *)p_src3_temp)[0];

                    temp = XT_MAX_SX2(i1, i2);
                    out = XT_MAX_SX2(temp, i3);
                    ((FLOAT32 *)p_dst_temp)[0] = out;
                }


                if(!pool_height)
                    break;

                p_src1 = p_dst;

                p_src2 = p_src3;
                INCR_ROW_IF_HEIGHT(p_src2, pool_height);

                p_src3 = p_src2;
                INCR_ROW_IF_HEIGHT(p_src3, pool_height);

            }while(1);
        }
        else
        {
            /* If there is no valid input present, fill the output with min_value */
            p_dst_pad = p_scratch + left_pad_aligned ;
            for(i = 0; i < input_width; i++)
            {
                p_dst_pad[i] = -INFINITY;
            }
        }

        /* Pool width processing */

        /* On scratch, compare width-wise with padding*/
        total_out_width = ALIGNED_SIZE(left_pad_aligned + input_width + right_pad + kernel_width, ALIGNMENT/sizeof(FLOAT32));
        scratch_width = x_padding + input_width + right_pad;
        p_dst = (xtfloatx2 *)((FLOAT32 *)p_scratch + total_out_width);
        pool_width = kernel_width;

        p_src1 = (xtfloatx2 *)((FLOAT32 *)p_scratch + left_pad_aligned - x_padding);
        pool_width--;

        p_src2 = p_src1;
        INC_1_IF_WIDTH(p_src2, pool_width);

        p_src3 = p_src2;
        INC_1_IF_WIDTH(p_src3, pool_width);

        do
        {
            p_dst_temp = p_dst;
            p_src1_temp = p_src1;
            p_src2_temp = p_src2;
            p_src3_temp = p_src3;

            /* prime */
            align_src1 = XT_LASX2PP(p_src1_temp);
            align_src2 = XT_LASX2PP(p_src2_temp);
            align_src3 = XT_LASX2PP(p_src3_temp);

            for(i = 0; i < (scratch_width >> 1); i++)
            {
                xtfloatx2 temp , src1, src2, src3, out;

                XT_LASX2IP(src1, align_src1, p_src1_temp);
                XT_LASX2IP(src2, align_src2, p_src2_temp);
                XT_LASX2IP(src3, align_src3, p_src3_temp);

                temp = XT_MAX_SX2(src1, src2);
                out = XT_MAX_SX2(temp, src3);
                XT_SSX2IP(out, p_dst_temp, 8);
            }

            /* reminder loop for scratch_width */
             if(scratch_width & 1)
             {
                xtfloatx2 temp , src1, src2, src3, out;

                src1 = ((FLOAT32 *)p_src1_temp)[0];
                src2 = ((FLOAT32 *)p_src2_temp)[0];
                src3 = ((FLOAT32 *)p_src3_temp)[0];

                temp = XT_MAX_SX2(src1, src2);
                out = XT_MAX_SX2(temp, src3);
                ((FLOAT32 *)p_dst_temp)[0] = out;
             }

            if(!pool_width)
                break;

            /* Setup next iteration */
            p_src1 = p_dst;
            p_src2 = p_src3;
            INC_1_IF_WIDTH(p_src2, pool_width);
            p_src3 = p_src2;
            INC_1_IF_WIDTH(p_src3, pool_width);

        }while(1);

        FLOAT32 *ptr_out1 = p_scratch + total_out_width;
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            p_out[itr_oh * out_width * 1 /* out_stride */ + itr_ow * 1 /* out_stride */] = ptr_out1[itr_ow * x_stride];
        }
    }
}

WORD32 xa_nn_maxpool_f32(
      FLOAT32* __restrict__ p_out,
const FLOAT32* __restrict__ p_inp,
      WORD32  input_height,
      WORD32  input_width,
      WORD32  input_channels,
      WORD32  kernel_height,
      WORD32  kernel_width,
      WORD32  x_stride,
      WORD32  y_stride,
      WORD32  x_padding,
      WORD32  y_padding,
      WORD32  out_height,
      WORD32  out_width,
#ifdef NNLIB_V2
      WORD32  inp_data_format,
#endif
      WORD32  out_data_format,
      VOID   *p_scratch)
{
    WORD32 err = 0;

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0) && (out_data_format != 1), -1);
#ifdef NNLIB_V2
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0) && (inp_data_format != 1), -1);

    // Different I/O formats (not supported!)
    XA_NNLIB_ARG_CHK_COND((out_data_format != inp_data_format), -1);
#endif

    if((input_channels == 1) || (out_data_format == 1))
    {
        err = xa_nn_maxpool_init(-1
                ,p_scratch
                );
        if(err<0)
            return err;

        xa_nn_maxpool_state_t *p_state = (xa_nn_maxpool_state_t *)p_scratch;
        FLOAT32 *p_scratch_in = (FLOAT32 *)(p_state->p_scratch);
        int itr_ic;
        const FLOAT32 *pt_inp; 
        FLOAT32 *pt_out;

        for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
        {
            pt_inp = &p_inp[itr_ic * input_height * input_width];
            pt_out = &p_out[itr_ic * out_height * out_width];

            maxpool_f32_hw(pt_out
                    ,pt_inp
                    ,input_height
                    ,input_width
                    ,kernel_height
                    ,kernel_width
                    ,x_stride
                    ,y_stride
                    ,x_padding
                    ,y_padding
                    ,out_height
                    ,out_width
                    ,p_scratch_in
                    );
        }
    }
    else
    {
        void *p_scratch_aligned = (void *)ALIGN_PTR(p_scratch, ALIGNMENT);

        xa_nn_maxpool_f32_hwc(p_out
                ,p_inp
                ,input_height
                ,input_width
                ,input_channels
                ,kernel_height
                ,kernel_width
                ,x_stride
                ,y_stride
                ,x_padding
                ,y_padding
                ,out_height
                ,out_width
                ,p_scratch_aligned);
    }
    return 0;
}
#endif /* #if !HAVE_VFPU */

