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
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"

#include "xa_nnlib_common.h"

#define INCR_N_ROW(ptr, n) \
    ptr = (ae_int16x4 *)((WORD16 *)(ptr) + (n) * (input_width));

#define INCR_ROW_IF_HEIGHT(ptr, height) \
        if(height) \
        { \
            INCR_N_ROW(ptr, 1); \
            height--; \
        }

#define INC_1_IF_WIDTH(ptr, width) \
        if(width) \
        { \
            ptr = (ae_int16x4 *)((WORD16 *)ptr + 1); \
            width--; \
        }

#if XCHAL_HAVE_HIFI1

#define MAX_16X4(id1, id0) \
        id1 = AE_MAX16(id1, id0);\

#else

#define MAX_16X4(id1, id0) \
        b0 = AE_LT16(id1, id0); \
        AE_MOVT16X4(id1, id0, b0); \

#endif

/* Max pooling without using extra copy of input data
 * Works with unaligned input, output.
 */
static void maxpool_16(
      WORD16* __restrict__ p_out,
const WORD16* __restrict__ p_inp,
      WORD32  input_height,
      WORD32  input_width,
      WORD32  kernel_height,
      WORD32  kernel_width,
      WORD32  x_stride,
      WORD32  y_stride,
      WORD32  x_padding,
      WORD32  y_padding,
      WORD32  out_height,
      WORD32  out_width,
      pVOID   p_scratch_in)
{
    WORD16 *p_scratch = (WORD16 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int left_pad_aligned, right_pad, total_out_width, scratch_width;
    ae_int16x4 * p_src1, * p_src2, * p_src3;
    ae_int16x4 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    ae_int16x4 *p_dst, *p_dst_temp;
    ae_valign align_src1, align_src2, align_src3;
    int i;
    WORD16 *p_dst_pad;
#if !XCHAL_HAVE_HIFI1
    xtbool4 b0;
#endif

    left_pad_aligned = ALIGNED_SIZE(x_padding, ALIGNMENT/sizeof(WORD16));

    /* Left padding of temporary output with min_value */
    p_dst_pad = p_scratch;
    for(i = 0; i < left_pad_aligned; i++)
    {
        p_dst_pad[i] = 0x8000;
    }

    total_out_width = XT_MAX(input_width + x_padding, (out_width - 1) * x_stride + kernel_width);
    right_pad = total_out_width - (x_padding + input_width);

    /* Right padding of temporary output with min_value,
     * add kernel_width values more for the aligning load operations */
    p_dst_pad = p_scratch + left_pad_aligned + input_width;
    for(i = 0; i < right_pad + kernel_width; i++)
    {
        p_dst_pad[i] = 0x8000;
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

        p_dst = (ae_int16x4 *)((WORD16 *)p_scratch + left_pad_aligned);

        if(pool_height)
        {
            p_src1 = (ae_int16x4 *)p_inp;
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
                align_src1 = AE_LA64_PP(p_src1_temp);
                align_src2 = AE_LA64_PP(p_src2_temp);
                align_src3 = AE_LA64_PP(p_src3_temp);

                //TODO: improve packing, use AE_MAX32 ?
                for(i = 0; i < (input_width >> 2); i++)
                {
                    ae_int16x4 i1, i2, i3;

                    AE_LA16X4_IP(i1, align_src1, p_src1_temp);
                    AE_LA16X4_IP(i2, align_src2, p_src2_temp);
                    AE_LA16X4_IP(i3, align_src3, p_src3_temp);

                    MAX_16X4(i1, i2);
                    MAX_16X4(i1, i3);
                    AE_S16X4_IP(i1, p_dst_temp, 8);
                }

                /* reminder loop for input_width */
                for(i = 0; i < (input_width & 3); i++)
                {
                    ae_int16x4 i1, i2, i3;

                    AE_L16_IP(i1, (ae_int16 *)p_src1_temp, 2);
                    AE_L16_IP(i2, (ae_int16 *)p_src2_temp, 2);
                    AE_L16_IP(i3, (ae_int16 *)p_src3_temp, 2);

                    MAX_16X4(i1, i2);
                    MAX_16X4(i1, i3);
                    AE_S16_0_IP(i1, (ae_int16 *)p_dst_temp, 2);
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
                p_dst_pad[i] = 0x8000;
            }
        }

        /* Pool width processing */

        /* On scratch, compare width-wise with padding*/
        total_out_width = ALIGNED_SIZE(left_pad_aligned + input_width + right_pad + kernel_width, ALIGNMENT/sizeof(WORD16));
        scratch_width = x_padding + input_width + right_pad;
        p_dst = (ae_int16x4 *)((WORD16 *)p_scratch + total_out_width);
        pool_width = kernel_width;

        p_src1 = (ae_int16x4 *)((WORD16 *)p_scratch + left_pad_aligned - x_padding);
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
            align_src1 = AE_LA64_PP(p_src1_temp);
            align_src2 = AE_LA64_PP(p_src2_temp);
            align_src3 = AE_LA64_PP(p_src3_temp);

            for(i = 0; i < (scratch_width >> 2); i++)
            {
                ae_int16x4 src1, src2, src3;

                AE_LA16X4_IP(src1, align_src1, p_src1_temp);
                AE_LA16X4_IP(src2, align_src2, p_src2_temp);
                AE_LA16X4_IP(src3, align_src3, p_src3_temp);

                MAX_16X4(src1, src2);
                MAX_16X4(src1, src3);
                AE_S16X4_IP(src1, p_dst_temp, 8);
            }

            /* reminder loop for scratch_width */
            for(i = 0; i < (scratch_width & 3); i++)
            {
                ae_int16x4 src1, src2, src3;

                AE_L16_IP(src1, (ae_int16 *)p_src1_temp, 2);
                AE_L16_IP(src2, (ae_int16 *)p_src2_temp, 2);
                AE_L16_IP(src3, (ae_int16 *)p_src3_temp, 2);

                MAX_16X4(src1, src2);
                MAX_16X4(src1, src3);
                AE_S16_0_IP(src1, (ae_int16 *)p_dst_temp, 2);
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

        WORD16 *ptr_out1 = p_scratch + total_out_width;
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            p_out[itr_oh * out_width * 1 /* out_stride */ + itr_ow * 1 /* out_stride */] = ptr_out1[itr_ow * x_stride];
        }
    }
}

WORD32 xa_nn_maxpool_16(
      WORD16* __restrict__ p_out,
const WORD16* __restrict__ p_inp,
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
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
#ifndef NNLIB_V2
    XA_NNLIB_ARG_CHK_COND((out_data_format != 1), -1);
#else
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0) && (out_data_format != 1), -1);
#endif
#ifdef NNLIB_V2
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0) && (inp_data_format != 1), -1);
    // Different I/O data formats (not supported!)
    XA_NNLIB_ARG_CHK_COND((out_data_format != inp_data_format), -1);
#endif

    if((input_channels == 1) || (out_data_format == 1))
    {
        err = xa_nn_maxpool_init(16
                ,p_scratch
                );
        if(err<0)
            return err;

        xa_nn_maxpool_state_t *p_state = (xa_nn_maxpool_state_t *)p_scratch;
        WORD16 *p_scratch_in = (WORD16 *)(p_state->p_scratch);
        int itr_ic;
        const WORD16 *pt_inp; 
        WORD16 *pt_out;

        for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
        {
            pt_inp = &p_inp[itr_ic * input_height * input_width];
            pt_out = &p_out[itr_ic * out_height * out_width];

            maxpool_16(pt_out
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
#ifdef NNLIB_V2
    else
    {
        void *p_scratch_aligned = (void *)ALIGN_PTR(p_scratch, ALIGNMENT);

        xa_nn_maxpool_16_hwc(p_out
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
#endif
    return 0;
}

