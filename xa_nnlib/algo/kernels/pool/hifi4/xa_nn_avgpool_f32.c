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
#include "xa_nn_avgpool_state.h"
#include "xa_nnlib_err_chk.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_avgpool_f32,(
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
    VOID *handle))
#else /* #if !HAVE_VFPU */
static void avgpool_f32_hw(
      FLOAT32* __restrict__ p_out,
const FLOAT32* __restrict__ p_inp,
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
      WORD32  out_plane_size,
      WORD32  not_last_channel,
      pVOID   p_scratch_in)
{
    FLOAT32 *p_scratch = (FLOAT32 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int left_pad_aligned, right_pad, total_out_width, scratch_width;
    xtfloatx2 * p_src1, * p_src2;
    xtfloatx2 * __restrict p_src1_temp, * __restrict p_src2_temp;
    xtfloatx2 * p_src0_temp;
    xtfloatx2 *p_dst, *p_dst_temp;
    ae_valign align_src1, align_src2;
    int i;
    FLOAT32 *p_dst_pad;


    left_pad_aligned = ALIGNED_SIZE(x_padding, ALIGNMENT/sizeof(FLOAT32));

    /* Left padding of temporary output with min_value */
    p_dst_pad = p_scratch;
    for(i = 0; i < left_pad_aligned; i++)
    {
        p_dst_pad[i] = 0.0f;
    }

    total_out_width = XT_MAX(input_width + x_padding, (out_width - 1) * x_stride + kernel_width);
    right_pad = total_out_width - (x_padding + input_width);

    /* Right padding of temporary output with min_value,
     * add kernel_width values more for the aligning load operations */
    p_dst_pad = p_scratch + left_pad_aligned + input_width;
    for(i = 0; i < right_pad + kernel_width; i++)
    {
        p_dst_pad[i] = 0.0f;
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
            p_src1 = (xtfloatx2 *)((FLOAT32 *)p_src1 + start_row*input_width);
            pool_height--;
            p_dst_temp = p_dst;
            p_src0_temp = p_src1;

            /* prime */
            align_src1 = XT_LASX2PP(p_src0_temp);

            for(i = 0; i < (input_width >> 1); i++)
            {
                xtfloatx2 i1;
                XT_LASX2IP(i1, align_src1, p_src0_temp);
                XT_SSX2IP(i1, p_dst_temp, 8);
            }

            /* reminder loop for input_width */
            if(input_width & 1)
            {
                xtfloatx2 out;
                out = ((FLOAT32 *)p_src0_temp)[0];
                ((FLOAT32 *)p_dst_temp)[0] = out;
            }

            p_src2 = p_src1;
            p_src1 = p_dst;
            /* Compare three rows per iteration */
            while(pool_height)
            {
                p_src2 = (xtfloatx2 *)((FLOAT32 *)p_src2 + input_width);
                p_dst_temp = p_dst;
                p_src1_temp = p_src1;
                p_src2_temp = p_src2;

                /* prime */
                align_src2 = XT_LASX2PP(p_src2_temp);

                for(i = 0; i < (input_width >> 1); i++)
                {
                    xtfloatx2 i1, i2, out;

                    XT_LSX2IP(i1, p_src1_temp, 8);
                    XT_LASX2IP(i2, align_src2, p_src2_temp);

                    out = XT_ADD_SX2(i1, i2);
                    XT_SSX2IP(out, p_dst_temp, 8);
                }

                /* reminder loop for input_width */
                if(input_width & 1)
                {
                    xtfloatx2 i1, i2, out;

                    i1 = ((FLOAT32 *)p_src1_temp)[0];
                    i2 = ((FLOAT32 *)p_src2_temp)[0];

                    out = XT_ADD_SX2(i1, i2);
                    ((FLOAT32 *)p_dst_temp)[0] = out;
                }
                pool_height--;
            };
        }
        else
        {
            /* If there is no valid input present, fill the output with min_value */
            p_dst_pad = p_scratch + left_pad_aligned ;
            for(i = 0; i < input_width; i++)
            {
                p_dst_pad[i] = 0.0f;
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
        p_dst_temp = p_dst;
        p_src0_temp = p_src1;

        /* prime */
        align_src1 = XT_LASX2PP(p_src0_temp);

        for(i = 0; i < (scratch_width >> 1); i++)
        {
            xtfloatx2 src1;
            XT_LASX2IP(src1, align_src1, p_src0_temp);
            XT_SSX2IP(src1, p_dst_temp, 8);
        }

        /* reminder loop for scratch_width */
        if(scratch_width & 1)
        {
           xtfloatx2 src1;
           src1 = ((FLOAT32 *)p_src0_temp)[0];
           ((FLOAT32 *)p_dst_temp)[0] = src1;
        }

        p_src2 = p_src1;
        p_src1 = p_dst;

        while(pool_width > 0)
        {
            p_src2 = (xtfloatx2 *)((FLOAT32 *)p_src2 + 1);
            p_dst_temp = p_dst;
            p_src1_temp = p_src1;
            p_src2_temp = p_src2;

            /* prime */
            align_src2 = XT_LASX2PP(p_src2_temp);

            for(i = 0; i < (scratch_width >> 1); i++)
            {
                xtfloatx2 src1, src2, out;
                XT_LSX2IP(src1, p_src1_temp, 8);
                XT_LASX2IP(src2, align_src2, p_src2_temp);
                out = XT_ADD_SX2(src1, src2);
                XT_SSX2IP(out, p_dst_temp, 8);
            }

            /* reminder loop for scratch_width */
             if(scratch_width & 1)
             {
                xtfloatx2 src1, src2, out;

                src1 = ((FLOAT32 *)p_src1_temp)[0];
                src2 = ((FLOAT32 *)p_src2_temp)[0];

                out = XT_ADD_SX2(src1, src2);
                ((FLOAT32 *)p_dst_temp)[0] = out;
             }
             pool_width--;
        };

        FLOAT32 *ptr_out1 = (FLOAT32 *)((FLOAT32 *)p_scratch + total_out_width);
        FLOAT32 den_inv;
        if(not_last_channel)
        {
            for(itr_ow = 0; itr_ow < out_width; itr_ow++)
            {
                den_inv = p_out[itr_oh*out_width+itr_ow];
                p_out[itr_oh*out_width+itr_ow] = XT_MUL_S(ptr_out1[itr_ow*x_stride], den_inv);
                /* store 1/den for next channel */
                p_out[out_plane_size + itr_oh*out_width+itr_ow] = den_inv;
            }
        }
        else
        {
            for(itr_ow = 0; itr_ow < out_width; itr_ow++)
            {
                den_inv = p_out[itr_oh*out_width+itr_ow];
                p_out[itr_oh*out_width+itr_ow] = XT_MUL_S(ptr_out1[itr_ow*x_stride], den_inv);
            }
        }
    }
}

WORD32 xa_nn_avgpool_f32(
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
      VOID *p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
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
        xa_nn_avgpool_init(-1,
                p_scratch,
                out_height,
                out_width);

        xa_nn_avgpool_state_t *p_state = (xa_nn_avgpool_state_t *)p_scratch;
        FLOAT32 *p_tmp_out = (FLOAT32 *)(p_state->p_tmp_out);
        int itr_ic, itr_oh, itr_ow;
        const FLOAT32 *pt_inp; 
        FLOAT32 *pt_out;

        /* Calculate denominators for division */
        for(itr_oh = 0; itr_oh < out_height; itr_oh++)
        {
            int kernel_x_start, kernel_x_end, kernel_y_start, kernel_y_end;
            kernel_y_start = itr_oh*y_stride - y_padding;
            kernel_y_end = kernel_y_start + kernel_height;
            LIMIT(kernel_y_start, 0, input_height)
            LIMIT(kernel_y_end, 0, input_height)
            for(itr_ow = 0; itr_ow < out_width; itr_ow++)
            {
                kernel_x_start = itr_ow*x_stride - x_padding;
                kernel_x_end = kernel_x_start + kernel_width;
                LIMIT(kernel_x_start, 0, input_width)
                LIMIT(kernel_x_end, 0, input_width)
                FLOAT32 den = (FLOAT32)((kernel_y_end-kernel_y_start)*(kernel_x_end-kernel_x_start));
                p_out[itr_oh*out_width+itr_ow] = XT_MAX_S(XT_RECIP_S(den), 0.0f);
            }
        }

        for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
        {
            pt_inp = &p_inp[itr_ic * input_height * input_width];
            pt_out = &p_out[itr_ic * out_height * out_width];

            avgpool_f32_hw(pt_out
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
                    ,out_height*out_width
                    ,(input_channels-itr_ic-1)
                    ,p_tmp_out
                    );
        }
    }
    else
    {
        FLOAT32 *p_rec_den, *p_den, *p_zeros_mem;
        void *p_scratch_aligned;
        int itr_oh, itr_ow;

        p_scratch_aligned = (void *)ALIGN_PTR(p_scratch, ALIGNMENT);

        p_rec_den = (FLOAT32 *)((WORD8 *)p_scratch_aligned +
            2*ALIGNED_SIZE((sizeof(FLOAT32) * input_channels * input_width), ALIGNMENT));

        p_den = p_rec_den;

        /* Calculate denominators for division */
        for(itr_oh = 0; itr_oh < out_height; itr_oh++)
        {
            int kernel_x_start, kernel_x_end, kernel_y_start, kernel_y_end;
            kernel_y_start = itr_oh*y_stride - y_padding;
            kernel_y_end = kernel_y_start + kernel_height;

            LIMIT(kernel_y_start, 0, input_height)
            LIMIT(kernel_y_end, 0, input_height)

            for(itr_ow = 0; itr_ow < out_width; itr_ow++)
            {
                kernel_x_start = itr_ow*x_stride - x_padding;
                kernel_x_end = kernel_x_start + kernel_width;

                LIMIT(kernel_x_start, 0, input_width)
                LIMIT(kernel_x_end, 0, input_width)

                FLOAT32 den = (FLOAT32)((kernel_y_end-kernel_y_start)*(kernel_x_end-kernel_x_start));
                p_rec_den[itr_oh*out_width+itr_ow] = XT_MAX_S(XT_RECIP_S(den), 0.0f);
            }
        }

        p_rec_den = (FLOAT32 *)((WORD8 *)p_scratch_aligned + ALIGNED_SIZE((sizeof(FLOAT32) * input_channels * input_width), ALIGNMENT));
        p_zeros_mem = p_rec_den;
        for(itr_oh = 0; itr_oh < input_channels*input_width; itr_oh++)
        {
            p_rec_den[itr_oh] = 0;
        }


        xa_nn_avgpool_f32_hwc(p_out
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
                ,p_scratch_aligned
                ,p_zeros_mem
                ,p_den);
    }
    return 0;
}
#endif /* #if !HAVE_VFPU */

