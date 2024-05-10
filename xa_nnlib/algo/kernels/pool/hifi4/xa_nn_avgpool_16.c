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

static void avgpool_16(
      WORD16 *__restrict__ p_out,
const WORD16 *__restrict__ p_inp,
      WORD32 *p_den_height,
      WORD32 *p_den_width,
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
    WORD32 *p_scratch = (WORD32 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int left_pad_aligned, right_pad, total_out_width, scratch_width;
    ae_int16x4 * p_src1, * p_src2;
    ae_int16x4 * __restrict p_src1_temp, * __restrict p_src2_temp;
    ae_int32x2 * p_wsrc1, * p_wsrc2;
    ae_int32x2 * __restrict p_wsrc1_temp, * __restrict p_wsrc2_temp;
    ae_int32x2 *p_dst, *p_dst_temp;
    ae_valign align_src1, align_src2;
    ae_valign align_wsrc1, align_wsrc2;
    int i;
    WORD32 *p_dst_pad;


    left_pad_aligned = ALIGNED_SIZE(x_padding, ALIGNMENT/sizeof(WORD32));

    /* Left padding of temporary output with min_value */
    p_dst_pad = p_scratch;
    for(i = 0; i < left_pad_aligned; i++)
    {
        p_dst_pad[i] = 0;
    }

    total_out_width = XT_MAX(input_width + x_padding, (out_width - 1) * x_stride + kernel_width);
    right_pad = total_out_width - (x_padding + input_width);

    /* Right padding of temporary output with min_value,
     * add kernel_width values more for the aligning load operations */
    p_dst_pad = p_scratch + left_pad_aligned + input_width;
    for(i = 0; i < right_pad + kernel_width; i++)
    {
        p_dst_pad[i] = 0;
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

        p_dst = (ae_int32x2 *)((WORD32 *)p_scratch + left_pad_aligned);

        if(pool_height)
        {
            p_src1 = (ae_int16x4 *)p_inp;
            p_src1 = (ae_int16x4 *)((WORD16 *)p_src1 + start_row*input_width);
            pool_height--;
            p_dst_temp = p_dst;
            p_src1_temp = p_src1;

            /* prime */
            align_src1 = AE_LA64_PP(p_src1_temp);

            for(i = 0; i < (input_width >> 2); i++)
            {
                ae_int16x4 i1;
                ae_int32x2 wi1, wi2;
                AE_LA16X4_IP(i1, align_src1, p_src1_temp);
                wi1 = AE_SEXT32X2D16_32(i1);
                wi2 = AE_SEXT32X2D16_10(i1);
                AE_S32X2_IP(wi1, p_dst_temp, 8);
                AE_S32X2_IP(wi2, p_dst_temp, 8);
            }

            /* reminder loop for input_width */
            for(i = 0; i < (input_width & 3); i++)
            {
                ae_int16x4 i1;
                i1 = ((ae_int16 *)p_src1_temp)[i];
                ((ae_int32 *)p_dst_temp)[i] = AE_SEXT32X2D16_32(i1);
            }

            p_src2 = p_src1;
            p_wsrc1 = p_dst;
            /* Compare three rows per iteration */
            while(pool_height)
            {
                p_src2 = (ae_int16x4 *)((WORD16 *)p_src2 + input_width);
                p_dst_temp = p_dst;
                p_wsrc1_temp = p_wsrc1;
                p_src2_temp = p_src2;

                /* prime */
                align_src2 = AE_LA64_PP(p_src2_temp);

                for(i = 0; i < (input_width >> 2); i++)
                {
                    ae_int16x4 i1, one = AE_MOVDA16(1);
                    ae_int32x2 wi1, wi2;

                    AE_L32X2_IP(wi1, p_wsrc1_temp, 8);
                    AE_L32X2_IP(wi2, p_wsrc1_temp, 8);
                    AE_LA16X4_IP(i1, align_src2, p_src2_temp);
                    AE_MULA16X4(wi1, wi2, i1, one);
                    AE_S32X2_IP(wi1, p_dst_temp, 8);
                    AE_S32X2_IP(wi2, p_dst_temp, 8);
                }

                /* reminder loop for input_width */
                for(i = 0; i < (input_width & 3); i++)
                {
                    ae_int16x4 i1;
                    ae_int32x2 wi1, out;

                    wi1 = ((ae_int32 *)p_wsrc1_temp)[i];
                    i1 = ((ae_int16 *)p_src2_temp)[i];
                    out = AE_ADD32(wi1, AE_SEXT32X2D16_32(i1));
                    ((ae_int32 *)p_dst_temp)[i] = out;
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
                p_dst_pad[i] = 0;
            }
        }

        /* Pool width processing */

        /* On scratch, compare width-wise with padding*/
        total_out_width = ALIGNED_SIZE(left_pad_aligned + input_width + right_pad + kernel_width, ALIGNMENT/sizeof(WORD32));
        scratch_width = x_padding + input_width + right_pad;
        p_dst = (ae_int32x2 *)((WORD32 *)p_scratch + total_out_width);
        pool_width = kernel_width;

        p_wsrc1 = (ae_int32x2 *)((WORD32 *)p_scratch + left_pad_aligned - x_padding);
        pool_width--;
        p_dst_temp = p_dst;
        p_wsrc1_temp = p_wsrc1;

        /* prime */
        align_wsrc1 = AE_LA64_PP(p_wsrc1_temp);

        for(i = 0; i < (scratch_width >> 1); i++)
        {
            ae_int32x2 wsrc1;
            AE_LA32X2_IP(wsrc1, align_wsrc1, p_wsrc1_temp);
            AE_S32X2_IP(wsrc1, p_dst_temp, 8);
        }

        /* reminder loop for scratch_width */
        if(scratch_width & 1)
        {
           ae_int32x2 wsrc1;
           wsrc1 = ((ae_int32 *)p_wsrc1_temp)[0];
           ((ae_int32 *)p_dst_temp)[0] = wsrc1;
        }

        p_wsrc2 = p_wsrc1;
        p_wsrc1 = p_dst;

        while(pool_width > 0)
        {
            p_wsrc2 = (ae_int32x2 *)((WORD32 *)p_wsrc2 + 1);
            p_dst_temp = p_dst;
            p_wsrc1_temp = p_wsrc1;
            p_wsrc2_temp = p_wsrc2;

            /* prime */
            align_wsrc2 = AE_LA64_PP(p_wsrc2_temp);

            for(i = 0; i < (scratch_width >> 1); i++)
            {
                ae_int32x2 wsrc1, wsrc2, out;
                AE_L32X2_IP(wsrc1, p_wsrc1_temp, 8);
                AE_LA32X2_IP(wsrc2, align_wsrc2, p_wsrc2_temp);
                out = AE_ADD32S(wsrc1, wsrc2);
                AE_S32X2_IP(out, p_dst_temp, 8);
            }

            /* reminder loop for scratch_width */
             if(scratch_width & 1)
             {
                ae_int32x2 wsrc1, wsrc2, out;

                wsrc1 = ((ae_int32 *)p_wsrc1_temp)[0];
                wsrc2 = ((ae_int32 *)p_wsrc2_temp)[0];

                out = AE_ADD32S(wsrc1, wsrc2);
                ((ae_int32 *)p_dst_temp)[0] = out;
             }
             pool_width--;
        };

        WORD32 *ptr_out1 = (WORD32 *)((WORD32 *)p_scratch + total_out_width);
        ae_int32x2 d_tmp32, d_out1;
        ae_int64 d_tmp;
        if(kernel_height * kernel_width <= 1024)
        {
          WORD32 den_hw;
          for(itr_ow = 0; itr_ow < out_width; itr_ow++)
          {
            den_hw = inv_256_tbl[p_den_height[itr_oh] * p_den_width[itr_ow]];
            d_out1 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride]);
            d_tmp32 = AE_MOVDA32(den_hw);
            #if XCHAL_HAVE_HIFI1
            d_tmp32 = AE_MULFP32X2RS_L(d_out1, d_tmp32);
            #else
            d_tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32);
            #endif
            p_out[itr_oh*out_width+itr_ow] = AE_SAT16X4(d_tmp32, d_tmp32);
          }
        }
        else
        {
          ae_int32x2 den_h, den_w;
          den_h = AE_MOVDA32(inv_256_tbl[p_den_height[itr_oh]]);
          for(itr_ow = 0; itr_ow < out_width; itr_ow++)
          {
            den_w = AE_MOVDA32(inv_256_tbl[p_den_width[itr_ow]]);
            d_out1 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride]);
            d_tmp = AE_MUL32U_LL(den_h, den_w);
            /* Max value of den_h or den_w is 0x80000000
            so 1 left shift is possible without overflow */
            d_tmp32 = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);
#if XCHAL_HAVE_HIFI1
            d_tmp32 = AE_MULFP32X2RS_L(d_out1, d_tmp32);
#else
            d_tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32);
#endif
            p_out[itr_oh*out_width+itr_ow] = AE_SAT16X4(d_tmp32, d_tmp32);
          }
        }
    }
}

WORD32 xa_nn_avgpool_16(
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
      VOID *p_scratch)
{
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
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
#ifndef NNLIB_V2
    XA_NNLIB_ARG_CHK_COND((out_data_format != 1), -1);
#else
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0) && (out_data_format != 1), -1);
#endif
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND((kernel_height > 1024), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_width > 1024), -1);

#ifdef NNLIB_V2
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0) && (inp_data_format != 1), -1);
    // Different I/O data formats (not supported!)
    XA_NNLIB_ARG_CHK_COND((out_data_format != inp_data_format), -1);
#endif

    if((input_channels == 1) || (out_data_format == 1))
    {
        xa_nn_avgpool_init(16,
                p_scratch,
                out_height,
                out_width);

        xa_nn_avgpool_state_t *p_state = (xa_nn_avgpool_state_t *)p_scratch;
        int itr_ic, itr_oh, itr_ow;
        const WORD16 *pt_inp; 
        WORD16 *pt_out;
        WORD32 *p_tmp_out = (WORD32 *)(p_state->p_tmp_out);

        /* Calculate denominators for division */
        int kernel_x_start, kernel_x_end, kernel_y_start, kernel_y_end;
        for(itr_oh = 0; itr_oh < out_height; itr_oh++)
        {
            kernel_y_start = itr_oh*y_stride - y_padding;
            kernel_y_end = kernel_y_start + kernel_height;
            LIMIT(kernel_y_start, 0, input_height)
            LIMIT(kernel_y_end, 0, input_height)
            p_state->p_den_height[itr_oh] = (kernel_y_end - kernel_y_start);
        }
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            kernel_x_start = itr_ow*x_stride - x_padding;
            kernel_x_end = kernel_x_start + kernel_width;
            LIMIT(kernel_x_start, 0, input_width)
            LIMIT(kernel_x_end, 0, input_width)
            p_state->p_den_width[itr_ow] = (kernel_x_end - kernel_x_start);
        }

        for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
        {
            pt_inp = &p_inp[itr_ic * input_height * input_width];
            pt_out = &p_out[itr_ic * out_height * out_width];

            avgpool_16(pt_out
                    ,pt_inp
                    ,p_state->p_den_height
                    ,p_state->p_den_width
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
                    ,p_tmp_out
                    );
        }
    }
#ifdef NNLIB_V2
    else
    {
        int i;
        void *p_scratch_aligned;
        WORD8 *p_zeros, *p_zeros_mem;
        WORD32 *p_rec_den, *p_den_height, *p_den_width;
        WORD32 *p_s;
        int kernel_x_start, kernel_x_end, kernel_y_start, kernel_y_end;
        int cw_plane_size, zero_mem_bytes;


        cw_plane_size = input_width * input_channels;
        p_scratch_aligned = (void *)ALIGN_PTR(p_scratch, ALIGNMENT);

        p_rec_den = (WORD32 *)p_scratch_aligned;
        p_den_height = p_rec_den;
        for(i = 0; i < out_height; i++)
        {
            kernel_y_start = i*y_stride - y_padding;
            kernel_y_end = kernel_y_start + kernel_height;
            LIMIT(kernel_y_start, 0, input_height)
            LIMIT(kernel_y_end, 0, input_height)
            *p_rec_den++ = (kernel_y_end - kernel_y_start);
        }

        p_den_width = (WORD32 *)((WORD8 *)p_scratch_aligned + ALIGNED_SIZE(sizeof(WORD32)*out_height, ALIGNMENT));
        p_rec_den = (WORD32 *)p_den_width;

        for(i = 0; i < out_width; i++)
        {
            kernel_x_start = i*x_stride - x_padding;
            kernel_x_end = kernel_x_start + kernel_width;
            LIMIT(kernel_x_start, 0, input_width)
            LIMIT(kernel_x_end, 0, input_width)
            *p_rec_den++ = (kernel_x_end - kernel_x_start);
        }

        p_s = (WORD32 *)((WORD8 *)p_den_width + ALIGNED_SIZE(sizeof(WORD32)*out_width, ALIGNMENT));
        p_rec_den = p_s;

        p_zeros = (WORD8 *)((WORD8 *)p_s + ALIGNED_SIZE(sizeof(WORD32)*cw_plane_size, ALIGNMENT));
        p_zeros = (WORD8 *)((WORD8 *)p_zeros + ALIGNED_SIZE(sizeof(WORD32)*input_channels, ALIGNMENT));
        p_zeros_mem = p_zeros;
        zero_mem_bytes = XT_MAX(sizeof(WORD16)*cw_plane_size, sizeof(WORD32)*input_channels);

        for(i = 0; i < zero_mem_bytes; i++)
        {
            *p_zeros++ = 0;
        }

        xa_nn_avgpool_16_hwc_32(p_out
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
                ,p_s
                ,(void *)p_zeros_mem
                ,p_den_height
                ,p_den_width);
    }
#endif

    return 0;
}

