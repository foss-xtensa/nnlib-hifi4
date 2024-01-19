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
#include "xa_nnlib_common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_avgpool_state.h"
#include "xa_nnlib_err_chk.h"

static void avgpool_asym8_hw(
      UWORD8 *__restrict__ p_out,
const UWORD8 *__restrict__ p_inp,
      WORD32 *p_den_height,
      WORD32 *p_den_width,
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
    WORD32 *p_scratch = (WORD32 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int left_pad_aligned, right_pad, total_out_width, scratch_width;
    int pre_loop_count, loop_count, post_loop_count;
    WORD8 * p_src1, * p_src2;
    WORD8 * __restrict p_src1_temp, * __restrict p_src2_temp;
    ae_int32x2 * p_wsrc1, * p_wsrc2;
    ae_int32x2 * __restrict p_wsrc1_temp, * __restrict p_wsrc2_temp;
    ae_int32x2 *p_dst, *p_dst_temp;
    ae_valign align_wsrc1, align_wsrc2;
    ae_valign align_dst;
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
            p_src1 = (WORD8 *)p_inp;
            p_src1 = p_src1 + start_row*input_width;
            pool_height--;
            p_dst_temp = p_dst;
            p_src1_temp = p_src1;
            pre_loop_count = (int)((unsigned)ALIGN_PTR(p_src1_temp, 4) - (unsigned)p_src1_temp);
            loop_count = (input_width - pre_loop_count)>>2;
            post_loop_count = input_width - pre_loop_count - (loop_count<<2);

            ae_int32 *p_dst_temp32 = (ae_int32 *)p_dst_temp;
#pragma no_unroll
            for(i = 0; i < pre_loop_count; i++)
            {
                ae_int16x4 i1;
                ae_int32x2 wi1;
                i1 = AE_MOVDA16((UWORD8)*p_src1_temp++);
                wi1 = AE_SEXT32X2D16_32(i1);
                *p_dst_temp32++ = wi1;
            }

            p_dst_temp = (ae_int32x2 *)p_dst_temp32;
            align_dst = AE_ZALIGN64();
            for(i = 0; i < loop_count; i++)
            {
                ae_int16x4 i1;
                ae_int32x2 wi1, wi2;
#if XCHAL_HAVE_HIFI1
                AE_L8X4U_IP(i1, p_src1_temp, 4);
                wi1 = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_7362(AE_ZERO16(), i1));
                //wi1 = AE_SEXT32X2D16_32(i1);
                wi2 = AE_SEXT32X2D16_10(i1);
#else
                AE_L8X4F_IP(i1, p_src1_temp, 4);
                i1 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(i1), 8));
                wi1 = AE_SEXT32X2D16_32(i1);
                wi2 = AE_SEXT32X2D16_10(i1);
#endif
                AE_SA32X2_IP(wi1, align_dst, p_dst_temp);
                AE_SA32X2_IP(wi2, align_dst, p_dst_temp);
            }
            AE_SA64POS_FP(align_dst, p_dst_temp);

            /* reminder loop for input_width */
#pragma no_unroll
            for(i = 0; i < post_loop_count; i++)
            {
                ae_int16x4 i1;
                ae_int32x2 wi1;
                i1 = AE_MOVDA16((UWORD8)p_src1_temp[i]);
                wi1 = AE_SEXT32X2D16_32(i1);
                ((ae_int32 *)p_dst_temp)[i] = wi1;
            }

            p_src2 = p_src1;
            p_wsrc1 = p_dst;
            /* Compare three rows per iteration */
            while(pool_height)
            {
                p_src2 = p_src2 + input_width;
                p_dst_temp = p_dst;
                p_wsrc1_temp = p_wsrc1;
                p_src2_temp = p_src2;

                pre_loop_count = (int)((unsigned)ALIGN_PTR(p_src2_temp, 4) - (unsigned)p_src2_temp);
                loop_count = (input_width - pre_loop_count)>>2;
                post_loop_count = input_width - pre_loop_count - (loop_count<<2);

                ae_int32 *p_wsrc1_temp32 = (ae_int32 *)p_wsrc1_temp;
                p_dst_temp32 = (ae_int32 *)p_dst_temp;
#pragma no_unroll
                for(i = 0; i < pre_loop_count; i++)
                {
                    ae_int16x4 i1, shift = AE_MOVDA16(1);
                    ae_int32x2 wi1, wi2=0;

                    wi1 = *p_wsrc1_temp32++;
                    i1 = AE_MOVDA16((UWORD8)*p_src2_temp++);
                    AE_MULA16X4(wi1, wi2, i1, shift);
                    *p_dst_temp32++ = wi1;
                }

                p_wsrc1_temp = (ae_int32x2 *)p_wsrc1_temp32;
                p_dst_temp = (ae_int32x2 *)p_dst_temp32;
                /* prime */
                align_wsrc1 = AE_LA64_PP(p_wsrc1_temp);
                align_dst = AE_ZALIGN64();
                for(i = 0; i < loop_count; i++)
                {
                    ae_int16x4 i1, one = AE_MOVDA16(1);
                    ae_int32x2 wi1, wi2;

                    AE_LA32X2_IP(wi1, align_wsrc1, p_wsrc1_temp);
                    AE_LA32X2_IP(wi2, align_wsrc1, p_wsrc1_temp);
#if XCHAL_HAVE_HIFI1
                    AE_L8X4U_IP(i1, p_src2_temp, 4);
#else
                    AE_L8X4F_IP(i1, p_src2_temp, 4);
                    i1 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(i1), 8));
#endif
                    AE_MULA16X4(wi1, wi2, i1, one);
                    AE_SA32X2_IP(wi1, align_dst, p_dst_temp);
                    AE_SA32X2_IP(wi2, align_dst, p_dst_temp);
                }
                AE_SA64POS_FP(align_dst, p_dst_temp);

                /* reminder loop for input_width */
#pragma no_unroll
                for(i = 0; i < post_loop_count; i++)
                {
                    ae_int16x4 i1, shift = AE_MOVDA16(1);
                    ae_int32x2 wi1, wi2=0;

                    wi1 = ((ae_int32 *)p_wsrc1_temp)[i];
                    i1 = AE_MOVDA16((UWORD8)p_src2_temp[i]);
                    AE_MULA16X4(wi1, wi2, i1, shift);
                    ((ae_int32 *)p_dst_temp)[i] = wi1;
                }
                pool_height--;
            };
        }
        else
        {
            /* If there is no valid input present, fill the output with 0 */
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
                AE_S8_0_I_HIFI1(AE_MOVINT16X4_FROMINT32X2(d_tmp32), (WORD8*)(p_out+(itr_oh*out_width)+itr_ow), 0);
                #else
                d_tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32);
                p_out[itr_oh*out_width+itr_ow] = (UWORD8)AE_MOVAD32_L(AE_SRAI32(d_tmp32, 0));
                #endif
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
                #if XCHAL_HAVE_HIFI1
                d_tmp32 = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);
                d_tmp32 = AE_MULFP32X2RS_L(d_out1, d_tmp32);
                AE_S8_0_I_HIFI1(AE_MOVINT16X4_FROMINT32X2(d_tmp32), (WORD8*)(p_out+(itr_oh*out_width)+itr_ow), 0);
                #else
                d_tmp32 = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);
                d_tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32);
                p_out[itr_oh*out_width+itr_ow] = (UWORD8)AE_MOVAD32_L(AE_SRAI32(d_tmp32, 0));
                #endif
            }
        }
    }
}

WORD32 xa_nn_avgpool_asym8(
      UWORD8* __restrict__ p_out,
const UWORD8* __restrict__ p_inp,
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
      WORD32  inp_data_format,
      WORD32  out_data_format,
      VOID    *p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND((kernel_height > 256), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_width > 256), -1);

    XA_NNLIB_ARG_CHK_COND((out_data_format != 0) && (out_data_format != 1), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0) && (inp_data_format != 1), -1);

    // Different I/O data formats (not supported!)
    XA_NNLIB_ARG_CHK_COND((out_data_format != inp_data_format), -1);

    if((input_channels == 1) || (out_data_format == 1))
    {
        xa_nn_avgpool_init(-3,
                p_scratch,
                out_height,
                out_width);

        xa_nn_avgpool_state_t *p_state = (xa_nn_avgpool_state_t *)p_scratch;
        int itr_ic, itr_oh, itr_ow;
        const UWORD8 *pt_inp; 
        UWORD8 *pt_out;
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

            avgpool_asym8_hw(pt_out
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

        if(kernel_height <= (int)MAX_HEIGHT_16_BIT_ACC)
        {
            p_zeros = (WORD8 *)((WORD8 *)p_s + ALIGNED_SIZE(sizeof(WORD16)*cw_plane_size, ALIGNMENT));
            p_zeros = (WORD8 *)((WORD8 *)p_zeros + ALIGNED_SIZE(sizeof(WORD32)*input_channels, ALIGNMENT));
            p_zeros_mem = p_zeros;
            zero_mem_bytes = XT_MAX(sizeof(UWORD8)*cw_plane_size, sizeof(WORD16)*input_channels);
        }
        else
        {
            p_zeros = (WORD8 *)((WORD8 *)p_s + ALIGNED_SIZE(sizeof(WORD32)*cw_plane_size, ALIGNMENT));
            p_zeros = (WORD8 *)((WORD8 *)p_zeros + ALIGNED_SIZE(sizeof(WORD32)*input_channels, ALIGNMENT));
            p_zeros_mem = p_zeros;
            zero_mem_bytes = XT_MAX(sizeof(UWORD8)*cw_plane_size, sizeof(WORD32)*input_channels);
        }

        for(i = 0; i < zero_mem_bytes; i++)
        {
            *p_zeros++ = 0;
        }

        if(kernel_height <= (int)MAX_HEIGHT_16_BIT_ACC)
        {
            xa_nn_avgpool_asym8_hwc_16(p_out
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
        else
        {
            xa_nn_avgpool_asym8_hwc_32(p_out
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
    }

    return 0;
}


