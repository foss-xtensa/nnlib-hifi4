/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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
#include "xa_nn_maxpool_state.h"
#include <math.h>

#define STORE_8X4_FROM_16X4(out_ptr, val){\
    int o1, o2, o3, o4;\
    o1 = AE_MOVAD16_3(val);\
    o2 = AE_MOVAD16_2(val);\
    o3 = AE_MOVAD16_1(val);\
    o4 = AE_MOVAD16_0(val);\
    *out_ptr++ = (UWORD8)o1;\
    *out_ptr++ = (UWORD8)o2;\
    *out_ptr++ = (UWORD8)o3;\
    *out_ptr++ = (UWORD8)o4;\
}

#define MAX_16X4(out, id2, id1, id0) {\
        out = id1;\
        b0 = AE_LT16(id1, id0); \
        AE_MOVT16X4(out, id0, b0);\
        b0 = AE_LT16(out, id2); \
        AE_MOVT16X4(out, id2, b0);\
}

#define INCR_N_PLANE_1(ptr, n, plane_size) \
    ptr = (ptr) + ((n) * (plane_size));

#define INCR_N_PLANE(ptr, n, plane_size) \
    ptr = (ptr) + ((n) * (plane_size));

#define INCR_PLANE_IF_HEIGHT(ptr, height, plane_size) \
        if(height) \
        { \
            INCR_N_PLANE(ptr, 1, plane_size); \
            height--; \
        }

#define INCR_N_ROW(ptr, n, row_size) \
    ptr = (ptr) + ((n) * (row_size));

#define INCR_ROW_IF_WIDTH(ptr, width, row_size) \
        if(width) \
        { \
            INCR_N_ROW(ptr, 1, row_size); \
            width--; \
        }

/* Max pooling without using extra copy of input data
 * Works with unaligned input, output.
 */

void xa_nn_maxpool_asym8_hwc(
      UWORD8* __restrict__ p_out,
const UWORD8* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in)
{
    WORD16 *p_scratch = (WORD16 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int plane_size;
    xtbool4 b0;
    WORD8 * p_src1, * p_src2, * p_src3;
    WORD16 * p_src1_w, * p_src2_w, * p_src3_w;
    WORD8 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    ae_int16x4 * __restrict p_src1_temp_w, * __restrict p_src2_temp_w, * __restrict p_src3_temp_w;
    ae_int16x4 * p_dst, *p_dst_temp;
    UWORD8 *p_out_temp;
    ae_int16x4 * p_src1_scratch;
    ae_valign align_src1, align_src2, align_src3, align_dst;

    ALIGN_REGISTER_TYPE i1_la, i2_la, i3_la;

    int i;
    WORD16 *p_dst_pad;

    plane_size = input_width * input_channels;
    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row, end_row;
        int start_plane, end_plane;


        /* Pool height processing */
        /* Processing width-channel planes for pool_height no. of planes  */
        /* Calculating max of k_h w-c planes and saving into the scratch memory*/
        /* Compare the input w-c planes (width-channel planes) for the required pooling height and store to the scratch */
        start_plane  = itr_oh * y_stride - y_padding;
        end_plane = start_plane + kernel_height;
        LIMIT(start_plane , 0, input_height);
        LIMIT(end_plane , 0, input_height);
        pool_height = end_plane - start_plane;
        p_dst = (ae_int16x4 *)p_scratch ;

        if(pool_height)
        {
            p_src1 = (WORD8 *)p_inp;
            INCR_N_PLANE(p_src1, start_plane, plane_size);
            pool_height--;

            p_src2 = p_src1;
            INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

            p_src3 = p_src2;
            INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

            align_dst = AE_ZALIGN64(); // zero alignment reg
            /* 1st instance: Compare three rows per iteration */
            {
                p_dst_temp = p_dst;
                p_src1_temp = p_src1;
                p_src2_temp = p_src2;
                p_src3_temp = p_src3;

                PRIME_8X4U(p_src1_temp, i1_la);
                PRIME_8X4U(p_src2_temp, i2_la);
                PRIME_8X4U(p_src3_temp, i3_la);

                for(i = 0; i < (plane_size >> 2); i++)
                {
                    ae_int16x4 i1, i2, i3, out;

                    AE_LA8X4U_IP(i1, i1_la, p_src1_temp);
                    AE_LA8X4U_IP(i2, i2_la, p_src2_temp);
                    AE_LA8X4U_IP(i3, i3_la, p_src3_temp);

                    MAX_16X4(out, i3, i2, i1)
                    AE_SA16X4_IP(out, align_dst, p_dst_temp);
                }

                AE_SA64POS_FP(align_dst, p_dst_temp); // finalize the stream

                /* remainder loop */
                for(i = 0; i < (plane_size & 3); i++)
                {
                    ae_int16x4 i1, i2, i3, out;
                    i1 = AE_MOVDA16(((UWORD8 *)p_src1_temp)[i] );
                    i2 = AE_MOVDA16(((UWORD8 *)p_src2_temp)[i] );
                    i3 = AE_MOVDA16(((UWORD8 *)p_src3_temp)[i] );

                    MAX_16X4(out, i3, i2, i1)
                    AE_S16_0_IP(out, (ae_int16 *)p_dst_temp, 2);
                }
            }

            if(pool_height)
            {
                p_src2 = p_src3;
                INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

                p_src3 = p_src2;
                INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

                do
                {
                    p_dst_temp = p_dst;
                    p_src1_scratch = p_dst;
                    p_src2_temp = p_src2;
                    p_src3_temp = p_src3;

                    PRIME_8X4U(p_src2_temp, i2_la);
                    PRIME_8X4U(p_src3_temp, i3_la);

                    align_dst = AE_ZALIGN64(); // zero alignment reg
                    align_src1 = AE_LA64_PP(p_src1_scratch);

                    for(i = 0; i < (plane_size >> 2); i++)
                    {
                        ae_int16x4 i1, i2, i3, out;

                        AE_LA16X4_IP(i1, align_src1, p_src1_scratch);
                        AE_LA8X4U_IP(i2, i2_la, p_src2_temp);
                        AE_LA8X4U_IP(i3, i3_la, p_src3_temp);

                        MAX_16X4(out, i3, i2, i1)
                        AE_SA16X4_IP(out, align_dst, p_dst_temp);
                    }

                    AE_SA64POS_FP(align_dst, p_dst_temp); // finalize the stream

                    /* remainder loop */
                    for(i = 0; i < (plane_size & 3); i++)
                    {
                        ae_int16x4 i1, i2, i3, out;

                        AE_L16_IP(i1,  (ae_int16 *)p_src1_scratch, 2);
                        i2 = AE_MOVDA16(((UWORD8 *)p_src2_temp)[i] );
                        i3 = AE_MOVDA16(((UWORD8 *)p_src2_temp)[i] );

                        MAX_16X4(out, i3, i2, i1)
                        AE_S16_0_IP(out, (ae_int16 *)p_dst_temp, 2);
                    }

                    if(!pool_height)
                        break;

                    p_src2 = p_src3;
                    INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

                    p_src3 = p_src2;
                    INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

                }while(1);
            }
        }
        else
        {
            /* If there is no valid input present, fill the output with min_value */
            p_dst_pad = (WORD16 *)p_scratch;
            for(i = 0; i < plane_size; i++)
            {
                p_dst_pad[i] =  0;
            }
        }

        /* Pool width processing */
        /* Processing the output of the height processing block (which is a w-c plane); along width */
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            start_row  = itr_ow * x_stride - x_padding;
            end_row = start_row + kernel_width;
            LIMIT(start_row , 0, input_width);
            LIMIT(end_row , 0, input_width);
            pool_width = end_row - start_row;
            p_out_temp = p_out + (itr_oh*out_width*input_channels) + (itr_ow*input_channels);
            p_dst = (ae_int16x4 *)((WORD16 *)p_scratch + plane_size);
            p_dst_temp = p_dst;

            if(pool_width)
            {
                p_src1_w = (WORD16 *)p_scratch;
                INCR_N_ROW(p_src1_w, start_row, input_channels);
                pool_width--;

                p_src2_w = p_src1_w;
                INCR_ROW_IF_WIDTH(p_src2_w, pool_width, input_channels);

                p_src3_w = p_src2_w;
                INCR_ROW_IF_WIDTH(p_src3_w, pool_width, input_channels);

                /* Compare three rows per iteration */
                do
                {
                    p_dst_temp = p_dst;
                    p_src1_temp_w = (ae_int16x4 *)p_src1_w;
                    p_src2_temp_w = (ae_int16x4 *)p_src2_w;
                    p_src3_temp_w = (ae_int16x4 *)p_src3_w;

                    /* prime */
                    align_src1 = AE_LA64_PP(p_src1_temp_w);
                    align_src2 = AE_LA64_PP(p_src2_temp_w);
                    align_src3 = AE_LA64_PP(p_src3_temp_w);
                    align_dst = AE_ZALIGN64(); // zero alignment reg

                    for(i = 0; i < (input_channels >> 2); i++)
                    {
                        ae_int16x4 i1, i2, i3, out;

                        AE_LA16X4_IP(i1, align_src1, p_src1_temp_w);
                        AE_LA16X4_IP(i2, align_src2, p_src2_temp_w);
                        AE_LA16X4_IP(i3, align_src3, p_src3_temp_w);

                        MAX_16X4(out, i3, i2, i1)

                        AE_SA16X4_IP(out, align_dst, p_dst_temp);
                    }

                    AE_SA64POS_FP(align_dst, p_dst_temp); // finalize the stream

                    /* remainder loop */
                    for(i = 0; i < (input_channels & 3); i++)
                    {
                        ae_int16x4 i1, i2, i3, out;

                        i1 = ((WORD16 *)p_src1_temp_w)[i];
                        i2 = ((WORD16 *)p_src2_temp_w)[i];
                        i3 = ((WORD16 *)p_src3_temp_w)[i];

                        MAX_16X4(out, i3, i2, i1)

                        AE_S16_0_IP(out, (ae_int16 *)p_dst_temp, 2);
                    }


                    if(!pool_width)
                        break;

                    p_src1_w = (WORD16 *)p_dst;

                    p_src2_w = p_src3_w;
                    INCR_ROW_IF_WIDTH(p_src2_w, pool_width, input_channels);

                    p_src3_w = p_src2_w;
                    INCR_ROW_IF_WIDTH(p_src3_w, pool_width, input_channels);

                }while(1);

                // Saving Output
                p_dst_pad = (WORD16 *)p_dst;
                for(i=0; i<input_channels; i++)
                {
                    p_out_temp[i] = (UWORD8)p_dst_pad[i];
                }
            }
            else
            {
                /* If there is no valid input present, fill the output with min_value */
                for(i = 0; i < input_channels; i++)
                {
                    p_out_temp[i] = (UWORD8)0x0;
                }
            }
        }
    }
}


