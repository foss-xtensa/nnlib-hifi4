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
#include "xa_nnlib_common.h"
#include "xa_nn_maxpool_state.h"

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

#if XCHAL_HAVE_HIFI1

#define MAX_16X4(out, id2, id1, id0) {\
        out = AE_MAX16(id1, id0);\
        out = AE_MAX16(id2, out);\
}

#else

#define MAX_16X4(out, id2, id1, id0) {\
        out = id1;\
        b0 = AE_LT16(id1, id0); \
        AE_MOVT16X4(out, id0, b0);\
        b0 = AE_LT16(out, id2); \
        AE_MOVT16X4(out, id2, b0);\
}

#endif




/* Max pooling without using extra copy of input data
 * Works with unaligned input, output.
 */

void xa_nn_maxpool_8_hwc(
      WORD8* __restrict__ p_out,
const WORD8* __restrict__ p_inp,
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
    /* aligning to the next 8-byte boundary to be able to use AE_[L,S]16X4 */
    WORD16 *p_scratch = (WORD16 *)ALIGN_PTR(p_scratch_in, 8);

    int itr_oh, itr_ow;
    const int plane_size = input_width * input_channels;
#if !XCHAL_HAVE_HIFI1
    xtbool4 b0;
#endif
    WORD8 * p_src1, * p_src2, * p_src3;
    WORD16 * p_src1_w, * p_src2_w, * p_src3_w;
    WORD8 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    ae_int16x4 * __restrict p_src1_temp_w, * __restrict p_src2_temp_w, * __restrict p_src3_temp_w;
    ae_int16x4 * __restrict__ p_dst, * __restrict__ p_dst_temp;
    WORD8 *p_out_temp;
    ae_int16x4 * p_src1_scratch;
    ae_valign align_src1, align_src2, align_src3/*, align_dst*/;
    ALIGN_REGISTER_TYPE i1_la, i2_la, i3_la;
    int i;
    WORD16 *p_dst_pad;

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

            //align_dst = AE_ZALIGN64(); // zero alignment reg
            /* 1st instance: Compare three rows per iteration */
            {
                p_dst_temp = p_dst;
                p_src1_temp = p_src1;
                p_src2_temp = p_src2;
                p_src3_temp = p_src3;

                PRIME_8X4F(p_src1_temp, i1_la);
                PRIME_8X4F(p_src2_temp, i2_la);
                PRIME_8X4F(p_src3_temp, i3_la);

                for(i = 0; i < (plane_size >> 2); i++)
                {
                    ae_int16x4 i1, i2, i3, out;

                    AE_LA8X4F_IP(i1, i1_la, p_src1_temp);
                    AE_LA8X4F_IP(i2, i2_la, p_src2_temp);
                    AE_LA8X4F_IP(i3, i3_la, p_src3_temp);

                    MAX_16X4(out, i3, i2, i1)
                    AE_S16X4_IP(out, p_dst_temp, 8);
                }

                /* remainder loop */
                for(i = 0; i < (plane_size & 3); i++)
                {
                    ae_int16x4 i1, i2, i3, out;
                    i1 = AE_MOVDA16(((WORD8 *)p_src1_temp)[i] );
                    i1 = AE_SLAI16S(i1, 8);

                    i2 = AE_MOVDA16(((WORD8 *)p_src2_temp)[i] );
                    i2 = AE_SLAI16S(i2, 8);

                    i3 = AE_MOVDA16(((WORD8 *)p_src3_temp)[i] );
                    i3 = AE_SLAI16S(i3, 8);

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

                    PRIME_8X4F(p_src2_temp, i2_la);
                    PRIME_8X4F(p_src3_temp, i3_la);

                    for(i = 0; i < (plane_size >> 2); i++)
                    {
                        ae_int16x4 i1, i2, i3, out;

                        AE_L16X4_IP(i1, p_src1_scratch, 8);
                        AE_LA8X4F_IP(i2, i2_la, p_src2_temp);
                        AE_LA8X4F_IP(i3, i3_la, p_src3_temp);

                        MAX_16X4(out, i3, i2, i1)
                        AE_S16X4_IP(out, p_dst_temp, 8);
                    }

                    /* remainder loop */
                    for(i = 0; i < (plane_size & 3); i++)
                    {
                        ae_int16x4 i1, i2, i3, out;

                        AE_L16_IP(i1,  (ae_int16 *)p_src1_scratch, 2);

                        i2 = AE_MOVDA16(((WORD8 *)p_src2_temp)[i] );
                        i2 = AE_SLAI16S(i2, 8);

                        i3 = AE_MOVDA16(((WORD8 *)p_src3_temp)[i] );
                        i3 = AE_SLAI16S(i3, 8);

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
                p_dst_pad[i] =  (WORD16)0x8000;
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
            p_dst = (ae_int16x4 *)ALIGN_PTR((WORD16 *)p_scratch + plane_size, ALIGNMENT);

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

                    for(i = 0; i < (input_channels >> 2); i++)
                    {
                        ae_int16x4 i1, i2, i3, out;

                        AE_LA16X4_IP(i1, align_src1, p_src1_temp_w);
                        AE_LA16X4_IP(i2, align_src2, p_src2_temp_w);
                        AE_LA16X4_IP(i3, align_src3, p_src3_temp_w);

                        MAX_16X4(out, i3, i2, i1)

                        AE_S16X4_IP(out, p_dst_temp, 8);
                    }

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
                    p_out_temp[i] = (WORD8)(p_dst_pad[i] >> 8);
                }
            }
            else
            {
                /* If there is no valid input present, fill the output with min_value */
                for(i = 0; i < input_channels; i++)
                {
                    p_out_temp[i] = (WORD8)0x80;
                }
            }
        }
    }
}


