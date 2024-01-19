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
#include "xa_nn_avgpool_state.h"

#define INCR_N_PLANE(ptr, n, plane_size) \
    ptr = (ptr) + ((n) * (plane_size));

#define INCR_PLANE_IF_HEIGHT(ptr, height, plane_size) \
        if(height) \
        { \
            INCR_N_PLANE(ptr, 1, plane_size); \
            height--; \
        }\
        else\
        {\
            ptr = (WORD8 *)p_zeros_mem;\
        }

#define INCR_N_ROW(ptr, n, row_size) \
    ptr = (ptr) + ((n) * (row_size));

#define INCR_ROW_IF_WIDTH_32(ptr, width, row_size) \
        if(width)\
        { \
            INCR_N_ROW(ptr, 1, row_size);\
            width--;\
        }\
        else\
        {\
            ptr = (WORD32 *)p_zeros_mem;\
        }

#define INCR_ROW_IF_WIDTH_16(ptr, width, row_size) \
        if(width)\
        { \
            INCR_N_ROW(ptr, 1, row_size);\
            width--;\
        }\
        else\
        {\
            ptr = (WORD16 *)p_zeros_mem;\
        }




/* Average pooling without using extra copy of input data
 * Works with unaligned input, output.
 */

void xa_nn_avgpool_asym8_hwc_16(
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
      pVOID    p_scratch_in,
      pVOID    p_zeros_mem,
      WORD32   *p_den_height,
      WORD32   *p_den_width)
{
    /* p_scratch_in is always aligned to ALIGNMENT(8)*/
    WORD16 *p_scratch = (WORD16 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int plane_size;
    WORD8 * p_src1, * p_src2, * p_src3;
    WORD16 * p_src1_w, * p_src2_w, * p_src3_w;
    WORD8 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    ae_int16x4 * __restrict p_src1_temp_w, * __restrict p_src2_temp_w, * __restrict p_src3_temp_w;
    ae_int16x4 * p_dst, *p_dst_temp;
    ae_int32x2 * p_dst_temp_w, *p_src1_32x2;
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
        /* Calculating avg of k_h w-c planes and saving into the scratch memory*/

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

                    out = AE_ADD16S(i1, i2);
                    out = AE_ADD16S(out, i3);

                    AE_SA16X4_IP(out, align_dst, p_dst_temp);
                }

                AE_SA64POS_FP(align_dst, p_dst_temp); // finalize the stream

                /* remainder loop for input_width */
                for(i = 0; i < (plane_size & 3); i++)
                {
                    ae_int16x4 i1, i2, i3, out;
                    i1 = AE_MOVDA16(((UWORD8 *)p_src1_temp)[i] );
                    i2 = AE_MOVDA16(((UWORD8 *)p_src2_temp)[i] );
                    i3 = AE_MOVDA16(((UWORD8 *)p_src3_temp)[i] );

                    out = AE_ADD16S(i1, i2);
                    out = AE_ADD16S(out, i3);

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

                        out = AE_ADD16S(i1, i2);
                        out = AE_ADD16S(out, i3);

                        AE_SA16X4_IP(out, align_dst, p_dst_temp);
                    }

                    AE_SA64POS_FP(align_dst, p_dst_temp); // finalize the stream

                    /* remainder loop */
                    for(i = 0; i < (plane_size & 3); i++)
                    {
                        ae_int16x4 i1, i2, i3, out;

                        AE_L16_IP(i1,  (ae_int16 *)p_src1_scratch, 2);
                        i2 = AE_MOVDA16(((UWORD8 *)p_src2_temp)[i] );
                        i3 = AE_MOVDA16(((UWORD8 *)p_src3_temp)[i] );

                        out = AE_ADD16S(i1, i2);
                        out = AE_ADD16S(out, i3);

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
            /* If there is no valid input present, fill the output with zeros */
            p_dst_pad = (WORD16 *)p_scratch;
            for(i = 0; i < plane_size; i++)
            {
                p_dst_pad[i] =  0; //-INFINITY;
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
            p_dst = (ae_int16x4 *)((WORD8 *)p_scratch + ALIGNED_SIZE(sizeof(WORD16)*plane_size, ALIGNMENT));

            if(pool_width)
            {
                p_src1_w = (WORD16 *)p_scratch;
                INCR_N_ROW(p_src1_w, start_row, input_channels);
                pool_width--;

                p_src2_w = p_src1_w;
                INCR_ROW_IF_WIDTH_16(p_src2_w, pool_width, input_channels);

                p_src3_w = p_src2_w;
                INCR_ROW_IF_WIDTH_16(p_src3_w, pool_width, input_channels);

                // 1st instance
                {
                    p_dst_temp_w = (ae_int32x2 *)p_dst;
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
                        ae_int16x4 i1, i2, i3;
                        ae_int16x4 one = AE_MOVDA16(1);
                        ae_int32x2 wout1, wout2;

                        AE_LA16X4_IP(i1, align_src1, p_src1_temp_w);
                        AE_LA16X4_IP(i2, align_src2, p_src2_temp_w);
                        AE_LA16X4_IP(i3, align_src3, p_src3_temp_w);

                        AE_MUL16X4 (wout1, wout2, i1, one);
                        AE_MULA16X4(wout1, wout2, i2, one);
                        AE_MULA16X4(wout1, wout2, i3, one);

                        AE_SA32X2_IP(wout1, align_dst, p_dst_temp_w);
                        AE_SA32X2_IP(wout2, align_dst, p_dst_temp_w);
                    }

                    AE_SA64POS_FP(align_dst, p_dst_temp_w); // finalize the stream

                    /* remainder loop */
                    for(i = 0; i < (input_channels & 3); i++)
                    {
                        ae_int16x4 i1, i2, i3;
                        ae_int32x2 wout1, wout2;
                        ae_int16x4 one = AE_MOVDA16(1);

                        i1 = AE_MOVDA16(((WORD16 *)p_src1_temp_w)[i]);
                        i2 = AE_MOVDA16(((WORD16 *)p_src2_temp_w)[i]);
                        i3 = AE_MOVDA16(((WORD16 *)p_src3_temp_w)[i]);

                        AE_MUL16X4 (wout1, wout2, i1, one);
                        AE_MULA16X4(wout1, wout2, i2, one);
                        AE_MULA16X4(wout1, wout2, i3, one);

                        AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp_w, sizeof(WORD32));
                    }
                }

                if(pool_width)
                {
                    p_src2_w = p_src3_w;
                    INCR_ROW_IF_WIDTH_16(p_src2_w, pool_width, input_channels);

                    p_src3_w = p_src2_w;
                    INCR_ROW_IF_WIDTH_16(p_src3_w, pool_width, input_channels);

                    /* Compare three rows per iteration */
                    do
                    {
                        p_dst_temp_w = (ae_int32x2 *)p_dst;
                        p_src1_32x2 = (ae_int32x2 *)p_dst;
                        p_src2_temp_w = (ae_int16x4 *)p_src2_w;
                        p_src3_temp_w = (ae_int16x4 *)p_src3_w;

                        /* prime */
                        align_src1 = AE_LA64_PP(p_src1_32x2);
                        align_src2 = AE_LA64_PP(p_src2_temp_w);
                        align_src3 = AE_LA64_PP(p_src3_temp_w);
                        align_dst = AE_ZALIGN64(); // zero alignment reg

                        for(i = 0; i < (input_channels >> 2); i++)
                        {
                            ae_int16x4 i2, i3;
                            ae_int16x4 one = AE_MOVDA16(1);
                            ae_int32x2 wout1, wout2;

                            AE_LA32X2_IP(wout1, align_src1, p_src1_32x2);
                            AE_LA32X2_IP(wout2, align_src1, p_src1_32x2);
                            AE_LA16X4_IP(i2, align_src2, p_src2_temp_w);
                            AE_LA16X4_IP(i3, align_src3, p_src3_temp_w);

                            AE_MULA16X4(wout1, wout2, i2, one);
                            AE_MULA16X4(wout1, wout2, i3, one);

                            AE_SA32X2_IP(wout1, align_dst, p_dst_temp_w);
                            AE_SA32X2_IP(wout2, align_dst, p_dst_temp_w);
                        }

                        AE_SA64POS_FP(align_dst, p_dst_temp_w); // finalize the stream

                        /* remainder loop */
                        for(i = 0; i < (input_channels & 3); i++)
                        {
                            ae_int16x4 i2, i3;
                            ae_int16x4 one = AE_MOVDA16(1);
                            ae_int32x2 wout1, wout2;
                            WORD32 *p_w = (WORD32 *)p_src1_32x2;

                            wout1 = AE_MOVDA32(p_w[i]);
                            wout2 = wout1;
                            i2 = ((WORD16 *)p_src2_temp_w)[i];
                            i3 = ((WORD16 *)p_src3_temp_w)[i];

                            AE_MULA16X4(wout1, wout2, i2, one);
                            AE_MULA16X4(wout1, wout2, i3, one);

                            AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp_w, sizeof(WORD32));
                        }


                        if(!pool_width)
                            break;

                        p_src2_w = p_src3_w;
                        INCR_ROW_IF_WIDTH_16(p_src2_w, pool_width, input_channels);

                        p_src3_w = p_src2_w;
                        INCR_ROW_IF_WIDTH_16(p_src3_w, pool_width, input_channels);

                    }while(1);
                }

                // Saving Output
                ae_int32x2 den_h, den_w, d_tmp32, d_out1, d_tmp32hw;
                ae_int64 d_tmp;
                WORD32 *p_out1;

                p_out1 = (WORD32 *)p_dst;

                if(kernel_height * kernel_width <= 1024)
                {
                    d_tmp32hw = AE_MOVDA32(inv_256_tbl[p_den_height[itr_oh] * p_den_width[itr_ow]]);
                }
                else
                {
                    den_h = AE_MOVDA32(inv_256_tbl[p_den_height[itr_oh]]);
                    den_w = AE_MOVDA32(inv_256_tbl[p_den_width[itr_ow]]);
                    d_tmp = AE_MUL32U_LL(den_h, den_w);

                    /* Max value of den_h or den_w is 0x80000000
                       so 1 left shift is possible without overflow */
                    d_tmp32hw = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);
                }

                for(i=0; i<input_channels; i++)
                {
                    d_out1 = AE_MOVDA32(p_out1[i]);
#if XCHAL_HAVE_HIFI1
                    d_tmp32 = AE_MULFP32X2RS_L(d_out1, d_tmp32hw);
                    AE_S8_0_IP_HIFI1(AE_MOVINT16X4_FROMINT32X2(d_tmp32),(WORD8*)p_out_temp, 1);
#else
                    d_tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32hw);
                    p_out_temp[i] = (UWORD8)AE_MOVAD32_L(AE_SRAI32(d_tmp32, 0));
#endif
                }
            }
            else
            {
                /* If there is no valid input present, fill the output with zeros*/
                for(i = 0; i < input_channels; i++)
                {
                    p_out_temp[i] = (UWORD8)0x0;
                }
            }
        }
    }
}

void xa_nn_avgpool_asym8_hwc_32(
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
      pVOID    p_scratch_in,
      pVOID    p_zeros_mem,
      WORD32   *p_den_height,
      WORD32   *p_den_width)
{
    /* p_scratch_in is always aligned to ALIGNMENT(8)*/
    WORD16 *p_scratch = (WORD16 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int plane_size;
    int i;
    WORD8 * p_src1, * p_src2, * p_src3;
    WORD8 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    ae_int32x2 * p_src1_scratch;
    WORD32 * p_src1_w, * p_src2_w, * p_src3_w;
    ae_int32x2 * __restrict p_src1_temp_w, * __restrict p_src2_temp_w, * __restrict p_src3_temp_w;
    ae_int32x2 * p_dst, *p_dst_temp;
    UWORD8 *p_out_temp;
    ae_valign align_src1, align_src2, align_src3, align_dst;
    ALIGN_REGISTER_TYPE i1_la, i2_la, i3_la;

    WORD32 *p_dst_pad;

    plane_size = input_width * input_channels;
    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row, end_row;
        int start_plane, end_plane;


        /* Pool height processing */
        /* Processing width-channel planes for pool_height no. of planes  */
        /* Calculating avg of k_h w-c planes and saving into the scratch memory*/
        start_plane  = itr_oh * y_stride - y_padding;
        end_plane = start_plane + kernel_height;
        LIMIT(start_plane , 0, input_height);
        LIMIT(end_plane , 0, input_height);
        pool_height = end_plane - start_plane;
        p_dst = (ae_int32x2 *)p_scratch ;

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
            /* 1st instance: Add three rows per iteration */
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
                    ae_int16x4 i1, i2, i3;
                    ae_int32x2 wout1, wout2;
                    ae_int16x4 one = AE_MOVDA16(1);

                    AE_LA8X4U_IP(i1, i1_la, p_src1_temp);
                    AE_LA8X4U_IP(i2, i2_la, p_src2_temp);
                    AE_LA8X4U_IP(i3, i3_la, p_src3_temp);

                    AE_MUL16X4 (wout1, wout2, i1, one);
                    AE_MULA16X4(wout1, wout2, i2, one);
                    AE_MULA16X4(wout1, wout2, i3, one);

                    AE_SA32X2_IP(wout1, align_dst, p_dst_temp);
                    AE_SA32X2_IP(wout2, align_dst, p_dst_temp);
                }

                AE_SA64POS_FP(align_dst, p_dst_temp); // finalize the stream

                /* remainder loop */
                for(i = 0; i < (plane_size & 3); i++)
                {
                    ae_int16x4 i1, i2, i3;
                    ae_int32x2 wout1, wout2;
                    ae_int16x4 one = AE_MOVDA16(1);

                    i1 = AE_MOVDA16(((UWORD8 *)p_src1_temp)[i] );
                    i2 = AE_MOVDA16(((UWORD8 *)p_src2_temp)[i] );
                    i3 = AE_MOVDA16(((UWORD8 *)p_src3_temp)[i] );

                    AE_MUL16X4 (wout1, wout2, i1, one);
                    AE_MULA16X4(wout1, wout2, i2, one);
                    AE_MULA16X4(wout1, wout2, i3, one);

                    AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp, sizeof(WORD32));
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

                    align_dst = AE_ZALIGN64(); // zero alignment reg
                    align_src1 = AE_LA64_PP(p_src1_scratch);

                    PRIME_8X4U(p_src2_temp, i2_la);
                    PRIME_8X4U(p_src3_temp, i3_la);

                    for(i = 0; i < (plane_size >> 2); i++)
                    {
                        ae_int16x4 i2, i3;
                        ae_int32x2 wout1, wout2;
                        ae_int16x4 one = AE_MOVDA16(1);

                        AE_LA32X2_IP(wout1, align_src1, p_src1_scratch);
                        AE_LA32X2_IP(wout2, align_src1, p_src1_scratch);
                        AE_LA8X4U_IP(i2, i2_la, p_src2_temp);
                        AE_LA8X4U_IP(i3, i3_la, p_src3_temp);

                        AE_MULA16X4(wout1, wout2, i2, one);
                        AE_MULA16X4(wout1, wout2, i3, one);

                        AE_SA32X2_IP(wout1, align_dst, p_dst_temp);
                        AE_SA32X2_IP(wout2, align_dst, p_dst_temp);
                    }

                    AE_SA64POS_FP(align_dst, p_dst_temp); // finalize the stream

                    /* remainder loop */
                    for(i = 0; i < (plane_size & 3); i++)
                    {
                        ae_int16x4 i2, i3;
                        ae_int32x2 wout1, wout2;
                        ae_int16x4 one = AE_MOVDA16(1);
                        WORD32 *p_w = (WORD32 *)p_src1_scratch;

                        wout1 = AE_MOVDA32(p_w[i]);
                        wout2 = wout1;

                        i2 = AE_MOVDA16(((UWORD8 *)p_src2_temp)[i] );
                        i3 = AE_MOVDA16(((UWORD8 *)p_src3_temp)[i] );

                        AE_MULA16X4(wout1, wout2, i2, one);
                        AE_MULA16X4(wout1, wout2, i3, one);

                        AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp, sizeof(WORD32));
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
            /* If there is no valid input present, fill the output with zeros */
            p_dst_pad = (WORD32 *)p_scratch;
            for(i = 0; i < plane_size; i++)
            {
                p_dst_pad[i] = 0;
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
            p_dst = (ae_int32x2 *)((WORD8 *)p_scratch + ALIGNED_SIZE(sizeof(WORD32)*plane_size, ALIGNMENT));

            if(pool_width)
            {
                p_src1_w = (WORD32 *)p_scratch;
                INCR_N_ROW(p_src1_w, start_row, input_channels);
                pool_width--;

                p_src2_w = p_src1_w;
                INCR_ROW_IF_WIDTH_32(p_src2_w, pool_width, input_channels);

                p_src3_w = p_src2_w;
                INCR_ROW_IF_WIDTH_32(p_src3_w, pool_width, input_channels);

                /* Add three rows per iteration */
                do
                {
                    p_dst_temp = p_dst;
                    p_src1_temp_w = (ae_int32x2 *)p_src1_w;
                    p_src2_temp_w = (ae_int32x2 *)p_src2_w;
                    p_src3_temp_w = (ae_int32x2 *)p_src3_w;

                    /* prime */
                    align_src1 = AE_LA64_PP(p_src1_temp_w);
                    align_src2 = AE_LA64_PP(p_src2_temp_w);
                    align_src3 = AE_LA64_PP(p_src3_temp_w);
                    align_dst = AE_ZALIGN64(); // zero alignment reg

                    for(i = 0; i < (input_channels >> 1); i++)
                    {
                        ae_int32x2 i1, i2, i3, out;

                        AE_LA32X2_IP(i1, align_src1, p_src1_temp_w);
                        AE_LA32X2_IP(i2, align_src2, p_src2_temp_w);
                        AE_LA32X2_IP(i3, align_src3, p_src3_temp_w);

                        out = AE_ADD32S(i1, i2);
                        out = AE_ADD32S(out, i3);

                        AE_SA32X2_IP(out, align_dst, p_dst_temp);
                    }

                    AE_SA64POS_FP(align_dst, p_dst_temp); // finalize the stream

                    /* remainder loop */
                    if(input_channels & 1)
                    {
                        ae_int32x2 i1, i2, i3, out;

                        i1 = AE_MOVDA32(((WORD32 *)p_src1_temp_w)[i]);
                        i2 = AE_MOVDA32(((WORD32 *)p_src2_temp_w)[i]);
                        i3 = AE_MOVDA32(((WORD32 *)p_src3_temp_w)[i]);

                        out = AE_ADD32S(i1, i2);
                        out = AE_ADD32S(out, i3);

                        AE_S32_L_IP(out, (ae_int32 *)p_dst_temp, sizeof(WORD32));
                    }


                    if(!pool_width)
                        break;

                    p_src1_w = (WORD32 *)p_dst;

                    p_src2_w = p_src3_w;
                    INCR_ROW_IF_WIDTH_32(p_src2_w, pool_width, input_channels);

                    p_src3_w = p_src2_w;
                    INCR_ROW_IF_WIDTH_32(p_src3_w, pool_width, input_channels);

                }while(1);

                // Saving Output
                ae_int32x2 den_h, den_w, d_tmp32, d_out1, d_tmp32hw;
                ae_int64 d_tmp;
                WORD32 *p_out1;

                p_out1 = (WORD32 *)p_dst;

                if(kernel_height * kernel_width <= 1024)
                {
                    d_tmp32hw = AE_MOVDA32(inv_256_tbl[p_den_height[itr_oh] * p_den_width[itr_ow]]);
                }
                else
                {
                    den_h = AE_MOVDA32(inv_256_tbl[p_den_height[itr_oh]]);
                    den_w = AE_MOVDA32(inv_256_tbl[p_den_width[itr_ow]]);
                    d_tmp = AE_MUL32U_LL(den_h, den_w);

                    /* Max value of den_h or den_w is 0x80000000
                       so 1 left shift is possible without overflow */
                    d_tmp32hw = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);
                }

                for(i=0; i<input_channels; i++)
                {
                    d_out1 = AE_MOVDA32(p_out1[i]);
                    d_tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32hw);
                    p_out_temp[i] = (UWORD8)AE_MOVAD32_L(AE_SRAI32(d_tmp32, 0));
                }
            }
            else
            {
                /* If there is no valid input present, fill the output with zeros*/
                for(i = 0; i < input_channels; i++)
                {
                    p_out_temp[i] = (UWORD8)0x0;
                }
            }
        }
    }
}


