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
#include "xa_type_def.h"
#include "common.h"
#include "xa_nnlib_err_chk.h"

#define ALIGNMENT   8   /* 8 bytes alignment */

#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#define MAX_16X4(id1, id0) \
        b0 = AE_LT16(id1, id0); \
        AE_MOVT16X4(id1, id0, b0);

#define MIN_16X4(id1, id0) \
        b0 = AE_LT16(id1, id0); \
        AE_MOVF16X4(id1, id0, b0);

#define LIMIT(out, inp, min, max){\
        out = min;\
        MAX_16X4(out, inp);\
        MIN_16X4(out, max);\
}

#define STORE_8X4_FROM_16X4(out_ptr, val){\
    int o1, o2, o3, o4;\
    o1 = AE_MOVAD16_3(val);\
    o2 = AE_MOVAD16_2(val);\
    o3 = AE_MOVAD16_1(val);\
    o4 = AE_MOVAD16_0(val);\
    *out_ptr++ = (WORD8)o1;\
    *out_ptr++ = (WORD8)o2;\
    *out_ptr++ = (WORD8)o3;\
    *out_ptr++ = (WORD8)o4;\
}

#define MAX_WORD16 (int)0x00007fff
#define MIN_WORD16 (int)0xffff8000

/*
 * inp: p_vec: 2 byte aligned input pointer
 * out: p_out: 2 byte aligned input pointer */
WORD32 xa_nn_vec_activation_min_max_16_16(WORD16 * __restrict__ p_out,
                                      const  WORD16 * __restrict__ p_vec,
                                      int    activation_min,
                                      int    activation_max,
                                      WORD32 vec_length)
{
    int i;
    ae_int16x4 x, y, min, max;
    ae_valign align_src, align_dst;
    xtbool4 b0;

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

    WORD16 *p_o = p_out;
    WORD16 *p_v = (WORD16 *)p_vec;

    min  = AE_MOVDA16(activation_min);
    max  = AE_MOVDA16(activation_max);

    align_src = AE_LA64_PP((ae_int16x4 *)p_v);
    align_dst = AE_ZALIGN64(); // zero alignment reg

    if((activation_max >= MAX_WORD16) && (activation_min <= MIN_WORD16))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(x, align_src, (ae_int16x4 *)p_v);
            AE_SA16X4_IP(x, align_dst, (ae_int16x4 *)p_o);
        }

        AE_SA64POS_FP(align_dst, p_o); // finalize the stream

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L16_IP(x, (ae_int16 *)p_v, sizeof(ae_int16));
            AE_S16_0_IP(x, (ae_int16 *)p_o, sizeof(ae_int16));
        }
    }
    else if((activation_max < MAX_WORD16) && (activation_min <= MIN_WORD16))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(x, align_src, (ae_int16x4 *)p_v);

            b0 = AE_LT16(x, max);
            AE_MOVF16X4(x, max, b0);

            AE_SA16X4_IP(x, align_dst, (ae_int16x4 *)p_o);
        }

        AE_SA64POS_FP(align_dst, p_o); // finalize the stream

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L16_IP(y, (ae_int16 *)p_v, sizeof(ae_int16));

            b0 = AE_LT16(y, max);
            AE_MOVF16X4(y, max, b0);

            AE_S16_0_IP(y, (ae_int16 *)p_o, sizeof(ae_int16));
        }
    }
    else if((activation_max >= MAX_WORD16) && (activation_min > MIN_WORD16))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(y, align_src, (ae_int16x4 *)p_v);

            b0 = AE_LT16(y, min);
            AE_MOVT16X4(y, min, b0);

            AE_SA16X4_IP(y, align_dst, (ae_int16x4 *)p_o);
        }

        AE_SA64POS_FP(align_dst, p_o); // finalize the stream

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L16_IP(y, (ae_int16 *)p_v, sizeof(ae_int16));

            b0 = AE_LT16(y, min);
            AE_MOVT16X4(y, min, b0);

            AE_S16_0_IP(y, (ae_int16 *)p_o, sizeof(ae_int16));
        }
    }
    else
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(x, align_src, (ae_int16x4 *)p_v);
            LIMIT(y, x, min, max)
            AE_SA16X4_IP(y, align_dst, (ae_int16x4 *)p_o);
        }

        AE_SA64POS_FP(align_dst, p_o); // finalize the stream

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L16_IP(x, (ae_int16 *)p_v, sizeof(ae_int16));
            LIMIT(y, x, min, max)
            AE_S16_0_IP(y, (ae_int16 *)p_o, sizeof(ae_int16));
        }
    }

    return 0;
}

/*
 * ReLU 16-bit:
 */
WORD32 xa_nn_vec_relu_16_16(
    WORD16       * __restrict__ p_out,
    const WORD16 * __restrict__ p_vec,
    WORD16       threshold,
    WORD32       vec_length)
{
    xa_nn_vec_activation_min_max_16_16(p_out,
                                      p_vec,
                                      0,
                                      threshold,
                                      vec_length);

    return 0;
}
/*
 * ReLU Standard 16-bit:
 */
WORD32 xa_nn_vec_relu_std_16_16(
    WORD16       * __restrict__ p_out,
    const WORD16 * __restrict__ p_vec,
    WORD32       vec_length)
{

    xa_nn_vec_activation_min_max_16_16(p_out,
                                      p_vec,
                                      0,
                                      MAX_WORD16,
                                      vec_length);
	return 0;
}
