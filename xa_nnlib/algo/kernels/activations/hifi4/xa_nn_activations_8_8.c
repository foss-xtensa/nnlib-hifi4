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
#include "xa_nnlib_err_chk.h"

#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#if XCHAL_HAVE_HIFI1

#define MAX_16X4(id1, id0) \
        id1 = AE_MAX16(id1, id0);\

#define MIN_16X4(id1, id0) \
        id1 = AE_MIN16(id1, id0);\

#define LIMIT(out, inp, min, max){\
        out = AE_MAX16(min, inp);\
        out = AE_MIN16(out, max);\
}

#define STORE_8X4_FROM_16X4(out_ptr, val){\
    AE_S8_0_IP_HIFI1(AE_SEL16_6543(val, val), out_ptr, sizeof(WORD8));\
    AE_S8_0_IP_HIFI1(AE_SEL16_5432(val, val), out_ptr, sizeof(WORD8));\
    AE_S8_0_IP_HIFI1(AE_SEL16_4321(val, val), out_ptr, sizeof(WORD8));\
    AE_S8_0_IP_HIFI1(val, out_ptr, sizeof(WORD8));\
}

#else

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

#endif //XCHAL_HAVE_HIFI1
#define MAX_WORD8 127
#define MIN_WORD8 -128

/*
 * inp: p_vec: 1 byte aligned input pointer
 * out: p_out: no alignment needed for output pointer*/
#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
WORD32 xa_nn_vec_activation_min_max_8_8(WORD8 * __restrict__ p_out,
                                      const  WORD8 * __restrict__ p_vec,
                                      int    activation_min,
                                      int    activation_max,
                                      WORD32 vec_length)
{
    int i;
    ae_int16x4 x, y, min, max;
    ae_valign align_in, align_out = AE_ZALIGN64();
    //xtbool4 b0;

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

    WORD8 *p_o = p_out;
    WORD8 *p_v = (WORD8 *)p_vec;

    min  = AE_MOVDA16(activation_min);
    max  = AE_MOVDA16(activation_max);

    // pre loop, active when input ptr is not 4 byte aligned
    int pre_loop_count=0;
    pre_loop_count = (int)((unsigned)ALIGN_PTR(p_v, 4) - (unsigned)p_v);
    pre_loop_count = (pre_loop_count < vec_length) ? pre_loop_count : vec_length;

    vec_length = vec_length - pre_loop_count;
    vec_length = (vec_length < 0) ? 0 : vec_length;

    for(i=0; i<pre_loop_count; i++)
    {
        AE_L8S_IP(x, p_v, 1);
        LIMIT(y, x, min, max)
        AE_S8_0_IP_HIFI1(y, p_o, 1);
    }
		
	align_in = AE_LA64_PP(p_v); 
    if((activation_max >= (int)MAX_WORD8) && (activation_min <= (int)MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA8X4S_IP(x, align_in, p_v);
            AE_SA8X4U_IP(x, align_out, (ae_int32*)p_o);
        }
		int rem_length = (vec_length & 3);
        {
            AE_LAV8X4S_XP(x, align_in, (ae_int8x4 *)p_v, rem_length);
            AE_SAV8X4U_XP(x, align_out, (ae_int8x4u *)p_o, rem_length);
        }
		AE_SA64POS_FP(align_out, p_o);

    }
    else if((activation_max < (int)MAX_WORD8) && (activation_min <= MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA8X4S_IP(x, align_in, p_v);
			
            x = AE_MIN16(x, max);
			
            AE_SA8X4U_IP(x, align_out, (ae_int32*)p_o);
        }

		int rem_length = (vec_length & 3);
        {
            AE_LAV8X4S_XP(x, align_in, (ae_int8x4 *)p_v, rem_length);
			
            x = AE_MIN16(x, max);
			
            AE_SAV8X4U_XP(x, align_out, (ae_int8x4u *)p_o, rem_length);
        }
        AE_SA64POS_FP(align_out, p_o);
    }
    else if((activation_max >= (int)MAX_WORD8) && (activation_min > MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA8X4S_IP(x, align_in, p_v);
			
            x = AE_MAX16(x, min);
			
            AE_SA8X4U_IP(x, align_out, (ae_int32*)p_o);
        }

		int rem_length = (vec_length & 3);
        {
            AE_LAV8X4S_XP(x, align_in, (ae_int8x4 *)p_v, rem_length);
			
            x = AE_MAX16(x, min);
			
            AE_SAV8X4U_XP(x, align_out, (ae_int8x4u *)p_o, rem_length);
        }
        AE_SA64POS_FP(align_out, p_o);
    }
    else
    {		
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA8X4S_IP(x, align_in, p_v);
			
            LIMIT(y, x, min, max) 
			
            AE_SA8X4U_IP(y, align_out, (ae_int32*)p_o);
        }

		int rem_length = (vec_length & 3);
        {
            AE_LAV8X4S_XP(x, align_in, (ae_int8x4 *)p_v, rem_length);
			
            LIMIT(y, x, min, max)
			
            AE_SAV8X4U_XP(y, align_out, (ae_int8x4u *)p_o, rem_length);
        }
        AE_SA64POS_FP(align_out, p_o);		
    }

    return 0;
}
#else
WORD32 xa_nn_vec_activation_min_max_8_8(WORD8 * __restrict__ p_out,
                                      const  WORD8 * __restrict__ p_vec,
                                      int    activation_min,
                                      int    activation_max,
                                      WORD32 vec_length)
{
    int i;
    ae_int16x4 x, y, min, max;
    ae_valign align_out = AE_ZALIGN64();
    //xtbool4 b0;

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

    WORD8 *p_o = p_out;
    WORD8 *p_v = (WORD8 *)p_vec;

    min  = AE_MOVDA16(activation_min);
    max  = AE_MOVDA16(activation_max);

    // pre loop, active when input ptr is not 4 byte aligned
    int pre_loop_count=0;
    pre_loop_count = (int)((unsigned)ALIGN_PTR(p_v, 4) - (unsigned)p_v);
    pre_loop_count = (pre_loop_count < vec_length) ? pre_loop_count : vec_length;

    vec_length = vec_length - pre_loop_count;
    vec_length = (vec_length < 0) ? 0 : vec_length;

    for(i=0; i<pre_loop_count; i++)
    {
        AE_L8S_IP(x, p_v, 1);
        LIMIT(y, x, min, max)
        AE_S8_0_IP_HIFI1(y, p_o, 1);
    }
  
    if((activation_max >= (int)MAX_WORD8) && (activation_min <= (int)MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4S_IP(x, p_v, 4*sizeof(WORD8));
            AE_SA8X4U_IP(x, align_out, (ae_int32*)p_o);
        }
        AE_SA64POS_FP(align_out, p_o);

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L8S_IP(x, p_v, sizeof(WORD8));
            AE_S8_0_IP_HIFI1(x, p_o, sizeof(WORD8));
        }
    }
    else if((activation_max < (int)MAX_WORD8) && (activation_min <= MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4S_IP(x, p_v, 4*sizeof(WORD8));
            x = AE_MIN16(x, max);
            AE_SA8X4U_IP(x, align_out, (ae_int32*)p_o);
        }
        AE_SA64POS_FP(align_out, p_o);

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L8S_IP(x, p_v, sizeof(WORD8));
            x = AE_MIN16(x, max);
            AE_S8_0_IP_HIFI1(x, p_o, sizeof(WORD8));
        }
    }
    else if((activation_max >= (int)MAX_WORD8) && (activation_min > MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4S_IP(x, p_v, 4*sizeof(WORD8));
            x = AE_MAX16(x, min);
            AE_SA8X4U_IP(x, align_out, (ae_int32*)p_o);
        }
        AE_SA64POS_FP(align_out, p_o);

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L8S_IP(x, p_v, sizeof(WORD8));
            x = AE_MAX16(x, min);
            AE_S8_0_IP_HIFI1(x, p_o, sizeof(WORD8));
        }
    }
    else
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4S_IP(x, p_v, 4*sizeof(WORD8));
            LIMIT(y, x, min, max) 
            AE_SA8X4U_IP(y, align_out, (ae_int32*)p_o);
        }
        AE_SA64POS_FP(align_out, p_o);

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L8S_IP(x, p_v, sizeof(WORD8));
            LIMIT(y, x, min, max)
            AE_S8_0_IP_HIFI1(y, p_o, sizeof(WORD8));
        }
    }

    return 0;
}
#endif
#else
WORD32 xa_nn_vec_activation_min_max_8_8(WORD8 * __restrict__ p_out,
                                      const  WORD8 * __restrict__ p_vec,
                                      int    activation_min,
                                      int    activation_max,
                                      WORD32 vec_length)
{
    int i;
    ae_int16x4 x, y, min, max;
    xtbool4 b0;

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

    WORD8 *p_o = p_out;
    WORD8 *p_v = (WORD8 *)p_vec;

    min  = AE_MOVDA16(activation_min);
    max  = AE_MOVDA16(activation_max);

    // pre loop, active when input ptr is not 4 byte aligned
    int pre_loop_count=0;
    pre_loop_count = (int)((unsigned)ALIGN_PTR(p_v, 4) - (unsigned)p_v);
    pre_loop_count = (pre_loop_count < vec_length) ? pre_loop_count : vec_length;

    vec_length = vec_length - pre_loop_count;
    vec_length = (vec_length < 0) ? 0 : vec_length;

    for(i=0; i<pre_loop_count; i++)
    {
        int i1;
        i1 = ((WORD8)*p_v++);
        x  = AE_MOVDA16(i1);
        LIMIT(y, x, min, max)
        i1 = AE_MOVAD16_3(y);
        *p_o++ = (WORD8)i1;
    }

    if((activation_max >= (int)MAX_WORD8) && (activation_min <= (int)MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4F_IP(x, p_v, 4*sizeof(WORD8));
            y = AE_SRAI16(x, 8);

            STORE_8X4_FROM_16X4(p_o, y)
        }
        for(i=0; i < (vec_length & 3); i++)
        {
            int i1;
            i1 = (WORD8) p_v[i];
            y  = AE_MOVDA16(i1);

            i1 = AE_MOVAD16_3(y);
            *p_o++ = (WORD8)i1;
        }
    }
    else if((activation_max < (int)MAX_WORD8) && (activation_min <= MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4F_IP(x, p_v, 4*sizeof(WORD8));
            y = AE_SRAI16(x, 8);

            b0 = AE_LT16(y, max);
            AE_MOVF16X4(y, max, b0);

            STORE_8X4_FROM_16X4(p_o, y)
        }
        for(i=0; i < (vec_length & 3); i++)
        {
            int i1;
            i1 = (WORD8) p_v[i];
            y  = AE_MOVDA16(i1);

            b0 = AE_LT16(y, max);
            AE_MOVF16X4(y, max, b0);

            i1 = AE_MOVAD16_3(y);
            *p_o++ = (WORD8)i1;
        }
    }
    else if((activation_max >= (int)MAX_WORD8) && (activation_min > MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4F_IP(x, p_v, 4*sizeof(WORD8));
            y = AE_SRAI16(x, 8);

            b0 = AE_LT16(y, min);
            AE_MOVT16X4(y, min, b0);

            STORE_8X4_FROM_16X4(p_o, y)
        }
        for(i=0; i < (vec_length & 3); i++)
        {
            int i1;
            i1 = (WORD8) p_v[i];
            y  = AE_MOVDA16(i1);

            b0 = AE_LT16(y, min);
            AE_MOVT16X4(y, min, b0);

            i1 = AE_MOVAD16_3(y);
            *p_o++ = (WORD8)i1;
        }
    }
    else
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4F_IP(x, p_v, 4*sizeof(WORD8));
            x = AE_SRAI16(x, 8);
            LIMIT(y, x, min, max)
            STORE_8X4_FROM_16X4(p_o, y)
        }
        for(i=0; i < (vec_length & 3); i++)
        {
            int i1;
            i1 = (WORD8) p_v[i];
            x  = AE_MOVDA16(i1);
            LIMIT(y, x, min, max)
            i1 = AE_MOVAD16_3(y);
            *p_o++ = (WORD8)i1;
        }
    }

    return 0;
}
#endif
/*
 * ReLU 8-bit:
 */
WORD32 xa_nn_vec_relu_8_8(
    WORD8        * __restrict__ p_out,
    const WORD8  * __restrict__ p_vec,
    WORD8       threshold,
    WORD32       vec_length)
{
    xa_nn_vec_activation_min_max_8_8( p_out,
                                      p_vec,
                                      0,
                                      threshold,
                                      vec_length);
    return 0;
}

/*
 * ReLU Standard 8-bit:
 */
WORD32 xa_nn_vec_relu_std_8_8(
    WORD8        * __restrict__ p_out,
    const WORD8  * __restrict__ p_vec,
    WORD32       vec_length)
{
    xa_nn_vec_activation_min_max_8_8( p_out,
                                      p_vec,
                                      0,
                                      MAX_WORD8,
                                      vec_length);
	return 0;
}
