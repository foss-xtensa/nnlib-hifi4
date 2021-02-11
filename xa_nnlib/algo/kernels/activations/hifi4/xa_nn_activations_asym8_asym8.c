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
#include "xa_nnlib_common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"


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

#define STORE_8X4_FROM_32X4(out_ptr, val12, val34){\
    int o1, o2, o3, o4;\
    o1 = AE_MOVAD32_H(val12);\
    o2 = AE_MOVAD32_L(val12);\
    o3 = AE_MOVAD32_H(val34);\
    o4 = AE_MOVAD32_L(val34);\
    *out_ptr++ = (UWORD8)o1;\
    *out_ptr++ = (UWORD8)o2;\
    *out_ptr++ = (UWORD8)o3;\
    *out_ptr++ = (UWORD8)o4;\
}

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

#define MultiplyByQuantizedMultiplierGreaterThanOne(y, x, multiplier, lsh) {\
    y = AE_SLAA32(x, lsh);\
    y = AE_MULFP32X2RAS(y, multiplier);\
}

#define MultiplyByQuantizedMultiplierSmallerThanOneExp(prod, val, multiplier, lsh) {\
    ae_int64 temp64_h, temp64_l;\
    prod = AE_MULFP32X2RAS(val, multiplier);\
    temp64_h = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(prod, ZERO));\
    temp64_l = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(prod, ZERO));\
    temp64_h = AE_SLAA64S(temp64_h, lsh);\
    temp64_l = AE_SLAA64S(temp64_l, lsh);\
    prod = AE_ROUND32X2F64SSYM(temp64_h, temp64_l);\
}

#define ROUNDING_HALF_SUM(s, a){\
    ae_int64 max32;\
    ae_int64 r=-1;\
    xtbool br;\
    max32 = Q31;\
    s = AE_ADD64(max32, a);\
    br = AE_LE64((ae_int64)0, s);\
    AE_MOVT64(r, (ae_int64)1, br);\
    s = AE_SRAI64(AE_ADD64(s,r), 1);\
}

#define EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL(y_out, inp)\
{\
    ae_int32x2 x1_in, x2, x3, x4, x4_by_4, y1, y2, y3, y4, y5, y6;\
\
    x1_in = AE_ADD32S(inp, CT_1_BY_8);\
    x2 = AE_MULFP32X2RS(x1_in, x1_in);\
    x3 = AE_MULFP32X2RS(x2, x1_in);\
    x4 = AE_MULFP32X2RS(x2, x2);\
    x4_by_4 = AE_SRAI32R(x4, 2);\
    y1 = AE_ADD32S(x4_by_4, x3);\
    y2 = AE_MULFP32X2RS(y1, CT_1_BY_3);\
    y3 = AE_ADD32S(y2, x2);\
    y4 = AE_SRAI32R(y3, 1);\
\
    y5 = AE_ADD32S(x1_in, y4); \
    y6 = AE_MULFP32X2RS(y5, CT);\
    y_out = AE_ADD32S(y6, CT);\
}

#define GEMMLOWP_EXP_BARREL_SHIFTER(out, exponent, FixedPointMultiplier, remainder)\
{\
    int shift_amount;\
    ae_int32x2 out1,  mask, scale;\
    xtbool2 b;\
\
    shift_amount = 27 + exponent;\
    scale = AE_SLAA32(ONE, shift_amount);\
\
    mask = AE_AND32(remainder,  scale);\
\
    b = AE_LT32(zero, mask);\
\
    out1 = AE_MULFP32X2RS(out, FixedPointMultiplier);\
    AE_MOVT32X2(out, out1, b);\
}

// calculates exp for inp < 0
#define EXP_Q26(y, inp)\
{\
    xtbool2 b;\
    ae_int32x2 x_in, x2, remainder;\
    ae_int32x2 a_mod_quater_minus_q_1_by_4;\
\
    x2 = AE_AND32(inp, mask_6fs);\
    a_mod_quater_minus_q_1_by_4 = AE_SUB32(x2, q_1_by_4);\
    x_in = AE_SLAI32(a_mod_quater_minus_q_1_by_4, 4);\
\
    EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL(y, x_in)\
\
    remainder = AE_SUB32(a_mod_quater_minus_q_1_by_4, inp);\
\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,-2, 1672461947, remainder);\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,-1, 1302514674, remainder);\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,0, 790015084,   remainder);\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,1, 290630308,   remainder);\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,2, 39332535,    remainder);\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,3, 720401,      remainder);\
\
    b = AE_EQ32(inp, zero);\
    AE_MOVT32X2(y, AE_MOVDA32(Q31), b);\
}

//extern ae_int32x2 one_over_one_plus_x_for_x_in_0_1(ae_int64 a);

//output: y1, y2 (ae_int32x2)
//input:  a1, a2 (ae_int32x2)
#define ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y1, y2, a1, a2){\
    ae_int64 s1, s2, s3, s4;\
    ae_int64 t1, t2, t3, t4;\
    ae_int32x2 half_den12, m1, x1, half_denominator_times_x1;\
    ae_int32x2 half_den34, m2, x2, half_denominator_times_x2;\
    ae_int32x2 one_minus_half_denominator_times_x1;\
    ae_int32x2 one_minus_half_denominator_times_x2;\
    ae_int32x2 CT_48_by_7, CT_neg_32_by_7, CT_F2_ONE;\
    int i;\
\
    CT_48_by_7 = AE_MOVDA32(constant_48_over_17);\
    CT_neg_32_by_7 = AE_MOVDA32(constant_neg_32_over_17);\
    CT_F2_ONE = AE_MOVDA32(F2_ONE);\
   \
    s1 = AE_MUL32_HH(a1, ONE);\
    s2 = AE_MUL32_LL(a1, ONE);\
    s3 = AE_MUL32_HH(a2, ONE);\
    s4 = AE_MUL32_LL(a2, ONE);\
\
    ROUNDING_HALF_SUM(t1, s1)\
    ROUNDING_HALF_SUM(t2, s2)\
    ROUNDING_HALF_SUM(t3, s3)\
    ROUNDING_HALF_SUM(t4, s4)\
\
    half_den12 = AE_MOVINT32X2_FROMINT64(t1);\
    half_den34 = AE_MOVINT32X2_FROMINT64(t2);\
    half_den12 = AE_SEL32_LL(half_den12, half_den34);\
\
    half_den34 = AE_MOVINT32X2_FROMINT64(t3);\
    m1 = AE_MOVINT32X2_FROMINT64(t4);\
    half_den34 = AE_SEL32_LL(half_den34, m1);\
\
    m1 = AE_MULFP32X2RS(half_den12, CT_neg_32_by_7);\
    x1 = AE_ADD32S(m1, CT_48_by_7);\
\
    m2 = AE_MULFP32X2RS(half_den34, CT_neg_32_by_7);\
    x2 = AE_ADD32S(m2, CT_48_by_7);\
\
    for(i=0; i<3; i++)\
    {\
        half_denominator_times_x1 = AE_MULFP32X2RS(x1, half_den12);\
        one_minus_half_denominator_times_x1 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x1);\
        half_denominator_times_x1 = AE_MULFP32X2RS(x1, half_den12);\
        m1 = AE_MULFP32X2RS(x1, one_minus_half_denominator_times_x1);\
        m1 = AE_SLAI32S(m1, 2);\
        x1 = AE_ADD32S(x1, m1);\
        \
        half_denominator_times_x2 = AE_MULFP32X2RS(x2, half_den34);\
        one_minus_half_denominator_times_x2 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x2);\
        half_denominator_times_x2 = AE_MULFP32X2RS(x2, half_den34);\
        m2 = AE_MULFP32X2RS(x2, one_minus_half_denominator_times_x2);\
        m2 = AE_SLAI32S(m2, 2);\
        x2 = AE_ADD32S(x2, m2);\
        \
    }\
\
    y1 = AE_SLAI32S(x1, 1);\
    y2 = AE_SLAI32S(x2, 1);\
\
}

static const int CONSTANT_TERM =  (0x70f5a894);
static const int CONSTANT_1_OVER_3 = (0x2aaaaaab);
static const int CONSTANT_1_OVER_8 = (0x10000000);
static const int ONE_QUATER_Q26 = (0x2000000); // Q5.27
static const int MASK = (0x1ffffff);
static const int Q31 = 0x7fffffff;
static const int constant_48_over_17 = 1515870810;
static const int constant_neg_32_over_17 = -1010580540;
static const int F2_ONE = 0x20000000;

WORD32 xa_nn_vec_sigmoid_asym8_asym8(UWORD8 *p_out,
                      const UWORD8 *p_vec,
                            WORD32 zero_point,
                            WORD32 input_range_radius,
                            WORD32 input_multiplier,
                            WORD32 input_left_shift,
                            WORD32 vec_length)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(UWORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND(((zero_point < 0) || (zero_point > 255)), -1);
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((input_left_shift < -31) || (input_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND((input_multiplier < 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_range_radius < 0), -1);

    int i;
    ae_int16x4 x;
    ae_int32x2 x32, x10, y32, y10, z32, z10;
    ae_int32x2 z, mul, zero;
    ae_int32x2 q31 = AE_MOVDA32(Q31);
    ae_int32x2 mask_6fs = AE_MOVDA32(MASK);
    ae_int32x2 q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
    ae_int32x2 CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
    ae_int32x2 CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
    ae_int32x2 CT = AE_MOVDA32(CONSTANT_TERM);
    ae_int32x2 ONE = AE_MOVDA32(1);
    ae_int32x2 CONST_256 = AE_MOVDA32(256);
    ae_int32x2 CONST_255 = AE_MOVDA32(255);
    ae_int32x2 radius, minus_radius;
    xtbool2 b32, b10, c32, c10, d32, d10;
    int pre_loop_count, main_loop_count, post_loop_count;

    if(vec_length > 3)
    {
        pre_loop_count = (int)((4 - ((int)p_vec & 0x3))&3);
        main_loop_count = vec_length - pre_loop_count;
        post_loop_count = (main_loop_count & 3);
        main_loop_count = main_loop_count >> 2;
    }
    else
    {
        pre_loop_count = 0;
        main_loop_count = 0;
        post_loop_count = vec_length;
    }

    UWORD8 *p_in  = (UWORD8 *)p_vec;
    UWORD8 *p_o = (UWORD8 *)p_out;

    radius = AE_MOVDA32(input_range_radius);
    minus_radius = AE_NEG32(radius);

    z = AE_MOVDA32(zero_point);
    mul = AE_MOVDA32(input_multiplier);
    zero = AE_ZERO32();

    __Pragma("no_unroll");
    for(i=0; i<pre_loop_count; i++)
    {
        int inp;

        inp = (int)*p_in++;

        x32 = AE_MOVDA32(inp);
        x32 = AE_SUB32S(x32, z);

        // set flag if x <= minus_radius
        b32 = AE_LE32(x32, minus_radius);

        // set flag if x < radius
        c32 = AE_LT32(x32, radius);
        MultiplyByQuantizedMultiplierGreaterThanOne(y32, x32, mul, input_left_shift)

        d32 = AE_LT32(y32, zero);

        // Computing Absolute value
        x32 = AE_ABS32(y32);
        y32 = AE_NEG32(x32);

        // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
        EXP_Q26(x32, y32)
        x10 = x32;

        ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, x32, x10)

        // if (dequantized_input < 0) output = 1 - sigmoid(abs(dequantized_input))
        AE_MOVT32X2(y32, AE_SUB32S(q31, y32), d32);

        // Downscale to 8 bit
        z32 = AE_SRAA32RS(y32, 23);

        // if(z == 256) z = 255;
        d32 = AE_EQ32(z32, CONST_256);
        AE_MOVT32X2(z32, CONST_255, d32);

        // if(inp_centered <= -radius) output = 0
        AE_MOVT32X2(z32, AE_ZERO32(), b32);

        // if(inp_centered >= radius) output = 255
        AE_MOVF32X2(z32, CONST_255, c32);

        inp = AE_MOVAD32_H(z32);
        *p_o++ = (UWORD8)inp;
    }

    WORD8 *p_in_t = (WORD8 *)p_in;
    for(i=0; i < main_loop_count; i++)
    {
        AE_L8X4F_IP(x, p_in_t, 4*sizeof(WORD8));
        x = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x), 8));
        x32 = AE_SEXT32X2D16_32(x);
        x10 = AE_SEXT32X2D16_10(x);

        x32 = AE_SUB32S(x32, z);
        x10 = AE_SUB32S(x10, z);

        // set flag if x <= minus_radius
        b32 = AE_LE32(x32, minus_radius);
        b10 = AE_LE32(x10, minus_radius);

        // set flag if x < radius
        c32 = AE_LT32(x32, radius);
        c10 = AE_LT32(x10, radius);

        MultiplyByQuantizedMultiplierGreaterThanOne(y32, x32, mul, input_left_shift)
        MultiplyByQuantizedMultiplierGreaterThanOne(y10, x10, mul, input_left_shift)

        d32 = AE_LT32(y32, zero);
        d10 = AE_LT32(y10, zero);

        // Computing Absolute value
        x32 = AE_ABS32(y32);
        x10 = AE_ABS32(y10);

        y32 = AE_NEG32(x32);
        y10 = AE_NEG32(x10);

        // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
        EXP_Q26(x32, y32)
        EXP_Q26(x10, y10)

        ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, x32, x10)

        // if (dequantized_input < 0) output = 1 - sigmoid(abs(dequantized_input))
        AE_MOVT32X2(y32, AE_SUB32S(q31, y32), d32);
        AE_MOVT32X2(y10, AE_SUB32S(q31, y10), d10);

        // Downscale to 8 bit
        z32 = AE_SRAA32RS(y32, 23);
        z10 = AE_SRAA32RS(y10, 23);

        // if(z == 256) z = 255;
        d32 = AE_EQ32(z32, CONST_256);
        d10 = AE_EQ32(z10, CONST_256);
        AE_MOVT32X2(z32, CONST_255, d32);
        AE_MOVT32X2(z10, CONST_255, d10);

        // if(inp_centered <= -radius) output = 0
        AE_MOVT32X2(z32, AE_ZERO32(), b32);
        AE_MOVT32X2(z10, AE_ZERO32(), b10);

        // if(inp_centered >= radius) output = 255
        AE_MOVF32X2(z32, CONST_255, c32);
        AE_MOVF32X2(z10, CONST_255, c10);

        STORE_8X4_FROM_32X4(p_o, z32, z10)
    }
    p_in = (UWORD8 *)p_in_t;

    __Pragma("no_unroll");
    for(i=0; i<post_loop_count; i++)
    {
        int inp;

        inp = (int)p_in[i];

        x32 = AE_MOVDA32(inp);
        x32 = AE_SUB32S(x32, z);

        // set flag if x <= minus_radius
        b32 = AE_LE32(x32, minus_radius);

        // set flag if x < radius
        c32 = AE_LT32(x32, radius);
        MultiplyByQuantizedMultiplierGreaterThanOne(y32, x32, mul, input_left_shift)

        d32 = AE_LT32(y32, zero);

        // Computing Absolute value
        x32 = AE_ABS32(y32);
        y32 = AE_NEG32(x32);

        // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
        EXP_Q26(x32, y32)
        x10 = x32;

        ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, x32, x10)

        // if (dequantized_input < 0) output = 1 - sigmoid(abs(dequantized_input))
        AE_MOVT32X2(y32, AE_SUB32S(q31, y32), d32);

        // Downscale to 8 bit
        z32 = AE_SRAA32RS(y32, 23);

        // if(z == 256) z = 255;
        d32 = AE_EQ32(z32, CONST_256);
        AE_MOVT32X2(z32, CONST_255, d32);

        // if(inp_centered <= -radius) output = 0
        AE_MOVT32X2(z32, AE_ZERO32(), b32);

        // if(inp_centered >= radius) output = 255
        AE_MOVF32X2(z32, CONST_255, c32);

        inp = AE_MOVAD32_H(z32);
        *p_o++ = (UWORD8)inp;
    }

    return 0;
}

WORD32 xa_nn_vec_sigmoid_asym8s_asym8s(WORD8 *p_out,
                      const WORD8 *p_vec,
                            WORD32 zero_point,
                            WORD32 input_range_radius,
                            WORD32 input_multiplier,
                            WORD32 input_left_shift,
                            WORD32 vec_length)
{
  return -1;
}

/*
 * inp: p_vec: 4 byte aligned input pointer
 * out: p_out: no alignment needed for output pointer*/
WORD32 xa_nn_vec_activation_min_max_asym8_asym8(UWORD8 * __restrict__ p_out,
                                      const  UWORD8 * __restrict__ p_vec,
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

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

    UWORD8 *p_o = p_out;
    UWORD8 *p_v = (UWORD8 *)p_vec;

    min  = AE_MOVDA16(activation_min);
    max  = AE_MOVDA16(activation_max);

    int pre_loop_count=0;
    // pre loop, active when input ptr is not 4 byte aligned
    pre_loop_count = (int)((unsigned)ALIGN_PTR(p_v, 4) - (unsigned)p_v);
    pre_loop_count = (pre_loop_count < vec_length) ? pre_loop_count : vec_length;

    vec_length = vec_length - pre_loop_count;
    vec_length = (vec_length < 0) ? 0 : vec_length;

    for(i=0; i<pre_loop_count; i++)
    {
        int i1;
        i1 = ((UWORD8)*p_v++);
        x  = AE_MOVDA16(i1);
        LIMIT(y, x, min, max)
        i1 = AE_MOVAD16_3(y);
        *p_o++ = (UWORD8)i1;
    }

    WORD8 *p_v_t = (WORD8 *)p_v;
    if((activation_max >= (int)255) && (activation_min <= (int)0))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4F_IP(x, p_v_t, 4*sizeof(WORD8));
            y = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x), 8));

            STORE_8X4_FROM_16X4(p_o, y)
        }
        p_v = (UWORD8 *)p_v_t;
        for(i=0; i < (vec_length & 3); i++)
        {
            int i1;
            i1 = (UWORD8) p_v[i];
            y  = AE_MOVDA16(i1);

            i1 = AE_MOVAD16_3(y);
            *p_o++ = (UWORD8)i1;
        }
    }
    else if((activation_max < (int)255) && (activation_min <= 0))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4F_IP(x, p_v_t, 4*sizeof(WORD8));
            y = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x), 8));

            b0 = AE_LT16(y, max);
            AE_MOVF16X4(y, max, b0);

            STORE_8X4_FROM_16X4(p_o, y)
        }
        p_v = (UWORD8 *)p_v_t;
        for(i=0; i < (vec_length & 3); i++)
        {
            int i1;
            i1 = (UWORD8) p_v[i];
            y  = AE_MOVDA16(i1);

            b0 = AE_LT16(y, max);
            AE_MOVF16X4(y, max, b0);

            i1 = AE_MOVAD16_3(y);
            *p_o++ = (UWORD8)i1;
        }
    }
    else if((activation_max >= (int)255) && (activation_min > 0))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4F_IP(x, p_v_t, 4*sizeof(WORD8));
            y = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x), 8));

            b0 = AE_LT16(y, min);
            AE_MOVT16X4(y, min, b0);

            STORE_8X4_FROM_16X4(p_o, y)
        }
        p_v = (UWORD8 *)p_v_t;
        for(i=0; i < (vec_length & 3); i++)
        {
            int i1;
            i1 = (UWORD8) p_v[i];
            y  = AE_MOVDA16(i1);

            b0 = AE_LT16(y, min);
            AE_MOVT16X4(y, min, b0);

            i1 = AE_MOVAD16_3(y);
            *p_o++ = (UWORD8)i1;
        }
    }
    else
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_L8X4F_IP(x, p_v_t, 4*sizeof(WORD8));
            x = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x), 8));
            LIMIT(y, x, min, max)
            STORE_8X4_FROM_16X4(p_o, y)
        }
        p_v = (UWORD8 *)p_v_t;
        for(i=0; i < (vec_length & 3); i++)
        {
            int i1;
            i1 = (UWORD8) p_v[i];
            x  = AE_MOVDA16(i1);
            LIMIT(y, x, min, max)
            i1 = AE_MOVAD16_3(y);
            *p_o++ = (UWORD8)i1;
        }
    }

    return 0;
}

WORD32 xa_nn_vec_relu_asym8u_asym8u( UWORD8 * __restrict__ p_out,
                    const   UWORD8 * __restrict__ p_vec,
                            WORD32 inp_zero_bias,
                            WORD32 out_multiplier,
                            WORD32 out_shift,
                            WORD32 out_zero_bias,
                            WORD32 quantized_activation_min,
                            WORD32 quantized_activation_max,
                            WORD32 vec_length)
{
  return -1;
}

WORD32 xa_nn_vec_relu_asym8s_asym8s( WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_vec,
                            WORD32 inp_zero_bias,
                            WORD32 out_multiplier,
                            WORD32 out_shift,
                            WORD32 out_zero_bias,
                            WORD32 quantized_activation_min,
                            WORD32 quantized_activation_max,
                            WORD32 vec_length)
{
  return -1;
}

WORD32 xa_nn_vec_prelu_asym8s_asym8s( WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_vec,
                    const   WORD8 * __restrict__ p_vec_alpha,
                            WORD32 inp_zero_bias,
                            WORD32 alpha_zero_bias,
                            WORD32 alpha_multiplier,
                            WORD32 alpha_shift,
                            WORD32 out_multiplier,
                            WORD32 out_shift,
                            WORD32 out_zero_bias,
                            WORD32 vec_length)
{
  return -1;
}

WORD32 xa_nn_vec_hard_swish_asym8s_asym8s( WORD8 * __restrict__ p_out,
                            const   WORD8 * __restrict__ p_vec,
                            WORD32 inp_zero_bias,
                            WORD16 reluish_multiplier,
                            WORD32 reluish_shift,
                            WORD16 out_multiplier,
                            WORD32 out_shift,
                            WORD32 out_zero_bias,
                            WORD32 vec_length)
{
  return -1;
}

WORD32 xa_nn_vec_tanh_asym8s_asym8s(WORD8 *p_out,
                      const WORD8 *p_vec,
                            WORD32 zero_point,
                            WORD32 input_range_radius,
                            WORD32 input_multiplier,
                            WORD32 input_left_shift,
                            WORD32 vec_length)
{
  return -1;
}

#if 0
enum ActivationFn {
    kActivationNone = 0,
    kActivationRelu,
    kActivationRelu1,
    kActivationRelu6,
    kActivationTanh,
    kActivationSignBit,
    kActivationSigmoid,
};

#define QUANTIZE(y, f){\
    xtfloat recip_scale, prod;\
    recip_scale = XT_FLOAT_S(input_scale, 0);\
    recip_scale = XT_RECIP_S(recip_scale);\
    prod = XT_MUL_S(recip_scale, XT_FLOAT_S(f, 0));\
    prod = XT_FIROUND_S(prod);\
    y = XT_ADD_S(prod, XT_FLOAT_S(input_offset, 0));\
}

#define CALCULATE_ACTIVATION_RANGE_ASYM8(activation){\
    if (activation == kActivationRelu)\
    {\
        QUANTIZE(y, 0.0)\
        act_min = XT_MAX_S(0, y);\
        act_max = 255;\
    }\
    else if (activation == kActivationRelu6) \
    {\
       QUANTIZE(y, 0.0)\
       act_min = XT_MAX_S(0, y);\
       QUANTIZE(y, 6.0)\
       act_max = XT_MIN_S(255, y);\
    }\
    else if (activation == kActivationRelu1) \
    {\
       QUANTIZE(y, -1.0)\
       act_min = XT_MAX_S(0, y);\
       QUANTIZE(y, 1.0)\
       act_max = XT_MIN_S(255, y);\
    }\
    else if (activation == kActivationNone)\
    {\
        act_min = 0;\
        act_max = 255;\
    }\
}


WORD32 xa_nn_vec_relu_asym8(
    UWORD8       * __restrict__ p_out,
    const UWORD8 * __restrict__ p_inp,
    WORD32       input_offset,
    WORD32       input_scale,
    WORD32       vec_length)
{
  xtfloat y, act_max, act_min;

  // Calculating act_min and act_max
  CALCULATE_ACTIVATION_RANGE_ASYM8(kActivationRelu)

  relu_asym8(p_out,
             p_inp,
             act_min,
             act_max,
             vec_length);
  return 0;
}


WORD32 xa_nn_vec_relu1_asym8(
    UWORD8       * __restrict__ p_out,
    const UWORD8 * __restrict__ p_inp,
    WORD32       input_offset,
    WORD32       input_scale,
    WORD32       vec_length)
{
  xtfloat y, act_max, act_min;
  // Calculating act_min and act_max
  CALCULATE_ACTIVATION_RANGE_ASYM8(kActivationRelu1)

  relu_asym8(p_out,
             p_inp,
             act_min,
             act_max,
             vec_length);

  return 0;
}

WORD32 xa_nn_vec_relu6_asym8(
    UWORD8       * __restrict__ p_out,
    const UWORD8 * __restrict__ p_inp,
    WORD32       input_offset,
    WORD32       input_scale,
    WORD32       vec_length)
{
  xtfloat y, act_max, act_min;
  // Calculating act_min and act_max
  CALCULATE_ACTIVATION_RANGE_ASYM8(kActivationRelu6)

  relu_asym8(p_out,
             p_inp,
             act_min,
             act_max,
             vec_length);

  return 0;
}
#endif

