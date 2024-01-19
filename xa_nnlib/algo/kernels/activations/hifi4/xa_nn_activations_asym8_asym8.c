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
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"

#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#if XCHAL_HAVE_HIFI1

#define MAX_16X4(id1, id0) \
        id1 = AE_MAX16(id1, id0);\

#define MIN_16X4(id1, id0) \
        id1 = AE_MIN16(id1, id0);\

#define LIMIT(out, inp, min, max){\
        out = min;\
        MAX_16X4(out, inp);\
        MIN_16X4(out, max);\
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
    *out_ptr++ = (UWORD8)o1;\
    *out_ptr++ = (UWORD8)o2;\
    *out_ptr++ = (UWORD8)o3;\
    *out_ptr++ = (UWORD8)o4;\
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

#endif

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
    xtbool2 b0;\
    ae_int32x2 x_in, x0, remainder;\
    ae_int32x2 a_mod_quater_minus_q_1_by_4;\
\
    x0 = AE_AND32(inp, mask_6fs);\
    a_mod_quater_minus_q_1_by_4 = AE_SUB32(x0, q_1_by_4);\
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
    b0 = AE_EQ32(inp, zero);\
    AE_MOVT32X2(y, AE_MOVDA32(Q31), b0);\
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
    int j;\
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
    for(j=0; j<3; j++)\
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
    XA_NNLIB_ARG_CLIP(input_left_shift, -31, 31);
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
#if XCHAL_HAVE_HIFI1
    ae_valign align_out = AE_ZALIGN64();
#endif

#if TFLITE_SINGLE_ROUNDING
    int left_shift  = input_left_shift;
    int right_shift = input_left_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    int left_shift  = input_left_shift<0?0: input_left_shift;
    int right_shift = input_left_shift>0?0:-input_left_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

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

        d32 = AE_LT32(x32, zero);

        MPY_BY_QUANT_MULT_X2_OUT32(y32, x32, mul, left_shift, right_shift)

        // Computing Absolute value
        x32 = AE_ABS32(y32);
        y32 = AE_NEG32(x32);

        // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
        EXP_Q26(x32, y32)
        x10 = x32;

        ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, x32, x10)

        // if (inp_centered < 0) output = 1 - sigmoid(abs(dequantized_input))
        AE_MOVT32X2(y32, AE_SUB32S(q31, y32), d32);

        // Downscale to 8 bit
        z32 = AE_SRAA32RS(y32, 23);

        // if(z == 256) z = 255;
        d32 = AE_EQ32(z32, CONST_256);
        AE_MOVT32X2(z32, CONST_255, d32);

        // if(inp_centered >= radius) output = 255
        AE_MOVF32X2(z32, CONST_255, c32);

        // if(inp_centered <= -radius) output = 0
        AE_MOVT32X2(z32, AE_ZERO32(), b32);

        inp = AE_MOVAD32_H(z32);
        *p_o++ = (UWORD8)inp;
    }

    WORD8 *p_in_t = (WORD8 *)p_in;
    for(i=0; i < main_loop_count; i++)
    {
#if XCHAL_HAVE_HIFI1
        AE_L8X4U_IP(x, p_in_t, 4*sizeof(WORD8));
#else
        AE_L8X4F_IP(x, p_in_t, 4*sizeof(WORD8));
        x = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x), 8));
#endif
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

        d32 = AE_LT32(x32, zero);
        d10 = AE_LT32(x10, zero);

        MPY_BY_QUANT_MULT_X2_OUT32(y32, x32, mul, left_shift, right_shift)
        MPY_BY_QUANT_MULT_X2_OUT32(y10, x10, mul, left_shift, right_shift)

        // Computing Absolute value
        x32 = AE_ABS32(y32);
        x10 = AE_ABS32(y10);

        y32 = AE_NEG32(x32);
        y10 = AE_NEG32(x10);

        // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
        EXP_Q26(x32, y32)
        EXP_Q26(x10, y10)

        ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, x32, x10)

        // if (inp_centered < 0) output = 1 - sigmoid(abs(dequantized_input))
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

        // if(inp_centered >= radius) output = 255
        AE_MOVF32X2(z32, CONST_255, c32);
        AE_MOVF32X2(z10, CONST_255, c10);

        // if(inp_centered <= -radius) output = 0
        AE_MOVT32X2(z32, AE_ZERO32(), b32);
        AE_MOVT32X2(z10, AE_ZERO32(), b10);

#if XCHAL_HAVE_HIFI1
        x = AE_CVT16X4(z32, z10);
        AE_SA8X4U_IP(x, align_out, (ae_int32*)p_o);
#else
        STORE_8X4_FROM_32X4(p_o, z32, z10)
#endif
    }
#if XCHAL_HAVE_HIFI1    
    AE_SA64POS_FP(align_out, p_o);
#endif
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

        d32 = AE_LT32(x32, zero);

        MPY_BY_QUANT_MULT_X2_OUT32(y32, x32, mul, left_shift, right_shift)

        // Computing Absolute value
        x32 = AE_ABS32(y32);
        y32 = AE_NEG32(x32);

        // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
        EXP_Q26(x32, y32)
        x10 = x32;

        ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, x32, x10)

        // if (inp_centered < 0) output = 1 - sigmoid(abs(dequantized_input))
        AE_MOVT32X2(y32, AE_SUB32S(q31, y32), d32);

        // Downscale to 8 bit
        z32 = AE_SRAA32RS(y32, 23);

        // if(z == 256) z = 255;
        d32 = AE_EQ32(z32, CONST_256);
        AE_MOVT32X2(z32, CONST_255, d32);

        // if(inp_centered >= radius) output = 255
        AE_MOVF32X2(z32, CONST_255, c32);

        // if(inp_centered <= -radius) output = 0
        AE_MOVT32X2(z32, AE_ZERO32(), b32);

        inp = AE_MOVAD32_H(z32);
        *p_o++ = (UWORD8)inp;
    }

    return 0;
}

#if !defined(USE_HIFI_ACT_TIE) || !defined(AE_TANH16X4) || !defined(AE_SIGMOID16X4)
static const int Q31_minus_1 = 0x7fffffff;
#endif

#define SUB_128(inp){\
  ae_int64 temp;\
  temp = AE_MOVINT64_FROMINT16X4(inp);\
  temp = AE_XOR(temp, offset_xor);\
  inp = AE_MOVINT16X4_FROMINT64(temp);\
}

//output: y1, y2 (ae_int32x2)
//input:  a1, a2 (ae_int32x2)
#define ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2_S(y1, y2, a1, a2){\
  ae_int64 s1, s2, s3, s4;\
  ae_int64 t1, t2, t3, t4;\
  ae_int32x2 half_den12, m1, x1, half_denominator_times_x1;\
  ae_int32x2 half_den34, m2, x2, half_denominator_times_x2;\
  ae_int32x2 one_minus_half_denominator_times_x1;\
  ae_int32x2 one_minus_half_denominator_times_x2;\
  ae_int32x2 CT_48_by_7, CT_neg_32_by_7, CT_F2_ONE;\
  int j;\
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
  m1 = AE_MULFP32X2RAS(half_den12, CT_neg_32_by_7);\
  m2 = AE_MULFP32X2RAS(half_den34, CT_neg_32_by_7);\
  x1 = AE_ADD32S(m1, CT_48_by_7);\
  x2 = AE_ADD32S(m2, CT_48_by_7);\
\
  for(j=0; j<3; j++)\
  {\
    half_denominator_times_x1 = AE_MULFP32X2RAS(x1, half_den12);\
    half_denominator_times_x2 = AE_MULFP32X2RAS(x2, half_den34);\
    one_minus_half_denominator_times_x1 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x1);\
    one_minus_half_denominator_times_x2 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x2);\
    m1 = AE_MULFP32X2RAS(x1, one_minus_half_denominator_times_x1);\
    m2 = AE_MULFP32X2RAS(x2, one_minus_half_denominator_times_x2);\
    m1 = AE_SLAI32S(m1, 2);\
    x1 = AE_ADD32S(x1, m1);\
  \
    m2 = AE_SLAI32S(m2, 2);\
    x2 = AE_ADD32S(x2, m2);\
  \
  }\
\
  y1 = AE_SLAI32S(x1, 1);\
  y2 = AE_SLAI32S(x2, 1);\
\
}

#define EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_S(y_out1, inp1)\
{\
  ae_int32x2 x1_in, x2, x3, x4, x4_by_4, y1, y2, y3, y4, y5, y6;\
\
  x1_in = AE_ADD32S(inp1, CT_1_BY_8);\
  x2 = AE_MULFP32X2RAS(x1_in, x1_in);\
  x3 = AE_MULFP32X2RAS(x2, x1_in);\
  x4 = AE_MULFP32X2RAS(x2, x2);\
  x4_by_4 = AE_SRAI32R(x4, 2);\
  y1 = AE_ADD32S(x4_by_4, x3);\
  y2 = AE_MULFP32X2RAS(y1, CT_1_BY_3);\
  y3 = AE_ADD32S(y2, x2);\
  y4 = AE_SRAI32R(y3, 1);\
\
  y5 = AE_ADD32S(x1_in, y4); \
  y6 = AE_MULFP32X2RAS(y5, CT);\
  y_out1 = AE_ADD32S(y6, CT);\
}

#define GEMMLOWP_EXP_BARREL_SHIFTER_S(out_1, exponent, FixedPointMultiplier, remainder1)\
{\
  int shift_amount;\
  ae_int32x2 out1,  mask1, scale;\
  xtbool2 b1;\
\
  shift_amount = 27 + exponent;\
  scale = AE_SLAA32(ONE, shift_amount);\
\
  mask1 = AE_AND32(remainder1,  scale);\
\
  b1 = AE_LT32(zero, mask1);\
\
  out1 = AE_MULFP32X2RAS(out_1, FixedPointMultiplier);\
  AE_MOVT32X2(out_1, out1, b1);\
}

#define EXP_Q26_S(y1, inp1)\
{\
  xtbool2 b;\
  ae_int32x2 x_in1, x0, remainder1;\
  ae_int32x2 a_mod_quater_minus_q_1_by_4_first;\
\
  x0 = AE_AND32(inp1, mask_6fs);\
  a_mod_quater_minus_q_1_by_4_first = AE_SUB32(x0, q_1_by_4);\
  x_in1 = AE_SLAI32(a_mod_quater_minus_q_1_by_4_first, 4);\
\
  EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_S(y1, x_in1)\
\
  remainder1 = AE_SUB32(a_mod_quater_minus_q_1_by_4_first, inp1);\
\
  GEMMLOWP_EXP_BARREL_SHIFTER_S(y1, -2, 1672461947, remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_S(y1, -1, 1302514674, remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_S(y1, 0, 790015084,   remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_S(y1, 1, 290630308,   remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_S(y1, 2, 39332535,    remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_S(y1, 3, 720401,      remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_S(y1, 4, 242,         remainder1);\
\
  b = AE_EQ32(inp1, zero);\
  AE_MOVT32X2(y1, AE_MOVDA32(Q31_minus_1), b);\
\
}

WORD32 xa_nn_vec_sigmoid_asym8s_asym8s(WORD8 *p_out,
                      const WORD8 *p_vec,
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((zero_point < -128) || (zero_point > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CLIP(input_left_shift, -31, 31);
  XA_NNLIB_ARG_CHK_COND((input_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_range_radius < 0), -1);

  /* Limit the input_range_radius value to int16, as we use ae_int16x4 data types for comparison */
  input_range_radius = (input_range_radius>32767) ? 32767 : input_range_radius;

#if defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4)
  WUR_AE_SAR(4);
#else
  ae_int32x2 mask_6fs = AE_MOVDA32(MASK);
  ae_int32x2 q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
  ae_int32x2 CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
  ae_int32x2 CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
  ae_int32x2 CT = AE_MOVDA32(CONSTANT_TERM);
  ae_int32x2 ONE = AE_MOVDA32(1);
  ae_int16x4 CONST_256_16x4 = AE_MOVDA16(256);
  ae_int32x2 exp_x10, exp_x32;
  xtbool4 f3210;
  ae_int32x2 m32, m10;
#endif /* defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4) */

  int i;
  int rem_length = (vec_length & 3);
  ae_int32x2 x32, x10;
  ae_int32x2 /* z,*/ mul, zero;
  ae_int16x4 CONST_255_16x4 = AE_MOVDA16(255);
//  ae_int32x2 radius, minus_radius;
  ae_int16x4 radius_16, minus_radius_16;
  xtbool4 b3210, d3210;
  ae_int32x2 dequantized_x32, dequantized_x10;
  ae_int16x4 m0, z_16x4; // m2, ;
  ae_int16x4 z10, zero_16x4;

  /* Second operand for XOR instruction used in SUB_128 and ADD_128*/
  ae_int64 offset_xor = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(128));

  WORD8 *p_in  = (WORD8 *)p_vec;
  WORD8 *p_o = (WORD8 *)p_out;

#if XCHAL_HAVE_HIFI1
  ae_valign align_src, align_dst;
  align_src = AE_LA64_PP(p_in);
  align_dst = AE_ZALIGN64();
#else
  ALIGN_REGISTER_TYPE align_src;
  PRIME_8X4F(p_in, align_src);
#endif

  //radius = AE_MOVDA32(input_range_radius);
  //minus_radius = AE_NEG32(radius);

  radius_16 = AE_MOVDA16(input_range_radius);
  minus_radius_16 = AE_NEG16S(radius_16);

  //z = AE_MOVDA32(zero_point);
  z_16x4 = AE_MOVDA16(zero_point);
  mul = AE_MOVDA32(input_multiplier);
  zero = AE_ZERO32();
  zero_16x4 = AE_ZERO16();

#if TFLITE_SINGLE_ROUNDING
    int left_shift  = input_left_shift;
    int right_shift = input_left_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    int left_shift  = input_left_shift<0?0: input_left_shift;
    int right_shift = input_left_shift>0?0:-input_left_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  for(i=0; i<(vec_length >> 2); i++)
  {
#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(m0, align_src, p_in);
#else
    AE_LA8X4F_IP(m0, align_src, p_in);
    m0 = AE_SRAI16(m0, 8);
#endif
    z10 = AE_SUB16(m0, z_16x4);

    // set flag if z <= minus_radius
    b3210 = AE_LE16(z10, minus_radius_16);

    // set flag if z < radius
    d3210 = AE_LT16(z10, radius_16);

    x32 = AE_SEXT32X2D16_32(z10);
    x10 = AE_SEXT32X2D16_10(z10);

    MPY_BY_QUANT_MULT_X2_OUT32(dequantized_x32, x32, mul, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2_OUT32(dequantized_x10, x10, mul, left_shift, right_shift);

#if defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4)
    (void)zero_16x4; (void)zero;
    x32 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(dequantized_x32), 15), AE_SRAA64(AE_CVT64F32_L(dequantized_x32), 15));
    x10 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(dequantized_x10), 15), AE_SRAA64(AE_CVT64F32_L(dequantized_x10), 15));

    z10 = AE_SAT16X4(x32, x10);

    z10 = AE_SIGMOID16X4(z10);

    z10 = AE_SRAA16S(z10, 8);
    z10 = AE_AND16(z10, AE_MOVDA16(0x00ff));
#else
    //set flag if z < 0
    f3210 = AE_LT16(z10, zero_16x4);

    x32 = AE_ABS32(dequantized_x32);
    x10 = AE_ABS32(dequantized_x10);
    x32 = AE_NEG32(x32);
    x10 = AE_NEG32(x10);

    // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
    EXP_Q26_S(exp_x32, x32);
    EXP_Q26_S(exp_x10, x10);

    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2_S(x32, x10, exp_x32, exp_x10)

    // Downscale to 8 bit
    m32 = AE_SRAA32RS(x32, 23);
    m10 = AE_SRAA32RS(x10, 23);
    // Due to rounding operation, sometimes value gets set to 256.
    // We need to saturate it to 255.
    // SATU8X4 used before store operation takes care of this.

    z10 = AE_CVT16X4(m32, m10);

    // if(inp_centered < 0) output = 1 - sigmoid(abs(dequantized_input))
    AE_MOVT16X4(z10, AE_SUB16S(CONST_256_16x4, z10), f3210);

#endif /* defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4) */

    // Computing Absolute value
    // if(inp_centered >= radius) output = 255
    AE_MOVF16X4(z10, CONST_255_16x4, d3210);

    // if(inp_centered <= -radius) output = 0
    AE_MOVT16X4(z10, AE_ZERO16(), b3210);

#if XCHAL_HAVE_HIFI1
    m0 = AE_SAT8U(z10);
#else
    m0 = z10;
    xtbool4 bsat4 = AE_LT16(CONST_255_16x4, m0);
    AE_MOVT16X4(m0, CONST_255_16x4 , bsat4);
#endif

    SUB_128(m0)

#if XCHAL_HAVE_HIFI1
    AE_SA8X4U_IP(m0, align_dst, (ae_int32 *)p_o);
#else
    STORE_8X4_FROM_16X4(p_o, m0);
#endif
  }

#if XCHAL_HAVE_HIFI1
  AE_SA64POS_FP(align_dst, p_o);
#endif

  for(i=0; i<rem_length; i++)
  {
#if XCHAL_HAVE_HIFI1
    AE_L8S_IP(m0, p_in, sizeof(WORD8));
#else
    m0 = (WORD16)*p_in++;
#endif
    z10 = AE_SUB16(m0, z_16x4);

    // set flag if z <= minus_radius
    b3210 = AE_LE16(z10, minus_radius_16);

    // set flag if z < radius
    d3210 = AE_LT16(z10, radius_16);

    x10 = AE_SEXT32X2D16_10(z10);

    MPY_BY_QUANT_MULT_X2_OUT32(dequantized_x10, x10, mul, left_shift, right_shift);

#if defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4)
    x10 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(dequantized_x10), 15), AE_SRAA64(AE_CVT64F32_L(dequantized_x10), 15));

    z10 = AE_SAT16X4(x10, x10);

    z10 = AE_SIGMOID16X4(z10);

    z10 = AE_SRAA16S(z10, 8);
    z10 = AE_AND16(z10, AE_MOVDA16(0x00ff));
#else
    //set flag if z < 0
    f3210 = AE_LT16(z10, zero_16x4);

    // Computing Absolute value
    x10 = AE_ABS32(dequantized_x10);
    x10 = AE_NEG32(x10);

    // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
    EXP_Q26_S(exp_x10, x10);
    exp_x32 = exp_x10;

    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2_S(x32, x10, exp_x32, exp_x10)

    // Downscale to 8 bit
    m10 = AE_SRAA32RS(x10, 23);
    // Due to rounding operation, sometimes value gets set to 256.
    // We need to saturate it to 255.
    // SATU8X8X16 used before store operation takes care of this.
    z10 = AE_CVT16X4(m10, m10);

    // if(inp_centered < 0) output = 1 - sigmoid(abs(dequantized_input))
    AE_MOVT16X4(z10, AE_SUB16S(CONST_256_16x4, z10), f3210);
#endif /* defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4) */

    // if(inp_centered >= radius) output = 255
    AE_MOVF16X4(z10, CONST_255_16x4, d3210);

    // if(inp_centered <= -radius) output = 0
    AE_MOVT16X4(z10, AE_ZERO16(), b3210);

#if XCHAL_HAVE_HIFI1
    m0 = AE_SAT8U(z10);
#else
    m0 = z10;
    xtbool4 bsat4 = AE_LT16(CONST_255_16x4, m0);
    AE_MOVT16X4(m0, CONST_255_16x4, bsat4);
#endif

    SUB_128(m0)

#if XCHAL_HAVE_HIFI1
    AE_S8_0_IP_HIFI1(m0, p_o, sizeof(WORD8));
#else
    WORD16 m0_out8 = m0;
    *p_o++ = (WORD8)m0_out8;
#endif
  }

  return 0;
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
#if XCHAL_HAVE_HIFI1
    ae_valign align_out = AE_ZALIGN64();
#endif

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
#if XCHAL_HAVE_HIFI1
            AE_L8X4U_IP(y, p_v_t, 4*sizeof(WORD8));
            AE_SA8X4U_IP(y, align_out, (ae_int32*)p_o);
#else
            AE_L8X4F_IP(x, p_v_t, 4*sizeof(WORD8));
            y = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x), 8));
            STORE_8X4_FROM_16X4(p_o, y)
#endif
        }
#if XCHAL_HAVE_HIFI1_
        AE_SA64POS_FP(align_out, p_o);
#endif
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
#if XCHAL_HAVE_HIFI1
            AE_L8X4U_IP(y, p_v_t, 4*sizeof(WORD8));
            MIN_16X4(y, max);
            AE_SA8X4U_IP(y, align_out, (ae_int32*)p_o);
#else
            AE_L8X4F_IP(x, p_v_t, 4*sizeof(WORD8));
            y = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x), 8));
            MIN_16X4(y, max);
            STORE_8X4_FROM_16X4(p_o, y)
#endif
        }
#if XCHAL_HAVE_HIFI1
        AE_SA64POS_FP(align_out, p_o);
#endif

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
#if XCHAL_HAVE_HIFI1
            AE_L8X4U_IP(y, p_v_t, 4*sizeof(WORD8));
            MAX_16X4(y, min)
            AE_SA8X4U_IP(y, align_out, (ae_int32*)p_o);
#else
            AE_L8X4F_IP(x, p_v_t, 4*sizeof(WORD8));
            y = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x), 8));
            MAX_16X4(y, min)
            STORE_8X4_FROM_16X4(p_o, y)
#endif
        }
#if XCHAL_HAVE_HIFI1
        AE_SA64POS_FP(align_out, p_o);
#endif

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
#if XCHAL_HAVE_HIFI1
            AE_L8X4U_IP(x, p_v_t, 4*sizeof(WORD8));
            LIMIT(y, x, min, max)
            AE_SA8X4U_IP(y, align_out, (ae_int32*)p_o);
#else
            AE_L8X4F_IP(x, p_v_t, 4*sizeof(WORD8));
            x = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x), 8));
            LIMIT(y, x, min, max)
            STORE_8X4_FROM_16X4(p_o, y)
#endif
        }
#if XCHAL_HAVE_HIFI1
        AE_SA64POS_FP(align_out, p_o);
#endif

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
  int i;

#if !XCHAL_HAVE_HIFI1  
  xtbool4 b0;
#endif

#if XCHAL_HAVE_HIFI1
  ae_valign align_dst;
  align_dst = AE_ZALIGN64();
#endif

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < 0) || (inp_zero_bias > 255)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < 0) || (out_zero_bias > 255)), -1);
  XA_NNLIB_ARG_CHK_COND(((quantized_activation_min < 0) || (quantized_activation_min > 255)), -1);
  XA_NNLIB_ARG_CHK_COND(((quantized_activation_max < 0) || (quantized_activation_max > 255)), -1);
  XA_NNLIB_ARG_CHK_COND((quantized_activation_max < quantized_activation_min), -1);

  int rem_length = (vec_length & 3);

  WORD8 *p_o = (WORD8 *)p_out;
  WORD8 *p_v = (WORD8 *)p_vec;

  ae_int16x4 inp_zb = AE_MOVDA16(inp_zero_bias);
  ae_int16x4 act_min = AE_MOVDA16(quantized_activation_min);
  ae_int16x4 act_max = AE_MOVDA16(quantized_activation_max);
  ae_int16x4 one = AE_MOVDA16(1);
  
#if TFLITE_SINGLE_ROUNDING
    int left_shift  = out_shift;
    int right_shift = out_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    int left_shift  = out_shift<0?0: out_shift;
    int right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ALIGN_REGISTER_TYPE align_src;
  PRIME_8X4U(p_v, align_src);

  for(i=0; i<(vec_length >> 2); i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_v0_0;
 
    ae_int32x2 d_w0_0, d_w0_1;

    AE_LA8X4U_IP(d_inp0, align_src, p_v);

    d_v0_0 = AE_SUB16S(d_inp0, inp_zb); 
    // Multiply with out multiplier
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 
    ae_int16x4 out0, out_minmax;
    MPY_BY_QUANT_MULT_X2X2_OUT16(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift); 
    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    //Clamp the output in the quantized activation range
    LIMIT(out_minmax, out0, act_min, act_max)
#if XCHAL_HAVE_HIFI1
    AE_SA8X4U_IP(out_minmax, align_dst, (ae_int32 *)p_o);
#else
     STORE_8X4_FROM_16X4(p_o, out_minmax);
#endif
  }

#if XCHAL_HAVE_HIFI1
  AE_SA64POS_FP(align_dst, p_o);
#endif

  //remainder loop
  for(i = 0; i<rem_length; i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_v0_0;
 
    ae_int32x2 d_w0_0, d_w0_1;
   
    d_inp0 = AE_MOVDA16((UWORD8)*p_v++);
    d_v0_0 = AE_SUB16S(d_inp0, inp_zb); 
    // Multiply with out multiplier
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 
    ae_int16x4 out0, out_minmax;
    MPY_BY_QUANT_MULT_X2X2_OUT16(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift); 
    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    //Clamp the output in the quantized activation range
    LIMIT(out_minmax, out0, act_min, act_max)
    WORD16 out8_0 = out_minmax;
    *p_o++ = (WORD8)out8_0;
  }

  return 0;
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
  int i;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((quantized_activation_min < -128) || (quantized_activation_min > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((quantized_activation_max < -128) || (quantized_activation_max > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((quantized_activation_max < quantized_activation_min), -1);

  int rem_length = (vec_length & 3);

  WORD8 *p_o = p_out;
  WORD8 *p_v = (WORD8 *)p_vec;

  ae_int16x4 inp_zb = AE_MOVDA16(inp_zero_bias);
  ae_int16x4 act_min = AE_MOVDA16(quantized_activation_min);
  ae_int16x4 act_max = AE_MOVDA16(quantized_activation_max);
  ae_int16x4 one = AE_MOVDA16(1);
  
#if TFLITE_SINGLE_ROUNDING
    int left_shift  = out_shift;
    int right_shift = out_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    int left_shift  = out_shift<0?0: out_shift;
    int right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ALIGN_REGISTER_TYPE align_src;
  PRIME_8X4F(p_v, align_src);
#if XCHAL_HAVE_HIFI1
  ALIGN_REGISTER_TYPE align_dst = AE_ZALIGN64();
#else
  xtbool4 b0;
#endif

  for(i=0; i<(vec_length >> 2); i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_v0_0;
 
    ae_int32x2 d_w0_0, d_w0_1;

#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(d_inp0, align_src, p_v);
#else
    AE_LA8X4F_IP(d_inp0, align_src, p_v);
    d_inp0 = AE_SRAI16(d_inp0, 8);
#endif

    d_v0_0 = AE_SUB16S(d_inp0, inp_zb); 
    // Multiply with out multiplier
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 
    ae_int16x4 out0, out_minmax;
    MPY_BY_QUANT_MULT_X2X2_OUT16(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift);
    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    //Clamp the output in the quantized activation range
    LIMIT(out_minmax, out0, act_min, act_max)
#if XCHAL_HAVE_HIFI1
    ae_int16x4 out8_0 = AE_SAT8S(out_minmax);
    AE_SA8X4U_IP(out8_0, align_dst, (ae_int32*)p_o);
#else
    STORE_8X4_FROM_16X4(p_o, out_minmax);
#endif
  }
#if XCHAL_HAVE_HIFI1
  AE_SA64POS_FP(align_dst, p_o);
#endif
  //remainder loop
  for(i = 0; i<rem_length; i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_v0_0;
 
    ae_int32x2 d_w0_0, d_w0_1;

#if XCHAL_HAVE_HIFI1
    AE_L8S_IP(d_inp0, p_v, sizeof(WORD8)); 
#else
     d_inp0 = (WORD16)*p_v++;
#endif
    d_v0_0 = AE_SUB16S(d_inp0, inp_zb); 
    // Multiply with out multiplier
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 
    ae_int16x4 out0, out_minmax;
    MPY_BY_QUANT_MULT_X2X2_OUT16(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift);
    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    //Clamp the output in the quantized activation range
    LIMIT(out_minmax, out0, act_min, act_max)
#if XCHAL_HAVE_HIFI1
    ae_int16x4 out8_0 = AE_SAT8S(out_minmax);
    AE_S8_0_IP_HIFI1(out8_0, p_o, sizeof(WORD8));
#else
    WORD16 out8_0 = out_minmax;
    *p_o++ = (WORD8)out8_0;
#endif
  }

  return 0;
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
  int i;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec_alpha, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -127) || (inp_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((alpha_zero_bias < -127) || (alpha_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((alpha_shift < -31) || (alpha_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((alpha_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);

  int rem_length = (vec_length & 3);

  WORD8 *p_o = p_out;
  WORD8 *p_v = (WORD8 *)p_vec;
  WORD8 *p_v_a = (WORD8 *)p_vec_alpha;

  ae_int16x4 inp_zb = AE_MOVDA16(-inp_zero_bias);
  ae_int16x4 alpha_zb = AE_MOVDA16(-alpha_zero_bias);
  ae_int16x4 one = AE_MOVDA16(1);
  ae_int16x4 zero = AE_ZERO16();
  
#if TFLITE_SINGLE_ROUNDING
    int left_shift  = out_shift;
    int right_shift = out_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift;

    int a_left_shift  = alpha_shift;
    int a_right_shift = alpha_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)a_right_shift;

#else /* #if TFLITE_SINGLE_ROUNDING */
    int left_shift  = out_shift<0?0: out_shift;
    int right_shift = out_shift>0?0:-out_shift;

    int a_left_shift  = alpha_shift<0?0: alpha_shift;
    int a_right_shift = alpha_shift>0?0:-alpha_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ALIGN_REGISTER_TYPE align_src, align_src1;
  PRIME_8X4F(p_v, align_src);
  PRIME_8X4F(p_v_a, align_src1);
#if XCHAL_HAVE_HIFI1
  ALIGN_REGISTER_TYPE align_dst = AE_ZALIGN64();
#else
  ae_int16x4 CONST_127_16x4 = AE_MOVDA16(127);
  ae_int16x4 CONST_MINUS_128_16x4 = AE_MOVDA16(-128);
#endif

  for(i=0; i<(vec_length >> 2); i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_a_inp0;
    ae_int16x4 d_v0_0;
    ae_int32x2 d_w0_0, d_w0_1;
    
#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(d_inp0, align_src, p_v);
    AE_LA8X4S_IP(d_a_inp0, align_src1, p_v_a);
#else
    AE_LA8X4F_IP(d_inp0, align_src, p_v);
    d_inp0 = AE_SRAI16(d_inp0, 8);
    AE_LA8X4F_IP(d_a_inp0, align_src1, p_v_a);
    d_a_inp0 = AE_SRAI16(d_a_inp0, 8);
#endif
    
    d_v0_0 = AE_SUB16S(d_inp0, inp_zb);  

    //Checking for input values less than inp_zero_bias
    xtbool4 sel0 = AE_LT16(d_v0_0, zero);
    
    // Multiply with out multiplier for input values >= 0
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 

    ae_int16x4 out0;
    MPY_BY_QUANT_MULT_X2X2_OUT16(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift);

    // Add alpha zero bias and multiply with alpha multiplier for input values < 0
    ae_int32x2 d_alpha_w0_0, d_alpha_w0_1;

    ae_int16x4 d_alpha_v0_0;
    d_alpha_v0_0 = AE_SUB16S(d_a_inp0, alpha_zb); 
   
    AE_MUL16X4(d_alpha_w0_0, d_alpha_w0_1, d_v0_0, d_alpha_v0_0);

    ae_int16x4 a_out0;
    MPY_BY_QUANT_MULT_X2X2_OUT16(a_out0, d_alpha_w0_0, d_alpha_w0_1, alpha_multiplier, a_left_shift, a_right_shift);

    AE_MOVT16X4(out0, a_out0, sel0);
    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
  
#if XCHAL_HAVE_HIFI1
    ae_int16x4 out8_0 = AE_SAT8S(out0);
    AE_SA8X4U_IP(out8_0, align_dst, (ae_int32*)p_o);
#else
    xtbool4 bsat4 = AE_LT16(CONST_127_16x4, out0);
    AE_MOVT16X4(out0, CONST_127_16x4 , bsat4);
    bsat4 = AE_LT16(out0, CONST_MINUS_128_16x4);
    AE_MOVT16X4(out0, CONST_MINUS_128_16x4 , bsat4);
    STORE_8X4_FROM_16X4(p_o, out0);
#endif
  }
#if XCHAL_HAVE_HIFI1
   AE_SA64POS_FP(align_dst, p_o);
#endif

  //remainder loop
  for(i=0; i<rem_length; i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_a_inp0;
    ae_int16x4 d_v0_0;
    ae_int32x2 d_w0_0, d_w0_1;
    
#if XCHAL_HAVE_HIFI1
    //AE_LA8X4S_IP(d_inp0, align_src, p_v);
    AE_L8S_IP(d_inp0, p_v, sizeof(WORD8)); 
    //AE_LA8X4S_IP(d_a_inp0, align_src1, p_v_a);
    AE_L8S_IP(d_a_inp0, p_v_a, sizeof(WORD8));
#else
     d_inp0 = (WORD16)*p_v++;
     d_a_inp0 = (WORD16)*p_v_a++;
#endif
    
    d_v0_0 = AE_SUB16S(d_inp0, inp_zb);  

    //Checking for input values less than inp_zero_bias
    xtbool4 sel0 = AE_LT16(d_v0_0, zero);
    
    // Multiply with out multiplier for input values >= 0
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 

    ae_int16x4 out0;
    MPY_BY_QUANT_MULT_X2X2_OUT16(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift);

    // Add alpha zero bias and multiply with alpha multiplier for input values < 0
    ae_int32x2 d_alpha_w0_0, d_alpha_w0_1;

    ae_int16x4 d_alpha_v0_0;
    d_alpha_v0_0 = AE_SUB16S(d_a_inp0, alpha_zb); 
   
    AE_MUL16X4(d_alpha_w0_0, d_alpha_w0_1, d_v0_0, d_alpha_v0_0);

    ae_int16x4 a_out0;
    MPY_BY_QUANT_MULT_X2X2_OUT16(a_out0, d_alpha_w0_0, d_alpha_w0_1, alpha_multiplier, a_left_shift, a_right_shift);
    AE_MOVT16X4(out0, a_out0, sel0);
    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
  
#if XCHAL_HAVE_HIFI1
    ae_int16x4 out8_0 = AE_SAT8S(out0);
    AE_S8_0_IP_HIFI1(out8_0, p_o, sizeof(WORD8));
#else
    xtbool4 bsat4 = AE_LT16(CONST_127_16x4, out0);
    AE_MOVT16X4(out0, CONST_127_16x4 , bsat4);
    bsat4 = AE_LT16(out0, CONST_MINUS_128_16x4);
    AE_MOVT16X4(out0, CONST_MINUS_128_16x4 , bsat4);
    WORD16 out8_0 = out0;
    *p_o++ = (WORD8)out8_0;
#endif
  }

  return 0;
}

WORD32 xa_nn_vec_leaky_relu_asym8s_asym8s( WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_vec,
                            WORD32 inp_zero_bias,
                            WORD32 alpha_multiplier,
                            WORD32 alpha_shift,
                            WORD32 out_multiplier,
                            WORD32 out_shift,
                            WORD32 out_zero_bias,
                            WORD32 vec_length)
{
  int i;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((alpha_shift < -31) || (alpha_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((alpha_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);

  int rem_length = (vec_length & 3);

  WORD8 *p_o = p_out;
  WORD8 *p_v = (WORD8 *)p_vec;

  ae_int16x4 inp_zb = AE_MOVDA16(inp_zero_bias);
  ae_int16x4 one = AE_MOVDA16(1);
  ae_int16x4 zero = AE_ZERO16();

#if TFLITE_SINGLE_ROUNDING
    int left_shift  = out_shift;
    int right_shift = out_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift;

    int a_left_shift  = alpha_shift;
    int a_right_shift = alpha_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)a_right_shift;

#else /* #if TFLITE_SINGLE_ROUNDING */
    int left_shift  = out_shift<0?0: out_shift;
    int right_shift = out_shift>0?0:-out_shift;

    int a_left_shift  = alpha_shift<0?0: alpha_shift;
    int a_right_shift = alpha_shift>0?0:-alpha_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

#if XCHAL_HAVE_HIFI1
  ae_valign align_src  = AE_LA64_PP((ae_int16x4 *)p_v);
  ae_valign align_dst = AE_ZALIGN64();
#else
  ae_int16x4 CONST_127_16x4 = AE_MOVDA16(127);
  ae_int16x4 CONST_MINUS_128_16x4 = AE_MOVDA16(-128);
  ALIGN_REGISTER_TYPE align_src;
  PRIME_8X4F(p_v, align_src);
#endif

#pragma concurrent
  for(i=0; i<(vec_length >> 2); i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_v0_0;
    ae_int32x2 d_w0_0, d_w0_1;
    ae_int32x2 d_alpha_w0_0, d_alpha_w0_1;

#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(d_inp0, align_src, p_v);
#else
    AE_LA8X4F_IP(d_inp0, align_src, p_v);
    d_inp0 = AE_SRAI16(d_inp0, 8);
#endif

    d_v0_0 = AE_SUB16(d_inp0, inp_zb);

    // Multiply with out multiplier for input values >= 0
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one);

    d_alpha_w0_0 = d_w0_0; d_alpha_w0_1 = d_w0_1;

    ae_int16x4 out0;
    MPY_BY_QUANT_MULT_X2X2_OUT16(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift);

    // Multiply with alpha multiplier for input values < 0
    ae_int16x4 a_out0;
    MPY_BY_QUANT_MULT_X2X2_OUT16(a_out0, d_alpha_w0_0, d_alpha_w0_1, alpha_multiplier, a_left_shift, a_right_shift);

    //Checking for input values less than zero
    xtbool4 sel0 = AE_LT16(d_v0_0, zero);
    AE_MOVT16X4(out0, a_out0, sel0);

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);

#if XCHAL_HAVE_HIFI1
    ae_int16x4 out8_0 = AE_SAT8S(out0);
    AE_SA8X4U_IP(out8_0, align_dst, (ae_int32 *)p_o);
#else
    xtbool4 bsat4 = AE_LT16(CONST_127_16x4, out0);
    AE_MOVT16X4(out0, CONST_127_16x4 , bsat4);
    bsat4 = AE_LT16(out0, CONST_MINUS_128_16x4);
    AE_MOVT16X4(out0, CONST_MINUS_128_16x4 , bsat4);
    STORE_8X4_FROM_16X4(p_o, out0);
#endif
  }
#if XCHAL_HAVE_HIFI1
  AE_SA64POS_FP(align_dst, p_o);
#endif

  //remainder loop for 3 elms
  for(i=0; i<rem_length; i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_v0_0;
    ae_int32x2 d_w0_0, d_w0_1;
    ae_int32x2 d_alpha_w0_0, d_alpha_w0_1;

#if XCHAL_HAVE_HIFI1
    AE_L8S_IP(d_inp0, (WORD8*)p_v, 1);
#else
     d_inp0 = (WORD16)*p_v++;
#endif

    d_v0_0 = AE_SUB16(d_inp0, inp_zb);

    //Checking for input values less than zero
    xtbool4 sel0 = AE_LT16(d_v0_0, zero);

    // Multiply with out multiplier for input values >= 0
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one);

    d_alpha_w0_0 = d_w0_0; d_alpha_w0_1 = d_w0_1;

    ae_int16x4 out0;
    MPY_BY_QUANT_MULT_X2X2_OUT16(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift);

    // Multiply with alpha multiplier for input values < 0
    ae_int16x4 a_out0;
    MPY_BY_QUANT_MULT_X2X2_OUT16(a_out0, d_alpha_w0_0, d_alpha_w0_1, alpha_multiplier, a_left_shift, a_right_shift);

    AE_MOVT16X4(out0, a_out0, sel0);

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);

#if XCHAL_HAVE_HIFI1
    ae_int16x4 out8_0 = AE_SAT8S(out0);
    AE_S8_0_IP_HIFI1(out8_0, (WORD8 *)p_o, 1);
#else
    xtbool4 bsat4 = AE_LT16(CONST_127_16x4, out0);
    AE_MOVT16X4(out0, CONST_127_16x4 , bsat4);
    bsat4 = AE_LT16(out0, CONST_MINUS_128_16x4);
    AE_MOVT16X4(out0, CONST_MINUS_128_16x4 , bsat4);
    WORD16 out8_0 = out0;
    *p_o++ = (WORD8)out8_0;
#endif
  }

  return 0;
}

static inline ae_int16x4 AE_SRAA16SYMS_LE(ae_int16x4 x, WORD32 shift)
{
    ae_int32x2 xh, xl;
    xh = AE_CVT32X2F16_32(x);
    xl = AE_CVT32X2F16_10(x);
    xh = AE_SRAA32RS(xh, shift);
    xl = AE_SRAA32RS(xl, shift);
    x = AE_ROUND16X4F32SSYM(xh, xl);
    
    return x;
}
#if XCHAL_HAVE_HIFI1
#define MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(inp1, multiplier, left_shift, right_shift, ext_shift) \
{\
  ae_int32x2 inp1_32h, inp1_32l;\
  AE_MUL16X4(inp1_32h, inp1_32l, inp1, AE_MOVDA16(left_shift)); \
  inp1 = AE_SAT16X4(inp1_32h, inp1_32l);\
  inp1 = AE_MULFP16X4RAS(inp1, multiplier); \
  AE_MUL16X4(inp1_32h, inp1_32l, inp1, AE_MOVDA16(ext_shift)); \
  inp1 = AE_SAT16X4(inp1_32h, inp1_32l);\
  inp1 = AE_SRAA16SYMS_LE(inp1, right_shift); \
}
#else
#define MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(inp1, multiplier, left_shift, right_shift, ext_shift) \
{\
  inp1 = AE_SLAA16S(inp1, left_shift); \
  inp1 = AE_MULFP16X4RAS(inp1, multiplier); \
  inp1 = AE_SLAA16S(inp1, ext_shift); \
  inp1 = AE_SRAA16SYMS_LE(inp1, right_shift); \
}
#endif

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
  int i;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((reluish_shift < -31) || (reluish_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((reluish_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);

  int rem_length = (vec_length & 3);

  WORD8 *p_o = p_out;
  WORD8 *p_v = (WORD8 *)p_vec;

  ae_int16x4 inp_zb = AE_MOVDA16(inp_zero_bias);
  ae_int16x4 hires_mul = AE_MOVDA16(128);
  
  int r_left_shift  = reluish_shift > 0 ? reluish_shift-1 : 0;
  int ext_lsh = reluish_shift > 0 ? 1 : 0;
  int r_right_shift = reluish_shift>0?0:-reluish_shift;

  /* Second operand for XOR instruction for ADD_32768*/
  ae_int64 offset_xor = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(32768));
  
  ALIGN_REGISTER_TYPE align_src;
  PRIME_8X4F(p_v, align_src);
#if XCHAL_HAVE_HIFI1
  ALIGN_REGISTER_TYPE align_dst = AE_ZALIGN64();
  r_left_shift = (1 << r_left_shift);
  ext_lsh = (1 << ext_lsh);
#else
  ae_int16x4 CONST_127_16x4 = AE_MOVDA16(127);
  ae_int16x4 CONST_MINUS_128_16x4 = AE_MOVDA16(-128);
#endif
  for(i=0; i<(vec_length >> 2); i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_v0_0;
    ae_int16x4 d_r0_0;
    ae_int16x4 d_w0_0;
    
#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(d_inp0, align_src, p_v);
#else
    AE_LA8X4F_IP(d_inp0, align_src, p_v);
    d_inp0 = AE_SRAI16(d_inp0, 8);
#endif
    
    d_v0_0 = AE_SUB16S( d_inp0, inp_zb); 

    //Shifting the result to MSB bits
    ae_int32x2 d_r0_0_h, d_r0_0_l;
    AE_MUL16X4(d_r0_0_h, d_r0_0_l, d_v0_0, hires_mul);
    d_r0_0 = AE_SAT16X4(d_r0_0_h, d_r0_0_l);

    // Multiply shifted result with out multiplier 
    d_w0_0 = AE_MULFP16X4RAS(d_r0_0, out_multiplier); 
    // Multiply the shifted result with reluish multiplier and apply reluish shift
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(d_r0_0, AE_MOVDA16(reluish_multiplier), r_left_shift, r_right_shift, ext_lsh); 

    // Bring the output in [0,1] range from [-1, 1] range.
    d_r0_0 = AE_MOVINT16X4_FROMINT64(AE_XOR(AE_MOVINT64_FROMINT16X4(d_r0_0), offset_xor)); 
    d_r0_0 = AE_SRAA16S(d_r0_0, 1);
    d_r0_0 = AE_AND16(d_r0_0, AE_MOVDA16(0x7fff));


    // Multiply inp*relu6(inp+3)
    ae_int16x4 out0, out0_rsh_val;
    out0 = AE_MULFP16X4S(d_w0_0, d_r0_0); 
    out0_rsh_val = AE_SRAI16(out0, 15);
    out0_rsh_val = AE_AND16(out0_rsh_val, AE_MOVDA16(1));
    out0 = AE_ADD16S(out0, out0_rsh_val);
    out0 = AE_SRAA16SYMS_LE(out0, -out_shift); 

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
  
#if XCHAL_HAVE_HIFI1
    ae_int16x4 out8_0 = AE_SAT8S(out0);
    AE_SA8X4U_IP(out8_0, align_dst, (ae_int32* )p_o);
#else
    xtbool4 bsat4 = AE_LT16(CONST_127_16x4, out0);
    AE_MOVT16X4(out0, CONST_127_16x4 , bsat4);
    bsat4 = AE_LT16(out0, CONST_MINUS_128_16x4);
    AE_MOVT16X4(out0, CONST_MINUS_128_16x4 , bsat4);
    STORE_8X4_FROM_16X4(p_o, out0);
#endif
  }
#if XCHAL_HAVE_HIFI1
   AE_SA64POS_FP(align_dst, p_o);
#endif
  //remainder loop
  for(i = 0; i< rem_length; i++)
  {
    ae_int16x4 d_inp0;
    ae_int16x4 d_v0_0;
    ae_int16x4 d_r0_0;
    ae_int16x4 d_w0_0;
    
#if XCHAL_HAVE_HIFI1
    AE_L8S_IP(d_inp0, p_v, sizeof(WORD8));
#else
    d_inp0 = (WORD16)*p_v++;
#endif
    
    d_v0_0 = AE_SUB16S( d_inp0, inp_zb); 

    //Shifting the result to MSB bits
    ae_int32x2 d_r0_0_h, d_r0_0_l;
	AE_MUL16X4(d_r0_0_h, d_r0_0_l, d_v0_0, hires_mul);
	d_r0_0 = AE_SAT16X4(d_r0_0_h, d_r0_0_l);

    // Multiply shifted result with out multiplier 
    d_w0_0 = AE_MULFP16X4RAS(d_r0_0, out_multiplier); 
    // Multiply the shifted result with reluish multiplier and apply reluish shift
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(d_r0_0, AE_MOVDA16(reluish_multiplier), r_left_shift, r_right_shift, ext_lsh); 

    // Bring the output in [0,1] range from [-1, 1] range.
    d_r0_0 = AE_MOVINT16X4_FROMINT64(AE_XOR(AE_MOVINT64_FROMINT16X4(d_r0_0), offset_xor)); 
    //d_r0_0 = AE_SRLI16(d_r0_0, 1);
    d_r0_0 = AE_SRAA16S(d_r0_0, 1);
    d_r0_0 = AE_AND16(d_r0_0, AE_MOVDA16(0x7fff));

    // Multiply inp*relu6(inp+3)
    ae_int16x4 out0, out0_rsh_val;
    out0 = AE_MULFP16X4S(d_w0_0, d_r0_0); 
    out0_rsh_val = AE_SRAI16(out0, 15);
    out0_rsh_val = AE_AND16(out0_rsh_val, AE_MOVDA16(1));
    out0 = AE_ADD16S(out0, out0_rsh_val);
    out0 = AE_SRAA16SYMS_LE(out0, -out_shift); 

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
  
#if XCHAL_HAVE_HIFI1
    ae_int16x4 out8_0 = AE_SAT8S(out0);
    AE_S8_0_IP_HIFI1(out8_0, p_o, sizeof(WORD8));
#else
    xtbool4 bsat4 = AE_LT16(CONST_127_16x4, out0);
    AE_MOVT16X4(out0, CONST_127_16x4 , bsat4);
    bsat4 = AE_LT16(out0, CONST_MINUS_128_16x4);
    AE_MOVT16X4(out0, CONST_MINUS_128_16x4 , bsat4);
    WORD16 out8_0 = out0;
    *p_o++ = (WORD8)out8_0;
#endif
  }

  return 0;
}

//output: y1, y2 (ae_int32x2)
//input:  a1, a2 (ae_int32x2)
#define ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y1, y2, a1, a2){\
  ae_int64 s1, s2, s3, s4;\
  ae_int64 t1, t2, t3, t4;\
  ae_int32x2 half_den12, m1, x1, half_denominator_times_x1;\
  ae_int32x2 half_den34, m2, x2, half_denominator_times_x2;\
  ae_int32x2 one_minus_half_denominator_times_x1;\
  ae_int32x2 one_minus_half_denominator_times_x2;\
  ae_int32x2 CT_48_by_7, CT_neg_32_by_7, CT_F2_ONE;\
  int j;\
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
  m1  = AE_MULFP32X2RAS(half_den12, CT_neg_32_by_7);\
  m2  = AE_MULFP32X2RAS(half_den34, CT_neg_32_by_7);\
  x1 = AE_ADD32S(m1, CT_48_by_7);\
  x2 = AE_ADD32S(m2, CT_48_by_7);\
\
  for(j=0; j<3; j++)\
  {\
    half_denominator_times_x1 = AE_MULFP32X2RAS(x1, half_den12);\
    half_denominator_times_x2 = AE_MULFP32X2RAS(x2, half_den34);\
    one_minus_half_denominator_times_x1 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x1);\
    one_minus_half_denominator_times_x2 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x2);\
    m1 = AE_MULFP32X2RAS(x1, one_minus_half_denominator_times_x1);\
    m2 = AE_MULFP32X2RAS(x2, one_minus_half_denominator_times_x2);\
    m1 = AE_SLAI32S(m1, 2);\
    x1 = AE_ADD32S(x1, m1);\
  \
    m2 = AE_SLAI32S(m2, 2);\
    x2 = AE_ADD32S(x2, m2);\
  \
  }\
\
  x1 = AE_SUB32S(x1, CT_F2_ONE);\
  x2 = AE_SUB32S(x2, CT_F2_ONE);\
  y1 = AE_SLAI32S(x1, 2);\
  y2 = AE_SLAI32S(x2, 2);\
\
}

WORD32 xa_nn_vec_tanh_asym8s_asym8s(WORD8 *p_out,
                      const WORD8 *p_vec,
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((zero_point < -128) || (zero_point > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CLIP(input_left_shift, -31, 31);
  XA_NNLIB_ARG_CHK_COND((input_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_range_radius < 0), -1);

  /* Limit the input_range_radius value to int16, as we use ae_int16x4 data types for comparison */
  input_range_radius = (input_range_radius>32767) ? 32767 : input_range_radius;

#if defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4)
  WUR_AE_SAR(4);
#else
  ae_int32x2 mask_6fs = AE_MOVDA32(MASK);
  ae_int32x2 q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
  ae_int32x2 CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
  ae_int32x2 CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
  ae_int32x2 CT = AE_MOVDA32(CONSTANT_TERM);
  ae_int32x2 ONE = AE_MOVDA32(1);
  xtbool4 f3210;
  ae_int32x2 exp_x32, exp_x10;
  ae_int32x2 m32, m10;
#endif /* defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4) */
  int i;
  int rem_length = (vec_length & 3);
  ae_int32x2 x32, x10;
  ae_int32x2 /*z,*/ mul, zero;
  ae_int16x4 CONST_127_16x4 = AE_MOVDA16(127);
  ae_int16x4 CONST_MINUS_128_16x4 = AE_MOVDA16(-128);
//  ae_int32x2 radius, minus_radius;
  ae_int16x4 radius_16, minus_radius_16;
  xtbool4 b3210, d3210;
  ae_int32x2  dequantized_x32, dequantized_x10;
  ae_int16x4 m0, z_16x4;
  ae_int16x4 z10, zero_16x4;

  WORD8 *p_in  = (WORD8 *)p_vec;
  WORD8 *p_o = (WORD8 *)p_out;

#if XCHAL_HAVE_HIFI1
  ae_valign align_src, align_dst;
  align_src = AE_LA64_PP(p_in);
  align_dst = AE_ZALIGN64();
#else
  ALIGN_REGISTER_TYPE align_src;
  PRIME_8X4F(p_in, align_src);
#endif

//  radius = AE_MOVDA32(input_range_radius);
//  minus_radius = AE_NEG32(radius);

  radius_16 = AE_MOVDA16(input_range_radius);
  minus_radius_16 = AE_NEG16S(radius_16);

  //z = AE_MOVDA32(zero_point);
  z_16x4 = AE_MOVDA16(zero_point);
  mul = AE_MOVDA32(input_multiplier);
  zero = AE_ZERO32();
  zero_16x4 = AE_ZERO16();

#if TFLITE_SINGLE_ROUNDING
    int left_shift  = input_left_shift;
    int right_shift = input_left_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    int left_shift  = input_left_shift<0?0: input_left_shift;
    int right_shift = input_left_shift>0?0:-input_left_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  for(i=0; i<(vec_length >> 2); i++)
  {
#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(m0, align_src, p_in);
#else
    AE_LA8X4F_IP(m0, align_src, p_in);
    m0 = AE_SRAI16(m0, 8);
#endif
    z10 = AE_SUB16(m0, z_16x4);

    // set flag if z <= minus_radius
    b3210 = AE_LE16(z10, minus_radius_16);

    // set flag if z < radius
    d3210 = AE_LT16(z10, radius_16);

    x32 = AE_SEXT32X2D16_32(z10);
    x10 = AE_SEXT32X2D16_10(z10);

    MPY_BY_QUANT_MULT_X2_OUT32(dequantized_x32, x32, mul, left_shift, right_shift)
    MPY_BY_QUANT_MULT_X2_OUT32(dequantized_x10, x10, mul, left_shift, right_shift)

#if defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4)
    (void)zero_16x4; (void)zero;
    x32 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(dequantized_x32), 15), AE_SRAA64(AE_CVT64F32_L(dequantized_x32), 15));
    x10 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(dequantized_x10), 15), AE_SRAA64(AE_CVT64F32_L(dequantized_x10), 15));

    z10 = AE_SAT16X4(x32, x10);

    z10 = AE_TANH16X4(z10);

    z10 = AE_SRAI16(z10, 8);
#else
    //set flag if z < 0
    f3210 = AE_LT16(z10, zero_16x4);

    // Computing Absolute value
    x32 = AE_ABS32(dequantized_x32);
    x10 = AE_ABS32(dequantized_x10);
    x32 = AE_NEG32(x32);
    x10 = AE_NEG32(x10);

    // Compute tanh i.e. one_minus_x_over_one_plus_x(exp(-2x))
    x32 = AE_SLAI32S(x32, 1);
    x10 = AE_SLAI32S(x10, 1);

    EXP_Q26_S(exp_x32, x32);
    EXP_Q26_S(exp_x10, x10);

    ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x32, x10, exp_x32, exp_x10)

    // Downscale to 8 bit
    m32 = AE_SRAA32RS(x32, 24);
    m10 = AE_SRAA32RS(x10, 24);
    // Due to rounding operation, sometimes value gets set to 128.
    // We need to saturate it to 127.
    // SAT8X4 used before store operation takes care of this.

    z10 = AE_CVT16X4(m32, m10);

    // if(inp_centered < 0) output = - tanh(abs(dequantized_input))
    AE_MOVT16X4(z10, AE_NEG16S(z10), f3210);
#endif /* defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4) */

    // if(inp_centered >= radius) output = 127
    AE_MOVF16X4(z10, CONST_127_16x4, d3210);

    // if(inp_centered <= -radius) output = -128
    AE_MOVT16X4(z10, CONST_MINUS_128_16x4, b3210);

#if XCHAL_HAVE_HIFI1
    m0 = AE_SAT8S(z10);
    AE_SA8X4U_IP(m0, align_dst, (ae_int32*)p_o);
#else
    m0 = z10;
    xtbool4 bsat4 = AE_LT16(CONST_127_16x4, m0);
    AE_MOVT16X4(m0, CONST_127_16x4 , bsat4);
    bsat4 = AE_LT16(m0, CONST_MINUS_128_16x4);
    AE_MOVT16X4(m0, CONST_MINUS_128_16x4 , bsat4);

    STORE_8X4_FROM_16X4(p_o, m0);
#endif
  }
#if XCHAL_HAVE_HIFI1
  AE_SA64POS_FP(align_dst, p_o);
#endif

  // remainder loop
  for(i=0; i< rem_length; i++)
  {
#if XCHAL_HAVE_HIFI1
    AE_L8S_IP(m0, p_in, sizeof(WORD8));
#else
    m0 = (WORD16)*p_in++;
#endif
    z10 = AE_SUB16(m0, z_16x4);

    // set flag if z <= minus_radius
    b3210 = AE_LE16(z10, minus_radius_16);

    // set flag if z < radius
    d3210 = AE_LT16(z10, radius_16);

    x10 = AE_SEXT32X2D16_10(z10);

    MPY_BY_QUANT_MULT_X2_OUT32(dequantized_x10, x10, mul, left_shift, right_shift)

#if defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4)
    x10 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(dequantized_x10), 15), AE_SRAA64(AE_CVT64F32_L(dequantized_x10), 15));

    z10 = AE_SAT16X4(x10, x10);

    z10 = AE_TANH16X4(z10);

    z10 = AE_SRAI16(z10, 8);
#else
    //set flag if z < 0
    f3210 = AE_LT16(z10, zero_16x4);

    // Computing Absolute value
    x10 = AE_ABS32(dequantized_x10);
    x10 = AE_NEG32(x10);

    // Compute tanh i.e. one_minus_x_over_one_plus_x(exp(-2x))
    x10 = AE_SLAI32S(x10, 1);

    EXP_Q26_S(exp_x10, x10);

    exp_x32 = exp_x10;
    ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x32, x10, exp_x32, exp_x10)

    // Downscale to 8 bit
    m10 = AE_SRAA32RS(x10, 24);
    // Due to rounding operation, sometimes value gets set to 128.
    // We need to saturate it to 127.
    // SAT8X4 used before store operation takes care of this.

    z10 = AE_CVT16X4(m10, m10);

    // if(inp_centered < 0) output = - tanh(abs(dequantized_input))
    AE_MOVT16X4(z10, AE_NEG16S(z10), f3210);
#endif /* defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4) */
    // if(inp_centered >= radius) output = 127
    AE_MOVF16X4(z10, CONST_127_16x4, d3210);

    // if(inp_centered <= -radius) output = -128
    AE_MOVT16X4(z10, CONST_MINUS_128_16x4, b3210);

#if XCHAL_HAVE_HIFI1
    m0 = AE_SAT8S(z10);
    AE_S8_0_IP_HIFI1(m0, p_o, sizeof(WORD8));
#else
    m0 = z10;
    xtbool4 bsat4 = AE_LT16(CONST_127_16x4, m0);
    AE_MOVT16X4(m0, CONST_127_16x4 , bsat4);
    bsat4 = AE_LT16(m0, CONST_MINUS_128_16x4);
    AE_MOVT16X4(m0, CONST_MINUS_128_16x4 , bsat4);

    WORD16 m0_out8 = m0;
    *p_o++ = (WORD8)m0_out8;
#endif
  }

  return 0;
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

