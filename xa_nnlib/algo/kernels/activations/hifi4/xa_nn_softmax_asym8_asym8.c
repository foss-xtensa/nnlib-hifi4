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
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros.h"

#define ALIGNMENT   8   /* 8 bytes alignment */
#define ALIGNED_SIZE(x, bytes)  (((x)+(bytes-1))&(~(bytes-1)))
#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#ifndef ENABLE_SCRATCH_SIZE_API_ONLY
#define SUB_128(inp){\
          ae_int64 temp;\
          temp = AE_MOVINT64_FROMINT32X2(inp);\
          temp = AE_XOR(temp, offset_xor);\
          inp = AE_MOVINT32X2_FROMINT64(temp);\
}

static const int CONSTANT_TERM =  (0x70f5a894);
static const int CONSTANT_1_OVER_3 = (0x2aaaaaab);
static const int CONSTANT_1_OVER_8 = (0x10000000);
static const int ONE_QUATER_Q26 = (0x1000000); // Q6.26
static const int MASK = (0xffffff);
static const int Q31 = 0x7fffffff;
static const int constant_48_over_17 = 1515870810;
static const int constant_neg_32_over_17 = -1010580540;
static const int F2_ONE = 0x20000000;


#if XCHAL_HAVE_HIFI1
#define MAX_16X4(id1, id0)\
    id1 = AE_MAX16(id1, id0);

#define STORE_8X4_FROM_32X4(out_ptr, val12, val34){\
    AE_S8_0_IP_HIFI1(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LH(val12, val12)), (WORD8 *)out_ptr, sizeof(WORD8));\
    AE_S8_0_IP_HIFI1(AE_MOVINT16X4_FROMINT32X2(val12), (WORD8 *)out_ptr, sizeof(WORD8));\
    AE_S8_0_IP_HIFI1(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LH(val34, val34)), (WORD8 *)out_ptr, sizeof(WORD8));\
    AE_S8_0_IP_HIFI1(AE_MOVINT16X4_FROMINT32X2(val34), (WORD8 *)out_ptr, sizeof(WORD8));\
}
#else
#define MAX_16X4(id1, id0) \
        b0 = AE_LT16(id1, id0); \
        AE_MOVT16X4(id1, id0, b0);

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

#define CLAMP_VAL(out, val, min, max){\
    ae_f32x2 temp_max;\
    temp_max = AE_MAX32(min, val);\
    out = AE_MIN32(temp_max, max);\
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


static ae_int32x2 one_over_one_plus_x_for_x_in_0_1(ae_int64 a)
{
    ae_int64 s;
    ae_int32x2 half_den, m, x, half_denominator_times_x;
    ae_int32x2 one_minus_half_denominator_times_x;
    ae_int32x2 CT_48_by_7, CT_neg_32_by_7, CT_F2_ONE;
    int i;

    CT_48_by_7 = AE_MOVDA32(constant_48_over_17);
    CT_neg_32_by_7 = AE_MOVDA32(constant_neg_32_over_17);
    CT_F2_ONE = AE_MOVDA32(F2_ONE);

    ROUNDING_HALF_SUM(s, a)

    half_den = AE_MOVINT32X2_FROMINT64(s);
    half_den = AE_SEL32_LL(half_den, half_den); // half denominator


    // Computation of x
    m = AE_MULFP32X2RS(half_den, CT_neg_32_by_7);
    x = AE_ADD32S(m, CT_48_by_7);

    for(i=0; i<3; i++)
    {
        half_denominator_times_x = AE_MULFP32X2RS(x, half_den);
        one_minus_half_denominator_times_x = AE_SUB32S(CT_F2_ONE, half_denominator_times_x);
        half_denominator_times_x = AE_MULFP32X2RS(x, half_den);
        m = AE_MULFP32X2RS(x, one_minus_half_denominator_times_x);
        m = AE_SLAI32S(m, 2);
        x = AE_ADD32S(x, m);
    }

    x = AE_SLAI32S(x, 1);

    return x;
}

static ae_int32x2 GetReciprocal(ae_int64 x, int x_integerbits, int *lsh)
{
    int headroom_plus_one;
    ae_int64 shifted_sum_minus_one, CT_Q31;
    ae_int64 shifted_sum;
    ae_int32x2 scale;

    headroom_plus_one = AE_NSA64(x) - 31;
    *lsh = x_integerbits - headroom_plus_one;


    CT_Q31 = Q31;
    shifted_sum = AE_SLAA64(x, headroom_plus_one);

    shifted_sum_minus_one = AE_SUB64(shifted_sum, CT_Q31);
    scale = one_over_one_plus_x_for_x_in_0_1(shifted_sum_minus_one);
    return scale;
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
    shift_amount = 26 + exponent;\
    scale = AE_SLAA32(ONE, shift_amount);\
\
    mask = AE_AND32(remainder,  scale);\
\
    b = AE_LT32(z, mask);\
\
    out1 = AE_MULFP32X2RS(out, FixedPointMultiplier);\
    AE_MOVT32X2(out, out1, b);\
}

#define EXP_Q26(y, inp)\
{\
    xtbool2 b1;\
    ae_int32x2 x_in, x0, remainder;\
    ae_int32x2 a_mod_quater_minus_q_1_by_4;\
\
    x0 = AE_AND32(inp, mask_6fs);\
    a_mod_quater_minus_q_1_by_4 = AE_SUB32(x0, q_1_by_4);\
    x_in = AE_SLAI32(a_mod_quater_minus_q_1_by_4, 5);\
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
    GEMMLOWP_EXP_BARREL_SHIFTER(y,4, 242,         remainder);\
\
    b1 = AE_EQ32(inp, z);\
    AE_MOVT32X2(y, AE_MOVDA32(Q31), b1);\
}


WORD32 xa_nn_vec_softmax_asym8_asym8( UWORD8 * __restrict__ p_out,
                    const   UWORD8 * __restrict__ p_vec,
                            WORD32  diffmin,
                            WORD32  input_beta_left_shift,
                            WORD32  input_beta_multiplier,
                            WORD32  vec_length,
                            pVOID   p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(UWORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((input_beta_left_shift < -31) || (input_beta_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND((input_beta_multiplier < 0), -1);

    int i;
    int shift_bits_reciprocal;
#if !XCHAL_HAVE_HIFI1
    xtbool4 b0;
#endif
    xtbool2 f32, f10;
    UWORD8 *p_in = (UWORD8 *)p_vec;
    WORD32 *p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);
    ae_int32x2 y32, y10, diff_min;
    ae_int32x2 dequantized_y32, dequantized_y10, a_min, a_max;
    ae_int32x2 exp_y32, exp_y10, sum_exp, recip_sum_exp, unsat_out32, unsat_out10, ONE;
#if !(XCHAL_HAVE_HIFI1 &( XCHAL_HW_VERSION >= RI9_HWVERSION ))
    ae_int32x2 out32, out10;
#endif
    ae_f16x4 x;
    ae_int16x4 temp16X4, m0, max;
    ae_int64 sum_exp_64;
    ae_valign align_dst;
    int pre_loop_count;
    int main_loop_count;
    int post_loop_count;
#if XCHAL_HAVE_HIFI1
    ae_valign align_out = AE_ZALIGN64();
#endif
 
    if(vec_length > 3)
    {
        pre_loop_count = (int)p_vec & 0x3;
        pre_loop_count = (4 - pre_loop_count) & 3;
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

    ae_int32x2 z = AE_ZERO32();
    ae_int32x2 CT, CT_1_BY_3, CT_1_BY_8;
    ae_int32x2 mask_6fs, q_1_by_4;
    CT = AE_MOVDA32(CONSTANT_TERM);
    CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
    CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
    mask_6fs = AE_MOVDA32(MASK);
    q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
    ONE = AE_MOVDA32(1);

    a_min = AE_ZERO32();
    a_max = AE_MOVDA32(255);
    // Calculating Max
    {
        m0 = AE_MOVDA16(0x8000);
        __Pragma("no_unroll");
        for(i=0; i < pre_loop_count; i++)
        {
            int i1;
            i1 = (WORD16)(*p_in++);
            temp16X4 = AE_MOVDA16(i1);
            MAX_16X4(m0, temp16X4)
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
            MAX_16X4(m0, x)
        }
        p_in = (UWORD8 *)p_in_t;
        __Pragma("no_unroll");
        for(i=0; i < post_loop_count; i++)
        {
            int i1;
            i1 = (WORD16) p_in[i];
            temp16X4 = AE_MOVDA16(i1);
            MAX_16X4(m0, temp16X4)
        }

        ae_int32x2 temp1, temp2;
        temp1 = AE_MOVINT32X2_FROMINT16X4(m0);
        temp2 = AE_SRAA32S(temp1, 16);
        temp1 = AE_SLAA32(temp1, 16);
        temp1 = AE_SRAA32S(temp1, 16);
        temp1 = AE_MAX32(temp1, temp2);
        temp2 = AE_SEL32_LH(temp1, temp1);
        temp2 = AE_MAX32(temp1, temp2);
        max = AE_MOVDA16(AE_MOVAD32_L(temp2));
    }

    diff_min = AE_MOVDA32(diffmin);
    sum_exp = z; // setting to zero

   __Pragma("no_unroll");
    p_in = (UWORD8 *)p_vec;
    for(i=0; i < pre_loop_count; i++)
    {
        int rem_x;

        rem_x = (WORD32)(*p_in++);
        rem_x = rem_x - AE_MOVAD16_0(max);
        y32 = AE_MOVDA32(rem_x);
        f32 = AE_LE32(diff_min, y32);

        MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y32, y32, input_beta_multiplier, input_beta_left_shift)
        EXP_Q26(exp_y32, dequantized_y32);
        AE_MOVF32X2(exp_y32, a_min, f32);
        AE_S32_L_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
        sum_exp = AE_ADD32S(sum_exp, exp_y32);
    }

    sum_exp = AE_SEL32_HH(sum_exp, z);
    align_dst = AE_ZALIGN64(); // zero alignment reg

    WORD8 *p_in_t = (WORD8 *)p_in;
    for(i=0; i < main_loop_count; i++)
    {
        AE_L8X4F_IP(x, p_in_t, 4*sizeof(WORD8));
        x = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x), 8));
        x = AE_SUB16S(x, max);

        y32 = AE_SEXT32X2D16_32(x);
        y10 = AE_SEXT32X2D16_10(x);
        f32 = AE_LE32(diff_min, y32);
        f10 = AE_LE32(diff_min, y10);


        MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y32, y32, input_beta_multiplier, input_beta_left_shift)
        EXP_Q26(exp_y32, dequantized_y32);
        AE_MOVF32X2(exp_y32, a_min, f32);
        AE_SA32X2_IP(exp_y32, align_dst, (ae_int32x2 *)p_exp);
        exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
        //exp_y32 = AE_SRAI32(exp_y32, 12);

        MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y10, y10, input_beta_multiplier, input_beta_left_shift)
        EXP_Q26(exp_y10, dequantized_y10);
        AE_MOVF32X2(exp_y10, a_min, f10);
        AE_SA32X2_IP(exp_y10, align_dst, (ae_int32x2 *)p_exp);
        exp_y10 = AE_SRAA32RS(exp_y10, (int)12);
        //exp_y10 = AE_SRAI32(exp_y10, 12);

        sum_exp = AE_ADD32S(sum_exp, exp_y32);
        sum_exp = AE_ADD32S(sum_exp, exp_y10);

    }
    sum_exp = AE_ADD32S_HL_LH(sum_exp, sum_exp);
    AE_SA64POS_FP(align_dst, p_exp); // finalize the stream

    p_in = (UWORD8 *)p_in_t;
   // remainder loop
   __Pragma("no_unroll");
    for(i=0; i < post_loop_count; i++)
    {
        int rem_x;

        rem_x = (WORD32) *p_in++;
        rem_x = rem_x - AE_MOVAD16_0(max);
        y32 = AE_MOVDA32(rem_x);
        f32 = AE_LE32(diff_min, y32);

        MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y32, y32, input_beta_multiplier, input_beta_left_shift)
        EXP_Q26(exp_y32, dequantized_y32);
        AE_MOVF32X2(exp_y32, a_min, f32);
        AE_S32_L_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
        sum_exp = AE_ADD32S(sum_exp, exp_y32);
    }

    sum_exp_64 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(sum_exp), 32);
    recip_sum_exp = GetReciprocal(sum_exp_64, 12, &shift_bits_reciprocal);

    p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);

    for(i=0; i<(vec_length >> 2); i++)
    {
        AE_L32X2_IP(exp_y32, (ae_int32x2 *)p_exp, 2*sizeof(WORD32));
        AE_L32X2_IP(exp_y10, (ae_int32x2 *)p_exp, 2*sizeof(WORD32));
        unsat_out32 = AE_MULFP32X2RAS(exp_y32, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 8);
        unsat_out10 = AE_MULFP32X2RAS(exp_y10, recip_sum_exp);
        unsat_out10 = AE_SRAA32RS(unsat_out10, shift_bits_reciprocal + 31 - 8);

#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
        // clamped_out
        ae_int8x8 clamped = AE_SATU8X4X32_H(unsat_out32, unsat_out10);
        // Store Output
        AE_SAV8X8_XP(clamped, align_out, (ae_int8x8 *)p_out, 4);
#else
        CLAMP_VAL(out32, unsat_out32, a_min, a_max);
        CLAMP_VAL(out10, unsat_out10, a_min, a_max);
        ae_f16x4 x_temp = AE_CVT16X4(out32, out10);
        AE_SA8X4U_IP(x_temp, align_out, (ae_int32*)p_out);
#endif
#else
        CLAMP_VAL(out32, unsat_out32, a_min, a_max);
        CLAMP_VAL(out10, unsat_out10, a_min, a_max);
        STORE_8X4_FROM_32X4(p_out, out32, out10)
#endif
    }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_out);
#endif

    // remainder loop
    __Pragma("no_unroll");
    for(i=0; i < (vec_length & 3); i++)
    {
#if !(XCHAL_HAVE_HIFI1 &( XCHAL_HW_VERSION >= RI9_HWVERSION ))
        int o1;
#endif
        AE_L32_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        unsat_out32 = AE_MULFP32X2RAS(exp_y32, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 8);
#if (XCHAL_HAVE_HIFI1 &( XCHAL_HW_VERSION >= RI9_HWVERSION ))
        // clamped_out
        ae_int8x8 clamped = AE_SATU8X4X32_L(unsat_out32, unsat_out32);
        // Store Output
        AE_S8_0_IP(clamped, (ae_int8 *)p_out, 1);
#else
        CLAMP_VAL(out32, unsat_out32, a_min, a_max);
        o1 = AE_MOVAD32_H(out32);
        *p_out++ = (UWORD8)o1;
#endif

#if XCHAL_HAVE_HIFI1
        (void)a_max; /* Unused in HiFi1. This removes LLVM15 warning */
#endif
    }

    return 0;
}

WORD32 xa_nn_vec_softmax_asym8s_asym8s( WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_vec,
                            WORD32  diffmin,
                            WORD32  input_beta_left_shift,
                            WORD32  input_beta_multiplier,
                            WORD32  vec_length,
                            pVOID   p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((input_beta_left_shift < 0) || (input_beta_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND((input_beta_multiplier < 0), -1);

    int i;
    int shift_bits_reciprocal;
#if !XCHAL_HAVE_HIFI1
    xtbool4 b0;
#endif
    xtbool2 f32, f10;
    WORD8 *p_in = (WORD8 *)p_vec;
    WORD32 *p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);
    ae_int32x2 y32, y10, diff_min;
    ae_int32x2 dequantized_y32, dequantized_y10, a_min, a_max;
    ae_int32x2 exp_y32, exp_y10, sum_exp, recip_sum_exp, unsat_out32, unsat_out10, out32, out10, ONE;
    ae_f16x4 x;
    ae_int16x4 temp16X4, m0, max;
    ae_int64 sum_exp_64;
    ae_valign align_dst;
    /* Second operand for XOR instruction used in SUB_128 */
    ae_int64 offset_xor = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(128));
    int pre_loop_count;
    int main_loop_count;
    int post_loop_count;
#if XCHAL_HAVE_HIFI1
    ae_valign align_out = AE_ZALIGN64();
#endif

    if(vec_length > 3)
    {
        pre_loop_count = (int)p_vec & 0x3;
        pre_loop_count = (4 - pre_loop_count) & 3;
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

    ae_int32x2 z = AE_ZERO32();
    ae_int32x2 CT, CT_1_BY_3, CT_1_BY_8;
    ae_int32x2 mask_6fs, q_1_by_4;
    CT = AE_MOVDA32(CONSTANT_TERM);
    CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
    CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
    mask_6fs = AE_MOVDA32(MASK);
    q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
    ONE = AE_MOVDA32(1);

    a_min = AE_ZERO32();
    a_max = AE_MOVDA32(255);
    // Calculating Max
    {
        m0 = AE_MOVDA16(0x8000);
        __Pragma("no_unroll");
        for(i=0; i < pre_loop_count; i++)
        {
            int i1;
            i1 = (WORD16)(*p_in++);
            temp16X4 = AE_MOVDA16(i1);
            MAX_16X4(m0, temp16X4)
        }
        for(i=0; i < main_loop_count; i++)
        {
#if XCHAL_HAVE_HIFI1
            AE_L8X4S_IP(x, p_in, 4*sizeof(WORD8));
#else
            AE_L8X4F_IP(x, p_in, 4*sizeof(WORD8));
            x = AE_SRAI16(x,8);
#endif
            MAX_16X4(m0, x)
        }
        __Pragma("no_unroll");
        for(i=0; i < post_loop_count; i++)
        {
            int i1;
            i1 = (WORD16) p_in[i];
            temp16X4 = AE_MOVDA16(i1);
            MAX_16X4(m0, temp16X4)
        }

        ae_int32x2 temp1, temp2;
        temp1 = AE_MOVINT32X2_FROMINT16X4(m0);
        temp2 = AE_SRAA32S(temp1, 16);
        temp1 = AE_SLAA32(temp1, 16);
        temp1 = AE_SRAA32S(temp1, 16);
        temp1 = AE_MAX32(temp1, temp2);
        temp2 = AE_SEL32_LH(temp1, temp1);
        temp2 = AE_MAX32(temp1, temp2);
        max = AE_MOVDA16(AE_MOVAD32_L(temp2));
    }

    diff_min = AE_MOVDA32(diffmin);
    sum_exp = z; // setting to zero

   __Pragma("no_unroll");
    p_in = (WORD8 *)p_vec;
    for(i=0; i < pre_loop_count; i++)
    {
        int rem_x;

        rem_x = (WORD32)(*p_in++);
        rem_x = rem_x - AE_MOVAD16_0(max);
        y32 = AE_MOVDA32(rem_x);
        f32 = AE_LE32(diff_min, y32);

        MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y32, y32, input_beta_multiplier, input_beta_left_shift)
        EXP_Q26(exp_y32, dequantized_y32);
        AE_MOVF32X2(exp_y32, a_min, f32);
        AE_S32_L_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
        sum_exp = AE_ADD32S(sum_exp, exp_y32);
    }

    sum_exp = AE_SEL32_HH(sum_exp, z);
    align_dst = AE_ZALIGN64(); // zero alignment reg

    for(i=0; i < main_loop_count; i++)
    {
#if XCHAL_HAVE_HIFI1
         AE_L8X4S_IP(x, p_in, 4*sizeof(WORD8));
#else
         AE_L8X4F_IP(x, p_in, 4*sizeof(WORD8));
         x = AE_SRAI16(x,8);
#endif
        x = AE_SUB16S(x, max);

        y32 = AE_SEXT32X2D16_32(x);
        y10 = AE_SEXT32X2D16_10(x);
        f32 = AE_LE32(diff_min, y32);
        f10 = AE_LE32(diff_min, y10);


        MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y32, y32, input_beta_multiplier, input_beta_left_shift)
        EXP_Q26(exp_y32, dequantized_y32);
        AE_MOVF32X2(exp_y32, a_min, f32);
        AE_SA32X2_IP(exp_y32, align_dst, (ae_int32x2 *)p_exp);
        exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
        //exp_y32 = AE_SRAI32(exp_y32, 12);

        MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y10, y10, input_beta_multiplier, input_beta_left_shift)
        EXP_Q26(exp_y10, dequantized_y10);
        AE_MOVF32X2(exp_y10, a_min, f10);
        AE_SA32X2_IP(exp_y10, align_dst, (ae_int32x2 *)p_exp);
        exp_y10 = AE_SRAA32RS(exp_y10, (int)12);
        //exp_y10 = AE_SRAI32(exp_y10, 12);

        sum_exp = AE_ADD32S(sum_exp, exp_y32);
        sum_exp = AE_ADD32S(sum_exp, exp_y10);

    }
    sum_exp = AE_ADD32S_HL_LH(sum_exp, sum_exp);
    AE_SA64POS_FP(align_dst, p_exp); // finalize the stream

   // remainder loop
   __Pragma("no_unroll");
    for(i=0; i < post_loop_count; i++)
    {
        int rem_x;

        rem_x = (WORD32) *p_in++;
        rem_x = rem_x - AE_MOVAD16_0(max);
        y32 = AE_MOVDA32(rem_x);
        f32 = AE_LE32(diff_min, y32);

        MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y32, y32, input_beta_multiplier, input_beta_left_shift)
        EXP_Q26(exp_y32, dequantized_y32);
        AE_MOVF32X2(exp_y32, a_min, f32);
        AE_S32_L_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
        sum_exp = AE_ADD32S(sum_exp, exp_y32);
    }

    sum_exp_64 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(sum_exp), 32);
    recip_sum_exp = GetReciprocal(sum_exp_64, 12, &shift_bits_reciprocal);

    p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);

    for(i=0; i<(vec_length >> 2); i++)
    {
        AE_L32X2_IP(exp_y32, (ae_int32x2 *)p_exp, 2*sizeof(WORD32));
        AE_L32X2_IP(exp_y10, (ae_int32x2 *)p_exp, 2*sizeof(WORD32));

        unsat_out32 = AE_MULFP32X2RAS(exp_y32, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 8);
        CLAMP_VAL(out32, unsat_out32, a_min, a_max);
        SUB_128(out32)

        unsat_out10 = AE_MULFP32X2RAS(exp_y10, recip_sum_exp);
        unsat_out10 = AE_SRAA32RS(unsat_out10, shift_bits_reciprocal + 31 - 8);
        CLAMP_VAL(out10, unsat_out10, a_min, a_max);
        SUB_128(out10)
#if XCHAL_HAVE_HIFI1
        ae_f16x4 x_temp = AE_CVT16X4(out32, out10);
        AE_SA8X4U_IP(x_temp, align_out, (ae_int32*)p_out);
#else
        STORE_8X4_FROM_32X4(p_out, out32, out10)
#endif
    }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_out);
#endif

    // remainder loop
    __Pragma("no_unroll");
    for(i=0; i < (vec_length & 3); i++)
    {
        int o1;
        AE_L32_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        unsat_out32 = AE_MULFP32X2RAS(exp_y32, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 8);
        CLAMP_VAL(out32, unsat_out32, a_min, a_max);
        SUB_128(out32)

        o1 = AE_MOVAD32_H(out32);
        *p_out++ = (WORD8)o1;
    }

    return 0;
}

WORD32 xa_nn_vec_softmax_asym8s_16( WORD16 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_vec,
                            WORD32  diffmin,
                            WORD32  input_beta_left_shift,
                            WORD32  input_beta_multiplier,
                            WORD32  vec_length,
                            pVOID   p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((input_beta_left_shift < 0) || (input_beta_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND((input_beta_multiplier < 0), -1);

    int i;
    int shift_bits_reciprocal;
#if !XCHAL_HAVE_HIFI1
    xtbool4 b0;
#endif
    xtbool2 f32, f10;
    WORD8 *p_in = (WORD8 *)p_vec;
    WORD32 *p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);
    ae_int32x2 y32, y10, diff_min;
    ae_int32x2 dequantized_y32, dequantized_y10, a_min, a_max;
    ae_int32x2 exp_y32, exp_y10, sum_exp, recip_sum_exp, unsat_out32, unsat_out10, out32, out10, ONE;
    ae_f16x4 x;
    ae_int16x4 temp16X4, m0, max;
    ae_int64 sum_exp_64;
    ae_valign align_dst;
    int pre_loop_count;
    int main_loop_count;
    int post_loop_count;

    if(vec_length > 3)
    {
        pre_loop_count = (int)p_vec & 0x3;
        pre_loop_count = (4 - pre_loop_count) & 3;
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

    ae_int32x2 z = AE_ZERO32();
    ae_int32x2 CT, CT_1_BY_3, CT_1_BY_8;
    ae_int32x2 mask_6fs, q_1_by_4;
    CT = AE_MOVDA32(CONSTANT_TERM);
    CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
    CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
    mask_6fs = AE_MOVDA32(MASK);
    q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
    ONE = AE_MOVDA32(1);

    a_min = AE_ZERO32();
    a_max = AE_MOVDA32(65535);

    // Calculating Max
    {
        m0 = AE_MOVDA16(0x8000);
        __Pragma("no_unroll");
        for(i=0; i < pre_loop_count; i++)
        {
            int i1;
            i1 = (WORD16)(*p_in++);
            temp16X4 = AE_MOVDA16(i1);
            MAX_16X4(m0, temp16X4)
        }
        for(i=0; i < main_loop_count; i++)
        {
#if XCHAL_HAVE_HIFI1
            AE_L8X4S_IP(x, p_in, 4*sizeof(WORD8));
#else
            AE_L8X4F_IP(x, p_in, 4*sizeof(WORD8));
            x = AE_SRAI16(x,8);
#endif
            MAX_16X4(m0, x)
        }

        if(post_loop_count & 0x2)
        {
            temp16X4 = AE_MOVDA16X2(p_in[0], p_in[1]);
            p_in += 2;
            MAX_16X4(m0, temp16X4)
        }

        if(post_loop_count & 0x1)
        {
            temp16X4 = AE_MOVDA16(*p_in++);
            MAX_16X4(m0, temp16X4)
        }

        ae_int32x2 temp1, temp2;
        temp1 = AE_MOVINT32X2_FROMINT16X4(m0);
        temp2 = AE_SRAA32S(temp1, 16);
        temp1 = AE_SLAA32(temp1, 16);
        temp1 = AE_SRAA32S(temp1, 16);
        temp1 = AE_MAX32(temp1, temp2);
        temp2 = AE_SEL32_LH(temp1, temp1);
        temp2 = AE_MAX32(temp1, temp2);
        max = AE_MOVDA16(AE_MOVAD32_L(temp2));
    }

    diff_min = AE_MOVDA32(diffmin);
    sum_exp = z; // setting to zero

   __Pragma("no_unroll");
    p_in = (WORD8 *)p_vec;
    for(i=0; i < pre_loop_count; i++)
    {
        int rem_x;

        rem_x = (WORD32)(*p_in++);
        rem_x = rem_x - AE_MOVAD16_0(max);
        y32 = AE_MOVDA32(rem_x);
        f32 = AE_LE32(diff_min, y32);

        MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y32, y32, input_beta_multiplier, input_beta_left_shift)
        EXP_Q26(exp_y32, dequantized_y32);
        AE_MOVF32X2(exp_y32, a_min, f32);
        AE_S32_L_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
        sum_exp = AE_ADD32S(sum_exp, exp_y32);
    }

    sum_exp = AE_SEL32_HH(sum_exp, z);
    align_dst = AE_ZALIGN64(); // zero alignment reg

    for(i=0; i < main_loop_count; i++)
    {
        AE_L8X4F_IP(x, p_in, 4*sizeof(WORD8));
        x = AE_SRAI16(x,8);
        x = AE_SUB16S(x, max);

        y32 = AE_SEXT32X2D16_32(x);
        y10 = AE_SEXT32X2D16_10(x);
        f32 = AE_LE32(diff_min, y32);
        f10 = AE_LE32(diff_min, y10);


        MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y32, y32, input_beta_multiplier, input_beta_left_shift)
        EXP_Q26(exp_y32, dequantized_y32);
        AE_MOVF32X2(exp_y32, a_min, f32);
        AE_SA32X2_IP(exp_y32, align_dst, (ae_int32x2 *)p_exp);
        exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
        //exp_y32 = AE_SRAI32(exp_y32, 12);

        MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y10, y10, input_beta_multiplier, input_beta_left_shift)
        EXP_Q26(exp_y10, dequantized_y10);
        AE_MOVF32X2(exp_y10, a_min, f10);
        AE_SA32X2_IP(exp_y10, align_dst, (ae_int32x2 *)p_exp);
        exp_y10 = AE_SRAA32RS(exp_y10, (int)12);
        //exp_y10 = AE_SRAI32(exp_y10, 12);

        sum_exp = AE_ADD32S(sum_exp, exp_y32);
        sum_exp = AE_ADD32S(sum_exp, exp_y10);

    }

   // remainder loop
   if(post_loop_count & 0x2)
   {
     ae_int16x4 rem_x;

     rem_x = AE_MOVDA16X2(p_in[0],p_in[1]);
     p_in += 2;
     rem_x = AE_SUB16S(rem_x, max);

     y32 = AE_SEXT32X2D16_10(rem_x);
     f32 = AE_LE32(diff_min, y32);

     MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y32, y32, input_beta_multiplier, input_beta_left_shift)
     EXP_Q26(exp_y32, dequantized_y32);
     AE_MOVF32X2(exp_y32, a_min, f32);
     AE_SA32X2_IP(exp_y32, align_dst, (ae_int32x2 *)p_exp);

     exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
     sum_exp = AE_ADD32S(sum_exp, exp_y32);
   }

   sum_exp = AE_ADD32S_HL_LH(sum_exp, sum_exp);
   AE_SA64POS_FP(align_dst, p_exp); // finalize the stream

   if(post_loop_count & 0x1)
   {
     ae_int16x4 rem_x;

     rem_x = AE_MOVDA16(*p_in++);
     rem_x = AE_SUB16S(rem_x, max);
     y32 = AE_SEXT32X2D16_10(rem_x);
     f32 = AE_LE32(diff_min, y32);

     MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(dequantized_y32, y32, input_beta_multiplier, input_beta_left_shift)
     EXP_Q26(exp_y32, dequantized_y32);
     AE_MOVF32X2(exp_y32, a_min, f32);
     AE_S32_L_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

     exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
     sum_exp = AE_ADD32S(sum_exp, exp_y32);
   }

    sum_exp_64 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(sum_exp), 32);
    recip_sum_exp = GetReciprocal(sum_exp_64, 12, &shift_bits_reciprocal);

    p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);

    for(i=0; i<(vec_length >> 2); i++)
    {
        AE_L32X2_IP(exp_y32, (ae_int32x2 *)p_exp, 2*sizeof(WORD32));
        AE_L32X2_IP(exp_y10, (ae_int32x2 *)p_exp, 2*sizeof(WORD32));

        unsat_out32 = AE_MULFP32X2RAS(exp_y32, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 16);
        CLAMP_VAL(out32, unsat_out32, a_min, a_max);
        out32 = AE_SUB32(out32, AE_MOVDA32(32768));

        unsat_out10 = AE_MULFP32X2RAS(exp_y10, recip_sum_exp);
        unsat_out10 = AE_SRAA32RS(unsat_out10, shift_bits_reciprocal + 31 - 16);
        CLAMP_VAL(out10, unsat_out10, a_min, a_max);
        out10 = AE_SUB32(out10, AE_MOVDA32(32768)); 
        m0 = AE_CVT16X4(out32, out10);
        AE_SA16X4_IP(m0, align_dst, (ae_int16x4 *)p_out);
    }
    AE_SA64POS_FP(align_dst, p_out); // finalize the stream

    // remainder loop
    if(vec_length & 0x2)
    {
        AE_L32X2_IP(exp_y32, (ae_int32x2 *)p_exp, 2*sizeof(WORD32));

        unsat_out32 = AE_MULFP32X2RAS(exp_y32, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 16);
        CLAMP_VAL(out32, unsat_out32, a_min, a_max);
        out32 = AE_SUB32(out32, AE_MOVDA32(32768));
        m0 = AE_CVT16X4(out32, out32);

        AE_S16_0_IP(AE_SEL16I(m0, m0, 4) , (ae_int16 *)p_out, 2);
        AE_S16_0_IP(m0, (ae_int16 *)p_out, 2);
    }

    if(vec_length & 0x1)
    {
        AE_L32_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        unsat_out32 = AE_MULFP32X2RAS(exp_y32, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 16);
        CLAMP_VAL(out32, unsat_out32, a_min, a_max);
        out32 = AE_SUB32(out32, AE_MOVDA32(32768));
        m0 = AE_CVT16X4(out32, out32);

        AE_S16_0_IP(m0, (ae_int16 *)p_out, 2);
    }

    return 0;
}
#endif // #ifndef ENABLE_SCRATCH_SIZE_API_ONLY

int get_softmax_scratch_size(int inp_precision, int out_precision, int length)
{
    XA_NNLIB_ARG_CHK_COND((length <= 0), -1);
    int size_of_one_elm_in_bytes, total_bytes;
    (void) out_precision;

    /* This function returns scratch size required by softmax implementation in bytes
       scratch memory is needed to save exponents of inputs computed in the function,
       every exponent is computed as 32 bit (4 bytes) number currently*/
    switch(inp_precision)
    {
        case 8:
            size_of_one_elm_in_bytes = 4;
            break;
        case 16:
            size_of_one_elm_in_bytes = 4;
            break;
        case 32:
            size_of_one_elm_in_bytes = 4;
            break;
        case -1:
            size_of_one_elm_in_bytes = 4;
            break;
        case -3:
            size_of_one_elm_in_bytes = 4;
            break;
        case -4:
            size_of_one_elm_in_bytes = 4;
            break;
        default:
            return -1;
    }

    total_bytes = size_of_one_elm_in_bytes*length;
    total_bytes = ALIGNED_SIZE(total_bytes, ALIGNMENT);

    return total_bytes;
}

