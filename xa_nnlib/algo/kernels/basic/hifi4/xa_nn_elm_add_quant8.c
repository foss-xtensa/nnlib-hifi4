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
#include "xa_nn_basic_state.h"
#include "xa_nnlib_common_macros.h"

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

#if XCHAL_HAVE_HIFI1

#define MAX_16X4(id1, id0) \
        id1 = AE_MAX16(id1, id0);
#define MIN_16X4(id1, id0) \
        id1 = AE_MIN16(id1, id0);
#define LIMIT16X4(out, inp, min, max){\
        out = min;\
        MAX_16X4(out, inp);\
        MIN_16X4(out, max);\
}

#else

#define MAX_16X4(id1, id0) { \
        xtbool4 bool4 = AE_LT16(id1, id0); \
        AE_MOVT16X4(id1, id0, bool4);\
}
#define MIN_16X4(id1, id0) { \
        xtbool4 bool4 = AE_LT16(id1, id0); \
        AE_MOVF16X4(id1, id0, bool4);\
}
#define LIMIT16X4(out, inp, min, max){\
        out = min;\
        MAX_16X4(out, inp);\
        MIN_16X4(out, max);\
}
#endif

#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_add_asym8xasym8_asym8(UWORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   UWORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const   UWORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(UWORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((out_zero_bias < 0) || (out_zero_bias > 255)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -255) || (inp1_zero_bias > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -255) || (inp2_zero_bias > 0)), -1);
    XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_multiplier < 0) || (inp1_multiplier < 0) || (inp2_multiplier < 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < 0) || (out_activation_min > 255)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < 0) || (out_activation_max > 255)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    int i;
    UWORD8 *out = p_out;
    WORD8 *p_i1 = (WORD8 *)p_inp1;
    WORD8 *p_i2 = (WORD8 *)p_inp2;

    ae_f16x4 x1, x2;
    ae_int32x2 temp;
    ae_f16x4 temp16X4, zero_bias1, zero_bias2;
    ae_f32x2 op_zero_bias, activation_min, activation_max;

    // Taking zero_bias into 16X4 variable
    temp = AE_MOVDA32(inp1_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias1 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

    temp = AE_MOVDA32(inp2_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias2 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

    op_zero_bias = AE_MOVDA32(out_zero_bias);

    activation_min = AE_MOVDA32(out_activation_min);
    activation_max = AE_MOVDA32(out_activation_max);

    ae_valign align_out, i1_a, i2_a;
    align_out = AE_ZALIGN64();
    i1_a = AE_LA64_PP(p_i1);
    i2_a = AE_LA64_PP(p_i2);

    for(i=0;i < num_elm>>2;i++)
    {
        ae_f16x4 v1, v2;
        ae_f32x2 shifted_v1, shifted_v2;
        ae_f32x2 shifted_v3, shifted_v4;
        ae_f32x2 scaled_v1, scaled_v2;
        ae_f32x2 scaled_v3, scaled_v4;
        ae_f32x2 raw_sum12, raw_sum34;
        ae_f32x2 raw_out12, raw_out34;
        ae_f32x2 clamped_out12, clamped_out34;


        AE_LA8X4U_IP(x1, i1_a, p_i1);
        AE_LA8X4U_IP(x2, i2_a, p_i2);

        v1 = AE_ADD16(x1, zero_bias1);
        v2 = AE_ADD16(x2, zero_bias2);

        shifted_v1 = AE_SEXT32X2D16_32(v1);
        shifted_v2 = AE_SEXT32X2D16_10(v1);
        shifted_v3 = AE_SEXT32X2D16_32(v2);
        shifted_v4 = AE_SEXT32X2D16_10(v2);

        shifted_v1 = AE_SLAA32S(shifted_v1, left_shift);
        shifted_v2 = AE_SLAA32S(shifted_v2, left_shift);
        shifted_v3 = AE_SLAA32S(shifted_v3, left_shift);
        shifted_v4 = AE_SLAA32S(shifted_v4, left_shift);


        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v1, shifted_v1, inp1_multiplier, inp1_left_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v2, shifted_v2, inp1_multiplier, inp1_left_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v3, shifted_v3, inp2_multiplier, inp2_left_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v4, shifted_v4, inp2_multiplier, inp2_left_shift)

        // Raw Sum
        raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);
        raw_sum34 = AE_ADD32S(scaled_v2, scaled_v4);

        // Raw Output
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out12, raw_sum12, out_multiplier, out_left_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out34, raw_sum34, out_multiplier, out_left_shift)
        raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);
        raw_out34 = AE_ADD32S(raw_out34, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out12, raw_out12, activation_min, activation_max)
        CLAMP_VAL(clamped_out34, raw_out34, activation_min, activation_max)

        // Store Output
        ae_int16x4 temp = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(clamped_out12),AE_MOVINT16X4_FROMINT32X2(clamped_out34));
        AE_SA8X4U_IP(temp, align_out, (ae_int32 *)out);

    }

    // Remainder Loop
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
int rem_itr = (num_elm & 3);
if(rem_itr)
{
        ae_f16x4 v1, v2;
        ae_f32x2 shifted_v1, shifted_v2;
        ae_f32x2 shifted_v3, shifted_v4;
        ae_f32x2 scaled_v1, scaled_v2;
        ae_f32x2 scaled_v3, scaled_v4;
        ae_f32x2 raw_sum12, raw_sum34;
        ae_f32x2 raw_out12, raw_out34;
        ae_f32x2 clamped_out12, clamped_out34;


        AE_LAV8X4U_XP(x1, i1_a, (ae_int8x4u *)p_i1, rem_itr);
        AE_LAV8X4U_XP(x2, i2_a, (ae_int8x4u *)p_i2, rem_itr);

        v1 = AE_ADD16(x1, zero_bias1);
        v2 = AE_ADD16(x2, zero_bias2);

        shifted_v1 = AE_SEXT32X2D16_32(v1);
        shifted_v2 = AE_SEXT32X2D16_10(v1);
        shifted_v3 = AE_SEXT32X2D16_32(v2);
        shifted_v4 = AE_SEXT32X2D16_10(v2);

        shifted_v1 = AE_SLAA32S(shifted_v1, left_shift);
        shifted_v2 = AE_SLAA32S(shifted_v2, left_shift);
        shifted_v3 = AE_SLAA32S(shifted_v3, left_shift);
        shifted_v4 = AE_SLAA32S(shifted_v4, left_shift);


        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v1, shifted_v1, inp1_multiplier, inp1_left_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v2, shifted_v2, inp1_multiplier, inp1_left_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v3, shifted_v3, inp2_multiplier, inp2_left_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v4, shifted_v4, inp2_multiplier, inp2_left_shift)

        // Raw Sum
        raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);
        raw_sum34 = AE_ADD32S(scaled_v2, scaled_v4);

        // Raw Output
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out12, raw_sum12, out_multiplier, out_left_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out34, raw_sum34, out_multiplier, out_left_shift)
        raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);
        raw_out34 = AE_ADD32S(raw_out34, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out12, raw_out12, activation_min, activation_max)
        CLAMP_VAL(clamped_out34, raw_out34, activation_min, activation_max)

        // Store Output
        ae_int16x4 temp = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(clamped_out12),AE_MOVINT16X4_FROMINT32X2(clamped_out34));
        AE_SAV8X4U_XP(temp, align_out, (ae_int8x4u *)out, rem_itr);
}
    AE_SA64POS_FP(align_out, out);
#else
    AE_SA64POS_FP(align_out, out);
    for(i=0; i < (num_elm & 3); i++)
    {
        ae_f16x4 v1, v2;
        ae_f32x2 shifted_v1;
        ae_f32x2 shifted_v3;
        ae_f32x2 scaled_v1;
        ae_f32x2 scaled_v3;
        ae_f32x2 raw_sum12;
        ae_f32x2 raw_out12;
        ae_f32x2 clamped_out12;

        WORD16 i1;
        
        AE_L8U_IP(x1, p_i1, 1);
        AE_L8U_IP(x2, p_i2, 1);
        
        v1 = AE_ADD16(x1, zero_bias1);
        v2 = AE_ADD16(x2, zero_bias2);

        shifted_v1 = AE_SEXT32X2D16_32(v1);
        shifted_v3 = AE_SEXT32X2D16_32(v2);

        shifted_v1 = AE_SLAA32S(shifted_v1, left_shift);
        shifted_v3 = AE_SLAA32S(shifted_v3, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v1, shifted_v1, inp1_multiplier, inp1_left_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v3, shifted_v3, inp2_multiplier, inp2_left_shift)

        // Raw Sum
        raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);

        // Raw Output
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out12, raw_sum12, out_multiplier, out_left_shift)
        raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out12, raw_out12, activation_min, activation_max)

        // Store Output
        i1 = AE_MOVAD32_H(clamped_out12);
        *out++ = (UWORD8) i1;
    }
#endif
    return 0;
}  
#else
WORD32 xa_nn_elm_add_asym8xasym8_asym8(UWORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   UWORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const   UWORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(UWORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((out_zero_bias < 0) || (out_zero_bias > 255)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -255) || (inp1_zero_bias > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -255) || (inp2_zero_bias > 0)), -1);
    XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_multiplier < 0) || (inp1_multiplier < 0) || (inp2_multiplier < 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < 0) || (out_activation_min > 255)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < 0) || (out_activation_max > 255)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    int i;
    UWORD8 *out = p_out;
    WORD8 *p_i1 = (WORD8 *)p_inp1;
    WORD8 *p_i2 = (WORD8 *)p_inp2;

    ae_f16x4 x1, x2;
    ae_int32x2 temp;
    ae_f16x4 temp16X4, zero_bias1, zero_bias2;
    ae_f32x2 op_zero_bias, activation_min, activation_max;

    // Taking zero_bias into 16X4 variable
    temp = AE_MOVDA32(inp1_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias1 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

    temp = AE_MOVDA32(inp2_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias2 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

    op_zero_bias = AE_MOVDA32(out_zero_bias);

    activation_min = AE_MOVDA32(out_activation_min);
    activation_max = AE_MOVDA32(out_activation_max);

    if(((((unsigned)p_i1)&3) == 0) && ((((unsigned)p_i2)&3) == 0))
    {
        for(i=0;i < num_elm>>2;i++)
        {
            ae_f16x4 v1, v2;
            ae_f32x2 shifted_v1, shifted_v2;
            ae_f32x2 shifted_v3, shifted_v4;
            ae_f32x2 scaled_v1, scaled_v2;
            ae_f32x2 scaled_v3, scaled_v4;
            ae_f32x2 raw_sum12, raw_sum34;
            ae_f32x2 raw_out12, raw_out34;
            ae_f32x2 clamped_out12, clamped_out34;


            AE_L8X4F_IP(x1, p_i1, 4*sizeof(WORD8));
            AE_L8X4F_IP(x2, p_i2, 4*sizeof(WORD8));

            x1 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x1), 8));
            x2 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x2), 8));

            v1 = AE_ADD16(x1, zero_bias1);
            v2 = AE_ADD16(x2, zero_bias2);

            shifted_v1 = AE_SEXT32X2D16_32(v1);
            shifted_v2 = AE_SEXT32X2D16_10(v1);
            shifted_v3 = AE_SEXT32X2D16_32(v2);
            shifted_v4 = AE_SEXT32X2D16_10(v2);

            shifted_v1 = AE_SLAA32S(shifted_v1, left_shift);
            shifted_v2 = AE_SLAA32S(shifted_v2, left_shift);
            shifted_v3 = AE_SLAA32S(shifted_v3, left_shift);
            shifted_v4 = AE_SLAA32S(shifted_v4, left_shift);


            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v1, shifted_v1, inp1_multiplier, inp1_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v2, shifted_v2, inp1_multiplier, inp1_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v3, shifted_v3, inp2_multiplier, inp2_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v4, shifted_v4, inp2_multiplier, inp2_left_shift)

            // Raw Sum
            raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);
            raw_sum34 = AE_ADD32S(scaled_v2, scaled_v4);

            // Raw Output
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out12, raw_sum12, out_multiplier, out_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out34, raw_sum34, out_multiplier, out_left_shift)
            raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);
            raw_out34 = AE_ADD32S(raw_out34, op_zero_bias);

            // clamped_out
            CLAMP_VAL(clamped_out12, raw_out12, activation_min, activation_max)
            CLAMP_VAL(clamped_out34, raw_out34, activation_min, activation_max)

            // Store Output
            STORE_8X4_FROM_32X4(out, clamped_out12, clamped_out34)
        }
    }
    else
    {
        ALIGN_REGISTER_TYPE i1_a, i2_a;

        PRIME_8X4U(p_i1, i1_a);
        PRIME_8X4U(p_i2, i2_a);
        for(i=0;i < num_elm>>2;i++)
        {
            ae_f16x4 v1, v2;
            ae_f32x2 shifted_v1, shifted_v2;
            ae_f32x2 shifted_v3, shifted_v4;
            ae_f32x2 scaled_v1, scaled_v2;
            ae_f32x2 scaled_v3, scaled_v4;
            ae_f32x2 raw_sum12, raw_sum34;
            ae_f32x2 raw_out12, raw_out34;
            ae_f32x2 clamped_out12, clamped_out34;


            AE_LA8X4U_IP(x1, i1_a, p_i1);
            AE_LA8X4U_IP(x2, i2_a, p_i2);

            v1 = AE_ADD16(x1, zero_bias1);
            v2 = AE_ADD16(x2, zero_bias2);

            shifted_v1 = AE_SEXT32X2D16_32(v1);
            shifted_v2 = AE_SEXT32X2D16_10(v1);
            shifted_v3 = AE_SEXT32X2D16_32(v2);
            shifted_v4 = AE_SEXT32X2D16_10(v2);

            shifted_v1 = AE_SLAA32S(shifted_v1, left_shift);
            shifted_v2 = AE_SLAA32S(shifted_v2, left_shift);
            shifted_v3 = AE_SLAA32S(shifted_v3, left_shift);
            shifted_v4 = AE_SLAA32S(shifted_v4, left_shift);


            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v1, shifted_v1, inp1_multiplier, inp1_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v2, shifted_v2, inp1_multiplier, inp1_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v3, shifted_v3, inp2_multiplier, inp2_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v4, shifted_v4, inp2_multiplier, inp2_left_shift)

            // Raw Sum
            raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);
            raw_sum34 = AE_ADD32S(scaled_v2, scaled_v4);

            // Raw Output
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out12, raw_sum12, out_multiplier, out_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out34, raw_sum34, out_multiplier, out_left_shift)
            raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);
            raw_out34 = AE_ADD32S(raw_out34, op_zero_bias);

            // clamped_out
            CLAMP_VAL(clamped_out12, raw_out12, activation_min, activation_max)
            CLAMP_VAL(clamped_out34, raw_out34, activation_min, activation_max)

            // Store Output
            STORE_8X4_FROM_32X4(out, clamped_out12, clamped_out34)
        }
    }
    // Remainder Loop
    for(i=0; i < (num_elm & 3); i++)
    {
        ae_f16x4 v1, v2;
        ae_f32x2 shifted_v1;
        ae_f32x2 shifted_v3;
        ae_f32x2 scaled_v1;
        ae_f32x2 scaled_v3;
        ae_f32x2 raw_sum12;
        ae_f32x2 raw_out12;
        ae_f32x2 clamped_out12;

        WORD16 i1, i2;

        i1 = (WORD16) *((UWORD8 *)p_i1 + i);
        i2 = (WORD16) *((UWORD8 *)p_i2 + i);

        x1 = AE_MOVDA16(i1);
        x2 = AE_MOVDA16(i2);

        v1 = AE_ADD16(x1, zero_bias1);
        v2 = AE_ADD16(x2, zero_bias2);

        shifted_v1 = AE_SEXT32X2D16_32(v1);
        shifted_v3 = AE_SEXT32X2D16_32(v2);

        shifted_v1 = AE_SLAA32S(shifted_v1, left_shift);
        shifted_v3 = AE_SLAA32S(shifted_v3, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v1, shifted_v1, inp1_multiplier, inp1_left_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v3, shifted_v3, inp2_multiplier, inp2_left_shift)

        // Raw Sum
        raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);

        // Raw Output
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out12, raw_sum12, out_multiplier, out_left_shift)
        raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out12, raw_out12, activation_min, activation_max)

        // Store Output
        i1 = (WORD16)(AE_MOVAD32_H(clamped_out12));
        *out++ = (UWORD8) i1;
    }

    return 0;
}
#endif

#if XCHAL_HAVE_HIFI1

#if XCHAL_HAVE_HIFI1S
WORD32 xa_nn_elm_add_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
    XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    int i;
    WORD8 *p_a = (WORD8 *)p_inp1;
    WORD8 *p_b = (WORD8 *)p_inp2;
    WORD8 *p_c =          p_out;

    ae_int8x8 amin8 = AE_MOVDA8(out_activation_min);
    ae_int8x8 amax8 = AE_MOVDA8(out_activation_max);

    const ae_int32x2 zc = AE_MOVDA32( out_zero_bias);
    
    ae_int8x8 za8 = AE_MOVDA8(-inp1_zero_bias);
    ae_int8x8 zb8 = AE_MOVDA8(-inp2_zero_bias);

    // intermediate results and scratch registers
    ae_int16x4 a0_3, a4_7, b0_3, b4_7;

    ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
    ae_int32x2 shifted_b0_1, shifted_b2_3, shifted_b4_5, shifted_b6_7;

    ae_int32x2 scaled_a0_1, scaled_a2_3, scaled_a4_5, scaled_a6_7;
    ae_int32x2 scaled_b0_1, scaled_b2_3, scaled_b4_5, scaled_b6_7;

    ae_int32x2 raw_sum0_1, raw_sum2_3, raw_sum4_5, raw_sum6_7;
    ae_int32x2 out0_1, out2_3, out4_5, out6_7;

    const int num_simd8_ops = num_elm/8;
    const int num_scalar_ops = num_elm%8;

    ae_valign va_a, va_b, va_c;
    va_a = AE_LA64_PP(p_a);
    va_b = AE_LA64_PP(p_b);
    va_c = AE_ZALIGN64();

#if TFLITE_SINGLE_ROUNDING
    inp1_left_shift = 31 - inp1_left_shift;
    inp1_left_shift = inp1_left_shift << 16 | inp1_left_shift;
    inp2_left_shift = 31 - inp2_left_shift;
    inp2_left_shift = inp2_left_shift << 16 | inp2_left_shift;
    out_left_shift = 31 - out_left_shift;
    out_left_shift = out_left_shift << 16 | out_left_shift;
#endif

    for(i=0; i<num_simd8_ops; i++){

        ae_int8x8 a0_7, b0_7;
        AE_LA8X8_IP(a0_7, va_a, (ae_int8x8*)p_a);
        AE_LA8X8_IP(b0_7, va_b, (ae_int8x8*)p_b);
        AE_SUBW8(a0_3, a4_7, a0_7, za8);
        AE_SUBW8(b0_3, b4_7, b0_7, zb8);
        
        shifted_a0_1 = AE_SEXT32X2D16_32(a0_3);
        shifted_a2_3 = AE_SEXT32X2D16_10(a0_3);
        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);
        
        shifted_a4_5 = AE_SEXT32X2D16_32(a4_7);
        shifted_a6_7 = AE_SEXT32X2D16_10(a4_7);
        shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
        shifted_a6_7 = AE_SLAA32S(shifted_a6_7,left_shift);
        
        shifted_b0_1 = AE_SEXT32X2D16_32(b0_3);
        shifted_b2_3 = AE_SEXT32X2D16_10(b0_3);
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);
        
        shifted_b4_5 = AE_SEXT32X2D16_32(b4_7);
        shifted_b6_7 = AE_SEXT32X2D16_10(b4_7);            
        shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
        shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);
        
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_a0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_a2_3, shifted_a2_3, inp1_multiplier, inp1_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_a4_5, shifted_a4_5, inp1_multiplier, inp1_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_a6_7, shifted_a6_7, inp1_multiplier, inp1_left_shift, inp1_left_shift);

        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_b0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_b2_3, shifted_b2_3, inp2_multiplier, inp2_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_b4_5, shifted_b4_5, inp2_multiplier, inp2_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_b6_7, shifted_b6_7, inp2_multiplier, inp2_left_shift, inp1_left_shift);
#else
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a2_3, shifted_a2_3, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a4_5, shifted_a4_5, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a6_7, shifted_a6_7, inp1_multiplier, inp1_left_shift);
        
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b2_3, shifted_b2_3, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b4_5, shifted_b4_5, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b6_7, shifted_b6_7, inp2_multiplier, inp2_left_shift);
#endif
        
        // Raw Sum
        raw_sum0_1 = AE_ADD32S(scaled_a0_1, scaled_b0_1);
        raw_sum2_3 = AE_ADD32S(scaled_a2_3, scaled_b2_3);
        raw_sum4_5 = AE_ADD32S(scaled_a4_5, scaled_b4_5);
        raw_sum6_7 = AE_ADD32S(scaled_a6_7, scaled_b6_7);
        
        // Raw Output
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out0_1, raw_sum0_1, out_multiplier, out_left_shift, out_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out2_3, raw_sum2_3, out_multiplier, out_left_shift, out_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out4_5, raw_sum4_5, out_multiplier, out_left_shift, out_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out6_7, raw_sum6_7, out_multiplier, out_left_shift, out_left_shift);
#else
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out0_1, raw_sum0_1, out_multiplier, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out2_3, raw_sum2_3, out_multiplier, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out4_5, raw_sum4_5, out_multiplier, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out6_7, raw_sum6_7, out_multiplier, out_left_shift);
#endif        

        out0_1 = AE_ADD32S(out0_1, zc);
        out2_3 = AE_ADD32S(out2_3, zc);
        out4_5 = AE_ADD32S(out4_5, zc);
        out6_7 = AE_ADD32S(out6_7, zc);
        
        ae_int8x8 res0_3 = AE_SAT8X4X32_H(out0_1, out2_3);
        ae_int8x8 res4_7 = AE_SAT8X4X32_H(out4_5, out6_7);
        ae_int8x8 res0_7 = AE_SEL8I(res0_3, res4_7, 1);
        res0_7 = AE_MIN8(res0_7, amax8);
        res0_7 = AE_MAX8(res0_7, amin8);

        AE_SA8X8_IP(res0_7, va_c, (ae_int8x8 *)p_c);
    }

    if(num_scalar_ops){

        ae_int8x8 a0_7, b0_7;
        AE_LAV8X8_XP(a0_7, va_a, (ae_int8x8*)p_a, num_scalar_ops);
        AE_LAV8X8_XP(b0_7, va_b, (ae_int8x8*)p_b, num_scalar_ops);
        AE_SUBW8(a0_3, a4_7, a0_7, za8);
        AE_SUBW8(b0_3, b4_7, b0_7, zb8);
        
        shifted_a0_1 = AE_SEXT32X2D16_32(a0_3);
        shifted_a2_3 = AE_SEXT32X2D16_10(a0_3);
        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);
        
        shifted_a4_5 = AE_SEXT32X2D16_32(a4_7);
        shifted_a6_7 = AE_SEXT32X2D16_10(a4_7);
        shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
        shifted_a6_7 = AE_SLAA32S(shifted_a6_7,left_shift);
        
        shifted_b0_1 = AE_SEXT32X2D16_32(b0_3);
        shifted_b2_3 = AE_SEXT32X2D16_10(b0_3);
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);
        
        shifted_b4_5 = AE_SEXT32X2D16_32(b4_7);
        shifted_b6_7 = AE_SEXT32X2D16_10(b4_7);            
        shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
        shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);
        
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_a0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_a2_3, shifted_a2_3, inp1_multiplier, inp1_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_a4_5, shifted_a4_5, inp1_multiplier, inp1_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_a6_7, shifted_a6_7, inp1_multiplier, inp1_left_shift, inp1_left_shift);

        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_b0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_b2_3, shifted_b2_3, inp2_multiplier, inp2_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_b4_5, shifted_b4_5, inp2_multiplier, inp2_left_shift, inp1_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(scaled_b6_7, shifted_b6_7, inp2_multiplier, inp2_left_shift, inp1_left_shift);
#else
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a2_3, shifted_a2_3, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a4_5, shifted_a4_5, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a6_7, shifted_a6_7, inp1_multiplier, inp1_left_shift);
        
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b2_3, shifted_b2_3, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b4_5, shifted_b4_5, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b6_7, shifted_b6_7, inp2_multiplier, inp2_left_shift);
#endif        
        // Raw Difference
        raw_sum0_1 = AE_ADD32S(scaled_a0_1, scaled_b0_1);
        raw_sum2_3 = AE_ADD32S(scaled_a2_3, scaled_b2_3);
        raw_sum4_5 = AE_ADD32S(scaled_a4_5, scaled_b4_5);
        raw_sum6_7 = AE_ADD32S(scaled_a6_7, scaled_b6_7);
        
        // Raw Output
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out0_1, raw_sum0_1, out_multiplier, out_left_shift, out_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out2_3, raw_sum2_3, out_multiplier, out_left_shift, out_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out4_5, raw_sum4_5, out_multiplier, out_left_shift, out_left_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out6_7, raw_sum6_7, out_multiplier, out_left_shift, out_left_shift);
#else
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out0_1, raw_sum0_1, out_multiplier, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out2_3, raw_sum2_3, out_multiplier, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out4_5, raw_sum4_5, out_multiplier, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out6_7, raw_sum6_7, out_multiplier, out_left_shift);
#endif
        
        out0_1 = AE_ADD32S(out0_1, zc);
        out2_3 = AE_ADD32S(out2_3, zc);
        out4_5 = AE_ADD32S(out4_5, zc);
        out6_7 = AE_ADD32S(out6_7, zc);
        
        ae_int8x8 res0_3 = AE_SAT8X4X32_H(out0_1, out2_3);
        ae_int8x8 res4_7 = AE_SAT8X4X32_H(out4_5, out6_7);
        ae_int8x8 res0_7 = AE_SEL8I(res0_3, res4_7, 1);
        res0_7 = AE_MIN8(res0_7, amax8);
        res0_7 = AE_MAX8(res0_7, amin8);

        AE_SAV8X8_XP(res0_7, va_c, (ae_int8x8 *)p_c, num_scalar_ops);
    }
    AE_SA64POS_FP(va_c, p_c);

    return 0; 
}
#else /* XCHAL_HAVE_HIFI1S */
WORD32 xa_nn_elm_add_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
    XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    int i;
    WORD8 *p_a = (WORD8 *)p_inp1;
    WORD8 *p_b = (WORD8 *)p_inp2;
    WORD8 *p_c =          p_out;

    //ae_int8x8 a0_7, b0_7;
    const ae_int32x2 activation_min = AE_MOVDA32(out_activation_min);
    const ae_int32x2 activation_max = AE_MOVDA32(out_activation_max);

    const ae_int16x4  za = -inp1_zero_bias;
    const ae_int16x4  zb = -inp2_zero_bias;
    const ae_int32x2 zc = AE_MOVDA32( out_zero_bias);
    
    const ae_int32x2 ma = AE_MOVDA32(inp1_multiplier);
    const ae_int32x2 mb = AE_MOVDA32(inp2_multiplier);
    const ae_int32x2 mc = AE_MOVDA32( out_multiplier);  // Multiplier into 32x2 variable

    // intermediate results and scratch registers
    ae_int16x4 a0_3, a4_7, b0_3, b4_7;

    ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
    ae_int32x2 shifted_b0_1, shifted_b2_3, shifted_b4_5, shifted_b6_7;

    ae_int32x2 scaled_a0_1, scaled_a2_3, scaled_a4_5, scaled_a6_7;
    ae_int32x2 scaled_b0_1, scaled_b2_3, scaled_b4_5, scaled_b6_7;

    ae_int32x2 raw_sum0_1, raw_sum2_3, raw_sum4_5, raw_sum6_7;
    ae_int32x2 out0_1, out2_3, out4_5, out6_7;

    ae_int16x4 temp1, temp2;

    const int num_simd8_ops = num_elm >> 3;
    const int num_scalar_ops = num_elm & 0x7;

    ae_valign va_a, va_b, va_c;
    va_a = AE_LA64_PP(p_a);
    va_b = AE_LA64_PP(p_b);
    va_c = AE_ZALIGN64();

    for(i=0; i<num_simd8_ops; i++)
    {
        AE_LA8X4S_IP(a0_3, va_a, p_a);
        AE_LA8X4S_IP(a4_7, va_a, p_a);
        AE_LA8X4S_IP(b0_3, va_b, p_b);
        AE_LA8X4S_IP(b4_7, va_b, p_b);
                    
        a0_3 = AE_SUB16(a0_3, za);
        a4_7 = AE_SUB16(a4_7, za);
        b0_3 = AE_SUB16(b0_3, zb);
        b4_7 = AE_SUB16(b4_7, zb);
                
        shifted_a0_1 = AE_SEXT32X2D16_32(a0_3);
        shifted_a2_3 = AE_SEXT32X2D16_10(a0_3);
        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);
            
        shifted_a4_5 = AE_SEXT32X2D16_32(a4_7);
        shifted_a6_7 = AE_SEXT32X2D16_10(a4_7);
        shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
        shifted_a6_7 = AE_SLAA32S(shifted_a6_7, left_shift);
            
        shifted_b0_1 = AE_SEXT32X2D16_32(b0_3);
        shifted_b2_3 = AE_SEXT32X2D16_10(b0_3); 
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);
            
        shifted_b4_5 = AE_SEXT32X2D16_32(b4_7);
        shifted_b6_7 = AE_SEXT32X2D16_10(b4_7);
        shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
        shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a0_1, shifted_a0_1, ma, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a2_3, shifted_a2_3, ma, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a4_5, shifted_a4_5, ma, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a6_7, shifted_a6_7, ma, inp1_left_shift);
        
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0_1, shifted_b0_1, mb, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b2_3, shifted_b2_3, mb, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b4_5, shifted_b4_5, mb, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b6_7, shifted_b6_7, mb, inp2_left_shift);
        
        // Raw sum
        raw_sum0_1 = AE_ADD32S(scaled_a0_1, scaled_b0_1);
        raw_sum2_3 = AE_ADD32S(scaled_a2_3, scaled_b2_3);
        raw_sum4_5 = AE_ADD32S(scaled_a4_5, scaled_b4_5);
        raw_sum6_7 = AE_ADD32S(scaled_a6_7, scaled_b6_7);
        
        // Raw Output
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out0_1, raw_sum0_1, mc, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out2_3, raw_sum2_3, mc, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out4_5, raw_sum4_5, mc, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out6_7, raw_sum6_7, mc, out_left_shift);
            
        out0_1 = AE_ADD32S(out0_1, zc);
        out2_3 = AE_ADD32S(out2_3, zc);
        out4_5 = AE_ADD32S(out4_5, zc);
        out6_7 = AE_ADD32S(out6_7, zc);

        /* Clamped out */
        out0_1 = AE_MAX32(out0_1, activation_min);
        out0_1 = AE_MIN32(out0_1, activation_max);
        out2_3 = AE_MAX32(out2_3, activation_min);
        out2_3 = AE_MIN32(out2_3, activation_max);
        out4_5 = AE_MAX32(out4_5, activation_min);
        out4_5 = AE_MIN32(out4_5, activation_max);
        out6_7 = AE_MAX32(out6_7, activation_min);
        out6_7 = AE_MIN32(out6_7, activation_max);          
        
        /* Store output */

        temp1 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out0_1),AE_MOVINT16X4_FROMINT32X2(out2_3));
        temp2 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out4_5),AE_MOVINT16X4_FROMINT32X2(out6_7));

        AE_SA8X4U_IP(temp1, va_c, (ae_int32 *)p_c);
        AE_SA8X4U_IP(temp2, va_c, (ae_int32 *)p_c);
        }
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    if((0 < num_scalar_ops)& (num_scalar_ops < 4))
    {
        AE_LAV8X4S_XP(a0_3, va_a, (ae_int8x4 *)p_a, num_scalar_ops);
        AE_LAV8X4S_XP(b0_3, va_b, (ae_int8x4 *)p_b, num_scalar_ops);

        a0_3 = AE_SUB16(a0_3, za);
        b0_3 = AE_SUB16(b0_3, zb);

        shifted_a0_1 = AE_SEXT32X2D16_32(a0_3);
        shifted_a2_3 = AE_SEXT32X2D16_10(a0_3);
        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

        shifted_b0_1 = AE_SEXT32X2D16_32(b0_3);
        shifted_b2_3 = AE_SEXT32X2D16_10(b0_3); 
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a0_1, shifted_a0_1, ma, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a2_3, shifted_a2_3, ma, inp1_left_shift);
        
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0_1, shifted_b0_1, mb, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b2_3, shifted_b2_3, mb, inp2_left_shift);

        // Raw sum
        raw_sum0_1 = AE_ADD32S(scaled_a0_1, scaled_b0_1);
        raw_sum2_3 = AE_ADD32S(scaled_a2_3, scaled_b2_3);
        
        // Raw Output
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out0_1, raw_sum0_1, mc, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out2_3, raw_sum2_3, mc, out_left_shift);
        
        out0_1 = AE_ADD32S(out0_1, zc);
        out2_3 = AE_ADD32S(out2_3, zc);

        /* Clamped out */
        out0_1 = AE_MAX32(out0_1, activation_min);
        out0_1 = AE_MIN32(out0_1, activation_max);
        out2_3 = AE_MAX32(out2_3, activation_min);
        out2_3 = AE_MIN32(out2_3, activation_max);

        /* Store output */

        temp1 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out0_1),AE_MOVINT16X4_FROMINT32X2(out2_3));

        AE_SAV8X4U_XP(temp1, va_c, (ae_int8x4u *)p_c, num_scalar_ops);
    }
    else if(num_scalar_ops >0)
    {
        int rem_itr = (num_scalar_ops&3);
        AE_LA8X4S_IP(a0_3, va_a, p_a);
        AE_LAV8X4S_XP(a4_7, va_a, (ae_int8x4 *)p_a, rem_itr);
        AE_LA8X4S_IP(b0_3, va_b, p_b);
        AE_LAV8X4S_XP(b4_7, va_b, (ae_int8x4 *)p_b, rem_itr);
                    
        a0_3 = AE_SUB16(a0_3, za);
        a4_7 = AE_SUB16(a4_7, za);
        b0_3 = AE_SUB16(b0_3, zb);
        b4_7 = AE_SUB16(b4_7, zb);
                
        shifted_a0_1 = AE_SEXT32X2D16_32(a0_3);
        shifted_a2_3 = AE_SEXT32X2D16_10(a0_3);
        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);
            
        shifted_a4_5 = AE_SEXT32X2D16_32(a4_7);
        shifted_a6_7 = AE_SEXT32X2D16_10(a4_7);
        shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
        shifted_a6_7 = AE_SLAA32S(shifted_a6_7, left_shift);
            
        shifted_b0_1 = AE_SEXT32X2D16_32(b0_3);
        shifted_b2_3 = AE_SEXT32X2D16_10(b0_3); 
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);
            
        shifted_b4_5 = AE_SEXT32X2D16_32(b4_7);
        shifted_b6_7 = AE_SEXT32X2D16_10(b4_7);
        shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
        shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a0_1, shifted_a0_1, ma, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a2_3, shifted_a2_3, ma, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a4_5, shifted_a4_5, ma, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a6_7, shifted_a6_7, ma, inp1_left_shift);
        
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0_1, shifted_b0_1, mb, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b2_3, shifted_b2_3, mb, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b4_5, shifted_b4_5, mb, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b6_7, shifted_b6_7, mb, inp2_left_shift);
        
        // Raw sum
        raw_sum0_1 = AE_ADD32S(scaled_a0_1, scaled_b0_1);
        raw_sum2_3 = AE_ADD32S(scaled_a2_3, scaled_b2_3);
        raw_sum4_5 = AE_ADD32S(scaled_a4_5, scaled_b4_5);
        raw_sum6_7 = AE_ADD32S(scaled_a6_7, scaled_b6_7);
        
        // Raw Output
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out0_1, raw_sum0_1, mc, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out2_3, raw_sum2_3, mc, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out4_5, raw_sum4_5, mc, out_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out6_7, raw_sum6_7, mc, out_left_shift);
        
        out0_1 = AE_ADD32S(out0_1, zc);
        out2_3 = AE_ADD32S(out2_3, zc);
        out4_5 = AE_ADD32S(out4_5, zc);
        out6_7 = AE_ADD32S(out6_7, zc);

        /* Clamped out */
        out0_1 = AE_MAX32(out0_1, activation_min);
        out0_1 = AE_MIN32(out0_1, activation_max);
        out2_3 = AE_MAX32(out2_3, activation_min);
        out2_3 = AE_MIN32(out2_3, activation_max);
        out4_5 = AE_MAX32(out4_5, activation_min);
        out4_5 = AE_MIN32(out4_5, activation_max);
        out6_7 = AE_MAX32(out6_7, activation_min);
        out6_7 = AE_MIN32(out6_7, activation_max);          
        
        /* Store output */

        temp1 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out0_1),AE_MOVINT16X4_FROMINT32X2(out2_3));
        temp2 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out4_5),AE_MOVINT16X4_FROMINT32X2(out6_7));

        AE_SA8X4U_IP(temp1, va_c, (ae_int32 *)p_c);
        AE_SAV8X4U_XP(temp2, va_c, (ae_int8x4u *)p_c, rem_itr);
    }
    AE_SA64POS_FP(va_c, p_c);
#else
    AE_SA64POS_FP(va_c, p_c);
    for(i=0; i<num_scalar_ops; i++) {

        ae_int32 a, b;
        ae_int32x2 res;

        a = (ae_int32)(p_a[i] + inp1_zero_bias);            // add input biases
        b = (ae_int32)(p_b[i] + inp2_zero_bias);

        a = AE_SLAA32S(a, left_shift);
        b = AE_SLAA32S(b, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(a, a, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(b, b, inp2_multiplier, inp2_left_shift);

        res = AE_ADD32S(a, b);                              // add inputs to one 32-bit res

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(res, res, out_multiplier, out_left_shift);

        res = AE_ADD32S(res, out_zero_bias);                // add out zero bias

        res = AE_MAX32(res, activation_min);
        res = AE_MIN32(res, activation_max);

        p_c[i] = (WORD8)AE_MOVAD32_L(res);

    }
#endif
    return 0; 
}
#endif /* XCHAL_HAVE_HIFI1S */

#else
WORD32 xa_nn_elm_add_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
    XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    int i;
    WORD8 *p_a = (WORD8 *)p_inp1;
    WORD8 *p_b = (WORD8 *)p_inp2;
    WORD8 *p_c =          p_out;

    //ae_int8x8 a0_7, b0_7;
    const ae_int32x2 activation_min = AE_MOVDA32(out_activation_min);
    const ae_int32x2 activation_max = AE_MOVDA32(out_activation_max);

    const ae_int16x4  za = -inp1_zero_bias;
    const ae_int16x4  zb = -inp2_zero_bias;
    const ae_int32x2 zc = AE_MOVDA32( out_zero_bias);
    
    xtbool io_pointers_aligned =    ((uintptr_t)p_a%8 == 0) &&
                                    ((uintptr_t)p_b%8 == 0) &&
                                    ((uintptr_t)p_c%8 == 0);


    // intermediate results and scratch registers
    ae_int16x4 a0_3, a4_7, b0_3, b4_7;

    ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
    ae_int32x2 shifted_b0_1, shifted_b2_3, shifted_b4_5, shifted_b6_7;

    ae_int32x2 scaled_a0_1, scaled_a2_3, scaled_a4_5, scaled_a6_7;
    ae_int32x2 scaled_b0_1, scaled_b2_3, scaled_b4_5, scaled_b6_7;

    ae_int32x2 raw_sum0_1, raw_sum2_3, raw_sum4_5, raw_sum6_7;
    ae_int32x2 out0_1, out2_3, out4_5, out6_7;

    ae_int16x4 ONE_16X4 = AE_MOVDA16(1);

    const int num_simd8_ops = num_elm/8;
    const int num_scalar_ops = num_elm%8;

    if(io_pointers_aligned){
        for(i=0; i<num_simd8_ops; i++){
            AE_L8X4F_IP(a0_3, p_a, 4);
            AE_L8X4F_IP(a4_7, p_a, 4);
            AE_L8X4F_IP(b0_3, p_b, 4);
            AE_L8X4F_IP(b4_7, p_b, 4);
            
            a0_3 = AE_SRAI16(a0_3, 8);
            a4_7 = AE_SRAI16(a4_7, 8);
            b0_3 = AE_SRAI16(b0_3, 8);
            b4_7 = AE_SRAI16(b4_7, 8);
            
            a0_3 = AE_SUB16(a0_3, za);
            a4_7 = AE_SUB16(a4_7, za);
            b0_3 = AE_SUB16(b0_3, zb);
            b4_7 = AE_SUB16(b4_7, zb);
            
            AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, ONE_16X4);
            shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
            shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);
            
            AE_MUL16X4(shifted_a4_5, shifted_a6_7, a4_7, ONE_16X4);
            shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
            shifted_a6_7 = AE_SLAA32S(shifted_a6_7, left_shift);
            
            AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, ONE_16X4);
            shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
            shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);
            
            AE_MUL16X4(shifted_b4_5, shifted_b6_7, b4_7, ONE_16X4);
            shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
            shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);

            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a2_3, shifted_a2_3, inp1_multiplier, inp1_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a4_5, shifted_a4_5, inp1_multiplier, inp1_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a6_7, shifted_a6_7, inp1_multiplier, inp1_left_shift);


            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b2_3, shifted_b2_3, inp2_multiplier, inp2_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b4_5, shifted_b4_5, inp2_multiplier, inp2_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b6_7, shifted_b6_7, inp2_multiplier, inp2_left_shift);

            // Raw Sum
            raw_sum0_1 = AE_ADD32S(scaled_a0_1, scaled_b0_1);
            raw_sum2_3 = AE_ADD32S(scaled_a2_3, scaled_b2_3);
            raw_sum4_5 = AE_ADD32S(scaled_a4_5, scaled_b4_5);
            raw_sum6_7 = AE_ADD32S(scaled_a6_7, scaled_b6_7);

            // Raw Output
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out0_1, raw_sum0_1, out_multiplier, out_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out2_3, raw_sum2_3, out_multiplier, out_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out4_5, raw_sum4_5, out_multiplier, out_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out6_7, raw_sum6_7, out_multiplier, out_left_shift);

            out0_1 = AE_ADD32S(out0_1, zc);
            out2_3 = AE_ADD32S(out2_3, zc);
            out4_5 = AE_ADD32S(out4_5, zc);
            out6_7 = AE_ADD32S(out6_7, zc);

            out0_1 = AE_MAX32(out0_1, activation_min);
            out0_1 = AE_MIN32(out0_1, activation_max);
            out2_3 = AE_MAX32(out2_3, activation_min);
            out2_3 = AE_MIN32(out2_3, activation_max);
            out4_5 = AE_MAX32(out4_5, activation_min);
            out4_5 = AE_MIN32(out4_5, activation_max);
            out6_7 = AE_MAX32(out6_7, activation_min);
            out6_7 = AE_MIN32(out6_7, activation_max);          
            
            STORE_8X4_FROM_32X4(p_c, out0_1, out2_3);
            STORE_8X4_FROM_32X4(p_c, out4_5, out6_7);
        }
    } else {
        ALIGN_REGISTER_TYPE va_a, va_b;
        PRIME_8X4F(p_a, va_a);
        PRIME_8X4F(p_b, va_b);

        for(i=0; i<num_simd8_ops; i++){
            AE_LA8X4F_IP(a0_3, va_a, p_a);
            AE_LA8X4F_IP(a4_7, va_a, p_a);
            AE_LA8X4F_IP(b0_3, va_b, p_b);
            AE_LA8X4F_IP(b4_7, va_b, p_b);
            
            a0_3 = AE_SRAI16(a0_3, 8);
            a4_7 = AE_SRAI16(a4_7, 8);
            b0_3 = AE_SRAI16(b0_3, 8);
            b4_7 = AE_SRAI16(b4_7, 8);
            
            a0_3 = AE_SUB16(a0_3, za);
            a4_7 = AE_SUB16(a4_7, za);
            b0_3 = AE_SUB16(b0_3, zb);
            b4_7 = AE_SUB16(b4_7, zb);
            
            AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, ONE_16X4);
            shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
            shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);
            
            AE_MUL16X4(shifted_a4_5, shifted_a6_7, a4_7, ONE_16X4);
            shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
            shifted_a6_7 = AE_SLAA32S(shifted_a6_7, left_shift);
            
            AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, ONE_16X4);
            shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
            shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);
            
            AE_MUL16X4(shifted_b4_5, shifted_b6_7, b4_7, ONE_16X4);
            shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
            shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);

            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a2_3, shifted_a2_3, inp1_multiplier, inp1_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a4_5, shifted_a4_5, inp1_multiplier, inp1_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a6_7, shifted_a6_7, inp1_multiplier, inp1_left_shift);
            
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b2_3, shifted_b2_3, inp2_multiplier, inp2_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b4_5, shifted_b4_5, inp2_multiplier, inp2_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b6_7, shifted_b6_7, inp2_multiplier, inp2_left_shift);
            
            // Raw sum
            raw_sum0_1 = AE_ADD32S(scaled_a0_1, scaled_b0_1);
            raw_sum2_3 = AE_ADD32S(scaled_a2_3, scaled_b2_3);
            raw_sum4_5 = AE_ADD32S(scaled_a4_5, scaled_b4_5);
            raw_sum6_7 = AE_ADD32S(scaled_a6_7, scaled_b6_7);
            
            // Raw Output
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out0_1, raw_sum0_1, out_multiplier, out_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out2_3, raw_sum2_3, out_multiplier, out_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out4_5, raw_sum4_5, out_multiplier, out_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out6_7, raw_sum6_7, out_multiplier, out_left_shift);
            
            out0_1 = AE_ADD32S(out0_1, zc);
            out2_3 = AE_ADD32S(out2_3, zc);
            out4_5 = AE_ADD32S(out4_5, zc);
            out6_7 = AE_ADD32S(out6_7, zc);

            /* Clamped out */
            out0_1 = AE_MAX32(out0_1, activation_min);
            out0_1 = AE_MIN32(out0_1, activation_max);
            out2_3 = AE_MAX32(out2_3, activation_min);
            out2_3 = AE_MIN32(out2_3, activation_max);
            out4_5 = AE_MAX32(out4_5, activation_min);
            out4_5 = AE_MIN32(out4_5, activation_max);
            out6_7 = AE_MAX32(out6_7, activation_min);
            out6_7 = AE_MIN32(out6_7, activation_max);          
            
            /* Store output */
            STORE_8X4_FROM_32X4(p_c, out0_1, out2_3);
            STORE_8X4_FROM_32X4(p_c, out4_5, out6_7);   
        }
    }

    for(i=0; i<num_scalar_ops; i++) {
        ae_int32 a, b;
        ae_int32x2 res;

        a = (ae_int32)(p_a[i] + inp1_zero_bias);            // add input biases
        b = (ae_int32)(p_b[i] + inp2_zero_bias);

        a = AE_SLAA32S(a, left_shift);
        b = AE_SLAA32S(b, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(a, a, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(b, b, inp2_multiplier, inp2_left_shift);

        res = AE_ADD32S(a, b);                              // add inputs to one 32-bit res

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(res, res, out_multiplier, out_left_shift);

        res = AE_ADD32S(res, out_zero_bias);                // add out zero bias

        res = AE_MAX32(res, activation_min);
        res = AE_MIN32(res, activation_max);

        p_c[i] = (WORD8)AE_MOVAD32_L(res);
    }

    return 0; 
}
#endif

static void internal_elm_add_broadcast_2D_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  out_lc,
                            WORD32  in_lc)
{
  int i, j;
  WORD8 * __restrict__ p_a = (WORD8 *)p_inp1;
  WORD8 * __restrict__ p_b = (WORD8 *)p_inp2;
  WORD8 *__restrict__ p_c;

  const ae_int16x4 za = AE_MOVDA16(-inp1_zero_bias);
  const ae_int16x4 zb = AE_MOVDA16(-inp2_zero_bias);

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b0_3, b4_7;

  ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
  ae_int32x2 shifted_b0_1, shifted_b2_3, shifted_b4_5, shifted_b6_7;

  ae_int32x2 raw_sum0_1, raw_sum2_3, raw_sum4_5, raw_sum6_7;

  ae_int16x4 out0, out1;
  ae_int16x4 out2, out3;

  int num_simd8_ops;
  int num_scalar_ops;

  if(out_lc == 1)
  {
    num_simd8_ops = in_lc >> 3;
    num_scalar_ops = in_lc & 7;
  }
  else
  {
    num_simd8_ops = (in_lc >> 4) << 1;
    num_scalar_ops = in_lc & 15;
  }
  for(i = 0; i < out_lc; i++)
  {
    p_a = (WORD8 *)&p_inp1[i * in_lc];
    p_b = (WORD8 *)p_inp2;
    p_c = (WORD8 *)&p_out[i * in_lc];

    xtbool io_pointers_aligned = ((uintptr_t)p_a%4 == 0) && ((uintptr_t)p_b%4==0) && ((uintptr_t)p_c%4==0);

    if (io_pointers_aligned)
    {
      for(j = 0; j < num_simd8_ops; j++)
      {
#if XCHAL_HAVE_HIFI1
        AE_L8X4S_IP(a0_3, p_a, 4);
        AE_L8X4S_IP(a4_7, p_a, 4);
        AE_L8X4S_IP(b0_3, p_b, 4);
        AE_L8X4S_IP(b4_7, p_b, 4);
#else
        AE_L8X4F_IP(a0_3, p_a, 4);
        AE_L8X4F_IP(a4_7, p_a, 4);
        AE_L8X4F_IP(b0_3, p_b, 4);
        AE_L8X4F_IP(b4_7, p_b, 4);

        a0_3 = AE_SRAI16(a0_3, 8);
        a4_7 = AE_SRAI16(a4_7, 8);
        b0_3 = AE_SRAI16(b0_3, 8);
        b4_7 = AE_SRAI16(b4_7, 8);
#endif
        // Add input zero bias
        a0_3 = AE_SUB16S(a0_3, za);
        a4_7 = AE_SUB16S(a4_7, za);
        b0_3 = AE_SUB16S(b0_3, zb);
        b4_7 = AE_SUB16S(b4_7, zb);
        // LSH (and promote to 32-bit)
        AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

        AE_MUL16X4(shifted_a4_5, shifted_a6_7, a4_7, AE_MOVDA16(1));
        shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
        shifted_a6_7 = AE_SLAA32S(shifted_a6_7, left_shift);

        AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, AE_MOVDA16(1));
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);

        AE_MUL16X4(shifted_b4_5, shifted_b6_7, b4_7, AE_MOVDA16(1));
        shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
        shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);

        raw_sum0_1 = raw_sum2_3 = raw_sum4_5 = raw_sum6_7 = AE_ZERO32();
        // Scaled input
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_b0_1, shifted_b2_3, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_b4_5, shifted_b6_7, inp2_multiplier, inp2_left_shift);
        // Raw Output
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_sum0_1, raw_sum2_3, out_multiplier, out_left_shift, out_zero_bias);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out1, raw_sum4_5, raw_sum6_7, out_multiplier, out_left_shift, out_zero_bias);
        // Clamp output
        LIMIT16X4(out2, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        LIMIT16X4(out3, out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
#if XCHAL_HAVE_HIFI1
        AE_S8X4_IP(out2,(ae_int32 *)p_c, 4);
        AE_S8X4_IP(out3,(ae_int32 *)p_c, 4);
#else
        STORE_8X4_FROM_16X4(p_c, out2);
        STORE_8X4_FROM_16X4(p_c, out3);
#endif
     }
   }
   else
   {
#if XCHAL_HAVE_HIFI1
    ALIGN_REGISTER_TYPE align_out = AE_ZALIGN64();
#endif
     ALIGN_REGISTER_TYPE va_a, va_b;
     PRIME_8X4F(p_a, va_a);
     PRIME_8X4F(p_b, va_b);
     
     for(j = 0; j < num_simd8_ops; j++)
     {
#if XCHAL_HAVE_HIFI1
       AE_LA8X4S_IP(a0_3, va_a, p_a);
       AE_LA8X4S_IP(a4_7, va_a, p_a);
       AE_LA8X4S_IP(b0_3, va_b, p_b);
       AE_LA8X4S_IP(b4_7, va_b, p_b);
#else
       AE_LA8X4F_IP(a0_3, va_a, p_a);
       AE_LA8X4F_IP(a4_7, va_a, p_a);
       AE_LA8X4F_IP(b0_3, va_b, p_b);
       AE_LA8X4F_IP(b4_7, va_b, p_b);

       a0_3 = AE_SRAI16(a0_3, 8);
       a4_7 = AE_SRAI16(a4_7, 8);
       b0_3 = AE_SRAI16(b0_3, 8);
       b4_7 = AE_SRAI16(b4_7, 8);
#endif
       // Add input zero bias
       a0_3 = AE_SUB16S(a0_3, za);
       a4_7 = AE_SUB16S(a4_7, za);
       b0_3 = AE_SUB16S(b0_3, zb);
       b4_7 = AE_SUB16S(b4_7, zb);

       // LSH (and promote to 32-bit)
       AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
       shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
       shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

       AE_MUL16X4(shifted_a4_5, shifted_a6_7, a4_7, AE_MOVDA16(1));
       shifted_a4_5 = AE_SLAA32S(shifted_a4_5, left_shift);
       shifted_a6_7 = AE_SLAA32S(shifted_a6_7, left_shift);
       AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, AE_MOVDA16(1));

       shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
       shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);
       AE_MUL16X4(shifted_b4_5, shifted_b6_7, b4_7, AE_MOVDA16(1));
       shifted_b4_5 = AE_SLAA32S(shifted_b4_5, left_shift);
       shifted_b6_7 = AE_SLAA32S(shifted_b6_7, left_shift);
       raw_sum0_1 = raw_sum2_3 = raw_sum4_5 = raw_sum6_7 = AE_ZERO32();
       // Scaled input
       MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
       MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_b0_1, shifted_b2_3, inp2_multiplier, inp2_left_shift);
       MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);
       MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_b4_5, shifted_b6_7, inp2_multiplier, inp2_left_shift);
       // Raw Output
       MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_sum0_1, raw_sum2_3, out_multiplier, out_left_shift, out_zero_bias);
       MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out1, raw_sum4_5, raw_sum6_7, out_multiplier, out_left_shift, out_zero_bias);
       // Clamp output
       LIMIT16X4(out2, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
       LIMIT16X4(out3, out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
#if XCHAL_HAVE_HIFI1
        AE_SA8X4U_IP(out2, align_out, (ae_int32 *)p_c);
        AE_SA8X4U_IP(out3, align_out, (ae_int32 *)p_c);    
#else
       STORE_8X4_FROM_16X4(p_c, out2);
       STORE_8X4_FROM_16X4(p_c, out3);
#endif
    }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_c);
#endif  
   }
  }

  if(num_scalar_ops!=0){
    for(i = 0; i < out_lc; i++)
    {
      p_a = (WORD8 *)&p_inp1[i * in_lc + (num_simd8_ops << 3)];
      p_c = (WORD8 *)&p_out[i * in_lc + (num_simd8_ops << 3)];
      p_b = (WORD8 *)&p_inp2[num_simd8_ops << 3];
      for(j = 0; j< num_scalar_ops; j++)
      {
        b0_3 = AE_MOVDA16(p_b[j]);
        a0_3 = AE_MOVDA16(p_a[j]);
        a0_3 = AE_SUB16S(a0_3, za);
        b0_3 = AE_SUB16S(b0_3, zb);
        AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
        AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, AE_MOVDA16(1));
        shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
        shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
        raw_sum0_1 = AE_ZERO32();
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2_OUT32(raw_sum0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
        MPY_BY_QUANT_MACC_ST_ONE_EXP_X2_OUT32(raw_sum0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift);
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT16_ZB(out0, raw_sum0_1, out_multiplier, out_left_shift, out_zero_bias);
        LIMIT16X4(out1, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        *(WORD8 *)p_c = (WORD8)AE_MOVAD16_0(out1);
        p_c++;
     }
   }
  }
}

static void internal_elm_add_broadcast_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  int i;
  WORD8 * __restrict__ p_a = (WORD8 *)p_inp1;
  WORD8 * __restrict__ p_b = (WORD8 *)p_inp2;
  WORD8 *__restrict__ p_c =          p_out;

  ae_int16x4 a0_7, b;

  ae_int16x4  za = AE_MOVDA16(-inp1_zero_bias);
  ae_int16x4  zb = AE_MOVDA16(-inp2_zero_bias);
  WORD32 a_ls, a_mult, b_ls, b_mult;
  a_ls = inp1_left_shift;
  a_mult = inp1_multiplier;
  b_ls = inp2_left_shift;
  b_mult = inp2_multiplier;

  ae_int16x4 a0_3, b0;
  ae_int32x2 shifted_a0_1, shifted_a2_3;
  ae_int32x2 shifted_b0, shifted_b1;
  ae_int32x2 scaled_b0;

  ae_int32x2 raw_sum0_1, raw_sum2_3;
  ae_int16x4 out0, out1;

  const int num_simd4_ops = num_elm >> 2;
  const int num_scalar_ops = num_elm & 3;

  xtbool io_pointers_aligned = ((uintptr_t)p_inp1%4 == 0) && ((uintptr_t)p_inp2%4==0) && ((uintptr_t)p_out%4==0);

  if(io_pointers_aligned)
  {
    b = AE_MOVDA16(p_b[0]);
    b0 = AE_SUB16(b, zb);

    AE_MUL16X4(shifted_b0, shifted_b1, b0, AE_MOVDA16(1));
    shifted_b0 = AE_SLAA32S(shifted_b0, left_shift);
    shifted_b1 = AE_SLAA32S(shifted_b1, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0, shifted_b0, b_mult, b_ls);

    for(i=0; i<num_simd4_ops; i++)
    {
#if XCHAL_HAVE_HIFI1
      AE_L8X4S_IP(a0_7, p_a, 4);
#else
      AE_L8X4F_IP(a0_7, p_a, 4);
      // Add input zero bias
      a0_7 = AE_SRAI16(a0_7, 8);
#endif
      a0_3 = AE_SUB16(a0_7, za);

      // LSH (and promote to 32-bit)
      AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
      shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
      shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

      raw_sum0_1 = raw_sum2_3 = scaled_b0;
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, a_mult, a_ls);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_sum0_1, raw_sum2_3, out_multiplier, out_left_shift, out_zero_bias);
      LIMIT16X4(out1, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
#if XCHAL_HAVE_HIFI1
      AE_S8X4_IP(out1,(ae_int32 *)p_c, 4);
#else
      STORE_8X4_FROM_16X4(p_c, out1);
#endif
    }
  }
  else
  {
    ALIGN_REGISTER_TYPE va_a;
#if XCHAL_HAVE_HIFI1
    ALIGN_REGISTER_TYPE align_out = AE_ZALIGN64();
#endif
    PRIME_8X4F(p_a, va_a);

    b = AE_MOVDA16(p_b[0]);
    b0 = AE_SUB16(b, zb);

    AE_MUL16X4(shifted_b0, shifted_b1, b0, AE_MOVDA16(1));
    shifted_b0 = AE_SLAA32S(shifted_b0, left_shift);
    shifted_b1 = AE_SLAA32S(shifted_b1, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0, shifted_b0, b_mult, b_ls);

    for(i=0; i<num_simd4_ops; i++)
    {
#if XCHAL_HAVE_HIFI1
      AE_LA8X4S_IP(a0_7, va_a, p_a);
#else
      AE_LA8X4F_IP(a0_7, va_a, p_a);
      // Add input zero bias
      a0_7 = AE_SRAI16(a0_7, 8);
#endif
      a0_3 = AE_SUB16(a0_7, za);

      // LSH (and promote to 32-bit)
      AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
      shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
      shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

      raw_sum0_1 = raw_sum2_3 = scaled_b0;
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, a_mult, a_ls);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_sum0_1, raw_sum2_3, out_multiplier, out_left_shift, out_zero_bias);
      LIMIT16X4(out1, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
#if XCHAL_HAVE_HIFI1
      AE_SA8X4U_IP(out1, align_out, (ae_int32 *)p_c);    
#else
      STORE_8X4_FROM_16X4(p_c, out1);
#endif
    }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_c);
#endif
  }
  for(i=0; i<num_scalar_ops; i++)
  {
    a0_7 = AE_MOVDA16(p_a[i]);
    a0_3 = AE_SUB16(a0_7, za);

    AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, AE_MOVDA16(1));
    shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
    shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);

    raw_sum0_1 = raw_sum2_3 = scaled_b0;
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2_OUT32(raw_sum0_1, shifted_a0_1, a_mult, a_ls);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT16_ZB(out0, raw_sum0_1, out_multiplier, out_left_shift, out_zero_bias);
    LIMIT16X4(out1, out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
    *p_c =  (WORD8)AE_MOVAD16_3(out1);
    p_c++;
  }
}

WORD32 xa_nn_elm_add_broadcast_4D_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                      const WORD8 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                      const WORD8 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

  /* Check shapes */
  int i;
  for(i = 0; i < 4; i++)
  {
    if((p_inp1_shape[i] != p_inp2_shape[i] && p_inp1_shape[i] != 1 && p_inp2_shape[i] != 1) ||
       (p_out_shape[i] != (p_inp1_shape[i] > p_inp2_shape[i] ? p_inp1_shape[i] : p_inp2_shape[i])))
    {
      return -1;
    }
  }

  WORD32 inp1_strides[4], inp2_strides[4];
  inp1_strides[3] = 1;
  inp2_strides[3] = 1;
  for(i = 2; i >= 0; i--)
  {
    ae_int32x2 d_str, d_shape;
    d_str = AE_MOVDA32X2(inp1_strides[i + 1], inp2_strides[i + 1]);
    d_shape = AE_MOVDA32X2(p_inp1_shape[i + 1], p_inp2_shape[i + 1]);
    d_str = AE_MULP32X2(d_str, d_shape);
    inp1_strides[i] = AE_MOVAD32_H(d_str);
    inp2_strides[i] = AE_MOVAD32_L(d_str);
  }

  int need_broadcast = 0;
  int inp1_const = 1, inp2_const = 1;
  for(i = 0; i < 4; i++)
  {
    if(p_inp1_shape[i] != p_inp2_shape[i])
    {
      if(p_inp1_shape[i] == 1)
        inp1_strides[i] = 0;
      else
        inp2_strides[i] = 0;

      need_broadcast = 1;
    }
    if(p_inp1_shape[i] != 1)
      inp1_const &= 0;
    if(p_inp2_shape[i] != 1)
      inp2_const &= 0;
  }
  int itr0, itr1, itr2;

  WORD8 *p_out_tmp = p_out;
  const WORD8 *__restrict__ p_inp1_tmp = p_inp1;
  const WORD8 *__restrict__ p_inp2_tmp = p_inp2;
  if(need_broadcast == 0)
  {
    internal_elm_add_broadcast_2D_asym8sxasym8s_asym8s(
                p_out,
                out_zero_bias,
                out_left_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1,
                inp1_zero_bias,
                inp1_left_shift,
                inp1_multiplier,
                p_inp2,
                inp2_zero_bias,
                inp2_left_shift,
                inp2_multiplier,
                left_shift,
                1,
                p_out_shape[0] * inp1_strides[0]);
  }
  else if(inp1_strides[3] == inp2_strides[3])
  {
    WORD32 in_lc, out_lc;
    WORD32 inp1_zb, inp1_ls, inp1_mult;
    WORD32 inp2_zb, inp2_ls, inp2_mult;

    inp1_zb = inp1_zero_bias;
    inp1_ls = inp1_left_shift;
    inp1_mult = inp1_multiplier;
    inp2_zb = inp2_zero_bias;
    inp2_ls = inp2_left_shift;
    inp2_mult = inp2_multiplier;

    in_lc = p_out_shape[2] * p_out_shape[3];
    out_lc = 1;
    if(inp1_strides[2] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp2_ls = inp1_left_shift;
      inp2_mult = inp1_multiplier;
      inp1_zb = inp2_zero_bias;
      inp1_ls = inp2_left_shift;
      inp1_mult = inp2_multiplier;
      const WORD8 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;

      int tmp_strides[2];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];

      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];

      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      in_lc = p_out_shape[3];
      out_lc = p_out_shape[2];
    }
    else if(inp2_strides[2] == 0)
    {
      in_lc = p_out_shape[3];
      out_lc = p_out_shape[2];
    }

    for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const WORD8 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const WORD8 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        internal_elm_add_broadcast_2D_asym8sxasym8s_asym8s(
            p_out_tmp,
            out_zero_bias,
            out_left_shift,
            out_multiplier,
            out_activation_min,
            out_activation_max,
            p_inp1_tmp0,
            inp1_zb,
            inp1_ls,
            inp1_mult,
            p_inp2_tmp0,
            inp2_zb,
            inp2_ls,
            inp2_mult,
            left_shift,
            out_lc,
            in_lc);
        p_out_tmp += in_lc * out_lc;
        p_inp1_tmp0 += inp1_strides[1];
        p_inp2_tmp0 += inp2_strides[1];
      }
      p_inp1_tmp += inp1_strides[0];
      p_inp2_tmp += inp2_strides[0];
    }
  }
  else if(inp1_const == 1 || inp2_const == 1)
  {
    WORD32 inp1_zb, inp1_ls, inp1_mult;
    WORD32 inp2_zb, inp2_ls, inp2_mult;
    inp1_zb = inp1_zero_bias;
    inp1_ls = inp1_left_shift;
    inp1_mult = inp1_multiplier;
    inp2_zb = inp2_zero_bias;
    inp2_ls = inp2_left_shift;
    inp2_mult = inp2_multiplier;
    if(inp1_const == 1)
    {
      inp2_zb = inp1_zero_bias;
      inp2_ls = inp1_left_shift;
      inp2_mult = inp1_multiplier;
      inp1_zb = inp2_zero_bias;
      inp1_ls = inp2_left_shift;
      inp1_mult = inp2_multiplier;

      const WORD8 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
    }
    {
      internal_elm_add_broadcast_asym8sxasym8s_asym8s(
          p_out_tmp,
          out_zero_bias,
          out_left_shift,
          out_multiplier,
          out_activation_min,
          out_activation_max,
          p_inp1_tmp,
          inp1_zb,
          inp1_ls,
          inp1_mult,
          p_inp2_tmp,
          inp2_zb,
          inp2_ls,
          inp2_mult,
          left_shift,
          p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3]);
    }
  }
  else
  {
    WORD32 inp1_zb, inp1_ls, inp1_mult;
    WORD32 inp2_zb, inp2_ls, inp2_mult;
    inp1_zb = inp1_zero_bias;
    inp1_ls = inp1_left_shift;
    inp1_mult = inp1_multiplier;
    inp2_zb = inp2_zero_bias;
    inp2_ls = inp2_left_shift;
    inp2_mult = inp2_multiplier;
    if(inp1_strides[3] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp2_ls = inp1_left_shift;
      inp2_mult = inp1_multiplier;
      inp1_zb = inp2_zero_bias;
      inp1_ls = inp2_left_shift;
      inp1_mult = inp2_multiplier;
      const WORD8 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;

      int tmp_strides[3];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];
      tmp_strides[2] = inp1_strides[2];

      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];
      inp1_strides[2] = inp2_strides[2];

      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      inp2_strides[2] = tmp_strides[2];
    }
    for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const WORD8 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const WORD8 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        const WORD8 *__restrict__ p_inp1_tmp1 = p_inp1_tmp0;
        const WORD8 *__restrict__ p_inp2_tmp1 = p_inp2_tmp0;
        for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
        {
          {
            internal_elm_add_broadcast_asym8sxasym8s_asym8s(
                p_out_tmp,
                out_zero_bias,
                out_left_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1_tmp1,
                inp1_zb,
                inp1_ls,
                inp1_mult,
                p_inp2_tmp1,
                inp2_zb,
                inp2_ls,
                inp2_mult,
                left_shift,
                p_out_shape[3]);
          }
          p_out_tmp += p_out_shape[3];
          p_inp1_tmp1 += inp1_strides[2];
          p_inp2_tmp1 += inp2_strides[2];
        }
        p_inp1_tmp0 += inp1_strides[1];
        p_inp2_tmp0 += inp2_strides[1];
      }
      p_inp1_tmp += inp1_strides[0];
      p_inp2_tmp += inp2_strides[0];
    }
  }
  return 0;
}
