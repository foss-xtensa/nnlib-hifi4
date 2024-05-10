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
#include "xa_nnlib_quant_macros.h"

#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_mul_asym8xasym8_asym8(UWORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   UWORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const   UWORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
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
    XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
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
#if TFLITE_SINGLE_ROUNDING
    int left_shift = out_shift;
    int right_shift = out_shift;
    /* Single rounding doesn't need two shifts */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    int left_shift = out_shift < 0 ? 0 : out_shift;
    int right_shift = out_shift > 0 ? 0 : -out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

    // Taking input zero_bias into 16X4 variable
    temp = AE_MOVDA32X2(inp1_zero_bias, inp1_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias1 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

    temp = AE_MOVDA32X2(inp2_zero_bias, inp2_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias2 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);


    // Taking into 32x2 variable
    op_zero_bias = AE_MOVDA32X2(out_zero_bias, out_zero_bias);

    activation_min = AE_MOVDA32X2(out_activation_min, out_activation_min);
    activation_max = AE_MOVDA32X2(out_activation_max, out_activation_max);

    ae_valign align_out, i1_a, i2_a;
    align_out = AE_ZALIGN64();
    i1_a = AE_LA64_PP(p_i1);
    i2_a = AE_LA64_PP(p_i2);

    for(i=0;i < num_elm>>2;i++)
    {
        ae_f16x4 v1, v2;
        ae_f32x2 prod32, prod10;
        ae_f32x2 clamped_out32, clamped_out10;
        ae_f32x2 unclamped_out32, unclamped_out10;


        AE_LA8X4U_IP(x1, i1_a, p_i1);
        AE_LA8X4U_IP(x2, i2_a, p_i2);

        v1 = AE_ADD16(x1, zero_bias1);
        v2 = AE_ADD16(x2, zero_bias2);

        AE_MUL16X4(prod32, prod10, v1, v2);

        // unclamped result
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out10, prod10, out_multiplier, left_shift, right_shift)
        unclamped_out32 = AE_ADD32S(unclamped_out32, op_zero_bias);
        unclamped_out10 = AE_ADD32S(unclamped_out10, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)
        CLAMP_VAL(clamped_out10, unclamped_out10, activation_min, activation_max)

        // Store Output
        ae_int16x4 temp = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(clamped_out32),AE_MOVINT16X4_FROMINT32X2(clamped_out10));
        AE_SA8X4U_IP(temp, align_out, (ae_int32 *)out);
    }
    // Remainder Loop
    p_i1 = (WORD8 *)p_inp1 + (num_elm & ~3);
    p_i2 = (WORD8 *)p_inp2 + (num_elm & ~3);
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    int rem_itr = (num_elm & 3);
    if(rem_itr)
    {
        ae_f16x4 v1, v2;
        ae_f32x2 prod32, prod10;
        ae_f32x2 clamped_out32, clamped_out10;
        ae_f32x2 unclamped_out32, unclamped_out10;


        AE_LAV8X4U_XP(x1, i1_a, (ae_int8x4u *)p_i1, rem_itr);
        AE_LAV8X4U_XP(x2, i2_a, (ae_int8x4u *)p_i2, rem_itr);

        v1 = AE_ADD16(x1, zero_bias1);
        v2 = AE_ADD16(x2, zero_bias2);

        AE_MUL16X4(prod32, prod10, v1, v2);

        // unclamped result
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out10, prod10, out_multiplier, left_shift, right_shift)
        unclamped_out32 = AE_ADD32S(unclamped_out32, op_zero_bias);
        unclamped_out10 = AE_ADD32S(unclamped_out10, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)
        CLAMP_VAL(clamped_out10, unclamped_out10, activation_min, activation_max)

        // Store Output
        ae_int16x4 temp = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(clamped_out32),AE_MOVINT16X4_FROMINT32X2(clamped_out10));
        AE_SAV8X4U_XP(temp, align_out, (ae_int8x4u *)out, rem_itr);
    }
    AE_SA64POS_FP(align_out, out);
#else
    AE_SA64POS_FP(align_out, out);

    for(i=0; i < (num_elm & 3); i++)
    {
        ae_f16x4 v1, v2;
        ae_f32x2 prod32, prod10;
        ae_f32x2 clamped_out32;
        ae_f32x2 unclamped_out32;

        WORD16 i1, i2;

        i1 = (WORD16) *((UWORD8 *)p_i1 + i);
        i2 = (WORD16) *((UWORD8 *)p_i2 + i);

        x1 = AE_MOVDA16(i1);
        x2 = AE_MOVDA16(i2);

        v1 = AE_ADD16(x1, zero_bias1);
        v2 = AE_ADD16(x2, zero_bias2);

        AE_MUL16X4(prod32, prod10, v1, v2);

        // unclamped result
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
        unclamped_out32 = AE_ADD32S(unclamped_out32, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)

        // Store Output
        i1 = AE_MOVAD32_H(clamped_out32);
        *out++ = (UWORD8) i1;
    }
#endif
    return 0;
}
#else
WORD32 xa_nn_elm_mul_asym8xasym8_asym8(UWORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   UWORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const   UWORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
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
    XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
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
#if TFLITE_SINGLE_ROUNDING
    int left_shift = out_shift;
    int right_shift = out_shift;
    /* Single rounding doesn't need two shifts */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    int left_shift = out_shift < 0 ? 0 : out_shift;
    int right_shift = out_shift > 0 ? 0 : -out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

    // Taking input zero_bias into 16X4 variable
    temp = AE_MOVDA32X2(inp1_zero_bias, inp1_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias1 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

    temp = AE_MOVDA32X2(inp2_zero_bias, inp2_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias2 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);


    // Taking into 32x2 variable
    op_zero_bias = AE_MOVDA32X2(out_zero_bias, out_zero_bias);

    activation_min = AE_MOVDA32X2(out_activation_min, out_activation_min);
    activation_max = AE_MOVDA32X2(out_activation_max, out_activation_max);

    if(((((unsigned)p_i1)&3) == 0) && ((((unsigned)p_i2)&3) == 0))
    {
        for(i=0;i < num_elm>>2;i++)
        {
            ae_f16x4 v1, v2;
            ae_f32x2 prod32, prod10;
            ae_f32x2 clamped_out32, clamped_out10;
            ae_f32x2 unclamped_out32, unclamped_out10;


            AE_L8X4F_IP(x1, p_i1, 4*sizeof(WORD8));
            AE_L8X4F_IP(x2, p_i2, 4*sizeof(WORD8));

            x1 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x1), 8));
            x2 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x2), 8));

            v1 = AE_ADD16(x1, zero_bias1);
            v2 = AE_ADD16(x2, zero_bias2);

            AE_MUL16X4(prod32, prod10, v1, v2);

            // unclamped result
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out10, prod10, out_multiplier, left_shift, right_shift)
            unclamped_out32 = AE_ADD32S(unclamped_out32, op_zero_bias);
            unclamped_out10 = AE_ADD32S(unclamped_out10, op_zero_bias);

            // clamped_out
            CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)
            CLAMP_VAL(clamped_out10, unclamped_out10, activation_min, activation_max)

            // Store Output
            STORE_8X4_FROM_32X4(out, clamped_out32, clamped_out10)
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
            ae_f32x2 prod32, prod10;
            ae_f32x2 clamped_out32, clamped_out10;
            ae_f32x2 unclamped_out32, unclamped_out10;


            AE_LA8X4U_IP(x1, i1_a, p_i1);
            AE_LA8X4U_IP(x2, i2_a, p_i2);

            v1 = AE_ADD16(x1, zero_bias1);
            v2 = AE_ADD16(x2, zero_bias2);

            AE_MUL16X4(prod32, prod10, v1, v2);

            // unclamped result
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out10, prod10, out_multiplier, left_shift, right_shift)
            unclamped_out32 = AE_ADD32S(unclamped_out32, op_zero_bias);
            unclamped_out10 = AE_ADD32S(unclamped_out10, op_zero_bias);

            // clamped_out
            CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)
            CLAMP_VAL(clamped_out10, unclamped_out10, activation_min, activation_max)

            // Store Output
            STORE_8X4_FROM_32X4(out, clamped_out32, clamped_out10)
        }
    }

    p_i1 = (WORD8 *)p_inp1 + (num_elm & ~3);
    p_i2 = (WORD8 *)p_inp2 + (num_elm & ~3);

    // Remainder Loop
    for(i=0; i < (num_elm & 3); i++)
    {
        ae_f16x4 v1, v2;
        ae_f32x2 prod32, prod10;
        ae_f32x2 clamped_out32;
        ae_f32x2 unclamped_out32;

        WORD16 i1, i2;

        i1 = (WORD16) *((UWORD8 *)p_i1 + i);
        i2 = (WORD16) *((UWORD8 *)p_i2 + i);

        x1 = AE_MOVDA16(i1);
        x2 = AE_MOVDA16(i2);

        v1 = AE_ADD16(x1, zero_bias1);
        v2 = AE_ADD16(x2, zero_bias2);

        AE_MUL16X4(prod32, prod10, v1, v2);

        // unclamped result
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
        unclamped_out32 = AE_ADD32S(unclamped_out32, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)

        // Store Output
        i1 = (WORD16)(AE_MOVAD32_H(clamped_out32));
        *out++ = (UWORD8) i1;
    }

    return 0;
}
#endif

#if XCHAL_HAVE_HIFI1

WORD32 xa_nn_elm_mul_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
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
    XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    unsigned int i = 0;

    /* c = ( a + za ) * ( b + zb ) */
    ae_int16x4 a0_3, a4_7, b0_3, b4_7;
    ae_int32x2 res0_1, res2_3, res4_5, res6_7;

    ae_int32x2 zc = AE_MOVDA32( out_zero_bias);


#if TFLITE_SINGLE_ROUNDING
    int l_shift = out_shift;
    int r_shift = out_shift;

#if XCHAL_HAVE_HIFI1S
    l_shift = 31 - l_shift;
    l_shift = l_shift << 16 | l_shift;
#endif    
    /* Single rounding doesn't need two shifts */
    (void)r_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    int l_shift = out_shift >= 0 ?   out_shift : 0;
    int r_shift = out_shift <  0 ?  -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */


    WORD8 *in1 = (WORD8 *)p_inp1;
    WORD8 *in2 = (WORD8 *)p_inp2;

    unsigned int num_simd8_ops = num_elm/8;
    unsigned int num_scalar_ops = num_elm%8;
    
    ae_valign va_in1, va_in2, va_out;
    va_in1 = AE_LA64_PP(in1);
    va_in2 = AE_LA64_PP(in2);
    va_out = AE_ZALIGN64();
    
#if XCHAL_HAVE_HIFI1S
    ae_int8x8 za8 = AE_MOVDA8(-inp1_zero_bias);
    ae_int8x8 zb8 = AE_MOVDA8(-inp2_zero_bias);
    ae_int8x8 amin8 = AE_MOVDA8(out_activation_min);
    ae_int8x8 amax8 = AE_MOVDA8(out_activation_max);

    for(i=0; i<num_simd8_ops; i++)
    {
        ae_int8x8 a0_7, b0_7;

        AE_LA8X8_IP(a0_7, va_in1, (ae_int8x8 *)in1);
        AE_LA8X8_IP(b0_7, va_in2, (ae_int8x8 *)in2);

        AE_SUBW8(a0_3, a4_7, a0_7, za8);
        AE_SUBW8(b0_3, b4_7, b0_7, zb8);

        AE_MUL16X4(res0_1, res2_3, a0_3, b0_3);                     // a & b are 9-bit vals in 16-bit containers.
        AE_MUL16X4(res4_5, res6_7, a4_7, b4_7);                     // res, therefore is 18-bit val in 32-bit container.
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(res0_1, res0_1, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(res2_3, res2_3, out_multiplier, l_shift, r_shift);

        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(res4_5, res4_5, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(res6_7, res6_7, out_multiplier, l_shift, r_shift);
#else        
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res0_1, res0_1, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res2_3, res2_3, out_multiplier, l_shift, r_shift);

        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res4_5, res4_5, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res6_7, res6_7, out_multiplier, l_shift, r_shift);
#endif
        /* add output zero bias */
        res0_1 = AE_ADD32S(res0_1, zc);     res2_3 = AE_ADD32S(res2_3, zc);
        res4_5 = AE_ADD32S(res4_5, zc);     res6_7 = AE_ADD32S(res6_7, zc);

        ae_int8x8 res0_3 = AE_SAT8X4X32_H(res0_1, res2_3);
        ae_int8x8 res4_7 = AE_SAT8X4X32_H(res4_5, res6_7);
        ae_int8x8 res0_7 = AE_SEL8I(res0_3, res4_7, 1);
        res0_7 = AE_MIN8(res0_7, amax8);
        res0_7 = AE_MAX8(res0_7, amin8);
        /* Clamp to activation max/min */

        AE_SA8X8_IP(res0_7, va_out, (ae_int8x8 *)p_out);
    }
    if(num_scalar_ops)
    {
        ae_int8x8 a0_7, b0_7;

        AE_LAV8X8_XP(a0_7, va_in1, (ae_int8x8 *)in1, num_scalar_ops);
        AE_LAV8X8_XP(b0_7, va_in2, (ae_int8x8 *)in2, num_scalar_ops);

        AE_SUBW8(a0_3, a4_7, a0_7, za8);
        AE_SUBW8(b0_3, b4_7, b0_7, zb8);

        AE_MUL16X4(res0_1, res2_3, a0_3, b0_3);                     // a & b are 9-bit vals in 16-bit containers.
        AE_MUL16X4(res4_5, res6_7, a4_7, b4_7);                     // res, therefore is 18-bit val in 32-bit container.
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(res0_1, res0_1, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(res2_3, res2_3, out_multiplier, l_shift, r_shift);

        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(res4_5, res4_5, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(res6_7, res6_7, out_multiplier, l_shift, r_shift);
#else        
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res0_1, res0_1, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res2_3, res2_3, out_multiplier, l_shift, r_shift);

        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res4_5, res4_5, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res6_7, res6_7, out_multiplier, l_shift, r_shift);
#endif
        /* add output zero bias */
        res0_1 = AE_ADD32S(res0_1, zc);     res2_3 = AE_ADD32S(res2_3, zc);
        res4_5 = AE_ADD32S(res4_5, zc);     res6_7 = AE_ADD32S(res6_7, zc);

        ae_int8x8 res0_3 = AE_SAT8X4X32_H(res0_1, res2_3);
        ae_int8x8 res4_7 = AE_SAT8X4X32_H(res4_5, res6_7);
        ae_int8x8 res0_7 = AE_SEL8I(res0_3, res4_7, 1);
        res0_7 = AE_MIN8(res0_7, amax8);
        res0_7 = AE_MAX8(res0_7, amin8);
        /* Clamp to activation max/min */

        AE_SAV8X8_XP(res0_7, va_out, (ae_int8x8 *)p_out, num_scalar_ops);
    }
    AE_SA64POS_FP(va_out, p_out);
#else
    ae_int16x4 za = AE_MOVDA16(inp1_zero_bias);     // replicate 16LSBs of input into 16x4 output
    ae_int16x4 zb = AE_MOVDA16(inp2_zero_bias);     // zero_bias is already signed, no need for ZE
    ae_int32x2 activation_max = AE_MOVDA32(out_activation_max);
    ae_int32x2 activation_min = AE_MOVDA32(out_activation_min);
    ae_int16x4 temp1, temp2;                        //intermediate results

    for(i=0; i<num_simd8_ops; i++)
    {

        AE_LA8X4S_IP(a0_3, va_in1, in1);    AE_LA8X4S_IP(a4_7, va_in1, in1);
        AE_LA8X4S_IP(b0_3, va_in2, in2);    AE_LA8X4S_IP(b4_7, va_in2, in2);

        a0_3 = AE_ADD16(a0_3, za);  a4_7 = AE_ADD16(a4_7, za);      // Add zero points
        b0_3 = AE_ADD16(b0_3, zb);  b4_7 = AE_ADD16(b4_7, zb);

        AE_MUL16X4(res0_1, res2_3, a0_3, b0_3);                     // a & b are 9-bit vals in 16-bit containers.
        AE_MUL16X4(res4_5, res6_7, a4_7, b4_7);                     // res, therefore is 18-bit val in 32-bit container.
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res0_1, res0_1, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res2_3, res2_3, out_multiplier, l_shift, r_shift);

        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res4_5, res4_5, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res6_7, res6_7, out_multiplier, l_shift, r_shift);


        /* add output zero bias */
        res0_1 = AE_ADD32S(res0_1, zc);     res2_3 = AE_ADD32S(res2_3, zc);
        res4_5 = AE_ADD32S(res4_5, zc);     res6_7 = AE_ADD32S(res6_7, zc);

        /* Clamp to activation max/min */
        res0_1 = AE_MAX32(res0_1, activation_min);
        res0_1 = AE_MIN32(res0_1, activation_max);
        res2_3 = AE_MAX32(res2_3, activation_min);
        res2_3 = AE_MIN32(res2_3, activation_max);
        res4_5 = AE_MAX32(res4_5, activation_min);
        res4_5 = AE_MIN32(res4_5, activation_max);
        res6_7 = AE_MAX32(res6_7, activation_min);
        res6_7 = AE_MIN32(res6_7, activation_max);

        temp1 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(res0_1),AE_MOVINT16X4_FROMINT32X2(res2_3));
        temp2 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(res4_5),AE_MOVINT16X4_FROMINT32X2(res6_7));

        AE_SA8X4U_IP(temp1, va_out, (ae_int32 *)p_out);
        AE_SA8X4U_IP(temp2, va_out, (ae_int32 *)p_out);
        
    }
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    if(num_scalar_ops>4)
    {
        AE_LA8X4S_IP(a0_3, va_in1, in1);    AE_LAV8X4S_XP(a4_7, va_in1, (ae_int8x4 *)in1, (num_scalar_ops&3));
        AE_LA8X4S_IP(b0_3, va_in2, in2);    AE_LAV8X4S_XP(b4_7, va_in2, (ae_int8x4 *)in2, (num_scalar_ops&3));

        a0_3 = AE_ADD16(a0_3, za);  a4_7 = AE_ADD16(a4_7, za);      // Add zero points
        b0_3 = AE_ADD16(b0_3, zb);  b4_7 = AE_ADD16(b4_7, zb);

        AE_MUL16X4(res0_1, res2_3, a0_3, b0_3);                     // a & b are 9-bit vals in 16-bit containers.
        AE_MUL16X4(res4_5, res6_7, a4_7, b4_7);                     // res, therefore is 18-bit val in 32-bit container.
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res0_1, res0_1, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res2_3, res2_3, out_multiplier, l_shift, r_shift);

        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res4_5, res4_5, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res6_7, res6_7, out_multiplier, l_shift, r_shift);


        /* add output zero bias */
        res0_1 = AE_ADD32S(res0_1, zc);     res2_3 = AE_ADD32S(res2_3, zc);
        res4_5 = AE_ADD32S(res4_5, zc);     res6_7 = AE_ADD32S(res6_7, zc);

        /* Clamp to activation max/min */
        res0_1 = AE_MAX32(res0_1, activation_min);
        res0_1 = AE_MIN32(res0_1, activation_max);
        res2_3 = AE_MAX32(res2_3, activation_min);
        res2_3 = AE_MIN32(res2_3, activation_max);
        res4_5 = AE_MAX32(res4_5, activation_min);
        res4_5 = AE_MIN32(res4_5, activation_max);
        res6_7 = AE_MAX32(res6_7, activation_min);
        res6_7 = AE_MIN32(res6_7, activation_max);

        temp1 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(res0_1),AE_MOVINT16X4_FROMINT32X2(res2_3));
        temp2 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(res4_5),AE_MOVINT16X4_FROMINT32X2(res6_7));

        AE_SA8X4U_IP(temp1, va_out, (ae_int32 *)p_out);
        AE_SAV8X4U_XP(temp2, va_out, (ae_int8x4u *)p_out, (num_scalar_ops&3));
    }
    else if(num_scalar_ops > 0)
    {
        AE_LAV8X4S_XP(a0_3, va_in1, (ae_int8x4 *)in1, (num_scalar_ops));
        AE_LAV8X4S_XP(b0_3, va_in2, (ae_int8x4 *)in2, (num_scalar_ops));

        a0_3 = AE_ADD16(a0_3, za);// Add zero points
        b0_3 = AE_ADD16(b0_3, zb);

        AE_MUL16X4(res0_1, res2_3, a0_3, b0_3);                     // a & b are 9-bit vals in 16-bit containers.res, therefore is 18-bit val in 32-bit container.
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res0_1, res0_1, out_multiplier, l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res2_3, res2_3, out_multiplier, l_shift, r_shift);

        /* add output zero bias */
        res0_1 = AE_ADD32S(res0_1, zc);     res2_3 = AE_ADD32S(res2_3, zc);

        /* Clamp to activation max/min */
        res0_1 = AE_MAX32(res0_1, activation_min);
        res0_1 = AE_MIN32(res0_1, activation_max);
        res2_3 = AE_MAX32(res2_3, activation_min);
        res2_3 = AE_MIN32(res2_3, activation_max);

        temp1 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(res0_1),AE_MOVINT16X4_FROMINT32X2(res2_3));

        AE_SAV8X4U_XP(temp1, va_out, (ae_int8x4u *)p_out, (num_scalar_ops));
    }
    AE_SA64POS_FP(va_out, p_out);
#else
    AE_SA64POS_FP(va_out, p_out);

    for(i=0; i<num_scalar_ops; i++)
    {
        ae_int32 tmp = (in1[i] + inp1_zero_bias) *
                            ( in2[i] + inp2_zero_bias );

        ae_int32x2 res = tmp;

        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res, res, out_multiplier, l_shift, r_shift);

        res = AE_ADD32S(res, zc);

        res = AE_MAX32(res, activation_min);
        res = AE_MIN32(res, activation_max);

        *p_out = (WORD8)AE_MOVAD32_L(res);
         p_out++;
    }
#endif
#endif
    return 0;
}

#else

WORD32 xa_nn_elm_mul_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
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
    XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    unsigned int i = 0;

    /* c = ( a + za ) * ( b + zb ) */
    ae_int16x4 a0_3, a4_7, b0_3, b4_7;

    ae_int32x2 res0_1, res2_3, res4_5, res6_7;

    ae_int16x4 za = AE_MOVDA16(inp1_zero_bias);     // replicate 16LSBs of input into 16x4 output
    ae_int16x4 zb = AE_MOVDA16(inp2_zero_bias);     // zero_bias is already signed, no need for ZE
    ae_int32x2 zc = AE_MOVDA32( out_zero_bias);

#if TFLITE_SINGLE_ROUNDING
    int l_shift = out_shift;
    int r_shift = out_shift;
    /* Single rounding doesn't need two shifts */
    (void)r_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    int l_shift = out_shift >= 0 ?   out_shift : 0;
    int r_shift = out_shift <  0 ?  -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */

    ae_int32x2 activation_max = AE_MOVDA32(out_activation_max);
    ae_int32x2 activation_min = AE_MOVDA32(out_activation_min);

    WORD8 *in1 = (WORD8 *)p_inp1;
    WORD8 *in2 = (WORD8 *)p_inp2;

    xtbool io_pointers_aligned = ((uintptr_t)in1%4 == 0) && ((uintptr_t)in2%4==0) && ((uintptr_t)p_out%4==0);

    unsigned int num_simd8_ops = num_elm/8;
    unsigned int num_scalar_ops = num_elm%8;

    if(io_pointers_aligned){
        for(i=0; i<num_simd8_ops; i++){

            AE_L8X4F_IP(a0_3, in1, 4);  AE_L8X4F_IP(a4_7, in1, 4);      // Load 8bit and SignEx to 16bit
            AE_L8X4F_IP(b0_3, in2, 4);  AE_L8X4F_IP(b4_7, in2, 4);
            a0_3 = AE_SRAI16(a0_3, 8);
            b0_3 = AE_SRAI16(b0_3, 8);
            a4_7 = AE_SRAI16(a4_7, 8);
            b4_7 = AE_SRAI16(b4_7, 8);

            a0_3 = AE_ADD16(a0_3, za);  a4_7 = AE_ADD16(a4_7, za);      // Add zero points
            b0_3 = AE_ADD16(b0_3, zb);  b4_7 = AE_ADD16(b4_7, zb);

            AE_MUL16X4(res0_1, res2_3, a0_3, b0_3);                     // a & b are 9-bit vals in 16-bit containers.
            AE_MUL16X4(res4_5, res6_7, a4_7, b4_7);                     // res, therefore is 18-bit val in 32-bit container.

            MPY_BY_QUANT_MULT_SLS_X2_OUT32(res0_1, res0_1, out_multiplier, l_shift, r_shift);
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(res2_3, res2_3, out_multiplier, l_shift, r_shift);

            MPY_BY_QUANT_MULT_SLS_X2_OUT32(res4_5, res4_5, out_multiplier, l_shift, r_shift);
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(res6_7, res6_7, out_multiplier, l_shift, r_shift);

            /* add output zero bias */
            res0_1 = AE_ADD32S(res0_1, zc);     res2_3 = AE_ADD32S(res2_3, zc);
            res4_5 = AE_ADD32S(res4_5, zc);     res6_7 = AE_ADD32S(res6_7, zc);

            /* Clamp to activation max/min */
            res0_1 = AE_MAX32(res0_1, activation_min);
            res0_1 = AE_MIN32(res0_1, activation_max);
            res2_3 = AE_MAX32(res2_3, activation_min);
            res2_3 = AE_MIN32(res2_3, activation_max);
            res4_5 = AE_MAX32(res4_5, activation_min);
            res4_5 = AE_MIN32(res4_5, activation_max);
            res6_7 = AE_MAX32(res6_7, activation_min);
            res6_7 = AE_MIN32(res6_7, activation_max);

            STORE_8X4_FROM_32X4(p_out, res0_1, res2_3);
            STORE_8X4_FROM_32X4(p_out, res4_5, res6_7);
        }
    }else{
        ALIGN_REGISTER_TYPE va_in1, va_in2;
        PRIME_8X4F(in1, va_in1);
        PRIME_8X4F(in2, va_in2);
        for(i=0; i<num_simd8_ops; i++){

            AE_LA8X4F_IP(a0_3, va_in1, in1);    AE_LA8X4F_IP(a4_7, va_in1, in1);
            AE_LA8X4F_IP(b0_3, va_in2, in2);    AE_LA8X4F_IP(b4_7, va_in2, in2);
            a0_3 = AE_SRAI16(a0_3, 8);
            a4_7 = AE_SRAI16(a4_7, 8);
            b0_3 = AE_SRAI16(b0_3, 8);
            b4_7 = AE_SRAI16(b4_7, 8);

            a0_3 = AE_ADD16(a0_3, za);  a4_7 = AE_ADD16(a4_7, za);      // Add zero points
            b0_3 = AE_ADD16(b0_3, zb);  b4_7 = AE_ADD16(b4_7, zb);

            AE_MUL16X4(res0_1, res2_3, a0_3, b0_3);                     // a & b are 9-bit vals in 16-bit containers.
            AE_MUL16X4(res4_5, res6_7, a4_7, b4_7);                     // res, therefore is 18-bit val in 32-bit container.

            MPY_BY_QUANT_MULT_SLS_X2_OUT32(res0_1, res0_1, out_multiplier, l_shift, r_shift);
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(res2_3, res2_3, out_multiplier, l_shift, r_shift);

            MPY_BY_QUANT_MULT_SLS_X2_OUT32(res4_5, res4_5, out_multiplier, l_shift, r_shift);
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(res6_7, res6_7, out_multiplier, l_shift, r_shift);

            /* add output zero bias */
            res0_1 = AE_ADD32S(res0_1, zc);     res2_3 = AE_ADD32S(res2_3, zc);
            res4_5 = AE_ADD32S(res4_5, zc);     res6_7 = AE_ADD32S(res6_7, zc);

            /* Clamp to activation max/min */
            res0_1 = AE_MAX32(res0_1, activation_min);
            res0_1 = AE_MIN32(res0_1, activation_max);
            res2_3 = AE_MAX32(res2_3, activation_min);
            res2_3 = AE_MIN32(res2_3, activation_max);
            res4_5 = AE_MAX32(res4_5, activation_min);
            res4_5 = AE_MIN32(res4_5, activation_max);
            res6_7 = AE_MAX32(res6_7, activation_min);
            res6_7 = AE_MIN32(res6_7, activation_max);

            STORE_8X4_FROM_32X4(p_out, res0_1, res2_3);
            STORE_8X4_FROM_32X4(p_out, res4_5, res6_7);
        }
    }

    for(i=0; i<num_scalar_ops; i++){
        ae_int32 tmp = (in1[i] + inp1_zero_bias) *
                            ( in2[i] + inp2_zero_bias );

        ae_int32x2 res = tmp;

        MPY_BY_QUANT_MULT_SLS_X2_OUT32(res, res, out_multiplier, l_shift, r_shift);

        res = AE_ADD32S(res, zc);

        res = AE_MAX32(res, activation_min);
        res = AE_MIN32(res, activation_max);

        *p_out = (WORD8)AE_MOVAD32_L(res);
        p_out++;
    }

    return 0;
}

#endif

#if XCHAL_HAVE_HIFI1S
static void internal_elm_mul_broadcast_2D_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  out_lc,
                            WORD32  in_lc)

{
  int i, j;
  WORD8 * __restrict__ p_a; 
  WORD8 * __restrict__ p_b; 
  WORD8 *__restrict__ p_c;

#if TFLITE_SINGLE_ROUNDING
  int l_shift = out_shift;
  int r_shift = out_shift;
  l_shift = 31 - l_shift;
  l_shift = l_shift << 16 | l_shift;
  /* Single rounding doesn't need two shifts */
  (void)r_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int l_shift = out_shift >= 0 ?   out_shift : 0;
  int r_shift = out_shift <  0 ?  -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  const ae_int16x4 za = AE_MOVDA16(-inp1_zero_bias);
  const ae_int16x4 zb = AE_MOVDA16(-inp2_zero_bias);

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b0_3, b4_7;

  ae_int32x2 raw_mul0_1, raw_mul2_3, raw_mul4_5, raw_mul6_7;
  ae_int32x2 out_mul0_1, out_mul2_3, out_mul4_5, out_mul6_7;

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

    ae_valign align_out = AE_ZALIGN64();

    ae_valign va_a, va_b;
    va_a = AE_LA64_PP(p_a);
    va_b = AE_LA64_PP(p_b);
    ae_int8x8 a0_7, b0_7;
    ae_int8x8 za8 = AE_MOVDA8(-inp1_zero_bias);
    ae_int8x8 zb8 = AE_MOVDA8(-inp2_zero_bias);
    ae_int8x8 amin8 = AE_MOVDA8(out_activation_min);
    ae_int8x8 amax8 = AE_MOVDA8(out_activation_max);
    
    for(j = 0; j < num_simd8_ops; j++)
    {
      AE_LA8X8_IP(a0_7, va_a, (ae_int8x8 *)p_a);
      AE_LA8X8_IP(b0_7, va_b, (ae_int8x8 *)p_b);
      AE_SUBW8(a0_3, a4_7, a0_7, za8);
      AE_SUBW8(b0_3, b4_7, b0_7, zb8);

      AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
      AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b4_7);

#if TFLITE_SINGLE_ROUNDING
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul4_5, raw_mul4_5, AE_MOVDA32(out_multiplier), l_shift, r_shift);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul6_7, raw_mul6_7, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#else
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul4_5, raw_mul4_5, AE_MOVDA32(out_multiplier), l_shift, r_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul6_7, raw_mul6_7, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#endif

      out_mul0_1 = AE_ADD32S(out_mul0_1, AE_MOVDA32(out_zero_bias));
      out_mul2_3 = AE_ADD32S(out_mul2_3, AE_MOVDA32(out_zero_bias));
      out_mul4_5 = AE_ADD32S(out_mul4_5, AE_MOVDA32(out_zero_bias));
      out_mul6_7 = AE_ADD32S(out_mul6_7, AE_MOVDA32(out_zero_bias));

      ae_int8x8 res0_3 = AE_SAT8X4X32_H(out_mul0_1, out_mul2_3);
      ae_int8x8 res4_7 = AE_SAT8X4X32_H(out_mul4_5, out_mul6_7);
      ae_int8x8 res0_7 = AE_SEL8I(res0_3, res4_7, 1);
      res0_7 = AE_MIN8(res0_7, amax8);
      res0_7 = AE_MAX8(res0_7, amin8);

      AE_SA8X8_IP(res0_7, align_out, (ae_int8x8 *)p_c);
    }
    AE_SA64POS_FP(align_out, p_c);
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
        AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#else        
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#endif        
        out_mul0_1 = AE_ADD32S(out_mul0_1, AE_MOVDA32(out_zero_bias));
        CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
        *(WORD8 *)p_c = (WORD8)AE_MOVAD32_H(out_mul0_1);
        p_c++;
     }
   }
  }
}
#else /* XCHAL_HAVE_HIFI1S */
static void internal_elm_mul_broadcast_2D_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  out_lc,
                            WORD32  in_lc)

{
  int i, j;
  WORD8 * __restrict__ p_a; 
  WORD8 * __restrict__ p_b; 
  WORD8 *__restrict__ p_c;

#if TFLITE_SINGLE_ROUNDING
  int l_shift = out_shift;
  int r_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)r_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int l_shift = out_shift >= 0 ?   out_shift : 0;
  int r_shift = out_shift <  0 ?  -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  const ae_int16x4 za = AE_MOVDA16(-inp1_zero_bias);
  const ae_int16x4 zb = AE_MOVDA16(-inp2_zero_bias);

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b0_3, b4_7;

  ae_int32x2 raw_mul0_1, raw_mul2_3, raw_mul4_5, raw_mul6_7;
  ae_int32x2 out_mul0_1, out_mul2_3, out_mul4_5, out_mul6_7;

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
#if XCHAL_HAVE_HIFI1
    ALIGN_REGISTER_TYPE align_out = AE_ZALIGN64();
#endif
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
        AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
        AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b4_7);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul4_5, raw_mul4_5, AE_MOVDA32(out_multiplier), l_shift, r_shift);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul6_7, raw_mul6_7, AE_MOVDA32(out_multiplier), l_shift, r_shift);

        out_mul0_1 = AE_ADD32S(out_mul0_1, AE_MOVDA32(out_zero_bias));
        out_mul2_3 = AE_ADD32S(out_mul2_3, AE_MOVDA32(out_zero_bias));
        out_mul4_5 = AE_ADD32S(out_mul4_5, AE_MOVDA32(out_zero_bias));
        out_mul6_7 = AE_ADD32S(out_mul6_7, AE_MOVDA32(out_zero_bias));

        CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
        CLAMP_VAL(out_mul2_3, out_mul2_3, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
        CLAMP_VAL(out_mul4_5, out_mul4_5, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
        CLAMP_VAL(out_mul6_7, out_mul6_7, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
#if XCHAL_HAVE_HIFI1
        ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(out_mul0_1), AE_MOVF16X4_FROMF32X2(out_mul2_3));
        AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_c);
        out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(out_mul4_5), AE_MOVF16X4_FROMF32X2(out_mul6_7));
        AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_c);    
#else
        STORE_8X4_FROM_32X4(p_c, out_mul0_1, out_mul2_3);
        STORE_8X4_FROM_32X4(p_c, out_mul4_5, out_mul6_7);
#endif
     }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_c);
#endif   
   }
   else
   {
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

       AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
       AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b4_7);

       MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
       MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
       MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul4_5, raw_mul4_5, AE_MOVDA32(out_multiplier), l_shift, r_shift);
       MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul6_7, raw_mul6_7, AE_MOVDA32(out_multiplier), l_shift, r_shift);

       out_mul0_1 = AE_ADD32S(out_mul0_1, AE_MOVDA32(out_zero_bias));
       out_mul2_3 = AE_ADD32S(out_mul2_3, AE_MOVDA32(out_zero_bias));
       out_mul4_5 = AE_ADD32S(out_mul4_5, AE_MOVDA32(out_zero_bias));
       out_mul6_7 = AE_ADD32S(out_mul6_7, AE_MOVDA32(out_zero_bias));

       CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
       CLAMP_VAL(out_mul2_3, out_mul2_3, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
       CLAMP_VAL(out_mul4_5, out_mul4_5, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
       CLAMP_VAL(out_mul6_7, out_mul6_7, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
#if XCHAL_HAVE_HIFI1
        ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(out_mul0_1), AE_MOVF16X4_FROMF32X2(out_mul2_3));
        AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_c);
        out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(out_mul4_5), AE_MOVF16X4_FROMF32X2(out_mul6_7));
        AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_c);    
#else
       STORE_8X4_FROM_32X4(p_c, out_mul0_1, out_mul2_3);
       STORE_8X4_FROM_32X4(p_c, out_mul4_5, out_mul6_7);
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
        AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
        out_mul0_1 = AE_ADD32S(out_mul0_1, AE_MOVDA32(out_zero_bias));
        CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
        *(WORD8 *)p_c = (WORD8)AE_MOVAD32_H(out_mul0_1);
        p_c++;
     }
   }
  }
}
#endif /* XCHAL_HAVE_HIFI1S */

#if XCHAL_HAVE_HIFI1S
static void internal_elm_mul_broadcast_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  num_elm)
{
#if TFLITE_SINGLE_ROUNDING
  int l_shift = out_shift;
  int r_shift = out_shift;
  l_shift = 31 - l_shift;
  l_shift = l_shift << 16 | l_shift;  
  /* Single rounding doesn't need two shifts */
  (void)r_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int l_shift = out_shift >= 0 ?   out_shift : 0;
  int r_shift = out_shift <  0 ?  -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  int i;
  WORD8 * __restrict__ p_a = (WORD8 *)p_inp1;
  WORD8 * __restrict__ p_b = (WORD8 *)p_inp2;
  WORD8 * __restrict__ p_c =          p_out;

  ae_int16x4 b;
  ae_int16x4  zb = AE_MOVDA16(-inp2_zero_bias);

  ae_int16x4 a0_3, a4_7, b0;

  ae_int32x2 raw_mul0_1, raw_mul2_3, raw_mul4_5, raw_mul6_7;
  ae_int32x2 out_mul0_1, out_mul2_3, out_mul4_5, out_mul6_7;

  const int num_simd8_ops = num_elm >> 3;
  const int num_scalar_ops = num_elm & 7;
  
  b = AE_MOVDA16(p_b[0]);
  b0 = AE_SUB16(b, zb);

  ae_valign va_a;
  va_a = AE_LA64_PP(p_a);
  ae_valign align_out = AE_ZALIGN64();

  ae_int8x8 a0_7;
  ae_int8x8 za8 = AE_MOVDA8(-inp1_zero_bias);
  ae_int8x8 amin8 = AE_MOVDA8(out_activation_min);
  ae_int8x8 amax8 = AE_MOVDA8(out_activation_max);

  for(i=0; i<num_simd8_ops; i++)
  {
    AE_LA8X8_IP(a0_7, va_a, (ae_int8x8 *)p_a);
    AE_SUBW8(a0_3, a4_7, a0_7, za8);

    // LSH (and promote to 32-bit)
    AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0);
    AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b0);

#if TFLITE_SINGLE_ROUNDING
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul4_5, raw_mul4_5, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul6_7, raw_mul6_7, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#else
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul4_5, raw_mul4_5, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul6_7, raw_mul6_7, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#endif
    out_mul0_1 = AE_ADD32S(out_mul0_1, AE_MOVDA32(out_zero_bias));
    out_mul2_3 = AE_ADD32S(out_mul2_3, AE_MOVDA32(out_zero_bias));
    out_mul4_5 = AE_ADD32S(out_mul4_5, AE_MOVDA32(out_zero_bias));
    out_mul6_7 = AE_ADD32S(out_mul6_7, AE_MOVDA32(out_zero_bias));

    ae_int8x8 res0_3 = AE_SAT8X4X32_H(out_mul0_1, out_mul2_3);
    ae_int8x8 res4_7 = AE_SAT8X4X32_H(out_mul4_5, out_mul6_7);
    ae_int8x8 res0_7 = AE_SEL8I(res0_3, res4_7, 1);
    res0_7 = AE_MIN8(res0_7, amax8);
    res0_7 = AE_MAX8(res0_7, amin8);
    AE_SA8X8_IP(res0_7, align_out, (ae_int8x8 *)p_c);
  }

  if(num_scalar_ops)
  {
    AE_LAV8X8_XP(a0_7, va_a, (ae_int8x8 *)p_a, num_scalar_ops);
    AE_SUBW8(a0_3, a4_7, a0_7, za8);

    AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0);
    AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b0);

#if TFLITE_SINGLE_ROUNDING
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul4_5, raw_mul4_5, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul6_7, raw_mul6_7, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#else
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul4_5, raw_mul4_5, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul6_7, raw_mul6_7, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#endif

    out_mul0_1 = AE_ADD32S(out_mul0_1, AE_MOVDA32(out_zero_bias));
    out_mul2_3 = AE_ADD32S(out_mul2_3, AE_MOVDA32(out_zero_bias));
    out_mul4_5 = AE_ADD32S(out_mul4_5, AE_MOVDA32(out_zero_bias));
    out_mul6_7 = AE_ADD32S(out_mul6_7, AE_MOVDA32(out_zero_bias));

    ae_int8x8 res0_3 = AE_SAT8X4X32_H(out_mul0_1, out_mul2_3);
    ae_int8x8 res4_7 = AE_SAT8X4X32_H(out_mul4_5, out_mul6_7);
    ae_int8x8 res0_7 = AE_SEL8I(res0_3, res4_7, 1);
    res0_7 = AE_MIN8(res0_7, amax8);
    res0_7 = AE_MAX8(res0_7, amin8);
    AE_SAV8X8_XP(res0_7, align_out, (ae_int8x8 *)p_c, num_scalar_ops);
  }
  AE_SA64POS_FP(align_out, p_c);
}

#else
static void internal_elm_mul_broadcast_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  num_elm)

{
#if TFLITE_SINGLE_ROUNDING
  int l_shift = out_shift;
  int r_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)r_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int l_shift = out_shift >= 0 ?   out_shift : 0;
  int r_shift = out_shift <  0 ?  -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  int i;
  WORD8 * __restrict__ p_a = (WORD8 *)p_inp1;
  WORD8 * __restrict__ p_b = (WORD8 *)p_inp2;
  WORD8 *__restrict__ p_c =          p_out;

  ae_int16x4 a0_7, b;

  ae_int16x4  za = AE_MOVDA16(-inp1_zero_bias);
  ae_int16x4  zb = AE_MOVDA16(-inp2_zero_bias);

  ae_int16x4 a0_3, b0;

  ae_int32x2 raw_mul0_1, raw_mul2_3;
  ae_int32x2 out_mul0_1, out_mul2_3;

  const int num_simd4_ops = num_elm >> 2;
  const int num_scalar_ops = num_elm & 3;
  
  b = AE_MOVDA16(p_b[0]);
  b0 = AE_SUB16(b, zb);
  xtbool io_pointers_aligned = ((uintptr_t)p_inp1%4 == 0) && ((uintptr_t)p_inp2%4==0) && ((uintptr_t)p_out%4==0);

  if(io_pointers_aligned)
  {

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
      AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0);

      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
      out_mul0_1 = AE_ADD32S(out_mul0_1, AE_MOVDA32(out_zero_bias));
      out_mul2_3 = AE_ADD32S(out_mul2_3, AE_MOVDA32(out_zero_bias));
      CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
      CLAMP_VAL(out_mul2_3, out_mul2_3, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
#if XCHAL_HAVE_HIFI1
      ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(out_mul0_1), AE_MOVF16X4_FROMF32X2(out_mul2_3));
      AE_S8X4_IP(out,(ae_int32 *)p_c, 4);
#else
      STORE_8X4_FROM_32X4(p_c, out_mul0_1, out_mul2_3);
#endif
    }
  }
  else
  {
    ALIGN_REGISTER_TYPE va_a;
    PRIME_8X4F(p_a, va_a);
#if XCHAL_HAVE_HIFI1
    ALIGN_REGISTER_TYPE align_out = AE_ZALIGN64();
#endif
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
      AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
      MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
      out_mul0_1 = AE_ADD32S(out_mul0_1, AE_MOVDA32(out_zero_bias));
      out_mul2_3 = AE_ADD32S(out_mul2_3, AE_MOVDA32(out_zero_bias));
      CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
      CLAMP_VAL(out_mul2_3, out_mul2_3, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
#if XCHAL_HAVE_HIFI1
      ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(out_mul0_1), AE_MOVF16X4_FROMF32X2(out_mul2_3));
      AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_c);
#else
      STORE_8X4_FROM_32X4(p_c, out_mul0_1, out_mul2_3);
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
    AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    out_mul0_1 = AE_ADD32S(out_mul0_1, AE_MOVDA32(out_zero_bias));
    CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
    *p_c = (WORD8)AE_MOVAD32_L(out_mul0_1);
    p_c++;
  }
}
#endif /* XCHAL_HAVE_HIFI1S */

WORD32 xa_nn_elm_mul_broadcast_4D_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                      const WORD8 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                            WORD32  inp1_zero_bias,
                      const WORD8 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape,
                            WORD32  inp2_zero_bias)
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
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
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
    // inp1_strides[i] = inp1_strides[i + 1] * p_inp1_shape[i + 1];
    // inp2_strides[i] = inp2_strides[i + 1] * p_inp2_shape[i + 1];
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
    internal_elm_mul_broadcast_2D_asym8sxasym8s_asym8s(
                p_out,
                out_zero_bias,
                out_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1,
                inp1_zero_bias,
                p_inp2,
                inp2_zero_bias,
                1,
                p_out_shape[0] * inp1_strides[0]);
  }
  else if(inp1_strides[3] == inp2_strides[3])
  {
    WORD32 in_lc, out_lc;
    WORD32 inp1_zb;
    WORD32 inp2_zb;

    inp1_zb = inp1_zero_bias;
    inp2_zb = inp2_zero_bias;

    in_lc = p_out_shape[2] * p_out_shape[3];
    out_lc = 1;
    if(inp1_strides[2] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp1_zb = inp2_zero_bias;
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
        internal_elm_mul_broadcast_2D_asym8sxasym8s_asym8s(
            p_out_tmp,
            out_zero_bias,
            out_shift,
            out_multiplier,
            out_activation_min,
            out_activation_max,
            p_inp1_tmp0,
            inp1_zb,
            p_inp2_tmp0,
            inp2_zb,
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
    WORD32 inp1_zb;
    WORD32 inp2_zb;
    inp1_zb = inp1_zero_bias;
    inp2_zb = inp2_zero_bias;
    if(inp1_strides[3] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp1_zb = inp2_zero_bias;
      const WORD8 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
    }

    internal_elm_mul_broadcast_asym8sxasym8s_asym8s(
        p_out_tmp,
        out_zero_bias,
        out_shift,
        out_multiplier,
        out_activation_min,
        out_activation_max,
        p_inp1_tmp,
        inp1_zb,
        p_inp2_tmp,
        inp2_zb,
        p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3]);
  }
  else
  {
    WORD32 inp1_zb;
    WORD32 inp2_zb;
    inp1_zb = inp1_zero_bias;
    inp2_zb = inp2_zero_bias;
    if(inp1_strides[3] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp1_zb = inp2_zero_bias;
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
            internal_elm_mul_broadcast_asym8sxasym8s_asym8s(
                p_out_tmp,
                out_zero_bias,
                out_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1_tmp1,
                inp1_zb,
                p_inp2_tmp1,
                inp2_zb,
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

