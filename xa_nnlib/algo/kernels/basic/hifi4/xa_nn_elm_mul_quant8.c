/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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
    int left_shift = out_shift < 0 ? 0 : out_shift;
    int right_shift = out_shift > 0 ? 0 : -out_shift;

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
        unclamped_out32 = AE_ADD32(unclamped_out32, op_zero_bias);
        unclamped_out10 = AE_ADD32(unclamped_out10, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)
        CLAMP_VAL(clamped_out10, unclamped_out10, activation_min, activation_max)

        // Store Output
	ae_int16x4 temp = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(clamped_out32),AE_MOVINT16X4_FROMINT32X2(clamped_out10));
        AE_SA8X4U_IP(temp, align_out, (ae_int32 *)out);
    }
	
    AE_SA64POS_FP(align_out, out);

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
        unclamped_out32 = AE_ADD32(unclamped_out32, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)

        // Store Output
        i1 = AE_MOVAD32_H(clamped_out32);
        *out++ = (UWORD8) i1;
    }

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
        i1 = AE_MOVAD32_H(clamped_out32);
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
	
    ae_int16x4 temp1, temp2;                        //intermediate results

    ae_int16x4 za = AE_MOVDA16(inp1_zero_bias);     // replicate 16LSBs of input into 16x4 output
    ae_int16x4 zb = AE_MOVDA16(inp2_zero_bias);     // zero_bias is already signed, no need for ZE
    ae_int32x2 zc = AE_MOVDA32( out_zero_bias);

    ae_f32x2 multiplier = AE_MOVDA32(out_multiplier);

    int l_shift = out_shift >= 0 ?   out_shift : 0;
    int r_shift = out_shift <  0 ?  -out_shift : 0;

    ae_int32x2 activation_max = AE_MOVDA32(out_activation_max);
    ae_int32x2 activation_min = AE_MOVDA32(out_activation_min);

    WORD8 *in1 = (WORD8 *)p_inp1;
    WORD8 *in2 = (WORD8 *)p_inp2;

    unsigned int num_simd8_ops = num_elm/8;
    unsigned int num_scalar_ops = num_elm%8;
	
    ae_valign va_in1, va_in2, va_out;
    va_in1 = AE_LA64_PP(in1);
    va_in2 = AE_LA64_PP(in2);
    va_out = AE_ZALIGN64();
	
    for(i=0; i<num_simd8_ops; i++)
    {

        AE_LA8X4S_IP(a0_3, va_in1, in1);    AE_LA8X4S_IP(a4_7, va_in1, in1);
        AE_LA8X4S_IP(b0_3, va_in2, in2);    AE_LA8X4S_IP(b4_7, va_in2, in2);

        a0_3 = AE_ADD16(a0_3, za);  a4_7 = AE_ADD16(a4_7, za);      // Add zero points
        b0_3 = AE_ADD16(b0_3, zb);  b4_7 = AE_ADD16(b4_7, zb);

        AE_MUL16X4(res0_1, res2_3, a0_3, b0_3);                     // a & b are 9-bit vals in 16-bit containers.
        AE_MUL16X4(res4_5, res6_7, a4_7, b4_7);                     // res, therefore is 18-bit val in 32-bit container.

        res0_1 = AE_SLAA32S(res0_1, l_shift);
        res2_3 = AE_SLAA32S(res2_3, l_shift);
        res0_1 = AE_MULFP32X2RAS(res0_1, multiplier);
        res2_3 = AE_MULFP32X2RAS(res2_3, multiplier);
        res0_1 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(res0_1), r_shift), AE_SRAA64(AE_CVT64F32_L(res0_1), r_shift));
        res2_3 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(res2_3), r_shift), AE_SRAA64(AE_CVT64F32_L(res2_3), r_shift));

        res4_5 = AE_SLAA32S(res4_5, l_shift);
        res6_7 = AE_SLAA32S(res6_7, l_shift);
        res4_5 = AE_MULFP32X2RAS(res4_5, multiplier);
        res6_7 = AE_MULFP32X2RAS(res6_7, multiplier);
        res4_5 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(res4_5), r_shift), AE_SRAA64(AE_CVT64F32_L(res4_5), r_shift));
        res6_7 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(res6_7), r_shift), AE_SRAA64(AE_CVT64F32_L(res6_7), r_shift));

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
    AE_SA64POS_FP(va_out, p_out);

    for(i=0; i<num_scalar_ops; i++)
    {
        ae_int32 tmp = (in1[i] + inp1_zero_bias) *
                            ( in2[i] + inp2_zero_bias );

        ae_int32x2 res = tmp;

        res = AE_SLAA32S(res, l_shift);
        res = AE_MULFP32X2RAS(res, multiplier);
        res = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(res), r_shift), AE_SRAA64(AE_CVT64F32_L(res), r_shift));

        res = AE_ADD32S(res, zc);

        res = AE_MAX32(res, activation_min);
        res = AE_MIN32(res, activation_max);

        *p_out = (WORD8)AE_MOVAD32_L(res);
         p_out++;
    }

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

