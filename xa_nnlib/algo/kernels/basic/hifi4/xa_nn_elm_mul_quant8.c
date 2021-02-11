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
#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"

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
            MULTIPLYBYQUANTIZEDMULTIPLIER_X2(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
            MULTIPLYBYQUANTIZEDMULTIPLIER_X2(unclamped_out10, prod10, out_multiplier, left_shift, right_shift)
            unclamped_out32 = AE_ADD32(unclamped_out32, op_zero_bias);
            unclamped_out10 = AE_ADD32(unclamped_out10, op_zero_bias);

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
            MULTIPLYBYQUANTIZEDMULTIPLIER_X2(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
            MULTIPLYBYQUANTIZEDMULTIPLIER_X2(unclamped_out10, prod10, out_multiplier, left_shift, right_shift)
            unclamped_out32 = AE_ADD32(unclamped_out32, op_zero_bias);
            unclamped_out10 = AE_ADD32(unclamped_out10, op_zero_bias);

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
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
        unclamped_out32 = AE_ADD32(unclamped_out32, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)

        // Store Output
        i1 = AE_MOVAD32_H(clamped_out32);
        *out++ = (UWORD8) i1;
    }

    return 0;
}

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
    return -1;
}
