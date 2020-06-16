/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
    XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 31)), -1);
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
    ae_f32x2 multiplier1, multiplier2, op_multiplier, op_zero_bias, activation_min, activation_max;
    ae_int32x2 ZERO = AE_ZERO32();

    // Taking zero_bias into 16X4 variable
    temp = AE_MOVDA32(inp1_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias1 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

    temp = AE_MOVDA32(inp2_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias2 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

    op_zero_bias = AE_MOVDA32(out_zero_bias);

    // Taking multiplier into 32x2 variable
    multiplier1 = AE_MOVDA32(inp1_multiplier);
    multiplier2 = AE_MOVDA32(inp2_multiplier);
    op_multiplier = AE_MOVDA32(out_multiplier);

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


            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v1, shifted_v1, multiplier1, inp1_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v2, shifted_v2, multiplier1, inp1_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v3, shifted_v3, multiplier2, inp2_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v4, shifted_v4, multiplier2, inp2_left_shift)

            // Raw Sum
            raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);
            raw_sum34 = AE_ADD32S(scaled_v2, scaled_v4);

            // Raw Output
            MultiplyByQuantizedMultiplierSmallerThanOneExp(raw_out12, raw_sum12, op_multiplier, out_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(raw_out34, raw_sum34, op_multiplier, out_left_shift)
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


            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v1, shifted_v1, multiplier1, inp1_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v2, shifted_v2, multiplier1, inp1_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v3, shifted_v3, multiplier2, inp2_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v4, shifted_v4, multiplier2, inp2_left_shift)

            // Raw Sum
            raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);
            raw_sum34 = AE_ADD32S(scaled_v2, scaled_v4);

            // Raw Output
            MultiplyByQuantizedMultiplierSmallerThanOneExp(raw_out12, raw_sum12, op_multiplier, out_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(raw_out34, raw_sum34, op_multiplier, out_left_shift)
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

        MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v1, shifted_v1, multiplier1, inp1_left_shift)
        MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v3, shifted_v3, multiplier2, inp2_left_shift)

        // Raw Sum
        raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);

        // Raw Output
        MultiplyByQuantizedMultiplierSmallerThanOneExp(raw_out12, raw_sum12, op_multiplier, out_left_shift)
        raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out12, raw_out12, activation_min, activation_max)

        // Store Output
        i1 = AE_MOVAD32_H(clamped_out12);
        *out++ = (UWORD8) i1;
    }

    return 0;
}


