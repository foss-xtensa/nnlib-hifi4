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
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"

#define MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(out, inp1, multiplier, right_shift) \
{\
  inp1 = AE_MULFP32X2RAS(inp1, ((multiplier))); \
  out = AE_MULFP32X2RS(inp1, right_shift); \
}

WORD32 xa_nn_elm_add_asym16sxasym16s_asym16s(WORD16 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   WORD16 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const   WORD16 * __restrict__ p_inp2,
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
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD16), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -32768) || (out_zero_bias > 32767)), -1);
    XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -32767) || (inp1_zero_bias > 32768)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -32767) || (inp2_zero_bias > 32768)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_multiplier < 0) || (inp1_multiplier < 0) || (inp2_multiplier < 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < -32768) || (out_activation_min > 32767)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < -32768) || (out_activation_max > 32767)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    /* Special case for SEANET testcases. Optimized. */
    if((out_left_shift <= 0) && (inp1_left_shift <= 0) && (inp2_left_shift <=0 ) && (left_shift <= 15))
    {
        int i;
        const ae_int16x4 *	p_i1;
        const ae_int16x4 *  p_i2;
        ae_int16x4 *	out_temp;

        p_i1    = (const ae_int16x4 *)p_inp1;
        p_i2    = (const ae_int16x4 *)p_inp2;
        out_temp     = (ae_int16x4 *)p_out;

        int inp1_right_shift = (0XFFFFFFFF << (31 + inp1_left_shift));

        int inp2_right_shift = (0XFFFFFFFF << (31 + inp2_left_shift));

        int out_right_shift = (0XFFFFFFFF << (31 + out_left_shift));

        WORD32 const1 = 1 << left_shift;
        ae_int32x2 const1_32x2 =  AE_MOVDA32X2(const1,const1);
            
        WORD32 const2_inp1 = inp1_zero_bias << left_shift;
        WORD32 const2_inp2 = inp2_zero_bias << left_shift;

        ae_int32x2 const2_32x2_LO_1 =  AE_MOVDA32X2(const2_inp1,const2_inp1);
        ae_int32x2 const2_32x2_LO_2 =  AE_MOVDA32X2(const2_inp2,const2_inp2);

        ae_f16x4 x1, x2;
        ae_int32x2 temp;
        ae_int16x4 clamped_out;
        ae_f16x4 temp16X4, zero_bias1, zero_bias2;
        ae_f32x2 multiplier1, multiplier2, op_multiplier, op_zero_bias, activation_min, activation_max;

        // Taking zero_bias into 16X4 variable
        temp = AE_MOVDA32(inp1_zero_bias);
        temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
        zero_bias1 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

        temp = AE_MOVDA32(inp2_zero_bias);
        temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
        zero_bias2 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

        op_zero_bias = AE_MOVDA32(out_zero_bias);

        // Taking multiplier into 32x2 variable
        multiplier1 = AE_NEG32S(AE_MOVDA32(inp1_multiplier));
        multiplier2 = AE_NEG32S(AE_MOVDA32(inp2_multiplier));
        op_multiplier = AE_NEG32S(AE_MOVDA32(out_multiplier));
 
        activation_min = AE_MOVDA32(out_activation_min);
        activation_max = AE_MOVDA32(out_activation_max);

        ae_valign i1_a, i2_a, out_a;
        i1_a = AE_LA64_PP(p_i1);
        i2_a = AE_LA64_PP(p_i2);
        out_a = AE_ZALIGN64();
            	
        for(i=0;i < num_elm>>2;i++)
        {
            ae_f32x2 scaled_v1, scaled_v2;
            ae_f32x2 scaled_v3, scaled_v4;
            ae_f32x2 raw_sum12, raw_sum34;
            ae_f32x2 raw_out12, raw_out34;
            ae_f32x2 d_0,d_1,d_2,d_3;
            
            AE_LA16X4_IP(x1,i1_a,p_i1);
            AE_LA16X4_IP(x2,i2_a,p_i2);
            
            d_0 = d_1 = const2_32x2_LO_1;
            d_2 = d_3 = const2_32x2_LO_2;
            
            AE_MULAP32X16X2_H(d_0,const1_32x2,(x1));
            AE_MULAP32X16X2_L(d_1,const1_32x2,(x1));
            
            AE_MULAP32X16X2_H(d_2,const1_32x2,(x2));
            AE_MULAP32X16X2_L(d_3,const1_32x2,(x2));
            
            MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v1, d_0, multiplier1, inp1_right_shift)
            MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v2, d_1, multiplier1, inp1_right_shift)
            MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v3, d_2, multiplier2, inp2_right_shift)
            MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v4, d_3, multiplier2, inp2_right_shift)
            
            // Raw Sum
            raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);
            raw_sum34 = AE_ADD32S(scaled_v2, scaled_v4);
            
            // Raw Output
            MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(raw_out12, raw_sum12, op_multiplier, out_right_shift)
            MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(raw_out34, raw_sum34, op_multiplier, out_right_shift)
            
            raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);
            raw_out34 = AE_ADD32S(raw_out34, op_zero_bias);
            
            // clamped_out			
            AE_MINMAX32(raw_out12,activation_min,activation_max);
            AE_MINMAX32(raw_out34,activation_min,activation_max);
            
            clamped_out = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(raw_out12), AE_MOVINT16X4_FROMINT32X2(raw_out34));	
            
            // Store Output
            AE_SA16X4_IP(clamped_out, out_a, out_temp);
        }
	AE_SA64POS_FP(out_a, out_temp);

        // Remainder Loop
        for(i=0; i < (num_elm & 3); i++)
        {
            ae_f32x2 scaled_v1;
            ae_f32x2 scaled_v3;
            ae_f32x2 raw_sum12;
            ae_f32x2 raw_out12;
            ae_f32x2 d_0,d_1,d_2,d_3;
            
            AE_L16_IP(x1,(ae_int16*)p_i1,+2);
            AE_L16_IP(x2,(ae_int16*)p_i2,+2);
            
            d_0 = d_1 = const2_32x2_LO_1;
            d_2 = d_3 = const2_32x2_LO_2;
            
            AE_MULAP32X16X2_H(d_0,const1_32x2,(x1));
            
            AE_MULAP32X16X2_H(d_2,const1_32x2,(x2));
            
            MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v1, d_0, multiplier1, inp1_right_shift)
            MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(scaled_v3, d_2, multiplier2, inp2_right_shift)
            
            // Raw Sum
            raw_sum12 = AE_ADD32S(scaled_v1, scaled_v3);
            
            // Raw Output
            MULTIPLYBYQUANTIZEDMULTIPLIER_RIGHT(raw_out12, raw_sum12, op_multiplier, out_right_shift)
            
            raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);
            
            // clamped_out			
            AE_MINMAX32(raw_out12,activation_min,activation_max);
            
            clamped_out = AE_MOVINT16X4_FROMINT32X2(raw_out12);	
            
            // Store Output
            AE_S16_0_IP(clamped_out,(ae_int16*)out_temp,+2);
        }
    }
    /* Generic case */
    else
    {
        int i;
        ae_int16x4 *p_a = (ae_int16x4 *)p_inp1;
        ae_int16x4 *p_b = (ae_int16x4 *)p_inp2;
        ae_int16x4 *p_c = (ae_int16x4 *) p_out;

        const ae_int32x2 activation_min = AE_MOVDA32(out_activation_min);
        const ae_int32x2 activation_max = AE_MOVDA32(out_activation_max);

        const ae_int32x2  za = -inp1_zero_bias;
        const ae_int32x2  zb = -inp2_zero_bias;
        const ae_int32x2 zc = AE_MOVDA32( out_zero_bias);
        
        xtbool io_pointers_aligned =    ((uintptr_t)p_a%8 == 0) &&
                                        ((uintptr_t)p_b%8 == 0) &&
                                        ((uintptr_t)p_c%8 == 0);


        // intermediate results and scratch registers
        ae_int16x4 a0_3, b0_3, sat_1;

        ae_int32x2 shifted_a0_1, shifted_a2_3;
        ae_int32x2 shifted_b0_1, shifted_b2_3;

        ae_int32x2 scaled_a0_1, scaled_a2_3;
        ae_int32x2 scaled_b0_1, scaled_b2_3;

        ae_int32x2 raw_sum0_1, raw_sum2_3;
        ae_int32x2 out0_1, out2_3;

        ae_int16x4 ONE_16X4 = AE_MOVDA16(1);

        const int num_simd4_ops = num_elm/4;
        const int num_scalar_ops = num_elm%4;

        if(io_pointers_aligned){
            for(i=0; i<num_simd4_ops; i++){
                AE_L16X4_IP(a0_3, p_a, 8);
                AE_L16X4_IP(b0_3, p_b, 8);
                
                AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, ONE_16X4);
                AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, ONE_16X4);

                shifted_a0_1 = AE_SUB32(shifted_a0_1, za);
                shifted_a2_3 = AE_SUB32(shifted_a2_3, za);
                shifted_b0_1 = AE_SUB32(shifted_b0_1, zb);
                shifted_b2_3 = AE_SUB32(shifted_b2_3, zb);
                
                shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
                shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);
                
                shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
                shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);
                
                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a2_3, shifted_a2_3, inp1_multiplier, inp1_left_shift);

                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift);
                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b2_3, shifted_b2_3, inp2_multiplier, inp2_left_shift);

                // Raw Sum
                raw_sum0_1 = AE_ADD32S(scaled_a0_1, scaled_b0_1);
                raw_sum2_3 = AE_ADD32S(scaled_a2_3, scaled_b2_3);

                // Raw Output
                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out0_1, raw_sum0_1, out_multiplier, out_left_shift);
                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out2_3, raw_sum2_3, out_multiplier, out_left_shift);

                out0_1 = AE_ADD32S(out0_1, zc);
                out2_3 = AE_ADD32S(out2_3, zc);

                out0_1 = AE_MAX32(out0_1, activation_min);
                out0_1 = AE_MIN32(out0_1, activation_max);
                out2_3 = AE_MAX32(out2_3, activation_min);
                out2_3 = AE_MIN32(out2_3, activation_max);
               
                sat_1 = AE_SAT16X4(out0_1, out2_3);
                AE_S16X4_IP(sat_1,p_c,+8);
            }
        } else {
            ae_valign va_a, va_b, va_dst;
            va_a = AE_LA64_PP(p_a);
            va_b = AE_LA64_PP(p_b);
            va_dst =  AE_ZALIGN64();

            for(i=0; i<num_simd4_ops; i++){
                AE_LA16X4_IP(a0_3, va_a, p_a);
                AE_LA16X4_IP(b0_3, va_b, p_b);
                
                AE_MUL16X4(shifted_a0_1, shifted_a2_3, a0_3, ONE_16X4);
                AE_MUL16X4(shifted_b0_1, shifted_b2_3, b0_3, ONE_16X4);

                shifted_a0_1 = AE_SUB32(shifted_a0_1, za);
                shifted_a2_3 = AE_SUB32(shifted_a2_3, za);
                shifted_b0_1 = AE_SUB32(shifted_b0_1, zb);
                shifted_b2_3 = AE_SUB32(shifted_b2_3, zb);

                shifted_a0_1 = AE_SLAA32S(shifted_a0_1, left_shift);
                shifted_a2_3 = AE_SLAA32S(shifted_a2_3, left_shift);
                shifted_b0_1 = AE_SLAA32S(shifted_b0_1, left_shift);
                shifted_b2_3 = AE_SLAA32S(shifted_b2_3, left_shift);
                
                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a0_1, shifted_a0_1, inp1_multiplier, inp1_left_shift);
                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_a2_3, shifted_a2_3, inp1_multiplier, inp1_left_shift);
                
                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0_1, shifted_b0_1, inp2_multiplier, inp2_left_shift);
                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b2_3, shifted_b2_3, inp2_multiplier, inp2_left_shift);
                
                // Raw sum
                raw_sum0_1 = AE_ADD32S(scaled_a0_1, scaled_b0_1);
                raw_sum2_3 = AE_ADD32S(scaled_a2_3, scaled_b2_3);
                
                // Raw Output
                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out0_1, raw_sum0_1, out_multiplier, out_left_shift);
                MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(out2_3, raw_sum2_3, out_multiplier, out_left_shift);
                
                out0_1 = AE_ADD32S(out0_1, zc);
                out2_3 = AE_ADD32S(out2_3, zc);

                /* Clamped out */
                out0_1 = AE_MAX32(out0_1, activation_min);
                out0_1 = AE_MIN32(out0_1, activation_max);
                out2_3 = AE_MAX32(out2_3, activation_min);
                out2_3 = AE_MIN32(out2_3, activation_max);
                
                /* Store output */
                sat_1 = AE_SAT16X4(out0_1, out2_3);
                AE_SA16X4_IP(sat_1, va_dst, p_c);
            }
            AE_SA64POS_FP(va_dst, p_c);
        }

        for(i=0; i<num_scalar_ops; i++) {
            ae_int32 a, b;
            ae_int32x2 res;

            a = (ae_int32)(((WORD16 *)p_a)[i] + inp1_zero_bias);            // add input biases
            b = (ae_int32)(((WORD16 *)p_b)[i] + inp2_zero_bias);

            a = AE_SLAA32S(a, left_shift);
            b = AE_SLAA32S(b, left_shift);

            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(a, a, inp1_multiplier, inp1_left_shift);
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(b, b, inp2_multiplier, inp2_left_shift);

            res = AE_ADD32S(a, b);                              // add inputs to one 32-bit res

            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(res, res, out_multiplier, out_left_shift);

            res = AE_ADD32S(res, out_zero_bias);                // add out zero bias

            res = AE_MAX32(res, activation_min);
            res = AE_MIN32(res, activation_max);

            ((WORD16 *)p_c)[i] = (WORD16)AE_MOVAD32_L(res);
        }
    }
    return 0;
}
