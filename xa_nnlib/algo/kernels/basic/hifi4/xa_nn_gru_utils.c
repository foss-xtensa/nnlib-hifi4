/*******************************************************************************
* Copyright (c) 2018-2025 Cadence Design Systems, Inc.
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
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_basic_state.h"
#include "xa_nnlib_quant_macros.h"

#if !XCHAL_HAVE_HIFI1
#if TFLITE_SINGLE_ROUNDING
#define MPY_BY_QUANT_MULT_OUT32(out, inp, multiplier, l_shift, r_shift) \
{ \
  ae_int64 out64_0; \
  out64_0 = AE_MUL32_HH(inp, AE_MOVDA32(multiplier)); \
  out64_0 = AE_SLAA64S(out64_0, 1 + l_shift); \
  out = AE_ROUND32X2F64SASYM(out64_0, out64_0); \
}
#else
#define MPY_BY_QUANT_MULT_OUT32(out, inp, multiplier, l_shift, r_shift) \
{ \
  out = AE_SLAA32S(inp, l_shift); \
  out = AE_MULFP32X2RAS(out, AE_MOVDA32(multiplier)); \
  out = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(out), r_shift), AE_SRAA64(AE_CVT64F32_H(out), r_shift)); \
}
#endif
#endif

WORD32 xa_nn_gru_hidden_state_update_8(WORD8* p_hidden_state,
                                      const WORD16* p_update_gate,
                                     const WORD16* p_modulated_state,
                                      WORD32 update_to_modulated_state_multiplier,
                                      WORD32 update_to_modulated_state_shift,
                                      WORD32 update_to_hidden_state_multiplier,
                                      WORD32 update_to_hidden_state_shift,
                                      WORD32 out_multiplier,
                                      WORD32 out_shift,
                                      WORD32 hidden_zero_bias,
                                      WORD32 num_elms)
{
    /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_hidden_state, -1);
  XA_NNLIB_ARG_CHK_PTR(p_update_gate, -1);
  XA_NNLIB_ARG_CHK_PTR(p_modulated_state, -1);
  /* Pointer alignment checks */ 
  XA_NNLIB_ARG_CHK_ALIGN(p_hidden_state, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_update_gate, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_modulated_state, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((update_to_modulated_state_shift < -31 || update_to_modulated_state_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((update_to_hidden_state_shift < -31 || update_to_hidden_state_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((num_elms <= 0),-1);
  XA_NNLIB_ARG_CHK_COND((hidden_zero_bias < -128 || hidden_zero_bias > 127), -1);

  int left_shift_utm, right_shift_utm;
  int left_shift_uth, right_shift_uth;
  int left_shift, right_shift;

#if TFLITE_SINGLE_ROUNDING
  left_shift_utm = update_to_modulated_state_shift;
  right_shift_utm = update_to_modulated_state_shift;

  left_shift_uth = update_to_hidden_state_shift;
  right_shift_uth = update_to_hidden_state_shift;

  left_shift = out_shift;
  right_shift = out_shift;

#if XCHAL_HAVE_HIFI1S
  left_shift_utm = 31 - left_shift_utm;
  left_shift_utm = (left_shift_utm << 16) | left_shift_utm;

  left_shift_uth = 31 - left_shift_uth;
  left_shift_uth = (left_shift_uth << 16) | left_shift_uth;

  left_shift = 31 - left_shift;
  left_shift = (left_shift << 16) | left_shift;
#endif 

  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift_utm;
  (void)right_shift_uth;
  (void)right_shift;

#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift_utm = update_to_modulated_state_shift > 0 ? update_to_modulated_state_shift : 0;
  right_shift_utm = update_to_modulated_state_shift < 0 ? -update_to_modulated_state_shift : 0;

  left_shift_uth = update_to_hidden_state_shift > 0 ? update_to_hidden_state_shift : 0;
  right_shift_uth = update_to_hidden_state_shift < 0 ? -update_to_hidden_state_shift : 0;

  left_shift = out_shift > 0 ? out_shift : 0;
  right_shift = out_shift < 0 ? -out_shift : 0;
#endif

  const WORD8 * ptr_hidden_in = (const WORD8 *)p_hidden_state;
  const ae_int16 * ptr_update = (const ae_int16 *)p_update_gate;
  const ae_int16 * ptr_modulated = (const ae_int16 *)p_modulated_state;
  WORD8 * ptr_hidden_out = (WORD8 *)p_hidden_state;

  ae_int16x4 d_update1, d_modulated1;
  ae_int32x2 int16_min = AE_MOVDA32(-32768);
  ae_int32x2 int16_max = AE_MOVDA32(32767);
  ae_int16x4 d_int16_max = AE_MOVDA16(32767);
  ae_int16x4 d_one16 = AE_MOVDA16(1);
  ae_int16x4 d_one_minus_update1;
  ae_int16x4 d_hidden_in1;
  ae_int32x2 d_out_zero_bias = AE_MOVDA32(hidden_zero_bias);
  WORD32 itr;


  ae_int32x2 d_update_times_hidden1 = 0, d_update_times_hidden2 = 0;
  ae_int32x2 d_update_times_modulated1 = 0, d_update_times_modulated2 = 0;
  ae_int32x2 q_update_times_modulated1, q_update_times_modulated2;
  ae_int32x2 q_update_times_hidden1, q_update_times_hidden2;
  ae_int32x2 d_final_sum1, d_final_sum2;
  ae_int32x2 q_final_sum1, q_final_sum2;
  ae_int16x4 d_hidden_zero_bias = AE_MOVDA16(hidden_zero_bias);
#if !XCHAL_HAVE_HIFI1 
  ae_int32x2 d_temp32x2;
  WORD32 pre_loop_count = 4 - ((uintptr_t)ptr_hidden_in & 0x3);
  pre_loop_count = (pre_loop_count==4)? 0: pre_loop_count;
  pre_loop_count = (pre_loop_count > num_elms) ? num_elms : pre_loop_count;
  num_elms -= pre_loop_count;
  
  for(itr=0; itr< pre_loop_count; itr++)
  {
    d_hidden_in1 = ((WORD16)(*ptr_hidden_in));
    ptr_hidden_in++;
    d_hidden_in1 = AE_SUB16(d_hidden_in1, d_hidden_zero_bias);
    AE_L16_IP(d_update1, (const ae_int16*)ptr_update, 2);
    AE_L16_IP(d_modulated1, (const ae_int16*)ptr_modulated, 2);

    //update_gate * hidden_state
    AE_MUL16X4(d_update_times_hidden1, d_temp32x2, d_update1, d_hidden_in1); // MUL16X4S
    MPY_BY_QUANT_MULT_OUT32(q_update_times_hidden1, d_update_times_hidden1, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
    AE_MINMAX32(q_update_times_hidden1, int16_min, int16_max);

    //(1.0 - update_gate) * modulated_state
    d_one_minus_update1 = AE_SUB16S(d_int16_max, d_update1);
    d_update1           = AE_ADD16S(d_one_minus_update1, d_one16);

    AE_MUL16X4(d_update_times_modulated1, d_temp32x2, d_update1, d_modulated1); // MUL16X4S
    MPY_BY_QUANT_MULT_OUT32(q_update_times_modulated1, d_update_times_modulated1, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
    AE_MINMAX32(q_update_times_modulated1, int16_min, int16_max);

    //(update_gate * hidden_state) + ((1.0 - update_gate) * modulated_state)
    d_final_sum1 = AE_ADD32S(q_update_times_modulated1, q_update_times_hidden1);
    MPY_BY_QUANT_MULT_OUT32(q_final_sum1, d_final_sum1, out_multiplier, left_shift, right_shift);
    q_final_sum1 = AE_ADD32S(q_final_sum1, d_out_zero_bias);
    
    //Saturate to 8-bit
    AE_MINMAX32(q_final_sum1, AE_MOVDA32(-128), AE_MOVDA32(127));

    *ptr_hidden_out = (WORD8)AE_MOVAD32_H(q_final_sum1);
    ptr_hidden_out++;
  }
#else
  ae_valign align_inp = AE_LA64_PP(ptr_hidden_in);
  ae_valign align_out = AE_ZALIGN64();
#endif

  ae_valign a_up = AE_LA64_PP(ptr_update);
  ae_valign a_mod = AE_LA64_PP(ptr_modulated);

#pragma concurrent
  for(itr = 0; itr < (num_elms >> 2) ; itr++)
  {
#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(d_hidden_in1, align_inp, ptr_hidden_in);
    d_hidden_in1 = AE_SUB16(d_hidden_in1, d_hidden_zero_bias);
#else
    AE_L8X4F_IP(d_hidden_in1, ptr_hidden_in, 4);
    d_hidden_in1 = AE_SRAI16(d_hidden_in1, 8);    
    d_hidden_in1 = AE_SUB16(d_hidden_in1, d_hidden_zero_bias);
#endif
    AE_LA16X4_IP(d_update1, a_up, ptr_update);
    AE_LA16X4_IP(d_modulated1, a_mod, ptr_modulated);

    //update_gate * hidden_state
    AE_MUL16X4(d_update_times_hidden1, d_update_times_hidden2, d_update1, d_hidden_in1);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(q_update_times_hidden1, q_update_times_hidden2, d_update_times_hidden1, d_update_times_hidden2, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
#else       
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_update_times_hidden1, d_update_times_hidden1, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_update_times_hidden2, d_update_times_hidden2, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
#endif  

    AE_MINMAX32(q_update_times_hidden1, int16_min, int16_max);
    AE_MINMAX32(q_update_times_hidden2, int16_min, int16_max);

    //(1.0 - update_gate) * modulated_state
    d_one_minus_update1 = AE_SUB16S(d_int16_max, d_update1);
    d_update1           = AE_ADD16S(d_one_minus_update1, d_one16);

    AE_MUL16X4(d_update_times_modulated1, d_update_times_modulated2, d_update1, d_modulated1);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(q_update_times_modulated1, q_update_times_modulated2, d_update_times_modulated1, d_update_times_modulated2, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
#else        
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_update_times_modulated1, d_update_times_modulated1, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_update_times_modulated2, d_update_times_modulated2, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
#endif
    AE_MINMAX32(q_update_times_modulated1, int16_min, int16_max);
    AE_MINMAX32(q_update_times_modulated2, int16_min, int16_max);

    //(update_gate * hidden_state) + ((1.0 - update_gate) * modulated_state)
    d_final_sum1 = AE_ADD32S(q_update_times_modulated1, q_update_times_hidden1);
    d_final_sum2 = AE_ADD32S(q_update_times_modulated2, q_update_times_hidden2);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(q_final_sum1, q_final_sum2, d_final_sum1, d_final_sum2, out_multiplier, left_shift, right_shift);
#else        
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_final_sum1, d_final_sum1, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_final_sum2, d_final_sum2, out_multiplier, left_shift, right_shift);
#endif
    q_final_sum1 = AE_ADD32S(q_final_sum1, d_out_zero_bias);
    q_final_sum2 = AE_ADD32S(q_final_sum2, d_out_zero_bias);

#if XCHAL_HAVE_HIFI1 
    // clamped_out
    ae_int8x8 clamped_01 = AE_SAT8X4X32_H(q_final_sum1, q_final_sum2);
    // Store Output
    AE_SAV8X8_XP(clamped_01, align_out, (ae_int8x8 *)ptr_hidden_out, 4);
#else
    //Saturate to 8-bit
    AE_MINMAX32(q_final_sum1, AE_MOVDA32(-128), AE_MOVDA32(127));
    AE_MINMAX32(q_final_sum2, AE_MOVDA32(-128), AE_MOVDA32(127));
    STORE_8X4_FROM_32X4(ptr_hidden_out, q_final_sum1, q_final_sum2);
#endif
  }

  WORD32 rem_elm = num_elms & 0x3;
#if XCHAL_HAVE_HIFI1
  if(rem_elm)
  {
    AE_LAV8X4S_XP(d_hidden_in1, align_inp, (ae_int8x4 *)ptr_hidden_in, rem_elm);
    d_hidden_in1 = AE_SUB16(d_hidden_in1, d_hidden_zero_bias);
    AE_LA16X4_IP(d_update1, a_up, ptr_update);
    AE_LA16X4_IP(d_modulated1, a_mod, ptr_modulated);

    //update_gate * hidden_state
    AE_MUL16X4(d_update_times_hidden1, d_update_times_hidden2, d_update1, d_hidden_in1);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(q_update_times_hidden1, q_update_times_hidden2, d_update_times_hidden1, d_update_times_hidden2, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
#else       
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_update_times_hidden1, d_update_times_hidden1, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_update_times_hidden2, d_update_times_hidden2, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
#endif  

    AE_MINMAX32(q_update_times_hidden1, int16_min, int16_max);
    AE_MINMAX32(q_update_times_hidden2, int16_min, int16_max);
    //(1.0 - update_gate) * modulated_state
    d_one_minus_update1 = AE_SUB16S(d_int16_max, d_update1);
    d_update1           = AE_ADD16S(d_one_minus_update1, d_one16);

    AE_MUL16X4(d_update_times_modulated1, d_update_times_modulated2, d_update1, d_modulated1);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(q_update_times_modulated1, q_update_times_modulated2, d_update_times_modulated1, d_update_times_modulated2, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
#else        
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_update_times_modulated1, d_update_times_modulated1, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_update_times_modulated2, d_update_times_modulated2, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
#endif  
    
    AE_MINMAX32(q_update_times_modulated1, int16_min, int16_max);
    AE_MINMAX32(q_update_times_modulated2, int16_min, int16_max);

    //(update_gate * hidden_state) + ((1.0 - update_gate) * modulated_state)
    d_final_sum1 = AE_ADD32S(q_update_times_modulated1, q_update_times_hidden1);
    d_final_sum2 = AE_ADD32S(q_update_times_modulated2, q_update_times_hidden2);
    
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(q_final_sum1, q_final_sum2, d_final_sum1, d_final_sum2, out_multiplier, left_shift, right_shift);
#else        
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_final_sum1, d_final_sum1, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(q_final_sum2, d_final_sum2, out_multiplier, left_shift, right_shift);
#endif  

    q_final_sum1 = AE_ADD32S(q_final_sum1, d_out_zero_bias);
    q_final_sum2 = AE_ADD32S(q_final_sum2, d_out_zero_bias);
    ae_int8x8 clamped_01 = AE_SAT8X4X32_H(q_final_sum1, q_final_sum2);

    AE_SAV8X8_XP(clamped_01, align_out, (ae_int8x8 *)ptr_hidden_out, rem_elm);
  }
  AE_SA64POS_FP(align_out, ptr_hidden_out);
#else
  for(itr=0; itr< rem_elm; itr++)
  {
    d_hidden_in1 = ((WORD16)(*ptr_hidden_in));
    ptr_hidden_in++;
    d_hidden_in1 = AE_SUB16(d_hidden_in1, d_hidden_zero_bias);
    AE_L16_IP(d_update1, (const ae_int16*)ptr_update, 2);
    AE_L16_IP(d_modulated1, (const ae_int16*)ptr_modulated, 2);
    //update_gate * hidden_state
    AE_MUL16X4(d_update_times_hidden1, d_temp32x2, d_update1, d_hidden_in1); // MUL16X4S
    MPY_BY_QUANT_MULT_OUT32(q_update_times_hidden1, d_update_times_hidden1, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
    AE_MINMAX32(q_update_times_hidden1, int16_min, int16_max);

    //(1.0 - update_gate) * modulated_state
    d_one_minus_update1 = AE_SUB16S(d_int16_max, d_update1);
    d_update1           = AE_ADD16S(d_one_minus_update1, d_one16);

    AE_MUL16X4(d_update_times_modulated1, d_temp32x2, d_update1, d_modulated1); // MUL16X4S
    MPY_BY_QUANT_MULT_OUT32(q_update_times_modulated1, d_update_times_modulated1, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
    AE_MINMAX32(q_update_times_modulated1, int16_min, int16_max);

    //(update_gate * hidden_state) + ((1.0 - update_gate) * modulated_state)
    d_final_sum1 = AE_ADD32S(q_update_times_modulated1, q_update_times_hidden1);
    MPY_BY_QUANT_MULT_OUT32(q_final_sum1, d_final_sum1, out_multiplier, left_shift, right_shift);
    q_final_sum1 = AE_ADD32S(q_final_sum1, d_out_zero_bias);
    
    //Saturate to 8-bit
    AE_MINMAX32(q_final_sum1, AE_MOVDA32(-128), AE_MOVDA32(127));

    *ptr_hidden_out = (WORD8)AE_MOVAD32_H(q_final_sum1);
    ptr_hidden_out++;
  }
#endif
  return 0;
}