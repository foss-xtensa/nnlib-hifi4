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

#define AE_SAT32X2_HIFI4(out32, inp64) \
    out32 = AE_TRUNCA32X2F64S(ZERO64, inp64, 32);

#define AE_MINMAX32_HIFI4(inout32, min32, max32) \
    inout32 = AE_MAX32(inout32, min32); \
    inout32 = AE_MIN32(inout32, max32);

/*----------------------------Main function---------------------------------*/
WORD32 xa_nn_dot_prod_16x16_asym8s(
         WORD8 * __restrict__ p_out,           /* pointer to output */
         const WORD16 * __restrict__ p_inp1_start,    /* pointer to input1 */
         const WORD16 * __restrict__ p_inp2_start,    /* pointer to input2 */
         const WORD32 * bias_ptr,
         WORD32 vec_length,
         WORD32 out_multiplier,
         WORD32 out_shift,
         WORD32 out_zero_bias,
         WORD32 vec_count)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_start, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_start, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_start, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_start, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec_count <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  int left_shift, right_shift;
  int loopcnt;
  const WORD32 bias_buffer[2] = {0, 0};
  const WORD32* p_bias_load;
  WORD32 bias_address_increment = sizeof(WORD32);

  if(bias_ptr == NULL)
  {
    p_bias_load = bias_buffer;
    bias_address_increment = 0;
  }
  else
  {
    p_bias_load = bias_ptr;
  }

#if TFLITE_SINGLE_ROUNDING
    left_shift = out_shift;
#if XCHAL_HAVE_HIFI1S
    left_shift = 31 - left_shift ;
    left_shift = left_shift << 16 | left_shift;
#endif    
    /* Single rounding requires only original shift value */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    left_shift = out_shift < 0 ? 0 : out_shift;
    right_shift = out_shift > 0 ? 0 : -out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */
  
  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);

  const ae_int16x4 *pt_inp1, *pt_inp2;
  ae_valign align_inp1, align_inp2;
  ae_int16x4 d_inp1_0;
  ae_int16x4 d_inp2_0;
  ae_int64 d_out64_0;
  ae_int32x2 d_out32;
  ae_int32x2 d_bias;
  int i;

 /* handle cases where vec_length is multiple of 8 */
  if(vec_length == 8)
  {
    /* Assumption: 
     * p_inp1_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * p_inp2_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * */
    pt_inp1 = (const ae_int16x4 *)((WORD16 *)p_inp1_start);
    pt_inp2 = (const ae_int16x4 *)((WORD16 *)p_inp2_start);

    align_inp1 = AE_LA64_PP(pt_inp1);
    align_inp2 = AE_LA64_PP(pt_inp2);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    ae_valign align_store = AE_ZALIGN64();
#endif

    /* TBD: multiple vec_count processing in a single loop can be done */
    for(loopcnt = 0; loopcnt < vec_count; loopcnt++)
    {
      AE_L32_XP(d_bias, (ae_int32 *)p_bias_load, bias_address_increment);

      d_out64_0 = ZERO64;
      AE_LA16X4_IP(d_inp1_0, align_inp1, pt_inp1);
      AE_LA16X4_IP(d_inp2_0, align_inp2, pt_inp2);
      AE_MULAAAAQ16(d_out64_0, d_inp1_0, d_inp2_0);
      AE_LA16X4_IP(d_inp1_0, align_inp1, pt_inp1);
      AE_LA16X4_IP(d_inp2_0, align_inp2, pt_inp2);
      AE_MULAAAAQ16(d_out64_0, d_inp1_0, d_inp2_0);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
      d_out32 = AE_TRUNCA32X2F64S(d_out64_0, d_out64_0, 32);
      d_out32 = AE_ADD32S(d_out32, d_bias);
      ae_int64 out64_tmp = AE_MUL32_LL(d_out32, AE_MOVDA32(out_multiplier));
      d_out32 = AE_ROUNDAV32X2F64SASYM(out64_tmp, out64_tmp, left_shift);    
      d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
      ae_int8x8 d_out_8b = AE_SAT8X4X32_H(d_out32,d_out32);
      AE_SAV8X8_XP(d_out_8b, align_store, (ae_int8x8 *)p_out, 1);
#else
      AE_SAT32X2_HIFI4(d_out32, d_out64_0);
      d_out32 = AE_ADD32S(d_out32, d_bias);
      MPY_BY_QUANT_MULT_X2_OUT32(d_out32, d_out32, out_multiplier, left_shift, right_shift)
      d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
      AE_MINMAX32_HIFI4(d_out32, min_int8, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(d_out32);
#endif      
    }
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    AE_SA64POS_FP(align_store, p_out);	
#endif    
  }
  else if(vec_length == 32)
  {
    /* Assumption: 
     * p_inp1_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * p_inp2_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * */
    pt_inp1 = (const ae_int16x4 *)((WORD16 *)p_inp1_start);
    pt_inp2 = (const ae_int16x4 *)((WORD16 *)p_inp2_start);

    align_inp1 = AE_LA64_PP(pt_inp1);
    align_inp2 = AE_LA64_PP(pt_inp2);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    ae_valign align_store = AE_ZALIGN64();
#endif
    /* TBD: multiple vec_count processing in a single loop can be done */
    for(loopcnt = 0; loopcnt < vec_count; loopcnt++)
    {
      AE_L32_XP(d_bias, (ae_int32 *)p_bias_load, bias_address_increment);

      d_out64_0 = ZERO64;
#if !(XCHAL_HAVE_HIFI1S)
#pragma loop_count min=3
#endif
      for(i = 0; i < (vec_length >> 2); i++)
      {
        AE_LA16X4_IP(d_inp1_0, align_inp1, pt_inp1);
        AE_LA16X4_IP(d_inp2_0, align_inp2, pt_inp2);
        AE_MULAAAAQ16(d_out64_0, d_inp1_0, d_inp2_0);
      }
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    d_out32 = AE_TRUNCA32X2F64S(d_out64_0, d_out64_0, 32);
    d_out32 = AE_ADD32S(d_out32, d_bias);
    ae_int64 out64_tmp = AE_MUL32_LL(d_out32, AE_MOVDA32(out_multiplier));
    d_out32 = AE_ROUNDAV32X2F64SASYM(out64_tmp, out64_tmp, left_shift);   
    d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
    ae_int8x8 d_out_8b = AE_SAT8X4X32_H(d_out32,d_out32);
    AE_SAV8X8_XP(d_out_8b, align_store, (ae_int8x8 *)p_out, 1);
#else
    AE_SAT32X2_HIFI4(d_out32, d_out64_0);
    d_out32 = AE_ADD32S(d_out32, d_bias);
    MPY_BY_QUANT_MULT_X2_OUT32(d_out32, d_out32, out_multiplier, left_shift, right_shift)    
    d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
    AE_MINMAX32_HIFI4(d_out32, min_int8, max_int8);
    *p_out++ = (WORD8)AE_MOVAD32_L(d_out32);
#endif
    }
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
  AE_SA64POS_FP(align_store, p_out);
#endif  

  }
  /* inp1 and inp2 8-byte aligned case */
  else if(((vec_length & 3) == 0) && (((int)p_inp1_start & 7) == 0) && (((int)p_inp2_start & 7) == 0))
  {
    /* Assumption: 
     * p_inp1_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * p_inp2_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * */
    pt_inp1 = (const ae_int16x4 *)((WORD16 *)p_inp1_start);
    pt_inp2 = (const ae_int16x4 *)((WORD16 *)p_inp2_start);

    /* TBD: multiple vec_count processing in a single loop can be done */
    for(loopcnt = 0; loopcnt < vec_count; loopcnt++)
    {
      AE_L32_XP(d_bias, (ae_int32 *)p_bias_load, bias_address_increment);

      d_out64_0 = ZERO64;

      for(i = 0; i < (vec_length >> 2); i++)
      {
        AE_L16X4_IP(d_inp1_0, pt_inp1, 8);
        AE_L16X4_IP(d_inp2_0, pt_inp2, 8);
        AE_MULAAAAQ16(d_out64_0, d_inp1_0, d_inp2_0);
      }
      AE_SAT32X2_HIFI4(d_out32, d_out64_0);
      d_out32 = AE_ADD32S(d_out32, d_bias);
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_out32, d_out32, out_multiplier, left_shift, right_shift)
#else
      MPY_BY_QUANT_MULT_X2_OUT32(d_out32, d_out32, out_multiplier, left_shift, right_shift)
#endif      
      d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
      AE_MINMAX32_HIFI4(d_out32, min_int8, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(d_out32);
    }
  }
   else if(((vec_length & 3) == 0) && (((int)p_inp1_start & 7) == 0))
  {
    /* Assumption: 
     * p_inp1_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * p_inp2_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * */
    pt_inp1 = (const ae_int16x4 *)((WORD16 *)p_inp1_start);
    pt_inp2 = (const ae_int16x4 *)((WORD16 *)p_inp2_start);

    align_inp2 = AE_LA64_PP(pt_inp2);
    /* TBD: multiple vec_count processing in a single loop can be done */
    for(loopcnt = 0; loopcnt < vec_count; loopcnt++)
    {
      AE_L32_XP(d_bias, (ae_int32 *)p_bias_load, bias_address_increment);

      d_out64_0 = ZERO64;

#pragma no_unroll
      for(i = 0; i < (vec_length >> 2); i++)
      {
        AE_L16X4_IP(d_inp1_0, pt_inp1, 8);
        AE_LA16X4_IP(d_inp2_0, align_inp2, pt_inp2);
        AE_MULAAAAQ16(d_out64_0, d_inp1_0, d_inp2_0);
      }
      AE_SAT32X2_HIFI4(d_out32, d_out64_0);
      d_out32 = AE_ADD32S(d_out32, d_bias);
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_out32, d_out32, out_multiplier, left_shift, right_shift)
#else
      MPY_BY_QUANT_MULT_X2_OUT32(d_out32, d_out32, out_multiplier, left_shift, right_shift)
#endif      
      d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
      AE_MINMAX32_HIFI4(d_out32, min_int8, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(d_out32);
    }
  }
  else if(((vec_length & 3) == 0))
  {
    /* Assumption: 
     * p_inp1_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * p_inp2_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * */
    pt_inp1 = (const ae_int16x4 *)((WORD16 *)p_inp1_start);
    pt_inp2 = (const ae_int16x4 *)((WORD16 *)p_inp2_start);

    align_inp1 = AE_LA64_PP(pt_inp1);
    align_inp2 = AE_LA64_PP(pt_inp2);
    /* TBD: multiple vec_count processing in a single loop can be done */
    for(loopcnt = 0; loopcnt < vec_count; loopcnt++)
    {
      AE_L32_XP(d_bias, (ae_int32 *)p_bias_load, bias_address_increment);

      d_out64_0 = ZERO64;

#pragma no_unroll
      for(i = 0; i < (vec_length >> 2); i++)
      {
        AE_LA16X4_IP(d_inp1_0, align_inp1, pt_inp1);
        AE_LA16X4_IP(d_inp2_0, align_inp2, pt_inp2);
        AE_MULAAAAQ16(d_out64_0, d_inp1_0, d_inp2_0);
      }
      AE_SAT32X2_HIFI4(d_out32, d_out64_0);
      d_out32 = AE_ADD32S(d_out32, d_bias);
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_out32, d_out32, out_multiplier, left_shift, right_shift)
#else
      MPY_BY_QUANT_MULT_X2_OUT32(d_out32, d_out32, out_multiplier, left_shift, right_shift)
#endif
      d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
      AE_MINMAX32_HIFI4(d_out32, min_int8, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(d_out32);
    }
  }
  else
  {
    for(loopcnt = 0; loopcnt < vec_count; loopcnt++)
    {
      pt_inp1 = (const ae_int16x4 *)((WORD16 *)p_inp1_start + (loopcnt * vec_length));
      pt_inp2 = (const ae_int16x4 *)((WORD16 *)p_inp2_start + (loopcnt * vec_length));
      align_inp1 = AE_LA64_PP(pt_inp1);
      align_inp2 = AE_LA64_PP(pt_inp2);
      d_out64_0 = ZERO64;

      AE_L32_XP(d_bias, (ae_int32 *)p_bias_load, bias_address_increment);

      for(i = 0; i < (vec_length >> 2); i++)
      {
        AE_LA16X4_IP(d_inp1_0, align_inp1, pt_inp1);
        AE_LA16X4_IP(d_inp2_0, align_inp2, pt_inp2);
        AE_MULAAAAQ16(d_out64_0, d_inp1_0, d_inp2_0);
      }
#if (( XCHAL_HW_VERSION >= RI9_HWVERSION )& (XCHAL_HAVE_HIFI1))
       int rem_len = (vec_length & 3);
       {
        AE_LAV16X4_XP(d_inp1_0, align_inp1, pt_inp1, (rem_len<<1));
        AE_LAV16X4_XP(d_inp2_0, align_inp2, pt_inp2, (rem_len<<1));
        AE_MULAAAAQ16(d_out64_0, d_inp1_0, d_inp2_0);
       }
#else
      for(i = 0; i < (vec_length & 3); i++)
      {
        AE_L16_IP(d_inp1_0, (ae_int16 *)pt_inp1, 2);
        AE_L16_IP(d_inp2_0, (ae_int16 *)pt_inp2, 2);
        AE_MULA16_00(d_out64_0, d_inp1_0, d_inp2_0);
      }
#endif
      AE_SAT32X2_HIFI4(d_out32, d_out64_0);
      d_out32 = AE_ADD32S(d_out32, d_bias);
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_out32, d_out32, out_multiplier, left_shift, right_shift)
#else
      MPY_BY_QUANT_MULT_X2_OUT32(d_out32, d_out32, out_multiplier, left_shift, right_shift)
#endif
      d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
      AE_MINMAX32_HIFI4(d_out32, min_int8, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(d_out32);
    }
  }
  return 0;
}
