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

static void internal_elm_mul_broadcast_2D_sym16sxsym16s_sym16s(WORD16 * __restrict__ p_out,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   WORD16 * __restrict__ p_inp1,
                    const   WORD16 * __restrict__ p_inp2,
                            WORD32  out_lc,
                            WORD32  in_lc)

{
  int i, j;
  WORD16 * __restrict__ p_a; 
  WORD16 * __restrict__ p_b; 
  WORD16 *__restrict__ p_c;

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
    p_a = (WORD16 *)&p_inp1[i * in_lc];
    p_b = (WORD16 *)p_inp2;
    p_c = (WORD16 *)&p_out[i * in_lc];

   ae_valign va_a, va_b, va_c;
   va_a = AE_LA64_PP(p_a);
   va_b = AE_LA64_PP(p_b);
   va_c = AE_ZALIGN64();
   
   for(j = 0; j < num_simd8_ops; j++)
   {
     AE_LA16X4_IP(a0_3, va_a, (ae_int16x4 *)p_a);
     AE_LA16X4_IP(a4_7, va_a, (ae_int16x4 *)p_a);
     AE_LA16X4_IP(b0_3, va_b, (ae_int16x4 *)p_b);
     AE_LA16X4_IP(b4_7, va_b, (ae_int16x4 *)p_b);

     AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
     AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b4_7);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
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

     CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
     CLAMP_VAL(out_mul2_3, out_mul2_3, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
     CLAMP_VAL(out_mul4_5, out_mul4_5, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
     CLAMP_VAL(out_mul6_7, out_mul6_7, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));

      ae_int16x4 outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out_mul0_1), AE_MOVINT16X4_FROMINT32X2(out_mul2_3));
      AE_SA16X4_IP(outval, va_c, (ae_int16x4 *)p_c);
      outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out_mul4_5), AE_MOVINT16X4_FROMINT32X2(out_mul6_7));
      AE_SA16X4_IP(outval, va_c, (ae_int16x4 *)p_c);
    }
    AE_SA64POS_FP(va_c, (ae_int16x4 *)p_c);
  }
#if XCHAL_HAVE_HIFI1 & (XCHAL_HW_VERSION >= RI9_HWVERSION)
  if(num_scalar_ops!=0){
    for(i = 0; i < out_lc; i++)
    {
      p_a = (WORD16 *)&p_inp1[i * in_lc + (num_simd8_ops << 3)];
      p_c = (WORD16 *)&p_out[i * in_lc + (num_simd8_ops << 3)];
      p_b = (WORD16 *)&p_inp2[num_simd8_ops << 3];
      ae_valign va_a, va_b, va_c;
      va_a = AE_LA64_PP(p_a);
      va_b = AE_LA64_PP(p_b);
      va_c = AE_ZALIGN64();
      {
          WORD32 num_scalar_ops0, num_scalar_ops1, num_scalar_ops2, num_scalar_ops3;
          num_scalar_ops0 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          num_scalar_ops1 = num_scalar_ops >= 8 ? 4 : (num_scalar_ops > 4? num_scalar_ops - 4 : 0); 
          num_scalar_ops2 = num_scalar_ops >= 12 ? 4 : (num_scalar_ops > 8? num_scalar_ops - 8 : 0);
          num_scalar_ops3 = num_scalar_ops > 12 ? num_scalar_ops - 12 : 0;

          AE_LAV16X4_XP(a0_3, va_a, (ae_int16x4 *)p_a, (num_scalar_ops0<<1));
          AE_LAV16X4_XP(a4_7, va_a, (ae_int16x4 *)p_a, (num_scalar_ops1<<1));         
          AE_LAV16X4_XP(b0_3, va_b, (ae_int16x4 *)p_b, (num_scalar_ops0<<1));
          AE_LAV16X4_XP(b4_7, va_b, (ae_int16x4 *)p_b, (num_scalar_ops1<<1));
          
          AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
          AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b4_7);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
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
          CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          CLAMP_VAL(out_mul2_3, out_mul2_3, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          CLAMP_VAL(out_mul4_5, out_mul4_5, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          CLAMP_VAL(out_mul6_7, out_mul6_7, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          
          ae_int16x4 outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out_mul0_1), AE_MOVINT16X4_FROMINT32X2(out_mul2_3));
          
          AE_SAV16X4_XP(outval, va_c, (ae_int16x4 *)p_c, (num_scalar_ops0<<1));
          outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out_mul4_5), AE_MOVINT16X4_FROMINT32X2(out_mul6_7));
          AE_SAV16X4_XP(outval, va_c, (ae_int16x4 *)p_c, (num_scalar_ops1<<1)); 
          
          AE_LAV16X4_XP(a0_3, va_a, (ae_int16x4 *)p_a, (num_scalar_ops2<<1));
          AE_LAV16X4_XP(a4_7, va_a, (ae_int16x4 *)p_a, (num_scalar_ops3<<1));         
          AE_LAV16X4_XP(b0_3, va_b, (ae_int16x4 *)p_b, (num_scalar_ops2<<1));
          AE_LAV16X4_XP(b4_7, va_b, (ae_int16x4 *)p_b, (num_scalar_ops3<<1));
          
          AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
          AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b4_7);

#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
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

          CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          CLAMP_VAL(out_mul2_3, out_mul2_3, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          CLAMP_VAL(out_mul4_5, out_mul4_5, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          CLAMP_VAL(out_mul6_7, out_mul6_7, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          
          outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out_mul0_1), AE_MOVINT16X4_FROMINT32X2(out_mul2_3));
          
          AE_SAV16X4_XP(outval, va_c, (ae_int16x4 *)p_c, (num_scalar_ops2<<1));
          outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out_mul4_5), AE_MOVINT16X4_FROMINT32X2(out_mul6_7));
          AE_SAV16X4_XP(outval, va_c, (ae_int16x4 *)p_c, (num_scalar_ops3<<1));           
      }
          AE_SA64POS_FP(va_c, (ae_int16x4 *)p_c);
    }
  }
#else
  if(num_scalar_ops!=0){
    for(i = 0; i < out_lc; i++)
    {
      p_a = (WORD16 *)&p_inp1[i * in_lc + (num_simd8_ops << 3)];
      p_c = (WORD16 *)&p_out[i * in_lc + (num_simd8_ops << 3)];
      p_b = (WORD16 *)&p_inp2[num_simd8_ops << 3];
      for(j = 0; j< num_scalar_ops; j++)
      {
        b0_3 = AE_MOVDA16(p_b[j]);
        a0_3 = AE_MOVDA16(p_a[j]);
        AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#else         
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#endif        
        CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
        *(WORD16 *)p_c = (WORD16)AE_MOVAD32_H(out_mul0_1);
        p_c++;
     }
   }
  }
#endif
}

static void internal_elm_mul_broadcast_sym16sxsym16s_sym16s(WORD16 * __restrict__ p_out,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   WORD16 * __restrict__ p_inp1,
                    const   WORD16 * __restrict__ p_inp2,
                            WORD32  num_elm)

{
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

  int i;
  WORD16 * __restrict__ p_a = (WORD16 *)p_inp1;
  WORD16 * __restrict__ p_b = (WORD16 *)p_inp2;
  WORD16 *__restrict__ p_c =          p_out;

  ae_int16x4 a0_3, b0;

  ae_int32x2 raw_mul0_1, raw_mul2_3;
  ae_int32x2 out_mul0_1, out_mul2_3;

  const int num_simd4_ops = num_elm >> 2;
  const int num_scalar_ops = num_elm & 3;
  
  b0 = AE_MOVDA16(p_b[0]);

  ae_valign va_a, va_c;
  va_a = AE_LA64_PP((ae_int16x4 *)p_a);
  va_c = AE_ZALIGN64();

  for(i=0; i<num_simd4_ops; i++)
  {

    AE_LA16X4_IP(a0_3, va_a, (ae_int16x4 *)p_a);

    // LSH (and promote to 32-bit)
    AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0);
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#else
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul2_3, raw_mul2_3, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#endif    
    CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
    CLAMP_VAL(out_mul2_3, out_mul2_3, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
    ae_int16x4 outval = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(out_mul0_1), AE_MOVINT16X4_FROMINT32X2(out_mul2_3));
    AE_SA16X4_IP(outval, va_c, (ae_int16x4 *)p_c);
  }
  AE_SA64POS_FP(va_c, (ae_int16x4 *)p_c);

  for(i=0; i<num_scalar_ops; i++)
  {
    a0_3 = AE_MOVDA16(p_a[i]);
    AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0);
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#else    
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out_mul0_1, raw_mul0_1, AE_MOVDA32(out_multiplier), l_shift, r_shift);
#endif    
    CLAMP_VAL(out_mul0_1, out_mul0_1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
    *p_c = (WORD16)AE_MOVAD32_L(out_mul0_1);
    p_c++;
  }
}

WORD32 xa_nn_elm_mul_broadcast_4D_sym16sxsym16s_sym16s(WORD16 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                      const WORD16 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const WORD16 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape
                            )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_min < -32768) || (out_activation_min > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_max < -32768) || (out_activation_max > 32767)), -1);
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

  WORD16 *p_out_tmp = p_out;
  const WORD16 *__restrict__ p_inp1_tmp = p_inp1;
  const WORD16 *__restrict__ p_inp2_tmp = p_inp2;
  if(need_broadcast == 0)
  {
    internal_elm_mul_broadcast_2D_sym16sxsym16s_sym16s(
                p_out,
                out_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1,
                p_inp2,
                1,
                p_out_shape[0] * inp1_strides[0]);
  }
  else if(inp1_strides[3] == inp2_strides[3])
  {
    WORD32 in_lc, out_lc;

    in_lc = p_out_shape[2] * p_out_shape[3];
    out_lc = 1;
    if(inp1_strides[2] == 0)
    {
      const WORD16 *tmp;
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
      const WORD16 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const WORD16 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        internal_elm_mul_broadcast_2D_sym16sxsym16s_sym16s(
            p_out_tmp,
            out_shift,
            out_multiplier,
            out_activation_min,
            out_activation_max,
            p_inp1_tmp0,
            p_inp2_tmp0,
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
    if(inp1_strides[3] == 0)
    {
      const WORD16 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
    }

    internal_elm_mul_broadcast_sym16sxsym16s_sym16s(
        p_out_tmp,
        out_shift,
        out_multiplier,
        out_activation_min,
        out_activation_max,
        p_inp1_tmp,
        p_inp2_tmp,
        p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3]);
  }
  else
  {
    if(inp1_strides[3] == 0)
    {
      const WORD16 *tmp;
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
      const WORD16 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const WORD16 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        const WORD16 *__restrict__ p_inp1_tmp1 = p_inp1_tmp0;
        const WORD16 *__restrict__ p_inp2_tmp1 = p_inp2_tmp0;
        for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
        {
          {
            internal_elm_mul_broadcast_sym16sxsym16s_sym16s(
                p_out_tmp,
                out_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1_tmp1,
                p_inp2_tmp1,
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

WORD32 xa_nn_elm_mul_sym16sxsym16s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   WORD16  * __restrict__ p_inp1,
                    const   WORD16  * __restrict__ p_inp2,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_max < out_activation_min) || (out_activation_max > 127)), -1);

  int i;
  ae_int16x4 * p_a;
  ae_int16x4 * p_b;
  WORD8  * p_c;

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

  // intermediate results and scratch registers
  ae_int16x4 a0_3, b0_3;

  ae_int32x2 raw_mul0_1, raw_mul2_3;

  ae_int32x2 out0, out1;

  int num_simd4_ops;
  int num_scalar_ops;

  num_simd4_ops = num_elm >> 2;
  num_scalar_ops = num_elm & 3;

  ae_valign va_a, va_b;

  p_a = (ae_int16x4 *)p_inp1;
  p_b = (ae_int16x4 *)p_inp2;
  p_c = (WORD8 *)p_out;

  va_a = AE_LA64_PP(p_a);
  va_b = AE_LA64_PP(p_b);

  for(i = 0; i < num_simd4_ops; i++)
  {
    AE_LA16X4_IP(a0_3, va_a, p_a);
    AE_LA16X4_IP(b0_3, va_b, p_b);

    AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out0, raw_mul0_1, out_multiplier, l_shift, r_shift);
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out1, raw_mul2_3, out_multiplier, l_shift, r_shift);  
#else
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out0, raw_mul0_1, out_multiplier, l_shift, r_shift);
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out1, raw_mul2_3, out_multiplier, l_shift, r_shift);  
#endif       
    out0 = AE_ADD32S(AE_MOVDA32(out_zero_bias), out0);
    out1 = AE_ADD32S(AE_MOVDA32(out_zero_bias), out1);  
    CLAMP_VAL(out0, out0, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
    CLAMP_VAL(out1, out1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
    STORE_8X4_FROM_32X4(p_c, out0, out1);
  }

  for(i = 0; i < num_scalar_ops; i++)
  {
    AE_L16_IP(a0_3, (ae_int16 *)p_a, sizeof(WORD16));
    AE_L16_IP(b0_3, (ae_int16 *)p_b, sizeof(WORD16));
    AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out0, raw_mul0_1, out_multiplier, l_shift, r_shift);
#else    
    MPY_BY_QUANT_MULT_SLS_X2_OUT32(out0, raw_mul0_1, out_multiplier, l_shift, r_shift);
#endif
    out0 = AE_ADD32S(AE_MOVDA32(out_zero_bias), out0);
    CLAMP_VAL(out0, out0, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
    *p_c++ = (WORD8)AE_MOVAD32_H(out0);
  }

  return 0;
}
