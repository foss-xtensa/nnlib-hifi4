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
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_quant_macros.h"

WORD32 xa_nn_elm_add_16x16_16(WORD16 * __restrict__ p_out,
                        const WORD16 * __restrict__ p_inp1,
                        const WORD16 * __restrict__ p_inp2,
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
  XA_NNLIB_ARG_CHK_COND((num_elm < 0), -1);

  int i;
  ae_int16x4 * __restrict__ p_a;
  ae_int16x4 * __restrict__ p_b;
  ae_int16x4 *__restrict__ p_c;

  // intermediate results and scratch registers
  ae_int16x4 a0_3, b0_3;

  ae_int16x4 out0;

  int num_simd4_ops;
  int num_scalar_ops;

  num_simd4_ops = num_elm >> 2;
  num_scalar_ops = num_elm & 3;

  ae_valign va_a, va_b, va_c;

  p_a = (ae_int16x4 *)p_inp1;
  p_b = (ae_int16x4 *)p_inp2;
  p_c = (ae_int16x4 *)p_out;

  va_a = AE_LA64_PP(p_a);
  va_b = AE_LA64_PP(p_b);
  va_c = AE_ZALIGN64();
  for(i = 0; i < num_simd4_ops; i++)
  {
    AE_LA16X4_IP(a0_3, va_a, p_a);
    AE_LA16X4_IP(b0_3, va_b, p_b);
    out0 = AE_ADD16S(a0_3, b0_3);
    AE_SA16X4_IP(out0, va_c, p_c);
  }
  AE_SA64POS_FP(va_c, p_c);

  for(i = 0; i < num_scalar_ops; i++)
  {
    AE_L16_IP(a0_3, (ae_int16 * )p_a, sizeof(WORD16));
    AE_L16_IP(b0_3, (ae_int16 * )p_b, sizeof(WORD16));
    out0 = AE_ADD16S(a0_3, b0_3);
    AE_S16_0_IP(out0, (ae_int16 * )p_c, sizeof(WORD16));    
  }
  
  return 0;
}

WORD32 xa_nn_lstm_cell_state_update_16(WORD16* p_cell_state,
                                           const WORD16* p_forget_gate,
                                           const WORD16* p_cell_gate,
                                           const WORD16* p_input_gate,
                                           WORD32 cell_to_forget_shift,
                                           WORD32 cell_to_input_shift,
                                           WORD32 quantized_cell_clip,
                                           WORD32 num_elms)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_cell_state, -1);
  XA_NNLIB_ARG_CHK_PTR(p_forget_gate, -1);
  XA_NNLIB_ARG_CHK_PTR(p_cell_gate, -1);
  XA_NNLIB_ARG_CHK_PTR(p_input_gate, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_cell_state, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_forget_gate, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_cell_gate, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_input_gate, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((cell_to_forget_shift < -31 || cell_to_forget_shift > -1), -1);
  XA_NNLIB_ARG_CHK_COND((cell_to_input_shift < -31 || cell_to_input_shift > -1), -1);
  XA_NNLIB_ARG_CHK_COND((num_elms < 0), -1);

  WORD32 ctof_right_shift, ctoi_right_shift;
#if TFLITE_SINGLE_ROUNDING
  ctof_right_shift = -cell_to_forget_shift;
  ctoi_right_shift = -cell_to_input_shift;
#else
  ctof_right_shift = -cell_to_forget_shift - 1;
  ctoi_right_shift = -cell_to_input_shift - 1;
#endif

  const ae_int16x4 *p16x4_cs_r, *p16x4_fg_r;
  const ae_int16x4 *p16x4_cg_r, *p16x4_ig_r;

  ae_int16x4* p16x4_cs_w;

  ae_int16x4 d_cs_r_0;
  ae_int16x4 d_fg_0;
  ae_int16x4 d_cg_0;
  ae_int16x4 d_ig_0;
  ae_int32x2 d_cs_w_0, d_cs_w_1;
  ae_int32x2 d_cgi_0, d_cgi_1;
  ae_int32x2 d_mul_0, d_mul_1;
  ae_int32x2 d_mul_4, d_mul_5;

  ae_int32x2 d_min, d_max;

  ae_valign d_fg_0_align, d_cg_0_align, d_ig_0_align;

  int i = 0;
  p16x4_cs_r = (const ae_int16x4*)p_cell_state;
  p16x4_fg_r = (const ae_int16x4*)p_forget_gate;
  p16x4_cg_r = (const ae_int16x4*)p_cell_gate;
  p16x4_ig_r = (const ae_int16x4*)p_input_gate;

  p16x4_cs_w = (ae_int16x4*)p_cell_state;

  if (quantized_cell_clip > 0) {
    d_min = AE_MOVDA32(-quantized_cell_clip);
    d_max = AE_MOVDA32(quantized_cell_clip);
  } else {
    d_min = AE_MOVDA32(-32768);
    d_max = AE_MOVDA32(32767);
  }
  int pre_loop_count = ((8 - (((unsigned)p_cell_state)&7))&7)>>1;
  pre_loop_count = pre_loop_count > num_elms ? num_elms : pre_loop_count;
  int core_loop_count = num_elms - pre_loop_count;
  int post_loop_count = core_loop_count & 3;

  for (i = 0; i < pre_loop_count; i++)
  {
    AE_L16_IP(d_cs_r_0, (ae_int16 *)p16x4_cs_r, sizeof(WORD16));
    AE_L16_IP(d_fg_0, (ae_int16 *)p16x4_fg_r, sizeof(WORD16));
    AE_L16_IP(d_cg_0, (ae_int16 *)p16x4_cg_r, sizeof(WORD16));
    AE_L16_IP(d_ig_0, (ae_int16 *)p16x4_ig_r, sizeof(WORD16));

    AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);

#if TFLITE_SINGLE_ROUNDING
    d_mul_0 = AE_SRAA32RS(d_mul_0, ctof_right_shift);
    d_mul_1 = AE_SRAA32RS(d_mul_1, ctof_right_shift);
    CLAMP_VAL(d_cs_w_0, d_mul_0, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cs_w_1, d_mul_1, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#else
    d_mul_0 = AE_SRAA32RS(d_mul_0, 1);
    d_mul_1 = AE_SRAA32RS(d_mul_1, 1);
    d_mul_0 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_0), ctof_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_0), ctof_right_shift));
    d_mul_1 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_1), ctof_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_1), ctof_right_shift));
    CLAMP_VAL(d_cs_w_0, d_mul_0, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cs_w_1, d_mul_1, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#endif

    AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_ig_0);

#if TFLITE_SINGLE_ROUNDING
    d_mul_4 = AE_SRAA32RS(d_mul_4, ctoi_right_shift);
    d_mul_5 = AE_SRAA32RS(d_mul_5, ctoi_right_shift);
    CLAMP_VAL(d_cgi_0, d_mul_4, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cgi_1, d_mul_5, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#else
    d_mul_4 = AE_SRAA32RS(d_mul_4, 1);
    d_mul_5 = AE_SRAA32RS(d_mul_5, 1);
    d_mul_4 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_4), ctoi_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_4), ctoi_right_shift));
    d_mul_5 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_5), ctoi_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_5), ctoi_right_shift));
    CLAMP_VAL(d_cgi_0, d_mul_4, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cgi_1, d_mul_5, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#endif

    d_cs_w_0 = AE_ADD32S(d_cs_w_0, d_cgi_0);
    d_cs_w_1 = AE_ADD32S(d_cs_w_1, d_cgi_1);

    CLAMP_VAL(d_cs_w_0, d_cs_w_0, d_min, d_max);
    CLAMP_VAL(d_cs_w_1, d_cs_w_1, d_min, d_max);

    ae_int16x4 output = AE_SAT16X4(d_cs_w_0, d_cs_w_1);
    AE_S16_0_IP(output, (ae_int16 *)p16x4_cs_w, sizeof(WORD16));
  }
  d_fg_0_align = AE_LA64_PP(p16x4_fg_r);
  d_cg_0_align = AE_LA64_PP(p16x4_cg_r);
  d_ig_0_align = AE_LA64_PP(p16x4_ig_r);
  for (i = 0; i < (core_loop_count >> 2); i++)
  {
    AE_L16X4_IP(d_cs_r_0, (ae_int16x4 *)p16x4_cs_r, 4 * sizeof(WORD16));
    AE_LA16X4_IP(d_fg_0, d_fg_0_align, (ae_int16x4 *)p16x4_fg_r);
    AE_LA16X4_IP(d_cg_0, d_cg_0_align, (ae_int16x4 *)p16x4_cg_r);
    AE_LA16X4_IP(d_ig_0, d_ig_0_align, (ae_int16x4 *)p16x4_ig_r);

    AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);

#if TFLITE_SINGLE_ROUNDING
    d_mul_0 = AE_SRAA32RS(d_mul_0, ctof_right_shift);
    d_mul_1 = AE_SRAA32RS(d_mul_1, ctof_right_shift);
    CLAMP_VAL(d_cs_w_0, d_mul_0, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cs_w_1, d_mul_1, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#else
    d_mul_0 = AE_SRAA32RS(d_mul_0, 1);
    d_mul_1 = AE_SRAA32RS(d_mul_1, 1);
    d_mul_0 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_0), ctof_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_0), ctof_right_shift));
    d_mul_1 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_1), ctof_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_1), ctof_right_shift));
    CLAMP_VAL(d_cs_w_0, d_mul_0, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cs_w_1, d_mul_1, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#endif

    AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_ig_0);

#if TFLITE_SINGLE_ROUNDING
    d_mul_4 = AE_SRAA32RS(d_mul_4, ctoi_right_shift);
    d_mul_5 = AE_SRAA32RS(d_mul_5, ctoi_right_shift);
    CLAMP_VAL(d_cgi_0, d_mul_4, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cgi_1, d_mul_5, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#else
    d_mul_4 = AE_SRAA32RS(d_mul_4, 1);
    d_mul_5 = AE_SRAA32RS(d_mul_5, 1);
    d_mul_4 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_4), ctoi_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_4), ctoi_right_shift));
    d_mul_5 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_5), ctoi_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_5), ctoi_right_shift));
    CLAMP_VAL(d_cgi_0, d_mul_4, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cgi_1, d_mul_5, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#endif

    d_cs_w_0 = AE_ADD32S(d_cs_w_0, d_cgi_0);
    d_cs_w_1 = AE_ADD32S(d_cs_w_1, d_cgi_1);

    CLAMP_VAL(d_cs_w_0, d_cs_w_0, d_min, d_max);
    CLAMP_VAL(d_cs_w_1, d_cs_w_1, d_min, d_max);

    ae_int16x4 output = AE_SAT16X4(d_cs_w_0, d_cs_w_1);
    AE_S16X4_IP(output, (ae_int16x4 *)p16x4_cs_w, 4 * sizeof(WORD16));
  }

  for (i = 0; i < post_loop_count; i++)
  {
    AE_L16_IP(d_cs_r_0, (ae_int16 *)p16x4_cs_r, sizeof(WORD16));
    AE_L16_IP(d_fg_0, (ae_int16 *)p16x4_fg_r, sizeof(WORD16));
    AE_L16_IP(d_cg_0, (ae_int16 *)p16x4_cg_r, sizeof(WORD16));
    AE_L16_IP(d_ig_0, (ae_int16 *)p16x4_ig_r, sizeof(WORD16));

    AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);

#if TFLITE_SINGLE_ROUNDING
    d_mul_0 = AE_SRAA32RS(d_mul_0, ctof_right_shift);
    d_mul_1 = AE_SRAA32RS(d_mul_1, ctof_right_shift);
    CLAMP_VAL(d_cs_w_0, d_mul_0, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cs_w_1, d_mul_1, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#else
    d_mul_0 = AE_SRAA32RS(d_mul_0, 1);
    d_mul_1 = AE_SRAA32RS(d_mul_1, 1);
    d_mul_0 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_0), ctof_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_0), ctof_right_shift));
    d_mul_1 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_1), ctof_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_1), ctof_right_shift));
    CLAMP_VAL(d_cs_w_0, d_mul_0, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cs_w_1, d_mul_1, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#endif

    AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_ig_0);

#if TFLITE_SINGLE_ROUNDING
    d_mul_4 = AE_SRAA32RS(d_mul_4, ctoi_right_shift);
    d_mul_5 = AE_SRAA32RS(d_mul_5, ctoi_right_shift);
    CLAMP_VAL(d_cgi_0, d_mul_4, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cgi_1, d_mul_5, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#else
    d_mul_4 = AE_SRAA32RS(d_mul_4, 1);
    d_mul_5 = AE_SRAA32RS(d_mul_5, 1);
    d_mul_4 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_4), ctoi_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_4), ctoi_right_shift));
    d_mul_5 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(d_mul_5), ctoi_right_shift), AE_SRAA64(AE_CVT64F32_L(d_mul_5), ctoi_right_shift));
    CLAMP_VAL(d_cgi_0, d_mul_4, AE_MOVDA32(-32768), AE_MOVDA32(32767));
    CLAMP_VAL(d_cgi_1, d_mul_5, AE_MOVDA32(-32768), AE_MOVDA32(32767));
#endif

    d_cs_w_0 = AE_ADD32S(d_cs_w_0, d_cgi_0);
    d_cs_w_1 = AE_ADD32S(d_cs_w_1, d_cgi_1);

    CLAMP_VAL(d_cs_w_0, d_cs_w_0, d_min, d_max);
    CLAMP_VAL(d_cs_w_1, d_cs_w_1, d_min, d_max);

    ae_int16x4 output = AE_SAT16X4(d_cs_w_0, d_cs_w_1);
    AE_S16_0_IP(output, (ae_int16 *)p16x4_cs_w, sizeof(WORD16));
  }
  return 0;
}
