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
#include "xa_type_def.h"
#include "xa_nn_common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_hifi_isa_compat.h"

static inline void compute_interpolation_values_integer(
    const WORD32 value, const WORD32 scale_10, const WORD32 shift,
    WORD32 input_size, WORD32* scaled_value, WORD32* lower_bound, WORD32* upper_bound)
{
  *scaled_value = value * scale_10 + shift;

  *lower_bound = XT_MIN(XT_MAX((*scaled_value >> 10), 0), input_size - 1);
  *upper_bound = XT_MIN(((*scaled_value + (1 << 10) - 1) >> 10), input_size - 1);
}

WORD32 xa_nn_resize_bilinear_8_8
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_inp
  ,WORD32  input_batch
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  out_batch
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  out_channels
  ,WORD32  height_scale_10
  ,WORD32  width_scale_10
  ,WORD32  height_shift
  ,WORD32  width_shift
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_batch <= 0 || input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_batch != input_batch || out_channels != input_channels), -1);

  int itr_n, itr_h, itr_w, itr_c;

  int width_off  = input_channels;
  int height_off = input_width * width_off;
  int batch_off  = input_height * height_off;

  WORD8 *ptmp_inp = (WORD8 *)p_inp, *ptmp_out = (WORD8 *)p_out;
  WORD8 *ptmp_inp_h0, *ptmp_inp_h1;
  WORD8 *ptmp_inp_h0w0, *ptmp_inp_h0w1, *ptmp_inp_h1w0, *ptmp_inp_h1w1;

  ae_int32x2 d_one_q10 = AE_MOVDA32(1 << 10);
  if((((unsigned)p_inp) & 3) == 0 && (input_channels & 3) == 0)
  {
    for (itr_n = 0; itr_n < out_batch; itr_n++)
    {
      for (itr_h = 0; itr_h < out_height; itr_h++)
      {
        WORD32 input_y, y0, y1;
        compute_interpolation_values_integer(itr_h, height_scale_10,
                                          height_shift, input_height,
                                          &input_y, &y0, &y1);
        ae_int32x2 y_m_y0_q10, one_m_y_m_y0_q10;
        y_m_y0_q10 = AE_MOVDA32(input_y - (y0 << 10));
        one_m_y_m_y0_q10 = AE_SUB32(d_one_q10, y_m_y0_q10);
        ptmp_inp_h0 = ptmp_inp + y0 * height_off;
        ptmp_inp_h1 = ptmp_inp + y1 * height_off;
        for (itr_w = 0; itr_w < out_width; itr_w++)
        {
          WORD32 input_x, x0, x1;
          compute_interpolation_values_integer(itr_w, width_scale_10,
                                            width_shift, input_width,
                                            &input_x, &x0, &x1);
          ae_int32x2 x_m_x0_q10, one_m_x_m_x0_q10;
          x_m_x0_q10 = AE_MOVDA32(input_x - (x0 << 10));
          one_m_x_m_x0_q10 = AE_SUB32(d_one_q10, x_m_x0_q10);
          ptmp_inp_h0w0 = ptmp_inp_h0 + x0 * width_off;
          ptmp_inp_h0w1 = ptmp_inp_h0 + x1 * width_off;
          ptmp_inp_h1w0 = ptmp_inp_h1 + x0 * width_off;
          ptmp_inp_h1w1 = ptmp_inp_h1 + x1 * width_off;
          ae_int32x2 mul_ll, mul_lu, mul_rl, mul_ru;
          mul_ll = AE_MULP32X2(one_m_x_m_x0_q10, one_m_y_m_y0_q10);
          mul_lu = AE_MULP32X2(one_m_x_m_x0_q10, y_m_y0_q10);
          mul_rl = AE_MULP32X2(x_m_x0_q10, one_m_y_m_y0_q10);
          mul_ru = AE_MULP32X2(x_m_x0_q10, y_m_y0_q10);
#pragma no_unroll
          for (itr_c = 0; itr_c < (out_channels >> 2); itr_c++)
          {
            ae_int16x4 d_inp_ll, d_inp_lu, d_inp_rl, d_inp_ru;
            AE_L8X4F_IP(d_inp_ll, ptmp_inp_h0w0, 4);
            AE_L8X4F_IP(d_inp_lu, ptmp_inp_h1w0, 4);
            AE_L8X4F_IP(d_inp_rl, ptmp_inp_h0w1, 4);
            AE_L8X4F_IP(d_inp_ru, ptmp_inp_h1w1, 4);
            d_inp_ll = AE_SRAI16(d_inp_ll, 8);
            d_inp_lu = AE_SRAI16(d_inp_lu, 8);
            d_inp_rl = AE_SRAI16(d_inp_rl, 8);
            d_inp_ru = AE_SRAI16(d_inp_ru, 8);

            ae_int32x2 d_out_0, d_out_1;

            d_out_0 = AE_MULP32X16X2_H(mul_ll, d_inp_ll);
            d_out_1 = AE_MULP32X16X2_L(mul_ll, d_inp_ll);

            AE_MULAP32X16X2_H(d_out_0, mul_lu, d_inp_lu);
            AE_MULAP32X16X2_L(d_out_1, mul_lu, d_inp_lu);

            AE_MULAP32X16X2_H(d_out_0, mul_rl, d_inp_rl);
            AE_MULAP32X16X2_L(d_out_1, mul_rl, d_inp_rl);

            AE_MULAP32X16X2_H(d_out_0, mul_ru, d_inp_ru);
            AE_MULAP32X16X2_L(d_out_1, mul_ru, d_inp_ru);

            ae_int16x4 d_out16;
#if TFLITE_SINGLE_ROUNDING
            d_out_0 = AE_SRAI32(d_out_0, 4);
            d_out_1 = AE_SRAI32(d_out_1, 4);
            d_out16 = AE_ROUND16X4F32SASYM(d_out_0, d_out_1);
            d_out16 = AE_SRAI16(AE_SLAI16S(d_out16, 8), 8);
#else
            d_out_0 = AE_MULFP32X2RS(d_out_0, AE_MOVDA32(1 << 11));
            d_out_1 = AE_MULFP32X2RS(d_out_1, AE_MOVDA32(1 << 11));
            d_out16 = AE_SAT16X4(d_out_0, d_out_1);
            d_out16 = AE_SRAI16(AE_SLAI16S(d_out16, 8), 8);
#endif  // TFLITE_SINGLE_ROUNDING
            *ptmp_out++ = (WORD8)AE_MOVAD16_3(d_out16);
            *ptmp_out++ = (WORD8)AE_MOVAD16_2(d_out16);
            *ptmp_out++ = (WORD8)AE_MOVAD16_1(d_out16);
            *ptmp_out++ = (WORD8)AE_MOVAD16_0(d_out16);
          }
        }
      }
      ptmp_inp += batch_off;
    }
  }
  else
  {
    for (itr_n = 0; itr_n < out_batch; itr_n++)
    {
      for (itr_h = 0; itr_h < out_height; itr_h++)
      {
        WORD32 input_y, y0, y1;
        compute_interpolation_values_integer(itr_h, height_scale_10,
                                          height_shift, input_height,
                                          &input_y, &y0, &y1);
        ae_int32x2 y_m_y0_q10, one_m_y_m_y0_q10;
        y_m_y0_q10 = AE_MOVDA32(input_y - (y0 << 10));
        one_m_y_m_y0_q10 = AE_SUB32(d_one_q10, y_m_y0_q10);
        ptmp_inp_h0 = ptmp_inp + y0 * height_off;
        ptmp_inp_h1 = ptmp_inp + y1 * height_off;
        for (itr_w = 0; itr_w < out_width; itr_w++)
        {
          WORD32 input_x, x0, x1;
          compute_interpolation_values_integer(itr_w, width_scale_10,
                                            width_shift, input_width,
                                            &input_x, &x0, &x1);
          ae_int32x2 x_m_x0_q10, one_m_x_m_x0_q10;
          x_m_x0_q10 = AE_MOVDA32(input_x - (x0 << 10));
          one_m_x_m_x0_q10 = AE_SUB32(d_one_q10, x_m_x0_q10);
          ptmp_inp_h0w0 = ptmp_inp_h0 + x0 * width_off;
          ptmp_inp_h0w1 = ptmp_inp_h0 + x1 * width_off;
          ptmp_inp_h1w0 = ptmp_inp_h1 + x0 * width_off;
          ptmp_inp_h1w1 = ptmp_inp_h1 + x1 * width_off;
          ALIGN_REGISTER_TYPE align_h0w0, align_h0w1, align_h1w0, align_h1w1;
          PRIME_8X4F(ptmp_inp_h0w0, align_h0w0);
          PRIME_8X4F(ptmp_inp_h0w1, align_h0w1);
          PRIME_8X4F(ptmp_inp_h1w0, align_h1w0);
          PRIME_8X4F(ptmp_inp_h1w1, align_h1w1);
          ae_int32x2 mul_ll, mul_lu, mul_rl, mul_ru;
          mul_ll = AE_MULP32X2(one_m_x_m_x0_q10, one_m_y_m_y0_q10);
          mul_lu = AE_MULP32X2(one_m_x_m_x0_q10, y_m_y0_q10);
          mul_rl = AE_MULP32X2(x_m_x0_q10, one_m_y_m_y0_q10);
          mul_ru = AE_MULP32X2(x_m_x0_q10, y_m_y0_q10);
#pragma concurrent
          for (itr_c = 0; itr_c < (out_channels >> 2); itr_c++)
          {
            ae_int16x4 d_inp_ll, d_inp_lu, d_inp_rl, d_inp_ru;
            AE_LA8X4F_IP(d_inp_ll, align_h0w0, ptmp_inp_h0w0);
            AE_LA8X4F_IP(d_inp_lu, align_h1w0, ptmp_inp_h1w0);
            AE_LA8X4F_IP(d_inp_rl, align_h0w1, ptmp_inp_h0w1);
            AE_LA8X4F_IP(d_inp_ru, align_h1w1, ptmp_inp_h1w1);
            d_inp_ll = AE_SRAI16(d_inp_ll, 8);
            d_inp_lu = AE_SRAI16(d_inp_lu, 8);
            d_inp_rl = AE_SRAI16(d_inp_rl, 8);
            d_inp_ru = AE_SRAI16(d_inp_ru, 8);

            ae_int32x2 d_out_0, d_out_1;

            d_out_0 = AE_MULP32X16X2_H(mul_ll, d_inp_ll);
            d_out_1 = AE_MULP32X16X2_L(mul_ll, d_inp_ll);

            AE_MULAP32X16X2_H(d_out_0, mul_lu, d_inp_lu);
            AE_MULAP32X16X2_L(d_out_1, mul_lu, d_inp_lu);

            AE_MULAP32X16X2_H(d_out_0, mul_rl, d_inp_rl);
            AE_MULAP32X16X2_L(d_out_1, mul_rl, d_inp_rl);

            AE_MULAP32X16X2_H(d_out_0, mul_ru, d_inp_ru);
            AE_MULAP32X16X2_L(d_out_1, mul_ru, d_inp_ru);

            ae_int16x4 d_out16;
#if TFLITE_SINGLE_ROUNDING
            d_out_0 = AE_SRAI32(d_out_0, 4);
            d_out_1 = AE_SRAI32(d_out_1, 4);
            d_out16 = AE_ROUND16X4F32SASYM(d_out_0, d_out_1);
            d_out16 = AE_SRAI16(AE_SLAI16S(d_out16, 8), 8);
#else
            d_out_0 = AE_MULFP32X2RS(d_out_0, AE_MOVDA32(1 << 11));
            d_out_1 = AE_MULFP32X2RS(d_out_1, AE_MOVDA32(1 << 11));
            d_out16 = AE_SAT16X4(d_out_0, d_out_1);
            d_out16 = AE_SRAI16(AE_SLAI16S(d_out16, 8), 8);
#endif  // TFLITE_SINGLE_ROUNDING
            *ptmp_out++ = (WORD8)AE_MOVAD16_3(d_out16);
            *ptmp_out++ = (WORD8)AE_MOVAD16_2(d_out16);
            *ptmp_out++ = (WORD8)AE_MOVAD16_1(d_out16);
            *ptmp_out++ = (WORD8)AE_MOVAD16_0(d_out16);
          }
#pragma concurrent
#pragma loop_count max=3
          for (itr_c = 0; itr_c < (out_channels & 3); itr_c++)
          {
            ae_int16x4 d_inp_ll_lu, d_inp_rl_ru, d_inp;
            d_inp_ll_lu = AE_MOVDA16X2((UWORD8)ptmp_inp_h0w0[itr_c], (UWORD8)ptmp_inp_h1w0[itr_c]);
            d_inp_rl_ru = AE_MOVDA16X2((UWORD8)ptmp_inp_h0w1[itr_c], (UWORD8)ptmp_inp_h1w1[itr_c]);
            d_inp = AE_SEL16_7632(d_inp_ll_lu, d_inp_rl_ru);
            d_inp = AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(AE_MOVINT32X2_FROMINT16X4(d_inp), 8));

            ae_int64 d_out64;
            ae_int32x2 d_out_0;

#ifdef AE_MULZAAAAQ32X16
            d_out64 = AE_MULZAAAAQ32X16(AE_SEL32_LL(mul_ll, mul_lu), AE_SEL32_LL(mul_rl, mul_ru), d_inp);
#else
            d_out64 = AE_MULZAAD32X16_H3_L2(AE_SEL32_LL(mul_ll, mul_lu), d_inp);
            AE_MULAAD32X16_H1_L0(d_out64, AE_SEL32_LL(mul_rl, mul_ru), d_inp);
#endif

            ae_int16x4 d_out16;
#if TFLITE_SINGLE_ROUNDING
            d_out_0 = AE_TRUNCA32X2F64S(d_out64, d_out64, 28-8);
            d_out16 = AE_ROUND16X4F32SASYM(d_out_0, d_out_0);
            d_out16 = AE_SRAI16(AE_SLAI16S(d_out16, 8), 8);
#else
            d_out_0 = AE_TRUNCA32X2F64S(d_out64, d_out64, 32-8);
            d_out_0 = AE_MULFP32X2RS(d_out_0, AE_MOVDA32(1 << 11));
            d_out16 = AE_SAT16X4(d_out_0, d_out_0);
            d_out16 = AE_SRAI16(AE_SLAI16S(d_out16, 8), 8);
#endif  // TFLITE_SINGLE_ROUNDING
            *ptmp_out++ = (WORD8)AE_MOVAD16_0(d_out16);
          }
        }
      }
      ptmp_inp += batch_off;
    }
  }

  return 0;
}

