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
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_hifi_isa_compat.h"

WORD32 xa_nn_batch_norm_3D_8_8(WORD8 * __restrict__ p_out,
                               const WORD8 * __restrict__ p_inp,
                               const WORD16 * __restrict__ p_alpha,
                               const WORD32 * __restrict__ p_beta,
                               WORD32 io_height,
                               WORD32 io_width,
                               WORD32 io_depth,
                               WORD32 out_shift,
                               WORD32 out_activation_min,
                               WORD32 out_activation_max,
                               WORD32 inp_data_format,
                               WORD32 out_data_format)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_alpha, -1);
  XA_NNLIB_ARG_CHK_PTR(p_beta, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_alpha, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_beta, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((io_height <= 0 || io_width <= 0 || io_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_min < -128 || out_activation_min > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min || out_activation_min > 127), -1);
  XA_NNLIB_ARG_CHK_COND((inp_data_format != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

  WORD32 i, j;
  WORD8 *ptmp_out;
  const WORD8 *ptmp_inp = p_inp;
  ae_int16x4 *ptmp_alpha;
  ae_int32x2 *ptmp_beta;
  ae_valign align_alpha, align_beta;
  ae_int32x2 d_min, d_max;
  d_min = AE_MOVDA32(out_activation_min);
  d_max = AE_MOVDA32(out_activation_max);

  if(((unsigned)p_inp & 3) == 0 && (io_depth & 3) == 0)
  {
    ptmp_alpha = (ae_int16x4 *)p_alpha;
    ptmp_beta = (ae_int32x2 *)p_beta;
    align_alpha = AE_LA64_PP(ptmp_alpha);
    align_beta = AE_LA64_PP(ptmp_beta);
    for(i = 0; i < (io_depth >> 2); i++)
    {
      ptmp_inp = (const WORD8 *)&p_inp[i<<2];
      ptmp_out = (WORD8 *)&p_out[i<<2];

      ae_int16x4 d_inp, d_alpha;
      ae_int32x2 d_beta0, d_beta1;
      ae_int32x2 d_res0, d_res1;
      AE_LA16X4_IP(d_alpha, align_alpha, ptmp_alpha);
      AE_LA32X2_IP(d_beta0, align_beta, ptmp_beta);
      AE_LA32X2_IP(d_beta1, align_beta, ptmp_beta);
#pragma concurrent
      for(j = 0; j < io_height * io_width; j++)
      {
        d_inp = AE_L8X4F_I(ptmp_inp, 0);
        d_inp = AE_SRAI16(d_inp, 8);
        AE_MUL16X4(d_res0, d_res1, d_inp, d_alpha);
        d_res0 = AE_ADD32S(d_res0, d_beta0);
        d_res1 = AE_ADD32S(d_res1, d_beta1);
        d_res0 = AE_SRAA32RS(d_res0, -out_shift);
        d_res1 = AE_SRAA32RS(d_res1, -out_shift);
        d_res0 = AE_MIN32(AE_MAX32(d_res0, d_min), d_max);
        d_res1 = AE_MIN32(AE_MAX32(d_res1, d_min), d_max);
        *ptmp_out++ = (WORD8)AE_MOVAD32_H(d_res0);
        *ptmp_out++ = (WORD8)AE_MOVAD32_L(d_res0);
        *ptmp_out++ = (WORD8)AE_MOVAD32_H(d_res1);
        *ptmp_out = (WORD8)AE_MOVAD32_L(d_res1);
        ptmp_inp += io_depth;
        ptmp_out += io_depth - 3;
      }
    }
  }
  else
  {
    ALIGN_REGISTER_TYPE align_inp;
    i = 0;
    if(io_depth >= 96)
    {
      for(i = 0; i < io_height * io_width; i++)
      {
        ptmp_out = (WORD8 *)&p_out[i*io_depth];
        ptmp_inp = (const WORD8 *)&p_inp[i*io_depth];

        ptmp_alpha = (ae_int16x4 *)p_alpha;
        ptmp_beta = (ae_int32x2 *)p_beta;

        align_alpha = AE_LA64_PP(ptmp_alpha);
        align_beta = AE_LA64_PP(ptmp_beta);
        PRIME_8X4F(ptmp_inp, align_inp);
#pragma concurrent
        for(j = 0; j < (io_depth >> 2); j++)
        {
          ae_int16x4 d_inp, d_alpha;
          ae_int32x2 d_beta0, d_beta1;
          ae_int32x2 d_res0, d_res1;
          AE_LA8X4F_IP(d_inp, align_inp, ptmp_inp);
          AE_LA16X4_IP(d_alpha, align_alpha, ptmp_alpha);
          AE_LA32X2_IP(d_beta0, align_beta, ptmp_beta);
          AE_LA32X2_IP(d_beta1, align_beta, ptmp_beta);
          d_inp = AE_SRAI16(d_inp, 8);
          AE_MUL16X4(d_res0, d_res1, d_inp, d_alpha);
          d_res0 = AE_ADD32S(d_res0, d_beta0);
          d_res1 = AE_ADD32S(d_res1, d_beta1);
          d_res0 = AE_NEG32S(d_res0);
          d_res0 = AE_MULFP32X2RAS(d_res0, AE_SRAA32(AE_MOVDA32(0x80000000), -out_shift));
          d_res1 = AE_NEG32S(d_res1);
          d_res1 = AE_MULFP32X2RAS(d_res1, AE_SRAA32(AE_MOVDA32(0x80000000), -out_shift));
          d_res0 = AE_MIN32(AE_MAX32(d_res0, d_min), d_max);
          d_res1 = AE_MIN32(AE_MAX32(d_res1, d_min), d_max);
          *ptmp_out++ = (WORD8)AE_MOVAD32_H(d_res0);
          *ptmp_out++ = (WORD8)AE_MOVAD32_L(d_res0);
          *ptmp_out++ = (WORD8)AE_MOVAD32_H(d_res1);
          *ptmp_out++ = (WORD8)AE_MOVAD32_L(d_res1);
        }
      }
      i = io_depth & (~3);
    }

    for(; i < (io_depth - 1); i+=2)
    {
      ae_int16x4 d_alpha;
      ae_int32x2 d_beta;
      ptmp_alpha = (ae_int16x4 *)p_alpha;
      ptmp_beta = (ae_int32x2 *)p_beta;
      d_alpha = AE_SEL16_7362(*(((ae_int16 *)ptmp_alpha) + i), *(((ae_int16 *)ptmp_alpha) + i + 1));
      d_beta = AE_SEL32_HH(*(((ae_int32 *)ptmp_beta) + i), *(((ae_int32 *)ptmp_beta) + i + 1));
      ptmp_inp = (const WORD8 *)&p_inp[i];
      ptmp_out = (WORD8 *)&p_out[i];
      for(j = 0; j < io_height * io_width; j++)
      {
        ae_int16x4 d_inp;
        ae_int32x2 d_res0, d_res1;
        d_inp = AE_MOVDA16X2(((const UWORD8 *)ptmp_inp)[j*io_depth], ((const UWORD8 *)ptmp_inp)[j*io_depth + 1]);
        d_inp = AE_SRAI16(AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(AE_MOVINT32X2_FROMINT16X4(d_inp), 8)), 8);
        AE_MUL16X4(d_res0, d_res1, d_inp, d_alpha);
        d_res0 = AE_ADD32S(d_res0, d_beta);
        d_res0 = AE_SRAA32RS(d_res0, -out_shift);
        d_res0 = AE_MIN32(AE_MAX32(d_res0, d_min), d_max);
        *ptmp_out++ = (WORD8)AE_MOVAD32_H(d_res0);
        *ptmp_out = (WORD8)AE_MOVAD32_L(d_res0);
        ptmp_out += io_depth - 1;
      }
    }

    if((io_depth & 1) != 0)
    {
      ae_int16x4 d_alpha;
      ae_int32x2 d_beta;
      ptmp_alpha = (ae_int16x4 *)p_alpha;
      ptmp_beta = (ae_int32x2 *)p_beta;
      d_alpha = *(((ae_int16 *)ptmp_alpha) + io_depth - 1);
      d_beta = *(((ae_int32 *)ptmp_beta) + io_depth - 1);
      ptmp_inp = (const WORD8 *)&p_inp[io_depth - 1];
      ptmp_out = (WORD8 *)&p_out[io_depth - 1];
      for(j = 0; j < io_height * io_width; j++)
      {
        ae_int16x4 d_inp;
        ae_int32x2 d_res0, d_res1;
        d_inp = AE_MOVDA16(ptmp_inp[j*io_depth]);
        AE_MUL16X4(d_res0, d_res1, d_inp, d_alpha);
        d_res0 = AE_ADD32S(d_res0, d_beta);
        d_res0 = AE_SRAA32RS(d_res0, -out_shift);
        d_res0 = AE_MIN32(AE_MAX32(d_res0, d_min), d_max);
        *ptmp_out = (WORD8)AE_MOVAD32_L(d_res0);
        ptmp_out += io_depth;
      }
    }
  }

  return 0;
}

