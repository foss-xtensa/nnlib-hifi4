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

#ifndef __XA_NN_CONV2D_DEPTHWISE_STATE_H__
#define __XA_NN_CONV2D_DEPTHWISE_STATE_H__

#include "xa_nn_circ_buf.h"

typedef struct _xa_nn_conv2d_dw_state_t
{
    xa_nn_circ_buf_t circ_buf;
    pVOID p_scratch;
} xa_nn_conv2d_dw_state_t;

VOID xa_nn_conv2d_depthwise_init
(pVOID p_scratch
 ,WORD32 input_height
 ,WORD32 input_width
 ,WORD32 input_channels
 ,WORD32 kernel_height
 ,WORD32 kernel_width
 ,WORD32 channels_multiplier
 ,WORD32 x_stride
 ,WORD32 y_stride
 ,WORD32 x_padding
 ,WORD32 y_padding
 ,WORD32 output_height
 ,WORD32 output_width
 ,WORD32 circ_buf_precision
 ,WORD32 inp_data_format
 );

#define COPY_KERNEL_TO_SCRATCH_8b(p_out, p_in, kh, kw, kw_pad) \
{ \
  int itr_kh, itr_kw; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    UWORD8 *pae_in = (UWORD8 *)(&p_in[itr_kh * kw]); \
    ae_int16 *pae_out = (ae_int16 *)(&p_out[itr_kh * kw_pad]); \
    ae_int16x4 d_tmp, d_tmp1; \
_Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < (kw >> 1); itr_kw++) \
    { \
      WORD16 a1, a2; \
      a1 = *pae_in++; \
      a2 = *pae_in++; \
      d_tmp = AE_MOVDA16(a1); \
      d_tmp1 = AE_MOVDA16(a2); \
      d_tmp1 = AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(AE_MOVINT32X2_FROMINT16X4(d_tmp1), 8)); \
      d_tmp = AE_OR16(d_tmp, d_tmp1); \
      *pae_out++ = d_tmp; \
    } \
    if(kw&1) \
    { \
      *(UWORD8 *)pae_out = *pae_in; \
    } \
  } \
}

#define COPY_KERNEL_TO_SCRATCH_16b(p_out, p_in, kh, kw, kw_pad) \
{ \
  int itr_kh, itr_kw; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    ae_int16x4 *pae_in = (ae_int16x4 *)(&p_in[itr_kh * kw]); \
    ae_int16x4 *pae_out = (ae_int16x4 *)(&p_out[itr_kh * kw_pad]); \
    ae_int16x4 d_tmp; \
    ae_valign in_a = AE_LA64_PP(pae_in); \
_Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < (kw >> 2); itr_kw++) \
    { \
      AE_LA16X4_IP(d_tmp, in_a, pae_in); \
      AE_S16X4_IP(d_tmp, pae_out, 8); \
    } \
    if(kw & 3) \
    { \
      AE_LA16X4_IP(d_tmp, in_a, pae_in); \
      ae_int64 d_tmp64 = AE_MOVINT64_FROMINT16X4(d_tmp); \
      d_tmp64 = AE_SRAA64(d_tmp64, 16 * (4 - (kw & 3))); \
      d_tmp64 = AE_SLAA64(d_tmp64, 16 * (4 - (kw & 3))); \
      d_tmp = AE_MOVINT16X4_FROMINT64(d_tmp64); \
      AE_S16X4_IP(d_tmp, pae_out, 8); \
    } \
  } \
}

#define COPY_KERNEL_TO_SCRATCH_F32(p_out, p_in, kh, kw, kw_pad) \
{ \
  int itr_kh, itr_kw; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    xtfloatx2 *pae_in = (xtfloatx2 *)(&p_in[itr_kh * kw]); \
    xtfloatx2 *pae_out = (xtfloatx2 *)(&p_out[itr_kh * kw_pad]); \
    xtfloatx2 d_tmp; \
    ae_valign in_a = XT_LASX2PP(pae_in); \
_Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < (kw >> 1); itr_kw++) \
    { \
      XT_LASX2IP(d_tmp, in_a, pae_in); \
      XT_SSX2IP(d_tmp, pae_out, 8); \
    } \
    if(kw & 1) \
    { \
      *(xtfloat *)pae_out = *(xtfloat *)pae_in; \
    } \
  } \
}

#define COPY_KERNEL_TO_SCRATCH_NHWC_4_8b(p_out, p_in, kh, kw, kc) \
{ \
  int itr_kh, itr_kw; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    const UWORD8 *pae_in; \
    ae_int16 *pae_out; \
    ae_int16x4 d_tmp, d_tmp1; \
_Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < kw; itr_kw++) \
    { \
      pae_in = (const UWORD8 *)(&p_in[(itr_kh * kw + itr_kw) * kc]); \
      pae_out = (ae_int16 *)(&p_out[(itr_kh * kw + itr_kw) * 4]); \
      WORD16 a1, a2; \
      a1 = *pae_in++; \
      a2 = *pae_in++; \
      d_tmp = AE_MOVDA16(a1); \
      d_tmp1 = AE_MOVDA16(a2); \
      d_tmp1 = AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(AE_MOVINT32X2_FROMINT16X4(d_tmp1), 8)); \
      d_tmp = AE_OR16(d_tmp, d_tmp1); \
      *pae_out++ = d_tmp; \
      a1 = *pae_in++; \
      a2 = *pae_in++; \
      d_tmp = AE_MOVDA16(a1); \
      d_tmp1 = AE_MOVDA16(a2); \
      d_tmp1 = AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(AE_MOVINT32X2_FROMINT16X4(d_tmp1), 8)); \
      d_tmp = AE_OR16(d_tmp, d_tmp1); \
      *pae_out++ = d_tmp; \
    } \
  } \
}

#endif /* #ifndef __XA_NN_CONV2D_DEPTHWISE_STATE_H__ */
