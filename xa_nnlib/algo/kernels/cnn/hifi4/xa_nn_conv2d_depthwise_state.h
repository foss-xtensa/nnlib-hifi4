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

#ifndef __XA_NN_CONV2D_DEPTHWISE_STATE_H__
#define __XA_NN_CONV2D_DEPTHWISE_STATE_H__

#include "xa_nn_circ_buf.h"
#include "xa_nnlib_common.h"

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
 ,pVOID p_pad_val
 );

VOID xa_nn_dilated_conv2d_depthwise_init
(pVOID p_scratch
 ,WORD32 input_height
 ,WORD32 input_width
 ,WORD32 input_channels
 ,WORD32 kernel_height
 ,WORD32 kernel_width
 ,WORD32 channels_multiplier
 ,WORD32 dilation_height
 ,WORD32 dilation_width
 ,WORD32 x_stride
 ,WORD32 y_stride
 ,WORD32 x_padding
 ,WORD32 y_padding
 ,WORD32 output_height
 ,WORD32 output_width
 ,WORD32 circ_buf_precision
 ,WORD32 inp_data_format
 ,pVOID p_pad_val
 );

#if XCHAL_HAVE_HIFI1 
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
#define COPY_KERNEL_TO_SCRATCH_8b(p_out, p_in, kh, kw, kw_pad) \
{ \
  int itr_kh, itr_kw; \
  ae_valign alignIn, alignOut; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    WORD8 *pae_in = (WORD8 *)(&p_in[itr_kh * kw]); \
    WORD8 *pae_out = (WORD8 *)(&p_out[itr_kh * kw_pad]); \
    alignIn = AE_LA64_PP(pae_in); \
    alignOut = AE_ZALIGN64(); \
    ae_int16x4 d_tmp; \
    _Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < (kw >> 2); itr_kw++) \
    { \
      AE_LA8X4S_IP(d_tmp, alignIn, pae_in); \
      AE_SA8X4U_IP(d_tmp, alignOut, (ae_int32 *)pae_out); \
    } \
  int rem_itr = (kw & 0x3);\
  if(rem_itr)\
  {\
      AE_LAV8X4S_XP(d_tmp, alignIn, (ae_int8x4 *)pae_in, rem_itr); \
      AE_SAV8X4U_XP(d_tmp, alignOut, (ae_int8x4u *)pae_out, rem_itr); \
  }\
    AE_SA64POS_FP(alignOut, pae_out); \
  }\
}\

#else
#define COPY_KERNEL_TO_SCRATCH_8b(p_out, p_in, kh, kw, kw_pad) \
{ \
  int itr_kh, itr_kw; \
  ae_valign alignIn, alignOut; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    WORD8 *pae_in = (WORD8 *)(&p_in[itr_kh * kw]); \
    WORD8 *pae_out = (WORD8 *)(&p_out[itr_kh * kw_pad]); \
    alignIn = AE_LA64_PP(pae_in); \
    alignOut = AE_ZALIGN64(); \
    ae_int16x4 d_tmp; \
    _Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < (kw >> 2); itr_kw++) \
    { \
      AE_LA8X4S_IP(d_tmp, alignIn, pae_in); \
      AE_SA8X4U_IP(d_tmp, alignOut, (ae_int32 *)pae_out); \
    } \
    AE_SA64POS_FP(alignOut, pae_out); \
    for(itr_kw=0; itr_kw < (kw & 0x3); itr_kw++){ \
      AE_L8S_IP(d_tmp, pae_in, 1); \
      AE_S8_0_IP_HIFI1(d_tmp, pae_out, 1);\
    } \
  } \
}
#endif
#else
#define COPY_KERNEL_TO_SCRATCH_8b(p_out, p_in, kh, kw, kw_pad) \
{ \
  int itr_kh, itr_kw; \
  ae_valign alignIn, alignOut; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    ae_int24x2 *pae_in = (ae_int24x2 *)(&p_in[itr_kh * kw]); \
    ae_int24x2 *pae_out = (ae_int24x2 *)(&p_out[itr_kh * kw_pad]); \
    alignIn = AE_LA64_PP(pae_in); \
    alignOut = AE_ZALIGN64(); \
    ae_int24x2 d_tmp; \
    int kw_by_6 =  AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(kw, 0x2AAAAAAB))); \
    int remainder_start = 6*kw_by_6; \
    _Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < kw_by_6; itr_kw++) \
    { \
      AE_LA24X2_IP(d_tmp, alignIn, pae_in); \
      AE_SA24X2_IP(d_tmp, alignOut, pae_out); \
    } \
    AE_SA64POS_FP(alignOut, pae_out); \
    for(itr_kw=remainder_start; itr_kw < kw; itr_kw++){ \
      ((WORD8 *)p_out)[(itr_kh * kw_pad) + itr_kw] = ((WORD8 *)p_in)[(itr_kh * kw) + itr_kw]; \
    } \
  } \
}
#endif

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

#if XCHAL_HAVE_HIFI1
#define COPY_KERNEL_TO_SCRATCH_NHWC_4_8b(p_out, p_in, kh, kw, kc) \
{ \
  int itr_kh, itr_kw; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    const WORD8 *pae_in; \
    WORD8 *pae_out; \
    ae_int16x4 d_tmp; \
    ae_valign align_out, align_in; \
    _Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < kw; itr_kw++) \
    { \
      pae_in = (const WORD8 *)(&p_in[(itr_kh * kw + itr_kw) * kc]); \
      pae_out = (WORD8 *)(&p_out[(itr_kh * kw + itr_kw) * 4]); \
      align_in = AE_LA64_PP(pae_in); \
      AE_LA8X4U_IP(d_tmp, align_in, pae_in); \
      AE_SA8X4U_IP(d_tmp, align_out, (ae_int32 *)pae_out); \
      AE_SA64POS_FP(align_out, pae_out); \
    } \
  } \
}
#else
#define COPY_KERNEL_TO_SCRATCH_NHWC_4_8b(p_out, p_in, kh, kw, kc) \
{ \
  int itr_kernel_height, itr_kernel_width; \
  for(itr_kernel_height = 0; itr_kernel_height < kh; itr_kernel_height++) \
  { \
    const UWORD8 *pae_in; \
    ae_int16 *pae_out; \
    ae_int16x4 d_tmp, d_tmp1; \
_Pragma("no_unroll") \
    for(itr_kernel_width = 0; itr_kernel_width < kw; itr_kernel_width++) \
    { \
      pae_in = (const UWORD8 *)(&p_in[(itr_kernel_height * kw + itr_kernel_width) * kc]); \
      pae_out = (ae_int16 *)(&p_out[(itr_kernel_height * kw + itr_kernel_width) * 4]); \
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
#endif

#endif /* #ifndef __XA_NN_CONV2D_DEPTHWISE_STATE_H__ */
