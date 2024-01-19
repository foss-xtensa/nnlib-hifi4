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

#ifndef __XA_NN_AVGPOOL_STATE_H__
#define __XA_NN_AVGPOOL_STATE_H__

#define ALIGNMENT   8   /* 8 bytes alignment */

#define ALIGNED_SIZE(x, bytes)  (((x)+(bytes-1))&(~(bytes-1)))
#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#define LIMIT(input, min, max) \
    input = XT_MAX(min, XT_MIN(max, input));

#define MAX_HEIGHT_16_BIT_ACC 127

#define MAX_16X4(out, id2, id1, id0) {\
        out = id1;\
        b0 = AE_LT16(id1, id0); \
        AE_MOVT16X4(out, id0, b0);\
        b0 = AE_LT16(out, id2); \
        AE_MOVT16X4(out, id2, b0);\
}

extern const unsigned int inv_256_tbl[257];
typedef struct _xa_nn_avgpool_state_t
{
    pWORD32 p_den_height;
    pWORD32 p_den_width;
    pVOID p_tmp_out;
} xa_nn_avgpool_state_t;

VOID xa_nn_avgpool_init(
    WORD32 inp_precision,
    pVOID  p_scratch,
    WORD32 out_height,
    WORD32 out_width);

void xa_nn_avgpool_f32_hwc(
      FLOAT32* __restrict__ p_out,
const FLOAT32* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in,
      FLOAT32  *p_zeros_mem,
      FLOAT32  *p_den);

void xa_nn_avgpool_asym8_hwc_16(
      UWORD8* __restrict__ p_out,
const UWORD8* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in,
      pVOID    p_zeros_mem,
      WORD32   *p_den_height,
      WORD32   *p_den_width);

void xa_nn_avgpool_asym8_hwc_32(
      UWORD8* __restrict__ p_out,
const UWORD8* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in,
      pVOID    p_zeros_mem,
      WORD32   *p_den_height,
      WORD32   *p_den_width);

void xa_nn_avgpool_8_hwc_16(
      WORD8* __restrict__ p_out,
const WORD8* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in,
      pVOID    p_zeros_mem,
      WORD32   *p_den_height,
      WORD32   *p_den_width);

void xa_nn_avgpool_8_hwc_32(
      WORD8* __restrict__ p_out,
const WORD8* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in,
      pVOID    p_zeros_mem,
      WORD32   *p_den_height,
      WORD32   *p_den_width);

void xa_nn_avgpool_16_hwc_32(
      WORD16* __restrict__ p_out,
const WORD16* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in,
      pVOID    p_zeros_mem,
      WORD32   *p_den_height,
      WORD32   *p_den_width);

#endif /* #ifndef __XA_NN_AVGPOOL_STATE_H__ */
