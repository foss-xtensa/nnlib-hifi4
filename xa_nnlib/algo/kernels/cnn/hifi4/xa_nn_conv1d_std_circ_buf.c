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
#include "xa_nn_conv1d_std_state.h"
#include <string.h>

WORD32 xa_nn_conv1d_std_getsize(
    WORD32 kernel_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 input_precision)
{
  XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
  XA_NNLIB_CHK_COND((input_width <= 0), -1);
  XA_NNLIB_CHK_COND((input_channels <= 0), -1);

  WORD32 mem_req = 0;
  WORD32 input_size;
  WORD32 align_size;

  mem_req += ALIGNED_SIZE(sizeof(xa_nn_conv_state_t), ALIGNMENT);
  switch(input_precision)
  {
    case 8:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    case 16:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case -1:
      input_size = sizeof(WORD32);
      align_size = ALIGNMENT>>2;
      break;
    case -3:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    default:
      return -1;
      break;
  }

  // Computing circular buffer size
  WORD32 input_channelsXwidth_pad = PADDED_SIZE(input_channels*input_width, align_size);
  WORD32 cir_buf_size_bytes = kernel_height * input_channelsXwidth_pad * input_size;

  /* scratch memory for convolution using matrix multiplication */
  mem_req += cir_buf_size_bytes;
  mem_req += BUS_WIDTH;

  return mem_req;
}

VOID xa_nn_conv1d_std_init_state(
    VOID *p_handle,
    VOID *p_kernel,
    WORD32 kernel_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 y_stride,
    WORD32 input_precision)
{
  WORD8 *p_mem = (WORD8 *)p_handle;
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_mem;
  size_t input_size;
  UWORD32 align_size;

  switch(input_precision)
  {
    case 8:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    case 16:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case 32:
    case -1:
      input_size = sizeof(WORD32);
      align_size = ALIGNMENT>>2;
      break;
    case -3:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    default:
      input_size = sizeof(WORD64);
      align_size = ALIGNMENT>>3;
      break;
  }

  p_mem += sizeof(xa_nn_conv_state_t);
  p_mem = ALIGNED_ADDR(p_mem, ALIGNMENT);


  if(((UWORD32)p_kernel & BUS_WIDTH_MASK) == ((UWORD32)p_mem & BUS_WIDTH_MASK))
  {
    p_mem += BUS_WIDTH; /* Add a offset to avoid banking stall */
  }

  p_state->cir_buf.p_begin = p_mem;
  p_state->cir_buf.p_curr = p_mem;

  // Computing circular buffer size
  WORD32 input_channelsXwidth_pad = PADDED_SIZE(input_channels*input_width, align_size);
  WORD32 cir_buf_size_bytes = kernel_height * input_channelsXwidth_pad * input_size;

  p_mem += cir_buf_size_bytes;
  p_state->cir_buf.p_end = p_mem;

  AE_SETCBEGIN0(p_state->cir_buf.p_begin);
  AE_SETCEND0(p_state->cir_buf.p_end);
  AE_ADDCIRC16X4_XC(p_state->cir_buf.p_curr, -y_stride * input_channelsXwidth_pad * input_size);

}

// Init (kernel_height - y_stride) x input_channelsXwidth_pad planes in circular buffer
VOID conv1d_std_init_cir_buf(
    WORD32 input_channels,
    WORD32 input_channelsXwidth_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 y_stride,
    WORD32 y_padding,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state)
{
  WORD32 k;
  VOID *p_inp = *pp_inp;
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, y_stride * input_channelsXwidth_pad * input_bytewidth);

  // Initialize circular buffer
  // Set y_padding rows of cir_buf with zero and remaining rows with input data
  WORD32 copy_y_pad_height = y_padding;
  WORD32 copy_inp_height = 0;
  if((kernel_height - y_stride) <= y_padding)
  {
    copy_y_pad_height = kernel_height - y_stride;
  }
  else
  {
    copy_inp_height = kernel_height - y_stride - y_padding;
  }
  for(k=0;k<copy_y_pad_height;k++)
  {
    memset(p_dst, 0, input_channelsXwidth_pad * input_bytewidth);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, input_channelsXwidth_pad * input_bytewidth);
  }
  for(k=0;k<copy_inp_height;k++)
  {
    xa_nn_memcpy(p_dst, p_inp, input_channels * input_width * input_bytewidth);
    memset(&p_dst[input_channels * input_width * input_bytewidth], 0, (input_channelsXwidth_pad - input_channels * input_width) * input_bytewidth);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, input_channelsXwidth_pad * input_bytewidth);
    p_inp += input_channels * input_width * input_bytewidth;
  }

  *pp_inp = p_inp;
}

// Add y_stride x input_channelsXwidth_pad new planes to circular buffer
VOID conv1d_std_update_cir_buf(
    WORD32 input_channels,
    WORD32 input_channelsXwidth_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 kernel_height,
    WORD32 y_stride,
    WORD32 y_padding,
    VOID **pp_inp,
    WORD32 idx_beg_inp_height_pad,
    xa_nn_conv_state_t *p_state)
{
  WORD32 k;
  VOID *p_inp = *pp_inp;

  // Copy 'y_stride' planes of data to circular buffer
  AE_ADDCIRC16X4_XC(p_state->cir_buf.p_curr, y_stride * input_channelsXwidth_pad * input_bytewidth);
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, (kernel_height - y_stride) * input_channelsXwidth_pad * input_bytewidth);

  // Set 'y_stride' rows of cir_buf with zero (from y_padding) and/or input data and/or zero (from y-bottom padding)
  WORD32 idx_end_inp_height_pad = idx_beg_inp_height_pad + y_stride;
  WORD32 copy_y_pad_height = 0;
  WORD32 copy_inp_height = 0;
  WORD32 copy_y_b_pad_height = 0;
  if(idx_beg_inp_height_pad < y_padding)
  {
    copy_y_pad_height = y_padding - idx_beg_inp_height_pad;
    copy_inp_height = idx_end_inp_height_pad - y_padding;
  }
  else if(idx_end_inp_height_pad <= y_padding + input_height)
  {
    copy_inp_height = y_stride;
  }
  else if(idx_beg_inp_height_pad < y_padding + input_height)
  {
    copy_inp_height = y_padding + input_height - idx_beg_inp_height_pad;
    copy_y_b_pad_height = idx_end_inp_height_pad - (y_padding + input_height);
  }
  else
  {
    copy_y_b_pad_height = y_stride;
  }

  for(k=0;k<copy_y_pad_height;k++)
  {
    memset(p_dst, 0, input_channelsXwidth_pad * input_bytewidth);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, input_channelsXwidth_pad * input_bytewidth);
  }
  for(k=0;k<copy_inp_height;k++)
  {
    xa_nn_memcpy(p_dst, p_inp, input_channels * input_width * input_bytewidth);
    memset(&p_dst[input_channels * input_width * input_bytewidth], 0, (input_channelsXwidth_pad - input_channels * input_width) * input_bytewidth);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, input_channelsXwidth_pad * input_bytewidth);
    p_inp += input_channels * input_width * input_bytewidth;
  }
  for(k=0;k<copy_y_b_pad_height;k++)
  {
    memset(p_dst, 0, input_channelsXwidth_pad * input_bytewidth);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, input_channelsXwidth_pad * input_bytewidth);
  }

  *pp_inp = p_inp;
}

// Init (kernel_height - y_stride) x input_channelsXwidth_pad planes in circular buffer
VOID conv1d_std_init_cir_buf_asym8(
    WORD32 input_channels,
    WORD32 input_channelsXwidth_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 y_stride,
    WORD32 y_padding,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val)
{
  WORD32 k;
  VOID *p_inp = *pp_inp;
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  UWORD8 pad_val_u8 = (UWORD8)pad_val;
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, y_stride * input_channelsXwidth_pad * input_bytewidth);

  // Initialize circular buffer
  // Set y_padding rows of cir_buf with zero and remaining rows with input data
  WORD32 copy_y_pad_height = y_padding;
  WORD32 copy_inp_height = 0;
  if((kernel_height - y_stride) <= y_padding)
  {
    copy_y_pad_height = kernel_height - y_stride;
  }
  else
  {
    copy_inp_height = kernel_height - y_stride - y_padding;
  }
  for(k=0;k<copy_y_pad_height;k++)
  {
    memset(p_dst, pad_val_u8, input_channelsXwidth_pad * input_bytewidth);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, input_channelsXwidth_pad * input_bytewidth);
  }
  for(k=0;k<copy_inp_height;k++)
  {
    xa_nn_memcpy(p_dst, p_inp, input_channels * input_width * input_bytewidth);
    memset(&p_dst[input_channels * input_width * input_bytewidth], pad_val_u8, (input_channelsXwidth_pad - input_channels * input_width) * input_bytewidth);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, input_channelsXwidth_pad * input_bytewidth);
    p_inp += input_channels * input_width * input_bytewidth;
  }

  *pp_inp = p_inp;
}

// Add y_stride x input_channelsXwidth_pad new planes to circular buffer
VOID conv1d_std_update_cir_buf_asym8(
    WORD32 input_channels,
    WORD32 input_channelsXwidth_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 kernel_height,
    WORD32 y_stride,
    WORD32 y_padding,
    VOID **pp_inp,
    WORD32 idx_beg_inp_height_pad,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val)
{
  WORD32 k;
  VOID *p_inp = *pp_inp;
  UWORD8 pad_val_u8 = (UWORD8)pad_val;

  // Copy 'y_stride' planes of data to circular buffer
  AE_ADDCIRC16X4_XC(p_state->cir_buf.p_curr, y_stride * input_channelsXwidth_pad * input_bytewidth);
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, (kernel_height - y_stride) * input_channelsXwidth_pad * input_bytewidth);

  // Set 'y_stride' rows of cir_buf with zero (from y_padding) and/or input data and/or zero (from y-bottom padding)
  WORD32 idx_end_inp_height_pad = idx_beg_inp_height_pad + y_stride;
  WORD32 copy_y_pad_height = 0;
  WORD32 copy_inp_height = 0;
  WORD32 copy_y_b_pad_height = 0;
  if(idx_beg_inp_height_pad < y_padding)
  {
    copy_y_pad_height = y_padding - idx_beg_inp_height_pad;
    copy_inp_height = idx_end_inp_height_pad - y_padding;
  }
  else if(idx_end_inp_height_pad <= y_padding + input_height)
  {
    copy_inp_height = y_stride;
  }
  else if(idx_beg_inp_height_pad < y_padding + input_height)
  {
    copy_inp_height = y_padding + input_height - idx_beg_inp_height_pad;
    copy_y_b_pad_height = idx_end_inp_height_pad - (y_padding + input_height);
  }
  else
  {
    copy_y_b_pad_height = y_stride;
  }

  for(k=0;k<copy_y_pad_height;k++)
  {
    memset(p_dst, pad_val_u8, input_channelsXwidth_pad * input_bytewidth);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, input_channelsXwidth_pad * input_bytewidth);
  }
  for(k=0;k<copy_inp_height;k++)
  {
    xa_nn_memcpy(p_dst, p_inp, input_channels * input_width * input_bytewidth);
    memset(&p_dst[input_channels * input_width * input_bytewidth], pad_val_u8, (input_channelsXwidth_pad - input_channels * input_width) * input_bytewidth);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, input_channelsXwidth_pad * input_bytewidth);
    p_inp += input_channels * input_width * input_bytewidth;
  }
  for(k=0;k<copy_y_b_pad_height;k++)
  {
    memset(p_dst, pad_val_u8, input_channelsXwidth_pad * input_bytewidth);
    AE_ADDCIRC16X4_XC((ae_int16x4*)p_dst, input_channelsXwidth_pad * input_bytewidth);
  }

  *pp_inp = p_inp;
}
