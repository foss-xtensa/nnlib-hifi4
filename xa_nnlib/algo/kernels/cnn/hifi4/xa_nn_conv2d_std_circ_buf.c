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
#include "xa_nn_conv2d_std_state.h"
#include <string.h>

WORD32 xa_nn_conv2d_std_getsize(
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 kernel_channels,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 x_stride,
    WORD32 x_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 output_channels,
    WORD32 input_precision,
    WORD32 kernel_precision,
    WORD32 dilation_height,
    WORD32 dilation_width,
    WORD32 out_data_format)
{
  XA_NNLIB_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_channels <= 0), -1);
  XA_NNLIB_CHK_COND((y_stride <= 0), -1);
  XA_NNLIB_CHK_COND((y_padding < 0), -1);
  XA_NNLIB_CHK_COND((x_stride <= 0), -1);
  XA_NNLIB_CHK_COND((x_padding < 0), -1);
  XA_NNLIB_CHK_COND((out_height <= 0), -1);
  XA_NNLIB_CHK_COND((out_width <= 0), -1);
  XA_NNLIB_CHK_COND((output_channels <= 0), -1);

  const int groups = input_channels/kernel_channels;
  XA_NNLIB_ARG_CHK_COND((groups<=0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_channels %kernel_channels)!=0),-1);
  XA_NNLIB_ARG_CHK_COND(((output_channels%groups)!=0),-1);
  const int filters_per_group = output_channels / groups;
  XA_NNLIB_ARG_CHK_COND((filters_per_group<=0),-1);


  /* Unused variables kept for future use */
  (void)kernel_precision;
  (void)dilation_height;
  (void)dilation_width;
  (void)out_data_format;

  WORD32 mem_req = 0;
  WORD32 input_size;
  WORD32 align_size;
  WORD32 inp_h, inp_w, ker_h, ker_w, x_str, y_str, x_pad, y_pad, out_h, out_w;

  if ((input_height == 1) && (kernel_height == 1) && (out_height == 1))
  {
    inp_h = input_width;
    inp_w = input_height;
    ker_h = kernel_width;
    ker_w = kernel_height;
    x_str = y_stride;
    y_str = x_stride;
    x_pad = y_padding;
    y_pad = x_padding;
    out_h = out_width;
    out_w = out_height;
  }
  else
  {
    inp_h = input_height;
    inp_w = input_width;
    ker_h = kernel_height;
    ker_w = kernel_width;
    x_str = x_stride;
    y_str = y_stride;
    x_pad = x_padding;
    y_pad = y_padding;
    out_h = out_height;
    out_w = out_width;
  }

  /* To remove LLVM15 warnings */
  (void)out_w; (void)x_str; (void)x_pad; (void)inp_w;

  mem_req += ALIGNMENT;
  mem_req += ALIGNED_SIZE(sizeof(xa_nn_conv_state_t), ALIGNMENT);
  /* Input precision is checked here */
  switch(input_precision)
  {
    case 8:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    case -7:
    case 16:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case -8:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case -1:
      input_size = sizeof(WORD32);
      align_size = ALIGNMENT>>2;
      break;
    case -3:
      input_size = sizeof(UWORD8);
      align_size = ALIGNMENT>>1;
      break;
    case -4:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    default:
      return -1;
      break;
  }

  // Computing circular buffer size
  // Determine y-bottom padding
  WORD32 y_b_pad = ker_h + (out_h - 1) * y_str - (y_pad + inp_h);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  WORD32 kernel_channels_pad;

#if !ENABLE_PADDING_CONV2D_STD
  if(input_precision == PREC_ASYM8S)
  {
    kernel_channels_pad = kernel_channels;
  }
  else
#endif
  {
    kernel_channels_pad = PADDED_SIZE(kernel_channels, align_size);
  }

  WORD32 total_input_height = y_pad + inp_h + y_b_pad;
  WORD32 cir_buf_size_bytes = total_input_height * ker_w * kernel_channels_pad * input_size;
  while(cir_buf_size_bytes%16 !=0)
  {
    cir_buf_size_bytes+= ker_w*kernel_channels_pad*input_size;
  }

  /* scratch memory for convolution using matrix multiplication */
  mem_req += ALIGNED_SIZE(cir_buf_size_bytes, ALIGNMENT);
  mem_req += BUS_WIDTH;

#if !ENABLE_PADDING_CONV2D_STD
  if(
      (input_precision != PREC_ASYM8S) &&
      (input_precision != PREC_F32) &&
      (input_precision != PREC_16) &&
      (kernel_channels_pad != kernel_channels)
    )
#else
  if(
      (input_precision != PREC_16) &&
      (kernel_channels_pad != kernel_channels)
    )
#endif
  {
    int padded_kernel_size = kernel_height * kernel_width * kernel_channels_pad * output_channels * input_size;
    mem_req += ALIGNED_SIZE(padded_kernel_size, ALIGNMENT);
  }

  return mem_req;
}


WORD32 xa_nn_conv2d_getsize(
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 kernel_channels,
    WORD32 dilation_height,
    WORD32 dilation_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 x_stride,
    WORD32 x_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 output_channels,
    WORD32 input_precision,
    WORD32 kernel_precision,
    WORD32 out_data_format)
{
  XA_NNLIB_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_channels <= 0), -1);
  XA_NNLIB_CHK_COND((y_stride <= 0), -1);
  XA_NNLIB_CHK_COND((y_padding < 0), -1);
  XA_NNLIB_CHK_COND((x_stride <= 0), -1);
  XA_NNLIB_CHK_COND((x_padding < 0), -1);
  XA_NNLIB_CHK_COND((out_height <= 0), -1);
  XA_NNLIB_CHK_COND((out_width <= 0), -1);
  XA_NNLIB_CHK_COND((output_channels <= 0), -1);

  const int groups = input_channels/kernel_channels;
  XA_NNLIB_ARG_CHK_COND((groups<=0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_channels %kernel_channels)!=0),-1);
  XA_NNLIB_ARG_CHK_COND(((output_channels%groups)!=0),-1);
  const int filters_per_group = output_channels / groups;
  XA_NNLIB_ARG_CHK_COND((filters_per_group<=0),-1);


  /* Unused variables kept for future use */
  (void)kernel_precision;
  (void)dilation_height;
  (void)dilation_width;

  WORD32 mem_req = 0;
  WORD32 input_size;
  WORD32 align_size;
  WORD32 inp_h, inp_w, ker_h, ker_w, x_str, y_str, x_pad, y_pad, out_h, out_w;

  if ((input_height == 1) && (kernel_height == 1) && (out_height == 1))
  {
    inp_h = input_width;
    inp_w = input_height;
    ker_h = kernel_width;
    ker_w = kernel_height;
    x_str = y_stride;
    y_str = x_stride;
    x_pad = y_padding;
    y_pad = x_padding;
    out_h = out_width;
    out_w = out_height;
  }
  else
  {
    inp_h = input_height;
    inp_w = input_width;
    ker_h = kernel_height;
    ker_w = kernel_width;
    x_str = x_stride;
    y_str = y_stride;
    x_pad = x_padding;
    y_pad = y_padding;
    out_h = out_height;
    out_w = out_width;
  }

  if((groups==input_channels) && (output_channels >=input_channels) && (input_precision==-4) && (out_data_format==0))
  {   
    WORD32 channel_multiplier=output_channels/input_channels;
    mem_req += ALIGNMENT;
    mem_req += PADDED_SIZE(output_channels*kernel_height*kernel_width, ALIGNMENT);
    mem_req += xa_nn_conv2d_depthwise_getsize(inp_h
                                              ,inp_w
                                              ,input_channels
                                              ,ker_h
                                              ,ker_w
                                              ,channel_multiplier
                                              ,x_str
                                              ,y_str
                                              ,x_pad
                                              ,y_pad
                                              ,out_h
                                              ,out_w
                                              ,input_precision
                                              ,0
                                              );

  }
  else
  {

  mem_req += ALIGNMENT;
  mem_req += ALIGNED_SIZE(sizeof(xa_nn_conv_state_t), ALIGNMENT);
  /* Input precision is checked here */
  switch(input_precision)
  {
    case 8:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    case -7:
    case 16:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case -8:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case -1:
      input_size = sizeof(WORD32);
      align_size = ALIGNMENT>>2;
      break;
    case -3:
      input_size = sizeof(UWORD8);
      align_size = ALIGNMENT>>1;
      break;
    case -4:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    default:
      return -1;
      break;
  }

  // Computing circular buffer size
  // Determine y-bottom padding
  WORD32 y_b_pad = ker_h + (out_h - 1) * y_str - (y_pad + inp_h);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  WORD32 kernel_channels_pad;

#if !ENABLE_PADDING_CONV2D_STD
  if(input_precision == PREC_ASYM8S)
  {
    kernel_channels_pad = kernel_channels;
  }
  else
#endif
  {
    kernel_channels_pad = PADDED_SIZE(kernel_channels, align_size);
  }

  WORD32 total_input_height = y_pad + inp_h + y_b_pad;
  WORD32 cir_buf_size_bytes = total_input_height * ker_w * kernel_channels_pad * input_size;
  while(cir_buf_size_bytes%16 !=0)
  {
    cir_buf_size_bytes+= ker_w*kernel_channels_pad*input_size;
  }

  /* scratch memory for convolution using matrix multiplication */
  mem_req += ALIGNED_SIZE(cir_buf_size_bytes, ALIGNMENT);
  mem_req += BUS_WIDTH;

#if !ENABLE_PADDING_CONV2D_STD
  if(
      (input_precision != PREC_ASYM8S) &&
      (input_precision != PREC_F32) &&
      (input_precision != PREC_16) &&
      (kernel_channels_pad != kernel_channels)
    )
#else
  if(
      (input_precision != PREC_16) &&
      (kernel_channels_pad != kernel_channels)
    )
#endif
  {
    int padded_kernel_size = kernel_height * kernel_width * kernel_channels_pad * output_channels * input_size;
    mem_req += ALIGNED_SIZE(padded_kernel_size, ALIGNMENT);
  }

  }
  return mem_req;
}

WORD32 xa_nn_conv2d_std_getsize_sym4s(
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 output_channels,
    WORD32 input_precision)
{
  return -1;
}

WORD32 xa_nn_dilated_conv2d_std_getsize(
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 output_channels,
    WORD32 input_precision,
    WORD32 dilation_height)
{
  XA_NNLIB_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
  XA_NNLIB_CHK_COND((dilation_height <= 0), -1);
  XA_NNLIB_CHK_COND((y_stride <= 0), -1);
  XA_NNLIB_CHK_COND((y_padding < 0), -1);
  XA_NNLIB_CHK_COND((out_height <= 0), -1);
  if(kernel_height==1)
    dilation_height = 1;
  WORD32 kernel_height_dilation = kernel_height + ( (dilation_height-1) * (kernel_height-1) );//dilation
  //XA_NNLIB_CHK_COND((kernel_height_dilation > input_height), -1);

  //if( (dilation_height>1) ) // Dilation not supporting stride presently
    //  XA_NNLIB_CHK_COND((y_stride > 1), -1);

  WORD32 mem_req = 0;
  WORD32 input_size;
  WORD32 align_size;
  WORD32 input_channels_pad;

  mem_req += ALIGNED_SIZE(sizeof(xa_nn_conv_state_t), ALIGNMENT);
  /* Input precision is checked here */
  switch(input_precision)
  {
    case 8:
    case -4:
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
      input_size = sizeof(UWORD8);
      align_size = ALIGNMENT>>1;
      break;
    default:
      return -1;
      break;
  }

  // Computing circular buffer size
  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height_dilation + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;


  WORD32 input_height_pad = y_padding + input_height + y_b_pad;

  XA_NNLIB_CHK_COND((kernel_height_dilation > input_height_pad), -1);

#if !ENABLE_PADDING_CONV2D_STD
  if(input_precision == PREC_8 || input_precision == PREC_ASYM8U || input_precision == PREC_ASYM8S) //TODO: remove the condition when the padding requirement is removed for other variants.
    input_channels_pad = input_channels;
  else
#endif
    input_channels_pad = PADDED_SIZE(input_channels, align_size);

  WORD32 total_input_height = ((y_padding + input_height + y_b_pad)/dilation_height) + 1;
  WORD32 cir_buf_size_bytes = total_input_height * kernel_width * input_channels_pad * input_size;
  while(cir_buf_size_bytes%16 !=0)
  {
      cir_buf_size_bytes += kernel_width*input_channels_pad*input_size;
  }
  /* scratch memory for convolution using matrix multiplication */
  mem_req += ALIGNED_SIZE(cir_buf_size_bytes, ALIGNMENT);
  mem_req += BUS_WIDTH;

#if !ENABLE_PADDING_CONV2D_STD
  if( (input_precision != PREC_ASYM8S) &&
      (input_precision != PREC_F32) &&
      (input_precision != PREC_16) &&
      (input_channels_pad != input_channels) )
#else
  if( (input_precision != PREC_F32) &&
      (input_precision != PREC_16) &&
      (input_channels_pad != input_channels) )
#endif
  {
    int padded_kernel_size = kernel_height_dilation * kernel_width *
                             input_channels_pad * output_channels * input_size;

    mem_req += ALIGNED_SIZE(padded_kernel_size, ALIGNMENT);
  }

  return mem_req;
}

#ifndef ENABLE_SCRATCH_SIZE_API_ONLY
VOID xa_nn_conv2d_std_init_state(
    VOID *p_scratch,
    VOID *p_kernel,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 output_channels,
    WORD32 input_precision,
    WORD32 kernel_precision)
{
  WORD8 *p_mem = (WORD8 *)p_scratch;
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_mem;
  size_t input_size;
  UWORD32 align_size;

  switch(input_precision)
  {
    case 8:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    case -8:
    case 16:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case -1:
      input_size = sizeof(WORD32);
      align_size = ALIGNMENT>>2;
      break;
    case -3:
      input_size = sizeof(UWORD8);
      align_size = ALIGNMENT>>1;
      break;
    case -4:
    case -5:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    default:
      input_size = 0;
      align_size = 0;
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
  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  WORD32 input_channels_pad;

#if !ENABLE_PADDING_CONV2D_STD
  if(input_precision == PREC_ASYM8S)
  {
    input_channels_pad = input_channels;
  }
  else
#endif
  {
#if HW_AE_ADDCIRC16X4_XC 
    /* Disbale padding for ic=1 (worst case scenario for performance), if hardware support exists.
     * Enabled only for conv2d_std_sym8sxasym8s variant */
    if(input_channels == 1 && kernel_precision == PREC_SYM8S && input_precision == PREC_ASYM8S)
    {
      input_channels_pad = 1;
    }
    else
#endif
    {
      input_channels_pad = PADDED_SIZE(input_channels, align_size);
    }
  }

  WORD32 cir_buf_size_bytes = (y_padding + input_height + y_b_pad) * kernel_width * input_channels_pad * input_size;
  while(cir_buf_size_bytes%16 !=0)
  {
    cir_buf_size_bytes+= kernel_width*input_channels_pad*input_size;
  }

  p_mem += cir_buf_size_bytes;
  p_state->cir_buf.p_end = p_mem;

  AE_SETCBEGIN0(p_state->cir_buf.p_begin);
  AE_SETCEND0(p_state->cir_buf.p_end);

  p_mem = ALIGNED_ADDR(p_mem, ALIGNMENT);

  p_state->p_kernel_padded = (void *)p_kernel;

#if !ENABLE_PADDING_CONV2D_STD
  if(
      (input_precision != PREC_ASYM8S) &&
      (input_precision != PREC_F32) &&
      (input_precision != PREC_16) &&
      (input_channels_pad != input_channels)
    )
#else
  if(
      (input_precision != PREC_16) &&
      (input_channels_pad != input_channels)
    )
#endif
  {
    int oc, kh, kw, kernel_size;
    p_state->p_kernel_padded = (void *)p_mem;

    switch(kernel_precision)
    {
      case 8:
        kernel_size = sizeof(WORD8);
        break;
      case 16:
        kernel_size = sizeof(WORD16);
        break;
      case -1:
        kernel_size = sizeof(WORD32);
        break;
      case -3:
        kernel_size = sizeof(UWORD8);
        break;
      case -4:
      case -5:
        kernel_size = sizeof(WORD8);
        break;
      default:
        kernel_size = 0;
        break;
    }

    memset(p_mem, 0, output_channels*kernel_height*kernel_width*input_channels_pad*kernel_size);
    pWORD8 p_src = (pWORD8) p_kernel;
    pWORD8 p_dst = (pWORD8) p_state->p_kernel_padded;

    for(oc = 0; oc < output_channels; oc++)
    for(kh = 0; kh < kernel_height; kh++)
    {
      for(kw = 0; kw < kernel_width; kw++)
      {
//        memcpy(p_dst, p_src, kernel_size * input_channels);
        for(int ii=0; ii<kernel_size * input_channels; ii++){
          p_dst[ii] = p_src[ii];
        }
        p_dst += kernel_size * input_channels_pad;
        p_src += kernel_size * input_channels;
      }
    }
  }

}

VOID xa_nn_conv2d_group_init_state(
    VOID *p_scratch,
    VOID *p_kernel,
    WORD32 input_height,
    WORD32 kernel_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 output_channels,
    WORD32 input_precision,
    WORD32 kernel_precision)
{
  WORD8 *p_mem = (WORD8 *)p_scratch;
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_mem;
  size_t input_size;
  UWORD32 align_size;

  switch(input_precision)
  {
    case 8:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    case -8:
    case 16:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case -1:
      input_size = sizeof(WORD32);
      align_size = ALIGNMENT>>2;
      break;
    case -3:
      input_size = sizeof(UWORD8);
      align_size = ALIGNMENT>>1;
      break;
    case -4:
    case -5:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    default:
      input_size = 0;
      align_size = 0;
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
  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  WORD32 kernel_channels_pad;

#if !ENABLE_PADDING_CONV2D_STD
  if(input_precision == PREC_ASYM8S)
  {
    kernel_channels_pad = kernel_channels;
  }
  else
#endif
  {
#if HW_AE_ADDCIRC16X4_XC 
    /* Disbale padding for ic=1 (worst case scenario for performance), if hardware support exists.
     * Enabled only for conv2d_std_sym8sxasym8s variant */
    if(kernel_channels == 1 && kernel_precision == PREC_SYM8S && input_precision == PREC_ASYM8S)
    {
      kernel_channels_pad = 1;
    }
    else
#endif
    {
      kernel_channels_pad = PADDED_SIZE(kernel_channels, align_size);
    }
  }

  WORD32 cir_buf_size_bytes = (y_padding + input_height + y_b_pad) * kernel_width * kernel_channels_pad * input_size;
  while(cir_buf_size_bytes%16 !=0)
  {
    cir_buf_size_bytes+= kernel_width*kernel_channels_pad*input_size;
  }

  p_mem += cir_buf_size_bytes;
  p_state->cir_buf.p_end = p_mem;

  AE_SETCBEGIN0(p_state->cir_buf.p_begin);
  AE_SETCEND0(p_state->cir_buf.p_end);

}

VOID xa_nn_conv2d_dilation_init_state(
    VOID *p_scratch,
    VOID *p_kernel,
    VOID *p_input)
{
	WORD8 *p_mem = (WORD8 *)p_scratch;
	xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_mem;

	  p_mem += sizeof(xa_nn_conv_state_t);
	  p_mem = ALIGNED_ADDR(p_mem, ALIGNMENT);


	  if(((UWORD32)p_kernel & BUS_WIDTH_MASK) == ((UWORD32)p_mem & BUS_WIDTH_MASK))
	  {
	    p_mem += BUS_WIDTH; /* Add a offset to avoid banking stall */
	  }
	  p_state->cir_buf.p_base = p_mem;
	  p_state->p_inp_base = p_input;
}


VOID xa_nn_dilated_conv2d_std_init_circ_buf(
    VOID *p_scratch,
    VOID *p_kernel,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height_dilation,
    WORD32 kernel_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 output_channels,
    WORD32 dilation_height,
    WORD32 dilation_h_offset,
    WORD32 input_precision,
    WORD32 kernel_precision)
{
  WORD8 *p_mem;// = (WORD8 *)p_scratch;
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  size_t input_size = 0;
  UWORD32 align_size = 0;
  WORD32 input_channels_pad;

  switch(input_precision)
  {
    case 8:
    case -4:
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
      input_size = sizeof(UWORD8);
      align_size = ALIGNMENT>>1;
      break;
    default:
      break;
  }

  p_state->cir_buf.p_begin = p_state->cir_buf.p_base;
  p_state->cir_buf.p_curr = p_state->cir_buf.p_begin;

  p_mem = p_state->cir_buf.p_begin;

  // Computing circular buffer size
  // Determine y-bottom padding
#if !ENABLE_PADDING_CONV2D_STD
  if(input_precision == PREC_8 || input_precision == PREC_ASYM8U || input_precision == PREC_ASYM8S) //TODO: remove the condition when the padding requirement is removed for other variants.
    input_channels_pad = input_channels;
  else
#endif
    input_channels_pad = PADDED_SIZE(input_channels, align_size);

  // calculate height for this offset case
  WORD32 y_b_pad_total = kernel_height_dilation + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad_total = y_b_pad_total < 0 ? 0 : y_b_pad_total;

  WORD32 total_height = (y_padding + input_height + y_b_pad_total);
  WORD32 height = (total_height/dilation_height) + (WORD32) (((total_height%dilation_height)-1)>=dilation_h_offset);

  WORD32 cir_buf_size_bytes = height * kernel_width * input_channels_pad * input_size;

  while(cir_buf_size_bytes%16 !=0)
  {
      cir_buf_size_bytes+= kernel_width*input_channels_pad*input_size;
  }

  p_mem += cir_buf_size_bytes;
  p_state->cir_buf.p_end = p_mem;

  AE_SETCBEGIN0(p_state->cir_buf.p_begin);
  AE_SETCEND0(p_state->cir_buf.p_end);
  
  p_mem = ALIGNED_ADDR(p_mem, ALIGNMENT);

  p_state->p_kernel_padded = (void *)p_kernel;

#if !ENABLE_PADDING_CONV2D_STD
  if( (input_precision != PREC_ASYM8S) &&
      (input_precision != PREC_F32) &&
      (input_precision != PREC_16) &&
      (input_channels_pad != input_channels) )
#else
  if( (input_precision != PREC_F32) &&
      (input_precision != PREC_16) &&
      (input_channels_pad != input_channels) )
#endif
  {
    int oc, kh, kw, kernel_size;
    p_state->p_kernel_padded = (void *)p_mem;

    switch(kernel_precision)
    {
      case 8:
        kernel_size = sizeof(WORD8);
        break;
      case 16:
        kernel_size = sizeof(WORD16);
        break;
      case -1:
        kernel_size = sizeof(WORD32);
        break;
      case -3:
        kernel_size = sizeof(UWORD8);
        break;
      case -4:
      case -5:
        kernel_size = sizeof(WORD8);
        break;
      default:
        kernel_size = 0;
        break;
    }

    pWORD8 p_src = (pWORD8) p_kernel;
    pWORD8 p_dst = (pWORD8) p_state->p_kernel_padded;

    for(oc = 0; oc < output_channels; oc++)
    {
      for(kh = 0; kh < kernel_height_dilation; kh++)
      {
        for(kw = 0; kw < kernel_width; kw++)
        {
          memcpy(p_dst, p_src, kernel_size * input_channels);
          p_dst += kernel_size * input_channels;
          p_src += kernel_size * input_channels;
      
          memset(p_dst, 0, kernel_size * (input_channels_pad - input_channels));
          p_dst += kernel_size * (input_channels_pad - input_channels);
        }
      }
    }
  }
}

VOID conv2d_std_init_cir_buf(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state)
{
  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  WORD32 planes_to_add = x_stride > kernel_width ? 0 : kernel_width - x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);

  // Initialize circular buffer
  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
  }

  // Set next 'input_height' rows of cir_buf with zero and/or input data
  WORD32 copy_x_pad_width = x_padding;
  WORD32 copy_inp_width = 0;
  WORD32 rem_copy_width = 0;
  if(planes_to_add <= x_padding)
  {
    copy_x_pad_width = planes_to_add;
  }
  else
  {
    copy_inp_width = planes_to_add - x_padding;
    rem_copy_width = XT_MAX(0, copy_inp_width - input_width);
    copy_inp_width = XT_MIN(copy_inp_width, input_width);
  }
  for(i=0;i<input_height;i++)
  {
    for(k=0;k<copy_x_pad_width;k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    for(k=0;k<copy_inp_width;k++)
    {
      memcpy(p_dst, p_inp, input_channels * input_bytewidth);
      memset(&p_dst[input_channels * input_bytewidth], 0, (input_channels_pad - input_channels) * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      p_inp += input_channels * input_bytewidth;
    }
    for(k=0;k<rem_copy_width;k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
    p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
  }

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
  }
  p_inp += (-input_height * input_width + copy_inp_width) * input_channels * input_bytewidth;
  *pp_inp = (VOID *)p_inp;
}

VOID conv2d_group_init_cir_buf(
    WORD32 input_channels,
    WORD32 kernel_channels_pad,
    WORD32 kernel_channels,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val)
{
  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  WORD32 planes_to_add = x_stride > kernel_width ? 0 : kernel_width - x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  UWORD8 pad_val_u8 = (UWORD8)pad_val;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);

  // Initialize circular buffer
  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, pad_val_u8, kernel_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);
  }

  // Set next 'input_height' rows of cir_buf with zero and/or input data
  WORD32 copy_x_pad_width = x_padding;
  WORD32 copy_inp_width = 0;
  WORD32 rem_copy_width = 0;
  if(planes_to_add <= x_padding)
  {
    copy_x_pad_width = planes_to_add;
  }
  else
  {
    copy_inp_width = planes_to_add - x_padding;
    rem_copy_width = XT_MAX(0, copy_inp_width - input_width);
    copy_inp_width = XT_MIN(copy_inp_width, input_width);
  }
  for(i=0;i<input_height;i++)
  {
    for(k=0;k<copy_x_pad_width;k++)
    {
      memset(p_dst, pad_val_u8, kernel_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
    }
    for(k=0;k<copy_inp_width;k++)
    {
      memcpy(p_dst, p_inp, kernel_channels * input_bytewidth);
      memset(&p_dst[kernel_channels * input_bytewidth], pad_val_u8, (kernel_channels_pad - kernel_channels) * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
      p_inp += input_channels * input_bytewidth;
    }
    for(k=0;k<rem_copy_width;k++)
    {
      memset(p_dst, pad_val_u8, kernel_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);
    p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
  }

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, pad_val_u8, kernel_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);
  }
  p_inp += (-input_height * input_width + copy_inp_width) * input_channels * input_bytewidth;
  *pp_inp = (VOID *)p_inp;
}

// Add x_stride (but not more than kernel_width) x (input_height x input_channels) new planes to circular buffer
// Slow version of conv2d_std_update_cir_buf with fewer requirements
VOID conv2d_std_update_cir_buf_slow(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    WORD32 idx_beg_inp_width_pad,
    xa_nn_conv_state_t *p_state)
{
  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  WORD32 planes_to_add = x_stride > kernel_width ? kernel_width : x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;

  // Copy 'planes_to_add' planes of data to circular buffer
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_state->cir_buf.p_curr, planes_to_add * input_channels_pad * input_bytewidth);
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);

  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
  }

  // Set next 'input_height' rows of cir_buf with zero (from x_padding) and/or input data and/or zero (from x-right padding)
  WORD32 idx_end_inp_width_pad = idx_beg_inp_width_pad + planes_to_add;
  WORD32 copy_x_pad_width = 0;
  WORD32 copy_inp_width = 0;
  WORD32 to_skip_inp_width = x_stride - planes_to_add;     // Non-zero for x_stride > kernel_width
  WORD32 copy_x_r_pad_width = 0;
  if(idx_beg_inp_width_pad < x_padding)
  {
    copy_x_pad_width = x_padding - idx_beg_inp_width_pad;
    copy_inp_width = idx_end_inp_width_pad - x_padding;
    copy_x_r_pad_width = XT_MAX(0, copy_inp_width - input_width);
    copy_inp_width = XT_MIN(copy_inp_width, input_width);
  }
  else if(idx_end_inp_width_pad <= x_padding + input_width)
  {
    copy_inp_width = planes_to_add;
  }
  else if(idx_beg_inp_width_pad < x_padding + input_width)
  {
    copy_inp_width = x_padding + input_width - idx_beg_inp_width_pad;
    copy_x_r_pad_width = idx_end_inp_width_pad - (x_padding + input_width);
  }
  else
  {
    copy_x_r_pad_width = planes_to_add;
  }

  const int size1 = input_channels * input_bytewidth;
  const int size2 = (input_channels_pad - input_channels) * input_bytewidth;
  const int size3 = input_channels_pad * input_bytewidth;
  if( (size1 < 32) && (size2 < 32) && (size3 < 32)){
  /* This case handle smaller sizes (<32) in which the functions memset/memcpy are not to be called */
    for(i=0;i<input_height;i++)
    {
      for(k=0;k<copy_x_pad_width;k++)
      {
        int ii;
        for(ii = 0; ii < size3; ii++) {
          p_dst[ii] = 0;
        }
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      }
      for(k=0;k<copy_inp_width;k++)
      {
        int ii;
        for(ii = 0; ii < size1; ii++){
          p_dst[ii] = p_inp[ii];
        }
        for(ii=0; ii < size2; ii++) {
          p_dst[size1 + ii] = 0;
        }
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
        p_inp += input_channels * input_bytewidth;
      }
      for(k=0;k<copy_x_r_pad_width;k++)
      {
        int ii;
        for(ii = 0; ii < size3; ii++) {
          p_dst[ii] = 0;
        }
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
      p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
    }
  } else {
    for(i=0;i<input_height;i++)
    {
      for(k=0;k<copy_x_pad_width;k++)
      {
        memset(p_dst, 0, input_channels_pad * input_bytewidth);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      }
      for(k=0;k<copy_inp_width;k++)
      {
        memcpy(p_dst, p_inp, input_channels * input_bytewidth);
        memset(&p_dst[input_channels * input_bytewidth], 0, (input_channels_pad - input_channels) * input_bytewidth);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
        p_inp += input_channels * input_bytewidth;
      }
      for(k=0;k<copy_x_r_pad_width;k++)
      {
        memset(p_dst, 0, input_channels_pad * input_bytewidth);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
      p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
    }
  }
  p_inp += (-input_height * input_width + copy_inp_width + to_skip_inp_width) * input_channels * input_bytewidth;

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
  }
  *pp_inp = (VOID *)p_inp;
}

VOID conv2d_group_update_cir_buf_slow(
    WORD32 input_channels,
    WORD32 kernel_channels_pad,
    WORD32 kernel_channels,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    WORD32 idx_beg_inp_width_pad,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val)
{
  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  UWORD8 pad_val_u8 = (UWORD8)pad_val;
  WORD32 planes_to_add = x_stride > kernel_width ? kernel_width : x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;

  // Copy 'planes_to_add' planes of data to circular buffer
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_state->cir_buf.p_curr, planes_to_add * kernel_channels_pad * input_bytewidth);
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);

  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, pad_val_u8, kernel_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);
  }

  // Set next 'input_height' rows of cir_buf with zero (from x_padding) and/or input data and/or zero (from x-right padding)
  WORD32 idx_end_inp_width_pad = idx_beg_inp_width_pad + planes_to_add;
  WORD32 copy_x_pad_width = 0;
  WORD32 copy_inp_width = 0;
  WORD32 to_skip_inp_width = x_stride - planes_to_add;     // Non-zero for x_stride > kernel_width
  WORD32 copy_x_r_pad_width = 0;
  if(idx_beg_inp_width_pad < x_padding)
  {
    copy_x_pad_width = x_padding - idx_beg_inp_width_pad;
    copy_inp_width = idx_end_inp_width_pad - x_padding;
    copy_x_r_pad_width = XT_MAX(0, copy_inp_width - input_width);
    copy_inp_width = XT_MIN(copy_inp_width, input_width);
  }
  else if(idx_end_inp_width_pad <= x_padding + input_width)
  {
    copy_inp_width = planes_to_add;
  }
  else if(idx_beg_inp_width_pad < x_padding + input_width)
  {
    copy_inp_width = x_padding + input_width - idx_beg_inp_width_pad;
    copy_x_r_pad_width = idx_end_inp_width_pad - (x_padding + input_width);
  }
  else
  {
    copy_x_r_pad_width = planes_to_add;
  }

  const int size1 = kernel_channels * input_bytewidth;
  const int size2 = (kernel_channels_pad - kernel_channels) * input_bytewidth;
  const int size3 = kernel_channels_pad * input_bytewidth;

  if ((kernel_channels <= 16) && ((kernel_channels_pad-kernel_channels) <= 16) && (kernel_channels_pad <= 16)) 
  {
    for(i=0;i<input_height;i++)
    {
      for(k=0;k<copy_x_pad_width;k++)
      {
        int ii;
        for(ii = 0; ii < size3; ii++) {
          p_dst[ii] = pad_val_u8;
        }
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
      }
      for(k=0;k<copy_inp_width;k++)
      {
        for (int j = 0; j < size1; ++j) {
          p_dst[j] = p_inp[j];
        }
        for(int ii=0; ii < size2; ii++) {
          p_dst[size1 + ii] = pad_val_u8;
        }
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
        p_inp += input_channels * input_bytewidth;
      }
      for(k=0;k<copy_x_r_pad_width;k++)
      {
        int ii;
        for(ii = 0; ii < size3; ii++) {
          p_dst[ii] = pad_val_u8;
        }
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);
      p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
    }
  
  }
  else 
  {
    for(i=0;i<input_height;i++)
    {
      for(k=0;k<copy_x_pad_width;k++)
      {
        memset(p_dst, pad_val_u8, kernel_channels_pad * input_bytewidth);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
      }
      for(k=0;k<copy_inp_width;k++)
      {
        memcpy(p_dst, p_inp, kernel_channels * input_bytewidth);
        memset(&p_dst[kernel_channels * input_bytewidth], pad_val_u8, (kernel_channels_pad - kernel_channels) * input_bytewidth);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
        p_inp += input_channels * input_bytewidth;
      }
      for(k=0;k<copy_x_r_pad_width;k++)
      {
        memset(p_dst, pad_val_u8, kernel_channels_pad * input_bytewidth);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);
      p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
    }
  }
  p_inp += (-input_height * input_width + copy_inp_width + to_skip_inp_width) * input_channels * input_bytewidth;

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, pad_val_u8, kernel_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);
  }
  *pp_inp = (VOID *)p_inp;
}

// Add x_stride (but not more than kernel_width) x (input_height x input_channels) new planes to circular buffer
VOID conv2d_std_update_cir_buf(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    WORD32 idx_beg_inp_width_pad,
    xa_nn_conv_state_t *p_state)
{
  if (y_padding != 0 || y_b_pad != 0 || x_padding != 0) {
    conv2d_std_update_cir_buf_slow(
      input_channels,
      input_channels_pad,
      input_bytewidth,
      input_width,
      input_height,
      y_padding,
      y_b_pad,
      x_padding,
      kernel_width,
      x_stride,
      pp_inp,
      idx_beg_inp_width_pad,
      p_state
    );
    return;
  }

  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  WORD32 planes_to_add = x_stride > kernel_width ? kernel_width : x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;

  // Copy 'planes_to_add' planes of data to circular buffer
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_state->cir_buf.p_curr, planes_to_add * input_channels_pad * input_bytewidth);
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);

  // Set next 'input_height' rows of cir_buf with zero (from x_padding) and/or input data and/or zero (from x-right padding)
  //WORD32 idx_end_inp_width_pad = idx_beg_inp_width_pad + planes_to_add;
  WORD32 copy_inp_width = planes_to_add;
  WORD32 to_skip_inp_width = x_stride - planes_to_add;     // Non-zero for x_stride > kernel_width

  int size = input_channels * input_bytewidth;
  if (size <= 32) {
    for(i=0;i<input_height;i++)
    {
      for(k=0;k<copy_inp_width;k++)
      {
        for (int j = 0; j < size; ++j) {
          p_dst[j] = p_inp[j];
        }
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
        p_inp += input_channels * input_bytewidth;
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
      p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
    }
  } else {
    for(i=0;i<input_height;i++)
    {
      for(k=0;k<copy_inp_width;k++)
      {
        memcpy(p_dst, p_inp, input_channels * input_bytewidth);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
        p_inp += input_channels * input_bytewidth;
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
      p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
    }
  }
  p_inp += (-input_height * input_width + copy_inp_width + to_skip_inp_width) * input_channels * input_bytewidth;

  *pp_inp = (VOID *)p_inp;
}

VOID conv2d_group_update_cir_buf(
    WORD32 input_channels,
    WORD32 kernel_channels_pad,
    WORD32 kernel_channels,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    WORD32 idx_beg_inp_width_pad,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val)
{
  if (y_padding != 0 || y_b_pad != 0 || x_padding != 0) {
    conv2d_group_update_cir_buf_slow(
      input_channels,
      kernel_channels_pad,
      kernel_channels,
      input_bytewidth,
      input_width,
      input_height,
      y_padding,
      y_b_pad,
      x_padding,
      kernel_width,
      x_stride,
      pp_inp,
      idx_beg_inp_width_pad,
      p_state,
      pad_val
    );
    return;
  }

  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  UWORD8 pad_val_u8 = (UWORD8)pad_val;
  WORD32 planes_to_add = x_stride > kernel_width ? kernel_width : x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;

  // Copy 'planes_to_add' planes of data to circular buffer
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_state->cir_buf.p_curr, planes_to_add * kernel_channels_pad * input_bytewidth);
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);

  // Set next 'input_height' rows of cir_buf with zero (from x_padding) and/or input data and/or zero (from x-right padding)
  //WORD32 idx_end_inp_width_pad = idx_beg_inp_width_pad + planes_to_add;
  WORD32 copy_inp_width = planes_to_add;
  WORD32 to_skip_inp_width = x_stride - planes_to_add;     // Non-zero for x_stride > kernel_width

  int size = kernel_channels * input_bytewidth;
  const int size2 = (kernel_channels_pad - kernel_channels) * input_bytewidth;
  if ((kernel_channels <= 16) && ((kernel_channels_pad-kernel_channels) <= 16)) {
    for(i=0;i<input_height;i++)
    {
      for(k=0;k<copy_inp_width;k++)
      {
        for (int j = 0; j < size; ++j) {
          p_dst[j] = p_inp[j];
        }
        for(int ii=0; ii < size2; ii++) {
          p_dst[size + ii] = pad_val_u8;
        }
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
        p_inp += input_channels * input_bytewidth;
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);
      p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
    }
  } else {
    for(i=0;i<input_height;i++)
    {
      for(k=0;k<copy_inp_width;k++)
      {
        memcpy(p_dst, p_inp, kernel_channels * input_bytewidth);
        memset(&p_dst[kernel_channels * input_bytewidth], pad_val_u8, (kernel_channels_pad - kernel_channels) * input_bytewidth);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, kernel_channels_pad * input_bytewidth);
        p_inp += input_channels * input_bytewidth;
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * kernel_channels_pad * input_bytewidth);
      p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
    }
  }
  p_inp += (-input_height * input_width + copy_inp_width + to_skip_inp_width) * input_channels * input_bytewidth;

  *pp_inp = (VOID *)p_inp;

}

VOID xa_nn_dilated_conv2d_std_load_cir_buf_asym8(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val,
    WORD32 dilation_height,
    WORD32 dilation_h_offset,
    WORD32 dilation_width,
    WORD32 dilation_w_offset,
    WORD32 x_padding_full,
    WORD32 *input_padding_consumed,
    WORD32 *input_width_consumed,
    WORD32 planes_to_add,
    WORD32 firstCall,
    WORD32 *circMatrixHeight,
    WORD32 widthIndexIteration,
    WORD32 x_stride_dilated,
    WORD32 heightIndexIteration)
{ 
  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  //WORD32 planes_to_add = x_stride > kernel_width ? 0 : kernel_width - x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;
  //ae_int8x8 zero_pad = AE_MOVDA8(pad_val);
  UWORD8 pad_val_u8 = (UWORD8)pad_val;
  //ae_int8x8 inp_val;
  (void) input_bytewidth;
  WORD32 y_padding_dilation;

  if(!firstCall)
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_state->cir_buf.p_curr, planes_to_add * input_channels_pad);
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad);

  WORD32 indexCorrectionDoneInHeight = 1;
  WORD32 heightIndexIterationModified = heightIndexIteration;
  y_padding_dilation = (y_padding / dilation_height) + (WORD32)(((y_padding%dilation_height)-1)>=dilation_h_offset);
  WORD32 y_padding_dilation_indexCorrected = y_padding_dilation - heightIndexIteration;
  if(y_padding_dilation_indexCorrected<0)
  {
      indexCorrectionDoneInHeight = 0;
      heightIndexIterationModified = -y_padding_dilation_indexCorrected;
      y_padding_dilation_indexCorrected = 0;
  }
  *circMatrixHeight = 0;
  *circMatrixHeight = *circMatrixHeight + y_padding_dilation_indexCorrected;
  // Initialize circular buffer
  /*if(input_channels == 1)
  {
    // Set first 'y_padding' rows of cir_buf to zero
    for(i=0;i<y_padding;i++)
    {
      for(k=0;k<planes_to_add;k++)
      {
        AE_S8_0_XC(zero_pad, (ae_int8 *)p_dst, 1);
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep);
    }
  }
  else*/
  {
    // Set first 'y_padding' rows of cir_buf to zero
    for(i=0;i<y_padding_dilation_indexCorrected;i++)
    {
      for(k=0;k<planes_to_add;k++)
      {
        memset(p_dst, pad_val_u8, input_channels_pad);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad);
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad);
    }
  }

  // Set next 'input_height' rows of cir_buf with zero and/or input data
    //estimate no.zeros for this offset

  ///Calculate number x padding remaining for this width offset which can participate in the convolution process
  WORD32 x_padding_full_dilation = (x_padding_full/dilation_width) + (WORD32) ( ((x_padding_full%dilation_width)-1) >= dilation_w_offset);//This is the contribution of zero padding(in total) towards this width offset
  WORD32 x_padding_dilation_initial_pad = ((x_padding_full-x_padding)/dilation_width) + (WORD32) ( (((x_padding_full-x_padding)%dilation_width)-1) >= dilation_w_offset); /// This offset's contribution which has been absorbed in initial analysis of zero padding
  WORD32 x_padding_dilation = x_padding_full_dilation - x_padding_dilation_initial_pad;//This is the num of zeros contribution from left padding for this dilation offset
  WORD32 indexCorrectionDoneInWidth = 1;
  WORD32 widthIndexIterationModified = widthIndexIteration;
  //Accounting for initial width index/point in this sub-matrix for this offset (This arises from stride implementation)
  WORD32 x_padding_dilation_postIndexCorrection = x_padding_dilation;// - widthIndexIteration;/// If this value lr. than zero implies first width-index inside this sub-matrix is inside input matrix after crossing left zero padding

          x_padding_dilation_postIndexCorrection = x_padding_dilation_postIndexCorrection - widthIndexIteration;
          if(x_padding_dilation_postIndexCorrection<0)
          {
              indexCorrectionDoneInWidth = 0;
              widthIndexIterationModified = -x_padding_dilation_postIndexCorrection;
              x_padding_dilation_postIndexCorrection = 0;
          }
          else
          {
              indexCorrectionDoneInWidth = 1;
              widthIndexIterationModified = 0;
          }

  x_padding_dilation = x_padding_dilation_postIndexCorrection - (*input_padding_consumed); /// When this loop called repeatedly; some of the input will be consumed discounting for that

  if(x_padding_dilation<0)
      x_padding_dilation = 0;/// This condition can occur when we are done with zero padding section in the prev. iteration(can be first iteration in corner case)


  /// Calculate number of input width/columns remaining for this width offset which can participate in the convolution process
  WORD32 x_padding_plus_input_dilation = ( (x_padding_full+input_width)/dilation_width) + (WORD32) ( (((x_padding_full+input_width)%dilation_width)-1) >= dilation_w_offset);//This is the num elements to be convolved for this offset in total(zeropad+input)
  WORD32 x_input_dilation = x_padding_plus_input_dilation - x_padding_full_dilation;// This is the number of elements from input that can potentially be populated
  WORD32 x_input_dilation_postIndexCorrection;
  WORD32 input_width_correction;

  if(indexCorrectionDoneInWidth==0)
  {
      x_input_dilation_postIndexCorrection = x_input_dilation - widthIndexIterationModified; // this value if -ve correction flows towards right z.p
      if(x_input_dilation_postIndexCorrection<0)
      {
          indexCorrectionDoneInWidth = 0;
          widthIndexIterationModified = -x_input_dilation_postIndexCorrection;
          x_input_dilation_postIndexCorrection = 0;
          input_width_correction = x_input_dilation;
      }
      else
      {
          indexCorrectionDoneInWidth = 1;
          input_width_correction = widthIndexIterationModified;
          widthIndexIterationModified = 0;

      }
  }
  else
  {
      x_input_dilation_postIndexCorrection = x_input_dilation;
      input_width_correction = 0;
  }

  WORD32 x_input_dilation_postIndexCorrection_total = x_input_dilation_postIndexCorrection;/// This is the total convoble area after adjustng for stride offset for this dilation_offset
  x_input_dilation_postIndexCorrection = x_input_dilation_postIndexCorrection - (*input_width_consumed);//consumedInput;/// When this loop called repeatedly; some of the input will be consumed discounting for that
  if(x_input_dilation_postIndexCorrection<0)
      x_input_dilation_postIndexCorrection = 0;/// This implies the control is to right padding

  WORD32 copy_x_pad_width, copy_x_r_pad_width, copy_inp_width;

  if(planes_to_add <= x_padding_dilation)
  {
    copy_x_pad_width = planes_to_add;
    copy_inp_width = 0;
    copy_x_r_pad_width = 0;
  }
  else if(planes_to_add <= (x_padding_dilation+x_input_dilation_postIndexCorrection) )
  {
      copy_x_pad_width = x_padding_dilation;
      copy_inp_width = planes_to_add - copy_x_pad_width;
      copy_x_r_pad_width = 0;
  }
  else
  {
      copy_x_pad_width = x_padding_dilation;
      copy_inp_width = x_input_dilation_postIndexCorrection;
      copy_x_r_pad_width = planes_to_add - (copy_x_pad_width+copy_inp_width) ;/// No need to calculate the right padding exactly as the loop outside i.e, calling function takes care of it
  }

  {
    // estimate total number of height values for height_offset value from the input matrix
    WORD32 input_padding_plus_height_dilation = ( (y_padding+input_height) / dilation_height) + (WORD32)((((y_padding+input_height)%dilation_height)-1)>=dilation_h_offset);
    WORD32 input_height_dilation = input_padding_plus_height_dilation - y_padding_dilation;//y_padding_dilation; /// This value is the height of the circular matrix that has to be iterated for non-zero input values i.e., without top padding and bottim padding iterations
    WORD32 input_height_dilation_indexCorrected = input_height_dilation;
    WORD32 input_height_correction = 0;
    if(indexCorrectionDoneInHeight==0)
    {
        input_height_dilation_indexCorrected = input_height_dilation_indexCorrected - heightIndexIterationModified;
        if(input_height_dilation_indexCorrected<0)
        {
            indexCorrectionDoneInHeight = 0;
            heightIndexIterationModified = -input_height_dilation_indexCorrected;
            input_height_dilation_indexCorrected =  0;
            input_height_correction = input_height_dilation;
        }
        else
        {
            indexCorrectionDoneInHeight = 1;
            input_height_correction = heightIndexIterationModified;
            heightIndexIterationModified = 0;

        }
    }
    *circMatrixHeight = *circMatrixHeight + input_height_dilation_indexCorrected;

    /// estimate the offset needed in the input matrix for this height offset
    WORD32 index_0_input_dilation_height_offset =  (y_padding % dilation_height) ; ///This value represent 0th index in input matrix (post top padding) correspond to which offset in height's dilation scale
    WORD32 input_offset_height_dilation = (dilation_h_offset - index_0_input_dilation_height_offset + dilation_height)%dilation_height;// "index_0_input_dilation_height_offset" represent the dilation offset corresponding to 0 th row of input but, the target is to reach "dilation_h_offset" in dilation scale. This calculation helps reach there from "index_0_input_dilation_height_offset"

    p_inp = p_inp + (input_offset_height_dilation * input_width * input_channels); // This offsets the pointer as per the dilation offset in height dimension for stride=1. While supporting stride find the point inside sub matrix that is the starting point
    p_inp = p_inp + (input_height_correction * dilation_height * input_width * input_channels);///This accounts for offset i.e., initial index that arises out of stride support
    /// In the above calculation of pointer ystride is not brought into calculation, in height dimension Ystride will be handled by core convolution code

    //for(i=0;i<input_height_dilation;i++)
    for(i=0;i<input_height_dilation_indexCorrected;i++)
    {
      for(k=0;k<copy_x_pad_width;k++)
      {
        memset(p_dst, pad_val_u8, input_channels_pad);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad);
      }
      WORD32 index_0_input_dilation_offset =  (x_padding_full % dilation_width) ; ///This represent 0th index in input matrix correspond to which offset in dilation
      WORD32 input_offset_dilation = (dilation_w_offset - index_0_input_dilation_offset + dilation_width)%dilation_width;// This is the offset corresponding to the present width offset
      p_inp = p_inp + (  (input_offset_dilation + (*input_width_consumed)*dilation_width)   *input_channels);/// This is the offset corresponding
      p_inp = p_inp + input_width_correction * dilation_width * input_channels;
      // Pointer Offset in width dimension here does not have an exclusive mention as explained below:
      // a) If stride value is smaller than the kernel then "planes_to_add" would be loaded with "stride" value outside the call. This data will begin from the width index after dropping xstride values. So, no need to exclusively mention this
      // b) If stride value is gr. than kernel then "strideConsumption" would be accounted in the total consumed value and the next index would start appropriately accounting for stride in an indirect fashion


      for(k=0;k<copy_inp_width;k++)
      {
        memcpy(p_dst, p_inp, input_channels);
        memset(&p_dst[input_channels], pad_val_u8, (input_channels_pad - input_channels));
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad);
        p_inp += (input_channels*dilation_width);
      }
      for(k=0;k<copy_x_r_pad_width;k++)
      {
          memset(p_dst, pad_val_u8, input_channels_pad);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad);
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad);
      p_inp += ( (input_width - ((copy_inp_width*dilation_width)+(input_offset_dilation + (*input_width_consumed)*dilation_width) + (input_width_correction * dilation_width)  ) ) + ((dilation_height-1)*input_width) )* input_channels;

    }

    if ( (copy_inp_width >0) && (x_stride_dilated>kernel_width) )
        *input_width_consumed = *input_width_consumed + x_stride_dilated - copy_x_pad_width;/// Account for stride consumption only if there was any consumption. Reduce whatever was consumed in left zp
    else
        *input_width_consumed = *input_width_consumed + copy_inp_width;

    if(x_input_dilation_postIndexCorrection_total < (*input_width_consumed) )
        *input_width_consumed = x_input_dilation_postIndexCorrection_total;


    if ( (copy_x_pad_width >0) && (x_stride_dilated>kernel_width) )
        *input_padding_consumed = *input_padding_consumed + x_stride_dilated ;
    else
        *input_padding_consumed = *input_padding_consumed + copy_x_pad_width ;

    if(x_padding_dilation_postIndexCorrection <  (*input_padding_consumed) )
        *input_padding_consumed = x_padding_dilation_postIndexCorrection;

    /// Similar consumption calculation is not needed for right padding. This is because in right padding number of points will be lesser than kernel width as the outside function would have absolved all other right padding indices implying there would not be more than one call to fill right padding as a part of circular matrix loading


    WORD32 input_height_toppadding_plus_input_plus_bottom_padding =  ((y_padding+input_height+y_b_pad) / dilation_height) + (WORD32)((((y_padding+input_height+y_b_pad)%dilation_height)-1)>=dilation_h_offset);// This is the total number of input points used for convolution for this height offset value
    WORD32 y_b_pad_dilation = input_height_toppadding_plus_input_plus_bottom_padding - (y_padding_dilation+input_height_dilation);/// This calculates number of bottom padding points for this dilation offset i.e., dilation_h_offset

    WORD32 input_bpadding_dilation_indexCorrected = y_b_pad_dilation;

    if(indexCorrectionDoneInHeight==0)
    {
        input_bpadding_dilation_indexCorrected = input_bpadding_dilation_indexCorrected - heightIndexIterationModified;
    }
    *circMatrixHeight = *circMatrixHeight + input_bpadding_dilation_indexCorrected;
    // Set last 'y_b_pad' rows of cir_buf to zero
    for(i=0;i<input_bpadding_dilation_indexCorrected;i++)
    {
      for(k=0;k<planes_to_add;k++)
      {
        memset(p_dst, pad_val_u8, input_channels_pad);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad);
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad);
    }

  }
}

VOID conv2d_std_init_cir_buf_asym8(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val)
{
  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  WORD32 planes_to_add = x_stride > kernel_width ? 0 : kernel_width - x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  UWORD8 pad_val_u8 = (UWORD8)pad_val;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);

  // Initialize circular buffer
  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
  }

  // Set next 'input_height' rows of cir_buf with zero and/or input data
  WORD32 copy_x_pad_width = x_padding;
  WORD32 copy_inp_width = 0;
  WORD32 rem_copy_width = 0;
  if(planes_to_add <= x_padding)
  {
    copy_x_pad_width = planes_to_add;
  }
  else
  {
    copy_inp_width = planes_to_add - x_padding;
    rem_copy_width = XT_MAX(0, copy_inp_width - input_width);
    copy_inp_width = XT_MIN(copy_inp_width, input_width);
  }
  for(i=0;i<input_height;i++)
  {
    for(k=0;k<copy_x_pad_width;k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    for(k=0;k<copy_inp_width;k++)
    {
      memcpy(p_dst, p_inp, input_channels * input_bytewidth);
      memset(&p_dst[input_channels * input_bytewidth], pad_val_u8, (input_channels_pad - input_channels) * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      p_inp += input_channels * input_bytewidth;
    }
    for(k=0;k<rem_copy_width;k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
    p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
  }

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
  }
  p_inp += (-input_height * input_width + copy_inp_width) * input_channels * input_bytewidth;
  *pp_inp = (VOID *)p_inp;
}

// Add x_stride (but not more than kernel_width) x (input_height x input_channels) new planes to circular buffer
VOID conv2d_std_update_cir_buf_asym8(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    WORD32 idx_beg_inp_width_pad,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val)
{
  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  UWORD8 pad_val_u8 = (UWORD8)pad_val;
  WORD32 planes_to_add = x_stride > kernel_width ? kernel_width : x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;

  // Copy 'planes_to_add' planes of data to circular buffer
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_state->cir_buf.p_curr, planes_to_add * input_channels_pad * input_bytewidth);
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);

  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
  }

  // Set next 'input_height' rows of cir_buf with zero (from x_padding) and/or input data and/or zero (from x-right padding)
  WORD32 idx_end_inp_width_pad = idx_beg_inp_width_pad + planes_to_add;
  WORD32 copy_x_pad_width = 0;
  WORD32 copy_inp_width = 0;
  WORD32 to_skip_inp_width = x_stride - planes_to_add;     // Non-zero for x_stride > kernel_width
  WORD32 copy_x_r_pad_width = 0;
  if(idx_beg_inp_width_pad < x_padding)
  {
    copy_x_pad_width = x_padding - idx_beg_inp_width_pad;
    copy_inp_width = idx_end_inp_width_pad - x_padding;
    copy_x_r_pad_width = XT_MAX(0, copy_inp_width - input_width);
    copy_inp_width = XT_MIN(copy_inp_width, input_width);
  }
  else if(idx_end_inp_width_pad <= x_padding + input_width)
  {
    copy_inp_width = planes_to_add;
  }
  else if(idx_beg_inp_width_pad < x_padding + input_width)
  {
    copy_inp_width = x_padding + input_width - idx_beg_inp_width_pad;
    copy_x_r_pad_width = idx_end_inp_width_pad - (x_padding + input_width);
  }
  else
  {
    copy_x_r_pad_width = planes_to_add;
  }

  const int size1 = input_channels * input_bytewidth;
  const int size2 = (input_channels_pad - input_channels) * input_bytewidth;
  const int size3 = input_channels_pad * input_bytewidth;
  if( (size1 < 16) && (size2 < 16) && (size3 < 16)){
  /* This case handle smaller sizes (<16) in which the functions memset/memcpy are not to be called */
    for(i=0;i<input_height;i++)
    {
      for(k=0;k<copy_x_pad_width;k++)
      {
        int ii;
        for(ii = 0; ii < size3; ii++) {
          p_dst[ii] = pad_val_u8;
        }
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      }
      for(k=0;k<copy_inp_width;k++)
      {
        int ii;
        for(ii = 0; ii < size1; ii++){
          p_dst[ii] = p_inp[ii];
        }
        for(ii=0; ii < size2; ii++) {
          p_dst[size1 + ii] = pad_val_u8;
        }
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
        p_inp += input_channels * input_bytewidth;
      }
      for(k=0;k<copy_x_r_pad_width;k++)
      {
        int ii;
        for(ii = 0; ii < size3; ii++) {
          p_dst[ii] = pad_val_u8;
        }
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
      p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
    }
  } else {
    for(i=0;i<input_height;i++)
    {
      for(k=0;k<copy_x_pad_width;k++)
      {
        memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      }
      for(k=0;k<copy_inp_width;k++)
      {
        memcpy(p_dst, p_inp, input_channels * input_bytewidth);
        memset(&p_dst[input_channels * input_bytewidth], pad_val_u8, (input_channels_pad - input_channels) * input_bytewidth);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
        p_inp += input_channels * input_bytewidth;
      }
      for(k=0;k<copy_x_r_pad_width;k++)
      {
        memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      }
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
      p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
    }
  }

  p_inp += (-input_height * input_width + copy_inp_width + to_skip_inp_width) * input_channels * input_bytewidth;

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, planes_to_keep * input_channels_pad * input_bytewidth);
  }
  *pp_inp = (VOID *)p_inp;
}
#endif // #ifndef ENABLE_SCRATCH_SIZE_API_ONLY

