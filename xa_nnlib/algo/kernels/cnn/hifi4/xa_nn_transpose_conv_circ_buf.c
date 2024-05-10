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
#include "xa_nn_transpose_conv_state.h"
#include "xa_nnlib_common_macros.h"

WORD32 xa_nn_transpose_conv_getsize
(
  WORD32 input_height 
 ,WORD32 input_width 
 ,WORD32 input_channels
 ,WORD32 kernel_height 
 ,WORD32 kernel_width 
 ,WORD32 x_stride
 ,WORD32 y_stride
 ,WORD32 output_height
 ,WORD32 output_width
 ,WORD32 output_channels
 ,WORD32 kernel_precision
 ,WORD32 output_precision
 )
{
    XA_NNLIB_CHK_COND((input_height <= 0), -1);
    XA_NNLIB_CHK_COND((input_width <= 0), -1);
    XA_NNLIB_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
    XA_NNLIB_CHK_COND((x_stride <= 0), -1);
    XA_NNLIB_CHK_COND((y_stride <= 0), -1);
    XA_NNLIB_CHK_COND((output_height <= 0), -1);
    XA_NNLIB_CHK_COND((output_width <= 0), -1);
    XA_NNLIB_CHK_COND((output_channels <= 0), -1);

    WORD32 scratch_bytewidth = 0;
    WORD32 input_size;
    WORD32 kernel_size;
    WORD32 align_size;
    WORD32 total_size = 0;

    switch (output_precision)
    {
        case -8: /* For sym16s */
            input_size = sizeof(WORD16);
            scratch_bytewidth = 8; /* 64b scratch */
            align_size = ALIGNMENT >> 1;
            break;
        case -4: /* For asym8s */
            input_size = sizeof(WORD8);
#if XCHAL_HAVE_HIFI1S
            scratch_bytewidth = 4; /* 32b scratch for HiFi1s*/
#else            
            scratch_bytewidth = 8; /* 64b scratch */
#endif            
            align_size = ALIGNMENT >> 1;
            break;            
        case -1: /* For float32 */
            input_size = sizeof(FLOAT32);
            scratch_bytewidth = 8; /* 32bx2 scratch */
            align_size = ALIGNMENT >> 2;
            break;			
        default:
            return -1; /* Returning due to invalid input */
            break;
    }

    switch (kernel_precision)
    {
        case -5: /* For sym8s */
            kernel_size = sizeof(WORD8);
            break;
        case -1:
            kernel_size = sizeof(FLOAT32);
            break;	
        default:
            return -1; /* Returning due to invalid prec */
            break;
    }

    int ker_grt_inp = (kernel_width > input_width || kernel_height > input_height);
    int str_leq_ker = (x_stride <= kernel_width && y_stride <= kernel_height);
    if(!ker_grt_inp && str_leq_ker)
    {
      total_size += ALIGNED_SIZE(sizeof(xa_nn_conv_state_t), 8);
      int subkerX_max = (kernel_width + x_stride - 1) / x_stride;
      int subkerY_max = (kernel_height + y_stride - 1) / y_stride;
      int n_subker = x_stride * y_stride;
	  WORD32 input_channels_pad = PADDED_SIZE(input_channels, align_size);
      WORD32 kernel_bytes = PADDED_SIZE(subkerX_max * subkerY_max * input_channels_pad * output_channels * n_subker * kernel_size, 8);
      WORD32 cir_buf_size_bytes = (2*(subkerY_max-1) + input_height) * subkerX_max * input_channels_pad * input_size;
      while(cir_buf_size_bytes%8 !=0)
      {
          cir_buf_size_bytes+= subkerX_max*input_channels_pad*input_size;
      }
      total_size += kernel_bytes + cir_buf_size_bytes; 
      total_size += BUS_WIDTH;
      total_size = PADDED_SIZE(total_size, align_size);
    }
    else
    {
      total_size = (output_height) * (output_width) * (output_channels) * (scratch_bytewidth);
    }
    return total_size;
}

#ifndef ENABLE_SCRATCH_SIZE_API_ONLY
VOID xa_nn_transpose_conv_init_state(
    VOID *p_scratch,
    VOID *p_kernel,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 input_precision)
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
  WORD32 input_channels_pad = PADDED_SIZE(input_channels,align_size);
  WORD32 cir_buf_size_bytes = (2*(kernel_height-1) + input_height) * kernel_width * input_channels_pad * input_size;
  while(cir_buf_size_bytes%8 !=0)
  {
    cir_buf_size_bytes+= kernel_width*input_channels_pad*input_size;
  }
  p_mem += cir_buf_size_bytes;
  p_state->cir_buf.p_end = p_mem;

  AE_SETCBEGIN0(p_state->cir_buf.p_begin);
  AE_SETCEND0(p_state->cir_buf.p_end);
}
#endif /* #ifndef ENABLE_SCRATCH_SIZE_API_ONLY */

