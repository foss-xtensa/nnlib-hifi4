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

static WORD32 conv_y_top_pad(
    WORD32 y_padding,
    WORD32 kernel_height,
    WORD32 y_stride,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_height_offset,
    WORD16 *p_bias,
    WORD16 *p_out,
    WORD32 bias_shift,
    WORD32 acc_shift)
{
  WORD32 i,j;
  WORD32 out_height_over_y_pad = (y_padding - kernel_height)/y_stride + 1;
  out_height_over_y_pad = out_height_over_y_pad > out_height ? out_height : out_height_over_y_pad;

  /* When kernel convolves over y-top pad region only, output is just bias */
  for(i=0;i<out_height_over_y_pad;i++)
  {
    for(j=0;j<out_channels;j++)
    {
#if XCHAL_HAVE_HIFI1
      ae_int16x4 acc16;
      ae_int64 acc;
      acc16 = AE_L16_I((ae_int16 *)&p_bias[j], 0);
      acc = AE_SRAI64(  AE_MOVINT64_FROMINT16X4(acc16), 48);
      acc = AE_SLAA64S(acc, bias_shift);
      acc = AE_SLAA64S(acc, acc_shift);
      ae_int32x2 sat_acc = AE_ROUND32F64SSYM(acc);
      p_out[i*out_height_offset+j*out_channels_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_acc, sat_acc));
#else
      ae_int64 acc = p_bias[j];
      acc = AE_SLAA64S(acc, bias_shift);
      acc = AE_SLAA64S(acc, acc_shift);
      p_out[i*out_height_offset+j*out_channels_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(acc),16),-16));
#endif
    }
  }
  return out_height_over_y_pad;
}

static WORD32 conv_y_bottom_pad(
    WORD32 y_padding,
    WORD32 input_height,
    WORD32 y_stride,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_height_offset,
    WORD16 *p_bias,
    WORD16 *p_out,
    WORD32 bias_shift,
    WORD32 acc_shift)
{
  WORD32 i,j;
  WORD32 idx_out_height_over_y_b_pad = (y_padding + input_height + y_stride - 1)/y_stride + 1;
  WORD32 out_height_over_y_b_pad = out_height - idx_out_height_over_y_b_pad;

  /* When kernel convolves over y-bottom pad region only, output is just bias */
  for(i=idx_out_height_over_y_b_pad;i<out_height;i++)
  {
    for(j=0;j<out_channels;j++)
    {
#if XCHAL_HAVE_HIFI1
      ae_int16x4 acc16;
      ae_int64 acc;
      acc16 = AE_L16_I((ae_int16 *)&p_bias[j], 0);
      acc = AE_SRAI64(  AE_MOVINT64_FROMINT16X4(acc16), 48);
      acc = AE_SLAA64S(acc, bias_shift);
      acc = AE_SLAA64S(acc, acc_shift);
      ae_int32x2 sat_acc = AE_ROUND32F64SSYM(acc);
      p_out[i*out_height_offset+j*out_channels_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_acc, sat_acc));
#else
      ae_int64 acc = p_bias[j];
      acc = AE_SLAA64S(acc, bias_shift);
      acc = AE_SLAA64S(acc, acc_shift);
      p_out[i*out_height_offset+j*out_channels_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(acc),16),-16));
#endif
    }
  }
  return out_height_over_y_b_pad;
}


WORD32 xa_nn_conv1d_std_16x16(
    WORD16* __restrict__ p_out,
    WORD16* __restrict__ p_inp,
    WORD16* __restrict__ p_kernel,
    WORD16* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 out_channels,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 bias_shift,
    WORD32 acc_shift,
    WORD32 out_data_format,
    VOID *p_scratch)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_kernel, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND((y_stride > kernel_height), -1);

  WORD32 j;
  WORD32 input_bytewidth = sizeof(*p_inp);
  VOID *pp_inp = (VOID *)p_inp;

  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  xa_nn_conv1d_std_init_state((void*)p_state,(void*)p_kernel,kernel_height,input_width,input_channels,y_stride,input_bytewidth*8);

  WORD32 out_channels_offset = out_data_format ? out_height : 1;
  WORD32 out_height_offset = out_data_format ? 1: out_channels;

  WORD32 y_padding_var = y_padding;
  WORD32 input_channelsXwidth_pad = PADDED_SIZE(input_channels*input_width, (ALIGNMENT>>1));

  // Limit effective bias_shift and acc_shift to [-63 ... 63]
  bias_shift = bias_shift > 63 ? 63 : bias_shift < -63 ? -63 : bias_shift;
  /* +48 to move acc to upper 16bits, as TRUNC keeps upper 32bits and ROUND keeps upper 16bits */
  acc_shift = acc_shift + 32;
  acc_shift = acc_shift > 63 ? 63 : acc_shift < -63 ? -63 : acc_shift;


  /* When kernel convolves over y-top pad region only */
  WORD32 out_height_over_y_pad = 0;
  if(y_padding_var >= kernel_height)
  {
    out_height_over_y_pad = conv_y_top_pad(y_padding, kernel_height, y_stride, out_height, out_channels, out_channels_offset, out_height_offset, p_bias, p_out, bias_shift, acc_shift);
    y_padding_var -= out_height_over_y_pad * y_stride;
  }


  /* When kernel convolves over y-bottom pad region only */
  WORD32 out_height_over_y_b_pad = 0;
  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;
  if(y_b_pad >= kernel_height)
  {
    out_height_over_y_b_pad = conv_y_bottom_pad(y_padding, input_height, y_stride, out_height, out_channels, out_channels_offset, out_height_offset, p_bias, p_out, bias_shift, acc_shift);
  }


  /* When kernel convolves over input region */
  p_out += out_height_over_y_pad * out_height_offset;

  // Initialize circular buffer

  conv1d_std_init_cir_buf(input_channels, input_channelsXwidth_pad, input_bytewidth, input_width, kernel_height, y_stride, y_padding_var, (VOID**)&pp_inp, p_state);

  // Index to padded input height
  WORD32 idx_beg_inp_height_pad = kernel_height - y_stride;

  // Process Loop to compute one output line [out_channels] per iteration
  for(j=0;j<out_height-out_height_over_y_pad-out_height_over_y_b_pad;j++)
  {
    // Add y_stride x input_channelsXwidth_pad new planes to circular buffer
    conv1d_std_update_cir_buf(input_channels, input_channelsXwidth_pad, input_bytewidth, input_width, input_height, kernel_height, y_stride, y_padding_var, (VOID**)&pp_inp, idx_beg_inp_height_pad, p_state);

    // Update index to input width padded
    idx_beg_inp_height_pad += y_stride;

    // Convolution using matXvec with vec as circular buffer
    xa_nn_matXvec_16x16_16_circ_nb
      (p_out /* output */
       ,p_kernel /* mat: rows x cols */
       ,p_state->cir_buf.p_curr/* vec: cols */
       ,p_bias /* bias */
       ,out_channels /* rows */
       ,input_channelsXwidth_pad * kernel_height /* cols */
       ,out_channels_offset
       ,bias_shift
       ,acc_shift
      );

    p_out += out_height_offset;
  }

  return 0;
}

