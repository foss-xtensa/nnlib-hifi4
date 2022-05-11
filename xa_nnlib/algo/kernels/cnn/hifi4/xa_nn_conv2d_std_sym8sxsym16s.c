/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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

static inline ae_int32x2 MultiplyByQuantizedMultiplier_ref(ae_int64 d_x,
                                             int32_t quantized_multiplier,
                                             int shift){
  // Inputs:
  // - quantized_multiplier has fixed point at bit 31
  // - shift is -31 to +7 (negative for right shift)
  //
  // Assumptions: The following input ranges are assumed
  // - quantize_scale>=0  (the usual range is (1<<30) to (1>>31)-1)
  // - scaling is chosen so final scaled result fits in int32_t
  // - input x is in the range -(1<<47) <= x < (1<<47)
/* shift_val  = -31 to 7
 * total_shift = 46 to 8
 * new_shift = 46-32 to 8-32
 * */
  ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
  ae_int32x2 d_red_mul32 = AE_SEXT32X2D16_32(d_red_mul16); //upper 32
  ae_int64 qL = AE_MUL32U_LL(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x));
  ae_int64 qH = AE_SLAI64(AE_MUL32_LH(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x)), 32);
  ae_int64 q = AE_ADD64(qL, qH);
  q = AE_SRAA64(q, (-shift-17));
  ae_int32x2 result = AE_ROUND32F64SASYM(q);
  return result;
}

static WORD32 conv_x_left_pad(
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const WORD64* __restrict__ p_bias,
    WORD16 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 out_width_over_x_pad = (x_padding - kernel_width)/x_stride + 1;
  out_width_over_x_pad = out_width_over_x_pad > out_width ? out_width : out_width_over_x_pad;
  ae_int16x4 d1;

  /* When kernel convolves over x-left pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = 0; j < out_width_over_x_pad; j++)
    {
      ae_int16 *ptrout = (ae_int16*)&p_out[i * out_height_offset + j * out_width_offset];
      ae_int64 *pbias = (ae_int64*)p_bias;
      ae_int64 q1;
      for(k = 0; k < out_channels; k++)
      {
	AE_L64_IP(q1, pbias, 8);
	ae_int32x2 acc = MultiplyByQuantizedMultiplier_ref(q1, p_out_multiplier[k], p_out_shift[k]);
	d1 = AE_SAT16X4(acc, acc);
	AE_S16_0_XP(d1, ptrout, out_channels_offset*sizeof(WORD16));
      }
    }
  }
  return out_width_over_x_pad;
}

static WORD32 conv_x_right_pad(
    WORD32 x_padding,
    WORD32 input_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const WORD64* __restrict__ p_bias,
    WORD16 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 idx_out_width_over_x_r_pad = (x_padding + input_width + x_stride - 1)/x_stride + 1;
  WORD32 out_width_over_x_r_pad = out_width - idx_out_width_over_x_r_pad;
  ae_int16x4 d1;

  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = idx_out_width_over_x_r_pad; j < out_width; j++)
    {
      ae_int16 *ptrout = (ae_int16*)&p_out[i * out_height_offset + j * out_width_offset];
      ae_int64 *pbias = (ae_int64*)p_bias;
      ae_int64 q1;
      for(k = 0; k < out_channels; k++)
      {
	AE_L64_IP(q1, pbias, 8);
	ae_int32x2 acc = MultiplyByQuantizedMultiplier_ref(q1, p_out_multiplier[k], p_out_shift[k]);
	d1 = AE_SAT16X4(acc, acc);
	AE_S16_0_XP(d1, ptrout, out_channels_offset*sizeof(WORD16));
      }
    }
  }
  return out_width_over_x_r_pad;
}

WORD32 xa_nn_conv2d_std_per_chan_sym8sxsym16s(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD64* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch)
{
   /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  //XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  //XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);

  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  WORD32 j;
  WORD32 input_bytewidth = 2;
  VOID *pp_inp = (VOID *)p_inp;

  p_scratch = ALIGNED_ADDR(p_scratch, ALIGNMENT);
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  xa_nn_conv2d_std_init_state((void*)p_state
      ,(void*)p_kernel
      ,input_height
      ,input_channels
      ,kernel_height
      ,kernel_width
      ,x_stride,y_stride
      ,y_padding
      ,out_height
      ,out_channels
      ,PREC_SYM16S
      ,PREC_SYM8S);

  WORD32 out_channels_offset = out_data_format ? out_height * out_width : 1;
  WORD32 out_height_offset = out_data_format ? out_width : out_width * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;

  WORD32 x_padding_var = x_padding;

#if !ENABLE_PADDING_CONV2D_STD
  WORD32 input_channels_pad = input_channels;
#else
  WORD32 input_channels_pad = PADDED_SIZE(input_channels, (ALIGNMENT>>1));
#endif

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
  if(x_padding_var >= kernel_width)
  {
    out_width_over_x_pad = conv_x_left_pad(x_padding, kernel_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
    x_padding_var -= out_width_over_x_pad * x_stride;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = kernel_width + (out_width - 1) * x_stride - (x_padding + input_width);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= kernel_width)
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_padding, input_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
  }

  /* When kernel convolves over input region */
  p_out += out_width_over_x_pad * out_width_offset;
  // Initialize circular buffer
  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  conv2d_std_init_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, p_state, -input_zero_bias);

  // Index to padded input width
  WORD32 idx_beg_inp_width_pad = kernel_width - x_stride;
  idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;


  // Process Loop to compute one output plane [out_height x out_channels] per iteration
  for(j=0;j<out_width-out_width_over_x_pad-out_width_over_x_r_pad;j++)
  {
    // Add x_stride x (input_height x input_channels) new planes to circular buffer
    conv2d_std_update_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state, -input_zero_bias);

    // Update index to input width padded
    idx_beg_inp_width_pad += x_stride;

    // Convolution using matXvec with matrix as circular buffer
    xa_nn_matXvec_sym8sxsym16s_sym16s_circ
      (p_out /* output */
       ,p_state->cir_buf.p_curr/* matrix: rows x cols */
       ,p_state->p_kernel_padded /* vec: cols */
       ,p_bias /* bias */
       ,out_height /* rows */
       ,input_channels_pad * kernel_width * kernel_height /* cols */
       ,input_channels_pad * kernel_width * y_stride/* row_offset */
       ,out_channels /* vec_count */
       ,input_channels_pad * kernel_width * kernel_height /* vec_stride */
       ,out_channels_offset /* out_col_offset */
       ,out_height_offset /* out_row_offset */
       ,input_zero_bias
       ,p_out_multiplier
       ,p_out_shift
       ,out_zero_bias
      );
    p_out += out_width_offset;
  }

  return 0;
}

