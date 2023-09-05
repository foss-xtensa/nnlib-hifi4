/*******************************************************************************
* Copyright (c) 2018-2023 Cadence Design Systems, Inc.
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
    const WORD32* __restrict__ p_bias,
    WORD8 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 out_width_over_x_pad = (x_padding - kernel_width)/x_stride + 1;
  WORD32 left_shift, right_shift;
  out_width_over_x_pad = out_width_over_x_pad > out_width ? out_width : out_width_over_x_pad;

  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);

  /* When kernel convolves over x-left pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = 0; j < out_width_over_x_pad; j++)
    {
      for(k = 0; k < out_channels; k++)
      {
#if TFLITE_SINGLE_ROUNDING
        left_shift  = p_out_shift[k];
        /* Single rounding macro doesn't need two shifts so this is not used */
        (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
        left_shift  = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
#endif /* #if TFLITE_SINGLE_ROUNDING */          
#if XCHAL_HAVE_HIFI1
        ae_int32x2 acc = AE_L32_I((ae_int32*)&p_bias[k], 0);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
        AE_S8_0_X_HIFI1( AE_MOVINT16X4_FROMINT32X2(acc), (WORD8 *)p_out, (i * out_height_offset + j * out_width_offset + k * out_channels_offset));
#else

        ae_int32x2 acc = AE_MOVDA32(p_bias[k]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
#if 0
        AE_MINMAX32(acc, min_int8, max_int8);
#else
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
#endif
        p_out[i * out_height_offset + j * out_width_offset + k * out_channels_offset] = (UWORD8)AE_MOVAD32_L(acc);
#endif
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
    const WORD32* __restrict__ p_bias,
    WORD8 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 idx_out_width_over_x_r_pad = (x_padding + input_width + x_stride - 1)/x_stride + 1;
  WORD32 left_shift, right_shift;
  WORD32 out_width_over_x_r_pad = out_width - idx_out_width_over_x_r_pad;

  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);

  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = idx_out_width_over_x_r_pad; j < out_width; j++)
    {
      for(k = 0; k < out_channels; k++)
      {
#if TFLITE_SINGLE_ROUNDING
        left_shift  = p_out_shift[k];
        /* Single rounding macro doesn't need two shifts so this is not used */
        (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
        left_shift  = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
#endif /* #if TFLITE_SINGLE_ROUNDING */          
#if XCHAL_HAVE_HIFI1
        ae_int32x2 acc = AE_L32_I((ae_int32*)&p_bias[k], 0);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
        AE_S8_0_X_HIFI1( AE_MOVINT16X4_FROMINT32X2(acc), (WORD8 *)p_out, (i * out_height_offset + j * out_width_offset + k * out_channels_offset));
#else
        ae_int32x2 acc = AE_MOVDA32(p_bias[k]);
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
#if 0
        AE_MINMAX32(acc, min_int8, max_int8);
#else
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
#endif
        p_out[i * out_height_offset + j * out_width_offset + k * out_channels_offset] = (UWORD8)AE_MOVAD32_L(acc);
#endif
      }
    }
  }
  return out_width_over_x_r_pad;
}


WORD32 xa_nn_conv2d_group_per_chan_sym8sxasym8s(
    WORD8* __restrict__ p_out,
    const WORD8* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD32* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 kernel_channels,
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
  //XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(UWORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_scratch, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);

  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  const int groups = input_channels/kernel_channels;
  XA_NNLIB_ARG_CHK_COND((groups<=0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_channels %kernel_channels)!=0),-1);
  XA_NNLIB_ARG_CHK_COND(((out_channels%groups)!=0),-1);
  const int kernels_per_group = out_channels / groups;
  XA_NNLIB_ARG_CHK_COND((kernels_per_group<=0),-1);

  WORD32 j;
  WORD32 input_bytewidth = 1;
  VOID *pp_inp = (VOID *)p_inp;
  WORD8* __restrict__ tmp_out;

  p_scratch = ALIGNED_ADDR(p_scratch, ALIGNMENT);
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;

  WORD32 out_channels_offset = out_data_format ? out_height * out_width : 1;
  WORD32 out_height_offset = out_data_format ? out_width : out_width * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;

  WORD32 x_padding_var = x_padding;
  WORD32 kernel_channels_pad;

#if !ENABLE_PADDING_CONV2D_STD
  kernel_channels_pad = kernel_channels;
#else
#if HW_AE_ADDCIRC16X4_XC
  if(kernel_channels == 1){
    kernel_channels_pad = 1;
  }
  else
#endif
  {
    kernel_channels_pad = PADDED_SIZE(kernel_channels, (ALIGNMENT>>1));
  }
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

  xa_nn_conv2d_std_init_state((void*)p_state
      ,(void*)p_kernel
      ,input_height
      ,kernel_channels
      ,kernel_height
      ,kernel_width
      ,y_stride
      ,y_padding
      ,out_height
      ,out_channels
      ,PREC_ASYM8S
      ,PREC_SYM8S);

  for (int grp_i = 0; grp_i < groups; ++grp_i)
  {
    tmp_out=p_out+grp_i*kernels_per_group*out_channels_offset;
    xa_nn_conv2d_group_init_state((void*)p_state
        ,(void*)p_kernel
        ,input_height
        ,kernel_channels
        ,kernel_height
        ,kernel_width
        ,y_stride
        ,y_padding
        ,out_height
        ,out_channels
        ,PREC_ASYM8S
        ,PREC_SYM8S);

    pp_inp = (VOID *)(p_inp+grp_i*kernel_channels);
    conv2d_group_init_cir_buf_asym8(input_channels, kernel_channels_pad,kernel_channels,input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, p_state, -input_zero_bias);
    
      // Index to padded input width
    WORD32 idx_beg_inp_width_pad = kernel_width - x_stride;
    idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;
    // Process Loop to compute one output plane [out_height x out_channels] per iteration
    for(j=0;j<out_width-out_width_over_x_pad-out_width_over_x_r_pad;j++)
    {
      // Add x_stride x (input_height x input_channels) new planes to circular buffer
      conv2d_group_update_cir_buf_asym8(input_channels, kernel_channels_pad,kernel_channels,input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state, -input_zero_bias);
      // Update index to input width padded
      idx_beg_inp_width_pad += x_stride;

      // Convolution using matXvec with matrix as circular buffer
      xa_nn_matXvec_sym8sxasym8s_asym8s_circ
        (tmp_out /* output */
        ,p_state->cir_buf.p_curr/* matrix: rows x cols */
        ,(p_state->p_kernel_padded+grp_i*kernels_per_group*kernel_channels_pad*kernel_width*kernel_height) /* vec: cols */
        ,(p_bias+grp_i*kernels_per_group) /* bias */
        ,out_height /* rows */
        ,kernel_channels_pad * kernel_width * kernel_height /* cols */
        ,kernel_channels_pad * kernel_width * y_stride/* row_offset */
        ,kernels_per_group /* vec_count */
        ,kernel_channels_pad * kernel_width * kernel_height /* vec_stride */
        ,out_channels_offset /* out_col_offset */
        ,out_height_offset /* out_row_offset */
        ,input_zero_bias
        ,(p_out_multiplier+grp_i*kernels_per_group)
        ,(p_out_shift+grp_i*kernels_per_group)
        ,out_zero_bias
        );

      tmp_out += out_width_offset;
    }
  
  }

  return 0;
}


WORD32 xa_nn_conv2d_group_sym8sxasym8s(
    WORD8* __restrict__ p_out,
    const WORD8* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD32* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
	  WORD32 kernel_channels,
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

    int ret = 0;
    ret |=xa_nn_conv2d_group_per_chan_sym8sxasym8s(
        p_out,
        p_inp,
        p_kernel,
        p_bias,
        input_height,
        input_width,
        input_channels,
        kernel_height,
        kernel_width,
        kernel_channels,
        out_channels,
        x_stride,
        y_stride,
        x_padding,
        y_padding,
        out_height,
        out_width,
        input_zero_bias,
        p_out_multiplier,
        p_out_shift,
        out_zero_bias,
        out_data_format,
        p_scratch);

    return ret;

}

