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
#ifndef  __XA_NN_CONV2D_STD_STATE_H__
#define  __XA_NN_CONV2D_STD_STATE_H__

#include "xa_type_def.h"

#define ALIGNED_SIZE( size, align ) \
  ( (size_t)(size) + (align) - 1 )

#define ALIGNED_ADDR( addr, align ) \
  (void*)( ( (UWORD32)(addr) + ( (align) - 1 ) ) & ~( (align) - 1 ) )

#define PADDED_SIZE( size, align ) \
  ( ( (size_t)(size) + (align) - 1 ) & ~( (align) - 1 ) )

#define BUS_WIDTH (8)
#define BUS_WIDTH_MASK (0xf)

typedef enum xa_nn_conv_datafmt_t{
  HWC=0
} xa_nn_conv_datafmt_t;

typedef struct _circular_buf_t{
  VOID *p_begin;
  VOID *p_end;
  VOID *p_curr;
  VOID *p_base;
} circular_buf_t;

typedef struct _xa_nn_conv_state_t{
  circular_buf_t cir_buf;
  void* p_kernel_padded;
  VOID* p_inp_base;
} xa_nn_conv_state_t;

VOID xa_nn_conv2d_dilation_init_state(
    VOID *p_scratch,
    VOID *p_kernel,
    VOID *p_input);

VOID xa_nn_conv2d_std_init_state(
    VOID *p_handle,
    VOID *p_kernel,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 input_precision,
    WORD32 kernel_precision);

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
    WORD32 kernel_precision);    

VOID xa_nn_dilated_conv2d_std_init_circ_buf(
    VOID *p_handle,
    VOID *p_kernel,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height_dilation,
    WORD32 kernel_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 dilation_height,
    WORD32 dilation_h_offset,
    WORD32 input_precision,
    WORD32 kernel_precision);

WORD32 xa_nn_matXvec_8x16_16_circ(
    WORD16 * __restrict__ p_out,
    WORD16 * __restrict__ p_mat,
    WORD8  * __restrict__ p_vec,
    WORD16 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_offset,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 bias_shift,
    WORD32 acc_shift);

WORD32 xa_nn_matXvec_8x8_8_circ(
    WORD8  * __restrict__ p_out,
    WORD8  * __restrict__ p_mat,
    WORD8  * __restrict__ p_vec,
    WORD8  * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_offset,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 bias_shift,
    WORD32 acc_shift);

WORD32 xa_nn_matXvec_16x16_16_circ(
    WORD16 * __restrict__ p_out,
    WORD16 * __restrict__ p_mat,
    WORD16 * __restrict__ p_vec,
    WORD16 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_offset,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 bias_shift,
    WORD32 acc_shift);

WORD32 xa_nn_matXvec_f32_circ(
    FLOAT32 * __restrict__ p_out,
    FLOAT32 * __restrict__ p_mat,
    const FLOAT32 * __restrict__ p_vec,
    const FLOAT32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_offset,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_col_offset,
    WORD32 out_row_offset);

WORD32 xa_nn_matXvec_asym8xasym8_asym8_circ(
    UWORD8 * __restrict__ p_out,
    UWORD8 * __restrict__ p_mat1,
    const UWORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_stride,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 mat1_offset,
    WORD32 vec1_offset,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_offset);

WORD32 xa_nn_matXvec_sym8sxsym16s_sym16s_circ(
    WORD16 * __restrict__ p_out,
    WORD16 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD64 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_stride,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift);

WORD32 xa_nn_matXvec_sym8sxasym8s_asym8s_circ(
    WORD8 * __restrict__ p_out,
    WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_stride,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 mat1_offset,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_offset);

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
    xa_nn_conv_state_t *p_state);

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
    WORD32 pad_val);

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
    xa_nn_conv_state_t *p_state);

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
    WORD32 pad_val);

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
    xa_nn_conv_state_t *p_state);

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
    WORD32 pad_val);

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
    WORD32 pad_val);

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
    WORD32 heightIndexIteration);

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
    WORD32 pad_val);

#endif /* __XA_NN_CONV2D_STD_STATE_H__ */

