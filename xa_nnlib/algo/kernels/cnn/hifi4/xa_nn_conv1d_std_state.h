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
#ifndef  __XA_NN_CONV1D_STD_STATE_H__
#define  __XA_NN_CONV1D_STD_STATE_H__

#include "xa_nn_conv2d_std_state.h"

VOID xa_nn_conv1d_std_init_state(
    VOID *p_handle,
    VOID *p_kernel,
    WORD32 kernel_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 y_stride,
    WORD32 input_precision);

VOID conv1d_std_init_cir_buf(
    WORD32 input_channels,
    WORD32 input_channelsXwidth_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 y_stride,
    WORD32 y_padding,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state);

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
    xa_nn_conv_state_t *p_state);

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
    WORD32 pad_val);

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
    WORD32 pad_val);

WORD32 xa_nn_matXvec_8x16_16_circ_nb(
    WORD16 *p_out,
    WORD8  *p_mat,
    WORD16 *p_vec,
    WORD16 *p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 out_offset,
    WORD32 bias_shift,
    WORD32 acc_shift);

WORD32 xa_nn_matXvec_8x8_8_circ_nb(
    WORD8 *p_out,
    WORD8 *p_mat,
    WORD8 *p_vec,
    WORD8 *p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 out_offset,
    WORD32 bias_shift,
    WORD32 acc_shift);

WORD32 xa_nn_matXvec_16x16_16_circ_nb(
    WORD16 *p_out,
    WORD16 *p_mat,
    WORD16 *p_vec,
    WORD16 *p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 out_offset,
    WORD32 bias_shift,
    WORD32 acc_shift);

WORD32 xa_nn_matXvec_f32_circ_nb(
    FLOAT32 *p_out,
    FLOAT32 *p_mat,
    FLOAT32 *p_vec,
    FLOAT32 *p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 out_offset);

WORD32 xa_nn_matXvec_asym8xasym8_asym8_circ_nb(
    UWORD8 * __restrict__ p_out,
    UWORD8 * __restrict__ p_mat1,
    UWORD8 * __restrict__ p_vec1,
    WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 out_stride,
    WORD32 mat1_offset,
    WORD32 vec1_offset,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_offset);

#endif /* __XA_NN_CONV1D_STD_STATE_H__ */

