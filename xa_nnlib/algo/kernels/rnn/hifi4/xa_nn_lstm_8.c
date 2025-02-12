/*******************************************************************************
* Copyright (c) 2018-2025 Cadence Design Systems, Inc.
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
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"

#include "xa_nnlib_common.h"

WORD32 xa_nn_matmul_sym8sxasym8s_sym16s(
    WORD16 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride,                      
    WORD32 vec1_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift);
    
WORD32 xa_nn_lstm_getsize(
    WORD32 n_batch,
    WORD32 n_itr,
    WORD32 n_cell,
    WORD32 cell_state_precision)
{
  WORD32 total_scratch_size;
  WORD32 W_fc_out_size, U_fc_out_size;

  W_fc_out_size = sizeof(WORD16) * 4 * n_itr * n_batch * n_cell;
  U_fc_out_size = sizeof(WORD16) * 4 * n_batch * n_cell;

  total_scratch_size = W_fc_out_size + U_fc_out_size;

  return total_scratch_size;
}

static void xa_nn_lstm_gate_integer_8x8_16(
    WORD16 *p_out,
    WORD16 *fc_out_W_ptr,
    WORD16 *fc_out_U_ptr,
    WORD32 n_batch,
    WORD32 n_cell,
    WORD32 batch_offset_out,
    WORD32 batch_offset_W,
    WORD32 batch_offset_U,
    WORD32 activation)
{
  int itr_b;

  {
    for(itr_b = 0; itr_b < n_batch; itr_b++)
    {
      xa_nn_elm_add_16x16_16(fc_out_U_ptr + itr_b * batch_offset_U,
                             fc_out_W_ptr + itr_b * batch_offset_W,
                             fc_out_U_ptr + itr_b * batch_offset_U,
                             n_cell);
    }
  }

  switch (activation)
  {
    case 0:
      for(itr_b = 0; itr_b < n_batch; itr_b++)
      {
        xa_nn_vec_sigmoid_sym16s_sym16s(p_out + itr_b * batch_offset_out,
                                        fc_out_U_ptr + itr_b * batch_offset_U,
                                        0,
                                        0,
                                        n_cell);
      }
      break;
    case 1:
      for(itr_b = 0; itr_b < n_batch; itr_b++)
      {
        xa_nn_vec_tanh_sym16s_sym16s(p_out + itr_b * batch_offset_out,
                                     fc_out_U_ptr + itr_b * batch_offset_U,
                                     0,
                                     0,
                                     n_cell);
      }
      break;
  }
}

static void xa_nn_update_lstm_cell(
    WORD16 *cell_state,
    WORD16 *input_gate,
    WORD16 *forget_gate,
    WORD16 *cell_gate,
    WORD32 n_batch,
    WORD32 n_cell,
    WORD32 batch_offset,
    WORD32 cell_state_scale,
    WORD32 use_cifg,
    WORD16 clip)
{
  int itr_b;
  if(use_cifg == 0)
  {
    for(itr_b = 0; itr_b < n_batch; itr_b++)
    {
      xa_nn_lstm_cell_state_update_16(cell_state + itr_b * batch_offset,
                                      forget_gate + itr_b * batch_offset,
                                      cell_gate + itr_b * batch_offset,
                                      input_gate + itr_b * batch_offset,
                                      -(15),
                                      -(30 + cell_state_scale),
                                      clip,
                                      n_cell);
    }
  }
  else
  {
    return;
  }
}

static void xa_nn_lstm_output_integer_16(
    VOID *output_state,
    WORD16 *cell_state,
    WORD16 *output_gate,
    WORD32 out_precision,
    WORD32 n_batch,
    WORD32 n_cell,
    WORD32 batch_offset_og,
    WORD32 cell_state_scale,
    WORD32 hidden_multiplier,
    WORD32 hidden_shift,
    WORD32 hidden_zp,
    void *scratch)
{
  WORD16 *scratch0 = (WORD16 *)scratch;
  int itr_b;
  WORD32 tanh_mul, tanh_shift;
  tanh_shift = 15 + cell_state_scale - 3;
  tanh_mul = 0;

  if (tanh_shift < 0)
  {
    tanh_shift = -tanh_shift;
#if (defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4))
    tanh_mul = 1;
#else
    tanh_mul = 3;
#endif
  }

  for(itr_b = 0; itr_b < n_batch; itr_b++)
  {
    xa_nn_vec_tanh_sym16s_sym16s(scratch0,
                                 cell_state + itr_b * n_cell,
                                 tanh_mul,
                                 tanh_shift,
                                 n_cell);
    if(out_precision == 8 || out_precision == -5)
    {
      xa_nn_elm_mul_sym16sxsym16s_asym8s(output_state + itr_b * n_cell,
                                         hidden_zp,
                                         hidden_shift,
                                         hidden_multiplier,
                                         -128,
                                         127,
                                         output_gate + itr_b * batch_offset_og,
                                         scratch0,
                                         n_cell);
    }
    else
    {
      WORD32 inout_shape[4] = {1, 1, 1, 1};
      inout_shape[3] = n_cell;
      xa_nn_elm_mul_broadcast_4D_sym16sxsym16s_sym16s(output_state + itr_b * n_cell,
                                                      inout_shape,
                                                      hidden_shift,
                                                      hidden_multiplier,
                                                      -32768,
                                                      32767,
                                                      output_gate + itr_b * batch_offset_og,
                                                      inout_shape,
                                                      scratch0,
                                                      inout_shape);
    }
  }
}

WORD32 xa_nn_lstm_sym8sxasym8s_16(
    WORD8*  p_out,                      /* out */
    WORD8*  p_hidden_state,             /* inout */
    WORD16* p_cell_state,               /* inout */
    lstm_weights_ptrs *p_lstm_weights,  /* input */
    lstm_bias_ptrs    *p_lstm_biases,   /* input */
    WORD8*  p_inp,                      /* input */
    WORD32 inp_size,
    WORD32 hidden_size,
    WORD32 out_size,
    WORD32 n_batch,
    WORD32 n_itr,
    WORD32 n_cell,
    lstm_quant_params *p_lstm_qp,
    lstm_flags *p_lstm_flags,
    void*  p_scratch)
{
  /* NULL Pointer Checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_hidden_state, -1);
  XA_NNLIB_ARG_CHK_PTR(p_cell_state, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  if(!p_lstm_flags->use_cifg)
  {
    XA_NNLIB_ARG_CHK_PTR(p_lstm_weights->p_ig_W, -1);
    XA_NNLIB_ARG_CHK_PTR(p_lstm_weights->p_ig_U, -1);
  }
  XA_NNLIB_ARG_CHK_PTR(p_lstm_weights->p_fg_W, -1);
  XA_NNLIB_ARG_CHK_PTR(p_lstm_weights->p_fg_U, -1);
  XA_NNLIB_ARG_CHK_PTR(p_lstm_weights->p_cg_W, -1);
  XA_NNLIB_ARG_CHK_PTR(p_lstm_weights->p_cg_U, -1);
  XA_NNLIB_ARG_CHK_PTR(p_lstm_weights->p_og_W, -1);
  XA_NNLIB_ARG_CHK_PTR(p_lstm_weights->p_og_U, -1);
  XA_NNLIB_ARG_CHK_PTR(p_lstm_biases->p_ig_W_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_lstm_biases->p_fg_W_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_lstm_biases->p_cg_W_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_lstm_biases->p_og_W_bias, -1);
  
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer Alignment Checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_cell_state, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_lstm_biases->p_ig_W_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_lstm_biases->p_fg_W_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_lstm_biases->p_cg_W_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_lstm_biases->p_og_W_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, 8, 1);
  /* Check FC Quant Parameters */
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->ig_W_out_shift < -31 || p_lstm_qp->ig_W_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->fg_W_out_shift < -31 || p_lstm_qp->fg_W_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->cg_W_out_shift < -31 || p_lstm_qp->cg_W_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->og_W_out_shift < -31 || p_lstm_qp->og_W_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->ig_U_out_shift < -31 || p_lstm_qp->ig_U_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->fg_U_out_shift < -31 || p_lstm_qp->fg_U_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->cg_U_out_shift < -31 || p_lstm_qp->cg_U_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->og_U_out_shift < -31 || p_lstm_qp->og_U_out_shift > 31), -1);
  /* Parameter checks */
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->input_zero_bias < -127 || p_lstm_qp->input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->hidden_zero_bias < -128 || p_lstm_qp->hidden_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->hidden_shift < -31 || p_lstm_qp->hidden_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_lstm_qp->cell_state_scale < -29 || p_lstm_qp->cell_state_scale > 1), -1);
  XA_NNLIB_ARG_CHK_COND((inp_size <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((n_batch <= 0 || n_cell <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((n_itr <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_size != n_cell || hidden_size != n_cell), -1);
  XA_NNLIB_ARG_CHK_COND((p_lstm_flags->use_cifg != 0), -1);

  WORD16 *ig_fc_W_out_ptr, *fg_fc_W_out_ptr, *cg_fc_W_out_ptr, *og_fc_W_out_ptr;
  WORD16 *ig_fc_U_out_ptr, *fg_fc_U_out_ptr, *cg_fc_U_out_ptr, *og_fc_U_out_ptr;
  WORD32 ret;

  WORD32 use_cifg, time_major;
  
  use_cifg = p_lstm_flags->use_cifg;
  time_major = p_lstm_flags->time_major;

  if(!use_cifg)
  {
    ig_fc_W_out_ptr = (WORD16 *)p_scratch;
    p_scratch = (void *)((WORD16 *)p_scratch + n_batch * n_itr * n_cell);
  }
  fg_fc_W_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void *)((WORD16 *)p_scratch + n_batch * n_itr * n_cell);
  cg_fc_W_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void *)((WORD16 *)p_scratch + n_batch * n_itr * n_cell);
  og_fc_W_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void *)((WORD16 *)p_scratch + n_batch * n_itr * n_cell);

  if(!use_cifg)
  {
    ig_fc_U_out_ptr = (WORD16 *)p_scratch;
    p_scratch = (void *)((WORD16 *)p_scratch + n_batch * n_cell);
  }
  fg_fc_U_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void *)((WORD16 *)p_scratch + n_batch * n_cell);
  cg_fc_U_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void *)((WORD16 *)p_scratch + n_batch * n_cell);
  og_fc_U_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void *)((WORD16 *)p_scratch + n_batch * n_cell);

  WORD32 itr_t, itr_b;
  if(!use_cifg)
  {
    ret = xa_nn_matmul_sym8sxasym8s_sym16s(ig_fc_W_out_ptr,
                                           p_lstm_weights->p_ig_W,
                                           p_inp,
                                           p_lstm_biases->p_ig_W_bias,
                                           n_cell,
                                           inp_size,
                                           inp_size,
                                           n_itr * n_batch,
                                           inp_size,
                                           n_cell,
                                           1,
                                           p_lstm_qp->input_zero_bias,
                                           p_lstm_qp->ig_W_out_multiplier,
                                           p_lstm_qp->ig_W_out_shift);
    if(ret != 0)
      return ret;
  }
  ret = xa_nn_matmul_sym8sxasym8s_sym16s(fg_fc_W_out_ptr,
                                         p_lstm_weights->p_fg_W,
                                         p_inp,
                                         p_lstm_biases->p_fg_W_bias,
                                         n_cell,
                                         inp_size,
                                         inp_size,
                                         n_itr * n_batch,
                                         inp_size,
                                         n_cell,
                                         1,
                                         p_lstm_qp->input_zero_bias,
                                         p_lstm_qp->fg_W_out_multiplier,
                                         p_lstm_qp->fg_W_out_shift);
  if(ret != 0)
    return ret;
  ret = xa_nn_matmul_sym8sxasym8s_sym16s(cg_fc_W_out_ptr,
                                         p_lstm_weights->p_cg_W,
                                         p_inp,
                                         p_lstm_biases->p_cg_W_bias,
                                         n_cell,
                                         inp_size,
                                         inp_size,
                                         n_itr * n_batch,
                                         inp_size,
                                         n_cell,
                                         1,
                                         p_lstm_qp->input_zero_bias,
                                         p_lstm_qp->cg_W_out_multiplier,
                                         p_lstm_qp->cg_W_out_shift);
  if(ret != 0)
    return ret;
  ret = xa_nn_matmul_sym8sxasym8s_sym16s(og_fc_W_out_ptr,
                                         p_lstm_weights->p_og_W,
                                         p_inp,
                                         p_lstm_biases->p_og_W_bias,
                                         n_cell,
                                         inp_size,
                                         inp_size,
                                         n_itr * n_batch,
                                         inp_size,
                                         n_cell,
                                         1,
                                         p_lstm_qp->input_zero_bias,
                                         p_lstm_qp->og_W_out_multiplier,
                                         p_lstm_qp->og_W_out_shift);
  if(ret != 0)
    return ret;

  for(itr_t = 0; itr_t < n_itr; itr_t++)
  {
    for(itr_b = 0; itr_b < n_batch; itr_b++)
    {
      WORD32 offset_fc_out = itr_b * n_cell;    // Recurrent FCs can only be done for 1 time iteration at a time
      WORD32 offset_hidden = itr_b * n_cell;
      /* Input Gate FC */
      if(!use_cifg)
      {
        ret = xa_nn_matXvec_out_stride_sym8sxasym8s_16(ig_fc_U_out_ptr + offset_fc_out,
                                                       p_lstm_weights->p_ig_U,
                                                       p_hidden_state + offset_hidden,
                                                       NULL,
                                                       n_cell,
                                                       hidden_size,
                                                       hidden_size,
                                                       1,
                                                       -p_lstm_qp->hidden_zero_bias,
                                                       p_lstm_qp->ig_U_out_multiplier,
                                                       p_lstm_qp->ig_U_out_shift);
        if(ret != 0)
          return ret;
      }
      /* Forget Gate FC */
      ret = xa_nn_matXvec_out_stride_sym8sxasym8s_16(fg_fc_U_out_ptr + offset_fc_out,
                                                     p_lstm_weights->p_fg_U,
                                                     p_hidden_state + offset_hidden,
                                                     NULL,
                                                     n_cell,
                                                     hidden_size,
                                                     hidden_size,
                                                     1,
                                                     -p_lstm_qp->hidden_zero_bias,
                                                     p_lstm_qp->fg_U_out_multiplier,
                                                     p_lstm_qp->fg_U_out_shift);
      if(ret != 0)
        return ret;
      /* Cell Gate FC */
      ret = xa_nn_matXvec_out_stride_sym8sxasym8s_16(cg_fc_U_out_ptr + offset_fc_out,
                                                     p_lstm_weights->p_cg_U,
                                                     p_hidden_state + offset_hidden,
                                                     NULL,
                                                     n_cell,
                                                     hidden_size,
                                                     hidden_size,
                                                     1,
                                                     -p_lstm_qp->hidden_zero_bias,
                                                     p_lstm_qp->cg_U_out_multiplier,
                                                     p_lstm_qp->cg_U_out_shift);
      if(ret != 0)
        return ret;
      /* Output Gate FC */
      ret = xa_nn_matXvec_out_stride_sym8sxasym8s_16(og_fc_U_out_ptr + offset_fc_out,
                                                     p_lstm_weights->p_og_U,
                                                     p_hidden_state + offset_hidden,
                                                     NULL,
                                                     n_cell,
                                                     hidden_size,
                                                     hidden_size,
                                                     1,
                                                     -p_lstm_qp->hidden_zero_bias,
                                                     p_lstm_qp->og_U_out_multiplier,
                                                     p_lstm_qp->og_U_out_shift);
      if(ret != 0)
        return ret;
    }

    WORD32 W_fc_out_offset = time_major ? itr_t * n_batch * n_cell : itr_t * n_cell;
    if(!use_cifg)
    {
      xa_nn_lstm_gate_integer_8x8_16(ig_fc_U_out_ptr,
                                     ig_fc_W_out_ptr + W_fc_out_offset,
                                     ig_fc_U_out_ptr,
                                     n_batch,
                                     n_cell,
                                     n_cell,
                                     time_major ? n_cell : n_cell * n_itr,
                                     n_cell,
                                     0);
    }
    xa_nn_lstm_gate_integer_8x8_16(fg_fc_U_out_ptr,
                                   fg_fc_W_out_ptr + W_fc_out_offset,
                                   fg_fc_U_out_ptr,
                                   n_batch,
                                   n_cell,
                                   n_cell,
                                   time_major ? n_cell : n_cell * n_itr,
                                   n_cell,
                                   0);
    xa_nn_lstm_gate_integer_8x8_16(cg_fc_U_out_ptr,
                                   cg_fc_W_out_ptr + W_fc_out_offset,
                                   cg_fc_U_out_ptr,
                                   n_batch,
                                   n_cell,
                                   n_cell,
                                   time_major ? n_cell : n_cell * n_itr,
                                   n_cell,
                                   1);
    xa_nn_update_lstm_cell(p_cell_state,
                           ig_fc_U_out_ptr,
                           fg_fc_U_out_ptr,
                           cg_fc_U_out_ptr,
                           n_batch,
                           n_cell,
                           n_cell,
                           p_lstm_qp->cell_state_scale,
                           use_cifg,
                           p_lstm_qp->quantized_cell_clip);
    xa_nn_lstm_gate_integer_8x8_16(og_fc_U_out_ptr,
                                   og_fc_W_out_ptr + W_fc_out_offset,
                                   og_fc_U_out_ptr,
                                   n_batch,
                                   n_cell,
                                   n_cell,
                                   time_major ? n_cell : n_cell * n_itr,
                                   n_cell,
                                   0);
    /* ig_fc_U_out_ptr reused as scratch here */
    xa_nn_lstm_output_integer_16(p_hidden_state,
                                 p_cell_state,
                                 og_fc_U_out_ptr,
                                 8,
                                 n_batch,
                                 n_cell,
                                 n_cell,
                                 p_lstm_qp->cell_state_scale,
                                 p_lstm_qp->hidden_multiplier,
                                 p_lstm_qp->hidden_shift,
                                 p_lstm_qp->hidden_zero_bias,
                                 ig_fc_U_out_ptr);
    if(time_major)
    {
      MEMCPY_8b(&p_out[itr_t*n_batch*n_cell], p_hidden_state, (WORD32)(sizeof(WORD8) * n_batch * n_cell));
    }
    else
    {
      for(itr_b = 0; itr_b < n_batch; itr_b++)
      {
        MEMCPY_8b(&p_out[(itr_t + itr_b * n_itr) * n_cell], &p_hidden_state[itr_b * n_cell], (WORD32)(sizeof(WORD8) * n_cell));
      }
    }
  }
  return 0;
}