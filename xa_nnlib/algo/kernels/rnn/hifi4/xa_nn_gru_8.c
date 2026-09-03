/*******************************************************************************
* Copyright (c) 2018-2026 Cadence Design Systems, Inc.
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

#ifndef ENABLE_SCRATCH_SIZE_API_ONLY
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
#endif

WORD32 xa_nn_gru_getsize(
  WORD32 n_batch,
  WORD32 n_itr,
  WORD32 hidden_size,
  WORD32 hidden_precision)
{
#ifndef ENABLE_SCRATCH_SIZE_API_ONLY
  XA_NNLIB_ARG_CHK_COND((n_batch <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((n_itr <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((hidden_size <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((hidden_precision != PREC_ASYM8S), -1);
#endif
  WORD32 total_scratch_size;
  WORD32 W_fc_out_size, U_fc_out_size, hidden_state_size;

  W_fc_out_size = sizeof(WORD16) * 4 * n_itr * n_batch * hidden_size;
  U_fc_out_size = sizeof(WORD16) * 4 * n_batch * hidden_size;

  hidden_state_size = sizeof(WORD8) * n_batch * hidden_size;

  total_scratch_size = W_fc_out_size + U_fc_out_size + hidden_state_size;

  return total_scratch_size;
}

#ifndef ENABLE_SCRATCH_SIZE_API_ONLY
static void xa_nn_gru_gate_integer_8x8_16(
  WORD16 *p_out,
  WORD16 *fc_out_W_ptr,
  WORD16 *fc_out_U_ptr,
  WORD16 *reset_gate,
  WORD32 n_batch,
  WORD32 hidden_size,
  WORD32 batch_offset_W,
  WORD32 rg_fcU_out_multiplier,
  WORD32 rg_fcU_out_shift 
)
{
  if(reset_gate)
  {
    WORD32 shape[4]={1,1,n_batch, hidden_size};
    xa_nn_elm_mul_broadcast_4D_sym16sxsym16s_sym16s(
      fc_out_U_ptr,
      shape,
      rg_fcU_out_shift,
      rg_fcU_out_multiplier,
      -32768,
      32767,
      fc_out_U_ptr,
      shape,
      reset_gate,
      shape
    );
  }
  WORD32 itr_b;
  for(itr_b = 0; itr_b < n_batch; itr_b++)
  {
    xa_nn_elm_add_16x16_16(
      fc_out_U_ptr + itr_b * hidden_size,
      fc_out_W_ptr + itr_b * batch_offset_W,
      fc_out_U_ptr + itr_b * hidden_size,
      hidden_size
    );
  }

  if(reset_gate)
  {
    xa_nn_vec_tanh_sym16s_sym16s(
      p_out,
      fc_out_U_ptr,
      0,
      0,
      n_batch * hidden_size
    );
  }
  else
  {
    xa_nn_vec_sigmoid_sym16s_sym16s(
      p_out,
      fc_out_U_ptr,
      0,
      0,
      n_batch * hidden_size
    );
  }  
}

WORD32 xa_nn_gru_sym8sxasym8s(
  WORD8* p_out,
  const WORD8* p_hidden_state,
  const gru_weights_ptrs *p_gru_weights,
  const gru_bias_ptrs *p_gru_biases,
  const WORD8* p_inp,
  WORD32 inp_size,
  WORD32 hidden_size,
  WORD32 out_size,
  WORD32 n_batch,
  WORD32 n_itr,
  const gru_quant_params *p_gru_qp,
  gru_flags *p_gru_flags,
  void* p_scratch
)
{
 WORD32 time_major, back;

 time_major = p_gru_flags->time_major;
 back = p_gru_flags->back; 
  /* NULL Pointer Checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_hidden_state, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_weights->p_rg_W, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_weights->p_rg_U, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_weights->p_ug_W, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_weights->p_ug_U, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_weights->p_ms_W, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_weights->p_ms_U, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_biases->p_rg_W_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_biases->p_ug_W_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_biases->p_ms_W_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_biases->p_rg_U_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_biases->p_ug_U_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_gru_biases->p_ms_U_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer Alignment Checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_gru_biases->p_rg_W_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_gru_biases->p_ug_W_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_gru_biases->p_ms_W_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_gru_biases->p_rg_U_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_gru_biases->p_ug_U_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_gru_biases->p_ms_U_bias, sizeof(WORD32), -1);
  /* Check FC Quant Parameters */
  XA_NNLIB_ARG_CHK_COND((p_gru_qp->rg_W_out_shift < -31 || p_gru_qp->rg_W_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_gru_qp->ug_W_out_shift < -31 || p_gru_qp->ug_W_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_gru_qp->ms_W_out_shift < -31 || p_gru_qp->ms_W_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_gru_qp->rg_U_out_shift < -31 || p_gru_qp->rg_U_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_gru_qp->ug_U_out_shift < -31 || p_gru_qp->ug_U_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_gru_qp->ms_U_out_shift < -31 || p_gru_qp->ms_U_out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((p_gru_qp->rg_fcU_out_shift < -31 || p_gru_qp->rg_fcU_out_shift > 31), -1);
  /* Parameter checks */
  XA_NNLIB_ARG_CHK_COND((p_gru_qp->input_zero_bias < -127 || p_gru_qp->input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((p_gru_qp->hidden_zero_bias < -128 || p_gru_qp->hidden_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((p_gru_qp->hidden_shift < -31      || p_gru_qp->hidden_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((inp_size <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((n_batch <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((n_itr <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((hidden_size <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_size != hidden_size), -1);
  XA_NNLIB_ARG_CHK_COND((time_major != 0 && time_major != 1), -1);

  WORD16 *rg_fc_W_out_ptr, *ug_fc_W_out_ptr, *ms_fc_W_out_ptr;
  WORD16 *rg_fc_U_out_ptr, *ug_fc_U_out_ptr, *ms_fc_U_out_ptr;
  WORD8 *updated_hidden_state;
  WORD32 ret;
  
  rg_fc_W_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void*)((WORD16 *)p_scratch + n_itr * n_batch * hidden_size);
  ug_fc_W_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void*)((WORD16 *)p_scratch + n_itr * n_batch * hidden_size);
  ms_fc_W_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void*)((WORD16 *)p_scratch + n_itr * n_batch * hidden_size);

  rg_fc_U_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void*)((WORD16 *)p_scratch + n_batch * hidden_size);
  ug_fc_U_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void*)((WORD16 *)p_scratch + n_batch * hidden_size);
  ms_fc_U_out_ptr = (WORD16 *)p_scratch;
  p_scratch = (void*)((WORD16 *)p_scratch + n_batch * hidden_size);

  updated_hidden_state = (WORD8 *)p_scratch;
  p_scratch = (void*)((WORD8 *)p_scratch + n_batch * hidden_size);

  MEMCPY_8b(updated_hidden_state, p_hidden_state, n_batch * hidden_size);

  ret = xa_nn_matmul_sym8sxasym8s_sym16s(
          rg_fc_W_out_ptr,
          p_gru_weights->p_rg_W,
          p_inp,
          p_gru_biases->p_rg_W_bias,
          hidden_size,
          inp_size,
          inp_size,
          n_itr * n_batch,
          inp_size,
          hidden_size,
          1,
          p_gru_qp->input_zero_bias,
          p_gru_qp->rg_W_out_multiplier,
          p_gru_qp->rg_W_out_shift
        );        
  if(ret != 0)
    return ret;
  
  ret = xa_nn_matmul_sym8sxasym8s_sym16s(
          ug_fc_W_out_ptr,
          p_gru_weights->p_ug_W,
          p_inp,
          p_gru_biases->p_ug_W_bias,
          hidden_size,
          inp_size,
          inp_size,
          n_itr * n_batch,
          inp_size,
          hidden_size,
          1,
          p_gru_qp->input_zero_bias,
          p_gru_qp->ug_W_out_multiplier,
          p_gru_qp->ug_W_out_shift
        );  
  if(ret != 0)
    return ret;
  
  ret = xa_nn_matmul_sym8sxasym8s_sym16s(
        ms_fc_W_out_ptr,
        p_gru_weights->p_ms_W,
        p_inp,
        p_gru_biases->p_ms_W_bias,
        hidden_size,
        inp_size,
        inp_size,
        n_itr * n_batch,
        inp_size,
        hidden_size,
        1,
        p_gru_qp->input_zero_bias,
        p_gru_qp->ms_W_out_multiplier,
        p_gru_qp->ms_W_out_shift
      );  
  if(ret != 0)
    return ret;
  
  WORD32 itr_t, itr_b;
  for(itr_t = 0; itr_t < n_itr; itr_t++)
  {
    ret = xa_nn_matmul_sym8sxasym8s_sym16s(
            rg_fc_U_out_ptr,
            p_gru_weights->p_rg_U,
            updated_hidden_state,
            p_gru_biases->p_rg_U_bias,
            hidden_size,
            hidden_size,
            hidden_size,
            n_batch,
            hidden_size,
            hidden_size,
            1,
            -p_gru_qp->hidden_zero_bias,
            p_gru_qp->rg_U_out_multiplier,
            p_gru_qp->rg_U_out_shift
          );
    if(ret != 0)
      return ret;
    
    ret = xa_nn_matmul_sym8sxasym8s_sym16s(
            ug_fc_U_out_ptr,
            p_gru_weights->p_ug_U,
            updated_hidden_state,
            p_gru_biases->p_ug_U_bias,
            hidden_size,
            hidden_size,
            hidden_size,
            n_batch,
            hidden_size,
            hidden_size,
            1,
            -p_gru_qp->hidden_zero_bias,
            p_gru_qp->ug_U_out_multiplier,
            p_gru_qp->ug_U_out_shift
          );
    if(ret != 0)
      return ret;
    
    ret = xa_nn_matmul_sym8sxasym8s_sym16s(
            ms_fc_U_out_ptr,
            p_gru_weights->p_ms_U,
            updated_hidden_state,
            p_gru_biases->p_ms_U_bias,
            hidden_size,
            hidden_size,
            hidden_size,
            n_batch,
            hidden_size,
            hidden_size,
            1,
            -p_gru_qp->hidden_zero_bias,
            p_gru_qp->ms_U_out_multiplier,
            p_gru_qp->ms_U_out_shift
          );
    if(ret != 0)
      return ret;

    WORD32 W_fc_out_offset = 0;
    if(back) {
       W_fc_out_offset = time_major ? (n_itr - itr_t - 1) * n_batch * hidden_size :(n_itr - itr_t - 1) * hidden_size;
    }
    else{
       W_fc_out_offset = time_major ? itr_t * n_batch * hidden_size : itr_t * hidden_size;
    }
    xa_nn_gru_gate_integer_8x8_16(
      rg_fc_U_out_ptr,
      rg_fc_W_out_ptr + W_fc_out_offset,
      rg_fc_U_out_ptr,
      NULL,
      n_batch,
      hidden_size,
      time_major ? hidden_size : hidden_size * n_itr,
      0,
      0
    );

    xa_nn_gru_gate_integer_8x8_16(
      ug_fc_U_out_ptr,
      ug_fc_W_out_ptr + W_fc_out_offset,
      ug_fc_U_out_ptr,
      NULL,
      n_batch,
      hidden_size,
      time_major ? hidden_size : hidden_size * n_itr,
      0,
      0
    );

    xa_nn_gru_gate_integer_8x8_16(
      ms_fc_U_out_ptr,
      ms_fc_W_out_ptr + W_fc_out_offset,
      ms_fc_U_out_ptr,
      rg_fc_U_out_ptr,
      n_batch,
      hidden_size,
      time_major ? hidden_size : hidden_size * n_itr,
      p_gru_qp->rg_fcU_out_multiplier,
      p_gru_qp->rg_fcU_out_shift
    );

    xa_nn_gru_hidden_state_update_8(
      updated_hidden_state,
      ug_fc_U_out_ptr,
      ms_fc_U_out_ptr,
      p_gru_qp->ug_ms_out_multiplier,
      p_gru_qp->ug_ms_out_shift,
      p_gru_qp->ug_hidden_out_multiplier,
      p_gru_qp->ug_hidden_out_shift,
      p_gru_qp->hidden_multiplier,
      p_gru_qp->hidden_shift,
      p_gru_qp->hidden_zero_bias,
      n_batch * hidden_size
    );

    /* Memcpy hidden state to output */
    if(time_major)
    {
      if(back){ 
        MEMCPY_8b(&p_out[(n_itr - itr_t - 1) * n_batch * hidden_size], updated_hidden_state, (WORD32)(sizeof(WORD8) * n_batch * hidden_size));
      }
      else{
        MEMCPY_8b(&p_out[itr_t * n_batch * hidden_size], updated_hidden_state, (WORD32)(sizeof(WORD8) * n_batch * hidden_size));
      } 
    }
    else
    {
      for(itr_b = 0; itr_b < n_batch; itr_b++)
      {
        if(back){
          MEMCPY_8b(&p_out[((n_itr - itr_t - 1) + itr_b * n_itr) * hidden_size], &updated_hidden_state[itr_b * hidden_size], (WORD32)(sizeof(WORD8) * hidden_size));
        }
        else{
          MEMCPY_8b(&p_out[(itr_t + itr_b * n_itr) * hidden_size], &updated_hidden_state[itr_b * hidden_size], (WORD32)(sizeof(WORD8) * hidden_size));
        }
      }
    }
  }
  return 0;
}
#endif /* #ifndef ENABLE_SCRATCH_SIZE_API_ONLY */

