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
#include <string.h>
#include "xa_nnlib_common.h"
#include "xa_nnlib_lstm_api.h"

#ifdef hifi4
#define XA_PAD_BYTES   8
#define ALIGN_MEM(_sptr) (((unsigned)((_sptr)+7))&(~7))
#define ALIGN_SIZE(n) (((n)+7)&(~7))
#endif
#ifdef hifi5
#define XA_PAD_BYTES   16
#define ALIGN_MEM(_sptr) (((unsigned)((_sptr)+15))&(~15))
#define ALIGN_SIZE(n) (((n)+15)&(~15))
#endif

#define scratch_alloc(_sptr, p, type, sz) { p = (type *)_sptr; _sptr += ALIGN_MEM(sz * sizeof(type));}
#define CHECK_PTR(ptr, err) if(NULL == ptr) return err;
#define CHECK_PTR_ALIGN(ptr, alignment, err) if((((unsigned)(ptr))&(alignment-1)) != 0) return err;

#define CHECK_MTX_SHAPE(p_shape, rows_shape, cols_shape)      \
{                                                             \
  if (!(p_shape.shape_type == SHAPE_MATRIX_T))                \
  {                                                           \
    return XA_NNLIB_FATAL_INVALID_SHAPE;                      \
  }                                                           \
  if (!(p_shape.dim.matrix.rows == rows_shape             &&  \
         p_shape.dim.matrix.cols == cols_shape            &&  \
         p_shape.dim.matrix.row_offset == cols_shape      &&  \
         p_shape.n_shapes == 1))                              \
  {                                                           \
    return XA_NNLIB_FATAL_INVALID_SHAPE;                      \
  }                                                           \
}

#define CHECK_VEC_SHAPE(p_shape, length_shape)                \
{                                                             \
  if(!(p_shape.shape_type == SHAPE_VECTOR_T))                 \
  {                                                           \
    return XA_NNLIB_FATAL_INVALID_SHAPE;                      \
  }                                                           \
  if (!(p_shape.dim.vector.length == length_shape &&          \
        p_shape.n_shapes == 1))                               \
  {                                                           \
    return XA_NNLIB_FATAL_INVALID_SHAPE;                      \
  }                                                           \
}

#define CHECK_IO_SHAPE(p_shape)                               \
{                                                             \
  if(!(p_shape->shape_type == SHAPE_VECTOR_T))                \
  {                                                           \
    return XA_NNLIB_FATAL_INVALID_SHAPE;                      \
  }                                                           \
}

typedef struct _lstm_state_t
{
  vect_t *prev_h;
  int *prev_c;
  xa_nnlib_lstm_weights_t weights;
  xa_nnlib_lstm_biases_t biases;
  int in_feats;
  int out_feats;
  int pad;
  int precision;
  int bias_shift;
  int matmul_lsh;
  int h_lsh;
  int fXprev_c_lsh;
  int iXc_hat_lsh;
} lstm_state_t;

typedef struct _temp_mem_t
{
  Int32 *vec;
} temp_mem_t;

typedef struct _scratch_mem_t
{
  vect_t *f_f;
  vect_t *i_f_or_o_f;
  vect_t *c_hat_f_or_tanh_c_f;
  temp_mem_t temp_mem;
} scratch_mem_t;

static void vec_elem_mul_16x32plus16x16_16(Int32 * __restrict__ output, Int16 * __restrict__ input_1, Int32 * __restrict__ input_2, Int16 * __restrict__ input_3, Int16 * __restrict__ input_4, int fXprev_c_lsh, int iXc_hat_lsh, int num_elm)
{
#pragma aligned(output, 8)
#pragma aligned(input_1, 8)
#pragma aligned(input_2, 8)
#pragma aligned(input_3, 8)
#pragma aligned(input_4, 8)

  int i;
  ae_f16x4 *inp1 = (ae_f16x4 *)input_1, *inp3 = (ae_f16x4 *)input_3, *inp4 = (ae_f16x4 *)input_4;
  ae_f32x2 *inp2 = (ae_f32x2 *)input_2, *out = (ae_f32x2 *)output;
  for(i=0;i<num_elm>>2;i++)
  {
    ae_f32x2 res1_1 = AE_SLAA32S(AE_MULFP32X16X2RS_H(inp2[2*i], inp1[i]), fXprev_c_lsh+15);
    ae_f32x2 res2_1 = AE_SLAA32S(AE_MULFP32X16X2RS_L_S2(inp2[2*i+1], inp1[i]), fXprev_c_lsh+15);
    ae_f32x2 res1_2, res2_2;
    AE_MULF16X4SS(res1_2, res2_2, inp3[i], inp4[i]);
    res1_2 = AE_SLAA32S(res1_2, iXc_hat_lsh-1);
    res2_2 = AE_SLAA32S(res2_2, iXc_hat_lsh-1);
    out[2*i] = AE_ADD32S(res1_1, res1_2);
    out[2*i+1] = AE_ADD32S(res2_1, res2_2);
  }

}

static void lstm_output_kernel_16x16_16(Int16 * __restrict__ output, Int16 * __restrict__ prev_output, Int16 * __restrict__ input_1, Int16 * __restrict__ input_2, int lsh, int num_elm)
{
#pragma aligned(output, 8)
#pragma aligned(prev_output, 8)
#pragma aligned(input_1, 8)
#pragma aligned(input_1, 8)
  int i;
  ae_f16x4 *inp1 = (ae_f16x4 *)input_1, *inp2 = (ae_f16x4 *)input_2, *out = (ae_f16x4 *)output, *prev_h = (ae_f16x4 *)prev_output;
  for(i=0;i<num_elm>>2;i++)
  {
    out[i] = prev_h[i] = AE_SLAA16S(AE_MULFP16X4S(inp1[i], inp2[i]), lsh);
  }
}

static Int32 validate_config(xa_nnlib_lstm_init_config_t *config)
{
  if(config->in_feats < 4 || config->in_feats > 2048 || (config->in_feats&3) != 0)
    return XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_IN_FEATS;

  if(config->out_feats < 4 || config->out_feats > 2048 || (config->out_feats&3) != 0)
    return XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_OUT_FEATS;

  if((config->precision != XA_NNLIB_LSTM_16bx16b) && (config->precision != XA_NNLIB_LSTM_8bx16b))
    return XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_PRECISION;

  if(config->coeff_Qformat < 0 || config->coeff_Qformat > 15)
    return XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_COEFF_QFORMAT;

  if(config->cell_Qformat < 0 || config->cell_Qformat > 25)
    return XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_CELL_QFORMAT;

  if(config->io_Qformat < 0 || config->io_Qformat > 15)
    return XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_IO_QFORMAT;

  if((config->pad !=0) && (config->pad != 1))
    return XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_MEMBANK_PADDING;

  return XA_NNLIB_NO_ERROR;
}

Int32 xa_nnlib_lstm_get_persistent_fast(
     xa_nnlib_lstm_init_config_t *config )
{
  int persistent_size, ret;
  CHECK_PTR(config, XA_NNLIB_FATAL_MEM_ALLOC);

  ret = validate_config(config);
  if(ret != XA_NNLIB_NO_ERROR)
    return ret;

  persistent_size  = ALIGN_SIZE(sizeof(lstm_state_t));
  // Size of prev_h and prev_c
  persistent_size += ALIGN_SIZE(config->out_feats * sizeof(vect_t));
  persistent_size += ALIGN_SIZE(config->out_feats * sizeof(int));

  return persistent_size;
}

Int32 xa_nnlib_lstm_get_scratch_fast(
       xa_nnlib_lstm_init_config_t *config )
{
  int scratch_size, ret;
  CHECK_PTR(config, XA_NNLIB_FATAL_MEM_ALLOC);

  ret = validate_config(config);
  if(ret != XA_NNLIB_NO_ERROR)
    return ret;

  scratch_size = ALIGN_SIZE(sizeof(scratch_mem_t));
  scratch_size += 3 * ALIGN_SIZE(config->out_feats * sizeof(vect_t));
#ifdef MODEL_FLT64
  scratch_size += 0;
#elif MODEL_INT16
  scratch_size += ALIGN_SIZE(1 * config->out_feats * sizeof(Int32));    //vect scratch
#endif

  return scratch_size;
}

int __attribute__((optimize ("-O0"))) xa_nnlib_lstm_init(
    xa_nnlib_handle_t handle,
    xa_nnlib_lstm_init_config_t *config )
{
  lstm_state_t *lstm;
  int ret;

  CHECK_PTR(handle, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(config, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR_ALIGN(handle, 8, XA_NNLIB_FATAL_MEM_ALIGN);

  ret = validate_config(config);
  if(ret != XA_NNLIB_NO_ERROR)
    return ret;

  lstm = (lstm_state_t *) handle;
  memset(lstm,0, sizeof(lstm_state_t));

  lstm->in_feats   = config->in_feats;
  lstm->out_feats  = config->out_feats;
  lstm->pad        = config->pad;
  lstm->precision  = config->precision;
  lstm->bias_shift   = (config->io_Qformat + config->coeff_Qformat) - 15;
  lstm->matmul_lsh = 25 - (config->coeff_Qformat + config->io_Qformat);  // Input to sigmoid function should be 6.25
  lstm->fXprev_c_lsh = config->cell_Qformat - (15 + config->cell_Qformat);  // For Q15xQ25 to cell_Qformat conversion
  lstm->iXc_hat_lsh = config->cell_Qformat - (15 + 15);  // For Q15xQ15 to cell_Qformat conversion
  lstm->h_lsh = config->io_Qformat - 15;  // For Q15 to io_Qformat conversion

  lstm->prev_h = (vect_t *)ALIGN_MEM((char *)handle + sizeof(lstm_state_t));
  memset(lstm->prev_h,0, config->out_feats * sizeof(vect_t));

  lstm->prev_c = (int *)ALIGN_MEM((char *)lstm->prev_h + config->out_feats * sizeof(vect_t));
  memset(lstm->prev_c,0, config->out_feats * sizeof(int));

  return XA_NNLIB_NO_ERROR;
}

int xa_nnlib_lstm_set_config(
  xa_nnlib_handle_t handle,
  xa_nnlib_lstm_param_id_t param_id,
  void *params )
{
  lstm_state_t *lstm;
  CHECK_PTR(handle, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(params, XA_NNLIB_FATAL_MEM_ALLOC);

  CHECK_PTR_ALIGN(handle, 8, XA_NNLIB_FATAL_MEM_ALIGN);
  CHECK_PTR_ALIGN(params, 4, XA_NNLIB_FATAL_MEM_ALIGN);

  lstm = (lstm_state_t *) handle;

  switch(param_id)
  {
    case XA_NNLIB_LSTM_WEIGHT:
    {
      xa_nnlib_lstm_weights_t *p_weights;
      p_weights = (xa_nnlib_lstm_weights_t *)params;

      if(lstm->precision == XA_NNLIB_LSTM_16bx16b)
      {
          CHECK_MTX_SHAPE(p_weights->weights16.shape_w_xf, lstm->out_feats, lstm->in_feats)
          CHECK_MTX_SHAPE(p_weights->weights16.shape_w_xi, lstm->out_feats, lstm->in_feats)
          CHECK_MTX_SHAPE(p_weights->weights16.shape_w_xc, lstm->out_feats, lstm->in_feats)
          CHECK_MTX_SHAPE(p_weights->weights16.shape_w_xo, lstm->out_feats, lstm->in_feats)

          CHECK_MTX_SHAPE(p_weights->weights16.shape_w_hf, lstm->out_feats, lstm->out_feats)
          CHECK_MTX_SHAPE(p_weights->weights16.shape_w_hi, lstm->out_feats, lstm->out_feats)
          CHECK_MTX_SHAPE(p_weights->weights16.shape_w_hc, lstm->out_feats, lstm->out_feats)
          CHECK_MTX_SHAPE(p_weights->weights16.shape_w_ho, lstm->out_feats, lstm->out_feats)

          lstm->weights.weights16.w_xf = p_weights->weights16.w_xf;
          lstm->weights.weights16.w_xi = p_weights->weights16.w_xi;
          lstm->weights.weights16.w_xc = p_weights->weights16.w_xc;
          lstm->weights.weights16.w_xo = p_weights->weights16.w_xo;

          lstm->weights.weights16.w_hf = p_weights->weights16.w_hf;
          lstm->weights.weights16.w_hi = p_weights->weights16.w_hi;
          lstm->weights.weights16.w_hc = p_weights->weights16.w_hc;
          lstm->weights.weights16.w_ho = p_weights->weights16.w_ho;
      }
      else if(lstm->precision == XA_NNLIB_LSTM_8bx16b)
      {
          CHECK_MTX_SHAPE(p_weights->weights8.shape_w_xf, lstm->out_feats, lstm->in_feats)
          CHECK_MTX_SHAPE(p_weights->weights8.shape_w_xi, lstm->out_feats, lstm->in_feats)
          CHECK_MTX_SHAPE(p_weights->weights8.shape_w_xc, lstm->out_feats, lstm->in_feats)
          CHECK_MTX_SHAPE(p_weights->weights8.shape_w_xo, lstm->out_feats, lstm->in_feats)

          CHECK_MTX_SHAPE(p_weights->weights8.shape_w_hf, lstm->out_feats, lstm->out_feats)
          CHECK_MTX_SHAPE(p_weights->weights8.shape_w_hi, lstm->out_feats, lstm->out_feats)
          CHECK_MTX_SHAPE(p_weights->weights8.shape_w_hc, lstm->out_feats, lstm->out_feats)
          CHECK_MTX_SHAPE(p_weights->weights8.shape_w_ho, lstm->out_feats, lstm->out_feats)

          lstm->weights.weights8.w_xf = p_weights->weights8.w_xf;
          lstm->weights.weights8.w_xi = p_weights->weights8.w_xi;
          lstm->weights.weights8.w_xc = p_weights->weights8.w_xc;
          lstm->weights.weights8.w_xo = p_weights->weights8.w_xo;

          lstm->weights.weights8.w_hf = p_weights->weights8.w_hf;
          lstm->weights.weights8.w_hi = p_weights->weights8.w_hi;
          lstm->weights.weights8.w_hc = p_weights->weights8.w_hc;
          lstm->weights.weights8.w_ho = p_weights->weights8.w_ho;
      }
    }
    break;

    case XA_NNLIB_LSTM_BIAS:
    {
      xa_nnlib_lstm_biases_t *p_biases;
      p_biases = (xa_nnlib_lstm_biases_t *)params;

      CHECK_VEC_SHAPE(p_biases->shape_b_f, lstm->out_feats)
      CHECK_VEC_SHAPE(p_biases->shape_b_i, lstm->out_feats)
      CHECK_VEC_SHAPE(p_biases->shape_b_c, lstm->out_feats)
      CHECK_VEC_SHAPE(p_biases->shape_b_o, lstm->out_feats)

      lstm->biases.b_f = p_biases->b_f;
      lstm->biases.b_i = p_biases->b_i;
      lstm->biases.b_c = p_biases->b_c;
      lstm->biases.b_o = p_biases->b_o;
    }
    break;

    case XA_NNLIB_LSTM_RESTORE_CONTEXT_OUTPUT:
    {
      vect_t *prev_h;
      prev_h = (vect_t *)params;

      memcpy(lstm->prev_h,prev_h,lstm->out_feats * sizeof(vect_t));
    }
    break;

    case XA_NNLIB_LSTM_RESTORE_CONTEXT_CELL:
    {
      int *prev_c;
      prev_c = (int *)params;

      memcpy(lstm->prev_c,prev_c,lstm->out_feats * sizeof(int));
    }
    break;

    default:
    return XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_PARAM_ID;
  }

  return XA_NNLIB_NO_ERROR;
}

int xa_nnlib_lstm_get_config(
  xa_nnlib_handle_t handle,
  xa_nnlib_lstm_param_id_t param_id,
  void *params )
{
  lstm_state_t *lstm;

  CHECK_PTR(handle, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(params, XA_NNLIB_FATAL_MEM_ALLOC);

  CHECK_PTR_ALIGN(handle, 8, XA_NNLIB_FATAL_MEM_ALIGN);
  CHECK_PTR_ALIGN(params, 4, XA_NNLIB_FATAL_MEM_ALIGN);

  lstm = (lstm_state_t *) handle;

  switch(param_id)
  {
    case XA_NNLIB_LSTM_WEIGHT:
    {
      xa_nnlib_lstm_weights_t *p_weights;
      p_weights = (xa_nnlib_lstm_weights_t *)params;

      if(lstm->precision == XA_NNLIB_LSTM_16bx16b)
      {
          memcpy(&(p_weights->weights16.shape_w_xf), &(lstm->weights.weights16.shape_w_xf), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights16.shape_w_xi), &(lstm->weights.weights16.shape_w_xi), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights16.shape_w_xc), &(lstm->weights.weights16.shape_w_xc), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights16.shape_w_xo), &(lstm->weights.weights16.shape_w_xo), sizeof(xa_nnlib_shape_t));

          memcpy(&(p_weights->weights16.shape_w_hf), &(lstm->weights.weights16.shape_w_hf), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights16.shape_w_hi), &(lstm->weights.weights16.shape_w_hi), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights16.shape_w_hc), &(lstm->weights.weights16.shape_w_hc), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights16.shape_w_ho), &(lstm->weights.weights16.shape_w_ho), sizeof(xa_nnlib_shape_t));

          p_weights->weights16.w_xf = lstm->weights.weights16.w_xf;
          p_weights->weights16.w_xi = lstm->weights.weights16.w_xi;
          p_weights->weights16.w_xc = lstm->weights.weights16.w_xc;
          p_weights->weights16.w_xo = lstm->weights.weights16.w_xo;

          p_weights->weights16.w_hf = lstm->weights.weights16.w_hf;
          p_weights->weights16.w_hi = lstm->weights.weights16.w_hi;
          p_weights->weights16.w_hc = lstm->weights.weights16.w_hc;
          p_weights->weights16.w_ho = lstm->weights.weights16.w_ho;
      }
      else if(lstm->precision == XA_NNLIB_LSTM_8bx16b)
      {
          memcpy(&(p_weights->weights8.shape_w_xf), &(lstm->weights.weights8.shape_w_xf), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights8.shape_w_xi), &(lstm->weights.weights8.shape_w_xi), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights8.shape_w_xc), &(lstm->weights.weights8.shape_w_xc), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights8.shape_w_xo), &(lstm->weights.weights8.shape_w_xo), sizeof(xa_nnlib_shape_t));

          memcpy(&(p_weights->weights8.shape_w_hf), &(lstm->weights.weights8.shape_w_hf), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights8.shape_w_hi), &(lstm->weights.weights8.shape_w_hi), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights8.shape_w_hc), &(lstm->weights.weights8.shape_w_hc), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights8.shape_w_ho), &(lstm->weights.weights8.shape_w_ho), sizeof(xa_nnlib_shape_t));

          p_weights->weights8.w_xf = lstm->weights.weights8.w_xf;
          p_weights->weights8.w_xi = lstm->weights.weights8.w_xi;
          p_weights->weights8.w_xc = lstm->weights.weights8.w_xc;
          p_weights->weights8.w_xo = lstm->weights.weights8.w_xo;

          p_weights->weights8.w_hf = lstm->weights.weights8.w_hf;
          p_weights->weights8.w_hi = lstm->weights.weights8.w_hi;
          p_weights->weights8.w_hc = lstm->weights.weights8.w_hc;
          p_weights->weights8.w_ho = lstm->weights.weights8.w_ho;
      }
    }
    break;

    case XA_NNLIB_LSTM_BIAS:
    {
      xa_nnlib_lstm_biases_t *p_biases;
      p_biases = (xa_nnlib_lstm_biases_t *)params;

      memcpy(&(p_biases->shape_b_f), &(lstm->biases.shape_b_f), sizeof(xa_nnlib_shape_t));
      memcpy(&(p_biases->shape_b_i), &(lstm->biases.shape_b_i), sizeof(xa_nnlib_shape_t));
      memcpy(&(p_biases->shape_b_c), &(lstm->biases.shape_b_c), sizeof(xa_nnlib_shape_t));
      memcpy(&(p_biases->shape_b_o), &(lstm->biases.shape_b_o), sizeof(xa_nnlib_shape_t));

      p_biases->b_f = lstm->biases.b_f;
      p_biases->b_i = lstm->biases.b_i;
      p_biases->b_c = lstm->biases.b_c;
      p_biases->b_o = lstm->biases.b_o;
    }
    break;

    case XA_NNLIB_LSTM_INPUT_SHAPE:
    {
      xa_nnlib_shape_t *inp_shape;
      inp_shape = (xa_nnlib_shape_t *)params;
      inp_shape->dim.vector.length = lstm->in_feats;
      inp_shape->shape_type = SHAPE_VECTOR_T;
      inp_shape->n_shapes = 1;
      inp_shape->shape_offset = -1;
    }
    break;

    case XA_NNLIB_LSTM_OUTPUT_SHAPE:
    {
      xa_nnlib_shape_t *out_shape;
      out_shape = (xa_nnlib_shape_t *)params;
      out_shape->dim.vector.length = lstm->out_feats;
      out_shape->shape_type = SHAPE_VECTOR_T;
      out_shape->n_shapes = 1;
      out_shape->shape_offset = -1;
    }
    break;

    case XA_NNLIB_LSTM_RESTORE_CONTEXT_OUTPUT:
    {
      vect_t *prev_h;
      prev_h = (vect_t *)params;

      memcpy(prev_h,lstm->prev_h,lstm->out_feats * sizeof(vect_t));
    }
    break;

    case XA_NNLIB_LSTM_RESTORE_CONTEXT_CELL:
    {
      int *prev_c;
      prev_c = (int *)params;

      memcpy(prev_c,lstm->prev_c,lstm->out_feats * sizeof(int));
    }
    break;

    default:
    return XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_PARAM_ID;
  }

  return XA_NNLIB_NO_ERROR;
}

int xa_nnlib_lstm_process(xa_nnlib_handle_t handle,
    void *scratch,
    void *input,
    void *output,
    xa_nnlib_shape_t *p_in_shape,
    xa_nnlib_shape_t *p_out_shape)
{
  lstm_state_t *lstm;
  scratch_mem_t *scratch_mem;

  CHECK_PTR(handle, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(scratch, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(input, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(output, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(p_in_shape, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(p_out_shape, XA_NNLIB_FATAL_MEM_ALLOC);

  CHECK_PTR_ALIGN(handle, 8, XA_NNLIB_FATAL_MEM_ALIGN);
  CHECK_PTR_ALIGN(scratch, 8, XA_NNLIB_FATAL_MEM_ALIGN);
  CHECK_PTR_ALIGN(input, 8, XA_NNLIB_FATAL_MEM_ALIGN);
  CHECK_PTR_ALIGN(output, 8, XA_NNLIB_FATAL_MEM_ALIGN);
  CHECK_PTR_ALIGN(p_in_shape, 4, XA_NNLIB_FATAL_MEM_ALIGN);
  CHECK_PTR_ALIGN(p_out_shape, 4, XA_NNLIB_FATAL_MEM_ALIGN);

  CHECK_IO_SHAPE(p_in_shape);
  CHECK_IO_SHAPE(p_out_shape);

  lstm = (lstm_state_t *) handle;

  if(p_out_shape->dim.vector.length < lstm->out_feats)
  {
    return XA_NNLIB_LSTM_EXECUTE_FATAL_INSUFFICIENT_OUTPUT_BUFFER_SPACE;
  }

  if(p_in_shape->dim.vector.length < lstm->in_feats)
  {
    return XA_NNLIB_LSTM_EXECUTE_FATAL_INSUFFICIENT_DATA;
  }

  p_in_shape->dim.vector.length = lstm->in_feats;
  p_out_shape->dim.vector.length = lstm->out_feats;

  //setup scratch
  {
    char *sptr = (char *)scratch;

    scratch_alloc(sptr, scratch_mem,   scratch_mem_t,  1 );

    scratch_alloc(sptr, scratch_mem->f_f, vect_t, lstm->out_feats);
    scratch_alloc(sptr, scratch_mem->i_f_or_o_f, vect_t, lstm->out_feats);
    scratch_alloc(sptr, scratch_mem->c_hat_f_or_tanh_c_f, vect_t, lstm->out_feats);

#ifdef MODEL_FLT64
    scratch_mem->temp_mem.vec = NULL ;

#elif MODEL_INT16
    scratch_alloc(sptr, scratch_mem->temp_mem.vec, Int32, lstm->out_feats);

#endif
  }

#ifdef MODEL_INT16
  if(lstm->precision == XA_NNLIB_LSTM_16bx16b)
  {

    xa_nn_matXvec_16x16_16_sigmoid(
        scratch_mem->f_f,
        lstm->weights.weights16.w_xf,
        lstm->weights.weights16.w_hf,
        input,
        lstm->prev_h,
        lstm->biases.b_f,
        lstm->out_feats,
        lstm->in_feats,
        lstm->out_feats,
        lstm->in_feats + lstm->pad*XA_PAD_BYTES,
        lstm->out_feats + lstm->pad*XA_PAD_BYTES,
        lstm->matmul_lsh,
        lstm->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

    xa_nn_matXvec_16x16_16_sigmoid(
        scratch_mem->i_f_or_o_f,
        lstm->weights.weights16.w_xi,
        lstm->weights.weights16.w_hi,
        input,
        lstm->prev_h,
        lstm->biases.b_i,
        lstm->out_feats,
        lstm->in_feats,
        lstm->out_feats,
        lstm->in_feats + lstm->pad*XA_PAD_BYTES,
        lstm->out_feats + lstm->pad*XA_PAD_BYTES,
        lstm->matmul_lsh,
        lstm->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

    xa_nn_matXvec_16x16_16_tanh(
        scratch_mem->c_hat_f_or_tanh_c_f,
        lstm->weights.weights16.w_xc,
        lstm->weights.weights16.w_hc,
        input,
        lstm->prev_h,
        lstm->biases.b_c,
        lstm->out_feats,
        lstm->in_feats,
        lstm->out_feats,
        lstm->in_feats + lstm->pad*XA_PAD_BYTES,
        lstm->out_feats + lstm->pad*XA_PAD_BYTES,
        lstm->matmul_lsh,
        lstm->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

    vec_elem_mul_16x32plus16x16_16(
        lstm->prev_c,
        scratch_mem->f_f,
        lstm->prev_c,
        scratch_mem->i_f_or_o_f,
        scratch_mem->c_hat_f_or_tanh_c_f,
        lstm->fXprev_c_lsh,
        lstm->iXc_hat_lsh,
        lstm->out_feats);

    xa_nn_matXvec_16x16_16_sigmoid(
        scratch_mem->i_f_or_o_f,
        lstm->weights.weights16.w_xo,
        lstm->weights.weights16.w_ho,
        input,
        lstm->prev_h,
        lstm->biases.b_o,
        lstm->out_feats,
        lstm->in_feats,
        lstm->out_feats,
        lstm->in_feats + lstm->pad*XA_PAD_BYTES,
        lstm->out_feats + lstm->pad*XA_PAD_BYTES,
        lstm->matmul_lsh,
        lstm->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

    xa_nn_vec_tanh_32_16(
        scratch_mem->c_hat_f_or_tanh_c_f,
        lstm->prev_c,
        lstm->out_feats);

    lstm_output_kernel_16x16_16(
        (vect_t*)output,
        lstm->prev_h,
        scratch_mem->i_f_or_o_f,
        scratch_mem->c_hat_f_or_tanh_c_f,
        lstm->h_lsh,
        lstm->out_feats);

  }
  else if(lstm->precision == XA_NNLIB_LSTM_8bx16b)
  {

    xa_nn_matXvec_8x16_16_sigmoid(
        scratch_mem->f_f,
        lstm->weights.weights8.w_xf,
        lstm->weights.weights8.w_hf,
        input,
        lstm->prev_h,
        lstm->biases.b_f,
        lstm->out_feats,
        lstm->in_feats,
        lstm->out_feats,
        lstm->in_feats + lstm->pad*XA_PAD_BYTES,
        lstm->out_feats + lstm->pad*XA_PAD_BYTES,
        lstm->matmul_lsh,
        lstm->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

    xa_nn_matXvec_8x16_16_sigmoid(
        scratch_mem->i_f_or_o_f,
        lstm->weights.weights8.w_xi,
        lstm->weights.weights8.w_hi,
        input,
        lstm->prev_h,
        lstm->biases.b_i,
        lstm->out_feats,
        lstm->in_feats,
        lstm->out_feats,
        lstm->in_feats + lstm->pad*XA_PAD_BYTES,
        lstm->out_feats + lstm->pad*XA_PAD_BYTES,
        lstm->matmul_lsh,
        lstm->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

    xa_nn_matXvec_8x16_16_tanh(
        scratch_mem->c_hat_f_or_tanh_c_f,
        lstm->weights.weights8.w_xc,
        lstm->weights.weights8.w_hc,
        input,
        lstm->prev_h,
        lstm->biases.b_c,
        lstm->out_feats,
        lstm->in_feats,
        lstm->out_feats,
        lstm->in_feats + lstm->pad*XA_PAD_BYTES,
        lstm->out_feats + lstm->pad*XA_PAD_BYTES,
        lstm->matmul_lsh,
        lstm->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

    vec_elem_mul_16x32plus16x16_16(
        lstm->prev_c,
        scratch_mem->f_f,
        lstm->prev_c,
        scratch_mem->i_f_or_o_f,
        scratch_mem->c_hat_f_or_tanh_c_f,
        lstm->fXprev_c_lsh,
        lstm->iXc_hat_lsh,
        lstm->out_feats);

    xa_nn_matXvec_8x16_16_sigmoid(
        scratch_mem->i_f_or_o_f,
        lstm->weights.weights8.w_xo,
        lstm->weights.weights8.w_ho,
        input,
        lstm->prev_h,
        lstm->biases.b_o,
        lstm->out_feats,
        lstm->in_feats,
        lstm->out_feats,
        lstm->in_feats + lstm->pad*XA_PAD_BYTES,
        lstm->out_feats + lstm->pad*XA_PAD_BYTES,
        lstm->matmul_lsh,
        lstm->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

    xa_nn_vec_tanh_32_16(
        scratch_mem->c_hat_f_or_tanh_c_f,
        lstm->prev_c,
        lstm->out_feats);

    lstm_output_kernel_16x16_16(
        (vect_t*)output,
        lstm->prev_h,
        scratch_mem->i_f_or_o_f,
        scratch_mem->c_hat_f_or_tanh_c_f,
        lstm->h_lsh,
        lstm->out_feats);

  }
#endif

  return XA_NNLIB_NO_ERROR;
}
