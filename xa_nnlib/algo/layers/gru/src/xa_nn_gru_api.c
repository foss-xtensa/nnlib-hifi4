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
#include "xa_nnlib_gru_api.h"
#include "xa_nnlib_common_fpu.h"

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

extern void xa_nn_elm_mul_16x16_16(WORD16 * __restrict__ output, const WORD16 * __restrict__ input_1, const WORD16 * __restrict__ input_2, WORD32 num_elm);

typedef struct _gru_state_t
{
  void *prev_h;
  xa_nnlib_gru_weights_t weights;
  xa_nnlib_gru_biases_t biases;
  int in_feats;
  int out_feats;
  int pad;
  int precision;
  int bias_shift;
  int matmul_lsh;
  int tanh_lsh;
  int split_bias;
} gru_state_t;

typedef struct _temp_mem_t
{
  Int32 *vec;
} temp_mem_t;

typedef struct _scratch_mem_t
{
  vect_t *z_or_r;
  vect_t *r_x_prev_h;
  vect_t *h;
  Int32  *sum_part1;
  Int32  *sum_part2;
  temp_mem_t temp_mem;
} scratch_mem_t;

#if HAVE_VFPU
typedef struct _temp_mem_t_f32
{
  FLOAT32 *vec;
} temp_mem_t_f32;

typedef struct _scratch_mem_t_f32
{
  FLOAT32 *z_or_r;
  FLOAT32 *r_x_prev_h;
  FLOAT32 *h;
  FLOAT32  *sum_part1;
  FLOAT32  *sum_part2;
  temp_mem_t_f32 temp_mem;
} scratch_mem_t_f32;
#endif

static Int32 validate_config(xa_nnlib_gru_init_config_t *config)
{
  if(config->in_feats < 4 || config->in_feats > 2048 || (config->in_feats&3) != 0)
    return XA_NNLIB_GRU_CONFIG_FATAL_INVALID_IN_FEATS;

  if(config->out_feats < 4 || config->out_feats > 2048 || (config->out_feats&3) != 0)
    return XA_NNLIB_GRU_CONFIG_FATAL_INVALID_OUT_FEATS;

  if((config->precision != XA_NNLIB_GRU_16bx16b) && (config->precision != XA_NNLIB_GRU_8bx16b) && (config->precision != XA_NNLIB_GRU_flt32xflt32))
    return XA_NNLIB_GRU_CONFIG_FATAL_INVALID_PRECISION;

  if(config->coeff_Qformat < 0 || config->coeff_Qformat > 15)
    return XA_NNLIB_GRU_CONFIG_FATAL_INVALID_COEFF_QFORMAT;

  if(config->io_Qformat < 0 || config->io_Qformat > 15)
    return XA_NNLIB_GRU_CONFIG_FATAL_INVALID_IO_QFORMAT;

  if((config->pad !=0) && (config->pad != 1))
    return XA_NNLIB_GRU_CONFIG_FATAL_INVALID_MEMBANK_PADDING;

  if((config->split_bias !=0) && (config->split_bias != 1))
    return XA_NNLIB_GRU_CONFIG_FATAL_INVALID_SPLIT_BIAS;

  return XA_NNLIB_NO_ERROR;
}

static void apply_inplace_lsh(Int16 *dst_src, Int32 len, Int32 lsh)
{
  int itr = 0;
  ae_f16x4 _tmp_var_1;
  ae_f16x4 _tmp_var_2;
  Int16 _tmp_var_3;
  ae_f16x4 *ptr = NULL;

  for(itr = 0; itr < (len & ~(4-1)); itr += 4)
  {
    /* Loading 4 values at a time */
    ptr = (ae_f16x4 *) &dst_src[itr];
    _tmp_var_1 = *ptr;

    _tmp_var_2 = AE_SLAA16S(_tmp_var_1, lsh);
    *ptr = _tmp_var_2;
  }
  for(; itr < len; itr++)
  {
    /* Loading one value at a time */
    _tmp_var_3 = dst_src[itr];
    _tmp_var_1 = (ae_f16x4) _tmp_var_3;

    _tmp_var_3 = AE_SLAA16S(_tmp_var_1, lsh);
    dst_src[itr] = _tmp_var_3;
  }
}

Int32 xa_nnlib_gru_get_persistent_fast(
     xa_nnlib_gru_init_config_t *config )
{
  int persistent_size, ret;
  CHECK_PTR(config, XA_NNLIB_FATAL_MEM_ALLOC);

  ret = validate_config(config);
  if(ret != XA_NNLIB_NO_ERROR)
    return ret;

  persistent_size  = ALIGN_SIZE(sizeof(gru_state_t));
#if HAVE_VFPU
  if(config->precision == XA_NNLIB_GRU_flt32xflt32){
	  persistent_size += ALIGN_SIZE(config->out_feats * sizeof(FLOAT32));
  } 
  else 
#endif  
  {
      persistent_size += ALIGN_SIZE(config->out_feats * sizeof(vect_t));
  }

  return persistent_size;
}

Int32 xa_nnlib_gru_get_scratch_fast(
       xa_nnlib_gru_init_config_t *config )
{
  int scratch_size, ret;
  CHECK_PTR(config, XA_NNLIB_FATAL_MEM_ALLOC);

  ret = validate_config(config);
  if(ret != XA_NNLIB_NO_ERROR)
    return ret;

  scratch_size = ALIGN_SIZE(sizeof(scratch_mem_t));
#if HAVE_VFPU
  if(config->precision == XA_NNLIB_GRU_flt32xflt32) {
    scratch_size += 3 * ALIGN_SIZE(config->out_feats * sizeof(FLOAT32));
  } 
  else
#endif   
  {
    scratch_size += 3 * ALIGN_SIZE(config->out_feats * sizeof(vect_t));
  }

  if(config->split_bias == 1){
    /* For split bias implementation, two extra arrays are needed to hold intermediate sums */
    scratch_size += 2 * ALIGN_SIZE(config->out_feats * sizeof(Int32));
  }
#ifdef MODEL_FLT64
  scratch_size += 0;
#elif MODEL_INT16
  scratch_size += ALIGN_SIZE(1 * config->out_feats * sizeof(Int32));    //vect scratch
#endif

  return scratch_size;
}

int __attribute__((optimize ("-O0"))) xa_nnlib_gru_init(
    xa_nnlib_handle_t handle,
    xa_nnlib_gru_init_config_t *config )
{
  gru_state_t *gru;
  int ret;

  CHECK_PTR(handle, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(config, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR_ALIGN(handle, 8, XA_NNLIB_FATAL_MEM_ALIGN);

  ret = validate_config(config);
  if(ret != XA_NNLIB_NO_ERROR)
    return ret;

  gru = (gru_state_t *) handle;
  memset(gru,0, sizeof(gru_state_t));

  gru->in_feats   = config->in_feats;
  gru->out_feats  = config->out_feats;
  gru->pad        = config->pad;
  gru->precision  = config->precision;
  gru->bias_shift   = (config->io_Qformat + config->coeff_Qformat) - 15;
  gru->matmul_lsh = 25 - (config->coeff_Qformat + config->io_Qformat);  // Input to sigmoid function should be 6.25
  gru->tanh_lsh   = config->io_Qformat - 15;  // For Q15 to io_Qformat conversion
  gru->split_bias = config->split_bias;

  gru->prev_h = (void *)ALIGN_MEM((char *)handle + sizeof(gru_state_t));
#if HAVE_VFPU
  if(gru->precision == XA_NNLIB_GRU_flt32xflt32) {
	memset(gru->prev_h,0, config->out_feats * sizeof(FLOAT32));
  } 
  else 
#endif
  {
    memset(gru->prev_h,0, config->out_feats * sizeof(vect_t));
  }

  return XA_NNLIB_NO_ERROR;
}

int xa_nnlib_gru_set_config(
  xa_nnlib_handle_t handle,
  xa_nnlib_gru_param_id_t param_id,
  void *params )
{
  gru_state_t *gru;
  CHECK_PTR(handle, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(params, XA_NNLIB_FATAL_MEM_ALLOC);

  CHECK_PTR_ALIGN(handle, 8, XA_NNLIB_FATAL_MEM_ALIGN);
  CHECK_PTR_ALIGN(params, 4, XA_NNLIB_FATAL_MEM_ALIGN);

  gru = (gru_state_t *) handle;

  switch(param_id)
  {
    case XA_NNLIB_GRU_WEIGHT:
    {
      xa_nnlib_gru_weights_t *p_weights;
      p_weights = (xa_nnlib_gru_weights_t *)params;

      if(gru->precision == XA_NNLIB_GRU_16bx16b)
      {
          CHECK_MTX_SHAPE(p_weights->weights16.shape_w_z, gru->out_feats, gru->in_feats)
          CHECK_MTX_SHAPE(p_weights->weights16.shape_w_r, gru->out_feats, gru->in_feats)
          CHECK_MTX_SHAPE(p_weights->weights16.shape_w_h, gru->out_feats, gru->in_feats)

          CHECK_MTX_SHAPE(p_weights->weights16.shape_u_z, gru->out_feats, gru->out_feats)
          CHECK_MTX_SHAPE(p_weights->weights16.shape_u_r, gru->out_feats, gru->out_feats)
          CHECK_MTX_SHAPE(p_weights->weights16.shape_u_h, gru->out_feats, gru->out_feats)

          gru->weights.weights16.w_z = p_weights->weights16.w_z;
          gru->weights.weights16.u_z = p_weights->weights16.u_z;
          gru->weights.weights16.w_r = p_weights->weights16.w_r;
          gru->weights.weights16.u_r = p_weights->weights16.u_r;
          gru->weights.weights16.w_h = p_weights->weights16.w_h;
          gru->weights.weights16.u_h = p_weights->weights16.u_h;
      }
      else if(gru->precision == XA_NNLIB_GRU_8bx16b)
      {
          CHECK_MTX_SHAPE(p_weights->weights8.shape_w_z, gru->out_feats, gru->in_feats)
          CHECK_MTX_SHAPE(p_weights->weights8.shape_w_r, gru->out_feats, gru->in_feats)
          CHECK_MTX_SHAPE(p_weights->weights8.shape_w_h, gru->out_feats, gru->in_feats)

          CHECK_MTX_SHAPE(p_weights->weights8.shape_u_z, gru->out_feats, gru->out_feats)
          CHECK_MTX_SHAPE(p_weights->weights8.shape_u_r, gru->out_feats, gru->out_feats)
          CHECK_MTX_SHAPE(p_weights->weights8.shape_u_h, gru->out_feats, gru->out_feats)

          gru->weights.weights8.w_z = p_weights->weights8.w_z;
          gru->weights.weights8.u_z = p_weights->weights8.u_z;
          gru->weights.weights8.w_r = p_weights->weights8.w_r;
          gru->weights.weights8.u_r = p_weights->weights8.u_r;
          gru->weights.weights8.w_h = p_weights->weights8.w_h;
          gru->weights.weights8.u_h = p_weights->weights8.u_h;
      }
#if HAVE_VFPU
	  else if(gru->precision == XA_NNLIB_GRU_flt32xflt32)
      {
          CHECK_MTX_SHAPE(p_weights->weightsf32.shape_w_z, gru->out_feats, gru->in_feats)
          CHECK_MTX_SHAPE(p_weights->weightsf32.shape_w_r, gru->out_feats, gru->in_feats)
          CHECK_MTX_SHAPE(p_weights->weightsf32.shape_w_h, gru->out_feats, gru->in_feats)

          CHECK_MTX_SHAPE(p_weights->weightsf32.shape_u_z, gru->out_feats, gru->out_feats)
          CHECK_MTX_SHAPE(p_weights->weightsf32.shape_u_r, gru->out_feats, gru->out_feats)
          CHECK_MTX_SHAPE(p_weights->weightsf32.shape_u_h, gru->out_feats, gru->out_feats)

          gru->weights.weightsf32.w_z = p_weights->weightsf32.w_z;
          gru->weights.weightsf32.u_z = p_weights->weightsf32.u_z;
          gru->weights.weightsf32.w_r = p_weights->weightsf32.w_r;
          gru->weights.weightsf32.u_r = p_weights->weightsf32.u_r;
          gru->weights.weightsf32.w_h = p_weights->weightsf32.w_h;
          gru->weights.weightsf32.u_h = p_weights->weightsf32.u_h;
      }
#endif      
    }
    break;

    case XA_NNLIB_GRU_BIAS:
    {
      xa_nnlib_gru_biases_t *p_biases;
      p_biases = (xa_nnlib_gru_biases_t *)params;

      CHECK_VEC_SHAPE(p_biases->shape_b_z, gru->out_feats)
      CHECK_VEC_SHAPE(p_biases->shape_b_r, gru->out_feats)
      CHECK_VEC_SHAPE(p_biases->shape_b_h, gru->out_feats)

      gru->biases.b_z = p_biases->b_z;
      gru->biases.b_r = p_biases->b_r;
      gru->biases.b_h = p_biases->b_h;

      if(gru->split_bias){
        CHECK_VEC_SHAPE(p_biases->shape_bs_z, gru->out_feats)
        CHECK_VEC_SHAPE(p_biases->shape_bs_r, gru->out_feats)
        CHECK_VEC_SHAPE(p_biases->shape_bs_h, gru->out_feats)
        gru->biases.bs_z = p_biases->bs_z;
        gru->biases.bs_r = p_biases->bs_r;
        gru->biases.bs_h = p_biases->bs_h;
      }
    }
    break;

    case XA_NNLIB_GRU_RESTORE_CONTEXT:
    {
#if HAVE_VFPU
		if(gru->precision == XA_NNLIB_GRU_flt32xflt32){
            FLOAT32 *prev_h;
            prev_h = (FLOAT32 *)params;
            memcpy(gru->prev_h,prev_h,gru->out_feats * sizeof(FLOAT32));
		} 
    else 
#endif
    {
            vect_t *prev_h;
            prev_h = (vect_t *)params;
            memcpy(gru->prev_h,prev_h,gru->out_feats * sizeof(vect_t));
		}
    }
    break;

    default:
    return XA_NNLIB_GRU_CONFIG_FATAL_INVALID_PARAM_ID;
  }

  return XA_NNLIB_NO_ERROR;
}

int xa_nnlib_gru_get_config(
  xa_nnlib_handle_t handle,
  xa_nnlib_gru_param_id_t param_id,
  void *params )
{
  gru_state_t *gru;

  CHECK_PTR(handle, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(params, XA_NNLIB_FATAL_MEM_ALLOC);

  CHECK_PTR_ALIGN(handle, 8, XA_NNLIB_FATAL_MEM_ALIGN);
  CHECK_PTR_ALIGN(params, 4, XA_NNLIB_FATAL_MEM_ALIGN);

  gru = (gru_state_t *) handle;

  switch(param_id)
  {
    case XA_NNLIB_GRU_WEIGHT:
    {
      xa_nnlib_gru_weights_t *p_weights;
      p_weights = (xa_nnlib_gru_weights_t *)params;

      if(gru->precision == XA_NNLIB_GRU_16bx16b)
      {
          memcpy(&(p_weights->weights16.shape_w_z), &(gru->weights.weights16.shape_w_z), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights16.shape_u_z), &(gru->weights.weights16.shape_u_z), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights16.shape_w_r), &(gru->weights.weights16.shape_w_r), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights16.shape_u_r), &(gru->weights.weights16.shape_u_r), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights16.shape_w_h), &(gru->weights.weights16.shape_w_h), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights16.shape_u_h), &(gru->weights.weights16.shape_u_h), sizeof(xa_nnlib_shape_t));

          p_weights->weights16.w_z = gru->weights.weights16.w_z;
          p_weights->weights16.u_z = gru->weights.weights16.u_z;
          p_weights->weights16.w_r = gru->weights.weights16.w_r;
          p_weights->weights16.u_r = gru->weights.weights16.u_r;
          p_weights->weights16.w_h = gru->weights.weights16.w_h;
          p_weights->weights16.u_h = gru->weights.weights16.u_h;
      }
      else if(gru->precision == XA_NNLIB_GRU_8bx16b)
      {
          memcpy(&(p_weights->weights8.shape_w_z), &(gru->weights.weights8.shape_w_z), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights8.shape_u_z), &(gru->weights.weights8.shape_u_z), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights8.shape_w_r), &(gru->weights.weights8.shape_w_r), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights8.shape_u_r), &(gru->weights.weights8.shape_u_r), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights8.shape_w_h), &(gru->weights.weights8.shape_w_h), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weights8.shape_u_h), &(gru->weights.weights8.shape_u_h), sizeof(xa_nnlib_shape_t));

          p_weights->weights8.w_z = gru->weights.weights8.w_z;
          p_weights->weights8.u_z = gru->weights.weights8.u_z;
          p_weights->weights8.w_r = gru->weights.weights8.w_r;
          p_weights->weights8.u_r = gru->weights.weights8.u_r;
          p_weights->weights8.w_h = gru->weights.weights8.w_h;
          p_weights->weights8.u_h = gru->weights.weights8.u_h;
      }
#if HAVE_VFPU
	  else if(gru->precision == XA_NNLIB_GRU_flt32xflt32)
      {
          memcpy(&(p_weights->weightsf32.shape_w_z), &(gru->weights.weightsf32.shape_w_z), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weightsf32.shape_u_z), &(gru->weights.weightsf32.shape_u_z), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weightsf32.shape_w_r), &(gru->weights.weightsf32.shape_w_r), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weightsf32.shape_u_r), &(gru->weights.weightsf32.shape_u_r), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weightsf32.shape_w_h), &(gru->weights.weightsf32.shape_w_h), sizeof(xa_nnlib_shape_t));
          memcpy(&(p_weights->weightsf32.shape_u_h), &(gru->weights.weightsf32.shape_u_h), sizeof(xa_nnlib_shape_t));

          p_weights->weightsf32.w_z = gru->weights.weightsf32.w_z;
          p_weights->weightsf32.u_z = gru->weights.weightsf32.u_z;
          p_weights->weightsf32.w_r = gru->weights.weightsf32.w_r;
          p_weights->weightsf32.u_r = gru->weights.weightsf32.u_r;
          p_weights->weightsf32.w_h = gru->weights.weightsf32.w_h;
          p_weights->weightsf32.u_h = gru->weights.weightsf32.u_h;
      }
#endif
    }
    break;

    case XA_NNLIB_GRU_BIAS:
    {
      xa_nnlib_gru_biases_t *p_biases;
      p_biases = (xa_nnlib_gru_biases_t *)params;

      memcpy(&(p_biases->shape_b_z), &(gru->biases.shape_b_z), sizeof(xa_nnlib_shape_t));
      memcpy(&(p_biases->shape_b_r), &(gru->biases.shape_b_r), sizeof(xa_nnlib_shape_t));
      memcpy(&(p_biases->shape_b_h), &(gru->biases.shape_b_h), sizeof(xa_nnlib_shape_t));

      p_biases->b_z = gru->biases.b_z;
      p_biases->b_r = gru->biases.b_r;
      p_biases->b_h = gru->biases.b_h;

      if(gru->split_bias){
        memcpy(&(p_biases->shape_bs_z), &(gru->biases.shape_bs_z), sizeof(xa_nnlib_shape_t));
        memcpy(&(p_biases->shape_bs_r), &(gru->biases.shape_bs_r), sizeof(xa_nnlib_shape_t));
        memcpy(&(p_biases->shape_bs_h), &(gru->biases.shape_bs_h), sizeof(xa_nnlib_shape_t));
        p_biases->bs_z = gru->biases.bs_z;
        p_biases->bs_r = gru->biases.bs_r;
        p_biases->bs_h = gru->biases.bs_h;
      }
    }
    break;

    case XA_NNLIB_GRU_INPUT_SHAPE:
    {
      xa_nnlib_shape_t *inp_shape;
      inp_shape = (xa_nnlib_shape_t *)params;
      inp_shape->dim.vector.length = gru->in_feats;
      inp_shape->shape_type = SHAPE_VECTOR_T;
      inp_shape->n_shapes = 1;
      inp_shape->shape_offset = -1;
    }
    break;

    case XA_NNLIB_GRU_OUTPUT_SHAPE:
    {
      xa_nnlib_shape_t *out_shape;
      out_shape = (xa_nnlib_shape_t *)params;
      out_shape->dim.vector.length = gru->out_feats;
      out_shape->shape_type = SHAPE_VECTOR_T;
      out_shape->n_shapes = 1;
      out_shape->shape_offset = -1;
    }
    break;

    case XA_NNLIB_GRU_RESTORE_CONTEXT:
    {
#if HAVE_VFPU
	  if(gru->precision == XA_NNLIB_GRU_flt32xflt32) {
        FLOAT32 *prev_h;
        prev_h = (FLOAT32 *)params;
        memcpy(prev_h,gru->prev_h,gru->out_feats * sizeof(FLOAT32));
	  } 
    else 
#endif    
    {
        vect_t *prev_h;
        prev_h = (vect_t *)params;
        memcpy(prev_h,gru->prev_h,gru->out_feats * sizeof(vect_t));
	  }
    }
    break;

    default:
    return XA_NNLIB_GRU_CONFIG_FATAL_INVALID_PARAM_ID;
  }

  return XA_NNLIB_NO_ERROR;
}

static void internal_xa_nn_elm_mul_16x32_32(WORD32 * __restrict__ p_out,
                      const WORD16 * __restrict__ p_inp1,
                      const WORD32 * __restrict__ p_inp2,
                      WORD32 num_elm)
{
    int i;
    ae_f16x4 *inp1 = (ae_f16x4 *)p_inp1;
    ae_f32x2 *inp2 = (ae_f32x2 *)p_inp2;
    ae_f32x2 *out = (ae_f32x2 *)p_out;

    for(i=0;i < num_elm>>2;i++)
    {
        ae_f16x4 in1;
        ae_f32x2 in2_0, in2_1, out0, out1;
        
        AE_L16X4_IP(in1, inp1, 8);
        AE_L32X2_IP(in2_0, inp2, 8);
        AE_L32X2_IP(in2_1, inp2, 8);
        
        out0 = AE_MULFP32X16X2RS_H(in2_0, in1);
        out1 = AE_MULFP32X16X2RS_L(in2_1, in1);
        
        AE_S32X2_IP(out0, out, 8);
        AE_S32X2_IP(out1, out, 8);      
    }
}

static WORD32 internal_xa_nn_elm_add_32x32_32(WORD32 * __restrict__ p_out,
                               const WORD32 * __restrict__ p_inp1,
                               const WORD32 * __restrict__ p_inp2,
                               WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, 2*sizeof(WORD32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, 2*sizeof(WORD32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, 2*sizeof(WORD32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    int i;
    ae_int32x2 *inp1 = (ae_int32x2 *)p_inp1;
    ae_int32x2 *inp2 = (ae_int32x2 *)p_inp2;
    ae_int32x2 *out =  (ae_int32x2 *)p_out;
    ae_int32x2 x1, x2, y;

    for(i=0;i < num_elm>>1;i++)
    {
        AE_L32X2_IP(x1, inp1, 2*sizeof(WORD32));
        AE_L32X2_IP(x2, inp2, 2*sizeof(WORD32));
        y = AE_ADD32S(x1, x2);
        AE_S32X2_IP( y, out,  2*sizeof(WORD32));
    }

    return 0;
}

#if HAVE_VFPU
static WORD32 xa_nn_vec_interpolation_f32(FLOAT32 * __restrict__ p_out,
         const FLOAT32 * __restrict__ p_ifact,
         FLOAT32 * __restrict__ p_inp1,
         const FLOAT32 * __restrict__ p_inp2,
         WORD32 num_elements)
{
    XA_NNLIB_ARG_CHK_PTR(p_out,    -1);
    XA_NNLIB_ARG_CHK_PTR(p_ifact,  -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1,   -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2,   -1);
    XA_NNLIB_ARG_CHK_COND(((num_elements&3) != 0), -1);

    int i;

    xtfloatx2 *p_fi = (xtfloatx2 *)p_ifact;
    xtfloatx2 *p_si = (xtfloatx2 *)p_inp1;
    xtfloatx2 *p_ti = (xtfloatx2 *)p_inp2;
    xtfloatx2 *p_r  = (xtfloatx2 *)p_out, one;

    one = XT_SEL32_HH_SX2(1.0f, 1.0f);

    for(i=0; i<num_elements >> 1; i++)
    {
        p_si[i] = p_r[i]  = XT_ADD_SX2(XT_MUL_SX2(p_fi[i], p_si[i]), XT_MUL_SX2(one-p_fi[i], p_ti[i]));
    }

    return 0;
}
#endif

int xa_nnlib_gru_process(xa_nnlib_handle_t handle,
    void *scratch,
    void *input,
    void *output,
    xa_nnlib_shape_t *p_in_shape,
    xa_nnlib_shape_t *p_out_shape )
{
  gru_state_t *gru;
  scratch_mem_t *scratch_mem;
#if HAVE_VFPU
  scratch_mem_t_f32 *scratch_mem_f32;
#endif
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

  gru = (gru_state_t *) handle;

  if(p_out_shape->dim.vector.length < gru->out_feats)
  {
    return XA_NNLIB_GRU_EXECUTE_FATAL_INSUFFICIENT_OUTPUT_BUFFER_SPACE;
  }

  if(p_in_shape->dim.vector.length < gru->in_feats)
  {
    return XA_NNLIB_GRU_EXECUTE_FATAL_INSUFFICIENT_DATA;
  }

  p_in_shape->dim.vector.length = gru->in_feats;
  p_out_shape->dim.vector.length = gru->out_feats;

  //setup scratch
  {
    char *sptr = (char *)scratch;
#if HAVE_VFPU
    if(gru->precision == XA_NNLIB_GRU_flt32xflt32){
        scratch_alloc(sptr, scratch_mem_f32,   scratch_mem_t_f32,  1 );

        scratch_alloc(sptr, scratch_mem_f32->z_or_r, FLOAT32, gru->out_feats);
        scratch_alloc(sptr, scratch_mem_f32->r_x_prev_h, FLOAT32, gru->out_feats);
        scratch_alloc(sptr, scratch_mem_f32->h, FLOAT32, gru->out_feats);

        if(gru->split_bias == 1) {
          /* Additional storage sum_part1/2 are required only in split_bias case */
          scratch_alloc(sptr, scratch_mem_f32->sum_part1, FLOAT32, gru->out_feats);
          scratch_alloc(sptr, scratch_mem_f32->sum_part2, FLOAT32, gru->out_feats);
        }		
	} 
  else 
#endif
  {
        scratch_alloc(sptr, scratch_mem,   scratch_mem_t,  1 );

        scratch_alloc(sptr, scratch_mem->z_or_r, vect_t, gru->out_feats);
        scratch_alloc(sptr, scratch_mem->r_x_prev_h, vect_t, gru->out_feats);
        scratch_alloc(sptr, scratch_mem->h, vect_t, gru->out_feats);

        if(gru->split_bias == 1) {
          /* Additional storage sum_part1/2 are required only in split_bias case */
          scratch_alloc(sptr, scratch_mem->sum_part1, Int32, gru->out_feats);
          scratch_alloc(sptr, scratch_mem->sum_part2, Int32, gru->out_feats);
        }
	}

#ifdef MODEL_FLT64
    scratch_mem->temp_mem.vec = NULL ;

#elif MODEL_INT16
#if HAVE_VFPU
  if(gru->precision == XA_NNLIB_GRU_flt32xflt32){
        scratch_alloc(sptr, scratch_mem_f32->temp_mem.vec, FLOAT32, gru->out_feats);
	}
  else 
#endif  
  {
		scratch_alloc(sptr, scratch_mem->temp_mem.vec, Int32, gru->out_feats);
	}

#endif
  }

#ifdef MODEL_INT16
  if(gru->precision == XA_NNLIB_GRU_16bx16b)
  {

    if(gru->split_bias == 1){
      xa_nn_matXvec_16x16_32( scratch_mem->sum_part1,
        gru->weights.weights16.w_r, NULL,
        input, NULL, gru->biases.bs_r,
        gru->out_feats, gru->in_feats, 0,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 1), 0,
        gru->matmul_lsh, gru->bias_shift);
      xa_nn_matXvec_16x16_32( scratch_mem->sum_part2,
        gru->weights.weights16.u_r, NULL,
        gru->prev_h, NULL, gru->biases.b_r,
        gru->out_feats, gru->out_feats, 0,
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 1), 0,
        gru->matmul_lsh, gru->bias_shift);
      internal_xa_nn_elm_add_32x32_32(scratch_mem->temp_mem.vec, scratch_mem->sum_part1, scratch_mem->sum_part2, gru->out_feats);
      xa_nn_vec_sigmoid_32_16(scratch_mem->z_or_r, scratch_mem->temp_mem.vec, gru->out_feats); /* Compute r */

      
      xa_nn_matXvec_16x16_32( scratch_mem->sum_part1,
        gru->weights.weights16.u_h, NULL,
        gru->prev_h, NULL, gru->biases.b_h,
        gru->out_feats, gru->out_feats, 0,
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 1), 0,
        gru->matmul_lsh, gru->bias_shift);
      internal_xa_nn_elm_mul_16x32_32(scratch_mem->sum_part2, scratch_mem->z_or_r, scratch_mem->sum_part1, gru->out_feats);
      xa_nn_matXvec_16x16_32( scratch_mem->sum_part1,
        gru->weights.weights16.w_h, NULL,
        input, NULL, gru->biases.bs_h,
        gru->out_feats, gru->in_feats, 0,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 1), 0,
        gru->matmul_lsh, gru->bias_shift);
      internal_xa_nn_elm_add_32x32_32(scratch_mem->temp_mem.vec, scratch_mem->sum_part1, scratch_mem->sum_part2, gru->out_feats);
      xa_nn_vec_tanh_32_16(scratch_mem->h, scratch_mem->temp_mem.vec, gru->out_feats); /* compute h */
      
      apply_inplace_lsh(scratch_mem->h, gru->out_feats, gru->tanh_lsh);

       
      xa_nn_matXvec_16x16_32( scratch_mem->sum_part1,
        gru->weights.weights16.w_z, NULL,
        input, NULL, gru->biases.bs_z,
        gru->out_feats, gru->in_feats, 0,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 1), 0,
        gru->matmul_lsh, gru->bias_shift);
      xa_nn_matXvec_16x16_32( scratch_mem->sum_part2,
        gru->weights.weights16.u_z, NULL,
        gru->prev_h, NULL, gru->biases.b_z,
        gru->out_feats, gru->out_feats, 0,
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 1), 0,
        gru->matmul_lsh, gru->bias_shift);
      internal_xa_nn_elm_add_32x32_32(scratch_mem->temp_mem.vec, scratch_mem->sum_part1, scratch_mem->sum_part2, gru->out_feats);
      xa_nn_vec_sigmoid_32_16(scratch_mem->z_or_r, scratch_mem->temp_mem.vec, gru->out_feats); /* Compute z */

      xa_nn_vec_interpolation_q15((vect_t *)output,
        scratch_mem->z_or_r,
        gru->prev_h, scratch_mem->h, gru->out_feats);

    } else {

      xa_nn_matXvec_16x16_16_sigmoid(
        scratch_mem->z_or_r,
        gru->weights.weights16.w_r,
        gru->weights.weights16.u_r,
        input,
        gru->prev_h,
        gru->biases.b_r,
        gru->out_feats,
        gru->in_feats,
        gru->out_feats,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 1),
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 1),
        gru->matmul_lsh,
        gru->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

      xa_nn_elm_mul_16x16_16(scratch_mem->r_x_prev_h, scratch_mem->z_or_r, gru->prev_h, gru->out_feats);

      xa_nn_matXvec_16x16_16_tanh(
        scratch_mem->h,
        gru->weights.weights16.w_h,
        gru->weights.weights16.u_h,
        input,
        scratch_mem->r_x_prev_h,
        gru->biases.b_h,
        gru->out_feats,
        gru->in_feats,
        gru->out_feats,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 1),
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 1),
        gru->matmul_lsh,
        gru->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

      apply_inplace_lsh(scratch_mem->h, gru->out_feats, gru->tanh_lsh);

      xa_nn_matXvec_16x16_16_sigmoid(
        scratch_mem->z_or_r,
        gru->weights.weights16.w_z,
        gru->weights.weights16.u_z,
        input,
        gru->prev_h,
        gru->biases.b_z,
        gru->out_feats,
        gru->in_feats,
        gru->out_feats,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 1),
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 1),
        gru->matmul_lsh,
        gru->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

    //h_t step
      xa_nn_vec_interpolation_q15((vect_t *)output,
        scratch_mem->z_or_r,
        gru->prev_h,
        scratch_mem->h,
        gru->out_feats);
    }
  }
  else if(gru->precision == XA_NNLIB_GRU_8bx16b)
  {

    if(gru->split_bias == 1){
      xa_nn_matXvec_8x16_32( scratch_mem->sum_part1,
        gru->weights.weights8.w_r, NULL,
        input, NULL, gru->biases.bs_r,
        gru->out_feats, gru->in_feats, 0,
        gru->in_feats + gru->pad*XA_PAD_BYTES, 0,
        gru->matmul_lsh, gru->bias_shift);
      xa_nn_matXvec_8x16_32( scratch_mem->sum_part2,
        gru->weights.weights8.u_r, NULL,
        gru->prev_h, NULL, gru->biases.b_r,
        gru->out_feats, gru->out_feats, 0,
        gru->out_feats + gru->pad*XA_PAD_BYTES, 0,
        gru->matmul_lsh, gru->bias_shift);
      internal_xa_nn_elm_add_32x32_32(scratch_mem->temp_mem.vec, scratch_mem->sum_part1, scratch_mem->sum_part2, gru->out_feats);
      xa_nn_vec_sigmoid_32_16(scratch_mem->z_or_r, scratch_mem->temp_mem.vec, gru->out_feats); /* Compute r */

      
      xa_nn_matXvec_8x16_32( scratch_mem->sum_part1,
        gru->weights.weights8.u_h, NULL,
        gru->prev_h, NULL, gru->biases.b_h,
        gru->out_feats, gru->out_feats, 0,
        gru->out_feats + gru->pad*XA_PAD_BYTES, 0,
        gru->matmul_lsh, gru->bias_shift);
      internal_xa_nn_elm_mul_16x32_32(scratch_mem->sum_part2, scratch_mem->z_or_r, scratch_mem->sum_part1, gru->out_feats);
      xa_nn_matXvec_8x16_32( scratch_mem->sum_part1,
        gru->weights.weights8.w_h, NULL,
        input, NULL, gru->biases.bs_h,
        gru->out_feats, gru->in_feats, 0,
        gru->in_feats + gru->pad*XA_PAD_BYTES, 0,
        gru->matmul_lsh, gru->bias_shift);
      internal_xa_nn_elm_add_32x32_32(scratch_mem->temp_mem.vec, scratch_mem->sum_part1, scratch_mem->sum_part2, gru->out_feats);
      xa_nn_vec_tanh_32_16(scratch_mem->h, scratch_mem->temp_mem.vec, gru->out_feats); /* compute h */
      
      apply_inplace_lsh(scratch_mem->h, gru->out_feats, gru->tanh_lsh);

       
      xa_nn_matXvec_8x16_32( scratch_mem->sum_part1,
        gru->weights.weights8.w_z, NULL,
        input, NULL, gru->biases.bs_z,
        gru->out_feats, gru->in_feats, 0,
        gru->in_feats + gru->pad*XA_PAD_BYTES, 0,
        gru->matmul_lsh, gru->bias_shift);
      xa_nn_matXvec_8x16_32( scratch_mem->sum_part2,
        gru->weights.weights8.u_z, NULL,
        gru->prev_h, NULL, gru->biases.b_z,
        gru->out_feats, gru->out_feats, 0,
        gru->out_feats + gru->pad*XA_PAD_BYTES, 0,
        gru->matmul_lsh, gru->bias_shift);
      internal_xa_nn_elm_add_32x32_32(scratch_mem->temp_mem.vec, scratch_mem->sum_part1, scratch_mem->sum_part2, gru->out_feats);
      xa_nn_vec_sigmoid_32_16(scratch_mem->z_or_r, scratch_mem->temp_mem.vec, gru->out_feats); /* Compute z */

      xa_nn_vec_interpolation_q15((vect_t *)output,
        scratch_mem->z_or_r,
        gru->prev_h, scratch_mem->h, gru->out_feats);

    } else {
      xa_nn_matXvec_8x16_16_sigmoid(
        scratch_mem->z_or_r,
        gru->weights.weights8.w_r,
        gru->weights.weights8.u_r,
        input,
        gru->prev_h,
        gru->biases.b_r,
        gru->out_feats,
        gru->in_feats,
        gru->out_feats,
        gru->in_feats + gru->pad*XA_PAD_BYTES,
        gru->out_feats + gru->pad*XA_PAD_BYTES,
        gru->matmul_lsh,
        gru->bias_shift,
        16,
        scratch_mem->temp_mem.vec);
      xa_nn_elm_mul_16x16_16(scratch_mem->r_x_prev_h, scratch_mem->z_or_r, gru->prev_h, gru->out_feats);

      xa_nn_matXvec_8x16_16_tanh(
        scratch_mem->h,
        gru->weights.weights8.w_h,
        gru->weights.weights8.u_h,
        input,
        scratch_mem->r_x_prev_h,
        gru->biases.b_h,
        gru->out_feats,
        gru->in_feats,
        gru->out_feats,
        gru->in_feats + gru->pad*XA_PAD_BYTES,
        gru->out_feats + gru->pad*XA_PAD_BYTES,
        gru->matmul_lsh,
        gru->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

      apply_inplace_lsh(scratch_mem->h, gru->out_feats, gru->tanh_lsh);

      xa_nn_matXvec_8x16_16_sigmoid(
        scratch_mem->z_or_r,
        gru->weights.weights8.w_z,
        gru->weights.weights8.u_z,
        input,
        gru->prev_h,
        gru->biases.b_z,
        gru->out_feats,
        gru->in_feats,
        gru->out_feats,
        gru->in_feats + gru->pad*XA_PAD_BYTES,
        gru->out_feats + gru->pad*XA_PAD_BYTES,
        gru->matmul_lsh,
        gru->bias_shift,
        16,
        scratch_mem->temp_mem.vec);

      //h_t step
      xa_nn_vec_interpolation_q15((vect_t *)output,
        scratch_mem->z_or_r,
        gru->prev_h,
        scratch_mem->h,
        gru->out_feats);
    }
  }
#if HAVE_VFPU
  else if(gru->precision == XA_NNLIB_GRU_flt32xflt32)
  {
    if(gru->split_bias == 1){
      xa_nn_matXvec_f32xf32_f32( scratch_mem_f32->sum_part1,
        gru->weights.weightsf32.w_r, NULL,
        input, NULL, gru->biases.bs_r,
        gru->out_feats, gru->in_feats, 0,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 2), 0);
      xa_nn_matXvec_f32xf32_f32( scratch_mem_f32->sum_part2,
        gru->weights.weightsf32.u_r, NULL,
        gru->prev_h, NULL, gru->biases.b_r,
        gru->out_feats, gru->out_feats, 0,
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 2), 0);
      xa_nn_elm_add_f32xf32_f32(scratch_mem_f32->temp_mem.vec, scratch_mem_f32->sum_part1, scratch_mem_f32->sum_part2, gru->out_feats);
      xa_nn_vec_sigmoid_f32_f32(scratch_mem_f32->z_or_r, scratch_mem_f32->temp_mem.vec, gru->out_feats); /* Compute r */

      xa_nn_matXvec_f32xf32_f32( scratch_mem_f32->sum_part1,
        gru->weights.weightsf32.u_h, NULL,
        gru->prev_h, NULL, gru->biases.b_h,
        gru->out_feats, gru->out_feats, 0,
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 2), 0);
      xa_nn_elm_mul_f32xf32_f32(scratch_mem_f32->sum_part2, scratch_mem_f32->z_or_r, scratch_mem_f32->sum_part1, gru->out_feats);
      xa_nn_matXvec_f32xf32_f32( scratch_mem_f32->sum_part1,
        gru->weights.weightsf32.w_h, NULL,
        input, NULL, gru->biases.bs_h,
        gru->out_feats, gru->in_feats, 0,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 2), 0);
      xa_nn_elm_add_f32xf32_f32(scratch_mem_f32->temp_mem.vec, scratch_mem_f32->sum_part1, scratch_mem_f32->sum_part2, gru->out_feats);
      xa_nn_vec_tanh_f32_f32(scratch_mem_f32->h, scratch_mem_f32->temp_mem.vec, gru->out_feats); /* compute h */

      xa_nn_matXvec_f32xf32_f32( scratch_mem_f32->sum_part1,
        gru->weights.weightsf32.w_z, NULL,
        input, NULL, gru->biases.bs_z,
        gru->out_feats, gru->in_feats, 0,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 2), 0);
      xa_nn_matXvec_f32xf32_f32( scratch_mem_f32->sum_part2,
        gru->weights.weightsf32.u_z, NULL,
        gru->prev_h, NULL, gru->biases.b_z,
        gru->out_feats, gru->out_feats, 0,
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 2), 0);
      xa_nn_elm_add_f32xf32_f32(scratch_mem_f32->temp_mem.vec, scratch_mem_f32->sum_part1, scratch_mem_f32->sum_part2, gru->out_feats);
      xa_nn_vec_sigmoid_f32_f32(scratch_mem_f32->z_or_r, scratch_mem_f32->temp_mem.vec, gru->out_feats); /* Compute z */

      xa_nn_vec_interpolation_f32((FLOAT32 *)output,
        scratch_mem_f32->z_or_r,
        gru->prev_h, scratch_mem_f32->h, gru->out_feats);

    } 
    else {
      xa_nn_matXvec_f32xf32_f32_sigmoid(
        scratch_mem_f32->z_or_r,
        gru->weights.weightsf32.w_r,
        gru->weights.weightsf32.u_r,
        input,
        gru->prev_h,
        gru->biases.b_r,
        gru->out_feats,
        gru->in_feats,
        gru->out_feats,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 2),
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 2),
        scratch_mem_f32->temp_mem.vec);
      xa_nn_elm_mul_f32xf32_f32(scratch_mem_f32->r_x_prev_h, scratch_mem_f32->z_or_r, gru->prev_h, gru->out_feats);

      xa_nn_matXvec_f32xf32_f32_tanh(
        scratch_mem_f32->h,
        gru->weights.weightsf32.w_h,
        gru->weights.weightsf32.u_h,
        input,
        scratch_mem_f32->r_x_prev_h,
        gru->biases.b_h,
        gru->out_feats,
        gru->in_feats,
        gru->out_feats,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 2),
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 2),
        scratch_mem_f32->temp_mem.vec);

     // apply_inplace_lsh(scratch_mem_f32->h, gru->out_feats, gru->tanh_lsh); //Not needed

      xa_nn_matXvec_f32xf32_f32_sigmoid(
        scratch_mem_f32->z_or_r,
        gru->weights.weightsf32.w_z,
        gru->weights.weightsf32.u_z,
        input,
        gru->prev_h,
        gru->biases.b_z,
        gru->out_feats,
        gru->in_feats,
        gru->out_feats,
        gru->in_feats + (gru->pad*XA_PAD_BYTES >> 2),
        gru->out_feats + (gru->pad*XA_PAD_BYTES >> 2),
        scratch_mem_f32->temp_mem.vec);

      //h_t step
      xa_nn_vec_interpolation_f32((FLOAT32 *)output,
        scratch_mem_f32->z_or_r,
        gru->prev_h,
        scratch_mem_f32->h,
        gru->out_feats);
    }
  }
#endif
#endif

  return XA_NNLIB_NO_ERROR;
}
