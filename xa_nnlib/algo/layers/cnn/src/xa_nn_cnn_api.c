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
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_cnn_api.h"

#define ALIGN_SIZE(n) (((n)+7)&(~7))
#define CHECK_PTR(ptr, err) if(NULL == ptr) return err;
#define CHECK_PTR_ALIGN(ptr, alignment, err) if((((unsigned)(ptr))&(alignment-1)) != 0) return err;

#define  IO_PRECISION_BITS(prec) ((prec == XA_NNLIB_CNN_16bx16b || prec == XA_NNLIB_CNN_8bx16b) ? 16 : ((prec == XA_NNLIB_CNN_8bx8b)   ?  8 : -1))
#define  KER_PRECISION_BITS(prec) ((prec == XA_NNLIB_CNN_8bx8b || prec == XA_NNLIB_CNN_8bx16b) ? 8 : ((prec == XA_NNLIB_CNN_16bx16b)   ?  16 : -1))
#define  IO_PRECISION_BYTES(prec) ((prec == XA_NNLIB_CNN_16bx16b || prec == XA_NNLIB_CNN_8bx16b) ?  2 : ((prec == XA_NNLIB_CNN_8bx8b)   ?  1 : 4))

/* HiFi_LE: clang warning - always true condition checks
Changed type in comparisons to shape.shape_type as they are verfied to be equal on first check */
#define CHECK_CUBE_DIMS(shape, type, err)                                               \
do {                                                                                    \
  if((shape.shape_type != type) ||                                                      \
     (shape.n_shapes != 1)      ||                                                      \
     (shape.shape_offset != -1))                                                        \
    return err;                                                                         \
  if(shape.dim.cube.height <= 0 ||                                                      \
     shape.dim.cube.width <= 0  ||                                                      \
     shape.dim.cube.depth <= 0)                                                         \
    return err;                                                                         \
  if(shape.shape_type == SHAPE_CUBE_DWH_T)                                              \
  {                                                                                     \
     if(shape.dim.cube.depth_offset != 1 ||                                             \
        shape.dim.cube.width_offset != shape.dim.cube.depth ||                          \
        shape.dim.cube.height_offset != (shape.dim.cube.depth*shape.dim.cube.width))    \
       return err;                                                                      \
  }                                                                                     \
  if(shape.shape_type == SHAPE_CUBE_WHD_T)                                              \
  {                                                                                     \
     if(shape.dim.cube.width_offset != 1 ||                                             \
        shape.dim.cube.height_offset != shape.dim.cube.width ||                         \
        shape.dim.cube.depth_offset != (shape.dim.cube.width*shape.dim.cube.height))    \
       return err;                                                                      \
  }                                                                                     \
} while(0)

#define CHECK_KERNEL_CUBE_DIMS(shape, type, err)                                        \
do {                                                                                    \
  if(shape.shape_type != type)                                                          \
    return err;                                                                         \
  if(shape.dim.cube.height <= 0 ||                                                      \
     shape.dim.cube.width <= 0  ||                                                      \
     shape.dim.cube.depth <= 0)                                                         \
    return err;                                                                         \
  if((shape.shape_type == SHAPE_CUBE_DWH_T) && (shape.dim.cube.depth_offset != 1))      \
    return err;                                                                         \
  if((shape.shape_type == SHAPE_CUBE_WHD_T) && (shape.dim.cube.width_offset != 1))      \
    return err;                                                                         \
} while(0)

#define FILL_SHAPE_MATRIX(shape, rows_shape, cols_shape)  \
{                                                         \
  shape.shape_type = SHAPE_MATRIX_T;                      \
  shape.dim.matrix.rows = rows_shape;                     \
  shape.dim.matrix.cols = cols_shape;                     \
  shape.dim.matrix.row_offset = cols_shape;               \
  shape.n_shapes = 1;                                     \
  shape.shape_offset = -1;                                \
}

#define FILL_SHAPE_CUBE(shape, nheight, nwidth, ndepth, type)     \
{                                                                 \
  shape.shape_type = type;                                        \
  shape.dim.cube.height = nheight;                                \
  shape.dim.cube.width = nwidth;                                  \
  shape.dim.cube.depth = ndepth;                                  \
  if (shape.shape_type == SHAPE_CUBE_DWH_T)                       \
  {                                                               \
    shape.dim.cube.depth_offset = 1;                              \
    shape.dim.cube.height_offset = ndepth*nwidth;                 \
    shape.dim.cube.width_offset = ndepth;                         \
  }                                                               \
  else if(shape.shape_type == SHAPE_CUBE_WHD_T)                   \
  {                                                               \
    shape.dim.cube.depth_offset = nwidth*nheight;                 \
    shape.dim.cube.height_offset = nwidth;                        \
    shape.dim.cube.width_offset = 1;                              \
  }                                                               \
  shape.n_shapes = 1;                                             \
  shape.shape_offset = -1;                                        \
}

#define MATCH_CUBE_DIMS(p_shape, ref_shape, err)                                        \
do {                                                                                    \
  if((p_shape->shape_type != ref_shape.shape_type)                           ||         \
     (p_shape->n_shapes != ref_shape.n_shapes)                               ||         \
     (p_shape->shape_offset != ref_shape.shape_offset)                       ||         \
     (p_shape->dim.cube.height != ref_shape.dim.cube.height)                 ||         \
     (p_shape->dim.cube.width != ref_shape.dim.cube.width)                   ||         \
     (p_shape->dim.cube.depth != ref_shape.dim.cube.depth)                   ||         \
     (p_shape->dim.cube.height_offset != ref_shape.dim.cube.height_offset)   ||         \
     (p_shape->dim.cube.width_offset != ref_shape.dim.cube.width_offset)     ||         \
     (p_shape->dim.cube.depth_offset != ref_shape.dim.cube.depth_offset))               \
    return err;                                                                         \
} while(0)

typedef struct _cnn_state_t
{
  xa_nnlib_cnn_init_config_t config;

  xa_nnlib_shape_t output_shape;

  void *kernel_std;
  void *kernel_ds_depth;
  void *kernel_ds_point;

  void *bias_std;
  void *bias_ds_depth;
  void *bias_ds_point;

} cnn_state_t;

typedef struct _temp_mem_t
{
  Int32 *vec;
} temp_mem_t;

typedef struct _scratch_mem_t
{
  vect_t *z_or_r;
  vect_t *r_x_prev_h;
  vect_t *h;
  temp_mem_t temp_mem;
} scratch_mem_t;

static Int32 validate_config(xa_nnlib_cnn_init_config_t *config)
{
  if((config->algo != XA_NNLIB_CNN_CONV1D_STD) &&
     (config->algo != XA_NNLIB_CNN_CONV2D_STD) &&
     (config->algo != XA_NNLIB_CNN_CONV2D_DS))
    return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_ALGO;

  if((config->precision != XA_NNLIB_CNN_16bx16b)
     && (config->precision != XA_NNLIB_CNN_8bx16b)
     && (config->precision != XA_NNLIB_CNN_8bx8b)
#if HAVE_VFPU
     && (config->precision != XA_NNLIB_CNN_f32xf32)
#endif
    )
    return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PRECISION;

  if(config->precision != XA_NNLIB_CNN_f32xf32)
  {
    if(config->bias_shift < -31 || config->bias_shift > 31)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_BIAS_SHIFT;

    if(config->acc_shift < -31 || config->acc_shift > 31)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_ACC_SHIFT;
  }

  if(config->x_stride <= 0 || config->y_stride <= 0)
    return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_STRIDE;

  if(config->x_padding < 0 || config->y_padding < 0)
    return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PADDING;

  if((config->output_height <= 0) || (config->output_width <= 0) || (config->output_channels <= 0))
    return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_OUTPUT_SHAPE;

  if(config->algo == XA_NNLIB_CNN_CONV1D_STD || config->algo == XA_NNLIB_CNN_CONV2D_STD)
  {
    CHECK_CUBE_DIMS(config->input_shape,    SHAPE_CUBE_DWH_T, XA_NNLIB_CNN_CONFIG_FATAL_INVALID_INPUT_SHAPE);
    CHECK_KERNEL_CUBE_DIMS(config->kernel_std_shape, SHAPE_CUBE_DWH_T, XA_NNLIB_CNN_CONFIG_FATAL_INVALID_KERNEL_SHAPE);
    if(config->bias_std_shape.shape_type != SHAPE_VECTOR_T)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_BIAS_SHAPE;

    /* kernel_std_depth must match input_depth */
    if(config->input_shape.dim.cube.depth != config->kernel_std_shape.dim.cube.depth)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PARAM_COMBINATION;

    /* output and bias dimensions check */
    if(config->output_channels != config->bias_std_shape.dim.vector.length)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PARAM_COMBINATION;

    /* x_stride must be <= kernel_std_width */
    if(config->x_stride > config->kernel_std_shape.dim.cube.width)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_STRIDE;

    /* output data format must be 0 (DWH) or 1 (WHD) */
    if ((config->output_format != 0) && (config->output_format != 1))
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_OUTPUT_SHAPE;

    if(config->algo == XA_NNLIB_CNN_CONV1D_STD)
    {
      /* output width must be 1 */
      if(config->output_width != 1)
        return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_OUTPUT_SHAPE;

      /* kernel_std_width must match input_width */
      if(config->input_shape.dim.cube.width != config->kernel_std_shape.dim.cube.width)
        return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PARAM_COMBINATION;

      /* kernel_std must be padded so as depth*width is multiple of 2 (float) or 4 (fixed) */
      if(config->precision == XA_NNLIB_CNN_f32xf32)
      {
        if(((config->kernel_std_shape.dim.cube.depth * config->kernel_std_shape.dim.cube.width + 1) & (~1)) != config->kernel_std_shape.dim.cube.height_offset)
          return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_KERNEL_SHAPE;
      }
      else
      {
        if(((config->kernel_std_shape.dim.cube.depth * config->kernel_std_shape.dim.cube.width + 3) & (~3)) != config->kernel_std_shape.dim.cube.height_offset)
          return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_KERNEL_SHAPE;
      }
    }
    else /* (config->algo == XA_NNLIB_CNN_CONV2D_STD) */
    {
      /* kernel_std must be padded so as depth is multiple of 2 (float) or 4 (fixed) */
      if(config->precision == XA_NNLIB_CNN_f32xf32)
      {
        if(((config->kernel_std_shape.dim.cube.depth+1)&(~1)) != config->kernel_std_shape.dim.cube.width_offset)
          return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_KERNEL_SHAPE;
      }
      else
      {
        if(((config->kernel_std_shape.dim.cube.depth+3)&(~3)) != config->kernel_std_shape.dim.cube.width_offset)
          return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_KERNEL_SHAPE;
      }

      /* x_stride must be <= kernel_std_width */
      if(config->x_stride > config->kernel_std_shape.dim.cube.width)
        return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_STRIDE;
    }
  }
  else if(config->algo == XA_NNLIB_CNN_CONV2D_DS)
  {
    CHECK_CUBE_DIMS(config->input_shape,  SHAPE_CUBE_WHD_T, XA_NNLIB_CNN_CONFIG_FATAL_INVALID_INPUT_SHAPE);
    CHECK_KERNEL_CUBE_DIMS(config->kernel_ds_depth_shape, SHAPE_CUBE_WHD_T, XA_NNLIB_CNN_CONFIG_FATAL_INVALID_KERNEL_SHAPE);

    /* output data format must be 1 (WHD) */
    if (config->output_format != 1)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_OUTPUT_SHAPE;

    if(config->bias_ds_depth_shape.shape_type != SHAPE_VECTOR_T)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_BIAS_SHAPE;

    if(config->kernel_ds_point_shape.shape_type != SHAPE_MATRIX_T)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_KERNEL_SHAPE;

    if(config->bias_ds_point_shape.shape_type != SHAPE_VECTOR_T)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_BIAS_SHAPE;

    /* input_channels * channels_multiplier must be multiple of 4 */
    if((config->input_shape.dim.cube.depth * config->channels_multiplier) & 3)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_INPUT_SHAPE;

    /* output_depth and bias_depth dimensions check */
    if(config->bias_ds_depth_shape.dim.vector.length != config->kernel_ds_depth_shape.dim.cube.depth)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PARAM_COMBINATION;

    /* output_point and bias_point dimensions check */
    if(config->bias_ds_point_shape.dim.vector.length != config->output_channels)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PARAM_COMBINATION;

    /* y_stride must be <= kernel_ds_depth_height */
    if(config->y_stride > config->kernel_ds_depth_shape.dim.cube.height)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_STRIDE;

    /* kernel_ds_depth_width must be padded to be multiple of 4 */
    if(((config->kernel_ds_depth_shape.dim.cube.width+3)&(~3)) != config->kernel_ds_depth_shape.dim.cube.height_offset)
      return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_KERNEL_SHAPE;
  }

  return XA_NNLIB_NO_ERROR;
}

Int32 xa_nnlib_cnn_get_persistent_fast(
     xa_nnlib_cnn_init_config_t *config )
{
  int persistent_size, ret;
  CHECK_PTR(config, XA_NNLIB_FATAL_MEM_ALLOC);

  ret = validate_config(config);
  if(ret != XA_NNLIB_NO_ERROR)
    return ret;

  persistent_size  = ALIGN_SIZE(sizeof(cnn_state_t));

  return persistent_size;
}

Int32 xa_nnlib_cnn_get_scratch_fast(
       xa_nnlib_cnn_init_config_t *config )
{
  int scratch_size = 0, ret;
  int inp_precision,ker_precision, out_bytewidth;
  CHECK_PTR(config, XA_NNLIB_FATAL_MEM_ALLOC);

  ret = validate_config(config);
  if(ret != XA_NNLIB_NO_ERROR)
    return ret;

  inp_precision = IO_PRECISION_BITS(config->precision);
  out_bytewidth = IO_PRECISION_BYTES(config->precision);

  if(config->algo == XA_NNLIB_CNN_CONV1D_STD)
  {
     scratch_size = xa_nn_conv1d_std_getsize(config->kernel_std_shape.dim.cube.height,
                                             config->input_shape.dim.cube.width,
                                             config->input_shape.dim.cube.depth,
                                             inp_precision);
  }
  else if(config->algo == XA_NNLIB_CNN_CONV2D_STD)
  {
     ker_precision=KER_PRECISION_BITS(config->precision);
     scratch_size = xa_nn_conv2d_std_getsize(config->input_shape.dim.cube.height,
                                             config->input_shape.dim.cube.width,
                                             config->input_shape.dim.cube.depth,
                                             config->kernel_std_shape.dim.cube.height,
                                             config->kernel_std_shape.dim.cube.width,
                                             config->kernel_std_shape.dim.cube.depth,
                                             config->y_stride,
                                             config->y_padding,
                                             config->x_stride,
                                             config->x_padding,
                                             config->output_height,
                                             config->output_width,
                                             config->output_channels,
                                             inp_precision,
                                             ker_precision,
                                             1,
                                             1,
                                             config->output_format
                                             );
  }
  else if(config->algo == XA_NNLIB_CNN_CONV2D_DS)
  {
    scratch_size = xa_nn_conv2d_depthwise_getsize(config->input_shape.dim.cube.height,
                                                  config->input_shape.dim.cube.width,
                                                  config->input_shape.dim.cube.depth,
                                                  config->kernel_ds_depth_shape.dim.cube.height,
                                                  config->kernel_ds_depth_shape.dim.cube.width,
                                                  config->channels_multiplier,
                                                  config->x_stride,
                                                  config->y_stride,
                                                  config->x_padding,
                                                  config->y_padding,
                                                  config->output_height,
                                                  config->output_width,
                                                  inp_precision,
                                                  1);   // WHD supported for all precisions
    scratch_size  = ALIGN_SIZE(scratch_size);
    scratch_size += out_bytewidth * config->output_height * config->output_width * config->input_shape.dim.cube.depth * config->channels_multiplier;
  }

  return scratch_size;
}

int __attribute__((optimize ("-O0"))) xa_nnlib_cnn_init(
    xa_nnlib_handle_t handle,
    xa_nnlib_cnn_init_config_t *config )
{
  cnn_state_t *cnn;
  int ret;

  CHECK_PTR(handle, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(config, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR_ALIGN(handle, 8, XA_NNLIB_FATAL_MEM_ALIGN);

  ret = validate_config(config);
  if(ret != XA_NNLIB_NO_ERROR)
    return ret;

  cnn = (cnn_state_t *) handle;
  memset(cnn,0, sizeof(cnn_state_t));

  memcpy(&cnn->config, config, sizeof(xa_nnlib_cnn_init_config_t));

  if(config->algo == XA_NNLIB_CNN_CONV1D_STD)
  {
    if(config->output_format == 0)
    {
      FILL_SHAPE_MATRIX(cnn->output_shape, config->output_height, config->output_channels)
    }
    else //(config->output_format == 1)
    {
      FILL_SHAPE_MATRIX(cnn->output_shape, config->output_channels, config->output_height)
    }
  }
  else if(config->algo == XA_NNLIB_CNN_CONV2D_STD)
  {
    if(config->output_format == 0)
    {
      FILL_SHAPE_CUBE(cnn->output_shape, config->output_height, config->output_width, config->output_channels, SHAPE_CUBE_DWH_T)
    }
    else //(config->output_format == 1)
    {
      FILL_SHAPE_CUBE(cnn->output_shape, config->output_height, config->output_width, config->output_channels, SHAPE_CUBE_WHD_T)
    }
  }
  else if(config->algo == XA_NNLIB_CNN_CONV2D_DS)
  {
    FILL_SHAPE_CUBE(cnn->output_shape, config->output_height, config->output_width, config->output_channels, SHAPE_CUBE_WHD_T)
  }

  return XA_NNLIB_NO_ERROR;
}

int xa_nnlib_cnn_set_config(
  xa_nnlib_handle_t handle,
  xa_nnlib_cnn_param_id_t param_id,
  void *params )
{
  cnn_state_t *cnn;
  xa_nnlib_cnn_init_config_t *config;

  CHECK_PTR(handle, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(params, XA_NNLIB_FATAL_MEM_ALLOC);

  CHECK_PTR_ALIGN(handle, 8, XA_NNLIB_FATAL_MEM_ALIGN);
  CHECK_PTR_ALIGN(params, 4, XA_NNLIB_FATAL_MEM_ALIGN);

  cnn = (cnn_state_t *) handle;
  config = (xa_nnlib_cnn_init_config_t *) &cnn->config;

  switch(param_id)
  {
    case XA_NNLIB_CNN_KERNEL:
    {
      void **pp_kernel = (void **) params;

      if(config->algo == XA_NNLIB_CNN_CONV1D_STD || config->algo == XA_NNLIB_CNN_CONV2D_STD)
      {
        cnn->kernel_std = pp_kernel[0];
      }
      else if(config->algo == XA_NNLIB_CNN_CONV2D_DS)
      {
        cnn->kernel_ds_depth = pp_kernel[0];
        cnn->kernel_ds_point = pp_kernel[1];
      }
    }
    break;

    case XA_NNLIB_CNN_BIAS:
    {
      void **pp_bias = (void **) params;

      CHECK_PTR(pp_bias[0], XA_NNLIB_FATAL_MEM_ALLOC);
      if(config->algo == XA_NNLIB_CNN_CONV1D_STD || config->algo == XA_NNLIB_CNN_CONV2D_STD)
      {
        cnn->bias_std = pp_bias[0];
      }
      else if(config->algo == XA_NNLIB_CNN_CONV2D_DS)
      {
        cnn->bias_ds_depth = pp_bias[0];
        cnn->bias_ds_point = pp_bias[1];
      }
    }
    break;

    default:
    return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PARAM_ID;
  }

  return XA_NNLIB_NO_ERROR;
}

int xa_nnlib_cnn_get_config(
  xa_nnlib_handle_t handle,
  xa_nnlib_cnn_param_id_t param_id,
  void *params )
{
  cnn_state_t *cnn;
  xa_nnlib_cnn_init_config_t *config;

  CHECK_PTR(handle, XA_NNLIB_FATAL_MEM_ALLOC);
  CHECK_PTR(params, XA_NNLIB_FATAL_MEM_ALLOC);

  CHECK_PTR_ALIGN(handle, 8, XA_NNLIB_FATAL_MEM_ALIGN);
  CHECK_PTR_ALIGN(params, 4, XA_NNLIB_FATAL_MEM_ALIGN);

  cnn = (cnn_state_t *) handle;
  config = (xa_nnlib_cnn_init_config_t *) &cnn->config;

  switch(param_id)
  {
    case XA_NNLIB_CNN_KERNEL:
    {
      void **pp_kernel = (void **) params;

      if(config->algo == XA_NNLIB_CNN_CONV1D_STD || config->algo == XA_NNLIB_CNN_CONV2D_STD)
      {
        pp_kernel[0] = cnn->kernel_std;
      }
      else if(config->algo == XA_NNLIB_CNN_CONV2D_DS)
      {
        pp_kernel[0] = cnn->kernel_ds_depth;
        pp_kernel[1] = cnn->kernel_ds_point;
      }
    }
    break;

    case XA_NNLIB_CNN_BIAS:
    {
      void **pp_bias = (void **) params;

      if(config->algo == XA_NNLIB_CNN_CONV1D_STD || config->algo == XA_NNLIB_CNN_CONV2D_STD)
      {
        pp_bias[0] = cnn->kernel_std;
      }
      else if(config->algo == XA_NNLIB_CNN_CONV2D_DS)
      {
        pp_bias[0] = cnn->kernel_ds_depth;
        pp_bias[1] = cnn->kernel_ds_point;
      }
    }
    break;

    case XA_NNLIB_CNN_INPUT_SHAPE:
    {
      xa_nnlib_shape_t *inp_shape = (xa_nnlib_shape_t *)params;
      memcpy(inp_shape, &config->input_shape, sizeof(xa_nnlib_shape_t));
    }
    break;

    case XA_NNLIB_CNN_OUTPUT_SHAPE:
    {
      xa_nnlib_shape_t *out_shape = (xa_nnlib_shape_t *)params;
      memcpy(out_shape, &cnn->output_shape, sizeof(xa_nnlib_shape_t));
    }
    break;

    default:
    return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PARAM_ID;
  }

  return XA_NNLIB_NO_ERROR;
}

int xa_nnlib_cnn_process(xa_nnlib_handle_t handle,
    void *scratch,
    void *input,
    void *output,
    xa_nnlib_shape_t *p_in_shape,
    xa_nnlib_shape_t *p_out_shape )
{
  cnn_state_t *cnn;
  xa_nnlib_cnn_init_config_t *config;
  int inp_precision;
  int err = 0;

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

  cnn = (cnn_state_t *) handle;
  config = (xa_nnlib_cnn_init_config_t *) &cnn->config;

  if(config->algo == XA_NNLIB_CNN_CONV1D_STD || config->algo == XA_NNLIB_CNN_CONV2D_STD)
  {
    CHECK_PTR(cnn->kernel_std, XA_NNLIB_FATAL_MEM_ALLOC);
    CHECK_PTR(cnn->bias_std,   XA_NNLIB_FATAL_MEM_ALLOC);
    CHECK_PTR_ALIGN(cnn->kernel_std, 8, XA_NNLIB_FATAL_MEM_ALIGN);
    CHECK_PTR_ALIGN(cnn->bias_std,   8, XA_NNLIB_FATAL_MEM_ALIGN);
  }
  else if(config->algo == XA_NNLIB_CNN_CONV2D_DS)
  {
    CHECK_PTR(cnn->kernel_ds_depth, XA_NNLIB_FATAL_MEM_ALLOC);
    CHECK_PTR(cnn->bias_ds_depth,   XA_NNLIB_FATAL_MEM_ALLOC);
    CHECK_PTR(cnn->kernel_ds_point, XA_NNLIB_FATAL_MEM_ALLOC);
    CHECK_PTR(cnn->bias_ds_point,   XA_NNLIB_FATAL_MEM_ALLOC);
    CHECK_PTR_ALIGN(cnn->kernel_ds_depth, 8, XA_NNLIB_FATAL_MEM_ALIGN);
    CHECK_PTR_ALIGN(cnn->bias_ds_depth,   8, XA_NNLIB_FATAL_MEM_ALIGN);
    CHECK_PTR_ALIGN(cnn->kernel_ds_point, 8, XA_NNLIB_FATAL_MEM_ALIGN);
    CHECK_PTR_ALIGN(cnn->bias_ds_point,   8, XA_NNLIB_FATAL_MEM_ALIGN);
  }

  MATCH_CUBE_DIMS(p_in_shape, config->input_shape, XA_NNLIB_CNN_EXECUTE_FATAL_INVALID_INPUT_SHAPE);

  inp_precision = IO_PRECISION_BITS(config->precision);

  if(config->algo == XA_NNLIB_CNN_CONV1D_STD)
  {
    switch(config->precision)
    {
      case XA_NNLIB_CNN_16bx16b:
      {
        err = xa_nn_conv1d_std_16x16(output,
                                     input,
                                     cnn->kernel_std,
                                     cnn->bias_std,
                                     config->input_shape.dim.cube.height,
                                     config->input_shape.dim.cube.width,
                                     config->input_shape.dim.cube.depth,
                                     config->kernel_std_shape.dim.cube.height,
                                     cnn->output_shape.dim.matrix.cols,
                                     config->y_stride,
                                     config->y_padding,
                                     cnn->output_shape.dim.matrix.rows,
                                     config->bias_shift,
                                     config->acc_shift,
                                     config->output_format,
                                     scratch);
      }
      break;
      case XA_NNLIB_CNN_8bx16b:
      {
        err = xa_nn_conv1d_std_8x16(output,
                                    input,
                                    cnn->kernel_std,
                                    cnn->bias_std,
                                    config->input_shape.dim.cube.height,
                                    config->input_shape.dim.cube.width,
                                    config->input_shape.dim.cube.depth,
                                    config->kernel_std_shape.dim.cube.height,
                                    cnn->output_shape.dim.matrix.cols,
                                    config->y_stride,
                                    config->y_padding,
                                    cnn->output_shape.dim.matrix.rows,
                                    config->bias_shift,
                                    config->acc_shift,
                                    config->output_format,
                                    scratch);
      }
      break;
      case XA_NNLIB_CNN_8bx8b:
      {
        err = xa_nn_conv1d_std_8x8(output,
                                   input,
                                   cnn->kernel_std,
                                   cnn->bias_std,
                                   config->input_shape.dim.cube.height,
                                   config->input_shape.dim.cube.width,
                                   config->input_shape.dim.cube.depth,
                                   config->kernel_std_shape.dim.cube.height,
                                   cnn->output_shape.dim.matrix.cols,
                                   config->y_stride,
                                   config->y_padding,
                                   cnn->output_shape.dim.matrix.rows,
                                   config->bias_shift,
                                   config->acc_shift,
                                   config->output_format,
                                   scratch);
      }
      break;
#if HAVE_VFPU
      case XA_NNLIB_CNN_f32xf32:
      {
        err = xa_nn_conv1d_std_f32(output,
                                   input,
                                   cnn->kernel_std,
                                   cnn->bias_std,
                                   config->input_shape.dim.cube.height,
                                   config->input_shape.dim.cube.width,
                                   config->input_shape.dim.cube.depth,
                                   config->kernel_std_shape.dim.cube.height,
                                   cnn->output_shape.dim.matrix.cols,
                                   config->y_stride,
                                   config->y_padding,
                                   cnn->output_shape.dim.matrix.rows,
                                   config->output_format,
                                   scratch);
      }
      break;
#else
      case XA_NNLIB_CNN_f32xf32:
      {
      }
      break;
#endif
    }
  }
  else if(config->algo == XA_NNLIB_CNN_CONV2D_STD)
  {
    switch(config->precision)
    {
      case XA_NNLIB_CNN_16bx16b:
      {
        err = xa_nn_conv2d_std_16x16(output,
                                     input,
                                     cnn->kernel_std,
                                     cnn->bias_std,
                                     config->input_shape.dim.cube.height,
                                     config->input_shape.dim.cube.width,
                                     config->input_shape.dim.cube.depth,
                                     config->kernel_std_shape.dim.cube.height,
                                     config->kernel_std_shape.dim.cube.width,
                                     cnn->output_shape.dim.cube.depth,
                                     config->x_stride,
                                     config->y_stride,
                                     config->x_padding,
                                     config->y_padding,
                                     cnn->output_shape.dim.cube.height,
                                     cnn->output_shape.dim.cube.width,
                                     config->bias_shift,
                                     config->acc_shift,
                                     config->output_format,
                                     scratch);
      }
      break;
      case XA_NNLIB_CNN_8bx16b:
      {
        err = xa_nn_conv2d_std_8x16(output,
                                    input,
                                    cnn->kernel_std,
                                    cnn->bias_std,
                                    config->input_shape.dim.cube.height,
                                    config->input_shape.dim.cube.width,
                                    config->input_shape.dim.cube.depth,
                                    config->kernel_std_shape.dim.cube.height,
                                    config->kernel_std_shape.dim.cube.width,
                                    cnn->output_shape.dim.cube.depth,
                                    config->x_stride,
                                    config->y_stride,
                                    config->x_padding,
                                    config->y_padding,
                                    cnn->output_shape.dim.cube.height,
                                    cnn->output_shape.dim.cube.width,
                                    config->bias_shift,
                                    config->acc_shift,
                                    config->output_format,
                                    scratch);
      }
      break;
      case XA_NNLIB_CNN_8bx8b:
      {
        err = xa_nn_conv2d_std_8x8(output,
                                   input,
                                   cnn->kernel_std,
                                   cnn->bias_std,
                                   config->input_shape.dim.cube.height,
                                   config->input_shape.dim.cube.width,
                                   config->input_shape.dim.cube.depth,
                                   config->kernel_std_shape.dim.cube.height,
                                   config->kernel_std_shape.dim.cube.width,
                                   cnn->output_shape.dim.cube.depth,
                                   config->x_stride,
                                   config->y_stride,
                                   config->x_padding,
                                   config->y_padding,
                                   cnn->output_shape.dim.cube.height,
                                   cnn->output_shape.dim.cube.width,
                                   config->bias_shift,
                                   config->acc_shift,
                                   config->output_format,
                                   scratch);
      }
      break;
#if HAVE_VFPU
      case XA_NNLIB_CNN_f32xf32:
      {
        err = xa_nn_conv2d_std_f32(output,
                                   input,
                                   cnn->kernel_std,
                                   cnn->bias_std,
                                   config->input_shape.dim.cube.height,
                                   config->input_shape.dim.cube.width,
                                   config->input_shape.dim.cube.depth,
                                   config->kernel_std_shape.dim.cube.height,
                                   config->kernel_std_shape.dim.cube.width,
                                   cnn->output_shape.dim.cube.depth,
                                   config->x_stride,
                                   config->y_stride,
                                   config->x_padding,
                                   config->y_padding,
                                   cnn->output_shape.dim.cube.height,
                                   cnn->output_shape.dim.cube.width,
                                   config->output_format,
                                   scratch);
      }
      break;
#else
      case XA_NNLIB_CNN_f32xf32:
      {
      }
      break;
#endif
    }
  }
  else if(config->algo == XA_NNLIB_CNN_CONV2D_DS)
  {
    int scratch_size;
    void *depthwise_out_scratch;

    scratch_size = xa_nn_conv2d_depthwise_getsize(config->input_shape.dim.cube.height,
                                                  config->input_shape.dim.cube.width,
                                                  config->input_shape.dim.cube.depth,
                                                  config->kernel_ds_depth_shape.dim.cube.height,
                                                  config->kernel_ds_depth_shape.dim.cube.width,
                                                  config->channels_multiplier,
                                                  config->x_stride,
                                                  config->y_stride,
                                                  config->x_padding,
                                                  config->y_padding,
                                                  cnn->output_shape.dim.cube.height,
                                                  cnn->output_shape.dim.cube.width,
                                                  inp_precision,
                                                  1);       //must be WHD
    scratch_size  = ALIGN_SIZE(scratch_size);
    depthwise_out_scratch = ((char *) scratch + scratch_size);

    switch(config->precision)
    {
      case XA_NNLIB_CNN_16bx16b:
      {
        err = xa_nn_conv2d_depthwise_16x16(depthwise_out_scratch,
                                           cnn->kernel_ds_depth,
                                           input,
                                           cnn->bias_ds_depth,
                                           config->input_shape.dim.cube.height,
                                           config->input_shape.dim.cube.width,
                                           config->input_shape.dim.cube.depth,
                                           config->kernel_ds_depth_shape.dim.cube.height,
                                           config->kernel_ds_depth_shape.dim.cube.width,
                                           config->channels_multiplier,
                                           config->x_stride,
                                           config->y_stride,
                                           config->x_padding,
                                           config->y_padding,
                                           cnn->output_shape.dim.cube.height,
                                           cnn->output_shape.dim.cube.width,
                                           config->acc_shift,
                                           config->bias_shift,
                                           1, //must be WHD
                                           0, //must be DWH
                                           scratch);

        if (err) break;

        err = xa_nn_conv2d_pointwise_16x16(output,
                                           cnn->kernel_ds_point,
                                           depthwise_out_scratch,
                                           cnn->bias_ds_point,
                                           cnn->output_shape.dim.cube.height,
                                           cnn->output_shape.dim.cube.width,
                                           config->input_shape.dim.cube.depth*config->channels_multiplier,
                                           cnn->output_shape.dim.cube.depth,
                                           config->acc_shift,
                                           config->bias_shift,
                                           config->output_format);
      }
      break;
      case XA_NNLIB_CNN_8bx16b:
      {
        err = xa_nn_conv2d_depthwise_8x16(depthwise_out_scratch,
                                          cnn->kernel_ds_depth,
                                          input,
                                          cnn->bias_ds_depth,
                                          config->input_shape.dim.cube.height,
                                          config->input_shape.dim.cube.width,
                                          config->input_shape.dim.cube.depth,
                                          config->kernel_ds_depth_shape.dim.cube.height,
                                          config->kernel_ds_depth_shape.dim.cube.width,
                                          config->channels_multiplier,
                                          config->x_stride,
                                          config->y_stride,
                                          config->x_padding,
                                          config->y_padding,
                                          cnn->output_shape.dim.cube.height,
                                          cnn->output_shape.dim.cube.width,
                                          config->acc_shift,
                                          config->bias_shift,
                                          1, //must be WHD
                                          0, //must be DWH
                                          scratch);

        if (err) break;

        err = xa_nn_conv2d_pointwise_8x16(output,
                                          cnn->kernel_ds_point,
                                          depthwise_out_scratch,
                                          cnn->bias_ds_point,
                                          cnn->output_shape.dim.cube.height,
                                          cnn->output_shape.dim.cube.width,
                                          config->input_shape.dim.cube.depth*config->channels_multiplier,
                                          cnn->output_shape.dim.cube.depth,
                                          config->acc_shift,
                                          config->bias_shift,
                                          config->output_format);
      }
      break;
      case XA_NNLIB_CNN_8bx8b:
      {
        err = xa_nn_conv2d_depthwise_8x8(depthwise_out_scratch,
                                         cnn->kernel_ds_depth,
                                         input,
                                         cnn->bias_ds_depth,
                                         config->input_shape.dim.cube.height,
                                         config->input_shape.dim.cube.width,
                                         config->input_shape.dim.cube.depth,
                                         config->kernel_ds_depth_shape.dim.cube.height,
                                         config->kernel_ds_depth_shape.dim.cube.width,
                                         config->channels_multiplier,
                                         config->x_stride,
                                         config->y_stride,
                                         config->x_padding,
                                         config->y_padding,
                                         cnn->output_shape.dim.cube.height,
                                         cnn->output_shape.dim.cube.width,
                                         config->acc_shift,
                                         config->bias_shift,
                                         1, //must be WHD
                                         0, //must be DWH
                                         scratch);

        if (err) break;

        err = xa_nn_conv2d_pointwise_8x8(output,
                                         cnn->kernel_ds_point,
                                         depthwise_out_scratch,
                                         cnn->bias_ds_point,
                                         cnn->output_shape.dim.cube.height,
                                         cnn->output_shape.dim.cube.width,
                                         config->input_shape.dim.cube.depth*config->channels_multiplier,
                                         cnn->output_shape.dim.cube.depth,
                                         config->acc_shift,
                                         config->bias_shift,
                                         config->output_format);
      }
      break;
#if HAVE_VFPU
      case XA_NNLIB_CNN_f32xf32:
      {
        err = xa_nn_conv2d_depthwise_f32(depthwise_out_scratch,
                                         cnn->kernel_ds_depth,
                                         input,
                                         cnn->bias_ds_depth,
                                         config->input_shape.dim.cube.height,
                                         config->input_shape.dim.cube.width,
                                         config->input_shape.dim.cube.depth,
                                         config->kernel_ds_depth_shape.dim.cube.height,
                                         config->kernel_ds_depth_shape.dim.cube.width,
                                         config->channels_multiplier,
                                         config->x_stride,
                                         config->y_stride,
                                         config->x_padding,
                                         config->y_padding,
                                         cnn->output_shape.dim.cube.height,
                                         cnn->output_shape.dim.cube.width,
                                         1, //must be WHD
                                         0, //must be DWH
                                         scratch);

        if (err) break;

        err = xa_nn_conv2d_pointwise_f32(output,
                                         cnn->kernel_ds_point,
                                         depthwise_out_scratch,
                                         cnn->bias_ds_point,
                                         cnn->output_shape.dim.cube.height,
                                         cnn->output_shape.dim.cube.width,
                                         config->input_shape.dim.cube.depth*config->channels_multiplier,
                                         cnn->output_shape.dim.cube.depth,
                                         config->output_format);

      }
      break;
#else
      case XA_NNLIB_CNN_f32xf32:
      {
      }
      break;
#endif
    }
  }

  if (!err) memcpy(p_out_shape, &cnn->output_shape, sizeof(xa_nnlib_shape_t));
  else
  {
    memset(p_out_shape, 0, sizeof(xa_nnlib_shape_t));
    //ASSERT - must not come here! For fail-safe, return invalid-param-combination error.
    return XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PARAM_COMBINATION;
  }

  return XA_NNLIB_NO_ERROR;
}

