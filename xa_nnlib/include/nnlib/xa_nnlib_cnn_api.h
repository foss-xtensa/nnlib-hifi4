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
#ifndef __XA_CNN_API_H__
#define __XA_CNN_API_H__

#include "xa_nnlib_standards.h"

#define XA_NNLIB_CNN    3

/* GET/SET Config Parameters                                */
typedef enum _xa_nnlib_cnn_param_id_t
{
    XA_NNLIB_CNN_KERNEL              = 1,             // GET/SET kernel
    XA_NNLIB_CNN_BIAS                = 2,             // GET/SET bias
    XA_NNLIB_CNN_INPUT_SHAPE         = 3,             // GET input shape information
    XA_NNLIB_CNN_OUTPUT_SHAPE        = 4              // GET output shape information
} xa_nnlib_cnn_param_id_t;

/* I/O Precision Settings */
typedef enum _xa_nnlib_cnn_precision_t
{
    XA_NNLIB_CNN_16bx16b             = 100,           // Coef: 16 bits, I/O: 16 bits Fixed Point
    XA_NNLIB_CNN_8bx16b              = 101,           // Coef: 8 bits, I/O: 16 bits Fixed Point
    XA_NNLIB_CNN_8bx8b               = 102,           // Coef: 8 bits, I/O: 8 bits Fixed Point
    XA_NNLIB_CNN_f32xf32             = 103            // Coef: float32, I/O: float32 floating Point
} xa_nnlib_cnn_precision_t;

/* Convolution Algorithm Settings */
typedef enum _xa_nnlib_cnn_algo_t
{
    XA_NNLIB_CNN_CONV1D_STD          = 200,           // Standard 1D Convolution
    XA_NNLIB_CNN_CONV2D_STD          = 201,           // Standard 2D Convolution
    XA_NNLIB_CNN_CONV2D_DS           = 203,           // Depthwise Separable 2D Convolution
} xa_nnlib_cnn_algo_t;

/************************************************************/
/* Class 1: Configuration Errors                            */
/************************************************************/
/* Nonfatal Errors */
/* None */

/* Fatal Errors */
typedef enum _xa_nnlib_fatal_config_cnn_error_code_t
{
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_ALGO              = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 0),
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PRECISION         = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 1),
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_BIAS_SHIFT        = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 2),
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_ACC_SHIFT         = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 3),
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_STRIDE            = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 4),
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PADDING           = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 5),
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_INPUT_SHAPE       = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 6),
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_OUTPUT_SHAPE      = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 7),
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_KERNEL_SHAPE      = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 8),
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_BIAS_SHAPE        = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 9),
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PARAM_ID          = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 10),
    XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PARAM_COMBINATION = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_CNN, 11),
} xa_nnlib_fatal_config_cnn_error_code_t;

/************************************************************/
/* Class 1: Execution Errors                                */
/************************************************************/
/* Nonfatal Errors */
/* None */

/* Fatal Errors */
/* None */


/* Structure for initial configuration */
typedef struct _xa_nnlib_cnn_init_config_t
{
    /* Shape */
    xa_nnlib_shape_t input_shape;           /* Input Dimensions */

    Int32 output_height;                    /* Output Height */
    Int32 output_width;                     /* Output Width */
    Int32 output_channels;                  /* Output Depth / Channels */
    Int32 output_format;                    /* Output format: 0 - Depth first (DepthWidthHeight), 1 - Depth last (WidthHeightDepth) */

    xa_nnlib_shape_t kernel_std_shape;      /* Standard 1D/2D Convolution Kernel (filter) Dimensions */
    xa_nnlib_shape_t kernel_ds_depth_shape; /* Depthwise Separable 2D Convolution - Depthwise Kernel (filter) Dimensions */
    xa_nnlib_shape_t kernel_ds_point_shape; /* Depthwise Separable 2D Convolution - Pointwise Kernel (filter) Dimensions */

    xa_nnlib_shape_t bias_std_shape;        /* Standard 1D/2D Convolution Bias Dimensions */
    xa_nnlib_shape_t bias_ds_depth_shape;   /* Depthwise Separable 2D Convolution - Depthwise Bias Dimensions */
    xa_nnlib_shape_t bias_ds_point_shape;   /* Depthwise Separable 2D Convolution - Pointwise Bias Dimensions */
    /* Precision */
    xa_nnlib_cnn_precision_t precision;     /* Kernel (filter), input, output precision setting */
    Int32 bias_shift;                       /* Q-format adjustment for bias before addition into accumulator, +/- value - left/right shift */
    Int32 acc_shift;                        /* Q-format adjustment for accumulator before rounding to result, +/- value - left/right shift */

    Int32 channels_multiplier;              /* Depthwise Separable 2D Convolution - channel multiplier */
    /* Padding to be added to input */
    Int32 x_padding;                        /* Left side padding */
    Int32 y_padding;                        /* Top padding */

    /* Kernel strides */
    Int32 x_stride;
    Int32 y_stride;

    /* Convolution algorithm */
    xa_nnlib_cnn_algo_t algo;

} xa_nnlib_cnn_init_config_t;


#if defined(__cplusplus)
extern "C" {
#endif    /* __cplusplus */

/************************************************************/
/* CNN Query Functions                                      */
/************************************************************/
Int32 xa_nnlib_cnn_get_persistent_fast( xa_nnlib_cnn_init_config_t *config);

Int32 xa_nnlib_cnn_get_scratch_fast( xa_nnlib_cnn_init_config_t *config);

/************************************************************/
/* CNN Initialization Function                              */
/************************************************************/
Int32 xa_nnlib_cnn_init(xa_nnlib_handle_t handle, xa_nnlib_cnn_init_config_t *config);

/************************************************************/
/* CNN Execution Functions                                  */
/************************************************************/
Int32 xa_nnlib_cnn_set_config(xa_nnlib_handle_t handle, xa_nnlib_cnn_param_id_t param_id, void *params);

Int32 xa_nnlib_cnn_get_config(xa_nnlib_handle_t handle, xa_nnlib_cnn_param_id_t param_id, void *params);

Int32 xa_nnlib_cnn_process(xa_nnlib_handle_t handle,
                           void *scratch,
                           void *input,
                           void *output,
                           xa_nnlib_shape_t *p_in_shape,
                           xa_nnlib_shape_t *p_out_shape );

#if defined(__cplusplus)
}
#endif    /* __cplusplus */

#endif  /* __XA_CNN_API_H__ */
