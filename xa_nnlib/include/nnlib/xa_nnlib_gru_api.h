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
#ifndef __XA_GRU_API_H__
#define __XA_GRU_API_H__

#include "xa_nnlib_standards.h"

#define XA_NNLIB_GRU    1

/* GET/SET Config Parameters                                */
typedef enum _xa_nnlib_gru_param_id_t
{
  XA_NNLIB_GRU_RESTORE_CONTEXT     = 0,             // GET/SET prev_h
  XA_NNLIB_GRU_WEIGHT              = 1,             // GET/SET weights
  XA_NNLIB_GRU_BIAS                = 2,             // GET/SET biases
  XA_NNLIB_GRU_INPUT_SHAPE         = 3,             // GET input shape information
  XA_NNLIB_GRU_OUTPUT_SHAPE        = 4              // GET output shape information
} xa_nnlib_gru_param_id_t;

/* I/O Precision Settings */
typedef enum _xa_nnlib_gru_precision_t
{
  XA_NNLIB_GRU_16bx16b             = 100,           // Coef: 16 bits, I/O: 16 bits Fixed Point
  XA_NNLIB_GRU_8bx16b              = 101,           // Coef: 8 bits, I/O: 16 bits Fixed Point
  XA_NNLIB_GRU_8bx8b               = 102,           // Not supported
  XA_NNLIB_GRU_flt16xflt16         = 103,           // Not supported
  XA_NNLIB_GRU_flt32xflt32         = 104
} xa_nnlib_gru_precision_t;


/************************************************************/
/* Class 1: Configuration Errors                            */
/************************************************************/
/* Nonfatal Errors */
/* None */

/* Fatal Errors */
typedef enum _xa_nnlib_fatal_config_gru_error_code_t
{
  XA_NNLIB_GRU_CONFIG_FATAL_INVALID_IN_FEATS         = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_GRU, 0),
  XA_NNLIB_GRU_CONFIG_FATAL_INVALID_OUT_FEATS        = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_GRU, 1),
  XA_NNLIB_GRU_CONFIG_FATAL_INVALID_PRECISION        = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_GRU, 2),
  XA_NNLIB_GRU_CONFIG_FATAL_INVALID_COEFF_QFORMAT    = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_GRU, 3),
  XA_NNLIB_GRU_CONFIG_FATAL_INVALID_IO_QFORMAT       = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_GRU, 4),
  XA_NNLIB_GRU_CONFIG_FATAL_INVALID_PARAM_ID         = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_GRU, 5),
  XA_NNLIB_GRU_CONFIG_FATAL_INVALID_MEMBANK_PADDING  = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_GRU, 6),
  XA_NNLIB_GRU_CONFIG_FATAL_INVALID_SPLIT_BIAS       = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_GRU, 7)
} xa_nnlib_fatal_config_gru_error_code_t;

/************************************************************/
/* Class 1: Execution Errors                                */
/************************************************************/
/* Nonfatal Errors */
/* None */

/* Fatal Errors */
typedef enum _xa_nnlib_fatal_exec_gru_error_code_t
{
  XA_NNLIB_GRU_EXECUTE_FATAL_INSUFFICIENT_OUTPUT_BUFFER_SPACE    = XA_ERROR_CODE(xa_severity_fatal, xa_class_execute, XA_NNLIB_GRU, 0),
  XA_NNLIB_GRU_EXECUTE_FATAL_INSUFFICIENT_DATA                   = XA_ERROR_CODE(xa_severity_fatal, xa_class_execute, XA_NNLIB_GRU, 1)
} xa_nnlib_fatal_exec_gru_error_code_t;


/* Structure for initial configuration */
typedef struct _xa_nnlib_gru_init_config_t
{
  /* Number of input features; 4-2048 (in step of 4) */
  Int32 in_feats;
  /* Number of output features; 4-2048 (in step of 4) */
  Int32 out_feats;
  /* Padding 8 bytes for HiFi4*/
  Int32 pad;
  /* Matrix input precision */
  Int32 mat_prec;
  /* Vector input precision */
  Int32 vec_prec;
  /* I/O precision setting */
  xa_nnlib_gru_precision_t precision;
  /* Number of fractional bits for weights and biases; 0-15 */
  Int16 coeff_Qformat;
  /* Number of fractional bits for input and output; 0-15 */
  Int16 io_Qformat;
  /* Flag to indicate if the biases are split. split_bias=1 indicates six bias vectors, otherwise three bias vectors */
  Int32 split_bias;
} xa_nnlib_gru_init_config_t;

/* Structure for getting/setting XA_NNLIB_GRU_WEIGHT parameter
. All pointer needs to be 8 bytes aligned.                  */
typedef union _xa_nnlib_gru_weights_t
{
    struct
    {
        coeff_t *w_z; xa_nnlib_shape_t shape_w_z;
        coeff_t *u_z; xa_nnlib_shape_t shape_u_z;
        coeff_t *w_r; xa_nnlib_shape_t shape_w_r;
        coeff_t *u_r; xa_nnlib_shape_t shape_u_r;
        coeff_t *w_h; xa_nnlib_shape_t shape_w_h;
        coeff_t *u_h; xa_nnlib_shape_t shape_u_h;
    }weights16;
    struct
    {
        coeff8_t *w_z; xa_nnlib_shape_t shape_w_z;
        coeff8_t *u_z; xa_nnlib_shape_t shape_u_z;
        coeff8_t *w_r; xa_nnlib_shape_t shape_w_r;
        coeff8_t *u_r; xa_nnlib_shape_t shape_u_r;
        coeff8_t *w_h; xa_nnlib_shape_t shape_w_h;
        coeff8_t *u_h; xa_nnlib_shape_t shape_u_h;
    }weights8;
	    struct
    {
        float *w_z; xa_nnlib_shape_t shape_w_z;
        float *u_z; xa_nnlib_shape_t shape_u_z;
        float *w_r; xa_nnlib_shape_t shape_w_r;
        float *u_r; xa_nnlib_shape_t shape_u_r;
        float *w_h; xa_nnlib_shape_t shape_w_h;
        float *u_h; xa_nnlib_shape_t shape_u_h;
    }weightsf32;
} xa_nnlib_gru_weights_t;

/* Structure for getting/setting XA_NNLIB_GRU_BIAS parameter.
 All pointer needs to be 8 bytes aligned.                   */
typedef struct _xa_nnlib_gru_biases_t
{
  void *b_z; xa_nnlib_shape_t shape_b_z;
  void *b_r; xa_nnlib_shape_t shape_b_r;
  void *b_h; xa_nnlib_shape_t shape_b_h;
  /* Following biases are used only for split-bias implementation */
  void *bs_z; xa_nnlib_shape_t shape_bs_z;
  void *bs_r; xa_nnlib_shape_t shape_bs_r;
  void *bs_h; xa_nnlib_shape_t shape_bs_h;
} xa_nnlib_gru_biases_t;

#if defined(__cplusplus)
extern "C" {
#endif    /* __cplusplus */

/************************************************************/
/* GRU Query Functions                                      */
/************************************************************/
Int32 xa_nnlib_gru_get_persistent_fast( xa_nnlib_gru_init_config_t *config);

Int32 xa_nnlib_gru_get_scratch_fast( xa_nnlib_gru_init_config_t *config);

/************************************************************/
/* GRU Initialization Function                              */
/************************************************************/
Int32 xa_nnlib_gru_init(xa_nnlib_handle_t handle, xa_nnlib_gru_init_config_t *config);

/************************************************************/
/* GRU Execution Functions                                  */
/************************************************************/
Int32 xa_nnlib_gru_set_config(xa_nnlib_handle_t handle, xa_nnlib_gru_param_id_t param_id, void *params);

Int32 xa_nnlib_gru_get_config(xa_nnlib_handle_t handle, xa_nnlib_gru_param_id_t param_id, void *params);

Int32 xa_nnlib_gru_process(xa_nnlib_handle_t handle,
    void *scratch,
    void *input,
    void *output,
    xa_nnlib_shape_t *p_in_shape,
    xa_nnlib_shape_t *p_out_shape );

#if defined(__cplusplus)
}
#endif    /* __cplusplus */

#endif  /* __XA_GRU_API_H__ */
