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
#ifndef __XA_LSTM_API_H__
#define __XA_LSTM_API_H__

#include "xa_nnlib_standards.h"

#define XA_NNLIB_LSTM    2

/* GET/SET Config Parameters                                */
typedef enum _xa_nnlib_lstm_param_id_t
{
  XA_NNLIB_LSTM_RESTORE_CONTEXT_OUTPUT = 0,             // GET/SET prev_h
  XA_NNLIB_LSTM_RESTORE_CONTEXT_CELL   = 1,             // GET/SET prev_c
  XA_NNLIB_LSTM_WEIGHT                 = 2,             // GET/SET weights
  XA_NNLIB_LSTM_BIAS                   = 3,             // GET/SET biases
  XA_NNLIB_LSTM_INPUT_SHAPE            = 4,             // GET input shape information
  XA_NNLIB_LSTM_OUTPUT_SHAPE           = 5,             // GET output shape information
  XA_NNLIB_LSTM_CELL_SHAPE             = 6              // GET cell shape information. Not Supported.
} xa_nnlib_lstm_param_id_t;

/* I/O Precision Settings */
typedef enum _xa_nnlib_lstm_precision_t
{
  XA_NNLIB_LSTM_16bx16b             = 100,           // Coef: 16 bits, I/O: 16 bits Fixed Point
  XA_NNLIB_LSTM_8bx16b              = 101,           // Coef: 8 bits, I/O: 16 bits Fixed Point
  XA_NNLIB_LSTM_8bx8b               = 102,           // Not supported
  XA_NNLIB_LSTM_flt16xflt16         = 103            // Not supported
} xa_nnlib_lstm_precision_t;


/************************************************************/
/* Class 1: Configuration Errors                            */
/************************************************************/
/* Nonfatal Errors */
/* None */

/* Fatal Errors */
typedef enum _xa_nnlib_fatal_config_lstm_error_code_t
{
  XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_IN_FEATS         = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_LSTM, 0),
  XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_OUT_FEATS        = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_LSTM, 1),
  XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_PRECISION        = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_LSTM, 2),
  XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_COEFF_QFORMAT    = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_LSTM, 3),
  XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_CELL_QFORMAT     = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_LSTM, 4),
  XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_IO_QFORMAT       = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_LSTM, 5),
  XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_PARAM_ID         = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_LSTM, 6),
  XA_NNLIB_LSTM_CONFIG_FATAL_INVALID_MEMBANK_PADDING  = XA_ERROR_CODE(xa_severity_fatal, xa_class_config, XA_NNLIB_LSTM, 7)
} xa_nnlib_fatal_config_lstm_error_code_t;

/************************************************************/
/* Class 1: Execution Errors                                */
/************************************************************/
/* Nonfatal Errors */
/* None */

/* Fatal Errors */
typedef enum _xa_nnlib_fatal_exec_lstm_error_code_t
{
  XA_NNLIB_LSTM_EXECUTE_FATAL_INSUFFICIENT_OUTPUT_BUFFER_SPACE    = XA_ERROR_CODE(xa_severity_fatal, xa_class_execute, XA_NNLIB_LSTM, 0),
  XA_NNLIB_LSTM_EXECUTE_FATAL_INSUFFICIENT_DATA                   = XA_ERROR_CODE(xa_severity_fatal, xa_class_execute, XA_NNLIB_LSTM, 1)
} xa_nnlib_fatal_exec_lstm_error_code_t;


/* Structure for initial configuration */
typedef struct _xa_nnlib_lstm_init_config_t
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
  xa_nnlib_lstm_precision_t precision;
  /* Number of fractional bits for weights and biases; 0-15 */
  Int16 coeff_Qformat;
  /* Number of fractional bits for cell state; 0-15 */
  Int16 cell_Qformat;
  /* Number of fractional bits for input and output; 0-15 */
  Int16 io_Qformat;
} xa_nnlib_lstm_init_config_t;

/* Structure for getting/setting XA_NNLIB_LSTM_WEIGHT parameter
. All pointer needs to be 8 bytes aligned.                  */
typedef union _xa_nnlib_lstm_weights_t
{
    struct
    {
        coeff_t *w_xf; xa_nnlib_shape_t shape_w_xf;
        coeff_t *w_hf; xa_nnlib_shape_t shape_w_hf;
        coeff_t *w_xi; xa_nnlib_shape_t shape_w_xi;
        coeff_t *w_hi; xa_nnlib_shape_t shape_w_hi;
        coeff_t *w_xc; xa_nnlib_shape_t shape_w_xc;
        coeff_t *w_hc; xa_nnlib_shape_t shape_w_hc;
        coeff_t *w_xo; xa_nnlib_shape_t shape_w_xo;
        coeff_t *w_ho; xa_nnlib_shape_t shape_w_ho;
    }weights16;
    struct
    {
        coeff8_t *w_xf; xa_nnlib_shape_t shape_w_xf;
        coeff8_t *w_hf; xa_nnlib_shape_t shape_w_hf;
        coeff8_t *w_xi; xa_nnlib_shape_t shape_w_xi;
        coeff8_t *w_hi; xa_nnlib_shape_t shape_w_hi;
        coeff8_t *w_xc; xa_nnlib_shape_t shape_w_xc;
        coeff8_t *w_hc; xa_nnlib_shape_t shape_w_hc;
        coeff8_t *w_xo; xa_nnlib_shape_t shape_w_xo;
        coeff8_t *w_ho; xa_nnlib_shape_t shape_w_ho;
    }weights8;
} xa_nnlib_lstm_weights_t;

/* Structure for getting/setting XA_NNLIB_LSTM_BIAS parameter.
 All pointer needs to be 8 bytes aligned.                   */
typedef struct _xa_nnlib_lstm_biases_t
{
  coeff_t *b_f; xa_nnlib_shape_t shape_b_f;
  coeff_t *b_i; xa_nnlib_shape_t shape_b_i;
  coeff_t *b_c; xa_nnlib_shape_t shape_b_c;
  coeff_t *b_o; xa_nnlib_shape_t shape_b_o;
} xa_nnlib_lstm_biases_t;

#if defined(__cplusplus)
extern "C" {
#endif    /* __cplusplus */

/************************************************************/
/* LSTM Query Functions                                      */
/************************************************************/
Int32 xa_nnlib_lstm_get_persistent_fast( xa_nnlib_lstm_init_config_t *config);

Int32 xa_nnlib_lstm_get_scratch_fast( xa_nnlib_lstm_init_config_t *config);

/************************************************************/
/* LSTM Initialization Function                              */
/************************************************************/
Int32 xa_nnlib_lstm_init(xa_nnlib_handle_t handle, xa_nnlib_lstm_init_config_t *config);

/************************************************************/
/* LSTM Execution Functions                                  */
/************************************************************/
Int32 xa_nnlib_lstm_set_config(xa_nnlib_handle_t handle, xa_nnlib_lstm_param_id_t param_id, void *params);

Int32 xa_nnlib_lstm_get_config(xa_nnlib_handle_t handle, xa_nnlib_lstm_param_id_t param_id, void *params);

Int32 xa_nnlib_lstm_process(xa_nnlib_handle_t handle,
    void *scratch,
    void *input,
    void *output,
    xa_nnlib_shape_t *p_in_shape,
    xa_nnlib_shape_t *p_out_shape);

#if defined(__cplusplus)
}
#endif    /* __cplusplus */

#endif  /* __XA_LSTM_API_H__ */
