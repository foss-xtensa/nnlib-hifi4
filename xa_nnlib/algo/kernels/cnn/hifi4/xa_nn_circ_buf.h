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

#ifndef __XA_NN_CIRC_BUF_H__
#define __XA_NN_CIRC_BUF_H__

#ifdef ENABLE_SCRATCH_SIZE_API_ONLY
#define xa_nn_circ_buf_nchw_getsize     xa_nn_circ_buf_nchw_getsize_hifi4
#define xa_nn_circ_buf_nhwc_getsize     xa_nn_circ_buf_nhwc_getsize_hifi4
#endif

#define OUT_HEIGHT_PER_ITER 2

#define ALIGNMENT   8   /* 8 bytes alignment */

#define ALIGNED_SIZE(x, bytes)  (((x)+(bytes-1))&(~(bytes-1)))
#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#define LIMIT(input, min, max) \
    input = XT_MAX(min, XT_MIN(max, input));

#define INIT_ROWS_ADDED(rows_added, y_padding) \
    rows_added = -y_padding;

#define UPDATE_ROWS_ADDED(rows_added, rows_to_add) \
    rows_added += rows_to_add;

#define INIT_ROWS_TO_ADD(rows_to_add, kernel_height, y_stride) \
    rows_to_add = kernel_height-y_stride;


#define CALC_ROWS_TO_ADD(rows_to_add, circ_out_height, y_stride) \
    rows_to_add = circ_out_height*y_stride;

/* rows_added starts from -y_padding and increases as we added rows to circular buffer,
   if rows_added is less than 0 we have to add that many zero rows at top (top_padding),
   if rows_added is more than input_height we have to add that many zero rows at bottom
   (bottom_padding) */
#define CALC_PADDINGS(top_pad, bottom_pad, rows_added, rows_to_add, input_height) \
    top_pad = -rows_added; \
    LIMIT(top_pad, 0, rows_to_add) \
    bottom_pad = rows_to_add - (input_height-rows_added); \
    LIMIT(bottom_pad, 0, rows_to_add) \


/* Calculate the input row index to which the pointer passed to circular buffer
   add row call should point, it must be between 0 to input_height-1 */
#define CALC_INP_ROW(input_row, rows_added, input_height) \
    input_row = rows_added; \
    LIMIT(input_row, 0, input_height-1)

/* Initialize the mechanism for adding rows from input to circular buffer */
#define CIRC_BUF_ADD_ROWS_INIT(rows_added, rows_to_add, top_pad, bottom_pad, input_row, \
    input_height, input_width, kernel_height, y_stride, x_padding, y_padding, p_circ_buf, \
    pt_inp) \
    INIT_ROWS_ADDED(rows_added, y_padding) \
    INIT_ROWS_TO_ADD(rows_to_add, kernel_height, y_stride) \
    CALC_PADDINGS(top_pad, bottom_pad, rows_added, rows_to_add, input_height) \
    CALC_INP_ROW(input_row, rows_added, input_height) \
    xa_nn_circ_buf_nchw_add_rows(p_circ_buf \
                                 ,&(pt_inp)[input_row*input_width] \
                                 ,x_padding \
                                 ,input_width \
                                 ,rows_to_add \
                                 ,top_pad \
                                 ,bottom_pad \
                                 ); \
    UPDATE_ROWS_ADDED(rows_added, rows_to_add) \

/* Add rows from input to circular buffer */
#define CIRC_BUF_ADD_ROWS(rows_added, rows_to_add, top_pad, bottom_pad, input_row, \
    input_height, input_width, circ_out_height, y_stride, x_padding, y_padding, \
    p_circ_buf, pt_inp) \
    CALC_ROWS_TO_ADD(rows_to_add, circ_out_height, y_stride) \
    CALC_PADDINGS(top_pad, bottom_pad, rows_added, rows_to_add, input_height) \
    CALC_INP_ROW(input_row, rows_added, input_height) \
    xa_nn_circ_buf_nchw_add_rows(p_circ_buf \
                                 ,&(pt_inp)[input_row*input_width] \
                                 ,x_padding \
                                 ,input_width \
                                 ,rows_to_add \
                                 ,top_pad \
                                 ,bottom_pad \
                                 ); \
    UPDATE_ROWS_ADDED(rows_added, rows_to_add) \

#define CIRC_BUF_ADD_ROWS_INIT_WITH_PAD_VAL(rows_added, rows_to_add, top_pad, bottom_pad, input_row, \
    input_height, input_width, kernel_height, y_stride, x_padding, y_padding, p_circ_buf, \
    pt_inp, p_pad_val) \
    INIT_ROWS_ADDED(rows_added, y_padding) \
    INIT_ROWS_TO_ADD(rows_to_add, kernel_height, y_stride) \
    CALC_PADDINGS(top_pad, bottom_pad, rows_added, rows_to_add, input_height) \
    CALC_INP_ROW(input_row, rows_added, input_height) \
    xa_nn_circ_buf_nchw_add_rows_with_pad_val(p_circ_buf \
                                              ,&(pt_inp)[input_row*input_width] \
                                              ,x_padding \
                                              ,input_width \
                                              ,rows_to_add \
                                              ,top_pad \
                                              ,bottom_pad \
                                              ,(p_pad_val) \
                                              ); \
    UPDATE_ROWS_ADDED(rows_added, rows_to_add) \

/* Add rows from input to circular buffer */
#define CIRC_BUF_ADD_ROWS_WITH_PAD_VAL(rows_added, rows_to_add, top_pad, bottom_pad, input_row, \
    input_height, input_width, circ_out_height, y_stride, x_padding, y_padding, \
    p_circ_buf, pt_inp, p_pad_val) \
    CALC_ROWS_TO_ADD(rows_to_add, circ_out_height, y_stride) \
    CALC_PADDINGS(top_pad, bottom_pad, rows_added, rows_to_add, input_height) \
    CALC_INP_ROW(input_row, rows_added, input_height) \
    xa_nn_circ_buf_nchw_add_rows_with_pad_val(p_circ_buf \
                                              ,&(pt_inp)[input_row*input_width] \
                                              ,x_padding \
                                              ,input_width \
                                              ,rows_to_add \
                                              ,top_pad \
                                              ,bottom_pad \
                                              ,(p_pad_val) \
                                              ); \
    UPDATE_ROWS_ADDED(rows_added, rows_to_add) \


typedef struct _xa_nn_circ_buf_t
{
    pVOID  p_begin                                             ; /* Start of circular buffer */
    pVOID  p_end                                               ; /* End of circular buffer */
    pVOID  p_curr                                              ; /* Start of current data in circular buffer */
    WORD32 bytewidth                                           ; /* Bytewidth of one element in circular buffer */
    WORD32 rows                                                ; /* Number of rows of circular buffer */
    WORD32 row_offset                                          ; /* Jump required to go next row but same column */
} xa_nn_circ_buf_t;

WORD32 xa_nn_circ_buf_nchw_getsize(
    WORD32 bytewidth,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 circ_buf_height,
    WORD32 output_width);

VOID xa_nn_circ_buf_nchw_init(
    xa_nn_circ_buf_t *p_circ_buf,
    pVOID p_mem,
    WORD32 bytewidth,
    WORD32 input_width,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 x_padding,
    WORD32 circ_buf_height,
    WORD32 output_width,
    pVOID p_pad_val);

void xa_nn_circ_buf_nchw_add_rows(
    xa_nn_circ_buf_t *p_circ_buf,
    const VOID *p_inp,
    WORD32 left_padding,
    WORD32 input_width,
    WORD32 n_rows,
    WORD32 top_pad,
    WORD32 bottom_pad);

void xa_nn_circ_buf_nchw_add_rows_with_pad_val(
    xa_nn_circ_buf_t *p_circ_buf,
    const VOID *p_inp,
    WORD32 left_padding,
    WORD32 input_width,
    WORD32 n_rows,
    WORD32 top_pad,
    WORD32 bottom_pad,
    pVOID p_pad_val);

/***********************************************************************************************/
/* Below macros and functions are for circular buffer for depth first input, here circular buffer
size is kernel_width*input_height*input_channels*channels_multiplier and buffer moves over input
in horizontal direction */

#define INIT_COLS_ADDED(cols_added, x_padding) \
    cols_added = -x_padding;

#define UPDATE_COLS_ADDED(cols_added, cols_to_add) \
    cols_added += cols_to_add;

#define INIT_COLS_TO_ADD(cols_to_add, kernel_width, x_stride) \
    cols_to_add = kernel_width-x_stride;


#define CALC_COLS_TO_ADD(cols_to_add, x_stride) \
    cols_to_add = x_stride;

/* cols_added starts from -x_padding and increases as we added columns to circular buffer,
   if cols_added is less than 0 we have to add that many zero columns at left (left_padding),
   if cols_added is more than input_width we have to add that many zero columns at right
   (right_padding) */
#define CALC_PADDINGS_LR(left_pad, right_pad, cols_added, cols_to_add, input_width) \
    left_pad = -cols_added; \
    LIMIT(left_pad, 0, cols_to_add) \
    right_pad = cols_to_add - (input_width-cols_added); \
    LIMIT(right_pad, 0, cols_to_add) \


/* Calculate the input column index to which the pointer passed to circular buffer
   add column call should point, it must be between 0 to input_width-1 */
#define CALC_INP_COL(input_col, cols_added, input_width) \
    input_col = cols_added; \
    LIMIT(input_col, 0, input_width-1)

/* Initialize the mechanism for adding columns from input to circular buffer */
#define CIRC_BUF_ADD_COLS_INIT(cols_added, cols_to_add, left_pad, right_pad, input_col, \
    input_height, input_width, input_channels, kernel_height, kernel_width, channels_multiplier, x_stride, \
    x_padding, y_padding, output_height, p_circ_buf, pt_inp) \
    INIT_COLS_ADDED(cols_added, x_padding) \
    INIT_COLS_TO_ADD(cols_to_add, kernel_width, x_stride) \
    CALC_PADDINGS_LR(left_pad, right_pad, cols_added, cols_to_add, input_width) \
    CALC_INP_COL(input_col, cols_added, input_width) \
    xa_nn_circ_buf_nhwc_add_cols(p_circ_buf \
                            ,&(pt_inp)[input_col*input_channels] \
                            ,y_padding \
                            ,input_height \
                            ,input_width \
                            ,input_channels \
                            ,kernel_height \
                            ,kernel_width \
                            ,channels_multiplier \
                            ,y_stride \
                            ,y_padding \
                            ,output_height \
                            ,cols_to_add \
                            ,left_pad \
                            ,right_pad \
                            ); \
    UPDATE_COLS_ADDED(cols_added, cols_to_add) \

/* Add columns from input to circular buffer */
#define CIRC_BUF_ADD_COLS(cols_added, cols_to_add, left_pad, right_pad, input_col, \
    input_height, input_width, input_channels, kernel_height, kernel_width, channels_multiplier, x_stride, \
    x_padding, y_padding, output_height, p_circ_buf, pt_inp) \
    CALC_COLS_TO_ADD(cols_to_add, x_stride) \
    CALC_PADDINGS_LR(left_pad, right_pad, cols_added, cols_to_add, input_width) \
    CALC_INP_COL(input_col, cols_added, input_width) \
    xa_nn_circ_buf_nhwc_add_cols(p_circ_buf \
                            ,&(pt_inp)[input_col*input_channels] \
                            ,y_padding \
                            ,input_height \
                            ,input_width \
                            ,input_channels \
                            ,kernel_height \
                            ,kernel_width \
                            ,channels_multiplier \
                            ,y_stride \
                            ,y_padding \
                            ,output_height \
                            ,cols_to_add \
                            ,left_pad \
                            ,right_pad \
                            ); \
    UPDATE_COLS_ADDED(cols_added, cols_to_add) \

/* Initialize the mechanism for adding columns from input to circular buffer */
#define CIRC_BUF_ADD_COLS_INIT_WITH_PAD_VAL(cols_added, cols_to_add, left_pad, right_pad, input_col, \
    input_height, input_width, input_channels, kernel_height, kernel_width, channels_multiplier, x_stride, \
    x_padding, y_padding, output_height, p_circ_buf, pt_inp, p_pad_val) \
    INIT_COLS_ADDED(cols_added, x_padding) \
    INIT_COLS_TO_ADD(cols_to_add, kernel_width, x_stride) \
    CALC_PADDINGS_LR(left_pad, right_pad, cols_added, cols_to_add, input_width) \
    CALC_INP_COL(input_col, cols_added, input_width) \
    xa_nn_circ_buf_nhwc_add_cols_with_pad_val(p_circ_buf \
                            ,&(pt_inp)[input_col*input_channels] \
                            ,y_padding \
                            ,input_height \
                            ,input_width \
                            ,input_channels \
                            ,kernel_height \
                            ,kernel_width \
                            ,channels_multiplier \
                            ,y_stride \
                            ,y_padding \
                            ,output_height \
                            ,cols_to_add \
                            ,left_pad \
                            ,right_pad \
                            ,p_pad_val \
                            ); \
    UPDATE_COLS_ADDED(cols_added, cols_to_add) \

/* Add columns from input to circular buffer */
#define CIRC_BUF_ADD_COLS_WITH_PAD_VAL(cols_added, cols_to_add, left_pad, right_pad, input_col, \
    input_height, input_width, input_channels, kernel_height, kernel_width, channels_multiplier, x_stride, \
    x_padding, y_padding, output_height, p_circ_buf, pt_inp, p_pad_val) \
    CALC_COLS_TO_ADD(cols_to_add, x_stride) \
    CALC_PADDINGS_LR(left_pad, right_pad, cols_added, cols_to_add, input_width) \
    CALC_INP_COL(input_col, cols_added, input_width) \
    xa_nn_circ_buf_nhwc_add_cols_with_pad_val(p_circ_buf \
                            ,&(pt_inp)[input_col*input_channels] \
                            ,y_padding \
                            ,input_height \
                            ,input_width \
                            ,input_channels \
                            ,kernel_height \
                            ,kernel_width \
                            ,channels_multiplier \
                            ,y_stride \
                            ,y_padding \
                            ,output_height \
                            ,cols_to_add \
                            ,left_pad \
                            ,right_pad \
                            ,p_pad_val \
                            ); \
    UPDATE_COLS_ADDED(cols_added, cols_to_add) \

/* Initialize the mechanism for adding columns from input to circular buffer for dilated conv2d depthwise */
#define DILATED_CIRC_BUF_ADD_COLS_INIT(cols_added, cols_to_add, left_pad, right_pad, input_col, \
    input_height, input_width, input_channels, kernel_height, kernel_width, channels_multiplier, \
    dilation_height, dilation_width, x_stride, y_stride, x_padding, y_padding, output_height, p_circ_buf, pt_inp) \
    INIT_COLS_ADDED(cols_added, x_padding) \
    INIT_COLS_TO_ADD(cols_to_add, kernel_width * dilation_width, x_stride) \
    CALC_PADDINGS_LR(left_pad, right_pad, cols_added, cols_to_add, input_width) \
    CALC_INP_COL(input_col, cols_added, input_width) \
    xa_nn_dilated_circ_buf_nhwc_add_cols(p_circ_buf \
                            ,&(pt_inp)[input_col*input_channels] \
                            ,y_padding \
                            ,input_height \
                            ,input_width \
                            ,input_channels \
                            ,kernel_height \
                            ,kernel_width \
                            ,channels_multiplier \
                            ,dilation_height \
                            ,dilation_width \
                            ,y_stride \
                            ,y_padding \
                            ,output_height \
                            ,cols_to_add \
                            ,left_pad \
                            ,right_pad \
                            ); \
    UPDATE_COLS_ADDED(cols_added, cols_to_add) \

/* Add columns from input to circular buffer for dilated conv2d depthwise */
#define DILATED_CIRC_BUF_ADD_COLS(cols_added, cols_to_add, left_pad, right_pad, input_col, \
    input_height, input_width, input_channels, kernel_height, kernel_width, channels_multiplier, \
    dilation_height, dilation_width, x_stride, y_stride, x_padding, y_padding, output_height, p_circ_buf, pt_inp) \
    CALC_COLS_TO_ADD(cols_to_add, x_stride) \
    CALC_PADDINGS_LR(left_pad, right_pad, cols_added, cols_to_add, input_width) \
    CALC_INP_COL(input_col, cols_added, input_width) \
    xa_nn_dilated_circ_buf_nhwc_add_cols(p_circ_buf \
                            ,&(pt_inp)[input_col*input_channels] \
                            ,y_padding \
                            ,input_height \
                            ,input_width \
                            ,input_channels \
                            ,kernel_height \
                            ,kernel_width \
                            ,channels_multiplier \
                            ,dilation_height \
                            ,dilation_width \
                            ,y_stride \
                            ,y_padding \
                            ,output_height \
                            ,cols_to_add \
                            ,left_pad \
                            ,right_pad \
                            ); \
    UPDATE_COLS_ADDED(cols_added, cols_to_add) \

#define DILATED_CIRC_BUF_ADD_COLS_INIT_WITH_PAD_VAL(cols_added, cols_to_add, left_pad, right_pad, input_col, \
    input_height, input_width, input_channels, kernel_height, kernel_width, channels_multiplier, \
    dilation_height, dilation_width, x_stride, y_stride, x_padding, y_padding, output_height, p_circ_buf, pt_inp, p_pad_val) \
    INIT_COLS_ADDED(cols_added, x_padding) \
    INIT_COLS_TO_ADD(cols_to_add, kernel_width * dilation_width, x_stride) \
    CALC_PADDINGS_LR(left_pad, right_pad, cols_added, cols_to_add, input_width) \
    CALC_INP_COL(input_col, cols_added, input_width) \
    xa_nn_dilated_circ_buf_nhwc_add_cols_with_pad_val(p_circ_buf \
                            ,&(pt_inp)[input_col*input_channels] \
                            ,y_padding \
                            ,input_height \
                            ,input_width \
                            ,input_channels \
                            ,kernel_height \
                            ,kernel_width \
                            ,channels_multiplier \
                            ,dilation_height \
                            ,dilation_width \
                            ,y_stride \
                            ,y_padding \
                            ,output_height \
                            ,cols_to_add \
                            ,left_pad \
                            ,right_pad \
                            ,p_pad_val \
                            ); \
    UPDATE_COLS_ADDED(cols_added, cols_to_add) \

/* Add columns from input to circular buffer for dilated conv2d depthwise */
#define DILATED_CIRC_BUF_ADD_COLS_WITH_PAD_VAL(cols_added, cols_to_add, left_pad, right_pad, input_col, \
    input_height, input_width, input_channels, kernel_height, kernel_width, channels_multiplier, \
    dilation_height, dilation_width, x_stride, y_stride, x_padding, y_padding, output_height, p_circ_buf, pt_inp, p_pad_val) \
    CALC_COLS_TO_ADD(cols_to_add, x_stride) \
    CALC_PADDINGS_LR(left_pad, right_pad, cols_added, cols_to_add, input_width) \
    CALC_INP_COL(input_col, cols_added, input_width) \
    xa_nn_dilated_circ_buf_nhwc_add_cols_with_pad_val(p_circ_buf \
                            ,&(pt_inp)[input_col*input_channels] \
                            ,y_padding \
                            ,input_height \
                            ,input_width \
                            ,input_channels \
                            ,kernel_height \
                            ,kernel_width \
                            ,channels_multiplier \
                            ,dilation_height \
                            ,dilation_width \
                            ,y_stride \
                            ,y_padding \
                            ,output_height \
                            ,cols_to_add \
                            ,left_pad \
                            ,right_pad \
                            ,p_pad_val \
                            ); \
    UPDATE_COLS_ADDED(cols_added, cols_to_add) \

#if 0 /* This function is unused in hifi4 nnlib */
WORD32 xa_nn_circ_buf_nhwc_getsize(
    WORD32 bytewidth,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 channels_multiplier,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height);
#endif

VOID xa_nn_circ_buf_nhwc_init(
    xa_nn_circ_buf_t *p_circ_buf,
    pVOID p_mem,
    WORD32 bytewidth,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 channels_multiplier,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height);

WORD32 xa_nn_dilated_circ_buf_nhwc_getsize(
    WORD32 bytewidth,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 channels_multiplier,
    WORD32 dilation_height,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height);

VOID xa_nn_dilated_circ_buf_nhwc_init(
    xa_nn_circ_buf_t *p_circ_buf,
    pVOID p_mem,
    WORD32 bytewidth,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 channels_multiplier,
    WORD32 dilation_height,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height);

void xa_nn_circ_buf_nhwc_add_cols(
    xa_nn_circ_buf_t *p_circ_buf,
    const VOID *p_inp,
    WORD32 top_padding,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 circ_buf_width,
    WORD32 channels_multiplier,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height,
    WORD32 n_cols,
    WORD32 left_pad,
    WORD32 right_pad);

void xa_nn_circ_buf_nhwc_add_cols_with_pad_val(
    xa_nn_circ_buf_t *p_circ_buf,
    const VOID *p_inp,
    WORD32 top_padding,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 circ_buf_width,
    WORD32 channels_multiplier,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height,
    WORD32 n_cols,
    WORD32 left_pad,
    WORD32 right_pad,
    pVOID  p_pad_val);

void xa_nn_dilated_circ_buf_nhwc_add_cols(
    xa_nn_circ_buf_t *p_circ_buf,
    const VOID *p_inp,
    WORD32 top_padding,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 circ_buf_width,
    WORD32 channels_multiplier,
    WORD32 dilation_height,
    WORD32 dilation_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height,
    WORD32 n_cols,
    WORD32 left_pad,
    WORD32 right_pad);

void xa_nn_dilated_circ_buf_nhwc_add_cols_with_pad_val(
    xa_nn_circ_buf_t *p_circ_buf,
    const VOID *p_inp,
    WORD32 top_padding,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 circ_buf_width,
    WORD32 channels_multiplier,
    WORD32 dilation_height,
    WORD32 dilation_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height,
    WORD32 n_cols,
    WORD32 left_pad,
    WORD32 right_pad,
    pVOID  p_pad_val);


#endif /* #ifndef __XA_NN_CIRC_BUF_H__ */
