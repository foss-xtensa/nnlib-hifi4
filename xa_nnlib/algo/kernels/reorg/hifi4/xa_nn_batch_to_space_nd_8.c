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
#include "xa_type_def.h"
#include "xa_nn_common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"

#include "xa_nnlib_common.h"

WORD32 xa_nn_batch_to_space_nd_8_8(
    WORD8 *__restrict__ p_out,
    const WORD32 *const p_out_shape,
    const WORD8 *__restrict__ p_inp,
    const WORD32 *const p_inp_shape,
    const WORD32 *const p_block_sizes,
    const WORD32 *const p_crop_sizes,
    WORD32  num_out_dims,
    WORD32  num_inp_dims)
{
  int i;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_block_sizes, -1);
  XA_NNLIB_ARG_CHK_PTR(p_crop_sizes, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);

  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_block_sizes, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_crop_sizes, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_inp_dims < 3 || num_inp_dims > 4), -1);
  XA_NNLIB_ARG_CHK_COND((num_out_dims != 4), -1);

  /* Shapes/Dimensions related checks */  
  for(i = 0; i < num_inp_dims; i++)
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[i] <= 0), -1);
  for(i = 0; i < (num_inp_dims - 2); i++)
    XA_NNLIB_ARG_CHK_COND((p_block_sizes[i] <= 0), -1);
  for(i = 0; i < 2*(num_inp_dims - 2); i++)
    XA_NNLIB_ARG_CHK_COND((p_crop_sizes[i] < 0), -1);
  for(i = 0; i < num_out_dims; i++)
    XA_NNLIB_ARG_CHK_COND((p_out_shape[i] <= 0), -1);

  int input_batch, input_height, input_width, input_channels;
  int out_batch, out_height, out_width, out_channels;
  int block_size_h, block_size_w;
  int crop_top, crop_left;

  input_batch = p_inp_shape[0];
  input_height = p_inp_shape[1];
  input_width = 1;
  input_channels = p_inp_shape[num_inp_dims - 1];

  out_batch = p_out_shape[0];
  out_height = p_out_shape[1];
  out_width = p_out_shape[2];
  out_channels = p_out_shape[3];

  block_size_h = p_block_sizes[0];
  block_size_w = 1;

  crop_top = p_crop_sizes[0];
  crop_left = 0;

  if(num_inp_dims == 4)
  {
    input_width = p_inp_shape[2];
    block_size_w = p_block_sizes[1];
    crop_left = p_crop_sizes[2];
  }

  XA_NNLIB_ARG_CHK_COND((input_batch != out_batch * block_size_h * block_size_w), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels != out_channels), -1);

  int itr_bh, itr_bw, itr_ih, itr_iw, itr_ib;
  int itr_ob, itr_oh, itr_ow;

  WORD8 *ptmp_inp, *ptmp_out;

  for(itr_bh = 0; itr_bh < block_size_h; itr_bh++)
  {
    for(itr_bw = 0; itr_bw < block_size_w; itr_bw++)
    {
      for(itr_ob = 0; itr_ob < out_batch; itr_ob++)
      {
        itr_ib = (itr_bh * block_size_w + itr_bw) * out_batch + itr_ob;
        for(itr_ih = 0; itr_ih < input_height; itr_ih++)
        {
          itr_oh = itr_ih * block_size_h + itr_bh - crop_top;
          if(itr_oh < 0 || itr_oh >= out_height)
            continue;
          for(itr_iw = 0; itr_iw < input_width; itr_iw++)
          {
            itr_ow = itr_iw * block_size_w + itr_bw - crop_left;
            if(itr_ow < 0 || itr_ow >= out_width)
              continue;
            ptmp_inp = (WORD8 *)(&p_inp[((itr_ib * input_height + itr_ih) * input_width + itr_iw) * input_channels]);
            ptmp_out = (WORD8 *)(&p_out[((itr_ob * out_height + itr_oh) * out_width + itr_ow) * out_channels]);
            MEMCPY_8b(ptmp_out, ptmp_inp, out_channels);
          }
        }
      }
    }
  }

  return 0;
}

