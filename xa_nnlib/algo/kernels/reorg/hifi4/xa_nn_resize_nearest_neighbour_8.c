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
#include "xa_nnlib_common_fpu.h"
#include "xa_nn_common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"

#include "xa_nnlib_common.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_resize_nearest_neighbour_8_8,
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_inp
  ,WORD32  input_batch
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  out_batch
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  out_channels
  ,FLOAT32 height_scale
  ,FLOAT32 width_scale
  ,FLOAT32 height_offset
  ,FLOAT32 width_offset
  ,WORD32  align_corners
  ))
#else
WORD32 xa_nn_resize_nearest_neighbour_8_8
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_inp
  ,WORD32  input_batch
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  out_batch
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  out_channels
  ,FLOAT32 height_scale
  ,FLOAT32 width_scale
  ,FLOAT32 height_offset
  ,FLOAT32 width_offset
  ,WORD32  align_corners
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_batch <= 0 || input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_batch != input_batch || out_channels != input_channels), -1);

  int itr_n, itr_h, itr_w;

  int width_off  = input_channels;
  int height_off = input_width * width_off;
  int batch_off  = input_height * height_off;

  WORD8 *ptmp_inp = (WORD8 *)p_inp, *ptmp_out = (WORD8 *)p_out;
  WORD8 *ptmp_inp_h, *ptmp_inp_w;

  for(itr_n = 0; itr_n < out_batch; itr_n++)
  {
    for(itr_h = 0; itr_h < out_height; itr_h++)
    {
      xtfloat outh_idx; 
      outh_idx = XT_ADD_S((xtfloat)itr_h, height_offset); 
      outh_idx = XT_MUL_S(outh_idx, height_scale); 
      outh_idx = align_corners ? XT_FIROUND_S(outh_idx) : XT_FIFLOOR_S(outh_idx); 
      outh_idx = XT_MIN_S(outh_idx, input_height - 1);
      outh_idx = XT_MAX_S(0, outh_idx);
      int outh = (int)(outh_idx);
      ptmp_inp_h = ptmp_inp + (outh * height_off);

      for(itr_w = 0; itr_w < out_width; itr_w++)
      {
        xtfloat outw_idx; 
        outw_idx = XT_ADD_S((xtfloat)itr_w, width_offset); 
        outw_idx = XT_MUL_S(outw_idx, width_scale); 
        outw_idx = align_corners ? XT_FIROUND_S(outw_idx) : XT_FIFLOOR_S(outw_idx); 
        outw_idx = XT_MIN_S(outw_idx, input_width - 1);
        outw_idx = XT_MAX_S(0, outw_idx);
        int outw = (int)(outw_idx);
        ptmp_inp_w = ptmp_inp_h + (outw * width_off);

        MEMCPY_8b(ptmp_out, ptmp_inp_w, input_channels);
        ptmp_out += input_channels;
      }
    }
    ptmp_inp += batch_off;
  }

  return 0;
}
#endif

