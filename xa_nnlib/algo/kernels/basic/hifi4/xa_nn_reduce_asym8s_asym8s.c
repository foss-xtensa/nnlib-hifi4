/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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
#include "common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_err_chk.h"

WORD32 xa_nn_reduce_max_getsize_nhwc(WORD32  inp_precision
                                    ,const WORD32 *const p_inp_shape
                                    ,WORD32 num_inp_dims
                                    ,const WORD32 *p_axis
                                    ,WORD32 num_axis_dims)
{
  return 0;
}

WORD32 xa_nn_reduce_max_4D_asym8s_asym8s(WORD8 * __restrict__ p_out
                                        ,const WORD32 *const p_out_shape
                                        ,const WORD8 * __restrict__ p_inp
                                        ,const WORD32 *const p_inp_shape
                                        ,const WORD32 * __restrict__ p_axis
                                        ,WORD32 num_out_dims
                                        ,WORD32 num_inp_dims
                                        ,WORD32 num_axis_dims
                                        ,pVOID p_scratch)
{
  return -1;
}
