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
/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"
#include "../../../ndsp/hifi4/include/NatureDSP_Signal_math.h"
#include "xa_nnlib_common.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_logn_f32_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp,
                WORD32 num_elm
             )
           )
#else
WORD32 xa_nn_elm_logn_f32_f32(FLOAT32 * __restrict__ p_out,
                              const FLOAT32 * __restrict__ p_inp,
                              WORD32 num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

  xa_nnlib_vec_lognf(p_out,
            p_inp,
            num_elm);

  return 0;
}
#endif /* !HAVE_VFPU */
