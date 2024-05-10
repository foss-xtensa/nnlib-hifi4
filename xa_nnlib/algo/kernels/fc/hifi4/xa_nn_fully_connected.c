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
#include "xa_nnlib_common_fpu.h"
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_err_chk.h"

#define ALIGNMENT   8

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_fully_connected_f32,
    (FLOAT32 *__restrict__ p_out
     ,const FLOAT32 *__restrict__ p_weight
     ,const FLOAT32 *__restrict__ p_inp
     ,const FLOAT32 *__restrict__ p_bias
     ,WORD32  weight_depth
     ,WORD32  out_depth
    )
    )
#else /* #if !HAVE_VFPU */
WORD32 xa_nn_fully_connected_f32
  (FLOAT32 *__restrict__ p_out
   ,const FLOAT32 *__restrict__ p_weight
   ,const FLOAT32 *__restrict__ p_inp
   ,const FLOAT32 *__restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_f32xf32_f32
    (p_out
     ,(FLOAT32 *)p_weight
     ,0
     ,(FLOAT32 *)p_inp
     ,0
     ,(FLOAT32 *)p_bias
     ,out_depth
     ,weight_depth
     ,0
     ,weight_depth
     ,0
    );
  return ret;
}
#endif /* #if !HAVE_VFPU */

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_fully_connected_f16,
    (WORD16 *__restrict__ p_out
     ,const WORD16 *__restrict__ p_weight
     ,const WORD16 *__restrict__ p_inp
     ,const WORD16 *__restrict__ p_bias
     ,WORD32  weight_depth
     ,WORD32  out_depth
    )
    )
#else /* #if !HAVE_HP_VFPU */
#ifndef hifi5
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_fully_connected_f16,
    (WORD16 *__restrict__ p_out
     ,const WORD16 *__restrict__ p_weight
     ,const WORD16 *__restrict__ p_inp
     ,const WORD16 *__restrict__ p_bias
     ,WORD32  weight_depth
     ,WORD32  out_depth
    )
    )
#else    
WORD32 xa_nn_fully_connected_f16
  (WORD16 *__restrict__ p_out
   ,const WORD16 *__restrict__ p_weight
   ,const WORD16 *__restrict__ p_inp
   ,const WORD16 *__restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_f16xf16_f16
    (p_out
     ,(WORD16 *)p_weight
     ,0
     ,(WORD16 *)p_inp
     ,0
     ,(WORD16 *)p_bias
     ,out_depth
     ,weight_depth
     ,0
     ,weight_depth
     ,0
    );
  return ret;
}
#endif
#endif /* #if !HAVE_HP_VFPU */

WORD32 xa_nn_fully_connected_16x16_16
  (pWORD16 __restrict__ p_out
   ,pWORD16  __restrict__ p_weight
   ,pWORD16 __restrict__ p_inp
   ,pWORD16 __restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
   ,WORD32  acc_shift
   ,WORD32  bias_shift
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND(((weight_depth&3) != 0), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_16x16_16
    (p_out
     ,p_weight
     ,0
     ,p_inp
     ,0
     ,p_bias
     ,out_depth
     ,weight_depth
     ,0
     ,weight_depth
     ,0
     ,acc_shift
     ,bias_shift
    );
  return ret;
}

WORD32 xa_nn_fully_connected_8x16_16
  (pWORD16 __restrict__ p_out
   ,pWORD8  __restrict__ p_weight
   ,pWORD16 __restrict__ p_inp
   ,pWORD16 __restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
   ,WORD32  acc_shift
   ,WORD32  bias_shift
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND(((weight_depth&3) != 0), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_8x16_16
    (p_out
     ,p_weight
     ,0
     ,p_inp
     ,0
     ,p_bias
     ,out_depth
     ,weight_depth
     ,0
     ,weight_depth
     ,0
     ,acc_shift
     ,bias_shift
    );
  return ret;
}

WORD32 xa_nn_fully_connected_8x8_8
  (pWORD8 __restrict__ p_out
   ,pWORD8  __restrict__ p_weight
   ,pWORD8 __restrict__ p_inp
   ,pWORD8 __restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
   ,WORD32  acc_shift
   ,WORD32  bias_shift
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND(((weight_depth&3) != 0), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_8x8_8
    (p_out
     ,p_weight
     ,0
     ,p_inp
     ,0
     ,p_bias
     ,out_depth
     ,weight_depth
     ,0
     ,weight_depth
     ,0
     ,acc_shift
     ,bias_shift
    );
  return ret;
}

WORD32 xa_nn_fully_connected_asym8xasym8_asym8
  (UWORD8 *__restrict__ p_out
   ,const UWORD8 *__restrict__ p_weight
   ,const UWORD8 *__restrict__ p_inp
   ,const WORD32 *__restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
   ,WORD32  input_zero_bias
   ,WORD32  weight_zero_bias
   ,WORD32  out_multiplier
   ,WORD32  out_shift
   ,WORD32  out_zero_bias
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -255 || input_zero_bias > 0), -1);
  XA_NNLIB_ARG_CHK_COND((weight_zero_bias < -255 || weight_zero_bias > 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < 0 || out_zero_bias > 255), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_asym8xasym8_asym8
    (p_out
     ,p_weight
     ,0
     ,p_inp
     ,0
     ,p_bias
     ,out_depth
     ,weight_depth
     ,0
     ,weight_depth
     ,0
     ,weight_zero_bias
     ,0
     ,input_zero_bias
     ,0
     ,out_multiplier
     ,out_shift
     ,out_zero_bias
    );
  return ret;
}

WORD32 xa_nn_fully_connected_sym8sxasym8s_asym8s
  (WORD8 *__restrict__ p_out
   ,const WORD8 *__restrict__ p_weight
   ,const WORD8 *__restrict__ p_inp
   ,const WORD32 *__restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
   ,WORD32  input_zero_bias
   ,WORD32  out_multiplier
   ,WORD32  out_shift
   ,WORD32  out_zero_bias
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
#if 0
  XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, ALIGNMENT, -1);
#else
  /* For TF Micro lite testing */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
#endif
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_sym8sxasym8s_asym8s
    (p_out
     ,p_weight
     ,0
     ,p_inp
     ,0
     ,p_bias
     ,out_depth
     ,weight_depth
     ,0
     ,weight_depth
     ,0
     ,input_zero_bias
     ,0
     ,out_multiplier
     ,out_shift
     ,out_zero_bias
    );
  return ret;
}

WORD32 xa_nn_fully_connected_sym8sxsym16s_sym16s
  (WORD16 *__restrict__ p_out
   ,const WORD8 *__restrict__ p_weight
   ,const WORD16 *__restrict__ p_inp
   ,const WORD64 *__restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
   ,WORD32  out_multiplier
   ,WORD32  out_shift
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 15), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_sym8sxsym16s_sym16s
    (p_out
     ,p_weight
     ,0
     ,p_inp
     ,0
     ,p_bias
     ,out_depth
     ,weight_depth
     ,0
     ,weight_depth
     ,0
     ,out_multiplier
     ,out_shift
    );
  return ret;
}

WORD32 xa_nn_fully_connected_asym8sxasym8s_asym8s
  (WORD8 *__restrict__ p_out
   ,const WORD8 *__restrict__ p_weight
   ,const WORD8 *__restrict__ p_inp
   ,const WORD32 *__restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
   ,WORD32  input_zero_bias
   ,WORD32  weight_zero_bias
   ,WORD32  out_multiplier
   ,WORD32  out_shift
   ,WORD32  out_zero_bias
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((weight_zero_bias < -127 || weight_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_asym8sxasym8s_asym8s
    (p_out
     ,p_weight
     ,0
     ,p_inp
     ,0
     ,p_bias
     ,out_depth
     ,weight_depth
     ,0
     ,weight_depth
     ,0
     ,weight_zero_bias
     ,0
     ,input_zero_bias
     ,0
     ,out_multiplier
     ,out_shift
     ,out_zero_bias
    );
  return ret;
}

WORD32 xa_nn_fully_connected_asym4sxasym8s_asym8s
  (WORD8 *__restrict__ p_out
   ,const WORD8 *__restrict__ p_weight
   ,const WORD8 *__restrict__ p_inp
   ,const WORD32 *__restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
   ,WORD32  input_zero_bias
   ,WORD32  weight_zero_bias
   ,WORD32  out_multiplier
   ,WORD32  out_shift
   ,WORD32  out_zero_bias
   ,VOID *p_scratch
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((weight_zero_bias < -127 || weight_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND(((weight_depth % 2) != 0), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_asym4sxasym8s_asym8s
    (p_out
     ,p_weight
     ,0
     ,p_inp
     ,0
     ,p_bias
     ,out_depth
     ,weight_depth
     ,0
     ,weight_depth
     ,0
     ,weight_zero_bias
     ,0
     ,input_zero_bias
     ,0
     ,out_multiplier
     ,out_shift
     ,out_zero_bias
     ,p_scratch
    );
  return ret;
}
