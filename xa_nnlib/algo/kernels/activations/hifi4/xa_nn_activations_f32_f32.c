/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
#include "common_fpu.h"
#include "xa_type_def.h"
#include "NatureDSP_Signal_math.h"
#include "xa_nnlib_err_chk.h"
#include <math.h>

#define ALIGNMENT   8   /* 8 bytes alignment */
#define LIMIT_SX2(out, inp, min, max){\
        out = XT_MAX_SX2(min, inp);\
        out = XT_MIN_SX2(out, max);\
}

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_activation_min_max_f32_f32,(               
            FLOAT32 *  p_out,        
    const   FLOAT32 *  p_vec,
            FLOAT32    activation_min,
            FLOAT32    activation_max,
            WORD32     vec_length))
#else
/*xa_nn_vec_activation_min_max_f32_f32() 
 * inp: p_vec: 8 byte aligned pointer
 * out: p_out: 8 byte aligned pointer */

WORD32 xa_nn_vec_activation_min_max_f32_f32(FLOAT32 * __restrict__ p_out, 
           const  FLOAT32 * __restrict__ p_vec, 
                  FLOAT32 activation_min,
                  FLOAT32 activation_max,
                  WORD32  vec_length)
{
    int i;
    xtfloatx2 x, y, min, max;
    xtfloatx2 *pi, *po; 

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);

    pi = (xtfloatx2 *)p_vec;
    po = (xtfloatx2 *)p_out;

    min  = (xtfloatx2) activation_min;
    max  = (xtfloatx2) activation_max;
    
    if(activation_max == INFINITY)
    {
        for(i=0; i<(vec_length >> 1); i++)
        {
            XT_LSX2IP(x, pi, sizeof(xtfloatx2));        
            y = XT_MAX_SX2(min, x);
            XT_SSX2IP(y, po, sizeof(xtfloatx2));
        }

        if(vec_length & 1)
        {   
            xtfloat x1;
            XT_LSIP(x1, (xtfloat *)pi, sizeof(xtfloat));        
            x = x1;
            y = XT_MAX_SX2(min, x);
            XT_SSIP(XT_HIGH_S(y), (xtfloat *)po, sizeof(xtfloat));
        }
    }
    else
    {   
        for(i=0; i<(vec_length >> 1); i++)
        {
            XT_LSX2IP(x, pi, sizeof(xtfloatx2));        
            LIMIT_SX2(y, x, min, max)
            XT_SSX2IP(y, po, sizeof(xtfloatx2));
        }

        if(vec_length & 1)
        {   
            xtfloat x1;
            XT_LSIP(x1, (xtfloat *)pi, sizeof(xtfloat));        
            x = x1;
            LIMIT_SX2(y, x, min, max)
            XT_SSIP(XT_HIGH_S(y), (xtfloat *)po, sizeof(xtfloat));
        }
    }
    
    return 0;    
}
#endif

#if HAVE_VFPU

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_sigmoid_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_sigmoid_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_sigmoidf(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_tanh_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_tanh_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_tanhf(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_relu_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    FLOAT32       threshold,                   
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_relu_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    FLOAT32       threshold,                   /* threshold, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_reluf(p_out, p_vec, threshold, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_relu1_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_relu1_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_reluf(p_out, p_vec, 1.0f, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_relu6_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_relu6_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_reluf(p_out, p_vec, 6.0f, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_softmax_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_softmax_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_softmaxf(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */
#endif

