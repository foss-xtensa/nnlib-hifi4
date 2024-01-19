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
#include "xa_type_def.h"
#include "../../../ndsp/hifi4/include/NatureDSP_Signal_math.h"
#include "xa_nnlib_err_chk.h"
#include <math.h>

#define LIMIT_SX2(out, inp, min, max){\
        out = XT_MAX_SX2(min, inp);\
        out = XT_MIN_SX2(max, out);\
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
    ae_valign align_inp, align_out; 

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

    pi = (xtfloatx2 *)p_vec;
    po = (xtfloatx2 *)p_out;

    min  = (xtfloatx2) activation_min;
    max  = (xtfloatx2) activation_max;

    align_inp = XT_LASX2PP(pi);
    align_out = AE_ZALIGN64();

    if(activation_max == INFINITY)
    {
        for(i=0; i<(vec_length >> 1); i++)
        {
            XT_LASX2IP(x, align_inp, pi);
            y = XT_MAX_SX2(min, x);
            XT_SASX2IP(y, align_out, po);
        }

        XT_SASX2POSFP(align_out, po);

        if(vec_length & 1)
        {
            x = p_vec[vec_length -1];
            p_out[vec_length - 1] = XT_MAX_SX2(min, x);
        }
    }
    else
    {
        for(i=0; i<(vec_length >> 1); i++)
        {
            XT_LASX2IP(x, align_inp, pi);
            LIMIT_SX2(y, x, min, max)
            XT_SASX2IP(y, align_out, po);
        }

        XT_SASX2POSFP(align_out, po);

        if(vec_length & 1)
        {
            x = p_vec[vec_length - 1];
            LIMIT_SX2(y, x, min, max);
            p_out[vec_length - 1] = y;
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
  xa_nnlib_vec_sigmoidf(p_out, p_vec, vec_length);
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
  xa_nnlib_vec_tanhf(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_relu_std_f32_f32,(
    FLOAT32       *  p_out,
    const FLOAT32 *  p_vec,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_relu_std_f32_f32(
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
#if !HAVE_VFPU && !HAVE_FPU
#elif HAVE_VFPU
    ae_valign aY;
    const xtfloatx2* restrict pX=(const xtfloatx2*)p_vec;
          xtfloatx2* restrict pY=(      xtfloatx2*)p_out;
    xtfloatx2 zero=XT_CONST_S(0);
    int n;
    int N = vec_length;
    if (N<=0) return 0;
    if (((uintptr_t)pX)&7)
    {
        xtfloat t;
        XT_LSIP(t,castxcc(xtfloat,pX),sizeof(xtfloat));
        t=XT_MAX_S(t,zero);
        XT_SSIP(t,castxcc(xtfloat,pY),sizeof(xtfloat));
        N--;
    }
    aY=AE_ZALIGN64();
    for(n=0; n<(N>>1); n++)
    {
        xtfloatx2 t;
        XT_LSX2IP(t,pX,sizeof(xtfloatx2));
        t=XT_MAX_SX2(t,zero);
        XT_SASX2IP(t,aY,pY);
    }
    AE_SA64POS_FP(aY,pY);
    if(N&1)
    {
        xtfloat t;
        t=XT_LSI((const xtfloat*)pX,0);
        t=XT_MAX_S(t,zero);
        XT_SSI(t,(xtfloat*)pY,0);
    }
#else
// code for scalar FPU
    const xtfloat* restrict pX=(const xtfloat*)p_vec;
          xtfloat* restrict pY=(      xtfloat*)p_out;
    xtfloat t,zero=XT_CONST_S(0);
    xtbool bbig,bneg;
    int n;
    int N = vec_length;
    for(n=0; n<N; n++)
    {
        XT_LSIP(t,pX,sizeof(float32_t));
        bneg=XT_OLT_S(t,zero);
        XT_MOVT_S(t,zero,bneg);
        XT_SSIP(t,pY,sizeof(float32_t));
    }
#endif
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
  xa_nnlib_vec_reluf(p_out, p_vec, threshold, vec_length);
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
  xa_nnlib_vec_reluf(p_out, p_vec, 1.0f, vec_length);
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
  xa_nnlib_vec_reluf(p_out, p_vec, 6.0f, vec_length);
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
  xa_nnlib_vec_softmaxf(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */
#endif

