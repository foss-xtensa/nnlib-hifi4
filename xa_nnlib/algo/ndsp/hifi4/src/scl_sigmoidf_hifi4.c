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
/* ------------------------------------------------------------------------ */
/*  IntegrIT, Ltd.   www.integrIT.com, info@integrIT.com                    */
/*                                                                          */
/* DSP Library                                                              */
/*                                                                          */
/* This library contains copyrighted materials, trade secrets and other     */
/* proprietary information of IntegrIT, Ltd. This software is licensed for  */
/* use with Cadence processor cores only and must not be used for any other */
/* processors and platforms. The license to use these sources was given to  */
/* Cadence, Inc. under Terms and Condition of a Software License Agreement  */
/* between Cadence, Inc. and IntegrIT, Ltd.                                 */
/* ------------------------------------------------------------------------ */
/*          Copyright (C) 2015-2018 IntegrIT, Limited.                      */
/*                      All Rights Reserved.                                */
/* ------------------------------------------------------------------------ */
/*
  NatureDSP Signal Processing Library. Vector matematics
    Sigmoid
    Code optimized for HiFi4 core
  IntegrIT, 2006-2018
*/
#include "../include/NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "xa_nn_common.h"
#include "xa_nnlib_common_fpu.h"
#include <math.h>

/*-------------------------------------------------------------------------
  Sigmoid
  The functions compute the sigmoid of input argument. 32-bit fixed-point
  functions accept inputs in Q6.25 and form outputs in Q16.15 format.

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB.
  f      floating point input, floating point output. Accuracy 2 ULP
  Input:
  x[N]   input data, Q6.25 or floating point
  N      length of vectors
  Output:
  y[N]   result, Q16.15 or floating point

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  return result, Q16.15 or floating point
-------------------------------------------------------------------------*/
#if !HAVE_VFPU && !HAVE_FPU
DISCARD_FUN_FOR_NONVOID_RETURN(float32_t,xa_nnlib_scl_sigmoidf,(float32_t x))
#else
float32_t xa_nnlib_scl_sigmoidf(float32_t x)
{
    static const union ufloat32uint32 c[]={{0x3fb8aa3b},{0x32a57060}};
    static const union ufloat32uint32 p[]={{0x39222a75},{0x3aaf9334},{0x3c1d94fc},{0x3d63578b},{0x3e75fdf0},{0x3f317218},{0x3f800000}};

    int s;
    int32_t n,n0,n1;
    xtfloat s0,s1;
    xtfloat x0,y,z,d,t;
    s=x<0.f;
    x=XT_NEG_S(XT_ABS_S(x));
#if HAVE_VFPU
    XT_MOVNEZ_S(x,-103.9721f,xtbool2_extract_0(XT_OLT_S(x,-103.9721f)));
#else
    XT_MOVNEZ_S(x,-103.9721f,XT_OLT_S(x,-103.9721f));
#endif
    /* compute d+n=log2(e)*x */
#if defined(XT_FIROUND_S)
    y=XT_FIROUND_S(XT_MUL_S(x,c[0].f));
#else
    y=XT_FLOAT_S(XT_ROUND_S(XT_MUL_S(x,c[0].f),0),0);
#endif
    d=XT_NEG_S(y);
    XT_MADDN_S(d,x,c[0].f);
    XT_MADDN_S(d,x,c[1].f);
    n=XT_TRUNC_S(y,0);
    /* approx 2^d */
    {
        xtfloat d2,z0,z1;
        d2=XT_MUL_S(d,d);
        z0=p[0].f;
        t =p[2].f; XT_MADDN_S(t,d2,z0); z0=t;
        t =p[4].f; XT_MADDN_S(t,d2,z0); z0=t;
        z1=p[1].f;
        t =p[3].f; XT_MADDN_S(t,d2,z1); z1=t;
        t =p[5].f; XT_MADDN_S(t,d2,z1); z1=t;
        XT_MADDN_S(z1,z0,d);
        z=z1;
    }
    t=XT_CONST_S(1); XT_MADDN_S(t,d,z); z=t;
    /* compute approx x0 - it does not give right values on denorm values but it is ok for further computing 1/(1+x) */
    s0=XT_WFR((XT_MAX((n+127),0)<<23));
    x0=z;
    x0=XT_MUL_S(x0,s0);
    /* simplified ldexpf */
    n0=(n>>1);
    n1=(n-n0);
    n1=(n1+127);
    n0=(n0+127);
    n1=(n1<<23);
    n0=(n0<<23);
    s0=XT_WFR(n0);
    s1=XT_WFR(n1);
    x=XT_MUL_S(XT_MUL_S(z,s0),s1);
    /* approx y=1/(1+x); */
    y=XT_RECIP_S(XT_ADD_S(XT_CONST_S(1),x0));
    t=XT_MUL_S(y,x);
    XT_MOVNEZ_S(y,t,s);
    return y;
}
#endif
