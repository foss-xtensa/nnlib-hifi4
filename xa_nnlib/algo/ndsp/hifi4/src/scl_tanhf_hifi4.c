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
    Hyperbolic Tangent
    Code optimized for HiFi4 core
  IntegrIT, 2006-2018
*/
#include <errno.h>
#include <fenv.h>

#include "../include/NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "xa_nn_common.h"
#include "xa_nnlib_common_fpu.h"

#include "xa_nnlib_common.h"

/* Tables and constants. */
#include "../include/tanhf_tbl.h"
#include "../include/expf_tbl.h"
#if 0
#include "baseop.h"
#endif
#include <math.h>
/* Constants and polynomial coeffs for exp(x) approximation. */
#include "../include/expf_tbl.h"
#include "../include/pow2f_tbl.h"

/* If non-zero, set errno and raise floating-point exceptions on errors. */
#if 0
float32_t halfexpf(float32_t* dy, float32_t x );
#endif

/*-------------------------------------------------------------------------
  Hyperbolic Tangent
  The functions compute the hyperbolic tangent of input argument. 32-bit
  fixed-point functions accept inputs in Q6.25 and form outputs in Q16.15
  format.

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB.
  f      floating point input, floating point output, Accuracy: 2 ULP
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
DISCARD_FUN_FOR_NONVOID_RETURN(float32_t,xa_nnlib_scl_tanhf,(float32_t x))
#else
float32_t xa_nnlib_scl_tanhf( float32_t x )
{
    float32_t zero, one, two, half, z, r, eps;
    float32_t y;
    float32_t p0, dy, y1;
    int32_t ux;
    int32_t e1, e2;
    int32_t SCF; /* Floating-point Status and Control Register values. */
#if HAVE_VFPU
    if ( xtbool2_extract_0(XT_UN_SX2(x,x)) )
#else
    if ( XT_UN_S(x,x))
#endif
    {
        __Pragma( "frequency_hint never" );
        errno = EDOM;
        return XT_ADD_S(x,x);
    }

    SCF = XT_RUR_FSR(); /* Sample floating-point exception flags. */

    zero = (float32_t)XT_CONST_S(0);
    one = (float32_t)XT_CONST_S(1);
    two = (float32_t)XT_CONST_S(2);
    half = (float32_t)XT_CONST_S(3);
    ux = XT_RFR(x);
    ux = (ux & 0x80000000);
    x = XT_ABS_S(x);
    if (x > xa_nnlib_halfln3.f)
    {
        /*
        * For a large input value tanh(x) is computed from exp(2*x)/2, using
        * the following identity: tanh(x) == 1 - 2/(exp(2*x)+1)
        */
        r = zero; XT_MADDN_S(r, two, x); x = r;
        {
            xtfloat t=(xtfloat)80.f;
            x = XT_MIN_S(x, t);
        }

        /* scale input to 1/ln(2) */
        p0 = XT_MUL_S(x, xa_nnlib_log2_e[0].f);
        #if defined(XT_FIROUND_S)
        p0 = XT_FIROUND_S(p0);
        #else
        p0 = XT_FLOAT_S(XT_ROUND_S(p0, 0), 0);
        #endif
        dy = XT_NEG_S(p0);
        XT_MADD_S(dy, x, xa_nnlib_log2_e[0].f);
        XT_MADD_S(dy, x, xa_nnlib_log2_e[1].f);
        /* compute 2^x */
        {
            float32_t y0, y2, y3, y4, y5, y6, dy2;
            dy2 = XT_MUL_S(dy, dy);
            y0 = xa_nnlib_pow2f_coef[0].f;
            y1 = xa_nnlib_pow2f_coef[1].f;
            y2 = xa_nnlib_pow2f_coef[2].f;
            y3 = xa_nnlib_pow2f_coef[3].f;
            y4 = xa_nnlib_pow2f_coef[4].f;
            y5 = xa_nnlib_pow2f_coef[5].f;
            y6 = xa_nnlib_pow2f_coef[6].f;
            XT_MADD_S(y1, y0, dy);
            XT_MADD_S(y3, y2, dy);
            XT_MADD_S(y5, y4, dy);

            XT_MADD_S(y3, y1, dy2);
            XT_MADD_S(y5, y3, dy2);
            XT_MADD_S(y6, y5, dy);
            y = y6;
        }

        /* resulted scaling */
        {
            xtfloat t;
            t=(xtfloat) 129.f;p0=XT_MIN_S(p0,t);
            t=(xtfloat)-151.f;p0=XT_MAX_S(p0,t);
        }

        /* Apply exponential part to the result */
        {
            uint32_t tmp, v1, v2;
            tmp = XT_TRUNC_S(p0, 0);
            tmp = tmp+254 - 30 - 1;
            v1 = (tmp>>1);
            v2 = (tmp-v1);
            e1 = (v1<<23);
            e2 = (v2<<23);
        }

        /*
        * Convert (y*2^(ex-30))/2 to floating-point p == exp(x)/2
        */
        r = XT_MUL_S(y, 1073741824.f);
        y = XT_MUL_S(r, XT_WFR(e2));
        y = XT_MUL_S(y, XT_WFR(e1));
        z = XT_ADD_S(y, half);
        /* Initial approximation for 1/y */
        r = XT_RECIP0_S(z);
        /* 2 Newton-Raphson iterations for 1/z  */
        eps = one; XT_MSUB_S(eps, z, r);
        XT_MADD_S(r, r, eps);
        eps = one; XT_MSUB_S(eps, z, r);
        XT_MADD_S(r, r, eps);
        z = XT_SUB_S(one, r);
    }
    else
    {
        /*
        * Use polynomial approximation for small input values. This branch is
        * also used for a NaN on input.
        */

        float32_t x2, x3, tn0, tn1, tn2, tn3;
        x2 = XT_MUL_S(x, x);
        x3 = XT_MUL_S(x, x2);
        tn0 = xa_nnlib_polytanhf_tbl[0].f;
        tn1 = xa_nnlib_polytanhf_tbl[1].f;
        tn2 = xa_nnlib_polytanhf_tbl[2].f;
        tn3 = xa_nnlib_polytanhf_tbl[3].f;
        XT_MADD_S(tn1, tn0, x2);
        XT_MADD_S(tn2, tn1, x2);
        XT_MADD_S(tn3, tn2, x2);
        z = x;
        XT_MADD_S(z, tn3, x3);
    }
    /* apply sign */
    XT_MOVT_S(z,XT_NEG_S(z),AE_MOVBA(((uint32_t)ux)>>31));

    XT_WUR_FSR(SCF);
    return (z);
} /* tanhf() */
#endif

