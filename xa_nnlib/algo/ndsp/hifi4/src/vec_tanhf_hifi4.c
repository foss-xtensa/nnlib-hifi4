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
#include "../include/NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "xa_nn_common.h"
#include "xa_nnlib_common_fpu.h"

#include "xa_nnlib_common.h"

/* Tables and constants. */
#include "../include/tanhf_tbl.h"
#include "../include/expf_tbl.h"
#include "../include/nanf_tbl.h"
#include "../include/pow2f_tbl.h"

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
DISCARD_FUN(void,xa_nnlib_vec_tanhf,(float32_t* y, const float32_t* x, int N))
#elif HAVE_VFPU
void xa_nnlib_vec_tanhf(float32_t* restrict y, const float32_t* restrict x, int N)
{
#define SCR_SZ (MAX_ALLOCA_SZ/(2*sizeof(float32_t)))
    float32_t ALIGN(8) scratch[SCR_SZ];
    const ae_int32* restrict pPolytanhf=(const ae_int32*)xa_nnlib_polytanhf_tbl;
          xtfloatx2 * restrict pScrWr;
    const xtfloatx2 * restrict pScrRd;
    ae_valign aX,aY;
    const xtfloatx2* restrict pX;
          xtfloatx2* restrict pY;
    xtfloatx2 one =XT_CONST_S(1);
    xtfloatx2 two =XT_CONST_S(2);
    xtfloatx2 half=XT_CONST_S(3);
    int n,m,M;
    if (N<=0) return;
    if (N&1) 
    {
        *y++=xa_nnlib_scl_tanhf(*x++); N--;
    }
    if (N<=0) return;
    for (m=0; m<N; m+=SCR_SZ/2,x+=SCR_SZ/2,y+=SCR_SZ/2)
    {
        M=XT_MIN(N-m,SCR_SZ/2);
        /*
        * For a large input value tanh(x) is computed from exp(2*x)/2, using
        * the following identity: tanh(x) == 1 - 2/(exp(2*x)+1)
        */
        /* argumant reduction phase */
        pX    =(const xtfloatx2*)x;
        aX=AE_LA64_PP(pX); 
        pScrWr=(      xtfloatx2*)scratch;
        for (n = 0; n < (M>>1); n++) 
        {
            xtfloatx2 d, p0, dy,t;
            XT_LASX2IP(d,aX,pX);
            d = XT_ABS_SX2(d);
            d=XT_MUL_SX2(two, d); 
            t=(xtfloatx2)80.f; d = XT_MIN_SX2(d, t);

            /* scale input to 1/ln(2) */
            p0 = XT_MUL_SX2(d, xa_nnlib_log2_e[0].f);
            #if defined(XT_FIROUND_SX2)
            p0 = XT_FIROUND_SX2(p0);
            #else
            p0 = XT_FLOAT_SX2(XT_ROUND_SX2(p0, 0), 0);
            #endif
            dy = XT_NEG_SX2(p0);

            XT_MADD_SX2(dy, d, xa_nnlib_log2_e[0].f);
            XT_MADD_SX2(dy, d, xa_nnlib_log2_e[1].f);
            XT_SSX2IP(dy ,pScrWr,sizeof(xtfloatx2));
            /* saturating p0 to the right values */
            t=(xtfloatx2) 129.f; p0=XT_MIN_SX2(p0,t);
            t=(xtfloatx2)-151.f; p0=XT_MAX_SX2(p0,t);
            XT_SSX2IP(p0,pScrWr,sizeof(xtfloatx2));
        }
        /* compute 2^x via polynomal appoximation */
        __Pragma("no_reorder")
        pScrRd=(const xtfloatx2*)scratch;
        pScrWr=(      xtfloatx2*)scratch;
        pPolytanhf=(const ae_int32*)xa_nnlib_pow2f_coef;
        for (n = 0; n < (M>>1); n++) 
        {
            xtfloatx2 dy, y0,y1, y2, y3, y4, y5, y6, y7, dy2;
            ae_int32x2 tmp;
            XT_LSX2IP(dy ,pScrRd,2*sizeof(xtfloatx2));
            dy2 = XT_MUL_SX2(dy, dy);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y2 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y3 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_XP(tmp,pPolytanhf,-4*(int)sizeof(float32_t));   y4 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            y5 = xa_nnlib_pow2f_coef[5].f;
            y6 = xa_nnlib_pow2f_coef[6].f;
            XT_MADD_SX2(y1, y0, dy);
            XT_MADD_SX2(y3, y2, dy);
            XT_MADD_SX2(y5, y4, dy);
            XT_MADD_SX2(y3, y1, dy2);
            XT_MADD_SX2(y5, y3, dy2);
            XT_MADD_SX2(y6, y5, dy);
            y7 = y6;
            XT_SSX2IP(y7 ,pScrWr,2*sizeof(xtfloatx2));
        }
        /* resulted scaling by 2^N and final Newton-Raphson phase */
        __Pragma("no_reorder")
        pScrRd=(const xtfloatx2*)scratch;
        pScrWr=(      xtfloatx2*)scratch;
        for (n = 0; n < (M>>1); n++) 
        {
            xtfloatx2  d, z, r, eps, p0;
            ae_int32x2 tmp, v1, v2, e1, e2;
            XT_LSX2IP(d ,pScrRd,sizeof(xtfloatx2));
            XT_LSX2IP(p0,pScrRd,sizeof(xtfloatx2));

            /* Apply exponential part to the result */
            tmp = XT_TRUNC_SX2(p0, 0);
            tmp = AE_ADD32(tmp,254 - 1);
            v1 = AE_SRLI32(tmp,1);
            v2 = AE_SUB32(tmp,v1);
            e1 = AE_SLLI32(v1,23);
            e2 = AE_SLLI32(v2,23);
            /*
            * Convert (y*2^(ex-30))/2 to floating-point p == exp(x)/2
            */
            d = XT_MUL_SX2(d, XT_AE_MOVXTFLOATX2_FROMINT32X2(e2));
            d = XT_MUL_SX2(d, XT_AE_MOVXTFLOATX2_FROMINT32X2(e1));
            z = XT_ADD_SX2(d, half);
            /* Initial approximation for 1/y */
            r = XT_RECIP0_SX2(z);
            /* 2 Newton-Raphson iterations for 1/z  */
            eps = one; XT_MSUB_SX2(eps, z, r);
            XT_MADD_SX2(r, r, eps);
            eps = one; XT_MSUB_SX2(eps, z, r);
            XT_MADD_SX2(r, r, eps);
            z = XT_SUB_SX2(one, r);
            XT_SSX2IP(z,pScrWr,2*sizeof(xtfloatx2));
        }        
        /* next, compute output for smaller argument 
           Use polynomial approximation for small input values. This branch is
           also used for a NaN on input.
        */
        __Pragma("no_reorder")
        pX    =(const xtfloatx2*)x;
        pScrWr=(( xtfloatx2*)scratch)+1;
        aX=AE_LA64_PP(pX); 
        pPolytanhf=(const ae_int32*)xa_nnlib_polytanhf_tbl;
        for (n = 0; n < (M>>1); n++) 
        {
            xtfloatx2 z, x1, x2, x3, tn0, tn1, tn2, tn3;
            XT_LASX2IP(x1,aX,pX);
            x1 = XT_ABS_SX2(x1);
            x2 = XT_MUL_SX2(x1, x1);
            x3 = XT_MUL_SX2(x1, x2);
            ae_int32x2 tmp;
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           tn0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           tn1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           tn2 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_XP(tmp,pPolytanhf,-3*(int)sizeof(float32_t));   tn3 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            XT_MADD_SX2(tn1, tn0, x2);
            XT_MADD_SX2(tn2, tn1, x2);
            XT_MADD_SX2(tn3, tn2, x2);
            z = x1;
            XT_MADD_SX2(z, tn3, x3);
            XT_SSX2IP(z,pScrWr,2*sizeof(xtfloatx2));
        }
        /* final stage: select right output and apply sign */
        __Pragma("no_reorder")
        pX    =(const xtfloatx2*)x;
        pY    =(      xtfloatx2*)y;
        pScrRd=(const xtfloatx2*)scratch;
        aX=AE_LA64_PP(pX); aY=AE_ZALIGN64();
        for (n = 0; n < (M>>1); n++) 
        {
            xtbool2 bbig,bsign;
            xtfloatx2 d, z, zbig;
            ae_int32x2 ux;
            XT_LASX2IP(d,aX,pX);
            ux = XT_AE_MOVINT32X2_FROMXTFLOATX2(d); 
            bsign=AE_LT32(ux,0);
            d = XT_ABS_SX2(d);
            bbig = XT_OLT_SX2(xa_nnlib_halfln3.f,d);
            XT_LSX2IP(zbig,pScrRd,sizeof(xtfloatx2));
            XT_LSX2IP(z   ,pScrRd,sizeof(xtfloatx2));
            XT_MOVT_SX2(z,zbig,bbig);
            /* apply sign */
            XT_MOVT_SX2(z,XT_NEG_SX2(z),bsign);
            XT_SASX2IP(z,aY,pY);
        }
        AE_SA64POS_FP(aY,pY);
    }
}
#else
// code for scalar FPU
void xa_nnlib_vec_tanhf(float32_t* restrict y, const float32_t* restrict x, int N)
{
    xtfloat zero, one, two, half;
    int n;
    const xtfloat* restrict pX=(const xtfloat*)x;
          xtfloat* restrict pY=(      xtfloat*)y;
    zero = XT_CONST_S(0);
    one  = XT_CONST_S(1);
    two  = XT_CONST_S(2);
    half = XT_CONST_S(3);
    for (n = 0; n < N; n++) 
    {
        xtbool bsmall;
        xtfloat x,y;
        xtfloat z, r, eps, zsmall,xin;
        xtfloat p0, dy, y1;
        int32_t ux;
        int32_t e1, e2;
        XT_LSIP(x,pX,sizeof(float32_t));
        ux = XT_RFR(x); 
        ux = (ux & 0x80000000);
        x = XT_ABS_S(x);
        bsmall = XT_OLT_S(xa_nnlib_halfln3.f,x);
        xin=x;
        /* compute output for smaller argument */
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
            zsmall = XT_SUB_S(one, r);
        }
        /* compute output for bigger argument */
        {
            /*
            * Use polynomial approximation for small input values. This branch is
            * also used for a NaN on input.
            */
            x=xin;
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
        XT_MOVT_S(z,zsmall,bsmall);
        /* apply sign */
        XT_MOVT_S(z,XT_NEG_S(z),AE_MOVBA(((uint32_t)ux)>>31));
        XT_SSIP(z,pY,sizeof(float32_t));
    }
}
#endif
