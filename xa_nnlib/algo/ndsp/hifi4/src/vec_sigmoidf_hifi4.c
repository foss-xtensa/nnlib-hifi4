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
DISCARD_FUN(void,xa_nnlib_vec_sigmoidf,(float32_t * y, const float32_t * x, int N))
#elif HAVE_VFPU
void xa_nnlib_vec_sigmoidf    (float32_t * y, const float32_t * x, int N)
{
#define SCR_SZ (MAX_ALLOCA_SZ/(2*sizeof(int32_t)))
    int32_t ALIGN(8) scratch[SCR_SZ*2];
    int m,M;
    static const union ufloat32uint32 c[]={{0x3fb8aa3b},{0x32a57060}}; 
    static const union ufloat32uint32 p[]={{0x39222a75},{0x3aaf9334},{0x3c1d94fc},{0x3d63578b},{0x3e75fdf0},{0x3f317218},{0x3f800000}};
    ae_valign aX,aY;
    const xtfloatx2 * restrict pX;
          xtfloatx2 * restrict pY;
    const ae_int32x2* restrict pScrRd;
          ae_int32x2* restrict pScrWr;
    const ae_int32  * restrict pP=(const ae_int32  *)p;

    int k;
    if (N<0) return;
    if (N&1) { *y++=xa_nnlib_scl_sigmoidf(*x++); N--; }
    if (N<0) return;
    for (m=0; m<N; m+=SCR_SZ,x+=SCR_SZ,y+=SCR_SZ)
    {
        M=XT_MIN(N-m,SCR_SZ);
        // First phase: argument reduction
        pScrWr=(ae_int32x2* )scratch;
        pX=(const xtfloatx2*)x;
        aX=AE_LA64_PP(pX); 
        for (k = 0; k < (M>>1); k++) 
        {
            //xtbool2 s;
            ae_int32x2 n;
            xtfloatx2 d0,d1,d2;
            XT_LASX2IP(d0,aX,pX);

            //s=XT_OLT_SX2(d0,XT_CONST_S(0));
            d0=XT_NEG_SX2(XT_ABS_SX2(d0));
            d0=XT_MAX_SX2(-103.9721f,d0);
            /* compute d1+n=log2(e)*x */
            #if defined(XT_FIROUND_SX2)
                d2=XT_FIROUND_SX2(XT_MUL_SX2(d0,c[0].f));
            #else
                d2=XT_FLOAT_SX2(XT_ROUND_SX2(XT_MUL_SX2(d0,c[0].f),0),0);
            #endif
            d1=XT_NEG_SX2(d2);
            XT_MADDN_SX2(d1,d0,c[0].f);
            XT_MADDN_SX2(d1,d0,c[1].f);
            n=XT_TRUNC_SX2(d2,0);
            AE_S32X2_IP(n,pScrWr,sizeof(ae_int32x2));
            XT_SSX2IP(d1,castxcc(xtfloatx2,pScrWr),sizeof(ae_int32x2));
        }
        // second phase: compute polynomial approximation
        __Pragma("no_reorder")
        pScrRd=( const ae_int32x2* )scratch;
        pScrWr=((      ae_int32x2* )scratch)+1;
        pX=(const xtfloatx2*)x;
        pY=(      xtfloatx2*)y;
        aX=AE_LA64_PP(pX); aY=AE_ZALIGN64();
        for (k = 0; k < (M>>1); k++) 
        {
            xtbool2 s;
            ae_int32x2 n;
            xtfloatx2 a,d,z,t,z0;
            XT_LASX2IP(a,aX,pX);
            AE_L32X2_IP(n,pScrRd,sizeof(ae_int32x2));
            XT_LSX2IP(d,castxcc(xtfloatx2,pScrRd),sizeof(ae_int32x2));

            s=XT_OLT_SX2(a,XT_CONST_S(0));
            /* approx 2^d */
            {
                xtfloatx2 d2,z1,z2;
                d2=XT_MUL_SX2(d,d);
                { ae_int32x2 tmp; AE_L32_IP(tmp,pP,sizeof(float32_t)); z2= XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); }
                { ae_int32x2 tmp; AE_L32_IP(tmp,pP,sizeof(float32_t)); z1= XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); }
                { ae_int32x2 tmp; AE_L32_IP(tmp,pP,sizeof(float32_t)); t = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); } XT_MADDN_SX2(t,d2,z2); z2=t;
                { ae_int32x2 tmp; AE_L32_IP(tmp,pP,sizeof(float32_t)); t = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); } XT_MADDN_SX2(t,d2,z1); z1=t;
                { ae_int32x2 tmp; AE_L32_IP(tmp,pP,sizeof(float32_t)); t = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); } XT_MADDN_SX2(t,d2,z2); z2=t;
                { ae_int32x2 tmp; AE_L32_XP(tmp,pP,-5*(int)sizeof(float32_t)); t = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); } XT_MADDN_SX2(t,d2,z1); z1=t;
                XT_MADDN_SX2(z1,z2,d);
                z=z1;
            }
            t=XT_CONST_S(1); XT_MADDN_SX2(t,d,z); z=t;
            t=XT_AE_MOVXTFLOATX2_FROMINT32X2(AE_SLLI32(AE_MAX32(AE_ADD32(n,127),0),23));
            z0=XT_MUL_SX2(z,t);
            XT_MOVT_SX2(z,XT_NEG_SX2(z),s); /* save orignal sign as a sign of polynomial */
            XT_SASX2IP(z,aY,pY);
            XT_SSX2IP(z0,castxcc(xtfloatx2,pScrWr),2*sizeof(ae_int32x2));
        }
        AE_SA64POS_FP(aY,pY);
        // last phase: scale polynomial to 2^n and compute 1/(1+x)
        __Pragma("no_reorder")
        pScrRd=(const ae_int32x2* )scratch;
        pX=(const xtfloatx2*)y;
        pY=(      xtfloatx2*)y;
        aX=AE_LA64_PP(pX); aY=AE_ZALIGN64();
        for (k = 0; k < (M>>1); k++) 
        {
            xtbool2 s;
            ae_int32x2 n,n0,n1;
            xtfloatx2 s0,s1,s2;
            xtfloatx2 x0,x1,z,t;
            XT_LASX2IP(z,aX,pX);
            AE_L32X2_IP(n,pScrRd,1*sizeof(ae_int32x2));
            s=XT_OLT_SX2(z,XT_CONST_S(0));  /* extract right sign */
            z=XT_ABS_SX2(z);
            XT_LSX2IP(x0,castxcc(xtfloatx2,pScrRd),1*sizeof(ae_int32x2));
            /* simplified ldexpf */
            n0=AE_SRAI32(n,1);
            n1=AE_SUB32(n,n0);
            n1=AE_ADD32(n1,127);
            n0=AE_ADD32(n0,127);
            n1=AE_SLLI32(n1,23);
            n0=AE_SLLI32(n0,23);
            s0=XT_AE_MOVXTFLOATX2_FROMINT32X2(n0);
            s1=XT_AE_MOVXTFLOATX2_FROMINT32X2(n1);
            s2=XT_MUL_SX2(XT_MUL_SX2(z,s0),s1);
            /* approx y=1/(1+s2); */
            x1=XT_RECIP_SX2(XT_ADD_SX2(XT_CONST_S(1),x0));
            t=XT_MUL_SX2(x1,s2);
            XT_MOVT_SX2(x1,t,s);
            XT_SASX2IP(x1,aY,pY);
        }    
        AE_SA64POS_FP(aY,pY);
    }
}
#else
// code for scalar FPU
void xa_nnlib_vec_sigmoidf    (float32_t * y, const float32_t * x, int N)
{
    static const union ufloat32uint32 c[]={{0x3fb8aa3b},{0x32a57060}}; 
    static const union ufloat32uint32 p[]={{0x39222a75},{0x3aaf9334},{0x3c1d94fc},{0x3d63578b},{0x3e75fdf0},{0x3f317218},{0x3f800000}};
    const xtfloat * restrict pX=(const xtfloat *)x;
          xtfloat * restrict pY=(      xtfloat *)y;
    int n;
    for (n = 0; n < N; n++)
    {
        xtbool s;
        int32_t n,n0,n1;
        xtfloat x,s0,s1;
        xtfloat x0,y,z,d,t;
        XT_LSIP(x,pX,sizeof(float32_t));
        s=XT_OLT_S(x,0.f);
        x=XT_NEG_S(XT_ABS_S(x));
        XT_MOVT_S(x,-103.9721f,XT_OLT_S(x,-103.9721f));
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
        XT_MOVT_S(y,t,s);
        XT_SSIP(y,pY,sizeof(float32_t));
   }
}
#endif
