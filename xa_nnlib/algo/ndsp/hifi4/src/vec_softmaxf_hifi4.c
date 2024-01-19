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
    Softmax
    Code optimized for HiFi4 core
  IntegrIT, 2006-2018
*/
#include "../include/NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "xa_nn_common.h"
#include "xa_nnlib_common_fpu.h"
#include "../include/inff_tbl.h"
#include "../include/nanf_tbl.h"
/*-------------------------------------------------------------------------
  Softmax
  The function computes the softmax (normalized exponential function) of 
  input data. 32-bit fixed-point functions accept inputs in Q6.25 and form 
  outputs in Q16.15 format. 

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB (see Note below)
  f      floating point input, floating point output

  Note: Accuracy of function may depend on amount of data and their 
  distribution. Given accuracy is achieved for N=2 for any pair of data 
  from input domain.

  Input:
  x[N]   input data, Q6.25 or floating point
  N      length of vectors
  Output:
  y[N]   result, Q16.15 or floating point

  Restriction:
  x,y should not overlap

-------------------------------------------------------------------------*/
#if !HAVE_VFPU && !HAVE_FPU
DISCARD_FUN(void,xa_nnlib_vec_softmaxf,(float32_t * y, const float32_t * x,int N))
#elif HAVE_VFPU
void xa_nnlib_vec_softmaxf    (float32_t * y, const float32_t * x,int N)
{
    ae_valign aX,aY;
    const xtfloatx2* restrict pX=(const xtfloatx2*)x;
          xtfloatx2* restrict pY=(      xtfloatx2*)y;
    int n;
    xtfloatx2 xmax,ysum,t;
    if (N<=0) return;
    /* compute maximum of x */
    xmax=xa_nnlib_minusInff.f;
    aX=AE_LA64_PP(pX);
    for (n=0; n<(N&~1); n+=2)
    {
        XT_LASX2IP(t,aX,pX);
        xmax=XT_MAX_SX2(xmax,t);
    }
    if (N&1)
    {
        t=XT_LSI((const xtfloat*)pX,0);
        xmax=XT_MAX_SX2(xmax,t);
    }
    t=XT_SEL32_LH_SX2(xmax,xmax);
    xmax=XT_MAX_SX2(xmax,t);

    /* subtract maximum of x from input data */
    {
        int M=N;
        pX=(const xtfloatx2*)x;
        if (((uintptr_t)pX)&7)
        {
            ae_int32x2 tmp;
            AE_L32_IP(tmp,castxcc(ae_int32,pX),sizeof(xtfloat));
            t=XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            t=XT_SUB_SX2(t,xmax);
            tmp=XT_AE_MOVINT32X2_FROMXTFLOATX2(t);
            AE_S32_L_IP(tmp,castxcc(ae_int32,pY),sizeof(xtfloat));
            M--;
        }
        aY=AE_ZALIGN64();
        for (n=0; n<(M&~1); n+=2)
        {
            XT_LSX2IP(t,pX,sizeof(xtfloatx2));
            t=XT_SUB_SX2(t,xmax);
            XT_SASX2IP(t,aY,pY);
        }
        AE_SA64POS_FP(aY,pY);
        if(M&1)
        {
            t=XT_LSI((const xtfloat*)pX,0);
            t=XT_SUB_SX2(t,xmax);
            XT_SSI(t,(xtfloat*)pY,0);
        }
    }
    /* compute exp() */
    xa_nnlib_vec_antilognf(y,y,N);
    /* sum results */
    pY=(xtfloatx2*)y;
    aY=AE_LA64_PP(pY);
    ysum=XT_CONST_S(0);
    {
        xtfloatx2 ysum1=XT_CONST_S(0);
        xtfloatx2 ysum2=XT_CONST_S(0);
        xtfloatx2 ysum3=XT_CONST_S(0);
        for (n=0; n<(N>>3); n++)
        {
            XT_LASX2IP(t,aY,pY); ysum =XT_ADD_SX2(ysum ,t);
            XT_LASX2IP(t,aY,pY); ysum1=XT_ADD_SX2(ysum1,t);
            XT_LASX2IP(t,aY,pY); ysum2=XT_ADD_SX2(ysum2,t);
            XT_LASX2IP(t,aY,pY); ysum3=XT_ADD_SX2(ysum3,t);
        }
        if (N&4)
        {
            XT_LASX2IP(t,aY,pY); ysum2=XT_ADD_SX2(ysum2,t);
            XT_LASX2IP(t,aY,pY); ysum3=XT_ADD_SX2(ysum3,t);
        }
        if (N&2)
        {
            XT_LASX2IP(t,aY,pY); ysum1=XT_ADD_SX2(ysum1,t);
        }
        ysum2=XT_ADD_SX2(ysum2,ysum3);
        ysum=XT_ADD_SX2(ysum,ysum1);
        ysum=XT_ADD_SX2(ysum,ysum2);
    }
    t=XT_SEL32_LH_SX2(ysum,ysum);
    ysum=XT_ADD_SX2(ysum,t);
    if(N&1)
    {
        t=XT_LSI((const xtfloat*)pY,0);
        ysum=XT_ADD_SX2(ysum,t);
    }
    ysum=XT_SEL32_HH_SX2(ysum,ysum);
    /* normalize output */
    ysum=XT_RECIP_S(ysum);
    __Pragma("no_reorder")
    {
        int M=N;
        pX=(xtfloatx2*)y;
        pY=(xtfloatx2*)y;
        aX=AE_LA64_PP(pX);
        if (((uintptr_t)pX)&7)
        {
            ae_int32x2 tmp;
            AE_L32_IP(tmp,castxcc(ae_int32,pX),sizeof(xtfloat));
            t=XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            t=XT_MUL_SX2(t,ysum);
            tmp=XT_AE_MOVINT32X2_FROMXTFLOATX2(t);
            AE_S32_L_IP(tmp,castxcc(ae_int32,pY),sizeof(xtfloat));
            M--;
        }

        aY=AE_ZALIGN64();
        for (n=0; n<(M&~1); n+=2)
        {
            XT_LSX2IP(t,pX,sizeof(xtfloatx2));
            t=XT_MUL_SX2(t,ysum);
            XT_SASX2IP(t,aY,pY);
        }
        AE_SA64POS_FP(aY,pY);
        if(M&1)
        {
            t=XT_LSI((const xtfloat*)pX,0);
            t=XT_MUL_SX2(t,ysum);
            XT_SSI(t,(xtfloat*)pY,0);
        }
    }
}
#else
// code for scalar FPU
void xa_nnlib_vec_softmaxf    (float32_t * y, const float32_t * x,int N)
{
    const xtfloat* restrict pX=(const xtfloat*)x;
          xtfloat* restrict pY=(      xtfloat*)y;
    int n;
    xtfloat xmax,ysum;
    if (N<0) return;
    /* compute maximum of x */
    xmax=xa_nnlib_minusInff.f;
    for (n=0; n<N; n++)
    {
        xtfloat t;
        XT_LSIP(t,pX,sizeof(xtfloat));
        XT_MOVT_S(xmax,t,XT_OLT_S(xmax,t));
    }
    /* subtract maximum of x from input data */
    pX=(const xtfloat*)x;
    for (n=0; n<N; n++)
    {
        xtfloat t;
        XT_LSIP(t,pX,sizeof(xtfloat));
        t=XT_SUB_S(t,xmax);
        XT_SSIP(t,pY,sizeof(xtfloat));
    }
    /* compute exp() */
    xa_nnlib_vec_antilognf(y,y,N);
    /* sum results */
    pY=(xtfloat*)y;
    ysum=XT_CONST_S(0);
    for (n=0; n<N; n++)
    {
        xtfloat t;
        XT_LSIP(t,pY,sizeof(xtfloat));
        ysum=XT_ADD_S(ysum,t);
    }
    /* normalize output */
    ysum=XT_RECIP_S(ysum);
    __Pragma("no_reorder")
    pX=(xtfloat*)y;
    pY=(xtfloat*)y;
    for (n=0; n<N; n++) 
    {
        xtfloat t;
        XT_LSIP(t,pX,sizeof(xtfloat));
        t=XT_MUL_S(t,ysum);
        XT_SSIP(t,pY,sizeof(xtfloat));
    }
}
#endif
