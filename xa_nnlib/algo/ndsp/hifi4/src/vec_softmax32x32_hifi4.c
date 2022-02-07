/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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
    Optimized code for HiFi4
  IntegrIT, 2006-2018
*/

#include "NatureDSP_types.h"
#include "NatureDSP_Signal_math.h"
#include "common.h"

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
#if XCHAL_HAVE_HIFI1
void vec_softmax32x32(int32_t * y, const int32_t * x, int N)
{
    /*
        Reference Matlab code:

        function y=softmax_32x32(x)
        % convert Q25 -> Q23 and scale by ln(2)
        b= double(max(x));
        x=x-b;
        x=round(pow2(double(x)*774541002,-29));

        % compute 2^x
        p=[14685058 114217091 514075394 1488269031 2147475316];
        %x=(-1.:pow2(1,-15):0);
        %y=2.^x;
        %p=polyfit(x,y,4);
        %round(pow2(p,23))
        p23=[57364      446161     2008107     5813551     8388575];

        x=floor(pow2(double(x),-2));
        x=pow2(double(x),-23);
        n=ceil(x);
        x=x-n;

        x=pow2(x,23);
        y=p23(1);
        y=sat32(round(pow2(y.*x,-23))+p23(2));
        y=sat32(round(pow2(y.*x,-23))+p23(3));
        y=sat32(round(pow2(y.*x,-23))+p23(4));
        y=sat32(round(pow2(y.*x,-23))+p23(5));
        y=round(pow2(y,n));
        X=sat32(y);
        Y = sum(X);%Q23
        n=log2(pow2(Y,-23));
        n=ceil(n);
        mantQ23=floor(pow2(Y,-n));
        x=double(mantQ23);
        y=(1.5*pow2(1,23)-abs(x)-1200*256).*sign(x);
        for k=1:2
            e=pow2(1,22)-round(pow2(x.*y,-23));
            e=e+e;
            y=y+round(pow2(y.*e,-23));
        end
        y=int32(round(pow2(y,-n+1))); %Q23
        y=int32(round(pow2(double(y).*X,-31)));
    */
  int n;
  static const int32_t polypow2_q23[] = { 57364, 446161, 2008107, 5813551, 8388575 };
        ae_int32x2 * restrict pYw = (      ae_int32x2 *)y;
  const ae_int32x2 * restrict pX  = (const ae_int32x2 *)x;
  const ae_int32x2 * restrict pYr = (const ae_int32x2 *)y;
  ae_valign x_align, yw_align, yr_align;
  ae_int32x2 X, Y, E, B, E_SUM;
  x_align = AE_LA64_PP(pX); 
  if (N <= 0) return;
  B = AE_MOVDA32(MIN_INT32);
  for (n = 0; n<(N>>1); n ++)
  {
    AE_LA32X2_IP(X, x_align, pX);
    B = AE_MAX32(X, B);
  }
  if (N&1)
  {
    X = AE_L32_I((const ae_int32*)pX, 0);
    B = AE_MAX32(X, B);
  }
  X = AE_SEL32_LH(B,B);
  B = AE_MAX32(X, B);
  pX = (const ae_int32x2 *)x;
  x_align = AE_LA64_PP(pX);
  yw_align = AE_ZALIGN64();

  E_SUM = AE_ZERO32();
  for (n = 0; n<(N>>1); n++)
  {
    ae_f32x2 t;
    AE_LA32X2_IP(X, x_align, pX);
    X = AE_SUB32S(X, B);
    X = AE_MULFP32X2RAS(X, AE_MOVDA32X2(774541002, 774541002));
    E = AE_SRAI32(X, 23);
    E = AE_ADD32(E, 1);
    X = AE_AND32(X, AE_MOVDA32(0x7fffff));
    X = AE_SUB32(X, AE_MOVDA32(0x800000));

    X = AE_SLAI32(X, 8);
    Y = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 0);
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 1); AE_MULAFP32X2RAS(t, X, Y); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 2); AE_MULAFP32X2RAS(t, X, Y); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 3); AE_MULAFP32X2RAS(t, X, Y); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 4); AE_MULAFP32X2RAS(t, X, Y); Y = t;

    X = AE_SLAA32S(Y, AE_MOVAD32_H(E));
    Y = AE_SLAA32S(Y, AE_MOVAD32_L(E));
    Y = AE_SEL32_HL(X, Y);
    E_SUM = AE_ADD32S(E_SUM, Y);
    AE_SA32X2_IP(Y, yw_align, pYw);
  }
  AE_SA64POS_FP(yw_align, pYw);
  if (N & 1)
  {
    ae_f32x2 t;
    X = AE_L32_I((const ae_int32*)pX, 0);
    X = AE_SUB32S(X, B);
    X = AE_MULFP32X2RAS(X, AE_MOVDA32X2(774541002, 774541002));
    E = AE_SRAI32(X, 23);
    E = AE_ADD32(E, 1);
    X = AE_AND32(X, AE_MOVDA32(0x7fffff));
    X = AE_SUB32(X, AE_MOVDA32(0x800000));

    X = AE_SLAI32(X, 8);
    Y = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 0);
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 1); AE_MULAFP32X2RAS(t, X, Y); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 2); AE_MULAFP32X2RAS(t, X, Y); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 3); AE_MULAFP32X2RAS(t, X, Y); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 4); AE_MULAFP32X2RAS(t, X, Y); Y = t;

    Y = AE_SLAA32S(Y, E);
    Y = AE_SEL32_HL(AE_ZERO32(), Y);
    E_SUM = AE_ADD32S(E_SUM, Y);
    AE_S32_L_I(Y, (ae_int32*)pYw, 0);
  }
  X = AE_SEL32_LH(E_SUM, E_SUM);
  E_SUM = AE_ADD32S(E_SUM, X);

  {
    unsigned nsa;
    xtbool2 isZero;
    ae_f32x2 t;
    X = E_SUM;
    
    nsa = AE_NSAZ32_L(X) - 8;
    X = AE_SLAA32S(X, nsa);
    nsa+=1;
    isZero = AE_EQ32(X, AE_ZERO32());
    /* first approximation */
    Y = AE_SUB32(AE_MOVDA32((int32_t)0xBB5000), X);
 
    t = AE_MOVF32X2_FROMINT32X2(AE_MOVDA32(0x400000)); AE_MULSFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); E = t;
    E = AE_ADD32(E, E);
    t = Y; AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(Y), AE_MOVF24X2_FROMINT32X2(E)); Y = t;
 
    t = AE_MOVF32X2_FROMINT32X2(AE_MOVDA32(0x400000)); AE_MULSFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); E = t;
    E = AE_ADD32(E, E);
    t = Y; AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(Y), AE_MOVF24X2_FROMINT32X2(E)); Y = t;
 
    Y = AE_SLAA32S(Y, nsa); /* Q23 */
  }
  pYr = (const ae_int32x2 *)y;
  pYw = (ae_int32x2 *)y;
  yr_align = AE_LA64_PP(pYr);
  yw_align = AE_ZALIGN64();

  for (n = 0; n<(N>>1); n++)
  {
    AE_LA32X2_IP(X, yr_align, pYr);
    X = AE_MULFP32X2RAS(X, Y);
    AE_SA32X2_IP(X, yw_align, pYw);
  }
  AE_SA64POS_FP(yw_align, pYw);
  if (N&1)
  {
    X = AE_L32_I((const ae_int32*)pYr, 0);
    X = AE_MULFP32X2RAS(X, Y);
    AE_S32_L_I(X, (ae_int32*)pYw, 0);
  }
} /* vec_softmax32x32() */

#else
void vec_softmax32x32(int32_t * y, const int32_t * x, int N)
{
    /*
        Reference Matlab code:

        function y=softmax_32x32(x)
        % convert Q25 -> Q23 and scale by ln(2)
        b= double(max(x));
        x=x-b;
        x=round(pow2(double(x)*774541002,-29));

        % compute 2^x
        p=[14685058 114217091 514075394 1488269031 2147475316];
        %x=(-1.:pow2(1,-15):0);
        %y=2.^x;
        %p=polyfit(x,y,4);
        %round(pow2(p,23))
        p23=[57364      446161     2008107     5813551     8388575];

        x=floor(pow2(double(x),-2));
        x=pow2(double(x),-23);
        n=ceil(x);
        x=x-n;

        x=pow2(x,23);
        y=p23(1);
        y=sat32(round(pow2(y.*x,-23))+p23(2));
        y=sat32(round(pow2(y.*x,-23))+p23(3));
        y=sat32(round(pow2(y.*x,-23))+p23(4));
        y=sat32(round(pow2(y.*x,-23))+p23(5));
        y=round(pow2(y,n));
        X=sat32(y);
        Y = sum(X);%Q23
        n=log2(pow2(Y,-23));
        n=ceil(n);
        mantQ23=floor(pow2(Y,-n));
        x=double(mantQ23);
        y=(1.5*pow2(1,23)-abs(x)-1200*256).*sign(x);
        for k=1:2
            e=pow2(1,22)-round(pow2(x.*y,-23));
            e=e+e;
            y=y+round(pow2(y.*e,-23));
        end
        y=int32(round(pow2(y,-n+1))); %Q23
        y=int32(round(pow2(double(y).*X,-31)));
    */
  int n;
  static const int32_t polypow2_q23[] = { 57364, 446161, 2008107, 5813551, 8388575 };
        ae_int32x2 * restrict pYw = (      ae_int32x2 *)y;
  const ae_int32x2 * restrict pX  = (const ae_int32x2 *)x;
  const ae_int32x2 * restrict pYr = (const ae_int32x2 *)y;
  ae_valign x_align, yw_align, yr_align;
  ae_int32x2 X, Y, E, B, E_SUM;
  x_align = AE_LA64_PP(pX); 
  if (N <= 0) return;
  B = AE_MOVDA32(MIN_INT32);
  for (n = 0; n<(N>>1); n ++)
  {
    AE_LA32X2_IP(X, x_align, pX);
    B = AE_MAX32(X, B);
  }
  if (N&1)
  {
    X = AE_L32_I((const ae_int32*)pX, 0);
    B = AE_MAX32(X, B);
  }
  X = AE_SEL32_LH(B,B);
  B = AE_MAX32(X, B);
  __Pragma("no_reorder");
  pX = (const ae_int32x2 *)x;
  x_align = AE_LA64_PP(pX);
  yw_align = AE_ZALIGN64();

  E_SUM = AE_ZERO32();
  for (n = 0; n<(N>>1); n++)
  {
    ae_f32x2 t;
    AE_LA32X2_IP(X, x_align, pX);
    X = AE_SUB32S(X, B);
    X = AE_MULFP32X2RAS(X, AE_MOVDA32X2(774541002, 774541002));
    E = AE_SRAI32(X, 23);
    E = AE_ADD32(E, 1);
    X = AE_AND32(X, AE_MOVDA32(0x7fffff));
    X = AE_SUB32(X, AE_MOVDA32(0x800000));

    Y = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 0);
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 1); AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 2); AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 3); AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 4); AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); Y = t;

    X = AE_SLAA32S(Y, AE_MOVAD32_H(E));
    Y = AE_SLAA32S(Y, AE_MOVAD32_L(E));
    Y = AE_SEL32_HL(X, Y);
    E_SUM = AE_ADD32S(E_SUM, Y);
    AE_SA32X2_IP(Y, yw_align, pYw);
  }
  AE_SA64POS_FP(yw_align, pYw);
  if (N & 1)
  {
    ae_f32x2 t;
    X = AE_L32_I((const ae_int32*)pX, 0);
    X = AE_SUB32S(X, B);
    X = AE_MULFP32X2RAS(X, AE_MOVDA32X2(774541002, 774541002));
    E = AE_SRAI32(X, 23);
    E = AE_ADD32(E, 1);
    X = AE_AND32(X, AE_MOVDA32(0x7fffff));
    X = AE_SUB32(X, AE_MOVDA32(0x800000));

    Y = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 0);
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 1); AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 2); AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 3); AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); Y = t;
    t = AE_L32_I((const ae_int32 *)polypow2_q23, 4 * 4); AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); Y = t;


    Y = AE_SLAA32S(Y, E);
    Y = AE_SEL32_HL(AE_ZERO32(), Y);
    E_SUM = AE_ADD32S(E_SUM, Y);
    AE_S32_L_I(Y, (ae_int32*)pYw, 0);
  }
  X = AE_SEL32_LH(E_SUM, E_SUM);
  E_SUM = AE_ADD32S(E_SUM, X);

  __Pragma("no_reorder");

  {
    unsigned nsa;
    xtbool2 isZero;
    ae_f32x2 t;
    X = E_SUM;
    
    nsa = AE_NSAZ32_L(X) - 8;
    X = AE_SLAA32S(X, nsa);
    nsa+=1;
    isZero = AE_EQ32(X, AE_ZERO32());
    /* first approximation */
    Y = AE_SUB32(AE_MOVDA32((int32_t)0xBB5000), X);
 
    t = AE_MOVF32X2_FROMINT32X2(AE_MOVDA32(0x400000)); AE_MULSFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); E = t;
    E = AE_ADD32(E, E);
    t = Y; AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(Y), AE_MOVF24X2_FROMINT32X2(E)); Y = t;
 
    t = AE_MOVF32X2_FROMINT32X2(AE_MOVDA32(0x400000)); AE_MULSFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); E = t;
    E = AE_ADD32(E, E);
    t = Y; AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(Y), AE_MOVF24X2_FROMINT32X2(E)); Y = t;
 
    Y = AE_SLAA32S(Y, nsa); /* Q23 */
  }
__Pragma("no_reorder");
  pYr = (const ae_int32x2 *)y;
  pYw = (ae_int32x2 *)y;
  yr_align = AE_LA64_PP(pYr);
  yw_align = AE_ZALIGN64();
  __Pragma("ymemory(pYr)");
  __Pragma("ymemory(pYw)");
  for (n = 0; n<(N>>1); n++)
  {
    AE_LA32X2_IP(X, yr_align, pYr);
    X = AE_MULFP32X2RAS(X, Y);
    AE_SA32X2_IP(X, yw_align, pYw);
  }
  AE_SA64POS_FP(yw_align, pYw);
  if (N&1)
  {
    X = AE_L32_I((const ae_int32*)pYr, 0);
    X = AE_MULFP32X2RAS(X, Y);
    AE_S32_L_I(X, (ae_int32*)pYw, 0);
  }
} /* vec_softmax32x32() */
#endif
