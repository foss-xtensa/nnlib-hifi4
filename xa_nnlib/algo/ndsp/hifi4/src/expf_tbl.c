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
    tables for expf(x) approximation
*/
#include "NatureDSP_types.h"
#include "../include/expf_tbl.h"
#include "xa_nn_common.h"

/*
   polynomial coefficients for 2^x in range 0...1

   derived by MATLAB code:
   order=6;
   x=(0:pow2(1,-16):1);
   y=2.^x;
   p=polyfit(x,y,order);
   p(order+1)=1;
   p(order)=p(order)-(sum(p)-2);
*/
const int32_t ALIGN(8) xa_nnlib_expftbl_Q30[8]=
{    234841,
    1329551,
   10400465,
   59570027,
  257946177,
  744260763,
 1073741824,
 0 /* Padding to allow for vector loads */
};

#if 0
const union ufloat32uint32 xa_nnlib_expfminmax[2]=  /* minimum and maximum arguments of expf() input */
{
  {0xc2ce8ed0},  /*-1.0327893066e+002f */
  {0x42b17218}   /* 8.8722839355e+001f */
};
#endif

const int32_t xa_nnlib_invln2_Q30=1549082005L; /* 1/ln(2), Q30 */

const union ufloat32uint32 ALIGN(8) xa_nnlib_log2_e[2] =
{
  { 0x3fb8aa3b }, /* 1.4426950216      */
  { 0x32a57060 }  /* 1.9259629891e-008 */
};

#if 0
/*
order=6;
x=(0:pow2(1,-16):1);
y=2.^x;
p=polyfit(x,y,order);
p(order+1)=1;
p(order)=p(order)-(sum(p)-2);
num2hex(single(p));
*/
const union ufloat32uint32 expftblf[] =
{
  { 0x39655635 },
  { 0x3aa24c7a },
  { 0x3c1eb2d1 },
  { 0x3d633ddb },
  { 0x3e75ff24 },
  { 0x3f317212 },
  { 0x3f800000 }
};
#endif
