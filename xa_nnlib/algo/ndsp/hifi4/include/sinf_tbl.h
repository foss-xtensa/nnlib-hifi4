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
/*          Copyright (C) 2015-2017 IntegrIT, Limited.                      */
/*                      All Rights Reserved.                                */
/* ------------------------------------------------------------------------ */

/*
    tables for sin(pi/2*x) approximation
*/
#ifndef SINFTBL_H__
#define SINFTBL_H__

#include "NatureDSP_types.h"
#include "xa_nn_common.h"

#define SINNCOSF_ALG 0 /* 0 - 2 ULP, 1 - 1 ULP */

externC const union ufloat32uint32 xa_nnlib_sinf_maxval; /* domain ranges - zero outside */

/* pi/4 represented as a sum of exacly represented numbers.
    derived from hex value of pi: 3.243F6A8885A308D313198A2E037073
*/
externC const union ufloat64uint64 ALIGN(8) xa_nnlib_pi4fc[];

/* 
   polynomial coefficients for sin(x)/x, [-pi/4...pi/4]
   derived by MATLAB code:
   s=pow2(1,-16); x=(s:s:pi/4); x=[-x(end:-1:1) x];
   y=sin(x)./x;
   p=polyfit(x,y,6); p=p(1:2:end); p(end)=[];
*/
externC const union ufloat32uint32 ALIGN(8) xa_nnlib_polysinf_tbl[];

/* 
   polynomial coefficients for cos(x), [-pi/4...pi/4]
   derived by MATLAB code:
   s=pow2(1,-16); x=(s:s:pi/4); x=[-x(end:-1:1) x];
   y=cos(x);
   p=polyfit(x,y,6); p=p(1:2:end); p(end)=[];
*/
externC const union ufloat32uint32 ALIGN(8) xa_nnlib_polycosf_tbl[];

#endif /* SINFTBL_H__ */
