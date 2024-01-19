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
    tables for expf(x) approximation
*/
#ifndef EXPFTBL_H__
#define EXPFTBL_H__

#include "NatureDSP_types.h"
#include "xa_nn_common.h"

/* Renaming the symbols to avoid multiple definitions */
#define expftblf    xa_ndsp_expftblf

/*
   polynomial coefficients for 2^x in range 0...1

   derived by MATLAB code:
   order=6;
   x=(0:pow2(1,-16):1);
   y=2.^x;
   p=polyfit(x,y,6);
   p(order+1)=1;
   p(order)=p(order)-(sum(p)-2);
*/
externC const int32_t xa_nnlib_expftbl_Q30[8];
externC const union ufloat32uint32 xa_nnlib_expfminmax[2];  /* minimum and maximum arguments of expf() input */
externC const int32_t xa_nnlib_invln2_Q30; /* 1/ln(2), Q30 */
externC const union ufloat32uint32 expftblf[7];
externC const union ufloat32uint32 xa_nnlib_log2_e[2];

#endif /* EXPFTBL_H__ */
