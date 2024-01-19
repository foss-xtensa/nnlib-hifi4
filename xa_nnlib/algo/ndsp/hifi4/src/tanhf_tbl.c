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
    tables for tanhf() approximation
*/

#include "NatureDSP_types.h"
#include "../include/tanhf_tbl.h"

/* polynomial approximation of tanh(x) in range [-log(3)/2...-log(3)/2]
    only odd coefficients are non zero
    s=pow2(2,-16);
    x=[s:s:log(3)/2+0.008]; x=[-x(end:-1:1) x];
    y=tanh(x); z=tanh(x)./x;
    p=polyfit(x,z,8);
    p=p(1:2:end); p(end)=[];
*/
const union ufloat32uint32 ALIGN(8) xa_nnlib_polytanhf_tbl[]=
{
    {0x3c86a7d1UL},/* 1.6437442973e-002*/
    {0xbd57b3abUL},/*-5.2661579102e-002*/
    {0x3e086615UL},/* 1.3320191205e-001*/
    {0xbeaaaa0fUL} /*-3.3332869411e-001*/
};

const union ufloat32uint32 xa_nnlib_halfln3={0x3f0c9f54UL} ; /* log(3)/2 - tanh(log(3)/2)==0.5 */
