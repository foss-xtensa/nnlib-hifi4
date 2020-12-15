/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
	NaN values for single precision routines
*/

#ifndef __NANF_TBL_H
#define __NANF_TBL_H

/* Portable data types. */
#include "NatureDSP_types.h"
/* Common utility macros. */
#include "common.h"

/* Renaming the symbols to avoid multiple definitions */
#define minus_qNaNf xa_ndsp_minus_qNaNf
#define minus_sNaNf xa_ndsp_minus_sNaNf
#define qNaNf       xa_ndsp_qNaNf
#define sNaNf       xa_ndsp_sNaNf

externC const union ufloat32uint32 sNaNf;       /* Signalling NaN          */
externC const union ufloat32uint32 qNaNf;       /* Quiet NaN               */
externC const union ufloat32uint32 minus_sNaNf; /* Negative Signalling NaN */
externC const union ufloat32uint32 minus_qNaNf; /* Negative Quiet NaN      */

#endif /* __NANF_TBL_H */
