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
/* Copyright (c) 2016 by Cadence Design Systems, Inc. ALL RIGHTS RESERVED.  */
/* These coded instructions, statements, and computer programs (“Cadence    */
/* Libraries”) are the copyrighted works of Cadence Design Systems Inc.	    */
/* Cadence IP is licensed for use with Cadence processor cores only and     */
/* must not be used for any other processors and platforms. Your use of the */
/* Cadence Libraries is subject to the terms of the license agreement you   */
/* have entered into with Cadence Design Systems, or a sublicense granted   */
/* to you by a direct Cadence licensee.                                     */
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
/*          Copyright (C) 2015-2016 IntegrIT, Limited.                      */
/*                      All Rights Reserved.                                */
/* ------------------------------------------------------------------------ */
/*
	NatureDSP_Signal library. FFT part
    Twiddle factor tables and descriptor structures for 32x32 FFTs
	IntegrIT, 2006-2017
*/

#ifndef __FFT_TWDIDDLES32X32_H__
#define __FFT_TWDIDDLES32X32_H__
#include "NatureDSP_types.h"
#include "NatureDSP_Signal_fft.h"



/*
    Pointer to functions which implement  complex 32x32 FFTs stage 
*/
typedef int (*fft_cplx32x32_stage_t)(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp); 

typedef const int32_t* cint32ptr_t;

/*
*   32x32-bit complex-valued FFT descriptor structure.
*/
typedef struct
{
    const int        N;
    /* Table of the stages for scaleOption = 2 */
    const fft_cplx32x32_stage_t *stages_s2; 
    /* Table of the stages for scaleOption = 3 */
    const fft_cplx32x32_stage_t *stages_s3;
    /* Tables params of the stages */
    const int *tw_step; 
    /* Twiddles */
    const cint32ptr_t *twd;

} fft_cplx32x32_descr_t;

/*
*   32x32-bit real-valued FFT descriptor structure.
*/
typedef struct
{
    // Handle of half-size complex FFT
    const fft_handle_t   cfft_hdl; 
    const int32_t  *twd;
} fft_real32x32_descr_t;

/*
*  32x32 FFT stages, scalingOption=2    
*/
extern int fft_stageS2_DFT2_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS2_DFT4_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS2_DFT3_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS2_DFT5_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS2_DFT2_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS2_DFT4_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS2_DFT3_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS2_DFT5_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);

extern int fft_stageS2_DFT2_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS2_DFT4_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS2_DFT3_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS2_DFT5_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS2_DFT8_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);

extern int fft_stageS2_DFT2_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS2_DFT3_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS2_DFT4_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS2_DFT5_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS2_DFT2_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS2_DFT3_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS2_DFT4_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS2_DFT5_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);

/*
*  32x32 FFT stages, scalingOption=3
*/
extern int fft_stageS3_DFT8_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp); 

extern int fft_stageS3_DFT2_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS3_DFT4_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS3_DFT3_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS3_DFT5_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS3_DFT2_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS3_DFT4_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS3_DFT3_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS3_DFT5_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);

extern int fft_stageS3_DFT2_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS3_DFT4_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS3_DFT3_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS3_DFT5_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);

extern int fft_stageS3_DFT2_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS3_DFT3_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS3_DFT4_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int fft_stageS3_DFT5_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS3_DFT2_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS3_DFT3_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS3_DFT4_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);
extern int ifft_stageS3_DFT5_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp);

extern const fft_cplx32x32_descr_t __cfft_descr16_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr32_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr64_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr128_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr256_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr512_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr1024_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr2048_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr4096_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr12_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr24_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr36_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr48_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr60_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr72_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr96_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr108_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr120_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr144_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr180_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr192_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr216_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr240_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr288_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr300_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr324_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr360_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr432_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr480_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr540_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr576_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr768_32x32;
extern const fft_cplx32x32_descr_t __cfft_descr960_32x32;

extern const fft_cplx32x32_descr_t __cifft_descr16_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr32_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr64_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr128_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr256_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr512_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr1024_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr2048_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr4096_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr12_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr24_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr36_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr48_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr60_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr72_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr96_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr108_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr120_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr144_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr180_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr192_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr216_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr240_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr288_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr300_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr324_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr360_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr432_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr480_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr540_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr576_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr768_32x32;
extern const fft_cplx32x32_descr_t __cifft_descr960_32x32;

#endif // __FFT_CPLX_TWD_H
