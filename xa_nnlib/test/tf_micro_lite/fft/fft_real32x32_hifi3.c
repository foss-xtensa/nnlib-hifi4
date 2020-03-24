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
/* Copyright (c) 2017 by Cadence Design Systems, Inc. ALL RIGHTS RESERVED.  */
/* These coded instructions, statements, and computer programs ("Cadence    */
/* Libraries") are the copyrighted works of Cadence Design Systems Inc.	    */
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
/*          Copyright (C) 2015-2017 IntegrIT, Limited.                      */
/*                      All Rights Reserved.                                */
/* ------------------------------------------------------------------------ */


#include "NatureDSP_Signal_fft.h"
#include "NatureDSP_Signal_vector.h"
#include "hifi_common.h"
#include "fft_twiddles32x32.h"


inline_ void _cmult32x32(ae_int32x2 *result, ae_int32x2 *x, ae_int32x2 *y)
{
#if (XCHAL_HAVE_HIFI3Z)
    ae_f32x2 z;

    z = AE_MULFCI32RAS(AE_MOVF32X2_FROMINT32X2(*x), AE_MOVF32X2_FROMINT32X2(*y));
    AE_MULFCR32RAS(z, AE_MOVF32X2_FROMINT32X2(*x), AE_MOVF32X2_FROMINT32X2(*y));
    *result = AE_MOVINT32X2_FROMF32X2(z);
#else

    ae_int32x2 b, c, d, ac, bd;
    c = AE_SEL32_HH(*y, *y);
    d = AE_SEL32_LL(*y, *y);
    b = AE_SEL32_LH(*x, *x);
    ac = AE_MULFP32X2RAS(*x, c);
    bd = AE_MULFP32X2RAS(b, d);
    ac = AE_SUBADD32S(ac, bd);
    *result = ac;
#endif
}


#define _CONJ32(_x) {_x = AE_SEL32_HL(_x, AE_NEG32S(_x) ); }

static int SpectrConv(complex_fract32 *y, int N, const complex_fract32 *twiddle_table, int twiddle_stride, int scalingOpt, int bexp)
{
    ae_int32x2  vA0, vA1, vB0, vB1, tw;
    ae_int32x2 * restrict p_x0, 
               * restrict p_x1, 
               * restrict ptw = (ae_int32x2*)(twiddle_table + twiddle_stride),
               * restrict p_y0, 
               * restrict p_y1;

    const int N4 = (N / 2 + 1) >> 1; /* Works for all even N */
/*----------------------------------------------------------------------------
Apply the in-place real-to-complex spectrum conversion

    MATLAB code:
    % x is input real sequence 1xN
    % y is output complex sequence 1xN
    y = fft(x(1:2:end) + 1j*x(2:2:end)); 
    N = length(x); 
    N4 = floor((N/2+1)/2); % N must be a multiple of 2
    twd = exp(-2*pi*1j*(0:N4-1)/N);
    a0 = y(1:N4);

    a1 = [y(1), y(N/2:-1:N/2-N4+2)];
    b0 = 1/2*(a0+conj(a1));
    b1 = 1/2*(a0-conj(a1))*-1j.*twd;
    a0 = b0+b1;
    a1 = b0-b1;
    if(mod(N,4))    
        y(1:N) = [a0, wrev(conj(a1(2:N4))), ...
        a1,    wrev(conj(a0(2:N4)))];    
    else
        y(1:N) = [a0,conj(y(N4+1)),wrev(conj(a1(2:N4))), ...
        a1,     y(N4+1) ,wrev(conj(a0(2:N4)))];
    end
*/
    int shift = (scalingOpt == 3) ?
        0 :
        (bexp < 1) ? 1 - bexp : 0;

    int  n;
    p_x0 = (ae_int32x2 *)(y);
    p_x1 = (ae_int32x2 *)(y + N / 2 - 1);
    p_y0 = (ae_int32x2 *)(y);
    p_y1 = (ae_int32x2 *)(y + N / 2);

    /*
    b0 = y[0];
    b0.s.re >>= shift;
    b0.s.im >>= shift;

    a0.s.re = L_add_ll(b0.s.re, b0.s.im);
    a0.s.im = 0;
    a1.s.re = L_sub_ll(b0.s.re, b0.s.im);
    a1.s.im = 0;

    a1 = conj_fr32c(a1); 
    y[0] = a0;
    y[N / 2] = a1;
    */
    AE_L32X2_IP(vB0, p_x0, 8);
    vB0 = AE_SRAA32(vB0, shift);
    vB1 = AE_SEL32_HH(vB0, vB0 );  
    vB0 = AE_SEL32_LL(vB0, AE_NEG32S(vB0));

    vB0 = AE_ADD32S(vB0, vB1); 

    vA0 = AE_SEL32_HH(vB0, AE_MOVI(0));
    vA1 = AE_SEL32_LL(vB0, AE_MOVI(0));

    AE_S32X2_IP(vA0, p_y0, sizeof(complex_fract32)); 
    AE_S32X2_XP(vA1, p_y1, -(int)sizeof(complex_fract32));

    /*15 cycles per pipeline stage in steady state with unroll=2*/
    __Pragma("loop_count min=3");
    for (n = 1; n < N4; n++)
    {
    /*  a0 = y[++k0];
        a1 = y[++k1];

        a0.s.re >>= shift;
        a0.s.im >>= shift;
        a1.s.re >>= shift;
        a1.s.im >>= shift;

        // b0 <- 1/2*(a0+conj(a1)); 
        b0.s.re = (fract32)(((int64_t)a0.s.re + a1.s.re) >> 1);
        b0.s.im = (fract32)(((int64_t)a0.s.im - a1.s.im) >> 1);
        // b1 <- 1/2*(a0-conj(a1))*-1j 
        b1.s.re = (fract32)(((int64_t)a0.s.im + a1.s.im) >> 1);
        b1.s.im = (fract32)(((int64_t)a1.s.re - a0.s.re) >> 1);

        b0.s.re = (fract32)vB0._[0];
        b0.s.im = (fract32)vB0._[1];
        b1.s.re = (fract32)vB1._[0];
        b1.s.im = (fract32)vB1._[1];

        // b1 <- b1*twd 
        b1 = mpy_CQ31CQ31(b1, twiddle_table[n * twiddle_stride]);        */

        AE_L32X2_IP(vB0, p_x0,  8);
        AE_L32X2_XP(vB1, p_x1, -8);
        AE_L32X2_XP(tw, ptw, twiddle_stride*sizeof(complex_fract32) ); 

        vB0 = AE_SRAA32(vB0, shift); 
        vB1 = AE_SRAA32(vB1, shift);

        // ADD/SUBB
        vA0 = AE_ADD32S(vB0, vB1);
        vA1 = AE_SUB32S(vB0, vB1);

        vB1 = AE_SEL32_LH(vA0, AE_NEG32S( vA1) );
        vB0 = AE_SEL32_HL(vA0, vA1);

        vB1 = AE_SRAI32(vB1, 1);
        vB0 = AE_SRAI32(vB0, 1);    


        _cmult32x32(&vB1, &vB1, &tw);
     /*  a0 <- b0+b1 
        a0.s.re = L_add_ll(b0.s.re, b1.s.re);
        a0.s.im = L_add_ll(b0.s.im, b1.s.im);
         a1 <- b0-b1 
        a1.s.re = L_sub_ll(b0.s.re, b1.s.re);
        a1.s.im = L_sub_ll(b0.s.im, b1.s.im);
        a1 = conj_fr32c(a1);
        y[k0] = a0;
        y[k1] = a1;  */

        vA0 = AE_ADD32S(vB0, vB1);
        vA1 = AE_SUB32S(vB0, vB1);
        _CONJ32(vA1);  // a1 = conj(a1)

        AE_S32X2_IP(vA0, p_y0,  sizeof(complex_fract32)); 
        AE_S32X2_XP(vA1, p_y1, -(int)sizeof(complex_fract32));
    }


    if (N & 3)
    {
        /* When N is not multiple of 4 */
        return shift; 
    }

    AE_L32X2_IP(vA0, p_x0, 8);
    vA0 = AE_SRAA32(vA0, shift);
    _CONJ32(vA0);  /* a0 = conj(a0) */
    AE_S32X2_IP(vA0, p_y0, sizeof(complex_fract32));

    return shift;
} /* SpectrConv */

/*-------------------------------------------------------------------------
  FFT on Real Data
  These functions make FFT on real data forming half of spectrum
      Scaling  : 
      +-------------------+----------------------------------------+
      |      Function     |           Scaling options              |
      +-------------------+----------------------------------------+
      |  fft_real16x16    |  2 - 16-bit dynamic scaling            | 
      |                   |  3 - fixed scaling before each stage   | 
      |  fft_real32x32    |  2 - 32-bit dynamic scaling            | 
      |                   |  3 - fixed scaling before each stage   | 
      |  fft_real32x16    |  3 - fixed scaling before each stage   | 
      |  fft_real24x24    |  0 - no scaling                        | 
      |                   |  1 - 24-bit scaling                    |
      |                   |  2 - 32-bit scaling on the first stage |
      |                   |  and 24-bit scaling later              |
      |                   |  3 - fixed scaling before each stage   |
      +-------------------+----------------------------------------+
  NOTES:
  1. Bit-reversal reordering is done here. 
  2. FFT runs in-place so INPUT DATA WILL APPEAR DAMAGED after the call.
  3. Real data FFT function calls fft_cplx() to apply complex FFT of size
     N/2 to input data and then transforms the resulting spectrum.
  4. 32x32 FFTs support mixed radix transforms 
  5. N - FFT size

  Precision:
  32x32  32-bit input/outputs, 32-bit twiddles
  24x24  24-bit input/outputs, 24-bit twiddles
  32x16  32-bit input/outputs, 16-bit twiddles
  16x16  16-bit input/outputs, 16-bit twiddles

  Input:
  x[N]          input signal
  scalingOpt    scaling option (see table above)
  Output:
  y[(N/2+1)*2]  output spectrum (positive side)

  Restrictions:
  x,y           should not overlap
  x,y           aligned on a 8-bytes boundary

-------------------------------------------------------------------------*/
int fft_real32x32(int32_t* y, int32_t* x, fft_handle_t h, int scalingOpt)
{
    int shift ; 
    int N, bexp; 
    fft_real32x32_descr_t *hr = (fft_real32x32_descr_t *)h; 
    fft_cplx32x32_descr_t *hc = (fft_cplx32x32_descr_t *)hr->cfft_hdl;
    N = 2 * hc->N; 

    NASSERT(scalingOpt==2 || scalingOpt==3); 
    NASSERT(x!=y); 
    NASSERT_ALIGN8(x); 
    NASSERT_ALIGN8(y);

    shift = fft_cplx32x32(y, x, hr->cfft_hdl, scalingOpt);

    if (scalingOpt == 2)
    {
        bexp = vec_bexp32(y, N);
    }
    else
    {
        bexp = 0;
    }

    shift += SpectrConv((complex_fract32*)y, N, (complex_fract32*)hr->twd, 1, scalingOpt, bexp);

    return shift;  
} /* fft_real32x32 */


