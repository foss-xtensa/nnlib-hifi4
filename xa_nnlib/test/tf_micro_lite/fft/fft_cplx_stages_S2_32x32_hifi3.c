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
#include "fft_twiddles32x32.h"
#include "hifi_common.h"

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
#define DFT4X1(x0, x1, x2, x3)\
{   \
\
    ae_int32x2 s0, s1, d0, d1;                                 \
    s0 = AE_ADD32S(x0, x2);                         \
    d0 = AE_SUB32S(x0, x2);                         \
    s1 = AE_ADD32S(x1, x3);                         \
    d1 = AE_SUB32S(x1, x3);                         \
    d1 = AE_SEL32_LH(d1, d1);                       \
    x0 = AE_ADD32S(s0, s1);                         \
    x2 = AE_SUB32S(s0, s1);                         \
    x1 = AE_ADDSUB32S(d0, d1);                      \
    x3 = AE_SUBADD32S(d0, d1);                      \
}
/*
DFT3 algorithm:
x - input complex vector
y - output complex vector
y = fft(x)
y = [ x(1) + x(2)  + x(3);
x(1) + (x(2) + x(3))*cos(2*pi/3) - 1j*(x(2) - x(3))*sin(2*pi/3);
x(1) + (x(2) + x(3))*cos(2*pi/3) + 1j*(x(2) - x(3))*sin(2*pi/3) ]

*/
#define DFT3X1(x0, x1, x2)\
{   \
\
    ae_int32x2 s0, s1, d0;                                 \
    ae_int32x2 c;                                              \
    c = AE_MOVDA32X2(0x0,0x6ED9EBA1);                          \
    s0 = AE_ADD32S(x1, x2);                                    \
    s1 = AE_ADD32S(x0, s0);                                    \
    s0 = AE_SRAI32(s0, 1);                                     \
    d0 = AE_SUB32S(x1, x2);                                    \
    _cmult32x32(&d0, &d0, &c);                                 \
    s0 = AE_SUB32S(x0, s0);                                    \
    x0 = s1;                                                   \
    x1 = AE_SUB32S(s0, d0);                                    \
    x2 = AE_ADD32S(s0, d0);                                    \
}       
/*
DFT5 algorithm:
x - input complex vector
y - output complex vector
y = fft(x)
w1 =  exp(-1j*2*pi/5);
w2 =  exp(-1j*2*pi*2/5);

y = zeros(5,1);
s1 = (x1+x4);
s2 = (x2 + x3);
d1 = (x1-x4);
d2 = (x2-x3);

y(1) = x0 + s1 + s2;
y(2) = x0 + (s1*real(w1) + s2*real(w2)) + 1j*(d1*imag(w1) + d2*imag(w2));
y(5) = x0 + (s1*real(w1) + s2*real(w2)) - 1j*(d1*imag(w1) + d2*imag(w2));
y(3) = x0 + (s1*real(w2) + s2*real(w1)) + 1j*(d1*imag(w2)  - d2*imag(w1));
y(4) = x0 + (s1*real(w2) + s2*real(w1)) - 1j*(d1*imag(w2)  - d2*imag(w1));

*/
#define DFT5X1(x0, x1, x2, x3, x4)\
{   \
\
  ae_int32x2 s1, s2, d1, d2;                                 \
  ae_int32x2 y0, y1, y2, y3, y4;                             \
  ae_int32x2 t0, t1, t2, t3;                             \
  ae_int32x2 real_w1, jimag_w1, real_w2, jimag_w2;           \
  real_w1 = AE_MOVDA32X2(0x278DDE6E, 0x0);                    \
  jimag_w1 = AE_MOVDA32X2(0x0,0x8643C7B3);                   \
  real_w2 = AE_MOVDA32X2(0x98722192,0x0);                    \
  jimag_w2 = AE_MOVDA32X2(0x0,0xB4C373EE);                   \
  s1 = AE_ADD32S(x1, x4);                                    \
  s2 = AE_ADD32S(x2, x3);                                    \
  d1 = AE_SUB32S(x1, x4);                                    \
  d2 = AE_SUB32S(x2, x3);                                    \
  y0 = AE_ADD32S(x0, AE_ADD32S(s1, s2));                     \
  _cmult32x32(&t0, &s1, &real_w1);                           \
  _cmult32x32(&t1, &s2, &real_w2);                           \
  y1 =AE_ADD32S(x0, AE_ADD32S(t0, t1));                      \
  y4 = y1;                                                   \
  _cmult32x32(&t2, &s1, &real_w2);                           \
  _cmult32x32(&t3, &s2, &real_w1);                           \
  y2 = AE_ADD32S(x0, AE_ADD32S(t2, t3));                     \
  y3 = y2;                                                   \
  _cmult32x32(&t0, &d1, &jimag_w1);                          \
  _cmult32x32(&t1, &d2, &jimag_w2);                          \
  _cmult32x32(&t2, &d2, &jimag_w1);                          \
  _cmult32x32(&t3, &d1, &jimag_w2);                          \
  t0 = AE_ADD32S(t0, t1);                                    \
  t1 = AE_SUB32S(t3, t2);                                    \
  x1 = AE_ADD32S(y1, t0);                                    \
  x4 = AE_SUB32S(y4, t0);                                    \
  x2 = AE_ADD32S(y2, t1);                                    \
  x3 = AE_SUB32S(y3, t1);                                    \
  x0= y0;                                                    \
}                                                       
/*
*  32x32 FFT stages, Radix 2, scalingOption=2
*/
int fft_stageS2_DFT2_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{
  const int stride = (N >> 1);
  int shift;
  int i;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict px1;
  ae_int32x2 * restrict py0;
  const ae_int32x2 * restrict ptwd;
  const int R = 2; // stage radix

  ae_int32x2 vmax;
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  const int min_shift = 2;
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);

  vmax = AE_MOVI(0);
  px0 = (ae_int32x2 *)(x + 0 * stride * 2);
  px1 = (ae_int32x2 *)(x + 1 * stride * 2);
  py0 = (ae_int32x2 *)(y);

  ptwd = (const ae_int32x2 *)tw;
  /* hifi3z: 5 cycles per stage, unroll=1*/
  __Pragma("loop_count min=3");
  for (i = 0; i < stride; i++)
  {
    ae_int32x2 x0, x1, y0, y1;
    ae_int32x2 tw1;
    ae_int32x2 t0;
    AE_L32X2_IP(x0, px0, sizeof(ae_int32x2));
    AE_L32X2_IP(x1, px1, sizeof(ae_int32x2));
    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));

    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);

    y0 = AE_ADD32S(x0, x1);
    y1 = AE_SUB32S(x0, x1);

    _cmult32x32(&y1, &y1, &tw1);
    AE_S32X2_IP(y0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(y1, py0, sizeof(ae_int32x2));

    t0 = AE_MAXABS32S(y0, y1);
    vmax = AE_OR32(t0, vmax);
  }

  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }
  *v *= R;
  return shift;
} /* fft_stageS2_DFT2_first_32x32() */

int fft_stageS2_DFT2_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{
  const int stride = (N >> 1);
  int shift;
  ae_int32x2 vmax;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;
  const int R = 2; // stage radix
  const int min_shift = 2; //todo possible mistake
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);

  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);
  int j;

  vmax = AE_MOVI(0);
  px0 = (ae_int32x2 *)((uintptr_t)x);
  py0 = (ae_int32x2 *)(0 * sizeof(ae_int32x2) + (uintptr_t)y);

  /* hifi3z: 5 cycles per stage, unroll=1*/
  __Pragma("loop_count min=3");
  for (j = 0; j < stride; j++)
  {
    ae_int32x2 x0, x1, y0, y1;
    ae_int32x2 t0;

    x1 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride);
    AE_L32X2_IP(x0, px0, sizeof(ae_int32x2));

    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);

    y0 = AE_ADD32S(x0, x1);
    y1 = AE_SUB32S(x0, x1);

    AE_S32X2_XP(y0, py0, stride * sizeof(ae_int32x2));
    AE_S32X2_XP(y1, py0, (-stride +1) * sizeof(ae_int32x2));
    t0 = AE_MAXABS32S(y0, y1);
    vmax = AE_OR32(t0, vmax);
  }
  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }
  *v *= R;
  return shift;

} /* fft_stageS2_DFT2_last_32x32() */
int fft_stageS2_DFT2_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{ 
  const int stride = (N >> 1);
  /* inner stage */
  ae_int32x2 vmax;
  int shift;
  int i;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;
  const ae_int32x2 * restrict ptwd;
  const int R = 2; // stage radix

  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  const int min_shift = 2;//todo possible mistake
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);

  vmax = AE_MOVI(0);

  {
#if 1
    int j;
    shift = 1;
    int _v = v[0];
    for (j = 0; j < _v; j++)
    {
      ae_int32x2 x0, x1, y0, y1;
      ae_int32x2 tw1;
      ptwd = (const ae_int32x2 *)tw;
      px0 = (ae_int32x2 *)(j * sizeof(ae_int32x2) + (uintptr_t)x);
      py0 = (ae_int32x2 *)(j * sizeof(ae_int32x2) + (uintptr_t)y);
      /* hifi3z: 11 cycles per stage, unroll=2*/
      __Pragma("loop_count min=1");
      for (i = 0; i < (stride / _v); i++)
      {
        ae_int32x2 t0;
        AE_L32X2_XP(tw1, ptwd, ((tw_step - 1) + 1)*sizeof(ae_int32x2));
        x1 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride);
        AE_L32X2_XP(x0, px0, _v * sizeof(ae_int32x2));

        x0 = AE_SRAA32RS(x0, shift);
        x1 = AE_SRAA32RS(x1, shift);

        y0 = AE_ADD32S(x0, x1);
        y1 = AE_SUB32S(x0, x1);

        _cmult32x32(&y1, &y1, &tw1);

        AE_S32X2_XP(y0, py0, _v * sizeof(ae_int32x2));
        AE_S32X2_XP(y1, py0, _v * sizeof(ae_int32x2));
        t0 = AE_MAXABS32S(y0, y1);
        vmax = AE_OR32(t0, vmax);

      }
    }
    {
      ae_int32x2 tmp;
      ae_int64   vL;
      int exp;
      tmp = AE_SEL32_LH(vmax, vmax);
      vmax = AE_OR32(tmp, vmax);
      vL = AE_MOVINT64_FROMINT32X2(vmax);
      exp = AE_NSA64(vL);
      exp = XT_MIN(31, exp);
      *bexp = (exp);
    }
#else
    int  flag = 0;
    int _v = v[0];
    const int tw_inc0 = sizeof(complex_fract32);
    ae_int32x2 start_incs = AE_MOVDA32X2((-1 * _v + 1)* sizeof(ae_int32x2),
      0 * tw_inc0);
    ae_int32x2 x0, x1, y0, y1;
    ae_int32x2 tw1;
    py0 = (ae_int32x2*)y;
    px0 = (ae_int32x2 *)(x);
    ptwd = (const ae_int32x2 *)tw;
    flag = _v;
    /* hifi3z: 8 cycles per stage, unroll=1 */
    __Pragma("loop_count min=1");
    for (i = 0; i< stride; i++)
    {
      int py_inc;// = (-1 * _v + 1) * sizeof(ae_int32x2) 
      int tw_inc;// = 0;
      ae_int32x2 t0;
      py_inc = AE_MOVAD32_H(start_incs);
      tw_inc = AE_MOVAD32_L(start_incs);

      flag--;
      XT_MOVEQZ(py_inc, (1)* sizeof(complex_fract32), flag);
      XT_MOVEQZ(tw_inc, (1 * (tw_step - 1)*sizeof(complex_fract32) + sizeof(complex_fract32)), flag);
      XT_MOVEQZ(flag, _v, flag);

      AE_L32X2_XP(tw1, ptwd, tw_inc);

      x1 = AE_L32X2_X(px0, 1 * stride*sizeof(complex_fract32));
      AE_L32X2_IP(x0, px0, sizeof(complex_fract32));

      x0 = AE_SRAA32RS(x0, shift);
      x1 = AE_SRAA32RS(x1, shift);

      y0 = AE_ADD32S(x0, x1);
      y1 = AE_SUB32S(x0, x1);

      _cmult32x32(&y1, &y1, &tw1);

      AE_S32X2_XP(y0, py0, _v * sizeof(ae_int32x2));
      AE_S32X2_XP(y1, py0, py_inc);
      t0 = AE_MAXABS32S(y0, y1);
      vmax = AE_OR32(t0, vmax);

    }
    {
      ae_int32x2 tmp;
      ae_int64   vL;
      int exp;
      tmp = AE_SEL32_LH(vmax, vmax);
      vmax = AE_OR32(tmp, vmax);
      vL = AE_MOVINT64_FROMINT32X2(vmax);
      exp = AE_NSA64(vL);
      exp = XT_MIN(31, exp);
      *bexp = (exp);
    }
#endif
  }
  *v *= R;
  return shift;
} /* fft_stageS2_DFT2_32x32() */

/*
*  32x32 FFT stages, Radix 3, scalingOption=2
*/
int fft_stageS2_DFT3_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{
  const int R = 3; // stage radix
  const int stride = N / R;
    int shift;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict px1;
  ae_int32x2 * restrict px2;
  ae_int32x2 * restrict py0;
  ae_int32x2 vmax;
  const ae_int32x2 * restrict ptwd;
  int min_shift = 3;
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  int i;

  px0 = (ae_int32x2 *)(x + 0 * stride * 2);
  px1 = (ae_int32x2 *)(x + 1 * stride * 2);
  px2 = (ae_int32x2 *)(x + 2 * stride * 2);

  py0 = (ae_int32x2 *)(y);

  ptwd = (const ae_int32x2 *)tw;
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);
  vmax = AE_MOVI(0);

  /* hifi3z: 10 cycles per stage, unroll=1*/
  __Pragma("loop_count min=2");
  for (i = 0; i <stride; i++)
  {
    ae_int32x2 x0, x1, x2;
    ae_int32x2 tw1, tw2;

    AE_L32X2_IP(x0, px0, sizeof(ae_int32x2));
    AE_L32X2_IP(x1, px1, sizeof(ae_int32x2));
    AE_L32X2_IP(x2, px2, sizeof(ae_int32x2));

    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);

    DFT3X1(x0, x1, x2);
    _cmult32x32(&x1, &x1, &tw1);
    _cmult32x32(&x2, &x2, &tw2);

    AE_S32X2_IP(x0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x1, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x2, py0, sizeof(ae_int32x2));

    x0 = AE_MAXABS32S(x0, x1);
    vmax = AE_MAXABS32S(vmax, x2);
    vmax = AE_MAX32(vmax, x0);

  }
  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }

  *v = v[0] * R;

  return shift;


} /* fft_stageS2_DFT3_first_32x32() */
int fft_stageS2_DFT3_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{
  const int R = 3; // stage radix
  const int stride = N / R;
    int shift;
  int min_shift = 3;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;
  ae_int32x2 vmax;
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  int j, _v;
  _v = v[0];
  px0 = (ae_int32x2 *)(0 * sizeof(ae_int32x2) + (uintptr_t)x);
  py0 = (ae_int32x2 *)(0 * sizeof(ae_int32x2) + (uintptr_t)y);

  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);
  vmax = AE_MOVI(0);

  /* hifi3z: 10 cycles per stage, unroll=1 */
  __Pragma("loop_count min=2");
  for (j = 0; j < _v; j++)
  {
    ae_int32x2 x0, x1, x2;

    x1 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride);
    x2 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 2);
    AE_L32X2_IP(x0, px0, sizeof(ae_int32x2));

    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);

    DFT3X1(x0, x1, x2);

    AE_S32X2_XP(x0, py0, stride * sizeof(ae_int32x2));
    AE_S32X2_XP(x1, py0, stride * sizeof(ae_int32x2));
    AE_S32X2_XP(x2, py0, (-2*stride+1) * sizeof(ae_int32x2));

    x0 = AE_ABS32(x0);
    x1 = AE_ABS32(x1);
    x2 = AE_ABS32(x2);

    x0 = AE_MAXABS32S(x0, x1);
    vmax = AE_MAXABS32S(vmax, x2);
    vmax = AE_MAX32(vmax, x0);

  }
  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }
  *v = v[0] * R;

  return shift;


} /* fft_stageS2_DFT3_last_32x32() */
int fft_stageS2_DFT3_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{ 
  const int R = 3; // stage radix
  const int stride = N / R;
  int shift;
  int min_shift = 3;
  ae_int32x2 vmax;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;
  const ae_int32x2 * restrict ptwd;
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);
  vmax = AE_MOVI(0);

  {
    int i;
    int _v;
    _v = v[0];
#if 0
    int j;
    __Pragma("loop_count min=1");
    for (j = 0; j < _v; j++)
    {
      ae_int32x2 x0, x1, x2;
      ae_int32x2 tw1, tw2;
      ptwd = (const ae_int32x2 *)tw;
      px0 = (ae_int32x2 *)(j * sizeof(ae_int32x2) + (uintptr_t)x);
      py0 = (ae_int32x2 *)(j * sizeof(ae_int32x2) + (uintptr_t)y);
      /* hifi3z: 11 cycles per stage, unroll=1*/
      __Pragma("loop_count min=1");
      for (i = 0; i < (stride / _v); i++)
      {
        ae_int32x2 t0;
        AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
        AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
        x1 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride);
        x2 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 2);
        AE_L32X2_XP(x0, px0, _v * sizeof(ae_int32x2));

        x0 = AE_SRAA32RS(x0, shift);
        x1 = AE_SRAA32RS(x1, shift);
        x2 = AE_SRAA32RS(x2, shift);

        DFT3X1(x0, x1, x2);

        _cmult32x32(&x1, &x1, &tw1);
        _cmult32x32(&x2, &x2, &tw2);
        AE_S32X2_XP(x0, py0, _v * sizeof(ae_int32x2));
        AE_S32X2_XP(x1, py0, _v * sizeof(ae_int32x2));
        AE_S32X2_XP(x2, py0, _v * sizeof(ae_int32x2));
        x0 = AE_MAXABS32S(x0, x1);
        vmax = AE_MAXABS32S(vmax, x2);
        vmax = AE_MAX32(vmax, x0);
      }
    }
#else
  {
    int flag = 0;
    const int tw_inc0 = sizeof(complex_fract32);
    ae_int32x2 start_incs = AE_MOVDA32X2((-2 * _v + 1)* sizeof(ae_int32x2),
      -1 * tw_inc0);
    ae_int32x2 x0, x1, x2;
    ae_int32x2 tw1, tw2;
    py0 = (ae_int32x2*)y;
    px0 = (ae_int32x2 *)(x);
    ptwd = (const ae_int32x2 *)tw;
    flag = _v;
    /* hifi3z: 13 cycles per stage, unroll=1*/
    __Pragma("loop_count min=1");
    for (i = 0; i< stride; i++)
    {
      int py_inc;// = (-2 * _v + 1)* sizeof(ae_int32x2)
      int tw_inc;// = -1*tw_inc0;
      py_inc = AE_MOVAD32_H(start_incs);
      tw_inc = AE_MOVAD32_L(start_incs);
      flag--;
      XT_MOVEQZ(py_inc, sizeof(complex_fract32), flag);
      XT_MOVEQZ(tw_inc, sizeof(complex_fract32), flag);
      XT_MOVEQZ(flag, _v, flag);

      AE_L32X2_XP(tw1, ptwd, tw_inc0);
      AE_L32X2_XP(tw2, ptwd, tw_inc);

      x1 = AE_L32X2_X(px0, 1 * stride*sizeof(complex_fract32));
      x2 = AE_L32X2_X(px0, 2 * stride*sizeof(complex_fract32));
      AE_L32X2_IP(x0, px0, sizeof(complex_fract32));

      x0 = AE_SRAA32RS(x0, shift);
      x1 = AE_SRAA32RS(x1, shift);
      x2 = AE_SRAA32RS(x2, shift);

      DFT3X1(x0, x1, x2);

      _cmult32x32(&x1, &x1, &tw1);
      _cmult32x32(&x2, &x2, &tw2);

      AE_S32X2_XP(x0, py0, _v * sizeof(ae_int32x2));
      AE_S32X2_XP(x1, py0, _v * sizeof(ae_int32x2));
      AE_S32X2_XP(x2, py0, py_inc);

      x0 = AE_MAXABS32S(x0, x1);
      vmax = AE_MAXABS32S(vmax, x2);
      vmax = AE_MAX32(vmax, x0);
    }
  }

#endif
  }
  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }
  *v = v[0] * R;

  return shift;
} /* fft_stageS2_DFT3_32x32() */

/*
*  32x32 FFT stages, Radix 4, scalingOption=2
*/
int fft_stageS2_DFT4_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{
  int i;
  int shift;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict px1;
  ae_int32x2 * restrict px2;
  ae_int32x2 * restrict px3;
  ae_int32x2 * restrict py0;
  const ae_int32x2 * restrict ptwd;
  const int R = 4; // stage radix
  const int stride = (N >> 2);
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  ae_int32x2 vmax;
  const int min_shift = 3;
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);

  vmax = AE_MOVI(0);
  px0 = (ae_int32x2 *)(x + 0 * stride * 2);
  px1 = (ae_int32x2 *)(x + 1 * stride * 2);
  px2 = (ae_int32x2 *)(x + 2 * stride * 2);
  px3 = (ae_int32x2 *)(x + 3 * stride * 2);

  py0 = (ae_int32x2 *)(y);

  ptwd = (const ae_int32x2 *)tw;
  __Pragma("loop_count min=3");
  /* hifi3z: 13 cycles per stage, unroll=1 */
  for (i = 0; i < stride; i++)
  {
    ae_int32x2 x0, x1, x2, x3;
    ae_int32x2 tw1, tw2, tw3;

    AE_L32X2_IP(x0, px0, sizeof(ae_int32x2));
    AE_L32X2_IP(x1, px1, sizeof(ae_int32x2));
    AE_L32X2_IP(x2, px2, sizeof(ae_int32x2));
    AE_L32X2_IP(x3, px3, sizeof(ae_int32x2));

    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
    AE_L32X2_XP(tw3, ptwd, (3 * tw_step - 2) * sizeof(ae_int32x2));

    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);
    x3 = AE_SRAA32RS(x3, shift);

    DFT4X1(x0, x1, x2, x3);

    _cmult32x32(&x1, &x1, &tw1);
    _cmult32x32(&x2, &x2, &tw2);
    _cmult32x32(&x3, &x3, &tw3);

    
    AE_S32X2_IP(x0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x1, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x2, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x3, py0, sizeof(ae_int32x2));

    x0 = AE_MAXABS32S(x0, x1);
    x1 = AE_MAXABS32S(x2, x3);
    vmax = AE_MAX32(vmax, x0);
    vmax = AE_MAX32(vmax, x1);
  }

  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }
    
  *v *= R;
  return shift;

} /* fft_stageS2_DFT4_first_32x32() */

int fft_stageS2_DFT4_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{
  int shift;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;

  const int R = 4; // stage radix
  const int stride = (N >> 2);
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  int j;
  ae_int32x2 vmax;

  const int min_shift = 2;
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);

  vmax = AE_MOVI(0);

  px0 = (ae_int32x2 *)(0 * sizeof(ae_int32x2) + (uintptr_t)x);
  py0 = (ae_int32x2 *)(0 * sizeof(ae_int32x2) + (uintptr_t)y);
  /* hifi3z: 10 cycles per stage, unroll=1*/
  __Pragma("loop_count min=3"); 
  for (j = 0; j < stride; j++)
  {
    ae_int32x2 x0, x1, x2, x3;

    x1 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride);
    x3 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 3);
    x2 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 2);
    AE_L32X2_IP(x0, px0, sizeof(ae_int32x2));

    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);
    x3 = AE_SRAA32RS(x3, shift);

    DFT4X1(x0, x1, x2, x3);

    AE_S32X2_XP(x0, py0, stride * sizeof(ae_int32x2));
    AE_S32X2_XP(x1, py0, stride * sizeof(ae_int32x2));
    AE_S32X2_XP(x2, py0, stride * sizeof(ae_int32x2));
    AE_S32X2_XP(x3, py0, (-3*stride+1) * sizeof(ae_int32x2));

    x0 = AE_MAXABS32S(x0, x1);
    x1 = AE_MAXABS32S(x2, x3);
    vmax = AE_MAX32(vmax, x0);
    vmax = AE_MAX32(vmax, x1);
  }
  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }

  *v *= R;
  return shift;

} /* fft_stageS2_DFT4_last_32x32() */

ALIGN(8) static const int32_t __fft8_tw1[] =
{
    (int32_t)0x7FFFFFFF, (int32_t)0x00000000,
    (int32_t)0x2D413CCD, (int32_t)0xD2BEC333,
    (int32_t)0x00000000, (int32_t)0xC0000000,
    (int32_t)0xD2BEC333, (int32_t)0xD2BEC333,
};

int fft_stageS2_DFT8_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{
    int shift;
    ae_int32x2 * restrict px0;
    ae_int32x2 * restrict py0;

    const int R = 8; // stage radix
    const int stride = (N >> 3);
    NASSERT_ALIGN8(x);
    NASSERT_ALIGN8(y);
    int j;
    ae_int32x2 vmax;
    ae_int32x2 tw1, tw2, tw3;
    const ae_int32x2 * restrict ptwd;

    px0 = (ae_int32x2 *)(7 * stride * sizeof(ae_int32x2)+(uintptr_t)x);
    py0 = (ae_int32x2 *)(0 * sizeof(ae_int32x2)+(uintptr_t)y);
    ptwd = (const ae_int32x2 *)__fft8_tw1;
    const int min_shift = 2;
    shift = min_shift - *bexp;
    ASSERT(shift>-32 && shift<32);

    vmax = AE_MOVI(0);

     /* hifi3z: 10 cycles per stage, unroll=1*/
    __Pragma("loop_count min=3");
    for (j = 0; j < stride; j++)
    {




        ae_int32x2 x0, x1, x2, x3;
        ae_int32x2 x4, x5, x6, x7;

        tw2 = AE_L32X2_I(ptwd, 2 * sizeof(ae_int32x2));
        tw1 = AE_L32X2_I(ptwd, 1 * sizeof(ae_int32x2));
        tw3 = AE_L32X2_I(ptwd, 3 * sizeof(ae_int32x2));

        AE_L32X2_XP(x7, px0, -1 * (sizeof(ae_int32x2)* stride));
        AE_L32X2_XP(x6, px0, -1 * (sizeof(ae_int32x2)* stride));
        AE_L32X2_XP(x5, px0, -1 * (sizeof(ae_int32x2)* stride));
        AE_L32X2_XP(x4, px0, -1 * (sizeof(ae_int32x2)* stride));
        AE_L32X2_XP(x3, px0, -1 * (sizeof(ae_int32x2)* stride));
        AE_L32X2_XP(x2, px0, -1 * (sizeof(ae_int32x2)* stride));
        AE_L32X2_XP(x1, px0, -1 * (sizeof(ae_int32x2)* stride));
        AE_L32X2_XP(x0, px0, sizeof(ae_int32x2)*(7 * stride + 1));

        x7 = AE_SRAA32RS(x7, shift);
        x6 = AE_SRAA32RS(x6, shift);
        x5 = AE_SRAA32RS(x5, shift);
        x4 = AE_SRAA32RS(x4, shift);
        x3 = AE_SRAA32RS(x3, shift);
        x2 = AE_SRAA32RS(x2, shift);
        x1 = AE_SRAA32RS(x1, shift);
        x0 = AE_SRAA32RS(x0, shift);

        DFT4X1(x0, x2, x4, x6);
        DFT4X1(x1, x3, x5, x7);

        _cmult32x32(&x3, &x3, &tw1);
        _cmult32x32(&x5, &x5, &tw2);
        _cmult32x32(&x7, &x7, &tw3);

        {
            ae_int32x2 s0, s1, s2, s3;
            ae_int32x2 d0, d1, d2, d3;

            x0 = AE_SRAI32(x0, 1);
            x1 = AE_SRAI32(x1, 1);
            x2 = AE_SRAI32(x2, 1);
            x4 = AE_SRAI32(x4, 1);
            x6 = AE_SRAI32(x6, 1);

            s0 = AE_ADD32S(x0, x1);   d0 = AE_SUB32S(x0, x1);
            s1 = AE_ADD32S(x2, x3);   d1 = AE_SUB32S(x2, x3);
            s2 = AE_ADD32S(x4, x5);   d2 = AE_SUB32S(x4, x5);
            s3 = AE_ADD32S(x6, x7);   d3 = AE_SUB32S(x6, x7);

            x0 = s0;        x4 = d0;
            x1 = s1;        x5 = d1;
            x2 = s2;        x6 = d2;
            x3 = s3;        x7 = d3;
        }

        AE_S32X2_XP(x0, py0, stride * sizeof(ae_int32x2));
        AE_S32X2_XP(x1, py0, stride * sizeof(ae_int32x2));
        AE_S32X2_XP(x2, py0, stride * sizeof(ae_int32x2));
        AE_S32X2_XP(x3, py0, stride * sizeof(ae_int32x2));
        AE_S32X2_XP(x4, py0, stride * sizeof(ae_int32x2));
        AE_S32X2_XP(x5, py0, stride * sizeof(ae_int32x2));
        AE_S32X2_XP(x6, py0, stride * sizeof(ae_int32x2));
        AE_S32X2_XP(x7, py0, (-7 * stride + 1)* sizeof(ae_int32x2));
    }


    *v *= R;
    return shift+1;

} /* fft_stageS2_DFT8_last_32x32() */


int fft_stageS2_DFT4_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{ 
  int i;
  int shift;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;
  const ae_int32x2 * restrict ptwd;
  const int R = 4; // stage radix
  const int stride = (N >> 2);
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  
  ae_int32x2 vmax, vmin;
  int _v;

  const int min_shift = 3;
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);

  vmax = AE_MOVI(0);
  vmin = AE_MOVI(0);
  _v = v[0];
#if 0
  int j;
  __Pragma("loop_count min=1");
  for (j = 0; j < _v; j++)
  {
    ae_int32x2 x0, x1, x2, x3;
    ae_int32x2 tw1, tw2, tw3;
    ptwd = (const ae_int32x2 *)tw;
    px0 = (ae_int32x2 *)(j * sizeof(ae_int32x2) + (uintptr_t)x);
    py0 = (ae_int32x2 *)(j * sizeof(ae_int32x2) + (uintptr_t)y);
    /* hifi3z: 13 cycles per stage, unroll=1*/
    __Pragma("loop_count min=1");
    for (i = 0; i < (stride / _v); i++)
    {
      AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
      AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
      AE_L32X2_XP(tw3, ptwd, (3 * (tw_step - 1) + 1)*sizeof(ae_int32x2));

      x1 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride);
      x3 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 3);
      x2 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 2);
      AE_L32X2_XP(x0, px0, _v * sizeof(ae_int32x2));

      x0 = AE_SRAA32RS(x0, shift);
      x1 = AE_SRAA32RS(x1, shift);
      x2 = AE_SRAA32RS(x2, shift);
      x3 = AE_SRAA32RS(x3, shift);

      DFT4X1(x0, x1, x2, x3);

      _cmult32x32(&x1, &x1, &tw1);
      _cmult32x32(&x2, &x2, &tw2);
      _cmult32x32(&x3, &x3, &tw3);

      AE_S32X2_XP(x0, py0, _v * sizeof(ae_int32x2));
      AE_S32X2_XP(x1, py0, _v * sizeof(ae_int32x2));
      AE_S32X2_XP(x2, py0, _v * sizeof(ae_int32x2));
      AE_S32X2_XP(x3, py0, _v * sizeof(ae_int32x2));

      x0 = AE_MAXABS32S(x0, x1);
      x1 = AE_MAXABS32S(x2, x3);
      vmax = AE_MAX32(vmax, x0);
      vmax = AE_MAX32(vmax, x1);

    }
  }
  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }
#else
  {
    int  flag = 0;
    const int tw_inc0 = sizeof(complex_fract32);
    ae_int32x2 start_incs = AE_MOVDA32X2((-3 * _v + 1)* sizeof(ae_int32x2),
                                         -2 * tw_inc0);
    ae_int32x2 x0, x1, x2, x3;
    ae_int32x2 tw1, tw2, tw3;
    py0 = (ae_int32x2*)y;
    px0 = (ae_int32x2 *)(x);
    ptwd = (const ae_int32x2 *)tw; 
    flag = _v;
    /* hifi3z: 18 cycles per stage, unroll=1 */
    __Pragma("loop_count min=3");
    for (i = 0; i< stride; i++)
    {
      int py_inc;// = (-3 * _v + 1)* sizeof(ae_int32x2)
      int tw_inc;// = -2*tw_inc0;

      py_inc = AE_MOVAD32_H(start_incs);
      tw_inc = AE_MOVAD32_L(start_incs);
      flag--;
      XT_MOVEQZ(py_inc, (1)* sizeof(complex_fract32), flag);
      XT_MOVEQZ(tw_inc, (3 * (tw_step - 1) + 1)*sizeof(ae_int32x2), flag);
      XT_MOVEQZ(flag, _v, flag);

      AE_L32X2_IP(tw1, ptwd, sizeof(complex_fract32));
      AE_L32X2_IP(tw2, ptwd, sizeof(complex_fract32));
      AE_L32X2_XP(tw3, ptwd, tw_inc);

      x1 = AE_L32X2_X(px0, 1*stride*sizeof(complex_fract32));
      x2 = AE_L32X2_X(px0, 2*stride*sizeof(complex_fract32));
      x3 = AE_L32X2_X(px0, 3*stride*sizeof(complex_fract32));
      AE_L32X2_IP(x0, px0, sizeof(complex_fract32));

      x0 = AE_SRAA32RS(x0, shift);
      x1 = AE_SRAA32RS(x1, shift);
      x2 = AE_SRAA32RS(x2, shift);
      x3 = AE_SRAA32RS(x3, shift);

      DFT4X1(x0, x1, x2, x3);

      _cmult32x32(&x1, &x1, &tw1);
      _cmult32x32(&x2, &x2, &tw2);
      _cmult32x32(&x3, &x3, &tw3);

      AE_S32X2_XP(x0, py0, _v * sizeof(ae_int32x2));
      AE_S32X2_XP(x1, py0, _v * sizeof(ae_int32x2));
      AE_S32X2_XP(x2, py0, _v * sizeof(ae_int32x2));
      AE_S32X2_XP(x3, py0, py_inc);
      x0 = AE_MAXABS32S(x0, x1);
      x1 = AE_MAXABS32S(x2, x3);
      vmax = AE_MAX32(vmax, x0);
      vmax = AE_MAX32(vmax, x1);

    }
  }

  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }

#endif

  *v *= R;
  return shift;
} /* fft_stageS2_DFT4_32x32() */
 /*
 *  32x32 FFT stages, Radix 5, scalingOption=2
 */
int fft_stageS2_DFT5_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{
  const int R = 5; // stage radix
  const int stride = N / R;
  int shift, min_shift;
  ae_int32x2 vmax;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict px1;
  ae_int32x2 * restrict px2;
  ae_int32x2 * restrict px3;
  ae_int32x2 * restrict px4;
  ae_int32x2 * restrict py0;
  const ae_int32x2 * restrict ptwd;
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  int i;

  px0 = (ae_int32x2 *)(x + 0 * stride * 2);
  px1 = (ae_int32x2 *)(x + 1 * stride * 2);
  px2 = (ae_int32x2 *)(x + 2 * stride * 2);
  px3 = (ae_int32x2 *)(x + 3 * stride * 2);
  px4 = (ae_int32x2 *)(x + 4 * stride * 2);

  py0 = (ae_int32x2 *)(y);

  ptwd = (const ae_int32x2 *)tw;
  min_shift = 4;
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);
  vmax = AE_MOVI(0);
  /* hifi3z: 31 cycles per stage, unroll=1*/
  __Pragma("loop_count min=6");
  for (i = 0; i <stride; i++)
  {
    ae_int32x2 x0, x1, x2, x3, x4;
    ae_int32x2 tw1, tw2, tw3, tw4;

    AE_L32X2_IP(x0, px0, sizeof(ae_int32x2));
    AE_L32X2_IP(x1, px1, sizeof(ae_int32x2));
    AE_L32X2_IP(x2, px2, sizeof(ae_int32x2));
    AE_L32X2_IP(x3, px3, sizeof(ae_int32x2));
    AE_L32X2_IP(x4, px4, sizeof(ae_int32x2));

    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw3, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw4, ptwd, sizeof(ae_int32x2));
    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);
    x3 = AE_SRAA32RS(x3, shift);
    x4 = AE_SRAA32RS(x4, shift);

    DFT5X1(x0, x1, x2, x3, x4);
    _cmult32x32(&x1, &x1, &tw1);
    _cmult32x32(&x2, &x2, &tw2);
    _cmult32x32(&x3, &x3, &tw3);
    _cmult32x32(&x4, &x4, &tw4);

    AE_S32X2_IP(x0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x1, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x2, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x3, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x4, py0, sizeof(ae_int32x2));

    x0 = AE_MAXABS32S(x0, x1);
    x1 = AE_MAXABS32S(x2, x3);
    vmax = AE_MAXABS32S(vmax, x4);
    vmax = AE_MAX32(vmax, x0);
    vmax = AE_MAX32(vmax, x1);
  }

  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }

  *v = v[0] * R;
  return shift;

} /* fft_stageS2_DFT5_first_32x32() */
int fft_stageS2_DFT5_last_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{
  const int R = 5; // stage radix
  const int stride = N / R;
  int shift, min_shift;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;
  ae_int32x2 vmax;
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  int j, _v;
  _v = v[0];
  px0 = (ae_int32x2 *)(0 * sizeof(ae_int32x2) + (uintptr_t)x);
  py0 = (ae_int32x2 *)(0 * sizeof(ae_int32x2) + (uintptr_t)y);
  min_shift = 3;
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);
  vmax = AE_MOVI(0);

  /* hifi3z: 22 cycles per stage, unroll=1*/
  __Pragma("loop_count min=6");
  for (j = 0; j < _v; j++)
  {
    ae_int32x2 x0, x1, x2, x3, x4;

    x1 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride);
    x2 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 2);
    x3 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 3);
    x4 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 4);
    AE_L32X2_IP(x0, px0, sizeof(ae_int32x2));

    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);
    x3 = AE_SRAA32RS(x3, shift);
    x4 = AE_SRAA32RS(x4, shift);

    DFT5X1(x0, x1, x2, x3, x4);

    AE_S32X2_XP(x0, py0, stride * sizeof(ae_int32x2));
    AE_S32X2_XP(x1, py0, stride * sizeof(ae_int32x2));
    AE_S32X2_XP(x2, py0, stride * sizeof(ae_int32x2));
    AE_S32X2_XP(x3, py0, stride * sizeof(ae_int32x2));
    AE_S32X2_XP(x4, py0, (-4*stride+1) * sizeof(ae_int32x2));
    x0 = AE_MAXABS32S(x0, x1);
    x1 = AE_MAXABS32S(x2, x3);
    vmax = AE_MAXABS32S(vmax, x4);
    vmax = AE_MAX32(vmax, x0);
    vmax = AE_MAX32(vmax, x1);
  }
  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }
  *v = v[0] * R;
  return shift;

} /* fft_stageS2_DFT5_last_32x32() */
int fft_stageS2_DFT5_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{
  const int R = 5; // stage radix
  const int stride = N / R;
  int shift, min_shift;
  ae_int32x2 vmax;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;
  const ae_int32x2 * restrict ptwd;
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  {
#if 1
    int i;
    int _v;
    min_shift = 4;
    shift = min_shift - *bexp;
    ASSERT(shift>-32 && shift<32);
    vmax = AE_MOVI(0);
    _v = v[0];
    int j;
    for (j = 0; j < _v; j++)
    {
      ae_int32x2 x0, x1, x2, x3, x4;
      ae_int32x2 tw1, tw2, tw3, tw4;
      ptwd = (const ae_int32x2 *)tw;
      px0 = (ae_int32x2 *)(j * sizeof(ae_int32x2) + (uintptr_t)x);
      py0 = (ae_int32x2 *)(j * sizeof(ae_int32x2) + (uintptr_t)y);
      /* hifi3z: 30 cycles per stage, unroll=1*/
      for (i = 0; i < (stride / _v); i++)
      {
        AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
        AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
        AE_L32X2_IP(tw3, ptwd, sizeof(ae_int32x2));
        AE_L32X2_IP(tw4, ptwd, sizeof(ae_int32x2));


        x1 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride);
        x2 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 2);
        x3 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 3);
        x4 = AE_L32X2_X(px0, sizeof(ae_int32x2) * stride * 4);
        AE_L32X2_XP(x0, px0, _v * sizeof(ae_int32x2));

        x0 = AE_SRAA32RS(x0, shift);
        x1 = AE_SRAA32RS(x1, shift);
        x2 = AE_SRAA32RS(x2, shift);
        x3 = AE_SRAA32RS(x3, shift);
        x4 = AE_SRAA32RS(x4, shift);

        DFT5X1(x0, x1, x2, x3, x4);

        _cmult32x32(&x1, &x1, &tw1);
        _cmult32x32(&x2, &x2, &tw2);
        _cmult32x32(&x3, &x3, &tw3);
        _cmult32x32(&x4, &x4, &tw4);
        AE_S32X2_XP(x0, py0, _v * sizeof(ae_int32x2));
        AE_S32X2_XP(x1, py0, _v * sizeof(ae_int32x2));
        AE_S32X2_XP(x2, py0, _v * sizeof(ae_int32x2));
        AE_S32X2_XP(x3, py0, _v * sizeof(ae_int32x2));
        AE_S32X2_XP(x4, py0, _v * sizeof(ae_int32x2));

        x0 = AE_MAXABS32S(x0, x1);
        x1 = AE_MAXABS32S(x2, x3);
        vmax = AE_MAXABS32S(vmax, x4);
        vmax = AE_MAX32(vmax, x0);
        vmax = AE_MAX32(vmax, x1);
      }
    }
    {
      ae_int32x2 tmp; 
      ae_int64   vL;
      int exp;
      tmp = AE_SEL32_LH(vmax, vmax);
      vmax = AE_OR32(tmp, vmax);
      vL = AE_MOVINT64_FROMINT32X2(vmax);
      exp = AE_NSA64(vL);
      exp = XT_MIN(31, exp);
      *bexp = (exp);
    }
#else
    {
      uint32_t  flag = 0;
      int _v, i;
      _v = v[0];
      min_shift = 4;
      shift = min_shift - *bexp;
      ASSERT(shift>-32 && shift<32);
      vmax = AE_MOVI(0);
      const int tw_inc0 = sizeof(complex_fract32);
      ae_int32x2 start_incs = AE_MOVDA32X2((-4 * _v + 1)* sizeof(ae_int32x2),
        -3 * tw_inc0);
      ae_int32x2 x0, x1, x2, x3, x4;
      ae_int32x2 tw1, tw2, tw3, tw4;
      py0 = (ae_int32x2*)y;
      px0 = (ae_int32x2 *)(x);
      ptwd = (const ae_int32x2 *)tw;
      flag = _v;
      /* hifi3z: 40 cycles per stage, unroll=1*/
      for (i = 0; i< stride; i++)
      {
        int py_inc;// = (-4 * _v + 1)* sizeof(ae_int32x2)
        int tw_inc;// = -3*tw_inc0;

        py_inc = AE_MOVAD32_H(start_incs);
        tw_inc = AE_MOVAD32_L(start_incs);
        flag--;

        XT_MOVEQZ(py_inc, (1)* sizeof(complex_fract32), flag);
        XT_MOVEQZ(tw_inc, sizeof(complex_fract32), flag);
        XT_MOVEQZ(flag, _v, flag);

        AE_L32X2_IP(tw1, ptwd, sizeof(complex_fract32));
        AE_L32X2_IP(tw2, ptwd, sizeof(complex_fract32));
        AE_L32X2_IP(tw3, ptwd, sizeof(complex_fract32));
        AE_L32X2_XP(tw4, ptwd, tw_inc);

        x1 = AE_L32X2_X(px0, 1 * stride*sizeof(complex_fract32));
        x2 = AE_L32X2_X(px0, 2 * stride*sizeof(complex_fract32));
        x3 = AE_L32X2_X(px0, 3 * stride*sizeof(complex_fract32));
        x4 = AE_L32X2_X(px0, 4 * stride*sizeof(complex_fract32));
        AE_L32X2_IP(x0, px0, sizeof(complex_fract32));

        x0 = AE_SRAA32RS(x0, shift);
        x1 = AE_SRAA32RS(x1, shift);
        x2 = AE_SRAA32RS(x2, shift);
        x3 = AE_SRAA32RS(x3, shift);
        x4 = AE_SRAA32RS(x4, shift);

        DFT5X1(x0, x1, x2, x3, x4);

        _cmult32x32(&x1, &x1, &tw1);
        _cmult32x32(&x2, &x2, &tw2);
        _cmult32x32(&x3, &x3, &tw3);
        _cmult32x32(&x4, &x4, &tw4);

        AE_S32X2_XP(x0, py0, _v * sizeof(ae_int32x2));
        AE_S32X2_XP(x1, py0, _v * sizeof(ae_int32x2));
        AE_S32X2_XP(x2, py0, _v * sizeof(ae_int32x2));
        AE_S32X2_XP(x3, py0, _v * sizeof(ae_int32x2));
        AE_S32X2_XP(x4, py0, py_inc);

        x0 = AE_MAXABS32S(x0, x1);
        x1 = AE_MAXABS32S(x2, x3);
        vmax = AE_MAXABS32S(vmax, x4);
        vmax = AE_MAX32(vmax, x0);
        vmax = AE_MAX32(vmax, x1);
      }
    }
    {
      ae_int32x2 tmp; 
      ae_int64   vL;
      int exp;
      tmp = AE_SEL32_LH(vmax, vmax);
      vmax = AE_OR32(tmp, vmax);
      vL = AE_MOVINT64_FROMINT32X2(vmax);
      exp = AE_NSA64(vL);
      exp = XT_MIN(31, exp);
      *bexp = (exp);
    }

#endif
  }

  *v = v[0] * R;
  return shift;
} /* fft_stageS2_DFT5_32x32() */

int ifft_stageS2_DFT2_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{ 
  int shift;
  ae_int32x2 vmax;
  int i;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;
  const ae_int32x2 * restrict ptwd;
  const int R = 2; // stage radix
  const int stride = (N >> 1);

  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  const int min_shift = 2;//todo possible mistake
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);
  py0 = (ae_int32x2 *)(y);

  ptwd = (const ae_int32x2 *)tw;
  vmax = AE_MOVI(0);
  px0 = (ae_int32x2*)(8 * (N - 1 * stride) + (uintptr_t)x);
  {
    /* First butterfly radix 2 */
    ae_int32x2 x0, x1, y0, y1;
    ae_int32x2 t0;
    ae_int32x2 tw1;

    AE_L32X2_XP(x1, px0, 8 * stride);
    x0 = AE_L32X2_X((ae_int32x2*)x, 0);

    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));

    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);

    y0 = AE_ADD32S(x0, x1);
    y1 = AE_SUB32S(x0, x1);

    _cmult32x32(&y1, &y1, &tw1);
    AE_S32X2_IP(y0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(y1, py0, sizeof(ae_int32x2));

    t0 = AE_MAXABS32S(y0, y1);
    vmax = AE_OR32(t0, vmax);
  }
  px0 = (ae_int32x2*)(8 * (N - 1 - 1 * stride) + (uintptr_t)x);
  /* hifi3z: 6 cycles per stage, unroll=1*/
  for (i = 1; i < stride; i++)
  { 
    ae_int32x2 x0, x1, y0, y1;
    ae_int32x2 t0;
    ae_int32x2 tw1;

    AE_L32X2_XP(x1, px0, 8 * stride);
    AE_L32X2_XP(x0, px0, (-1 * stride - 1) * 8);

    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));

    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);

    y0 = AE_ADD32S(x0, x1);
    y1 = AE_SUB32S(x0, x1);

    _cmult32x32(&y1, &y1, &tw1);
    AE_S32X2_IP(y0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(y1, py0, sizeof(ae_int32x2));

    t0 = AE_MAXABS32S(y0, y1);
    vmax = AE_OR32(t0, vmax);
  }
  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }
  *v *= R;
  return shift;
} /* ifft_stageS2_DFT2_first_32x32() */
int ifft_stageS2_DFT3_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{ 

  int shift;
  ae_int32x2 vmax;
  int min_shift = 3;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;
  const ae_int32x2 * restrict ptwd;
  const int R = 3; // stage radix
  const int stride = N / R;
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  int i;

  py0 = (ae_int32x2 *)(y);
  ptwd = (const ae_int32x2 *)tw;
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);
  vmax = AE_MOVI(0);
  px0 = (ae_int32x2*)(8 * (N - 2 * stride) + (uintptr_t)x);
  /*
  ifft(x) = fft( [ x(1), x(end:-1:2)] )
  */
  {
    /* First butterfly radix 3 */
    ae_int32x2 x0, x1, x2;
    ae_int32x2 tw1, tw2;

    AE_L32X2_XP(x2, px0, 8 * stride);
    AE_L32X2_XP(x1, px0, 8 * stride);
    x0 = AE_L32X2_X((ae_int32x2*)x, 0);

    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);

    DFT3X1(x0, x1, x2);
    _cmult32x32(&x1, &x1, &tw1);
    _cmult32x32(&x2, &x2, &tw2);

    AE_S32X2_IP(x0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x1, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x2, py0, sizeof(ae_int32x2));

    x0 = AE_MAXABS32S(x0, x1);
    vmax = AE_MAXABS32S(vmax, x2);
    vmax = AE_MAX32(vmax, x0);
  }
  px0 = (ae_int32x2*)(8 * (N - 1 - 2 * stride) + (uintptr_t)x);

  /* hifi3z: X cycles per stage, unroll=X*/
  for (i = 1; i <stride; i++)
  {
    ae_int32x2 x0, x1, x2;
    ae_int32x2 tw1, tw2;

    AE_L32X2_XP(x2, px0, 8 * stride);
    AE_L32X2_XP(x1, px0, 8 * stride);
    AE_L32X2_XP(x0, px0, (-2 * stride - 1) * 8);

    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);

    DFT3X1(x0, x1, x2);
    _cmult32x32(&x1, &x1, &tw1);
    _cmult32x32(&x2, &x2, &tw2);

    AE_S32X2_IP(x0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x1, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x2, py0, sizeof(ae_int32x2));

    x0 = AE_MAXABS32S(x0, x1);
    vmax = AE_MAXABS32S(vmax, x2);
    vmax = AE_MAX32(vmax, x0);
  }
  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_MAX32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }
  *v = v[0] * R;

  return shift;
} /* ifft_stageS2_DFT3_first_32x32() */
int ifft_stageS2_DFT4_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{
  int i;
  int shift;
  ae_int32x2 vmax;
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;
  const ae_int32x2 * restrict ptwd;
  const int R = 4; // stage radix
  const int stride = (N >> 2);
  const int min_shift = 3;
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);

  vmax = AE_MOVI(0);

  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);

  py0 = (ae_int32x2 *)(y);
  px0 = (ae_int32x2*)(8 * (N - 3 * stride) + (uintptr_t)x);
  ptwd = (const ae_int32x2 *)tw;

  __Pragma("loop_count min=3");

  /*
  ifft(x) = fft( [ x(1), x(end:-1:2)] )
  */
  {
    /* First butterfly radix 4 */
    ae_int32x2 x0, x1, x2, x3;
    ae_int32x2 tw1, tw2, tw3;

    AE_L32X2_XP(x3, px0, 8 * stride);
    AE_L32X2_XP(x2, px0, 8 * stride);
    AE_L32X2_XP(x1, px0, 8 * stride);
    x0 = AE_L32X2_X((ae_int32x2*)x, 0);

    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
    AE_L32X2_XP(tw3, ptwd, (3 * tw_step - 2) * sizeof(ae_int32x2));

    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);
    x3 = AE_SRAA32RS(x3, shift);
    
    DFT4X1(x0, x1, x2, x3);

    _cmult32x32(&x1, &x1, &tw1);
    _cmult32x32(&x2, &x2, &tw2);
    _cmult32x32(&x3, &x3, &tw3);

    AE_S32X2_IP(x0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x1, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x2, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x3, py0, sizeof(ae_int32x2));

    x0 = AE_MAXABS32S(x0, x1);
    x1 = AE_MAXABS32S(x2, x3);
    vmax = AE_MAX32(vmax, x0);
    vmax = AE_MAX32(vmax, x1);

  }
  px0 = (ae_int32x2*)(8 * (N - 1 - 3 * stride) + (uintptr_t)x);

  /* hifi3z: 14 cycles per pipeline stage in steady state with unroll=1*/
  for (i = 1; i < stride; i++)
  {
    ae_int32x2 x0, x1, x2, x3;
    ae_int32x2 tw1, tw2, tw3;

    AE_L32X2_XP(x3, px0, 8 * stride);
    AE_L32X2_XP(x2, px0, 8 * stride);
    AE_L32X2_XP(x1, px0, 8 * stride);
    AE_L32X2_XP(x0, px0, (-3 * stride - 1) * 8);

    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
    AE_L32X2_XP(tw3, ptwd, (3 * tw_step - 2) * sizeof(ae_int32x2));

    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);
    x3 = AE_SRAA32RS(x3, shift);

    DFT4X1(x0, x1, x2, x3);

    _cmult32x32(&x1, &x1, &tw1);
    _cmult32x32(&x2, &x2, &tw2);
    _cmult32x32(&x3, &x3, &tw3);

    AE_S32X2_IP(x0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x1, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x2, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x3, py0, sizeof(ae_int32x2));

    x0 = AE_MAXABS32S(x0, x1);
    x1 = AE_MAXABS32S(x2, x3);
    vmax = AE_MAX32(vmax, x0);
    vmax = AE_MAX32(vmax, x1);

  }
  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }
  *v = v[0] * R;
  return shift;

} /* ifft_stageS2_DFT4_first_32x32() */
int ifft_stageS2_DFT5_first_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{ 
  int shift, min_shift;
  const int R = 5; // stage radix
  ae_int32x2 * restrict px0;
  ae_int32x2 * restrict py0;
  const ae_int32x2 * restrict ptwd;
  const int stride = N / R;
  NASSERT_ALIGN8(x);
  NASSERT_ALIGN8(y);
  int i;
  ae_int32x2 vmax;

  px0 = (ae_int32x2*)(8 * (N - 4 * stride) + (uintptr_t)x);
  py0 = (ae_int32x2 *)(y);

  ptwd = (const ae_int32x2 *)tw;
  min_shift = 4;
  shift = min_shift - *bexp;
  ASSERT(shift>-32 && shift<32);
  vmax = AE_MOVI(0);

  /*
  ifft(x) = fft( [ x(1), x(end:-1:2)] )
  */
  {
    /* First butterfly radix 5 */
    ae_int32x2 x0, x1, x2, x3, x4;
    ae_int32x2 tw1, tw2, tw3, tw4;

    AE_L32X2_XP(x4, px0, 8 * stride);
    AE_L32X2_XP(x3, px0, 8 * stride);
    AE_L32X2_XP(x2, px0, 8 * stride);
    AE_L32X2_XP(x1, px0, 8 * stride);
    x0 = AE_L32X2_X((ae_int32x2*)x, 0);

    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw3, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw4, ptwd, sizeof(ae_int32x2));
    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);
    x3 = AE_SRAA32RS(x3, shift);
    x4 = AE_SRAA32RS(x4, shift);

    DFT5X1(x0, x1, x2, x3, x4);
    _cmult32x32(&x1, &x1, &tw1);
    _cmult32x32(&x2, &x2, &tw2);
    _cmult32x32(&x3, &x3, &tw3);
    _cmult32x32(&x4, &x4, &tw4);

    AE_S32X2_IP(x0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x1, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x2, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x3, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x4, py0, sizeof(ae_int32x2));

    x0 = AE_MAXABS32S(x0, x1);
    x1 = AE_MAXABS32S(x2, x3);
    vmax = AE_MAXABS32S(vmax, x4);
    vmax = AE_MAX32(vmax, x0);
    vmax = AE_MAX32(vmax, x1);
  }
  px0 = (ae_int32x2*)(8 * (N - 1 - 4 * stride) + (uintptr_t)x);


  /* hifi3z: 31 cycles per stage, unroll=1*/
  for (i = 1; i <stride; i++)
  {
    ae_int32x2 x0, x1, x2, x3, x4;
    ae_int32x2 tw1, tw2, tw3, tw4;

    AE_L32X2_XP(x4, px0, 8 * stride);
    AE_L32X2_XP(x3, px0, 8 * stride);
    AE_L32X2_XP(x2, px0, 8 * stride);
    AE_L32X2_XP(x1, px0, 8 * stride);
    AE_L32X2_XP(x0, px0, (-4 * stride - 1) * 8);

    AE_L32X2_IP(tw1, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw2, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw3, ptwd, sizeof(ae_int32x2));
    AE_L32X2_IP(tw4, ptwd, sizeof(ae_int32x2));
    x0 = AE_SRAA32RS(x0, shift);
    x1 = AE_SRAA32RS(x1, shift);
    x2 = AE_SRAA32RS(x2, shift);
    x3 = AE_SRAA32RS(x3, shift);
    x4 = AE_SRAA32RS(x4, shift);

    DFT5X1(x0, x1, x2, x3, x4);
    _cmult32x32(&x1, &x1, &tw1);
    _cmult32x32(&x2, &x2, &tw2);
    _cmult32x32(&x3, &x3, &tw3);
    _cmult32x32(&x4, &x4, &tw4);

    AE_S32X2_IP(x0, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x1, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x2, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x3, py0, sizeof(ae_int32x2));
    AE_S32X2_IP(x4, py0, sizeof(ae_int32x2));

    x0 = AE_MAXABS32S(x0, x1);
    x1 = AE_MAXABS32S(x2, x3);
    vmax = AE_MAXABS32S(vmax, x4);
    vmax = AE_MAX32(vmax, x0);
    vmax = AE_MAX32(vmax, x1);
  }
  {
    ae_int32x2 tmp;
    ae_int64   vL;
    int exp;
    tmp = AE_SEL32_LH(vmax, vmax);
    vmax = AE_OR32(tmp, vmax);
    vL = AE_MOVINT64_FROMINT32X2(vmax);
    exp = AE_NSA64(vL);
    exp = XT_MIN(31, exp);
    *bexp = (exp);
  }
  *v = v[0] * R;
  return shift;
} /* ifft_stageS2_DFT5_first_32x32() */



int ifft_stageS2_DFT2_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp){ return 0; }
int ifft_stageS2_DFT3_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp){ return 0; }
int ifft_stageS2_DFT4_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp)
{ 
  return 0;
}
int ifft_stageS2_DFT5_32x32(const int32_t *tw, const int32_t *x, int32_t *y, int N, int *v, int tw_step, int *bexp){ return 0; }


