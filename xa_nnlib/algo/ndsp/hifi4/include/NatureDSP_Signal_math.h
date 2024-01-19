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
#ifndef __NATUREDSP_SIGNAL_MATH_H__
#define __NATUREDSP_SIGNAL_MATH_H__

#include "NatureDSP_types.h"

#ifdef __cplusplus
extern "C" {
#endif


/*===========================================================================
  Mathematics:
  vec_recip            Reciprocal on Q31/Q15 Numbers
  vec_divide           Division
  vec_log              Logarithm
  vec_antilog          Antilogarithm
  vec_pow              Power function
  vec_sqrt             Square Root
  vec_rsqrt	           Reciprocal Square Root
  vec_sine,vec_cosine  Sine/Cosine
  vec_tan              Tangent
  vec_atan             Arctangent
  vec_atan2            Full Quadrant Arctangent
  vec_tanh             Hyperbolic Tangent
  vec_sigmoid          Sigmoid
  vec_softmax          Softmax
  vec_int2float        Integer to Float Conversion
  vec_float2int        Float to Integer Conversion
===========================================================================*/

/*-------------------------------------------------------------------------
  Reciprocal on Q63/Q31/Q15 Numbers
  These routines return the fractional and exponential portion of the
  reciprocal of a vector x of Q31 or Q15 numbers. Since the reciprocal is
  always greater than 1, it returns fractional portion frac in Q(31-exp)
  or Q(15-exp) format and exponent exp so true reciprocal value in the
  Q0.31/Q0.15 may be found by shifting fractional part left by exponent
  value.

  Mantissa accuracy is 1 LSB, so relative accuracy is:
  vec_recip16x16, scl_recip16x16                   6.2e-5
  scl_recip32x32                                   2.4e-7
  vec_recip32x32                                   9.2e-10
  vec_recip64x64                                   2.2e-19

  Precision:
  64x64  64-bit input, 64-bit output.
  32x32  32-bit input, 32-bit output.
  16x16  16-bit input, 16-bit output.

  Input:
  x[N]    input data, Q63, Q31 or Q15
  N       length of vectors

  Output:
  frac[N] fractional part of result, Q(63-exp), Q(31-exp) or Q(15-exp)
  exp[N]  exponent of result

  Restriction:
  x,frac,exp should not overlap

  PERFORMANCE NOTE:
  for optimum performance follow rules:
  frac,x - aligned on 8-byte boundary
  N      - multiple of 4 and >4

  Scalar versions:
  ----------------
  Return packed value:
  scl_recip64x64():
  bits 55:0 fractional part
  bits 63:56 exponent
  scl_recip32x32():
  bits 23:0 fractional part
  bits 31:24 exponent
  scl_recip16x16():
  bits 15:0 fractional part
  bits 31:16 exponent
-------------------------------------------------------------------------*/
void vec_recip64x64 (int64_t *  frac, int16_t *exp, const int64_t * x, int N);
void vec_recip32x32 (int32_t *  frac, int16_t *exp, const int32_t * x, int N);
void vec_recip16x16 (int16_t *  frac, int16_t *exp, const int16_t * x, int N);
uint64_t scl_recip64x64 (int64_t x);
uint32_t scl_recip32x32 (int32_t x);
uint32_t scl_recip16x16 (int16_t x);

/*-------------------------------------------------------------------------
  Division
  These routines perform pair-wise division of vectors written in Q63, Q31 or
  Q15 format. They return the fractional and exponential portion of the division
  result. Since the division may generate result greater than 1, it returns
  fractional portion frac in Q(63-exp), Q(31-exp) or Q(15-exp) format and
  exponent exp so true division result in the Q0.31 may be found by shifting
  fractional part left by exponent value.
  Additional routine makes integer division of 64-bit number to 32-bit
  denominator forming 32-bit result. If result is overflown, 0x7fffffff
  or 0x80000000 is returned depending on the signs of inputs.
  For division to 0, the result is not defined.

  Two versions of routines are available: regular versions (vec_divide64x32i,
  vec_divide64x64, vec_divide32x32, vec_divide16x16) work
  with arbitrary arguments, faster versions (vec_divide32x32_fast,
  vec_divide16x16_fast) apply some restrictions.

  Accuracy is measured as accuracy of fractional part (mantissa):
  vec_divide64x32i, scl_divide64x32                      :  1 LSB
  vec_divide64x64                                        :  2 LSB
  vec_divide32x32, vec_divide32x32_fast                  :  2 LSB (1.8e-9)
  scl_divide32x32                                        :  2 LSB (4.8e-7)
  vec_divide16x16, scl_divide16x16, vec_divide16x16_fast :  2 LSB (1.2e-4)

  Precision:
  64x32i integer division, 64-bit nominator, 32-bit denominator, 32-bit output.
  64x64  fractional division, 64-bit inputs, 64-bit output.
  32x32  fractional division, 32-bit inputs, 32-bit output.
  16x16  fractional division, 16-bit inputs, 16-bit output.

  Input:
  x[N]    nominator, 64-bit integer, Q63, Q31 or Q15
  y[N]    denominator, 32-bit integer, Q63, Q31 or Q15
  N       length of vectors
  Output:
  frac[N] fractional parts of result, Q(63-exp), Q(31-exp) or Q(15-exp)
  exp[N]  exponents of result

  Restriction:
  For regular versions (vec_divide64x32i, vec_divide64x64, vec_divide32x32,
  vec_divide16x16) :
  x,y,frac,exp should not overlap

  For faster versions (vec_divide32x32_fast, vec_divide16x16_fast) :
  x,y,frac,exp  should not overlap
  x,y,frac      to be aligned by 8-byte boundary, N - multiple of 4.

  Scalar versions:
  ----------------
  scl_divide64x32(): integer remainder
  Return packed value:
  scl_divide64x64():
  bits 55:0 fractional part
  bits 63:56 exponent
  scl_divide32x32():
  bits 23:0 fractional part
  bits 31:24 exponent
  scl_divide16x16():
  bits 15:0 fractional part
  bits 31:16 exponent
-------------------------------------------------------------------------*/
void vec_divide64x32i
                (int32_t * restrict frac,
                 const int64_t * restrict x,
                 const int32_t * restrict y, int N);
void vec_divide64x64
                (int64_t * restrict frac,
                 int16_t *exp,
                 const int64_t * restrict x,
                 const int64_t * restrict y, int N);
void vec_divide32x32
                (int32_t * restrict frac,
                 int16_t *exp,
                 const int32_t * restrict x,
                 const int32_t * restrict y, int N);
void vec_divide16x16
                (int16_t * restrict frac,
                 int16_t *exp,
                 const int16_t * restrict x,
                 const int16_t * restrict y, int N);
void vec_divide32x32_fast
                (int32_t * restrict frac,
                 int16_t * exp,
                 const int32_t * restrict x,
                 const int32_t * restrict y, int N);
void vec_divide16x16_fast
                (int16_t * restrict frac,
                 int16_t * exp,
                 const int16_t * restrict x,
                 const int16_t * restrict y, int N);

int32_t  scl_divide64x32(int64_t x,int32_t y);
uint64_t scl_divide64x64(int64_t x,int64_t y);
uint32_t scl_divide32x32(int32_t x,int32_t y);
uint32_t scl_divide16x16(int16_t x,int16_t y);

/*-------------------------------------------------------------------------
  Logarithm:
  Different kinds of logarithm (base 2, natural, base 10). Fixed point
  functions represent results in Q25 format or return 0x80000000 on negative
  of zero input.

  Precision:
  32x32  32-bit inputs, 32-bit outputs
  f      floating point

  Accuracy :
  vec_log2_32x32,scl_log2_32x32              730 (2.2e-5)
  vec_logn_32x32,scl_logn_32x32              510 (1.5e-5)
  vec_log10_32x32,scl_log10_32x32            230 (6.9e-6)
  floating point                             2 ULP

  NOTES:
  1.  Although 32 and 24 bit functions provide the same accuracy, 32-bit
      functions have better input/output resolution (dynamic range)
  2.  Scalar Floating point functions are compatible with standard ANSI C routines
      and set errno and exception flags accordingly.
  3.  Floating point functions limit the range of allowable input values:
      A) If x<0, the result is set to NaN. In addition, scalar floating point
         functions assign the value EDOM to errno and raise the "invalid"
         floating-point exception.
      B) If x==0, the result is set to minus infinity. Scalar floating  point
         functions assign the value ERANGE to errno and raise the "divide-by-zero"
         floating-point exception.

  Input:
  x[N]  input data, Q16.15 or floating point
  N     length of vectors
  Output:
  y[N]  result, Q25 or floating point

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  return result in Q25 or floating point
-------------------------------------------------------------------------*/
void vec_log2_32x32 (int32_t * restrict y,const int32_t * restrict x, int N);
void vec_logn_32x32 (int32_t * restrict y,const int32_t * restrict x, int N);
void vec_log10_32x32(int32_t * restrict y,const int32_t * restrict x, int N);
void vec_log2f     (float32_t * restrict y,const float32_t * restrict x, int N);
void xa_nnlib_vec_lognf     (float32_t * restrict y,const float32_t * restrict x, int N);
void vec_log10f    (float32_t * restrict y,const float32_t * restrict x, int N);
int32_t scl_log2_32x32 (int32_t x);
int32_t scl_logn_32x32 (int32_t x);
int32_t scl_log10_32x32(int32_t x);
float32_t scl_log2f (float32_t x);
float32_t scl_lognf (float32_t x);
float32_t scl_log10f(float32_t x);

/*-------------------------------------------------------------------------
  Antilogarithm
  These routines calculate antilogarithm (base2, natural and base10).
  Fixed-point functions accept inputs in Q25 and form outputs in Q16.15
  format and return 0x7FFFFFFF in case of overflow and 0 in case of
  underflow.

  Precision:
  32x32  32-bit inputs, 32-bit outputs. Accuracy: 8*e-6*y+1LSB
  f      floating point: Accuracy: 2 ULP
  NOTE:
  1.  Although 32 and 24 bit functions provide the similar accuracy, 32-bit
      functions have better input/output resolution (dynamic range).
  2.  Floating point functions are compatible with standard ANSI C routines
      and set errno and exception flags accordingly.

  Input:
  x[N]  input data,Q25 or floating point
  N     length of vectors
  Output:
  y[N]  output data,Q16.15 or floating point

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  fixed point functions return result in Q16.15

  PERFORMANCE NOTE:
  for optimum performance follow rules:
  x,y - aligned on 8-byte boundary
  N   - multiple of 2
-------------------------------------------------------------------------*/
void vec_antilog2_32x32 (int32_t * restrict y, const int32_t* restrict x, int N);
void vec_antilogn_32x32 (int32_t * restrict y, const int32_t* restrict x, int N);
void vec_antilog10_32x32(int32_t * restrict y, const int32_t* restrict x, int N);
void vec_antilog2f (float32_t * restrict y, const float32_t* restrict x, int N);
void xa_nnlib_vec_antilognf (float32_t * restrict y, const float32_t* restrict x, int N);
void vec_antilog10f(float32_t * restrict y, const float32_t* restrict x, int N);
int32_t scl_antilog2_32x32 (int32_t x);
int32_t scl_antilogn_32x32 (int32_t x);
int32_t scl_antilog10_32x32(int32_t x);
float32_t scl_antilog2f (float32_t x);
float32_t scl_antilognf (float32_t x);
float32_t scl_antilog10f(float32_t x);

/*-------------------------------------------------------------------------
  Power function
  This routine calculates power function for 32-bit fixed-point numbers.
  The  base is represented in Q31, the exponent is represented in Q6.25.
  Results are represented as normalized fixed point  number with separate
  mantissa in Q31 and exponent.
  NOTE: function returns 0 for negative or zero base

  Precision:
  32x32  32-bit inputs, 32-bit outputs. Accuracy: 2 ULP

  Input:
  x[N]  input data,Q0.31
  y[N]  input data,Q6.25
  N     length of vectors
  Output:
  m[N]  mantissa of output, Q31
  e[N]  exponent of output

  Restriction:
  x,y,m should not overlap
-------------------------------------------------------------------------*/
void vec_pow_32x32(int32_t *  m, int16_t *e,
                   const int32_t *  x, const int32_t*  y, int N);

/*-------------------------------------------------------------------------
  Square Root
  These routines calculate square root.
  NOTE: functions return 0x80000000 on negative argument for 32-bit outputs
  or 0x8000 for 16-bit outputs.
  Two versions of functions available: regular version (vec_sqrt16x16, vec_sqrt32x32,
  vec_sqrt32x16, vec_sqrt64x32) with arbitrary
  arguments and faster version (vec_sqrt32x32_fast) that
  apply some restrictions.

  Precision:
  16x16  16-bit inputs, 16-bit output. Accuracy: 2LSB
  32x32  32-bit inputs, 32-bit output. Accuracy: (2.6e-7*y+1LSB)
  32x16  32-bit input, 16-bit output.  Accuracy: 2 LSB
  64x32  64-bit inputs, 32-bit output. Accuracy: 2LSB

  Input:
  x[N]  input data, Q15, Q31, Q63
  N     length of vectors
  Output:
  y[N]  output data, Q15, Q31

  Restriction:
  Regular versions (vec_sqrt16x16, vec_sqrt32x32, vec_sqrt32x16, vec_sqrt64x32):
  x,y - should not overlap

  Faster versions (vec_sqrt32x32_fast):
  x,y - should not overlap
  x,y - aligned on 8-byte boundary
  N   - multiple of 2

  Scalar versions:
  ----------------
  return result, Q15, Q31
-------------------------------------------------------------------------*/
void vec_sqrt16x16      (int16_t* restrict y, const int16_t* restrict x, int N);
void vec_sqrt32x32      (int32_t* restrict y, const int32_t* restrict x, int N);
void vec_sqrt32x16      (int16_t* restrict y, const int32_t* restrict x, int N);
void vec_sqrt32x32_fast (int32_t* restrict y, const int32_t* restrict x, int N);
void vec_sqrt64x32      (int32_t* restrict y, const int64_t* restrict x, int N);
int16_t scl_sqrt16x16(int16_t x);
int16_t scl_sqrt32x16(int32_t x);
int32_t scl_sqrt32x32(int32_t x);
int32_t scl_sqrt64x32(int64_t x);

/*-------------------------------------------------------------------------
  Reciprocal Square Root
  These routines return the fractional and exponential portion of the
  reciprocal square root of a vector x of Q31 or Q15 numbers. Since the
  reciprocal square root is always greater than 1, they return fractional
  portion frac in Q(31-exp) or Q(15-exp) format and exponent exp so true
  reciprocal value in the Q0.31/Q0.15 may be found by shifting fractional
  part left by exponent value.

  Mantissa accuracy is 1 LSB, so relative accuracy is:
  vec_rsqrt16x16, scl_rsqrt16x16	6.2e-5
  scl_rsqrt32x32	                2.4e-7
  vec_rsqrt32x32	                9.2e-10

  Precision:
  16x16  16-bit inputs, 16-bit output. Accuracy: 2LSB
  32x32  32-bit inputs, 32-bit output. Accuracy: (2.6e-7*y+1LSB)

  Input:
  x[N]     input data, Q15, Q31
  N        length of vectors
  Output:
  frac[N]  fractional part of result, Q(31-exp) or Q(15-exp)
  exp[N]   exponent of result

  Restriction:
  x, fract, exp - should not overlap

  Scalar versions:
  ----------------
  Returned packed value:
  scl_rsqrt32x32():
  bits 23�0 fractional part
  bits 31�24 exponent
  scl_rsqrt16x16():
  bits 15�0 fractional part
  bits 31�16 exponent

-------------------------------------------------------------------------*/
void vec_rsqrt32x32 ( int32_t * frac, int16_t * exp, const int32_t * x, int N);
void vec_rsqrt16x16 ( int16_t * frac, int16_t * exp, const int16_t * x, int N);
uint32_t scl_rsqrt32x32(int32_t x);
uint32_t scl_rsqrt16x16(int16_t x);

/*-------------------------------------------------------------------------
  Sine/Cosine
  Fixed-point functions calculate sin(pi*x) or cos(pi*x) for numbers written
  in Q31 or Q15 format. Return results in the same format.
  Floating point functions compute sin(x) or cos(x)
  Two versions of functions available: regular version (vec_sine32x32,
  vec_cosine32x32, , xa_nnlib_vec_sinef, xa_nnlib_vec_cosinef)
  with arbitrary arguments and faster version (vec_sine32x32_fast,
  vec_cosine32x32_fast) that apply some restrictions.
  NOTE:
  1.  Scalar floating point functions are compatible with standard ANSI C
      routines and set errno and exception flags accordingly
  2.  Floating point functions limit the range of allowable input values:
      [-102940.0, 102940.0] Whenever the input value does not belong to this
      range, the result is set to NaN.

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 1700 (7.9e-7)
  f      floating point. Accuracy 2 ULP

  Input:
  x[N]  input data,Q31 or floating point
  N     length of vectors
  Output:
  y[N]  output data,Q31 or floating point

  Restriction:
  Regular versions (vec_sine32x32, vec_cosine32x32, xa_nnlib_vec_sinef,
  xa_nnlib_vec_cosinef):
  x,y - should not overlap

  Faster versions (vec_sine32x32_fast, vec_cosine32x32_fast):
  x,y - should not overlap
  x,y - aligned on 8-byte boundary
  N   - multiple of 2

  Scalar versions:
  ----------------
  return result in Q31 or floating point
-------------------------------------------------------------------------*/
void vec_sine32x32        (int32_t * restrict y, const int32_t * restrict x, int N);
void vec_sine32x32_fast   (int32_t * restrict y, const int32_t * restrict x, int N);
void vec_cosine32x32      (int32_t * restrict y, const int32_t * restrict x, int N);
void vec_cosine32x32_fast (int32_t * restrict y, const int32_t * restrict x, int N);
void xa_nnlib_vec_sinef     ( float32_t * restrict y, const float32_t * restrict x, int N);
void xa_nnlib_vec_cosinef   ( float32_t * restrict y, const float32_t * restrict x, int N);
int32_t scl_sine32x32   (int32_t x);
int32_t scl_cosine32x32 (int32_t x);
float32_t scl_sinef   (float32_t x);
float32_t scl_cosinef (float32_t x);

/*-------------------------------------------------------------------------
  Tangent
  Fixed point functions calculate tan(pi*x) for number written in Q31.
  Floating point functions compute tan(x)

  Precision:
  32x32  32-bit inputs, 32-bit outputs. Accuracy: (1.3e-4*y+1LSB)
                                        if abs(y)<=464873(14.19 in Q15)
                                        or abs(x)<pi*0.4776
  f      floating point.                Accuracy: 2 ULP

  NOTE:
  1.  Scalar floating point function is compatible with standard ANSI C routines
      and set errno and exception flags accordingly
  2.  Floating point functions limit the range of allowable input values: [-9099, 9099]
      Whenever the input value does not belong to this range, the result is set to NaN.

  Input:
  x[N]   input data,Q31 or floating point
  N      length of vectors
  Output:
  y[N]   result, Q16.15 or floating point

  Restriction:
  x,y - should not overlap

  PERFORMANCE NOTE:
  for optimum performance follow rules:
  x,z - aligned on 8-byte boundary
  N   - multiple of 2

  Scalar versions:
  ----------------
  Return result, Q16.15 or floating point
-------------------------------------------------------------------------*/
void vec_tan32x32 (int32_t * restrict y, const int32_t * restrict x, int N);
void vec_tanf (float32_t * restrict y, const float32_t * restrict x, int N);
int32_t   scl_tan32x32 (int32_t x);
float32_t scl_tanf   (float32_t x);

/*-------------------------------------------------------------------------
  Arctangent
  Functions calculate arctangent of number. Fixed point functions
  scale output to pi so it is always in range -0x20000000 : 0x20000000
  which corresponds to the real phases +pi/4 and represent input and output
  in Q31
  NOTE:
  1.  Scalar floating point function is compatible with standard ANSI C
      routines and sets errno and exception flags accordingly

  Accuracy:
  24 bit version: 74000 (3.4e-5)
  32 bit version: 42    (2.0e-8)
  floating point: 2 ULP

  Precision:
  32x32  32-bit inputs, 32-bit output.
  f      floating point

  Input:
  x[N]   input data, Q31 or floating point
  N      length of vectors
  Output:
  z[N]   result, Q31 or floating point

  Restriction:
  x,z should not overlap

  PERFORMANCE NOTE:
  for optimum performance follow rules:
  x,z - aligned on 8-byte boundary
  N   - multiple of 2

  Scalar versions:
  ----------------
  return result, Q31 or floating point
-------------------------------------------------------------------------*/
void vec_atan32x32 (int32_t * restrict z, const int32_t * restrict x, int N);
void vec_atanf (float32_t * restrict z, const float32_t * restrict x, int N);
int32_t   scl_atan32x32 (int32_t x);
float32_t scl_atanf   (float32_t x);

/*-------------------------------------------------------------------------
  Full-Quadrant Arc Tangent
  The functions compute the arc tangent of the ratios y[N]/x[N] and store the
  result to output vector z[N].
  Floating point functions output is in radians. Fixed point functions
  scale its output by pi.

  NOTE:
  1. Scalar floating point function is compatible with standard ANSI C routines and set
     errno and exception flags accordingly
  2. Scalar floating point function assigns EDOM to errno whenever y==0 and x==0.

  Accuracy:
  24 bit version: 768 (3.57e-7)
  floating point: 2 ULP

  Special cases:
       y    |   x   |  result   |  extra conditions
    --------|-------|-----------|---------------------
     +/-0   | -0    | +/-pi     |
     +/-0   | +0    | +/-0      |
     +/-0   |  x    | +/-pi     | x<0
     +/-0   |  x    | +/-0      | x>0
     y      | +/-0  | -pi/2     | y<0
     y      | +/-0  |  pi/2     | y>0
     +/-y   | -inf  | +/-pi     | finite y>0
     +/-y   | +inf  | +/-0      | finite y>0
     +/-inf | x     | +/-pi/2   | finite x
     +/-inf | -inf  | +/-3*pi/4 |
     +/-inf | +inf  | +/-pi/4   |

  Input:
    y[N]  vector of numerator values, Q31 or floating point
    x[N]  vector of denominator values, Q31 or floating point
    N     length of vectors
  Output:
    z[N]  results, Q31 or floating point

---------------------------------------------------------------------------*/
void vec_atan2f     (float32_t * z, const float32_t * y, const float32_t * x, int N);
float32_t scl_atan2f (float32_t y, float32_t x);

/*-------------------------------------------------------------------------
  Hyperbolic Tangent
  The functions compute the hyperbolic tangent of input argument. 32-bit
  fixed-point functions accept inputs in Q6.25 and form outputs in Q16.15
  format.

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB.
  f      floating point input, floating point output, Accuracy: 2 ULP
  Input:
  x[N]   input data, Q6.25 or floating point
  N      length of vectors
  Output:
  y[N]   result, Q16.15 or floating point

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  return result, Q16.15 or floating point
-------------------------------------------------------------------------*/
void xa_nnlib_vec_tanhf    (float32_t * y, const float32_t * x,int N);
int32_t scl_tanh32x32(int32_t x);
float32_t xa_nnlib_scl_tanhf  (float32_t x);
/*-------------------------------------------------------------------------
  Sigmoid
  The functions compute the sigmoid of input argument. 32-bit fixed-point
  functions accept inputs in Q6.25 and form outputs in Q16.15 format.

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB.
  f      floating point input, floating point output. Accuracy 2 ULP
  Input:
  x[N]   input data, Q6.25 or floating point
  N      length of vectors
  Output:
  y[N]   result, Q16.15 or floating point

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  return result, Q16.15 or floating point
-------------------------------------------------------------------------*/
void xa_nnlib_vec_sigmoidf    (float32_t * y, const float32_t * x, int N);
int32_t   scl_sigmoid32x32(int32_t x);
float32_t xa_nnlib_scl_sigmoidf    (float32_t x);
/*-------------------------------------------------------------------------
  Rectifier function
  The functions compute the rectifier linear unit function of input argument.
  32-bit fixed-point functions accept inputs in Q6.25 and form outputs in
  Q16.15 format. Parameter K allows to set upper threshold for proper
  compression of output signal.

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB.
  f      floating point input, floating point output. Accuracy 2 ULP
  Input:
  x[N]   input data, Q6.25 or floating point
  K      threshold, Q16.15 or floating point
  N      length of vectors
  Output:
  y[N]   result, Q16.15 or floating point

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  return result, Q16.15 or floating point
-------------------------------------------------------------------------*/

void xa_nnlib_vec_reluf     (float32_t * y, const float32_t * x, float32_t K, int N);
int32_t   scl_relu32x32 (int32_t   x, int32_t   K);
float32_t scl_reluf     (float32_t x, float32_t K);

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
void xa_nnlib_vec_softmaxf    (float32_t * y, const float32_t * x,int N);
/*-------------------------------------------------------------------------
  Integer to float conversion
  Routines convert integer to float and scale result up by 2^t.

  Precision:
  f     32-bit input, floating point output

  Input:
  x[N]  input data, integer
  t     scale factor
  N     length of vector
  Output:
  y[N]  conversion results, floating point

  Restriction:
  t should be in range -126...126
-------------------------------------------------------------------------*/
void   vec_int2float (float32_t * restrict y, const int32_t * restrict x, int t, int N);
float32_t scl_int2float (int32_t x, int t);

/*-------------------------------------------------------------------------
  Float to integer conversion
  routines scale floating point input down by 2^t and convert it to integer
  with saturation

  Precision:
  f     single precision floating point

  Input:
  x[N]  input data, floating point
  t     scale factor
  N     length of vector
  Output:
  y[N]  conversion results, integer

  Restriction:
  t should be in range -126...126
-------------------------------------------------------------------------*/
void   vec_float2int (int32_t * restrict y, const float32_t * restrict x, int t, int N);
int32_t scl_float2int (float32_t x, int t);

#ifdef __cplusplus
}
#endif

#endif/* __NATUREDSP_SIGNAL_MATH_H__ */
