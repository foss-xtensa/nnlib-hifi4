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
#include "xa_nnlib_common.h"
#include "../../../ndsp/hifi4/include/NatureDSP_Signal_math.h"
/*-------------------------------------------------------------------------
  Sigmoid
  The functions compute the sigmoid of input argument. 32-bit fixed-point
  functions accept inputs in Q6.25 and form outputs in Q0.15 format.

  Precision:
  32x16  32-bit inputs, 16-bit output. Accuracy: 2 LSB.

  Input:
  x[N]   input data, Q6.25
  N      length of vectors
  Output:
  y[N]   result, Q0.15

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  return result, Q0.15
-------------------------------------------------------------------------*/
WORD32 xa_nn_vec_sigmoid_32_16(
    WORD16       * __restrict__ y,             /* result, Q0.15 */
    const WORD32 * __restrict__ x,             /* input data, Q6.25 */
    WORD32       N)                            /* length of vectors */
{
    /*
    Reference Matlab code:

    % sigmoid function x in Q6.25, y in Q16.15
    function y=sigmoid_32x16(x)
        % convert Q25 -> Q23 and scale by ln(2)
        % x=round(double(x)*(0.25/log(2)));
        x=round(pow2(double(x)*774541002,-31));
        s=x<0;
        x=abs(x);
        % compute 2^-x
        n=bitshift(x,-23);
        mantQ23=bitand(x,8388607);
        % polynomial for 2^-x, for x=0...1, coeffients in Q23
        polypow2=[57364 -446161 2008107 -5813551 8388608];
        y=polypow2(1);
        y=round(pow2(y.*mantQ23,-23))+polypow2(2);
        y=round(pow2(y.*mantQ23,-23))+polypow2(3);
        y=round(pow2(y.*mantQ23,-23))+polypow2(4);
        y=round(pow2(y.*mantQ23,-23))+polypow2(5);
        x=bitshift(y,-n);

        % iterations to compute 1./(1+x) in Q23
        y=8053064-bitshift(x,-1);  % first approximation 0.96-x/2
        d=8388608-y-round(pow2(y.*x,-23));
        y=y+round(pow2(y.*d,-23));
        d=8388608-y-round(pow2(y.*x,-23));
        y=y+round(pow2(y.*d,-23));

        % apply sign
        y(s)=8388608-y(s);
        % scale to Q15 with rounding
        y=bitshift(y+128,-8);
    */

    XA_NNLIB_ARG_CHK_PTR(y, -1);
    XA_NNLIB_ARG_CHK_PTR(x, -1);
    XA_NNLIB_ARG_CHK_ALIGN(y, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(x, sizeof(WORD32), -1);

    static const int32_t polypow2[] = { 14685184, -114217216 , 514075392, -1488269056, 2147483647, 2061584302 };// coefficients in q31 format
    int n;
    ae_int32x2 X, X0, X1, E, Y, Z, D;
    ae_f32x2 t;
    ae_int16x4 Y_16;
    xtbool2 sign;
    const ae_int32x2 * restrict pX = (const ae_int32x2 *)x;
    const ae_int32x2 * restrict pX1 = (const ae_int32x2 *)x;
          // ae_int32x2 * restrict pY = (      ae_int32x2 *)y;
          ae_int16 * restrict pY = (ae_int16 *)y;
    ae_valign aX, aX1; //, aY;

    NASSERT(x);
    NASSERT(y);
    if (N <= 0) return -1;

    // aY = AE_ZALIGN64();
    aX = AE_LA64_PP(pX);
    aX1 = AE_LA64_PP(pX1);
    for (n = 0; n < (N >> 1); n++)
    {
        AE_LA32X2_IP(X, aX, pX);

        Z = AE_MULFP32X2RAS(X, AE_MOVDA32X2(774541002, 774541002));
        X = AE_ABS32S(Z);

        E = AE_SRAI32(X, 23);
        X = AE_AND32(X, AE_MOVDA32X2(0x007fffff, 0x007fffff));
        X = AE_SLAI32S(X, 8);

        Y = AE_L32_I((const ae_int32 *)polypow2, 4 * 0);
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 1); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 2); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 3); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 4); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        X0 = AE_SRAA32RS(Y, AE_MOVAD32_H(E));
        X1 = AE_SRAA32RS(Y, AE_MOVAD32_L(E));
        X = AE_SEL32_HL(X0, X1);

        Z = AE_SUB32(2061584302, AE_SRAI32(X, 1));
        t = AE_SUB32(2147483647, Z);AE_MULSFP32X2RAS(t, Z, X);D = t;
        AE_MULAFP32X2RAS(Z, Z, D);
        t = AE_SUB32(2147483647, Z);AE_MULSFP32X2RAS(t, Z, X);D = t;
        AE_MULAFP32X2RAS(Z, Z, D);

        Z = AE_SRAA32RS(Z, 16);

        AE_LA32X2_IP(X, aX1, pX1);
        sign = AE_LT32(X, 0);

        Y = AE_SUB32(32768, Z);
        AE_MOVT32X2(Z, Y, sign);

        // AE_SA32X2_IP(Z, aY, pY);
        Y_16 = AE_SAT16X4(Z, Z);
        *pY++ = AE_SEL16_6543(Y_16, Y_16);
        *pY++ = Y_16;
    }
    // AE_SA64POS_FP(aY, pY);

    if (N & 1)
    {
        X = AE_L32_I((const ae_int32 *)pX, 0);
        sign = AE_LT32(X, 0);

        Z = AE_MULFP32X2RAS(X, AE_MOVDA32X2(774541002, 774541002));
        X = AE_ABS32S(Z);

        E = AE_SRAI32(X, 23);
        X = AE_AND32(X, AE_MOVDA32X2(0x007fffff, 0x007fffff));
        X = AE_SLAI32S(X, 8);

        Y = AE_L32_I((const ae_int32 *)polypow2, 4 * 0);
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 1); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 2); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 3); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 4); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        X = AE_SRAA32RS(Y, AE_MOVAD32_H(E));

        Z = AE_SUB32(2061584302, AE_SRAI32(X, 1));
        t = AE_SUB32(2147483647, Z);AE_MULSFP32X2RAS(t, Z, X);D = t;
        AE_MULAFP32X2RAS(Z, Z, D);
        t = AE_SUB32(2147483647, Z);AE_MULSFP32X2RAS(t, Z, X);D = t;
        AE_MULAFP32X2RAS(Z, Z, D);
        Z = AE_SRAA32RS(Z, 16);

        Y = AE_SUB32(32768, Z);
        AE_MOVT32X2(Z, Y, sign);

        // AE_S32_L_I(Z, (ae_int32 *)pY, 0);
        Y_16 = AE_SAT16X4(Z, Z);
        *pY = Y_16;
    }

    return 0;
} /* xa_nn_vec_sigmoid_32_16() */

/*-------------------------------------------------------------------------
  Hyperbolic Tangent
  The functions compute the hyperbolic tangent of input argument. 32-bit
  fixed-point functions accept inputs in Q6.25 and form outputs in Q0.15
  format.

  Precision:
  32x16  32-bit inputs, 16-bit output. Accuracy: 2 LSB.

  Input:
  x[N]   input data, Q6.25
  N      length of vectors
  Output:
  y[N]   result, Q0.15

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  return result, Q0.15
-------------------------------------------------------------------------*/
WORD32 xa_nn_vec_tanh_32_16(
    WORD16       * __restrict__ y,             /* result, Q0.15 */
    const WORD32 * __restrict__ x,             /* input data, Q6.25 */
    WORD32       N)                            /* length of vectors */
{
    /*
    Reference Matlab code:
        function y=tanh_32x16(x)
        % convert Q25 -> Q23 and scale by ln(2)
        x=round(pow2(double(x)*774541002*2,-31));
        s=x<0;
        x=abs(x);
        % compute 2^-x
        n=bitshift(x,-23);
        mantQ23=bitand(x,8388607);
        % polynomial for 2^-x, for x=0...1, coeffients in Q23
        polypow2=[57364 -446161 2008107 -5813551 8388608];
        y=polypow2(1);
        y=round(pow2(y.*mantQ23,-23))+polypow2(2);
        y=round(pow2(y.*mantQ23,-23))+polypow2(3);
        y=round(pow2(y.*mantQ23,-23))+polypow2(4);
        y=round(pow2(y.*mantQ23,-23))+polypow2(5);
        x=bitshift(y,-n);

        % iterations to compute 1./(1+x) in Q23
        y=8053064-bitshift(x,-1);  % first approximation 0.96-x/2
        d=8388608-y-round(pow2(y.*x,-23));
        y=y+round(pow2(y.*d,-23));
        d=8388608-y-round(pow2(y.*x,-23));
        y=y+round(pow2(y.*d,-23));
        % scale by (1-x)
        y=round(pow2(y.*(8388608-x),-23));
        % scale to Q15 with rounding
        y=bitshift(y+128,-8);

        % apply sign
        y(s)=-y(s);
    */
    static const int32_t polypow2[] = { 14685184, -114217216 , 514075392, -1488269056, 2147483647 };// coefficients in q31 format
    int n;
    ae_int32x2 X, X0, X1, E, Y, Z, D;
    ae_f32x2 t;
    ae_int16x4 Y_16;
    xtbool2 sign;
    const ae_int32x2 * restrict pX = (const ae_int32x2 *)x;
    const ae_int32x2 * restrict pX1 = (const ae_int32x2 *)x;
          // ae_int32x2 * restrict pY = (      ae_int32x2 *)y;
          ae_int16   * restrict pY = (      ae_int16 *)y;
    ae_valign aX, aX1; //, aY;

    NASSERT(x);
    NASSERT(y);
    if (N <= 0) return -1;

    // aY = AE_ZALIGN64();
    aX = AE_LA64_PP(pX);
    aX1 = AE_LA64_PP(pX1);
    for (n = 0; n < (N >> 1); n++)
    {
        AE_LA32X2_IP(X, aX, pX);

        Z = AE_MULFP32X2RAS(X, AE_MOVDA32X2(1549082005, 1549082005));
        X = AE_ABS32S(Z);

        E = AE_SRAI32(X, 23);
        X = AE_AND32(X, AE_MOVDA32X2(0x007fffff, 0x007fffff));
        X = AE_SLAI32S(X, 8);

        Y = AE_L32_I((const ae_int32 *)polypow2, 4 * 0);
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 1); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 2); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 3); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 4); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        X0 = AE_SRAA32RS(Y, AE_MOVAD32_H(E));
        X1 = AE_SRAA32RS(Y, AE_MOVAD32_L(E));
        X = AE_SEL32_HL(X0, X1);

        Z = AE_SUB32(2061584302, AE_SRAI32(X, 1));
        t = AE_SUB32(2147483647, Z);AE_MULSFP32X2RAS(t, Z, X);D = t;
        AE_MULAFP32X2RAS(Z, Z, D);
        t = AE_SUB32(2147483647, Z);AE_MULSFP32X2RAS(t, Z, X);D = t;
        AE_MULAFP32X2RAS(Z, Z, D);
        Y = AE_SUB32(2147483647, X);
        Z = AE_MULFP32X2RAS(Z, Y);

        AE_LA32X2_IP(X, aX1, pX1);
        sign = AE_LT32(X, 0);
        X = AE_NEG32S(Z);
        AE_MOVT32X2(Z, X, sign);

        // AE_SA32X2_IP(Z, aY, pY);
        Y_16 = AE_ROUND16X4F32SASYM(Z, Z);
        *pY++ = AE_SEL16_6543(Y_16, Y_16);
        *pY++ = Y_16;
    }
    //AE_SA64POS_FP(aY, pY);

    if (N & 1)
    {
        X = AE_L32_I((const ae_int32 *)pX, 0);
        sign = AE_LT32(X, 0);

        Z = AE_MULFP32X2RAS(X, AE_MOVDA32X2(1549082005, 1549082005));
        X = AE_ABS32S(Z);

        E = AE_SRAI32(X, 23);
        X = AE_AND32(X, AE_MOVDA32X2(0x007fffff, 0x007fffff));
        X = AE_SLAI32S(X, 8);

        Y = AE_L32_I((const ae_int32 *)polypow2, 4 * 0);
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 1); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 2); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 3); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        t = AE_L32_I((const ae_int32 *)polypow2, 4 * 4); AE_MULAFP32X2RAS(t, X, Y); Y = t;
        X = AE_SRAA32RS(Y, AE_MOVAD32_H(E));

        Z = AE_SUB32(2061584302, AE_SRAI32(X, 1));
        t = AE_SUB32(2147483647, Z);AE_MULSFP32X2RAS(t, Z, X);D = t;
        AE_MULAFP32X2RAS(Z, Z, D);
        t = AE_SUB32(2147483647, Z);AE_MULSFP32X2RAS(t, Z, X);D = t;
        AE_MULAFP32X2RAS(Z, Z, D);
        Y = AE_SUB32(2147483647, X);
        Z = AE_MULFP32X2RAS(Z, Y);

        X = AE_NEG32S(Z);
        AE_MOVT32X2(Z, X, sign);

        // AE_S32_L_I(Z, (ae_int32 *)pY, 0);
        Y_16 = AE_ROUND16X4F32SASYM(Z, Z);
        *pY = Y_16;
    }

    return 0;
} /* xa_nn_vec_tanh_32_16() */

