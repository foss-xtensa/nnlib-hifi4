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

/* DSP Library API */
#include "../include/NatureDSP_Signal_math.h"
/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"
/* Tables */
#include "../include/expf_tbl.h"
/* sNaN/qNaN, single precision. */
#include "../include/nanf_tbl.h"

#if !HAVE_VFPU && !HAVE_FPU
DISCARD_FUN(void,xa_nnlib_vec_antilognf,( float32_t * restrict y, const float32_t* restrict x, int N ))
#elif HAVE_VFPU
/*===========================================================================
  Vector matematics:
  vec_antilog          Antilogarithm         
===========================================================================*/

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
void xa_nnlib_vec_antilognf( float32_t * restrict y, const float32_t* restrict x, int N )
{
  /*
    int32_t t, y;
    int e;
    int64_t a;

    if (isnan(x)) return x;
    if (x>xa_nnlib_expfminmax[1].f) x = xa_nnlib_expfminmax[1].f;
    if (x<xa_nnlib_expfminmax[0].f) x = xa_nnlib_expfminmax[0].f;

    / scale input to 1/ln(2) and convert to Q31 /
    x = frexpf(x, &e);

    t = (int32_t)STDLIB_MATH(ldexpf)(x, e + 24);
    a = ((int64_t)t*xa_nnlib_invln2_Q30) >> 22; / Q24*Q30->Q32 /
    t = ((uint32_t)a) >> 1;
    e = (int32_t)(a >> 32);
    / compute 2^t in Q30 where t is in Q31 /
    y = xa_nnlib_expftbl_Q30[0];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[1];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[2];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[3];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[4];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[5];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[6];
    / convert back to the floating point /
    x = STDLIB_MATH(ldexpf)((float32_t)y, e - 30);
  */

  const xtfloatx2 *          X0  = (xtfloatx2*)x;
        xtfloatx2 * restrict Y   = (xtfloatx2*)y;
  const ae_int32  * restrict TBL = (ae_int32 *)xa_nnlib_expftbl_Q30;

  ae_valign X0_va, Y_va;
  
  xtfloatx2 x0, x1, y0, y1;
  ae_int32x2 tb0, tb1, tb2, tb3, tb4, tb5, tb6;
  ae_int32x2 u0, e0, e1, n0;
  ae_int64 wh, wl;
  ae_f32x2 f0;
  xtbool2 b_nan;

  int n;

  if ( N<=0 ) return;

  X0_va = AE_LA64_PP(X0);
  Y_va = AE_ZALIGN64();

  for ( n=0; n<(N>>1); n++ )
  {
    XT_LASX2IP(x0, X0_va, X0);
    b_nan = XT_UN_SX2(x0, x0);

    /* scale input by 1/ln(2) and convert to Q31 */
    u0 = XT_TRUNC_SX2(x0, 24);
    wh = AE_MUL32_HH(u0, xa_nnlib_invln2_Q30);
    wl = AE_MUL32_LL(u0, xa_nnlib_invln2_Q30);
    e0 = AE_TRUNCA32X2F64S(wh, wl, -22);
    wh = AE_SLLI64(wh, 32-22);
    wl = AE_SLLI64(wl, 32-22);
    u0 = AE_TRUNCI32X2F64S(wh, wl, 0);
    u0 = AE_SRLI32(u0, 1);

    AE_L32_IP(tb0, TBL, sizeof(int32_t));
    AE_L32_IP(tb1, TBL, sizeof(int32_t));
    AE_L32_IP(tb2, TBL, sizeof(int32_t));
    AE_L32_IP(tb3, TBL, sizeof(int32_t));
    AE_L32_IP(tb4, TBL, sizeof(int32_t));
    AE_L32_IP(tb5, TBL, sizeof(int32_t));
    AE_L32_XP(tb6, TBL, -6*(int)sizeof(int32_t));

    n0 = tb0; f0 = tb1;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0; f0 = tb2;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0; f0 = tb3;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0; f0 = tb4;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0; f0 = tb5;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0; f0 = tb6;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0;

    x0 = XT_FLOAT_SX2(n0,30);

    e0 = AE_ADD32(e0, 254);
    e1 = AE_SRAI32(e0, 1);
    e0 = AE_SUB32(e0, e1);
    e0 = AE_SLAI32(e0, 23);
    e1 = AE_SLAI32(e1, 23);
    y0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(e0);
    y1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(e1);

    XT_MOVT_SX2(y1, xa_nnlib_qNaNf.f, b_nan);

    y0 = XT_MUL_SX2(y0, y1);
    y0 = XT_MUL_SX2(x0, y0);

    XT_SASX2IP(y0, Y_va, Y);
  }

  XT_SASX2POSFP(Y_va, Y);

  if ( N&1 )
  {
    x0 = XT_LSI( (xtfloat*)X0, 0 );

    b_nan = XT_UN_SX2(x0, x0);

    /* scale input by 1/ln(2) and convert to Q31 */
    u0 = XT_TRUNC_SX2(x0, 24);
    wh = AE_MUL32_HH(u0, xa_nnlib_invln2_Q30);
    wl = AE_MUL32_LL(u0, xa_nnlib_invln2_Q30);
    e0 = AE_TRUNCA32X2F64S(wh, wl, -22);
    wh = AE_SLLI64(wh, 32-22);
    wl = AE_SLLI64(wl, 32-22);
    u0 = AE_TRUNCI32X2F64S(wh, wl, 0);
    u0 = AE_SRLI32(u0, 1);

    tb0 = xa_nnlib_expftbl_Q30[0];
    tb1 = xa_nnlib_expftbl_Q30[1];
    tb2 = xa_nnlib_expftbl_Q30[2];
    tb3 = xa_nnlib_expftbl_Q30[3];
    tb4 = xa_nnlib_expftbl_Q30[4];
    tb5 = xa_nnlib_expftbl_Q30[5];
    tb6 = xa_nnlib_expftbl_Q30[6];

    n0 = tb0; f0 = tb1;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0; f0 = tb2;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0; f0 = tb3;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0; f0 = tb4;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0; f0 = tb5;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0; f0 = tb6;
    AE_MULAFP32X2RAS(f0, u0, n0); n0 = f0;

    x1 = XT_FLOAT_SX2(n0,30);

    e0 = AE_ADD32(e0, 254);
    e1 = AE_SRAI32(e0, 1);
    e0 = AE_SUB32(e0, e1);
    e0 = AE_SLAI32(e0, 23);
    e1 = AE_SLAI32(e1, 23);
    y0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(e0);
    y1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(e1);

    XT_MOVT_SX2(y1, x0, b_nan);

    y0 = XT_MUL_SX2(y0, y1);
    y0 = XT_MUL_SX2(x1, y0);

    XT_SSI( y0, (xtfloat*)Y, 0 );
  }

} /* xa_nnlib_vec_antilognf() */ 

#elif HAVE_FPU
#define sz_i32 (int)sizeof(int32_t)
#define sz_f32 (int)sizeof(float32_t)

/*===========================================================================
  Vector matematics:
  vec_antilog          Antilogarithm         
===========================================================================*/

/*-------------------------------------------------------------------------
  Antilogarithm
  These routines calculate antilogarithm (base2, natural and base10). 
  Fixed-point functions accept inputs in Q25 and form outputs in Q16.15 
  format and return 0x7FFFFFFF in case of overflow and 0 in case of 
  underflow.

  Precision:
  32x32  32-bit inputs, 32-bit outputs. Accuracy: 4*e-5*y+1LSB
  f      floating point: Accuracy: 2 ULP
  NOTE:
  1.  Floating point functions are compatible with standard ANSI C routines 
      and set errno and exception flags accordingly

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

-------------------------------------------------------------------------*/

void xa_nnlib_vec_antilognf( float32_t * restrict y, const float32_t* restrict x, int N )
{
  /*
    int32_t t, y;
    int e;
    int64_t a;

    if (isnan(x)) return x;
    if (x>xa_nnlib_expfminmax[1].f) x = xa_nnlib_expfminmax[1].f;
    if (x<xa_nnlib_expfminmax[0].f) x = xa_nnlib_expfminmax[0].f;

    / scale input to 1/ln(2) and convert to Q31 /
    x = frexpf(x, &e);

    t = (int32_t)STDLIB_MATH(ldexpf)(x, e + 24);
    a = ((int64_t)t*xa_nnlib_invln2_Q30) >> 22; / Q24*Q30->Q32 /
    t = ((uint32_t)a) >> 1;
    e = (int32_t)(a >> 32);
    / compute 2^t in Q30 where t is in Q31 /
    y = xa_nnlib_expftbl_Q30[0];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[1];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[2];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[3];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[4];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[5];
    y = satQ31((((int64_t)t*y) + (1LL << (31 - 1))) >> 31) + xa_nnlib_expftbl_Q30[6];
    / convert back to the floating point /
    x = STDLIB_MATH(ldexpf)((float32_t)y, e - 30);
  */

  const xtfloat  *          X0  = (xtfloat*)x;
  const xtfloat  *          X1  = (xtfloat*)x;
  const ae_int32 *          TBL = (ae_int32*)xa_nnlib_expftbl_Q30;
        xtfloat  * restrict Y   = (xtfloat*)y;

  xtfloat    x0, x1, x0_, x1_, y0, y1, z0, z1;
  ae_int32x2 tb0, tb1, tb2, tb3, tb4, tb5, tb6;
  int32_t    e0, e1, n0, n1, u0, u1;
  xtbool     b_nan0, b_nan1;
  ae_int64   w0, w1, r0, r1;

  int n;

  if ( N<=0 ) return;

  for ( n=0; n<(N>>1); n++ )
  {
    ae_f32x2   v01, f01, m01;
    ae_int32x2 e01, g01, u01, t01;

	XT_LSIP(x0, X0, sz_f32);
	XT_LSIP(x1, X0, sz_f32);

    /* scale input by 1/ln(2) and convert to Q31 */
    u0 = XT_TRUNC_S( XT_MUL_S( x0, XT_FLOAT_S( 1<<9, 0 ) ), 15 );
    u1 = XT_TRUNC_S( XT_MUL_S( x1, XT_FLOAT_S( 1<<9, 0 ) ), 15 );

	w0 = AE_MUL32_HH(u0, xa_nnlib_invln2_Q30);
	w1 = AE_MUL32_HH(u1, xa_nnlib_invln2_Q30);
    e0 = ae_int32x2_rtor_int32(AE_TRUNCA32X2F64S(w0, w0, -22));
    e1 = ae_int32x2_rtor_int32(AE_TRUNCA32X2F64S(w1, w1, -22));
    r0 = AE_SLLI64(w0, 32-22);
    r1 = AE_SLLI64(w1, 32-22);
    u0 = ae_int32x2_rtor_int32(AE_TRUNCI32X2F64S(r0, r0, 0));
    u1 = ae_int32x2_rtor_int32(AE_TRUNCI32X2F64S(r1, r1, 0));
    u01 = AE_MOVDA32X2(u1, u0);
	u01 = AE_SRLI32(u01, 1);

    tb0 = AE_L32_I( TBL, 0*sz_f32 );
    tb1 = AE_L32_I( TBL, 1*sz_f32 );
    tb2 = AE_L32_I( TBL, 2*sz_f32 );
    tb3 = AE_L32_I( TBL, 3*sz_f32 );
    tb4 = AE_L32_I( TBL, 4*sz_f32 );
    tb5 = AE_L32_I( TBL, 5*sz_f32 );
    tb6 = AE_L32_I( TBL, 6*sz_f32 );

	v01 = AE_MOVF32X2_FROMINT32X2(u01);

	m01 = tb0; f01 = tb1;
    AE_MULAFP32X2RAS(f01, v01, m01); m01 = f01; f01 = tb2;
    AE_MULAFP32X2RAS(f01, v01, m01); m01 = f01; f01 = tb3;
    AE_MULAFP32X2RAS(f01, v01, m01); m01 = f01; f01 = tb4;
    AE_MULAFP32X2RAS(f01, v01, m01); m01 = f01; f01 = tb5;
    AE_MULAFP32X2RAS(f01, v01, m01); m01 = f01; f01 = tb6;
    AE_MULAFP32X2RAS(f01, v01, m01); m01 = f01;
	
    n0 = AE_MOVAD32_L(m01);
    n1 = AE_MOVAD32_H(m01);

    x0 = XT_MUL_S( XT_FLOAT_S( n0, 15 ), XT_FLOAT_S( 1, 15 ) );
    x1 = XT_MUL_S( XT_FLOAT_S( n1, 15 ), XT_FLOAT_S( 1, 15 ) );

    e01 = AE_MOVDA32X2(e1, e0);
    e01 = AE_ADD32(e01, 254);
    g01 = AE_SRAI32(e01, 1);
    e01 = AE_SUB32(e01, g01);
    u01 = AE_SLLI32(e01, 23);
    t01 = AE_SLLI32(g01, 23);
    y0 = XT_WFR(AE_MOVAD32_L(u01));
    y1 = XT_WFR(AE_MOVAD32_H(u01));
    z0 = XT_WFR(AE_MOVAD32_L(t01));
    z1 = XT_WFR(AE_MOVAD32_H(t01));

	XT_LSIP(x0_, X1, sz_f32);
	XT_LSIP(x1_, X1, sz_f32);
	b_nan0 = XT_UN_S(x0_, x0_);
	b_nan1 = XT_UN_S(x1_, x1_);
    XT_MOVT_S(z0, xa_nnlib_qNaNf.f, b_nan0);
    XT_MOVT_S(z1, xa_nnlib_qNaNf.f, b_nan1);

    y0 = XT_MUL_S(y0, z0);
    y1 = XT_MUL_S(y1, z1);
    y0 = XT_MUL_S(x0, y0);
    y1 = XT_MUL_S(x1, y1);

	XT_SSIP(y0, Y, sz_f32);
	XT_SSIP(y1, Y, sz_f32);
  }
  if (N & 1)
  {
    ae_f32x2 f0;
    ae_f32   v0, m0;
    ae_int32 g0;

	XT_LSIP(x0, X0, sz_f32);
    b_nan0 = XT_UN_S(x0, x0);

    /* scale input by 1/ln(2) and convert to Q31 */
    u0 = XT_TRUNC_S( XT_MUL_S( x0, XT_FLOAT_S( 1<<9, 0 ) ), 15 );

	w0 = AE_MUL32_HH(u0, xa_nnlib_invln2_Q30);
    e0 = ae_int32x2_rtor_int32(AE_TRUNCA32X2F64S(w0, w0, -22));
    r0 = AE_SLLI64(w0, 32-22);
    u0 = ae_int32x2_rtor_int32(AE_TRUNCI32X2F64S(r0, r0, 0));
    u0 = XT_SRLI(u0, 1);

    tb0 = AE_L32_I(TBL, 0*sz_f32);
    tb1 = AE_L32_I(TBL, 1*sz_f32);
    tb2 = AE_L32_I(TBL, 2*sz_f32);
    tb3 = AE_L32_I(TBL, 3*sz_f32);
    tb4 = AE_L32_I(TBL, 4*sz_f32);
    tb5 = AE_L32_I(TBL, 5*sz_f32);
    tb6 = AE_L32_I(TBL, 6*sz_f32);

    v0 = u0;

    m0 = tb0; f0 = tb1;
    AE_MULAFP32X2RAS(f0, v0, m0); m0 = f0; f0 = tb2;
    AE_MULAFP32X2RAS(f0, v0, m0); m0 = f0; f0 = tb3;
    AE_MULAFP32X2RAS(f0, v0, m0); m0 = f0; f0 = tb4;
    AE_MULAFP32X2RAS(f0, v0, m0); m0 = f0; f0 = tb5;
    AE_MULAFP32X2RAS(f0, v0, m0); m0 = f0; f0 = tb6;
    AE_MULAFP32X2RAS(f0, v0, m0); m0 = f0;

    n0 = ae_f32_rtor_int32(m0);

    x0 = XT_MUL_S( XT_FLOAT_S( n0, 15 ), XT_FLOAT_S( 1, 15 ) );

    e0 = XT_ADD(e0, 254);
    g0 = XT_SRAI(e0, 1);
    e0 = XT_SUB(e0, g0);
    e0 = XT_SLLI(e0, 23);
    g0 = XT_SLLI(g0, 23);
    y0 = XT_WFR(e0);
    z0 = XT_WFR(g0);

	XT_LSIP(x0_, X1, sz_f32);
	b_nan0 = XT_UN_S(x0_, x0_);
    XT_MOVT_S(z0, xa_nnlib_qNaNf.f, b_nan0);

    y0 = XT_MUL_S(y0, z0);
    y0 = XT_MUL_S(x0, y0);

	XT_SSIP(y0, Y, sz_f32);
  }
} /* xa_nnlib_vec_antilognf() */ 
#endif
