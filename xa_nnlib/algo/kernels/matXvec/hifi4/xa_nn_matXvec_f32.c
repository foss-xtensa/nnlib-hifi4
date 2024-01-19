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
/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros.h"

#if HAVE_VFPU

#if HAVE_VFPU
/*#define ENABLE_PRAGMA*/

#define SZ_F32 (sizeof(FLOAT32))
WORD32 static dual_mtx_vecmpyf_bias_add( FLOAT32 * z,
     const FLOAT32 * x,  const FLOAT32 * y, const FLOAT32 * v, const FLOAT32 * w,
     const FLOAT32 * b, int rows, int cols1, int cols2, int row_stride1, int row_stride2 )
{
  const xtfloatx2 *restrict px0;
  const xtfloatx2 *restrict px1;
  const xtfloatx2 *restrict px2;
  const xtfloatx2 *restrict px3;
  const xtfloatx2 *restrict pv0;
  const xtfloatx2 *restrict pv1;
  const xtfloatx2 *restrict pv2;
  const xtfloatx2 *restrict pv3;
  const xtfloatx2 *restrict py;
  const xtfloatx2 *restrict pw;
  const xtfloatx2 *restrict pb;
        xtfloatx2 *restrict pz;
        xtfloat *restrict pz_;
  xtfloatx2 b0, b1;
  xtfloatx2 y0, y1;
  xtfloatx2 w0, w1;
  xtfloatx2 z0, z1;
  xtfloat z0_, b0_;
  xtfloatx2 x00, x01, x10, x11,
            x20, x21, x30, x31;
  xtfloatx2 v00, v01, v10, v11,
            v20, v21, v30, v31;
  xtfloatx2 acc00, acc01, acc10, acc11,
            acc20, acc21, acc30, acc31;
  int m, n, k;

  NASSERT(x);
  NASSERT(y);
  NASSERT(v);
  NASSERT(w);
  NASSERT(z);
//  NASSERT(b);
  NASSERT((z != x) && (z != y) && (z != v) && (z != w) && (z != b));
  NASSERT_ALIGN(x,8);
  NASSERT_ALIGN(y,8);
  NASSERT_ALIGN(v,8);
  NASSERT_ALIGN(w,8);
  NASSERT_ALIGN(z,8);
  NASSERT_ALIGN(b,8);
  NASSERT(cols1%4==0);
  NASSERT(cols2%4==0);
  NASSERT(row_stride1%4==0);
  NASSERT(row_stride2%4==0);

  pz = (xtfloatx2 *)z;
  pb = (const xtfloatx2 *)(b);

  if (y && w)
  {
    /* Compute by 4 values */
#if defined(ENABLE_PRAGMA)
    __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
    for (m = 0; m < (rows>>2); m++)
    {
      px0 = (const xtfloatx2 *)(x+(4*m*row_stride1));
      pv0 = (const xtfloatx2 *)(v+(4*m*row_stride2));

      px1 = (const xtfloatx2 *)((FLOAT32 *)px0+row_stride1);
      px2 = (const xtfloatx2 *)((FLOAT32 *)px1+row_stride1);
      px3 = (const xtfloatx2 *)((FLOAT32 *)px2+row_stride1);
      py  = (const xtfloatx2 *)(y);
      pv1 = (const xtfloatx2 *)((FLOAT32 *)pv0+row_stride2);
      pv2 = (const xtfloatx2 *)((FLOAT32 *)pv1+row_stride2);
      pv3 = (const xtfloatx2 *)((FLOAT32 *)pv2+row_stride2);
      pw  = (const xtfloatx2 *)(w);

      acc00 = acc01 = acc10 = acc11 =
      acc20 = acc21 = acc30 = acc31 =  (xtfloatx2)0.0f;

      b0 = b1 = (xtfloatx2)0.0f;
      if(b != NULL){
        XT_LSX2IP(b0, pb, SZ_F32*2);
        XT_LSX2IP(b1, pb, SZ_F32*2);
      }

#if defined(ENABLE_PRAGMA)
      __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
      for (n = 0; n < (cols1>>2); n++)
      {
        XT_LSX2IP(x00, px0, SZ_F32*2);
        XT_LSX2IP(x01, px0, SZ_F32*2);
        XT_LSX2IP(x10, px1, SZ_F32*2);
        XT_LSX2IP(x11, px1, SZ_F32*2);
        XT_LSX2IP(x20, px2, SZ_F32*2);
        XT_LSX2IP(x21, px2, SZ_F32*2);
        XT_LSX2IP(x30, px3, SZ_F32*2);
        XT_LSX2IP(x31, px3, SZ_F32*2);

        XT_LSX2IP( y0,  py, SZ_F32*2);
        XT_LSX2IP( y1,  py, SZ_F32*2);

        XT_MADD_SX2(acc00, x00, y0);
        XT_MADD_SX2(acc01, x01, y1);
        XT_MADD_SX2(acc10, x10, y0);
        XT_MADD_SX2(acc11, x11, y1);
        XT_MADD_SX2(acc20, x20, y0);
        XT_MADD_SX2(acc21, x21, y1);
        XT_MADD_SX2(acc30, x30, y0);
        XT_MADD_SX2(acc31, x31, y1);
      }
      acc00 = acc00 + acc01;
      acc10 = acc10 + acc11;
      y0 = XT_SEL32_HL_SX2(acc00, acc10);
      y1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = y0 + y1;

      acc20 = acc20 + acc21;
      acc30 = acc30 + acc31;
      y0 = XT_SEL32_HL_SX2(acc20, acc30);
      y1 = XT_SEL32_LH_SX2(acc20, acc30);
      z1 = y0 + y1;


      acc00 = acc01 = acc10 = acc11 =
      acc20 = acc21 = acc30 = acc31 =  (xtfloatx2)0.0f;

#if defined(ENABLE_PRAGMA)
      __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
      for (k = 0; k < (cols2>>2); k++)
      {
        XT_LSX2IP(v00, pv0, SZ_F32*2);
        XT_LSX2IP(v01, pv0, SZ_F32*2);
        XT_LSX2IP(v10, pv1, SZ_F32*2);
        XT_LSX2IP(v11, pv1, SZ_F32*2);
        XT_LSX2IP(v20, pv2, SZ_F32*2);
        XT_LSX2IP(v21, pv2, SZ_F32*2);
        XT_LSX2IP(v30, pv3, SZ_F32*2);
        XT_LSX2IP(v31, pv3, SZ_F32*2);

        XT_LSX2IP( w0,  pw, SZ_F32*2);
        XT_LSX2IP( w1,  pw, SZ_F32*2);

        XT_MADD_SX2(acc00, v00, w0);
        XT_MADD_SX2(acc01, v01, w1);
        XT_MADD_SX2(acc10, v10, w0);
        XT_MADD_SX2(acc11, v11, w1);
        XT_MADD_SX2(acc20, v20, w0);
        XT_MADD_SX2(acc21, v21, w1);
        XT_MADD_SX2(acc30, v30, w0);
        XT_MADD_SX2(acc31, v31, w1);
      }
      acc00 = acc00 + acc01;
      acc10 = acc10 + acc11;
      w0 = XT_SEL32_HL_SX2(acc00, acc10);
      w1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = z0 + w0;
      z0 = z0 + w1;
      z0 = z0 + b0;

      acc20 = acc20 + acc21;
      acc30 = acc30 + acc31;
      w0 = XT_SEL32_HL_SX2(acc20, acc30);
      w1 = XT_SEL32_LH_SX2(acc20, acc30);
      z1 = z1 + w0;
      z1 = z1 + w1;
      z1 = z1 + b1;

      XT_SSX2IP(z0, pz, SZ_F32*2);
      XT_SSX2IP(z1, pz, SZ_F32*2);
    }

    /* Compute last (rows%4) output element */
    for (m = rows&(~3); m < rows; m++)
    {
      px0 = (const xtfloatx2 *)(x+m*row_stride1);
      py  = (const xtfloatx2 *)(y);
      pz_ = (xtfloat *)(z+m);
      pv0 = (const xtfloatx2 *)(v+m*row_stride2);
      pw  = (const xtfloatx2 *)(w);

      b0_ = 0.0f;
      if(b != NULL){
        b0_ = b[m];
      }
      acc00 = acc01 = (xtfloatx2)0.0f;

#if defined(ENABLE_PRAGMA)
      __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
      for (n = 0; n < (cols1>>2); n++)
      {
        XT_LSX2IP(x00, px0, SZ_F32*2);
        XT_LSX2IP(x01, px0, SZ_F32*2);
        XT_LSX2IP(y0,  py, SZ_F32*2);
        XT_LSX2IP(y1,  py, SZ_F32*2);

        XT_MADD_SX2(acc00, x00, y0);
        XT_MADD_SX2(acc01, x01, y1);
      }
      acc00 = acc00 + acc01;

      acc20 = acc21 = (xtfloatx2)0.0f;
#if defined(ENABLE_PRAGMA)
      __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
      for (k = 0; k < (cols2>>2); k++)
      {
        XT_LSX2IP(v00, pv0, SZ_F32*2);
        XT_LSX2IP(v01, pv0, SZ_F32*2);
        XT_LSX2IP(w0,  pw, SZ_F32*2);
        XT_LSX2IP(w1,  pw, SZ_F32*2);

        XT_MADD_SX2(acc20, v00, w0);
        XT_MADD_SX2(acc21, v01, w1);
      }
      acc20 = acc20 + acc21;
      acc00 = acc00 + acc20;

      z0_ = XT_RADD_SX2(acc00);
      z0_ = z0_ + b0_;

      XT_SSIP(z0_, pz_, sizeof(FLOAT32));
    }
    return 0;
  }
  else
  {
    /* Compute by 4 values */
#if defined(ENABLE_PRAGMA)
    __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
    for (m = 0; m < (rows>>2); m++)
    {
      px0 = (const xtfloatx2 *)(x+(4*m*row_stride1));

      px1 = (const xtfloatx2 *)((FLOAT32 *)px0+row_stride1);
      px2 = (const xtfloatx2 *)((FLOAT32 *)px1+row_stride1);
      px3 = (const xtfloatx2 *)((FLOAT32 *)px2+row_stride1);
      py  = (const xtfloatx2 *)(y);

      acc00 = acc01 = acc10 = acc11 =
      acc20 = acc21 = acc30 = acc31 =  (xtfloatx2)0.0f;

      b0 = b1 = (xtfloatx2)0.0f;
      if(b != NULL){
        XT_LSX2IP(b0, pb, SZ_F32*2);
        XT_LSX2IP(b1, pb, SZ_F32*2);
      }

#if defined(ENABLE_PRAGMA)
      __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
      for (n = 0; n < (cols1>>2); n++)
      {
        XT_LSX2IP(x00, px0, SZ_F32*2);
        XT_LSX2IP(x01, px0, SZ_F32*2);
        XT_LSX2IP(x10, px1, SZ_F32*2);
        XT_LSX2IP(x11, px1, SZ_F32*2);
        XT_LSX2IP(x20, px2, SZ_F32*2);
        XT_LSX2IP(x21, px2, SZ_F32*2);
        XT_LSX2IP(x30, px3, SZ_F32*2);
        XT_LSX2IP(x31, px3, SZ_F32*2);

        XT_LSX2IP( y0,  py, SZ_F32*2);
        XT_LSX2IP( y1,  py, SZ_F32*2);

        XT_MADD_SX2(acc00, x00, y0);
        XT_MADD_SX2(acc01, x01, y1);
        XT_MADD_SX2(acc10, x10, y0);
        XT_MADD_SX2(acc11, x11, y1);
        XT_MADD_SX2(acc20, x20, y0);
        XT_MADD_SX2(acc21, x21, y1);
        XT_MADD_SX2(acc30, x30, y0);
        XT_MADD_SX2(acc31, x31, y1);
      }
      acc00 = acc00 + acc01;
      acc10 = acc10 + acc11;
      y0 = XT_SEL32_HL_SX2(acc00, acc10);
      y1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = y0 + y1;
      z0 = z0 + b0;

      acc20 = acc20 + acc21;
      acc30 = acc30 + acc31;
      y0 = XT_SEL32_HL_SX2(acc20, acc30);
      y1 = XT_SEL32_LH_SX2(acc20, acc30);
      z1 = y0 + y1;
      z1 = z1 + b1;

      XT_SSX2IP(z0, pz, SZ_F32*2);
      XT_SSX2IP(z1, pz, SZ_F32*2);
    }

    /* Compute last (rows%4) output element */
    for (m = rows&(~3); m < rows; m++)
    {
      px0 = (const xtfloatx2 *)(x+m*row_stride1);
      py  = (const xtfloatx2 *)(y);
      pz_ = (xtfloat *)(z+m);

      b0_ = 0.0f;
      if(b != NULL){
        b0_ = b[m];
      }
      acc00 = acc01 = (xtfloatx2)0.0f;

#if defined(ENABLE_PRAGMA)
      __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
      for (n = 0; n < (cols1>>2); n++)
      {
        XT_LSX2IP(x00, px0, SZ_F32*2);
        XT_LSX2IP(x01, px0, SZ_F32*2);
        XT_LSX2IP(y0,  py, SZ_F32*2);
        XT_LSX2IP(y1,  py, SZ_F32*2);

        XT_MADD_SX2(acc00, x00, y0);
        XT_MADD_SX2(acc01, x01, y1);
      }
      acc00 = acc00 + acc01;

      z0_ = XT_RADD_SX2(acc00);
      z0_ = z0_ + b0_;

      XT_SSIP(z0_, pz_, sizeof(FLOAT32));
    }
    return 0;
  }
} /* dual_mtx_vecmpyf_bias_add() */

WORD32 static dual_mtx_vecmpyf_bias_add_generic( FLOAT32 * z,
     const FLOAT32 * x,  const FLOAT32 * y, const FLOAT32 * v, const FLOAT32 * w,
     const FLOAT32 * b, int rows, int cols1, int cols2, int row_stride1, int row_stride2 )
{
  const xtfloatx2 *restrict px0;
  const xtfloatx2 *restrict px1;
  const xtfloatx2 *restrict pv0;
  const xtfloatx2 *restrict pv1;
  const xtfloatx2 *restrict py;
  const xtfloatx2 *restrict pw;
  const xtfloat   *restrict pb;
        xtfloatx2 *restrict pz;
        xtfloat *restrict pz_;
  xtfloatx2 b0;
  xtfloatx2 y0, y1;
  xtfloatx2 w0, w1;
  xtfloatx2 z0;
  xtfloat z0_, b0_;
  xtfloatx2 x00, x01, x10, x11;
  xtfloatx2 v00, v01, v10, v11;
  xtfloatx2 acc00, acc01, acc10, acc11;
  ae_valign x0_a, x1_a, y_a;
  ae_valign v0_a, v1_a, w_a;
  int m, n, k;

  NASSERT(x);
  NASSERT(y);
  NASSERT(v);
  NASSERT(w);
  NASSERT(z);
//  NASSERT(b);
  NASSERT((z != x) && (z != y) && (z != v) && (z != w) && (z != b));
  NASSERT_ALIGN(x,4);
  NASSERT_ALIGN(y,4);
  NASSERT_ALIGN(v,4);
  NASSERT_ALIGN(w,4);
  NASSERT_ALIGN(z,4);
//  NASSERT_ALIGN(b,4);

  pz = (xtfloatx2 *)z;
  pb = (const xtfloat *)(b);

  ae_valign z_a = AE_ZALIGN64();
  if (y && w)
  {
    for (m = 0; m < (rows>>1); m++)
    {
      px0 = (const xtfloatx2 *)(x+(2*m*row_stride1));
      pv0 = (const xtfloatx2 *)(v+(2*m*row_stride2));

      px1 = (const xtfloatx2 *)((FLOAT32 *)px0+row_stride1);
      py  = (const xtfloatx2 *)(y);
      pv1 = (const xtfloatx2 *)((FLOAT32 *)pv0+row_stride2);
      pw  = (const xtfloatx2 *)(w);

      x0_a = XT_LASX2PP(px0);
      x1_a = XT_LASX2PP(px1);
      y_a = XT_LASX2PP(py);

      acc00 = acc01 = acc10 = acc11 = (xtfloatx2)0.0f;

      b0 = (xtfloatx2)0.0f;
      if(b != NULL){
        b0 = XT_SEL32_LL_SX2((xtfloatx2)(pb[(m<<1)+0]), (xtfloatx2)(pb[(m<<1)+1]));
      }

      for (n = 0; n < (cols1>>2); n++)
      {
        XT_LASX2IP(x00, x0_a, px0);
        XT_LASX2IP(x01, x0_a, px0);
        XT_LASX2IP(x10, x1_a, px1);
        XT_LASX2IP(x11, x1_a, px1);

        XT_LASX2IP( y0, y_a,  py);
        XT_LASX2IP( y1, y_a,  py);

        XT_MADD_SX2(acc00, x00, y0);
        XT_MADD_SX2(acc01, x01, y1);
        XT_MADD_SX2(acc10, x10, y0);
        XT_MADD_SX2(acc11, x11, y1);
      }
      acc00 = acc00 + acc01;
      acc10 = acc10 + acc11;
      y0 = XT_SEL32_HL_SX2(acc00, acc10);
      y1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = y0 + y1;

      acc00 = 0.0f;
      for(k = 0; k < (cols1&3); k++)
      {
          x00 = XT_SEL32_LL_SX2((xtfloatx2)(*(((xtfloat *)px0)+k)), (xtfloatx2)(*(((xtfloat *)px1)+k)));
          XT_MADD_SX2(acc00, x00, (xtfloatx2)(*(((xtfloat *)py)+k)));
      }
      z0 = z0 + acc00;

      v0_a = XT_LASX2PP(pv0);
      v1_a = XT_LASX2PP(pv1);
      w_a = XT_LASX2PP(pw);

      acc00 = acc01 = acc10 = acc11 = (xtfloatx2)0.0f;

      for (k = 0; k < (cols2>>2); k++)
      {
        XT_LASX2IP(v00, v0_a, pv0);
        XT_LASX2IP(v01, v0_a, pv0);
        XT_LASX2IP(v10, v1_a, pv1);
        XT_LASX2IP(v11, v1_a, pv1);

        XT_LASX2IP( w0, w_a,  pw);
        XT_LASX2IP( w1, w_a,  pw);

        XT_MADD_SX2(acc00, v00, w0);
        XT_MADD_SX2(acc01, v01, w1);
        XT_MADD_SX2(acc10, v10, w0);
        XT_MADD_SX2(acc11, v11, w1);
      }
      acc00 = acc00 + acc01;
      acc10 = acc10 + acc11;
      w0 = XT_SEL32_HL_SX2(acc00, acc10);
      w1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = z0 + w0;
      z0 = z0 + w1;

      acc00 = 0.0f;
      for(k = 0; k < (cols2&3); k++)
      {
          v00 = XT_SEL32_LL_SX2((xtfloatx2)(*(((xtfloat *)pv0)+k)), (xtfloatx2)(*(((xtfloat *)pv1)+k)));
          XT_MADD_SX2(acc00, v00, (xtfloatx2)(*(((xtfloat *)pw)+k)));
      }
      z0 = z0 + acc00;

      /* Add bias */
      z0 = z0 + b0;

      XT_SASX2IP(z0, z_a, pz);
    }
    XT_SASX2POSFP(z_a, pz);

    /* Compute last (rows%2) output element */
    for (m = rows&(~1); m < rows; m++)
    {
      px0 = (const xtfloatx2 *)(x+m*row_stride1);
      py  = (const xtfloatx2 *)(y);
      pz_ = (xtfloat *)(z+m);
      pv0 = (const xtfloatx2 *)(v+m*row_stride2);
      pw  = (const xtfloatx2 *)(w);

      x0_a = XT_LASX2PP(px0);
      y_a = XT_LASX2PP(py);

      b0_ = 0.0f;
      if(b != NULL){
        b0_ = b[m];
      }
      acc00 = acc01 = (xtfloatx2)0.0f;

      for (n = 0; n < (cols1>>2); n++)
      {
        XT_LASX2IP(x00, x0_a, px0);
        XT_LASX2IP(x01, x0_a, px0);
        XT_LASX2IP(y0, y_a,  py);
        XT_LASX2IP(y1, y_a,  py);

        XT_MADD_SX2(acc00, x00, y0);
        XT_MADD_SX2(acc01, x01, y1);
      }
      acc00 = acc00 + acc01;
      z0_ = XT_RADD_SX2(acc00);

      for(n = 0; n < (cols1&3); n++)
      {
          XT_MADD_S(z0_, *(((xtfloat *)px0)+n), *(((xtfloat *)py)+n));
      }

      v0_a = XT_LASX2PP(pv0);
      w_a = XT_LASX2PP(pw);
      acc00 = acc01 = (xtfloatx2)0.0f;
      for (k = 0; k < (cols2>>2); k++)
      {
        XT_LASX2IP(v00, v0_a, pv0);
        XT_LASX2IP(v01, v0_a, pv0);
        XT_LASX2IP(w0, w_a,  pw);
        XT_LASX2IP(w1, w_a,  pw);

        XT_MADD_SX2(acc00, v00, w0);
        XT_MADD_SX2(acc01, v01, w1);
      }
      acc00 = acc00 + acc01;
      z0_ = z0_ + XT_RADD_SX2(acc00);

      for(n = 0; n < (cols2&3); n++)
      {
          XT_MADD_S(z0_, *(((xtfloat *)pv0)+n), *(((xtfloat *)pw)+n));
      }

      z0_ = z0_ + b0_;

      XT_SSIP(z0_, pz_, sizeof(FLOAT32));
    }
    return 0;
  }
  else
  {
    for (m = 0; m < (rows>>1); m++)
    {
      px0 = (const xtfloatx2 *)(x+(2*m*row_stride1));

      px1 = (const xtfloatx2 *)((FLOAT32 *)px0+row_stride1);
      py  = (const xtfloatx2 *)(y);

      x0_a = XT_LASX2PP(px0);
      x1_a = XT_LASX2PP(px1);
      y_a = XT_LASX2PP(py);

      acc00 = acc01 = acc10 = acc11 = (xtfloatx2)0.0f;

      b0 = (xtfloatx2)0.0f;
      if(b != NULL){
        b0 = XT_SEL32_LL_SX2((xtfloatx2)(pb[(m<<1)+0]), (xtfloatx2)(pb[(m<<1)+1]));
      }

      for (n = 0; n < (cols1>>2); n++)
      {
        XT_LASX2IP(x00, x0_a, px0);
        XT_LASX2IP(x01, x0_a, px0);
        XT_LASX2IP(x10, x1_a, px1);
        XT_LASX2IP(x11, x1_a, px1);

        XT_LASX2IP( y0, y_a,  py);
        XT_LASX2IP( y1, y_a,  py);

        XT_MADD_SX2(acc00, x00, y0);
        XT_MADD_SX2(acc01, x01, y1);
        XT_MADD_SX2(acc10, x10, y0);
        XT_MADD_SX2(acc11, x11, y1);
      }
      acc00 = acc00 + acc01;
      acc10 = acc10 + acc11;
      y0 = XT_SEL32_HL_SX2(acc00, acc10);
      y1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = y0 + y1;

      acc00 = 0.0f;
      for(k = 0; k < (cols1&3); k++)
      {
          x00 = XT_SEL32_LL_SX2((xtfloatx2)(*(((xtfloat *)px0)+k)), (xtfloatx2)(*(((xtfloat *)px1)+k)));
          XT_MADD_SX2(acc00, x00, (xtfloatx2)(*(((xtfloat *)py)+k)));
      }
      z0 = z0 + acc00;

      /* Add bias */
      z0 = z0 + b0;

      XT_SASX2IP(z0, z_a, pz);
    }
    XT_SASX2POSFP(z_a, pz);

    /* Compute last (rows%2) output element */
    for (m = rows&(~1); m < rows; m++)
    {
      px0 = (const xtfloatx2 *)(x+m*row_stride1);
      py  = (const xtfloatx2 *)(y);
      pz_ = (xtfloat *)(z+m);

      x0_a = XT_LASX2PP(px0);
      y_a = XT_LASX2PP(py);

      b0_ = 0.0f;
      if(b != NULL){
        b0_ = b[m];
      }
      acc00 = acc01 = (xtfloatx2)0.0f;

      for (n = 0; n < (cols1>>2); n++)
      {
        XT_LASX2IP(x00, x0_a, px0);
        XT_LASX2IP(x01, x0_a, px0);
        XT_LASX2IP(y0, y_a,  py);
        XT_LASX2IP(y1, y_a,  py);

        XT_MADD_SX2(acc00, x00, y0);
        XT_MADD_SX2(acc01, x01, y1);
      }
      acc00 = acc00 + acc01;
      z0_ = XT_RADD_SX2(acc00);

      for(n = 0; n < (cols1&3); n++)
      {
          XT_MADD_S(z0_, *(((xtfloat *)px0)+n), *(((xtfloat *)py)+n));
      }

      z0_ = z0_ + b0_;

      XT_SSIP(z0_, pz_, sizeof(FLOAT32));
    }
    return 0;
  }
} /* dual_mtx_vecmpyf_bias_add_generic() */

#endif /* HAVE_VFPU */


/*-------------------------------------------------------------------------
  xa_nn_matXvec_f32xf32_f32_sigmoid
  This function computes the sigmoid operated over dual matrix vector
  multiplication with added bias vector value (the most fundamental DNN
  operation). The inputs and output are all 32 bit float numbers.

  Precision:
  f32xf32_f32  32-bit float inputs, 32-bit float output.

  Input:
  p_mat1         first matrix pointer,                32-bit float
  p_mat2         second matrix pointer,               32-bit float
  p_vec1         first vector pointer,                32-bit float
  p_vec2         second vector pointer,               32-bit float
  p_bias         bias vector pointer,                 32-bit float
  rows           number of rows,                      32 bit integer
  cols1          number of columns of first matrix,   32 bit integer
  cols2          number of columns of second matrix,  32 bit integer
  row_stride1    row offset of first matrix,          32 bit integer
  row_stride2    row offset of second matrix,         32 bit integer
  p_scratch      intermediate scratch vector pointer, 32-bit float
  Output:
  p_out          result vector pointer,               32-bit float

  Restriction:
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should hold
  valid addresses in the memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should not
  overlap in the memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should be 8 byte
  boundaries aligned in the memory space
  cols1, cols2, row_stride1, row_stride2 should be multiple of 4
-------------------------------------------------------------------------*/
#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_matXvec_f32xf32_f32_sigmoid,(
    FLOAT32  *  p_out,
    FLOAT32  *  p_mat1,
    FLOAT32  *  p_mat2,
    FLOAT32  *  p_vec1,
    FLOAT32  *  p_vec2,
    FLOAT32  *  p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    FLOAT32  * __restrict__ p_scratch))
#else
WORD32  xa_nn_matXvec_f32xf32_f32_sigmoid(
    FLOAT32  * __restrict__ p_out,
    FLOAT32  * __restrict__ p_mat1,
    FLOAT32  * __restrict__ p_mat2,
    FLOAT32  * __restrict__ p_vec1,
    FLOAT32  * __restrict__ p_vec2,
    FLOAT32  * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    FLOAT32  * __restrict__ p_scratch)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
//  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out , sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);


  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
  }

  WORD32 ret = 0, k;

  if(((cols1&3) == 0) && ((cols2&3) == 0) && ((row_stride1&1) == 0) && ((row_stride2&1) == 0) &&
     ((((unsigned)p_out)&7) == 0) && ((((unsigned)p_mat1)&7) == 0) && ((((unsigned)p_vec1)&7) == 0) &&
     ((((unsigned)p_mat2)&7) == 0) && ((((unsigned)p_vec2)&7) == 0) && ((((unsigned)p_bias)&7) == 0))
  {
    ret = dual_mtx_vecmpyf_bias_add(p_scratch, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }
  else
  {
    ret = dual_mtx_vecmpyf_bias_add_generic(p_scratch, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }

  if (0 == ret)
  {
    xa_nn_vec_sigmoid_f32_f32(p_out, p_scratch, rows);
  }
  else if (-1 == ret)
  {
    /* In erroneous case, populate output with zeros. */
    for (k = 0; k < rows; k++)
    {
      p_out[k] = 0.0f;
    }
  }

  return ret;
}
#endif /* !HAVE_VFPU */


/*-------------------------------------------------------------------------
  xa_nn_matXvec_f32xf32_f32_tanh
  This function computes the tanh operated over dual matrix vector
  multiplication with added bias vector value (the most fundamental DNN
  operation). The inputs and output are all 32 bit float numbers.

  Precision:
  f32xf32_f32  32-bit float inputs, 32-bit float output.

  Input:
  p_mat1         first matrix pointer,                32-bit float
  p_mat2         second matrix pointer,               32-bit float
  p_vec1         first vector pointer,                32-bit float
  p_vec2         second vector pointer,               32-bit float
  p_bias         bias vector pointer,                 32-bit float
  rows           number of rows,                      32 bit integer
  cols1          number of columns of first matrix,   32 bit integer
  cols2          number of columns of second matrix,  32 bit integer
  row_stride1    row offset of first matrix,          32 bit integer
  row_stride2    row offset of second matrix,         32 bit integer
  p_scratch      intermediate scratch vector pointer, 32-bit float
  Output:
  p_out          result vector pointer,               32-bit float

  Restriction:
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should hold
  valid addresses in the memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should not
  overlap in the memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should be 8 byte
  boundaries aligned in the memory space
  cols1, cols2, row_stride1, row_stride2 should be multiple of 4
-------------------------------------------------------------------------*/
#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_matXvec_f32xf32_f32_tanh,(
    FLOAT32  *  p_out,
    FLOAT32  *  p_mat1,
    FLOAT32  *  p_mat2,
    FLOAT32  *  p_vec1,
    FLOAT32  *  p_vec2,
    FLOAT32  *  p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    FLOAT32  * __restrict__ p_scratch))
#else
WORD32  xa_nn_matXvec_f32xf32_f32_tanh(
    FLOAT32  * __restrict__ p_out,
    FLOAT32  * __restrict__ p_mat1,
    FLOAT32  * __restrict__ p_mat2,
    FLOAT32  * __restrict__ p_vec1,
    FLOAT32  * __restrict__ p_vec2,
    FLOAT32  * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    FLOAT32  * __restrict__ p_scratch)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
 // XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out , sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
  }

  WORD32 ret = 0, k;
  if(((cols1&3) == 0) && ((cols2&3) == 0) && ((row_stride1&1) == 0) && ((row_stride2&1) == 0) &&
     ((((unsigned)p_out)&7) == 0) && ((((unsigned)p_mat1)&7) == 0) && ((((unsigned)p_vec1)&7) == 0) &&
     ((((unsigned)p_mat2)&7) == 0) && ((((unsigned)p_vec2)&7) == 0) && ((((unsigned)p_bias)&7) == 0))
  {
    ret = dual_mtx_vecmpyf_bias_add(p_scratch, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }
  else
  {
    ret = dual_mtx_vecmpyf_bias_add_generic(p_scratch, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }

  if (0 == ret)
  {
    xa_nn_vec_tanh_f32_f32(p_out, p_scratch, rows);
  }
  else if (-1 == ret)
  {
    /* In erroneous case, populate output with zeros. */
    for (k = 0; k < rows; k++)
    {
      p_out[k] = 0.0f;
    }
  }

  return ret;
}
#endif /* !HAVE_VFPU */


/*-------------------------------------------------------------------------
  xa_nn_matXvec_f32xf32_f32
  This function computes the dual matrix vector multiplication with added
  bias vector value (the most fundamental DNN operation). The inputs and
  output are all 32 bit float numbers.

  Precision:
  f32xf32_f32  32-bit float inputs, 32-bit float output.

  Input:
  p_mat1         first matrix pointer,                32-bit float
  p_mat2         second matrix pointer,               32-bit float
  p_vec1         first vector pointer,                32-bit float
  p_vec2         second vector pointer,               32-bit float
  p_bias         bias vector pointer,                 32-bit float
  rows           number of rows,                      32 bit integer
  cols1          number of columns of first matrix,   32 bit integer
  cols2          number of columns of second matrix,  32 bit integer
  row_stride1    row offset of first matrix,          32 bit integer
  row_stride2    row offset of second matrix,         32 bit integer
  Output:
  p_out          result vector pointer,               32-bit float

  Restriction:
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias should hold valid addresses
  in the memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias should not overlap in the
  memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias should be 4 byte boundaries
  aligned in the memory space
-------------------------------------------------------------------------*/
#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_matXvec_f32xf32_f32,(
    FLOAT32  *  p_out,
    const FLOAT32  *  p_mat1,
    const FLOAT32  *  p_mat2,
    const FLOAT32  *  p_vec1,
    const FLOAT32  *  p_vec2,
    const FLOAT32  *  p_bias,
    WORD32 rows, WORD32 cols1, WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2))
#else
WORD32  xa_nn_matXvec_f32xf32_f32(
    FLOAT32  * __restrict__ p_out,
    const FLOAT32  * __restrict__ p_mat1,
    const FLOAT32  * __restrict__ p_mat2,
    const FLOAT32  * __restrict__ p_vec1,
    const FLOAT32  * __restrict__ p_vec2,
    const FLOAT32  * __restrict__ p_bias,
    WORD32 rows, WORD32 cols1, WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
//  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
  }

  WORD32 ret = 0, k;
  if(((cols1&3) == 0) && ((cols2&3) == 0) && ((row_stride1&1) == 0) && ((row_stride2&1) == 0) &&
     ((((unsigned)p_out)&7) == 0) && ((((unsigned)p_mat1)&7) == 0) && ((((unsigned)p_vec1)&7) == 0) &&
     ((((unsigned)p_mat2)&7) == 0) && ((((unsigned)p_vec2)&7) == 0) && ((((unsigned)p_bias)&7) == 0))
  {
    ret = dual_mtx_vecmpyf_bias_add(p_out, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }
  else
  {
    ret = dual_mtx_vecmpyf_bias_add_generic(p_out, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }

  if (-1 == ret)
  {
    /* In erroneous case, populate output with zeros. */
    for (k = 0; k < rows; k++)
    {
      p_out[k] = 0.0f;
    }
  }

  return ret;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_dot_prod_f32xf32_f32,(
    FLOAT32  *  p_out,
    FLOAT32  *  p_inp1,
    FLOAT32  *  p_inp2,
    WORD32 vec_length,
    WORD32 num_vecs))
#else
WORD32 xa_nn_dot_prod_f32xf32_f32(
         FLOAT32 * __restrict__ p_out,          /* pointer to output */
         const FLOAT32 * __restrict__ p_inp1,   /* pointer to input1 */
         const FLOAT32 * __restrict__ p_inp2,   /* pointer to input2 */
         WORD32 vec_length,
         WORD32 num_vecs)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((num_vecs <= 0), -1);

    xtfloatx2 *pt_inp1, *pt_inp2;
    xtfloatx2 d_inp1, d_inp2;
    xtfloatx2 d_acc;
    float d_out;
    int i, j;

    if(((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0) && ((vec_length&1) == 0))
    {
        pt_inp1 = (xtfloatx2 *)p_inp1;
        pt_inp2 = (xtfloatx2 *)p_inp2;
        for(i = 0; i < num_vecs; i++)
        {
            d_acc = (xtfloatx2)0.0f;
            for(j = 0; j < (vec_length>>1); j++)
            {
                d_inp1 = *pt_inp1++;
                d_inp2 = *pt_inp2++;
                XT_MADD_SX2(d_acc, d_inp1, d_inp2);
            }
            d_out = XT_ADD_S(XT_HIGH_S(d_acc), XT_LOW_S(d_acc));
            *(float *)(&p_out[i]) = d_out;
        }
    }
    else
    {
        ae_valign inp1_a, inp2_a;

        for(i = 0; i < num_vecs; i++)
        {
            pt_inp1 = (xtfloatx2 *)(&p_inp1[i*vec_length]);
            pt_inp2 = (xtfloatx2 *)(&p_inp2[i*vec_length]);
            inp1_a = XT_LASX2PP(pt_inp1);
            inp2_a = XT_LASX2PP(pt_inp2);
            d_acc = (xtfloatx2)0.0f;
            for(j = 0; j < (vec_length>>1); j++)
            {
                XT_LASX2IP(d_inp1, inp1_a, pt_inp1);
                XT_LASX2IP(d_inp2, inp2_a, pt_inp2);
                XT_MADD_SX2(d_acc, d_inp1, d_inp2);
            }
            d_out = XT_ADD_S(XT_HIGH_S(d_acc), XT_LOW_S(d_acc));
            float *pt_inp1u, *pt_inp2u;
            pt_inp1u = (float *)pt_inp1;
            pt_inp2u = (float *)pt_inp2;
            if((vec_length&1) != 0)
            {
                XT_MADD_S(d_out, pt_inp1u[0], pt_inp2u[0]);
            }
            *(float *)(&p_out[i]) = d_out;
        }
    }
    return 0;
}
#endif /* #if !HAVE_VFPU */

#endif

