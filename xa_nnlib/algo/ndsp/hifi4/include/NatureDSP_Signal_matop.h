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
#ifndef __NATUREDSP_SIGNAL_MATOP_H__
#define __NATUREDSP_SIGNAL_MATOP_H__

#include "NatureDSP_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
  Matrix Operations:
  mtx_mpy              Matrix Multiply
  mtx_vecmpy           Matrix by Vector Multiply
===========================================================================*/

/*-------------------------------------------------------------------------
  Matrix Multiply
  These functions compute the expression z = 2^lsh * x * y for the matrices
  x and y. The columnar dimension of x must match the row dimension of y.
  The resulting matrix has the same number of rows as x and the same number
  of columns as y.
  Transposing API allows to interpret input yt as transposed matrix y.

  NOTE: lsh factor is not relevant for floating point routines.

  Functions require scratch memory for storing intermediate data. This
  scratch memory area should be aligned on 8 byte boundary and its size is
  calculated by dedicated scratch allocation functions.

  Two versions of functions available: regular version (mtx_mpy[t]32x32,
  mtx_mpy[t]16x16, mtx_mpy[t]8x16, mtx_mpy[t]8x8, mtx[t]_mpyf) with
  arbitrary arguments and faster version (mtx_mpy[t]32x32_fast,
  mtx_mpy[t]16x16_fast, mtx_mpy[t]8x16_fast, mtx_mpy[t]8x8_fast,
  mtx_mpy[t]f_fast) that apply some restrictions

  Precision:
  32x32 32-bit inputs, 32-bit output
  16x16 16-bit inputs, 16-bit output
  8x8   8-bit inputs, 8-bit output
  8x16  8/16-bit inputs, 16-bit output
  f     floating point

  Input:
  x[M*N]      input matrix x, Q7, Q15, Q31 or floating point
  y[N*P]      input matrix y, Q7, Q15, Q31 or floating point
  yt[P*N]     transposed input matrix y. Q31,Q15, Q7 floating point. (for
              transposing API only)
  M           number of rows in matrix x and z
  N           number of columns in matrix x and number of rows in matrix y
  P           number of columns in matrices y and z
  lsh         left shift applied to the result (applied to the fixed-
              point functions only)
  Output:
  z[M*P]      output matrix z, Q7, Q15, Q31 or floating point
  Scratch:
  pScr        size in bytes defined by corresponding scratch allocation
              functions

  Restrictions:
  For regular routines mpy[t]32x32, mtx_mpy[t]16x16, mtx_mpy[t]8x8,
  mtx_mpy[t]8x16, mtx_mpy[t]f):
  pScr    aligned on 8-byte boundary
  x,y,z   should not overlap

  For faster routines (mtx_mpy[t]32x32_fast, mtx_mpy[t]16x16_fast,
  mtx_mpy[t]8x8_fast, mtx_mpy[t]8x16_fast,
  mtx_mpy[t]f_fast):
  x,y,z        should not overlap
  x,y,z,pScr   aligned on 8-byte boundary
  M,N,P        multiplies of 4
  lsh         should be in range:
              -31...31 for mtx_mpy32x32, mtx_mpy32x32_fast
              -15...15 for mtx_mpy16x16, mtx_mpy16x16_fast, mtx_mpy[t]8x8,
                       mtx_mpy[t]8x8_fast, mtx_mpy[t]8x16,
                       mtx_mpy[t]8x16_fast

-------------------------------------------------------------------------*/
void mtx_mpy32x32 ( void* pScr,
                    int32_t* restrict z,
              const int32_t* restrict x,
              const int32_t* restrict y,
              int M, int N, int P, int lsh );
void mtx_mpy16x16 ( void* pScr,
                    int16_t* restrict z,
              const int16_t* restrict x,
              const int16_t* restrict y,
              int M, int N, int P, int lsh );
void mtx_mpy8x8  (  void* pScr,
                     int8_t*  z, const int8_t*  x, const int8_t*  y,
                     int M, int N, int P, int lsh );
void mtx_mpy8x16 (  void* pScr,
                     int16_t*  z, const int8_t*  x, const int16_t*  y,
                     int M, int N, int P, int lsh );
void mtx_mpy32x32_fast ( void* pScr,int32_t* restrict z,
                   const int32_t* restrict x,
                   const int32_t* restrict y,
                   int M, int N, int P, int lsh );
void mtx_mpy16x16_fast ( void* pScr, int16_t* restrict z,
                   const int16_t* restrict x,
                   const int16_t* restrict y,
                   int M, int N, int P, int lsh );
void mtx_mpy8x8_fast(  void* pScr,
                     int8_t*  z, const int8_t*  x, const int8_t*  y,
                     int M, int N, int P, int lsh );
void mtx_mpy8x16_fast (  void* pScr,
                     int16_t*  z, const int8_t*  x, const int16_t*  y,
                     int M, int N, int P, int lsh );

void mtx_mpyf ( void* pScr, float32_t* restrict z,
          const float32_t* restrict x,
          const float32_t* restrict y,
          int M, int N, int P);
void mtx_mpyf_fast ( void* pScr, float32_t* restrict z,
               const float32_t* restrict x,
               const float32_t* restrict y,
               int M, int N, int P);
// Transposing API:
void mtx_mpyt16x16 ( void* pScr,
                     int16_t*  z, const int16_t*  x, const int16_t*  yt,
                     int M, int N, int P, int lsh );
void mtx_mpyt8x8 ( void* pScr,
                     int8_t*  z, const int8_t*  x, const int8_t*  yt,
                     int M, int N, int P, int lsh );
void mtx_mpyt8x16( void* pScr,
                     int16_t*  z, const int8_t*  x, const int16_t*  yt,
                     int M, int N, int P, int lsh );
void mtx_mpyt32x32 ( void* pScr,
                     int32_t*  z, const int32_t*  x, const int32_t*  yt,
                     int M, int N, int P, int lsh );
void mtx_mpytf (     void* pScr,
                     float32_t* z, const float32_t* x, const float32_t* yt,
                     int M, int N, int P);
void mtx_mpyt16x16_fast (  void* pScr,
                     int16_t*  z, const int16_t*  x, const int16_t*  yt,
                     int M, int N, int P, int lsh );
void mtx_mpyt8x8_fast (  void* pScr,
                     int8_t*  z, const int8_t*  x, const int8_t*  yt,
                     int M, int N, int P, int lsh );
void mtx_mpyt8x16_fast (  void* pScr,
                     int16_t*  z, const int8_t*  x, const int16_t*  yt,
                     int M, int N, int P, int lsh );
void mtx_mpyt32x32_fast (  void* pScr,
                     int32_t*  z, const int32_t*  x, const int32_t*  yt,
                     int M, int N, int P, int lsh );
void mtx_mpytf_fast( void* pScr,
                     float32_t* z, const float32_t* x, const float32_t* yt,
                     int M, int N, int P);
// scratch allocation functions
size_t mtx_mpy16x16_getScratchSize      (int M, int N, int P);
size_t mtx_mpy8x8_getScratchSize        (int M, int N, int P);
size_t mtx_mpy8x16_getScratchSize       (int M, int N, int P);
size_t mtx_mpy32x32_getScratchSize      (int M, int N, int P);
size_t mtx_mpyf_getScratchSize          (int M, int N, int P);
size_t mtx_mpy16x16_fast_getScratchSize (int M, int N, int P);
size_t mtx_mpy8x8_fast_getScratchSize   (int M, int N, int P);
size_t mtx_mpy8x16_fast_getScratchSize  (int M, int N, int P);
size_t mtx_mpy32x32_fast_getScratchSize (int M, int N, int P);
size_t mtx_mpyf_fast_getScratchSize     (int M, int N, int P);

size_t mtx_mpyt16x16_getScratchSize     (int M, int N, int P);
size_t mtx_mpyt8x8_getScratchSize       (int M, int N, int P);
size_t mtx_mpyt8x16_getScratchSize      (int M, int N, int P);
size_t mtx_mpyt32x32_getScratchSize     (int M, int N, int P);
size_t mtx_mpytf_getScratchSize         (int M, int N, int P);
size_t mtx_mpyt16x16_fast_getScratchSize(int M, int N, int P);
size_t mtx_mpyt8x8_fast_getScratchSize       (int M, int N, int P);
size_t mtx_mpyt8x16_fast_getScratchSize      (int M, int N, int P);
size_t mtx_mpyt32x32_fast_getScratchSize(int M, int N, int P);
size_t mtx_mpytf_fast_getScratchSize    (int M, int N, int P);

/*-------------------------------------------------------------------------
  Matrix by Vector Multiply
  These functions compute the expression z = 2^lsh * x * y for the matrices
  x and vector y.
  NOTE: lsh factor is not relevant for floating point routines.

  Two versions of functions available: regular version (mtx_vecmpy32x32,
  mtx_vecmpy16x16, mtx_vecmpy8x8, mtx_vecmpy8x16, mtx_vecmpyf) with arbitrary
  arguments and faster version (mtx_vecmpy32x32_fast, mtx_vecmpy16x16_fast,
  mtx_vecmpy8x8_fast, mtx_vecmpy8x16_fast,  mtx_vecmpyf_fast) that apply
  some restrictions

  Precision:
  32x32 32-bit input, 32-bit output
  16x16 16-bit input, 16-bit output
  8x8   8-bit inputs, 8-bit output
  8x16  8/16-bit inputs, 16-bit output
  f     floating point

  Input:
  x[M*N] input matrix,Q31,Q15 or floating point
  y[N]   input vector,Q31,Q15 or floating point
  M      number of rows in matrix x
  N      number of columns in matrix x
  lsh    additional left shift(applied to the fixed-
         point functions only)
  Output:
  z[M]   output vector,Q31,Q15 or floating point

  Restriction:
  For regular routines (mtx_vecmpy32x32, mtx_vecmpy16x16, mtx_vecmpy8x8,
  mtx_vecmpy8x16,  mtx_vecmpyf)
  x,y,z should not overlap

  For faster routines  (mtx_vecmpy32x32_fast, mtx_vecmpy16x16_fast,
  mtx_vecmpy8x8_fast, mtx_vecmpy8x16_fast, mtx_vecmpyf_fast)
  x,y,z   should not overlap
  x,y     aligned on 8-byte boundary
  N, M    multiples of 4
  lsh     should be in range:
          -31...31 for mtx_vecmpy32x32, mtx_vecmpy32x32_fast
          -15...15 for mtx_vecmpy16x16, mtx_vecmpy16x16_fast,
                   mtx_vecmpy8x8_fast, mtx_vecmpy8x16_fast
-------------------------------------------------------------------------*/
void mtx_vecmpy32x32 ( int32_t* restrict z,
                 const int32_t* restrict x,
                 const int32_t* restrict y,
                 int M, int N, int lsh);
void mtx_vecmpy16x16 ( int16_t* restrict z,
                 const int16_t* restrict x,
                 const int16_t* restrict y,
                 int M, int N, int lsh);
void mtx_vecmpy8x8 ( int8_t*  z,
               const int8_t*  x,
               const int8_t*  y,
               int M, int N, int lsh);
void mtx_vecmpy8x16( int16_t*  z,
               const int8_t *  x,
               const int16_t*  y,
               int M, int N, int lsh);
void mtx_vecmpy32x32_fast ( int32_t* restrict z,
                      const int32_t* restrict x,
                      const int32_t* restrict y,
                      int M, int N, int lsh);
void mtx_vecmpy16x16_fast ( int16_t* restrict z,
                      const int16_t* restrict x,
                      const int16_t* restrict y,
                      int M, int N, int lsh);
void mtx_vecmpy8x8_fast ( int8_t*  z,
               const int8_t*  x,
               const int8_t*  y,
               int M, int N, int lsh);
void mtx_vecmpy8x16_fast ( int16_t*  z,
               const int8_t *  x,
               const int16_t*  y,
               int M, int N, int lsh);
void mtx_vecmpyf ( float32_t* restrict z,
             const float32_t* restrict x,
             const float32_t* restrict y,
             int M, int N);
void mtx_vecmpyf_fast ( float32_t* restrict z,
                  const float32_t* restrict x,
                  const float32_t* restrict y,
                  int M, int N);

/*-------------------------------------------------------------------------
  Matrix Transpose
  These functions transpose matrices.

  Precision:
  32x32 32-bit input, 32-bit output
  16x16 16-bit input, 16-bit output
  8x8   8-bit inputs, 8-bit output
  f     floating point

  Input:
  x[M][N] input matrix,Q31,Q15,Q7 or floating point
  M       number of rows in matrix x
  N       number of columns in matrix x
  Output:
  y[N][M] output vector,Q31,Q15,Q7 or floating point

  Restriction:
  For regular routines (mtx_transpose_32x32, mtx_transpose_16x16,
  mtx_transpose_8x8, mtx_transposef):
  x,y should not overlap

  For faster routines (mtx_transpose 32x32_fast, mtx_transpose 16x16_fast,
  mtx_transpose_8x8_fast, mtx_transposef_fast)
  x,y   should not overlap
  x,y   aligned on 8-byte boundary
  N and M are multiples of 4
-------------------------------------------------------------------------*/
void mtx_transpose32x32 (int32_t*    y, const int32_t*     x, int M, int N);
void mtx_transpose16x16 (int16_t*    y, const int16_t*     x, int M, int N);
void mtx_transpose8x8   (int8_t *    y, const int8_t *     x, int M, int N);
void mtx_transposef     (float32_t*  y, const float32_t *  x, int M, int N);

void mtx_transpose32x32_fast (int32_t  *  y, const int32_t*     x, int M, int N);
void mtx_transpose16x16_fast (int16_t  *  y, const int16_t*     x, int M, int N);
void mtx_transpose8x8_fast   (int8_t   *  y, const int8_t *     x, int M, int N);
void mtx_transposef_fast     (float32_t*  y, const float32_t *  x, int M, int N);

#ifdef __cplusplus
}
#endif

#endif/* __NATUREDSP_SIGNAL_MATOP_H__ */
