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
#include "xa_nnlib_common_macros.h"

WORD32 xa_nn_matXvec_8x16_16(
    WORD16 * __restrict__ p_out,           /* output */
    WORD8 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
    WORD8 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
    WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
    WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
    WORD16 * __restrict__ p_bias,          /* bias */
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,                    /* row stride for matrix1 */
    WORD32 row_stride2,                    /* row stride for matrix2 */
    WORD32 acc_shift,                        /* out accumulator shift amount */
    WORD32 bias_shift)                       /* bias shift amount */
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, (ALIGNMENT>>1), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND(((cols1&3) != 0), -1);
  XA_NNLIB_ARG_CHK_COND(((row_stride1&3) != 0), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, (ALIGNMENT>>1), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND(((cols2&3) != 0), -1);
    XA_NNLIB_ARG_CHK_COND(((row_stride2&3) != 0), -1);
  }

  /* Iterators used in for loops */
  int m_itr, c_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;

#define UNROLL_SETUP_ACC          SETUP_ACC_FOR_8bx16b
#define UNROLL_SETUP_MAT1         SETUP_MAT1_8b
#define UNROLL_SETUP_MAT2         SETUP_MAT2_8b
#define UNROLL_KERNEL_MAT1_VEC1   KERNEL_MAT1_VEC1_8b_16b
#define UNROLL_KERNEL_MAT2_VEC2   KERNEL_MAT2_VEC2_8b_16b
#define UNROLL_STORE_ACC          STORE_ACC_8bx16b_AT_OUT_16b
#define SETUP_VEC1                SETUP_VEC1_16b
#define SETUP_VEC2                SETUP_VEC2_16b
#define LOAD_VEC1                 LOAD_VEC1_16b
#define LOAD_VEC2                 LOAD_VEC2_16b
#define SETUP_BIAS                SETUP_BIAS_16b
#define UNROLL_ADD_BIAS_ACC       ADD_BIAS_16b_ACC_FOR_8bx16b
#if !XA_HAVE_HIFI3_CORE
#define SETUP_VEC1_X2             SETUP_VEC1_16b_x2
#define SETUP_VEC2_X2             SETUP_VEC2_16b_x2
#define LOAD_VEC1_X2              LOAD_VEC1_16b_x2
#define LOAD_VEC2_X2              LOAD_VEC2_16b_x2
#define UNROLL_SETUP_MAT1_X2      SETUP_MAT1_8b_x2
#define UNROLL_SETUP_MAT2_X2      SETUP_MAT2_8b_x2
#define UNROLL_KERNEL_MAT1_VEC1_X2    KERNEL_MAT1_VEC1_8b_16b_x2
#define UNROLL_KERNEL_MAT2_VEC2_X2    KERNEL_MAT2_VEC2_8b_16b_x2
#endif

  ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(WORD8, WORD16, WORD16);

  if (p_mat2 && p_vec2)
  {
    /* All four pointers are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
      {
        SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE        
        SETUP_VEC1_X2; SETUP_MAT1_X2;
        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
        {
          LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
        }
        if((cols1 & (7)) !=0){
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        }
        SETUP_VEC2_X2; SETUP_MAT2_X2;
        for(c_itr = 0; c_itr < (cols2 >> 3); c_itr++)
        {
          LOAD_VEC2_X2 ; KERNEL_MAT2_VEC2_X2 ;
        }
        if((cols2 & (7)) !=0){
          LOAD_VEC2; KERNEL_MAT2_VEC2;
        }
#else
        SETUP_VEC1; SETUP_MAT1;
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        }
        SETUP_VEC2; SETUP_MAT2;
        for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
        {
          LOAD_VEC2; KERNEL_MAT2_VEC2;
        } 
#endif        
        ADD_BIAS_ACC; STORE_ACC;
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
        }
        SETUP_VEC2; UNROLL_SETUP_MAT2(0);
        for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
        {
          LOAD_VEC2; UNROLL_KERNEL_MAT2_VEC2(0);
        }
        UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
      }
    }
  }
  else
  {
    /* Only mat1, vec1 are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
      {
        SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE         
        SETUP_VEC1_X2; SETUP_MAT1_X2;
        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
        {
          LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
        }
        if((cols1 & (7)) !=0){
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        }
#else
        SETUP_VEC1; SETUP_MAT1;
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        }   
#endif        
        ADD_BIAS_ACC; STORE_ACC;
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
        }
        UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
      }
    }
  }

  /* Undefining the defined macro to make them available for reuse */

#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef UNROLL_KERNEL_MAT1_VEC1
#undef UNROLL_KERNEL_MAT2_VEC2
#undef UNROLL_STORE_ACC
#undef SETUP_VEC1
#undef SETUP_VEC2
#undef LOAD_VEC1
#undef LOAD_VEC2
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
#if !XA_HAVE_HIFI3_CORE
#undef UNROLL_SETUP_MAT1_X2
#undef UNROLL_SETUP_MAT2_X2
#undef UNROLL_KERNEL_MAT1_VEC1_X2
#undef UNROLL_KERNEL_MAT2_VEC2_X2
#undef SETUP_VEC1_X2
#undef SETUP_VEC2_X2
#undef LOAD_VEC1_X2
#undef LOAD_VEC2_X2
#endif
  return 0;
}

WORD32 xa_nn_matXvec_8x16_32(
    WORD32 * __restrict__ p_out,           /* output */
    WORD8 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
    WORD8 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
    WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
    WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
    WORD16 * __restrict__ p_bias,          /* bias */
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,                    /* row stride for matrix1 */
    WORD32 row_stride2,                    /* row stride for matrix2 */
    WORD32 acc_shift,                        /* out accumulator shift amount */
    WORD32 bias_shift)                       /* bias shift amount */

{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, (ALIGNMENT>>1), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND(((cols1&3) != 0), -1);
  XA_NNLIB_ARG_CHK_COND(((row_stride1&3) != 0), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, (ALIGNMENT>>1), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND(((cols2&3) != 0), -1);
    XA_NNLIB_ARG_CHK_COND(((row_stride2&3) != 0), -1);
  }

  /* Iterators used in for loops */
  int m_itr, c_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;

#define UNROLL_SETUP_ACC            SETUP_ACC_FOR_8bx16b
#define UNROLL_SETUP_MAT1           SETUP_MAT1_8b
#define UNROLL_SETUP_MAT2           SETUP_MAT2_8b
#define UNROLL_KERNEL_MAT1_VEC1     KERNEL_MAT1_VEC1_8b_16b
#define UNROLL_KERNEL_MAT2_VEC2     KERNEL_MAT2_VEC2_8b_16b
#define UNROLL_STORE_ACC            STORE_ACC_8bx16b_AT_OUT_32b
#define SETUP_VEC1                  SETUP_VEC1_16b
#define SETUP_VEC2                  SETUP_VEC2_16b
#define LOAD_VEC1                   LOAD_VEC1_16b
#define LOAD_VEC2                   LOAD_VEC2_16b
#define SETUP_BIAS                  SETUP_BIAS_16b
#define UNROLL_ADD_BIAS_ACC         ADD_BIAS_16b_ACC_FOR_8bx16b
#if !XA_HAVE_HIFI3_CORE
#define SETUP_VEC1_X2               SETUP_VEC1_16b_x2
#define SETUP_VEC2_X2               SETUP_VEC2_16b_x2
#define LOAD_VEC1_X2                LOAD_VEC1_16b_x2
#define LOAD_VEC2_X2                LOAD_VEC2_16b_x2
#define UNROLL_SETUP_MAT1_X2        SETUP_MAT1_8b_x2
#define UNROLL_SETUP_MAT2_X2        SETUP_MAT2_8b_x2
#define UNROLL_KERNEL_MAT1_VEC1_X2  KERNEL_MAT1_VEC1_8b_16b_x2
#define UNROLL_KERNEL_MAT2_VEC2_X2  KERNEL_MAT2_VEC2_8b_16b_x2
#endif
  ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(WORD8, WORD16, WORD32);

  if (p_mat2 && p_vec2)
  {
    /* All four pointers are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
      {
        SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE        
        SETUP_VEC1_X2; SETUP_MAT1_X2;
        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
        {
          LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
        }
        if((cols1 & (7)) !=0){
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        }
        SETUP_VEC2_X2; SETUP_MAT2_X2;
        for(c_itr = 0; c_itr < (cols2 >> 3); c_itr++)
        {
          LOAD_VEC2_X2 ; KERNEL_MAT2_VEC2_X2 ;
        }
        if((cols2 & (7)) !=0){
          LOAD_VEC2; KERNEL_MAT2_VEC2;
        }
#else
        SETUP_VEC1; SETUP_MAT1;
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        }
        SETUP_VEC2; SETUP_MAT2;
        for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
        {
          LOAD_VEC2; KERNEL_MAT2_VEC2;
        } 
#endif        
        ADD_BIAS_ACC; STORE_ACC;
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
        }

        SETUP_VEC2; UNROLL_SETUP_MAT2(0);
        for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
        {
          LOAD_VEC2; UNROLL_KERNEL_MAT2_VEC2(0);
        }
        UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
      }
    }
  }
  else
  {
    /* Only mat1, vec1 are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
      {
        SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE        
        SETUP_VEC1_X2; SETUP_MAT1_X2;
        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
        {
          LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
        }
        if((cols1 & (7)) !=0){
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        }
#else
        SETUP_VEC1; SETUP_MAT1;
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        } 
#endif        
        ADD_BIAS_ACC; STORE_ACC;
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
        }
        UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
      }
    }
  }

  /* Undefining the defined macro to make them available for reuse */
#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef UNROLL_KERNEL_MAT1_VEC1
#undef UNROLL_KERNEL_MAT2_VEC2
#undef UNROLL_STORE_ACC
#undef SETUP_VEC1
#undef SETUP_VEC2
#undef LOAD_VEC1
#undef LOAD_VEC2
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
#if !XA_HAVE_HIFI3_CORE
#undef UNROLL_SETUP_MAT1_X2
#undef UNROLL_SETUP_MAT2_X2
#undef UNROLL_KERNEL_MAT1_VEC1_X2
#undef UNROLL_KERNEL_MAT2_VEC2_X2
#undef SETUP_VEC1_X2
#undef SETUP_VEC2_X2
#undef LOAD_VEC1_X2
#undef LOAD_VEC2_X2
#endif
  return 0;
}


WORD32 xa_nn_matXvec_8x16_64(
    WORD64 * __restrict__ p_out,           /* output */
    WORD8 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
    WORD8 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
    WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
    WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
    WORD16 * __restrict__ p_bias,          /* bias */
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,                    /* row stride for matrix1 */
    WORD32 row_stride2,                    /* row stride for matrix2 */
    WORD32 acc_shift,                        /* out accumulator shift amount */
    WORD32 bias_shift)                       /* bias shift amount */
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD64), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, (ALIGNMENT>>1), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND(((cols1&3) != 0), -1);
  XA_NNLIB_ARG_CHK_COND(((row_stride1&3) != 0), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, (ALIGNMENT>>1), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND(((cols2&3) != 0), -1);
    XA_NNLIB_ARG_CHK_COND(((row_stride2&3) != 0), -1);
  }

  /* Iterators used in for loops */
  int m_itr, c_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;

#define UNROLL_SETUP_ACC            SETUP_ACC_FOR_8bx16b
#define UNROLL_SETUP_MAT1           SETUP_MAT1_8b
#define UNROLL_SETUP_MAT2           SETUP_MAT2_8b
#define UNROLL_KERNEL_MAT1_VEC1     KERNEL_MAT1_VEC1_8b_16b
#define UNROLL_KERNEL_MAT2_VEC2     KERNEL_MAT2_VEC2_8b_16b
#define UNROLL_STORE_ACC            STORE_ACC_8bx16b_AT_OUT_64b
#define SETUP_VEC1                  SETUP_VEC1_16b
#define SETUP_VEC2                  SETUP_VEC2_16b
#define LOAD_VEC1                   LOAD_VEC1_16b
#define LOAD_VEC2                   LOAD_VEC2_16b
#define SETUP_BIAS                  SETUP_BIAS_16b
#define UNROLL_ADD_BIAS_ACC         ADD_BIAS_16b_ACC_FOR_8bx16b
#if !XA_HAVE_HIFI3_CORE
#define SETUP_VEC1_X2               SETUP_VEC1_16b_x2
#define SETUP_VEC2_X2               SETUP_VEC2_16b_x2
#define LOAD_VEC1_X2                LOAD_VEC1_16b_x2
#define LOAD_VEC2_X2                LOAD_VEC2_16b_x2
#define UNROLL_SETUP_MAT1_X2        SETUP_MAT1_8b_x2
#define UNROLL_SETUP_MAT2_X2        SETUP_MAT2_8b_x2
#define UNROLL_KERNEL_MAT1_VEC1_X2  KERNEL_MAT1_VEC1_8b_16b_x2
#define UNROLL_KERNEL_MAT2_VEC2_X2  KERNEL_MAT2_VEC2_8b_16b_x2
#endif
  if (p_mat2 && p_vec2)
  {
    /* All four pointers are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
      {
        SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE        
        SETUP_VEC1_X2; SETUP_MAT1_X2;
        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
        {
          LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
        }
        if((cols1 & (7)) !=0){
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        }
        SETUP_VEC2_X2; SETUP_MAT2_X2;
        for(c_itr = 0; c_itr < (cols2 >> 3); c_itr++)
        {
          LOAD_VEC2_X2 ; KERNEL_MAT2_VEC2_X2 ;
        }
        if((cols2 & (7)) !=0){
          LOAD_VEC2; KERNEL_MAT2_VEC2;
        }
#else
        SETUP_VEC1; SETUP_MAT1;
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        }
        SETUP_VEC2; SETUP_MAT2;
        for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
        {
          LOAD_VEC2; KERNEL_MAT2_VEC2;
        } 
#endif        
        ADD_BIAS_ACC; STORE_ACC;
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
        }

        SETUP_VEC2; UNROLL_SETUP_MAT2(0);
        for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
        {
          LOAD_VEC2; UNROLL_KERNEL_MAT2_VEC2(0);
        }
        UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
      }
    }
  }
  else
  {
    /* Only mat1, vec1 are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
      {
        SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE        
        SETUP_VEC1_X2; SETUP_MAT1_X2;
        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
        {
          LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
        }
        if((cols1 & (7)) !=0){
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        }
#else
        SETUP_VEC1; SETUP_MAT1;
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; KERNEL_MAT1_VEC1;
        } 
#endif        
        ADD_BIAS_ACC; STORE_ACC;
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
        }
        UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
      }
    }
  }
  /* Undefining the defined macro to make them available for reuse */
#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef UNROLL_KERNEL_MAT1_VEC1
#undef UNROLL_KERNEL_MAT2_VEC2
#undef UNROLL_STORE_ACC
#undef SETUP_VEC1
#undef SETUP_VEC2
#undef LOAD_VEC1
#undef LOAD_VEC2
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
#if !XA_HAVE_HIFI3_CORE
#undef UNROLL_SETUP_MAT1_X2
#undef UNROLL_SETUP_MAT2_X2
#undef UNROLL_KERNEL_MAT1_VEC1_X2
#undef UNROLL_KERNEL_MAT2_VEC2_X2
#undef SETUP_VEC1_X2
#undef SETUP_VEC2_X2
#undef LOAD_VEC1_X2
#undef LOAD_VEC2_X2
#endif
  return 0;
}

WORD32 xa_nn_matXvec_8x16_16_tanh(
    WORD16 * __restrict__ p_out,     /* output */
    WORD8 * __restrict__ p_mat1,     /* matrix1: rows x cols1 */
    WORD8 * __restrict__ p_mat2,     /* matrix2: rows x cols2 */
    WORD16 * __restrict__ p_vec1,    /* vec1: cols1 x 1 */
    WORD16 * __restrict__ p_vec2,    /* vec2: cols2 x 1 */
    VOID   * __restrict__ p_bias,    /* bias */
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,              /* row stride for matrix1 */
    WORD32 row_stride2,              /* row stride for matrix2 */
    WORD32 acc_shift,                  /* out accumulator shift amount */
    WORD32 bias_shift,                 /* bias shift amount */
    WORD32 bias_precision,           /* 16 or 64 */
    VOID   * __restrict__ p_scratch) /* Scratch pointer arg, only if required */
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, (ALIGNMENT>>1), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_precision != 16 && bias_precision != 64), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND(((cols1&3) != 0), -1);
  XA_NNLIB_ARG_CHK_COND(((row_stride1&3) != 0), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, (ALIGNMENT>>1), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND(((cols2&3) != 0), -1);
    XA_NNLIB_ARG_CHK_COND(((row_stride2&3) != 0), -1);
  }

  /* Iterators used in for loops */
  int m_itr, c_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;

#define UNROLL_SETUP_ACC            SETUP_ACC_FOR_8bx16b
#define UNROLL_SETUP_MAT1           SETUP_MAT1_8b
#define UNROLL_SETUP_MAT2           SETUP_MAT2_8b
#define UNROLL_KERNEL_MAT1_VEC1     KERNEL_MAT1_VEC1_8b_16b
#define UNROLL_KERNEL_MAT2_VEC2     KERNEL_MAT2_VEC2_8b_16b
#define UNROLL_STORE_ACC            STORE_ACC_8bx16b_AT_SCRATCH_32b
#define SETUP_VEC1                  SETUP_VEC1_16b
#define SETUP_VEC2                  SETUP_VEC2_16b
#define LOAD_VEC1                   LOAD_VEC1_16b
#define LOAD_VEC2                   LOAD_VEC2_16b
#if !XA_HAVE_HIFI3_CORE
#define SETUP_VEC1_X2               SETUP_VEC1_16b_x2
#define SETUP_VEC2_X2               SETUP_VEC2_16b_x2
#define LOAD_VEC1_X2                LOAD_VEC1_16b_x2
#define LOAD_VEC2_X2                LOAD_VEC2_16b_x2
#define UNROLL_SETUP_MAT1_X2        SETUP_MAT1_8b_x2
#define UNROLL_SETUP_MAT2_X2        SETUP_MAT2_8b_x2
#define UNROLL_KERNEL_MAT1_VEC1_X2  KERNEL_MAT1_VEC1_8b_16b_x2
#define UNROLL_KERNEL_MAT2_VEC2_X2  KERNEL_MAT2_VEC2_8b_16b_x2
#endif

  ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(WORD8, WORD16, WORD32);

  switch(bias_precision)
  {
    default:
    case 16:
      {
#define SETUP_BIAS              SETUP_BIAS_16b
#define UNROLL_ADD_BIAS_ACC     ADD_BIAS_16b_ACC_FOR_8bx16b
        if (p_mat2 && p_vec2)
        {
          /* All four pointers are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
            {
              SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE              
              SETUP_VEC1_X2; SETUP_MAT1_X2;
              for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
              {
                LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
              }
              if((cols1 & (7)) !=0){
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
              SETUP_VEC2_X2; SETUP_MAT2_X2;
              for(c_itr = 0; c_itr < (cols2 >> 3); c_itr++)
              {
                LOAD_VEC2_X2 ; KERNEL_MAT2_VEC2_X2 ;
              }
              if((cols2 & (7)) !=0){
                LOAD_VEC2; KERNEL_MAT2_VEC2;
              }
#else
              SETUP_VEC1; SETUP_MAT1;
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
              SETUP_VEC2; SETUP_MAT2;
              for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
              {
                LOAD_VEC2; KERNEL_MAT2_VEC2;
              } 
#endif              
              ADD_BIAS_ACC; STORE_ACC;
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {
              UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
              }

              SETUP_VEC2; UNROLL_SETUP_MAT2(0);
              for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
              {
                LOAD_VEC2; UNROLL_KERNEL_MAT2_VEC2(0);
              }
              UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
            }
          }
        }
        else
        {
          /* Only mat1, vec1 are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
            {
              SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE              
              SETUP_VEC1_X2; SETUP_MAT1_X2;
              for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
              {
                LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
              }
              if((cols1 & (7)) !=0){
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
#else
              SETUP_VEC1; SETUP_MAT1;
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              } 
#endif              
              ADD_BIAS_ACC; STORE_ACC;
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {
              UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
              }
              UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
            }
          }
        }

        break;
        /* Undefining the defined macro to make them available for reuse */
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
      }
    case 64:
      {
#define SETUP_BIAS              SETUP_BIAS_64b
#define UNROLL_ADD_BIAS_ACC     ADD_BIAS_64b_ACC_FOR_8bx16b
        if (p_mat2 && p_vec2)
        {
          /* All four pointers are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
            {
              SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE              
              SETUP_VEC1_X2; SETUP_MAT1_X2;
              for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
              {
                LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
              }
              if((cols1 & (7)) !=0){
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
              SETUP_VEC2_X2; SETUP_MAT2_X2;
              for(c_itr = 0; c_itr < (cols2 >> 3); c_itr++)
              {
                LOAD_VEC2_X2 ; KERNEL_MAT2_VEC2_X2 ;
              }
              if((cols2 & (7)) !=0){
                LOAD_VEC2; KERNEL_MAT2_VEC2;
              }
#else
              SETUP_VEC1; SETUP_MAT1;
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
              SETUP_VEC2; SETUP_MAT2;
              for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
              {
                LOAD_VEC2; KERNEL_MAT2_VEC2;
              } 
#endif              
              ADD_BIAS_ACC; STORE_ACC;
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {
              UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
              }

              SETUP_VEC2; UNROLL_SETUP_MAT2(0);
              for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
              {
                LOAD_VEC2; UNROLL_KERNEL_MAT2_VEC2(0);
              }
              UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
            }
          }
        }
        else
        {
          /* Only mat1, vec1 are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
            {
              SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE              
              SETUP_VEC1_X2; SETUP_MAT1_X2;
              for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
              {
                LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
              }
              if((cols1 & (7)) !=0){
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
#else
              SETUP_VEC1; SETUP_MAT1;
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              } 
#endif              
              ADD_BIAS_ACC; STORE_ACC;
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {
              UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
              }
              UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
            }
          }
        }

        break;
        /* Undefining the defined macro to make them available for reuse */
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
      }
#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef UNROLL_KERNEL_MAT1_VEC1
#undef UNROLL_KERNEL_MAT2_VEC2
#undef UNROLL_STORE_ACC
#undef SETUP_VEC1
#undef SETUP_VEC2
#undef LOAD_VEC1
#undef LOAD_VEC2
#if !XA_HAVE_HIFI3_CORE
#undef UNROLL_SETUP_MAT1_X2
#undef UNROLL_SETUP_MAT2_X2
#undef UNROLL_KERNEL_MAT1_VEC1_X2
#undef UNROLL_KERNEL_MAT2_VEC2_X2
#undef SETUP_VEC1_X2
#undef SETUP_VEC2_X2
#undef LOAD_VEC1_X2
#undef LOAD_VEC2_X2
#endif
  }

  xa_nn_vec_tanh_32_16((pWORD16) p_out, (pWORD32) p_scratch, rows);

  return 0;
}

WORD32 xa_nn_matXvec_8x16_16_sigmoid(
    WORD16 * __restrict__ p_out,     /* output */
    WORD8 * __restrict__ p_mat1,     /* matrix1: rows x cols1 */
    WORD8 * __restrict__ p_mat2,     /* matrix2: rows x cols2 */
    WORD16 * __restrict__ p_vec1,    /* vec1: cols1 x 1 */
    WORD16 * __restrict__ p_vec2,    /* vec2: cols2 x 1 */
    VOID   * __restrict__ p_bias,    /* bias */
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,              /* row stride for matrix1 */
    WORD32 row_stride2,              /* row stride for matrix2 */
    WORD32 acc_shift,                  /* out accumulator shift amount */
    WORD32 bias_shift,                 /* bias shift amount */
    WORD32 bias_precision,           /* 16 or 64 */
    VOID   * __restrict__ p_scratch) /* Scratch pointer arg, only if required */
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, (ALIGNMENT>>1), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_precision != 16 && bias_precision != 64), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND(((cols1&3) != 0), -1);
  XA_NNLIB_ARG_CHK_COND(((row_stride1&3) != 0), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, (ALIGNMENT>>1), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND(((cols2&3) != 0), -1);
    XA_NNLIB_ARG_CHK_COND(((row_stride2&3) != 0), -1);
  }

  /* Iterators used in for loops */
  int m_itr, c_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;

#define UNROLL_SETUP_ACC            SETUP_ACC_FOR_8bx16b
#define UNROLL_SETUP_MAT1           SETUP_MAT1_8b
#define UNROLL_SETUP_MAT2           SETUP_MAT2_8b
#define UNROLL_KERNEL_MAT1_VEC1     KERNEL_MAT1_VEC1_8b_16b
#define UNROLL_KERNEL_MAT2_VEC2     KERNEL_MAT2_VEC2_8b_16b
#define UNROLL_STORE_ACC            STORE_ACC_8bx16b_AT_SCRATCH_32b
#define SETUP_VEC1                  SETUP_VEC1_16b
#define SETUP_VEC2                  SETUP_VEC2_16b
#define LOAD_VEC1                   LOAD_VEC1_16b
#define LOAD_VEC2                   LOAD_VEC2_16b
#if !XA_HAVE_HIFI3_CORE
#define SETUP_VEC1_X2               SETUP_VEC1_16b_x2
#define SETUP_VEC2_X2               SETUP_VEC2_16b_x2
#define LOAD_VEC1_X2                LOAD_VEC1_16b_x2
#define LOAD_VEC2_X2                LOAD_VEC2_16b_x2
#define UNROLL_SETUP_MAT1_X2        SETUP_MAT1_8b_x2
#define UNROLL_SETUP_MAT2_X2        SETUP_MAT2_8b_x2
#define UNROLL_KERNEL_MAT1_VEC1_X2  KERNEL_MAT1_VEC1_8b_16b_x2
#define UNROLL_KERNEL_MAT2_VEC2_X2  KERNEL_MAT2_VEC2_8b_16b_x2
#endif

  ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(WORD8, WORD16, WORD32);

  switch(bias_precision)
  {
    default:
    case 16:
      {
#define SETUP_BIAS              SETUP_BIAS_16b
#define UNROLL_ADD_BIAS_ACC     ADD_BIAS_16b_ACC_FOR_8bx16b
        if (p_mat2 && p_vec2)
        {
          /* All four pointers are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
            {
              SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE              
              SETUP_VEC1_X2; SETUP_MAT1_X2;
              for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
              {
                LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
              }
              if((cols1 & (7)) !=0){
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
              SETUP_VEC2_X2; SETUP_MAT2_X2;
              for(c_itr = 0; c_itr < (cols2 >> 3); c_itr++)
              {
                LOAD_VEC2_X2 ; KERNEL_MAT2_VEC2_X2 ;
              }
              if((cols2 & (7)) !=0){
                LOAD_VEC2; KERNEL_MAT2_VEC2;
              }
#else
              SETUP_VEC1; SETUP_MAT1;
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
              SETUP_VEC2; SETUP_MAT2;
              for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
              {
                LOAD_VEC2; KERNEL_MAT2_VEC2;
              } 
#endif              
              ADD_BIAS_ACC; STORE_ACC;
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {
              UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
              }

              SETUP_VEC2; UNROLL_SETUP_MAT2(0);
              for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
              {
                LOAD_VEC2; UNROLL_KERNEL_MAT2_VEC2(0);
              }
              UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
            }
          }
        }
        else
        {
          /* Only mat1, vec1 are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
            {
              SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE              
              SETUP_VEC1_X2; SETUP_MAT1_X2;
              for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
              {
                LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
              }
              if((cols1 & (7)) !=0){
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
#else
              SETUP_VEC1; SETUP_MAT1;
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              } 
#endif              
              ADD_BIAS_ACC; STORE_ACC;
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {
              UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
              }
              UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
            }
          }
        }

        break;
        /* Undefining the defined macro to make them available for reuse */
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
      }
    case 64:
      {
#define SETUP_BIAS              SETUP_BIAS_64b
#define UNROLL_ADD_BIAS_ACC     ADD_BIAS_64b_ACC_FOR_8bx16b
        if (p_mat2 && p_vec2)
        {
          /* All four pointers are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
            {
              SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE              
              SETUP_VEC1_X2; SETUP_MAT1_X2;
              for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
              {
                LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
              }
              if((cols1 & (7)) !=0){
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
              SETUP_VEC2_X2; SETUP_MAT2_X2;
              for(c_itr = 0; c_itr < (cols2 >> 3); c_itr++)
              {
                LOAD_VEC2_X2 ; KERNEL_MAT2_VEC2_X2 ;
              }
              if((cols2 & (7)) !=0){
                LOAD_VEC2; KERNEL_MAT2_VEC2;
              }
#else
              SETUP_VEC1; SETUP_MAT1;
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
              SETUP_VEC2; SETUP_MAT2;
              for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
              {
                LOAD_VEC2; KERNEL_MAT2_VEC2;
              } 
#endif              
              ADD_BIAS_ACC; STORE_ACC;
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {
              UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
              }

              SETUP_VEC2; UNROLL_SETUP_MAT2(0);
              for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
              {
                LOAD_VEC2; UNROLL_KERNEL_MAT2_VEC2(0);
              }
              UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
            }
          }
        }
        else
        {
          /* Only mat1, vec1 are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
            {
              SETUP_ACC; 
#if !XA_HAVE_HIFI3_CORE              
              SETUP_VEC1_X2; SETUP_MAT1_X2;
              for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
              {
                LOAD_VEC1_X2; KERNEL_MAT1_VEC1_X2;
              }
              if((cols1 & (7)) !=0){
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              }
#else
              SETUP_VEC1; SETUP_MAT1;
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; KERNEL_MAT1_VEC1;
              } 
#endif              
              ADD_BIAS_ACC; STORE_ACC;
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {
              UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0) ;
              }
              UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
            }
          }
        }
        break;
        /* Undefining the defined macro to make them available for reuse */
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
      }
#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef UNROLL_KERNEL_MAT1_VEC1
#undef UNROLL_KERNEL_MAT2_VEC2
#undef UNROLL_STORE_ACC
#undef SETUP_VEC1
#undef SETUP_VEC2
#undef LOAD_VEC1
#undef LOAD_VEC2
#if !XA_HAVE_HIFI3_CORE
#undef UNROLL_SETUP_MAT1_X2
#undef UNROLL_SETUP_MAT2_X2
#undef UNROLL_KERNEL_MAT1_VEC1_X2
#undef UNROLL_KERNEL_MAT2_VEC2_X2
#undef SETUP_VEC1_X2
#undef SETUP_VEC2_X2
#undef LOAD_VEC1_X2
#undef LOAD_VEC2_X2
#endif
  }

  xa_nn_vec_sigmoid_32_16((pWORD16) p_out, (pWORD32) p_scratch, rows);

  return 0;
}
