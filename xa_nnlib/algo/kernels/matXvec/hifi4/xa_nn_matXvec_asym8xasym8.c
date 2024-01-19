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

#ifdef ROW_UNROLL
#undef ROW_UNROLL
#endif
#define ROW_UNROLL  4

#include "xa_nnlib_common_macros.h"

WORD32 xa_nn_matXvec_asym8xasym8_asym8(
    UWORD8 * __restrict__ p_out,
    const UWORD8 * __restrict__ p_mat1,
    const UWORD8 * __restrict__ p_mat2,
    const UWORD8 * __restrict__ p_vec1,
    const UWORD8 * __restrict__ p_vec2,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    WORD32 mat1_zero_bias,
    WORD32 mat2_zero_bias,
    WORD32 vec1_zero_bias,
    WORD32 vec2_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((mat1_zero_bias < -255 || mat1_zero_bias > 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -255 || vec1_zero_bias > 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < 0 || out_zero_bias > 255), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
    XA_NNLIB_ARG_CHK_COND((mat2_zero_bias < -255 || mat2_zero_bias > 0), -1);
    XA_NNLIB_ARG_CHK_COND((vec2_zero_bias < -255 || vec2_zero_bias > 0), -1);
  }

  if(((((uintptr_t)p_out)&7) == 0) && ((((uintptr_t)p_mat1)&7) == 0) && ((((uintptr_t)p_mat2)&7) == 0) &&
      ((((uintptr_t)p_vec1)&7) == 0) && ((((uintptr_t)p_vec2)&7) == 0) && ((((uintptr_t)p_bias)&3) == 0) &&
      ((cols1&3) == 0) && ((cols2&3) == 0) && ((row_stride1&3) == 0) && ((row_stride2&3) == 0))
  {
    /* Iterators used in for loops */
    int m_itr, c_itr;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    /* Shifts to match with Tensorflow */
    int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
    left_shift = out_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    left_shift = out_shift<0?0:out_shift;
    right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

#define UNROLL_SETUP_ACC            SETUP_ACC_FOR_ASYM8bxASYM8b
#define UNROLL_SETUP_MAT1           SETUP_MAT1_ASYM8b
#define UNROLL_SETUP_MAT2           SETUP_MAT2_ASYM8b
#define UNROLL_KERNEL_MAT1_VEC1     KERNEL_MAT1_VEC1_ASYM8b_ASYM8b
#define UNROLL_KERNEL_MAT2_VEC2     KERNEL_MAT2_VEC2_ASYM8b_ASYM8b
#define UNROLL_ADJUST_ACC           ADJUST_ACC_ASYM8b
#define UNROLL_STORE_ACC            STORE_ACC_ASYM8bxASYM8b_AT_OUT_ASYM8b
#define SETUP_VEC1                  SETUP_VEC1_ASYM8b
#define SETUP_VEC2                  SETUP_VEC2_ASYM8b
#define LOAD_VEC1                   LOAD_VEC1_ASYM8b
#define LOAD_VEC2                   LOAD_VEC2_ASYM8b
#define SETUP_BIAS                  SETUP_BIAS_ASYM8b
#define UNROLL_ADD_BIAS_ACC         ADD_BIAS_ASYM8b_ACC_FOR_ASYM8bxASYM8b

    SETUP_BIAS;

    for(m_itr = 0; m_itr < (rows & ~3); m_itr += 4)
    {
      ae_int64 _ae_int64_acc_0, _ae_int64_acc_1, _ae_int64_acc_2, _ae_int64_acc_3;
      _ae_int64_acc_0 = _ae_int64_acc_1 = _ae_int64_acc_2 = _ae_int64_acc_3 = 0;

      ae_int16x4 _ae_int16x4_vec, _ae_int16x4_mat_0, _ae_int16x4_mat_1,_ae_int16x4_mat_2, _ae_int16x4_mat_3;
      _ae_int16x4_vec = _ae_int16x4_mat_0 = _ae_int16x4_mat_1 = _ae_int16x4_mat_2 = _ae_int16x4_mat_3 = 0;

      /* mat1*vec1 */
      WORD8 * __restrict__ p_vec1_0 = (WORD8 *) p_vec1;
      WORD8 * __restrict__ p_mat1_0 = (WORD8 *) &p_mat1[(m_itr+0)*row_stride1];
      WORD8 * __restrict__ p_mat1_1 = (WORD8 *) &p_mat1[(m_itr+1)*row_stride1];
      WORD8 * __restrict__ p_mat1_2 = (WORD8 *) &p_mat1[(m_itr+2)*row_stride1];
      WORD8 * __restrict__ p_mat1_3 = (WORD8 *) &p_mat1[(m_itr+3)*row_stride1];

      ae_int64 acc_e = 0;
#if XCHAL_HAVE_HIFI4
        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
        {
        ae_int16x4 _ae_int16x4_vec_I, _ae_int16x4_mat_0_I, _ae_int16x4_mat_1_I, _ae_int16x4_mat_2_I, _ae_int16x4_mat_3_I;

        _ae_int16x4_vec_I = AE_L8X4F_I(p_vec1_0, INCREMENT_IN_BYTES_FOR_WORD8X4);
        _ae_int16x4_vec_I = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec_I), 8));
        _ae_int16x4_vec_I = AE_ADD16(_ae_int16x4_vec_I, AE_MOVDA16(vec1_zero_bias));
        AE_L8X4F_IP(_ae_int16x4_vec, p_vec1_0, 2 * 4);
        _ae_int16x4_vec = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec), 8));
        _ae_int16x4_vec = AE_ADD16(_ae_int16x4_vec, AE_MOVDA16(vec1_zero_bias));

        _ae_int16x4_mat_0_I = AE_L8X4F_I(p_mat1_0, 4);
        _ae_int16x4_mat_0_I = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_0_I), 8));
        AE_L8X4F_IP(_ae_int16x4_mat_0, p_mat1_0, 2 * 4);
        _ae_int16x4_mat_0 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_0), 8));

        _ae_int16x4_mat_1_I = AE_L8X4F_I(p_mat1_1, 4);
        _ae_int16x4_mat_1_I = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_1_I), 8));
        AE_L8X4F_IP(_ae_int16x4_mat_1, p_mat1_1, 2 * 4);
        _ae_int16x4_mat_1 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_1), 8));

        _ae_int16x4_mat_2_I = AE_L8X4F_I(p_mat1_2, 4);
        _ae_int16x4_mat_2_I = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_2_I), 8));
        AE_L8X4F_IP(_ae_int16x4_mat_2, p_mat1_2, 2 * 4);
        _ae_int16x4_mat_2 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_2), 8));

        _ae_int16x4_mat_3_I = AE_L8X4F_I(p_mat1_3, 4);
        _ae_int16x4_mat_3_I = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_3_I), 8));
        AE_L8X4F_IP(_ae_int16x4_mat_3, p_mat1_3, 2 * 4);
        _ae_int16x4_mat_3 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_3), 8));

        AE_MULAAAAQ16(acc_e, _ae_int16x4_vec, AE_MOVDA16(mat1_zero_bias));
        AE_MULAAAAQ16(acc_e, _ae_int16x4_vec_I, AE_MOVDA16(mat1_zero_bias));
        AE_MULAAAAQ16(_ae_int64_acc_0, _ae_int16x4_vec, _ae_int16x4_mat_0);
        AE_MULAAAAQ16(_ae_int64_acc_0, _ae_int16x4_vec_I, _ae_int16x4_mat_0_I);
        AE_MULAAAAQ16(_ae_int64_acc_1, _ae_int16x4_vec, _ae_int16x4_mat_1);
        AE_MULAAAAQ16(_ae_int64_acc_1, _ae_int16x4_vec_I, _ae_int16x4_mat_1_I);
        AE_MULAAAAQ16(_ae_int64_acc_2, _ae_int16x4_vec, _ae_int16x4_mat_2);
        AE_MULAAAAQ16(_ae_int64_acc_2, _ae_int16x4_vec_I, _ae_int16x4_mat_2_I);
        AE_MULAAAAQ16(_ae_int64_acc_3, _ae_int16x4_vec, _ae_int16x4_mat_3);
        AE_MULAAAAQ16(_ae_int64_acc_3, _ae_int16x4_vec_I, _ae_int16x4_mat_3_I);
      }
      if((cols1 & 7) != 0)
      {
        AE_L8X4F_IP(_ae_int16x4_vec, p_vec1_0, INCREMENT_IN_BYTES_FOR_WORD8X4);
        _ae_int16x4_vec = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec), 8));
        _ae_int16x4_vec = AE_ADD16(_ae_int16x4_vec, AE_MOVDA16(vec1_zero_bias));
        AE_MULAAAAQ16(acc_e, _ae_int16x4_vec, AE_MOVDA16(mat1_zero_bias));

        AE_L8X4F_IP(_ae_int16x4_mat_0, p_mat1_0, 4);
        _ae_int16x4_mat_0 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_0), 8));
        AE_MULAAAAQ16(_ae_int64_acc_0, _ae_int16x4_vec, _ae_int16x4_mat_0);

        AE_L8X4F_IP(_ae_int16x4_mat_1, p_mat1_1, 4);
        _ae_int16x4_mat_1 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_1), 8));
        AE_MULAAAAQ16(_ae_int64_acc_1, _ae_int16x4_vec, _ae_int16x4_mat_1);

        AE_L8X4F_IP(_ae_int16x4_mat_2, p_mat1_2, 4);
        _ae_int16x4_mat_2 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_2), 8));
        AE_MULAAAAQ16(_ae_int64_acc_2, _ae_int16x4_vec, _ae_int16x4_mat_2);

        AE_L8X4F_IP(_ae_int16x4_mat_3, p_mat1_3, 4);
        _ae_int16x4_mat_3 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_3), 8));
        AE_MULAAAAQ16(_ae_int64_acc_3, _ae_int16x4_vec, _ae_int16x4_mat_3);
      }
#else /* XCHAL_HAVE_HIFI4 */
      for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
      {
#if XCHAL_HAVE_HIFI1
        AE_L8X4U_IP(_ae_int16x4_vec, p_vec1_0, INCREMENT_IN_BYTES_FOR_WORD8X4);
#else
        AE_L8X4F_IP(_ae_int16x4_vec, p_vec1_0, INCREMENT_IN_BYTES_FOR_WORD8X4);
        _ae_int16x4_vec = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec), 8));
#endif
        _ae_int16x4_vec = AE_ADD16(_ae_int16x4_vec, AE_MOVDA16(vec1_zero_bias));

        AE_MULAAAAQ16(acc_e, _ae_int16x4_vec, AE_MOVDA16(mat1_zero_bias));

#if XCHAL_HAVE_HIFI1
        AE_L8X4U_IP(_ae_int16x4_mat_0, p_mat1_0, 4);
#else
        AE_L8X4F_IP(_ae_int16x4_mat_0, p_mat1_0, 4);
        _ae_int16x4_mat_0 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_0), 8));
#endif
        AE_MULAAAAQ16(_ae_int64_acc_0, _ae_int16x4_vec, _ae_int16x4_mat_0);

#if XCHAL_HAVE_HIFI1
        AE_L8X4U_IP(_ae_int16x4_mat_1, p_mat1_1, 4);
#else
        AE_L8X4F_IP(_ae_int16x4_mat_1, p_mat1_1, 4);
        _ae_int16x4_mat_1 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_1), 8));
#endif
        AE_MULAAAAQ16(_ae_int64_acc_1, _ae_int16x4_vec, _ae_int16x4_mat_1);

#if XCHAL_HAVE_HIFI1
        AE_L8X4U_IP(_ae_int16x4_mat_2, p_mat1_2, 4);
#else
        AE_L8X4F_IP(_ae_int16x4_mat_2, p_mat1_2, 4);
        _ae_int16x4_mat_2 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_2), 8));
#endif
        AE_MULAAAAQ16(_ae_int64_acc_2, _ae_int16x4_vec, _ae_int16x4_mat_2);

#if XCHAL_HAVE_HIFI1
        AE_L8X4U_IP(_ae_int16x4_mat_3, p_mat1_3, 4);
#else
        AE_L8X4F_IP(_ae_int16x4_mat_3, p_mat1_3, 4);
        _ae_int16x4_mat_3 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_3), 8));
#endif
        AE_MULAAAAQ16(_ae_int64_acc_3, _ae_int16x4_vec, _ae_int16x4_mat_3);
      }
#endif /* XCHAL_HAVE_HIFI4  */
      _ae_int64_acc_0 = AE_ADD64(_ae_int64_acc_0, acc_e);
      _ae_int64_acc_1 = AE_ADD64(_ae_int64_acc_1, acc_e);
      _ae_int64_acc_2 = AE_ADD64(_ae_int64_acc_2, acc_e);
      _ae_int64_acc_3 = AE_ADD64(_ae_int64_acc_3, acc_e);


      if (p_mat2 && p_vec2){
        /* mat2*vec2 */
        WORD8 * __restrict__ p_vec2_0 = (WORD8 *) p_vec2;
        WORD8 * __restrict__ p_mat2_0 = (WORD8 *) &p_mat2[(m_itr+0)*row_stride2];
        WORD8 * __restrict__ p_mat2_1 = (WORD8 *) &p_mat2[(m_itr+1)*row_stride2];
        WORD8 * __restrict__ p_mat2_2 = (WORD8 *) &p_mat2[(m_itr+2)*row_stride2];
        WORD8 * __restrict__ p_mat2_3 = (WORD8 *) &p_mat2[(m_itr+3)*row_stride2];

        acc_e = 0;
        for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
        {
#if XCHAL_HAVE_HIFI1
          AE_L8X4U_IP(_ae_int16x4_vec, p_vec2_0, 4);
#else
          AE_L8X4F_IP(_ae_int16x4_vec, p_vec2_0, 4);
          _ae_int16x4_vec = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec), 8));
#endif
          _ae_int16x4_vec = AE_ADD16(_ae_int16x4_vec, AE_MOVDA16(vec2_zero_bias));

          AE_MULAAAAQ16(acc_e, _ae_int16x4_vec, AE_MOVDA16(mat2_zero_bias));

#if XCHAL_HAVE_HIFI1
          AE_L8X4U_IP(_ae_int16x4_mat_0, p_mat2_0, 4);
#else
          AE_L8X4F_IP(_ae_int16x4_mat_0, p_mat2_0, 4);
          _ae_int16x4_mat_0 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_0), 8));
#endif
          AE_MULAAAAQ16(_ae_int64_acc_0, _ae_int16x4_vec, _ae_int16x4_mat_0);

#if XCHAL_HAVE_HIFI1
          AE_L8X4U_IP(_ae_int16x4_mat_1, p_mat2_1, 4);
#else
          AE_L8X4F_IP(_ae_int16x4_mat_1, p_mat2_1, 4);
          _ae_int16x4_mat_1 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_1), 8));
#endif
          AE_MULAAAAQ16(_ae_int64_acc_1, _ae_int16x4_vec, _ae_int16x4_mat_1);

#if XCHAL_HAVE_HIFI1
          AE_L8X4U_IP(_ae_int16x4_mat_2, p_mat2_2, 4);
#else
          AE_L8X4F_IP(_ae_int16x4_mat_2, p_mat2_2, 4);
          _ae_int16x4_mat_2 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_2), 8));
#endif
          AE_MULAAAAQ16(_ae_int64_acc_2, _ae_int16x4_vec, _ae_int16x4_mat_2);

#if XCHAL_HAVE_HIFI1
          AE_L8X4U_IP(_ae_int16x4_mat_3, p_mat2_3, 4);
#else
          AE_L8X4F_IP(_ae_int16x4_mat_3, p_mat2_3, 4);
          _ae_int16x4_mat_3 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat_3), 8));
#endif
          AE_MULAAAAQ16(_ae_int64_acc_3, _ae_int16x4_vec, _ae_int16x4_mat_3);
        }

        _ae_int64_acc_0 = AE_ADD64(_ae_int64_acc_0, acc_e);
        _ae_int64_acc_1 = AE_ADD64(_ae_int64_acc_1, acc_e);
        _ae_int64_acc_2 = AE_ADD64(_ae_int64_acc_2, acc_e);
        _ae_int64_acc_3 = AE_ADD64(_ae_int64_acc_3, acc_e);
      }
      ADD_BIAS_ACC;
      ADJUST_ACC;

      STORE_ACC;
    }

    for(; m_itr < rows; m_itr++)
    {
      UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
      for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
      {
        LOAD_VEC1; UNROLL_KERNEL_MAT1_VEC1(0);
      }

      if (p_mat2 && p_vec2){
        SETUP_VEC2; UNROLL_SETUP_MAT2(0);
        for(c_itr = 0; c_itr < (cols2 >> 2); c_itr++)
        {
          LOAD_VEC2; UNROLL_KERNEL_MAT2_VEC2(0);
        }
      }

      UNROLL_ADD_BIAS_ACC(0);
      UNROLL_ADJUST_ACC(0);

      UNROLL_STORE_ACC(0);
    }

    /* Undefining the defined macro to make them available for reuse */
#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef UNROLL_KERNEL_MAT1_VEC1
#undef UNROLL_KERNEL_MAT2_VEC2
#undef UNROLL_ADJUST_ACC
#undef UNROLL_STORE_ACC
#undef SETUP_VEC1
#undef SETUP_VEC2
#undef LOAD_VEC1
#undef LOAD_VEC2
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
  }
  else
  {
    int left_shift, right_shift;
    const WORD8 *p_mat1_0;
    const WORD8 *p_mat1_1;
    const WORD8 *p_mat1_2;
    const WORD8 *p_mat2_0;
    const WORD8 *p_mat2_1;
    const WORD8 *p_mat2_2;
    const WORD8 *p_vec1_0;
    const WORD8 *p_vec2_0;
    ae_int32x2 db0, db1;
    ae_int16x4 dm0, dm1, dm2;
    ae_int16x4 dv0;
    ae_int64 d_acc0, d_acc1, d_acc2;

    ALIGN_REGISTER_TYPE mat1_0_a, mat1_1_a, mat1_2_a, vec1_0_a;

    ALIGN_REGISTER_TYPE mat2_0_a, mat2_1_a, mat2_2_a, vec2_0_a;

    ae_int32x2 dm0_32, dm1_32, dv0_32, d_acc0_32, d_acc1_32;
    int m, n, k;

#if TFLITE_SINGLE_ROUNDING
    left_shift = out_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    left_shift = out_shift<0?0:out_shift;
    right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

    for (m = 0; m < (rows-2); m+=3)
    {
      p_mat1_0 = (const WORD8 *)(p_mat1+(m*row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      p_mat1_1 = (const WORD8 *)(p_mat1_0+row_stride1);
      p_mat1_2 = (const WORD8 *)(p_mat1_1+row_stride1);

      PRIME_8X4U(p_mat1_0, mat1_0_a);
      PRIME_8X4U(p_mat1_1, mat1_1_a);
      PRIME_8X4U(p_mat1_2, mat1_2_a);
      PRIME_8X4U(p_vec1_0, vec1_0_a);

      d_acc0 = d_acc1 = d_acc2 = AE_ZERO64();

      db0 = AE_MOVDA32X2(p_bias[m], p_bias[m+1]);
      db1 = AE_MOVDA32(p_bias[m+2]);

      for (n = 0; n < (cols1>>2); n++)
      {
        AE_LA8X4U_IP(dm0, mat1_0_a, p_mat1_0);
        AE_LA8X4U_IP(dm1, mat1_1_a, p_mat1_1);
        AE_LA8X4U_IP(dm2, mat1_2_a, p_mat1_2);
        AE_LA8X4U_IP(dv0, vec1_0_a, p_vec1_0);

        dm0 = AE_ADD16(dm0, AE_MOVDA16(mat1_zero_bias));
        dm1 = AE_ADD16(dm1, AE_MOVDA16(mat1_zero_bias));
        dm2 = AE_ADD16(dm2, AE_MOVDA16(mat1_zero_bias));
        dv0 = AE_ADD16(dv0, AE_MOVDA16(vec1_zero_bias));

        AE_MULAAAAQ16(d_acc0, dm0, dv0);
        AE_MULAAAAQ16(d_acc1, dm1, dv0);
        AE_MULAAAAQ16(d_acc2, dm2, dv0);
      }

      for(k = 0; k < (cols1&3); k++)
      {
        dm0_32 = AE_MOVDA32X2(*(((const UWORD8 *)p_mat1_0)+k), *(((const UWORD8 *)p_mat1_1)+k));
        dm1_32 = AE_MOVDA32(*(((const UWORD8 *)p_mat1_2)+k));
        dv0_32 = AE_MOVDA32(*(((const UWORD8 *)p_vec1_0)+k));

        dm0_32 = AE_ADD32(dm0_32, AE_MOVDA32(mat1_zero_bias));
        dm1_32 = AE_ADD32(dm1_32, AE_MOVDA32(mat1_zero_bias));
        dv0_32 = AE_ADD32(dv0_32, AE_MOVDA32(vec1_zero_bias));

        AE_MULA32_HL(d_acc0, dm0_32, dv0_32);
        AE_MULA32_LL(d_acc1, dm0_32, dv0_32);
        AE_MULA32_LL(d_acc2, dm1_32, dv0_32);
      }

      if (p_mat2 && p_vec2)
      {
        p_mat2_0 = (const WORD8 *)(p_mat2+(m*row_stride2));
        p_vec2_0 = (const WORD8 *)(p_vec2);
        p_mat2_1 = (const WORD8 *)(p_mat2_0+row_stride2);
        p_mat2_2 = (const WORD8 *)(p_mat2_1+row_stride2);

        PRIME_8X4U(p_mat2_0, mat2_0_a);
        PRIME_8X4U(p_mat2_1, mat2_1_a);
        PRIME_8X4U(p_mat2_2, mat2_2_a);
        PRIME_8X4U(p_vec2_0, vec2_0_a);

        for (n = 0; n < (cols2>>2); n++)
        {
          AE_LA8X4U_IP(dm0, mat2_0_a, p_mat2_0);
          AE_LA8X4U_IP(dm1, mat2_1_a, p_mat2_1);
          AE_LA8X4U_IP(dm2, mat2_2_a, p_mat2_2);
          AE_LA8X4U_IP(dv0, vec2_0_a, p_vec2_0);

          dm0 = AE_ADD16(dm0, AE_MOVDA16(mat2_zero_bias));
          dm1 = AE_ADD16(dm1, AE_MOVDA16(mat2_zero_bias));
          dm2 = AE_ADD16(dm2, AE_MOVDA16(mat2_zero_bias));
          dv0 = AE_ADD16(dv0, AE_MOVDA16(vec2_zero_bias));

          AE_MULAAAAQ16(d_acc0, dm0, dv0);
          AE_MULAAAAQ16(d_acc1, dm1, dv0);
          AE_MULAAAAQ16(d_acc2, dm2, dv0);
        }

        for(k = 0; k < (cols2&3); k++)
        {
          dm0_32 = AE_MOVDA32X2(*(((const UWORD8 *)p_mat2_0)+k), *(((const UWORD8 *)p_mat2_1)+k));
          dm1_32 = AE_MOVDA32(*(((const UWORD8 *)p_mat2_2)+k));
          dv0_32 = AE_MOVDA32(*(((const UWORD8 *)p_vec2_0)+k));

          dm0_32 = AE_ADD32(dm0_32, AE_MOVDA32(mat2_zero_bias));
          dm1_32 = AE_ADD32(dm1_32, AE_MOVDA32(mat2_zero_bias));
          dv0_32 = AE_ADD32(dv0_32, AE_MOVDA32(vec2_zero_bias));

          AE_MULA32_HL(d_acc0, dm0_32, dv0_32);
          AE_MULA32_LL(d_acc1, dm0_32, dv0_32);
          AE_MULA32_LL(d_acc2, dm1_32, dv0_32);
        }
      }
      d_acc0_32 = AE_TRUNCA32X2F64S(d_acc0, d_acc1, 32);
      d_acc1_32 = AE_TRUNCA32F64S(d_acc2, 32);
      /* Add bias */
      d_acc0_32 = AE_ADD32S(d_acc0_32, db0);
      d_acc1_32 = AE_ADD32S(d_acc1_32, db1);

      MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_32, d_acc0_32, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_X2_OUT32(d_acc1_32, d_acc1_32, out_multiplier, left_shift, right_shift);
      d_acc0_32 = AE_ADD32S(d_acc0_32, out_zero_bias);
      d_acc1_32 = AE_ADD32S(d_acc1_32, out_zero_bias);
      d_acc0_32 = AE_MAX32(AE_MIN32(d_acc0_32, AE_MOVDA32(255)), AE_ZERO32());
      *p_out++ = (UWORD8)AE_MOVAD32_H(d_acc0_32);
      *p_out++ = (UWORD8)AE_MOVAD32_L(d_acc0_32);
      d_acc1_32 = AE_MAX32(AE_MIN32(d_acc1_32, AE_MOVDA32(255)), AE_ZERO32());
      *p_out++ = (UWORD8)AE_MOVAD32_L(d_acc1_32);
    }

    /* Compute last (rows%3) output element */
    for (; m < rows; m++)
    {
      p_mat1_0 = (const WORD8 *)(p_mat1+(m*row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      PRIME_8X4U(p_mat1_0, mat1_0_a);
      PRIME_8X4U(p_vec1_0, vec1_0_a);

      d_acc0 = AE_ZERO64();

      db0 = AE_MOVDA32(p_bias[m]);

      for (n = 0; n < (cols1>>2); n++)
      {
        AE_LA8X4U_IP(dm0, mat1_0_a, p_mat1_0);
        AE_LA8X4U_IP(dv0, vec1_0_a, p_vec1_0);

        dm0 = AE_ADD16(dm0, AE_MOVDA16(mat1_zero_bias));
        dv0 = AE_ADD16(dv0, AE_MOVDA16(vec1_zero_bias));

        AE_MULAAAAQ16(d_acc0, dm0, dv0);
      }

      for(k = 0; k < (cols1&3); k++)
      {
        dm0_32 = AE_MOVDA32(*(((const UWORD8 *)p_mat1_0)+k));
        dv0_32 = AE_MOVDA32(*(((const UWORD8 *)p_vec1_0)+k));

        dm0_32 = AE_ADD32(dm0_32, AE_MOVDA32(mat1_zero_bias));
        dv0_32 = AE_ADD32(dv0_32, AE_MOVDA32(vec1_zero_bias));

        AE_MULA32_LL(d_acc0, dm0_32, dv0_32);
      }

      if (p_mat2 && p_vec2)
      {
        p_mat2_0 = (const WORD8 *)(p_mat2+(m*row_stride2));
        p_vec2_0 = (const WORD8 *)(p_vec2);

        PRIME_8X4U(p_mat2_0, mat2_0_a);
        PRIME_8X4U(p_vec2_0, vec2_0_a);

        for (n = 0; n < (cols2>>2); n++)
        {
          AE_LA8X4U_IP(dm0, mat2_0_a, p_mat2_0);
          AE_LA8X4U_IP(dv0, vec2_0_a, p_vec2_0);

          dm0 = AE_ADD16(dm0, AE_MOVDA16(mat2_zero_bias));
          dv0 = AE_ADD16(dv0, AE_MOVDA16(vec2_zero_bias));

          AE_MULAAAAQ16(d_acc0, dm0, dv0);
        }

        for(k = 0; k < (cols2&3); k++)
        {
          dm0_32 = AE_MOVDA32(*(((const UWORD8 *)p_mat2_0)+k));
          dv0_32 = AE_MOVDA32(*(((const UWORD8 *)p_vec2_0)+k));

          dm0_32 = AE_ADD32(dm0_32, AE_MOVDA32(mat2_zero_bias));
          dv0_32 = AE_ADD32(dv0_32, AE_MOVDA32(vec2_zero_bias));

          AE_MULA32_HL(d_acc0, dm0_32, dv0_32);
        }
      }
      d_acc0_32 = AE_TRUNCA32X2F64S(d_acc0, d_acc0, 32);

      /* Add bias */
      d_acc0_32 = AE_ADD32S(d_acc0_32, db0);

      MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_32, d_acc0_32, out_multiplier, left_shift, right_shift);
      d_acc0_32 = AE_ADD32S(d_acc0_32, out_zero_bias);
      d_acc0_32 = AE_MAX32(AE_MIN32(d_acc0_32, AE_MOVDA32(255)), AE_ZERO32());
      *p_out++ = (UWORD8)AE_MOVAD32_L(d_acc0_32);
    }
  }
  return 0;
}
