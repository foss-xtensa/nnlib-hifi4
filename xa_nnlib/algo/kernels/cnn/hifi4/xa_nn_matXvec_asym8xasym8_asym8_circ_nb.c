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
#include "xa_type_def.h"
#include "xa_nn_common.h"
#include "xa_nn_conv2d_std_state.h"

#include "xa_nnlib_common.h"
#include "xa_nnlib_quant_macros.h"

#define ZERO64  AE_ZERO64()

#define ROW_UNROLL  4

#define SETUP_ACC_FOR_ASYM8bxASYM8b(idx_row) \
  ae_int64 _ae_int64_acc_ ##idx_row = ZERO64; \

#define SETUP_VEC_ASYM8b \
  ae_int16x4 _ae_int16x4_vec = AE_ZERO16(); \
  WORD8 *_WORD8_p_vec  = (WORD8 *)p_vec1; \

#define SETUP_MAT1_ASYM8b(idx_row) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx_row = AE_ZERO16(); \
  WORD8 *_WORD8_p_mat1_ ## idx_row = (WORD8 *) (&p_mat1[(m_itr+idx_row)*cols]); \

#if XCHAL_HAVE_HIFI1
#define LOAD_VEC_ASYM8b \
  _ae_int16x4_vec = AE_L8X4U_I(_WORD8_p_vec, 0); \
  AE_ADDCIRC16X4_XC((ae_int16x4 *)_WORD8_p_vec, 4*sizeof(WORD8)); \
  _ae_int16x4_vec = AE_ADD16(_ae_int16x4_vec, AE_MOVDA16(vec1_offset));

#define LOAD_ROW_MAT1_ASYM8b(idx_row) \
  AE_L8X4U_IP(_ae_int16x4_mat1_ ##idx_row, _WORD8_p_mat1_ ##idx_row, 4*sizeof(WORD8)); \
  _ae_int16x4_mat1_ ##idx_row = AE_ADD16(_ae_int16x4_mat1_ ##idx_row, AE_MOVDA16(mat1_offset));
#else
#define LOAD_VEC_ASYM8b \
  _ae_int16x4_vec = AE_L8X4F_I(_WORD8_p_vec, 0); \
  AE_ADDCIRC16X4_XC((ae_int16x4 *)_WORD8_p_vec, 4*sizeof(WORD8)); \
  _ae_int16x4_vec = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec), 8)); \
  _ae_int16x4_vec = AE_ADD16(_ae_int16x4_vec, AE_MOVDA16(vec1_offset));

#define LOAD_ROW_MAT1_ASYM8b(idx_row) \
  AE_L8X4F_IP(_ae_int16x4_mat1_ ##idx_row, _WORD8_p_mat1_ ##idx_row, 4*sizeof(WORD8)); \
  _ae_int16x4_mat1_ ##idx_row = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat1_ ##idx_row), 8)); \
  _ae_int16x4_mat1_ ##idx_row = AE_ADD16(_ae_int16x4_mat1_ ##idx_row, AE_MOVDA16(mat1_offset));
#endif

#define KERNEL_MAT1_VEC_ASYM8b_ASYM8b(idx_row) \
  LOAD_ROW_MAT1_ASYM8b(idx_row); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx_row, _ae_int16x4_vec, _ae_int16x4_mat1_ ## idx_row); \

#define ADD_BIAS_ASYM8b_ACC_FOR_ASYM8bxASYM8b(idx_row) \
  ae_int64 _ae_int64_sat_bias_ ##idx_row = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(p_bias[m_itr + idx_row])), 32); \
  _ae_int64_acc_ ## idx_row = AE_ADD64S(_ae_int64_acc_ ## idx_row, _ae_int64_sat_bias_ ##idx_row); \

/* Output scaling according to Tensorflow logic; following are steps:
    1. If left_shift is to be done, do it in 32-bit without saturation
    2. Multiply by out_multiplier: 32x32 multiplcation to 32 bit output
    with asymmetric rounding and saturation
    3. If right_shift is to be done, do it with symmetric rounding
    4. Add out_offset */
#if XCHAL_HAVE_HIFI1
#define ADJUST_ACC_ASYM8b(idx_row) \
  ae_int32x2 _ae_int32x2_acc_ ##idx_row; \
  MPY_BY_QUANT_MULT_X2_OUT32(_ae_int32x2_acc_ ##idx_row, AE_MOVINT32X2_FROMINT64(_ae_int64_acc_ ##idx_row), out_multiplier, left_shift, right_shift); \
  (_ae_int32x2_acc_ ##idx_row) = AE_ADD32S(_ae_int32x2_acc_ ##idx_row, AE_MOVDA32(out_offset)); \

/* Saturate result to unsigned 8 bit (0-255) and store */
#define STORE_ACC_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row) \
  _ae_int32x2_acc_ ##idx_row = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ##idx_row, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  AE_S8_0_I_HIFI1(AE_MOVINT16X4_FROMINT32X2(_ae_int32x2_acc_ ##idx_row), ((WORD8 *)p_out + (m_itr + idx_row)*out_stride), 0); \

#else
#define ADJUST_ACC_ASYM8b(idx_row) \
  ae_int32x2 _ae_int32x2_acc_ ##idx_row; \
  MPY_BY_QUANT_MULT_X2_OUT32(_ae_int32x2_acc_ ##idx_row, AE_MOVINT32X2_FROMINT64(_ae_int64_acc_ ##idx_row), out_multiplier, left_shift, right_shift); \
  (_ae_int32x2_acc_ ##idx_row) = AE_ADD32S(_ae_int32x2_acc_ ##idx_row, AE_MOVDA32(out_offset)); \

/* Saturate result to unsigned 8 bit (0-255) and store */
#define STORE_ACC_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row) \
  _ae_int32x2_acc_ ##idx_row = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ##idx_row, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  (*((UWORD8 *) (&p_out[(m_itr + idx_row)*out_stride]))) = (UWORD8)AE_MOVAD32_L(_ae_int32x2_acc_ ##idx_row); \

#endif

#if (ROW_UNROLL == 1)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)
#define LOAD_MAT1            UNROLL_LOAD_MAT1(0)
#define KERNEL_MAT1_VEC      UNROLL_KERNEL_MAT1_VEC(0)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)
#define STORE_ACC            UNROLL_STORE_ACC(0)

#elif (ROW_UNROLL == 2)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)            UNROLL_SETUP_ACC(1)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)
#define LOAD_MAT1            UNROLL_LOAD_MAT1(0)            UNROLL_LOAD_MAT1(1)
#define KERNEL_MAT1_VEC      UNROLL_KERNEL_MAT1_VEC(0)      UNROLL_KERNEL_MAT1_VEC(1)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)         UNROLL_ADD_BIAS_ACC(1)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)           UNROLL_ADJUST_ACC(1)
#define STORE_ACC            UNROLL_STORE_ACC(0)            UNROLL_STORE_ACC(1)

#elif (ROW_UNROLL == 4)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)            UNROLL_SETUP_ACC(1)            UNROLL_SETUP_ACC(2)            UNROLL_SETUP_ACC(3)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)           UNROLL_SETUP_MAT1(2)           UNROLL_SETUP_MAT1(3)
#define KERNEL_MAT1_VEC      UNROLL_KERNEL_MAT1_VEC(0)      UNROLL_KERNEL_MAT1_VEC(1)      UNROLL_KERNEL_MAT1_VEC(2)      UNROLL_KERNEL_MAT1_VEC(3)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)         UNROLL_ADD_BIAS_ACC(1)         UNROLL_ADD_BIAS_ACC(2)         UNROLL_ADD_BIAS_ACC(3)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)           UNROLL_ADJUST_ACC(1)           UNROLL_ADJUST_ACC(2)           UNROLL_ADJUST_ACC(3)
#define STORE_ACC            UNROLL_STORE_ACC(0)            UNROLL_STORE_ACC(1)            UNROLL_STORE_ACC(2)            UNROLL_STORE_ACC(3)

#elif (ROW_UNROLL == 8)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)            UNROLL_SETUP_ACC(1)            UNROLL_SETUP_ACC(2)            UNROLL_SETUP_ACC(3)            UNROLL_SETUP_ACC(4)            UNROLL_SETUP_ACC(5)            UNROLL_SETUP_ACC(6)            UNROLL_SETUP_ACC(7)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)           UNROLL_SETUP_MAT1(2)           UNROLL_SETUP_MAT1(3)           UNROLL_SETUP_MAT1(4)           UNROLL_SETUP_MAT1(5)           UNROLL_SETUP_MAT1(6)           UNROLL_SETUP_MAT1(7)
#define LOAD_MAT1            UNROLL_LOAD_MAT1(0)            UNROLL_LOAD_MAT1(1)            UNROLL_LOAD_MAT1(2)            UNROLL_LOAD_MAT1(3)            UNROLL_LOAD_MAT1(4)            UNROLL_LOAD_MAT1(5)            UNROLL_LOAD_MAT1(6)            UNROLL_LOAD_MAT1(7)
#define KERNEL_MAT1_VEC      UNROLL_KERNEL_MAT1_VEC(0)      UNROLL_KERNEL_MAT1_VEC(1)      UNROLL_KERNEL_MAT1_VEC(2)      UNROLL_KERNEL_MAT1_VEC(3)      UNROLL_KERNEL_MAT1_VEC(4)      UNROLL_KERNEL_MAT1_VEC(5)      UNROLL_KERNEL_MAT1_VEC(6)      UNROLL_KERNEL_MAT1_VEC(7)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)         UNROLL_ADD_BIAS_ACC(1)         UNROLL_ADD_BIAS_ACC(2)         UNROLL_ADD_BIAS_ACC(3)         UNROLL_ADD_BIAS_ACC(4)         UNROLL_ADD_BIAS_ACC(5)         UNROLL_ADD_BIAS_ACC(6)         UNROLL_ADD_BIAS_ACC(7)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)           UNROLL_ADJUST_ACC(1)           UNROLL_ADJUST_ACC(2)           UNROLL_ADJUST_ACC(3)           UNROLL_ADJUST_ACC(4)           UNROLL_ADJUST_ACC(5)           UNROLL_ADJUST_ACC(6)           UNROLL_ADJUST_ACC(7)
#define STORE_ACC            UNROLL_STORE_ACC(0)            UNROLL_STORE_ACC(1)            UNROLL_STORE_ACC(2)            UNROLL_STORE_ACC(3)            UNROLL_STORE_ACC(4)            UNROLL_STORE_ACC(5)            UNROLL_STORE_ACC(6)            UNROLL_STORE_ACC(7)

#endif /* (ROW_UNROLL == 1) */

WORD32 xa_nn_matXvec_asym8xasym8_asym8_circ_nb(
    UWORD8 * __restrict__ p_out,
    UWORD8 * __restrict__ p_mat1,
    UWORD8 * __restrict__ p_vec1,
    WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 out_stride,
    WORD32 mat1_offset,
    WORD32 vec1_offset,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_offset)
{

  /* Iterators used in for loops */
  int m_itr, c_itr;
  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;

  if((out_shift > 31) || (out_shift < -31))
  {
    return -1;
  }

  if (!p_bias)
  {
    return -1;
  }

#define UNROLL_SETUP_ACC                        SETUP_ACC_FOR_ASYM8bxASYM8b
#define UNROLL_SETUP_MAT1                       SETUP_MAT1_ASYM8b
#define SETUP_VEC                               SETUP_VEC_ASYM8b
#define LOAD_VEC                                LOAD_VEC_ASYM8b
#define UNROLL_KERNEL_MAT1_VEC                  KERNEL_MAT1_VEC_ASYM8b_ASYM8b
#define UNROLL_ADD_BIAS_ACC                     ADD_BIAS_ASYM8b_ACC_FOR_ASYM8bxASYM8b
#define UNROLL_ADJUST_ACC                       ADJUST_ACC_ASYM8b
#define UNROLL_STORE_ACC                        STORE_ACC_ASYM8bxASYM8b_AT_OUT_ASYM8b

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  right_shift = out_shift;
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

#if XCHAL_HAVE_HIFI4
  if(p_mat1 && p_vec1)
  {
    for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
    {
      SETUP_ACC;
      SETUP_VEC;
      SETUP_MAT1;
      for(c_itr = 0; c_itr < (cols >> 3); c_itr++)
      {
        LOAD_VEC;
        KERNEL_MAT1_VEC;
        LOAD_VEC;
        KERNEL_MAT1_VEC;
      }
      if((cols & 7) != 0)
      {
        LOAD_VEC;
        KERNEL_MAT1_VEC;
      }
      ADD_BIAS_ACC;
      ADJUST_ACC;
      STORE_ACC;
    }
    for(; m_itr < rows; m_itr++)
    {
      UNROLL_SETUP_ACC(0);
      SETUP_VEC;
      UNROLL_SETUP_MAT1(0);
      for(c_itr = 0; c_itr < (cols >> 2); c_itr++)
      {
          LOAD_VEC;
          UNROLL_KERNEL_MAT1_VEC(0);
      }
      UNROLL_ADD_BIAS_ACC(0);
      UNROLL_ADJUST_ACC(0);
      UNROLL_STORE_ACC(0);
    }
  }
#else
  if(p_mat1 && p_vec1)
  {
    for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
    {
      SETUP_ACC;
      SETUP_VEC;
      SETUP_MAT1;
      for(c_itr = 0; c_itr < (cols >> 2); c_itr++)
      {
        LOAD_VEC;
        KERNEL_MAT1_VEC;
      }
      ADD_BIAS_ACC;
      ADJUST_ACC;
      STORE_ACC;
    }
    for(; m_itr < rows; m_itr++)
    {
      UNROLL_SETUP_ACC(0);
      SETUP_VEC;
      UNROLL_SETUP_MAT1(0);
      for(c_itr = 0; c_itr < (cols >> 2); c_itr++)
      {
          LOAD_VEC;
          UNROLL_KERNEL_MAT1_VEC(0);
      }
      UNROLL_ADD_BIAS_ACC(0);
      UNROLL_ADJUST_ACC(0);
      UNROLL_STORE_ACC(0);
    }
  }
#endif
  else
  {
    return -1;
  }
  return 0;
}
