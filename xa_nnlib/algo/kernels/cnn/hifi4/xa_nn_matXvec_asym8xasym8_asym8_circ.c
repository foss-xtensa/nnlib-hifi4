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
#define VEC_UNROLL  2

#define SETUP_BIAS_BATCH_ASYM8b(idx_row, idx_vec) \
  ae_int64 _ae_int64_sat_bias_ ##idx_row ##_ ##idx_vec = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(p_bias[vec_itr + idx_vec])), 32); \

#define SETUP_BIAS_BATCH_ROW_ASYM8b(idx_row) \
  SETUP_BIAS_BATCH_VEC_UNROLL(idx_row) \

#define SETUP_ACC_BATCH_ROW_FOR_ASYM8bxASYM8b SETUP_ACC_BATCH_VEC_UNROLL

#define SETUP_ACC_BATCH_FOR_ASYM8bxASYM8b(idx_row,idx_vec) \
  ae_int64 _ae_int64_acc_ ##idx_row ##_ ##idx_vec = ZERO64; \

#define SETUP_VEC_BATCH_ASYM8b(idx_vec) \
  ae_int16x4 _ae_int16x4_vec_batch_ ##idx_vec  = AE_ZERO16(); \
  WORD8 *_WORD8_p_vec_batch_ ##idx_vec  = (WORD8 *)(&p_vec1[(vec_itr + idx_vec)*vec_stride]); \

#define SETUP_MAT1_ASYM8b(idx) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx = AE_ZERO16(); \
  WORD8 *_WORD8_p_mat1_ ## idx = (WORD8 *) p_mat1; \
  AE_ADDCIRC16X4_XC((ae_int16x4 *)_WORD8_p_mat1_ ##idx, (m_itr+idx)*row_stride1); \

#if XCHAL_HAVE_HIFI1
#define LOAD_VEC_BATCH_ASYM8b(idx_vec) \
  AE_L8X4U_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, 4*sizeof(WORD8)); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_offset));
#define LOAD_ROW_MAT1_ASYM8b(idx_row) \
  _ae_int16x4_mat1_ ##idx_row = AE_L8X4U_I(_WORD8_p_mat1_ ##idx_row, 0); \
  AE_ADDCIRC16X4_XC((ae_int16x4 *)_WORD8_p_mat1_ ##idx_row, 4*sizeof(WORD8)); \
  _ae_int16x4_mat1_ ##idx_row = AE_ADD16(_ae_int16x4_mat1_ ##idx_row, AE_MOVDA16(mat1_offset));
#else
#define LOAD_VEC_BATCH_ASYM8b(idx_vec) \
  AE_L8X4F_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, 4*sizeof(WORD8)); \
  _ae_int16x4_vec_batch_ ##idx_vec  = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec_batch_ ##idx_vec), 8)); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_offset));
#define LOAD_ROW_MAT1_ASYM8b(idx_row) \
  _ae_int16x4_mat1_ ##idx_row = AE_L8X4F_I(_WORD8_p_mat1_ ##idx_row, 0); \
  AE_ADDCIRC16X4_XC((ae_int16x4 *)_WORD8_p_mat1_ ##idx_row, 4*sizeof(WORD8)); \
  _ae_int16x4_mat1_ ##idx_row = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat1_ ##idx_row), 8)); \
  _ae_int16x4_mat1_ ##idx_row = AE_ADD16(_ae_int16x4_mat1_ ##idx_row, AE_MOVDA16(mat1_offset));
#endif

#define KERNEL_MAT1_VEC_BATCH_ROW_ASYM8b_ASYM8b(idx_row) \
  KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row); \

#define KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(idx_row, idx_vec) \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx_row ##_ ##idx_vec, _ae_int16x4_vec_batch_ ##idx_vec, _ae_int16x4_mat1_ ## idx_row); \

#define ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b(idx_row) \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row); \

#define ADD_BIAS_BATCH_ASYM8b_ACC_FOR_ASYM8bxASYM8b(idx_row,idx_vec) \
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_ADD64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int64_sat_bias_ ##idx_row ##_ ##idx_vec); \

/* Output scaling according to Tensorflow logic; following are steps:
    1. If left_shift is to be done, do it in 32-bit without saturation
    2. Multiply by out_multiplier: 32x32 multiplcation to 32 bit output
    with asymmetric rounding and saturation
    3. If right_shift is to be done, do it with symmetric rounding
    4. Add out_offset */
#define ADJUST_ACC_BATCH_ROW_ASYM8b(idx_row) \
  ADJUST_ACC_BATCH_VEC_UNROLL(idx_row); \

#if XCHAL_HAVE_HIFI1
#define ADJUST_ACC_BATCH_ASYM8b(idx_row, idx_vec) \
  ae_int32x2 _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_SLAA32(AE_MOVINT32X2_FROMINT64(_ae_int64_acc_ ##idx_row ##_ ##idx_vec), left_shift); \
  MPY_BY_QUANT_MULT_X2_OUT32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVINT32X2_FROMINT64(_ae_int64_acc_ ##idx_row ##_ ##idx_vec), out_multiplier, left_shift, right_shift); \
  (_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec) = AE_ADD32S(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(out_offset)); \


#define STORE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row,idx_vec) \
  _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(0)), AE_MOVDA32(255)); \
    AE_S8_0_I_HIFI1(AE_MOVINT16X4_FROMINT32X2(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec), ((WORD8 *)(&p_out[(vec_itr + idx_vec)*out_col_offset + (m_itr + idx_row)*out_row_offset])) , 0); \

#else
#define ADJUST_ACC_BATCH_ASYM8b(idx_row, idx_vec) \
  ae_int32x2 _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec; \
  MPY_BY_QUANT_MULT_X2_OUT32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVINT32X2_FROMINT64(_ae_int64_acc_ ##idx_row ##_ ##idx_vec), out_multiplier, left_shift, right_shift); \
  (_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec) = AE_ADD32S(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(out_offset)); \

#define STORE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row,idx_vec) \
  _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  (*((UWORD8 *) (&p_out[(vec_itr + idx_vec)*out_col_offset + (m_itr + idx_row)*out_row_offset]))) = (UWORD8)AE_MOVAD32_L(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec); \

#endif

/* Saturate result to unsigned 8 bit (0-255) and store */
#define STORE_ACC_BATCH_ROW_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row) \
  STORE_ACC_BATCH_VEC_UNROLL(idx_row); \

#if (ROW_UNROLL == 1)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)
#define SETUP_ROW_SUM_MAT1   UNROLL_SETUP_ROW_SUM_MAT1(0)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)

#elif (ROW_UNROLL == 2)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)            UNROLL_SETUP_ACC(1)
#define SETUP_ROW_SUM_MAT1   UNROLL_SETUP_ROW_SUM_MAT1(0)   UNROLL_SETUP_ROW_SUM_MAT1(1)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)

#elif (ROW_UNROLL == 4)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)           UNROLL_SETUP_MAT1(2)           UNROLL_SETUP_MAT1(3)

#elif (ROW_UNROLL == 8)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)            UNROLL_SETUP_ACC(1)            UNROLL_SETUP_ACC(2)            UNROLL_SETUP_ACC(3)            UNROLL_SETUP_ACC(4)            UNROLL_SETUP_ACC(5)            UNROLL_SETUP_ACC(6)            UNROLL_SETUP_ACC(7)
#define SETUP_ROW_SUM_MAT1   UNROLL_SETUP_ROW_SUM_MAT1(0)   UNROLL_SETUP_ROW_SUM_MAT1(1)   UNROLL_SETUP_ROW_SUM_MAT1(2)   UNROLL_SETUP_ROW_SUM_MAT1(3)   UNROLL_SETUP_ROW_SUM_MAT1(4)   UNROLL_SETUP_ROW_SUM_MAT1(5)   UNROLL_SETUP_ROW_SUM_MAT1(6)   UNROLL_SETUP_ROW_SUM_MAT1(7)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)           UNROLL_SETUP_MAT1(2)           UNROLL_SETUP_MAT1(3)           UNROLL_SETUP_MAT1(4)           UNROLL_SETUP_MAT1(5)           UNROLL_SETUP_MAT1(6)           UNROLL_SETUP_MAT1(7)

#endif /* (ROW_UNROLL == 1) */

#if (ROW_UNROLL == 4 && VEC_UNROLL == 2)

#define SETUP_VEC_BATCH                             UNROLL_SETUP_VEC_BATCH(0)               UNROLL_SETUP_VEC_BATCH(1)

#define SETUP_BIAS_BATCH                            UNROLL_ROW_SETUP_BIAS_BATCH(0)          UNROLL_ROW_SETUP_BIAS_BATCH(1)          UNROLL_ROW_SETUP_BIAS_BATCH(2)      UNROLL_ROW_SETUP_BIAS_BATCH(3)
#define SETUP_BIAS_BATCH_VEC_UNROLL(idx_row)        UNROLL_SETUP_BIAS_BATCH(idx_row,0)      UNROLL_SETUP_BIAS_BATCH(idx_row,1)
#define SETUP_BIAS_BATCH_TAIL                       UNROLL_SETUP_BIAS_BATCH(0,0)            UNROLL_SETUP_BIAS_BATCH(1,0)            UNROLL_SETUP_BIAS_BATCH(2,0)        UNROLL_SETUP_BIAS_BATCH(3,0)

#define SETUP_ACC_BATCH                             UNROLL_ROW_SETUP_ACC_BATCH(0)           UNROLL_ROW_SETUP_ACC_BATCH(1)           UNROLL_ROW_SETUP_ACC_BATCH(2)       UNROLL_ROW_SETUP_ACC_BATCH(3)
#define SETUP_ACC_BATCH_VEC_UNROLL(idx_row)         UNROLL_SETUP_ACC_BATCH(idx_row,0)       UNROLL_SETUP_ACC_BATCH(idx_row,1)
#define SETUP_ACC_BATCH_TAIL                        UNROLL_SETUP_ACC_BATCH(0,0)             UNROLL_SETUP_ACC_BATCH(1,0)             UNROLL_SETUP_ACC_BATCH(2,0)         UNROLL_SETUP_ACC_BATCH(3,0)

#define LOAD_VEC_BATCH                              UNROLL_LOAD_VEC_BATCH(0)                UNROLL_LOAD_VEC_BATCH(1)
#define LOAD_MAT1                                   UNROLL_LOAD_ROW_MAT1(0)                 UNROLL_LOAD_ROW_MAT1(1)                 UNROLL_LOAD_ROW_MAT1(2)             UNROLL_LOAD_ROW_MAT1(3)

#define KERNEL_MAT1_VEC_BATCH                       UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(0)     UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(1)     UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(2) UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(3)
#define KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row)   UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row,0) UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row,1)
#define KERNEL_MAT1_VEC_BATCH_TAIL                  UNROLL_KERNEL_MAT1_VEC_BATCH(0,0)       UNROLL_KERNEL_MAT1_VEC_BATCH(1,0)       UNROLL_KERNEL_MAT1_VEC_BATCH(2,0)   UNROLL_KERNEL_MAT1_VEC_BATCH(3,0)

#define ADD_BIAS_ACC_BATCH                          UNROLL_ROW_ADD_BIAS_ACC(0)              UNROLL_ROW_ADD_BIAS_ACC(1)              UNROLL_ROW_ADD_BIAS_ACC(2)          UNROLL_ROW_ADD_BIAS_ACC(3)
#define ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row)      UNROLL_ADD_BIAS_ACC_BATCH(idx_row,0)    UNROLL_ADD_BIAS_ACC_BATCH(idx_row,1)
#define ADD_BIAS_ACC_BATCH_TAIL                     UNROLL_ADD_BIAS_ACC_BATCH(0,0)          UNROLL_ADD_BIAS_ACC_BATCH(1,0)          UNROLL_ADD_BIAS_ACC_BATCH(2,0)      UNROLL_ADD_BIAS_ACC_BATCH(3,0)

#define STORE_ACC_BATCH                             UNROLL_ROW_STORE_ACC(0)                 UNROLL_ROW_STORE_ACC(1)                 UNROLL_ROW_STORE_ACC(2)             UNROLL_ROW_STORE_ACC(3)
#define STORE_ACC_BATCH_VEC_UNROLL(idx_row)         UNROLL_STORE_ACC_BATCH(idx_row,0)       UNROLL_STORE_ACC_BATCH(idx_row,1)
#define STORE_ACC_BATCH_TAIL                        UNROLL_STORE_ACC_BATCH(0,0)             UNROLL_STORE_ACC_BATCH(1,0)             UNROLL_STORE_ACC_BATCH(2,0)         UNROLL_STORE_ACC_BATCH(3,0)

#define ADJUST_ACC_BATCH                            UNROLL_ROW_ADJUST_ACC(0)                UNROLL_ROW_ADJUST_ACC(1)                UNROLL_ROW_ADJUST_ACC(2)            UNROLL_ROW_ADJUST_ACC(3)
#define ADJUST_ACC_BATCH_VEC_UNROLL(idx_row)        UNROLL_ADJUST_ACC_BATCH(idx_row,0)      UNROLL_ADJUST_ACC_BATCH(idx_row,1)
#define ADJUST_ACC_BATCH_TAIL                       UNROLL_ADJUST_ACC_BATCH(0, 0)           UNROLL_ADJUST_ACC_BATCH(1, 0)           UNROLL_ADJUST_ACC_BATCH(2, 0)       UNROLL_ADJUST_ACC_BATCH(3, 0)

#endif /* (ROW_UNROLL == 4 && VEC_UNROLL == 2)*/


WORD32 xa_nn_matXvec_asym8xasym8_asym8_circ(
    UWORD8 * __restrict__ p_out,
    UWORD8 * __restrict__ p_mat1,
    const UWORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_stride,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 mat1_offset,
    WORD32 vec1_offset,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_offset)
{

  /* Iterators used in for loops */
  int m_itr, c_itr, vec_itr;
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

#define UNROLL_ROW_SETUP_ACC_BATCH              SETUP_ACC_BATCH_ROW_FOR_ASYM8bxASYM8b
#define UNROLL_SETUP_ACC_BATCH                  SETUP_ACC_BATCH_FOR_ASYM8bxASYM8b
#define UNROLL_SETUP_MAT1                       SETUP_MAT1_ASYM8b
#define UNROLL_SETUP_VEC_BATCH                  SETUP_VEC_BATCH_ASYM8b
#define UNROLL_ROW_SETUP_BIAS_BATCH             SETUP_BIAS_BATCH_ROW_ASYM8b
#define UNROLL_SETUP_BIAS_BATCH                 SETUP_BIAS_BATCH_ASYM8b
#define UNROLL_LOAD_VEC_BATCH                   LOAD_VEC_BATCH_ASYM8b
#define UNROLL_LOAD_ROW_MAT1                    LOAD_ROW_MAT1_ASYM8b
#define UNROLL_ROW_KERNEL_MAT1_VEC_BATCH        KERNEL_MAT1_VEC_BATCH_ROW_ASYM8b_ASYM8b
#define UNROLL_KERNEL_MAT1_VEC_BATCH            KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b
#define UNROLL_ROW_ADD_BIAS_ACC                 ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b
#define UNROLL_ADD_BIAS_ACC_BATCH               ADD_BIAS_BATCH_ASYM8b_ACC_FOR_ASYM8bxASYM8b
#define UNROLL_ROW_ADJUST_ACC                   ADJUST_ACC_BATCH_ROW_ASYM8b
#define UNROLL_ADJUST_ACC_BATCH                 ADJUST_ACC_BATCH_ASYM8b
#define UNROLL_ROW_STORE_ACC                    STORE_ACC_BATCH_ROW_ASYM8bxASYM8b_AT_OUT_ASYM8b
#define UNROLL_STORE_ACC_BATCH                  STORE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  right_shift = out_shift;
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  if(p_mat1 && p_vec1)
  {
    for(vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr+= VEC_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
      {
        SETUP_BIAS_BATCH;
        SETUP_ACC_BATCH;
        SETUP_VEC_BATCH;
        SETUP_MAT1;
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC_BATCH;
          LOAD_MAT1;
          KERNEL_MAT1_VEC_BATCH;
        }
        ADD_BIAS_ACC_BATCH;
        ADJUST_ACC_BATCH;
        STORE_ACC_BATCH;
      }
      for(; m_itr < rows; m_itr++)
      {
        UNROLL_ROW_SETUP_BIAS_BATCH(0);
        UNROLL_ROW_SETUP_ACC_BATCH(0);
        SETUP_VEC_BATCH;
        UNROLL_SETUP_MAT1(0);
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          LOAD_VEC_BATCH;
          UNROLL_LOAD_ROW_MAT1(0);
          UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(0);
        }
        UNROLL_ROW_ADD_BIAS_ACC(0);
        UNROLL_ROW_ADJUST_ACC(0);
        UNROLL_ROW_STORE_ACC(0);
      }
    }
    /* Tail loop for vec unroll */
    for(; vec_itr < vec_count; vec_itr++)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
      {
        SETUP_BIAS_BATCH_TAIL;
        SETUP_ACC_BATCH_TAIL;
        UNROLL_SETUP_VEC_BATCH(0);
        SETUP_MAT1;
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
          UNROLL_LOAD_VEC_BATCH(0);
          LOAD_MAT1;
          KERNEL_MAT1_VEC_BATCH_TAIL;
        }
        ADD_BIAS_ACC_BATCH_TAIL;
        ADJUST_ACC_BATCH_TAIL;
        STORE_ACC_BATCH_TAIL;
      }
      for(; m_itr < rows; m_itr++)
      {
        UNROLL_SETUP_BIAS_BATCH(0,0);
        UNROLL_SETUP_ACC_BATCH(0,0);
        UNROLL_SETUP_VEC_BATCH(0);
        UNROLL_SETUP_MAT1(0);
        for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
        {
            UNROLL_LOAD_VEC_BATCH(0);
            UNROLL_LOAD_ROW_MAT1(0);
            UNROLL_KERNEL_MAT1_VEC_BATCH(0,0);
        }
        UNROLL_ADD_BIAS_ACC_BATCH(0,0);
        UNROLL_ADJUST_ACC_BATCH(0,0);
        UNROLL_STORE_ACC_BATCH(0,0);
      }
    }
  }
  else
  {
    return -1;
  }
  return 0;
}
