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
#include <stddef.h>
#include "xa_nnlib_common.h"

#define ZERO64   AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0))

#if defined(CUST_UNROLL) && (CUST_UNROLL != 0)
#define UNROLL_S CUST_UNROLL
#else
#define UNROLL_S  8 /// Optimal unroll
#endif

#define SETUP_ROW_S(N) \
  ae_int64 accu1_ ##N;\
  WORD8 *p_mat1_ ##N = &p_mat[(row+N)*cols]; \
  accu1_ ##N = ZERO64;


#if XCHAL_HAVE_HIFI1
#define KERNEL_ROW_S(N) \
{ \
  ae_int16x4 temp_in1; \
  AE_L8X4S_IP(temp_in1, p_mat1_ ##N, 4); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
}
#define KERNEL_ROW_S_I(N) \
{ \
  ae_int16x4 temp_in1; \
  AE_L8X4S_IP(temp_in1, p_mat1_ ##N, 4); \
  AE_L16X4_XC(temp_src1, p_src1, 8); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
}
#define STORE_ROW_S(N) \
  ae_int64 temp1_ ##N = p_bias[row+N];            \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , bias_shift);\
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N);\
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , acc_shift);\
  ae_int32x2 sat_acc1_ ##N = AE_ROUND32F64SSYM(accu1_ ##N);\
  p_out[(row+N)*out_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_acc1_ ##N, sat_acc1_ ##N)); \

#else

#if !XA_HAVE_HIFI3_CORE
#define KERNEL_ROW_S_TAIL(N) \
{ \
  ae_int16x4 temp_in1; \
  AE_L8X4F_IP(temp_in1, p_mat1_ ##N, 4); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
}
#define KERNEL_ROW_S_I_TAIL(N) \
{ \
  ae_int16x4 temp_in1; \
  AE_L8X4F_IP(temp_in1, p_mat1_ ##N, 4); \
  AE_L16X4_XC(temp_src1, p_src1, 8); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
}

#define KERNEL_ROW_S(N) \
{ \
  ae_int16x4 temp_in1, temp_in1_I; \
  temp_in1_I = AE_L8X4F_I(p_mat1_ ##N, 4); \
  AE_L8X4F_IP(temp_in1, p_mat1_ ##N, 8); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
  AE_MULAAAAQ16(accu1_ ##N, temp_src2, temp_in1_I);\
}
#define KERNEL_ROW_S_I(N) \
{ \
  ae_int16x4 temp_in1, temp_in1_I; \
  temp_in1_I = AE_L8X4F_I(p_mat1_ ##N, 4); \
  AE_L8X4F_IP(temp_in1, p_mat1_ ##N, 8); \
  AE_L16X4_XC(temp_src1, p_src1, 8); \
  AE_L16X4_XC(temp_src2, p_src1, 8); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
  AE_MULAAAAQ16(accu1_ ##N, temp_src2, temp_in1_I);\
}
#else/* !XA_HAVE_HIFI3_CORE */

#define KERNEL_ROW_S_I_TAIL KERNEL_ROW_S_I
#define KERNEL_ROW_S(N) \
{ \
  ae_int16x4 temp_in1; \
  AE_L8X4F_IP(temp_in1, p_mat1_ ##N, 4); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
}
#define KERNEL_ROW_S_I(N) \
{ \
  ae_int16x4 temp_in1; \
  AE_L8X4F_IP(temp_in1, p_mat1_ ##N, 4); \
  AE_L16X4_XC(temp_src1, p_src1, 8); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
}

#endif/* !XA_HAVE_HIFI3_CORE */
#define STORE_ROW_S(N) \
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , -8);\
  ae_int64 temp1_ ##N = p_bias[row+N];            \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , bias_shift);\
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N);\
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , acc_shift);\
  p_out[(row+N)*out_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_ ##N),16),-16)); \

#endif

#if (UNROLL_S == 1)
#define SETUP_S SETUP_ROW_S(0)
#define KERNEL_S KERNEL_ROW_S_I(0)
#define KERNEL_S_TAIL KERNEL_ROW_S_I_TAIL(0)
#define STORE_S STORE_ROW_S(0)

#elif (UNROLL_S == 2)
#define SETUP_S  SETUP_ROW_S(0)  SETUP_ROW_S(1)
#define KERNEL_S KERNEL_ROW_S_I(0) KERNEL_ROW_S(1)
#define KERNEL_S_TAIL KERNEL_ROW_S_I_TAIL(0) KERNEL_ROW_S_TAIL(1)
#define STORE_S  STORE_ROW_S(0)  STORE_ROW_S(1)

#elif (UNROLL_S == 4)
#define SETUP_S  SETUP_ROW_S(0)  SETUP_ROW_S(1)  SETUP_ROW_S(2)  SETUP_ROW_S(3)
#define KERNEL_S KERNEL_ROW_S_I(0) KERNEL_ROW_S(1) KERNEL_ROW_S(2) KERNEL_ROW_S(3)
#define KERNEL_S_TAIL KERNEL_ROW_S_I_TAIL(0) KERNEL_ROW_S_TAIL(1) KERNEL_ROW_S_TAIL(2) KERNEL_ROW_S_TAIL(3)
#define STORE_S  STORE_ROW_S(0)  STORE_ROW_S(1)  STORE_ROW_S(2)  STORE_ROW_S(3)
#elif (UNROLL_S == 8)
#define SETUP_S   SETUP_ROW_S(0)  SETUP_ROW_S(1)  SETUP_ROW_S(2)  SETUP_ROW_S(3)  SETUP_ROW_S(4)  SETUP_ROW_S(5)  SETUP_ROW_S(6)  SETUP_ROW_S(7)
#define KERNEL_S KERNEL_ROW_S_I(0) KERNEL_ROW_S(1) KERNEL_ROW_S(2) KERNEL_ROW_S(3) KERNEL_ROW_S(4) KERNEL_ROW_S(5) KERNEL_ROW_S(6) KERNEL_ROW_S(7)
#define KERNEL_S_TAIL KERNEL_ROW_S_I_TAIL(0) KERNEL_ROW_S_TAIL(1) KERNEL_ROW_S_TAIL(2) KERNEL_ROW_S_TAIL(3) KERNEL_ROW_S_TAIL(4) KERNEL_ROW_S_TAIL(5) KERNEL_ROW_S_TAIL(6) KERNEL_ROW_S_TAIL(7)
#define STORE_S   STORE_ROW_S(0)  STORE_ROW_S(1)  STORE_ROW_S(2)  STORE_ROW_S(3)  STORE_ROW_S(4)  STORE_ROW_S(5)  STORE_ROW_S(6)  STORE_ROW_S(7)

#endif

WORD32 xa_nn_matXvec_8x16_16_circ_nb(
  WORD16 * __restrict__ p_out,
  WORD8  * __restrict__ p_mat,
  WORD16 * __restrict__ p_vec,
  WORD16 * __restrict__ p_bias,
  WORD32 rows,
  WORD32 cols,
  WORD32 out_offset,
  WORD32 bias_shift,
  WORD32 acc_shift)
{
#if (XCHAL_HAVE_HIFI1 || XA_HAVE_HIFI3_CORE)
  WORD32 row, col;
  ae_int16x4 temp_src1;

  if ((NULL == p_out) || (NULL == p_mat) || (NULL == p_vec))
  {
    return -1;
  }

  if ((0 >= rows ) || (0 >= cols ) || (cols & 0x3))
  {
    return -2;
  }

  row = 0;

  if(rows >= UNROLL_S)
  {
    for (row = 0; row < ( rows & ~(UNROLL_S-1)) ; row+=UNROLL_S)
    {
      ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
      SETUP_S;
#pragma ymemory (p_mat1_0)
#pragma ymemory (p_mat1_1)
#pragma ymemory (p_mat1_2)
#pragma ymemory (p_mat1_3)
      for (col = 0; col < (cols>>2); col++) {
        KERNEL_S;
      }
      STORE_S;
    }
  }
  // Handle remaining rows
  for (; row < rows ; row++)
  {
    ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
    SETUP_ROW_S(0);
    for (col = 0; col < (cols>>2); col++) {
      KERNEL_ROW_S_I(0);
    }
    STORE_ROW_S(0);
  }

  return 0;
#else
  WORD32 row, col;
  ae_int16x4 temp_src1, temp_src2;

  if ((NULL == p_out) || (NULL == p_mat) || (NULL == p_vec))
  {
    return -1;
  }

  if ((0 >= rows ) || (0 >= cols ) || (cols & 0x3))
  {
    return -2;
  }

  row = 0;

  if(rows >= UNROLL_S)
  {
    for (row = 0; row < ( rows & ~(UNROLL_S-1)) ; row+=UNROLL_S)
    {
      ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
      SETUP_S;
      for (col = 0; col < (cols >> 3); col++) {
        KERNEL_S;
      }
      if((cols & (7)) !=0){
        KERNEL_S_TAIL;
      }
      STORE_S;
    }
  }
  // Handle remaining rows
  for (; row < rows ; row++)
  {
    ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
    SETUP_ROW_S(0);
    for (col = 0; col < (cols>>2); col++) {
      KERNEL_ROW_S_I_TAIL(0);
    }
    STORE_ROW_S(0);
  }
  return 0;
#endif
}

