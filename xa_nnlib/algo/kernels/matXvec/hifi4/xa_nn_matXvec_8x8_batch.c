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
    #ifdef ROW_UNROLL
        #undef ROW_UNROLL
        #define ROW_UNROLL 4
    #else
    #define ROW_UNROLL 4
    #endif
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common.h"

/*----------------------------Main function---------------------------------*/

WORD32 xa_nn_matXvec_batch_8x8_32(

         WORD32 ** __restrict__ p_out,          /* array of output pointers */
         WORD8 *  __restrict__ p_mat1,         /* matrix1: rows x cols1 */
         WORD8 ** __restrict__ p_vec1,         /* vec1: cols1 x 1 */
         WORD8 *  __restrict__ p_bias,         /* bias TBD: Need array? */
         WORD32 rows,
         WORD32 cols1,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift,                       /* bias shift amount */
         WORD32 vec_count)                      /* number of vectors: 2, 4, 2n */
{
    int i;
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    for(i = 0; i < vec_count; i++)
    {
      XA_NNLIB_ARG_CHK_PTR(p_out[i], -1);
      XA_NNLIB_ARG_CHK_PTR(p_vec1[i], -1);
    }
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32 *), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_mat1, (ALIGNMENT>>1), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD8 *), -1);
    for(i = 0; i < vec_count; i++)
    {
      XA_NNLIB_ARG_CHK_ALIGN(p_out[i], sizeof(WORD32), -1);
      XA_NNLIB_ARG_CHK_ALIGN(p_vec1[i], (ALIGNMENT>>1), -1);
    }
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
    XA_NNLIB_ARG_CHK_COND((vec_count <= 0), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND(((cols1&3) != 0), -1);
    XA_NNLIB_ARG_CHK_COND(((row_stride1&3) != 0), -1);

    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    vec_itr = 0;

    #define VEC_UNROLL 2
    #define UNROLL_ROW_SETUP_ACC_BATCH          SETUP_ACC_BATCH_ROW_FOR_8bx8b
    #define UNROLL_SETUP_ACC_BATCH              SETUP_ACC_BATCH_FOR_8bx8b
    #define UNROLL_SETUP_MAT1                   SETUP_MAT1_8b
    #define UNROLL_SETUP_VEC_BATCH              SETUP_VEC_BATCH_8b
    #define SETUP_BIAS                          SETUP_BIAS_8b_BATCH
    #define UNROLL_LOAD_VEC_BATCH               LOAD_VEC_BATCH_8b
    #define UNROLL_LOAD_ROW_MAT1                LOAD_ROW_MAT1_8b
    #define LOAD_BIAS                           LOAD_BIAS_8b_FOR_8bx8b
    #define UNROLL_ROW_KERNEL_MAT1_VEC_BATCH    KERNEL_MAT1_VEC_BATCH_ROW_8b_8b
    #define UNROLL_KERNEL_MAT1_VEC_BATCH        KERNEL_MAT1_VEC_BATCH_8b_8b
    #define UNROLL_ROW_ADD_BIAS_ACC             ADD_BIAS_BATCH_ROW_8b_ACC_FOR_8bx8b
    #define UNROLL_ADD_BIAS_ACC_BATCH           ADD_BIAS_BATCH_8b_ACC_FOR_8bx8b
#if XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= RI9_HWVERSION)
	#define UNROLL_ROW_STORE_ACC                STORE_ACC_BATCH_ROW_8bx8b_AT_OUT_32bx2
#else
    #define UNROLL_ROW_STORE_ACC                STORE_ACC_BATCH_ROW_8bx8b_AT_OUT_32b
#endif
    #define UNROLL_STORE_ACC_BATCH              STORE_ACC_BATCH_8bx8b_AT_OUT_32b
#if XCHAL_HAVE_HIFI4
    #define UNROLL_SETUP_MAT1_X2                SETUP_MAT1_8b_x2
    #define UNROLL_SETUP_VEC_BATCH_X2           SETUP_VEC_BATCH_8b_x2
    #define UNROLL_LOAD_VEC_BATCH_X2            LOAD_VEC_BATCH_8b_x2
    #define UNROLL_LOAD_ROW_MAT1_X2             LOAD_ROW_MAT1_8b_x2
    #define UNROLL_ROW_KERNEL_MAT1_VEC_BATCH_X2 KERNEL_MAT1_VEC_BATCH_ROW_8b_8b_X2
    #define UNROLL_KERNEL_MAT1_VEC_BATCH_X2     KERNEL_MAT1_VEC_BATCH_8b_8b_x2
#endif

    ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(WORD8, WORD8, WORD32);

    if(rows > ROW_UNROLL)
    {
        if(vec_count > VEC_UNROLL)
        {
            for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
            {
                SETUP_BIAS;
                for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
                {
                    SETUP_ACC_BATCH;
#if XCHAL_HAVE_HIFI4
                    SETUP_VEC_BATCH_X2;
                    SETUP_MAT1_X2;
                    for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
                    {
                        LOAD_VEC_BATCH_X2;
                        LOAD_MAT1_X2;
                        KERNEL_MAT1_VEC_BATCH_X2;
                    }
                    if((cols1 & 7) != 0)
                    {
                        LOAD_VEC_BATCH;
                        LOAD_MAT1;
                        KERNEL_MAT1_VEC_BATCH;
                    }
#else
                    SETUP_VEC_BATCH;
                    SETUP_MAT1;
                    for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                    {
                        LOAD_VEC_BATCH;
                        LOAD_MAT1;
                        KERNEL_MAT1_VEC_BATCH;
                    }
#endif
                    ADD_BIAS_ACC_BATCH;
                    STORE_ACC_BATCH;
                }

                for(; m_itr < rows; m_itr++)
                {
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
                    UNROLL_ROW_STORE_ACC(0);
                }
            }
        }
        {
            /* Tail loop for vec unroll */
            for(; vec_itr < vec_count; vec_itr++)
            {
                SETUP_BIAS;
                for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
                {
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
                    STORE_ACC_BATCH_TAIL;
                }

                for(; m_itr < rows; m_itr++)
                {
                    UNROLL_SETUP_ACC_BATCH(0,0);
                    UNROLL_SETUP_VEC_BATCH(0);
                    UNROLL_SETUP_MAT1(0);

                    for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                    {
                        UNROLL_LOAD_VEC_BATCH(0);
                        UNROLL_LOAD_ROW_MAT1(0);
                        UNROLL_KERNEL_MAT1_VEC_BATCH(0,0);
                    }

                    LOAD_BIAS;
                    UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                    UNROLL_STORE_ACC_BATCH(0,0);
                }
            }
        }
    }
    else
    {
        if(vec_count > VEC_UNROLL)
        {
            for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
            {
                SETUP_BIAS;
                for(m_itr = 0; m_itr < rows; m_itr++)
                {
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
                    UNROLL_ROW_STORE_ACC(0);

                }
            }
        }
        { /* Tail loop for vec unroll */
            for(; vec_itr < vec_count; vec_itr++)
            {
                SETUP_BIAS;

                for(m_itr = 0; m_itr < rows; m_itr++)
                {
                    UNROLL_SETUP_ACC_BATCH(0,0);
                    UNROLL_SETUP_VEC_BATCH(0);
                    UNROLL_SETUP_MAT1(0);

                    for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                    {
                        UNROLL_LOAD_VEC_BATCH(0);
                        UNROLL_LOAD_ROW_MAT1(0);
                        UNROLL_KERNEL_MAT1_VEC_BATCH(0,0);
                    }

                    LOAD_BIAS;
                    UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                    UNROLL_STORE_ACC_BATCH(0,0);
                }
            }
        }
    }


    #undef UNROLL_ROW_SETUP_ACC_BATCH
    #undef UNROLL_SETUP_ACC_BATCH
    #undef UNROLL_SETUP_MAT1
    #undef UNROLL_SETUP_VEC_BATCH
    #undef SETUP_BIAS
    #undef UNROLL_LOAD_VEC_BATCH
    #undef UNROLL_LOAD_ROW_MAT1
    #undef LOAD_BIAS
    #undef UNROLL_ROW_KERNEL_MAT1_VEC_BATCH
    #undef UNROLL_KERNEL_MAT1_VEC_BATCH
    #undef UNROLL_ROW_ADD_BIAS_ACC
    #undef UNROLL_ADD_BIAS_ACC_BATCH
    #undef UNROLL_ROW_STORE_ACC
    #undef UNROLL_STORE_ACC_BATCH
    #undef VEC_UNROLL
    #undef ROW_UNROLL
#if XCHAL_HAVE_HIFI4
    #undef UNROLL_SETUP_MAT1_X2                
    #undef UNROLL_SETUP_VEC_BATCH_X2           
    #undef UNROLL_LOAD_VEC_BATCH_X2            
    #undef UNROLL_LOAD_ROW_MAT1_X2             
    #undef UNROLL_ROW_KERNEL_MAT1_VEC_BATCH_X2 
    #undef UNROLL_KERNEL_MAT1_VEC_BATCH_X2     
#endif
    return 0;
}

#define AE_MULA8Q8X8_HIFI1(d_acc0, d_acc1, mat1_00, mat1_10, mat1_20, mat1_30, vec1_00) \
    tmp_acc64_0 = AE_MULZAAAAQ16(mat1_00, vec1_00); \
    tmp_acc64_1 = AE_MULZAAAAQ16(mat1_10, vec1_00); \
    tmp_acc64_2 = AE_MULZAAAAQ16(mat1_20, vec1_00); \
    tmp_acc64_3 = AE_MULZAAAAQ16(mat1_30, vec1_00); \
    d_acc0 = AE_ADD32(d_acc0, AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(tmp_acc64_0), AE_MOVINT32X2_FROMINT64(tmp_acc64_1))); \
    d_acc1 = AE_ADD32(d_acc1, AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(tmp_acc64_2), AE_MOVINT32X2_FROMINT64(tmp_acc64_3))); \

WORD32 xa_nn_matXvec_acc_batch_sym8sx8_asym16s(
         WORD16 * __restrict__ p_out,           /* output pointer */
         const WORD8 *  __restrict__ p_mat1,    /* matrix1: rows x cols1 */
         const WORD8 * __restrict__ p_vec1,     /* vec1: cols1 x vec_count */
         const WORD32 *  __restrict__ p_bias,   /* bias: rows x 1 */
         WORD32 rows,
         WORD32 cols1,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 out_multiplier,                 /* out multiplier for quantization */
         WORD32 out_shift,                      /* out shift for quantization */
         WORD32 out_zero_bias,						          /* out zero bias for quantization */
         WORD32 vec_count)                      /* number of vectors */
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -32768 || out_zero_bias > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((vec_count < 0), -1);

  /* Iterators used in for loops */
  int m_itr, c_itr, vec_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  vec_itr = 0;
  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
#if XCHAL_HAVE_HIFI1S
  left_shift = 31 - left_shift;
  left_shift = (left_shift << 16) | (left_shift); 
#endif  
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift > 0 ? out_shift : 0;
  right_shift = out_shift < 0 ? -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */

#if XCHAL_HAVE_HIFI1
    ae_valign align_out0, align_out1;
    align_out0 = AE_ZALIGN64();
    align_out1 = AE_ZALIGN64();
#endif

#if XCHAL_HAVE_HIFI1S
  if((cols1&7) == 0 && ((row_stride1&7) == 0) && ((uintptr_t)p_mat1&7) == 0 && ((uintptr_t)p_vec1&7) == 0)
  {
    for (vec_itr = 0; vec_itr < (vec_count - (2 - 1)) ; vec_itr+=2)
    {
      ae_int16x4 *p_out0_16 = (ae_int16x4 *)&p_out[vec_itr*rows];
      ae_int16x4 *p_out1_16 = (ae_int16x4 *)&p_out[(vec_itr+1)*rows];

      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();
        ae_int32x2 d_acc0_1 = AE_ZERO32();
        ae_int32x2 d_acc1_1 = AE_ZERO32();

        ae_int64 acc64_0, acc64_1, acc64_2, acc64_3, acc64_4, acc64_5, acc64_6, acc64_7;
        acc64_0 = acc64_1 = acc64_2 = acc64_3 = acc64_4 = acc64_5 = acc64_6 = acc64_7 = 0;

        ae_int32x2 acc32_0, acc32_1, acc32_2, acc32_3, acc32_4, acc32_5, acc32_6, acc32_7;
        acc32_0 = acc32_1 = acc32_2 = acc32_3 = acc32_4 = acc32_5 = acc32_6 = acc32_7 = 0;

        ae_int8x8 mat_00, mat_10, mat_20, mat_30;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_mat1_1 = (WORD8 *)&p_mat1[(m_itr+1)*row_stride1];
        WORD8 *p_mat1_2 = (WORD8 *)&p_mat1[(m_itr+2)*row_stride1];
        WORD8 *p_mat1_3 = (WORD8 *)&p_mat1[(m_itr+3)*row_stride1];

        ae_int8x8 vec_00, vec_10;
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];
        WORD8 *p_vec1_1 = (WORD8 *)&p_vec1[(vec_itr+1)*cols1];

        for(c_itr = 0; c_itr < (cols1>>3); c_itr++)
        {
          AE_L8X8_IP(mat_00, (ae_int8x8 *)p_mat1_0, 8);
          AE_L8X8_IP(mat_10, (ae_int8x8 *)p_mat1_1, 8);
          AE_L8X8_IP(mat_20, (ae_int8x8 *)p_mat1_2, 8);
          AE_L8X8_IP(mat_30, (ae_int8x8 *)p_mat1_3, 8);
          AE_L8X8_IP(vec_00, (ae_int8x8 *)p_vec1_0, 8);
          AE_L8X8_IP(vec_10, (ae_int8x8 *)p_vec1_1, 8);

          AE_MULAAAAQ8(acc32_0, mat_00, vec_00);
          AE_MULAAAAQ8(acc32_1, mat_10, vec_00);
          AE_MULAAAAQ8(acc32_2, mat_20, vec_00);
          AE_MULAAAAQ8(acc32_3, mat_30, vec_00);

          AE_MULAAAAQ8(acc32_4, mat_00, vec_10);
          AE_MULAAAAQ8(acc32_5, mat_10, vec_10);
          AE_MULAAAAQ8(acc32_6, mat_20, vec_10);
          AE_MULAAAAQ8(acc32_7, mat_30, vec_10);
        }

        acc64_0 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_0, acc32_0)), 0);
        acc64_1 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_1, acc32_1)), 0);
        acc64_2 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_2, acc32_2)), 0);
        acc64_3 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_3, acc32_3)), 0);
        acc64_4 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_4, acc32_4)), 0);
        acc64_5 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_5, acc32_5)), 0);
        acc64_6 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_6, acc32_6)), 0);
        acc64_7 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_7, acc32_7)), 0);

        d_acc0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_0), AE_MOVINT32X2_FROMINT64(acc64_1));
        d_acc1_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_2), AE_MOVINT32X2_FROMINT64(acc64_3));
        d_acc0_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_4), AE_MOVINT32X2_FROMINT64(acc64_5));
        d_acc1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_6), AE_MOVINT32X2_FROMINT64(acc64_7));

        {
          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
          d_acc0_1 = AE_ADD32S(d_acc0_1, d_bias01);
          d_acc1_1 = AE_ADD32S(d_acc1_1, d_bias23);
        }
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        d_acc1_0 = AE_ADD32S(d_acc1_0, AE_MOVDA32(out_zero_bias));
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_1 = AE_ADD32S(d_acc0_1, AE_MOVDA32(out_zero_bias));
        d_acc1_1 = AE_ADD32S(d_acc1_1, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
          out0 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr+2], p_out[(vec_itr+1)*rows+m_itr+3]);
          d_acc0_1 = AE_ADD32S(d_acc0_1, out0);
          d_acc1_1 = AE_ADD32S(d_acc1_1, out1);
        }

        ae_int16x4 ae_int16x4_out;
        ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
        AE_SA16X4_IP(ae_int16x4_out, align_out0, p_out0_16);
        ae_int16x4_out = AE_SAT16X4(d_acc0_1, d_acc1_1);
        AE_SA16X4_IP(ae_int16x4_out, align_out1, p_out1_16);
      }
      AE_SA64POS_FP(align_out0, p_out0_16);
      AE_SA64POS_FP(align_out1, p_out1_16);

#pragma no_unroll
      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();

        ae_int64 tmp_acc64_0, tmp_acc64_1;
        ae_int32x2 acc32_0, acc32_1;
        acc32_0 = acc32_1 = 0;
        ae_int8x8 mat1_00, vec1_00, vec1_10;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];
        WORD8 *p_vec1_1 = (WORD8 *)&p_vec1[(vec_itr+1)*cols1];

        for(c_itr = 0; c_itr < (cols1>>3); c_itr++)
        {
          AE_L8X8_IP(mat1_00, (ae_int8x8 *)p_mat1_0, 8);
          AE_L8X8_IP(vec1_00, (ae_int8x8 *)p_vec1_0, 8);
          AE_L8X8_IP(vec1_10, (ae_int8x8 *)p_vec1_1, 8);

          AE_MULAAAAQ8(acc32_0, mat1_00, vec1_00);
          AE_MULAAAAQ8(acc32_1, mat1_00, vec1_10);
        }

        {
          tmp_acc64_0 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_0, acc32_0)), 0);
          tmp_acc64_1 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_1, acc32_1)), 0);
          d_acc0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(tmp_acc64_0), AE_MOVINT32X2_FROMINT64(tmp_acc64_1));
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);

        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
      }
    }
    for (; vec_itr < vec_count ; vec_itr++)
    {
      ae_int16x4 *p_out0_16 = (ae_int16x4 *)&p_out[vec_itr*rows];

      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();

        ae_int64 tmp_acc64_0, tmp_acc64_1, tmp_acc64_2, tmp_acc64_3;
        ae_int32x2 acc32_0, acc32_1, acc32_2, acc32_3;
        acc32_0 = acc32_1 = acc32_2 = acc32_3 = 0;
        ae_int8x8 mat1_00, mat1_10, mat1_20, mat1_30;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_mat1_1 = (WORD8 *)&p_mat1[(m_itr+1)*row_stride1];
        WORD8 *p_mat1_2 = (WORD8 *)&p_mat1[(m_itr+2)*row_stride1];
        WORD8 *p_mat1_3 = (WORD8 *)&p_mat1[(m_itr+3)*row_stride1];

        ae_int8x8 vec1_00;
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];

        for(c_itr = 0; c_itr < (cols1>>3); c_itr++)
        {
          AE_L8X8_IP(mat1_00, (ae_int8x8 *)p_mat1_0, 8);
          AE_L8X8_IP(mat1_10, (ae_int8x8 *)p_mat1_1, 8);
          AE_L8X8_IP(mat1_20, (ae_int8x8 *)p_mat1_2, 8);
          AE_L8X8_IP(mat1_30, (ae_int8x8 *)p_mat1_3, 8);
          AE_L8X8_IP(vec1_00, (ae_int8x8 *)p_vec1_0, 8);

          AE_MULAAAAQ8(acc32_0, mat1_00, vec1_00);
          AE_MULAAAAQ8(acc32_1, mat1_10, vec1_00);
          AE_MULAAAAQ8(acc32_2, mat1_20, vec1_00);
          AE_MULAAAAQ8(acc32_3, mat1_30, vec1_00);
        }

        {
          tmp_acc64_0 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_0, acc32_0)), 0);
          tmp_acc64_1 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_1, acc32_1)), 0);
          d_acc0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(tmp_acc64_0), AE_MOVINT32X2_FROMINT64(tmp_acc64_1));
          tmp_acc64_2 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_2, acc32_2)), 0);
          tmp_acc64_3 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_3, acc32_3)), 0);
          d_acc1_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(tmp_acc64_2), AE_MOVINT32X2_FROMINT64(tmp_acc64_3));

          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
        }
#if TFLITE_SINGLE_ROUNDING   
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        d_acc1_0 = AE_ADD32S(d_acc1_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
        AE_SA16X4_IP(_ae_int16x4_out, align_out0, p_out0_16);
      }
      AE_SA64POS_FP(align_out0, p_out0_16);

#pragma no_unroll
      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0;
        ae_int64 d64_acc0 = AE_ZERO64();

        ae_int16x4 _ae8x8_mat1_00, _ae8x8_vec1_00;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];

        for(c_itr = 0; c_itr < (cols1>>2); c_itr++)
        {
          AE_L8X4S_IP(_ae8x8_mat1_00, p_mat1_0, 4);
          AE_L8X4S_IP(_ae8x8_vec1_00, p_vec1_0, 4);

          AE_MULAAAAQ16(d64_acc0, _ae8x8_mat1_00, _ae8x8_vec1_00);
        }
        ae_int32x2 tmp = AE_MOVINT32X2_FROMINT64(AE_SLAI64S(d64_acc0, 32));
        d_acc0_0 = AE_SEL32_HH(tmp, tmp);
        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32(p_out[vec_itr*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
      }
    }
  }
  else
#endif  
  if((cols1&3) == 0 && ((row_stride1&3) == 0) && ((uintptr_t)p_mat1&3) == 0 && ((uintptr_t)p_vec1&3) == 0)
  {
    for (vec_itr = 0; vec_itr < (vec_count - (2 - 1)) ; vec_itr+=2)
    {
#if XCHAL_HAVE_HIFI1
      ae_int16x4 *p_out0_16 = (ae_int16x4 *)&p_out[vec_itr*rows];
      ae_int16x4 *p_out1_16 = (ae_int16x4 *)&p_out[(vec_itr+1)*rows];
#endif
      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();
        ae_int32x2 d_acc0_1 = AE_ZERO32();
        ae_int32x2 d_acc1_1 = AE_ZERO32();

        ae_int64 acc64_0, acc64_1, acc64_2, acc64_3, acc64_4, acc64_5, acc64_6, acc64_7;
        acc64_0 = acc64_1 = acc64_2 = acc64_3 = acc64_4 = acc64_5 = acc64_6 = acc64_7 = 0;

        ae_int16x4 mat_00, mat_10, mat_20, mat_30;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_mat1_1 = (WORD8 *)&p_mat1[(m_itr+1)*row_stride1];
        WORD8 *p_mat1_2 = (WORD8 *)&p_mat1[(m_itr+2)*row_stride1];
        WORD8 *p_mat1_3 = (WORD8 *)&p_mat1[(m_itr+3)*row_stride1];

        ae_int16x4 vec_00, vec_10;
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];
        WORD8 *p_vec1_1 = (WORD8 *)&p_vec1[(vec_itr+1)*cols1];

        for(c_itr = 0; c_itr < (cols1>>2); c_itr++)
        {
#if XCHAL_HAVE_HIFI1
          AE_L8X4S_IP(mat_00, p_mat1_0, 4);
          AE_L8X4S_IP(mat_10, p_mat1_1, 4);
          AE_L8X4S_IP(mat_20, p_mat1_2, 4);
          AE_L8X4S_IP(mat_30, p_mat1_3, 4);
          AE_L8X4S_IP(vec_00, p_vec1_0, 4);
          AE_L8X4S_IP(vec_10, p_vec1_1, 4);
#else
          AE_L8X4F_IP(mat_00, p_mat1_0, 4);
          AE_L8X4F_IP(mat_10, p_mat1_1, 4);
          AE_L8X4F_IP(mat_20, p_mat1_2, 4);
          AE_L8X4F_IP(mat_30, p_mat1_3, 4);
          AE_L8X4F_IP(vec_00, p_vec1_0, 4);
          AE_L8X4F_IP(vec_10, p_vec1_1, 4);
#endif

          AE_MULAAAAQ16(acc64_0, mat_00, vec_00);
          AE_MULAAAAQ16(acc64_1, mat_10, vec_00);
          AE_MULAAAAQ16(acc64_2, mat_20, vec_00);
          AE_MULAAAAQ16(acc64_3, mat_30, vec_00);

          AE_MULAAAAQ16(acc64_4, mat_00, vec_10);
          AE_MULAAAAQ16(acc64_5, mat_10, vec_10);
          AE_MULAAAAQ16(acc64_6, mat_20, vec_10);
          AE_MULAAAAQ16(acc64_7, mat_30, vec_10);
        }

#if !XCHAL_HAVE_HIFI1
        acc64_0 = AE_SRAI64(acc64_0, 16);
        acc64_1 = AE_SRAI64(acc64_1, 16);
        acc64_2 = AE_SRAI64(acc64_2, 16);
        acc64_3 = AE_SRAI64(acc64_3, 16);
        acc64_4 = AE_SRAI64(acc64_4, 16);
        acc64_5 = AE_SRAI64(acc64_5, 16);
        acc64_6 = AE_SRAI64(acc64_6, 16);
        acc64_7 = AE_SRAI64(acc64_7, 16);
#endif

        d_acc0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_0), AE_MOVINT32X2_FROMINT64(acc64_1));
        d_acc1_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_2), AE_MOVINT32X2_FROMINT64(acc64_3));
        d_acc0_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_4), AE_MOVINT32X2_FROMINT64(acc64_5));
        d_acc1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_6), AE_MOVINT32X2_FROMINT64(acc64_7));

        {
          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
          d_acc0_1 = AE_ADD32S(d_acc0_1, d_bias01);
          d_acc1_1 = AE_ADD32S(d_acc1_1, d_bias23);
        }

#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        d_acc1_0 = AE_ADD32S(d_acc1_0, AE_MOVDA32(out_zero_bias));

#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_1 = AE_ADD32S(d_acc0_1, AE_MOVDA32(out_zero_bias));
        d_acc1_1 = AE_ADD32S(d_acc1_1, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
          out0 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr+2], p_out[(vec_itr+1)*rows+m_itr+3]);
          d_acc0_1 = AE_ADD32S(d_acc0_1, out0);
          d_acc1_1 = AE_ADD32S(d_acc1_1, out1);
        }

        ae_int16x4 ae_int16x4_out;
        ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
#if XCHAL_HAVE_HIFI1
        AE_SA16X4_IP(ae_int16x4_out, align_out0, p_out0_16);
#else
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr  ] = AE_SEL16_6543(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = ae_int16x4_out;
#endif

        ae_int16x4_out = AE_SAT16X4(d_acc0_1, d_acc1_1);
#if XCHAL_HAVE_HIFI1
        AE_SA16X4_IP(ae_int16x4_out, align_out1, p_out1_16);
#else
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr  ] = AE_SEL16_6543(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+1] = AE_SEL16_5432(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+2] = AE_SEL16_4321(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+3] = ae_int16x4_out;
#endif
      }

#if XCHAL_HAVE_HIFI1
      AE_SA64POS_FP(align_out0, p_out0_16);
      AE_SA64POS_FP(align_out1, p_out1_16);
#endif

#pragma no_unroll
      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc0_1 = AE_ZERO32();

        ae_int64 tmp_acc64_0, tmp_acc64_1, tmp_acc64_2, tmp_acc64_3;

        ae_int16x4 mat1_00, vec1_00, vec1_10;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];
        WORD8 *p_vec1_1 = (WORD8 *)&p_vec1[(vec_itr+1)*cols1];

        for(c_itr = 0; c_itr < (cols1>>2); c_itr++)
        {
          AE_L8X4S_IP(mat1_00, p_mat1_0, 4);
          AE_L8X4S_IP(vec1_00, p_vec1_0, 4);
          AE_L8X4S_IP(vec1_10, p_vec1_1, 4);

          AE_MULA8Q8X8_HIFI1(d_acc0_0, d_acc0_1, vec1_00, vec1_10, vec1_00, vec1_10, mat1_00);
        }

        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else     
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);

        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
      }
    }
    for (; vec_itr < vec_count ; vec_itr++)
    {
#if XCHAL_HAVE_HIFI1
      ae_int16x4 *p_out0_16 = (ae_int16x4 *)&p_out[vec_itr*rows];
#endif

      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();

        ae_int64 tmp_acc64_0, tmp_acc64_1, tmp_acc64_2, tmp_acc64_3;
        ae_int16x4 _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_mat1_1 = (WORD8 *)&p_mat1[(m_itr+1)*row_stride1];
        WORD8 *p_mat1_2 = (WORD8 *)&p_mat1[(m_itr+2)*row_stride1];
        WORD8 *p_mat1_3 = (WORD8 *)&p_mat1[(m_itr+3)*row_stride1];

        ae_int16x4 _ae8x8_vec1_00;
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];

        for(c_itr = 0; c_itr < (cols1>>2); c_itr++)
        {
          AE_L8X4S_IP(_ae8x8_mat1_00, p_mat1_0, 4);
          AE_L8X4S_IP(_ae8x8_mat1_10, p_mat1_1, 4);
          AE_L8X4S_IP(_ae8x8_mat1_20, p_mat1_2, 4);
          AE_L8X4S_IP(_ae8x8_mat1_30, p_mat1_3, 4);
          AE_L8X4S_IP(_ae8x8_vec1_00, p_vec1_0, 4);

          AE_MULA8Q8X8_HIFI1(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_00);
        }

        {
          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
        }

#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)        
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        d_acc1_0 = AE_ADD32S(d_acc1_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
#if XCHAL_HAVE_HIFI1
        AE_SA16X4_IP(_ae_int16x4_out, align_out0, p_out0_16);
#else
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = (_ae_int16x4_out);
#endif
      }

#if XCHAL_HAVE_HIFI1
      AE_SA64POS_FP(align_out0, p_out0_16);
#endif

#pragma no_unroll
      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0;
        ae_int64 d64_acc0 = AE_ZERO64();

        ae_int16x4 _ae8x8_mat1_00, _ae8x8_vec1_00;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];

        for(c_itr = 0; c_itr < (cols1>>2); c_itr++)
        {
          AE_L8X4S_IP(_ae8x8_mat1_00, p_mat1_0, 4);
          AE_L8X4S_IP(_ae8x8_vec1_00, p_vec1_0, 4);

          AE_MULAAAAQ16(d64_acc0, _ae8x8_mat1_00, _ae8x8_vec1_00);
        }
        ae_int32x2 tmp = AE_MOVINT32X2_FROMINT64(AE_SLAI64S(d64_acc0, 32));
        d_acc0_0 = AE_SEL32_HH(tmp, tmp);
        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32(p_out[vec_itr*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
      }
    }
  }
  else
  {
    for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
    {
#if XCHAL_HAVE_HIFI1
      ae_int16x4 *p_out0_16 = (ae_int16x4 *)&p_out[vec_itr*rows];
#endif
      for(m_itr = 0; m_itr < (rows & ~3); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();

        ae_int16x4 vec_00, mat_00, mat_10, mat_20, mat_30;

        ae_int64 acc64_0, acc64_1, acc64_2, acc64_3;
        acc64_0 = acc64_1 = acc64_2 = acc64_3 = 0;

        WORD8 *p_vec_0 = (WORD8 *) &p_vec1[vec_itr*cols1];

        WORD8 *p_mat_0 = (WORD8 *) &p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_mat_1 = (WORD8 *) &p_mat1[(m_itr+1)*row_stride1];
        WORD8 *p_mat_2 = (WORD8 *) &p_mat1[(m_itr+2)*row_stride1];
        WORD8 *p_mat_3 = (WORD8 *) &p_mat1[(m_itr+3)*row_stride1];

        /* Compatibility ALIGN_REGISTER_TYPE is int16x4 for (use by) AE_LA8X4F_IP implementation
         * on HiFi1/4, so can use 5 of them
         */
        ALIGN_REGISTER_TYPE align_p_vec_0,
            align_p_mat_0, align_p_mat_1, align_p_mat_2, align_p_mat_3;

        PRIME_8X4F(p_mat_0, align_p_mat_0);
        PRIME_8X4F(p_mat_1, align_p_mat_1);
        PRIME_8X4F(p_mat_2, align_p_mat_2);
        PRIME_8X4F(p_mat_3, align_p_mat_3);
        PRIME_8X4F(p_vec_0, align_p_vec_0);

        int cols_count=cols1&(~3);
        for(c_itr = 0; c_itr < cols_count>>2; c_itr++)
        {
#if XCHAL_HAVE_HIFI1
          AE_LA8X4S_IP(vec_00, align_p_vec_0, p_vec_0);
                  
          AE_LA8X4S_IP(mat_00, align_p_mat_0, p_mat_0);
          AE_LA8X4S_IP(mat_10, align_p_mat_1, p_mat_1);
          AE_LA8X4S_IP(mat_20, align_p_mat_2, p_mat_2);
          AE_LA8X4S_IP(mat_30, align_p_mat_3, p_mat_3);
#else
          AE_LA8X4F_IP(vec_00, align_p_vec_0, p_vec_0);

          AE_LA8X4F_IP(mat_00, align_p_mat_0, p_mat_0);
          AE_LA8X4F_IP(mat_10, align_p_mat_1, p_mat_1);
          AE_LA8X4F_IP(mat_20, align_p_mat_2, p_mat_2);
          AE_LA8X4F_IP(mat_30, align_p_mat_3, p_mat_3);
#endif

          AE_MULAAAAQ16(acc64_0, mat_00, vec_00);
          AE_MULAAAAQ16(acc64_1, mat_10, vec_00);
          AE_MULAAAAQ16(acc64_2, mat_20, vec_00);
          AE_MULAAAAQ16(acc64_3, mat_30, vec_00);
        }

#if !XCHAL_HAVE_HIFI1
        acc64_0 = AE_SRAI64(acc64_0, 16);
        acc64_1 = AE_SRAI64(acc64_1, 16);
        acc64_2 = AE_SRAI64(acc64_2, 16);
        acc64_3 = AE_SRAI64(acc64_3, 16);
#endif

        d_acc0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_0), AE_MOVINT32X2_FROMINT64(acc64_1));
        d_acc1_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_2), AE_MOVINT32X2_FROMINT64(acc64_3));

        int rem_itr = 0;
        for(rem_itr = 0 ; rem_itr <(cols1 - cols_count); rem_itr++)
        {
          mat_00 = AE_MOVDA16((WORD16)*p_mat_0); p_mat_0++;
          mat_10 = AE_MOVDA16((WORD16)*p_mat_1); p_mat_1++;
          mat_20 = AE_MOVDA16((WORD16)*p_mat_2); p_mat_2++;
          mat_30 = AE_MOVDA16((WORD16)*p_mat_3); p_mat_3++;
          vec_00 = AE_MOVDA16((WORD16)*p_vec_0); p_vec_0++;

          /* mul_mtx = (ae_int16x4){mat1_00, mat1_10, mat1_20, mat1_30} */
          ae_int16x4 mul_mtx = AE_SEL16_7531(
            AE_SEL16_5432(mat_00, mat_10), AE_SEL16_5432(mat_20, mat_30) );

          AE_MULA16X4(d_acc0_0, d_acc1_0, mul_mtx, vec_00);
        }
        {
          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc1_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc1_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        d_acc1_0 = AE_ADD32S(d_acc1_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr  ], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
        }

        ae_int16x4 ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);

#if XCHAL_HAVE_HIFI1
        AE_SA16X4_IP(ae_int16x4_out, align_out0, p_out0_16);
#else
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr  ] = AE_SEL16_6543(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = ae_int16x4_out;
#endif
      }

#if XCHAL_HAVE_HIFI1
      AE_SA64POS_FP(align_out0, p_out0_16);
#endif

      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0;
        ae_int64 d64_acc0 = AE_ZERO64();

        ae_int16x4 _ae8x8_mat1_00, _ae8x8_vec1_00;
        ALIGN_REGISTER_TYPE _align_ae8x16_p_mat1_0, _align_ae8x16_p_vec1_0;

        WORD8 *_ae8x16_p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *_ae8x16_p_vec1_0 = (WORD8 *)&p_vec1[vec_itr*cols1];

        PRIME_8X4F(_ae8x16_p_mat1_0, _align_ae8x16_p_mat1_0);
        PRIME_8X4F(_ae8x16_p_vec1_0, _align_ae8x16_p_vec1_0);

        int cols_count=cols1&(~3);
        for(c_itr = 0; c_itr < cols_count>>2; c_itr++)
        {

#if XCHAL_HAVE_HIFI1
          AE_LA8X4S_IP(_ae8x8_mat1_00, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0);
          AE_LA8X4S_IP(_ae8x8_vec1_00, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
#else
          AE_LA8X4F_IP(_ae8x8_mat1_00, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0);
          AE_LA8X4F_IP(_ae8x8_vec1_00, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);

          _ae8x8_mat1_00 = AE_SRAI16(_ae8x8_mat1_00, 8);
          _ae8x8_vec1_00 = AE_SRAI16(_ae8x8_vec1_00, 8);
#endif

          AE_MULAAAAQ16(d64_acc0, _ae8x8_mat1_00, _ae8x8_vec1_00);
        }

        int rem_itr = 0;
        ae_int64 acc_tmp = AE_ZERO64();
        for(rem_itr = 0 ; rem_itr <(cols1 - cols_count); rem_itr++)
        {
          _ae8x8_mat1_00 = AE_MOVDA16( (WORD16)*_ae8x16_p_mat1_0); _ae8x16_p_mat1_0++;
          _ae8x8_vec1_00 = AE_MOVDA16( (WORD16)*_ae8x16_p_vec1_0); _ae8x16_p_vec1_0++;

          AE_MULAAAAQ16(acc_tmp, _ae8x8_mat1_00, _ae8x8_vec1_00);
        }
        acc_tmp = AE_SRAI64(acc_tmp, 2);
        d64_acc0 = AE_ADD64(d64_acc0, acc_tmp);
        ae_int32x2 tmp = AE_MOVINT32X2_FROMINT64(AE_SLAI64S(d64_acc0, 32));
        d_acc0_0 = AE_SEL32_HH(tmp, tmp);
        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32(p_out[vec_itr*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = _ae_int16x4_out;
      }
    }
  }

  return 0;
}

#define AE_MINMAX32_HIFI4(acc, min, max) \
    acc = AE_MAX32(min, AE_MIN32(max, acc));

WORD32 xa_nn_matXvec_acc_batch_sym8sx8_asym16s_hU(
         WORD16 * __restrict__ p_out,           /* output pointer */
         const WORD8 *  __restrict__ p_mat1,    /* matrix1: rows x cols1 */
         const WORD8 * __restrict__ p_vec1,     /* vec1: cols1 x vec_count */
         const WORD32 *  __restrict__ p_bias,   /* bias: rows x 1 */
         WORD32 rows,
         WORD32 cols1,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 out_multiplier,                 /* out multiplier for quantization */
         WORD32 out_shift,                      /* out shift for quantization */
         WORD32 out_zero_bias,						          /* out zero bias for quantization */
         WORD32 vec_count)                      /* number of vectors */
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -32768 || out_zero_bias > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((vec_count < 0), -1);

  /* Iterators used in for loops */
  int m_itr, c_itr, vec_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  vec_itr = 0;
  int left_shift, right_shift;

  ae_int32x2 d_outzb = AE_MOVDA32(out_zero_bias);

  out_shift = out_shift - 1;
  d_outzb = AE_SRAI32(d_outzb, 1);

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
#if XCHAL_HAVE_HIFI1S
  left_shift = 31 - left_shift;
  left_shift = (left_shift << 16) | (left_shift); 
#endif    
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift > 0 ? out_shift : 0;
  right_shift = out_shift < 0 ? -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */

#if XCHAL_HAVE_HIFI1
    ae_valign align_out0, align_out1;
    align_out0 = AE_ZALIGN64();
    align_out1 = AE_ZALIGN64();
#endif

  ae_int32x2 min_word16 = AE_MOVDA32(-32768);
  ae_int32x2 max_word16 = AE_MOVDA32(32767);

#if XCHAL_HAVE_HIFI1S
  if((cols1&7) == 0 && ((row_stride1&7) == 0) && ((uintptr_t)p_mat1&7) == 0 && ((uintptr_t)p_vec1&7) == 0)
  {
    for (vec_itr = 0; vec_itr < (vec_count - (2 - 1)) ; vec_itr+=2)
    {
      ae_int16x4 *p_out0_16 = (ae_int16x4 *)&p_out[vec_itr*rows];
      ae_int16x4 *p_out1_16 = (ae_int16x4 *)&p_out[(vec_itr+1)*rows];

      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();
        ae_int32x2 d_acc0_1 = AE_ZERO32();
        ae_int32x2 d_acc1_1 = AE_ZERO32();

        ae_int64 acc64_0, acc64_1, acc64_2, acc64_3, acc64_4, acc64_5, acc64_6, acc64_7;
        acc64_0 = acc64_1 = acc64_2 = acc64_3 = acc64_4 = acc64_5 = acc64_6 = acc64_7 = 0;

        ae_int32x2 acc32_0, acc32_1, acc32_2, acc32_3, acc32_4, acc32_5, acc32_6, acc32_7;
        acc32_0 = acc32_1 = acc32_2 = acc32_3 = acc32_4 = acc32_5 = acc32_6 = acc32_7 = 0;

        ae_int8x8 mat_00, mat_10, mat_20, mat_30;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_mat1_1 = (WORD8 *)&p_mat1[(m_itr+1)*row_stride1];
        WORD8 *p_mat1_2 = (WORD8 *)&p_mat1[(m_itr+2)*row_stride1];
        WORD8 *p_mat1_3 = (WORD8 *)&p_mat1[(m_itr+3)*row_stride1];

        ae_int8x8 vec_00, vec_10;
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];
        WORD8 *p_vec1_1 = (WORD8 *)&p_vec1[(vec_itr+1)*cols1];

        for(c_itr = 0; c_itr < (cols1>>3); c_itr++)
        {
          AE_L8X8_IP(mat_00, (ae_int8x8 *)p_mat1_0, 8);
          AE_L8X8_IP(mat_10, (ae_int8x8 *)p_mat1_1, 8);
          AE_L8X8_IP(mat_20, (ae_int8x8 *)p_mat1_2, 8);
          AE_L8X8_IP(mat_30, (ae_int8x8 *)p_mat1_3, 8);
          AE_L8X8_IP(vec_00, (ae_int8x8 *)p_vec1_0, 8);
          AE_L8X8_IP(vec_10, (ae_int8x8 *)p_vec1_1, 8);

          AE_MULAAAAQ8(acc32_0, mat_00, vec_00);
          AE_MULAAAAQ8(acc32_1, mat_10, vec_00);
          AE_MULAAAAQ8(acc32_2, mat_20, vec_00);
          AE_MULAAAAQ8(acc32_3, mat_30, vec_00);

          AE_MULAAAAQ8(acc32_4, mat_00, vec_10);
          AE_MULAAAAQ8(acc32_5, mat_10, vec_10);
          AE_MULAAAAQ8(acc32_6, mat_20, vec_10);
          AE_MULAAAAQ8(acc32_7, mat_30, vec_10);
        }

        acc64_0 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_0, acc32_0)), 0);
        acc64_1 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_1, acc32_1)), 0);
        acc64_2 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_2, acc32_2)), 0);
        acc64_3 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_3, acc32_3)), 0);
        acc64_4 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_4, acc32_4)), 0);
        acc64_5 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_5, acc32_5)), 0);
        acc64_6 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_6, acc32_6)), 0);
        acc64_7 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_7, acc32_7)), 0);

        d_acc0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_0), AE_MOVINT32X2_FROMINT64(acc64_1));
        d_acc1_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_2), AE_MOVINT32X2_FROMINT64(acc64_3));
        d_acc0_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_4), AE_MOVINT32X2_FROMINT64(acc64_5));
        d_acc1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_6), AE_MOVINT32X2_FROMINT64(acc64_7));

        {
          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
          d_acc0_1 = AE_ADD32S(d_acc0_1, d_bias01);
          d_acc1_1 = AE_ADD32S(d_acc1_1, d_bias23);
        }

#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)  
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        d_acc1_0 = AE_ADD32S(d_acc1_0, d_outzb);

#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S) 
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_1 = AE_ADD32S(d_acc0_1, d_outzb);
        d_acc1_1 = AE_ADD32S(d_acc1_1, d_outzb);
        AE_MINMAX32_HIFI4(d_acc0_0, min_word16, max_word16);
        AE_MINMAX32_HIFI4(d_acc1_0, min_word16, max_word16);
        AE_MINMAX32_HIFI4(d_acc0_1, min_word16, max_word16);
        AE_MINMAX32_HIFI4(d_acc1_1, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        d_acc1_0 = AE_SLAI32(d_acc1_0, 1);
        d_acc0_1 = AE_SLAI32(d_acc0_1, 1);
        d_acc1_1 = AE_SLAI32(d_acc1_1, 1);
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
          out0 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr+2], p_out[(vec_itr+1)*rows+m_itr+3]);
          d_acc0_1 = AE_ADD32S(d_acc0_1, out0);
          d_acc1_1 = AE_ADD32S(d_acc1_1, out1);
        }

        ae_int16x4 ae_int16x4_out;
        ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
        AE_SA16X4_IP(ae_int16x4_out, align_out0, p_out0_16);
        ae_int16x4_out = AE_SAT16X4(d_acc0_1, d_acc1_1);
        AE_SA16X4_IP(ae_int16x4_out, align_out1, p_out1_16);
      }
      AE_SA64POS_FP(align_out0, p_out0_16);
      AE_SA64POS_FP(align_out1, p_out1_16);

#pragma no_unroll
      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();

        ae_int64 tmp_acc64_0, tmp_acc64_1;
        ae_int32x2 acc32_0, acc32_1;
        acc32_0 = acc32_1 = 0;
        ae_int8x8 mat1_00, vec1_00, vec1_10;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];
        WORD8 *p_vec1_1 = (WORD8 *)&p_vec1[(vec_itr+1)*cols1];

        for(c_itr = 0; c_itr < (cols1>>3); c_itr++)
        {
          AE_L8X8_IP(mat1_00, (ae_int8x8 *)p_mat1_0, 8);
          AE_L8X8_IP(vec1_00, (ae_int8x8 *)p_vec1_0, 8);
          AE_L8X8_IP(vec1_10, (ae_int8x8 *)p_vec1_1, 8);

          AE_MULAAAAQ8(acc32_0, mat1_00, vec1_00);
          AE_MULAAAAQ8(acc32_1, mat1_00, vec1_10);
        }

        {
          tmp_acc64_0 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_0, acc32_0)), 0);
          tmp_acc64_1 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_1, acc32_1)), 0);
          d_acc0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(tmp_acc64_0), AE_MOVINT32X2_FROMINT64(tmp_acc64_1));
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        AE_MINMAX32_HIFI4(d_acc0_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);

        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
      }
    }
    for (; vec_itr < vec_count ; vec_itr++)
    {
      ae_int16x4 *p_out0_16 = (ae_int16x4 *)&p_out[vec_itr*rows];

      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();

        ae_int64 tmp_acc64_0, tmp_acc64_1, tmp_acc64_2, tmp_acc64_3;
        ae_int32x2 acc32_0, acc32_1, acc32_2, acc32_3;
        acc32_0 = acc32_1 = acc32_2 = acc32_3 = 0;

        ae_int8x8 mat1_00, mat1_10, mat1_20, mat1_30;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_mat1_1 = (WORD8 *)&p_mat1[(m_itr+1)*row_stride1];
        WORD8 *p_mat1_2 = (WORD8 *)&p_mat1[(m_itr+2)*row_stride1];
        WORD8 *p_mat1_3 = (WORD8 *)&p_mat1[(m_itr+3)*row_stride1];

        ae_int8x8 vec1_00;
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];

        for(c_itr = 0; c_itr < (cols1>>3); c_itr++)
        {
          AE_L8X8_IP(mat1_00, (ae_int8x8 *)p_mat1_0, 8);
          AE_L8X8_IP(mat1_10, (ae_int8x8 *)p_mat1_1, 8);
          AE_L8X8_IP(mat1_20, (ae_int8x8 *)p_mat1_2, 8);
          AE_L8X8_IP(mat1_30, (ae_int8x8 *)p_mat1_3, 8);
          AE_L8X8_IP(vec1_00, (ae_int8x8 *)p_vec1_0, 8);

          AE_MULAAAAQ8(acc32_0, mat1_00, vec1_00);
          AE_MULAAAAQ8(acc32_1, mat1_10, vec1_00);
          AE_MULAAAAQ8(acc32_2, mat1_20, vec1_00);
          AE_MULAAAAQ8(acc32_3, mat1_30, vec1_00);
        }

        {
          tmp_acc64_0 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_0, acc32_0)), 0);
          tmp_acc64_1 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_1, acc32_1)), 0);
          d_acc0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(tmp_acc64_0), AE_MOVINT32X2_FROMINT64(tmp_acc64_1));
          tmp_acc64_2 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_2, acc32_2)), 0);
          tmp_acc64_3 = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_ADD32_HL_LH(acc32_3, acc32_3)), 0);
          d_acc1_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(tmp_acc64_2), AE_MOVINT32X2_FROMINT64(tmp_acc64_3));

          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S) 
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        d_acc1_0 = AE_ADD32S(d_acc1_0, d_outzb);
        AE_MINMAX32_HIFI4(d_acc0_0, min_word16, max_word16);
        AE_MINMAX32_HIFI4(d_acc1_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        d_acc1_0 = AE_SLAI32(d_acc1_0, 1);
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
        AE_SA16X4_IP(_ae_int16x4_out, align_out0, p_out0_16);
      }
      AE_SA64POS_FP(align_out0, p_out0_16);

#pragma no_unroll
      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0;
        ae_int64 d64_acc0 = AE_ZERO64();

        ae_int16x4 _ae8x8_mat1_00, _ae8x8_vec1_00;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];

        for(c_itr = 0; c_itr < (cols1>>2); c_itr++)
        {
          AE_L8X4S_IP(_ae8x8_mat1_00, p_mat1_0, 4);
          AE_L8X4S_IP(_ae8x8_vec1_00, p_vec1_0, 4);

          AE_MULAAAAQ16(d64_acc0, _ae8x8_mat1_00, _ae8x8_vec1_00);
        }
        ae_int32x2 tmp = AE_MOVINT32X2_FROMINT64(AE_SLAI64S(d64_acc0, 32));
        d_acc0_0 = AE_SEL32_HH(tmp, tmp);
        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        AE_MINMAX32_HIFI4(d_acc0_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32(p_out[vec_itr*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
      }
    }
  }
  else
#endif
  if((cols1&3) == 0 && ((row_stride1&3) == 0) && ((uintptr_t)p_mat1&3) == 0 && ((uintptr_t)p_vec1&3) == 0)
  {
    for (vec_itr = 0; vec_itr < (vec_count - (2 - 1)) ; vec_itr+=2)
    {
#if XCHAL_HAVE_HIFI1
      ae_int16x4 *p_out0_16 = (ae_int16x4 *)&p_out[vec_itr*rows];
      ae_int16x4 *p_out1_16 = (ae_int16x4 *)&p_out[(vec_itr+1)*rows];
#endif
      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();
        ae_int32x2 d_acc0_1 = AE_ZERO32();
        ae_int32x2 d_acc1_1 = AE_ZERO32();

        ae_int64 acc64_0, acc64_1, acc64_2, acc64_3, acc64_4, acc64_5, acc64_6, acc64_7;
        acc64_0 = acc64_1 = acc64_2 = acc64_3 = acc64_4 = acc64_5 = acc64_6 = acc64_7 = 0;

        ae_int16x4 mat_00, mat_10, mat_20, mat_30;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_mat1_1 = (WORD8 *)&p_mat1[(m_itr+1)*row_stride1];
        WORD8 *p_mat1_2 = (WORD8 *)&p_mat1[(m_itr+2)*row_stride1];
        WORD8 *p_mat1_3 = (WORD8 *)&p_mat1[(m_itr+3)*row_stride1];

        ae_int16x4 vec_00, vec_10;
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];
        WORD8 *p_vec1_1 = (WORD8 *)&p_vec1[(vec_itr+1)*cols1];

        for(c_itr = 0; c_itr < (cols1>>2); c_itr++)
        {
#if XCHAL_HAVE_HIFI1
          AE_L8X4S_IP(mat_00, p_mat1_0, 4);
          AE_L8X4S_IP(mat_10, p_mat1_1, 4);
          AE_L8X4S_IP(mat_20, p_mat1_2, 4);
          AE_L8X4S_IP(mat_30, p_mat1_3, 4);
          AE_L8X4S_IP(vec_00, p_vec1_0, 4);
          AE_L8X4S_IP(vec_10, p_vec1_1, 4);
#else
          AE_L8X4F_IP(mat_00, p_mat1_0, 4);
          AE_L8X4F_IP(mat_10, p_mat1_1, 4);
          AE_L8X4F_IP(mat_20, p_mat1_2, 4);
          AE_L8X4F_IP(mat_30, p_mat1_3, 4);
          AE_L8X4F_IP(vec_00, p_vec1_0, 4);
          AE_L8X4F_IP(vec_10, p_vec1_1, 4);
#endif

          AE_MULAAAAQ16(acc64_0, mat_00, vec_00);
          AE_MULAAAAQ16(acc64_1, mat_10, vec_00);
          AE_MULAAAAQ16(acc64_2, mat_20, vec_00);
          AE_MULAAAAQ16(acc64_3, mat_30, vec_00);

          AE_MULAAAAQ16(acc64_4, mat_00, vec_10);
          AE_MULAAAAQ16(acc64_5, mat_10, vec_10);
          AE_MULAAAAQ16(acc64_6, mat_20, vec_10);
          AE_MULAAAAQ16(acc64_7, mat_30, vec_10);
        }

#if !XCHAL_HAVE_HIFI1
        acc64_0 = AE_SRAI64(acc64_0, 16);
        acc64_1 = AE_SRAI64(acc64_1, 16);
        acc64_2 = AE_SRAI64(acc64_2, 16);
        acc64_3 = AE_SRAI64(acc64_3, 16);
        acc64_4 = AE_SRAI64(acc64_4, 16);
        acc64_5 = AE_SRAI64(acc64_5, 16);
        acc64_6 = AE_SRAI64(acc64_6, 16);
        acc64_7 = AE_SRAI64(acc64_7, 16);
#endif

        d_acc0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_0), AE_MOVINT32X2_FROMINT64(acc64_1));
        d_acc1_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_2), AE_MOVINT32X2_FROMINT64(acc64_3));
        d_acc0_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_4), AE_MOVINT32X2_FROMINT64(acc64_5));
        d_acc1_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_6), AE_MOVINT32X2_FROMINT64(acc64_7));

        {
          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
          d_acc0_1 = AE_ADD32S(d_acc0_1, d_bias01);
          d_acc1_1 = AE_ADD32S(d_acc1_1, d_bias23);
        }

#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)  
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        d_acc1_0 = AE_ADD32S(d_acc1_0, d_outzb);

#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S) 
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_1 = AE_ADD32S(d_acc0_1, d_outzb);
        d_acc1_1 = AE_ADD32S(d_acc1_1, d_outzb);
        AE_MINMAX32_HIFI4(d_acc0_0, min_word16, max_word16);
        AE_MINMAX32_HIFI4(d_acc1_0, min_word16, max_word16);
        AE_MINMAX32_HIFI4(d_acc0_1, min_word16, max_word16);
        AE_MINMAX32_HIFI4(d_acc1_1, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        d_acc1_0 = AE_SLAI32(d_acc1_0, 1);
        d_acc0_1 = AE_SLAI32(d_acc0_1, 1);
        d_acc1_1 = AE_SLAI32(d_acc1_1, 1);
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
          out0 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr+2], p_out[(vec_itr+1)*rows+m_itr+3]);
          d_acc0_1 = AE_ADD32S(d_acc0_1, out0);
          d_acc1_1 = AE_ADD32S(d_acc1_1, out1);
        }

        ae_int16x4 ae_int16x4_out;
        ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
#if XCHAL_HAVE_HIFI1
        AE_SA16X4_IP(ae_int16x4_out, align_out0, p_out0_16);
#else
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr  ] = AE_SEL16_6543(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = ae_int16x4_out;
#endif

        ae_int16x4_out = AE_SAT16X4(d_acc0_1, d_acc1_1);
#if XCHAL_HAVE_HIFI1
        AE_SA16X4_IP(ae_int16x4_out, align_out1, p_out1_16);
#else
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr  ] = AE_SEL16_6543(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+1] = AE_SEL16_5432(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+2] = AE_SEL16_4321(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+3] = ae_int16x4_out;
#endif
      }

#if XCHAL_HAVE_HIFI1
      AE_SA64POS_FP(align_out0, p_out0_16);
      AE_SA64POS_FP(align_out1, p_out1_16);
#endif

#pragma no_unroll
      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc0_1 = AE_ZERO32();

        ae_int64 tmp_acc64_0, tmp_acc64_1, tmp_acc64_2, tmp_acc64_3;

        ae_int16x4 mat1_00, vec1_00, vec1_10;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];
        WORD8 *p_vec1_1 = (WORD8 *)&p_vec1[(vec_itr+1)*cols1];

        for(c_itr = 0; c_itr < (cols1>>2); c_itr++)
        {
          AE_L8X4S_IP(mat1_00, p_mat1_0, 4);
          AE_L8X4S_IP(vec1_00, p_vec1_0, 4);
          AE_L8X4S_IP(vec1_10, p_vec1_1, 4);

          AE_MULA8Q8X8_HIFI1(d_acc0_0, d_acc0_1, vec1_00, vec1_10, vec1_00, vec1_10, mat1_00);
        }

        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        AE_MINMAX32_HIFI4(d_acc0_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);

        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
      }
    }
    for (; vec_itr < vec_count ; vec_itr++)
    {
#if XCHAL_HAVE_HIFI1
      ae_int16x4 *p_out0_16 = (ae_int16x4 *)&p_out[vec_itr*rows];
#endif

      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();

        ae_int64 tmp_acc64_0, tmp_acc64_1, tmp_acc64_2, tmp_acc64_3;
        ae_int16x4 _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_mat1_1 = (WORD8 *)&p_mat1[(m_itr+1)*row_stride1];
        WORD8 *p_mat1_2 = (WORD8 *)&p_mat1[(m_itr+2)*row_stride1];
        WORD8 *p_mat1_3 = (WORD8 *)&p_mat1[(m_itr+3)*row_stride1];

        ae_int16x4 _ae8x8_vec1_00;
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];

        for(c_itr = 0; c_itr < (cols1>>2); c_itr++)
        {
          AE_L8X4S_IP(_ae8x8_mat1_00, p_mat1_0, 4);
          AE_L8X4S_IP(_ae8x8_mat1_10, p_mat1_1, 4);
          AE_L8X4S_IP(_ae8x8_mat1_20, p_mat1_2, 4);
          AE_L8X4S_IP(_ae8x8_mat1_30, p_mat1_3, 4);
          AE_L8X4S_IP(_ae8x8_vec1_00, p_vec1_0, 4);

          AE_MULA8Q8X8_HIFI1(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_00);
        }

        {
          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S) 
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        d_acc1_0 = AE_ADD32S(d_acc1_0, d_outzb);
        AE_MINMAX32_HIFI4(d_acc0_0, min_word16, max_word16);
        AE_MINMAX32_HIFI4(d_acc1_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        d_acc1_0 = AE_SLAI32(d_acc1_0, 1);
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
#if XCHAL_HAVE_HIFI1
        AE_SA16X4_IP(_ae_int16x4_out, align_out0, p_out0_16);
#else
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = (_ae_int16x4_out);
#endif
      }

#if XCHAL_HAVE_HIFI1
      AE_SA64POS_FP(align_out0, p_out0_16);
#endif

#pragma no_unroll
      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0;
        ae_int64 d64_acc0 = AE_ZERO64();

        ae_int16x4 _ae8x8_mat1_00, _ae8x8_vec1_00;

        WORD8 *p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_vec1_0 = (WORD8 *)&p_vec1[(vec_itr+0)*cols1];

        for(c_itr = 0; c_itr < (cols1>>2); c_itr++)
        {
          AE_L8X4S_IP(_ae8x8_mat1_00, p_mat1_0, 4);
          AE_L8X4S_IP(_ae8x8_vec1_00, p_vec1_0, 4);

          AE_MULAAAAQ16(d64_acc0, _ae8x8_mat1_00, _ae8x8_vec1_00);
        }
        ae_int32x2 tmp = AE_MOVINT32X2_FROMINT64(AE_SLAI64S(d64_acc0, 32));
        d_acc0_0 = AE_SEL32_HH(tmp, tmp);
        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        AE_MINMAX32_HIFI4(d_acc0_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32(p_out[vec_itr*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
      }
    }
  }
  else
  {
    for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
    {
#if XCHAL_HAVE_HIFI1
      ae_int16x4 *p_out0_16 = (ae_int16x4 *)&p_out[vec_itr*rows];
#endif
      for(m_itr = 0; m_itr < (rows & ~3); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();

        ae_int16x4 vec_00, mat_00, mat_10, mat_20, mat_30;

        ae_int64 acc64_0, acc64_1, acc64_2, acc64_3;
        acc64_0 = acc64_1 = acc64_2 = acc64_3 = 0;

        WORD8 *p_vec_0 = (WORD8 *) &p_vec1[vec_itr*cols1];

        WORD8 *p_mat_0 = (WORD8 *) &p_mat1[(m_itr+0)*row_stride1];
        WORD8 *p_mat_1 = (WORD8 *) &p_mat1[(m_itr+1)*row_stride1];
        WORD8 *p_mat_2 = (WORD8 *) &p_mat1[(m_itr+2)*row_stride1];
        WORD8 *p_mat_3 = (WORD8 *) &p_mat1[(m_itr+3)*row_stride1];

        /* Compatibility ALIGN_REGISTER_TYPE is int16x4 for (use by) AE_LA8X4F_IP implementation
         * on HiFi1/4, so can use 5 of them
         */
        ALIGN_REGISTER_TYPE align_p_vec_0,
            align_p_mat_0, align_p_mat_1, align_p_mat_2, align_p_mat_3;

        PRIME_8X4F(p_mat_0, align_p_mat_0);
        PRIME_8X4F(p_mat_1, align_p_mat_1);
        PRIME_8X4F(p_mat_2, align_p_mat_2);
        PRIME_8X4F(p_mat_3, align_p_mat_3);
        PRIME_8X4F(p_vec_0, align_p_vec_0);

        int cols_count=cols1&(~3);
        for(c_itr = 0; c_itr < cols_count>>2; c_itr++)
        {
#if XCHAL_HAVE_HIFI1
          AE_LA8X4S_IP(vec_00, align_p_vec_0, p_vec_0);
                  
          AE_LA8X4S_IP(mat_00, align_p_mat_0, p_mat_0);
          AE_LA8X4S_IP(mat_10, align_p_mat_1, p_mat_1);
          AE_LA8X4S_IP(mat_20, align_p_mat_2, p_mat_2);
          AE_LA8X4S_IP(mat_30, align_p_mat_3, p_mat_3);
#else
          AE_LA8X4F_IP(vec_00, align_p_vec_0, p_vec_0);

          AE_LA8X4F_IP(mat_00, align_p_mat_0, p_mat_0);
          AE_LA8X4F_IP(mat_10, align_p_mat_1, p_mat_1);
          AE_LA8X4F_IP(mat_20, align_p_mat_2, p_mat_2);
          AE_LA8X4F_IP(mat_30, align_p_mat_3, p_mat_3);
#endif

          AE_MULAAAAQ16(acc64_0, mat_00, vec_00);
          AE_MULAAAAQ16(acc64_1, mat_10, vec_00);
          AE_MULAAAAQ16(acc64_2, mat_20, vec_00);
          AE_MULAAAAQ16(acc64_3, mat_30, vec_00);
        }

#if !XCHAL_HAVE_HIFI1
        acc64_0 = AE_SRAI64(acc64_0, 16);
        acc64_1 = AE_SRAI64(acc64_1, 16);
        acc64_2 = AE_SRAI64(acc64_2, 16);
        acc64_3 = AE_SRAI64(acc64_3, 16);
#endif

        d_acc0_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_0), AE_MOVINT32X2_FROMINT64(acc64_1));
        d_acc1_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc64_2), AE_MOVINT32X2_FROMINT64(acc64_3));

        int rem_itr = 0;
        for(rem_itr = 0 ; rem_itr <(cols1 - cols_count); rem_itr++)
        {
          mat_00 = AE_MOVDA16((WORD16)*p_mat_0); p_mat_0++;
          mat_10 = AE_MOVDA16((WORD16)*p_mat_1); p_mat_1++;
          mat_20 = AE_MOVDA16((WORD16)*p_mat_2); p_mat_2++;
          mat_30 = AE_MOVDA16((WORD16)*p_mat_3); p_mat_3++;
          vec_00 = AE_MOVDA16((WORD16)*p_vec_0); p_vec_0++;

          /* mul_mtx = (ae_int16x4){mat1_00, mat1_10, mat1_20, mat1_30} */
          ae_int16x4 mul_mtx = AE_SEL16_7531(
            AE_SEL16_5432(mat_00, mat_10), AE_SEL16_5432(mat_20, mat_30) );

          AE_MULA16X4(d_acc0_0, d_acc1_0, mul_mtx, vec_00);
        }
        {
          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc1_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc1_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        d_acc1_0 = AE_ADD32S(d_acc1_0, d_outzb);
        AE_MINMAX32_HIFI4(d_acc0_0, min_word16, max_word16);
        AE_MINMAX32_HIFI4(d_acc1_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        d_acc1_0 = AE_SLAI32(d_acc1_0, 1);
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr  ], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
        }

        ae_int16x4 ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);

#if XCHAL_HAVE_HIFI1
        AE_SA16X4_IP(ae_int16x4_out, align_out0, p_out0_16);
#else
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr  ] = AE_SEL16_6543(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(ae_int16x4_out, ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = ae_int16x4_out;
#endif
      }

#if XCHAL_HAVE_HIFI1
      AE_SA64POS_FP(align_out0, p_out0_16);
#endif

      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0;
        ae_int64 d64_acc0 = AE_ZERO64();

        ae_int16x4 _ae8x8_mat1_00, _ae8x8_vec1_00;
        ALIGN_REGISTER_TYPE _align_ae8x16_p_mat1_0, _align_ae8x16_p_vec1_0;

        WORD8 *_ae8x16_p_mat1_0 = (WORD8 *)&p_mat1[(m_itr+0)*row_stride1];
        WORD8 *_ae8x16_p_vec1_0 = (WORD8 *)&p_vec1[vec_itr*cols1];

        PRIME_8X4F(_ae8x16_p_mat1_0, _align_ae8x16_p_mat1_0);
        PRIME_8X4F(_ae8x16_p_vec1_0, _align_ae8x16_p_vec1_0);

        int cols_count=cols1&(~3);
        for(c_itr = 0; c_itr < cols_count>>2; c_itr++)
        {

#if XCHAL_HAVE_HIFI1
          AE_LA8X4S_IP(_ae8x8_mat1_00, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0);
          AE_LA8X4S_IP(_ae8x8_vec1_00, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
#else
          AE_LA8X4F_IP(_ae8x8_mat1_00, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0);
          AE_LA8X4F_IP(_ae8x8_vec1_00, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);

          _ae8x8_mat1_00 = AE_SRAI16(_ae8x8_mat1_00, 8);
          _ae8x8_vec1_00 = AE_SRAI16(_ae8x8_vec1_00, 8);
#endif

          AE_MULAAAAQ16(d64_acc0, _ae8x8_mat1_00, _ae8x8_vec1_00);
        }

        int rem_itr = 0;
        ae_int64 acc_tmp = AE_ZERO64();
        for(rem_itr = 0 ; rem_itr <(cols1 - cols_count); rem_itr++)
        {
          _ae8x8_mat1_00 = AE_MOVDA16( (WORD16)*_ae8x16_p_mat1_0); _ae8x16_p_mat1_0++;
          _ae8x8_vec1_00 = AE_MOVDA16( (WORD16)*_ae8x16_p_vec1_0); _ae8x16_p_vec1_0++;

          AE_MULAAAAQ16(acc_tmp, _ae8x8_mat1_00, _ae8x8_vec1_00);
        }
        acc_tmp = AE_SRAI64(acc_tmp, 2);
        d64_acc0 = AE_ADD64(d64_acc0, acc_tmp);
        ae_int32x2 tmp = AE_MOVINT32X2_FROMINT64(AE_SLAI64S(d64_acc0, 32));
        d_acc0_0 = AE_SEL32_HH(tmp, tmp);
        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (TFLITE_SINGLE_ROUNDING && XCHAL_HAVE_HIFI1S)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        AE_MINMAX32_HIFI4(d_acc0_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32(p_out[vec_itr*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = _ae_int16x4_out;
      }
    }
  }

  return 0;
}
