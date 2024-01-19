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

/*----------------------------Main function---------------------------------*/

WORD32 xa_nn_matmul_asym8xasym8_asym8(
    UWORD8 * __restrict__ p_out,
    const UWORD8 * __restrict__ p_mat1,
    const UWORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride,                      
    WORD32 mat1_zero_bias,
    WORD32 vec1_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    /* Pointer alignment checks */
    //XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    //XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(WORD8), -1);
    //XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
    XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
    XA_NNLIB_ARG_CHK_COND((mat1_zero_bias < -255 || mat1_zero_bias > 0), -1);
    XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -255 || vec1_zero_bias > 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
    XA_NNLIB_ARG_CHK_COND((out_zero_bias < 0 || out_zero_bias > 255), -1);

    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    /* Shifts to match with Tensorflow */
    int left_shift, right_shift;

    #define UNROLL_ROW_SETUP_ACC_BATCH              SETUP_ACC_BATCH_ROW_FOR_ASYM8bxASYM8b
    #define UNROLL_SETUP_ACC_BATCH                  SETUP_ACC_BATCH_FOR_ASYM8bxASYM8b
    #define UNROLL_SETUP_MAT1                       SETUP_MAT1_ASYM8b
    #define UNROLL_SETUP_VEC_BATCH                  SETUP_VEC_OFFSET_BATCH_ASYM8b
    #define SETUP_BIAS                              SETUP_BIAS_ASYM8b
    #define UNROLL_LOAD_VEC_BATCH                   LOAD_VEC_BATCH_ASYM8b
    #define UNROLL_LOAD_ROW_MAT1                    LOAD_ROW_MAT1_ASYM8b
    #define LOAD_BIAS                               LOAD_BIAS_ASYM8b_MATMUL
    #define UNROLL_ROW_KERNEL_MAT1_VEC_BATCH        KERNEL_MAT1_VEC_BATCH_ROW_ASYM8b_ASYM8b
    #define UNROLL_KERNEL_MAT1_VEC_BATCH            KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b
    #define UNROLL_ROW_ADD_BIAS_ACC                 ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL
    #define UNROLL_ADD_BIAS_ACC_BATCH               ADD_BIAS_BATCH_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL
    #define UNROLL_ROW_ADJUST_ACC                   ADJUST_ACC_BATCH_ROW_ASYM8b
    #define UNROLL_ADJUST_ACC_BATCH                 ADJUST_ACC_BATCH_ASYM8b
    #define UNROLL_ROW_STORE_ACC                    STORE_ACC_BATCH_ROW_ASYM8bxASYM8b_AT_OUT_ASYM8b
    #define UNROLL_STORE_ACC_BATCH                  STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b

#if TFLITE_SINGLE_ROUNDING
    left_shift = out_shift;
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    left_shift = out_shift<0?0:out_shift;
    right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

    int chk_align = 0;  
    CHK_MATMUL_ALIGN(chk_align, p_mat1, (ALIGNMENT>>1), p_vec1, (ALIGNMENT>>1), cols1, row_stride1, vec_offset, 4);
    if(chk_align)
    {
        for(vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr+=VEC_UNROLL)
        {
            SETUP_BIAS;
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
            {
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
                ADJUST_ACC_BATCH_TAIL;
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
                UNROLL_ADJUST_ACC_BATCH(0,0);
                UNROLL_STORE_ACC_BATCH(0,0);
              }
        }
      
    /* Undefining the defined macro to make them available for reuse */
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
    #undef UNROLL_ROW_ADJUST_ACC
    #undef UNROLL_ADJUST_ACC_BATCH
    #undef UNROLL_ROW_STORE_ACC
    #undef UNROLL_STORE_ACC_BATCH
    #undef VEC_UNROLL
    #undef ROW_UNROLL
    }
    else if (p_mat1 && p_vec1)
    {
        #define ROW_UNROLL 2
        #define VEC_UNROLL 2
        #define UNROLL_SETUP_ACC_BATCH              SETUP_ACC_BATCH_FOR_ASYM8bxASYM8b
        #define SETUP_BIAS                          SETUP_BIAS_ASYM8b
        #define LOAD_BIAS                           LOAD_BIAS_ASYM8b_MATMUL
        #define UNROLL_ADD_BIAS_ACC_BATCH           ADD_BIAS_BATCH_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL
        #define UNROLL_ADJUST_ACC_BATCH             ADJUST_ACC_BATCH_ASYM8b
        for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
        {
            SETUP_BIAS;
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
            {
                UNROLL_SETUP_ACC_BATCH(0,0);
                UNROLL_SETUP_ACC_BATCH(0,1);
                UNROLL_SETUP_ACC_BATCH(1,0);
                UNROLL_SETUP_ACC_BATCH(1,1);
                SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(0);
                SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(1);
                SETUP_MAT1_ASYM8b_UNALIGNED(0);
                SETUP_MAT1_ASYM8b_UNALIGNED(1);

                int cols1_count = cols1- cols1%4;
                for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
                {
                    LOAD_VEC_BATCH_ASYM8b_UNALIGNED(0);
                    LOAD_VEC_BATCH_ASYM8b_UNALIGNED(1);
                    LOAD_ROW_MAT1_ASYM8b_UNALIGNED(0);
                    LOAD_ROW_MAT1_ASYM8b_UNALIGNED(1);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(1,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,1);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(1,1);
                }
                #pragma no_unroll
                for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                {
                    LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(0);
                    LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(1);
                    LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(0);
                    LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(1);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(1,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,1);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(1,1);
                }

                ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL(0);
                ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL(1);
                ADJUST_ACC_BATCH_ROW_ASYM8b(0);
                ADJUST_ACC_BATCH_ROW_ASYM8b(1);
                STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0,0);
                STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(1,0);
                STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0,1);
                STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(1,1);
            }
            //Remaining row
            for(; m_itr < rows; m_itr++)
            {
                UNROLL_SETUP_ACC_BATCH(0,0);
                UNROLL_SETUP_ACC_BATCH(0,1);
                SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(0);
                SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(1);
                SETUP_MAT1_ASYM8b_UNALIGNED(0);
                int cols1_count = cols1- cols1%4;

                for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
                {
                    LOAD_VEC_BATCH_ASYM8b_UNALIGNED(0);
                    LOAD_VEC_BATCH_ASYM8b_UNALIGNED(1);
                    LOAD_ROW_MAT1_ASYM8b_UNALIGNED(0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,1);
                }
                #pragma no_unroll
                for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                {
                    LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(0);
                    LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(1);
                    LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,1);
                }
                ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL(0);
                ADJUST_ACC_BATCH_ROW_ASYM8b(0);
                STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0,0);
                STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0,1);
            }

        }
        {
            /* Tail loop for vec unroll */
            for(; vec_itr < vec_count; vec_itr++)
            {
                SETUP_BIAS;
                for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
                {
                    UNROLL_SETUP_ACC_BATCH(0,0);
                    UNROLL_SETUP_ACC_BATCH(1,0);
                    SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(0);
                    SETUP_MAT1_ASYM8b_UNALIGNED(0);
                    SETUP_MAT1_ASYM8b_UNALIGNED(1);
                    int cols1_count = cols1 - cols1%4;

                    for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
                    {
                        LOAD_VEC_BATCH_ASYM8b_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8b_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8b_UNALIGNED(1);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,0);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(1,0);
                    }
                #pragma no_unroll
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(1);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,0);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(1,0);
                    }  

                    LOAD_BIAS; 
                    UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                    UNROLL_ADJUST_ACC_BATCH(0,0);
                    LOAD_BIAS; 
                    UNROLL_ADD_BIAS_ACC_BATCH(1,0);
                    UNROLL_ADJUST_ACC_BATCH(1,0);
                
                    STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0,0);
                    STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(1,0);
                }

                for(; m_itr < rows; m_itr++)
                {
                    UNROLL_SETUP_ACC_BATCH(0,0);
                    SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(0);
                    SETUP_MAT1_ASYM8b_UNALIGNED(0);
                    int cols1_count = cols1 - cols1%4;

                    for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
                    {
                        LOAD_VEC_BATCH_ASYM8b_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8b_UNALIGNED(0);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,0);
                    }
                #pragma no_unroll
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(0);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,0);
                    }

                    LOAD_BIAS;
                    UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                    UNROLL_ADJUST_ACC_BATCH(0,0);
                    STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0,0);
                }
            }
        }
    }
    else
    {
        return -1;
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
    #undef UNROLL_ROW_ADJUST_ACC
    #undef UNROLL_ADJUST_ACC_BATCH
    #undef UNROLL_ROW_STORE_ACC
    #undef UNROLL_STORE_ACC_BATCH
    #undef VEC_UNROLL
    #undef ROW_UNROLL

    return 0;
}
