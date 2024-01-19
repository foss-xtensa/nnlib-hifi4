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

WORD32 xa_nn_matXvec_batch_8x16_64(

         WORD64 ** __restrict__ p_out,          /* array of output pointers */
         WORD8 *  __restrict__ p_mat1,         /* matrix1: rows x cols1 */
         WORD16 ** __restrict__ p_vec1,         /* vec1: cols1 x 1 */
         WORD16 *  __restrict__ p_bias,         /* bias TBD: Need array? */
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
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD64 *), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_mat1, (ALIGNMENT>>1), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD16 *), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
    for(i = 0; i < vec_count; i++)
    {
      XA_NNLIB_ARG_CHK_ALIGN(p_out[i], sizeof(WORD64), -1);
      XA_NNLIB_ARG_CHK_ALIGN(p_vec1[i], ALIGNMENT, -1);
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
    #define UNROLL_ROW_SETUP_ACC_BATCH          SETUP_ACC_BATCH_ROW_FOR_8bx16b
    #define UNROLL_SETUP_ACC_BATCH              SETUP_ACC_BATCH_FOR_8bx16b
    #define UNROLL_SETUP_MAT1                   SETUP_MAT1_8b
    #define UNROLL_SETUP_VEC_BATCH              SETUP_VEC_BATCH_16b
    #define SETUP_BIAS                          SETUP_BIAS_16b
    #define UNROLL_LOAD_VEC_BATCH               LOAD_VEC_BATCH_16b
    #define UNROLL_LOAD_ROW_MAT1                LOAD_ROW_MAT1_8b
    #define LOAD_BIAS                           LOAD_BIAS_16b_FOR_8bx16b
    #define UNROLL_ROW_KERNEL_MAT1_VEC_BATCH    KERNEL_MAT1_VEC_BATCH_ROW_8b_16b
    #define UNROLL_KERNEL_MAT1_VEC_BATCH        KERNEL_MAT1_VEC_BATCH_8b_16b
    #define UNROLL_ROW_ADD_BIAS_ACC             ADD_BIAS_BATCH_ROW_16b_ACC_FOR_8bx16b
    #define UNROLL_ADD_BIAS_ACC_BATCH           ADD_BIAS_BATCH_16b_ACC_FOR_8bx16b
    #define UNROLL_ROW_STORE_ACC                STORE_ACC_BATCH_ROW_8bx16b_AT_OUT_64b
    #define UNROLL_STORE_ACC_BATCH              STORE_ACC_BATCH_8bx16b_AT_OUT_64b

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
                    SETUP_VEC_BATCH;
                    SETUP_MAT1;

                    for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                    {
                        LOAD_VEC_BATCH;
                        LOAD_MAT1;
                        KERNEL_MAT1_VEC_BATCH;
                    }

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

    return 0;
}

