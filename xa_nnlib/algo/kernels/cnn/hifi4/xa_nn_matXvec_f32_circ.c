/*******************************************************************************
* Copyright (c) 2018-2025 Cadence Design Systems, Inc.
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
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_matXvec_f32_circ,(
    FLOAT32 *__restrict__ p_out,
    FLOAT32 * __restrict__ p_mat,
    FLOAT32 * __restrict__ p_vec,
    FLOAT32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_offset,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 bias_row_offset,
    WORD32 out_col_offset,
    WORD32 out_row_offset))
#else /* #if !HAVE_VFPU */
#ifdef ROW_UNROLL
#undef ROW_UNROLL
#endif
#define ROW_UNROLL 4

#define INCREMENT_IN_BYTES_FOR_FLOAT32x2    2*sizeof(FLOAT32)

#define SETUP_ACC_BATCH_ROW_FOR_f32(idx_row)\
    SETUP_ACC_BATCH_VEC_UNROLL(idx_row);

#define SETUP_ACC_BATCH_FOR_f32(idx_row,idx_vec) \
    xtfloatx2 _xtfloatx2_acc_ ##idx_row ##_ ##idx_vec = (xtfloatx2)0.0f; \
    xtfloat _xtfloat_acc_ ##idx_row ##_ ##idx_vec = (xtfloat) 0.0f;

#define SETUP_MAT_f32(idx) \
    xtfloatx2 _xtfloatx2_mat_ ## idx = (xtfloatx2)0.0f; \
    xtfloatx2 *_xtfloatx2_p_mat_ ## idx = (xtfloatx2 *) p_mat; \
    AE_ADDCIRC16X4_XC((ae_int16x4 *)_xtfloatx2_p_mat_ ##idx, (m_itr+idx)*row_offset*sizeof(FLOAT32));

#define SETUP_VEC_OFFSET_BATCH_f32(idx_vec)\
    xtfloatx2 _xtfloatx2_vec_batch_ ##idx_vec  = (xtfloatx2)0.0f ; \
    xtfloatx2 *_xtfloatx2_p_vec_batch_ ##idx_vec  = (xtfloatx2 *)(&p_vec[(vec_itr + idx_vec)*vec_offset]); \
    ae_valign _xtfloatx2_vec_batch_aligned_ ##idx_vec = AE_LA64_PP(_xtfloatx2_p_vec_batch_ ##idx_vec);

#define SETUP_BIAS_BATCH_ROW_FOR_f32(idx_row)\
    SETUP_BIAS_BATCH_VEC_UNROLL(idx_row);

#define SETUP_BIAS_BATCH_FOR_f32(idx_row,idx_vec) \
    xtfloat _xtfloat_bias_ ##idx_row ##_ ##idx_vec = 0.0f ; \
    if(p_bias != NULL){ \
    _xtfloat_bias_ ##idx_row ##_ ##idx_vec = p_bias[(vec_itr + idx_vec)]; \
    } \

#define LOAD_VEC_BATCH_f32(idx_vec) \
    XT_LASX2IP(_xtfloatx2_vec_batch_ ##idx_vec, _xtfloatx2_vec_batch_aligned_ ##idx_vec, _xtfloatx2_p_vec_batch_ ##idx_vec);

#define LOAD_ROW_MAT_f32(idx) \
    XT_LSX2XC(_xtfloatx2_mat_ ## idx, _xtfloatx2_p_mat_ ## idx, INCREMENT_IN_BYTES_FOR_FLOAT32x2);

#define KERNEL_MAT_VEC_BATCH_ROW_f32(idx_row)\
    KERNEL_MAT_VEC_BATCH_VEC_UNROLL(idx_row);\

#define KERNEL_MAT_VEC_BATCH_f32(idx_row,idx_vec) \
    XT_MADD_SX2(_xtfloatx2_acc_ ##idx_row ##_ ##idx_vec, _xtfloatx2_vec_batch_ ##idx_vec, _xtfloatx2_mat_ ##idx_row);

#define ADD_BIAS_BATCH_ROW_ACC_FOR_f32(idx_row)\
    ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);

#define ADD_BIAS_BATCH_ACC_FOR_f32(idx_row,idx_vec)\
    _xtfloat_acc_ ##idx_row ##_ ##idx_vec = XT_RADD_SX2(_xtfloatx2_acc_ ##idx_row ##_ ##idx_vec);\
    _xtfloat_acc_ ##idx_row ##_ ##idx_vec = XT_ADD_S(_xtfloat_acc_ ##idx_row ##_ ##idx_vec, _xtfloat_bias_ ##idx_row ##_ ##idx_vec);

#define STORE_ACC_BATCH_ROW_AT_OUT_f32(idx_row)\
    STORE_ACC_BATCH_VEC_UNROLL(idx_row);

#define STORE_ACC_BATCH_AT_OUT_OFFSET_f32(idx_row,idx_vec) \
    /*p_out value stored in a tmp pointer to make it inout for ISA */\
    p_out_tmp = &(p_out[(vec_itr + idx_vec)*out_col_offset + (m_itr + idx_row)*out_row_offset]);\
    XT_SSIP(_xtfloat_acc_ ##idx_row ##_ ##idx_vec,p_out_tmp,0);

#define VEC_UNROLL 2
#define UNROLL_ROW_SETUP_ACC_BATCH          SETUP_ACC_BATCH_ROW_FOR_f32
#define UNROLL_SETUP_ACC_BATCH              SETUP_ACC_BATCH_FOR_f32
#define UNROLL_SETUP_MAT                    SETUP_MAT_f32
#define UNROLL_SETUP_VEC_BATCH              SETUP_VEC_OFFSET_BATCH_f32
#define UNROLL_ROW_SETUP_BIAS_BATCH         SETUP_BIAS_BATCH_ROW_FOR_f32
#define UNROLL_SETUP_BIAS_BATCH             SETUP_BIAS_BATCH_FOR_f32
#define UNROLL_LOAD_VEC_BATCH               LOAD_VEC_BATCH_f32
#define UNROLL_LOAD_ROW_MAT                 LOAD_ROW_MAT_f32
#define UNROLL_ROW_KERNEL_MAT_VEC_BATCH     KERNEL_MAT_VEC_BATCH_ROW_f32
#define UNROLL_KERNEL_MAT_VEC_BATCH         KERNEL_MAT_VEC_BATCH_f32
#define UNROLL_ROW_ADD_BIAS_ACC             ADD_BIAS_BATCH_ROW_ACC_FOR_f32
#define UNROLL_ADD_BIAS_ACC_BATCH           ADD_BIAS_BATCH_ACC_FOR_f32
#define UNROLL_ROW_STORE_ACC                STORE_ACC_BATCH_ROW_AT_OUT_f32
#define UNROLL_STORE_ACC_BATCH              STORE_ACC_BATCH_AT_OUT_OFFSET_f32

/* ==================================================================================================== */
#undef SETUP_MAT
#if (ROW_UNROLL == 1)
#define SETUP_MAT           UNROLL_SETUP_MAT(0)
#elif (ROW_UNROLL == 2)
#define SETUP_MAT           UNROLL_SETUP_MAT(0)           UNROLL_SETUP_MAT(1)
#elif (ROW_UNROLL == 4)
#define SETUP_MAT           UNROLL_SETUP_MAT(0)           UNROLL_SETUP_MAT(1)           UNROLL_SETUP_MAT(2)           UNROLL_SETUP_MAT(3)
#elif (ROW_UNROLL == 8)
#define SETUP_MAT           UNROLL_SETUP_MAT(0)           UNROLL_SETUP_MAT(1)           UNROLL_SETUP_MAT(2)           UNROLL_SETUP_MAT(3)           UNROLL_SETUP_MAT(4)           UNROLL_SETUP_MAT(5)           UNROLL_SETUP_MAT(6)           UNROLL_SETUP_MAT(7)
#endif /* (ROW_UNROLL == 1) */

#if (ROW_UNROLL == 4 && VEC_UNROLL == 2)

#define SETUP_VEC_BATCH     UNROLL_SETUP_VEC_BATCH(0)   UNROLL_SETUP_VEC_BATCH(1)

#define SETUP_BIAS_BATCH     UNROLL_ROW_SETUP_BIAS_BATCH(0)     UNROLL_ROW_SETUP_BIAS_BATCH(1)     UNROLL_ROW_SETUP_BIAS_BATCH(2)     UNROLL_ROW_SETUP_BIAS_BATCH(3)
#define SETUP_BIAS_BATCH_VEC_UNROLL(idx_row)     UNROLL_SETUP_BIAS_BATCH(idx_row,0)   UNROLL_SETUP_BIAS_BATCH(idx_row,1)
#define SETUP_BIAS_BATCH_TAIL    UNROLL_SETUP_BIAS_BATCH(0,0)     UNROLL_SETUP_BIAS_BATCH(1,0)     UNROLL_SETUP_BIAS_BATCH(2,0)     UNROLL_SETUP_BIAS_BATCH(3,0)

#define SETUP_ACC_BATCH     UNROLL_ROW_SETUP_ACC_BATCH(0)     UNROLL_ROW_SETUP_ACC_BATCH(1)     UNROLL_ROW_SETUP_ACC_BATCH(2)     UNROLL_ROW_SETUP_ACC_BATCH(3)
#define SETUP_ACC_BATCH_VEC_UNROLL(idx_row)     UNROLL_SETUP_ACC_BATCH(idx_row,0)   UNROLL_SETUP_ACC_BATCH(idx_row,1)
#define SETUP_ACC_BATCH_TAIL    UNROLL_SETUP_ACC_BATCH(0,0)     UNROLL_SETUP_ACC_BATCH(1,0)     UNROLL_SETUP_ACC_BATCH(2,0)     UNROLL_SETUP_ACC_BATCH(3,0)

#define LOAD_VEC_BATCH      UNROLL_LOAD_VEC_BATCH(0)    UNROLL_LOAD_VEC_BATCH(1)
#define LOAD_MAT            UNROLL_LOAD_ROW_MAT(0)     UNROLL_LOAD_ROW_MAT(1)     UNROLL_LOAD_ROW_MAT(2)     UNROLL_LOAD_ROW_MAT(3)

#define KERNEL_MAT_VEC_BATCH       UNROLL_ROW_KERNEL_MAT_VEC_BATCH(0)     UNROLL_ROW_KERNEL_MAT_VEC_BATCH(1)     UNROLL_ROW_KERNEL_MAT_VEC_BATCH(2)     UNROLL_ROW_KERNEL_MAT_VEC_BATCH(3)
#define KERNEL_MAT_VEC_BATCH_VEC_UNROLL(idx_row)   UNROLL_KERNEL_MAT_VEC_BATCH(idx_row,0)     UNROLL_KERNEL_MAT_VEC_BATCH(idx_row,1)
#define KERNEL_MAT_VEC_BATCH_TAIL  UNROLL_KERNEL_MAT_VEC_BATCH(0,0)   UNROLL_KERNEL_MAT_VEC_BATCH(1,0)   UNROLL_KERNEL_MAT_VEC_BATCH(2,0)   UNROLL_KERNEL_MAT_VEC_BATCH(3,0)

#define ADD_BIAS_ACC_BATCH      UNROLL_ROW_ADD_BIAS_ACC(0)      UNROLL_ROW_ADD_BIAS_ACC(1)      UNROLL_ROW_ADD_BIAS_ACC(2)      UNROLL_ROW_ADD_BIAS_ACC(3)
#define ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row)      UNROLL_ADD_BIAS_ACC_BATCH(idx_row,0)     UNROLL_ADD_BIAS_ACC_BATCH(idx_row,1)
#define ADD_BIAS_ACC_BATCH_TAIL   UNROLL_ADD_BIAS_ACC_BATCH(0,0)     UNROLL_ADD_BIAS_ACC_BATCH(1,0)     UNROLL_ADD_BIAS_ACC_BATCH(2,0)     UNROLL_ADD_BIAS_ACC_BATCH(3,0)

#define STORE_ACC_BATCH     UNROLL_ROW_STORE_ACC(0)     UNROLL_ROW_STORE_ACC(1)     UNROLL_ROW_STORE_ACC(2)     UNROLL_ROW_STORE_ACC(3)
#define STORE_ACC_BATCH_VEC_UNROLL(idx_row)     UNROLL_STORE_ACC_BATCH(idx_row,0)     UNROLL_STORE_ACC_BATCH(idx_row,1)
#define STORE_ACC_BATCH_TAIL    UNROLL_STORE_ACC_BATCH(0,0)     UNROLL_STORE_ACC_BATCH(1,0)     UNROLL_STORE_ACC_BATCH(2,0)     UNROLL_STORE_ACC_BATCH(3,0)

#endif /* (ROW_UNROLL == 4 && VEC_UNROLL == 2)*/

#if NO_AGGR_FLOAT_OPT

#define INCREMENT_IN_BYTES_FOR_FLOAT32 sizeof(FLOAT32)

WORD32 xa_nn_matXvec_f32_circ(
    FLOAT32 *__restrict__ p_out,            /* output pointer */
    FLOAT32 *__restrict__ p_mat,            /* matrix: rows x cols */
    const FLOAT32 *__restrict__ p_vec,            /* vec: cols x 1 */
    const FLOAT32 *__restrict__ p_bias,           /* bias TBD: Need array? */
    WORD32 rows,                            /* Number of rows in matrix */
    WORD32 cols,                            /* Number of columns in matrix */
    WORD32 row_offset,                      /* row stride for matrix */
    WORD32 vec_count,                       /* number of vectors: 2, 4, 2n */
    WORD32 vec_offset,                      /* offset from current to next vector */
    WORD32 out_col_offset,
    WORD32 out_row_offset)
{
    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
    xtfloat* p_out_tmp;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    vec_itr = 0;

    {
        for (vec_itr = 0; vec_itr < (vec_count & (~0x3)); vec_itr+=4)
        {
            m_itr = 0;
            for(; m_itr < (rows & (~0x1)); m_itr+=2)
            {
                xtfloat _xtfloat_bias_0 = ZERO_S();
                xtfloat _xtfloat_bias_1 = ZERO_S();
                xtfloat _xtfloat_bias_2 = ZERO_S();
                xtfloat _xtfloat_bias_3 = ZERO_S();
                xtfloatx2 _xtfloatx2_bias_01 = ZERO_S(), _xtfloatx2_bias_23 = ZERO_S();
                xtfloatx2 acc_row0_vec01 = ZERO_SX2(), acc_row0_vec23 = ZERO_SX2();
                xtfloatx2 acc_row1_vec01 = ZERO_SX2(), acc_row1_vec23 = ZERO_SX2();
                xtfloatx2 x00, x11;
                xtfloatx2 vec01;
                xtfloatx2 vec23;
                if(p_bias != NULL){
                    _xtfloat_bias_0 = p_bias[vec_itr];
                    _xtfloat_bias_1 = p_bias[vec_itr+1];
                    _xtfloat_bias_2 = p_bias[vec_itr+2];
                    _xtfloat_bias_3 = p_bias[vec_itr+3];
                    _xtfloatx2_bias_01 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_0), AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_1));
                    _xtfloatx2_bias_23 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_2), AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_3));
                }
 
                ae_int64 *p64_x0 = (ae_int64 *) p_mat;
                ae_int64 *p64_x1 = (ae_int64 *) p_mat;
                xtfloat *p_vec0  = (xtfloat *)(&p_vec[(vec_itr)*vec_offset]);
                xtfloat *p_vec1  = (xtfloat *)(&p_vec[(vec_itr+1)*vec_offset]);
                xtfloat *p_vec2  = (xtfloat *)(&p_vec[(vec_itr+2)*vec_offset]);
                xtfloat *p_vec3  = (xtfloat *)(&p_vec[(vec_itr+3)*vec_offset]);
 
                xtfloat _xtfloat_mat_0 = ZERO_S();
                xtfloat _xtfloat_mat_1 = ZERO_S();
                AE_ADDCIRC_XC(p64_x0, (m_itr)*row_offset*sizeof(FLOAT32));
                AE_ADDCIRC_XC(p64_x1, (m_itr+1)*row_offset*sizeof(FLOAT32));
                xtfloat *px0 = (xtfloat*)p64_x0;
                xtfloat *px1 = (xtfloat*)p64_x1;
                int k;

                for(k = 0; k < cols; k++, p_vec0++, p_vec1++, p_vec2++, p_vec3++)
                {
                    vec01 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec0)), AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec1)));
                    vec23 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec2)), AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec3)));
                    
                    AE_LSXC(_xtfloat_mat_0, px0, sizeof(xtfloat));
                    AE_LSXC(_xtfloat_mat_1, px1, sizeof(xtfloat));
                    x00 = AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_mat_0);
                    x11 = AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_mat_1);
                    
                    acc_row0_vec01 = ADD_SX2(acc_row0_vec01, MUL_SX2(x00, vec01));
                    acc_row0_vec23 = ADD_SX2(acc_row0_vec23, MUL_SX2(x00, vec23));
                    acc_row1_vec01 = ADD_SX2(acc_row1_vec01, MUL_SX2(x11, vec01));
                    acc_row1_vec23 = ADD_SX2(acc_row1_vec23, MUL_SX2(x11, vec23));
                }
                acc_row0_vec01 = ADD_SX2(acc_row0_vec01, _xtfloatx2_bias_01);
                acc_row0_vec23 = ADD_SX2(acc_row0_vec23, _xtfloatx2_bias_23);
                acc_row1_vec01 = ADD_SX2(acc_row1_vec01, _xtfloatx2_bias_01);
                acc_row1_vec23 = ADD_SX2(acc_row1_vec23, _xtfloatx2_bias_23);
 
                p_out_tmp = (xtfloat*)(&(p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(HIGH_S(acc_row0_vec01),p_out_tmp,0);
                p_out_tmp = (xtfloat*)(&(p_out[(vec_itr+1)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(LOW_S(acc_row0_vec01),p_out_tmp,0);
                p_out_tmp = (xtfloat*)(&(p_out[(vec_itr+2)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(HIGH_S(acc_row0_vec23),p_out_tmp,0);
                p_out_tmp = (xtfloat*)(&(p_out[(vec_itr+3)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(LOW_S(acc_row0_vec23),p_out_tmp,0);
                p_out_tmp = (xtfloat*)(&(p_out[(vec_itr)*out_col_offset + (m_itr+1)*out_row_offset]));
                AE_SSIP(HIGH_S(acc_row1_vec01),p_out_tmp,0);
                p_out_tmp = (xtfloat*)(&(p_out[(vec_itr+1)*out_col_offset + (m_itr+1)*out_row_offset]));
                AE_SSIP(LOW_S(acc_row1_vec01),p_out_tmp,0);
                p_out_tmp = (xtfloat*)(&(p_out[(vec_itr+2)*out_col_offset + (m_itr+1)*out_row_offset]));
                AE_SSIP(HIGH_S(acc_row1_vec23),p_out_tmp,0);
                p_out_tmp = (xtfloat*)(&(p_out[(vec_itr+3)*out_col_offset + (m_itr+1)*out_row_offset]));
                AE_SSIP(LOW_S(acc_row1_vec23),p_out_tmp,0);
            }
            for (; m_itr < rows ; m_itr++)
            {
                xtfloat _xtfloat_bias_0 = ZERO_S();
                xtfloat _xtfloat_bias_1 = ZERO_S();
                xtfloat _xtfloat_bias_2 = ZERO_S();
                xtfloat _xtfloat_bias_3 = ZERO_S();
                xtfloatx2 _xtfloatx2_bias_01 = ZERO_S(), _xtfloatx2_bias_23 = ZERO_S();
                xtfloatx2 acc_row0_vec01 = ZERO_SX2(), acc_row0_vec23 = ZERO_SX2();
                xtfloatx2 x00;
                xtfloatx2 vec01;
                xtfloatx2 vec23;
                if(p_bias != NULL){
                    _xtfloat_bias_0 = p_bias[vec_itr];
                    _xtfloat_bias_1 = p_bias[vec_itr+1];
                    _xtfloat_bias_2 = p_bias[vec_itr+2];
                    _xtfloat_bias_3 = p_bias[vec_itr+3];
                    _xtfloatx2_bias_01 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_0), AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_1));
                    _xtfloatx2_bias_23 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_2), AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_3));
                }
 
                ae_int64 *p64_x0 = (ae_int64 *) p_mat;
                xtfloat *p_vec0  = (xtfloat *)(&p_vec[(vec_itr)*vec_offset]);
                xtfloat *p_vec1  = (xtfloat *)(&p_vec[(vec_itr+1)*vec_offset]);
                xtfloat *p_vec2  = (xtfloat *)(&p_vec[(vec_itr+2)*vec_offset]);
                xtfloat *p_vec3  = (xtfloat *)(&p_vec[(vec_itr+3)*vec_offset]);
 
                xtfloat _xtfloat_mat_0 = ZERO_S();
                AE_ADDCIRC_XC(p64_x0, (m_itr)*row_offset*sizeof(FLOAT32));
                xtfloat *px0 = (xtfloat*)p64_x0;
                int k;
                for(k = 0; k < cols; k++, p_vec0++, p_vec1++, p_vec2++, p_vec3++)
                {
                    vec01 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec0)), AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec1)));
                    vec23 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec2)), AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec3)));
                    
                    AE_LSXC(_xtfloat_mat_0, px0, sizeof(xtfloat));
                    x00 = AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_mat_0);
                    
                    acc_row0_vec01 = ADD_SX2(acc_row0_vec01, MUL_SX2(x00, vec01));
                    acc_row0_vec23 = ADD_SX2(acc_row0_vec23, MUL_SX2(x00, vec23));
                }
                acc_row0_vec01 = ADD_SX2(acc_row0_vec01, _xtfloatx2_bias_01);
                acc_row0_vec23 = ADD_SX2(acc_row0_vec23, _xtfloatx2_bias_23);
 
                p_out_tmp = (xtfloat *)(&(p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(HIGH_S(acc_row0_vec01),p_out_tmp,0);
                p_out_tmp = (xtfloat *)(&(p_out[(vec_itr+1)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(LOW_S(acc_row0_vec01),p_out_tmp,0);
                p_out_tmp = (xtfloat *)(&(p_out[(vec_itr+2)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(HIGH_S(acc_row0_vec23),p_out_tmp,0);
                p_out_tmp = (xtfloat *)(&(p_out[(vec_itr+3)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(LOW_S(acc_row0_vec23),p_out_tmp,0);
            }
        }
        for (; vec_itr < (vec_count & (~0x1)); vec_itr+=2)
        {
            for(m_itr = 0; m_itr < (rows & (~0x1)); m_itr+=2)
            {
                xtfloat _xtfloat_bias_0 = ZERO_S();
                xtfloat _xtfloat_bias_1 = ZERO_S();
                xtfloatx2 acc_row0_vec01 = ZERO_SX2();
                xtfloatx2 acc_row1_vec01 = ZERO_SX2();
                xtfloatx2 x00, x01;
                xtfloatx2 vec0_0;
                xtfloatx2 vec1_0;
                xtfloatx2 vec01;
                if(p_bias != NULL){
                _xtfloat_bias_0 = p_bias[vec_itr];
                _xtfloat_bias_1 = p_bias[vec_itr+1];
                }
 
                ae_int64 *p64_x0 = (ae_int64 *) p_mat;
                ae_int64 *p64_x1 = (ae_int64 *) p_mat;
                xtfloat *p_vec0  = (xtfloat *)(&p_vec[(vec_itr)*vec_offset]);
                xtfloat *p_vec1  = (xtfloat *)(&p_vec[(vec_itr+1)*vec_offset]);
 
                xtfloat _xtfloat_mat_0 = ZERO_S();
                xtfloat _xtfloat_mat_1 = ZERO_S();
                AE_ADDCIRC_XC(p64_x0, (m_itr)*row_offset*sizeof(FLOAT32));
                AE_ADDCIRC_XC(p64_x1, (m_itr+1)*row_offset*sizeof(FLOAT32));
                xtfloat *px0 = (xtfloat *)p64_x0;
                xtfloat *px1 = (xtfloat *)p64_x1;
                int k;

                for(k = 0; k < cols; k++, p_vec0++, p_vec1++)
                {
                    AE_LSXC(_xtfloat_mat_0, px0, sizeof(xtfloat));
                    AE_LSXC(_xtfloat_mat_1, px1, sizeof(xtfloat));
                    x00 = AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_mat_0);
                    x01 = AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_mat_1);
                    vec0_0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec0));
                    vec1_0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec1));
                    vec01 = XT_SEL32_HH_SX2(vec0_0, vec1_0);
                    acc_row0_vec01 = ADD_SX2(acc_row0_vec01, MUL_SX2(x00, vec01));
                    acc_row1_vec01 = ADD_SX2(acc_row1_vec01, MUL_SX2(x01, vec01));
                }
                acc_row0_vec01 = ADD_SX2(acc_row0_vec01, XT_SEL32_HH_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_0), AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_1)));
                acc_row1_vec01 = ADD_SX2(acc_row1_vec01, XT_SEL32_HH_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_0), AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_bias_1)));
 
                p_out_tmp = (xtfloat *)(&(p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(HIGH_S(acc_row0_vec01),p_out_tmp,0);
                p_out_tmp = (xtfloat *)(&(p_out[(vec_itr)*out_col_offset + (m_itr+1)*out_row_offset]));
                AE_SSIP(HIGH_S(acc_row1_vec01),p_out_tmp,0);
                p_out_tmp = (xtfloat *)(&(p_out[(vec_itr+1)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(LOW_S(acc_row0_vec01),p_out_tmp,0);
                p_out_tmp = (xtfloat *)(&(p_out[(vec_itr+1)*out_col_offset + (m_itr+1)*out_row_offset]));
                AE_SSIP(LOW_S(acc_row1_vec01),p_out_tmp,0);
            }
            for (; m_itr < rows ; m_itr++)
            {
                xtfloat _xtfloat_bias_0 = ZERO_S();
                xtfloat _xtfloat_bias_1 = ZERO_S();
                xtfloatx2 acc_row0_vec0 = ZERO_SX2();
                xtfloatx2 x00;
                xtfloatx2 vec0_0;
                xtfloatx2 vec1_0;
                xtfloatx2 vec01;
                if(p_bias != NULL){
                _xtfloat_bias_0 = p_bias[vec_itr];
                _xtfloat_bias_1 = p_bias[vec_itr+1];
                }
 
                ae_int64 *p64_x0 = (ae_int64 *) p_mat;
                xtfloat *p_vec0  = (xtfloat *)(&p_vec[(vec_itr)*vec_offset]);
                xtfloat *p_vec1  = (xtfloat *)(&p_vec[(vec_itr+1)*vec_offset]);
 
                xtfloat _xtfloat_mat_0 = ZERO_S();
                AE_ADDCIRC_XC(p64_x0, (m_itr)*row_offset*sizeof(FLOAT32));
                xtfloat *px0 = (xtfloat*)p64_x0;
                int k;
                for(k = 0; k < cols; k++, p_vec0++, p_vec1++)
                {
                    AE_LSXC(_xtfloat_mat_0, px0, sizeof(xtfloat));
                    x00 = AE_MOVXTFLOATX2_FROMXTFLOAT(_xtfloat_mat_0);
                    vec0_0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec0));
                    vec1_0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec1));
                    vec01 = XT_SEL32_HH_SX2(vec0_0, vec1_0);
                    acc_row0_vec0 = ADD_SX2(acc_row0_vec0, MUL_SX2(x00, vec01));
                }
                acc_row0_vec0 = ADD_SX2(acc_row0_vec0, XT_SEL32_HH_SX2(_xtfloat_bias_0, _xtfloat_bias_1));
 
                p_out_tmp = (xtfloat *)(&(p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(HIGH_S(acc_row0_vec0),p_out_tmp,0);
                p_out_tmp = (xtfloat *)(&(p_out[(vec_itr+1)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(LOW_S(acc_row0_vec0),p_out_tmp,0);
            }
        }
        if(vec_count & 0x1)
        {
            vec_itr = vec_count -1;
            for(m_itr = 0; m_itr < (rows); m_itr ++)
            {
                xtfloat _xtfloat_bias = ZERO_S();
                if(p_bias != NULL){
                _xtfloat_bias = p_bias[vec_itr];
                }
                xtfloat _xtfloat_temp_0_0 = ZERO_S();
                xtfloat _xtfloat_acc_0_0 = ZERO_S();
                xtfloat _xtfloat_vec_batch_0  = ZERO_S() ;
                xtfloat *_xtfloat_p_vec_batch_0  = (xtfloat *)(&p_vec[(vec_itr)*vec_offset]);
                xtfloat _xtfloat_mat_0 = ZERO_S();
                ae_int16x4 *_xt16x4_p_mat_0 = (ae_int16x4 *) p_mat;
                AE_ADDCIRC16X4_XC(_xt16x4_p_mat_0, (m_itr)*row_offset*sizeof(FLOAT32));
                xtfloat *_xtfloat_p_mat_0 = (xtfloat*)_xt16x4_p_mat_0;
                for(c_itr = 0; c_itr < cols; c_itr++)
                {
                    AE_LSIP(_xtfloat_vec_batch_0, _xtfloat_p_vec_batch_0, INCREMENT_IN_BYTES_FOR_FLOAT32);
                    AE_LSXC(_xtfloat_mat_0, _xtfloat_p_mat_0, INCREMENT_IN_BYTES_FOR_FLOAT32);
                    _xtfloat_temp_0_0 = MUL_S(_xtfloat_vec_batch_0,_xtfloat_mat_0);

                    _xtfloat_acc_0_0 = ADD_S(_xtfloat_acc_0_0,_xtfloat_temp_0_0);
                }
                _xtfloat_acc_0_0=ADD_S(_xtfloat_acc_0_0,_xtfloat_bias);
                p_out_tmp = (xtfloat *)(&(p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]));
                AE_SSIP(_xtfloat_acc_0_0,p_out_tmp,0);
            }
        }
    }
    return 0;
}
#else
WORD32 xa_nn_matXvec_f32_circ(
    FLOAT32 *__restrict__ p_out,            /* output pointer */
    FLOAT32 *__restrict__ p_mat,            /* matrix: rows x cols */
    const FLOAT32 *__restrict__ p_vec,            /* vec: cols x 1 */
    const FLOAT32 *__restrict__ p_bias,           /* bias TBD: Need array? */
    WORD32 rows,                            /* Number of rows in matrix */
    WORD32 cols,                            /* Number of columns in matrix */
    WORD32 row_offset,                      /* row stride for matrix */
    WORD32 vec_count,                       /* number of vectors: 2, 4, 2n */
    WORD32 vec_offset,                      /* offset from current to next vector */
    WORD32 out_col_offset,
    WORD32 out_row_offset)
{
    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
    xtfloat* p_out_tmp;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    vec_itr = 0;

    if(rows >= ROW_UNROLL)
    {
        if(vec_count >= VEC_UNROLL)
        {
            for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
            {
                for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
                {
                    SETUP_BIAS_BATCH;
                    SETUP_ACC_BATCH;
                    SETUP_VEC_BATCH;
                    SETUP_MAT;

                    for(c_itr = 0; c_itr < (cols >> 1); c_itr++)
                    {
                        LOAD_VEC_BATCH;
                        LOAD_MAT;
                        KERNEL_MAT_VEC_BATCH;
                    }

                    ADD_BIAS_ACC_BATCH;
                    STORE_ACC_BATCH;
                }

                for(; m_itr < rows; m_itr++)
                {
                    UNROLL_ROW_SETUP_BIAS_BATCH(0);
                    UNROLL_ROW_SETUP_ACC_BATCH(0);
                    SETUP_VEC_BATCH;
                    UNROLL_SETUP_MAT(0);

                    for(c_itr = 0; c_itr < (cols >> 1); c_itr++)
                    {
                        LOAD_VEC_BATCH;
                        UNROLL_LOAD_ROW_MAT(0);
                        UNROLL_ROW_KERNEL_MAT_VEC_BATCH(0);
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
                for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
                {
                    SETUP_BIAS_BATCH_TAIL;
                    SETUP_ACC_BATCH_TAIL;
                    UNROLL_SETUP_VEC_BATCH(0);
                    SETUP_MAT;

                    for(c_itr = 0; c_itr < (cols >> 1); c_itr++)
                    {
                        UNROLL_LOAD_VEC_BATCH(0);
                        LOAD_MAT;
                        KERNEL_MAT_VEC_BATCH_TAIL;
                    }

                    ADD_BIAS_ACC_BATCH_TAIL;
                    STORE_ACC_BATCH_TAIL;
                }

                for(; m_itr < rows; m_itr++)
                {
                    UNROLL_SETUP_BIAS_BATCH(0,0);
                    UNROLL_SETUP_ACC_BATCH(0,0);
                    UNROLL_SETUP_VEC_BATCH(0);
                    UNROLL_SETUP_MAT(0);

                    for(c_itr = 0; c_itr < (cols >> 1); c_itr++)
                    {
                        UNROLL_LOAD_VEC_BATCH(0);
                        UNROLL_LOAD_ROW_MAT(0);
                        UNROLL_KERNEL_MAT_VEC_BATCH(0,0);
                    }

                    UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                    UNROLL_STORE_ACC_BATCH(0,0);
                }
            }
        }
    }
    else
    {
        if(vec_count >= VEC_UNROLL)
        {
            for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
            {
                for(m_itr = 0; m_itr < rows; m_itr++)
                {
                    UNROLL_ROW_SETUP_BIAS_BATCH(0);
                    UNROLL_ROW_SETUP_ACC_BATCH(0);
                    SETUP_VEC_BATCH;
                    UNROLL_SETUP_MAT(0);

                    for(c_itr = 0; c_itr < (cols >> 1); c_itr++)
                    {
                        LOAD_VEC_BATCH;
                        UNROLL_LOAD_ROW_MAT(0);
                        UNROLL_ROW_KERNEL_MAT_VEC_BATCH(0);
                    }

                    UNROLL_ROW_ADD_BIAS_ACC(0);
                    UNROLL_ROW_STORE_ACC(0);

                }
            }
        }
        { /* Tail loop for vec unroll */
            for(; vec_itr < vec_count; vec_itr++)
            {
                for(m_itr = 0; m_itr < rows; m_itr++)
                {
                    UNROLL_SETUP_BIAS_BATCH(0,0);
                    UNROLL_SETUP_ACC_BATCH(0,0);
                    UNROLL_SETUP_VEC_BATCH(0);
                    UNROLL_SETUP_MAT(0);

                    for(c_itr = 0; c_itr < (cols >> 1); c_itr++)
                    {
                        UNROLL_LOAD_VEC_BATCH(0);
                        UNROLL_LOAD_ROW_MAT(0);
                        UNROLL_KERNEL_MAT_VEC_BATCH(0,0);
                    }

                    UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                    UNROLL_STORE_ACC_BATCH(0,0);
                }
            }
        }
    }
    return 0;
}
#endif
#endif /* #if !HAVE_VFPU */
