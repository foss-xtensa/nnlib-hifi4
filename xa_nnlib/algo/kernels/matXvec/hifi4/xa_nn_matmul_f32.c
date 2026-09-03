/*******************************************************************************
* Copyright (c) 2018-2026 Cadence Design Systems, Inc.
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

#ifdef ROW_UNROLL
    #undef ROW_UNROLL
    #define ROW_UNROLL 4
#else
    #define ROW_UNROLL 4
#endif

#include "xa_nnlib_common_macros.h"

#ifndef ZERO_SX2
#define ZERO_SX2 XT_ZERO_SX2
#endif

#ifndef ZERO_S
#define ZERO_S XT_ZERO_S
#endif

/*----------------------------Main function---------------------------------*/

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_matmul_f32xf32_f32,(
    FLOAT32 * __restrict__ p_out,        
    const FLOAT32 * __restrict__ p_mat1, 
    const FLOAT32 * __restrict__ p_vec1, 
    const FLOAT32 * __restrict__ p_bias, 
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,                   
    WORD32 vec_count,                     
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride))                      

#else

#if NO_AGGR_FLOAT_OPT
/* Using the 4 row 1 vec function defined in xa_nn_matXvec_f32.c for xa_nn_matXvec_f32() kernel */
static inline void _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
    (xtfloatx2* out_0_0
    ,xtfloatx2* out_1_0
    ,xtfloat*   px0
    ,xtfloat*   py
    ,WORD32     cols1
    ,WORD32     row_stride1
    )
{
  xtfloatx2 acc00, acc20;
  xtfloatx2 x00, x20;
  xtfloatx2 y0;

  xtfloat* px1 = px0 + 4*row_stride1; //next 4th row 
  xtfloat* px2 = px1 + 4*row_stride1; //next 4th row
  xtfloat* px3 = px2 + 4*row_stride1; //next 4th row 
  
  xtfloatx2 z0 = *out_0_0;
  xtfloatx2 z1 = *out_1_0;

  /* Pre loop computation */
  int k;

  acc00 = ZERO_SX2();
  acc20 = ZERO_SX2();
  for(k = 0; k < cols1; k++, px0++, px1++, px2++, px3++, py++)
  {
      x00 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(px0)), AE_MOVXTFLOATX2_FROMXTFLOAT(*(px1)));
      x20 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(px2)), AE_MOVXTFLOATX2_FROMXTFLOAT(*(px3)));
      y0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*(py));
      acc00 = XT_ADD_SX2(acc00, XT_MUL_SX2(x00, y0));
      acc20 = XT_ADD_SX2(acc20, XT_MUL_SX2(x20, y0));
  }
  z0 = XT_ADD_SX2(z0, acc00);
  z1 = XT_ADD_SX2(z1, acc20);

  *out_0_0 = z0;
  *out_1_0 = z1;
}

static inline void _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
    (xtfloatx2* out_0_0
    ,xtfloatx2* out_1_0
    ,xtfloatx2* out_0_1
    ,xtfloatx2* out_1_1
    ,xtfloatx2* out_0_2
    ,xtfloatx2* out_1_2
    ,xtfloatx2* out_0_3
    ,xtfloatx2* out_1_3
    ,xtfloat*   px0
    ,xtfloat*   p_vec0
    ,WORD32     cols1
    ,WORD32     row_stride1
    ,WORD32     vec_offset
    )
{
  xtfloatx2 acc_row0_vec0, acc_row0_vec1;
  xtfloatx2 acc_row2_vec0, acc_row2_vec1;
  xtfloatx2 acc_row0_vec2, acc_row0_vec3;
  xtfloatx2 acc_row2_vec2, acc_row2_vec3;
  xtfloatx2 x00;
  xtfloatx2 x20;
  xtfloatx2 vec0_0;
  xtfloatx2 vec1_0;
  xtfloatx2 vec2_0;
  xtfloatx2 vec3_0;
 
  xtfloat* px1 = px0 + 4*row_stride1; //next 4th row
  xtfloat* px2 = px1 + 4*row_stride1; //next 4th row
  xtfloat* px3 = px2 + 4*row_stride1; //next 4th row

  xtfloat *p_vec1  = (xtfloat *)(p_vec0 + vec_offset);
  xtfloat *p_vec2  = (xtfloat *)(p_vec1 + vec_offset);
  xtfloat *p_vec3  = (xtfloat *)(p_vec2 + vec_offset);
  
  xtfloatx2 z0 = *out_0_0;
  xtfloatx2 z1 = *out_1_0;
  xtfloatx2 z2 = *out_0_1;
  xtfloatx2 z3 = *out_1_1;
  xtfloatx2 z4 = *out_0_2;
  xtfloatx2 z5 = *out_1_2;
  xtfloatx2 z6 = *out_0_3;
  xtfloatx2 z7 = *out_1_3;
 
  int k;
  
  acc_row0_vec0 = acc_row2_vec0 = ZERO_SX2();
  acc_row0_vec1 = acc_row2_vec1 = ZERO_SX2();
  acc_row0_vec2 = acc_row2_vec2 = ZERO_SX2();
  acc_row0_vec3 = acc_row2_vec3 = ZERO_SX2();
  for(k = 0; k < cols1; k++, px0++, px1++, px2++, px3++, p_vec0++, p_vec1++, p_vec2++, p_vec3++)
  {
      x00 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(px0)), AE_MOVXTFLOATX2_FROMXTFLOAT(*(px1)));
      x20 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(px2)), AE_MOVXTFLOATX2_FROMXTFLOAT(*(px3)));
      vec0_0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec0));
      vec1_0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec1));
      vec2_0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec2));
      vec3_0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*(p_vec3));
      
      acc_row0_vec0 = XT_ADD_SX2(acc_row0_vec0, XT_MUL_SX2(x00, vec0_0));
      acc_row2_vec0 = XT_ADD_SX2(acc_row2_vec0, XT_MUL_SX2(x20, vec0_0));
      acc_row0_vec1 = XT_ADD_SX2(acc_row0_vec1, XT_MUL_SX2(x00, vec1_0));
      acc_row2_vec1 = XT_ADD_SX2(acc_row2_vec1, XT_MUL_SX2(x20, vec1_0));
      acc_row0_vec2 = XT_ADD_SX2(acc_row0_vec2, XT_MUL_SX2(x00, vec2_0));
      acc_row2_vec2 = XT_ADD_SX2(acc_row2_vec2, XT_MUL_SX2(x20, vec2_0));
      acc_row0_vec3 = XT_ADD_SX2(acc_row0_vec3, XT_MUL_SX2(x00, vec3_0));
      acc_row2_vec3 = XT_ADD_SX2(acc_row2_vec3, XT_MUL_SX2(x20, vec3_0));
  }
  z0 = XT_ADD_SX2(z0, acc_row0_vec0);
  z1 = XT_ADD_SX2(z1, acc_row2_vec0);
  z2 = XT_ADD_SX2(z2, acc_row0_vec1);
  z3 = XT_ADD_SX2(z3, acc_row2_vec1);
  z4 = XT_ADD_SX2(z4, acc_row0_vec2);
  z5 = XT_ADD_SX2(z5, acc_row2_vec2);
  z6 = XT_ADD_SX2(z6, acc_row0_vec3);
  z7 = XT_ADD_SX2(z7, acc_row2_vec3);

  *out_0_0 = z0;
  *out_1_0 = z1;
  *out_0_1 = z2;
  *out_1_1 = z3;
  *out_0_2 = z4;
  *out_1_2 = z5;
  *out_0_3 = z6;
  *out_1_3 = z7;
}

static inline void _xa_nn_dot_product_1_row_4_vecs_unaligned
    (xtfloatx2* out_0_0
    ,xtfloatx2* out_1_0
    ,xtfloat*   py
    ,xtfloat*   pv0
    ,WORD32     cols1
    ,WORD32     vec_offset
    )
{
  xtfloatx2 acc00;
  xtfloatx2 acc20, acc21, acc22, acc23;
  xtfloatx2 acc30, acc31, acc32, acc33;
  xtfloatx2 v00;
  xtfloatx2 v20;
  xtfloatx2 y0;

  xtfloat* pv1 = pv0 + vec_offset; //next vec 
  xtfloat* pv2 = pv1 + vec_offset; //next vec
  xtfloat* pv3 = pv2 + vec_offset; //next vec 
  
  xtfloatx2 z0 = *out_0_0;
  xtfloatx2 z1 = *out_1_0;

  int k;

  acc00 = ZERO_SX2();
  acc20 = acc21 = acc30 = acc31 = ZERO_SX2();
  acc22 = acc23 = acc32 = acc33 = ZERO_SX2();

  acc20 =  XT_ADD_SX2(XT_ADD_SX2(acc20, acc21), XT_ADD_SX2(acc22, acc23));
  acc30 =  XT_ADD_SX2(XT_ADD_SX2(acc30, acc31), XT_ADD_SX2(acc32, acc33));

  acc00 = ZERO_SX2();
  acc20 = ZERO_SX2();
  for(k = 0; k < cols1; k++, pv0++, pv1++, pv2++, pv3++, py++)
  {
      v00 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(pv0)), AE_MOVXTFLOATX2_FROMXTFLOAT(*(pv1)));
      v20 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(pv2)), AE_MOVXTFLOATX2_FROMXTFLOAT(*(pv3)));
      y0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*(py));
      acc00 = XT_ADD_SX2(acc00, XT_MUL_SX2(v00, y0));
      acc20 = XT_ADD_SX2(acc20, XT_MUL_SX2(v20, y0));
  }
  z0 = XT_ADD_SX2(z0, acc00);
  z1 = XT_ADD_SX2(z1, acc20);

  *out_0_0 = z0;
  *out_1_0 = z1;
}

WORD32 xa_nn_matmul_f32xf32_f32(
    FLOAT32 * __restrict__ p_out,          
    const FLOAT32 * __restrict__ p_mat1,   
    const FLOAT32 * __restrict__ p_vec1,   
    const FLOAT32 * __restrict__ p_bias,   
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,                    
    WORD32 vec_count,                      
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride)                      
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
    XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
  
    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    vec_itr = 0;

    {
      for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
      {
        xtfloat *p_out_0 = (xtfloat*)(p_out + (vec_itr + 0)*out_offset);
        xtfloat *p_out_1 = (xtfloat*)(p_out + (vec_itr + 1)*out_offset);
        xtfloat *p_out_2 = (xtfloat*)(p_out + (vec_itr + 2)*out_offset);
        xtfloat *p_out_3 = (xtfloat*)(p_out + (vec_itr + 3)*out_offset);
        int ii;
        for(m_itr = 0; m_itr < (rows & ~(16 - 1)); m_itr += 16)
        {
          for(ii = 0; ii < 4; ii++)
          {
            xtfloat *p_out_0_ii = p_out_0 + (m_itr + ii) * out_stride;
            xtfloat *p_out_1_ii = p_out_1 + (m_itr + ii) * out_stride;
            xtfloat *p_out_2_ii = p_out_2 + (m_itr + ii) * out_stride;
            xtfloat *p_out_3_ii = p_out_3 + (m_itr + ii) * out_stride;
            /* Init out registers with bias */
            xtfloatx2 z0, z1, z2, z3;
            xtfloatx2 z4, z5, z6, z7;
            z0 = z1 = z2 = z3 = ZERO_SX2();
            z4 = z5 = z6 = z7 = ZERO_SX2();
            if(p_bias != NULL)
            {
              z6 = z4 = z2 = z0 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(p_bias[m_itr+ii+0]), AE_MOVXTFLOATX2_FROMXTFLOAT(p_bias[m_itr+ii+4]));
              z7 = z5 = z3 = z1 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(p_bias[m_itr+ii+8]), AE_MOVXTFLOATX2_FROMXTFLOAT(p_bias[m_itr+ii+12]));
            }
            
            xtfloat *p_mat = (xtfloat *)(p_mat1+((m_itr+ii)*row_stride1));
            xtfloat *p_vec = (xtfloat *)(p_vec1+(vec_itr*vec_offset));

            _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
              (&z0
              ,&z1
              ,&z2
              ,&z3
              ,&z4
              ,&z5
              ,&z6
              ,&z7
              ,(xtfloat *)p_mat
              ,(xtfloat *)p_vec
              ,cols1
              ,row_stride1
              ,vec_offset
              );
            
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z0,z0)), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z0), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z1,z1)), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z1), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z2,z2)), p_out_1_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z2), p_out_1_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z3,z3)), p_out_1_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z3), p_out_1_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z4,z4)), p_out_2_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z4), p_out_2_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z5,z5)), p_out_2_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z5), p_out_2_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z6,z6)), p_out_3_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z6), p_out_3_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z7,z7)), p_out_3_ii, 4*out_stride*sizeof(xtfloat));
            XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z7), p_out_3_ii, 4*out_stride*sizeof(xtfloat));
          }
        }
        p_out_0 = p_out_0 + (rows & (~15)) * out_stride;
        p_out_1 = p_out_1 + (rows & (~15)) * out_stride;
        p_out_2 = p_out_2 + (rows & (~15)) * out_stride;
        p_out_3 = p_out_3 + (rows & (~15)) * out_stride;
        
        //Remaining (rows % 16) rows
        for(m_itr = (rows & ~(15)); m_itr < rows; m_itr++)
        {
          /* Init out registers with bias */
          xtfloatx2 z0, z1;
          z0 = z1 = ZERO_SX2();
          if(p_bias != NULL)
          {
            z0 = z1 = AE_MOVXTFLOATX2_FROMXTFLOAT(p_bias[m_itr]);
          }
          
          xtfloat *p_mat = (xtfloat *)(p_mat1+(m_itr*row_stride1));
          xtfloat *p_vec = (xtfloat *)(p_vec1+(vec_itr*vec_offset));

          _xa_nn_dot_product_1_row_4_vecs_unaligned
            (&z0
            ,&z1
            ,(xtfloat *)p_mat
            ,(xtfloat *)p_vec
            ,cols1
            ,vec_offset
            );
         
          XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z0,z0)), p_out_0, out_stride*sizeof(xtfloat));
          XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z0), p_out_1, out_stride*sizeof(xtfloat));
          XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z1,z1)), p_out_2, out_stride*sizeof(xtfloat));
          XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z1), p_out_3, out_stride*sizeof(xtfloat));
        }
      }
      /* Tail loop for vec unroll */
      for(vec_itr = (vec_count & ~(3)); vec_itr < vec_count; vec_itr++)
      {
        int ii;
        xtfloat *p_out_0 = (xtfloat *)(p_out + (vec_itr*out_offset));
        for(m_itr = 0; m_itr < (rows & ~(16 - 1)); m_itr += 16)
        {
          for(ii = 0; ii < 4; ii++)
          {
              xtfloat *p_out_0_ii = p_out_0 + (m_itr + ii) * out_stride;
              /* Init out registers with bias */
              xtfloatx2 z0, z1;
              z0 = z1 = ZERO_SX2();
              if(p_bias != NULL)
              {
                z0 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(p_bias[m_itr+ii+0]), AE_MOVXTFLOATX2_FROMXTFLOAT(p_bias[m_itr+ii+4]));
                z1 = XT_SEL32_LL_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(p_bias[m_itr+ii+8]), AE_MOVXTFLOATX2_FROMXTFLOAT(p_bias[m_itr+ii+12]));
              }
              
              xtfloat *p_mat = (xtfloat *)(p_mat1+((m_itr+ii)*row_stride1));
              xtfloat *p_vec = (xtfloat *)(p_vec1+(vec_itr*vec_offset));

              _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
                (&z0
                ,&z1
                ,(xtfloat *)p_mat
                ,(xtfloat *)p_vec
                ,cols1
                ,row_stride1
                );
              
              XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z0,z0)), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z0), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(XT_SEL32_HH_SX2(z1,z1)), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(z1), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
            }
        }

        p_out_0 = p_out_0 + (rows & (~15)) * out_stride;
        xtfloat bias = ZERO_S();
        xtfloat *pbias = (xtfloat *) p_bias + m_itr;
        for(m_itr = (rows & ~(15)); m_itr < rows; m_itr++)
        {
          xtfloatx2 acc_row0_vec0;
          xtfloatx2 _xtfloatx2_temp;
          xtfloatx2 vec_batch_0_0;
          xtfloatx2 mat1_0_0;
          
          xtfloat *p_vec_batch_0  = (xtfloat *)(p_vec1 + (vec_itr + 0)*vec_offset);
          xtfloat *p_mat1_0 = (xtfloat *) &p_mat1[(m_itr+0)*row_stride1];
          
          acc_row0_vec0 = ZERO_SX2();
          
          /* Remainder loop for cols1 */
          for(c_itr = 0; c_itr < cols1; c_itr++, 
              p_vec_batch_0++, p_mat1_0++)
          {
              vec_batch_0_0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*((xtfloat *)p_vec_batch_0));
              mat1_0_0 = AE_MOVXTFLOATX2_FROMXTFLOAT(*((xtfloat *)p_mat1_0));
              _xtfloatx2_temp = XT_MUL_SX2(vec_batch_0_0, mat1_0_0);
              acc_row0_vec0 = XT_ADD_SX2(acc_row0_vec0, _xtfloatx2_temp);
          }
          if(p_bias!=NULL)
          {
            XT_LSIP(bias, pbias, 4);
            acc_row0_vec0 = AE_MOVXTFLOATX2_FROMXTFLOAT(XT_ADD_S(AE_MOVXTFLOAT_FROMXTFLOATX2(acc_row0_vec0), bias));
          }
         
          XT_SSXP(AE_MOVXTFLOAT_FROMXTFLOATX2(acc_row0_vec0), p_out_0, out_stride*sizeof(xtfloat));
        }
      }
    }

    return 0;
}
#else
WORD32 xa_nn_matmul_f32xf32_f32(

    FLOAT32 * __restrict__ p_out,          
    const FLOAT32 * __restrict__ p_mat1,   
    const FLOAT32 * __restrict__ p_vec1,   
    const FLOAT32 * __restrict__ p_bias,   
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,                    
    WORD32 vec_count,                      
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride)                      
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
    XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
  
    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
    xtfloat* p_out_tmp;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    vec_itr = 0;

    #define VEC_UNROLL 2
    #define UNROLL_ROW_SETUP_ACC_BATCH          SETUP_ACC_BATCH_ROW_FOR_f32
    #define UNROLL_SETUP_ACC_BATCH              SETUP_ACC_BATCH_FOR_f32_MATMUL
    #define UNROLL_SETUP_MAT1                   SETUP_MAT1_f32
    #define UNROLL_SETUP_VEC_BATCH              SETUP_VEC_OFFSET_BATCH_f32
    #define SETUP_BIAS                          SETUP_BIAS_f32
    #define UNROLL_LOAD_VEC_BATCH               LOAD_VEC_BATCH_f32
    #define UNROLL_LOAD_ROW_MAT1                LOAD_ROW_MAT1_f32
    #define LOAD_BIAS                           LOAD_BIAS_f32_MATMUL
    #define UNROLL_ROW_KERNEL_MAT1_VEC_BATCH    KERNEL_MAT1_VEC_BATCH_ROW_f32
    #define UNROLL_KERNEL_MAT1_VEC_BATCH        KERNEL_MAT1_VEC_BATCH_f32
    #define UNROLL_ROW_ADD_BIAS_ACC             ADD_BIAS_BATCH_ROW_ACC_FOR_f32_MATMUL
    #define UNROLL_ADD_BIAS_ACC_BATCH           ADD_BIAS_BATCH_ACC_FOR_f32_MATMUL
    #define UNROLL_ROW_STORE_ACC                STORE_ACC_BATCH_ROW_AT_OUT_f32
    #define UNROLL_STORE_ACC_BATCH              STORE_STRIDE_ACC_BATCH_AT_OUT_f32

    int chk_align = 0;
    CHK_MATMUL_ALIGN(chk_align, p_mat1, 2 * sizeof(FLOAT32), p_vec1, 2 * sizeof(FLOAT32), cols1, row_stride1, vec_offset, 2);
    if(chk_align)
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

                    for(c_itr = 0; c_itr < (cols1 >> 1); c_itr++)
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

                    for(c_itr = 0; c_itr < (cols1 >> 1); c_itr++)
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

                    for(c_itr = 0; c_itr < (cols1 >> 1); c_itr++)
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

                    for(c_itr = 0; c_itr < (cols1 >> 1); c_itr++)
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
    #undef ROW_UNROLL
    #undef VEC_UNROLL
    }
    else if (p_mat1 && p_vec1)
    {
        #define ROW_UNROLL 2
        #define VEC_UNROLL 2
        #define UNROLL_SETUP_ACC_BATCH              SETUP_ACC_BATCH_FOR_f32_MATMUL
        #define SETUP_BIAS                          SETUP_BIAS_f32
        #define LOAD_BIAS                           LOAD_BIAS_f32_MATMUL
        #define UNROLL_ADD_BIAS_ACC_BATCH           ADD_BIAS_BATCH_ACC_FOR_f32_MATMUL
        if(vec_count > VEC_UNROLL)
        {
            for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
            {
                SETUP_BIAS;
                for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
                {
                    UNROLL_SETUP_ACC_BATCH(0,0);
                    UNROLL_SETUP_ACC_BATCH(0,1);
                    UNROLL_SETUP_ACC_BATCH(1,0);
                    UNROLL_SETUP_ACC_BATCH(1,1);
                    SETUP_VEC_OFFSET_BATCH_f32_UNALIGNED(0);
                    SETUP_VEC_OFFSET_BATCH_f32_UNALIGNED(1);
                    SETUP_MAT1_f32_UNALIGNED(0);
                    SETUP_MAT1_f32_UNALIGNED(1);

                    int cols1_count = cols1- cols1%2;
                    for(c_itr = 0; c_itr < (cols1_count >> 1); c_itr++)
                    {
                        LOAD_VEC_BATCH_f32_UNALIGNED(0);
                        LOAD_VEC_BATCH_f32_UNALIGNED(1);
                        LOAD_ROW_MAT1_f32_UNALIGNED(0);
                        LOAD_ROW_MAT1_f32_UNALIGNED(1);
                        KERNEL_MAT1_VEC_BATCH_f32(0,0);
                        KERNEL_MAT1_VEC_BATCH_f32(1,0);
                        KERNEL_MAT1_VEC_BATCH_f32(0,1);
                        KERNEL_MAT1_VEC_BATCH_f32(1,1);
                    }
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_VEC_BATCH_f32_SINGLE_UNALIGNED(0);
                        LOAD_VEC_BATCH_f32_SINGLE_UNALIGNED(1);
                        LOAD_ROW_MAT1_f32_SINGLE_UNALIGNED(0);
                        LOAD_ROW_MAT1_f32_SINGLE_UNALIGNED(1);
                        KERNEL_MAT1_VEC_BATCH_f32_SINGLE_UNALIGNED(0,0);
                        KERNEL_MAT1_VEC_BATCH_f32_SINGLE_UNALIGNED(1,0);
                        KERNEL_MAT1_VEC_BATCH_f32_SINGLE_UNALIGNED(0,1);
                        KERNEL_MAT1_VEC_BATCH_f32_SINGLE_UNALIGNED(1,1);
                    }

                    ADD_BIAS_BATCH_ROW_ACC_FOR_f32(0);
                    ADD_BIAS_BATCH_ROW_ACC_FOR_f32(1);
                    STORE_STRIDE_ACC_BATCH_AT_OUT_f32(0,0);
                    STORE_STRIDE_ACC_BATCH_AT_OUT_f32(1,0);
                    STORE_STRIDE_ACC_BATCH_AT_OUT_f32(0,1);
                    STORE_STRIDE_ACC_BATCH_AT_OUT_f32(1,1);
                }
                //Remaining row
                for(; m_itr < rows; m_itr++)
                {
                    UNROLL_SETUP_ACC_BATCH(0,0);
                    UNROLL_SETUP_ACC_BATCH(0,1);
                    SETUP_VEC_OFFSET_BATCH_f32_UNALIGNED(0);
                    SETUP_VEC_OFFSET_BATCH_f32_UNALIGNED(1);
                    SETUP_MAT1_f32_UNALIGNED(0);
                    int cols1_count = cols1- cols1%2;

                    for(c_itr = 0; c_itr < (cols1_count >> 1); c_itr++)
                    {
                        LOAD_VEC_BATCH_f32_UNALIGNED(0);
                        LOAD_VEC_BATCH_f32_UNALIGNED(1);
                        LOAD_ROW_MAT1_f32_UNALIGNED(0);
                        KERNEL_MAT1_VEC_BATCH_f32(0,0);
                        KERNEL_MAT1_VEC_BATCH_f32(0,1);
                    }
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_VEC_BATCH_f32_SINGLE_UNALIGNED(0);
                        LOAD_VEC_BATCH_f32_SINGLE_UNALIGNED(1);
                        LOAD_ROW_MAT1_f32_SINGLE_UNALIGNED(0);
                        KERNEL_MAT1_VEC_BATCH_f32_SINGLE_UNALIGNED(0,0);
                        KERNEL_MAT1_VEC_BATCH_f32_SINGLE_UNALIGNED(0,1);
                    }
                    ADD_BIAS_BATCH_ROW_ACC_FOR_f32(0);
                    STORE_STRIDE_ACC_BATCH_AT_OUT_f32(0,0);
                    STORE_STRIDE_ACC_BATCH_AT_OUT_f32(0,1);
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
                    UNROLL_SETUP_ACC_BATCH(0,0);
                    UNROLL_SETUP_ACC_BATCH(1,0);
                    SETUP_VEC_OFFSET_BATCH_f32_UNALIGNED(0);
                    SETUP_MAT1_f32_UNALIGNED(0);
                    SETUP_MAT1_f32_UNALIGNED(1);
                    int cols1_count = cols1 - cols1%2;

                    for(c_itr = 0; c_itr < (cols1_count >> 1); c_itr++)
                    {
                        LOAD_VEC_BATCH_f32_UNALIGNED(0);
                        LOAD_ROW_MAT1_f32_UNALIGNED(0);
                        LOAD_ROW_MAT1_f32_UNALIGNED(1);
                        KERNEL_MAT1_VEC_BATCH_f32(0,0);
                        KERNEL_MAT1_VEC_BATCH_f32(1,0);
                    }
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_VEC_BATCH_f32_SINGLE_UNALIGNED(0);
                        LOAD_ROW_MAT1_f32_SINGLE_UNALIGNED(0);
                        LOAD_ROW_MAT1_f32_SINGLE_UNALIGNED(1);
                        KERNEL_MAT1_VEC_BATCH_f32_SINGLE_UNALIGNED(0,0);
                        KERNEL_MAT1_VEC_BATCH_f32_SINGLE_UNALIGNED(1,0);
                    }  

                    LOAD_BIAS; 
                    UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                    LOAD_BIAS; 
                    UNROLL_ADD_BIAS_ACC_BATCH(1,0);
                
                    STORE_STRIDE_ACC_BATCH_AT_OUT_f32(0,0);
                    STORE_STRIDE_ACC_BATCH_AT_OUT_f32(1,0);
                }

                for(; m_itr < rows; m_itr++)
                {
                    UNROLL_SETUP_ACC_BATCH(0,0);
                    SETUP_VEC_OFFSET_BATCH_f32_UNALIGNED(0);
                    SETUP_MAT1_f32_UNALIGNED(0);
                    int cols1_count = cols1 - cols1%2;

                    for(c_itr = 0; c_itr < (cols1_count >> 1); c_itr++)
                    {
                        LOAD_VEC_BATCH_f32_UNALIGNED(0);
                        LOAD_ROW_MAT1_f32_UNALIGNED(0);
                        KERNEL_MAT1_VEC_BATCH_f32(0,0);
                    }
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_VEC_BATCH_f32_SINGLE_UNALIGNED(0);
                        LOAD_ROW_MAT1_f32_SINGLE_UNALIGNED(0);
                        KERNEL_MAT1_VEC_BATCH_f32_SINGLE_UNALIGNED(0,0);
                    }

                    LOAD_BIAS;
                    UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                    STORE_STRIDE_ACC_BATCH_AT_OUT_f32(0,0);
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
    #undef UNROLL_ROW_STORE_ACC
    #undef UNROLL_STORE_ACC_BATCH
    #undef VEC_UNROLL
    #undef ROW_UNROLL

    return 0;
}
#endif
#endif

WORD32 xa_nn_matmul_v2_f32xf32_f32(
    FLOAT32 * __restrict__ p_out,          
    const FLOAT32 * __restrict__ p_mat1,   
    const FLOAT32 * __restrict__ p_mat2,
    const FLOAT32 * __restrict__ pt_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_stride,
    WORD32 vec_count,                      
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride,
    FLOAT32 out_activation_min,
    FLOAT32 out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
    return -1;
}