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

#if XCHAL_HAVE_HIFI1S
#include "xa_nnlib_common_macros.h"
static WORD32 internal_calc_mzbsum(WORD32 mat_zero_bias, WORD32 vec_zero_bias, const WORD8 * __restrict__ p_vec, int cols1)
{
  if(mat_zero_bias == 0){
    return 0;
  }

  WORD32 sum_mzb32 = 0, c_itr;
  WORD32 preloop_cnt = (8 - ((unsigned)p_vec-(((unsigned)p_vec)&~0x7))) & 0x07;
  if(preloop_cnt > cols1) { preloop_cnt = 0;}
  cols1 = cols1 - preloop_cnt;

  for(c_itr = 0; c_itr < preloop_cnt; c_itr++){
    int vecval = *p_vec++;
    sum_mzb32 += (vecval+vec_zero_bias)*mat_zero_bias;
  }
  
  ae_int64 sum_mzb = (ae_int64)sum_mzb32;
  ae_int16x4 mzb_16x4 = AE_MOVDA16(mat_zero_bias);
  ae_int8x8 vzb_8x8 = AE_MOVDA8(-vec_zero_bias);
  ae_int8x8 d_vec0;
  ae_int16x4 vec0_zb_0, vec0_zb_1;

  for(c_itr = 0; c_itr < cols1>>3; c_itr++){
    AE_L8X8_IP(d_vec0, (ae_int8x8 *)p_vec, 8);
    AE_SUBW8(vec0_zb_0, vec0_zb_1, d_vec0, vzb_8x8);
    AE_MULAAAAQ16(sum_mzb, mzb_16x4, vec0_zb_0);
    AE_MULAAAAQ16(sum_mzb, mzb_16x4, vec0_zb_1);
  }
  sum_mzb32 = AE_MOVINT32X2_FROMINT64(sum_mzb);

  for(c_itr = 0; c_itr < (cols1&0x7); c_itr++){
    int vecval = *p_vec++;
    sum_mzb32 += (vecval+vec_zero_bias)*mat_zero_bias;
  }
  return sum_mzb32;
}

WORD32 xa_nn_matmul_asym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
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
    XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
    XA_NNLIB_ARG_CHK_COND((vec_count <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
    XA_NNLIB_ARG_CHK_COND((mat1_zero_bias < -127 || mat1_zero_bias > 128), -1);
    XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
    XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
    XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

    int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
    left_shift = 31 - out_shift;
    left_shift = left_shift << 16 | left_shift;       
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    left_shift = out_shift<0?0:out_shift;
    right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */
    int m_itr = 0, v_itr = 0, c_itr = 0;

    if(((((unsigned)p_mat1&7) == 0) && ((row_stride1&7) == 0))  || ((((unsigned)p_vec1&7) == 0) && ((vec_offset&7) == 0)))
    {
      if(((((unsigned)p_mat1&7) == 0) && ((row_stride1&7) == 0)))
      {
          for(; m_itr < (rows &~ 0x03); m_itr+=4)
          {
              ae_int32x2 acc_00_hl, acc_01_hl;
              ae_int32x2 acc_10_hl, acc_11_hl;
              ae_int32x2 acc_20_hl, acc_21_hl;
              ae_int32x2 acc_30_hl, acc_31_hl;
              WORD8 *p_dst_0 = p_out + (m_itr * out_stride);
              WORD8 *p_dst_1 = p_out + ((m_itr + 1) * out_stride);
              WORD8 *p_dst_2 = p_out + ((m_itr + 2) * out_stride);
              WORD8 *p_dst_3 = p_out + ((m_itr + 3) * out_stride);
              for(v_itr = 0; v_itr < (vec_count &~ 0x01); v_itr+=2)
              {
                  acc_00_hl = acc_01_hl = acc_10_hl = acc_11_hl = acc_20_hl = acc_21_hl = acc_30_hl = acc_31_hl = 0;
                  WORD8* vec_ptr_0 = (WORD8*)&p_vec1[v_itr * vec_offset];
                  WORD8* vec_ptr_1 = (WORD8*)&p_vec1[(v_itr + 1) * vec_offset];
                  WORD8 *mat_ptr_0 = (WORD8*)&p_mat1[m_itr * row_stride1];
                  WORD8 *mat_ptr_1 = (WORD8*)&p_mat1[(m_itr + 1)* row_stride1];
                  WORD8 *mat_ptr_2 = (WORD8*)&p_mat1[(m_itr + 2)* row_stride1];
                  WORD8 *mat_ptr_3 = (WORD8*)&p_mat1[(m_itr + 3)* row_stride1];

                  WORD32 mat1_zb_sum_0 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_0, cols1);
                  WORD32 mat1_zb_sum_1 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_1, cols1);
                  ae_int32x2 mat1_zb_sum_0_x2 = mat1_zb_sum_0;
                  ae_int32x2 mat1_zb_sum_1_x2 = mat1_zb_sum_1;
                  ae_valign align_v0 = AE_LA64_PP(vec_ptr_0);
                  ae_valign align_v1 = AE_LA64_PP(vec_ptr_1);
                  ae_int8x8 vec0, vec1;
                  ae_int8x8 mat0, mat1, mat2, mat3;
                  ae_int16x4 vec0_zb_0, vec0_zb_1;
                  ae_int16x4 vec1_zb_0, vec1_zb_1;
                  ae_int8x8 vzb = AE_MOVDA8(-vec1_zero_bias);
                  for(c_itr=0; c_itr < cols1 >> 3; c_itr++)
                  {
                      AE_L8X8_IP(mat0, (ae_int8x8 *)mat_ptr_0, 8);
                      AE_L8X8_IP(mat1, (ae_int8x8 *)mat_ptr_1, 8);
                      AE_L8X8_IP(mat2, (ae_int8x8 *)mat_ptr_2, 8);
                      AE_L8X8_IP(mat3, (ae_int8x8 *)mat_ptr_3, 8);
                      AE_LA8X8_IP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0);
                      AE_LA8X8_IP(vec1, align_v1, (ae_int8x8 *)vec_ptr_1);
                      AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                      AE_SUBW8(vec1_zb_0, vec1_zb_1, vec1, vzb);
                      AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_01_hl, vec1_zb_0, vec1_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_11_hl, vec1_zb_0, vec1_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_20_hl, vec0_zb_0, vec0_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_21_hl, vec1_zb_0, vec1_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_30_hl, vec0_zb_0, vec0_zb_1, mat3);
                      AE_MULAAAA16Q8(acc_31_hl, vec1_zb_0, vec1_zb_1, mat3);
                  }
                  int rem_elms_shift = 64 - ((cols1 & 7) * 8);
                  if( (cols1&7 ))
                  {
                      AE_L8X8_IP(mat0, (ae_int8x8 *)mat_ptr_0, 8);
                      AE_L8X8_IP(mat1, (ae_int8x8 *)mat_ptr_1, 8);
                      AE_L8X8_IP(mat2, (ae_int8x8 *)mat_ptr_2, 8);
                      AE_L8X8_IP(mat3, (ae_int8x8 *)mat_ptr_3, 8);
                      mat0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat0), rem_elms_shift), rem_elms_shift));
                      mat1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1), rem_elms_shift), rem_elms_shift));
                      mat2 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat2), rem_elms_shift), rem_elms_shift));
                      mat3 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat3), rem_elms_shift), rem_elms_shift));
                      AE_LAV8X8_XP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0, (cols1&7));
                      AE_LAV8X8_XP(vec1, align_v1, (ae_int8x8 *)vec_ptr_1, (cols1&7));
                      AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                      AE_SUBW8(vec1_zb_0, vec1_zb_1, vec1, vzb);
                      AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_01_hl, vec1_zb_0, vec1_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_11_hl, vec1_zb_0, vec1_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_20_hl, vec0_zb_0, vec0_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_21_hl, vec1_zb_0, vec1_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_30_hl, vec0_zb_0, vec0_zb_1, mat3);
                      AE_MULAAAA16Q8(acc_31_hl, vec1_zb_0, vec1_zb_1, mat3);
                  }
                  if(p_bias != NULL){
                    ae_int32x2 bias0 = AE_MOVDA32X2(p_bias[m_itr], AE_ZERO32());
                    ae_int32x2 bias1 = AE_MOVDA32X2(p_bias[m_itr + 1], AE_ZERO32());
                    ae_int32x2 bias2 = AE_MOVDA32X2(p_bias[m_itr + 2], AE_ZERO32());
                    ae_int32x2 bias3 = AE_MOVDA32X2(p_bias[m_itr + 3], AE_ZERO32());
                    acc_00_hl = AE_ADD32S(acc_00_hl, bias0);
                    acc_01_hl = AE_ADD32S(acc_01_hl, bias0);
                    acc_10_hl = AE_ADD32S(acc_10_hl, bias1);
                    acc_11_hl = AE_ADD32S(acc_11_hl, bias1);
                    acc_20_hl = AE_ADD32S(acc_20_hl, bias2);
                    acc_21_hl = AE_ADD32S(acc_21_hl, bias2);
                    acc_30_hl = AE_ADD32S(acc_30_hl, bias3);
                    acc_31_hl = AE_ADD32S(acc_31_hl, bias3);                  
                  }
                  ae_int32x2 accum_vec0_0 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_00_hl, acc_00_hl), AE_ADD32_HL_LH(acc_10_hl, acc_10_hl));
                  ae_int32x2 accum_vec1_0 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_01_hl, acc_01_hl), AE_ADD32_HL_LH(acc_11_hl, acc_11_hl));
                  ae_int32x2 accum_vec0_1 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_20_hl, acc_20_hl), AE_ADD32_HL_LH(acc_30_hl, acc_30_hl));
                  ae_int32x2 accum_vec1_1 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_21_hl, acc_21_hl), AE_ADD32_HL_LH(acc_31_hl, acc_31_hl));

                  accum_vec0_0 = AE_ADD32S(accum_vec0_0, mat1_zb_sum_0_x2);
                  accum_vec0_1 = AE_ADD32S(accum_vec0_1, mat1_zb_sum_0_x2);
                  accum_vec1_0 = AE_ADD32S(accum_vec1_0, mat1_zb_sum_1_x2);
                  accum_vec1_1 = AE_ADD32S(accum_vec1_1, mat1_zb_sum_1_x2);
          
                  ae_int32x2 output_vec0_0, output_vec0_1, output_vec1_0, output_vec1_1;
#if TFLITE_SINGLE_ROUNDING
                  MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(output_vec0_0, output_vec0_1, accum_vec0_0, accum_vec0_1, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(output_vec1_0, output_vec1_1, accum_vec1_0, accum_vec1_1, out_multiplier, left_shift, right_shift);
#else                  
                  MPY_BY_QUANT_MULT_X2X2_OUT32(output_vec0_0, output_vec0_1, accum_vec0_0, accum_vec0_1, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_X2X2_OUT32(output_vec1_0, output_vec1_1, accum_vec1_0, accum_vec1_1, out_multiplier, left_shift, right_shift);
#endif
                  output_vec0_0 = AE_ADD32S(output_vec0_0, AE_MOVDA32(out_zero_bias));
                  output_vec0_1 = AE_ADD32S(output_vec0_1, AE_MOVDA32(out_zero_bias));
                  output_vec1_0 = AE_ADD32S(output_vec1_0, AE_MOVDA32(out_zero_bias));
                  output_vec1_1 = AE_ADD32S(output_vec1_1, AE_MOVDA32(out_zero_bias));

                  ae_int8x8 output = AE_SAT8X4X32_H(output_vec0_0, output_vec1_0);
                  ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  ae_int8x8 output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  ae_int8x8 output10 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 40));
                  ae_int8x8 output11 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 32));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_1, out_offset);
                  AE_S8_0_XP(output10, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output11, (ae_int8 *)p_dst_1, out_offset);        

                  output   = AE_SAT8X4X32_H(output_vec0_1, output_vec1_1);
                  output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  output10 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 40));
                  output11 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 32));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_2, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_3, out_offset);
                  AE_S8_0_XP(output10, (ae_int8 *)p_dst_2, out_offset);
                  AE_S8_0_XP(output11, (ae_int8 *)p_dst_3, out_offset);
              }
              for(; v_itr < (vec_count); v_itr++)
              {
                  acc_00_hl = acc_10_hl = acc_20_hl = acc_30_hl = 0;
                  WORD8* vec_ptr_0 = (WORD8*)&p_vec1[v_itr * vec_offset];
                  WORD8 *mat_ptr_0 = (WORD8*)&p_mat1[m_itr * row_stride1];
                  WORD8 *mat_ptr_1 = (WORD8*)&p_mat1[(m_itr + 1)* row_stride1];
                  WORD8 *mat_ptr_2 = (WORD8*)&p_mat1[(m_itr + 2)* row_stride1];
                  WORD8 *mat_ptr_3 = (WORD8*)&p_mat1[(m_itr + 3)* row_stride1];

                  WORD32 mat1_zb_sum_0 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_0, cols1);
                  ae_int32x2 mat1_zb_sum_0_x2 = mat1_zb_sum_0;
                  ae_valign align_v0 = AE_LA64_PP(vec_ptr_0);
                  ae_int8x8 vec0;
                  ae_int8x8 mat0, mat1, mat2, mat3;
                  ae_int16x4 vec0_zb_0, vec0_zb_1;
                  ae_int8x8 vzb = AE_MOVDA8(-vec1_zero_bias);
                  for(c_itr=0; c_itr < cols1 >> 3; c_itr++)
                  {
                      AE_L8X8_IP(mat0, (ae_int8x8 *)mat_ptr_0, 8);
                      AE_L8X8_IP(mat1, (ae_int8x8 *)mat_ptr_1, 8);
                      AE_L8X8_IP(mat2, (ae_int8x8 *)mat_ptr_2, 8);
                      AE_L8X8_IP(mat3, (ae_int8x8 *)mat_ptr_3, 8);
                      AE_LA8X8_IP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0);
                      AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                      AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_20_hl, vec0_zb_0, vec0_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_30_hl, vec0_zb_0, vec0_zb_1, mat3);
                  }
                  int rem_elms_shift = 64 - ((cols1 & 7) * 8);
                  if( (cols1&7 ))
                  {
                      AE_L8X8_IP(mat0, (ae_int8x8 *)mat_ptr_0, 8);
                      AE_L8X8_IP(mat1, (ae_int8x8 *)mat_ptr_1, 8);
                      AE_L8X8_IP(mat2, (ae_int8x8 *)mat_ptr_2, 8);
                      AE_L8X8_IP(mat3, (ae_int8x8 *)mat_ptr_3, 8);
                      mat0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat0), rem_elms_shift), rem_elms_shift));
                      mat1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1), rem_elms_shift), rem_elms_shift));
                      mat2 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat2), rem_elms_shift), rem_elms_shift));
                      mat3 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat3), rem_elms_shift), rem_elms_shift));
                      AE_LAV8X8_XP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0, (cols1&7));
                      AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                      AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_20_hl, vec0_zb_0, vec0_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_30_hl, vec0_zb_0, vec0_zb_1, mat3);
                  }
                  if(p_bias != NULL){
                    ae_int32x2 bias0 = AE_MOVDA32X2(p_bias[m_itr], AE_ZERO32());
                    ae_int32x2 bias1 = AE_MOVDA32X2(p_bias[m_itr + 1], AE_ZERO32());
                    ae_int32x2 bias2 = AE_MOVDA32X2(p_bias[m_itr + 2], AE_ZERO32());
                    ae_int32x2 bias3 = AE_MOVDA32X2(p_bias[m_itr + 3], AE_ZERO32());
                    acc_00_hl = AE_ADD32S(acc_00_hl, bias0);
                    acc_10_hl = AE_ADD32S(acc_10_hl, bias1);
                    acc_20_hl = AE_ADD32S(acc_20_hl, bias2);
                    acc_30_hl = AE_ADD32S(acc_30_hl, bias3);                  
                  }
                  ae_int32x2 accum_vec0_0 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_00_hl, acc_00_hl), AE_ADD32_HL_LH(acc_10_hl, acc_10_hl));
                  ae_int32x2 accum_vec0_1 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_20_hl, acc_20_hl), AE_ADD32_HL_LH(acc_30_hl, acc_30_hl));

                  accum_vec0_0 = AE_ADD32S(accum_vec0_0, mat1_zb_sum_0_x2);
                  accum_vec0_1 = AE_ADD32S(accum_vec0_1, mat1_zb_sum_0_x2);
          
                  ae_int32x2 output_vec0_0, output_vec0_1;
#if TFLITE_SINGLE_ROUNDING
                  MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(output_vec0_0, output_vec0_1, accum_vec0_0, accum_vec0_1, out_multiplier, left_shift, right_shift);
#else                  
                  MPY_BY_QUANT_MULT_X2X2_OUT32(output_vec0_0, output_vec0_1, accum_vec0_0, accum_vec0_1, out_multiplier, left_shift, right_shift);
#endif                  

                  output_vec0_0 = AE_ADD32S(output_vec0_0, AE_MOVDA32(out_zero_bias));
                  output_vec0_1 = AE_ADD32S(output_vec0_1, AE_MOVDA32(out_zero_bias));

                  ae_int8x8 output = AE_SAT8X4X32_H(output_vec0_0, output_vec0_0);
                  ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  ae_int8x8 output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_1, out_offset);     

                  output   = AE_SAT8X4X32_H(output_vec0_1, output_vec0_1);
                  output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_2, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_3, out_offset);
              }              
          }
      }
      else
      {
          for(; m_itr < (rows &~ 0x03); m_itr+=4)
          {
              ae_int32x2 acc_00_hl, acc_01_hl;
              ae_int32x2 acc_10_hl, acc_11_hl;
              ae_int32x2 acc_20_hl, acc_21_hl;
              ae_int32x2 acc_30_hl, acc_31_hl;
              WORD8 *p_dst_0 = p_out + (m_itr * out_stride);
              WORD8 *p_dst_1 = p_out + ((m_itr + 1) * out_stride);
              WORD8 *p_dst_2 = p_out + ((m_itr + 2) * out_stride);
              WORD8 *p_dst_3 = p_out + ((m_itr + 3) * out_stride);
              for(v_itr = 0; v_itr < (vec_count &~ 0x01); v_itr+=2)
              {
                  acc_00_hl = acc_01_hl = acc_10_hl = acc_11_hl = acc_20_hl = acc_21_hl = acc_30_hl = acc_31_hl = 0;
                  WORD8* vec_ptr_0 = (WORD8*)&p_vec1[v_itr * vec_offset];
                  WORD8* vec_ptr_1 = (WORD8*)&p_vec1[(v_itr + 1) * vec_offset];
                  WORD8 *mat_ptr_0 = (WORD8*)&p_mat1[m_itr * row_stride1];
                  WORD8 *mat_ptr_1 = (WORD8*)&p_mat1[(m_itr + 1)* row_stride1];
                  WORD8 *mat_ptr_2 = (WORD8*)&p_mat1[(m_itr + 2)* row_stride1];
                  WORD8 *mat_ptr_3 = (WORD8*)&p_mat1[(m_itr + 3)* row_stride1];

                  WORD32 mat1_zb_sum_0 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_0, cols1);
                  WORD32 mat1_zb_sum_1 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_1, cols1);
                  ae_int32x2 mat1_zb_sum_0_x2 = mat1_zb_sum_0;
                  ae_int32x2 mat1_zb_sum_1_x2 = mat1_zb_sum_1;
                  ae_valign align_m0 = AE_LA64_PP(mat_ptr_0);
                  ae_valign align_m1 = AE_LA64_PP(mat_ptr_1);
                  ae_valign align_m2 = AE_LA64_PP(mat_ptr_2);
                  ae_valign align_m3 = AE_LA64_PP(mat_ptr_3);
                  ae_int8x8 vec0, vec1;
                  ae_int8x8 mat0, mat1, mat2, mat3;
                  ae_int16x4 vec0_zb_0, vec0_zb_1;
                  ae_int16x4 vec1_zb_0, vec1_zb_1;
                  ae_int8x8 vzb = AE_MOVDA8(-vec1_zero_bias);
                  for(c_itr=0; c_itr < cols1 >> 3; c_itr++)
                  {
                      AE_LA8X8_IP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0);
                      AE_LA8X8_IP(mat1, align_m1, (ae_int8x8 *)mat_ptr_1);
                      AE_LA8X8_IP(mat2, align_m2, (ae_int8x8 *)mat_ptr_2);
                      AE_LA8X8_IP(mat3, align_m3, (ae_int8x8 *)mat_ptr_3);
                      AE_L8X8_IP(vec0, (ae_int8x8 *)vec_ptr_0, 8);
                      AE_L8X8_IP(vec1, (ae_int8x8 *)vec_ptr_1, 8);
                      AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                      AE_SUBW8(vec1_zb_0, vec1_zb_1, vec1, vzb);
                      AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_01_hl, vec1_zb_0, vec1_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_11_hl, vec1_zb_0, vec1_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_20_hl, vec0_zb_0, vec0_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_21_hl, vec1_zb_0, vec1_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_30_hl, vec0_zb_0, vec0_zb_1, mat3);
                      AE_MULAAAA16Q8(acc_31_hl, vec1_zb_0, vec1_zb_1, mat3);
                  }
                  int rem_elms_shift = 64 - ((cols1 & 7) * 8);
                  if( (cols1&7 ))
                  {
                      AE_LAV8X8_XP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0, (cols1&7));
                      AE_LAV8X8_XP(mat1, align_m1, (ae_int8x8 *)mat_ptr_1, (cols1&7));
                      AE_LAV8X8_XP(mat2, align_m2, (ae_int8x8 *)mat_ptr_2, (cols1&7));
                      AE_LAV8X8_XP(mat3, align_m3, (ae_int8x8 *)mat_ptr_3, (cols1&7));
                      AE_L8X8_IP(vec0, (ae_int8x8 *)vec_ptr_0, 8);
                      AE_L8X8_IP(vec1, (ae_int8x8 *)vec_ptr_1, 8);
                      vec0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0), rem_elms_shift), rem_elms_shift));
                      vec1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1), rem_elms_shift), rem_elms_shift));                      
                      AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                      AE_SUBW8(vec1_zb_0, vec1_zb_1, vec1, vzb);
                      AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_01_hl, vec1_zb_0, vec1_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_11_hl, vec1_zb_0, vec1_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_20_hl, vec0_zb_0, vec0_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_21_hl, vec1_zb_0, vec1_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_30_hl, vec0_zb_0, vec0_zb_1, mat3);
                      AE_MULAAAA16Q8(acc_31_hl, vec1_zb_0, vec1_zb_1, mat3);
                  }
                  if(p_bias != NULL){
                    ae_int32x2 bias0 = AE_MOVDA32X2(p_bias[m_itr], AE_ZERO32());
                    ae_int32x2 bias1 = AE_MOVDA32X2(p_bias[m_itr + 1], AE_ZERO32());
                    ae_int32x2 bias2 = AE_MOVDA32X2(p_bias[m_itr + 2], AE_ZERO32());
                    ae_int32x2 bias3 = AE_MOVDA32X2(p_bias[m_itr + 3], AE_ZERO32());
                    acc_00_hl = AE_ADD32S(acc_00_hl, bias0);
                    acc_01_hl = AE_ADD32S(acc_01_hl, bias0);
                    acc_10_hl = AE_ADD32S(acc_10_hl, bias1);
                    acc_11_hl = AE_ADD32S(acc_11_hl, bias1);
                    acc_20_hl = AE_ADD32S(acc_20_hl, bias2);
                    acc_21_hl = AE_ADD32S(acc_21_hl, bias2);
                    acc_30_hl = AE_ADD32S(acc_30_hl, bias3);
                    acc_31_hl = AE_ADD32S(acc_31_hl, bias3);                  
                  }
                  ae_int32x2 accum_vec0_0 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_00_hl, acc_00_hl), AE_ADD32_HL_LH(acc_10_hl, acc_10_hl));
                  ae_int32x2 accum_vec1_0 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_01_hl, acc_01_hl), AE_ADD32_HL_LH(acc_11_hl, acc_11_hl));
                  ae_int32x2 accum_vec0_1 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_20_hl, acc_20_hl), AE_ADD32_HL_LH(acc_30_hl, acc_30_hl));
                  ae_int32x2 accum_vec1_1 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_21_hl, acc_21_hl), AE_ADD32_HL_LH(acc_31_hl, acc_31_hl));

                  accum_vec0_0 = AE_ADD32S(accum_vec0_0, mat1_zb_sum_0_x2);
                  accum_vec0_1 = AE_ADD32S(accum_vec0_1, mat1_zb_sum_0_x2);
                  accum_vec1_0 = AE_ADD32S(accum_vec1_0, mat1_zb_sum_1_x2);
                  accum_vec1_1 = AE_ADD32S(accum_vec1_1, mat1_zb_sum_1_x2);
          
                  ae_int32x2 output_vec0_0, output_vec0_1, output_vec1_0, output_vec1_1;
#if TFLITE_SINGLE_ROUNDING
                  MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(output_vec0_0, output_vec0_1, accum_vec0_0, accum_vec0_1, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(output_vec1_0, output_vec1_1, accum_vec1_0, accum_vec1_1, out_multiplier, left_shift, right_shift);
#else                   
                  MPY_BY_QUANT_MULT_X2X2_OUT32(output_vec0_0, output_vec0_1, accum_vec0_0, accum_vec0_1, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_X2X2_OUT32(output_vec1_0, output_vec1_1, accum_vec1_0, accum_vec1_1, out_multiplier, left_shift, right_shift);
#endif

                  output_vec0_0 = AE_ADD32S(output_vec0_0, AE_MOVDA32(out_zero_bias));
                  output_vec0_1 = AE_ADD32S(output_vec0_1, AE_MOVDA32(out_zero_bias));
                  output_vec1_0 = AE_ADD32S(output_vec1_0, AE_MOVDA32(out_zero_bias));
                  output_vec1_1 = AE_ADD32S(output_vec1_1, AE_MOVDA32(out_zero_bias));

                  ae_int8x8 output = AE_SAT8X4X32_H(output_vec0_0, output_vec1_0);
                  ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  ae_int8x8 output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  ae_int8x8 output10 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 40));
                  ae_int8x8 output11 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 32));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_1, out_offset);
                  AE_S8_0_XP(output10, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output11, (ae_int8 *)p_dst_1, out_offset);        

                  output   = AE_SAT8X4X32_H(output_vec0_1, output_vec1_1);
                  output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  output10 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 40));
                  output11 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 32));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_2, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_3, out_offset);
                  AE_S8_0_XP(output10, (ae_int8 *)p_dst_2, out_offset);
                  AE_S8_0_XP(output11, (ae_int8 *)p_dst_3, out_offset);
              }
              for(; v_itr < (vec_count); v_itr++)
              {
                  acc_00_hl = acc_10_hl = acc_20_hl = acc_30_hl = 0;
                  WORD8* vec_ptr_0 = (WORD8*)&p_vec1[v_itr * vec_offset];
                  WORD8 *mat_ptr_0 = (WORD8*)&p_mat1[m_itr * row_stride1];
                  WORD8 *mat_ptr_1 = (WORD8*)&p_mat1[(m_itr + 1)* row_stride1];
                  WORD8 *mat_ptr_2 = (WORD8*)&p_mat1[(m_itr + 2)* row_stride1];
                  WORD8 *mat_ptr_3 = (WORD8*)&p_mat1[(m_itr + 3)* row_stride1];

                  WORD32 mat1_zb_sum_0 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_0, cols1);
                  ae_int32x2 mat1_zb_sum_0_x2 = mat1_zb_sum_0;
                  ae_valign align_m0 = AE_LA64_PP(mat_ptr_0);
                  ae_valign align_m1 = AE_LA64_PP(mat_ptr_1);
                  ae_valign align_m2 = AE_LA64_PP(mat_ptr_2);
                  ae_valign align_m3 = AE_LA64_PP(mat_ptr_3);
                  ae_int8x8 vec0;
                  ae_int8x8 mat0, mat1, mat2, mat3;
                  ae_int16x4 vec0_zb_0, vec0_zb_1;
                  ae_int8x8 vzb = AE_MOVDA8(-vec1_zero_bias);
                  for(c_itr=0; c_itr < cols1 >> 3; c_itr++)
                  {
                      AE_LA8X8_IP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0);
                      AE_LA8X8_IP(mat1, align_m1, (ae_int8x8 *)mat_ptr_1);
                      AE_LA8X8_IP(mat2, align_m2, (ae_int8x8 *)mat_ptr_2);
                      AE_LA8X8_IP(mat3, align_m3, (ae_int8x8 *)mat_ptr_3);
                      AE_L8X8_IP(vec0, (ae_int8x8 *)vec_ptr_0, 8);
                      AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                      AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_20_hl, vec0_zb_0, vec0_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_30_hl, vec0_zb_0, vec0_zb_1, mat3);
                  }
                  int rem_elms_shift = 64 - ((cols1 & 7) * 8);
                  if( (cols1&7 ))
                  {
                      AE_LAV8X8_XP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0, (cols1&7));
                      AE_LAV8X8_XP(mat1, align_m0, (ae_int8x8 *)mat_ptr_1, (cols1&7));
                      AE_LAV8X8_XP(mat2, align_m0, (ae_int8x8 *)mat_ptr_2, (cols1&7));
                      AE_LAV8X8_XP(mat3, align_m0, (ae_int8x8 *)mat_ptr_3, (cols1&7));
                      AE_L8X8_IP(vec0, (ae_int8x8 *)vec_ptr_0, 8);
                      vec0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0), rem_elms_shift), rem_elms_shift));
                      AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                      AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                      AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
                      AE_MULAAAA16Q8(acc_20_hl, vec0_zb_0, vec0_zb_1, mat2);
                      AE_MULAAAA16Q8(acc_30_hl, vec0_zb_0, vec0_zb_1, mat3);
                  }
                  if(p_bias != NULL){
                    ae_int32x2 bias0 = AE_MOVDA32X2(p_bias[m_itr], AE_ZERO32());
                    ae_int32x2 bias1 = AE_MOVDA32X2(p_bias[m_itr + 1], AE_ZERO32());
                    ae_int32x2 bias2 = AE_MOVDA32X2(p_bias[m_itr + 2], AE_ZERO32());
                    ae_int32x2 bias3 = AE_MOVDA32X2(p_bias[m_itr + 3], AE_ZERO32());
                    acc_00_hl = AE_ADD32S(acc_00_hl, bias0);
                    acc_10_hl = AE_ADD32S(acc_10_hl, bias1);
                    acc_20_hl = AE_ADD32S(acc_20_hl, bias2);
                    acc_30_hl = AE_ADD32S(acc_30_hl, bias3);                  
                  }
                  ae_int32x2 accum_vec0_0 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_00_hl, acc_00_hl), AE_ADD32_HL_LH(acc_10_hl, acc_10_hl));
                  ae_int32x2 accum_vec0_1 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_20_hl, acc_20_hl), AE_ADD32_HL_LH(acc_30_hl, acc_30_hl));

                  accum_vec0_0 = AE_ADD32S(accum_vec0_0, mat1_zb_sum_0_x2);
                  accum_vec0_1 = AE_ADD32S(accum_vec0_1, mat1_zb_sum_0_x2);
          
                  ae_int32x2 output_vec0_0, output_vec0_1;
#if TFLITE_SINGLE_ROUNDING
                  MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(output_vec0_0, output_vec0_1, accum_vec0_0, accum_vec0_1, out_multiplier, left_shift, right_shift);
#else                   
                  MPY_BY_QUANT_MULT_X2X2_OUT32(output_vec0_0, output_vec0_1, accum_vec0_0, accum_vec0_1, out_multiplier, left_shift, right_shift);
#endif
                  output_vec0_0 = AE_ADD32S(output_vec0_0, AE_MOVDA32(out_zero_bias));
                  output_vec0_1 = AE_ADD32S(output_vec0_1, AE_MOVDA32(out_zero_bias));

                  ae_int8x8 output = AE_SAT8X4X32_H(output_vec0_0, output_vec0_0);
                  ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  ae_int8x8 output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_1, out_offset);     

                  output   = AE_SAT8X4X32_H(output_vec0_1, output_vec0_1);
                  output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_2, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_3, out_offset);
              }              
          }        
      }
    }

    for(; m_itr < (rows &~ 0x01); m_itr+=2)
    {
        ae_int32x2 acc_00_hl, acc_01_hl;
        ae_int32x2 acc_10_hl, acc_11_hl;
        WORD8 *p_dst_0 = p_out + (m_itr * out_stride);
        WORD8 *p_dst_1 = p_out + ((m_itr + 1) * out_stride);
        for(v_itr = 0; v_itr < (vec_count &~ 0x01); v_itr+=2)
        {
            acc_00_hl = acc_01_hl = acc_10_hl = acc_11_hl = 0;
            WORD8* vec_ptr_0 = (WORD8*)&p_vec1[v_itr * vec_offset];
            WORD8* vec_ptr_1 = (WORD8*)&p_vec1[(v_itr + 1) * vec_offset];
            WORD8 *mat_ptr_0 = (WORD8*)&p_mat1[m_itr * row_stride1];
            WORD8 *mat_ptr_1 = (WORD8*)&p_mat1[(m_itr + 1)* row_stride1];                
            WORD32 mat1_zb_sum_0 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_0, cols1);
            WORD32 mat1_zb_sum_1 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_1, cols1);
            ae_int32x2 mat1_zb_sum_0_x2 = mat1_zb_sum_0;
            ae_int32x2 mat1_zb_sum_1_x2 = mat1_zb_sum_1;
            ae_valign align_v0 = AE_LA64_PP(vec_ptr_0);
            ae_valign align_v1 = AE_LA64_PP(vec_ptr_1);
            ae_valign align_m0 = AE_LA64_PP(mat_ptr_0);
            ae_valign align_m1 = AE_LA64_PP(mat_ptr_1);
            ae_int8x8 vec0, vec1, mat0, mat1;
            ae_int16x4 vec0_zb_0, vec0_zb_1;
            ae_int16x4 vec1_zb_0, vec1_zb_1;
            ae_int8x8 vzb = AE_MOVDA8(-vec1_zero_bias);
            for(c_itr=0; c_itr < cols1 >> 3; c_itr++)
            {
                AE_LA8X8_IP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0);
                AE_LA8X8_IP(mat1, align_m1, (ae_int8x8 *)mat_ptr_1);                    
                AE_LA8X8_IP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0);
                AE_LA8X8_IP(vec1, align_v1, (ae_int8x8 *)vec_ptr_1);
                AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                AE_SUBW8(vec1_zb_0, vec1_zb_1, vec1, vzb);
                AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                AE_MULAAAA16Q8(acc_01_hl, vec1_zb_0, vec1_zb_1, mat0);
                AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
                AE_MULAAAA16Q8(acc_11_hl, vec1_zb_0, vec1_zb_1, mat1);
            }
            if( (cols1&7 ))
            {
                AE_LAV8X8_XP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0, (cols1&7));
                AE_LAV8X8_XP(mat1, align_m1, (ae_int8x8 *)mat_ptr_1, (cols1&7));                    
                AE_LAV8X8_XP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0, (cols1&7));
                AE_LAV8X8_XP(vec1, align_v1, (ae_int8x8 *)vec_ptr_1, (cols1&7));
                AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                AE_SUBW8(vec1_zb_0, vec1_zb_1, vec1, vzb);
                AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                AE_MULAAAA16Q8(acc_01_hl, vec1_zb_0, vec1_zb_1, mat0);
                AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
                AE_MULAAAA16Q8(acc_11_hl, vec1_zb_0, vec1_zb_1, mat1);                   
            }
            if(p_bias != NULL){
              ae_int32x2 bias0 = AE_MOVDA32X2(p_bias[m_itr], AE_ZERO32());
              ae_int32x2 bias1 = AE_MOVDA32X2(p_bias[m_itr + 1], AE_ZERO32());
              acc_00_hl = AE_ADD32S(acc_00_hl, bias0);
              acc_01_hl = AE_ADD32S(acc_01_hl, bias0);
              acc_10_hl = AE_ADD32S(acc_10_hl, bias1);
              acc_11_hl = AE_ADD32S(acc_11_hl, bias1);
            }
            ae_int32x2 accum_vec0 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_00_hl, acc_00_hl), AE_ADD32_HL_LH(acc_10_hl, acc_10_hl));
            ae_int32x2 accum_vec1 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_01_hl, acc_01_hl), AE_ADD32_HL_LH(acc_11_hl, acc_11_hl));
            accum_vec0 = AE_ADD32S(accum_vec0, mat1_zb_sum_0_x2);
            accum_vec1 = AE_ADD32S(accum_vec1, mat1_zb_sum_1_x2);
    
            ae_int32x2 output_vec0, output_vec1;
#if TFLITE_SINGLE_ROUNDING
            MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI1S(output_vec0, output_vec1, accum_vec0, accum_vec1, out_multiplier, left_shift, right_shift);
#else             
            MPY_BY_QUANT_MULT_X2X2_OUT32(output_vec0, output_vec1, accum_vec0, accum_vec1, out_multiplier, left_shift, right_shift);
#endif            
            output_vec0 = AE_ADD32S(output_vec0, AE_MOVDA32(out_zero_bias));
            output_vec1 = AE_ADD32S(output_vec1, AE_MOVDA32(out_zero_bias));

            ae_int8x8 output = AE_SAT8X4X32_H(output_vec0, output_vec1);
            ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
            ae_int8x8 output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
            ae_int8x8 output10 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 40));
            ae_int8x8 output11 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 32));

            AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
            AE_S8_0_XP(output01, (ae_int8 *)p_dst_1, out_offset);
            AE_S8_0_XP(output10, (ae_int8 *)p_dst_0, out_offset);
            AE_S8_0_XP(output11, (ae_int8 *)p_dst_1, out_offset);             
        }
        for(; v_itr < (vec_count); v_itr++)
        {
            acc_00_hl = acc_10_hl = 0;
            WORD8* vec_ptr_0 = (WORD8*)&p_vec1[v_itr * vec_offset];
            WORD8 *mat_ptr_0 = (WORD8*)&p_mat1[m_itr * row_stride1];
            WORD8 *mat_ptr_1 = (WORD8*)&p_mat1[(m_itr + 1)* row_stride1];                
            WORD32 mat1_zb_sum_0 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_0, cols1);
            ae_int32x2 mat1_zb_sum_0_x2 = mat1_zb_sum_0;
            ae_valign align_v0 = AE_LA64_PP(vec_ptr_0);
            ae_valign align_m0 = AE_LA64_PP(mat_ptr_0);
            ae_valign align_m1 = AE_LA64_PP(mat_ptr_1);
            ae_int8x8 vec0, mat0, mat1;
            ae_int16x4 vec0_zb_0, vec0_zb_1;
            ae_int8x8 vzb = AE_MOVDA8(-vec1_zero_bias);
            for(c_itr=0; c_itr < cols1 >> 3; c_itr++)
            {
                AE_LA8X8_IP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0);
                AE_LA8X8_IP(mat1, align_m1, (ae_int8x8 *)mat_ptr_1);                    
                AE_LA8X8_IP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0);
                AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
            }
            if( (cols1&7 ))
            {
                AE_LAV8X8_XP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0, (cols1&7));
                AE_LAV8X8_XP(mat1, align_m1, (ae_int8x8 *)mat_ptr_1, (cols1&7));                    
                AE_LAV8X8_XP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0, (cols1&7));
                AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                AE_MULAAAA16Q8(acc_10_hl, vec0_zb_0, vec0_zb_1, mat1);
            }
            if(p_bias != NULL){
              ae_int32x2 bias0 = AE_MOVDA32X2(p_bias[m_itr], AE_ZERO32());
              ae_int32x2 bias1 = AE_MOVDA32X2(p_bias[m_itr + 1], AE_ZERO32());
              acc_00_hl = AE_ADD32S(acc_00_hl, bias0);
              acc_10_hl = AE_ADD32S(acc_10_hl, bias1);
            }
            ae_int32x2 accum_vec0 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_00_hl, acc_00_hl), AE_ADD32_HL_LH(acc_10_hl, acc_10_hl));
            accum_vec0 = AE_ADD32S(accum_vec0, mat1_zb_sum_0_x2);
    
            ae_int32x2 output_vec0;
#if TFLITE_SINGLE_ROUNDING
            MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(output_vec0, accum_vec0, out_multiplier, left_shift, right_shift);
#else            
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec0, accum_vec0, out_multiplier, left_shift, right_shift);
#endif            
            output_vec0 = AE_ADD32S(output_vec0, AE_MOVDA32(out_zero_bias));

            ae_int8x8 output = AE_SAT8X4X32_H(output_vec0, output_vec0);
            ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
            ae_int8x8 output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));

            AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
            AE_S8_0_XP(output01, (ae_int8 *)p_dst_1, out_offset);         
        }
    }
    for(; m_itr < rows; m_itr++)
    {
        ae_int32x2 acc_00_hl, acc_01_hl;
        WORD8 *p_dst_0 = p_out + (m_itr * out_stride);
        v_itr = 0;
        for(; v_itr < (vec_count &~ 0x01); v_itr+=2)
        {
            acc_00_hl = acc_01_hl = 0;
            WORD8* vec_ptr_0 = (WORD8*)&p_vec1[v_itr * vec_offset];
            WORD8* vec_ptr_1 = (WORD8*)&p_vec1[(v_itr + 1) * vec_offset];
            WORD8 *mat_ptr_0 = (WORD8*)&p_mat1[m_itr * row_stride1];              
            WORD32 mat1_zb_sum_0 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_0, cols1);
            WORD32 mat1_zb_sum_1 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_1, cols1);
            ae_int32x2 mat1_zb_sum_x2 = AE_MOVDA32X2(mat1_zb_sum_0, mat1_zb_sum_1);
            ae_valign align_v0 = AE_LA64_PP(vec_ptr_0);
            ae_valign align_v1 = AE_LA64_PP(vec_ptr_1);
            ae_valign align_m0 = AE_LA64_PP(mat_ptr_0);
            ae_int8x8 vec0, vec1, mat0;
            ae_int16x4 vec0_zb_0, vec0_zb_1;
            ae_int16x4 vec1_zb_0, vec1_zb_1;
            ae_int8x8 vzb = AE_MOVDA8(-vec1_zero_bias);
            for(c_itr=0; c_itr < cols1 >> 3; c_itr++)
            {
                AE_LA8X8_IP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0);
                AE_LA8X8_IP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0);
                AE_LA8X8_IP(vec1, align_v1, (ae_int8x8 *)vec_ptr_1);
                AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                AE_SUBW8(vec1_zb_0, vec1_zb_1, vec1, vzb);
                AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                AE_MULAAAA16Q8(acc_01_hl, vec1_zb_0, vec1_zb_1, mat0);
            }
            if( (cols1&7 ))
            {
                AE_LAV8X8_XP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0, (cols1&7));
                AE_LAV8X8_XP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0, (cols1&7));
                AE_LAV8X8_XP(vec1, align_v1, (ae_int8x8 *)vec_ptr_1, (cols1&7));
                AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                AE_SUBW8(vec1_zb_0, vec1_zb_1, vec1, vzb);
                AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
                AE_MULAAAA16Q8(acc_01_hl, vec1_zb_0, vec1_zb_1, mat0);                 
            }
            if(p_bias != NULL){
              ae_int32x2 bias0 = AE_MOVDA32X2(p_bias[m_itr], AE_ZERO32());
              acc_00_hl = AE_ADD32S(acc_00_hl, bias0);
              acc_01_hl = AE_ADD32S(acc_01_hl, bias0);
            }
            ae_int32x2 accum_row0 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_00_hl, acc_00_hl), AE_ADD32_HL_LH(acc_01_hl, acc_01_hl));
            accum_row0 = AE_ADD32S(accum_row0, mat1_zb_sum_x2);
      
            ae_int32x2 output_row0;
#if TFLITE_SINGLE_ROUNDING
            MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(output_row0, accum_row0, out_multiplier, left_shift, right_shift);
#else
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_row0, accum_row0, out_multiplier, left_shift, right_shift);
#endif            
            output_row0 = AE_ADD32S(output_row0, AE_MOVDA32(out_zero_bias));

            ae_int8x8 output = AE_SAT8X4X32_H(output_row0, output_row0);
            ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
            ae_int8x8 output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));

            AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
            AE_S8_0_XP(output01, (ae_int8 *)p_dst_0, out_offset);
        }
        for(; v_itr < (vec_count); v_itr++)
        {
            acc_00_hl = 0;
            WORD8* vec_ptr_0 = (WORD8*)&p_vec1[v_itr * vec_offset];
            WORD8 *mat_ptr_0 = (WORD8*)&p_mat1[m_itr * row_stride1];              
            WORD32 mat1_zb_sum_0 =  internal_calc_mzbsum(mat1_zero_bias, vec1_zero_bias, vec_ptr_0, cols1);
            ae_int32x2 mat1_zb_sum_x2 = AE_MOVDA32(mat1_zb_sum_0);
            ae_valign align_v0 = AE_LA64_PP(vec_ptr_0);
            ae_valign align_m0 = AE_LA64_PP(mat_ptr_0);
            ae_int8x8 vec0, mat0;
            ae_int16x4 vec0_zb_0, vec0_zb_1;
            ae_int8x8 vzb = AE_MOVDA8(-vec1_zero_bias);
            for(c_itr=0; c_itr < cols1 >> 3; c_itr++)
            {
                AE_LA8X8_IP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0);
                AE_LA8X8_IP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0);
                AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
            }
            if( (cols1&7 ))
            {
                AE_LAV8X8_XP(mat0, align_m0, (ae_int8x8 *)mat_ptr_0, (cols1&7));
                AE_LAV8X8_XP(vec0, align_v0, (ae_int8x8 *)vec_ptr_0, (cols1&7));
                AE_SUBW8(vec0_zb_0, vec0_zb_1, vec0, vzb);
                AE_MULAAAA16Q8(acc_00_hl, vec0_zb_0, vec0_zb_1, mat0);
            }
            if(p_bias != NULL){
              ae_int32x2 bias0 = AE_MOVDA32X2(p_bias[m_itr], AE_ZERO32());
              acc_00_hl = AE_ADD32S(acc_00_hl, bias0);
            }
            ae_int32x2 accum_row0 = AE_SEL32_HH(AE_ADD32_HL_LH(acc_00_hl, acc_00_hl), AE_ADD32_HL_LH(acc_00_hl, acc_00_hl));
            accum_row0 = AE_ADD32S(accum_row0, mat1_zb_sum_x2);
      
            ae_int32x2 output_row0;
#if TFLITE_SINGLE_ROUNDING
            MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(output_row0, accum_row0, out_multiplier, left_shift, right_shift);
#else            
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_row0, accum_row0, out_multiplier, left_shift, right_shift);
#endif            
            output_row0 = AE_ADD32S(output_row0, AE_MOVDA32(out_zero_bias));

            ae_int8x8 output = AE_SAT8X4X32_H(output_row0, output_row0);
            ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));

            AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
        }
    }
    return 0;
}
#else // XCHAL_HAVE_HIFI1S

#ifdef ROW_UNROLL
#undef ROW_UNROLL
#endif
#define ROW_UNROLL  4

#include "xa_nnlib_common_macros.h"

/*----------------------------Main function---------------------------------*/
WORD32 xa_nn_matmul_asym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
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
    XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
    XA_NNLIB_ARG_CHK_COND((vec_count <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
    XA_NNLIB_ARG_CHK_COND((mat1_zero_bias < -127 || mat1_zero_bias > 128), -1);
    XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
    XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
    XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

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
    #define UNROLL_LOAD_VEC_BATCH                   LOAD_VEC_BATCH_ASYM8bs
    #define UNROLL_LOAD_ROW_MAT1                    LOAD_ROW_MAT1_ASYM8bs
    #define LOAD_BIAS                               LOAD_BIAS_ASYM8b_MATMUL
    #define UNROLL_ROW_KERNEL_MAT1_VEC_BATCH        KERNEL_MAT1_VEC_BATCH_ROW_ASYM8b_ASYM8b
    #define UNROLL_KERNEL_MAT1_VEC_BATCH            KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b
    #define UNROLL_ROW_ADD_BIAS_ACC                 ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL
    #define UNROLL_ADD_BIAS_ACC_BATCH               ADD_BIAS_BATCH_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL
    #define UNROLL_ROW_ADJUST_ACC                   ADJUST_ACC_BATCH_ROW_ASYM8b
    #define UNROLL_ADJUST_ACC_BATCH                 ADJUST_ACC_BATCH_ASYM8b
    #define UNROLL_ROW_STORE_ACC                    STORE_ACC_BATCH_ROW_ASYM8bxASYM8b_AT_OUT_ASYM8b
    #define UNROLL_STORE_ACC_BATCH                  STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs

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
                    LOAD_VEC_BATCH_ASYM8bs_UNALIGNED(0);
                    LOAD_VEC_BATCH_ASYM8bs_UNALIGNED(1);
                    LOAD_ROW_MAT1_ASYM8bs_UNALIGNED(0);
                    LOAD_ROW_MAT1_ASYM8bs_UNALIGNED(1);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(1,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,1);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(1,1);
                }
                #pragma no_unroll
                for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                {
                    LOAD_VEC_BATCH_ASYM8bs_SINGLE_UNALIGNED(0);
                    LOAD_VEC_BATCH_ASYM8bs_SINGLE_UNALIGNED(1);
                    LOAD_ROW_MAT1_ASYM8bs_SINGLE_UNALIGNED(0);
                    LOAD_ROW_MAT1_ASYM8bs_SINGLE_UNALIGNED(1);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(1,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,1);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(1,1);
                }

                ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL(0);
                ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL(1);
                ADJUST_ACC_BATCH_ROW_ASYM8b(0);
                ADJUST_ACC_BATCH_ROW_ASYM8b(1);
                STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(0,0);
                STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(1,0);
                STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(0,1);
                STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(1,1);
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
                    LOAD_VEC_BATCH_ASYM8bs_UNALIGNED(0);
                    LOAD_VEC_BATCH_ASYM8bs_UNALIGNED(1);
                    LOAD_ROW_MAT1_ASYM8bs_UNALIGNED(0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,1);
                }
                #pragma no_unroll
                for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                {
                    LOAD_VEC_BATCH_ASYM8bs_SINGLE_UNALIGNED(0);
                    LOAD_VEC_BATCH_ASYM8bs_SINGLE_UNALIGNED(1);
                    LOAD_ROW_MAT1_ASYM8bs_SINGLE_UNALIGNED(0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,0);
                    KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,1);
                }
                ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL(0);
                ADJUST_ACC_BATCH_ROW_ASYM8b(0);
                STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(0,0);
                STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(0,1);
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
                        LOAD_VEC_BATCH_ASYM8bs_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8bs_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8bs_UNALIGNED(1);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,0);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(1,0);
                    }
                #pragma no_unroll
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_VEC_BATCH_ASYM8bs_SINGLE_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8bs_SINGLE_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8bs_SINGLE_UNALIGNED(1);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,0);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(1,0);
                    }  

                    LOAD_BIAS; 
                    UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                    UNROLL_ADJUST_ACC_BATCH(0,0);
                    LOAD_BIAS; 
                    UNROLL_ADD_BIAS_ACC_BATCH(1,0);
                    UNROLL_ADJUST_ACC_BATCH(1,0);
                
                    STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(0,0);
                    STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(1,0);
                }

                for(; m_itr < rows; m_itr++)
                {
                    UNROLL_SETUP_ACC_BATCH(0,0);
                    SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(0);
                    SETUP_MAT1_ASYM8b_UNALIGNED(0);
                    int cols1_count = cols1 - cols1%4;

                    for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
                    {
                        LOAD_VEC_BATCH_ASYM8bs_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8bs_UNALIGNED(0);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0,0);
                    }
                #pragma no_unroll
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_VEC_BATCH_ASYM8bs_SINGLE_UNALIGNED(0);
                        LOAD_ROW_MAT1_ASYM8bs_SINGLE_UNALIGNED(0);
                        KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0,0);
                    }

                    LOAD_BIAS;
                    UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                    UNROLL_ADJUST_ACC_BATCH(0,0);
                    STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(0,0);
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
#endif // XCHAL_HAVE_HIFI1S
