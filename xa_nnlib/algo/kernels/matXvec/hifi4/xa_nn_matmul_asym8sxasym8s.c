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
#include "xa_nnlib_common.h"

#if XCHAL_HAVE_HIFI1S
#include "xa_nnlib_common_macros.h"
static WORD32 internal_calc_mzbsum(WORD32 mat_zero_bias, WORD32 vec_zero_bias, const WORD8 * __restrict__ p_vec, int cols1)
{
  if(mat_zero_bias == 0){
    return 0;
  }

  WORD32 c_itr;
  ae_int64 sum_mzb = (ae_int64)0;
  ae_int16x4 mzb_16x4 = AE_MOVDA16(mat_zero_bias);
  ae_int8x8 vzb_8x8 = AE_MOVDA8(-vec_zero_bias);
  ae_int8x8 d_vec0;
  ae_int16x4 vec0_zb_0, vec0_zb_1;

  ae_valign align_vec = AE_LA64_PP(p_vec);
  for(c_itr = 0; c_itr < cols1>>3; c_itr++){
    AE_LA8X8_IP(d_vec0, align_vec, (ae_int8x8 *)p_vec);
    AE_SUBW8(vec0_zb_0, vec0_zb_1, d_vec0, vzb_8x8);
    AE_MULAAAAQ16(sum_mzb, mzb_16x4, vec0_zb_0);
    AE_MULAAAAQ16(sum_mzb, mzb_16x4, vec0_zb_1);
  }
  WORD32 sum_mzb32 = AE_MOVINT32X2_FROMINT64(sum_mzb);

  for(c_itr = 0; c_itr < (cols1&0x7); c_itr++){
    int vecval = *p_vec++;
    sum_mzb32 += (vecval+vec_zero_bias)*mat_zero_bias;
  }
  return sum_mzb32;
}

WORD32 xa_nn_matmul_v2_asym8sxasym8s_asym8s(
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
    WORD32 out_zero_bias,
    WORD32 out_activation_min,
    WORD32 out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
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
    /* MinMax activation range check */
    XA_NNLIB_ARG_CHK_COND((out_activation_min < -128 || out_activation_min > 127), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < -128 || out_activation_max > 127), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

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
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec0_0, accum_vec0_0, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec0_1, accum_vec0_1, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec1_0, accum_vec1_0, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec1_1, accum_vec1_1, out_multiplier, left_shift, right_shift);
#endif
                  output_vec0_0 = AE_ADD32S(output_vec0_0, AE_MOVDA32(out_zero_bias));
                  output_vec0_1 = AE_ADD32S(output_vec0_1, AE_MOVDA32(out_zero_bias));
                  output_vec1_0 = AE_ADD32S(output_vec1_0, AE_MOVDA32(out_zero_bias));
                  output_vec1_1 = AE_ADD32S(output_vec1_1, AE_MOVDA32(out_zero_bias));

                  ae_int8x8 output = AE_SAT8X4X32_H(output_vec0_0, output_vec1_0);
                  output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
                  ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  ae_int8x8 output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  ae_int8x8 output10 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 40));
                  ae_int8x8 output11 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 32));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_1, out_offset);
                  AE_S8_0_XP(output10, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output11, (ae_int8 *)p_dst_1, out_offset);        

                  output   = AE_SAT8X4X32_H(output_vec0_1, output_vec1_1);
                  output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
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
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec0_0, accum_vec0_0, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec0_1, accum_vec0_1, out_multiplier, left_shift, right_shift);
#endif                  

                  output_vec0_0 = AE_ADD32S(output_vec0_0, AE_MOVDA32(out_zero_bias));
                  output_vec0_1 = AE_ADD32S(output_vec0_1, AE_MOVDA32(out_zero_bias));

                  ae_int8x8 output = AE_SAT8X4X32_H(output_vec0_0, output_vec0_0);
                  output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
                  ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  ae_int8x8 output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_1, out_offset);     

                  output   = AE_SAT8X4X32_H(output_vec0_1, output_vec0_1);
                  output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
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
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec0_0, accum_vec0_0, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec0_1, accum_vec0_1, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec1_0, accum_vec1_0, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec1_1, accum_vec1_1, out_multiplier, left_shift, right_shift);
#endif

                  output_vec0_0 = AE_ADD32S(output_vec0_0, AE_MOVDA32(out_zero_bias));
                  output_vec0_1 = AE_ADD32S(output_vec0_1, AE_MOVDA32(out_zero_bias));
                  output_vec1_0 = AE_ADD32S(output_vec1_0, AE_MOVDA32(out_zero_bias));
                  output_vec1_1 = AE_ADD32S(output_vec1_1, AE_MOVDA32(out_zero_bias));

                  ae_int8x8 output = AE_SAT8X4X32_H(output_vec0_0, output_vec1_0);
                  output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
                  ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  ae_int8x8 output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  ae_int8x8 output10 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 40));
                  ae_int8x8 output11 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 32));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_1, out_offset);
                  AE_S8_0_XP(output10, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output11, (ae_int8 *)p_dst_1, out_offset);        

                  output   = AE_SAT8X4X32_H(output_vec0_1, output_vec1_1);
                  output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
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
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec0_0, accum_vec0_0, out_multiplier, left_shift, right_shift);
                  MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec0_1, accum_vec0_1, out_multiplier, left_shift, right_shift);
#endif
                  output_vec0_0 = AE_ADD32S(output_vec0_0, AE_MOVDA32(out_zero_bias));
                  output_vec0_1 = AE_ADD32S(output_vec0_1, AE_MOVDA32(out_zero_bias));

                  ae_int8x8 output = AE_SAT8X4X32_H(output_vec0_0, output_vec0_0);
                  output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
                  ae_int8x8 output00 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 56));
                  ae_int8x8 output01 = AE_MOVINT8X8_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT8X8(output), 48));
                  AE_S8_0_XP(output00, (ae_int8 *)p_dst_0, out_offset);
                  AE_S8_0_XP(output01, (ae_int8 *)p_dst_1, out_offset);     

                  output   = AE_SAT8X4X32_H(output_vec0_1, output_vec0_1);
                  output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
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
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec0, accum_vec0, out_multiplier, left_shift, right_shift);
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(output_vec1, accum_vec1, out_multiplier, left_shift, right_shift);
#endif            
            output_vec0 = AE_ADD32S(output_vec0, AE_MOVDA32(out_zero_bias));
            output_vec1 = AE_ADD32S(output_vec1, AE_MOVDA32(out_zero_bias));

            ae_int8x8 output = AE_SAT8X4X32_H(output_vec0, output_vec1);
            output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
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
            output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
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
            output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
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
            output = AE_MIN8(AE_MAX8(output, AE_MOVDA8(out_activation_min)), AE_MOVDA8(out_activation_max));
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
static WORD32 internal_calc_mzbsum_align(WORD32 mat_zero_bias, WORD32 vec_zero_bias, const WORD8 * __restrict__ p_vec, int cols1)
{
/* p_vec is aligned to 4-byte boundary, cols1 is multiple of 4*/
  if(mat_zero_bias == 0){
    return 0;
  }
  WORD32 sum_mzb32 = cols1*mat_zero_bias*vec_zero_bias, c_itr;
  
  ae_int64 sum_mzb = (ae_int64)sum_mzb32;
  ae_int16x4 mzb_16x4 = AE_MOVDA16(mat_zero_bias);
  ae_int16x4 d_vec0;

  for(c_itr = 0; c_itr < cols1>>2; c_itr++){
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(d_vec0, p_vec, 4);
#else    
    AE_L8X4F_IP(d_vec0, p_vec, 4);
    d_vec0 = AE_SRAI16(d_vec0, 8);
#endif
    AE_MULAAAAQ16(sum_mzb, mzb_16x4, d_vec0);
  }
  sum_mzb32 = AE_MOVINT32X2_FROMINT64(sum_mzb);
  return (sum_mzb32);
}


/*----------------------------Main function---------------------------------*/
WORD32 xa_nn_matmul_v2_asym8sxasym8s_asym8s(
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
    WORD32 out_zero_bias,
    WORD32 out_activation_min,
    WORD32 out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
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
    /* MinMax activation range check */
    XA_NNLIB_ARG_CHK_COND((out_activation_min < -128 || out_activation_min > 127), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < -128 || out_activation_max > 127), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
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

    int chk_align = 0;  
    CHK_MATMUL_ALIGN(chk_align, p_mat1, (ALIGNMENT>>1), p_vec1, (ALIGNMENT>>1), cols1, row_stride1, vec_offset, 4);
    if(chk_align)
    {
        for(vec_itr = 0; vec_itr < (vec_count & ~(2 -1)); vec_itr+=2)
        {
            WORD32 *p_bias32 = (WORD32 *) p_bias;
            WORD32 mat1_zb_sum_0 =  internal_calc_mzbsum_align(mat1_zero_bias, vec1_zero_bias, (WORD8 *)(p_vec1 + (vec_itr + 0)*vec_offset), cols1);
            WORD32 mat1_zb_sum_1 =  internal_calc_mzbsum_align(mat1_zero_bias, vec1_zero_bias, (WORD8 *)(p_vec1 + (vec_itr + 1)*vec_offset), cols1);
            ae_int32x2 mat1_zb_01 = AE_MOVDA32X2(mat1_zb_sum_0, mat1_zb_sum_1);

            for(m_itr = 0; m_itr < (rows & ~(4 -1)); m_itr += 4)
            {
                ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_0_1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_1_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_1_1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_2_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_2_1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_3_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_3_1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));

                ae_int16x4 vec_batch_0, vec_batch_1;
                WORD8 *p_vec_batch_0 = (WORD8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                WORD8 *p_vec_batch_1 = (WORD8 *)(p_vec1 + (vec_itr + 1)*vec_offset);

                ae_int16x4 mat1_0, mat1_1, mat1_2, mat1_3; 
                WORD8 *p_mat1_0 = (WORD8 *) &p_mat1[(m_itr+0)*row_stride1];
                WORD8 *p_mat1_1 = (WORD8 *) &p_mat1[(m_itr+1)*row_stride1];
                WORD8 *p_mat1_2 = (WORD8 *) &p_mat1[(m_itr+2)*row_stride1];
                WORD8 *p_mat1_3 = (WORD8 *) &p_mat1[(m_itr+3)*row_stride1];

#if XCHAL_HAVE_HIFI1
                for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                {
                    AE_L8X4S_IP(vec_batch_0, p_vec_batch_0, (1 * 4));
                    AE_L8X4S_IP(vec_batch_1, p_vec_batch_1, (1 * 4));
                    vec_batch_0 = AE_ADD16(vec_batch_0, AE_MOVDA16(vec1_zero_bias));
                    vec_batch_1 = AE_ADD16(vec_batch_1, AE_MOVDA16(vec1_zero_bias));
                    AE_L8X4S_IP(mat1_0, p_mat1_0, (1 * 4));
                    AE_L8X4S_IP(mat1_1, p_mat1_1, (1 * 4));
                    AE_L8X4S_IP(mat1_2, p_mat1_2, (1 * 4));
                    AE_L8X4S_IP(mat1_3, p_mat1_3, (1 * 4));
                    AE_MULAAAAQ16(acc_0_0, vec_batch_0, mat1_0);
                    AE_MULAAAAQ16(acc_0_1, vec_batch_1, mat1_0);
                    AE_MULAAAAQ16(acc_1_0, vec_batch_0, mat1_1);
                    AE_MULAAAAQ16(acc_1_1, vec_batch_1, mat1_1);
                    AE_MULAAAAQ16(acc_2_0, vec_batch_0, mat1_2);
                    AE_MULAAAAQ16(acc_2_1, vec_batch_1, mat1_2);
                    AE_MULAAAAQ16(acc_3_0, vec_batch_0, mat1_3);
                    AE_MULAAAAQ16(acc_3_1, vec_batch_1, mat1_3);
                }
#else
                for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                {                   
                    AE_L8X4F_IP(vec_batch_0, p_vec_batch_0, (1 * 4));
                    vec_batch_0 = AE_SRAI16(vec_batch_0, 8);
                    vec_batch_0 = AE_ADD16(vec_batch_0, AE_MOVDA16(vec1_zero_bias));
                    AE_L8X4F_IP(vec_batch_1, p_vec_batch_1, (1 * 4));
                    vec_batch_1 = AE_SRAI16(vec_batch_1, 8);
                    vec_batch_1 = AE_ADD16(vec_batch_1, AE_MOVDA16(vec1_zero_bias));

                    AE_L8X4F_IP(mat1_0, p_mat1_0, (1 * 4));
                    AE_L8X4F_IP(mat1_1, p_mat1_1, (1 * 4));
                    AE_L8X4F_IP(mat1_2, p_mat1_2, (1 * 4));
                    AE_L8X4F_IP(mat1_3, p_mat1_3, (1 * 4));

                    AE_MULAAAAQ16(acc_0_0, vec_batch_0, mat1_0);
                    AE_MULAAAAQ16(acc_0_1, vec_batch_1, mat1_0);
                    AE_MULAAAAQ16(acc_1_0, vec_batch_0, mat1_1);
                    AE_MULAAAAQ16(acc_1_1, vec_batch_1, mat1_1);
                    AE_MULAAAAQ16(acc_2_0, vec_batch_0, mat1_2);
                    AE_MULAAAAQ16(acc_2_1, vec_batch_1, mat1_2);
                    AE_MULAAAAQ16(acc_3_0, vec_batch_0, mat1_3);
                    AE_MULAAAAQ16(acc_3_1, vec_batch_1, mat1_3);
                }

                acc_0_0 = AE_SRAI64(acc_0_0, 8);
                acc_0_1 = AE_SRAI64(acc_0_1, 8);
                acc_1_0 = AE_SRAI64(acc_1_0, 8);
                acc_1_1 = AE_SRAI64(acc_1_1, 8);
                acc_2_0 = AE_SRAI64(acc_2_0, 8);
                acc_2_1 = AE_SRAI64(acc_2_1, 8);
                acc_3_0 = AE_SRAI64(acc_3_0, 8);
                acc_3_1 = AE_SRAI64(acc_3_1, 8);
#endif
                ae_int32x2 acc_32x2_0, acc_32x2_1, acc_32x2_2, acc_32x2_3;
    
                acc_32x2_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc_0_0), AE_MOVINT32X2_FROMINT64(acc_0_1));
                acc_32x2_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc_1_0), AE_MOVINT32X2_FROMINT64(acc_1_1));
                acc_32x2_2 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc_2_0), AE_MOVINT32X2_FROMINT64(acc_2_1));
                acc_32x2_3 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc_3_0), AE_MOVINT32X2_FROMINT64(acc_3_1));

                acc_32x2_0 = AE_ADD32(acc_32x2_0, mat1_zb_01);
                acc_32x2_1 = AE_ADD32(acc_32x2_1, mat1_zb_01);
                acc_32x2_2 = AE_ADD32(acc_32x2_2, mat1_zb_01);
                acc_32x2_3 = AE_ADD32(acc_32x2_3, mat1_zb_01);

                if(p_bias != (void *)0){
                    WORD32 bias0 = *p_bias32++;
                    WORD32 bias1 = *p_bias32++;
                    WORD32 bias2 = *p_bias32++;
                    WORD32 bias3 = *p_bias32++;
                    acc_32x2_0 = AE_ADD32(acc_32x2_0, AE_MOVDA32(bias0)); 
                    acc_32x2_1 = AE_ADD32(acc_32x2_1, AE_MOVDA32(bias1)); 
                    acc_32x2_2 = AE_ADD32(acc_32x2_2, AE_MOVDA32(bias2)); 
                    acc_32x2_3 = AE_ADD32(acc_32x2_3, AE_MOVDA32(bias3)); 
                }

                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_0, acc_32x2_0, out_multiplier, left_shift, right_shift);
                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_1, acc_32x2_1, out_multiplier, left_shift, right_shift);
                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_2, acc_32x2_2, out_multiplier, left_shift, right_shift);
                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_3, acc_32x2_3, out_multiplier, left_shift, right_shift);

                acc_32x2_0 = AE_ADD32S(acc_32x2_0, AE_MOVDA32(out_zero_bias));
                acc_32x2_1 = AE_ADD32S(acc_32x2_1, AE_MOVDA32(out_zero_bias));
                acc_32x2_2 = AE_ADD32S(acc_32x2_2, AE_MOVDA32(out_zero_bias));
                acc_32x2_3 = AE_ADD32S(acc_32x2_3, AE_MOVDA32(out_zero_bias));                

                acc_32x2_0 = AE_MIN32(AE_MAX32(acc_32x2_0, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max)); 
                acc_32x2_1 = AE_MIN32(AE_MAX32(acc_32x2_1, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max)); 
                acc_32x2_2 = AE_MIN32(AE_MAX32(acc_32x2_2, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max)); 
                acc_32x2_3 = AE_MIN32(AE_MAX32(acc_32x2_3, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max)); 

                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_H(acc_32x2_0);
                (*((WORD8 *) p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_0);
                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride)) = (WORD8)AE_MOVAD32_H(acc_32x2_1); 
                (*((WORD8 *) p_out + (vec_itr + 1)*out_offset + (m_itr + 1)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_1);
                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 2)*out_stride)) = (WORD8)AE_MOVAD32_H(acc_32x2_2); 
                (*((WORD8 *) p_out + (vec_itr + 1)*out_offset + (m_itr + 2)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_2);
                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 3)*out_stride)) = (WORD8)AE_MOVAD32_H(acc_32x2_3); 
                (*((WORD8 *) p_out + (vec_itr + 1)*out_offset + (m_itr + 3)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_3);
            }

            for(; m_itr < rows; m_itr++)
            {
                ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_0_1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int16x4 vec_batch_0 = AE_MOVDA16(0);
                WORD8 *p_vec_batch_0 = (WORD8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int16x4 vec_batch_1 = AE_MOVDA16(0);
                WORD8 *p_vec_batch_1 = (WORD8 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                ae_int16x4 mat1_0 = AE_MOVDA16(0);
                WORD8 *p_mat1_0 = (WORD8 *) &p_mat1[(m_itr+0)*row_stride1];

#if XCHAL_HAVE_HIFI1
                for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                {
                    AE_L8X4S_IP(vec_batch_0, p_vec_batch_0, (1 * 4));
                    AE_L8X4S_IP(vec_batch_1, p_vec_batch_1, (1 * 4));
                    vec_batch_0 = AE_ADD16(vec_batch_0, AE_MOVDA16(vec1_zero_bias));
                    vec_batch_1 = AE_ADD16(vec_batch_1, AE_MOVDA16(vec1_zero_bias));
                    AE_L8X4S_IP(mat1_0, p_mat1_0, (1 * 4));                 
                    AE_MULAAAAQ16(acc_0_0, vec_batch_0, mat1_0);
                    AE_MULAAAAQ16(acc_0_1, vec_batch_1, mat1_0);
                }
#else
                for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                {
                    AE_L8X4F_IP(vec_batch_0, p_vec_batch_0, (1 * 4));
                    vec_batch_0 = AE_SRAI16(vec_batch_0, 8);
                    vec_batch_0 = AE_ADD16(vec_batch_0, AE_MOVDA16(vec1_zero_bias));
                    AE_L8X4F_IP(vec_batch_1, p_vec_batch_1, (1 * 4));
                    vec_batch_1 = AE_SRAI16(vec_batch_1, 8);
                    vec_batch_1 = AE_ADD16(vec_batch_1, AE_MOVDA16(vec1_zero_bias));
                    AE_L8X4F_IP(mat1_0, p_mat1_0, (1 * 4));
                    AE_MULAAAAQ16(acc_0_0, vec_batch_0, mat1_0);
                    AE_MULAAAAQ16(acc_0_1, vec_batch_1, mat1_0);
                }

                acc_0_0 = AE_SRAI64(acc_0_0, 8);
                acc_0_1 = AE_SRAI64(acc_0_1, 8);
#endif
                ae_int32x2 acc_32x2_0;
                acc_32x2_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc_0_0), AE_MOVINT32X2_FROMINT64(acc_0_1));

                acc_32x2_0 = AE_ADD32(acc_32x2_0, mat1_zb_01);
                if(p_bias!=(void *)0) {
                    WORD32 bias0 = *p_bias32++;
                    acc_32x2_0 = AE_ADD32(acc_32x2_0, AE_MOVDA32(bias0)); 
                }

                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_0, acc_32x2_0, out_multiplier, left_shift, right_shift);
                acc_32x2_0 = AE_ADD32S(acc_32x2_0, AE_MOVDA32(out_zero_bias));
                acc_32x2_0 = AE_MIN32(AE_MAX32(acc_32x2_0, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max));
                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_H(acc_32x2_0);
                (*((WORD8 *) p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_0);
            }
        }

        for(; vec_itr < vec_count; vec_itr++)
        {
            WORD32 *p_bias32 = (WORD32 *) p_bias;
            for(m_itr = 0; m_itr < (rows & ~(4 -1)); m_itr += 4)
            {
                ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_1_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_2_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_3_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int16x4 vec_batch_0 = AE_MOVDA16(0);
                WORD8 *p_vec_batch_0 = (WORD8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int16x4 mat1_0 = AE_MOVDA16(0);
                WORD8 *p_mat1_0 = (WORD8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_int16x4 mat1_1 = AE_MOVDA16(0);
                WORD8 *p_mat1_1 = (WORD8 *) &p_mat1[(m_itr+1)*row_stride1];
                ae_int16x4 mat1_2 = AE_MOVDA16(0);
                WORD8 *p_mat1_2 = (WORD8 *) &p_mat1[(m_itr+2)*row_stride1];
                ae_int16x4 mat1_3 = AE_MOVDA16(0);
                WORD8 *p_mat1_3 = (WORD8 *) &p_mat1[(m_itr+3)*row_stride1];

#if XCHAL_HAVE_HIFI1
                for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                {
                    AE_L8X4S_IP(vec_batch_0, p_vec_batch_0, (1 * 4));
                    AE_L8X4S_IP(mat1_0, p_mat1_0, (1 * 4));
                    AE_L8X4S_IP(mat1_1, p_mat1_1, (1 * 4));
                    AE_L8X4S_IP(mat1_2, p_mat1_2, (1 * 4));
                    AE_L8X4S_IP(mat1_3, p_mat1_3, (1 * 4));     
                    vec_batch_0 = AE_ADD16(vec_batch_0, AE_MOVDA16(vec1_zero_bias));
                    mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));
                    mat1_1 = AE_ADD16(mat1_1, AE_MOVDA16(mat1_zero_bias));
                    mat1_2 = AE_ADD16(mat1_2, AE_MOVDA16(mat1_zero_bias));
                    mat1_3 = AE_ADD16(mat1_3, AE_MOVDA16(mat1_zero_bias));

                    AE_MULAAAAQ16(acc_0_0, vec_batch_0, mat1_0);
                    AE_MULAAAAQ16(acc_1_0, vec_batch_0, mat1_1);
                    AE_MULAAAAQ16(acc_2_0, vec_batch_0, mat1_2);
                    AE_MULAAAAQ16(acc_3_0, vec_batch_0, mat1_3);
                }
#else
                for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                {
                    AE_L8X4F_IP(vec_batch_0, p_vec_batch_0, (1 * 4));
                    vec_batch_0 = AE_SRAI16(vec_batch_0, 8);
                    vec_batch_0 = AE_ADD16(vec_batch_0, AE_MOVDA16(vec1_zero_bias));
                    AE_L8X4F_IP(mat1_0, p_mat1_0, (1 * 4));
                    mat1_0 = AE_SRAI16(mat1_0, 8);
                    mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));
                    AE_L8X4F_IP(mat1_1, p_mat1_1, (1 * 4));
                    mat1_1 = AE_SRAI16(mat1_1, 8);
                    mat1_1 = AE_ADD16(mat1_1, AE_MOVDA16(mat1_zero_bias));
                    AE_L8X4F_IP(mat1_2, p_mat1_2, (1 * 4));
                    mat1_2 = AE_SRAI16(mat1_2, 8);
                    mat1_2 = AE_ADD16(mat1_2, AE_MOVDA16(mat1_zero_bias));
                    AE_L8X4F_IP(mat1_3, p_mat1_3, (1 * 4));
                    mat1_3 = AE_SRAI16(mat1_3, 8);
                    mat1_3 = AE_ADD16(mat1_3, AE_MOVDA16(mat1_zero_bias));
                    AE_MULAAAAQ16(acc_0_0, vec_batch_0, mat1_0);
                    AE_MULAAAAQ16(acc_1_0, vec_batch_0, mat1_1);
                    AE_MULAAAAQ16(acc_2_0, vec_batch_0, mat1_2);
                    AE_MULAAAAQ16(acc_3_0, vec_batch_0, mat1_3);
                }
#endif
                ae_int32x2 acc_32x2_0, acc_32x2_1;
    
                acc_32x2_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc_0_0), AE_MOVINT32X2_FROMINT64(acc_1_0));
                acc_32x2_1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc_2_0), AE_MOVINT32X2_FROMINT64(acc_3_0));

                if(p_bias != (void *)0){
                    WORD32 bias0 = *p_bias32++;
                    WORD32 bias1 = *p_bias32++;
                    WORD32 bias2 = *p_bias32++;
                    WORD32 bias3 = *p_bias32++;
                   acc_32x2_0 = AE_ADD32(acc_32x2_0, AE_MOVDA32X2(bias0, bias1)); 
                   acc_32x2_1 = AE_ADD32(acc_32x2_1, AE_MOVDA32X2(bias2, bias3)); 
                }

                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_0, acc_32x2_0, out_multiplier, left_shift, right_shift);
                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_1, acc_32x2_1, out_multiplier, left_shift, right_shift);

                acc_32x2_0 = AE_ADD32S(acc_32x2_0, AE_MOVDA32(out_zero_bias));
                acc_32x2_1 = AE_ADD32S(acc_32x2_1, AE_MOVDA32(out_zero_bias));

                acc_32x2_0 = AE_MIN32(AE_MAX32(acc_32x2_0, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max)); 
                acc_32x2_1 = AE_MIN32(AE_MAX32(acc_32x2_1, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max)); 

                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_H(acc_32x2_0);
                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_0); 
                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 2)*out_stride)) = (WORD8)AE_MOVAD32_H(acc_32x2_1); 
                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 3)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_1); 
            }

            for(; m_itr < rows; m_itr++)
            {
                ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int16x4 vec_batch_0 = AE_MOVDA16(0);
                WORD8 *p_vec_batch_0 = (WORD8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int16x4 mat1_0 = AE_MOVDA16(0); WORD8 *p_mat1_0 = (WORD8 *) &p_mat1[(m_itr+0)*row_stride1];

#if XCHAL_HAVE_HIFI1
                for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                {
                    AE_L8X4S_IP(vec_batch_0, p_vec_batch_0, (1 * 4));
                    AE_L8X4S_IP(mat1_0, p_mat1_0, (1 * 4));
                    vec_batch_0 = AE_ADD16(vec_batch_0, AE_MOVDA16(vec1_zero_bias));
                    mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));                    
                    AE_MULAAAAQ16(acc_0_0, vec_batch_0, mat1_0);
                }
#else
                for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
                {              
                    AE_L8X4F_IP(vec_batch_0, p_vec_batch_0, (1 * 4));
                    vec_batch_0 = AE_SRAI16(vec_batch_0, 8);
                    vec_batch_0 = AE_ADD16(vec_batch_0, AE_MOVDA16(vec1_zero_bias));
                    AE_L8X4F_IP(mat1_0, p_mat1_0, (1 * 4));
                    mat1_0 = AE_SRAI16(mat1_0, 8);
                    mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));
                    AE_MULAAAAQ16(acc_0_0, vec_batch_0, mat1_0);
                }
#endif
                ae_int32x2 acc_32x2_0;
                acc_32x2_0 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(acc_0_0), AE_MOVINT32X2_FROMINT64(acc_0_0));

                if(p_bias != (void *)0){
                    WORD32 bias0 = *p_bias32++;
                   acc_32x2_0 = AE_ADD32(acc_32x2_0, AE_MOVDA32X2(bias0, bias0)); 
                }

                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_0, acc_32x2_0, out_multiplier, left_shift, right_shift);
                acc_32x2_0 = AE_ADD32S(acc_32x2_0, AE_MOVDA32(out_zero_bias));
                acc_32x2_0 = AE_MIN32(AE_MAX32(acc_32x2_0, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max)); 
                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_H(acc_32x2_0);
            }
        }
    }
    else if (p_mat1 && p_vec1)
    {
        for (vec_itr = 0; vec_itr < (vec_count & ~(2-1)); vec_itr += 2)
        {
            WORD32 bias32;
            ae_int64 sat_bias;
            WORD32 *ptr_bias = (WORD32 *) p_bias;
            for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
            {
            	ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
            	ae_int64 acc_0_1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
            	ae_int64 acc_1_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
            	ae_int64 acc_1_1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));

                ae_int16x4 vec0, vec1, mat1_0, mat1_1;   
                vec0 = vec1 = mat1_0 = mat1_1 = ZERO16X4;
                WORD8 *ptr_vec0  = (WORD8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                WORD8 *ptr_vec1  = (WORD8 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                WORD8 *p_mat1_0 = (WORD8 *) &p_mat1[(m_itr+0)*row_stride1];
                WORD8 *p_mat1_1 = (WORD8 *) &p_mat1[(m_itr+1)*row_stride1];

#if XCHAL_HAVE_HIFI1
                ae_valign align_vec0, align_vec1, align_mat1_0, align_mat1_1;
                align_vec0 = AE_LA64_PP(ptr_vec0);
                align_vec1 = AE_LA64_PP(ptr_vec1);
                align_mat1_0 = AE_LA64_PP(p_mat1_0);
                align_mat1_1 = AE_LA64_PP(p_mat1_1);

                int cols1_count = cols1- cols1%4;
                for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
                {
              	  AE_LA8X4S_IP(vec0, align_vec0, ptr_vec0);
              	  vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));
                  AE_LA8X4S_IP(vec1, align_vec1, ptr_vec1);
            	  vec1 = AE_ADD16(vec1, AE_MOVDA16(vec1_zero_bias));
            	  AE_LA8X4S_IP(mat1_0, align_mat1_0, p_mat1_0);
            	  mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));
            	  AE_LA8X4S_IP(mat1_1, align_mat1_1, p_mat1_1);
            	  mat1_1 = AE_ADD16(mat1_1, AE_MOVDA16(mat1_zero_bias));
            	  AE_MULAAAAQ16(acc_0_0, vec0, mat1_0);
            	  AE_MULAAAAQ16(acc_1_0, vec0, mat1_1);
            	  AE_MULAAAAQ16(acc_0_1, vec1, mat1_0);
            	  AE_MULAAAAQ16(acc_1_1, vec1, mat1_1);
                }
#else
                ALIGN_REGISTER_TYPE align_vec0;
                PRIME_8X4F(ptr_vec0, align_vec0);
                ALIGN_REGISTER_TYPE align_vec1;
                PRIME_8X4F(ptr_vec1, align_vec1);
                ALIGN_REGISTER_TYPE align_mat1_0;
                PRIME_8X4F(p_mat1_0, align_mat1_0);
                ALIGN_REGISTER_TYPE align_mat1_1;
                PRIME_8X4F(p_mat1_1, align_mat1_1);

                int cols1_count = cols1- cols1%4;
                for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
                {
              	  AE_LA8X4F_IP(vec0, align_vec0, ptr_vec0);
              	  vec0  = AE_SRAI16(vec0, 8);
              	  vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));
                  AE_LA8X4F_IP(vec1, align_vec1, ptr_vec1);
            	  vec1  = AE_SRAI16(vec1, 8);
            	  vec1 = AE_ADD16(vec1, AE_MOVDA16(vec1_zero_bias));
            	  AE_LA8X4F_IP(mat1_0, align_mat1_0, p_mat1_0);
            	  mat1_0 = AE_SRAI16(mat1_0, 8);
            	  mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));
            	  AE_LA8X4F_IP(mat1_1, align_mat1_1, p_mat1_1);
            	  mat1_1 = AE_SRAI16(mat1_1, 8);
            	  mat1_1 = AE_ADD16(mat1_1, AE_MOVDA16(mat1_zero_bias));
            	  AE_MULAAAAQ16(acc_0_0, vec0, mat1_0);
            	  AE_MULAAAAQ16(acc_1_0, vec0, mat1_1);
            	  AE_MULAAAAQ16(acc_0_1, vec1, mat1_0);
            	  AE_MULAAAAQ16(acc_1_1, vec1, mat1_1);
                }
#endif                
                #pragma no_unroll
                for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                {
                  vec0 = AE_MOVDA16((short)*(ptr_vec0)); 
                  vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));

                  vec1 = AE_MOVDA16((short)*(ptr_vec1));
                  vec1 = AE_ADD16(vec1, AE_MOVDA16(vec1_zero_bias));
                  mat1_0 = AE_MOVDA16((short)*(p_mat1_0));
                  mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));
                  mat1_1 = AE_MOVDA16((short)*(p_mat1_1));
                  mat1_1 = AE_ADD16(mat1_1, AE_MOVDA16(mat1_zero_bias));

                  AE_MULA16_00(acc_0_0, vec0, mat1_0);
                  AE_MULA16_00(acc_1_0, vec0, mat1_1);
                  AE_MULA16_00(acc_0_1, vec1, mat1_0);
                  AE_MULA16_00(acc_1_1, vec1, mat1_1);

                  ptr_vec0++;
                  ptr_vec1++;                  
                  p_mat1_0++;
                  p_mat1_1++;

                }
                
                if(p_bias!=(void *)0)
                {
                  bias32 = *ptr_bias++;
                  sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(bias32)), 32);
                  acc_0_0 = AE_ADD64S(acc_0_0, sat_bias);
                  acc_0_1 = AE_ADD64S(acc_0_1, sat_bias);
                  bias32 = *ptr_bias++;
                  sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(bias32)), 32);
                  acc_1_0 = AE_ADD64S(acc_1_0, sat_bias);
                  acc_1_1 = AE_ADD64S(acc_1_1, sat_bias);
                }

                ae_int32x2 acc_32x2_0, acc_32x2_1, acc_32x2_2, acc_32x2_3;
                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_0, AE_MOVINT32X2_FROMINT64(acc_0_0), out_multiplier, left_shift, right_shift);
                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_1, AE_MOVINT32X2_FROMINT64(acc_0_1), out_multiplier, left_shift, right_shift);
                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_2, AE_MOVINT32X2_FROMINT64(acc_1_0), out_multiplier, left_shift, right_shift);
                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_3, AE_MOVINT32X2_FROMINT64(acc_1_1), out_multiplier, left_shift, right_shift);

                acc_32x2_0 = AE_ADD32S(acc_32x2_0, AE_MOVDA32(out_zero_bias));
                acc_32x2_1 = AE_ADD32S(acc_32x2_1, AE_MOVDA32(out_zero_bias));
                acc_32x2_2 = AE_ADD32S(acc_32x2_2, AE_MOVDA32(out_zero_bias));
                acc_32x2_3 = AE_ADD32S(acc_32x2_3, AE_MOVDA32(out_zero_bias));

                acc_32x2_0 = AE_MIN32(AE_MAX32(acc_32x2_0, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max));
                acc_32x2_2 = AE_MIN32(AE_MAX32(acc_32x2_2, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max));
                acc_32x2_1 = AE_MIN32(AE_MAX32(acc_32x2_1, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max));
                acc_32x2_3 = AE_MIN32(AE_MAX32(acc_32x2_3, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max));

                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_0);
                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_2);
                (*((WORD8 *) p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_1);
                (*((WORD8 *) p_out + (vec_itr + 1)*out_offset + (m_itr + 1)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_3);
            }
            //Remaining row
            for(; m_itr < rows; m_itr++)
            {
            	ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
            	ae_int64 acc_0_1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));

            	ae_int16x4 vec0, vec1, mat1_0; 
                vec0 = vec1 = mat1_0  = ZERO16X4;

            	WORD8 *ptr_vec0  = (WORD8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
            	WORD8 *ptr_vec1  = (WORD8 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                WORD8 *p_mat1_0 = (WORD8 *) &p_mat1[(m_itr+0)*row_stride1];

#if XCHAL_HAVE_HIFI1
                ae_valign align_vec0, align_vec1, align_mat1_0;
                align_vec0 = AE_LA64_PP(ptr_vec0);
                align_vec1 = AE_LA64_PP(ptr_vec1);
                align_mat1_0 = AE_LA64_PP(p_mat1_0);

                int cols1_count = cols1- cols1%4;

                for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
                {
                  AE_LA8X4S_IP(vec0, align_vec0, ptr_vec0);
                  vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));
                  AE_LA8X4S_IP(vec1, align_vec1, ptr_vec1);
                  vec1 = AE_ADD16(vec1, AE_MOVDA16(vec1_zero_bias));
                  AE_LA8X4S_IP(mat1_0, align_mat1_0, p_mat1_0);
                  mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));

                  AE_MULAAAAQ16(acc_0_0, vec0, mat1_0);
                  AE_MULAAAAQ16(acc_0_1, vec1, mat1_0);
                }
#else
            	ALIGN_REGISTER_TYPE align_vec0;
            	PRIME_8X4F(ptr_vec0, align_vec0);
            	ALIGN_REGISTER_TYPE align_vec1;
            	PRIME_8X4F(ptr_vec1, align_vec1);
                ALIGN_REGISTER_TYPE align_mat1_0;
                PRIME_8X4F(p_mat1_0, align_mat1_0);

                int cols1_count = cols1- cols1%4;

                for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
                {
                  AE_LA8X4F_IP(vec0, align_vec0, ptr_vec0);
                  vec0  = AE_SRAI16(vec0, 8);
                  vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));

                  AE_LA8X4F_IP(vec1, align_vec1, ptr_vec1);
                  vec1  = AE_SRAI16(vec1, 8);
                  vec1 = AE_ADD16(vec1, AE_MOVDA16(vec1_zero_bias));

                  AE_LA8X4F_IP(mat1_0, align_mat1_0, p_mat1_0);
                  mat1_0 = AE_SRAI16(mat1_0, 8);
                  mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));

                  AE_MULAAAAQ16(acc_0_0, vec0, mat1_0);
                  AE_MULAAAAQ16(acc_0_1, vec1, mat1_0);
                }
#endif
                #pragma no_unroll
                for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                {
                  vec0 = AE_MOVDA16((short)*(ptr_vec0));
                  vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));
                  vec1 = AE_MOVDA16((short)*(ptr_vec1));
                  vec1 = AE_ADD16(vec1, AE_MOVDA16(vec1_zero_bias));
                  mat1_0 = AE_MOVDA16((short)*(p_mat1_0));
                  mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));

                  AE_MULA16_00(acc_0_0, vec0, mat1_0);
                  AE_MULA16_00(acc_0_1, vec1, mat1_0);

                  ptr_vec0++;
                  ptr_vec1++;
                  p_mat1_0++;
                }
                if(p_bias!=(void *)0)
                {
                  bias32 = *ptr_bias++;
                  sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(bias32)), 32);
                  acc_0_0 = AE_ADD64S(acc_0_0, sat_bias); 
                  acc_0_1 = AE_ADD64S(acc_0_1, sat_bias);
                }

                ae_int32x2 acc_32x2_0, acc_32x2_1;
                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_0, AE_MOVINT32X2_FROMINT64(acc_0_0), out_multiplier, left_shift, right_shift);
                MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_1, AE_MOVINT32X2_FROMINT64(acc_0_1), out_multiplier, left_shift, right_shift);

                acc_32x2_0 = AE_ADD32S(acc_32x2_0, AE_MOVDA32(out_zero_bias));
                acc_32x2_1 = AE_ADD32S(acc_32x2_1, AE_MOVDA32(out_zero_bias));

                acc_32x2_0 = AE_MIN32(AE_MAX32(acc_32x2_0, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max));
                acc_32x2_1 = AE_MIN32(AE_MAX32(acc_32x2_1, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max));

                (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_0);
                (*((WORD8 *) p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_1);
            }

        }
        /* Tail loop for vec unroll */
        for(; vec_itr < vec_count; vec_itr++)
        {
            WORD32 bias32;
            ae_int64 sat_bias;
            WORD32 *ptr_bias = (WORD32 *) p_bias;
            for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
            {
              ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
              ae_int64 acc_1_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));

              ae_int16x4  vec0, mat1_0,  mat1_1;
              vec0 = mat1_0  = mat1_1 = ZERO16X4;

              WORD8 *ptr_vec0  = (WORD8 *)(p_vec1 + (vec_itr + 0)*vec_offset);       
              WORD8 *p_mat1_0 = (WORD8 *) &p_mat1[(m_itr+0)*row_stride1];         
              WORD8 *p_mat1_1 = (WORD8 *) &p_mat1[(m_itr+1)*row_stride1];

#if XCHAL_HAVE_HIFI1
              ae_valign align_vec0, align_mat1_0, align_mat1_1;
              align_vec0 = AE_LA64_PP(ptr_vec0);
              align_mat1_0 = AE_LA64_PP(p_mat1_0);
              align_mat1_1 = AE_LA64_PP(p_mat1_1);

              int cols1_count = cols1 - cols1%4;

              for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
              {
                AE_LA8X4S_IP(vec0, align_vec0, ptr_vec0);
                vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));
                AE_LA8X4S_IP(mat1_0, align_mat1_0, p_mat1_0);
                mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));
                AE_LA8X4S_IP(mat1_1, align_mat1_1, p_mat1_1);
                mat1_1 = AE_ADD16(mat1_1, AE_MOVDA16(mat1_zero_bias));

                AE_MULAAAAQ16(acc_0_0, vec0, mat1_0);
                AE_MULAAAAQ16(acc_1_0, vec0, mat1_1);
              }
#else
              ALIGN_REGISTER_TYPE align_vec0;
              PRIME_8X4F(ptr_vec0, align_vec0);
              ALIGN_REGISTER_TYPE align_mat1_0;
              PRIME_8X4F(p_mat1_0, align_mat1_0);
              ALIGN_REGISTER_TYPE align_mat1_1;
              PRIME_8X4F(p_mat1_1, align_mat1_1);

              int cols1_count = cols1 - cols1%4;

              for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
              {
                AE_LA8X4F_IP(vec0, align_vec0, ptr_vec0);
                vec0  = AE_SRAI16(vec0, 8);
                vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));

                AE_LA8X4F_IP(mat1_0, align_mat1_0, p_mat1_0);
                mat1_0 = AE_SRAI16(mat1_0, 8);
                mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));

                AE_LA8X4F_IP(mat1_1, align_mat1_1, p_mat1_1);
                mat1_1 = AE_SRAI16(mat1_1, 8);
                mat1_1 = AE_ADD16(mat1_1, AE_MOVDA16(mat1_zero_bias));

                AE_MULAAAAQ16(acc_0_0, vec0, mat1_0);
                AE_MULAAAAQ16(acc_1_0, vec0, mat1_1);
              }
#endif                  
#pragma no_unroll
              for(c_itr = cols1_count; c_itr < cols1; c_itr++)
              {
                vec0 = AE_MOVDA16((short)*(ptr_vec0));
                vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));
                mat1_0 = AE_MOVDA16((short)*(p_mat1_0));
                mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));
                mat1_1 = AE_MOVDA16((short)*(p_mat1_1));
                mat1_1 = AE_ADD16(mat1_1, AE_MOVDA16(mat1_zero_bias));

                AE_MULA16_00(acc_0_0, vec0, mat1_0);
                AE_MULA16_00(acc_1_0, vec0, mat1_1);

                ptr_vec0++;
                p_mat1_0++;
                p_mat1_1++;
              }  
              if(p_bias!=(void *)0)
              {
                bias32 = *ptr_bias++;
                sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(bias32)), 32);
                acc_0_0 = AE_ADD64S(acc_0_0, sat_bias);
                bias32 = *ptr_bias++;
                sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(bias32)), 32);
                acc_1_0 = AE_ADD64S(acc_1_0, sat_bias);                        
              }

              ae_int32x2 acc_32x2_0, acc_32x2_2;
              MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_0, AE_MOVINT32X2_FROMINT64(acc_0_0), out_multiplier, left_shift, right_shift);
              MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_2, AE_MOVINT32X2_FROMINT64(acc_1_0), out_multiplier, left_shift, right_shift);
              acc_32x2_0 = AE_ADD32S(acc_32x2_0, AE_MOVDA32(out_zero_bias));
              acc_32x2_2 = AE_ADD32S(acc_32x2_2, AE_MOVDA32(out_zero_bias));
              acc_32x2_0 = AE_MIN32(AE_MAX32(acc_32x2_0, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max));
              acc_32x2_2 = AE_MIN32(AE_MAX32(acc_32x2_2, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max));

              (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_0);
              (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_2);
            }

            for(; m_itr < rows; m_itr++)
            {
              ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
              ae_int16x4 vec0, mat1_0; 
              vec0 = mat1_0  = ZERO16X4;

              WORD8 *ptr_vec0  = (WORD8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
              WORD8 *p_mat1_0 = (WORD8 *) &p_mat1[(m_itr+0)*row_stride1];

#if XCHAL_HAVE_HIFI1
              ae_valign align_vec0, align_mat1_0;
              align_vec0 = AE_LA64_PP(ptr_vec0);
              align_mat1_0 = AE_LA64_PP(p_mat1_0);

              int cols1_count = cols1 - cols1%4;

              for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
              {
                AE_LA8X4S_IP(vec0, align_vec0, ptr_vec0);
                vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));
                AE_LA8X4S_IP(mat1_0, align_mat1_0, p_mat1_0);
                mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));
                AE_MULAAAAQ16(acc_0_0, vec0, mat1_0);
              }
#else
              ALIGN_REGISTER_TYPE align_vec0;
              PRIME_8X4F(ptr_vec0, align_vec0);
              ALIGN_REGISTER_TYPE align_mat1_0;
              PRIME_8X4F(p_mat1_0, align_mat1_0);

              int cols1_count = cols1 - cols1%4;

              for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
              {
                AE_LA8X4F_IP(vec0, align_vec0, ptr_vec0);
                vec0  = AE_SRAI16(vec0, 8);
                vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));
                AE_LA8X4F_IP(mat1_0, align_mat1_0, p_mat1_0);
                mat1_0 = AE_SRAI16(mat1_0, 8);
                mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));

                AE_MULAAAAQ16(acc_0_0, vec0, mat1_0);
              }
#endif                  
#pragma no_unroll
              for(c_itr = cols1_count; c_itr < cols1; c_itr++)
              {
                vec0 = AE_MOVDA16((short)*(ptr_vec0));                    	
                vec0 = AE_ADD16(vec0, AE_MOVDA16(vec1_zero_bias));
                mat1_0 = AE_MOVDA16((short)*(p_mat1_0));	
                mat1_0 = AE_ADD16(mat1_0, AE_MOVDA16(mat1_zero_bias));

                AE_MULA16_00(acc_0_0, vec0, mat1_0);

                ptr_vec0++;
                p_mat1_0++;
              }

              if(p_bias!=(void *)0)
              {
                bias32 = *ptr_bias++;
                sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(bias32)), 32);
                acc_0_0 = AE_ADD64S(acc_0_0, sat_bias);
              }

              ae_int32x2 acc_32x2_0;
              MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc_32x2_0, AE_MOVINT32X2_FROMINT64(acc_0_0), out_multiplier, left_shift, right_shift);
              acc_32x2_0 = AE_ADD32S(acc_32x2_0, AE_MOVDA32(out_zero_bias));
              acc_32x2_0 = AE_MIN32(AE_MAX32(acc_32x2_0, AE_MOVDA32(out_activation_min)), AE_MOVDA32(out_activation_max));
              (*((WORD8 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)AE_MOVAD32_L(acc_32x2_0);
            }
        }
    }
    else
    {
        return -1;
    }

    return 0;
}
#endif // XCHAL_HAVE_HIFI1S

/* Legacy implementation provided with call through _fast API */
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
  WORD32 ret;
  ret = xa_nn_matmul_v2_asym8sxasym8s_asym8s(
                 p_out, p_mat1, p_vec1, p_bias,
                 rows, cols1, row_stride1, vec_count, vec_offset, out_offset, out_stride,
                 mat1_zero_bias, vec1_zero_bias, out_multiplier, out_shift, out_zero_bias, -128, 127, NULL);
  return ret;
}
