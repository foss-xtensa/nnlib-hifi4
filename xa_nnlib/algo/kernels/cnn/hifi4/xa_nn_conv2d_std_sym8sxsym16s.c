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
#include "xa_nnlib_common_macros.h"
#include "xa_nn_conv2d_std_state.h"

static inline ae_int32x2 MultiplyByQuantizedMultiplier_ref(ae_int64 d_x,
                                             int32_t quantized_multiplier,
                                             int shift){
  // Inputs:
  // - quantized_multiplier has fixed point at bit 31
  // - shift is -31 to +7 (negative for right shift)
  //
  // Assumptions: The following input ranges are assumed
  // - quantize_scale>=0  (the usual range is (1<<30) to (1>>31)-1)
  // - scaling is chosen so final scaled result fits in int32_t
  // - input x is in the range -(1<<47) <= x < (1<<47)
/* shift_val  = -31 to 7
 * total_shift = 46 to 8
 * new_shift = 46-32 to 8-32
 * */
  ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
  ae_int32x2 d_red_mul32 = AE_SEXT32X2D16_32(d_red_mul16); //upper 32
  ae_int64 qL = AE_MUL32U_LL(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x));
  ae_int64 qH = AE_SLAI64(AE_MUL32_LH(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x)), 32);
  ae_int64 q = AE_ADD64(qL, qH);
  q = AE_SLAA64S(q, (shift+17));
  ae_int32x2 result = AE_ROUND32F64SASYM(q);
  return result;
}

#if XCHAL_HAVE_HIFI1S
static inline ae_int32x2 MultiplyByQuantizedMultiplier_x2_opt(ae_int64 d_x1, ae_int64 d_x2,
                                             int32_t quantized_multiplier,
                                             int shift) {
  ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
  ae_int64 q1 = AE_MUL48X16_0(d_x1, d_red_mul16);
  ae_int64 q2 = AE_MUL48X16_0(d_x2, d_red_mul16);
  ae_int32x2 result = AE_ROUNDAV32X2F64SASYM (q1, q2, shift);
  return result;
}
#else
static inline ae_int32x2 MultiplyByQuantizedMultiplier_x2_opt(ae_int64 d_x1, ae_int64 d_x2,
                                             int32_t quantized_multiplier,
                                             int shift) {
  ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
  ae_int32x2 d_red_mul32 = AE_SEXT32X2D16_32(d_red_mul16);
  ae_int64 qL1 = AE_MUL32U_LL(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x1));
  ae_int64 qL2 = AE_MUL32U_LL(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x2));
  ae_int64 qH1 = AE_SLAI64(AE_MUL32_LH(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x1)), 32);
  ae_int64 qH2 = AE_SLAI64(AE_MUL32_LH(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x2)), 32);
  ae_int64 q1 = AE_ADD64(qL1, qH1);
  ae_int64 q2 = AE_ADD64(qL2, qH2);
  q1 = AE_SLAA64S(q1, (shift+17));
  q2 = AE_SLAA64S(q2, (shift+17));
  ae_int32x2 result = AE_ROUND32X2F64SASYM(q1, q2);
  return result;
}
#endif

static WORD32 conv_x_left_pad(
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const WORD64* __restrict__ p_bias,
    WORD16 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_activation_min,
    WORD32 out_activation_max)
{
  WORD32 i,j,k;
  WORD32 out_width_over_x_pad = (x_padding - kernel_width)/x_stride + 1;
  out_width_over_x_pad = out_width_over_x_pad > out_width ? out_width : out_width_over_x_pad;
  ae_int16x4 d1;

  /* When kernel convolves over x-left pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = 0; j < out_width_over_x_pad; j++)
    {
      ae_int16 *ptrout = (ae_int16*)&p_out[i * out_height_offset + j * out_width_offset];
      ae_int64 *pbias = (ae_int64*)p_bias;
      ae_int64 q1;
      for(k = 0; k < out_channels; k++)
      {
        if(p_bias != NULL){  
          AE_L64_IP(q1, pbias, 8);
        }
        else{
          q1 = 0;
        }
	ae_int32x2 acc = MultiplyByQuantizedMultiplier_ref(q1, p_out_multiplier[k], p_out_shift[k]);
    AE_MINMAX32(acc, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
	d1 = AE_SAT16X4(acc, acc);
	AE_S16_0_XP(d1, ptrout, out_channels_offset*sizeof(WORD16));
      }
    }
  }
  return out_width_over_x_pad;
}

static WORD32 conv_x_right_pad(
    WORD32 x_padding,
    WORD32 input_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const WORD64* __restrict__ p_bias,
    WORD16 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_activation_min,
    WORD32 out_activation_max)
{
  WORD32 i,j,k;
  WORD32 idx_out_width_over_x_r_pad = (x_padding + input_width + x_stride - 1)/x_stride + 1;
  WORD32 out_width_over_x_r_pad = out_width - idx_out_width_over_x_r_pad;
  ae_int16x4 d1;

  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = idx_out_width_over_x_r_pad; j < out_width; j++)
    {
      ae_int16 *ptrout = (ae_int16*)&p_out[i * out_height_offset + j * out_width_offset];
      ae_int64 *pbias = (ae_int64*)p_bias;
      ae_int64 q1;
      for(k = 0; k < out_channels; k++)
      {
        if(p_bias != NULL){  
          AE_L64_IP(q1, pbias, 8);
        }
        else{
          q1 = 0;
        }
	ae_int32x2 acc = MultiplyByQuantizedMultiplier_ref(q1, p_out_multiplier[k], p_out_shift[k]);
    AE_MINMAX32(acc, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
	d1 = AE_SAT16X4(acc, acc);
	AE_S16_0_XP(d1, ptrout, out_channels_offset*sizeof(WORD16));
      }
    }
  }
  return out_width_over_x_r_pad;
}

static WORD32 xa_nn_conv2d_std_per_chan_sym8sxsym16s_no_circ_buf(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD64* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    WORD32 out_activation_min,
    WORD32 out_activation_max
    )
{
      
    const WORD16 *p_dst0_0 = p_out + 0;
    const WORD16 *p_dst0_1 = p_out + 1;
    const WORD16 *p_dst0_2 = p_out + 2;
    const WORD16 *p_dst0_3 = p_out + 3;
    const WORD16 *p_dst1_0 = p_out + out_channels + 0;
    const WORD16 *p_dst1_1 = p_out + out_channels + 1;
    const WORD16 *p_dst1_2 = p_out + out_channels + 2;
    const WORD16 *p_dst1_3 = p_out + out_channels + 3;
    int kernel_out_ch_offset = kernel_height * kernel_width * input_channels;
    int input_x_offset = (input_channels * x_stride) / 4;
    int p_inp_vec_stride = (input_width * input_channels) / 4;
    int p_kern_vec_stride = kernel_width * input_channels;
    int vec_len = kernel_width * input_channels;
    for (int out_y = 0; out_y < out_height; ++out_y) {
      for (int out_x = 0; out_x < out_width; out_x += 2) {
        for (int out_ch = 0; out_ch < out_channels; out_ch += 4) {
          ae_int64 out0_0 = p_bias[out_ch + 0];
          ae_int64 out0_1 = p_bias[out_ch + 1];
          ae_int64 out0_2 = p_bias[out_ch + 2];
          ae_int64 out0_3 = p_bias[out_ch + 3];
          ae_int64 out1_0 = p_bias[out_ch + 0];
          ae_int64 out1_1 = p_bias[out_ch + 1];
          ae_int64 out1_2 = p_bias[out_ch + 2];
          ae_int64 out1_3 = p_bias[out_ch + 3];
         
#if !XCHAL_HAVE_HIFI1S
          out0_0 = AE_SLAI64(out0_0, 8);
          out0_1 = AE_SLAI64(out0_1, 8);
          out0_2 = AE_SLAI64(out0_2, 8);
          out0_3 = AE_SLAI64(out0_3, 8);
          out1_0 = AE_SLAI64(out1_0, 8);
          out1_1 = AE_SLAI64(out1_1, 8);
          out1_2 = AE_SLAI64(out1_2, 8);
          out1_3 = AE_SLAI64(out1_3, 8);
#endif
          
          int in_x_o = out_x * x_stride;
          int in_y_o = out_y * y_stride - y_padding;
          int k_y_min = -in_y_o;
          int k_y_max = input_height - in_y_o;
          k_y_min = (k_y_min < 0) ? 0 : k_y_min;
          k_y_min = (k_y_min < kernel_height) ? k_y_min : kernel_height;
          k_y_max = (k_y_max < 0) ? 0 : k_y_max;
          k_y_max = (k_y_max < kernel_height) ? k_y_max : kernel_height;
          const ae_int16x4 *p_inp_vec =
              (ae_int16x4 *)&p_inp[((in_y_o + k_y_min) * input_width + in_x_o) *
                                      input_channels +
                                  0];
          const WORD8 *p_kern_vec =
              &p_kernel[(((out_ch + 0) * kernel_height + k_y_min) * kernel_width +
                        0) *
                            input_channels +
                        0];
          for (int k_y = k_y_min; k_y < k_y_max; ++k_y) {
            const ae_int16x4 *p_inp_vec0 = p_inp_vec;
            const ae_int16x4 *p_inp_vec1 = p_inp_vec + input_x_offset;
            const WORD8 *p_kern_vec0 = p_kern_vec;
            const WORD8 *p_kern_vec1 = p_kern_vec0 + kernel_out_ch_offset;
            const WORD8 *p_kern_vec2 = p_kern_vec1 + kernel_out_ch_offset;
            const WORD8 *p_kern_vec3 = p_kern_vec2 + kernel_out_ch_offset;
            p_inp_vec += p_inp_vec_stride;
            p_kern_vec += p_kern_vec_stride;
            ae_int16x4 d_inp0;
            ae_int16x4 d_inp1;

#if XCHAL_HAVE_HIFI1S
            ae_int16x4 d_inp0_1, d_inp1_1;
            ae_int8x8 d_kern0, d_kern1, d_kern2, d_kern3;
            ae_valign align_k0 = AE_LA64_PP(p_kern_vec0);
            ae_valign align_k1 = AE_LA64_PP(p_kern_vec1);
            ae_valign align_k2 = AE_LA64_PP(p_kern_vec2);
            ae_valign align_k3 = AE_LA64_PP(p_kern_vec3);
            int i;
            for (i = 0; i < (vec_len&~0x07); i += 8) {
              AE_L16X4_IP(d_inp0, p_inp_vec0, 8);
              AE_L16X4_IP(d_inp1, p_inp_vec1, 8);
              AE_L16X4_IP(d_inp0_1, p_inp_vec0, 8);
              AE_L16X4_IP(d_inp1_1, p_inp_vec1, 8);
              AE_LA8X8_IP(d_kern0, align_k0, (ae_int8x8 *)p_kern_vec0);
              AE_LA8X8_IP(d_kern1, align_k1, (ae_int8x8 *)p_kern_vec1);
              AE_LA8X8_IP(d_kern2, align_k2, (ae_int8x8 *)p_kern_vec2);
              AE_LA8X8_IP(d_kern3, align_k3, (ae_int8x8 *)p_kern_vec3);

              AE_MULAO8X16(out0_0, d_inp0, d_inp0_1, d_kern0);
              AE_MULAO8X16(out0_1, d_inp0, d_inp0_1, d_kern1);
              AE_MULAO8X16(out0_2, d_inp0, d_inp0_1, d_kern2);
              AE_MULAO8X16(out0_3, d_inp0, d_inp0_1, d_kern3);
              AE_MULAO8X16(out1_0, d_inp1, d_inp1_1, d_kern0);
              AE_MULAO8X16(out1_1, d_inp1, d_inp1_1, d_kern1);
              AE_MULAO8X16(out1_2, d_inp1, d_inp1_1, d_kern2);
              AE_MULAO8X16(out1_3, d_inp1, d_inp1_1, d_kern3);
            }

            if (vec_len&0x07) {
              /* As vec_len is multiple of 4, only 4 elements are now remaining */
              AE_L16X4_IP(d_inp0, p_inp_vec0, 8);
              AE_L16X4_IP(d_inp1, p_inp_vec1, 8);
              d_inp0_1 = AE_ZERO16();
              AE_LAV8X8_XP(d_kern0, align_k0, (ae_int8x8 *)p_kern_vec0, 4);
              AE_LAV8X8_XP(d_kern1, align_k1, (ae_int8x8 *)p_kern_vec1, 4);
              AE_LAV8X8_XP(d_kern2, align_k2, (ae_int8x8 *)p_kern_vec2, 4);
              AE_LAV8X8_XP(d_kern3, align_k3, (ae_int8x8 *)p_kern_vec3, 4);

              AE_MULAO8X16(out0_0, d_inp0, d_inp0_1, d_kern0);
              AE_MULAO8X16(out0_1, d_inp0, d_inp0_1, d_kern1);
              AE_MULAO8X16(out0_2, d_inp0, d_inp0_1, d_kern2);
              AE_MULAO8X16(out0_3, d_inp0, d_inp0_1, d_kern3);
              AE_MULAO8X16(out1_0, d_inp1, d_inp0_1, d_kern0);
              AE_MULAO8X16(out1_1, d_inp1, d_inp0_1, d_kern1);
              AE_MULAO8X16(out1_2, d_inp1, d_inp0_1, d_kern2);
              AE_MULAO8X16(out1_3, d_inp1, d_inp0_1, d_kern3);
            }
#else
            ae_int16x4 d_kern0, d_kern1, d_kern2, d_kern3;
#if XCHAL_HAVE_HIFI4 /* Following code avoids membank conflicts*/
            for (int i = 0; i < (vec_len&~0x07); i += 8) {
              ae_int16x4 d_inp0_1, d_inp1_1;
              ae_int16x4 d_kern0_1, d_kern1_1, d_kern2_1, d_kern3_1;

              d_inp0_1 = AE_L16X4_I(p_inp_vec0, 8);
              AE_L16X4_IP(d_inp0, p_inp_vec0, 16);
              d_inp1_1 = AE_L16X4_I(p_inp_vec1, 8);
              AE_L16X4_IP(d_inp1, p_inp_vec1, 16);

              d_kern0_1 = AE_L8X4F_I(p_kern_vec0, 4);
              AE_L8X4F_IP(d_kern0, p_kern_vec0, 8);
              d_kern1_1 = AE_L8X4F_I(p_kern_vec1, 4);
              AE_L8X4F_IP(d_kern1, p_kern_vec1, 8);
              d_kern2_1 = AE_L8X4F_I(p_kern_vec2, 4);
              AE_L8X4F_IP(d_kern2, p_kern_vec2, 8);
              d_kern3_1 = AE_L8X4F_I(p_kern_vec3, 4);
              AE_L8X4F_IP(d_kern3, p_kern_vec3, 8);

              AE_MULAAAAQ16(out0_0, d_inp0, d_kern0);
              AE_MULAAAAQ16(out0_1, d_inp0, d_kern1);
              AE_MULAAAAQ16(out0_2, d_inp0, d_kern2);
              AE_MULAAAAQ16(out0_3, d_inp0, d_kern3);
              AE_MULAAAAQ16(out1_0, d_inp1, d_kern0);
              AE_MULAAAAQ16(out1_1, d_inp1, d_kern1);
              AE_MULAAAAQ16(out1_2, d_inp1, d_kern2);
              AE_MULAAAAQ16(out1_3, d_inp1, d_kern3);

              AE_MULAAAAQ16(out0_0, d_inp0_1, d_kern0_1);
              AE_MULAAAAQ16(out0_1, d_inp0_1, d_kern1_1);
              AE_MULAAAAQ16(out0_2, d_inp0_1, d_kern2_1);
              AE_MULAAAAQ16(out0_3, d_inp0_1, d_kern3_1);
              AE_MULAAAAQ16(out1_0, d_inp1_1, d_kern0_1);
              AE_MULAAAAQ16(out1_1, d_inp1_1, d_kern1_1);
              AE_MULAAAAQ16(out1_2, d_inp1_1, d_kern2_1);
              AE_MULAAAAQ16(out1_3, d_inp1_1, d_kern3_1);
            }
            if(vec_len%8 != 0) {
              AE_L16X4_IP(d_inp0, p_inp_vec0, 8);
              AE_L16X4_IP(d_inp1, p_inp_vec1, 8);
              AE_L8X4F_IP(d_kern0, p_kern_vec0, 4);
              AE_L8X4F_IP(d_kern1, p_kern_vec1, 4);
              AE_L8X4F_IP(d_kern2, p_kern_vec2, 4);
              AE_L8X4F_IP(d_kern3, p_kern_vec3, 4);
              AE_MULAAAAQ16(out0_0, d_inp0, d_kern0);
              AE_MULAAAAQ16(out0_1, d_inp0, d_kern1);
              AE_MULAAAAQ16(out0_2, d_inp0, d_kern2);
              AE_MULAAAAQ16(out0_3, d_inp0, d_kern3);
              AE_MULAAAAQ16(out1_0, d_inp1, d_kern0);
              AE_MULAAAAQ16(out1_1, d_inp1, d_kern1);
              AE_MULAAAAQ16(out1_2, d_inp1, d_kern2);
              AE_MULAAAAQ16(out1_3, d_inp1, d_kern3);
            }
#else /* XCHAL_HAVE_HIFI4 */
            for (int i = 0; i < vec_len; i += 4) {
              AE_L16X4_IP(d_inp0, p_inp_vec0, 8);
              AE_L16X4_IP(d_inp1, p_inp_vec1, 8);
              AE_L8X4F_IP(d_kern0, p_kern_vec0, 4);
              AE_L8X4F_IP(d_kern1, p_kern_vec1, 4);
              AE_L8X4F_IP(d_kern2, p_kern_vec2, 4);
              AE_L8X4F_IP(d_kern3, p_kern_vec3, 4);
              AE_MULAAAAQ16(out0_0, d_inp0, d_kern0);
              AE_MULAAAAQ16(out0_1, d_inp0, d_kern1);
              AE_MULAAAAQ16(out0_2, d_inp0, d_kern2);
              AE_MULAAAAQ16(out0_3, d_inp0, d_kern3);
              AE_MULAAAAQ16(out1_0, d_inp1, d_kern0);
              AE_MULAAAAQ16(out1_1, d_inp1, d_kern1);
              AE_MULAAAAQ16(out1_2, d_inp1, d_kern2);
              AE_MULAAAAQ16(out1_3, d_inp1, d_kern3);
            }
#endif /* XCHAL_HAVE_HIFI4 */
#endif
          }

#if XCHAL_HAVE_HIFI1S
          WORD32 out_shift_0 = 15 - p_out_shift[out_ch+0];
          out_shift_0 = ( out_shift_0  << 16 ) | out_shift_0;
          WORD32 out_shift_1 = 15 - p_out_shift[out_ch+1];
          out_shift_1 = ( out_shift_1  << 16 ) | out_shift_1;
          WORD32 out_shift_2 = 15 - p_out_shift[out_ch+2];
          out_shift_2 = ( out_shift_2  << 16 ) | out_shift_2;
          WORD32 out_shift_3 = 15 - p_out_shift[out_ch+3];
          out_shift_3 = ( out_shift_3  << 16 ) | out_shift_3;
#else
          out0_0 = AE_SRAI64(out0_0, 8);
          out0_1 = AE_SRAI64(out0_1, 8);
          out0_2 = AE_SRAI64(out0_2, 8);
          out0_3 = AE_SRAI64(out0_3, 8);
          out1_0 = AE_SRAI64(out1_0, 8);
          out1_1 = AE_SRAI64(out1_1, 8);
          out1_2 = AE_SRAI64(out1_2, 8);
          out1_3 = AE_SRAI64(out1_3, 8);
          WORD32 out_shift_0 = p_out_shift[out_ch+0];
          WORD32 out_shift_1 = p_out_shift[out_ch+1];
          WORD32 out_shift_2 = p_out_shift[out_ch+2];
          WORD32 out_shift_3 = p_out_shift[out_ch+3];
#endif
          ae_int32x2 acc_vec0 = MultiplyByQuantizedMultiplier_x2_opt(
              out0_0, out1_0, p_out_multiplier[out_ch + 0],
              out_shift_0);
          ae_int32x2 acc_vec1 = MultiplyByQuantizedMultiplier_x2_opt(
              out0_1, out1_1, p_out_multiplier[out_ch + 1],
              out_shift_1);
          ae_int32x2 acc_vec2 = MultiplyByQuantizedMultiplier_x2_opt(
              out0_2, out1_2, p_out_multiplier[out_ch + 2],
              out_shift_2);
          ae_int32x2 acc_vec3 = MultiplyByQuantizedMultiplier_x2_opt(
              out0_3, out1_3, p_out_multiplier[out_ch + 3],
              out_shift_3);
          AE_MINMAX32(acc_vec0, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          AE_MINMAX32(acc_vec1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          AE_MINMAX32(acc_vec2, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          AE_MINMAX32(acc_vec3, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          ae_int16x4 d1 = AE_SAT16X4(acc_vec0, acc_vec1);
          ae_int16x4 d2 = AE_SAT16X4(acc_vec2, acc_vec3);
          AE_S16_0_XP(AE_SEL16_6543(d1, d1), (ae_int16 *)p_dst0_0, 8);
          AE_S16_0_XP(AE_SEL16_5432(d1, d1), (ae_int16 *)p_dst1_0, 8);
          AE_S16_0_XP(AE_SEL16_4321(d1, d1), (ae_int16 *)p_dst0_1, 8);
          AE_S16_0_XP(d1, (ae_int16 *)p_dst1_1, 8);
          AE_S16_0_XP(AE_SEL16_6543(d2, d2), (ae_int16 *)p_dst0_2, 8);
          AE_S16_0_XP(AE_SEL16_5432(d2, d2), (ae_int16 *)p_dst1_2, 8);
          AE_S16_0_XP(AE_SEL16_4321(d2, d2), (ae_int16 *)p_dst0_3, 8);
          AE_S16_0_XP(d2, (ae_int16 *)p_dst1_3, 8);
        }
        p_dst0_0 += out_channels;
        p_dst0_1 += out_channels;
        p_dst0_2 += out_channels;
        p_dst0_3 += out_channels;
        p_dst1_0 += out_channels;
        p_dst1_1 += out_channels;
        p_dst1_2 += out_channels;
        p_dst1_3 += out_channels;
      }
    }
  return 0;
}

/* This helper function aligns each row of filter (kernel) to 4-byte boundary. This aligns all filter loads */
static VOID* align_weightbuffer_rows(VOID *p_scratch /*dest*/, const WORD8 *p_kernel /*src*/, int oc, int kh , int kw, int kc)
{
    WORD8 *p_dst_orig = ALIGNED_ADDR(p_scratch, ALIGNMENT);
    WORD8 *p_dst = p_dst_orig;
    const WORD8 *p_src = p_kernel;

    int row_length     = kw*kc;
    int row_length_pad = PADDED_SIZE(kw*kc, 4);

    if( row_length_pad == row_length){
      return (VOID *)p_kernel;
    }

    int i, ii, l;

    for(l = 0; l < oc; l++){
        for(i = 0; i < kh; i++) {
            for(ii = 0; ii < row_length; ii++){
              *p_dst = *p_src;
              p_dst++;
              p_src++;
            }
            for(ii = 0; ii < (row_length_pad - row_length); ii++) {
              *p_dst++ = 0;
            }
        }
    }

    return p_dst_orig;
}

static WORD32 xa_nn_conv2d_std_per_chan_sym8sxsym16s_no_circ_buf_vec_unaligned(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD64* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    WORD32 out_activation_min,
    WORD32 out_activation_max
    )
{
    const WORD16 *p_dst0_0 = p_out + 0;
    const WORD16 *p_dst0_1 = p_out + 1;
    const WORD16 *p_dst0_2 = p_out + 2;
    const WORD16 *p_dst0_3 = p_out + 3;
    const WORD16 *p_dst1_0 = p_out + out_channels + 0;
    const WORD16 *p_dst1_1 = p_out + out_channels + 1;
    const WORD16 *p_dst1_2 = p_out + out_channels + 2;
    const WORD16 *p_dst1_3 = p_out + out_channels + 3;

    int kernel_out_ch_offset = kernel_height * PADDED_SIZE(kernel_width * input_channels,4);
    int input_x_offset = (input_channels * x_stride)*sizeof(WORD16); /* value in bytes*/
    int p_inp_vec_stride = (input_width * input_channels)*sizeof(WORD16); /* value in bytes */
    /* Pad kernel horizontal slice to ensure 4-byte alignement */
    int p_kern_vec_stride = PADDED_SIZE(kernel_width * input_channels, 4); 
    /* vec_len is also padded to allow over-run in core-loop, zero padding of kernel ensures correct output*/
    int vec_len = PADDED_SIZE(kernel_width * input_channels, 4); 

    for (int out_y = 0; out_y < out_height; ++out_y) {
      for (int out_x = 0; out_x < out_width; out_x += 2) {
        for (int out_ch = 0; out_ch < out_channels; out_ch += 4) {
          ae_int64 out0_0 = p_bias[out_ch + 0];
          ae_int64 out0_1 = p_bias[out_ch + 1];
          ae_int64 out0_2 = p_bias[out_ch + 2];
          ae_int64 out0_3 = p_bias[out_ch + 3];
          ae_int64 out1_0 = p_bias[out_ch + 0];
          ae_int64 out1_1 = p_bias[out_ch + 1];
          ae_int64 out1_2 = p_bias[out_ch + 2];
          ae_int64 out1_3 = p_bias[out_ch + 3];
          out0_0 = AE_SLAI64(out0_0, 8);
          out0_1 = AE_SLAI64(out0_1, 8);
          out0_2 = AE_SLAI64(out0_2, 8);
          out0_3 = AE_SLAI64(out0_3, 8);
          out1_0 = AE_SLAI64(out1_0, 8);
          out1_1 = AE_SLAI64(out1_1, 8);
          out1_2 = AE_SLAI64(out1_2, 8);
          out1_3 = AE_SLAI64(out1_3, 8);
          
          int in_x_o = out_x * x_stride;
          int in_y_o = out_y * y_stride - y_padding;
          int k_y_min = -in_y_o;
          int k_y_max = input_height - in_y_o;
          k_y_min = (k_y_min < 0) ? 0 : k_y_min;
          k_y_min = (k_y_min < kernel_height) ? k_y_min : kernel_height;
          k_y_max = (k_y_max < 0) ? 0 : k_y_max;
          k_y_max = (k_y_max < kernel_height) ? k_y_max : kernel_height;
          const ae_int16x4 *p_inp_vec =
              (ae_int16x4 *)&p_inp[((in_y_o + k_y_min) * input_width + in_x_o) *
                                      input_channels +
                                  0];
          const WORD8 *p_kern_vec =
              &p_kernel[(out_ch * kernel_height + k_y_min) * PADDED_SIZE(kernel_width * input_channels,4)
                        ];

          for (int k_y = k_y_min; k_y < k_y_max; ++k_y) {
            const ae_int16x4 *p_inp_vec0 = p_inp_vec;
            const ae_int16x4 *p_inp_vec1 = (ae_int16x4 *)((WORD8 *)p_inp_vec + input_x_offset);
            const WORD8 *p_kern_vec0 = p_kern_vec;
            const WORD8 *p_kern_vec1 = p_kern_vec0 + kernel_out_ch_offset;
            const WORD8 *p_kern_vec2 = p_kern_vec1 + kernel_out_ch_offset;
            const WORD8 *p_kern_vec3 = p_kern_vec2 + kernel_out_ch_offset;
            p_inp_vec = (ae_int16x4 *)((WORD8*)p_inp_vec +  p_inp_vec_stride);
            p_kern_vec += p_kern_vec_stride;
            ae_int16x4 d_inp0;
            ae_int16x4 d_inp1;
            ae_valign vec0_align = AE_LA64_PP(p_inp_vec0);
            ae_valign vec1_align = AE_LA64_PP(p_inp_vec1);

            ae_int16x4 d_kern0, d_kern1, d_kern2, d_kern3;
            for (int i = 0; i < vec_len; i += 4) {
              AE_LA16X4_IP(d_inp0, vec0_align, p_inp_vec0);
              AE_LA16X4_IP(d_inp1, vec1_align, p_inp_vec1);
              AE_L8X4F_IP(d_kern0, p_kern_vec0, 4);
              AE_L8X4F_IP(d_kern1, p_kern_vec1, 4);
              AE_L8X4F_IP(d_kern2, p_kern_vec2, 4);
              AE_L8X4F_IP(d_kern3, p_kern_vec3, 4);
              AE_MULAAAAQ16(out0_0, d_inp0, d_kern0);
              AE_MULAAAAQ16(out0_1, d_inp0, d_kern1);
              AE_MULAAAAQ16(out0_2, d_inp0, d_kern2);
              AE_MULAAAAQ16(out0_3, d_inp0, d_kern3);
              AE_MULAAAAQ16(out1_0, d_inp1, d_kern0);
              AE_MULAAAAQ16(out1_1, d_inp1, d_kern1);
              AE_MULAAAAQ16(out1_2, d_inp1, d_kern2);
              AE_MULAAAAQ16(out1_3, d_inp1, d_kern3);
            }
          }
          out0_0 = AE_SRAI64(out0_0, 8);
          out0_1 = AE_SRAI64(out0_1, 8);
          out0_2 = AE_SRAI64(out0_2, 8);
          out0_3 = AE_SRAI64(out0_3, 8);
          out1_0 = AE_SRAI64(out1_0, 8);
          out1_1 = AE_SRAI64(out1_1, 8);
          out1_2 = AE_SRAI64(out1_2, 8);
          out1_3 = AE_SRAI64(out1_3, 8);
#if XCHAL_HAVE_HIFI1S
          WORD32 out_shift_0 = 15 - p_out_shift[out_ch+0];
          out_shift_0 = ( out_shift_0  << 16 ) | out_shift_0;
          WORD32 out_shift_1 = 15 - p_out_shift[out_ch+1];
          out_shift_1 = ( out_shift_1  << 16 ) | out_shift_1;
          WORD32 out_shift_2 = 15 - p_out_shift[out_ch+2];
          out_shift_2 = ( out_shift_2  << 16 ) | out_shift_2;
          WORD32 out_shift_3 = 15 - p_out_shift[out_ch+3];
          out_shift_3 = ( out_shift_3  << 16 ) | out_shift_3;
#else
          WORD32 out_shift_0 = p_out_shift[out_ch+0];
          WORD32 out_shift_1 = p_out_shift[out_ch+1];
          WORD32 out_shift_2 = p_out_shift[out_ch+2];
          WORD32 out_shift_3 = p_out_shift[out_ch+3];
#endif

          ae_int32x2 acc_vec0 = MultiplyByQuantizedMultiplier_x2_opt(
              out0_0, out1_0, p_out_multiplier[out_ch + 0],
              out_shift_0);
          ae_int32x2 acc_vec1 = MultiplyByQuantizedMultiplier_x2_opt(
              out0_1, out1_1, p_out_multiplier[out_ch + 1],
              out_shift_1);
          ae_int32x2 acc_vec2 = MultiplyByQuantizedMultiplier_x2_opt(
              out0_2, out1_2, p_out_multiplier[out_ch + 2],
              out_shift_2);
          ae_int32x2 acc_vec3 = MultiplyByQuantizedMultiplier_x2_opt(
              out0_3, out1_3, p_out_multiplier[out_ch + 3],
              out_shift_3);
          AE_MINMAX32(acc_vec0, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          AE_MINMAX32(acc_vec1, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          AE_MINMAX32(acc_vec2, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          AE_MINMAX32(acc_vec3, AE_MOVDA32(out_activation_min), AE_MOVDA32(out_activation_max));
          ae_int16x4 d1 = AE_SAT16X4(acc_vec0, acc_vec1);
          ae_int16x4 d2 = AE_SAT16X4(acc_vec2, acc_vec3);
          AE_S16_0_XP(AE_SEL16_6543(d1, d1), (ae_int16 *)p_dst0_0, 8);
          AE_S16_0_XP(AE_SEL16_5432(d1, d1), (ae_int16 *)p_dst1_0, 8);
          AE_S16_0_XP(AE_SEL16_4321(d1, d1), (ae_int16 *)p_dst0_1, 8);
          AE_S16_0_XP(d1, (ae_int16 *)p_dst1_1, 8);
          AE_S16_0_XP(AE_SEL16_6543(d2, d2), (ae_int16 *)p_dst0_2, 8);
          AE_S16_0_XP(AE_SEL16_5432(d2, d2), (ae_int16 *)p_dst1_2, 8);
          AE_S16_0_XP(AE_SEL16_4321(d2, d2), (ae_int16 *)p_dst0_3, 8);
          AE_S16_0_XP(d2, (ae_int16 *)p_dst1_3, 8);
        }
        p_dst0_0 += out_channels;
        p_dst0_1 += out_channels;
        p_dst0_2 += out_channels;
        p_dst0_3 += out_channels;
        p_dst1_0 += out_channels;
        p_dst1_1 += out_channels;
        p_dst1_2 += out_channels;
        p_dst1_3 += out_channels;
      }
    }
  return 0;
}



static inline WORD32 gcd(WORD32 a, WORD32 b)
{
    while (a != b)
    {
        if (a > b)
        {
            return gcd(a - b, b);
        }
        else
        {
            return gcd(a, b - a);
        }
    }
    return a;
}

WORD32 xa_nn_dilated_conv2d_std_v2_per_chan_sym8sxsym16s(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD64* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch,
    WORD32 dilation_height,
    WORD32 dilation_width,
    WORD32 out_activation_min,
    WORD32 out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
  WORD16* __restrict__ p_out_base;
  p_out_base = p_out;
  WORD32 circMatrixHeight = 0;

  if(kernel_height==1)
    dilation_height = 1;
  if(kernel_width==1)
    dilation_width = 1;

  WORD32 kernel_height_dilation = kernel_height + ( (dilation_height-1) * (kernel_height-1) );//dilation
  WORD32 kernel_width_dilation = kernel_width + ( (dilation_width-1) * (kernel_width-1) );//dilation
   /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_height <= 0 || dilation_width <= 0), -1);//dilation
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);
  /* MinMax activation check */
  XA_NNLIB_ARG_CHK_COND((out_activation_min < -32768) || (out_activation_min > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < -32768) || (out_activation_max > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  WORD32 input_bytewidth = 2;
  VOID *pp_inp = (VOID *)p_inp;
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  WORD32 out_channels_offset = out_data_format ? out_height * out_width : 1;
  WORD32 out_height_offset = out_data_format ? out_width : out_width * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;
  WORD32 x_padding_var = x_padding;
  WORD32 dilation_w_offset, dilation_h_offset;
  WORD32 out_iteraions;
  
#if !ENABLE_PADDING_CONV2D_STD
  WORD32 input_channels_pad = input_channels;
#else
  WORD32 input_channels_pad = PADDED_SIZE(input_channels, (ALIGNMENT>>1));
#endif

  // Initialize start of the circular buffer
  xa_nn_conv2d_dilation_init_state((void*)p_state,(void*)p_kernel, (void*)pp_inp);

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
    if(x_padding_var >= kernel_width_dilation)//dilation
  {
    out_width_over_x_pad = conv_x_left_pad(x_padding, kernel_width_dilation, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_activation_min, out_activation_max);
    x_padding_var -= out_width_over_x_pad * x_stride;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = kernel_width_dilation + (out_width - 1) * x_stride - (x_padding + input_width);//dilation
  XA_NNLIB_ARG_CHK_COND((x_r_pad<0), -1);
  if(x_r_pad >= kernel_width_dilation)//dilation
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_padding, input_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_activation_min, out_activation_max);
  }


  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height_dilation + (out_height - 1) * y_stride - (y_padding + input_height);
  XA_NNLIB_ARG_CHK_COND((y_b_pad<0), -1);

  XA_NNLIB_ARG_CHK_COND((kernel_height_dilation > ( y_padding + input_height + y_b_pad)), -1);//dilation
  XA_NNLIB_ARG_CHK_COND((kernel_width_dilation  > ( x_padding + input_width  + x_r_pad)), -1);//dilation

  for(dilation_w_offset =0; dilation_w_offset<dilation_width; dilation_w_offset++ )
  {
	  /// Calculate number of left padding zeros for this particular width offset
	  WORD32 x_padding_dilation_initial_pad = ((x_padding-x_padding_var)/dilation_width) + (WORD32) ( (((x_padding-x_padding_var)%dilation_width)-1) >= dilation_w_offset); /// This offset's contribution which has been absorbed in initial analysis of zero padding

	  WORD32 x_stride_dilated = x_stride / gcd(x_stride, dilation_width);

	  WORD32 widthIndexIteration, firstWidthIndexNr, firstWidthIndex;
	  ///Check whether for a given width offset if there does exist a first column/width entry in this sub-matrix; if there are no width entries skip the entire row-offset
	  for(widthIndexIteration=0;widthIndexIteration<x_stride_dilated;widthIndexIteration++)
	  {
		  firstWidthIndexNr = (dilation_w_offset + (widthIndexIteration * dilation_width));
		  firstWidthIndex = firstWidthIndexNr / x_stride;
		  if(firstWidthIndex*x_stride == firstWidthIndexNr)
			  break;
	  }
	  if(widthIndexIteration==x_stride_dilated) //No more iterations for this width offset as the first index does not exist
		  continue;

	  WORD32 adjustZpAndOffsetIndex;// In the sub-matrix some of the initial left padding values might be consumed by conv_x_left_pad() function and then there is an index offset. The index offset is a value which is oblivious to zero padding or input matrix or so on.
	  /// This is the number of points that needs to be skipped in the sub-matrix for the first convolution to happen in polyphase. There is a chance that conv_x_left_pad() consumed more or less than this offset. In either case the pointer has to be appropriately adjusted for this so as to consume the correct point in convolution
	  //// The variable "adjustZpAndOffsetIndex" is the new offset keeping in mind both the initial offset for sub-matrix and number of points consumed in conv_x_left_pad(). This becomse the new offset even inside circular matrix loading function from where the convolution is to begin
	  if(x_padding_dilation_initial_pad <= widthIndexIteration)
		  adjustZpAndOffsetIndex = widthIndexIteration - x_padding_dilation_initial_pad;
	  else
	  {
		  adjustZpAndOffsetIndex = (  (x_padding_dilation_initial_pad - widthIndexIteration) /  x_stride_dilated  );// This is floor ;
		  adjustZpAndOffsetIndex = adjustZpAndOffsetIndex + (((x_padding_dilation_initial_pad - widthIndexIteration) - (adjustZpAndOffsetIndex*x_stride_dilated))>0);/// ceil implementation
		  adjustZpAndOffsetIndex = widthIndexIteration + (adjustZpAndOffsetIndex * x_stride_dilated);
		  adjustZpAndOffsetIndex = adjustZpAndOffsetIndex - x_padding_dilation_initial_pad;
	  }


	  //// Calculations for out points for this width offset
	  WORD32 totalPointsParticipatingInConvolution = x_padding + input_width + (x_r_pad - (out_width_over_x_r_pad*x_stride));//Note:x_padding is added here and not x_padding_var because this is discounted later by sub x_padding_dilation_initial_pad
	  WORD32 pointsParticipatingInConvolutionForThisOffset = ( (totalPointsParticipatingInConvolution)/dilation_width) + (WORD32) ( (((totalPointsParticipatingInConvolution)%dilation_width)-1) >= dilation_w_offset);
	  pointsParticipatingInConvolutionForThisOffset = pointsParticipatingInConvolutionForThisOffset - x_padding_dilation_initial_pad;

	  if(  (pointsParticipatingInConvolutionForThisOffset - adjustZpAndOffsetIndex) < kernel_width) // After identifying the first index value check if there are enough points to convolve;if not break again; There is also no reason to check for higher values of firstIndex further
	  		continue;
	  WORD32 out_points_for_this_xyoffset = ((pointsParticipatingInConvolutionForThisOffset - adjustZpAndOffsetIndex) - kernel_width)/x_stride_dilated + 1;/// This represents total num of times the conv needs to be called

	  for(dilation_h_offset =0; dilation_h_offset<dilation_height; dilation_h_offset++ )
	  {
		  {

			  WORD32 input_padding_consumed =0;
			  WORD32 input_width_consumed = 0;

			  WORD32 y_stride_dilated = y_stride / gcd(y_stride, dilation_height); // This is the new stride value in height dimension
			  ///Check whether for a given height offset if there does exist a height entry in this sub-matrix;
			  ///if there are no width entries skip the entire height-offset
			  WORD32 heightIndexIteration, firstHeightIndexNr,firstHeightIndex ;
			  for(heightIndexIteration=0;heightIndexIteration<y_stride_dilated;heightIndexIteration++)
			  {
				  firstHeightIndexNr = (dilation_h_offset + (heightIndexIteration * dilation_height));
				  firstHeightIndex = firstHeightIndexNr / y_stride;
				  if(firstHeightIndex*y_stride == firstHeightIndexNr)
					  break;
			  }

			  WORD32 heightOfCircMatrix = ((y_padding + input_height + y_b_pad)/dilation_height) + (WORD32) ((((y_padding + input_height + y_b_pad)%dilation_height)-1)>=dilation_h_offset);// Height of circular matrix for a given offset value
			  if(heightIndexIteration==y_stride_dilated) //No more iterations for this height offset as the first index does not exist
				  continue;
			  else if( (heightOfCircMatrix- heightIndexIteration) < kernel_height) // After identifying the first index value check if there are enough points to convolve;if not break again; There is also no reason to check for higher values of firstIndex further
				  continue;

			  /// Initialize circular buffer end/height/size based on the dilation offset
			  xa_nn_dilated_conv2d_std_init_circ_buf(
                  (void*)p_state, (void*)p_kernel,
                   input_height, input_channels,
                   kernel_height_dilation, kernel_width,
                   y_stride, y_padding,
                   out_height, out_channels,
                   dilation_height, dilation_h_offset,
                   PREC_SYM16S, PREC_SYM8S); //dilation

#ifdef polyphase_debug
			  p_buff_circ_deb = p_state->cir_buf.p_curr;
#endif
			  WORD32 planesToAdd = (kernel_width - x_stride_dilated);
			  planesToAdd = (planesToAdd>0) ? planesToAdd : 0;
			  xa_nn_dilated_conv2d_std_load_cir_buf(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, (VOID**)&pp_inp, p_state, -input_zero_bias, dilation_height, dilation_h_offset, dilation_width, dilation_w_offset, x_padding, &input_padding_consumed, &input_width_consumed, planesToAdd,1,&circMatrixHeight, adjustZpAndOffsetIndex, x_stride_dilated, heightIndexIteration);

			  WORD32 outPointerHeightOffset = (dilation_h_offset + (heightIndexIteration*dilation_height) ) / y_stride; // In stride =1 case heightIndexIteration = 0// Refer to the PPT slide refering to formula to stich back matrix;last but 2 slide in PPT
			  WORD32 outPointerWidthOffset = (((x_padding_dilation_initial_pad + adjustZpAndOffsetIndex) * dilation_width) + dilation_w_offset) / x_stride;// The two addition terms take us to the point where conv. is going to start in this width_offset. Multiplication with dilation_width translates the same in linear domain. Adding the offset takes it to the right spot in the input matrix. Dividing by stride gives us the output width point
			  
			  p_out = p_out_base + ( outPointerHeightOffset * out_height_offset) + (outPointerWidthOffset*out_width_offset);//(dilation_w_offset * out_width_offset) + ( (left_pad_offset+out_width_over_x_pad) * out_width_offset);

#ifdef polyphase_debug
			  p_buff_circ_deb = p_state->cir_buf.p_curr;
#endif

			  //out_points_for_this_xyoffset = 0;//To be removed for debugging purpose
			  for(out_iteraions = 0;out_iteraions<out_points_for_this_xyoffset;out_iteraions++)
			  {
				  planesToAdd = x_stride_dilated;
				  if(planesToAdd>kernel_width)
					  planesToAdd = kernel_width;
    				  xa_nn_dilated_conv2d_std_load_cir_buf(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, (VOID**)&pp_inp, p_state, -input_zero_bias, dilation_height, dilation_h_offset, dilation_width, dilation_w_offset, x_padding, &input_padding_consumed, &input_width_consumed, planesToAdd,0,&circMatrixHeight, adjustZpAndOffsetIndex, x_stride_dilated, heightIndexIteration);

				    // Convolution using matXvec with matrix as circular buffer
				    xa_nn_matXvec_sym8sxsym16s_sym16s_circ
				      (p_out /* output */
				       ,p_state->cir_buf.p_curr/* matrix: rows x cols */
				       ,p_state->p_kernel_padded /* vec: cols */
				       ,p_bias /* bias */
				       ,((circMatrixHeight-kernel_height)/y_stride_dilated)+1//out_height /* rows */
				       ,input_channels_pad * kernel_width * kernel_height /* cols */
				       ,input_channels_pad * kernel_width * y_stride_dilated/* row_offset */
				       ,out_channels /* vec_count */
				       ,input_channels_pad * kernel_width * kernel_height /* vec_stride */
				       ,out_channels_offset /* out_col_offset */
				       ,out_height_offset * dilation_height /gcd(y_stride, dilation_height)  /* out_row_offset *//// mul by dilation_height
				       ,p_out_multiplier
				       ,p_out_shift
				       ,out_activation_min
				       ,out_activation_max
				       ,p_dma_cfg
				      );
				    p_out += (out_width_offset*dilation_width / gcd(x_stride, dilation_width) );//Mul by dilation width
			  }

		  }
	  }
  }
  return 0;
}

WORD32 xa_nn_dilated_conv2d_std_per_chan_sym8sxsym16s(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD64* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch,
    WORD32 dilation_height,
    WORD32 dilation_width)
{
  int ret;
  ret = xa_nn_dilated_conv2d_std_v2_per_chan_sym8sxsym16s(p_out, p_inp, p_kernel, p_bias,
                input_height, input_width, input_channels, kernel_height, kernel_width, out_channels, 
                x_stride, y_stride, x_padding, y_padding, out_height, out_width,
                input_zero_bias, p_out_multiplier, p_out_shift, out_zero_bias, out_data_format, p_scratch,
                dilation_height, dilation_width, -32768, 32767, NULL);
  return ret;
}

WORD32 xa_nn_conv2d_std_v2_per_chan_sym8sxsym16s(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD64* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch,
    WORD32 out_activation_min,
    WORD32 out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
   /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_min < -32768) || (out_activation_min > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < -32768) || (out_activation_max > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 15), -1);
  }

  if ( !(x_padding) && !(input_channels & 0x3) && !(out_channels & 0x3) && !(out_width & 0x1) && (out_data_format == 0) && ((out_width-1)*x_stride <=(input_width-kernel_width) ) && p_bias)
  {
    int ret_val=0;
    ret_val=xa_nn_conv2d_std_per_chan_sym8sxsym16s_no_circ_buf(p_out,
                                                              p_inp,
                                                              p_kernel,
                                                              p_bias,
                                                              input_height,
                                                              input_width,
                                                              input_channels,
                                                              kernel_height,
                                                              kernel_width,
                                                              out_channels,
                                                              x_stride,
                                                              y_stride,
                                                              x_padding,
                                                              y_padding,
                                                              out_height,
                                                              out_width,
                                                              input_zero_bias,
                                                              p_out_multiplier,
                                                              p_out_shift,
                                                              out_zero_bias,
                                                              out_data_format,
                                                              out_activation_min,
                                                              out_activation_max
                                                            );

    return ret_val;
  }

  if ( !(x_padding) && (input_channels == 2) && !(out_channels & 0x3) && !(out_width & 0x1) && (out_data_format == 0) && ((out_width-1)*x_stride <=(input_width-kernel_width) ) && p_bias)
  {
    int ret_val=0;
    VOID *p_kernel_padded = align_weightbuffer_rows(p_scratch /*dest*/, p_kernel /*src*/, out_channels, kernel_height, kernel_width, input_channels);
    ret_val=xa_nn_conv2d_std_per_chan_sym8sxsym16s_no_circ_buf_vec_unaligned(p_out,
                                                              p_inp,
                                                              p_kernel_padded,
                                                              p_bias,
                                                              input_height,
                                                              input_width,
                                                              input_channels,
                                                              kernel_height,
                                                              kernel_width,
                                                              out_channels,
                                                              x_stride,
                                                              y_stride,
                                                              x_padding,
                                                              y_padding,
                                                              out_height,
                                                              out_width,
                                                              input_zero_bias,
                                                              p_out_multiplier,
                                                              p_out_shift,
                                                              out_zero_bias,
                                                              out_data_format,
                                                              out_activation_min,
                                                              out_activation_max
                                                            );

    return ret_val;
  }

  WORD32 j;
  WORD32 input_bytewidth = 2;
  VOID *pp_inp = (VOID *)p_inp;

  p_scratch = ALIGNED_ADDR(p_scratch, ALIGNMENT);
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  WORD32 inp_h, inp_w, ker_h, ker_w, x_str, y_str, x_pad, y_pad, out_h, out_w;

  if ((input_height == 1) && (kernel_height == 1) && (out_height == 1))
  {
    inp_h = input_width;
    inp_w = input_height;
    ker_h = kernel_width;
    ker_w = kernel_height;
    x_str = y_stride;
    y_str = x_stride;
    x_pad = y_padding;
    y_pad = x_padding;
    out_h = out_width;
    out_w = out_height;
  }
  else
  {
    inp_h = input_height;
    inp_w = input_width;
    ker_h = kernel_height;
    ker_w = kernel_width;
    x_str = x_stride;
    y_str = y_stride;
    x_pad = x_padding;
    y_pad = y_padding;
    out_h = out_height;
    out_w = out_width;
  }

  xa_nn_conv2d_std_init_state((void*)p_state
    ,(void*)p_kernel
    ,inp_h
    ,input_channels
    ,ker_h
    ,ker_w
    ,y_str
    ,y_pad
    ,out_h
    ,out_channels
    ,PREC_SYM16S
    ,PREC_SYM8S);

  WORD32 out_channels_offset = out_data_format ? out_h * out_w : 1;
  WORD32 out_height_offset = out_data_format ? out_w : out_w * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;

  WORD32 x_padding_var = x_pad;

#if !ENABLE_PADDING_CONV2D_STD
  WORD32 input_channels_pad = input_channels;
#else
  WORD32 input_channels_pad; 
  input_channels_pad = PADDED_SIZE(input_channels, (ALIGNMENT>>1));
#endif

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
  if(x_padding_var >= ker_w)
  {
    out_width_over_x_pad = conv_x_left_pad(x_pad, ker_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_activation_min, out_activation_max);
    x_padding_var -= out_width_over_x_pad * x_str;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = ker_w + (out_w - 1) * x_str - (x_pad + inp_w);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= ker_w)
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_pad, inp_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_activation_min, out_activation_max);
  }

  /* When kernel convolves over input region */
  p_out += out_width_over_x_pad * out_width_offset;
  // Initialize circular buffer
  // Determine y-bottom padding
  WORD32 y_b_pad = ker_h + (out_h - 1) * y_str - (y_pad + inp_h);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  conv2d_std_init_cir_buf(input_channels, input_channels_pad, input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, p_state);

  // Index to padded input width
  WORD32 idx_beg_inp_width_pad = ker_w - x_str;
  idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;


  // Process Loop to compute one output plane [out_h x out_channels] per iteration
  for(j=0;j<out_w-out_width_over_x_pad-out_width_over_x_r_pad;j++)
  {
    // Add x_str x (inp_h x input_channels) new planes to circular buffer
    conv2d_std_update_cir_buf(input_channels, input_channels_pad, input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state);

    // Update index to input width padded
    idx_beg_inp_width_pad += x_str;

    // Convolution using matXvec with matrix as circular buffer
    xa_nn_matXvec_sym8sxsym16s_sym16s_circ
      (p_out /* output */
       ,p_state->cir_buf.p_curr/* matrix: rows x cols */
       ,p_state->p_kernel_padded /* vec: cols */
       ,p_bias /* bias */
       ,out_h /* rows */
       ,PADDED_SIZE(input_channels_pad * ker_w * ker_h, 4) /* cols */
       ,input_channels_pad * ker_w * y_str /* row_stride */
       ,out_channels /* vec_count */
       ,PADDED_SIZE(input_channels_pad * ker_w * ker_h,4) /* vec_stride */
       ,out_channels_offset /* out_col_offset */
       ,out_height_offset /* out_row_offset */
       ,p_out_multiplier
       ,p_out_shift
       ,out_activation_min
       ,out_activation_max
       ,p_dma_cfg
      );
    p_out += out_width_offset;
  }

  return 0;
}


WORD32 xa_nn_conv2d_std_per_chan_sym8sxsym16s(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD64* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch)
{
  int ret;
  ret = xa_nn_conv2d_std_v2_per_chan_sym8sxsym16s(p_out, p_inp, p_kernel, p_bias,
                input_height, input_width, input_channels, kernel_height, kernel_width, out_channels,
                x_stride, y_stride, x_padding, y_padding, out_height, out_width,
                input_zero_bias, p_out_multiplier, p_out_shift, out_zero_bias, out_data_format, p_scratch,
                -32768, 32767, NULL);
  return ret;

}
