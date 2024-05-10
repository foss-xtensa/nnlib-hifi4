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
  q = AE_SRAA64(q, (-shift-17));
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
  q1 = AE_SRAA64(q1, (-shift-17));
  q2 = AE_SRAA64(q2, (-shift-17));
  ae_int32x2 result = AE_ROUND32X2F64SASYM(q1, q2);
  return result;
}
#endif

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
    WORD32 out_data_format
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
    WORD32 out_data_format
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
    WORD32 * p_out_shift)
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
    WORD32 * p_out_shift)
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
	d1 = AE_SAT16X4(acc, acc);
	AE_S16_0_XP(d1, ptrout, out_channels_offset*sizeof(WORD16));
      }
    }
  }
  return out_width_over_x_r_pad;
}

WORD32 xa_nn_conv2d_per_chan_sym8sxsym16s(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD64* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 kernel_channels,
    WORD32 dilation_height,
    WORD32 dilation_width,
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
  XA_NNLIB_ARG_CHK_COND((kernel_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  //XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  //XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_height != 1), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_width != 1), -1);

  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 15), -1);
  }
    const int groups = input_channels/kernel_channels;
  XA_NNLIB_ARG_CHK_COND((groups<=0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_channels %kernel_channels)!=0),-1);
  XA_NNLIB_ARG_CHK_COND(((out_channels%groups)!=0),-1);
  const int kernels_per_group = out_channels / groups;
  XA_NNLIB_ARG_CHK_COND((kernels_per_group<=0),-1);

  if ( (groups == 1) && !(x_padding) && !(input_channels & 0x3) && !(out_channels & 0x3) && !(out_width & 0x1) && (out_data_format == 0) && ((out_width-1)*x_stride <=(input_width-kernel_width) ) && p_bias)
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
                                                              out_data_format
                                                            );

    return ret_val;
  }

  if ( (groups == 1) && !(x_padding) && (input_channels == 2) && !(out_channels & 0x3) && !(out_width & 0x1) && (out_data_format == 0) && ((out_width-1)*x_stride <=(input_width-kernel_width) ) && p_bias)
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
                                                              out_data_format
                                                            );

    return ret_val;
  }

  WORD32 j;
  WORD32 input_bytewidth = 2;
  VOID *pp_inp = (VOID *)p_inp;
  WORD16* __restrict__ tmp_out;

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
      ,kernel_channels
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
  WORD32 kernel_channels_pad;

#if !ENABLE_PADDING_CONV2D_STD
  kernel_channels_pad = kernel_channels;
#else
  kernel_channels_pad = PADDED_SIZE(kernel_channels, (ALIGNMENT>>1));
#endif

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
  if(x_padding_var >= ker_w)
  {
    out_width_over_x_pad = conv_x_left_pad(x_pad, ker_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift);
    x_padding_var -= out_width_over_x_pad * x_str;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = ker_w + (out_w - 1) * x_str - (x_pad + inp_w);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= ker_w)
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_pad, inp_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift);
  }

  /* When kernel convolves over input region */
  p_out += out_width_over_x_pad * out_width_offset;
  // Initialize circular buffer
  // Determine y-bottom padding
  WORD32 y_b_pad = ker_h + (out_h - 1) * y_str - (y_pad + inp_h);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  for (int grp_i = 0; grp_i < groups; ++grp_i)
  {
    tmp_out=p_out+grp_i*kernels_per_group*out_channels_offset;
    xa_nn_conv2d_group_init_state((void*)p_state
        ,(void*)p_kernel
        ,inp_h
        ,kernel_channels
        ,ker_h
        ,ker_w
        ,y_str
        ,y_pad
        ,out_h
        ,out_channels
        ,PREC_SYM16S
        ,PREC_SYM8S);
    pp_inp = (VOID *)(p_inp+grp_i*kernel_channels);
    conv2d_group_init_cir_buf(input_channels, kernel_channels_pad, kernel_channels,input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, p_state,0);

    // Index to padded input width
    WORD32 idx_beg_inp_width_pad = ker_w - x_str;
    idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;


  // Process Loop to compute one output plane [out_h x out_channels] per iteration
    for(j=0;j<out_w-out_width_over_x_pad-out_width_over_x_r_pad;j++)
    {
      // Add x_str x (inp_h x input_channels) new planes to circular buffer
      conv2d_group_update_cir_buf(input_channels, kernel_channels_pad,kernel_channels, input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state,0);

      // Update index to input width padded
      idx_beg_inp_width_pad += x_str;

      // Convolution using matXvec with matrix as circular buffer
      const WORD64 *p_bias_grp = NULL;
      if(p_bias != NULL){
        p_bias_grp = p_bias+grp_i*kernels_per_group;
      }
      xa_nn_matXvec_sym8sxsym16s_sym16s_circ
        (tmp_out /* output */
        ,p_state->cir_buf.p_curr/* matrix: rows x cols */
        ,(p_state->p_kernel_padded+grp_i*kernels_per_group*kernel_channels_pad*ker_w*ker_h) /* vec: cols */
        ,p_bias_grp /* bias */
        ,out_h /* rows */
        ,kernel_channels_pad * ker_w * ker_h /* cols */
        ,kernel_channels_pad * ker_w * y_str/* row_offset */
        ,kernels_per_group /* vec_count */
        ,kernel_channels_pad * ker_w * ker_h /* vec_stride */
        ,out_channels_offset /* out_col_offset */
        ,out_height_offset /* out_row_offset */
        ,(p_out_multiplier+grp_i*kernels_per_group)
        ,(p_out_shift+grp_i*kernels_per_group)
        );
      tmp_out += out_width_offset;
    }
  
  }

  return 0;
}
