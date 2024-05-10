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
#include "xa_type_def.h"
#include "xa_nn_common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"

#include "xa_nnlib_common.h"

#if 0 /* By default special case for 3x3 kernel is enabled */
  #define DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE
#endif

/* 2D Convolution implementation */
static inline void conv2d_nchw_sym8sxasym8s_hf4_convmul
(pWORD8 __restrict__ p_out  /* Output:  [Stream] [(out_stride): (actual_out_height): (actual_out_width)] */
 ,const WORD8 *__restrict__ p_ker  /* Kernel:  [Block] [1:             kernel_height:       kernel_width_pad] */
 ,const WORD8 *__restrict__ p_inp  /* Input:   [Block] [1:             input_height:        input_width] */
 ,WORD32 bias
 ,int input_height
 ,int input_width
 ,int kernel_height
 ,int kernel_width
 ,int actual_out_height      /* This is the actual output height, processing should be limited to it. */
 ,int actual_out_width       /* This is the actual output width, processing should be limited to it. */
 ,int out_stride
 ,int x_stride
 ,int y_stride
 ,WORD32  input_zero_bias
 ,WORD32  out_multiplier
 ,WORD32  out_shift
 ,WORD32  out_zero_bias
 ,pWORD32 __restrict__ p_scratch /* Scratch: [Block] [1:             (actual_out_height): (out_width)] */
 )
{
    /* Importance of actual_out_width, since we are appending zeros input left
     * and right side. No problem with left padding, but for right padding that
     * is done to make sure that input_width is multiple of 4. Here
     * 'output_width_for_x_stride_1' value is calculated based on this padded value. But
     * actually expected output width to pick correct values from 'output_width_for_x_stride_1' on
     * jumps of 'x_stride'. */

    int kernel_width_pad = (kernel_width+3)&(~3);
    int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);

    /* Generic case */
    int i, j, k, l;
    int output_height = input_height - kernel_height + 1;
    int output_width_for_x_stride_1;

    /* Here input_width is nothing but circ_buf_width, which is taken care to be
     * multiple of 4. */
    output_width_for_x_stride_1 = (1 + ((input_width - kernel_width)/1));
    /* output_width_for_x_stride_1 loop is unrolled by 4 so keeping this dimension to multiple of 4 */
    output_width_for_x_stride_1 = ALIGNED_SIZE(output_width_for_x_stride_1, (ALIGNMENT/2));

    /* Please note that below addition of 1 is done to adjust in C style indices
     * */
    if ((actual_out_height - 1) > ((output_height + 1) / (y_stride)))
    {
        return;
    }
    if ((actual_out_width - 1) > ((output_width_for_x_stride_1 + 1) / (x_stride)))
    {
        return;
    }

    ae_int64 accu_int64_0, accu_int64_1, accu_int64_2, accu_int64_3;
    ae_int32x2 *scratch_ptr = (ae_int32x2 *)p_scratch;

    ae_int16x4 d_input_zero_bias;
    d_input_zero_bias = AE_MOVDA16(input_zero_bias);
    ae_int32x2 _ae_int32_sat_bias;
    _ae_int32_sat_bias = AE_MOVDA32X2(bias, bias);

    if(kernel_width_pad==12)
    {
      ae_int16x4 d_inp00, d_inp01, d_inp02, d_inp03;
      ae_int16x4 d_ker0, d_ker1, d_ker2;
      ae_int16x4 d_inp1, d_inp2, d_inp3, d_inp4, d_inp5, d_inp6, d_inp7, d_inp8, d_inp9;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int32x2 *) (p_scratch + (i * output_width_for_x_stride_1));
#pragma loop_count min=1
        for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          const WORD8 *pt_ker = p_ker;
#pragma loop_count min=1
#pragma no_unroll
          for(k = 0; k < kernel_height; k++)
          {
            const WORD8 *pt_inp0 = (p_inp);
            AE_ADDCIRC16X4_XC
                ((ae_int16x4 *)pt_inp0
                  ,((sizeof(WORD8)) * ((i * y_stride * input_width) + j*4 + k*input_width))
                );
#if XCHAL_HAVE_HIFI1
            AE_L8X4S_IP(d_ker0, pt_ker, 4);
            AE_L8X4S_IP(d_ker1, pt_ker, 4);
            AE_L8X4S_IP(d_ker2, pt_ker, 4);
            AE_L8X4S_IP(d_inp00, pt_inp0, 4);
            AE_L8X4S_IP(d_inp01, pt_inp0, 4);
            AE_L8X4S_IP(d_inp02, pt_inp0, 4);
            AE_L8X4S_IP(d_inp03, pt_inp0, 4);
#else
            AE_L8X4F_IP(d_ker0, pt_ker, 4);
            AE_L8X4F_IP(d_ker1, pt_ker, 4);
            AE_L8X4F_IP(d_ker2, pt_ker, 4);
            AE_L8X4F_IP(d_inp00, pt_inp0, 4);
            AE_L8X4F_IP(d_inp01, pt_inp0, 4);
            AE_L8X4F_IP(d_inp02, pt_inp0, 4);
            AE_L8X4F_IP(d_inp03, pt_inp0, 4);
            d_ker0 = AE_SRAI16(d_ker0, 8);
            d_ker1 = AE_SRAI16(d_ker1, 8);
            d_ker2 = AE_SRAI16(d_ker2, 8);
            d_inp00 = AE_SRAI16(d_inp00, 8);
            d_inp01 = AE_SRAI16(d_inp01, 8);
            d_inp02 = AE_SRAI16(d_inp02, 8);
            d_inp03 = AE_SRAI16(d_inp03, 8);
#endif
            d_inp00 = AE_ADD16(d_inp00, d_input_zero_bias);
            d_inp01 = AE_ADD16(d_inp01, d_input_zero_bias);
            d_inp02 = AE_ADD16(d_inp02, d_input_zero_bias);
            d_inp03 = AE_ADD16(d_inp03, d_input_zero_bias);
            d_inp1 = AE_SEL16_6543(d_inp00, d_inp01);
            d_inp2 = AE_SEL16_5432(d_inp00, d_inp01);
            d_inp3 = AE_SEL16_4321(d_inp00, d_inp01);
            AE_MULAAAAQ16(accu_int64_0, d_inp00, d_ker0);
            AE_MULAAAAQ16(accu_int64_1, d_inp1, d_ker0);
            AE_MULAAAAQ16(accu_int64_2, d_inp2, d_ker0);
            AE_MULAAAAQ16(accu_int64_3, d_inp3, d_ker0);
            d_inp4 = AE_SEL16_6543(d_inp01, d_inp02);
            d_inp5 = AE_SEL16_5432(d_inp01, d_inp02);
            d_inp6 = AE_SEL16_4321(d_inp01, d_inp02);
            AE_MULAAAAQ16(accu_int64_0, d_inp01, d_ker1);
            AE_MULAAAAQ16(accu_int64_1, d_inp4, d_ker1);
            AE_MULAAAAQ16(accu_int64_2, d_inp5, d_ker1);
            AE_MULAAAAQ16(accu_int64_3, d_inp6, d_ker1);
            d_inp7 = AE_SEL16_6543(d_inp02, d_inp03);
            d_inp8 = AE_SEL16_5432(d_inp02, d_inp03);
            d_inp9 = AE_SEL16_4321(d_inp02, d_inp03);
            AE_MULAAAAQ16(accu_int64_0, d_inp02, d_ker2);
            AE_MULAAAAQ16(accu_int64_1, d_inp7, d_ker2);
            AE_MULAAAAQ16(accu_int64_2, d_inp8, d_ker2);
            AE_MULAAAAQ16(accu_int64_3, d_inp9, d_ker2);

          }
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_0), AE_MOVINT32X2_FROMINT64(accu_int64_1));
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_2), AE_MOVINT32X2_FROMINT64(accu_int64_3));
        }
      }
    }

    else if(kernel_width_pad==8)
    {
      ae_int16x4 d_inp00, d_inp01, d_inp02;
      ae_int16x4 d_ker0, d_ker1;
      ae_int16x4 d_inp1, d_inp2, d_inp3, d_inp4, d_inp5, d_inp6;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int32x2 *) (p_scratch + (i * output_width_for_x_stride_1));
#pragma loop_count min=1
        for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          const WORD8 *pt_ker = p_ker;
#pragma loop_count min=1
#pragma no_unroll
          for(k = 0; k < kernel_height; k++)
          {
            const WORD8 *pt_inp0 = (p_inp);
            AE_ADDCIRC16X4_XC
                ((ae_int16x4 *)pt_inp0
                  ,((sizeof(WORD8)) * ((i * y_stride * input_width) + j*4 + k*input_width))
                );
#if XCHAL_HAVE_HIFI1
            AE_L8X4S_IP(d_ker0, pt_ker, 4);
            AE_L8X4S_IP(d_ker1, pt_ker, 4);
            AE_L8X4S_IP(d_inp00, pt_inp0, 4);
            AE_L8X4S_IP(d_inp01, pt_inp0, 4);
            AE_L8X4S_IP(d_inp02, pt_inp0, 4);
#else
            AE_L8X4F_IP(d_ker0, pt_ker, 4);
            AE_L8X4F_IP(d_ker1, pt_ker, 4);
            AE_L8X4F_IP(d_inp00, pt_inp0, 4);
            AE_L8X4F_IP(d_inp01, pt_inp0, 4);
            AE_L8X4F_IP(d_inp02, pt_inp0, 4);
            d_ker0 = AE_SRAI16(d_ker0, 8);
            d_ker1 = AE_SRAI16(d_ker1, 8);
            d_inp00 = AE_SRAI16(d_inp00, 8);
            d_inp01 = AE_SRAI16(d_inp01, 8);
            d_inp02 = AE_SRAI16(d_inp02, 8);
#endif
            d_inp00 = AE_ADD16(d_inp00, d_input_zero_bias);
            d_inp01 = AE_ADD16(d_inp01, d_input_zero_bias);
            d_inp02 = AE_ADD16(d_inp02, d_input_zero_bias);
            d_inp1 = AE_SEL16_6543(d_inp00, d_inp01);
            d_inp2 = AE_SEL16_5432(d_inp00, d_inp01);
            d_inp3 = AE_SEL16_4321(d_inp00, d_inp01);
            AE_MULAAAAQ16(accu_int64_0, d_inp00, d_ker0);
            AE_MULAAAAQ16(accu_int64_1, d_inp1, d_ker0);
            AE_MULAAAAQ16(accu_int64_2, d_inp2, d_ker0);
            AE_MULAAAAQ16(accu_int64_3, d_inp3, d_ker0);
            d_inp4 = AE_SEL16_6543(d_inp01, d_inp02);
            d_inp5 = AE_SEL16_5432(d_inp01, d_inp02);
            d_inp6 = AE_SEL16_4321(d_inp01, d_inp02);
            AE_MULAAAAQ16(accu_int64_0, d_inp01, d_ker1);
            AE_MULAAAAQ16(accu_int64_1, d_inp4, d_ker1);
            AE_MULAAAAQ16(accu_int64_2, d_inp5, d_ker1);
            AE_MULAAAAQ16(accu_int64_3, d_inp6, d_ker1);

          }
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_0), AE_MOVINT32X2_FROMINT64(accu_int64_1));
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_2), AE_MOVINT32X2_FROMINT64(accu_int64_3));
        }
      }
    }

    else if(kernel_width_pad==4)
    {
      ae_int16x4 d_inp00, d_inp01;
      ae_int16x4 d_ker0;
      ae_int16x4 d_inp1, d_inp2, d_inp3;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int32x2 *) (p_scratch + (i * output_width_for_x_stride_1));
#pragma loop_count min=1
        for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          const WORD8 *pt_ker = p_ker;
#pragma loop_count min=1
#pragma no_unroll
          for(k = 0; k < kernel_height; k++)
          {
            const WORD8 *pt_inp0 = (p_inp);
            AE_ADDCIRC16X4_XC
                ((ae_int16x4 *)pt_inp0
                  ,((sizeof(WORD8)) * ((i * y_stride * input_width) + j*4 + k*input_width))
                );
#if XCHAL_HAVE_HIFI1
            AE_L8X4S_IP(d_ker0, pt_ker, 4);
            AE_L8X4S_IP(d_inp00, pt_inp0, 4);
            AE_L8X4S_IP(d_inp01, pt_inp0, 4);
#else
            AE_L8X4F_IP(d_ker0, pt_ker, 4);
            AE_L8X4F_IP(d_inp00, pt_inp0, 4);
            AE_L8X4F_IP(d_inp01, pt_inp0, 4);
            d_ker0 = AE_SRAI16(d_ker0, 8);
            d_inp00 = AE_SRAI16(d_inp00, 8);
            d_inp01 = AE_SRAI16(d_inp01, 8);
#endif
            d_inp00 = AE_ADD16(d_inp00, d_input_zero_bias);
            d_inp01 = AE_ADD16(d_inp01, d_input_zero_bias);
            d_inp1 = AE_SEL16_6543(d_inp00, d_inp01);
            d_inp2 = AE_SEL16_5432(d_inp00, d_inp01);
            d_inp3 = AE_SEL16_4321(d_inp00, d_inp01);
            AE_MULAAAAQ16(accu_int64_0, d_inp00, d_ker0);
            AE_MULAAAAQ16(accu_int64_1, d_inp1, d_ker0);
            AE_MULAAAAQ16(accu_int64_2, d_inp2, d_ker0);
            AE_MULAAAAQ16(accu_int64_3, d_inp3, d_ker0);

          }
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_0), AE_MOVINT32X2_FROMINT64(accu_int64_1));
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_2), AE_MOVINT32X2_FROMINT64(accu_int64_3));
        }
      }
    }

    else
    {
      for(i = 0; i < actual_out_height; i++)
      {
          scratch_ptr = (ae_int32x2 *) (p_scratch + (i * output_width_for_x_stride_1));
          for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
          {
              accu_int64_0 = AE_ZERO64();
              accu_int64_1 = AE_ZERO64();
              accu_int64_2 = AE_ZERO64();
              accu_int64_3 = AE_ZERO64();
#pragma loop_count min=1
              for(k = 0; k < kernel_height_pad; k += 2)
              {
                  const WORD8 *pt_inp0 = (p_inp);
                  AE_ADDCIRC16X4_XC
                      ((ae_int16x4 *)pt_inp0
                       ,((sizeof(WORD8)) * ((i * y_stride * input_width) + j*4 + k*input_width))
                      );
                  const WORD8 *pt_inp1 = (p_inp);
                  AE_ADDCIRC16X4_XC
                      ((ae_int16x4 *)pt_inp1
                       ,((sizeof(WORD8)) * ((i * y_stride * input_width) + j*4 + (k+1)*input_width))
                      );
                  const WORD8 *pt_ker0 = (p_ker + k*kernel_width_pad);
                  const WORD8 *pt_ker1 = (p_ker + (k+1)*kernel_width_pad);
                  ae_int16x4 d_ker0, d_ker1;
                  ae_int16x4 d_inp00, d_inp01, d_inp10, d_inp11, d_inp1, d_inp2, d_inp3, d_inp4, d_inp5, d_inp6;
#if XCHAL_HAVE_HIFI1
                  AE_L8X4S_IP(d_inp00, pt_inp0, 4);
                  AE_L8X4S_IP(d_inp10, pt_inp1, 4);
#else
                  AE_L8X4F_IP(d_inp00, pt_inp0, 4);
                  d_inp00 = AE_SRAI16(d_inp00, 8);
                  AE_L8X4F_IP(d_inp10, pt_inp1, 4);
                  d_inp10 = AE_SRAI16(d_inp10, 8);
#endif
                  d_inp00 = AE_ADD16(d_inp00, d_input_zero_bias);
                  d_inp10 = AE_ADD16(d_inp10, d_input_zero_bias);
#pragma loop_count min=1
#pragma no_unroll
                  for(l = 0; l < (kernel_width_pad>>2); l++)
                  {
#if XCHAL_HAVE_HIFI1
                      AE_L8X4S_IP(d_inp01, pt_inp0, 4);
                      AE_L8X4S_IP(d_inp11, pt_inp1, 4);
                      AE_L8X4S_IP(d_ker0, pt_ker0, 4);
                      AE_L8X4S_IP(d_ker1, pt_ker1, 4);
#else
                      AE_L8X4F_IP(d_inp01, pt_inp0, 4);
                      AE_L8X4F_IP(d_inp11, pt_inp1, 4);
                      AE_L8X4F_IP(d_ker0, pt_ker0, 4);
                      AE_L8X4F_IP(d_ker1, pt_ker1, 4);
                      d_inp01 = AE_SRAI16(d_inp01, 8);
                      d_inp11 = AE_SRAI16(d_inp11, 8);
                      d_ker0 = AE_SRAI16(d_ker0, 8);
                      d_ker1 = AE_SRAI16(d_ker1, 8);
#endif
                      d_inp01 = AE_ADD16(d_inp01, d_input_zero_bias);
                      d_inp11 = AE_ADD16(d_inp11, d_input_zero_bias);
                      d_inp1 = AE_SEL16_6543(d_inp00, d_inp01);
                      d_inp2 = AE_SEL16_5432(d_inp00, d_inp01);
                      d_inp3 = AE_SEL16_4321(d_inp00, d_inp01);
                      d_inp4 = AE_SEL16_6543(d_inp10, d_inp11);
                      d_inp5 = AE_SEL16_5432(d_inp10, d_inp11);
                      d_inp6 = AE_SEL16_4321(d_inp10, d_inp11);
                      AE_MULAAAAQ16(accu_int64_0, d_inp00, d_ker0);
                      AE_MULAAAAQ16(accu_int64_1, d_inp1, d_ker0);
                      AE_MULAAAAQ16(accu_int64_2, d_inp2, d_ker0);
                      AE_MULAAAAQ16(accu_int64_3, d_inp3, d_ker0);
                      AE_MULAAAAQ16(accu_int64_0, d_inp10, d_ker1);
                      AE_MULAAAAQ16(accu_int64_1, d_inp4, d_ker1);
                      AE_MULAAAAQ16(accu_int64_2, d_inp5, d_ker1);
                      AE_MULAAAAQ16(accu_int64_3, d_inp6, d_ker1);
                      d_inp00 = d_inp01;
                      d_inp10 = d_inp11;
                  }
              }
              *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_0), AE_MOVINT32X2_FROMINT64(accu_int64_1));
              *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_2), AE_MOVINT32X2_FROMINT64(accu_int64_3));
          }
      }
    }

    /* Here we store output based on strides. For values in a row, values
     * will be picked from it as per 'x_stride'. No need to worry about
     * height dimension, since we took care of it by efficient row
     * accesses. */
    ae_int32 *scratch_ptr1 = (ae_int32 *) p_scratch;
#if TFLITE_SINGLE_ROUNDING
  int left_shift = out_shift;
  int right_shift = out_shift;
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int left_shift = XT_MAX(0, out_shift);
  int right_shift = XT_MAX(0, -out_shift);
#endif /* #if TFLITE_SINGLE_ROUNDING */

    for(i = 0; i < actual_out_height; i++)
    {
        scratch_ptr1 = (ae_int32 *) p_scratch + (i * output_width_for_x_stride_1);
        WORD8 *out_ptr  = (WORD8 *) p_out + (i * out_stride * actual_out_width);
        ae_int32x2 accu_int32_0;


#pragma no_unroll
#pragma loop_count min=1
        for(j = 0; j < actual_out_width; j++)
        {
            accu_int32_0 = scratch_ptr1[(j * x_stride)];

            accu_int32_0 = AE_ADD32S(accu_int32_0, _ae_int32_sat_bias);

            MPY_BY_QUANT_MULT_X2_OUT32(accu_int32_0, accu_int32_0, out_multiplier, left_shift, right_shift)

            accu_int32_0 = AE_ADD32S(accu_int32_0, AE_MOVDA32X2(out_zero_bias, out_zero_bias));
            accu_int32_0 = AE_SRAI32(AE_SLAI32S(accu_int32_0, 24), 24);
#if XCHAL_HAVE_HIFI1
            AE_S8_0_XP_HIFI1(AE_MOVINT16X4_FROMINT32X2(accu_int32_0), out_ptr, out_stride);
#else
            out_ptr[(j * out_stride)] = (WORD8)AE_MOVAD32_L(accu_int32_0);
#endif
        }
    }
}

static void xa_nn_conv2d_depthwise_per_chan_nchw_sym8sxasym8s
(pWORD8 __restrict__ p_out
 ,const WORD8 *__restrict__ p_kernel
 ,const WORD8 *__restrict__ p_inp
 ,const WORD32 *__restrict__ p_bias
 ,WORD32  input_height
 ,WORD32  input_width
 ,WORD32  input_channels
 ,WORD32  kernel_height
 ,WORD32  kernel_width
 ,WORD32  channels_multiplier
 ,WORD32  x_stride
 ,WORD32  y_stride
 ,WORD32  x_padding
 ,WORD32  y_padding
 ,WORD32  out_height
 ,WORD32  out_width
 ,WORD32  input_zero_bias
 ,const WORD32  *p_out_multiplier
 ,const WORD32  *p_out_shift
 ,WORD32  out_zero_bias
,pVOID p_scratch
)
{ 
    int input_zero_bias_neg = -input_zero_bias;
    xa_nn_conv2d_depthwise_init
        (p_scratch
         ,input_height
         ,input_width
         ,input_channels
         ,kernel_height
         ,kernel_width
         ,channels_multiplier
         ,x_stride
         ,y_stride
         ,x_padding
         ,y_padding
         ,out_height
         ,out_width
         ,8
         ,1
         ,(pVOID)(&input_zero_bias_neg)
        );

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ic, itr_cm, itr_oh;
    int circ_out_height = (p_circ_buf->rows - kernel_height)/y_stride + 1;
    int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
    int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
    int rows_to_add, top_pad, bottom_pad, rows_added;
    int input_row;
    const WORD8 *pt_ker;
    const WORD8 *pt_inp;
    pWORD8 p_inp_circ;
    int i;
    WORD8 *p_kernel_padded = (WORD8 *)(p_state->p_scratch);
    p_kernel_padded = (WORD8 *)ALIGN_PTR(p_kernel_padded, 8);
    pWORD32 p_tmp_out = (pWORD32)(p_kernel_padded + kernel_height_pad * kernel_width_pad);
    p_tmp_out = (pWORD32)ALIGN_PTR(p_tmp_out, 8);

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    WORD32 bias = 0;

    /* Initialize whole scratch for padded kernel to padding value, after this
       we only have to copy actual kernel values, padding area should remain
       untouched */
    ae_int32x2 *pae_ker_pad = (ae_int32x2 *)p_kernel_padded;
    for(i = 0; i < ((kernel_height_pad * kernel_width_pad) >> 3); i++)
    {
      pae_ker_pad[i] = AE_ZERO32();
    }

    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = (const WORD8 *)&p_inp[itr_ic*input_height*input_width];

        CIRC_BUF_ADD_ROWS_INIT_WITH_PAD_VAL(rows_added
                ,rows_to_add
                ,top_pad
                ,bottom_pad
                ,input_row
                ,input_height
                ,input_width
                ,kernel_height
                ,y_stride
                ,x_padding
                ,y_padding
                ,p_circ_buf
                ,pt_inp
                ,&input_zero_bias_neg
                );

        for(itr_oh = 0; itr_oh < out_height - (circ_out_height - 1); itr_oh += circ_out_height)
        {
            CIRC_BUF_ADD_ROWS_WITH_PAD_VAL(rows_added
                    ,rows_to_add
                    ,top_pad
                    ,bottom_pad
                    ,input_row
                    ,input_height
                    ,input_width
                    ,circ_out_height
                    ,y_stride
                    ,x_padding
                    ,y_padding
                    ,p_circ_buf
                    ,pt_inp
                    ,&input_zero_bias_neg
                    );

            p_inp_circ = (WORD8 *)p_circ_buf->p_curr;

            for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
            {
                pt_ker = (const WORD8 *)&p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
                COPY_KERNEL_TO_SCRATCH_8b(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
                bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

                conv2d_nchw_sym8sxasym8s_hf4_convmul
                    ((pWORD8)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
                     ,p_kernel_padded
                     ,p_inp_circ
                     ,bias
                     ,p_circ_buf->rows
                     ,p_circ_buf->row_offset
                     ,kernel_height
                     ,kernel_width
                     ,circ_out_height
                     ,out_width
                     ,(input_channels * channels_multiplier)
                     ,x_stride
                     ,y_stride
                     ,input_zero_bias
                     ,p_out_multiplier[itr_ic * channels_multiplier + itr_cm]
                     ,p_out_shift[itr_ic * channels_multiplier + itr_cm]
                     ,out_zero_bias
                     ,p_tmp_out
                    );
            }
        }

        CIRC_BUF_ADD_ROWS_WITH_PAD_VAL(rows_added
                ,rows_to_add
                ,top_pad
                ,bottom_pad
                ,input_row
                ,input_height
                ,input_width
                ,circ_out_height
                ,y_stride
                ,x_padding
                ,y_padding
                ,p_circ_buf
                ,pt_inp
                ,&input_zero_bias_neg
                );

        p_inp_circ = (WORD8 *)p_circ_buf->p_curr;

        for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
        {
            pt_ker = (const WORD8 *)&p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
            COPY_KERNEL_TO_SCRATCH_8b(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
            bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

            conv2d_nchw_sym8sxasym8s_hf4_convmul
                ((pWORD8)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
                 ,p_kernel_padded
                 ,p_inp_circ
                 ,bias
                 ,p_circ_buf->rows
                 ,p_circ_buf->row_offset
                 ,kernel_height
                 ,kernel_width
                 ,(out_height - itr_oh)
                 ,out_width
                 ,(input_channels * channels_multiplier)
                 ,x_stride
                 ,y_stride
                 ,input_zero_bias
                 ,p_out_multiplier[itr_ic * channels_multiplier + itr_cm]
                 ,p_out_shift[itr_ic * channels_multiplier + itr_cm]
                 ,out_zero_bias
                 ,p_tmp_out
                );
        }
    }
}

/* 2D Convolution implementation */
static inline void conv2d_per_chan_nhwc_sym8sxasym8s
(pWORD8 __restrict__ p_out
 ,const WORD8 *__restrict__ p_ker
 ,const WORD8 *__restrict__ p_inp
 ,const WORD32 *p_bias
 ,int kernel_height
 ,int kernel_width
 ,int out_height
 ,int out_width
 ,int out_channels
 ,int y_stride
 ,WORD32  input_zero_bias
 ,const WORD32 *p_out_multiplier
 ,const WORD32 *p_out_shift
 ,WORD32  out_zero_bias
 ,pWORD32 __restrict__ p_scratch
 )
{
    WORD32 out_channels_pad;
    WORD32 i, itr_oh, itr_ch, itr_kw;
    pWORD8 pt_inp0, pt_inp1;
    const WORD8 *pt_ker;
    WORD8 *p_ker_scr;
    pWORD8 out_ptr0, out_ptr1;
    ae_int16x4 d_inp0, d_inp1, d_ker;
    const ae_int32x2 *pt_bias;
    ae_valign bias_a;
    ae_int32x2 d_acc0, d_acc1, d_bias0, d_bias1;
    ae_int32x2 d_acc2, d_acc3;
    ae_int16x4 d_acc16x4;

    ae_valign out_valign;
    WORD32 *p_out_multiplier_align = (WORD32 *)p_out_multiplier;
    out_valign = AE_LA64_PP(p_out_multiplier_align);

    out_channels_pad = (out_channels + 3)&(~3);

    pt_bias = (const ae_int32x2 *)p_bias;
    bias_a = AE_LA64_PP(pt_bias);
    for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
    {
        ae_int32x2 out_0, out_1;
        AE_LA32X2_IP(out_0, out_valign,(ae_int32x2 *)p_out_multiplier_align);
        AE_LA32X2_IP(out_1, out_valign, (ae_int32x2 *)p_out_multiplier_align);
        AE_LA32X2_IP(d_bias0, bias_a, pt_bias);
        AE_LA32X2_IP(d_bias1, bias_a, pt_bias);

        pt_ker = (const WORD8 *)(&p_ker[itr_ch]);
        p_ker_scr = (WORD8 *)p_scratch;
        COPY_KERNEL_TO_SCRATCH_NHWC_4_8b(p_ker_scr, pt_ker, kernel_height, kernel_width, out_channels);
        int l_shift[4], r_shift[4];
#if TFLITE_SINGLE_ROUNDING
        l_shift[0] = p_out_shift[itr_ch+0];
        l_shift[1] = p_out_shift[itr_ch+1];
        l_shift[2] = p_out_shift[itr_ch+2];
        l_shift[3] = p_out_shift[itr_ch+3];
        /* Single rounding macro doesn't need two shifts so this is not used */
        (void)r_shift[0];
        (void)r_shift[1];
        (void)r_shift[2];
        (void)r_shift[3];
#else /* #if TFLITE_SINGLE_ROUNDING */
        l_shift[0] = p_out_shift[itr_ch+0] < 0 ? 0 :  p_out_shift[itr_ch+0];
        r_shift[0] = p_out_shift[itr_ch+0] > 0 ? 0 : -p_out_shift[itr_ch+0];
        l_shift[1] = p_out_shift[itr_ch+1] < 0 ? 0 :  p_out_shift[itr_ch+1];
        r_shift[1] = p_out_shift[itr_ch+1] > 0 ? 0 : -p_out_shift[itr_ch+1];
        l_shift[2] = p_out_shift[itr_ch+2] < 0 ? 0 :  p_out_shift[itr_ch+2];
        r_shift[2] = p_out_shift[itr_ch+2] > 0 ? 0 : -p_out_shift[itr_ch+2];
        l_shift[3] = p_out_shift[itr_ch+3] < 0 ? 0 :  p_out_shift[itr_ch+3];
        r_shift[3] = p_out_shift[itr_ch+3] > 0 ? 0 : -p_out_shift[itr_ch+3];
#endif /* #if TFLITE_SINGLE_ROUNDING */

        for(itr_oh = 0; itr_oh < (out_height); itr_oh+=2)        
        {
            out_ptr0 = (WORD8 *)(&p_out[itr_oh*out_channels*out_width]);
            out_ptr1 = (WORD8 *)(&p_out[(itr_oh+1)*out_channels*out_width]);

            pt_inp0 = (WORD8 *)p_inp;
            pt_inp1 = (WORD8 *)p_inp;
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, itr_ch + itr_oh*y_stride*kernel_width*out_channels_pad);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, itr_ch + (itr_oh+1)*y_stride*kernel_width*out_channels_pad);
            p_ker_scr = (WORD8 *)p_scratch;
            d_acc0 = AE_ZERO32();
            d_acc1 = AE_ZERO32();
            d_acc2 = AE_ZERO32();
            d_acc3 = AE_ZERO32();
#pragma no_unroll
#pragma loop_count min=1
            for(itr_kw = 0; itr_kw < kernel_height * kernel_width; itr_kw++)
            {
#if XCHAL_HAVE_HIFI1
#if XCHAL_HAVE_HIFI1S
                AE_L8X4S_XC(d_inp0, pt_inp0, out_channels_pad);
                AE_L8X4S_XC(d_inp1, pt_inp1, out_channels_pad);
#else // XCHAL_HAVE_HIFI1S
                d_inp0 = AE_L8X4S_I(pt_inp0, 0);
                d_inp1 = AE_L8X4S_I(pt_inp1, 0);
#endif // XCHAL_HAVE_HIFI1S                
                AE_L8X4S_IP(d_ker, p_ker_scr, 4);
#else
                d_inp0 = AE_L8X4F_I(pt_inp0, 0);
                d_inp1 = AE_L8X4F_I(pt_inp1, 0);
                AE_L8X4F_IP(d_ker, p_ker_scr, 4);
                d_inp0 = AE_SRAI16(d_inp0, 8);
                d_inp1 = AE_SRAI16(d_inp1, 8);
                d_ker = AE_SRAI16(d_ker, 8);
#endif
                d_inp0 = AE_ADD16(d_inp0, AE_MOVDA16(input_zero_bias));
                d_inp1 = AE_ADD16(d_inp1, AE_MOVDA16(input_zero_bias));
                AE_MULA16X4(d_acc0, d_acc1, d_inp0, d_ker);
                AE_MULA16X4(d_acc2, d_acc3, d_inp1, d_ker);
#if !(XCHAL_HAVE_HIFI1S)
                AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, out_channels_pad);
                AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, out_channels_pad);
#endif                
            }
            d_acc0 = AE_ADD32S(d_acc0, d_bias0);
            d_acc1 = AE_ADD32S(d_acc1, d_bias1);
            d_acc2 = AE_ADD32S(d_acc2, d_bias0);
            d_acc3 = AE_ADD32S(d_acc3, d_bias1);
            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc0, d_acc0, out_0, l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc1, d_acc1, out_1, l_shift[2], l_shift[3], r_shift[2], r_shift[3]);

            d_acc0 = AE_ADD32S(d_acc0, AE_MOVDA32(out_zero_bias));
            d_acc1 = AE_ADD32S(d_acc1, AE_MOVDA32(out_zero_bias));
            d_acc0 = AE_SRAI32(AE_SLAI32S(d_acc0, 24), 24);
            d_acc1 = AE_SRAI32(AE_SLAI32S(d_acc1, 24), 24);

            d_acc16x4 = AE_SAT16X4(d_acc0, d_acc1);
#pragma no_unroll
            for(i = 0; i < XT_MIN(out_channels-itr_ch, 4); i++)
            {
                out_ptr0[itr_ch+i] = (UWORD8)AE_MOVAD16_3(d_acc16x4);
                d_acc16x4 = AE_SEL16_6543(d_acc16x4, d_acc16x4);
            }

            if(out_height - itr_oh >= 2)
            {
                MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc2, d_acc2, out_0, l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
                MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc3, d_acc3, out_1, l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
                d_acc2 = AE_ADD32S(d_acc2, AE_MOVDA32(out_zero_bias));
                d_acc3 = AE_ADD32S(d_acc3, AE_MOVDA32(out_zero_bias));
                d_acc2 = AE_SRAI32(AE_SLAI32S(d_acc2, 24), 24);
                d_acc3 = AE_SRAI32(AE_SLAI32S(d_acc3, 24), 24);

                d_acc16x4 = AE_SAT16X4(d_acc2, d_acc3);
#pragma no_unroll
                for(i = 0; i < XT_MIN(out_channels-itr_ch, 4); i++)
                {
                    out_ptr1[itr_ch+i] = (UWORD8)AE_MOVAD16_3(d_acc16x4);
                    d_acc16x4 = AE_SEL16_6543(d_acc16x4, d_acc16x4);
                }
            }
        }
    }
}

#ifndef DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE
/* Special case for kernel dimension 3x3 */
#if XCHAL_HAVE_HIFI1
static inline void conv2d_per_chan_nhwc_sym8sxasym8s_k3x3
(pWORD8 __restrict__ p_out
 ,const WORD8 *__restrict__ p_ker
 ,const WORD8 *__restrict__ p_inp
 ,const WORD32 *p_bias
 ,int kernel_width
 ,int out_height
 ,int out_width
 ,int out_channels
 ,int y_stride
 ,WORD32  input_zero_bias
 ,const WORD32 *p_out_multiplier
 ,const WORD32 *p_out_shift
 ,WORD32  out_zero_bias
 )
{
    WORD32 itr_oh, itr_ch, itr_kw;
    pWORD8 pt_inp0, pt_inp1;
    WORD8 *pt_ker;
    pWORD8 out_ptr0, out_ptr1;
    ae_int16x4 d_inp0, d_inp1, d_ker;
    const ae_int32x2 *pt_bias;
    ae_valign bias_a;
    ae_int32x2 d_acc0, d_acc1, d_bias0, d_bias1;
    ae_int32x2 d_acc2, d_acc3;

    ae_valign out_valign;
    WORD32 *p_out_multiplier_align = (WORD32 *)p_out_multiplier;
    out_valign = AE_LA64_PP(p_out_multiplier_align);

    pt_bias = (const ae_int32x2 *)p_bias;
    bias_a = AE_LA64_PP(pt_bias);
    for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
    {
	ae_int32x2 out_0, out_1;
        AE_LA32X2_IP(out_0, out_valign, (ae_int32x2 *)p_out_multiplier_align);
        AE_LA32X2_IP(out_1, out_valign, (ae_int32x2 *)p_out_multiplier_align);
        AE_LA32X2_IP(d_bias0, bias_a, pt_bias);
        AE_LA32X2_IP(d_bias1, bias_a, pt_bias);

        pt_ker = (WORD8 *)(&p_ker[itr_ch]);
        int i = 0;
        ae_int32x2 temp_acc0, temp_acc1;
        temp_acc0 = d_bias0;
        temp_acc1 = d_bias1;
        for(i=0; i<9; i++)
        {
            d_ker = AE_L8X4S_I(pt_ker, 0);
            AE_MULA16X4(temp_acc0, temp_acc1, d_ker, AE_MOVDA16(input_zero_bias));
            pt_ker += out_channels;
        }
        int l_shift[4], r_shift[4];
#if TFLITE_SINGLE_ROUNDING
        l_shift[0] = p_out_shift[itr_ch+0];
        l_shift[1] = p_out_shift[itr_ch+1];
        l_shift[2] = p_out_shift[itr_ch+2];
        l_shift[3] = p_out_shift[itr_ch+3];
#if XCHAL_HAVE_HIFI1S     
        l_shift[0] = ((31 - l_shift[1]) << 16) | (31 - l_shift[0]);
        l_shift[2] = ((31 - l_shift[3]) << 16) | (31 - l_shift[2]);
#endif
        /* Single rounding macro doesn't need two shifts so this is not used */
        (void)r_shift[0];
        (void)r_shift[1];
        (void)r_shift[2];
        (void)r_shift[3];
#else /* #if TFLITE_SINGLE_ROUNDING */
        l_shift[0] = p_out_shift[itr_ch+0] < 0 ? 0 :  p_out_shift[itr_ch+0];
        r_shift[0] = p_out_shift[itr_ch+0] > 0 ? 0 : -p_out_shift[itr_ch+0];
        l_shift[1] = p_out_shift[itr_ch+1] < 0 ? 0 :  p_out_shift[itr_ch+1];
        r_shift[1] = p_out_shift[itr_ch+1] > 0 ? 0 : -p_out_shift[itr_ch+1];
        l_shift[2] = p_out_shift[itr_ch+2] < 0 ? 0 :  p_out_shift[itr_ch+2];
        r_shift[2] = p_out_shift[itr_ch+2] > 0 ? 0 : -p_out_shift[itr_ch+2];
        l_shift[3] = p_out_shift[itr_ch+3] < 0 ? 0 :  p_out_shift[itr_ch+3];
        r_shift[3] = p_out_shift[itr_ch+3] > 0 ? 0 : -p_out_shift[itr_ch+3];
#endif /* #if TFLITE_SINGLE_ROUNDING */

        for(itr_oh = 0; itr_oh < (out_height); itr_oh+=2)        
        {
            out_ptr0 = (WORD8 *)(&p_out[itr_oh*out_channels*out_width]);
            out_ptr1 = (WORD8 *)(&p_out[(itr_oh+1)*out_channels*out_width]);

            pt_inp0 = (WORD8 *)p_inp;
            pt_inp1 = (WORD8 *)p_inp;
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, itr_ch + itr_oh*y_stride*kernel_width*out_channels);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, itr_ch + (itr_oh+1)*y_stride*kernel_width*out_channels);
            pt_ker = (WORD8 *)(&p_ker[itr_ch]);
            d_acc0 = temp_acc0;
            d_acc1 = temp_acc1;
            d_acc2 = temp_acc0;
            d_acc3 = temp_acc1;
#pragma no_unroll
#pragma loop_count min=9
            for(itr_kw = 0; itr_kw < 9; itr_kw++)
            {
#if XCHAL_HAVE_HIFI1S
                AE_L8X4S_XC(d_inp0, pt_inp0, out_channels);
                AE_L8X4S_XC(d_inp1, pt_inp1, out_channels);
#else // XCHAL_HAVE_HIFI1S
                d_inp0 = AE_L8X4S_I(pt_inp0, 0);
                d_inp1 = AE_L8X4S_I(pt_inp1, 0);
#endif // XCHAL_HAVE_HIFI1S
#if (XCHAL_HW_VERSION >= RI9_HWVERSION)
                AE_L8X4S_XP(d_ker, pt_ker, out_channels);
#else // (XCHAL_HW_VERSION >= RI9_HWVERSION)
                d_ker = AE_L8X4F_I(pt_ker, 0);
                d_ker = AE_SRAI16(d_ker, 8);
                pt_ker += out_channels;
#endif // (XCHAL_HW_VERSION >= RI9_HWVERSION)                
                AE_MULA16X4(d_acc0, d_acc1, d_inp0, d_ker);
                AE_MULA16X4(d_acc2, d_acc3, d_inp1, d_ker);
#if !(XCHAL_HAVE_HIFI1S)
                AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, out_channels);
                AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, out_channels);
#endif                
            }
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_REVERSE_OUTPUT_HIFI1S(d_acc0, d_acc0, out_0, l_shift[0]);
            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_REVERSE_OUTPUT_HIFI1S(d_acc1, d_acc1, out_1, l_shift[2]);

            d_acc0 = AE_ADD32S(d_acc0, AE_MOVDA32(out_zero_bias));
            d_acc1 = AE_ADD32S(d_acc1, AE_MOVDA32(out_zero_bias));
	
            ae_int8x8 d_acc8x8 = AE_SAT8X4X32_H(d_acc1, d_acc0);
            AE_S32_H_I(AE_MOVINT32X2_FROMINT8X8(d_acc8x8), (ae_int32*)&out_ptr0[itr_ch], 0);
			
            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_REVERSE_OUTPUT_HIFI1S(d_acc2, d_acc2, out_0, l_shift[0]);
            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32_REVERSE_OUTPUT_HIFI1S(d_acc3, d_acc3, out_1, l_shift[2]);
			
            d_acc2 = AE_ADD32S(d_acc2, AE_MOVDA32(out_zero_bias));
            d_acc3 = AE_ADD32S(d_acc3, AE_MOVDA32(out_zero_bias));
			
            d_acc8x8 = AE_SAT8X4X32_H(d_acc3, d_acc2);
            AE_S32_H_I(AE_MOVINT32X2_FROMINT8X8(d_acc8x8), (ae_int32*)&out_ptr1[itr_ch], 0);
#else

            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc0, d_acc0, out_0, l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc1, d_acc1, out_1, l_shift[2], l_shift[3], r_shift[2], r_shift[3]);

            d_acc0 = AE_ADD32S(d_acc0, AE_MOVDA32(out_zero_bias));
            d_acc1 = AE_ADD32S(d_acc1, AE_MOVDA32(out_zero_bias));
            d_acc0 = AE_SRAI32(AE_SLAI32S(d_acc0, 24), 24);
            d_acc1 = AE_SRAI32(AE_SLAI32S(d_acc1, 24), 24);

            ae_int16x4 d_acc16x4 = AE_SAT16X4(d_acc0, d_acc1);
            WORD8 *pout_ptr0 = &out_ptr0[itr_ch];
            AE_S8X4_I(d_acc16x4, pout_ptr0, 0);

            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc2, d_acc2, out_0, l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc3, d_acc3, out_1, l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
            d_acc2 = AE_ADD32S(d_acc2, AE_MOVDA32(out_zero_bias));
            d_acc3 = AE_ADD32S(d_acc3, AE_MOVDA32(out_zero_bias));
            d_acc2 = AE_SRAI32(AE_SLAI32S(d_acc2, 24), 24);
            d_acc3 = AE_SRAI32(AE_SLAI32S(d_acc3, 24), 24);

            d_acc16x4 = AE_SAT16X4(d_acc2, d_acc3);
            WORD8 *pout_ptr1 = &out_ptr1[itr_ch];
            AE_S8X4_I(d_acc16x4, pout_ptr1, 0);
#endif            
        }
    }
}
#else
static inline void conv2d_per_chan_nhwc_sym8sxasym8s_k3x3
(pWORD8 __restrict__ p_out
 ,const WORD8 *__restrict__ p_ker
 ,const WORD8 *__restrict__ p_inp
 ,const WORD32 *p_bias
 ,int kernel_width
 ,int out_height
 ,int out_width
 ,int out_channels
 ,int y_stride
 ,WORD32  input_zero_bias
 ,const WORD32 *p_out_multiplier
 ,const WORD32 *p_out_shift
 ,WORD32  out_zero_bias
 )
{
    WORD32 itr_oh, itr_ch, itr_kw;
    pWORD8 pt_inp0, pt_inp1;
    WORD8 *pt_ker;
    pWORD8 out_ptr0, out_ptr1;
    ae_int16x4 d_inp0, d_inp1, d_ker;
    const ae_int32x2 *pt_bias;
    ae_valign bias_a;
    ae_int32x2 d_acc0, d_acc1, d_bias0, d_bias1;
    ae_int32x2 d_acc2, d_acc3;
    ae_int16x4 d_acc16x4;

    ae_valign out_valign;
    WORD32 *p_out_multiplier_align = (WORD32 *)p_out_multiplier;
    out_valign = AE_LA64_PP(p_out_multiplier_align);

    pt_bias = (const ae_int32x2 *)p_bias;
    bias_a = AE_LA64_PP(pt_bias);
    for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
    {
        ae_int32x2 out_0, out_1;
        AE_LA32X2_IP(out_0, out_valign, (ae_int32x2 *)p_out_multiplier_align);
        AE_LA32X2_IP(out_1, out_valign, (ae_int32x2 *)p_out_multiplier_align);
        AE_LA32X2_IP(d_bias0, bias_a, pt_bias);
        AE_LA32X2_IP(d_bias1, bias_a, pt_bias);

        pt_ker = (WORD8 *)(&p_ker[itr_ch]);
        int i = 0;
        ae_int32x2 temp_acc0, temp_acc1;
        temp_acc0 = temp_acc1 = AE_ZERO32();
        for(i=0; i<9; i++)
        {
            d_ker = AE_L8X4F_I(pt_ker, 0);
            AE_MULA16X4(temp_acc0, temp_acc1, d_ker, AE_MOVDA16(input_zero_bias));
            pt_ker += out_channels;
        }
        int l_shift[4], r_shift[4];
#if TFLITE_SINGLE_ROUNDING
        l_shift[0] = p_out_shift[itr_ch+0];
        l_shift[1] = p_out_shift[itr_ch+1];
        l_shift[2] = p_out_shift[itr_ch+2];
        l_shift[3] = p_out_shift[itr_ch+3];
        /* Single rounding macro doesn't need two shifts so this is not used */
        (void)r_shift[0];
        (void)r_shift[1];
        (void)r_shift[2];
        (void)r_shift[3];
#else /* #if TFLITE_SINGLE_ROUNDING */
        l_shift[0] = p_out_shift[itr_ch+0] < 0 ? 0 :  p_out_shift[itr_ch+0];
        r_shift[0] = p_out_shift[itr_ch+0] > 0 ? 0 : -p_out_shift[itr_ch+0];
        l_shift[1] = p_out_shift[itr_ch+1] < 0 ? 0 :  p_out_shift[itr_ch+1];
        r_shift[1] = p_out_shift[itr_ch+1] > 0 ? 0 : -p_out_shift[itr_ch+1];
        l_shift[2] = p_out_shift[itr_ch+2] < 0 ? 0 :  p_out_shift[itr_ch+2];
        r_shift[2] = p_out_shift[itr_ch+2] > 0 ? 0 : -p_out_shift[itr_ch+2];
        l_shift[3] = p_out_shift[itr_ch+3] < 0 ? 0 :  p_out_shift[itr_ch+3];
        r_shift[3] = p_out_shift[itr_ch+3] > 0 ? 0 : -p_out_shift[itr_ch+3];
#endif /* #if TFLITE_SINGLE_ROUNDING */

        for(itr_oh = 0; itr_oh < (out_height); itr_oh+=2)        
        {
            out_ptr0 = (WORD8 *)(&p_out[itr_oh*out_channels*out_width]);
            out_ptr1 = (WORD8 *)(&p_out[(itr_oh+1)*out_channels*out_width]);

            pt_inp0 = (WORD8 *)p_inp;
            pt_inp1 = (WORD8 *)p_inp;
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, itr_ch + itr_oh*y_stride*kernel_width*out_channels);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, itr_ch + (itr_oh+1)*y_stride*kernel_width*out_channels);
            pt_ker = (WORD8 *)(&p_ker[itr_ch]);
            d_acc0 = temp_acc0;
            d_acc1 = temp_acc1;
            d_acc2 = temp_acc0;
            d_acc3 = temp_acc1;
#pragma no_unroll
#pragma loop_count min=9
            for(itr_kw = 0; itr_kw < 9; itr_kw++)
            {
                d_inp0 = AE_L8X4F_I(pt_inp0, 0);
                d_inp1 = AE_L8X4F_I(pt_inp1, 0);
                d_ker = AE_L8X4F_I(pt_ker, 0);
                d_ker = AE_SRAI16(d_ker, 8);

                AE_MULA16X4(d_acc0, d_acc1, d_inp0, d_ker);
                AE_MULA16X4(d_acc2, d_acc3, d_inp1, d_ker);
                AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, out_channels);
                AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, out_channels);
                pt_ker += out_channels;
            }

            d_acc0 = AE_SRAI32(d_acc0, 8);
            d_acc1 = AE_SRAI32(d_acc1, 8);
            d_acc2 = AE_SRAI32(d_acc2, 8);
            d_acc3 = AE_SRAI32(d_acc3, 8);

            d_acc0 = AE_ADD32S(d_acc0, d_bias0);
            d_acc1 = AE_ADD32S(d_acc1, d_bias1);
            d_acc2 = AE_ADD32S(d_acc2, d_bias0);
            d_acc3 = AE_ADD32S(d_acc3, d_bias1);
            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc0, d_acc0, out_0, l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc1, d_acc1, out_1, l_shift[2], l_shift[3], r_shift[2], r_shift[3]);

            d_acc0 = AE_ADD32S(d_acc0, AE_MOVDA32(out_zero_bias));
            d_acc1 = AE_ADD32S(d_acc1, AE_MOVDA32(out_zero_bias));
            d_acc0 = AE_SRAI32(AE_SLAI32S(d_acc0, 24), 24);
            d_acc1 = AE_SRAI32(AE_SLAI32S(d_acc1, 24), 24);

            d_acc16x4 = AE_SAT16X4(d_acc0, d_acc1);

            out_ptr0[itr_ch+0] = (UWORD8)AE_MOVAD16_3(d_acc16x4);
            out_ptr0[itr_ch+1] = (UWORD8)AE_MOVAD16_2(d_acc16x4);
            out_ptr0[itr_ch+2] = (UWORD8)AE_MOVAD16_1(d_acc16x4);
            out_ptr0[itr_ch+3] = (UWORD8)AE_MOVAD16_0(d_acc16x4);

            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc2, d_acc2, out_0, l_shift[0], l_shift[1], r_shift[0], r_shift[1]);
            MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(d_acc3, d_acc3, out_1, l_shift[2], l_shift[3], r_shift[2], r_shift[3]);
            d_acc2 = AE_ADD32S(d_acc2, AE_MOVDA32(out_zero_bias));
            d_acc3 = AE_ADD32S(d_acc3, AE_MOVDA32(out_zero_bias));
            d_acc2 = AE_SRAI32(AE_SLAI32S(d_acc2, 24), 24);
            d_acc3 = AE_SRAI32(AE_SLAI32S(d_acc3, 24), 24);

            d_acc16x4 = AE_SAT16X4(d_acc2, d_acc3);
            out_ptr1[itr_ch+0] = (UWORD8)AE_MOVAD16_3(d_acc16x4);
            out_ptr1[itr_ch+1] = (UWORD8)AE_MOVAD16_2(d_acc16x4);
            out_ptr1[itr_ch+2] = (UWORD8)AE_MOVAD16_1(d_acc16x4);
            out_ptr1[itr_ch+3] = (UWORD8)AE_MOVAD16_0(d_acc16x4);
        }
    }
}
#endif
#endif

static void xa_nn_conv2d_depthwise_per_chan_nhwc_sym8sxasym8s
(pWORD8 __restrict__ p_out
 ,const WORD8 *__restrict__ p_kernel
 ,const WORD8 *__restrict__ p_inp
 ,const WORD32 *__restrict__ p_bias
 ,WORD32  input_height
 ,WORD32  input_width
 ,WORD32  input_channels
 ,WORD32  kernel_height
 ,WORD32  kernel_width
 ,WORD32  channels_multiplier
 ,WORD32  x_stride
 ,WORD32  y_stride
 ,WORD32  x_padding
 ,WORD32  y_padding
 ,WORD32  out_height
 ,WORD32  out_width
 ,WORD32  input_zero_bias
 ,const WORD32  *p_out_multiplier
 ,const WORD32  *p_out_shift
 ,WORD32  out_zero_bias
,pVOID p_scratch
)
{
    int temp_pad_val = 0;
    xa_nn_conv2d_depthwise_init
        (p_scratch
         ,input_height
         ,input_width
         ,input_channels
         ,kernel_height
         ,kernel_width
         ,channels_multiplier
         ,x_stride
         ,y_stride
         ,x_padding
         ,y_padding
         ,out_height
         ,out_width
         ,8
         ,0
         ,(pVOID)(&temp_pad_val)
        );

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ow;
    int cols_to_add, left_pad, right_pad, cols_added;
    int input_col;
    int input_zero_bias_neg = -input_zero_bias;
    const WORD8 *pt_inp;
    pWORD8 p_inp_circ;

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    pt_inp = (const WORD8 *)p_inp;

    CIRC_BUF_ADD_COLS_INIT_WITH_PAD_VAL(cols_added
            ,cols_to_add
            ,left_pad
            ,right_pad
            ,input_col
            ,input_height
            ,input_width
            ,input_channels
            ,kernel_height
            ,kernel_width
            ,channels_multiplier
            ,x_stride
            ,x_padding
            ,y_padding
            ,out_height
            ,p_circ_buf
            ,pt_inp
            ,&input_zero_bias_neg
            );

#ifndef DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE
    if((channels_multiplier == 1) &&
       (kernel_height == 3) &&
       (kernel_width == 3) &&
       ((((unsigned)p_inp) & 3) == 0) &&
       ((((unsigned)p_kernel) & 3) == 0) &&
       ((((unsigned)p_out) & 3) == 0) &&
       ((input_channels & 0x3) == 0) &&
       ((out_height & 0x1) == 0)
      )
    {
      for(itr_ow = 0; itr_ow < out_width; itr_ow++)
      {
          CIRC_BUF_ADD_COLS_WITH_PAD_VAL(cols_added
                  ,cols_to_add
                  ,left_pad
                  ,right_pad
                  ,input_col
                  ,input_height
                  ,input_width
                  ,input_channels
                  ,kernel_height
                  ,kernel_width
                  ,channels_multiplier
                  ,x_stride
                  ,x_padding
                  ,y_padding
                  ,out_height
                  ,p_circ_buf
                  ,pt_inp
                  ,&input_zero_bias_neg
                  );

          p_inp_circ = (WORD8 *)p_circ_buf->p_curr;

          conv2d_per_chan_nhwc_sym8sxasym8s_k3x3
              ((pWORD8)(&p_out[itr_ow*input_channels*channels_multiplier])
               ,p_kernel
               ,p_inp_circ
               ,p_bias
               ,kernel_width
               ,out_height
               ,out_width
               ,(input_channels * channels_multiplier)
               ,y_stride
               ,input_zero_bias
               ,p_out_multiplier
               ,p_out_shift
               ,out_zero_bias
              );
      }
    }
    else
#endif
    {
      for(itr_ow = 0; itr_ow < out_width; itr_ow++)
      {
          CIRC_BUF_ADD_COLS_WITH_PAD_VAL(cols_added
                  ,cols_to_add
                  ,left_pad
                  ,right_pad
                  ,input_col
                  ,input_height
                  ,input_width
                  ,input_channels
                  ,kernel_height
                  ,kernel_width
                  ,channels_multiplier
                  ,x_stride
                  ,x_padding
                  ,y_padding
                  ,out_height
                  ,p_circ_buf
                  ,pt_inp
                  ,&input_zero_bias_neg
                  );

          p_inp_circ = (WORD8 *)p_circ_buf->p_curr;

          conv2d_per_chan_nhwc_sym8sxasym8s
              ((pWORD8)(&p_out[itr_ow*input_channels*channels_multiplier])
               ,p_kernel
               ,p_inp_circ
               ,p_bias
               ,kernel_height
               ,kernel_width
               ,out_height
               ,out_width
               ,(input_channels * channels_multiplier)
               ,y_stride
               ,input_zero_bias
               ,p_out_multiplier
               ,p_out_shift
               ,out_zero_bias
               ,p_state->p_scratch
              );
      }
    }
}

WORD32 xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,const WORD32 *__restrict__ p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  x_stride
  ,WORD32  y_stride
  ,WORD32  x_padding
  ,WORD32  y_padding
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  input_zero_bias
  ,const WORD32 *p_out_multiplier
  ,const WORD32 *p_out_shift
  ,WORD32  out_zero_bias
  ,WORD32  inp_data_format
  ,WORD32  out_data_format
  ,pVOID p_scratch
  )
{
    int i;
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_zero_bias > 128 || input_zero_bias < -127), -1);
    for(i = 0; i < input_channels*channels_multiplier; i++)
      XA_NNLIB_ARG_CHK_COND((p_out_shift[i] < -31 || p_out_shift[i] > 31), -1);
    XA_NNLIB_ARG_CHK_COND((out_zero_bias > 127 || out_zero_bias < -128), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

    if(inp_data_format == 0)
    {
        xa_nn_conv2d_depthwise_per_chan_nhwc_sym8sxasym8s
            (p_out
             ,p_kernel
             ,p_inp
             ,p_bias
             ,input_height
             ,input_width
             ,input_channels
             ,kernel_height
             ,kernel_width
             ,channels_multiplier
             ,x_stride
             ,y_stride
             ,x_padding
             ,y_padding
             ,out_height
             ,out_width
             ,input_zero_bias
             ,p_out_multiplier
             ,p_out_shift
             ,out_zero_bias
             ,p_scratch);
    }
    else if(inp_data_format == 1)
    {
        xa_nn_conv2d_depthwise_per_chan_nchw_sym8sxasym8s
            (p_out
             ,p_kernel
             ,p_inp
             ,p_bias
             ,input_height
             ,input_width
             ,input_channels
             ,kernel_height
             ,kernel_width
             ,channels_multiplier
             ,x_stride
             ,y_stride
             ,x_padding
             ,y_padding
             ,out_height
             ,out_width
             ,input_zero_bias
             ,p_out_multiplier
             ,p_out_shift
             ,out_zero_bias
             ,p_scratch);
    }
    return 0;
}

/* 2D Convolution with dilation implementation */
static inline void dilated_conv2d_nchw_sym8sxasym8s_hf4_convmul
(pWORD8 __restrict__ p_out  /* Output:  [Stream] [(out_stride): (actual_out_height): (actual_out_width)] */
 ,const WORD8 *__restrict__ p_ker  /* Kernel:  [Block] [1:             kernel_height:       kernel_width_pad] */
 ,const WORD8 *__restrict__ p_inp  /* Input:   [Block] [1:             input_height:        input_width] */
 ,WORD32 bias
 ,int input_height
 ,int input_width
 ,int kernel_height
 ,int kernel_width
 ,int dilation_height
 ,int dilation_width
 ,int actual_out_height      /* This is the actual output height, processing should be limited to it. */
 ,int actual_out_width       /* This is the actual output width, processing should be limited to it. */
 ,int out_stride
 ,int x_stride
 ,int y_stride
 ,WORD32  input_zero_bias
 ,WORD32  out_multiplier
 ,WORD32  out_shift
 ,WORD32  out_zero_bias
 ,pWORD32 __restrict__ p_scratch /* Scratch: [Block] [1:             (actual_out_height): (out_width)] */
 )
{
    /* Importance of actual_out_width, since we are appending zeros input left
     * and right side. No problem with left padding, but for right padding that
     * is done to make sure that input_width is multiple of 4. Here
     * 'output_width_for_x_stride_1' value is calculated based on this padded value. But
     * actually expected output width to pick correct values from 'output_width_for_x_stride_1' on
     * jumps of 'x_stride'. */

    int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
    int kernel_width_pad = (((kernel_width - 1) * dilation_width + 1)+3)&(~3);

    /* Generic case */
    int i, j, k, l;
    int output_height = input_height - kernel_height + 1;
    int output_width_for_x_stride_1;

    /* Here input_width is nothing but circ_buf_width, which is taken care to be
     * multiple of 4. */
    output_width_for_x_stride_1 = (1 + ((input_width - ((kernel_width - 1) * dilation_width + 1))/1));
    /* output_width_for_x_stride_1 loop is unrolled by 4 so keeping this dimension to multiple of 4 */
    output_width_for_x_stride_1 = ALIGNED_SIZE(output_width_for_x_stride_1, (ALIGNMENT/2));

    /* Please note that below addition of 1 is done to adjust in C style indices
     * */
    if ((actual_out_height - 1) > ((output_height + 1) / (y_stride)))
    {
        return;
    }
    if ((actual_out_width - 1) > ((output_width_for_x_stride_1 + 1) / (x_stride)))
    {
        return;
    }

    ae_int64 accu_int64_0, accu_int64_1, accu_int64_2, accu_int64_3;
    ae_int32x2 *scratch_ptr = (ae_int32x2 *)p_scratch;

    ae_int16x4 d_input_zero_bias;
    d_input_zero_bias = AE_MOVDA16(input_zero_bias);
    ae_int32x2 _ae_int32_sat_bias;
    _ae_int32_sat_bias = AE_MOVDA32X2(bias, bias);

    if(kernel_width_pad==12)
    {
      ae_int16x4 d_inp00, d_inp01, d_inp02, d_inp03;
      ae_int16x4 d_ker0, d_ker1, d_ker2;
      ae_int16x4 d_inp1, d_inp2, d_inp3, d_inp4, d_inp5, d_inp6, d_inp7, d_inp8, d_inp9;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int32x2 *) (p_scratch + (i * output_width_for_x_stride_1));
#pragma loop_count min=1
        for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          const WORD8 *pt_ker = p_ker;
#pragma loop_count min=1
#pragma no_unroll
          for(k = 0; k < kernel_height; k++)
          {
            const WORD8 *pt_inp0 = (p_inp);
            AE_ADDCIRC16X4_XC
                ((ae_int16x4 *)pt_inp0
                  ,((sizeof(WORD8)) * ((i * y_stride * input_width) + j*4 + k*input_width*dilation_height))
                );
#if XCHAL_HAVE_HIFI1
            AE_L8X4S_IP(d_ker0, pt_ker, 4);
            AE_L8X4S_IP(d_ker1, pt_ker, 4);
            AE_L8X4S_IP(d_ker2, pt_ker, 4);
            AE_L8X4S_IP(d_inp00, pt_inp0, 4);
            AE_L8X4S_IP(d_inp01, pt_inp0, 4);
            AE_L8X4S_IP(d_inp02, pt_inp0, 4);
            AE_L8X4S_IP(d_inp03, pt_inp0, 4);
#else
            AE_L8X4F_IP(d_ker0, pt_ker, 4);
            AE_L8X4F_IP(d_ker1, pt_ker, 4);
            AE_L8X4F_IP(d_ker2, pt_ker, 4);
            AE_L8X4F_IP(d_inp00, pt_inp0, 4);
            AE_L8X4F_IP(d_inp01, pt_inp0, 4);
            AE_L8X4F_IP(d_inp02, pt_inp0, 4);
            AE_L8X4F_IP(d_inp03, pt_inp0, 4);
            d_ker0 = AE_SRAI16(d_ker0, 8);
            d_ker1 = AE_SRAI16(d_ker1, 8);
            d_ker2 = AE_SRAI16(d_ker2, 8);
            d_inp00 = AE_SRAI16(d_inp00, 8);
            d_inp01 = AE_SRAI16(d_inp01, 8);
            d_inp02 = AE_SRAI16(d_inp02, 8);
            d_inp03 = AE_SRAI16(d_inp03, 8);
#endif
            d_inp00 = AE_ADD16(d_inp00, d_input_zero_bias);
            d_inp01 = AE_ADD16(d_inp01, d_input_zero_bias);
            d_inp02 = AE_ADD16(d_inp02, d_input_zero_bias);
            d_inp03 = AE_ADD16(d_inp03, d_input_zero_bias);
            d_inp1 = AE_SEL16_6543(d_inp00, d_inp01);
            d_inp2 = AE_SEL16_5432(d_inp00, d_inp01);
            d_inp3 = AE_SEL16_4321(d_inp00, d_inp01);
            AE_MULAAAAQ16(accu_int64_0, d_inp00, d_ker0);
            AE_MULAAAAQ16(accu_int64_1, d_inp1, d_ker0);
            AE_MULAAAAQ16(accu_int64_2, d_inp2, d_ker0);
            AE_MULAAAAQ16(accu_int64_3, d_inp3, d_ker0);
            d_inp4 = AE_SEL16_6543(d_inp01, d_inp02);
            d_inp5 = AE_SEL16_5432(d_inp01, d_inp02);
            d_inp6 = AE_SEL16_4321(d_inp01, d_inp02);
            AE_MULAAAAQ16(accu_int64_0, d_inp01, d_ker1);
            AE_MULAAAAQ16(accu_int64_1, d_inp4, d_ker1);
            AE_MULAAAAQ16(accu_int64_2, d_inp5, d_ker1);
            AE_MULAAAAQ16(accu_int64_3, d_inp6, d_ker1);
            d_inp7 = AE_SEL16_6543(d_inp02, d_inp03);
            d_inp8 = AE_SEL16_5432(d_inp02, d_inp03);
            d_inp9 = AE_SEL16_4321(d_inp02, d_inp03);
            AE_MULAAAAQ16(accu_int64_0, d_inp02, d_ker2);
            AE_MULAAAAQ16(accu_int64_1, d_inp7, d_ker2);
            AE_MULAAAAQ16(accu_int64_2, d_inp8, d_ker2);
            AE_MULAAAAQ16(accu_int64_3, d_inp9, d_ker2);

          }
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_0), AE_MOVINT32X2_FROMINT64(accu_int64_1));
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_2), AE_MOVINT32X2_FROMINT64(accu_int64_3));
        }
      }
    }
    else if(kernel_width_pad==8)
    {
      ae_int16x4 d_inp00, d_inp01, d_inp02;
      ae_int16x4 d_ker0, d_ker1;
      ae_int16x4 d_inp1, d_inp2, d_inp3, d_inp4, d_inp5, d_inp6;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int32x2 *) (p_scratch + (i * output_width_for_x_stride_1));
#pragma loop_count min=1
        for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          const WORD8 *pt_ker = p_ker;
#pragma loop_count min=1
#pragma no_unroll
          for(k = 0; k < kernel_height; k++)
          {
            const WORD8 *pt_inp0 = (p_inp);
            AE_ADDCIRC16X4_XC
                ((ae_int16x4 *)pt_inp0
                  ,((sizeof(WORD8)) * ((i * y_stride * input_width) + j*4 + k*input_width*dilation_height))
                );
#if XCHAL_HAVE_HIFI1
            AE_L8X4S_IP(d_ker0, pt_ker, 4);
            AE_L8X4S_IP(d_ker1, pt_ker, 4);
            AE_L8X4S_IP(d_inp00, pt_inp0, 4);
            AE_L8X4S_IP(d_inp01, pt_inp0, 4);
            AE_L8X4S_IP(d_inp02, pt_inp0, 4);
#else
            AE_L8X4F_IP(d_ker0, pt_ker, 4);
            AE_L8X4F_IP(d_ker1, pt_ker, 4);
            AE_L8X4F_IP(d_inp00, pt_inp0, 4);
            AE_L8X4F_IP(d_inp01, pt_inp0, 4);
            AE_L8X4F_IP(d_inp02, pt_inp0, 4);
            d_ker0 = AE_SRAI16(d_ker0, 8);
            d_ker1 = AE_SRAI16(d_ker1, 8);
            d_inp00 = AE_SRAI16(d_inp00, 8);
            d_inp01 = AE_SRAI16(d_inp01, 8);
            d_inp02 = AE_SRAI16(d_inp02, 8);
#endif
            d_inp00 = AE_ADD16(d_inp00, d_input_zero_bias);
            d_inp01 = AE_ADD16(d_inp01, d_input_zero_bias);
            d_inp02 = AE_ADD16(d_inp02, d_input_zero_bias);
            d_inp1 = AE_SEL16_6543(d_inp00, d_inp01);
            d_inp2 = AE_SEL16_5432(d_inp00, d_inp01);
            d_inp3 = AE_SEL16_4321(d_inp00, d_inp01);
            AE_MULAAAAQ16(accu_int64_0, d_inp00, d_ker0);
            AE_MULAAAAQ16(accu_int64_1, d_inp1, d_ker0);
            AE_MULAAAAQ16(accu_int64_2, d_inp2, d_ker0);
            AE_MULAAAAQ16(accu_int64_3, d_inp3, d_ker0);
            d_inp4 = AE_SEL16_6543(d_inp01, d_inp02);
            d_inp5 = AE_SEL16_5432(d_inp01, d_inp02);
            d_inp6 = AE_SEL16_4321(d_inp01, d_inp02);
            AE_MULAAAAQ16(accu_int64_0, d_inp01, d_ker1);
            AE_MULAAAAQ16(accu_int64_1, d_inp4, d_ker1);
            AE_MULAAAAQ16(accu_int64_2, d_inp5, d_ker1);
            AE_MULAAAAQ16(accu_int64_3, d_inp6, d_ker1);

          }
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_0), AE_MOVINT32X2_FROMINT64(accu_int64_1));
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_2), AE_MOVINT32X2_FROMINT64(accu_int64_3));
        }
      }
    }
    else if(kernel_width_pad==4)
    {
      ae_int16x4 d_inp00, d_inp01;
      ae_int16x4 d_ker0;
      ae_int16x4 d_inp1, d_inp2, d_inp3;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int32x2 *) (p_scratch + (i * output_width_for_x_stride_1));
#pragma loop_count min=1
        for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          const WORD8 *pt_ker = p_ker;
#pragma loop_count min=1
#pragma no_unroll
          for(k = 0; k < kernel_height; k++)
          {
            const WORD8 *pt_inp0 = (p_inp);
            AE_ADDCIRC16X4_XC
                ((ae_int16x4 *)pt_inp0
                  ,((sizeof(WORD8)) * ((i * y_stride * input_width) + j*4 + k*input_width*dilation_height))
                );
#if XCHAL_HAVE_HIFI1
            AE_L8X4S_IP(d_ker0, pt_ker, 4);
            AE_L8X4S_IP(d_inp00, pt_inp0, 4);
            AE_L8X4S_IP(d_inp01, pt_inp0, 4);
#else
            AE_L8X4F_IP(d_ker0, pt_ker, 4);
            AE_L8X4F_IP(d_inp00, pt_inp0, 4);
            AE_L8X4F_IP(d_inp01, pt_inp0, 4);
            d_ker0 = AE_SRAI16(d_ker0, 8);
            d_inp00 = AE_SRAI16(d_inp00, 8);
            d_inp01 = AE_SRAI16(d_inp01, 8);
#endif
            d_inp00 = AE_ADD16(d_inp00, d_input_zero_bias);
            d_inp01 = AE_ADD16(d_inp01, d_input_zero_bias);
            d_inp1 = AE_SEL16_6543(d_inp00, d_inp01);
            d_inp2 = AE_SEL16_5432(d_inp00, d_inp01);
            d_inp3 = AE_SEL16_4321(d_inp00, d_inp01);
            AE_MULAAAAQ16(accu_int64_0, d_inp00, d_ker0);
            AE_MULAAAAQ16(accu_int64_1, d_inp1, d_ker0);
            AE_MULAAAAQ16(accu_int64_2, d_inp2, d_ker0);
            AE_MULAAAAQ16(accu_int64_3, d_inp3, d_ker0);

          }
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_0), AE_MOVINT32X2_FROMINT64(accu_int64_1));
          *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_2), AE_MOVINT32X2_FROMINT64(accu_int64_3));
        }
      }
    }
    else
    {
      for(i = 0; i < actual_out_height; i++)
      {
          scratch_ptr = (ae_int32x2 *) (p_scratch + (i * output_width_for_x_stride_1));
          for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
          {
              accu_int64_0 = AE_ZERO64();
              accu_int64_1 = AE_ZERO64();
              accu_int64_2 = AE_ZERO64();
              accu_int64_3 = AE_ZERO64();
#pragma loop_count min=1
              for(k = 0; k < kernel_height_pad; k += 2)
              {
                  const WORD8 *pt_inp0 = (p_inp);
                  AE_ADDCIRC16X4_XC
                      ((ae_int16x4 *)pt_inp0
                       ,((sizeof(WORD8)) * ((i * y_stride * input_width) + j*4 + k*input_width*dilation_height))
                      );
                  const WORD8 *pt_inp1 = (p_inp);
                  AE_ADDCIRC16X4_XC
                      ((ae_int16x4 *)pt_inp1
                       ,((sizeof(WORD8)) * ((i * y_stride * input_width) + j*4 + (k+1)*input_width*dilation_height))
                      );
                  const WORD8 *pt_ker0 = (p_ker + k*kernel_width_pad);
                  const WORD8 *pt_ker1 = (p_ker + (k+1)*kernel_width_pad);
                  ae_int16x4 d_ker0, d_ker1;
                  ae_int16x4 d_inp00, d_inp01, d_inp10, d_inp11, d_inp1, d_inp2, d_inp3, d_inp4, d_inp5, d_inp6;
#if XCHAL_HAVE_HIFI1
                  AE_L8X4S_IP(d_inp00, pt_inp0, 4);
                  AE_L8X4S_IP(d_inp10, pt_inp1, 4);
#else
                  AE_L8X4F_IP(d_inp00, pt_inp0, 4);
                  d_inp00 = AE_SRAI16(d_inp00, 8);
                  AE_L8X4F_IP(d_inp10, pt_inp1, 4);
                  d_inp10 = AE_SRAI16(d_inp10, 8);
#endif
                  d_inp00 = AE_ADD16(d_inp00, d_input_zero_bias);
                  d_inp10 = AE_ADD16(d_inp10, d_input_zero_bias);
#pragma loop_count min=1
#pragma no_unroll
                  for(l = 0; l < (kernel_width_pad>>2); l++)
                  {
#if XCHAL_HAVE_HIFI1
                      AE_L8X4S_IP(d_inp01, pt_inp0, 4);
                      AE_L8X4S_IP(d_inp11, pt_inp1, 4);
                      AE_L8X4S_IP(d_ker0, pt_ker0, 4);
                      AE_L8X4S_IP(d_ker1, pt_ker1, 4);
#else
                      AE_L8X4F_IP(d_inp01, pt_inp0, 4);
                      AE_L8X4F_IP(d_inp11, pt_inp1, 4);
                      AE_L8X4F_IP(d_ker0, pt_ker0, 4);
                      AE_L8X4F_IP(d_ker1, pt_ker1, 4);
                      d_inp01 = AE_SRAI16(d_inp01, 8);
                      d_inp11 = AE_SRAI16(d_inp11, 8);
                      d_ker0 = AE_SRAI16(d_ker0, 8);
                      d_ker1 = AE_SRAI16(d_ker1, 8);
#endif
                      d_inp01 = AE_ADD16(d_inp01, d_input_zero_bias);
                      d_inp11 = AE_ADD16(d_inp11, d_input_zero_bias);
                      d_inp1 = AE_SEL16_6543(d_inp00, d_inp01);
                      d_inp2 = AE_SEL16_5432(d_inp00, d_inp01);
                      d_inp3 = AE_SEL16_4321(d_inp00, d_inp01);
                      d_inp4 = AE_SEL16_6543(d_inp10, d_inp11);
                      d_inp5 = AE_SEL16_5432(d_inp10, d_inp11);
                      d_inp6 = AE_SEL16_4321(d_inp10, d_inp11);
                      AE_MULAAAAQ16(accu_int64_0, d_inp00, d_ker0);
                      AE_MULAAAAQ16(accu_int64_1, d_inp1, d_ker0);
                      AE_MULAAAAQ16(accu_int64_2, d_inp2, d_ker0);
                      AE_MULAAAAQ16(accu_int64_3, d_inp3, d_ker0);
                      AE_MULAAAAQ16(accu_int64_0, d_inp10, d_ker1);
                      AE_MULAAAAQ16(accu_int64_1, d_inp4, d_ker1);
                      AE_MULAAAAQ16(accu_int64_2, d_inp5, d_ker1);
                      AE_MULAAAAQ16(accu_int64_3, d_inp6, d_ker1);
                      d_inp00 = d_inp01;
                      d_inp10 = d_inp11;
                  }
              }
              *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_0), AE_MOVINT32X2_FROMINT64(accu_int64_1));
              *scratch_ptr++ = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu_int64_2), AE_MOVINT32X2_FROMINT64(accu_int64_3));
          }
      }
    }

    /* Here we store output based on strides. For values in a row, values
     * will be picked from it as per 'x_stride'. No need to worry about
     * height dimension, since we took care of it by efficient row
     * accesses. */
    ae_int32 *scratch_ptr1 = (ae_int32 *) p_scratch;
#if TFLITE_SINGLE_ROUNDING
  int left_shift = out_shift;
  int right_shift = out_shift;
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int left_shift = XT_MAX(0, out_shift);
  int right_shift = XT_MAX(0, -out_shift);
#endif /* #if TFLITE_SINGLE_ROUNDING */

    for(i = 0; i < actual_out_height; i++)
    {
        scratch_ptr1 = (ae_int32 *) p_scratch + (i * output_width_for_x_stride_1);
        WORD8 *out_ptr  = (WORD8 *) p_out + (i * out_stride * actual_out_width);
        ae_int32x2 accu_int32_0;


#pragma no_unroll
#pragma loop_count min=1
        for(j = 0; j < actual_out_width; j++)
        {
            accu_int32_0 = scratch_ptr1[(j * x_stride)];

            accu_int32_0 = AE_ADD32S(accu_int32_0, _ae_int32_sat_bias);

            MPY_BY_QUANT_MULT_X2_OUT32(accu_int32_0, accu_int32_0, out_multiplier, left_shift, right_shift)

            accu_int32_0 = AE_ADD32S(accu_int32_0, AE_MOVDA32X2(out_zero_bias, out_zero_bias));
            accu_int32_0 = AE_SRAI32(AE_SLAI32S(accu_int32_0, 24), 24);
#if XCHAL_HAVE_HIFI1
            AE_S8_0_XP_HIFI1(AE_MOVINT16X4_FROMINT32X2(accu_int32_0), out_ptr, out_stride);
#else
            out_ptr[(j * out_stride)] = (WORD8)AE_MOVAD32_L(accu_int32_0);
#endif
        }
    }
}

#define COPY_DILATED_KERNEL_TO_SCRATCH(p_out, p_in, kh, kw, kw_pad, d_w) \
{ \
  int itr_kh, itr_kw; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    WORD8 *pae_in = (WORD8 *)(&p_in[itr_kh * kw]); \
    WORD8 *pae_out = (WORD8 *)(&p_out[itr_kh * kw_pad]); \
    WORD8 d_tmp; \
    for(itr_kw = 0; itr_kw < kw; itr_kw++) \
    { \
      d_tmp = *pae_in++; \
      *pae_out = d_tmp; \
      pae_out += d_w;\
    } \
  } \
}

static void xa_nn_dilated_conv2d_depthwise_nchw_per_chan_sym8sxasym8s
    (pWORD8 __restrict__ p_out
    ,const WORD8 *__restrict__ p_kernel
    ,const WORD8 *__restrict__ p_inp
    ,const WORD32 *__restrict__ p_bias
    ,WORD32  input_height
    ,WORD32  input_width
    ,WORD32  input_channels
    ,WORD32  kernel_height
    ,WORD32  kernel_width
    ,WORD32  channels_multiplier
    ,WORD32  dilation_height
    ,WORD32  dilation_width
    ,WORD32  x_stride
    ,WORD32  y_stride
    ,WORD32  x_padding
    ,WORD32  y_padding
    ,WORD32  out_height
    ,WORD32  out_width
    ,WORD32  input_zero_bias
    ,const WORD32  *p_out_multiplier
    ,const WORD32  *p_out_shift
    ,WORD32  out_zero_bias
    ,WORD32  out_data_format
    ,pVOID p_scratch)
{
    int input_zero_bias_neg = -input_zero_bias;
    xa_nn_dilated_conv2d_depthwise_init
        (p_scratch
         ,input_height
         ,input_width
         ,input_channels
         ,kernel_height
         ,kernel_width 
         ,channels_multiplier
         ,dilation_height
         ,dilation_width
         ,x_stride
         ,y_stride
         ,x_padding
         ,y_padding
         ,out_height
         ,out_width
         ,8
         ,1
         ,(pVOID)(&input_zero_bias_neg)
        );

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ic, itr_cm, itr_oh;

    int circ_out_height = (p_circ_buf->rows - ((kernel_height - 1) * dilation_height + 1))/y_stride + 1;
    int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
    int kernel_width_pad = ALIGNED_SIZE((kernel_width - 1) * dilation_width + 1, 4);

    int rows_to_add, top_pad, bottom_pad, rows_added;
    int input_row;
    const WORD8 *pt_ker;
    const WORD8 *pt_inp;
    pWORD8 p_inp_circ;
    int i;
    WORD8 *p_kernel_padded = (WORD8 *)(p_state->p_scratch);
    p_kernel_padded = (WORD8 *)ALIGN_PTR(p_kernel_padded, 8);
    pWORD32 p_tmp_out = (pWORD32)(p_kernel_padded + kernel_height_pad * kernel_width_pad);
    p_tmp_out = (pWORD32)ALIGN_PTR(p_tmp_out, 8);

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    WORD32 bias = 0;

    /* Initialize whole scratch for padded kernel to padding value, after this
       we only have to copy actual kernel values, padding area should remain
       untouched */
    ae_int32x2 *pae_ker_pad = (ae_int32x2 *)p_kernel_padded;
    for(i = 0; i < ((kernel_height_pad * kernel_width_pad) >> 3); i++)
    {
      pae_ker_pad[i] = AE_ZERO32();
    }

    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = (const WORD8 *)&p_inp[itr_ic*input_height*input_width];

        CIRC_BUF_ADD_ROWS_INIT_WITH_PAD_VAL(rows_added
                ,rows_to_add
                ,top_pad
                ,bottom_pad
                ,input_row
                ,input_height
                ,input_width
                ,(kernel_height - 1) * dilation_height + 1
                ,y_stride
                ,x_padding
                ,y_padding
                ,p_circ_buf
                ,pt_inp
                ,&input_zero_bias_neg
                );

        for(itr_oh = 0; itr_oh < out_height - (circ_out_height - 1); itr_oh += circ_out_height)
        {
            CIRC_BUF_ADD_ROWS_WITH_PAD_VAL(rows_added
                    ,rows_to_add
                    ,top_pad
                    ,bottom_pad
                    ,input_row
                    ,input_height
                    ,input_width
                    ,circ_out_height
                    ,y_stride
                    ,x_padding
                    ,y_padding
                    ,p_circ_buf
                    ,pt_inp
                    ,&input_zero_bias_neg
                    );

            p_inp_circ = (WORD8 *)p_circ_buf->p_curr;

            for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
            {
                pt_ker = (const WORD8 *)&p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
                COPY_DILATED_KERNEL_TO_SCRATCH(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad, dilation_width);
                bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

                dilated_conv2d_nchw_sym8sxasym8s_hf4_convmul
                    ((pWORD8)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
                     ,p_kernel_padded
                     ,p_inp_circ
                     ,bias
                     ,p_circ_buf->rows
                     ,p_circ_buf->row_offset
                     ,kernel_height
                     ,kernel_width
                     ,dilation_height
                     ,dilation_width
                     ,circ_out_height
                     ,out_width
                     ,(input_channels * channels_multiplier)
                     ,x_stride
                     ,y_stride
                     ,input_zero_bias
                     ,p_out_multiplier[itr_ic * channels_multiplier + itr_cm]
                     ,p_out_shift[itr_ic * channels_multiplier + itr_cm]
                     ,out_zero_bias
                     ,p_tmp_out
                    );
            }
        }

        CIRC_BUF_ADD_ROWS_WITH_PAD_VAL(rows_added
                ,rows_to_add
                ,top_pad
                ,bottom_pad
                ,input_row
                ,input_height
                ,input_width
                ,circ_out_height
                ,y_stride
                ,x_padding
                ,y_padding
                ,p_circ_buf
                ,pt_inp
                ,&input_zero_bias_neg
                );

        p_inp_circ = (WORD8 *)p_circ_buf->p_curr;
        for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
        {
            pt_ker = (const WORD8 *)&p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
            COPY_DILATED_KERNEL_TO_SCRATCH(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad, dilation_width);
            bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

            dilated_conv2d_nchw_sym8sxasym8s_hf4_convmul
                ((pWORD8)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
                 ,p_kernel_padded
                 ,p_inp_circ
                 ,bias
                 ,p_circ_buf->rows
                 ,p_circ_buf->row_offset
                 ,kernel_height
                 ,kernel_width
                 ,dilation_height
                 ,dilation_width
                 ,(out_height - itr_oh)
                 ,out_width
                 ,(input_channels * channels_multiplier)
                 ,x_stride
                 ,y_stride
                 ,input_zero_bias
                 ,p_out_multiplier[itr_ic * channels_multiplier + itr_cm]
                 ,p_out_shift[itr_ic * channels_multiplier + itr_cm]
                 ,out_zero_bias
                 ,p_tmp_out
                );
        }
    }
}

static WORD32 gcd(WORD32 a, WORD32 b)
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

static void xa_nn_dilated_conv2d_depthwise_nhwc_per_chan_sym8sxasym8s(
  WORD8 *__restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,const WORD32 *__restrict__ p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  dilation_height
  ,WORD32  dilation_width
  ,WORD32  x_stride
  ,WORD32  y_stride
  ,WORD32  x_padding
  ,WORD32  y_padding
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  input_zero_bias
  ,const WORD32 *p_out_multiplier
  ,const WORD32 *p_out_shift
  ,WORD32  out_zero_bias
  ,WORD32  out_data_format
  ,pVOID p_scratch)
{
    int itr_ow;
    int itr_dh, itr_dw;
    int cols_to_add, left_pad, right_pad, cols_added;
    int input_col;
    const WORD8 *pt_inp;
    WORD8 *p_inp_circ;

    pt_inp = (const WORD8 *)p_inp;
    const WORD8 *pt_ker = p_kernel;

    WORD32 dh_count, dw_count;
    WORD32 y_padding_dh, x_padding_dw;
    WORD32 /*y_stride_dh,*/ x_stride_dw;
    WORD32 out_height_dh, out_width_dw;
    WORD32 rem_dh, rem_dw;
    WORD32 gcd_h, gcd_w;
    WORD32 y_stride_circ_buf;

    gcd_h = gcd(dilation_height, y_stride);
    gcd_w = gcd(dilation_width, x_stride);
    dh_count = dilation_height/gcd_h;
    dw_count = dilation_width/gcd_w;
    y_padding_dh = y_padding;
    //y_stride_dh = y_stride * dh_count;
    out_height_dh = out_height / dh_count;
    out_width_dw = out_width / dw_count;
    rem_dh = out_height - out_height_dh * dh_count;
    y_stride_circ_buf = y_stride / gcd_h;

    UWORD8 input_zero_bias_neg = -input_zero_bias;
    xa_nn_dilated_conv2d_depthwise_init(
            p_scratch,
            input_height,
            input_width,
            input_channels,
            kernel_height,
            kernel_width,
            channels_multiplier,
            dilation_height,
            dilation_width,
            x_stride,
            y_stride,
            x_padding,
            y_padding,
            out_height,
            out_width,
            8,
            0,
            (pVOID)(&input_zero_bias_neg));

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    for(itr_dh = 0; itr_dh < dh_count; itr_dh++, rem_dh--)
    {
        x_padding_dw = x_padding;
        x_stride_dw = x_stride * dw_count;
        rem_dw = out_width - out_width_dw * dw_count;

        WORD32 out_height_dh_cur = out_height_dh + (rem_dh > 0 ? 1 : 0);
        if(out_height_dh_cur < 1)
          break;
        
        for(itr_dw = 0; itr_dw < dw_count; itr_dw++, rem_dw--)
        {
            WORD32 out_width_dw_cur = out_width_dw + (rem_dw > 0 ? 1 : 0);
            DILATED_CIRC_BUF_ADD_COLS_INIT_WITH_PAD_VAL(
                    cols_added,
                    cols_to_add,
                    left_pad,
                    right_pad,
                    input_col,
                    input_height,
                    input_width,
                    input_channels,
                    kernel_height,
                    kernel_width,
                    channels_multiplier,
                    dilation_height,
                    dilation_width,
                    x_stride_dw,
                    y_stride_circ_buf,
                    x_padding_dw,
                    y_padding_dh,
                    out_height_dh_cur,
                    p_circ_buf,
                    pt_inp,
                    &input_zero_bias_neg);

            for(itr_ow = 0; itr_ow < out_width_dw_cur; itr_ow++)
            {
                WORD8 *pt_out = (WORD8 *)&p_out[(itr_dh * out_width + itr_dw + itr_ow * dw_count)*input_channels * channels_multiplier];
                DILATED_CIRC_BUF_ADD_COLS_WITH_PAD_VAL(
                        cols_added,
                        cols_to_add,
                        left_pad,
                        right_pad,
                        input_col,
                        input_height,
                        input_width,
                        input_channels,
                        kernel_height,
                        kernel_width,
                        channels_multiplier,
                        dilation_height,
                        dilation_width,
                        x_stride_dw,
                        y_stride_circ_buf,
                        x_padding_dw,
                        y_padding_dh,
                        out_height_dh_cur,
                        p_circ_buf,
                        pt_inp,
                        &input_zero_bias_neg);

                p_inp_circ = (WORD8 *)p_circ_buf->p_curr;

                conv2d_per_chan_nhwc_sym8sxasym8s
                        (pt_out
                        ,pt_ker
                        ,p_inp_circ
                        ,p_bias
                        ,kernel_height
                        ,kernel_width
                        ,out_height_dh_cur
                        ,out_width * dh_count
                        ,(input_channels * channels_multiplier)
                        ,y_stride_circ_buf
                        ,input_zero_bias
                        ,p_out_multiplier
                        ,p_out_shift
                        ,out_zero_bias
                        ,p_state->p_scratch);
            }
            x_padding_dw -= x_stride;
        }
        y_padding_dh -= y_stride;
    }
}


WORD32 xa_nn_dilated_conv2d_depthwise_per_chan_sym8sxasym8s
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,const WORD32 *__restrict__ p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  dilation_height
  ,WORD32  dilation_width
  ,WORD32  x_stride
  ,WORD32  y_stride
  ,WORD32  x_padding
  ,WORD32  y_padding
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  input_zero_bias
  ,const WORD32 *p_out_multiplier
  ,const WORD32 *p_out_shift
  ,WORD32  out_zero_bias
  ,WORD32  inp_data_format
  ,WORD32  out_data_format
  ,pVOID p_scratch)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shift, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shift, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_height <= 0 || dilation_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias > 128 || input_zero_bias < -127), -1);
  int i;
  for(i = 0; i < input_channels*channels_multiplier; i++)
    XA_NNLIB_ARG_CHK_COND((p_out_shift[i] < -31 || p_out_shift[i] > 31), -1);
  XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

    if(inp_data_format == 0)
    {
        xa_nn_dilated_conv2d_depthwise_nhwc_per_chan_sym8sxasym8s(
                p_out,
                p_kernel,
                p_inp,
                p_bias,
                input_height,
                input_width,
                input_channels,
                kernel_height,
                kernel_width,
                channels_multiplier,
                dilation_height,
                dilation_width,
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
                p_scratch);
    }
    else if(inp_data_format == 1)
    {
        xa_nn_dilated_conv2d_depthwise_nchw_per_chan_sym8sxasym8s(
                p_out,
                p_kernel,
                p_inp,
                p_bias,
                input_height,
                input_width,
                input_channels,
                kernel_height,
                kernel_width,
                channels_multiplier,
                dilation_height,
                dilation_width,
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
                p_scratch);
    }

    return 0;
}
