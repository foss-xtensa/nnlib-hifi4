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

#if XCHAL_HAVE_HIFI1S
static inline ae_int32x2 MultiplyByQuantizedMultiplier_ref(ae_int64 d_x,
                                             int32_t quantized_multiplier,
                                             int shift){
  ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
  ae_int64 q = AE_MUL48X16_0(d_x, d_red_mul16);
  ae_int32x2 result = AE_ROUNDAV32X2F64SASYM (q, q, 15 - shift);   // only lower 32 is valid result
  return result;
}
#else
static inline ae_int32x2 MultiplyByQuantizedMultiplier_ref(ae_int64 d_x,
                                             int32_t quantized_multiplier,
                                             int shift){
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
#endif

#if 0
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

static inline void conv2d_nchw_sym8sxsym16s_hf4_convmul
(pWORD16 __restrict__ p_out  /* Output:  [Stream] [(out_stride): (actual_out_height): (actual_out_width)] */
 ,const WORD8 *__restrict__ p_ker  /* Kernel:  [Block] [1:             kernel_height:       kernel_width_pad] */
 ,const WORD16 *__restrict__ p_inp  /* Input:   [Block] [1:             input_height:        input_width] */
 ,WORD64 bias
 ,int input_height
 ,int input_width
 ,int kernel_height
 ,int kernel_width
 ,int actual_out_height      /* This is the actual output height, processing should be limited to it. */
 ,int actual_out_width       /* This is the actual output width, processing should be limited to it. */
 ,int out_stride
 ,int x_stride
 ,int y_stride
 ,WORD32  out_multiplier
 ,WORD32  out_shift
 ,pWORD64 __restrict__ p_scratch /* Scratch: [Block] [1:             (actual_out_height): (out_width)] */
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
    ae_int64 *scratch_ptr = (ae_int64 *)p_scratch;

    ae_int64 _ae_int64_sat_bias;
    _ae_int64_sat_bias = bias;

    {
      for(i = 0; i < actual_out_height; i++)
      {
          scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
          for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
          {
              accu_int64_0 = AE_ZERO64();
              accu_int64_1 = AE_ZERO64();
              accu_int64_2 = AE_ZERO64();
              accu_int64_3 = AE_ZERO64();

              for(k = 0; k < kernel_height_pad; k += 2)
              {
                  ae_int16x4 *pt_inp0 = (ae_int16x4 *)(p_inp);
                  AE_ADDCIRC16X4_XC
                      (pt_inp0
                       ,((sizeof(WORD16)) * ((i * y_stride * input_width) + j*4 + k*input_width))
                      );
                  ae_int16x4 *pt_inp1 = (ae_int16x4 *)(p_inp);
                  AE_ADDCIRC16X4_XC
                      (pt_inp1
                       ,((sizeof(WORD16)) * ((i * y_stride * input_width) + j*4 + (k+1)*input_width))
                      );
                  pWORD8 pt_ker0 = (pWORD8)(p_ker + k*kernel_width_pad);
                  pWORD8 pt_ker1 = (pWORD8)(p_ker + (k+1)*kernel_width_pad);
                  ae_int16x4 d_inp1, d_inp2, d_inp3, d_inp4, d_inp5, d_inp6;
                  ae_int16x4 d_ker0, d_ker1;
                  ae_int16x4 d_inp00, d_inp01, d_inp10, d_inp11;

                  d_inp00 = *pt_inp0++;
                  d_inp10 = *pt_inp1++;

#pragma no_unroll /* (kernel_width_pad/4) is typically a small value, so disabling unroll */
                  for(l = 0; l < (kernel_width_pad>>2); l++)
                  {
                      d_inp01 = *pt_inp0++;
                      d_inp11 = *pt_inp1++;
#if XCHAL_HAVE_HIFI1
                      AE_L8X4S_IP(d_ker0, pt_ker0, 4);
                      AE_L8X4S_IP(d_ker1, pt_ker1, 4);
#else
                      AE_L8X4F_IP(d_ker0, pt_ker0, 4);
                      AE_L8X4F_IP(d_ker1, pt_ker1, 4);
#endif
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

              WORD32 _WORD32_scratch_j = (j << 2);
#if XCHAL_HAVE_HIFI1
              scratch_ptr[_WORD32_scratch_j + 0] = accu_int64_0;
              scratch_ptr[_WORD32_scratch_j + 1] = accu_int64_1;
              scratch_ptr[_WORD32_scratch_j + 2] = accu_int64_2;
              scratch_ptr[_WORD32_scratch_j + 3] = accu_int64_3;
#else
              scratch_ptr[_WORD32_scratch_j + 0] = AE_SRAI64(accu_int64_0, 8);
              scratch_ptr[_WORD32_scratch_j + 1] = AE_SRAI64(accu_int64_1, 8);
              scratch_ptr[_WORD32_scratch_j + 2] = AE_SRAI64(accu_int64_2, 8);
              scratch_ptr[_WORD32_scratch_j + 3] = AE_SRAI64(accu_int64_3, 8);
#endif
          }
      }
    }


    /* Here we store output based on strides. For values in a row, values
     * will be picked from it as per 'x_stride'. No need to worry about
     * height dimension, since we took care of it by efficient row
     * accesses. */
    scratch_ptr = (ae_int64 *) p_scratch;

    for(i = 0; i < actual_out_height; i++)
    {
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
        ae_int16 *out_ptr  = (ae_int16 *) p_out + (i * out_stride * actual_out_width);

        for(j = 0; j < actual_out_width; j++)
        {
            accu_int64_0 = scratch_ptr[(j * x_stride)];

            accu_int64_0 = AE_ADD64S(accu_int64_0, _ae_int64_sat_bias);
            ae_int32x2 outval32 = MultiplyByQuantizedMultiplier_ref(accu_int64_0, out_multiplier, out_shift);
            ae_int16x4 outval16 = AE_SAT16X4(outval32, outval32);
#if XCHAL_HAVE_HIFI1S
            out_ptr[(j * out_stride)] = AE_MOVAD16_0(outval16);
#else
            out_ptr[(j * out_stride)] = AE_MOVAD16_3(outval16);
#endif
        }
    }
}

static void xa_nn_conv2d_depthwise_per_chan_nchw_sym8sxsym16s
(pWORD16 __restrict__ p_out
 ,const WORD8 *__restrict__ p_kernel
 ,const WORD16 *__restrict__ p_inp
 ,const WORD64 *__restrict__ p_bias
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
 ,const WORD32  *p_out_multiplier
 ,const WORD32  *p_out_shift
,pVOID p_scratch
)
{
    int input_zero_bias_neg = 0;
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
         ,16
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
    const WORD16 *pt_inp;
    pWORD16 p_inp_circ;
    int i;

    WORD8 *p_kernel_padded = (WORD8 *)(p_state->p_scratch);
    p_kernel_padded = (WORD8 *)ALIGN_PTR(p_kernel_padded, 8);
    pWORD64 p_tmp_out = (pWORD64)(p_kernel_padded + kernel_height_pad * kernel_width_pad);
    p_tmp_out = (pWORD64)ALIGN_PTR(p_tmp_out, 8);

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    WORD64 bias = 0;

    /* Initialize whole scratch for padded kernel to padding value, after this
       we only have to copy actual kernel values, padding area should remain
       untouched */

    WORD8 *pae_ker_pad8 = (WORD8 *)p_kernel_padded;
    for(i = 0; i < (kernel_height_pad * kernel_width_pad); i++) {
      pae_ker_pad8[i] = 0;
    }

    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = (const WORD16 *)&p_inp[itr_ic*input_height*input_width];

        CIRC_BUF_ADD_ROWS_INIT(rows_added
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
                );

        for(itr_oh = 0; itr_oh < out_height - (circ_out_height - 1); itr_oh += circ_out_height)
        {
            CIRC_BUF_ADD_ROWS(rows_added
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
                    );

            p_inp_circ = p_circ_buf->p_curr;

            for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
            {
                pt_ker = (const WORD8 *)&p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
                COPY_KERNEL_TO_SCRATCH_8b(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
                bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

                conv2d_nchw_sym8sxsym16s_hf4_convmul
                    ((pWORD16)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
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
                     ,p_out_multiplier[itr_ic * channels_multiplier + itr_cm]
                     ,p_out_shift[itr_ic * channels_multiplier + itr_cm]
                     ,p_tmp_out
                    );
            }
        }

        CIRC_BUF_ADD_ROWS(rows_added
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
                );

        p_inp_circ = p_circ_buf->p_curr;


        for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
        {
            pt_ker = (const WORD8 *)&p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
            COPY_KERNEL_TO_SCRATCH_8b(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
            bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

            conv2d_nchw_sym8sxsym16s_hf4_convmul
                ((pWORD16)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
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
                 ,p_out_multiplier[itr_ic * channels_multiplier + itr_cm]
                 ,p_out_shift[itr_ic * channels_multiplier + itr_cm]
                 ,p_tmp_out
                );
        }
    }
}

static inline void conv2d_per_chan_nhwc_sym8sxsym16s
(pWORD16 __restrict__ p_out
 ,const WORD8 *__restrict__ p_ker
 ,const WORD16 *__restrict__ p_inp
 ,const WORD64 *p_bias
 ,int kernel_height
 ,int kernel_width
 ,int out_height
 ,int out_width
 ,int out_channels
 ,int y_stride
 ,const WORD32 *p_out_multiplier
 ,const WORD32 *p_out_shift
 ,pWORD32 __restrict__ p_scratch
 )
{
    WORD32 out_channels_pad;
    WORD32 itr_oh, itr_ch, itr_kw;
    ae_int16x4 * pt_inp0, *pt_inp1;
    const WORD8 *pt_ker;
    WORD8 *p_ker_scr;
    pWORD16 out_ptr0, out_ptr1;
    ae_int16x4 d_ker;
    const ae_int64 *pt_bias;
    ae_int64 d_acc0, d_acc1, d_acc2, d_acc3, d_acc4, d_acc5, d_acc6, d_acc7;
    ae_int64 d_bias0, d_bias1, d_bias2, d_bias3;
    ae_int16x4 d_acc16x4;

    out_channels_pad = (out_channels + 3)&(~3);

    pt_bias = (ae_int64 *)p_bias;
    for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
    {
        d_bias0 = *pt_bias++;
        d_bias1 = *pt_bias++;
        d_bias2 = *pt_bias++;
        d_bias3 = *pt_bias++;

        pt_ker = (const WORD8 *)(&p_ker[itr_ch]);
        p_ker_scr = (WORD8 *)p_scratch;
        COPY_KERNEL_TO_SCRATCH_NHWC_4_8b(p_ker_scr, pt_ker, kernel_height, kernel_width, out_channels);

        int r_shift[4];
        r_shift[0] = p_out_shift[itr_ch+0];
        r_shift[1] = p_out_shift[itr_ch+1];
        r_shift[2] = p_out_shift[itr_ch+2];
        r_shift[3] = p_out_shift[itr_ch+3];

        for(itr_oh = 0; itr_oh < (out_height); itr_oh+=2)        
        {
            out_ptr0 = (WORD16 *)(&p_out[itr_oh*out_channels*out_width]);
            out_ptr1 = (WORD16 *)(&p_out[(itr_oh+1)*out_channels*out_width]);

            pt_inp0 = (ae_int16x4 *)p_inp;
            pt_inp1 = (ae_int16x4 *)p_inp;
            AE_ADDCIRC16X4_XC(pt_inp0, 2*(itr_ch + itr_oh*y_stride*kernel_width*out_channels_pad));
            AE_ADDCIRC16X4_XC(pt_inp1, 2*(itr_ch + (itr_oh+1)*y_stride*kernel_width*out_channels_pad));
            p_ker_scr = (WORD8 *)p_scratch;
            d_acc0 = 0;
            d_acc1 = 0;
            d_acc2 = 0;
            d_acc3 = 0;
            d_acc4 = 0;
            d_acc5 = 0;
            d_acc6 = 0;
            d_acc7 = 0;
            
            /* The following do-while loop allows the use of 32 bit accumulator for 16bitx8bit mac operation.
             * The OverflowConst makes sure that 32 bit accumulator does not overflow, at the end of OverflowConst-times
             * accumulations, the 32-bit accumulator is offloaded to 64-bit accumulator. The 32-bit accumulator is also
             * reset for next batch. */

            int ker_loop_count = kernel_height * kernel_width;
            int OverflowConst = 64; /* 128 should also work*/
            do{
              int tmp_loop_cnt = XT_MIN(ker_loop_count, OverflowConst);
              ae_int32x2 d_acc32_0 = 0;
              ae_int32x2 d_acc32_1 = 0;
              ae_int32x2 d_acc32_2 = 0;
              ae_int32x2 d_acc32_3 = 0;

              for(itr_kw = 0; itr_kw < tmp_loop_cnt; itr_kw++)
              {
                  ae_int16x4 d_inp0, d_inp1;
                  AE_L16X4_XC(d_inp0, pt_inp0, out_channels_pad*2);
                  AE_L16X4_XC(d_inp1, pt_inp1, out_channels_pad*2);
#if XCHAL_HAVE_HIFI1
                  AE_L8X4S_IP(d_ker, p_ker_scr, 4);
#else
                  AE_L8X4F_IP(d_ker, p_ker_scr, 4);
                  d_ker = AE_SRAI16(d_ker, 8);
#endif
                  AE_MULA16X4(d_acc32_0, d_acc32_1, d_inp0, d_ker);
                  AE_MULA16X4(d_acc32_2, d_acc32_3, d_inp1, d_ker);
              }
              ae_int32x2 ONE32X2 = 1;
              AE_MULA32_HL(d_acc0, d_acc32_0, ONE32X2);
              AE_MULA32_LL(d_acc1, d_acc32_0, ONE32X2);
              AE_MULA32_HL(d_acc2, d_acc32_1, ONE32X2);
              AE_MULA32_LL(d_acc3, d_acc32_1, ONE32X2);
              AE_MULA32_HL(d_acc4, d_acc32_2, ONE32X2);
              AE_MULA32_LL(d_acc5, d_acc32_2, ONE32X2);
              AE_MULA32_HL(d_acc6, d_acc32_3, ONE32X2);
              AE_MULA32_LL(d_acc7, d_acc32_3, ONE32X2);
              ker_loop_count -= OverflowConst;
            } while(ker_loop_count > 0);

            d_acc0 = AE_ADD64(d_acc0, d_bias0);
            d_acc1 = AE_ADD64(d_acc1, d_bias1);
            d_acc2 = AE_ADD64(d_acc2, d_bias2);
            d_acc3 = AE_ADD64(d_acc3, d_bias3);
            d_acc4 = AE_ADD64(d_acc4, d_bias0);
            d_acc5 = AE_ADD64(d_acc5, d_bias1);
            d_acc6 = AE_ADD64(d_acc6, d_bias2);
            d_acc7 = AE_ADD64(d_acc7, d_bias3);

            ae_int32x2 tmp32_0 = MultiplyByQuantizedMultiplier_ref(d_acc0, p_out_multiplier[itr_ch + 0], r_shift[0]);
            ae_int32x2 tmp32_1 = MultiplyByQuantizedMultiplier_ref(d_acc1, p_out_multiplier[itr_ch + 1], r_shift[1]);
            ae_int32x2 tmp32_2 = MultiplyByQuantizedMultiplier_ref(d_acc2, p_out_multiplier[itr_ch + 2], r_shift[2]); 
            ae_int32x2 tmp32_3 = MultiplyByQuantizedMultiplier_ref(d_acc3, p_out_multiplier[itr_ch + 3], r_shift[3]); 

            ae_int32x2 d32_acc0 = AE_SEL32_LL(tmp32_0, tmp32_1);
            ae_int32x2 d32_acc1 = AE_SEL32_LL(tmp32_2, tmp32_3);

            d_acc16x4 = AE_SAT16X4(d32_acc0, d32_acc1);

#if XCHAL_HAVE_HIFI1S
            ae_valign align_out = AE_ZALIGN64();
            ae_int16x4 * st_ptr = (ae_int16x4 *)&out_ptr0[itr_ch];
            AE_SAV16X4_XP(d_acc16x4, align_out, st_ptr, (XT_MIN(out_channels-itr_ch, 4))<<1);
            AE_SA64POS_FP(align_out, st_ptr);
#else
            WORD32 i;
            for(i = 0; i < XT_MIN(out_channels-itr_ch, 4); i++)
            {
                out_ptr0[itr_ch+i] =(WORD16)(AE_MOVAD16_3(d_acc16x4));
                d_acc16x4 = AE_SEL16_6543(d_acc16x4, d_acc16x4);
            }
#endif

            if(out_height - itr_oh >= 2)
            {
              tmp32_0 = MultiplyByQuantizedMultiplier_ref(d_acc4, p_out_multiplier[itr_ch + 0], r_shift[0]); 
              tmp32_1 = MultiplyByQuantizedMultiplier_ref(d_acc5, p_out_multiplier[itr_ch + 1], r_shift[1]); 
              tmp32_2 = MultiplyByQuantizedMultiplier_ref(d_acc6, p_out_multiplier[itr_ch + 2], r_shift[2]); 
              tmp32_3 = MultiplyByQuantizedMultiplier_ref(d_acc7, p_out_multiplier[itr_ch + 3], r_shift[3]);
              ae_int32x2 d32_acc2 = AE_SEL32_LL(tmp32_0, tmp32_1);
              ae_int32x2 d32_acc3 = AE_SEL32_LL(tmp32_2, tmp32_3);
                
              d_acc16x4 = AE_SAT16X4(d32_acc2, d32_acc3);
#if XCHAL_HAVE_HIFI1S
              ae_valign align_out1 = AE_ZALIGN64();
              ae_int16x4 * st_ptr1 = (ae_int16x4 *)&out_ptr1[itr_ch];
              AE_SAV16X4_XP(d_acc16x4, align_out1, st_ptr1, (XT_MIN(out_channels-itr_ch, 4))<<1);
              AE_SA64POS_FP(align_out1, st_ptr1);
#else
              for(i = 0; i < XT_MIN(out_channels-itr_ch, 4); i++)
              {
                out_ptr1[itr_ch+i] = (WORD16)(AE_MOVAD16_3(d_acc16x4));
                d_acc16x4 = AE_SEL16_6543(d_acc16x4, d_acc16x4);
              }
#endif
            }
        }
    }
}


static void xa_nn_conv2d_depthwise_per_chan_nhwc_sym8sxsym16s
(pWORD16 __restrict__ p_out
 ,const WORD8 *__restrict__ p_kernel
 ,const WORD16 *__restrict__ p_inp
 ,const WORD64 *__restrict__ p_bias
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
 ,const WORD32  *p_out_multiplier
 ,const WORD32  *p_out_shift
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
         ,16
         ,0
         ,(pVOID)(&temp_pad_val)
        );

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ow;
    int cols_to_add, left_pad, right_pad, cols_added;
    int input_col;
    const WORD16 *pt_inp;
    pWORD16 p_inp_circ;

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    pt_inp = (const WORD16 *)p_inp;

    CIRC_BUF_ADD_COLS_INIT(cols_added
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
            );
    {
      for(itr_ow = 0; itr_ow < out_width; itr_ow++)
      {
          CIRC_BUF_ADD_COLS(cols_added
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
                  );

          p_inp_circ = (WORD16 *)p_circ_buf->p_curr;

          conv2d_per_chan_nhwc_sym8sxsym16s
              ((pWORD16)(&p_out[itr_ow*input_channels*channels_multiplier])
               ,p_kernel
               ,p_inp_circ
               ,p_bias
               ,kernel_height
               ,kernel_width
               ,out_height
               ,out_width
               ,(input_channels * channels_multiplier)
               ,y_stride
               ,p_out_multiplier
               ,p_out_shift
               ,p_state->p_scratch
              );
      }
    }
}


WORD32 xa_nn_conv2d_depthwise_per_chan_sym8sxsym16s
  (pWORD16 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD16 *__restrict__ p_inp
  ,const WORD64 *__restrict__ p_bias
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
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(input_zero_bias != 0, -1);
    for(i = 0; i < input_channels*channels_multiplier; i++)
      XA_NNLIB_ARG_CHK_COND((p_out_shift[i] < -31 || p_out_shift[i] > 15), -1);
    XA_NNLIB_ARG_CHK_COND((out_zero_bias != 0 ), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

    if(inp_data_format == 0)
    {
        xa_nn_conv2d_depthwise_per_chan_nhwc_sym8sxsym16s
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
             ,p_out_multiplier
             ,p_out_shift
             ,p_scratch);
    }
    else if(inp_data_format == 1)
    {
        xa_nn_conv2d_depthwise_per_chan_nchw_sym8sxsym16s
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
             ,p_out_multiplier
             ,p_out_shift
             ,p_scratch);
    }

    return 0;
}
