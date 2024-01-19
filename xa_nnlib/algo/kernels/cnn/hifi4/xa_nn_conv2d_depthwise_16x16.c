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

/* 2D Convolution implementation */
static inline void conv2d_nchw_16x16_hf4_convmul
(pWORD16 __restrict__ p_out  /* Output:  [Stream] [(out_stride): (actual_out_height): (actual_out_width)] */
 ,const WORD16 *__restrict__ p_ker  /* Kernel:  [Block] [1:             kernel_height:       kernel_width_pad] */
 ,const WORD16 *__restrict__ p_inp  /* Input:   [Block] [1:             input_height:        input_width] */
 ,WORD16 bias
 ,int input_height
 ,int input_width
 ,int kernel_height
 ,int kernel_width
 ,int actual_out_height       /* This is the actual output height, processing should be limited to it. */
 ,int actual_out_width        /* This is the actual output width, processing should be limited to it. */
 ,int out_stride
 ,int x_stride
 ,int y_stride
 ,int acc_shift
 ,int bias_shift
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
    _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) (*((ae_int16 *) &bias))), bias_shift);

    if(kernel_width_pad==12)
    {
      ae_int16x4 d_inp00, d_inp01, d_inp02, d_inp03;
      ae_int16x4 d_ker0, d_ker1, d_ker2;
      ae_int16x4 d_inp1, d_inp2, d_inp3, d_inp4, d_inp5, d_inp6, d_inp7, d_inp8, d_inp9;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
#pragma loop_count min=1
        for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          ae_int16x4 *pt_inp = (ae_int16x4 *)(p_inp);
          AE_ADDCIRC16X4_XC
              (pt_inp
                ,((sizeof(WORD16)) * ((i * y_stride * input_width) + j*4))
              );
          ae_int16x4 *pt_ker = (ae_int16x4 *)p_ker;
#pragma loop_count min=1
#pragma no_unroll
          for(k = 0; k < kernel_height; k++)
          {
            AE_L16X4_XC(d_inp00, pt_inp, 8);
            AE_L16X4_XC(d_inp01, pt_inp, 8);
            AE_L16X4_XC(d_inp02, pt_inp, 8);
            AE_L16X4_XC(d_inp03, pt_inp, (((sizeof(WORD16)) * input_width) - 24));
            d_ker0 = *pt_ker++;
            d_ker1 = *pt_ker++;
            d_ker2 = *pt_ker++;
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
          WORD32 _WORD32_scratch_j = (j << 2);

          scratch_ptr[_WORD32_scratch_j + 0] = accu_int64_0;
          scratch_ptr[_WORD32_scratch_j + 1] = accu_int64_1;
          scratch_ptr[_WORD32_scratch_j + 2] = accu_int64_2;
          scratch_ptr[_WORD32_scratch_j + 3] = accu_int64_3;
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
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
#pragma loop_count min=1
        for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          ae_int16x4 *pt_inp = (ae_int16x4 *)(p_inp);
          AE_ADDCIRC16X4_XC
              (pt_inp
                ,((sizeof(WORD16)) * ((i * y_stride * input_width) + j*4))
              );
          ae_int16x4 *pt_ker = (ae_int16x4 *)p_ker;
#pragma loop_count min=1
#pragma no_unroll
          for(k = 0; k < kernel_height; k++)
          {
            AE_L16X4_XC(d_inp00, pt_inp, 8);
            AE_L16X4_XC(d_inp01, pt_inp, 8);
            AE_L16X4_XC(d_inp02, pt_inp, (((sizeof(WORD16)) * input_width) - 16));
            d_ker0 = *pt_ker++;
            d_ker1 = *pt_ker++;
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
          WORD32 _WORD32_scratch_j = (j << 2);

          scratch_ptr[_WORD32_scratch_j + 0] = accu_int64_0;
          scratch_ptr[_WORD32_scratch_j + 1] = accu_int64_1;
          scratch_ptr[_WORD32_scratch_j + 2] = accu_int64_2;
          scratch_ptr[_WORD32_scratch_j + 3] = accu_int64_3;
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
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
#pragma loop_count min=1
        for(j = 0; j < (output_width_for_x_stride_1>>2); j++)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          ae_int16x4 *pt_inp = (ae_int16x4 *)(p_inp);
          AE_ADDCIRC16X4_XC
              (pt_inp
                ,((sizeof(WORD16)) * ((i * y_stride * input_width) + j*4))
              );
          ae_int16x4 *pt_ker = (ae_int16x4 *)p_ker;
#pragma loop_count min=1
#pragma no_unroll
          for(k = 0; k < kernel_height; k++)
          {
            AE_L16X4_XC(d_inp00, pt_inp, 8);
            AE_L16X4_XC(d_inp01, pt_inp, (((sizeof(WORD16)) * input_width) - 8));
            d_ker0 = *pt_ker++;
            d_inp1 = AE_SEL16_6543(d_inp00, d_inp01);
            d_inp2 = AE_SEL16_5432(d_inp00, d_inp01);
            d_inp3 = AE_SEL16_4321(d_inp00, d_inp01);
            AE_MULAAAAQ16(accu_int64_0, d_inp00, d_ker0);
            AE_MULAAAAQ16(accu_int64_1, d_inp1, d_ker0);
            AE_MULAAAAQ16(accu_int64_2, d_inp2, d_ker0);
            AE_MULAAAAQ16(accu_int64_3, d_inp3, d_ker0);
          }
          WORD32 _WORD32_scratch_j = (j << 2);

          scratch_ptr[_WORD32_scratch_j + 0] = accu_int64_0;
          scratch_ptr[_WORD32_scratch_j + 1] = accu_int64_1;
          scratch_ptr[_WORD32_scratch_j + 2] = accu_int64_2;
          scratch_ptr[_WORD32_scratch_j + 3] = accu_int64_3;
        }
      }
    }

    else
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
#pragma loop_count min=1
              for(k = 0; k < kernel_height; k++)
              {
                  ae_int16x4 *pt_inp = (ae_int16x4 *)(p_inp);
                  AE_ADDCIRC16X4_XC
                      (pt_inp
                       ,((sizeof(WORD16)) * ((i * y_stride * input_width) + j*4 + k*input_width))
                      );
                  ae_int16x4 *pt_ker = (ae_int16x4 *)(p_ker + k*kernel_width_pad);
                  ae_int16x4 d_inp, d_ker;
                  ae_int16x4 d_inp0, d_inp1, d_inp2, d_inp3;
                  d_inp0 = *pt_inp++;
#pragma loop_count min=1
#pragma no_unroll
                  for(l = 0; l < (kernel_width_pad>>2); l++)
                  {
                      d_inp = *pt_inp++;
                      d_ker = *pt_ker++;
                      d_inp1 = AE_SEL16_6543(d_inp0, d_inp);
                      d_inp2 = AE_SEL16_5432(d_inp0, d_inp);
                      d_inp3 = AE_SEL16_4321(d_inp0, d_inp);
                      AE_MULAAAAQ16(accu_int64_0, d_inp0, d_ker);
                      AE_MULAAAAQ16(accu_int64_1, d_inp1, d_ker);
                      AE_MULAAAAQ16(accu_int64_2, d_inp2, d_ker);
                      AE_MULAAAAQ16(accu_int64_3, d_inp3, d_ker);
                      d_inp0 = d_inp;
                  }
              }

              WORD32 _WORD32_scratch_j = (j << 2);

              scratch_ptr[_WORD32_scratch_j + 0] = accu_int64_0;
              scratch_ptr[_WORD32_scratch_j + 1] = accu_int64_1;
              scratch_ptr[_WORD32_scratch_j + 2] = accu_int64_2;
              scratch_ptr[_WORD32_scratch_j + 3] = accu_int64_3;

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
            accu_int64_0 = AE_SLAA64S(accu_int64_0, acc_shift);
            out_ptr[(j * out_stride)] =AE_SAT16X4(0, AE_ROUND32F64SSYM(accu_int64_0));
        }
    }
}

WORD32 xa_nn_conv2d_depthwise_nchw_16x16
(pWORD16 __restrict__ p_out
 ,const WORD16 *__restrict__ p_kernel
 ,const WORD16 *__restrict__ p_inp
 ,const WORD16 *__restrict__ p_bias
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
 ,WORD32  acc_shift
 ,WORD32  bias_shift
 ,pVOID p_scratch
)
{
    WORD16 pad_val = 0;
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
         ,(pVOID)(&pad_val)
        );

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ic, itr_cm, itr_oh, i;
    int circ_out_height = (p_circ_buf->rows - kernel_height)/y_stride + 1;
    int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
    int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
    int rows_to_add, top_pad, bottom_pad, rows_added;
    int input_row;
    const WORD16 *pt_ker;
    const WORD16 *pt_inp;
    pWORD16 p_inp_circ;

    WORD16 *p_kernel_padded = (WORD16 *)(p_state->p_scratch);
    p_kernel_padded = (WORD16 *)ALIGN_PTR(p_kernel_padded, 8);
    pWORD64 p_tmp_out = (pWORD64)(p_kernel_padded + kernel_height_pad * kernel_width_pad);
    p_tmp_out = (pWORD64)ALIGN_PTR(p_tmp_out, 8);

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    WORD16 bias = 0;

    acc_shift = acc_shift + 32;
    LIMIT_ACC_LSH;

    /* Initialize whole scratch for padded kernel to padding value, after this
       we only have to copy actual kernel values, padding area should remain
       untouched */
    ae_int16x4 *pae_ker_pad = (ae_int16x4 *)p_kernel_padded;
    for(i = 0; i < ((kernel_height_pad * kernel_width_pad) >> 2); i++)
    {
      pae_ker_pad[i] = AE_ZERO16();
    }

    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = &p_inp[itr_ic*input_height*input_width];

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

            p_inp_circ = (WORD16 *)p_circ_buf->p_curr;

            for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
            {
                pt_ker = &p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
                COPY_KERNEL_TO_SCRATCH_16b(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
                bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

                conv2d_nchw_16x16_hf4_convmul
                    ((&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
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
                     ,acc_shift
                     ,bias_shift
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

        p_inp_circ = (WORD16 *)p_circ_buf->p_curr;

        for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
        {
            pt_ker = &p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
            COPY_KERNEL_TO_SCRATCH_16b(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
            bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

            conv2d_nchw_16x16_hf4_convmul
                ((&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
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
                 ,acc_shift
                 ,bias_shift
                 ,p_tmp_out
                );
        }
    }

    return 0;
}

/* 2D Convolution implementation */
static inline void conv2d_nhwc_16x16
(pWORD16 __restrict__ p_out
 ,const WORD16 *__restrict__ p_ker
 ,const WORD16 *__restrict__ p_inp
 ,const WORD16 *p_bias
 ,int kernel_height
 ,int kernel_width
 ,int out_height
 ,int out_width
 ,int out_channels
 ,int y_stride
 ,WORD32  acc_shift
 ,WORD32  bias_shift
 )
{
    WORD32 ker_channels_pad, inp_channels_pad;
    WORD32 i, itr_oh, itr_ch, itr_kw;
    ae_int16x4 *pt_inp0, *pt_inp1, *pt_ker;
    pWORD16 out_ptr0, out_ptr1;
    ae_int16x4 d_inp0, d_inp1, d_ker;
    ae_int32x2 d32_ker0, d32_ker1;
    const ae_int16 *pt_bias;
    ae_valign ker_a;
    ae_int32x2 d_acc0, d_acc1;
    ae_int32x2 d_acc2, d_acc3;
    ae_int64 d64_bias0, d64_bias1, d64_bias2, d64_bias3;
    ae_int64 d64_acc0, d64_acc1, d64_acc2, d64_acc3;
    ae_int64 d64_acc4, d64_acc5, d64_acc6, d64_acc7;
    ae_int16x4 d_acc16x4;

    ker_channels_pad = out_channels;
    inp_channels_pad = (out_channels + 3)&(~3);

    for(itr_oh = 0; itr_oh < out_height; itr_oh+=2)
    {
        out_ptr0 = (WORD16 *)(&p_out[itr_oh*out_channels*out_width]);
        out_ptr1 = (WORD16 *)(&p_out[(itr_oh+1)*out_channels*out_width]);
        pt_bias = (const ae_int16 *)p_bias;
        for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
        {
            pt_inp0 = (ae_int16x4 *)p_inp;
            pt_inp1 = (ae_int16x4 *)p_inp;
            AE_ADDCIRC16X4_XC(pt_inp0, (itr_ch + itr_oh*y_stride*kernel_width*inp_channels_pad)*sizeof(WORD16));
            AE_ADDCIRC16X4_XC(pt_inp1, (itr_ch + (itr_oh+1)*y_stride*kernel_width*inp_channels_pad)*sizeof(WORD16));
            pt_ker = (ae_int16x4 *)(&p_ker[itr_ch]);
            ker_a = AE_LA64_PP(pt_ker);
            d64_acc0 = AE_ZERO64();
            d64_acc1 = AE_ZERO64();
            d64_acc2 = AE_ZERO64();
            d64_acc3 = AE_ZERO64();
            d64_acc4 = AE_ZERO64();
            d64_acc5 = AE_ZERO64();
            d64_acc6 = AE_ZERO64();
            d64_acc7 = AE_ZERO64();
#pragma loop_count min=1
#pragma no_unroll
            for(itr_kw = 0; itr_kw < kernel_height * kernel_width; itr_kw++)
            {
                AE_L16X4_XC(d_inp0, pt_inp0, inp_channels_pad*sizeof(WORD16));
                AE_L16X4_XC(d_inp1, pt_inp1, inp_channels_pad*sizeof(WORD16));
                AE_LA16X4_IP(d_ker, ker_a, pt_ker);
                pt_ker = (ae_int16x4 *)((WORD8 *)pt_ker + sizeof(WORD16) * (ker_channels_pad - 4));
                ker_a = AE_LA64_PP(pt_ker);
                /* Need to accumulate in 64 bit accumulator so converting
                muls 32x16 muls, so that mul and accumulation can be combined */
                d32_ker0 = AE_SEXT32X2D16_32(d_ker);
                d32_ker1 = AE_SEXT32X2D16_10(d_ker);
                AE_MULA32X16_H3(d64_acc0, d32_ker0, d_inp0);
                AE_MULA32X16_L2(d64_acc1, d32_ker0, d_inp0);
                AE_MULA32X16_H1(d64_acc2, d32_ker1, d_inp0);
                AE_MULA32X16_L0(d64_acc3, d32_ker1, d_inp0);
                AE_MULA32X16_H3(d64_acc4, d32_ker0, d_inp1);
                AE_MULA32X16_L2(d64_acc5, d32_ker0, d_inp1);
                AE_MULA32X16_H1(d64_acc6, d32_ker1, d_inp1);
                AE_MULA32X16_L0(d64_acc7, d32_ker1, d_inp1);

            }
            d64_bias0 = AE_MOVINT64_FROMINT16(pt_bias[itr_ch]);
            d64_bias1 = AE_MOVINT64_FROMINT16(pt_bias[itr_ch+1]);
            d64_bias2 = AE_MOVINT64_FROMINT16(pt_bias[itr_ch+2]);
            d64_bias3 = AE_MOVINT64_FROMINT16(pt_bias[itr_ch+3]);
            d64_bias0 = AE_SLAA64S(AE_SRAI64(d64_bias0, 48), bias_shift);
            d64_bias1 = AE_SLAA64S(AE_SRAI64(d64_bias1, 48), bias_shift);
            d64_bias2 = AE_SLAA64S(AE_SRAI64(d64_bias2, 48), bias_shift);
            d64_bias3 = AE_SLAA64S(AE_SRAI64(d64_bias3, 48), bias_shift);

            d64_acc0 = AE_ADD64S(d64_acc0, d64_bias0);
            d64_acc1 = AE_ADD64S(d64_acc1, d64_bias1);
            d64_acc2 = AE_ADD64S(d64_acc2, d64_bias2);
            d64_acc3 = AE_ADD64S(d64_acc3, d64_bias3);

            d64_acc0 = AE_SLAA64S(d64_acc0, acc_shift+32);
            d64_acc1 = AE_SLAA64S(d64_acc1, acc_shift+32);
            d64_acc2 = AE_SLAA64S(d64_acc2, acc_shift+32);
            d64_acc3 = AE_SLAA64S(d64_acc3, acc_shift+32);

            d_acc0 = AE_ROUND32X2F64SSYM(d64_acc0, d64_acc1);
            d_acc1 = AE_ROUND32X2F64SSYM(d64_acc2, d64_acc3);
            d_acc16x4 = AE_SAT16X4(d_acc0, d_acc1);
#pragma no_unroll
            for(i = 0; i < XT_MIN(out_channels-itr_ch, 4); i++)
            {
                d_acc16x4 = AE_SEL16_6543(d_acc16x4, d_acc16x4);
                *(ae_int16 *)(&out_ptr0[itr_ch+i]) = d_acc16x4;
            }

            if(out_height - itr_oh >= 2)
            {
                d64_acc4 = AE_ADD64S(d64_acc4, d64_bias0);
                d64_acc5 = AE_ADD64S(d64_acc5, d64_bias1);
                d64_acc6 = AE_ADD64S(d64_acc6, d64_bias2);
                d64_acc7 = AE_ADD64S(d64_acc7, d64_bias3);

                d64_acc4 = AE_SLAA64S(d64_acc4, acc_shift+32);
                d64_acc5 = AE_SLAA64S(d64_acc5, acc_shift+32);
                d64_acc6 = AE_SLAA64S(d64_acc6, acc_shift+32);
                d64_acc7 = AE_SLAA64S(d64_acc7, acc_shift+32);

                d_acc2 = AE_ROUND32X2F64SSYM(d64_acc4, d64_acc5);
                d_acc3 = AE_ROUND32X2F64SSYM(d64_acc6, d64_acc7);
                d_acc16x4 = AE_SAT16X4(d_acc2, d_acc3);
#pragma no_unroll
                for(i = 0; i < XT_MIN(out_channels-itr_ch, 4); i++)
                {
                    d_acc16x4 = AE_SEL16_6543(d_acc16x4, d_acc16x4);
                    *(ae_int16 *)(&out_ptr1[itr_ch+i]) = d_acc16x4;
                }
            }
        }
    }
}

static void xa_nn_conv2d_depthwise_nhwc_16x16
(pWORD16 __restrict__ p_out
 ,const WORD16 *__restrict__ p_kernel
 ,const WORD16 *__restrict__ p_inp
 ,const WORD16 *__restrict__ p_bias
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
 ,WORD32  acc_shift
 ,WORD32  bias_shift
 ,pVOID p_scratch
)
{
    WORD16 pad_val = 0;
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
         ,(pVOID)(&pad_val)
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

        conv2d_nhwc_16x16
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
             ,acc_shift
             ,bias_shift
            );
    }
}

WORD32 xa_nn_conv2d_depthwise_16x16
(pWORD16 __restrict__ p_out
 ,const WORD16 *__restrict__ p_kernel
 ,const WORD16 *__restrict__ p_inp
 ,const WORD16 *__restrict__ p_bias
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
 ,WORD32  acc_shift
 ,WORD32  bias_shift
 ,WORD32  inp_data_format
 ,WORD32  out_data_format
 ,pVOID p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
    XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

    if(inp_data_format == 0)
    {
        xa_nn_conv2d_depthwise_nhwc_16x16
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
             ,acc_shift
             ,bias_shift
             ,p_scratch);
    }
    else if(inp_data_format == 1)
    {
        xa_nn_conv2d_depthwise_nchw_16x16
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
             ,acc_shift
             ,bias_shift
             ,p_scratch);
    }
    return 0;
}
