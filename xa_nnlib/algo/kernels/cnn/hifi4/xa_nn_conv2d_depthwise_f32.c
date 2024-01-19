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
#include "xa_nnlib_common_fpu.h"
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"

#include "xa_nnlib_common.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_conv2d_depthwise_f32,(
            FLOAT32* __restrict__ p_out,
            const FLOAT32* __restrict__ p_kernel,
            const FLOAT32* __restrict__ p_inp,
            const FLOAT32* __restrict__ p_bias,
            WORD32  input_height,
            WORD32  input_width,
            WORD32  input_channels,
            WORD32  kernel_height,
            WORD32  kernel_width,
            WORD32  channels_multiplier,
            WORD32  x_stride,
            WORD32  y_stride,
            WORD32  x_padding,
            WORD32  y_padding,
            WORD32  out_height,
            WORD32  out_width,
            WORD32  inp_data_format,
            WORD32  out_data_format,
            pVOID p_scratch))

DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_dilated_conv2d_depthwise_f32,(
            FLOAT32* __restrict__ p_out,
            const FLOAT32* __restrict__ p_kernel,
            const FLOAT32* __restrict__ p_inp,
            const FLOAT32* __restrict__ p_bias,
            WORD32  input_height,
            WORD32  input_width,
            WORD32  input_channels,
            WORD32  kernel_height,
            WORD32  kernel_width,
            WORD32  channels_multiplier,
            WORD32  dilation_height,
            WORD32  dilation_width,
            WORD32  x_stride,
            WORD32  y_stride,
            WORD32  x_padding,
            WORD32  y_padding,
            WORD32  out_height,
            WORD32  out_width,
            WORD32  inp_data_format,
            WORD32  out_data_format,
            pVOID p_scratch))
#else /* #if !HAVE_VFPU */
static void convolve_nchw_f32(
        FLOAT32*  __restrict__ p_out,
        const FLOAT32* __restrict__ p_ker,
        const FLOAT32* __restrict__ p_inp,
        FLOAT32 bias,
        WORD32  input_width,
        WORD32  kernel_height,
        WORD32  kernel_width,
        WORD32  x_stride,
        WORD32  y_stride,
        WORD32  out_height,
        WORD32  out_width,
        WORD32  out_stride,
        pVOID   p_scratch)
{
    int itr_oh, itr_ow, itr_kh, itr_kw;
    int total_out_width = (input_width - kernel_width) + 1;
    int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
    xtfloatx2 *ptr_inp, *ptr_ker, *ptr_out;

    xtfloatx2 ker0, ker1, ker2, ker3, ker4, ker5;
    xtfloatx2 accu_x2_0, accu_x2_1;
    xtfloatx2 accu_x2_0_a, accu_x2_1_a;
    xtfloatx2 accu_x2_0_b, accu_x2_1_b;
    xtfloatx2 accu_x2_0_c, accu_x2_1_c;
    xtfloatx2 id4, id8, id12, id16, id20, id24, id28, id32;
    xtfloatx2 id5, id6, id7, id9, id10, id11, id13;

    if(kernel_width_pad == 12)
    {
      for(itr_oh=0; itr_oh<out_height; itr_oh++)
      {
          ptr_out = (xtfloatx2 *)p_scratch;
          for(itr_ow=0; itr_ow<((total_out_width+3)>>2); itr_ow++)
          {
              accu_x2_0 = XT_CONST_S(0);
              accu_x2_1 = XT_CONST_S(0);
              accu_x2_0_a = XT_CONST_S(0);
              accu_x2_1_a = XT_CONST_S(0);
              accu_x2_0_b = XT_CONST_S(0);
              accu_x2_1_b = XT_CONST_S(0);
              accu_x2_0_c = XT_CONST_S(0);
              accu_x2_1_c = XT_CONST_S(0);

              ptr_ker = (xtfloatx2 *)p_ker;
              ptr_inp = (xtfloatx2 *)p_inp;
              AE_ADDCIRC16X4_XC((ae_int16x4 *)ptr_inp, ((itr_oh*y_stride)*input_width+4*itr_ow)*sizeof(FLOAT32));
#pragma loop_count min=1
#pragma no_unroll
              for(itr_kh=0; itr_kh<kernel_height; itr_kh++)
              {
                  //Input loads
                  XT_LSX2XC(id4, ptr_inp, 8);
                  XT_LSX2XC(id8, ptr_inp, 8);
                  XT_LSX2XC(id12, ptr_inp, 8);
                  XT_LSX2XC(id16, ptr_inp, 8);
                  XT_LSX2XC(id20, ptr_inp, 8);
                  XT_LSX2XC(id24, ptr_inp, 8);
                  XT_LSX2XC(id28, ptr_inp, 8);
                  XT_LSX2XC(id32, ptr_inp, sizeof(FLOAT32)*(input_width - 14));

                  //Kernel Loads
                  XT_LSX2IP(ker0, ptr_ker, sizeof(xtfloatx2));
                  XT_LSX2IP(ker1, ptr_ker, sizeof(xtfloatx2));
                  XT_LSX2IP(ker2, ptr_ker, sizeof(xtfloatx2));
                  XT_LSX2IP(ker3, ptr_ker, sizeof(xtfloatx2));
                  XT_LSX2IP(ker4, ptr_ker, sizeof(xtfloatx2));
                  XT_LSX2IP(ker5, ptr_ker, sizeof(xtfloatx2));

                  id5 = XT_SEL32_HL_SX2(id8, id4);
                  id6 = XT_SEL32_HL_SX2(id12, id8);
                  id7 = XT_SEL32_HL_SX2(id16, id12);
                  id9 = XT_SEL32_HL_SX2(id20, id16);
                  id10 = XT_SEL32_HL_SX2(id24, id20);
                  id11 = XT_SEL32_HL_SX2(id28, id24);
                  id13 = XT_SEL32_HL_SX2(id32, id28);

                  XT_MADDMUX_S(accu_x2_0, ker0, id4, 0);
                  XT_MADDMUX_S(accu_x2_1, ker0, id8, 0);
                  XT_MADDMUX_S(accu_x2_0_a, ker0, id5, 5);
                  XT_MADDMUX_S(accu_x2_1_a, ker0, id6, 5);
                  XT_MADDMUX_S(accu_x2_0_b, ker1, id8, 0);
                  XT_MADDMUX_S(accu_x2_1_b, ker1, id12, 0);
                  XT_MADDMUX_S(accu_x2_0_c, ker1, id6, 5);
                  XT_MADDMUX_S(accu_x2_1_c, ker1, id7, 5);

                  XT_MADDMUX_S(accu_x2_0, ker2, id12, 0);
                  XT_MADDMUX_S(accu_x2_1, ker2, id16, 0);
                  XT_MADDMUX_S(accu_x2_0_a, ker2, id7, 5);
                  XT_MADDMUX_S(accu_x2_1_a, ker2, id9, 5);
                  XT_MADDMUX_S(accu_x2_0_b, ker3, id16, 0);
                  XT_MADDMUX_S(accu_x2_1_b, ker3, id20, 0);
                  XT_MADDMUX_S(accu_x2_0_c, ker3, id9, 5);
                  XT_MADDMUX_S(accu_x2_1_c, ker3, id10, 5);

                  XT_MADDMUX_S(accu_x2_0, ker4, id20, 0);
                  XT_MADDMUX_S(accu_x2_1, ker4, id24, 0);
                  XT_MADDMUX_S(accu_x2_0_a, ker4, id10, 5);
                  XT_MADDMUX_S(accu_x2_1_a, ker4, id11, 5);
                  XT_MADDMUX_S(accu_x2_0_b, ker5, id24, 0);
                  XT_MADDMUX_S(accu_x2_1_b, ker5, id28, 0);
                  XT_MADDMUX_S(accu_x2_0_c, ker5, id11, 5);
                  XT_MADDMUX_S(accu_x2_1_c, ker5, id13, 5);

              }
              accu_x2_0 += accu_x2_0_a;
              accu_x2_0_b += accu_x2_0_c;
              accu_x2_1 += accu_x2_1_a;
              accu_x2_1_b += accu_x2_1_c;
              accu_x2_0 += accu_x2_0_b;
              accu_x2_1 += accu_x2_1_b;

              *ptr_out++ = accu_x2_0;
              *ptr_out++ = accu_x2_1;
          }

          float *ptr_out1 = (float *)p_scratch;
          for(itr_ow = 0; itr_ow < out_width; itr_ow++)
          {
              p_out[itr_oh*out_width*out_stride+itr_ow*out_stride] = ptr_out1[itr_ow*x_stride] + bias;
          }
      }
    }
    else if(kernel_width_pad == 8)
    {
      for(itr_oh=0; itr_oh<out_height; itr_oh++)
      {
          ptr_out = (xtfloatx2 *)p_scratch;
          for(itr_ow=0; itr_ow<((total_out_width+3)>>2); itr_ow++)
          {
              accu_x2_0 = XT_CONST_S(0);
              accu_x2_1 = XT_CONST_S(0);
              accu_x2_0_a = XT_CONST_S(0);
              accu_x2_1_a = XT_CONST_S(0);
              accu_x2_0_b = XT_CONST_S(0);
              accu_x2_1_b = XT_CONST_S(0);
              accu_x2_0_c = XT_CONST_S(0);
              accu_x2_1_c = XT_CONST_S(0);

              ptr_ker = (xtfloatx2 *)p_ker;
              ptr_inp = (xtfloatx2 *)p_inp;
              AE_ADDCIRC16X4_XC((ae_int16x4 *)ptr_inp, ((itr_oh*y_stride)*input_width+4*itr_ow)*sizeof(FLOAT32));
#pragma loop_count min=1
#pragma no_unroll
              for(itr_kh=0; itr_kh<kernel_height; itr_kh++)
              {
                  //Input loads
                  XT_LSX2XC(id4, ptr_inp, 8);
                  XT_LSX2XC(id8, ptr_inp, 8);
                  XT_LSX2XC(id12, ptr_inp, 8);
                  XT_LSX2XC(id16, ptr_inp, 8);
                  XT_LSX2XC(id20, ptr_inp, 8);
                  XT_LSX2XC(id24, ptr_inp, sizeof(FLOAT32)*(input_width - 10));

                  //Kernel Loads
                  ker1 = XT_LSX2I(ptr_ker, 8);
                  ker2 = XT_LSX2I(ptr_ker, 16);
                  ker3 = XT_LSX2I(ptr_ker, 24);
                  XT_LSX2IP(ker0, ptr_ker, 4*sizeof(xtfloatx2));

                  id5 = XT_SEL32_HL_SX2(id8, id4);
                  id6 = XT_SEL32_HL_SX2(id12, id8);
                  id7 = XT_SEL32_HL_SX2(id16, id12);
                  id9 = XT_SEL32_HL_SX2(id20, id16);
                  id10 = XT_SEL32_HL_SX2(id24, id20);

                  XT_MADDMUX_S(accu_x2_0, ker0, id4, 0);
                  XT_MADDMUX_S(accu_x2_1, ker0, id8, 0);
                  XT_MADDMUX_S(accu_x2_0_a, ker0, id5, 5);
                  XT_MADDMUX_S(accu_x2_1_a, ker0, id6, 5);
                  XT_MADDMUX_S(accu_x2_0_b, ker1, id8, 0);
                  XT_MADDMUX_S(accu_x2_1_b, ker1, id12, 0);
                  XT_MADDMUX_S(accu_x2_0_c, ker1, id6, 5);
                  XT_MADDMUX_S(accu_x2_1_c, ker1, id7, 5);

                  XT_MADDMUX_S(accu_x2_0, ker2, id12, 0);
                  XT_MADDMUX_S(accu_x2_1, ker2, id16, 0);
                  XT_MADDMUX_S(accu_x2_0_a, ker2, id7, 5);
                  XT_MADDMUX_S(accu_x2_1_a, ker2, id9, 5);
                  XT_MADDMUX_S(accu_x2_0_b, ker3, id16, 0);
                  XT_MADDMUX_S(accu_x2_1_b, ker3, id20, 0);
                  XT_MADDMUX_S(accu_x2_0_c, ker3, id9, 5);
                  XT_MADDMUX_S(accu_x2_1_c, ker3, id10, 5);

              }
              accu_x2_0 += accu_x2_0_a;
              accu_x2_0_b += accu_x2_0_c;
              accu_x2_1 += accu_x2_1_a;
              accu_x2_1_b += accu_x2_1_c;
              accu_x2_0 += accu_x2_0_b;
              accu_x2_1 += accu_x2_1_b;

              *ptr_out++ = accu_x2_0;
              *ptr_out++ = accu_x2_1;
          }

          float *ptr_out1 = (float *)p_scratch;
          for(itr_ow = 0; itr_ow < out_width; itr_ow++)
          {
              p_out[itr_oh*out_width*out_stride+itr_ow*out_stride] = ptr_out1[itr_ow*x_stride] + bias;
          }
      }
    }
    else
    {
      /* No reminder loop, run extra iteration, extra output will be thrown away
         when we pick correct outputs using x_stride */
      for(itr_oh=0; itr_oh<out_height; itr_oh++)
      {
          ptr_out = (xtfloatx2 *)p_scratch;
          for(itr_ow=0; itr_ow<((total_out_width+3)>>2); itr_ow++)
          {
              accu_x2_0 = XT_CONST_S(0);
              accu_x2_1 = XT_CONST_S(0);
              accu_x2_0_a = XT_CONST_S(0);
              accu_x2_1_a = XT_CONST_S(0);
              accu_x2_0_b = XT_CONST_S(0);
              accu_x2_1_b = XT_CONST_S(0);
              accu_x2_0_c = XT_CONST_S(0);
              accu_x2_1_c = XT_CONST_S(0);

              ptr_ker = (xtfloatx2 *)p_ker;
#pragma loop_count min=1
              for(itr_kh=0; itr_kh<kernel_height; itr_kh++)
              {
                  ptr_inp = (xtfloatx2 *)p_inp;
                  AE_ADDCIRC16X4_XC((ae_int16x4 *)ptr_inp, ((itr_kh+itr_oh*y_stride)*input_width+4*itr_ow)*sizeof(FLOAT32));
#pragma loop_count min=1
#pragma no_unroll
                  for(itr_kw=0; itr_kw<(kernel_width_pad>>2); itr_kw++)
                  {
                      XT_LSX2XC(id4, ptr_inp, 8);
                      XT_LSX2XC(id8, ptr_inp, 8);
                      id5 = XT_SEL32_HL_SX2(id8, id4);

                      XT_LSX2IP(ker0, ptr_ker, 8);

                      XT_MADDMUX_S(accu_x2_0, ker0, id4, 0);
                      XT_MADDMUX_S(accu_x2_1, ker0, id8, 0);

                      XT_LSX2XC(id12, ptr_inp, 8);
                      id6 = XT_SEL32_HL_SX2(id12, id8);

                      XT_LSX2IP(ker1, ptr_ker, 8);

                      XT_MADDMUX_S(accu_x2_0_a, ker0, id5, 5);
                      XT_MADDMUX_S(accu_x2_1_a, ker0, id6, 5);

                      XT_MADDMUX_S(accu_x2_0_b, ker1, id8, 0);
                      XT_MADDMUX_S(accu_x2_1_b, ker1, id12, 0);

                      XT_LSX2XC(id16, ptr_inp, -8);
                      id7 = XT_SEL32_HL_SX2(id16, id12);

                      XT_MADDMUX_S(accu_x2_0_c, ker1, id6, 5);
                      XT_MADDMUX_S(accu_x2_1_c, ker1, id7, 5);
                  }
              }
              accu_x2_0 += accu_x2_0_a;
              accu_x2_0_b += accu_x2_0_c;
              accu_x2_1 += accu_x2_1_a;
              accu_x2_1_b += accu_x2_1_c;
              accu_x2_0 += accu_x2_0_b;
              accu_x2_1 += accu_x2_1_b;

              *ptr_out++ = accu_x2_0;
              *ptr_out++ = accu_x2_1;
          }

          float *ptr_out1 = (float *)p_scratch;
          for(itr_ow = 0; itr_ow < out_width; itr_ow++)
          {
              p_out[itr_oh*out_width*out_stride+itr_ow*out_stride] = ptr_out1[itr_ow*x_stride] + bias;
          }
      }
    }
}

static void xa_nn_conv2d_depthwise_nchw_f32(
        FLOAT32* __restrict__ p_out,
        const FLOAT32* __restrict__ p_kernel,
        const FLOAT32* __restrict__ p_inp,
        const FLOAT32* __restrict__ p_bias,
        WORD32  input_height,
        WORD32  input_width,
        WORD32  input_channels,
        WORD32  kernel_height,
        WORD32  kernel_width,
        WORD32  channels_multiplier,
        WORD32  x_stride,
        WORD32  y_stride,
        WORD32  x_padding,
        WORD32  y_padding,
        WORD32  out_height,
        WORD32  out_width,
        pVOID p_scratch)
{
    FLOAT32 pad_val = 0.0f;
    xa_nn_conv2d_depthwise_init(
            p_scratch,
            input_height,
            input_width,
            input_channels,
            kernel_height,
            kernel_width,
            channels_multiplier,
            x_stride,
            y_stride,
            x_padding,
            y_padding,
            out_height,
            out_width,
            -1,
            1,
            (pVOID)(&pad_val));

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ic, itr_cm, itr_oh, i;
    int circ_out_height = (p_circ_buf->rows - kernel_height)/y_stride + 1;
    int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
    int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
    int rows_to_add, top_pad, bottom_pad, rows_added;
    int input_row;
    FLOAT32 *pt_inp;
    const FLOAT32 *pt_ker;
    FLOAT32 *p_inp_circ;

    FLOAT32 *p_kernel_padded = (FLOAT32 *)(p_state->p_scratch);
    p_kernel_padded = (FLOAT32 *)ALIGN_PTR(p_kernel_padded, 8);
    FLOAT32 *p_tmp_out = (FLOAT32 *)(p_kernel_padded + kernel_height_pad * kernel_width_pad);
    p_tmp_out = (FLOAT32 *)ALIGN_PTR(p_tmp_out, 8);

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    /* Initialize whole scratch for padded kernel to padding value, after this
       we only have to copy actual kernel values, padding area should remain
       untouched */
    xtfloatx2 *pae_ker_pad = (xtfloatx2 *)p_kernel_padded;
    for(i = 0; i < ((kernel_height_pad * kernel_width_pad) >> 1); i++)
    {
      pae_ker_pad[i] = XT_CONST_S(0);
    }

    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = (FLOAT32 *)&p_inp[itr_ic*input_height*input_width];

        CIRC_BUF_ADD_ROWS_INIT(
                rows_added,
                rows_to_add,
                top_pad,
                bottom_pad,
                input_row,
                input_height,
                input_width,
                kernel_height,
                y_stride,
                x_padding,
                y_padding,
                p_circ_buf,
                pt_inp);
        for(itr_oh = 0; itr_oh < out_height - (circ_out_height - 1); itr_oh += circ_out_height)
        {
            CIRC_BUF_ADD_ROWS(
                    rows_added,
                    rows_to_add,
                    top_pad,
                    bottom_pad,
                    input_row,
                    input_height,
                    input_width,
                    circ_out_height,
                    y_stride,
                    x_padding,
                    y_padding,
                    p_circ_buf,
                    pt_inp);
            p_inp_circ = (FLOAT32 *)p_circ_buf->p_curr;
            for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
            {
                pt_ker = &p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
                COPY_KERNEL_TO_SCRATCH_F32(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);

                convolve_nchw_f32(
                        &p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)],
                        p_kernel_padded,
                        p_inp_circ,
                        p_bias[itr_ic*channels_multiplier+itr_cm],
                        p_circ_buf->row_offset,
                        kernel_height,
                        kernel_width,
                        x_stride,
                        y_stride,
                        circ_out_height,
                        out_width,
                        input_channels*channels_multiplier,
                        p_tmp_out);
            }
        }
        CIRC_BUF_ADD_ROWS(
                rows_added,
                rows_to_add,
                top_pad,
                bottom_pad,
                input_row,
                input_height,
                input_width,
                circ_out_height,
                y_stride,
                x_padding,
                y_padding,
                p_circ_buf,
                pt_inp);
        p_inp_circ = (FLOAT32 *)p_circ_buf->p_curr;
        for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
        {
            pt_ker = &p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
            COPY_KERNEL_TO_SCRATCH_F32(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);

            convolve_nchw_f32(
                    &p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)],
                    p_kernel_padded,
                    p_inp_circ,
                    p_bias[itr_ic*channels_multiplier+itr_cm],
                    p_circ_buf->row_offset,
                    kernel_height,
                    kernel_width,
                    x_stride,
                    y_stride,
                    (out_height-itr_oh),
                    out_width,
                    input_channels*channels_multiplier,
                    p_tmp_out);
        }
    }
}

static void dilated_convolve_nchw_f32(
    FLOAT32*  __restrict__ p_out,
    const FLOAT32* __restrict__ p_ker,
    const FLOAT32* __restrict__ p_inp,
    const FLOAT32* __restrict__ p_bias,
    WORD32   input_width,
    WORD32   kernel_height,
    WORD32   kernel_width,
    WORD32   dilation_height,
    WORD32   dilation_width,
    WORD32   x_stride,
    WORD32   y_stride,
    WORD32   out_height,
    WORD32   out_width,
    WORD32   out_stride,
    pVOID    p_scratch)
{
    int itr_oh, itr_ow, itr_kh, itr_kw;
    int total_out_width = (input_width - ((kernel_width - 1) * dilation_width + 1)) + 1;
    int kernel_width_pad = ALIGNED_SIZE((kernel_width - 1) * dilation_width + 1, 4);
    xtfloatx2 *ptr_inp, *ptr_ker, *ptr_out;

    xtfloatx2 ker0, ker1, ker2, ker3, ker4, ker5;
    xtfloatx2 accu_x2_0, accu_x2_1;
    xtfloatx2 accu_x2_0_a, accu_x2_1_a;
    xtfloatx2 accu_x2_0_b, accu_x2_1_b;
    xtfloatx2 accu_x2_0_c, accu_x2_1_c;
    xtfloatx2 id4, id8, id12, id16, id20, id24, id28, id32;
    xtfloatx2 id5, id6, id7, id9, id10, id11, id13;

    if(kernel_width_pad == 12)
    {
      for(itr_oh=0; itr_oh<out_height; itr_oh++)
      {
          ptr_out = (xtfloatx2 *)p_scratch;
          for(itr_ow=0; itr_ow<((total_out_width+3)>>2); itr_ow++)
          {
              accu_x2_0 = XT_CONST_S(0);
              accu_x2_1 = XT_CONST_S(0);
              accu_x2_0_a = XT_CONST_S(0);
              accu_x2_1_a = XT_CONST_S(0);
              accu_x2_0_b = XT_CONST_S(0);
              accu_x2_1_b = XT_CONST_S(0);
              accu_x2_0_c = XT_CONST_S(0);
              accu_x2_1_c = XT_CONST_S(0);

              ptr_ker = (xtfloatx2 *)p_ker;
              ptr_inp = (xtfloatx2 *)p_inp;
              AE_ADDCIRC16X4_XC((ae_int16x4 *)ptr_inp, ((itr_oh*y_stride)*input_width+4*itr_ow)*sizeof(FLOAT32));
#pragma loop_count min=1
#pragma no_unroll
              for(itr_kh=0; itr_kh<kernel_height; itr_kh++)
              {
                  //Input loads
                  XT_LSX2XC(id4, ptr_inp, 8);
                  XT_LSX2XC(id8, ptr_inp, 8);
                  XT_LSX2XC(id12, ptr_inp, 8);
                  XT_LSX2XC(id16, ptr_inp, 8);
                  XT_LSX2XC(id20, ptr_inp, 8);
                  XT_LSX2XC(id24, ptr_inp, 8);
                  XT_LSX2XC(id28, ptr_inp, 8);
                  XT_LSX2XC(id32, ptr_inp, sizeof(FLOAT32)*(dilation_height * input_width - 14));

                  //Kernel Loads
                  XT_LSX2IP(ker0, ptr_ker, sizeof(xtfloatx2));
                  XT_LSX2IP(ker1, ptr_ker, sizeof(xtfloatx2));
                  XT_LSX2IP(ker2, ptr_ker, sizeof(xtfloatx2));
                  XT_LSX2IP(ker3, ptr_ker, sizeof(xtfloatx2));
                  XT_LSX2IP(ker4, ptr_ker, sizeof(xtfloatx2));
                  XT_LSX2IP(ker5, ptr_ker, sizeof(xtfloatx2));

                  id5 = XT_SEL32_HL_SX2(id8, id4);
                  id6 = XT_SEL32_HL_SX2(id12, id8);
                  id7 = XT_SEL32_HL_SX2(id16, id12);
                  id9 = XT_SEL32_HL_SX2(id20, id16);
                  id10 = XT_SEL32_HL_SX2(id24, id20);
                  id11 = XT_SEL32_HL_SX2(id28, id24);
                  id13 = XT_SEL32_HL_SX2(id32, id28);

                  XT_MADDMUX_S(accu_x2_0, ker0, id4, 0);
                  XT_MADDMUX_S(accu_x2_1, ker0, id8, 0);
                  XT_MADDMUX_S(accu_x2_0_a, ker0, id5, 5);
                  XT_MADDMUX_S(accu_x2_1_a, ker0, id6, 5);
                  XT_MADDMUX_S(accu_x2_0_b, ker1, id8, 0);
                  XT_MADDMUX_S(accu_x2_1_b, ker1, id12, 0);
                  XT_MADDMUX_S(accu_x2_0_c, ker1, id6, 5);
                  XT_MADDMUX_S(accu_x2_1_c, ker1, id7, 5);

                  XT_MADDMUX_S(accu_x2_0, ker2, id12, 0);
                  XT_MADDMUX_S(accu_x2_1, ker2, id16, 0);
                  XT_MADDMUX_S(accu_x2_0_a, ker2, id7, 5);
                  XT_MADDMUX_S(accu_x2_1_a, ker2, id9, 5);
                  XT_MADDMUX_S(accu_x2_0_b, ker3, id16, 0);
                  XT_MADDMUX_S(accu_x2_1_b, ker3, id20, 0);
                  XT_MADDMUX_S(accu_x2_0_c, ker3, id9, 5);
                  XT_MADDMUX_S(accu_x2_1_c, ker3, id10, 5);

                  XT_MADDMUX_S(accu_x2_0, ker4, id20, 0);
                  XT_MADDMUX_S(accu_x2_1, ker4, id24, 0);
                  XT_MADDMUX_S(accu_x2_0_a, ker4, id10, 5);
                  XT_MADDMUX_S(accu_x2_1_a, ker4, id11, 5);
                  XT_MADDMUX_S(accu_x2_0_b, ker5, id24, 0);
                  XT_MADDMUX_S(accu_x2_1_b, ker5, id28, 0);
                  XT_MADDMUX_S(accu_x2_0_c, ker5, id11, 5);
                  XT_MADDMUX_S(accu_x2_1_c, ker5, id13, 5);

              }
              accu_x2_0 += accu_x2_0_a;
              accu_x2_0_b += accu_x2_0_c;
              accu_x2_1 += accu_x2_1_a;
              accu_x2_1_b += accu_x2_1_c;
              accu_x2_0 += accu_x2_0_b;
              accu_x2_1 += accu_x2_1_b;

              *ptr_out++ = accu_x2_0;
              *ptr_out++ = accu_x2_1;
          }

          float *ptr_out1 = (float *)p_scratch;
          for(itr_ow = 0; itr_ow < out_width; itr_ow++)
          {
              p_out[itr_oh*out_width*out_stride+itr_ow*out_stride] = ptr_out1[itr_ow*x_stride] + p_bias[0];
          }
      }
    }
    else if(kernel_width_pad == 8)
    {
      for(itr_oh=0; itr_oh<out_height; itr_oh++)
      {
          ptr_out = (xtfloatx2 *)p_scratch;
          for(itr_ow=0; itr_ow<((total_out_width+3)>>2); itr_ow++)
          {
              accu_x2_0 = XT_CONST_S(0);
              accu_x2_1 = XT_CONST_S(0);
              accu_x2_0_a = XT_CONST_S(0);
              accu_x2_1_a = XT_CONST_S(0);
              accu_x2_0_b = XT_CONST_S(0);
              accu_x2_1_b = XT_CONST_S(0);
              accu_x2_0_c = XT_CONST_S(0);
              accu_x2_1_c = XT_CONST_S(0);

              ptr_ker = (xtfloatx2 *)p_ker;
              ptr_inp = (xtfloatx2 *)p_inp;
              AE_ADDCIRC16X4_XC((ae_int16x4 *)ptr_inp, ((itr_oh*y_stride)*input_width+4*itr_ow)*sizeof(FLOAT32));
#pragma loop_count min=1
#pragma no_unroll
              for(itr_kh=0; itr_kh<kernel_height; itr_kh++)
              {
                  //Input loads
                  XT_LSX2XC(id4, ptr_inp, 8);
                  XT_LSX2XC(id8, ptr_inp, 8);
                  XT_LSX2XC(id12, ptr_inp, 8);
                  XT_LSX2XC(id16, ptr_inp, 8);
                  XT_LSX2XC(id20, ptr_inp, 8);
                  XT_LSX2XC(id24, ptr_inp, sizeof(FLOAT32)*(dilation_height * input_width - 10));

                  //Kernel Loads
                  ker1 = XT_LSX2I(ptr_ker, 8);
                  ker2 = XT_LSX2I(ptr_ker, 16);
                  ker3 = XT_LSX2I(ptr_ker, 24);
                  XT_LSX2IP(ker0, ptr_ker, 4*sizeof(xtfloatx2));

                  id5 = XT_SEL32_HL_SX2(id8, id4);
                  id6 = XT_SEL32_HL_SX2(id12, id8);
                  id7 = XT_SEL32_HL_SX2(id16, id12);
                  id9 = XT_SEL32_HL_SX2(id20, id16);
                  id10 = XT_SEL32_HL_SX2(id24, id20);

                  XT_MADDMUX_S(accu_x2_0, ker0, id4, 0);
                  XT_MADDMUX_S(accu_x2_1, ker0, id8, 0);
                  XT_MADDMUX_S(accu_x2_0_a, ker0, id5, 5);
                  XT_MADDMUX_S(accu_x2_1_a, ker0, id6, 5);
                  XT_MADDMUX_S(accu_x2_0_b, ker1, id8, 0);
                  XT_MADDMUX_S(accu_x2_1_b, ker1, id12, 0);
                  XT_MADDMUX_S(accu_x2_0_c, ker1, id6, 5);
                  XT_MADDMUX_S(accu_x2_1_c, ker1, id7, 5);

                  XT_MADDMUX_S(accu_x2_0, ker2, id12, 0);
                  XT_MADDMUX_S(accu_x2_1, ker2, id16, 0);
                  XT_MADDMUX_S(accu_x2_0_a, ker2, id7, 5);
                  XT_MADDMUX_S(accu_x2_1_a, ker2, id9, 5);
                  XT_MADDMUX_S(accu_x2_0_b, ker3, id16, 0);
                  XT_MADDMUX_S(accu_x2_1_b, ker3, id20, 0);
                  XT_MADDMUX_S(accu_x2_0_c, ker3, id9, 5);
                  XT_MADDMUX_S(accu_x2_1_c, ker3, id10, 5);

              }
              accu_x2_0 += accu_x2_0_a;
              accu_x2_0_b += accu_x2_0_c;
              accu_x2_1 += accu_x2_1_a;
              accu_x2_1_b += accu_x2_1_c;
              accu_x2_0 += accu_x2_0_b;
              accu_x2_1 += accu_x2_1_b;

              *ptr_out++ = accu_x2_0;
              *ptr_out++ = accu_x2_1;
          }

          float *ptr_out1 = (float *)p_scratch;
          for(itr_ow = 0; itr_ow < out_width; itr_ow++)
          {
              p_out[itr_oh*out_width*out_stride+itr_ow*out_stride] = ptr_out1[itr_ow*x_stride] + p_bias[0];
          }
      }
    }
    else
    {
      /* No reminder loop, run extra iteration, extra output will be thrown away
         when we pick correct outputs using x_stride */
      for(itr_oh=0; itr_oh<out_height; itr_oh++)
      {
          ptr_out = (xtfloatx2 *)p_scratch;
          for(itr_ow=0; itr_ow<((total_out_width+3)>>2); itr_ow++)
          {
              accu_x2_0 = XT_CONST_S(0);
              accu_x2_1 = XT_CONST_S(0);
              accu_x2_0_a = XT_CONST_S(0);
              accu_x2_1_a = XT_CONST_S(0);
              accu_x2_0_b = XT_CONST_S(0);
              accu_x2_1_b = XT_CONST_S(0);
              accu_x2_0_c = XT_CONST_S(0);
              accu_x2_1_c = XT_CONST_S(0);

              ptr_ker = (xtfloatx2 *)p_ker;
#pragma loop_count min=1
              for(itr_kh=0; itr_kh<kernel_height; itr_kh++)
              {
                  ptr_inp = (xtfloatx2 *)p_inp;
                  AE_ADDCIRC16X4_XC((ae_int16x4 *)ptr_inp, ((itr_kh*dilation_height+itr_oh*y_stride)*input_width+4*itr_ow)*sizeof(FLOAT32));
#pragma loop_count min=1
#pragma no_unroll
                  for(itr_kw=0; itr_kw<(kernel_width_pad>>2); itr_kw++)
                  {
                      XT_LSX2XC(id4, ptr_inp, 8);
                      XT_LSX2XC(id8, ptr_inp, 8);
                      id5 = XT_SEL32_HL_SX2(id8, id4);

                      XT_LSX2IP(ker0, ptr_ker, 8);

                      XT_MADDMUX_S(accu_x2_0, ker0, id4, 0);
                      XT_MADDMUX_S(accu_x2_1, ker0, id8, 0);

                      XT_LSX2XC(id12, ptr_inp, 8);
                      id6 = XT_SEL32_HL_SX2(id12, id8);

                      XT_LSX2IP(ker1, ptr_ker, 8);

                      XT_MADDMUX_S(accu_x2_0_a, ker0, id5, 5);
                      XT_MADDMUX_S(accu_x2_1_a, ker0, id6, 5);

                      XT_MADDMUX_S(accu_x2_0_b, ker1, id8, 0);
                      XT_MADDMUX_S(accu_x2_1_b, ker1, id12, 0);

                      XT_LSX2XC(id16, ptr_inp, -8);
                      id7 = XT_SEL32_HL_SX2(id16, id12);

                      XT_MADDMUX_S(accu_x2_0_c, ker1, id6, 5);
                      XT_MADDMUX_S(accu_x2_1_c, ker1, id7, 5);
                  }
              }
              accu_x2_0 += accu_x2_0_a;
              accu_x2_0_b += accu_x2_0_c;
              accu_x2_1 += accu_x2_1_a;
              accu_x2_1_b += accu_x2_1_c;
              accu_x2_0 += accu_x2_0_b;
              accu_x2_1 += accu_x2_1_b;

              *ptr_out++ = accu_x2_0;
              *ptr_out++ = accu_x2_1;
          }

          float *ptr_out1 = (float *)p_scratch;
          for(itr_ow = 0; itr_ow < out_width; itr_ow++)
          {
              p_out[itr_oh*out_width*out_stride+itr_ow*out_stride] = ptr_out1[itr_ow*x_stride] + p_bias[0];
          }
      }
    }

}

#define COPY_DILATED_KERNEL_TO_SCRATCH(p_out, p_in, kh, kw, kw_pad, d_w) \
{ \
  int itr_kh, itr_kw; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    xtfloat *pae_in = (xtfloat *)(&p_in[itr_kh * kw]); \
    xtfloat *pae_out = (xtfloat *)(&p_out[itr_kh * kw_pad]); \
_Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < kw; itr_kw++) \
    { \
      pae_out[itr_kw * d_w] = pae_in[itr_kw]; \
    } \
  } \
}

static void xa_nn_dilated_conv2d_depthwise_nchw_f32(
    FLOAT32* __restrict__ p_out,
    const FLOAT32* __restrict__ p_kernel,
    const FLOAT32* __restrict__ p_inp,
    const FLOAT32* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  channels_multiplier,
    WORD32  dilation_height,
    WORD32  dilation_width,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
    pVOID p_scratch)
{
    FLOAT32 pad_val = 0.0f;
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
            -1,
            1,
            (pVOID)(&pad_val));

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ic, itr_cm, itr_oh;
    int circ_out_height = (p_circ_buf->rows - ((kernel_height - 1) * dilation_height + 1))/y_stride + 1;
    int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
    int kernel_width_pad = ALIGNED_SIZE((kernel_width - 1) * dilation_width + 1, 4);
    int rows_to_add, top_pad, bottom_pad, rows_added;
    int input_row;
    const FLOAT32 *pt_inp, *pt_ker;
    FLOAT32 *p_inp_circ;
    int i;
    FLOAT32 *p_kernel_padded = (FLOAT32 *)(p_state->p_scratch);
    p_kernel_padded = (FLOAT32 *)ALIGN_PTR(p_kernel_padded, 16);
    FLOAT32 *p_tmp_out = (FLOAT32 *)(p_kernel_padded + kernel_height_pad * kernel_width_pad);
    p_tmp_out = (FLOAT32 *)ALIGN_PTR(p_tmp_out, 16);

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    /* Initialize whole scratch for padded kernel to padding value, after this
     we only have to copy actual kernel values, padding area should remain
     untouched */
    xtfloatx2 *pae_ker_pad = (xtfloatx2 *)p_kernel_padded;
    for(i = 0; i < ((kernel_height_pad * kernel_width_pad) >> 1); i++)
    {
        pae_ker_pad[i] = XT_CONST_S(0);
    }

    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = &p_inp[itr_ic*input_height*input_width];
        for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
        {
            pt_ker = &p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
            COPY_DILATED_KERNEL_TO_SCRATCH(p_kernel_padded, pt_ker, kernel_height, kernel_width,
                                           kernel_width_pad, dilation_width);

            CIRC_BUF_ADD_ROWS_INIT(rows_added
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
                                   )
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
                              )
                p_inp_circ = (FLOAT32 *)p_circ_buf->p_curr;
                dilated_convolve_nchw_f32(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)]
                            ,p_kernel_padded
                            ,p_inp_circ
                            ,&p_bias[itr_ic*channels_multiplier+itr_cm]
                            ,p_circ_buf->row_offset
                            ,kernel_height
                            ,kernel_width
                            ,dilation_height
                            ,dilation_width
                            ,x_stride
                            ,y_stride
                            ,circ_out_height
                            ,out_width
                            ,input_channels*channels_multiplier
                            ,p_tmp_out
                            );
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
                              )
            p_inp_circ = (FLOAT32 *)p_circ_buf->p_curr;
            dilated_convolve_nchw_f32(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)]
                        ,p_kernel_padded
                        ,p_inp_circ
                        ,&p_bias[itr_ic*channels_multiplier+itr_cm]
                        ,p_circ_buf->row_offset
                        ,kernel_height
                        ,kernel_width
                        ,dilation_height
                        ,dilation_width
                        ,x_stride
                        ,y_stride
                        ,(out_height-itr_oh)
                        ,out_width
                        ,input_channels*channels_multiplier
                        ,p_tmp_out
                        );
        }
    }
}


/* 2D Convolution implementation */
static inline void conv2d_nhwc_f32(
        FLOAT32 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_ker,
        const FLOAT32 *__restrict__ p_inp,
        const FLOAT32 *p_bias,
        int kernel_height,
        int kernel_width,
        int out_height,
        int out_width,
        int out_channels,
        int y_stride)
{
    WORD32 out_channels_pad;
    WORD32 i, itr_oh, itr_ch, itr_kw;
    xtfloatx2 *pt_inp0, *pt_inp1, *pt_ker;
    FLOAT32 *out_ptr0, *out_ptr1;
    xtfloatx2 d_inp0, d_inp1, d_ker0;
    xtfloatx2 d_inp2, d_inp3, d_ker1;
    const xtfloatx2 *pt_bias;
    ae_valign bias_a;
    ae_valign ker_a;
    xtfloatx2 d_acc0, d_acc1, d_bias0;
    xtfloatx2 d_acc2, d_acc3, d_bias1;

    out_channels_pad = (out_channels + 1)&(~1);

    for(itr_oh = 0; itr_oh < out_height; itr_oh+=2)
    {
        out_ptr0 = (FLOAT32 *)(&p_out[itr_oh*out_channels*out_width]);
        out_ptr1 = (FLOAT32 *)(&p_out[(itr_oh+1)*out_channels*out_width]);
        pt_bias = (const xtfloatx2 *)p_bias;
        bias_a = XT_LASX2PP(pt_bias);
        for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
        {
            pt_inp0 = (xtfloatx2 *)p_inp;
            pt_inp1 = (xtfloatx2 *)p_inp;
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, (itr_ch + itr_oh*y_stride*kernel_width*out_channels_pad)*sizeof(FLOAT32));
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, (itr_ch + (itr_oh+1)*y_stride*kernel_width*out_channels_pad)*sizeof(FLOAT32));
            pt_ker = (xtfloatx2 *)(&p_ker[itr_ch]);
            d_acc0 = XT_CONST_S(0);
            d_acc1 = XT_CONST_S(0);
            d_acc2 = XT_CONST_S(0);
            d_acc3 = XT_CONST_S(0);
            ker_a = XT_LASX2PP(pt_ker);
#pragma loop_count min=1
            for(itr_kw = 0; itr_kw < kernel_height * kernel_width; itr_kw++)
            {
                XT_LSX2XC(d_inp0, pt_inp0, 8);
                XT_LSX2XC(d_inp1, pt_inp1, 8);
                XT_LASX2IP(d_ker0, ker_a, pt_ker);
                XT_LSX2XC(d_inp2, pt_inp0, (out_channels_pad-2)*sizeof(FLOAT32));
                XT_LSX2XC(d_inp3, pt_inp1, (out_channels_pad-2)*sizeof(FLOAT32));
                XT_LASX2IP(d_ker1, ker_a, pt_ker);
                pt_ker = (xtfloatx2 *)((FLOAT32 *)pt_ker + (out_channels - 4));
                ker_a = XT_LASX2PP(pt_ker);
                XT_MADD_SX2(d_acc0, d_inp0, d_ker0);
                XT_MADD_SX2(d_acc1, d_inp1, d_ker0);
                XT_MADD_SX2(d_acc2, d_inp2, d_ker1);
                XT_MADD_SX2(d_acc3, d_inp3, d_ker1);
            }
            XT_LASX2IP(d_bias0, bias_a, pt_bias);
            XT_LASX2IP(d_bias1, bias_a, pt_bias);
            d_acc0 = XT_ADD_SX2(d_acc0, d_bias0);
            d_acc2 = XT_ADD_SX2(d_acc2, d_bias1);

#pragma no_unroll
            for(i = 0; i < XT_MIN(out_channels-itr_ch, 4); i++)
            {
                out_ptr0[itr_ch+i] = XT_HIGH_S(d_acc0);
                d_acc0 = XT_SEL32_LH_SX2(d_acc0, d_acc2);
                d_acc2 = XT_SEL32_LH_SX2(d_acc2, d_acc2);
            }
            if(out_height - itr_oh >= 2)
            {
                d_acc1 = XT_ADD_SX2(d_acc1, d_bias0);
                d_acc3 = XT_ADD_SX2(d_acc3, d_bias1);

#pragma no_unroll
                for(i = 0; i < XT_MIN(out_channels-itr_ch, 4); i++)
                {
                    out_ptr1[itr_ch+i] = XT_HIGH_S(d_acc1);
                    d_acc1 = XT_SEL32_LH_SX2(d_acc1, d_acc3);
                    d_acc3 = XT_SEL32_LH_SX2(d_acc3, d_acc3);
                }
            }
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

static void xa_nn_conv2d_depthwise_nhwc_f32(
        FLOAT32 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_kernel,
        const FLOAT32 *__restrict__ p_inp,
        const FLOAT32 *__restrict__ p_bias,
        WORD32  input_height,
        WORD32  input_width,
        WORD32  input_channels,
        WORD32  kernel_height,
        WORD32  kernel_width,
        WORD32  channels_multiplier,
        WORD32  x_stride,
        WORD32  y_stride,
        WORD32  x_padding,
        WORD32  y_padding,
        WORD32  out_height,
        WORD32  out_width,
        pVOID p_scratch)
{
    FLOAT32 pad_val = 0.0f;
    xa_nn_conv2d_depthwise_init(
            p_scratch,
            input_height,
            input_width,
            input_channels,
            kernel_height,
            kernel_width,
            channels_multiplier,
            x_stride,
            y_stride,
            x_padding,
            y_padding,
            out_height,
            out_width,
            -1,
            0,
            (pVOID)(&pad_val));

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ow;
    int cols_to_add, left_pad, right_pad, cols_added;
    int input_col;
    const FLOAT32 *pt_inp;
    FLOAT32 *p_inp_circ;

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    pt_inp = (const FLOAT32 *)p_inp;

    CIRC_BUF_ADD_COLS_INIT(
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
            x_stride,
            x_padding,
            y_padding,
            out_height,
            p_circ_buf,
            pt_inp);

    for(itr_ow = 0; itr_ow < out_width; itr_ow++)
    {
        CIRC_BUF_ADD_COLS(
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
                x_stride,
                x_padding,
                y_padding,
                out_height,
                p_circ_buf,
                pt_inp);

        p_inp_circ = (FLOAT32 *)p_circ_buf->p_curr;

        conv2d_nhwc_f32(
                (FLOAT32 *)(&p_out[itr_ow*input_channels*channels_multiplier]),
                p_kernel,
                p_inp_circ,
                p_bias,
                kernel_height,
                kernel_width,
                out_height,
                out_width,
                (input_channels * channels_multiplier),
                y_stride);
    }
}

static void xa_nn_dilated_conv2d_depthwise_nhwc_f32(
        FLOAT32 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_kernel,
        const FLOAT32 *__restrict__ p_inp,
        const FLOAT32 *__restrict__ p_bias,
        WORD32  input_height,
        WORD32  input_width,
        WORD32  input_channels,
        WORD32  kernel_height,
        WORD32  kernel_width,
        WORD32  channels_multiplier,
        WORD32  dilation_height,
        WORD32  dilation_width,
        WORD32  x_stride,
        WORD32  y_stride,
        WORD32  x_padding,
        WORD32  y_padding,
        WORD32  out_height,
        WORD32  out_width,
        pVOID p_scratch)
{
    int itr_ow;
    int itr_dh, itr_dw;
    int cols_to_add, left_pad, right_pad, cols_added;
    int input_col;
    const FLOAT32 *pt_inp;
    FLOAT32 *p_inp_circ;

    pt_inp = (const FLOAT32 *)p_inp;

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

    FLOAT32 pad_val = 0.0f;
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
            -1,
            0
            ,(pVOID)(&pad_val));
    xa_nn_circ_buf_t *p_state = (xa_nn_circ_buf_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = p_state;

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    for(itr_dh = 0; itr_dh < dh_count; itr_dh++, rem_dh--)
    {
        x_padding_dw = x_padding;
        x_stride_dw = x_stride * dw_count;
        rem_dw = out_width - out_width_dw * dw_count;

        WORD32 out_height_dh_cur = out_height_dh + (rem_dh > 0 ? 1 : 0);
        for(itr_dw = 0; itr_dw < dw_count; itr_dw++, rem_dw--)
        {
            WORD32 out_width_dw_cur = out_width_dw + (rem_dw > 0 ? 1 : 0);
            DILATED_CIRC_BUF_ADD_COLS_INIT(
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
                    pt_inp);

            for(itr_ow = 0; itr_ow < out_width_dw_cur; itr_ow++)
            {
                FLOAT32 *pt_out = (FLOAT32 *)&p_out[(itr_dh * out_width + itr_dw + itr_ow * dw_count)*input_channels * channels_multiplier];
                DILATED_CIRC_BUF_ADD_COLS(
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
                        pt_inp);

                p_inp_circ = (FLOAT32 *)p_circ_buf->p_curr;

                conv2d_nhwc_f32(
                        pt_out,
                        p_kernel,
                        p_inp_circ,
                        p_bias,
                        kernel_height,
                        kernel_width,
                        out_height_dh_cur,
                        out_width * dh_count,
                        (input_channels * channels_multiplier),
                        y_stride_circ_buf);
            }
            x_padding_dw -= x_stride;
        }
        y_padding_dh -= y_stride;
    }
}

WORD32 xa_nn_conv2d_depthwise_f32(
        FLOAT32* __restrict__ p_out,
        const FLOAT32* __restrict__ p_kernel,
        const FLOAT32* __restrict__ p_inp,
        const FLOAT32* __restrict__ p_bias,
        WORD32  input_height,
        WORD32  input_width,
        WORD32  input_channels,
        WORD32  kernel_height,
        WORD32  kernel_width,
        WORD32  channels_multiplier,
        WORD32  x_stride,
        WORD32  y_stride,
        WORD32  x_padding,
        WORD32  y_padding,
        WORD32  out_height,
        WORD32  out_width,
        WORD32  inp_data_format,
        WORD32  out_data_format,
        pVOID p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

    if(inp_data_format == 0)
    {
        xa_nn_conv2d_depthwise_nhwc_f32(
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
                x_stride,
                y_stride,
                x_padding,
                y_padding,
                out_height,
                out_width,
                p_scratch);
    }
    else if(inp_data_format == 1)
    {
        xa_nn_conv2d_depthwise_nchw_f32(
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
                x_stride,
                y_stride,
                x_padding,
                y_padding,
                out_height,
                out_width,
                p_scratch);
    }
    return 0;
}

WORD32 xa_nn_dilated_conv2d_depthwise_f32(
        FLOAT32* __restrict__ p_out,
        const FLOAT32* __restrict__ p_kernel,
        const FLOAT32* __restrict__ p_inp,
        const FLOAT32* __restrict__ p_bias,
        WORD32  input_height,
        WORD32  input_width,
        WORD32  input_channels,
        WORD32  kernel_height,
        WORD32  kernel_width,
        WORD32  channels_multiplier,
        WORD32  dilation_height,
        WORD32  dilation_width,
        WORD32  x_stride,
        WORD32  y_stride,
        WORD32  x_padding,
        WORD32  y_padding,
        WORD32  out_height,
        WORD32  out_width,
        WORD32  inp_data_format,
        WORD32  out_data_format,
        pVOID p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
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
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

    if(inp_data_format == 0)
    {
        xa_nn_dilated_conv2d_depthwise_nhwc_f32(
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
                p_scratch);
    }
    else if(inp_data_format == 1)
    {
        xa_nn_dilated_conv2d_depthwise_nchw_f32(
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
                p_scratch);
    }

    return 0;
}
#endif /* #if !HAVE_VFPU */
