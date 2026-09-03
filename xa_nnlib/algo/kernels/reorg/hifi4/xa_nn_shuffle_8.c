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
#include "xa_nnlib_common.h"

WORD32 xa_nn_shuffle_3D_8_8(WORD8 *__restrict__ p_out,
                            const WORD8 *__restrict__ p_inp,
                            WORD32 input_height, WORD32 input_width,
                            WORD32 input_channel, WORD32 output_height,
                            WORD32 output_width, WORD32 output_channel,
                            WORD32 interleave_groups) {
  /* NULL pointer check */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height != output_height), -1);
  XA_NNLIB_ARG_CHK_COND((input_width != output_width), -1);
  XA_NNLIB_ARG_CHK_COND((input_channel != output_channel), -1);
  XA_NNLIB_ARG_CHK_COND(
      (interleave_groups < 0 || interleave_groups > output_channel), -1);
  XA_NNLIB_ARG_CHK_COND((output_channel % interleave_groups != 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_height < 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_width < 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channel < 0), -1);
  WORD32 channel_per_group = output_channel / interleave_groups;
  WORD32 cpg, g;
  WORD8 *out_ptr, *out_ptr1, *inp_ptr;
  if (channel_per_group < 32 && interleave_groups > channel_per_group) {
    for (WORD32 w = 0; w < (output_width * output_height); w++) {
      out_ptr1 = (WORD8 *)(p_out + w * channel_per_group * interleave_groups);
      for (cpg = 0; cpg < channel_per_group; cpg++) {
        inp_ptr = (WORD8 *)(p_inp + w * output_channel + cpg);
        out_ptr = (WORD8 *)(out_ptr1 + cpg * interleave_groups);
        for (g = 0; g < interleave_groups; g++) {
          out_ptr[g] = inp_ptr[g * channel_per_group];
        }
      }
    }
  } else {
    for (g = 0; g < interleave_groups; g++) {
      out_ptr1 = (WORD8 *)(p_out + g);
      for (WORD32 w = 0; w < (output_width * output_height); w++) {
        inp_ptr = (WORD8 *)(p_inp + g * channel_per_group + w * output_channel);
        out_ptr =
            (WORD8 *)(out_ptr1 + w * channel_per_group * interleave_groups);
        for (cpg = 0; cpg < channel_per_group; cpg++) {
          out_ptr[cpg * interleave_groups] = inp_ptr[cpg];
        }
      }
    }
  }
  return 0;
}
