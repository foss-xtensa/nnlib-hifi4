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
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nn_transpose_conv_state.h"
#include <string.h>

#if HAVE_VFPU
static inline void tconv2d_f32xf32(FLOAT32* output_data,
		const FLOAT32* input_data,
		const FLOAT32* filter_data,
		const FLOAT32* bias_data,
		int stride_width, int stride_height,
		int pad_width, int pad_height,
		int input_depth, int output_depth,
		int input_height, int input_width,
		int filter_height, int filter_width,
		int output_height, int output_width,
		int num_elements,
		FLOAT32* scratch_buffer)
{
  /* scratch memory is twice as big as output buffer, and stores 2 elements per output to allow parallel mac operations */
  memset(scratch_buffer, 0, 2*num_elements*sizeof(FLOAT32));
  ae_int64 *pscratch64 = (ae_int64 *)scratch_buffer;

  int stride1 = filter_height*filter_width*input_depth*sizeof(FLOAT32);

  if(input_data && filter_data && output_data && scratch_buffer &&
            (((unsigned int)input_data&0x7)==0) && (((unsigned int)filter_data&0x7)==0) && (((unsigned int)output_data&0x7) == 0) &&
            (((unsigned int)scratch_buffer&0x7) == 0) && ((input_depth&0x3)==0) && ((filter_height*filter_width&0x3)==0))
  {
    for (int in_y = 0; in_y < input_height; ++in_y)
    {
      for (int in_x = 0; in_x < input_width; ++in_x)
      {
        const int out_x_orig = in_x*stride_width - pad_width;
        const int out_y_orig = in_y*stride_height - pad_height;
        int filt_x_min = -out_x_orig; 
        int filt_x_max = output_width - out_x_orig; 
        int filt_y_min = -out_y_orig; 
        int filt_y_max = output_height - out_y_orig; 
        filt_x_min = (filt_x_min < filter_width) ? filt_x_min : filter_width;
        filt_x_min = (filt_x_min < 0) ? 0 : filt_x_min;
        filt_x_max = (filt_x_max < filter_width) ? filt_x_max : filter_width;
        filt_x_max = (filt_x_max < 0) ? 0 : filt_x_max;
        filt_y_min = (filt_y_min < filter_height) ? filt_y_min : filter_height;
        filt_y_min = (filt_y_min < 0) ? 0 : filt_y_min;
        filt_y_max = (filt_y_max < filter_height) ? filt_y_max : filter_height;
        filt_y_max = (filt_y_max < 0) ? 0 : filt_y_max;
        FLOAT32 * __restrict__ pinp =  (FLOAT32*)&input_data[in_y*input_width*input_depth+in_x*input_depth];

        for (int in_channel = 0; in_channel < input_depth; in_channel+=4)
        {
          xtfloatx2 d_inp, d_inp1;
          XT_LSX2IP(d_inp, (xtfloatx2 *)pinp, 2*sizeof(FLOAT32));
          XT_LSX2IP(d_inp1, (xtfloatx2 *)pinp, 2*sizeof(FLOAT32));

          for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
          {
            for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
            {
              // Compute output element location.
              int out_x = out_x_orig + filter_x;//out_x_origin + filter_x;
              int out_y = out_y_orig + filter_y;//out_y_origin + filter_y;
              FLOAT32* __restrict__ pscratch_src = (FLOAT32*)&pscratch64[out_y*output_width*output_depth+out_x*output_depth];
              FLOAT32* __restrict__ pfilt = (FLOAT32*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + in_channel];

              for (int out_channel = 0; out_channel < output_depth>>2; ++out_channel)
              {
                xtfloatx2 d_fil0, d_fil1, d_fil2, d_fil3;
                xtfloatx2 d_fil10, d_fil11, d_fil12, d_fil13;
                xtfloatx2 d_scr0, d_scr1, d_scr2, d_scr3;

                d_scr0 = XT_LSX2I((xtfloatx2 *)pscratch_src, 0);
                d_scr1 = XT_LSX2I((xtfloatx2 *)pscratch_src, 8);
                d_scr2 = XT_LSX2I((xtfloatx2 *)pscratch_src, 16);
                d_scr3 = XT_LSX2I((xtfloatx2 *)pscratch_src, 24);

                XT_LSX2IP(d_fil0, (xtfloatx2*)pfilt, 8);
                XT_LSX2XP(d_fil10, (xtfloatx2*)pfilt, stride1 -8);
                XT_LSX2IP(d_fil1, (xtfloatx2*)pfilt, 8);
                XT_LSX2XP(d_fil11, (xtfloatx2*)pfilt, stride1 -8);
                XT_LSX2IP(d_fil2, (xtfloatx2*)pfilt, 8);
                XT_LSX2XP(d_fil12, (xtfloatx2*)pfilt, stride1 -8);
                XT_LSX2IP(d_fil3, (xtfloatx2*)pfilt, 8);
                XT_LSX2XP(d_fil13, (xtfloatx2*)pfilt, stride1 -8);

                d_scr0 += (d_inp*d_fil0);
                d_scr0 += (d_inp1*d_fil10);
                d_scr1 += (d_inp*d_fil1);
                d_scr1 += (d_inp1*d_fil11);
                d_scr2 += (d_inp*d_fil2);
                d_scr2 += (d_inp1*d_fil12);
                d_scr3 += (d_inp*d_fil3);
                d_scr3 += (d_inp1*d_fil13);

                XT_SSX2I(d_scr0, (xtfloatx2 *)pscratch_src, 0);
                XT_SSX2I(d_scr1, (xtfloatx2 *)pscratch_src, 8);
                XT_SSX2I(d_scr2, (xtfloatx2 *)pscratch_src, 16);
                XT_SSX2I(d_scr3, (xtfloatx2 *)pscratch_src, 24);
                pscratch_src += 8;
              }
              for (int out_channel = 0; out_channel < (output_depth&0x3); ++out_channel)
              {
                xtfloatx2 d_fil, d_fil1;
                XT_LSX2XP(d_fil, (xtfloatx2*)pfilt, 8);
                XT_LSX2XP(d_fil1, (xtfloatx2*)pfilt, stride1-8);
                xtfloatx2 d_scr = XT_LSX2I((xtfloatx2 *)pscratch_src, 0);
                d_scr += (d_inp*d_fil);
                d_scr += (d_inp1*d_fil1);
                XT_SSX2IP(d_scr, (xtfloatx2 *)pscratch_src, 2*sizeof(FLOAT32));
              }
            }
          }
        }
      }
    }
  } 
  else 
  {
    for (int in_y = 0; in_y < input_height; ++in_y)
    {
      for (int in_x = 0; in_x < input_width; ++in_x)
      {
        const int out_x_orig = in_x*stride_width - pad_width;
        const int out_y_orig = in_y*stride_height - pad_height;
        int filt_x_min = -out_x_orig; 
        int filt_x_max = output_width - out_x_orig; 
        int filt_y_min = -out_y_orig; 
        int filt_y_max = output_height - out_y_orig; 
        filt_x_min = (filt_x_min < filter_width) ? filt_x_min : filter_width;
        filt_x_min = (filt_x_min < 0) ? 0 : filt_x_min;
        filt_x_max = (filt_x_max < filter_width) ? filt_x_max : filter_width;
        filt_x_max = (filt_x_max < 0) ? 0 : filt_x_max;
        filt_y_min = (filt_y_min < filter_height) ? filt_y_min : filter_height;
        filt_y_min = (filt_y_min < 0) ? 0 : filt_y_min;
        filt_y_max = (filt_y_max < filter_height) ? filt_y_max : filter_height;
        filt_y_max = (filt_y_max < 0) ? 0 : filt_y_max;
        FLOAT32 * __restrict__ pinp =  (FLOAT32*)&input_data[in_y*input_width*input_depth+in_x*input_depth];

        ae_valign d_inp_align;
        xtfloatx2 *p_inp_align = (xtfloatx2 *)pinp;
        d_inp_align = XT_LASX2PP((xtfloatx2 *)p_inp_align);
        
        int in_channel;
        for (in_channel = 0; in_channel < (input_depth >> 2); in_channel++)
        {
          xtfloatx2 d_inp, d_inp1;

          XT_LASX2IP(d_inp, d_inp_align, (xtfloatx2 *)p_inp_align);
          XT_LASX2IP(d_inp1, d_inp_align, (xtfloatx2 *)p_inp_align);

          for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
          {
            for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
            {
              // Compute output element location.
              int out_x = out_x_orig + filter_x;
              int out_y = out_y_orig + filter_y;
              FLOAT32* __restrict__ pscratch_src = (FLOAT32*)&pscratch64[out_y*output_width*output_depth+out_x*output_depth];
              FLOAT32* __restrict__ pfilt = (FLOAT32*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + (in_channel << 2)];

              ae_valign filt_align;

              int out_channel = 0;
              for (out_channel = 0; out_channel < (output_depth >> 2) ; ++out_channel)
              {
                xtfloatx2 d_fil0, d_fil1, d_fil2, d_fil3;
                xtfloatx2 d_fil10, d_fil11, d_fil12, d_fil13;
                xtfloatx2 d_scr0, d_scr1, d_scr2, d_scr3;

                d_scr0 = XT_LSX2I((xtfloatx2 *)pscratch_src, 0);
                d_scr1 = XT_LSX2I((xtfloatx2 *)pscratch_src, 8);
                d_scr2 = XT_LSX2I((xtfloatx2 *)pscratch_src, 16);
                d_scr3 = XT_LSX2I((xtfloatx2 *)pscratch_src, 24);

                filt_align = XT_LASX2PP((xtfloatx2 *)pfilt);
                XT_LASX2IP(d_fil0 , filt_align, (xtfloatx2*)pfilt);
                XT_LASX2IP(d_fil10, filt_align, (xtfloatx2*)pfilt);
                pfilt += ((stride1 >> 2) -4);

                filt_align = XT_LASX2PP((xtfloatx2 *)pfilt);
                XT_LASX2IP(d_fil1 , filt_align, (xtfloatx2*)pfilt);
                XT_LASX2IP(d_fil11, filt_align, (xtfloatx2*)pfilt);
                pfilt += ((stride1 >> 2) -4);
                
                filt_align = XT_LASX2PP((xtfloatx2 *)pfilt);
                XT_LASX2IP(d_fil2 , filt_align, (xtfloatx2*)pfilt);
                XT_LASX2IP(d_fil12, filt_align, (xtfloatx2*)pfilt);
                pfilt += ((stride1 >> 2) -4);
                
                filt_align = XT_LASX2PP((xtfloatx2 *)pfilt);
                XT_LASX2IP(d_fil3 , filt_align, (xtfloatx2*)pfilt);
                XT_LASX2IP(d_fil13, filt_align, (xtfloatx2*)pfilt);
                pfilt += ((stride1 >> 2) -4);

                d_scr0 += (d_inp*d_fil0);
                d_scr0 += (d_inp1*d_fil10);
                d_scr1 += (d_inp*d_fil1);
                d_scr1 += (d_inp1*d_fil11);
                d_scr2 += (d_inp*d_fil2);
                d_scr2 += (d_inp1*d_fil12);
                d_scr3 += (d_inp*d_fil3);
                d_scr3 += (d_inp1*d_fil13);

                XT_SSX2I(d_scr0, (xtfloatx2 *)pscratch_src, 0);
                XT_SSX2I(d_scr1, (xtfloatx2 *)pscratch_src, 8);
                XT_SSX2I(d_scr2, (xtfloatx2 *)pscratch_src, 16);
                XT_SSX2I(d_scr3, (xtfloatx2 *)pscratch_src, 24);
                pscratch_src += 8;
              }
              
              for (out_channel = 0; out_channel < (output_depth&0x3); ++out_channel)
              {
                xtfloatx2 d_fil, d_fil1;
                filt_align = XT_LASX2PP((xtfloatx2 *)pfilt);
                XT_LASX2IP(d_fil,  filt_align, (xtfloatx2*)pfilt);
                XT_LASX2IP(d_fil1, filt_align, (xtfloatx2*)pfilt);
                pfilt += ((stride1 >> 2) -4);
                xtfloatx2 d_scr = XT_LSX2I((xtfloatx2 *)pscratch_src, 0);
                d_scr += (d_inp*d_fil);
                d_scr += (d_inp1*d_fil1);
                XT_SSX2IP(d_scr, (xtfloatx2 *)pscratch_src, 2*sizeof(FLOAT32));
              }
            }
          }
        }

        for (in_channel=0 ; in_channel < (input_depth&0x3); in_channel++)
        {
          xtfloat d_inp;
          xtfloatx2 d_inpx2;
          xtfloat zero = 0.0f;
          XT_LSIP(d_inp, (xtfloat *)p_inp_align, 4);
          for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
          {
            for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
            {
              // Compute output element location.
              int out_x = out_x_orig + filter_x;
              int out_y = out_y_orig + filter_y;
              FLOAT32* __restrict__ pscratch_src = (FLOAT32*)&pscratch64[out_y*output_width*output_depth+out_x*output_depth];
              FLOAT32* __restrict__ pfilt = (FLOAT32*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + (input_depth&~0x3) + in_channel];

              for (int out_channel = 0; out_channel < output_depth; out_channel++)
              {
                xtfloat   d_fil;
                xtfloatx2 d_filx2;
                xtfloatx2 d_scr0;

                d_scr0 = XT_LSX2I((xtfloatx2 *)pscratch_src, 0);
                XT_LSXP(d_fil, pfilt, stride1);
                d_inpx2 = XT_SEL32_LL_SX2((xtfloatx2)(d_inp), (xtfloatx2)(zero));
                d_filx2 = XT_SEL32_LL_SX2((xtfloatx2)(d_fil), (xtfloatx2)(zero));
                d_scr0 += (d_inpx2*d_filx2);
                XT_SSX2IP(d_scr0, (xtfloatx2 *)pscratch_src, 8);
              }
            }
          }
        }
      }
    } 
  }
  if(bias_data)
  {
    FLOAT32 *pbias = (FLOAT32*)bias_data;

    for (int out_channel = 0; out_channel < output_depth; ++out_channel)
    {
      xtfloatx2 acc;
      FLOAT32 dbias;
      FLOAT32 *pscratch = (FLOAT32*)&pscratch64[out_channel];
      FLOAT32 *pout = (FLOAT32*)&output_data[out_channel];
      XT_LSIP(dbias, pbias, sizeof(FLOAT32));

      for (int i = 0; i < (output_height*output_width); i++)
      {
        XT_LSX2XP(acc, (xtfloatx2 *)pscratch, 2*output_depth*sizeof(FLOAT32));
        XT_SSXP(XT_RADD_SX2(acc) + dbias, pout, output_depth*sizeof(FLOAT32));
      }
    }
  }
  else
  {
    FLOAT32 *pscratch = scratch_buffer;
    FLOAT32 *pout = (FLOAT32*)output_data;
    for (int i = 0; i < output_height*output_width; i++)
    {
      for (int out_channel = 0; out_channel < output_depth; ++out_channel)
      {
        xtfloatx2 acc;
        XT_LSX2IP(acc, (xtfloatx2 *)pscratch, 2*sizeof(FLOAT32));
        XT_SSIP(XT_RADD_SX2(acc), pout, sizeof(FLOAT32));
      }
    }
  }    
}

/* Handle sub-kernel formation and transpose */
static inline void tconv2d_std_reorder_kernel_f32
    (pVOID p_scratch
     ,const FLOAT32* p_kernel
     ,WORD32 kernel_height
     ,WORD32 kernel_width
     ,WORD32 input_channels
     ,WORD32 output_channels
     ,WORD32 x_stride
     ,WORD32 y_stride
     ,WORD32 subker_size
    )
{
  WORD32 kIdx, kIdy;
  WORD32 kernelIdx;

  WORD32 kx, ky, outCh, inIdx;
  WORD32 kxStart, kyStart;

  WORD32 input_channels_pad = PADDED_SIZE(input_channels, ALIGNMENT >> 2);
  WORD32 pitch_d = input_channels;
  WORD32 pitch_w = kernel_width * input_channels;
  WORD32 pitch_h = kernel_height * kernel_width * input_channels;

  WORD32 subkermax_w = (kernel_width + x_stride - 1) / x_stride;
  WORD32 subkermax_h = (kernel_height + y_stride - 1) / y_stride;
  
  FLOAT32 *p_ker;

  /* Conversion from NDWH -> DNWH,                       */
  /* transposing of kernels and formation of sub-kernels */
  for (kIdy = 0; kIdy < y_stride; kIdy++)
  {
    for (kIdx = 0; kIdx < x_stride; kIdx++)
    {
      kernelIdx = kIdy * x_stride + kIdx;
      FLOAT32 *p_dst = ((FLOAT32 *)p_scratch + kernelIdx * subker_size);

      kyStart = kernel_height - 1 - ((kernel_height + y_stride - kIdy - 1) % y_stride);
      kxStart = kernel_width - 1 - ((kernel_width + x_stride - kIdx - 1) % x_stride);
      WORD32 subker_w = (kernel_width + x_stride - kIdx - 1) / x_stride;
      WORD32 subker_h = (kernel_height + y_stride - kIdy - 1) / y_stride;

      for (outCh = 0; outCh < output_channels; outCh++)      /* N */
      {
        p_dst += (subkermax_h - subker_h) * subkermax_w * input_channels_pad; /* Add top padding to the subkernel */
        for (ky = kyStart; ky >= 0; ky -= y_stride)          /* H */
        {
          p_dst += (subkermax_w - subker_w) * input_channels_pad; /* Add left padding to the subkernel */
          for (kx = kxStart; kx >= 0; kx -= x_stride)        /* W */
          {
            p_ker = (FLOAT32 *)&p_kernel[inIdx = outCh * pitch_h + ky * pitch_w + kx * pitch_d];
            xa_nn_memcpy(p_dst, p_ker, input_channels * sizeof(FLOAT32));
            p_dst+=input_channels_pad;
          }
        }
      }
    }
  }
}

static inline void tconv_pad(
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const FLOAT32* __restrict__ p_bias,
    FLOAT32 *p_out,
    WORD32 idx_width,
    WORD32 idx_height)
{
  WORD32 i, j, k;

  /* When kernel has no valid input for convolution, output is just bias */
  if(p_bias != NULL){
    for(i = idx_height; i < out_height; i++)
    {
      for(j = idx_width; j < out_width; j++)
      {
        xtfloat *ptrout = (xtfloat*)&p_out[i * out_height_offset + j * out_width_offset];
        xtfloat *pbias = (xtfloat*)p_bias;
        xtfloat q1;

        for(k = 0; k < out_channels; k++)
        {
          XT_LSIP(q1, pbias, 4);
          XT_SSXP(q1, ptrout, out_channels_offset*sizeof(FLOAT32));
        }
      }
    }
  }
  else{
    for(i = idx_height; i < out_height; i++)
    {
      for(j = idx_width; j < out_width; j++)
      {
        xtfloat *ptrout = (xtfloat*)&p_out[i * out_height_offset + j * out_width_offset];
        xtfloat q1 = 0.0f;
        for(k = 0; k < out_channels; k++)
        {
          XT_SSXP(q1, ptrout, out_channels_offset*sizeof(FLOAT32));
        }
      }
    }
  }
}

static inline void transpose_conv2d_std_f32xf32(FLOAT32* output_data,
		const FLOAT32* input_data,
		const FLOAT32* filter_data,
		const FLOAT32* bias_data,
		int stride_width, int stride_height,
		int pad_width, int pad_height,
		int input_depth, int output_depth,
		int input_height, int input_width,
		int filter_height, int filter_width,
		int output_height, int output_width,
		pVOID scratch_buffer)
{
  /* Transpose and Reorder the kernel into sub-kernels */
  WORD32 subkerX_max = (filter_width + stride_width - 1) / stride_width;
  WORD32 subkerY_max = (filter_height + stride_height - 1) / stride_height;
  WORD32 n_subker = stride_width * stride_height;
  WORD32 input_depth_pad = PADDED_SIZE(input_depth, (ALIGNMENT>>2));
  WORD32 subker_size = subkerX_max * subkerY_max * input_depth_pad * output_depth;
  /* memset the kernel reordering memory on scratch */
  memset(scratch_buffer, (WORD8)0, subker_size * n_subker * sizeof(FLOAT32));

  tconv2d_std_reorder_kernel_f32(scratch_buffer, filter_data, filter_height, filter_width, input_depth, output_depth, stride_width, stride_height, subker_size);

  /* Calculate padding values */
  WORD32 x_pad = subkerX_max - 1;
  WORD32 y_pad = subkerY_max - 1;
  WORD32 y_b_pad = subkerY_max - 1;

  /* Calculate valid output dims */
  WORD32 orig_valid_out_h = XT_MIN(output_height, filter_height + stride_height * (input_height -1) - pad_height);
  WORD32 orig_valid_out_w = XT_MIN(output_width, filter_width + stride_width * (input_width -1) - pad_width);
  WORD32 valid_out_h = orig_valid_out_h + pad_height;
  WORD32 valid_out_w = orig_valid_out_w + pad_width;
  WORD32 out_h_per_subker = orig_valid_out_h / stride_height;
  WORD32 pad_h_per_subker = pad_height / stride_height;

  /* Calculate valid and actual output offsets */
  WORD32 out_data_format = 0; // NHWC
  WORD32 out_channels_offset = out_data_format ? valid_out_h * valid_out_w : 1;
  WORD32 final_out_channels_offset = out_data_format ? output_height * output_width : 1;
  WORD32 final_out_height_offset = out_data_format ? output_width : output_width * output_depth;
  WORD32 final_out_width_offset = out_data_format ? 1 : output_depth;

  /* Calculate pointers for different sections on scratch buffer */
  WORD32 kernel_size = PADDED_SIZE(subker_size * n_subker, 2);
  FLOAT32 *p_trp_ker = (FLOAT32 *)scratch_buffer; 
  FLOAT32 *p_scr_cnv = (FLOAT32 *)((FLOAT32 *)scratch_buffer + kernel_size);

  /* Handle cases that have less valid output dimension than the output dimension given by the user */
  if(((orig_valid_out_h) < output_height))
  {
    tconv_pad(output_width, output_height, output_depth, final_out_channels_offset, final_out_width_offset, final_out_height_offset, bias_data, output_data, 0, XT_MAX(0,orig_valid_out_h));
  }
  if((orig_valid_out_w) < output_width)
  {
    tconv_pad(output_width, output_height, output_depth, final_out_channels_offset, final_out_width_offset, final_out_height_offset, bias_data, output_data, XT_MAX(0,orig_valid_out_w), 0);
  }
  if((out_h_per_subker < 0))
  {
	tconv_pad(output_width, output_height, output_depth, final_out_channels_offset, final_out_width_offset, final_out_height_offset, bias_data, output_data, 0, 0);
	return;
  }

  WORD32 j;
  WORD32 input_bytewidth = 4;
  VOID *pp_inp = (VOID *)(input_data);

  /* Conv 2D Standard code init */
  /* Here the x-pad and y-pad values are controlled by the filter dimensions
   * x-r-pad = filter_width - 1 and y-b-pad = filter_height - 1
   * x_pad and y_pad depend on kernel dimension and the padding.
  */
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scr_cnv;
  xa_nn_transpose_conv_init_state((void*)p_state
      ,(void*)p_trp_ker
      ,input_height
      ,input_depth
      ,subkerY_max
      ,subkerX_max
      ,PREC_F32);

  /* When kernel convolves over input region */
  // Initialize circular buffer
  conv2d_std_init_cir_buf(input_depth, input_depth_pad, input_bytewidth, input_width, input_height, y_pad, y_b_pad, x_pad, subkerX_max, 1, (VOID**)&pp_inp, p_state);

  // Index to padded input width
  WORD32 idx_beg_inp_width_pad = subkerX_max - 1;
  idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;

  FLOAT32 *po_tmp;
  WORD32 rem_val_out_w = valid_out_w % stride_width;
  WORD32 pad_w = pad_width;
  
  // Process Loop to compute one output plane [out_height x out_channels] per iteration
  WORD32 out_w_looopcnt = valid_out_w / stride_width;
  for(j = 0; j < out_w_looopcnt; j++)
  {
    // Add x_stride x (input_height x input_channels) new planes to circular buffer
    conv2d_std_update_cir_buf(input_depth, input_depth_pad, input_bytewidth, input_width, input_height, y_pad, y_b_pad, x_pad, subkerX_max, 1, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state);

    // Update index to input width padded
    idx_beg_inp_width_pad += 1;

    int kernelIdx;
    for (int kIdx = 0; kIdx < stride_width; kIdx++, pad_w--)
    {
      WORD32 rem_val_out_h = (valid_out_h - pad_height) % stride_height;
      WORD32 is_pad_w = (pad_w > 0);

      if(!is_pad_w)
      {
        WORD32 pad_h_ky = stride_height - (pad_height % stride_height); // Required to handle valid inp_h for subkernel
        po_tmp = output_data;
        for (int kIdy = 0; kIdy < stride_height; kIdy++, rem_val_out_h--, pad_h_ky--)
        {
          kernelIdx = ((kIdy + pad_height) % stride_height) * stride_width + kIdx;
          FLOAT32 *p_subkernel = ((FLOAT32 *)p_trp_ker + kernelIdx * subker_size);
          WORD32 rem_out_h_per_subker = (rem_val_out_h > 0) ? 1 : 0; 

          // Adjust the circ_buf pointer as per pad_height
          WORD32 cir_buf_inp_offset = pad_h_per_subker * input_depth_pad * subkerX_max;
          cir_buf_inp_offset = (pad_h_ky > 0) ? cir_buf_inp_offset : cir_buf_inp_offset + input_depth_pad * subkerX_max;
          FLOAT32 *p_inp_cir_buf = p_state->cir_buf.p_curr;
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_inp_cir_buf, cir_buf_inp_offset * input_bytewidth);        
          // Convolution using matXvec with matrix as circular buffer
          xa_nn_matXvec_f32_circ
            (po_tmp /* output */
            ,p_inp_cir_buf/* matrix: rows x cols */
            ,p_subkernel /* vec: cols */
            ,(FLOAT32 *)bias_data /* bias */
            ,out_h_per_subker + rem_out_h_per_subker /* rows */
            ,input_depth_pad * subkerX_max * subkerY_max /* cols */
            ,input_depth_pad * subkerX_max /* row_offset */
            ,output_depth /* vec_count */
            ,input_depth_pad * subkerX_max * subkerY_max /* vec_stride */
            ,out_channels_offset /* out_col_offset */
            ,final_out_height_offset * stride_height /* out_row_offset */
            );
          po_tmp += final_out_height_offset;
        }
      }
      output_data = is_pad_w ? output_data : output_data + output_depth;
    }
  }

  /* Tail loop depending on remaining valid_out_width */
  if(rem_val_out_w)
  {
    // Add x_stride x (input_height x input_channels) new planes to circular buffer
    conv2d_std_update_cir_buf(input_depth, input_depth_pad, input_bytewidth, input_width, input_height, y_pad, y_b_pad, x_pad, subkerX_max, 1, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state);

    // Update index to input width padded
    idx_beg_inp_width_pad += 1;

    int kernelIdx;
    for (int kIdx = 0; kIdx < rem_val_out_w; kIdx++, pad_w--)
    {
      WORD32 rem_val_out_h = (valid_out_h - pad_height) % stride_height;
      WORD32 is_pad_w = (pad_w > 0);

      if(!is_pad_w)
      {
        WORD32 pad_h_ky = stride_height - (pad_height % stride_height); // Required to handle valid inp_h for subkernel
        po_tmp = output_data;
        for (int kIdy = 0; kIdy < stride_height; kIdy++, rem_val_out_h--, pad_h_ky--)
        {
          kernelIdx = ((kIdy + pad_height) % stride_height) * stride_width + kIdx;
          FLOAT32 *p_subkernel = ((FLOAT32 *)p_trp_ker + kernelIdx * subker_size);
          WORD32 rem_out_h_per_subker = (rem_val_out_h > 0) ? 1 : 0; 
          // Adjust the circ_buf pointer as per pad_height
          WORD32 cir_buf_inp_offset = pad_h_per_subker * input_depth_pad * subkerX_max;
          cir_buf_inp_offset = (pad_h_ky > 0) ? cir_buf_inp_offset : cir_buf_inp_offset + input_depth_pad * subkerX_max;
          FLOAT32 *p_inp_cir_buf = p_state->cir_buf.p_curr;
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_inp_cir_buf, cir_buf_inp_offset * input_bytewidth);
          
          // Convolution using matXvec with matrix as circular buffer
          xa_nn_matXvec_f32_circ
            (po_tmp /* output */
            ,p_inp_cir_buf/* matrix: rows x cols */
            ,p_subkernel /* vec: cols */
            ,(FLOAT32 *)bias_data /* bias */
            ,out_h_per_subker + rem_out_h_per_subker /* rows */
            ,input_depth_pad * subkerX_max * subkerY_max /* cols */
            ,input_depth_pad * subkerX_max /* row_offset */
            ,output_depth /* vec_count */
            ,input_depth_pad * subkerX_max * subkerY_max /* vec_stride */
            ,out_channels_offset /* out_col_offset */
            ,final_out_height_offset * stride_height /* out_row_offset */
            );   
          po_tmp += final_out_height_offset;
        }
      }
      output_data = is_pad_w ? output_data : output_data + output_depth;
    }
  }
}
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_transpose_conv_f32, (FLOAT32* output_data,
            const FLOAT32* input_data,
            const FLOAT32* filter_data,
            const FLOAT32* bias_data,
            int stride_width, int stride_height,
            int pad_width, int pad_height,
            int input_depth, int output_depth,
            int input_height, int input_width,
            int filter_height, int filter_width,
            int output_height, int output_width,
            int num_elements,
            void* scratch_buffer))
#else
WORD32 xa_nn_transpose_conv_f32(FLOAT32* output_data,
		const FLOAT32* input_data,
		const FLOAT32* filter_data,
		const FLOAT32* bias_data,
		int stride_width, int stride_height,
		int pad_width, int pad_height,
		int input_depth, int output_depth,
		int input_height, int input_width,
		int filter_height, int filter_width,
		int output_height, int output_width,
		int num_elements,
		void* scratch_buffer)
{
	/* NULL pointer checks */
	XA_NNLIB_ARG_CHK_PTR(output_data, -1);
	XA_NNLIB_ARG_CHK_PTR(filter_data, -1);
	XA_NNLIB_ARG_CHK_PTR(input_data, -1);
	XA_NNLIB_ARG_CHK_PTR(scratch_buffer, -1);
	/* Pointer alignment checks */
	XA_NNLIB_ARG_CHK_ALIGN(output_data, sizeof(FLOAT32), -1);
	XA_NNLIB_ARG_CHK_ALIGN(filter_data, sizeof(FLOAT32), -1);
	XA_NNLIB_ARG_CHK_ALIGN(input_data, sizeof(FLOAT32), -1);
	XA_NNLIB_ARG_CHK_ALIGN(bias_data, sizeof(FLOAT32), -1);
	XA_NNLIB_ARG_CHK_ALIGN(scratch_buffer, sizeof(FLOAT32), -1);
	/* Basic Parameter checks */
	XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((input_depth <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((filter_height <= 0 || filter_width <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((output_depth <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((stride_height <= 0 || stride_width <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((pad_height < 0 || pad_width < 0), -1);
	XA_NNLIB_ARG_CHK_COND((output_height <= 0 || output_width <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((num_elements <= 0), -1);

  int ker_grt_inp = (filter_width > input_width || filter_height > input_height);
  int str_leq_ker = (stride_width <= filter_width && stride_height <= filter_height);

  if(!ker_grt_inp && str_leq_ker)
  {
    transpose_conv2d_std_f32xf32(output_data, input_data, filter_data, bias_data,
    stride_width, stride_height, pad_width, pad_height, input_depth, output_depth,
		input_height, input_width, filter_height, filter_width,	output_height, output_width,
		scratch_buffer);
  }
  else
  {
    tconv2d_f32xf32(output_data, input_data, filter_data, bias_data,
    stride_width, stride_height, pad_width, pad_height, input_depth, output_depth,
		input_height, input_width, filter_height, filter_width,	output_height, output_width,
		num_elements, scratch_buffer);
  }

	return 0;
}
#endif
