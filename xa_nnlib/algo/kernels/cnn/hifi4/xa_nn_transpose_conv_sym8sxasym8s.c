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
#include "xa_nn_transpose_conv_state.h"
#include <string.h>

#if XCHAL_HAVE_HIFI1S
static inline void tconv2d_sym8sxasym8s(WORD8* output_data,
    const WORD8* input_data,
    const WORD8* filter_data,
    const WORD32* bias_data,
    int stride_width, int stride_height,
    int pad_width, int pad_height,
    int input_depth, int output_depth,
    int input_height, int input_width,
    int filter_height, int filter_width,
    int output_height, int output_width,
    int num_elements,
    int input_offset, int output_offset,
    int *output_shift, int *output_multiplier,
    int *scratch_buffer)
{
  ae_int32x2 *pscratch1 = (ae_int32x2*)scratch_buffer;
  ae_int32x2 dzero = AE_ZERO32();
  for(int i=0; i<(num_elements>>1); i++){
    AE_S32X2_IP(dzero, pscratch1, 8);
  }
  if(num_elements&0x1)
	AE_S32_L_IP(dzero, (ae_int32*)pscratch1, 4);

  ae_int32 *pscratch = (ae_int32*)scratch_buffer;

  int stride1 = filter_height*filter_width*input_depth;
  WORD8 *pinp;
  ae_int8x8 d_izb = AE_MOVDA8(input_offset);

  /*
   * Special case for input_depth multiple of 16
   */
  if(input_data && filter_data && scratch_buffer
     && (((unsigned int)input_data&0x7)==0) && (((unsigned int)filter_data&0x7)==0) && 
     (((unsigned int)scratch_buffer&0x7) == 0) && ((input_depth&0xF)==0) && ((filter_height*filter_width&0x3)==0))
  {
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
          pinp =  (WORD8*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
          for (int in_channel = 0; in_channel < input_depth; in_channel+=16)
          {
            ae_int8x8 d_fil, d_fil1;
            ae_int8x8 d_inp, d_inp1;
            ae_int16x4 dtmp1, dtmp2, dtmp3, dtmp4;
            AE_L8X8_IP(d_inp, (ae_int8x8*)pinp, 8);
            AE_L8X8_IP(d_inp1, (ae_int8x8*)pinp, 8);
            AE_ADDW8(dtmp1, dtmp2, d_inp, d_izb);
            AE_ADDW8(dtmp3, dtmp4, d_inp1, d_izb);

            for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
            {
              for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
              {
                // Compute output element location.
                int out_x = out_x_orig + filter_x;//out_x_origin + filter_x;
                int out_y = out_y_orig + filter_y;//out_y_origin + filter_y;
                ae_int32 *pscratch_src = (ae_int32*)&scratch_buffer[out_y*output_width*output_depth+out_x*output_depth];
                ae_int32x2 d_scr;
                WORD8* pfilt = (WORD8*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + in_channel];
                AE_L8X8_IP(d_fil, (ae_int8x8*)pfilt, 8);
                AE_L8X8_XP(d_fil1, (ae_int8x8*)pfilt, stride1-8);

                for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                {
                  d_scr =  AE_SEL32_LL(AE_ZERO32(), AE_L32_I(pscratch_src, 0));
                  AE_MULAAAA16Q8(d_scr, dtmp1, dtmp2, d_fil);
                  AE_MULAAAA16Q8(d_scr, dtmp3, dtmp4, d_fil1);
                  d_scr = AE_ADD32_HL_LH(d_scr, d_scr);
                  AE_L8X8_IP(d_fil, (ae_int8x8*)pfilt, 8);
                  AE_L8X8_XP(d_fil1, (ae_int8x8*)pfilt, stride1-8);
                  AE_S32_L_IP(d_scr, pscratch_src, sizeof(WORD32));
                }
              }
            }
          }
        }
      }
    }    
  }
  else if(input_data && filter_data && scratch_buffer && ((input_depth&0xF)==0) && ((filter_height*filter_width&0x3)==0))
  {    
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
          pinp =  (WORD8*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
          ae_valign align0 = AE_LA64_PP(pinp);
          for (int in_channel = 0; in_channel < input_depth; in_channel+=16)
          {
            ae_int8x8 d_fil, d_fil1;
            ae_int8x8 d_inp, d_inp1;
            ae_int16x4 dtmp1, dtmp2, dtmp3, dtmp4;
            AE_LA8X8_IP(d_inp, align0, (ae_int8x8*)pinp);
            AE_LA8X8_IP(d_inp1, align0, (ae_int8x8*)pinp);
            AE_ADDW8(dtmp1, dtmp2, d_inp, d_izb);
            AE_ADDW8(dtmp3, dtmp4, d_inp1, d_izb);

            for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
            {
              for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
              {
                int out_x = out_x_orig + filter_x;
                int out_y = out_y_orig + filter_y;
                ae_int32 *pscratch_src = (ae_int32*)&scratch_buffer[out_y*output_width*output_depth+out_x*output_depth];
                ae_int32x2 d_scr;
                WORD8* pfilt = (WORD8*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + in_channel];
                ae_valign align1 = AE_LA64_PP(pfilt);
                AE_LA8X8_IP(d_fil, align1, (ae_int8x8*)pfilt);
                AE_LA8X8_IP(d_fil1, align1, (ae_int8x8*)pfilt);
                pfilt += (stride1-16);

                for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                {
                  d_scr =  AE_SEL32_LL(AE_ZERO32(), AE_L32_I(pscratch_src, 0));
                  AE_MULAAAA16Q8(d_scr, dtmp1, dtmp2, d_fil);
                  AE_MULAAAA16Q8(d_scr, dtmp3, dtmp4, d_fil1);
                  d_scr = AE_ADD32_HL_LH(d_scr, d_scr);
                  ae_valign align1 = AE_LA64_PP(pfilt);
                  AE_LA8X8_IP(d_fil, align1, (ae_int8x8*)pfilt);
                  AE_LA8X8_IP(d_fil1, align1, (ae_int8x8*)pfilt);
                  pfilt += (stride1-16);
                  AE_S32_L_IP(d_scr, pscratch_src, sizeof(WORD32));
                }
              }
            }
          }
        }
      }
    }    
  }    
  else if(input_data && filter_data && scratch_buffer &&
      (((unsigned int)input_data&0x3)==0) && (((unsigned int)filter_data&0x3)==0) &&
      (((unsigned int)scratch_buffer&0x7) == 0) && ((input_depth&0x3)==0))
  {
    {
      //tbd : batch = 1, need to handle other values and in_x_min/max= 0 .. need toc heck for other values
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
          pinp =  (WORD8*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
          for (int in_channel = 0; in_channel < input_depth; in_channel+=4)
          {
            ae_int16x4 d_inp;
            AE_L8X4S_IP(d_inp, pinp, sizeof(WORD32));
            d_inp = AE_ADD16(d_inp, AE_MOVDA16(input_offset));
            for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
            {
              for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
              {
                // Compute output element location.
                int out_x = out_x_orig + filter_x;//out_x_origin + filter_x;
                int out_y = out_y_orig + filter_y;//out_y_origin + filter_y;
                ae_int32 *pscratch_src = (ae_int32*)&scratch_buffer[out_y*output_width*output_depth+out_x*output_depth];
                ae_int64 d_scr;
                WORD8* pfilt = (WORD8*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + in_channel];
                ae_int16x4 d_fil;
                AE_L8X4S_XP(d_fil, pfilt, stride1);
                for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                {
		              d_scr = AE_SRAI64(AE_CVT64F32_H(AE_L32_I(pscratch_src, 0)), 32);
                  AE_MULAAAAQ16(d_scr, d_inp, d_fil);
                  AE_L8X4S_XP(d_fil, pfilt, stride1);
                  AE_S32_L_IP(AE_MOVINT32X2_FROMINT64(d_scr), pscratch_src, sizeof(WORD32));
                }
              }
            }
          }
        }
      }
    }    
  }
  else
  {     
    {
      for (int in_y = 0; in_y < input_height; ++in_y)
      {
        for (int in_x = 0; in_x < input_width; ++in_x)
        {
          for (int in_channel = 0; in_channel < input_depth; ++in_channel)
          {
            const int out_x_origin = (in_x * stride_width) - pad_width;
            const int out_y_origin = (in_y * stride_height) - pad_height;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y)
            {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x)
              {
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) && (out_y < output_height))
                {
                  for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                  {
                    const int32_t input_value = input_data[((in_y)*input_width+in_x)*input_depth+in_channel] + input_offset;
                    const int32_t filter_value = filter_data[(((out_channel*filter_height)+filter_y)*filter_width+filter_x)*input_depth+in_channel];
                    scratch_buffer[((out_y)*output_width+out_x)*output_depth+out_channel] += (int32_t)(input_value * filter_value);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if(bias_data)
  {
    ae_int32x2 acc0, acc1;
    ae_int32x2 dbias;
    ae_int32 *pbias = (ae_int32*)bias_data;

    for (int out_channel = 0; out_channel < output_depth; ++out_channel)
    {
#if TFLITE_SINGLE_ROUNDING
      int left_shift = output_shift[out_channel];
      int right_shift = output_shift[out_channel];
      left_shift = 31 - left_shift;
      left_shift = left_shift << 16 | left_shift;      
      (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int left_shift = output_shift[out_channel] < 0 ? 0 : output_shift[out_channel];
      int right_shift = output_shift[out_channel] > 0 ? 0 : -output_shift[out_channel];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      pscratch = (ae_int32*)&scratch_buffer[out_channel];
      WORD8 *pout = (WORD8*)&output_data[out_channel];
      ae_int32 *pscratch1 = (ae_int32*)&scratch_buffer[((output_height*output_width)>>1)*output_depth+out_channel];
      WORD8 *pout1 = (WORD8*)&output_data[((output_height*output_width)>>1)*output_depth+out_channel];
      AE_L32_IP(dbias, pbias, sizeof(WORD32));
      AE_L32_XP(acc0, pscratch, output_depth*sizeof(WORD32));
      AE_L32_XP(acc1, pscratch1, output_depth*sizeof(WORD32));
      ae_int32 out_mult = output_multiplier[out_channel];
      for (int i = 0; i < ((output_height*output_width)>>1); i++)
      {
        ae_int32x2 out32;
        ae_int32x2 acc;
        acc = AE_SEL32_LL(acc0, acc1);
        acc = AE_ADD32(acc, dbias);
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out32, acc, out_mult, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(out32, acc, out_mult, left_shift, right_shift);
#endif        
        out32 = AE_ADD32(out32, AE_MOVDA32(output_offset));
        out32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(out32, AE_MOVDA32(-128)));
        AE_L32_XP(acc0, pscratch, output_depth*sizeof(WORD32));
        AE_L32_XP(acc1, pscratch1, output_depth*sizeof(WORD32));
        *pout = (WORD8)AE_MOVAD32_H(out32);
        pout+=output_depth;
        *pout1 = (WORD8)AE_MOVAD32_L(out32);
        pout1+=output_depth;
      }
      if((output_height*output_width) & 1)
      {
        ae_int16x4 out16;
        ae_int32x2 out1_32;
        acc1 = AE_ADD32(acc1, dbias);
#if TFLITE_SINGLE_ROUNDING
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out1_32, acc1, out_mult, left_shift, right_shift);
#else          
        MPY_BY_QUANT_MULT_X2_OUT32(out1_32, acc1, out_mult, left_shift, right_shift);
#endif        
        out1_32 = AE_ADD32(out1_32, AE_MOVDA32(output_offset));
        out1_32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(out1_32, AE_MOVDA32(-128)));
        out16 = AE_SAT16X4(out1_32, out1_32);
        *pout1 = (WORD8)AE_MOVAD16_0(out16);
      }
    }
  }
  else
  {
    ae_int32 *pscratch_test = (ae_int32*)scratch_buffer;
    WORD8 *pout = (WORD8*)output_data;
    for (int i = 0; i < output_height*output_width; i++)
    {   
      for (int out_channel = 0; out_channel < output_depth; ++out_channel)
      {
#if TFLITE_SINGLE_ROUNDING
        int left_shift = output_shift[out_channel];
        int right_shift = output_shift[out_channel];
        left_shift = 31 - left_shift;
        left_shift = left_shift << 16 | left_shift;        
        (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
        int left_shift = output_shift[out_channel] < 0 ? 0 : output_shift[out_channel];
        int right_shift = output_shift[out_channel] > 0 ? 0 : -output_shift[out_channel];
#endif /* #if TFLITE_SINGLE_ROUNDING */        
        ae_int32x2 acc;
        ae_int32x2 out0_32;
        ae_int16x4 out16;
        AE_L32_IP(acc, pscratch_test, sizeof(WORD32));
#if TFLITE_SINGLE_ROUNDING        
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(out0_32, acc, output_multiplier[out_channel], left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(out0_32, acc, output_multiplier[out_channel], left_shift, right_shift);
#endif        
        out0_32 = AE_ADD32(out0_32, AE_MOVDA32(output_offset));
        out0_32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(out0_32, AE_MOVDA32(-128))); 
        out16 = AE_SAT16X4(out0_32, out0_32);  
        *pout++ = (WORD8)AE_MOVAD16_0(out16);
      }
    }   
  }
}
#else // XCHAL_HAVE_HIFI1S
static inline void tconv2d_sym8sxasym8s(WORD8* output_data,
    const WORD8* input_data,
    const WORD8* filter_data,
    const WORD32* bias_data,
    int stride_width, int stride_height,
    int pad_width, int pad_height,
    int input_depth, int output_depth,
    int input_height, int input_width,
    int filter_height, int filter_width,
    int output_height, int output_width,
    int num_elements,
    int input_offset, int output_offset,
    int *output_shift, int *output_multiplier,
    int64_t* scratch_buffer)
{
  ae_int64 *pscratch = (ae_int64*)scratch_buffer;
	ae_int64 dzero = AE_ZERO64();
	for(int i=0; i<num_elements; i++){
		AE_S64_IP(dzero, pscratch, 8);
  }

  int stride1 = filter_height*filter_width*input_depth;
  WORD8 *pinp;

  /*
   * Special case for input_depth multiple of 16
   */
  if(input_data && filter_data && scratch_buffer &&
      (((unsigned int)input_data&0x3)==0) && (((unsigned int)filter_data&0x3)==0) && 
      (((unsigned int)scratch_buffer&0x7) == 0) && ((input_depth&0xF)==0) && ((filter_height*filter_width&0x3)==0))
  {
    {
      //tbd : batch = 1, need to handle other values and in_x_min/max= 0 .. need toc heck for other values
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
          pinp =  (WORD8*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
          for (int in_channel = 0; in_channel < input_depth; in_channel+=16)
          {
            ae_int16x4 d_inp, d_inp1, d_inp2, d_inp3;
            AE_L8X4F_IP(d_inp, pinp, sizeof(WORD32));
            AE_L8X4F_IP(d_inp1, pinp, sizeof(WORD32));
            AE_L8X4F_IP(d_inp2, pinp, sizeof(WORD32));
            AE_L8X4F_IP(d_inp3, pinp, sizeof(WORD32));
            d_inp = AE_SRAI16(d_inp, 8);
            d_inp1 = AE_SRAI16(d_inp1, 8);
            d_inp2 = AE_SRAI16(d_inp2, 8);
            d_inp3 = AE_SRAI16(d_inp3, 8);
            d_inp = AE_ADD16(d_inp, AE_MOVDA16(input_offset));
            d_inp1 = AE_ADD16(d_inp1,AE_MOVDA16(input_offset));
            d_inp2 = AE_ADD16(d_inp2,AE_MOVDA16(input_offset));
            d_inp3 = AE_ADD16(d_inp3,AE_MOVDA16(input_offset));
            for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
            {
              for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
              {
                // Compute output element location.
                int out_x = out_x_orig + filter_x;//out_x_origin + filter_x;
                int out_y = out_y_orig + filter_y;//out_y_origin + filter_y;
                ae_int64 *pscratch_src = (ae_int64*)&scratch_buffer[out_y*output_width*output_depth+out_x*output_depth];
                ae_int64 d_scr;
                WORD8* pfilt = (WORD8*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + in_channel];
                ae_int16x4 d_fil, d_fil1, d_fil2, d_fil3;

                AE_L8X4F_IP(d_fil, pfilt, sizeof(WORD32));
                AE_L8X4F_IP(d_fil1, pfilt, sizeof(WORD32));
                AE_L8X4F_IP(d_fil2, pfilt, sizeof(WORD32));
                AE_L8X4F_XP(d_fil3, pfilt, (stride1-12));

                for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                {
                  d_scr = AE_L64_I(pscratch_src, 0);
                  AE_MULAAAAQ16(d_scr, d_inp, d_fil);
                  AE_MULAAAAQ16(d_scr, d_inp1, d_fil1);
                  AE_MULAAAAQ16(d_scr, d_inp2, d_fil2);
                  AE_MULAAAAQ16(d_scr, d_inp3, d_fil3);

                  d_fil  = AE_L8X4F_I( pfilt, 0*sizeof(WORD32));
                  d_fil1 = AE_L8X4F_I( pfilt, 1*sizeof(WORD32));
                  d_fil2 = AE_L8X4F_I( pfilt, 2*sizeof(WORD32));
                  d_fil3 = AE_L8X4F_I( pfilt, 3*sizeof(WORD32));

                  pfilt += stride1;
                  AE_S64_IP(d_scr, pscratch_src, sizeof(WORD64));
                }
              }
            }
          }
        }
      }
    }    
  }
  else if(input_data && filter_data && scratch_buffer &&
      (((unsigned int)input_data&0x3)==0) && (((unsigned int)filter_data&0x3)==0) &&
      (((unsigned int)scratch_buffer&0x7) == 0) && ((input_depth&0x3)==0) && ((filter_height*filter_width&0x3)==0))
  {
    {
      //tbd : batch = 1, need to handle other values and in_x_min/max= 0 .. need toc heck for other values
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
          pinp =  (WORD8*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
          for (int in_channel = 0; in_channel < input_depth; in_channel+=4)
          {
            ae_int16x4 d_inp;
            AE_L8X4F_IP(d_inp, pinp, sizeof(WORD32));
            d_inp = AE_SRAI16(d_inp, 8);
            d_inp = AE_ADD16(d_inp, AE_MOVDA16(input_offset));
            for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
            {
              for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
              {
                // Compute output element location.
                int out_x = out_x_orig + filter_x;//out_x_origin + filter_x;
                int out_y = out_y_orig + filter_y;//out_y_origin + filter_y;
                ae_int64 *pscratch_src = (ae_int64*)&scratch_buffer[out_y*output_width*output_depth+out_x*output_depth];
                ae_int64 d_scr;
                WORD8* pfilt = (WORD8*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + in_channel];
                ae_int16x4 d_fil;

                AE_L8X4F_XP(d_fil, pfilt, stride1);

                for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                {
                  d_scr = AE_L64_I(pscratch_src, 0);
                  AE_MULAAAAQ16(d_scr, d_inp, d_fil);
                  AE_L8X4F_XP(d_fil, pfilt, stride1);
                  AE_S64_IP(d_scr, pscratch_src, sizeof(WORD64));
                }
              }
            }
          }
        }
      }
    }    
  }
  else
  {
    {
      for (int in_y = 0; in_y < input_height; ++in_y)
      {
        for (int in_x = 0; in_x < input_width; ++in_x)
        {
          for (int in_channel = 0; in_channel < input_depth; ++in_channel)
          {
            const int out_x_origin = (in_x * stride_width) - pad_width;
            const int out_y_origin = (in_y * stride_height) - pad_height;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y)
            {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x)
              {
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) && (out_y < output_height))
                {
                  for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                  {
                    const int32_t input_value = input_data[((in_y)*input_width+in_x)*input_depth+in_channel] + input_offset;
                    const int32_t filter_value = filter_data[(((out_channel*filter_height)+filter_y)*filter_width+filter_x)*input_depth+in_channel] << 8;
                    scratch_buffer[((out_y)*output_width+out_x)*output_depth+out_channel] += input_value * filter_value;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if(bias_data)
  {
    ae_int64 acc0, acc1;
    ae_int32x2 dbias;
    ae_int32 *pbias = (ae_int32*)bias_data;

    for (int out_channel = 0; out_channel < output_depth; ++out_channel)
    {
#if TFLITE_SINGLE_ROUNDING
      int left_shift = output_shift[out_channel];
      int right_shift = output_shift[out_channel];
      (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int left_shift = output_shift[out_channel] < 0 ? 0 : output_shift[out_channel];
      int right_shift = output_shift[out_channel] > 0 ? 0 : -output_shift[out_channel];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      pscratch = (ae_int64*)&scratch_buffer[out_channel];
      WORD8 *pout = (WORD8*)&output_data[out_channel];
      ae_int64 *pscratch1 = (ae_int64*)&scratch_buffer[((output_height*output_width)>>1)*output_depth+out_channel];
      WORD8 *pout1 = (WORD8*)&output_data[((output_height*output_width)>>1)*output_depth+out_channel];
      AE_L32_IP(dbias, pbias, sizeof(WORD32));
      AE_L64_XP(acc0, pscratch, output_depth*sizeof(WORD64));
      AE_L64_XP(acc1, pscratch1, output_depth*sizeof(WORD64));
      ae_int32 out_mult = output_multiplier[out_channel];
      for (int i = 0; i < ((output_height*output_width)>>1); i++)
      {
        ae_int32x2 out32;
        ae_int32x2 acc;
        acc0 = AE_SRAI64(acc0, 8);
        acc1 = AE_SRAI64(acc1, 8);           
        acc0 = AE_ADD64(acc0, AE_MOVINT64_FROMF32(dbias));
        acc1 = AE_ADD64(acc1, AE_MOVINT64_FROMF32(dbias));
        acc = AE_MOVDA32X2(AE_MOVINT32_FROMINT64(acc0), AE_MOVINT32_FROMINT64(acc1));
        MPY_BY_QUANT_MULT_X2_OUT32(out32, acc, out_mult, left_shift, right_shift);
        out32 = AE_ADD32(out32, AE_MOVDA32(output_offset));
        out32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(out32, AE_MOVDA32(-128)));
        AE_L64_XP(acc0, pscratch, output_depth*sizeof(WORD64));
        AE_L64_XP(acc1, pscratch1, output_depth*sizeof(WORD64));
        *pout = (WORD8)AE_MOVAD32_H(out32);
        pout+=output_depth;
        *pout1 = (WORD8)AE_MOVAD32_L(out32);
        pout1+=output_depth;
      }
      if((output_height*output_width) & 1)
      {
        ae_int16x4 out16;
        ae_int32x2 out1_32;
        acc1 = AE_SRAI64(acc1, 8);
        acc1 = AE_ADD64(acc1, AE_MOVINT64_FROMF32(dbias));   
        MPY_BY_QUANT_MULT_X2_OUT32(out1_32, AE_MOVDA32X2(AE_MOVINT32_FROMINT64(acc1), AE_MOVINT32_FROMINT64(acc1)), out_mult, left_shift, right_shift);
        out1_32 = AE_ADD32(out1_32, AE_MOVDA32(output_offset));
        out1_32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(out1_32, AE_MOVDA32(-128)));
        out16 = AE_SAT16X4(out1_32, out1_32);
        *pout1 = (WORD8)AE_MOVAD16_0(out16);
      }
    }
  }
  else
  {
    pscratch = (ae_int64*)scratch_buffer;
    WORD8 *pout = (WORD8*)output_data;
    for (int i = 0; i < output_height*output_width; i++)
    {   
      for (int out_channel = 0; out_channel < output_depth; ++out_channel)
      {
#if TFLITE_SINGLE_ROUNDING
        int left_shift = output_shift[out_channel];
        int right_shift = output_shift[out_channel];
        (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
        int left_shift = output_shift[out_channel] < 0 ? 0 : output_shift[out_channel];
        int right_shift = output_shift[out_channel] > 0 ? 0 : -output_shift[out_channel];
#endif /* #if TFLITE_SINGLE_ROUNDING */        
        ae_int64 acc;
        ae_int32x2 out0_32;
        ae_int16x4 out16;
        AE_L64_IP(acc, pscratch, sizeof(WORD64));
        acc = AE_SRAI64(acc, 8);   
        MPY_BY_QUANT_MULT_X2_OUT32(out0_32, AE_MOVDA32X2(AE_MOVINT32_FROMINT64(acc), AE_MOVINT32_FROMINT64(acc)), output_multiplier[out_channel], left_shift, right_shift);      
        out0_32 = AE_ADD32(out0_32, AE_MOVDA32(output_offset));
        out0_32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(out0_32, AE_MOVDA32(-128))); 
        out16 = AE_SAT16X4(out0_32, out0_32);  
        *pout++ = (WORD8)AE_MOVAD16_0(out16);
      }
    }   
  }
}
#endif

static inline void tconv2d_std_reorder_kernel_sym8s
    (pVOID p_scratch
     ,const WORD8* p_kernel
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

  WORD32 input_channels_pad = PADDED_SIZE(input_channels, ALIGNMENT >> 1);
  WORD32 pitch_d = input_channels;
  WORD32 pitch_w = kernel_width * input_channels;
  WORD32 pitch_h = kernel_height * kernel_width * input_channels;

  WORD32 subkermax_w = (kernel_width + x_stride - 1) / x_stride;
  WORD32 subkermax_h = (kernel_height + y_stride - 1) / y_stride;
  
  WORD8 *p_ker;

  /* Conversion from NDWH -> DNWH,                       */
  /* transposing of kernels and formation of sub-kernels */
  for (kIdy = 0; kIdy < y_stride; kIdy++)
  {
    for (kIdx = 0; kIdx < x_stride; kIdx++)
    {
      kernelIdx = kIdy * x_stride + kIdx;
      WORD8 *p_dst = ((WORD8 *)p_scratch + kernelIdx * subker_size);

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
            p_ker = (WORD8 *)&p_kernel[inIdx = outCh * pitch_h + ky * pitch_w + kx * pitch_d];
            memcpy(p_dst, p_ker, input_channels);
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
    const WORD32* __restrict__ p_bias,
    WORD8 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_offset,
    WORD32 idx_width,
    WORD32 idx_height)
{
  WORD32 i, j, k;
  /* When kernel has no valid input for convolution, output is just bias */
  for(i = idx_height; i < out_height; i++)
  {
    for(j = idx_width; j < out_width; j++)
    {
      WORD8 *ptrout = (WORD8*)&p_out[i * out_height_offset + j * out_width_offset];
      ae_int32 *pbias = (ae_int32*)p_bias;
      ae_int32x2 q1;
      for(k = 0; k < out_channels; k++)
      {
        if(p_bias != NULL){
          AE_L32_IP(q1, pbias, 4);
        }
        else{
          q1 = AE_MOVDA32(0);
        }
        ae_int32x2 acc;
        int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
        left_shift = right_shift = p_out_shift[k];
#if XCHAL_HAVE_HIFI1S
      left_shift = 31 - left_shift;
      left_shift = left_shift << 16 | left_shift;
#endif
        (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
        left_shift = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
#endif
#if (XCHAL_HAVE_HIFI1S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI1S(acc, q1, p_out_multiplier[k], left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc, q1, p_out_multiplier[k], left_shift, right_shift);
#endif        
        acc = AE_ADD32S(acc, AE_MOVDA32(out_offset));
        acc = AE_MIN32(AE_MOVDA32(127), AE_MAX32(acc, AE_MOVDA32(-128)));        
        *ptrout = (WORD8)AE_MOVAD32_H(acc);
        ptrout+=out_channels_offset;
      }
    }
  }
}

static inline void transpose_conv2d_std_sym8sxasym8s(WORD8* output_data,
    const WORD8* input_data,
    const WORD8* filter_data,
    const WORD32* bias_data,
    int stride_width, int stride_height,
    int pad_width, int pad_height,
    int input_depth, int output_depth,
    int input_height, int input_width,
    int filter_height, int filter_width,
    int output_height, int output_width,
    int num_elements,
    int input_offset, int output_offset,
    int *output_shift, int *output_multiplier,
    pVOID scratch_buffer)
{
  /* Transpose and Reorder the kernel into sub-kernels */
  WORD32 subkerX_max = (filter_width + stride_width - 1) / stride_width;
  WORD32 subkerY_max = (filter_height + stride_height - 1) / stride_height;
  WORD32 n_subker = stride_width * stride_height;
  WORD32 input_depth_pad = PADDED_SIZE(input_depth, (ALIGNMENT>>1));
  WORD32 subker_size = subkerX_max * subkerY_max * input_depth_pad * output_depth;
  /* memset the kernel reordering memory on scratch */
  memset(scratch_buffer, (WORD8)0, subker_size * n_subker);

  tconv2d_std_reorder_kernel_sym8s(scratch_buffer, filter_data, filter_height, filter_width, input_depth, output_depth, stride_width, stride_height, subker_size);

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
  WORD32 kernel_size = PADDED_SIZE(subker_size * n_subker, 4);
  WORD8 *p_trp_ker = (WORD8 *)scratch_buffer; 
  WORD8 *p_scr_cnv = (WORD8 *)((WORD8 *)scratch_buffer + kernel_size);

  /* Handle cases that have less valid output dimension than the output dimension given by the user */
  if(((orig_valid_out_h) < output_height))
  { 
    tconv_pad(output_width, output_height, output_depth, final_out_channels_offset, final_out_width_offset, final_out_height_offset, bias_data, output_data, output_multiplier, output_shift, output_offset, 0, XT_MAX(0,orig_valid_out_h));
  }
  if((orig_valid_out_w) < output_width)
  {
    tconv_pad(output_width, output_height, output_depth, final_out_channels_offset, final_out_width_offset, final_out_height_offset, bias_data, output_data, output_multiplier, output_shift, output_offset, XT_MAX(0,orig_valid_out_w), 0);
  }
  if((out_h_per_subker < 0))
  {
    tconv_pad(output_width, output_height, output_depth, final_out_channels_offset, final_out_width_offset, final_out_height_offset, bias_data, output_data, output_multiplier, output_shift, output_offset, 0, 0);
    return;
  }

  WORD32 j;
  WORD32 input_bytewidth = 1;
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
      ,PREC_ASYM8S);

  /* When kernel convolves over input region */
  // Initialize circular buffer
  conv2d_std_init_cir_buf_asym8(input_depth, input_depth_pad, input_bytewidth, input_width, input_height, y_pad, y_b_pad, x_pad, subkerX_max, 1, (VOID**)&pp_inp, p_state, -input_offset);

  // Index to padded input width
  WORD32 idx_beg_inp_width_pad = subkerX_max - 1;
  idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;

  WORD8 *po_tmp;
  WORD32 rem_val_out_w = valid_out_w % stride_width;
  WORD32 pad_w = pad_width;
  
  // Process Loop to compute one output plane [out_height x out_channels] per iteration
  WORD32 out_w_looopcnt = valid_out_w / stride_width;

  for(j = 0; j < out_w_looopcnt; j++)
  {
    // Add x_stride x (input_height x input_channels) new planes to circular buffer
    conv2d_std_update_cir_buf_asym8(input_depth, input_depth_pad, input_bytewidth, input_width, input_height, y_pad, y_b_pad, x_pad, subkerX_max, 1, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state, -input_offset);

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
          WORD8 *p_subkernel = ((WORD8 *)p_trp_ker + kernelIdx * subker_size);
          WORD32 rem_out_h_per_subker = (rem_val_out_h > 0) ? 1 : 0; 

          // Adjust the circ_buf pointer as per pad_height
          WORD32 cir_buf_inp_offset = pad_h_per_subker * input_depth_pad * subkerX_max;
          cir_buf_inp_offset = (pad_h_ky > 0) ? cir_buf_inp_offset : cir_buf_inp_offset + input_depth_pad * subkerX_max;
          WORD8 *p_inp_cir_buf = p_state->cir_buf.p_curr;
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_inp_cir_buf, cir_buf_inp_offset * input_bytewidth);        
          // Convolution using matXvec with matrix as circular buffer
          xa_nn_matXvec_sym8sxasym8s_asym8s_circ
          (po_tmp /* output */
           ,p_inp_cir_buf/* matrix: rows x cols */
           ,p_subkernel /* vec: cols */
           ,bias_data /* bias */
           ,out_h_per_subker + rem_out_h_per_subker /* rows */
           ,input_depth_pad * subkerX_max * subkerY_max /* cols */
           ,input_depth_pad * subkerX_max /* row_offset */
           ,output_depth /* vec_count */
           ,input_depth_pad * subkerX_max * subkerY_max /* vec_stride */
           ,out_channels_offset /* out_col_offset */
           ,final_out_height_offset * stride_height /* out_row_offset */
           ,input_offset
           ,output_multiplier
           ,output_shift
           ,output_offset
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
    conv2d_std_update_cir_buf_asym8(input_depth, input_depth_pad, input_bytewidth, input_width, input_height, y_pad, y_b_pad, x_pad, subkerX_max, 1, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state, -input_offset);

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
          WORD8 *p_subkernel = ((WORD8 *)p_trp_ker + kernelIdx * subker_size);
          WORD32 rem_out_h_per_subker = (rem_val_out_h > 0) ? 1 : 0; 
          // Adjust the circ_buf pointer as per pad_height
          WORD32 cir_buf_inp_offset = pad_h_per_subker * input_depth_pad * subkerX_max;
          cir_buf_inp_offset = (pad_h_ky > 0) ? cir_buf_inp_offset : cir_buf_inp_offset + input_depth_pad * subkerX_max;
          WORD8 *p_inp_cir_buf = p_state->cir_buf.p_curr;
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_inp_cir_buf, cir_buf_inp_offset * input_bytewidth);
          
          // Convolution using matXvec with matrix as circular buffer
          xa_nn_matXvec_sym8sxasym8s_asym8s_circ
          (po_tmp /* output */
           ,p_inp_cir_buf/* matrix: rows x cols */
           ,p_subkernel /* vec: cols */
           ,bias_data /* bias */
           ,out_h_per_subker + rem_out_h_per_subker /* rows */
           ,input_depth_pad * subkerX_max * subkerY_max /* cols */
           ,input_depth_pad * subkerX_max /* row_offset */
           ,output_depth /* vec_count */
           ,input_depth_pad * subkerX_max * subkerY_max /* vec_stride */
           ,out_channels_offset /* out_col_offset */
           ,final_out_height_offset * stride_height /* out_row_offset */
           ,input_offset
           ,output_multiplier
           ,output_shift
           ,output_offset
          );       
          po_tmp += final_out_height_offset;
        }
      }
      output_data = is_pad_w ? output_data : output_data + output_depth;
    }
  }  
}

int xa_nn_transpose_conv_sym8sxasym8s(WORD8* output_data,
    const WORD8* input_data,
    const WORD8* filter_data,
    const WORD32* bias_data,
    int stride_width, int stride_height,
    int pad_width, int pad_height,
    int input_depth, int output_depth,
    int input_height, int input_width,
    int filter_height, int filter_width,
    int output_height, int output_width,
    int num_elements,
    int input_offset, int output_offset,
    int *output_shift, int *output_multiplier,
    void* scratch_buffer)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(output_data, -1);
  XA_NNLIB_ARG_CHK_PTR(filter_data, -1);
  XA_NNLIB_ARG_CHK_PTR(input_data, -1);
  XA_NNLIB_ARG_CHK_PTR(scratch_buffer, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(output_data, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(filter_data, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(input_data, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(bias_data, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(scratch_buffer, sizeof(WORD64), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((filter_height <= 0 || filter_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((output_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((stride_height <= 0 || stride_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((pad_height < 0 || pad_width < 0), -1);
  XA_NNLIB_ARG_CHK_COND((output_height <= 0 || output_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((num_elements <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_offset < -127 || input_offset > 128), -1);
  XA_NNLIB_ARG_CHK_COND((output_offset < -128 || output_offset > 127), -1);

  int ker_grt_inp = (filter_width > input_width || filter_height > input_height);
  int str_leq_ker = (stride_width <= filter_width && stride_height <= filter_height);

  if(!ker_grt_inp && str_leq_ker)
  {
    transpose_conv2d_std_sym8sxasym8s(output_data, input_data, filter_data, bias_data,
      stride_width, stride_height, pad_width, pad_height, input_depth, output_depth,
      input_height, input_width, filter_height, filter_width, output_height, output_width,
      num_elements, input_offset, output_offset, output_shift, output_multiplier, scratch_buffer);
  }
  else
  {
#if XCHAL_HAVE_HIFI1S
    tconv2d_sym8sxasym8s(output_data, input_data, filter_data, bias_data,
      stride_width, stride_height, pad_width, pad_height, input_depth, output_depth,
      input_height, input_width, filter_height, filter_width, output_height, output_width,
      num_elements, input_offset, output_offset, output_shift, output_multiplier, (int32_t *)scratch_buffer);
#else    
    tconv2d_sym8sxasym8s(output_data, input_data, filter_data, bias_data,
      stride_width, stride_height, pad_width, pad_height, input_depth, output_depth,
      input_height, input_width, filter_height, filter_width, output_height, output_width,
      num_elements, input_offset, output_offset, output_shift, output_multiplier, (int64_t *)scratch_buffer);
#endif      
  }
  return 0;
}
