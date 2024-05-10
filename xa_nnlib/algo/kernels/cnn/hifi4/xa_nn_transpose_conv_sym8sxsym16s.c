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

#define MPY_BY_QUANT_MULT_ACC64_OUT32(out0, inp0, mult, l_shift) \
{ \
  ae_int32x2 d_red_mult = AE_SEXT32X2D16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult))); \
  ae_int32x2 d_red_mult_l16 = AE_CVT32X2F16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult)));  \
  ae_int32x2 d_inp0_h = AE_ROUND32F64SASYM(inp0); \
  ae_int64 q0_l; \
  q0_l = AE_MUL32_HH(d_red_mult, AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(inp0), AE_MOVINT32X2_FROMINT64(inp0))); \
  AE_MULAF32S_HH(q0_l, d_red_mult_l16, AE_SLAI32(d_inp0_h, 15)); \
  q0_l = AE_SLAA64(q0_l, (l_shift + 17)); \
  out0 = AE_ROUND32F64SASYM(q0_l); \
}

#if XCHAL_HAVE_HIFI1S
static inline ae_int32x2 MultiplyByQuantizedMultiplier_ref(ae_int64 d_x,
    int32_t quantized_multiplier,
    int shift){
    ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
    ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
    ae_int64 q = AE_MUL48X16_0(d_x, d_red_mul16);
    ae_int32x2 result = AE_ROUNDAV32X2F64SASYM (q, q, 15-shift);   // only lower 32 is valid result
    return result;
}

static inline ae_int32x2 MultiplyByQuantizedMultiplier_x2_opt(ae_int64 d_x1, ae_int64 d_x2,
    ae_int32x2 d_mul32,
    int shift) {
    ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_mul32, d_mul32);
    ae_int64 q1 = AE_MUL48X16_0(d_x1, d_red_mul16);
    ae_int64 q2 = AE_MUL48X16_0(d_x2, d_red_mul16);
    ae_int32x2 result = AE_ROUNDAV32X2F64SASYM (q1, q2, shift);  
    return result;
}
#else
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

static inline ae_int32x2 MultiplyByQuantizedMultiplier_x2_opt(ae_int64 d_x1, ae_int64 d_x2,
    ae_int32x2 d_red_mul32,
    int shift) {
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

static inline void tconv2d_sym8sxsym16s(WORD16* output_data,
    const WORD16* input_data,
    const WORD8* filter_data,
    const WORD64* bias_data,
    int stride_width, int stride_height,
    int pad_width, int pad_height,
    int input_depth, int output_depth,
    int input_height, int input_width,
    int filter_height, int filter_width,
    int output_height, int output_width,
    int num_elements,
    int *output_shift, int *output_multiplier,
    int64_t* scratch_buffer)
{
  ae_int64 *pscratch = (ae_int64*)scratch_buffer;
  ae_int64 dzero = AE_ZERO64();
  for(int i=0; i<num_elements; i++)
    AE_S64_IP(dzero, pscratch, 8);

  int stride1 = filter_height*filter_width*input_depth;
  WORD16 *pinp;

  /*
   * SEANet: special case for input_depth multiple of 16
   */
#if XCHAL_HAVE_HIFI1S
  if(input_data && filter_data && output_data && scratch_buffer &&
      (((unsigned int)input_data&0x7)==0) && (((unsigned int)filter_data&0x7)==0) && (((unsigned int)output_data&0x7) == 0) &&
      (((unsigned int)scratch_buffer&0x7) == 0) && ((input_depth&0xF)==0))
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
          pinp =  (WORD16*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
          int in_channel = 0;
          for (; in_channel + 15 < input_depth; in_channel+=16)
          {
            ae_int16x4 d_inp, d_inp1, d_inp2, d_inp3;
            AE_L16X4_IP(d_inp, (ae_int16x4*)pinp, sizeof(WORD64));
            AE_L16X4_IP(d_inp1, (ae_int16x4*)pinp, sizeof(WORD64));
            AE_L16X4_IP(d_inp2, (ae_int16x4*)pinp, sizeof(WORD64));
            AE_L16X4_IP(d_inp3, (ae_int16x4*)pinp, sizeof(WORD64));

            for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
            {
              for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
              {
                // Compute output element location.
                int out_x = out_x_orig + filter_x;
                int out_y = out_y_orig + filter_y;
                ae_int64 *pscratch_src = (ae_int64*)&scratch_buffer[out_y*output_width*output_depth+out_x*output_depth];
                ae_int64 d_scr;
                WORD8* pfilt = (WORD8*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + in_channel];
                ae_int8x8 d_fil, d_fil1;

                AE_L8X8_IP(d_fil, (ae_int8x8*)pfilt, 8);
                AE_L8X8_XP(d_fil1, (ae_int8x8*)pfilt, stride1-8);

                for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                {
                  d_scr = AE_L64_I(pscratch_src, 0);
                  AE_MULAO8X16 (d_scr, d_inp, d_inp1, d_fil);
                  AE_MULAO8X16 (d_scr, d_inp2, d_inp3, d_fil1);
                  AE_L8X8_IP(d_fil, (ae_int8x8*)pfilt, 8);
                  AE_L8X8_XP(d_fil1, (ae_int8x8*)pfilt, stride1-8);
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
#endif // XCHAL_HAVE_HIFI1S  
  if(input_data && filter_data && output_data && scratch_buffer &&
      (((unsigned int)input_data&0x7)==0) && (((unsigned int)filter_data&0x3)==0) && (((unsigned int)output_data&0x7) == 0) &&
      (((unsigned int)scratch_buffer&0x7) == 0) && ((input_depth&0x3)==0))
  {
#if XCHAL_HAVE_HIFI1S
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
          pinp =  (WORD16*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
          int in_channel = 0;
          for (; in_channel + 15 < input_depth; in_channel+=16)
          {
            ae_int16x4 d_inp, d_inp1, d_inp2, d_inp3;
            AE_L16X4_IP(d_inp, (ae_int16x4*)pinp, sizeof(WORD64));
            AE_L16X4_IP(d_inp1, (ae_int16x4*)pinp, sizeof(WORD64));
            AE_L16X4_IP(d_inp2, (ae_int16x4*)pinp, sizeof(WORD64));
            AE_L16X4_IP(d_inp3, (ae_int16x4*)pinp, sizeof(WORD64));

            for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
            {
              for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
              {
                // Compute output element location.
                int out_x = out_x_orig + filter_x;
                int out_y = out_y_orig + filter_y;
                ae_int64 *pscratch_src = (ae_int64*)&scratch_buffer[out_y*output_width*output_depth+out_x*output_depth];
                ae_int64 d_scr;
                WORD8* pfilt = (WORD8*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + in_channel];
                ae_int8x8 d_fil, d_fil1;
                ae_valign filter_align = AE_LA64_PP(pfilt);

                AE_LA8X8_IP(d_fil, filter_align, (ae_int8x8*)pfilt);
                AE_LA8X8_IP(d_fil1, filter_align, (ae_int8x8*)pfilt);
                pfilt += (stride1-16);

                for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                {
                  d_scr = AE_L64_I(pscratch_src, 0);
                  AE_MULAO8X16 (d_scr, d_inp, d_inp1, d_fil);
                  AE_MULAO8X16 (d_scr, d_inp2, d_inp3, d_fil1);
                  filter_align = AE_LA64_PP(pfilt);
                  AE_LA8X8_IP(d_fil, filter_align, (ae_int8x8*)pfilt);
                  AE_LA8X8_IP(d_fil1, filter_align, (ae_int8x8*)pfilt);
                  pfilt += (stride1-16);
                  AE_S64_IP(d_scr, pscratch_src, sizeof(WORD64));
                }
              }
            }
          }
          for (; in_channel + 3 < input_depth; in_channel+=4)
          {
            ae_int16x4 d_inp;
            AE_L16X4_IP(d_inp, (ae_int16x4*)pinp, sizeof(WORD64));

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

                AE_L8X4S_XP(d_fil, pfilt, stride1);

                for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                {
                  d_scr = AE_L64_I(pscratch_src, 0);
                  AE_MULAAAAQ16(d_scr, d_inp, d_fil);
                  AE_L8X4S_XP(d_fil, pfilt, stride1);
                  AE_S64_IP(d_scr, pscratch_src, sizeof(WORD64));
                }
              }
            }
          }
        }
      }
    }
#else	
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
          pinp =  (WORD16*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
          int in_channel = 0;
          for (; in_channel + 15 < input_depth; in_channel+=16)
          {
            ae_int16x4 d_inp, d_inp1, d_inp2, d_inp3;
            AE_L16X4_IP(d_inp, (ae_int16x4*)pinp, sizeof(WORD64));
            AE_L16X4_IP(d_inp1, (ae_int16x4*)pinp, sizeof(WORD64));
            AE_L16X4_IP(d_inp2, (ae_int16x4*)pinp, sizeof(WORD64));
            AE_L16X4_IP(d_inp3, (ae_int16x4*)pinp, sizeof(WORD64));

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
#if XCHAL_HAVE_HIFI1
                AE_L8X4S_IP(d_fil, pfilt, sizeof(WORD32));
                AE_L8X4S_IP(d_fil1, pfilt, sizeof(WORD32));
                AE_L8X4S_IP(d_fil2, pfilt, sizeof(WORD32));
                AE_L8X4S_XP(d_fil3, pfilt, (stride1-12));
#else
                AE_L8X4F_IP(d_fil, pfilt, sizeof(WORD32));
                AE_L8X4F_IP(d_fil1, pfilt, sizeof(WORD32));
                AE_L8X4F_IP(d_fil2, pfilt, sizeof(WORD32));
                AE_L8X4F_XP(d_fil3, pfilt, (stride1-12));
#endif  

                for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                {
                  d_scr = AE_L64_I(pscratch_src, 0);
                  AE_MULAAAAQ16(d_scr, d_inp, d_fil);
                  AE_MULAAAAQ16(d_scr, d_inp1, d_fil1);
                  AE_MULAAAAQ16(d_scr, d_inp2, d_fil2);
                  AE_MULAAAAQ16(d_scr, d_inp3, d_fil3);
#if XCHAL_HAVE_HIFI1
                  d_fil  = AE_L8X4S_I( pfilt, 0*sizeof(WORD32));
                  d_fil1 = AE_L8X4S_I( pfilt, 1*sizeof(WORD32));
                  d_fil2 = AE_L8X4S_I( pfilt, 2*sizeof(WORD32));
                  d_fil3 = AE_L8X4S_I( pfilt, 3*sizeof(WORD32));
#else
                  d_fil  = AE_L8X4F_I( pfilt, 0*sizeof(WORD32));
                  d_fil1 = AE_L8X4F_I( pfilt, 1*sizeof(WORD32));
                  d_fil2 = AE_L8X4F_I( pfilt, 2*sizeof(WORD32));
                  d_fil3 = AE_L8X4F_I( pfilt, 3*sizeof(WORD32));
#endif
                  pfilt += stride1;
                  AE_S64_IP(d_scr, pscratch_src, sizeof(WORD64));
                }
              }
            }
          }
          for (; in_channel + 3 < input_depth; in_channel+=4)
          {
            ae_int16x4 d_inp;
            AE_L16X4_IP(d_inp, (ae_int16x4*)pinp, sizeof(WORD64));

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
#if XCHAL_HAVE_HIFI1
                AE_L8X4S_XP(d_fil, pfilt, stride1);
#else
                AE_L8X4F_XP(d_fil, pfilt, stride1);
#endif
                for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                {
                  d_scr = AE_L64_I(pscratch_src, 0);
                  AE_MULAAAAQ16(d_scr, d_inp, d_fil);
#if XCHAL_HAVE_HIFI1
                  AE_L8X4S_XP(d_fil, pfilt, stride1);
#else
                  AE_L8X4F_XP(d_fil, pfilt, stride1);
#endif
                  AE_S64_IP(d_scr, pscratch_src, sizeof(WORD64));
                }
              }
            }
          }
        }
      }
    }
#endif	
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
                    const int32_t input_value = input_data[((in_y)*input_width+in_x)*input_depth+in_channel];
#if XCHAL_HAVE_HIFI1
                    const int32_t filter_value = filter_data[(((out_channel*filter_height)+filter_y)*filter_width+filter_x)*input_depth+in_channel];
#else
                    const int32_t filter_value = filter_data[(((out_channel*filter_height)+filter_y)*filter_width+filter_x)*input_depth+in_channel]<<8;
#endif
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
    ae_int32x2 dmul;
    ae_int64 acc, acc1, dbias;
    ae_int64 *pbias = (ae_int64*)bias_data;
    ae_int32 *pout_multiplier = (ae_int32*)output_multiplier;

    for (int out_channel = 0; out_channel < output_depth; ++out_channel)
    {
      int shift = output_shift[out_channel];
#if XCHAL_HAVE_HIFI1S  
      shift = 15 - shift;
      shift = (shift << 16) | (shift); 
#endif
      pscratch = (ae_int64*)&scratch_buffer[out_channel];
      ae_int16 *pout = (ae_int16*)&output_data[out_channel];
      ae_int64 *pscratch1 = (ae_int64*)&scratch_buffer[((output_height*output_width)>>1)*output_depth+out_channel];
      ae_int16 *pout1 = (ae_int16*)&output_data[((output_height*output_width)>>1)*output_depth+out_channel];
      AE_L64_IP(dbias, pbias, sizeof(WORD64));
      AE_L32_IP(dmul, pout_multiplier, sizeof(WORD32));
#if !XCHAL_HAVE_HIFI1S      
      ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(dmul, dmul);
      ae_int32x2 d_red_mul32 = AE_SEXT32X2D16_32(d_red_mul16);
#endif      
      AE_L64_XP(acc, pscratch, output_depth*sizeof(WORD64));
      AE_L64_XP(acc1, pscratch1, output_depth*sizeof(WORD64));
      for (int i = 0; i < ((output_height*output_width)>>1); i++)
      {
#if !XCHAL_HAVE_HIFI1
        acc = AE_SRAI64(acc, 8);
        acc1 = AE_SRAI64(acc1, 8);
#endif
        acc = AE_ADD64(acc, dbias);
        acc1 = AE_ADD64(acc1, dbias);
#if XCHAL_HAVE_HIFI1S        
        ae_int32x2 scaled_acc = MultiplyByQuantizedMultiplier_x2_opt(acc, acc1, dmul, shift);
#else
        ae_int32x2 scaled_acc = MultiplyByQuantizedMultiplier_x2_opt(acc, acc1, d_red_mul32, shift);
#endif        
        ae_int16x4 d1 = AE_SAT16X4(scaled_acc, scaled_acc);
        AE_L64_XP(acc, pscratch, output_depth*sizeof(WORD64));
        AE_L64_XP(acc1, pscratch1, output_depth*sizeof(WORD64));
        AE_S16_0_XP(AE_SEL16_4321(d1, d1), pout, output_depth*sizeof(WORD16));
        AE_S16_0_XP(d1, pout1, output_depth*sizeof(WORD16));
      }
      if((output_height*output_width) & 1)
      {
#if !XCHAL_HAVE_HIFI1
        acc1 = AE_SRAI64(acc1, 8);
#endif
        acc1 = AE_ADD64(acc1, dbias);
#if XCHAL_HAVE_HIFI1S          
        ae_int32x2 scaled_acc = MultiplyByQuantizedMultiplier_x2_opt(acc1, acc1, dmul, shift);
#else
        ae_int32x2 scaled_acc = MultiplyByQuantizedMultiplier_x2_opt(acc1, acc1, d_red_mul32, shift);
#endif        
        ae_int16x4 d1 = AE_SAT16X4(scaled_acc, scaled_acc);
        AE_S16_0_I(d1, pout1, 0);
      }
    }
  }
  else
  {
    pscratch = (ae_int64*)scratch_buffer;
    ae_int16 *pout = (ae_int16*)output_data;
    for (int i = 0; i < output_height*output_width; i++)
    {
      for (int out_channel = 0; out_channel < output_depth; ++out_channel)
      {
        ae_int64 acc;
        AE_L64_IP(acc, pscratch, sizeof(WORD64));
#if !XCHAL_HAVE_HIFI1
        acc = AE_SRAI64(acc, 8);
#endif
        ae_int32x2 scaled_acc = MultiplyByQuantizedMultiplier_ref(acc, output_multiplier[out_channel], output_shift[out_channel]);
        ae_int16x4 d1 = AE_SAT16X4(scaled_acc, scaled_acc);
        AE_S16_0_IP(d1, pout, sizeof(WORD16));
      }
    }
  }
}

/* Handle sub-kernel formation and transpose */
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
            xa_nn_memcpy(p_dst, p_ker, input_channels);
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
    const WORD64* __restrict__ p_bias,
    WORD16 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 idx_width,
    WORD32 idx_height)
{
  WORD32 i, j, k;
  ae_int16x4 d1;

  /* When kernel has no valid input for convolution, output is just bias */
  for(i = idx_height; i < out_height; i++)
  {
    for(j = idx_width; j < out_width; j++)
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
        ae_int32x2 acc;
        MPY_BY_QUANT_MULT_ACC64_OUT32(acc, q1, p_out_multiplier[k], p_out_shift[k]);
        d1 = AE_SAT16X4(acc, acc);
        AE_S16_0_XP(d1, ptrout, out_channels_offset*sizeof(WORD16));
      }
    }
  }
}

static inline void transpose_conv2d_std_sym8sxsym16s(WORD16* output_data,
    const WORD16* input_data,
    const WORD8* filter_data,
    const WORD64* bias_data,
    int stride_width, int stride_height,
    int pad_width, int pad_height,
    int input_depth, int output_depth,
    int input_height, int input_width,
    int filter_height, int filter_width,
    int output_height, int output_width,
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
  WORD32 kernel_size = PADDED_SIZE(subker_size * n_subker, 8);
  WORD8 *p_trp_ker = (WORD8 *)scratch_buffer; 
  WORD16 *p_scr_cnv = (WORD16 *)((WORD8 *)scratch_buffer + kernel_size);

  /* Handle cases that have less valid output dimension than the output dimension given by the user */
  if(((orig_valid_out_h) < output_height))
  {
    tconv_pad(output_width, output_height, output_depth, final_out_channels_offset, final_out_width_offset, final_out_height_offset, bias_data, output_data, output_multiplier, output_shift, 0, XT_MAX(0,orig_valid_out_h));
  }
  if((orig_valid_out_w) < output_width)
  {
    tconv_pad(output_width, output_height, output_depth, final_out_channels_offset, final_out_width_offset, final_out_height_offset, bias_data, output_data, output_multiplier, output_shift, XT_MAX(0,orig_valid_out_w), 0);
  }
  if((out_h_per_subker < 0))
  {
  tconv_pad(output_width, output_height, output_depth, final_out_channels_offset, final_out_width_offset, final_out_height_offset, bias_data, output_data, output_multiplier, output_shift, 0, 0);
  return;
  }

  WORD32 j;
  WORD32 input_bytewidth = 2;
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
      ,PREC_SYM16S);

  /* When kernel convolves over input region */
  // Initialize circular buffer
  conv2d_std_init_cir_buf(input_depth, input_depth_pad, input_bytewidth, input_width, input_height, y_pad, y_b_pad, x_pad, subkerX_max, 1, (VOID**)&pp_inp, p_state);

  // Index to padded input width
  WORD32 idx_beg_inp_width_pad = subkerX_max - 1;
  idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;

  WORD16 *po_tmp;
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
          WORD8 *p_subkernel = ((WORD8 *)p_trp_ker + kernelIdx * subker_size);
          WORD32 rem_out_h_per_subker = (rem_val_out_h > 0) ? 1 : 0; 

          // Adjust the circ_buf pointer as per pad_height
          WORD32 cir_buf_inp_offset = pad_h_per_subker * input_depth_pad * subkerX_max;
          cir_buf_inp_offset = (pad_h_ky > 0) ? cir_buf_inp_offset : cir_buf_inp_offset + input_depth_pad * subkerX_max;
          WORD16 *p_inp_cir_buf = p_state->cir_buf.p_curr;
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_inp_cir_buf, cir_buf_inp_offset * input_bytewidth);        
          // Convolution using matXvec with matrix as circular buffer
          xa_nn_matXvec_sym8sxsym16s_sym16s_circ
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
           ,output_multiplier
           ,output_shift
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
          WORD8 *p_subkernel = ((WORD8 *)p_trp_ker + kernelIdx * subker_size);
          WORD32 rem_out_h_per_subker = (rem_val_out_h > 0) ? 1 : 0; 
          // Adjust the circ_buf pointer as per pad_height
          WORD32 cir_buf_inp_offset = pad_h_per_subker * input_depth_pad * subkerX_max;
          cir_buf_inp_offset = (pad_h_ky > 0) ? cir_buf_inp_offset : cir_buf_inp_offset + input_depth_pad * subkerX_max;
          WORD16 *p_inp_cir_buf = p_state->cir_buf.p_curr;
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_inp_cir_buf, cir_buf_inp_offset * input_bytewidth);
          
          // Convolution using matXvec with matrix as circular buffer
          xa_nn_matXvec_sym8sxsym16s_sym16s_circ
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
           ,output_multiplier
           ,output_shift
          );
          po_tmp += final_out_height_offset;
        }
      }
      output_data = is_pad_w ? output_data : output_data + output_depth;
    }
  }
}

int xa_nn_transpose_conv_sym8sxsym16s(WORD16* output_data,
    const WORD16* input_data,
    const WORD8* filter_data,
    const WORD64* bias_data,
    int stride_width, int stride_height,
    int pad_width, int pad_height,
    int input_depth, int output_depth,
    int input_height, int input_width,
    int filter_height, int filter_width,
    int output_height, int output_width,
    int num_elements,
    int *output_shift, int *output_multiplier,
    void* scratch_buffer)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(output_data, -1);
  XA_NNLIB_ARG_CHK_PTR(filter_data, -1);
  XA_NNLIB_ARG_CHK_PTR(input_data, -1);
  XA_NNLIB_ARG_CHK_PTR(scratch_buffer, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(output_data, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(filter_data, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(input_data, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(bias_data, sizeof(WORD64), -1);
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

  int ker_grt_inp = (filter_width > input_width || filter_height > input_height);
  int str_leq_ker = (stride_width <= filter_width && stride_height <= filter_height);

  if(!ker_grt_inp && str_leq_ker)
  {
    transpose_conv2d_std_sym8sxsym16s(output_data, input_data, filter_data, bias_data,
    stride_width, stride_height, pad_width, pad_height, input_depth, output_depth,
    input_height, input_width, filter_height, filter_width,  output_height, output_width,
    output_shift, output_multiplier, scratch_buffer);
  }
  else
  {
    tconv2d_sym8sxsym16s(output_data, input_data, filter_data, bias_data,
    stride_width, stride_height, pad_width, pad_height, input_depth, output_depth,
    input_height, input_width, filter_height, filter_width,  output_height, output_width,
    num_elements, output_shift, output_multiplier, scratch_buffer);
  }

  return 0;
}

