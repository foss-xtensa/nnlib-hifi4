/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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
#include "common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros.h"

WORD32 xa_nn_transpose_conv_getsize
(WORD32 output_height
 ,WORD32 output_width
 ,WORD32 output_channels
 ,WORD32 output_precision
 )
{
    XA_NNLIB_CHK_COND((output_height <= 0), -1);
    XA_NNLIB_CHK_COND((output_width <= 0), -1);
    XA_NNLIB_CHK_COND((output_channels <= 0), -1);

    WORD32 scratch_bytewidth = 0;
    WORD32 total_size = 0;

    switch (output_precision)
    {
        case -8: /* For sym16s */
            scratch_bytewidth = 8; /* 64b scratch */
            break;
        default:
            return -1; /* Retunrning due to invalid input */
            break;
    }

    total_size = (output_height) * (output_width) * (output_channels) * (scratch_bytewidth);
    return total_size;
}

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
		int64_t* scratch_buffer)
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

	ae_int64 *pscratch = (ae_int64*)scratch_buffer;
	ae_int64 dzero = AE_ZERO64();
	for(int i=0; i<num_elements; i++)
		AE_S64_IP(dzero, pscratch, 8);

	int stride1 = filter_height*filter_width*input_depth;
	WORD16 *pinp;

	/*
	 * SEANet: special case for input_depth multiple of 16
	 */
	if(input_data && filter_data && output_data && scratch_buffer &&
			(((unsigned int)input_data&0x7)==0) && (((unsigned int)filter_data&0x3)==0) && (((unsigned int)output_data&0x7) == 0) &&
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
					pinp =  (WORD16*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
					for (int in_channel = 0; in_channel < input_depth; in_channel+=16)
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
									AE_L8X4F_IP(d_fil, pfilt, sizeof(WORD32));
									AE_L8X4F_IP(d_fil1, pfilt, sizeof(WORD32));
									AE_L8X4F_IP(d_fil2, pfilt, sizeof(WORD32));
									AE_L8X4F_XP(d_fil3, pfilt, (stride1-12));
									AE_S64_IP(d_scr, pscratch_src, sizeof(WORD64));
								}
							}
						}
					}
				}
			}
		}
	}
	else if(input_data && filter_data && output_data && scratch_buffer &&
			(((unsigned int)input_data&0x7)==0) && (((unsigned int)filter_data&0x3)==0) && (((unsigned int)output_data&0x7) == 0) &&
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
					pinp =  (WORD16*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
					for (int in_channel = 0; in_channel < input_depth; in_channel+=4)
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
										const int32_t input_value = input_data[((in_y)*input_width+in_x)*input_depth+in_channel];
										const int32_t filter_value = filter_data[(((out_channel*filter_height)+filter_y)*filter_width+filter_x)*input_depth+in_channel]<<8;
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
			ae_int64 *pscratch = (ae_int64*)&scratch_buffer[out_channel];
			ae_int16 *pout = (ae_int16*)&output_data[out_channel];
			ae_int64 *pscratch1 = (ae_int64*)&scratch_buffer[((output_height*output_width)>>1)*output_depth+out_channel];
			ae_int16 *pout1 = (ae_int16*)&output_data[((output_height*output_width)>>1)*output_depth+out_channel];
			AE_L64_IP(dbias, pbias, sizeof(WORD64));
			AE_L32_IP(dmul, pout_multiplier, sizeof(WORD32));
			ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(dmul, dmul);
			ae_int32x2 d_red_mul32 = AE_SEXT32X2D16_32(d_red_mul16);
			AE_L64_XP(acc, pscratch, output_depth*sizeof(WORD64));
			AE_L64_XP(acc1, pscratch1, output_depth*sizeof(WORD64));
			for (int i = 0; i < ((output_height*output_width)>>1); i++)
			{
				acc = AE_SRAI64(acc, 8);
				acc1 = AE_SRAI64(acc1, 8);
				acc = AE_ADD64(acc, dbias);
				acc1 = AE_ADD64(acc1, dbias);
				ae_int32x2 scaled_acc = MultiplyByQuantizedMultiplier_x2_opt(acc, acc1, d_red_mul32, shift);
				ae_int16x4 d1 = AE_SAT16X4(scaled_acc, scaled_acc);
				AE_L64_XP(acc, pscratch, output_depth*sizeof(WORD64));
				AE_L64_XP(acc1, pscratch1, output_depth*sizeof(WORD64));
				AE_S16_0_XP(AE_SEL16_4321(d1, d1), pout, output_depth*sizeof(WORD16));
				AE_S16_0_XP(d1, pout1, output_depth*sizeof(WORD16));
			}
			if((output_height*output_width) & 1)
			{
				acc1 = AE_SRAI64(acc1, 8);
				acc1 = AE_ADD64(acc1, dbias);
				ae_int32x2 scaled_acc = MultiplyByQuantizedMultiplier_x2_opt(acc1, acc1, d_red_mul32, shift);
				ae_int16x4 d1 = AE_SAT16X4(scaled_acc, scaled_acc);
				AE_S16_0_I(d1, pout1, 0);
			}
		}
	}
	else
	{
		ae_int64 *pscratch = (ae_int64*)scratch_buffer;
		ae_int16 *pout = (ae_int16*)output_data;
		for (int i = 0; i < output_height*output_width; i++)
		{
			for (int out_channel = 0; out_channel < output_depth; ++out_channel)
			{
				ae_int64 acc;
				AE_L64_IP(acc, pscratch, sizeof(WORD64));
				acc = AE_SRAI64(acc, 8);
				ae_int32x2 scaled_acc = MultiplyByQuantizedMultiplier_ref(acc, output_multiplier[out_channel], output_shift[out_channel]);
				ae_int16x4 d1 = AE_SAT16X4(scaled_acc, scaled_acc);
				AE_S16_0_IP(d1, pout, sizeof(WORD16));
			}
		}
	}

	return 0;
}
