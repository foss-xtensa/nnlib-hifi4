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

#if XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= RI9_HWVERSION)
static inline WORD32 xa_nn_memset_16_16( void *p_dst,
		WORD32 val,
		WORD32 n)
{
	WORD32 MEMCPY_16b_num_elements = n >> 1;
	ae_int16x4 d_inp0;
	d_inp0 = (WORD16)val;
	ae_int16 * __restrict__ pdest = (ae_int16 *)p_dst;
	ae_valign MEMCPY_16b_d_align = AE_ZALIGN64();
	for (int ii=0; ii < (MEMCPY_16b_num_elements >> 2); ii++) {
		AE_SA16X4_IP(d_inp0, MEMCPY_16b_d_align, (ae_int16x4 *)pdest);
	}
	AE_SAV16X4_XP(d_inp0, MEMCPY_16b_d_align, (ae_int16x4 *)pdest, ((MEMCPY_16b_num_elements&3)<<1));
	AE_SA64POS_FP(MEMCPY_16b_d_align, pdest);
	
	return 0;
}

static inline WORD32 xa_nn_memmove_16_16( void *p_dst,
		const void *p_src,
		WORD32 n)
{
	WORD32 MEMCPY_16b_num_elements = n >> 1;
	ae_int16x4 d_inp0;
	ae_int16 * __restrict__ psrc  = (ae_int16 *)p_src;
	ae_int16 * __restrict__ pdest = (ae_int16 *)p_dst;
	ae_valign MEMCPY_16b_s_align = AE_LA64_PP((ae_int16x4 *)psrc);
	ae_valign MEMCPY_16b_d_align = AE_ZALIGN64();
	for (int ii=0; ii < (MEMCPY_16b_num_elements >> 2); ii++) {
		AE_LA16X4_IP(d_inp0, MEMCPY_16b_s_align, (ae_int16x4 *)psrc);
		AE_SA16X4_IP(d_inp0, MEMCPY_16b_d_align, (ae_int16x4 *)pdest);
	}
	AE_LAV16X4_XP(d_inp0, MEMCPY_16b_s_align, (ae_int16x4 *)psrc, ((MEMCPY_16b_num_elements&3)<<1));
	AE_SAV16X4_XP(d_inp0, MEMCPY_16b_d_align, (ae_int16x4 *)pdest, ((MEMCPY_16b_num_elements&3)<<1));
	AE_SA64POS_FP(MEMCPY_16b_d_align, pdest);
	
	return 0;
}
#else
static inline WORD32 xa_nn_memset_16_16( void *p_dst,
		WORD32 val,
		WORD32 n)
{
	WORD32 MEMCPY_16b_num_elements = n >> 1;
	ae_int16x4 d_inp0, d_inp1;
	d_inp0 = (WORD16)val;
	d_inp1 = (WORD16)val;
	ae_int16 * __restrict__ pdest = (ae_int16 *)p_dst;
	ae_valign MEMCPY_16b_d_align = AE_ZALIGN64();
	for (int ii=0; ii < (MEMCPY_16b_num_elements >> 3); ii++) {
		AE_SA16X4_IP(d_inp0, MEMCPY_16b_d_align, (ae_int16x4 *)pdest);
		AE_SA16X4_IP(d_inp1, MEMCPY_16b_d_align, (ae_int16x4 *)pdest);
	}
	AE_SA64POS_FP(MEMCPY_16b_d_align, pdest);
	for (int ii = 0; ii<(MEMCPY_16b_num_elements&7); ii++) {
		AE_S16_0_IP(d_inp0, (ae_int16 *)pdest, sizeof(ae_int16));
	}
	return 0;
}

static inline WORD32 xa_nn_memmove_16_16( void *p_dst,
		const void *p_src,
		WORD32 n)
{
	WORD32 MEMCPY_16b_num_elements = n >> 1;
	ae_int16x4 d_inp0, d_inp1;
	ae_int16 * __restrict__ psrc  = (ae_int16 *)p_src;
	ae_int16 * __restrict__ pdest = (ae_int16 *)p_dst;
	ae_valign MEMCPY_16b_s_align = AE_LA64_PP((ae_int16x4 *)psrc);
	ae_valign MEMCPY_16b_d_align = AE_ZALIGN64();
	for (int ii=0; ii < (MEMCPY_16b_num_elements >> 3); ii++) {
		AE_LA16X4_IP(d_inp0, MEMCPY_16b_s_align, (ae_int16x4 *)psrc);
		AE_LA16X4_IP(d_inp1, MEMCPY_16b_s_align, (ae_int16x4 *)psrc);
		AE_SA16X4_IP(d_inp0, MEMCPY_16b_d_align, (ae_int16x4 *)pdest);
		AE_SA16X4_IP(d_inp1, MEMCPY_16b_d_align, (ae_int16x4 *)pdest);
	}
	AE_SA64POS_FP(MEMCPY_16b_d_align, pdest);
	for (int ii = 0; ii<(MEMCPY_16b_num_elements&7); ii++) {
		AE_L16_IP(d_inp0, (ae_int16 *)psrc, sizeof(ae_int16));
		AE_S16_0_IP(d_inp0, (ae_int16 *)pdest, sizeof(ae_int16));
	}
	return 0;
}
#endif

/*
 * Currently only supports upto 4D input tensors.
 * 1/2/3 D input tensors will be scaled up to 4D.
 * For example, 2x3 -> 1x1x2x3.
 */
WORD32 xa_nn_pad_16_16(WORD16 * __restrict__ p_out
		,const WORD32 *const p_out_shape
		,const WORD16 * __restrict__ p_inp
		,const WORD32 *const p_inp_shape
		,const WORD32 * __restrict__ p_pad_values
		,const WORD32 *const p_pad_shape
		,WORD32 num_out_dims
		,WORD32 num_inp_dims
		,WORD32 num_pad_dims
		,WORD32 pad_value)
{
	/* NULL pointer checks */
	XA_NNLIB_ARG_CHK_PTR(p_out, -1);
	XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
	XA_NNLIB_ARG_CHK_PTR(p_pad_values, -1);
	XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
	XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);
	XA_NNLIB_ARG_CHK_PTR(p_pad_shape, -1);

	/* Invalid input checks */
	XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > 4)), -1);
	XA_NNLIB_ARG_CHK_COND((num_out_dims != num_inp_dims), -1);
	XA_NNLIB_ARG_CHK_COND(((num_pad_dims < 0) || (num_pad_dims > 4)), -1);

	int itr = 0;
	for(itr=0; itr < num_inp_dims; itr++)
	{
		XA_NNLIB_ARG_CHK_COND((p_inp_shape[itr] <= 0), -1);
		XA_NNLIB_ARG_CHK_COND((p_out_shape[itr] <= 0), -1);
	}

	int pad_length = 1;
	for(itr=0; itr < num_pad_dims; itr++)
	{
		XA_NNLIB_ARG_CHK_COND((p_pad_shape[itr] <= 0), -1);
		pad_length *= p_pad_shape[itr];
	}
	XA_NNLIB_ARG_CHK_COND((pad_length != (2*num_out_dims)), -1);

	/* Output shape provided must be correctly computed based on input
	 * shape and pad values */
	/* Also, pad values should not be less than zero */
	for(itr=0; itr < num_out_dims; itr++)
	{
		int output_dim = p_out_shape[itr];
		int expected_dim = p_inp_shape[itr] + p_pad_values[2*itr] + p_pad_values[2*itr + 1];
		XA_NNLIB_ARG_CHK_COND(((output_dim != expected_dim) || (p_pad_values[2*itr] < 0) || (p_pad_values[2*itr + 1] < 0)), -1);
	}

	XA_NNLIB_ARG_CHK_COND(((pad_value < -32768) || (pad_value > 32767)), -1);

	/* Pointer alignment checks */
	XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
	XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
	XA_NNLIB_ARG_CHK_ALIGN(p_pad_values, sizeof(WORD32), -1);
	XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
	XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);
	XA_NNLIB_ARG_CHK_ALIGN(p_pad_shape, sizeof(WORD32), -1);

	/* Promoting lesser dim tensors to 4D tensors. Also shifting
	   padding values accordingly */
	int p_4D_inp_shape[4] = {1, 1, 1, 1};
	int p_4D_out_shape[4] = {1, 1, 1, 1};
	int p_pad_values_shifted[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	itr = num_inp_dims - 1;
	int count = 3;
#pragma loop_count max=4
#pragma loop_count min=1
	while(itr >= 0)
	{
		p_4D_inp_shape[count] = p_inp_shape[itr];
		p_4D_out_shape[count] = p_out_shape[itr];
		p_pad_values_shifted[2*count + 1] = p_pad_values[2*itr + 1];
		p_pad_values_shifted[2*count] = p_pad_values[2*itr];
		itr--;
		count--;
	}

	int output_batch  = p_4D_out_shape[0];
	int output_height = p_4D_out_shape[1];
	int output_width  = p_4D_out_shape[2];
	int output_depth  = p_4D_out_shape[3];

	int input_batch  = p_4D_inp_shape[0];
	int input_height = p_4D_inp_shape[1];
	int input_width  = p_4D_inp_shape[2];
	int input_depth  = p_4D_inp_shape[3];

	int left_b_padding = p_pad_values_shifted[0];
	int left_h_padding = p_pad_values_shifted[2];
	int left_w_padding = p_pad_values_shifted[4];
	int left_d_padding = p_pad_values_shifted[6];

	int right_b_padding = p_pad_values_shifted[1];
	int right_h_padding = p_pad_values_shifted[3];
	int right_w_padding = p_pad_values_shifted[5];
	int right_d_padding = p_pad_values_shifted[7];

	WORD16 *p_in = (WORD16 *)(p_inp);

	int output_index = (left_b_padding * output_height * output_width * output_depth);
	int input_index = 0;

	int itr_b, itr_h, itr_w;

	/* If only the outermost dimension needs padding, then just copy the (inner) sub-matrix and pad the top and bottom of output memory */
	if( !left_d_padding && !right_d_padding && !left_w_padding && !right_w_padding && !left_h_padding && !right_h_padding)
	{
		size_t sub_matrix_size = input_depth*input_width*input_height*sizeof(WORD16);

		xa_nn_memmove_16_16((void*)&p_out[left_b_padding*(sub_matrix_size>>1)], (void*)&p_in[input_index], input_batch * sub_matrix_size);

		if(left_b_padding){
			xa_nn_memset_16_16(p_out, pad_value, left_b_padding * sub_matrix_size);
		}

		if(right_b_padding){
			xa_nn_memset_16_16(&p_out[(left_b_padding+input_batch)*(sub_matrix_size>>1)], pad_value, right_b_padding * sub_matrix_size);
		}
	}
	else
	{
		xa_nn_memset_16_16(p_out, pad_value, (output_batch * output_height * output_width * output_depth * sizeof(WORD16)));

		if((!left_d_padding) && (!right_d_padding) && (!left_w_padding) && (!right_w_padding))
		{
			for(itr_b=0; itr_b<input_batch; itr_b++)
			{
				output_index = output_index + (left_h_padding * output_width * output_depth);
				xa_nn_memmove_16_16((void*)&p_out[output_index], (void*)&p_in[input_index], (input_depth*input_width*input_height*sizeof(WORD16)));
				output_index = output_index + (input_depth * input_width *input_height) + (right_h_padding * output_width * output_depth);
				input_index = input_index + (input_depth * input_width *input_height);
			}
		}
		else if((!left_d_padding) && (!right_d_padding))
		{
			for(itr_b=0; itr_b<input_batch; itr_b++)
			{
				output_index = output_index + (left_h_padding * output_width * output_depth);
				for(itr_h=0; itr_h<input_height; itr_h++)
				{
					output_index = output_index + (left_w_padding * output_depth);
					xa_nn_memmove_16_16((void*)&p_out[output_index], (void*)&p_in[input_index], (input_depth*input_width*sizeof(WORD16)));
					output_index = output_index + (input_depth * input_width) + (right_w_padding * output_depth);
					input_index = input_index + (input_depth * input_width);
				}
				output_index = output_index + (right_h_padding * output_width * output_depth);
			}
		}
		else
		{
			for(itr_b = 0; itr_b < input_batch; itr_b++)
			{
				output_index = output_index + (left_h_padding * output_width * output_depth);
				for(itr_h = 0; itr_h < input_height; itr_h++)
				{
					output_index = output_index + (left_w_padding * output_depth);
					for(itr_w = 0; itr_w < input_width; itr_w++)
					{
						output_index = output_index + left_d_padding;
						xa_nn_memmove_16_16((void*)&p_out[output_index], (void*)&p_in[input_index], (input_depth * sizeof(WORD16)));
						output_index = output_index + input_depth + right_d_padding;
						input_index = input_index + input_depth;
					}
					output_index = output_index + (right_w_padding * output_depth);
				}
				output_index = output_index + (right_h_padding * output_width * output_depth);
			}
		}
	}

	return 0;
}
