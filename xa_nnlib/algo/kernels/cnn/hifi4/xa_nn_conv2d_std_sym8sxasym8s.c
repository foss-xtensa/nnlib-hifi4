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
#include "xa_nn_conv2d_std_state.h"

#if 1


static WORD32 conv_x_left_pad(
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
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
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 out_width_over_x_pad = (x_padding - kernel_width)/x_stride + 1;
  WORD32 left_shift, right_shift;
  out_width_over_x_pad = out_width_over_x_pad > out_width ? out_width : out_width_over_x_pad;

  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);

  /* When kernel convolves over x-left pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = 0; j < out_width_over_x_pad; j++)
    {
      for(k = 0; k < out_channels; k++)
      {
#if TFLITE_SINGLE_ROUNDING
        left_shift  = p_out_shift[k];
        /* Single rounding macro doesn't need two shifts so this is not used */
        (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
        left_shift  = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
#endif /* #if TFLITE_SINGLE_ROUNDING */    
        ae_int32x2 acc;      
#if XCHAL_HAVE_HIFI1
        if(p_bias != NULL){
          acc = AE_L32_I((ae_int32*)&p_bias[k], 0);
        }
        else{
          acc = AE_MOVDA32(0);
        }
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
        AE_S8_0_X_HIFI1( AE_MOVINT16X4_FROMINT32X2(acc), (WORD8 *)p_out, (i * out_height_offset + j * out_width_offset + k * out_channels_offset));
#else
        if(p_bias != NULL){
          acc = AE_MOVDA32(p_bias[k]);
        }
        else{
          acc = AE_MOVDA32(0);
        }
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
#if 0
        AE_MINMAX32(acc, min_int8, max_int8);
#else
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
#endif
        p_out[i * out_height_offset + j * out_width_offset + k * out_channels_offset] = (UWORD8)AE_MOVAD32_L(acc);
#endif
      }
    }
  }
  return out_width_over_x_pad;
}

static WORD32 conv_x_right_pad(
    WORD32 x_padding,
    WORD32 input_width,
    WORD32 x_stride,
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
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 idx_out_width_over_x_r_pad = (x_padding + input_width + x_stride - 1)/x_stride + 1;
  WORD32 left_shift, right_shift;
  WORD32 out_width_over_x_r_pad = out_width - idx_out_width_over_x_r_pad;

  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);

  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = idx_out_width_over_x_r_pad; j < out_width; j++)
    {
      for(k = 0; k < out_channels; k++)
      {
#if TFLITE_SINGLE_ROUNDING
        left_shift  = p_out_shift[k];
        /* Single rounding macro doesn't need two shifts so this is not used */
        (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
        left_shift  = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
#endif /* #if TFLITE_SINGLE_ROUNDING */  
        ae_int32x2 acc;        
#if XCHAL_HAVE_HIFI1
        if(p_bias != NULL){
           acc = AE_L32_I((ae_int32*)&p_bias[k], 0);
        }
        else{
          acc = AE_MOVDA32(0);
        }
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
        AE_S8_0_X_HIFI1( AE_MOVINT16X4_FROMINT32X2(acc), (WORD8 *)p_out, (i * out_height_offset + j * out_width_offset + k * out_channels_offset));
#else
        if(p_bias != NULL){
          acc = AE_MOVDA32(p_bias[k]);
        }
        else{
          acc = AE_MOVDA32(0);
        }
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
#if 0
        AE_MINMAX32(acc, min_int8, max_int8);
#else
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
#endif
        p_out[i * out_height_offset + j * out_width_offset + k * out_channels_offset] = (UWORD8)AE_MOVAD32_L(acc);
#endif
      }
    }
  }
  return out_width_over_x_r_pad;
}
#endif

#ifdef polyphase_debug
#include<stdio.h>
void writingoutput(WORD8* __restrict__ p_out_base, WORD32 out_height, WORD32 out_width,WORD32 out_channels )
{
	int i,j, count;
	FILE * dataFilePr;
	count = 0;
	dataFilePr = fopen("C:/Users/hariev/Documents/file.txt", "w+");
	for(i=0;i<out_height;i++)
		for(j=0;j<out_width;j++)
		{
			fprintf(dataFilePr,"%d\n", *(p_out_base+count) );
			count = count + out_channels;
		}
	fclose(dataFilePr);
}
void manipulateinput(void* p_inp, WORD32 input_height, WORD32 input_width, WORD32 input_channels, void* p_ker, WORD32 kernel_height, WORD32 kernel_width, WORD32 output_channels, void* p_bias, WORD32* p_out_multiplier, WORD32* p_out_shift, WORD32* out_zero_bias, WORD32* input_zero_bias)
{
	WORD8* p_inp_debug;
	WORD8* p_ker_debug;
	WORD32* p_bias_debug;

	p_inp_debug  = (WORD8*)p_inp;
	p_ker_debug  = (WORD8*)p_ker;
	p_bias_debug = (WORD32*)p_bias;

	WORD32 iter = 0, i, k, j1, j2;
	for(k=0;k<input_height;k++)
		for(i=0;i<input_width;i++)
		{
			for(j1=0;j1<input_channels;j1++)
			{
				*p_inp_debug = iter;//14*k + 2*i;//iter;
				p_inp_debug++;
			}
			iter++;
			if(iter==8)
				iter = 0;
		}

	for(j2=0;j2<output_channels;j2++)
		for(k=0;k<kernel_height;k++)
			for(i=0;i<kernel_width;i++)
			{
				for(j1=0;j1<input_channels;j1++)
				{

					{
						*p_ker_debug = 1;
						//if( (k==0) && (i==0) && (j2==1))
							//*p_ker_debug = 1;
						p_ker_debug++;
					}
				}
			}

	for(k=0;k<output_channels;k++)
	{
		p_bias_debug[k] = 0;
		p_out_multiplier[k] = 1073741823;//1073741823;///2147483647;
		p_out_shift[k] = -2;
	}

	*out_zero_bias = 0;
	*input_zero_bias = 0;

}
#endif

WORD32 gcd(WORD32 a, WORD32 b)
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


WORD32 xa_nn_dilated_conv2d_std_per_chan_sym8sxasym8s(
    WORD8* __restrict__ p_out,
    const WORD8* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD32* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch,
    WORD32 dilation_height,
    WORD32 dilation_width)
{

	WORD8* __restrict__ p_out_base;
	p_out_base = p_out;
	//WORD32 dilation_height = 2, dilation_width = 3;//dilation
	WORD32 circMatrixHeight = 0;



#ifdef polyphase_debug
	/* Filling debug input data*/
	WORD32 base = 0;
	base = base * 5;
	VOID* p_inp_deb = (void*) p_inp;
	VOID* p_kernel_deb = (void*) p_kernel;
	VOID* p_bias_deb = (void*) p_bias;
	WORD8* p_buff_circ_deb;
  	manipulateinput((void*) p_inp_deb, input_height, input_width, input_channels, p_kernel_deb, kernel_height, kernel_width, out_channels, (void*) p_bias_deb, p_out_multiplier, p_out_shift, &out_zero_bias, &input_zero_bias);
#endif

	if(kernel_height==1)
  		dilation_height = 1;
  	if(kernel_width==1)
  		dilation_width = 1;

	WORD32 kernel_height_dilation = kernel_height + ( (dilation_height-1) * (kernel_height-1) );//dilation
	WORD32 kernel_width_dilation = kernel_width + ( (dilation_width-1) * (kernel_width-1) );//dilation
   /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  //XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(UWORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_height <= 0 || dilation_width <= 0), -1);//dilation
  //XA_NNLIB_ARG_CHK_COND((kernel_height_dilation > input_height), -1);//dilation
  //XA_NNLIB_ARG_CHK_COND((kernel_width_dilation > input_width), -1);//dilation
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  //XA_NNLIB_ARG_CHK_COND((y_stride != 1 || x_stride != 1), -1);//dilation
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);

  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  WORD32 input_bytewidth = 1;
  VOID *pp_inp = (VOID *)p_inp;
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  WORD32 out_channels_offset = out_data_format ? out_height * out_width : 1;
  WORD32 out_height_offset = out_data_format ? out_width : out_width * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;
  WORD32 x_padding_var = x_padding;
  WORD32 dilation_w_offset, dilation_h_offset;
  WORD32 out_iteraions;
  
#if !ENABLE_PADDING_CONV2D_STD
  WORD32 input_channels_pad = input_channels;
#else
  WORD32 input_channels_pad = PADDED_SIZE(input_channels, (ALIGNMENT>>1));
#endif

  // Initialize start of the circular buffer
  xa_nn_conv2d_dilation_init_state((void*)p_state,(void*)p_kernel, (void*)pp_inp);

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
    if(x_padding_var >= kernel_width_dilation)//dilation
  {
    //out_width_over_x_pad = conv_x_left_pad(x_padding, kernel_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);//dilation
    out_width_over_x_pad = conv_x_left_pad(x_padding, kernel_width_dilation, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
    x_padding_var -= out_width_over_x_pad * x_stride;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = kernel_width_dilation + (out_width - 1) * x_stride - (x_padding + input_width);//dilation
  //x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  XA_NNLIB_ARG_CHK_COND((x_r_pad<0), -1);
  if(x_r_pad >= kernel_width_dilation)//dilation
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_padding, input_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
  }


  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height_dilation + (out_height - 1) * y_stride - (y_padding + input_height);
  //y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;
  XA_NNLIB_ARG_CHK_COND((y_b_pad<0), -1);

  XA_NNLIB_ARG_CHK_COND((kernel_height_dilation > ( y_padding + input_height + y_b_pad)), -1);//dilation
  XA_NNLIB_ARG_CHK_COND((kernel_width_dilation  > ( x_padding + input_width  + x_r_pad)), -1);//dilation

  //WORD32 out_width_part_of_convolution = out_width-out_width_over_x_pad-out_width_over_x_r_pad;
  //WORD32 out_height_part_of_convolution = out_height;

  for(dilation_w_offset =0; dilation_w_offset<dilation_width; dilation_w_offset++ )
  {
	  /// Calculate number of left padding zeros for this particular width offset
	  WORD32 x_padding_dilation_initial_pad = ((x_padding-x_padding_var)/dilation_width) + (WORD32) ( (((x_padding-x_padding_var)%dilation_width)-1) >= dilation_w_offset); /// This offset's contribution which has been absorbed in initial analysis of zero padding

	  WORD32 x_stride_dilated = x_stride / gcd(x_stride, dilation_width);
	  //WORD32 out_points_for_this_xyoffset = ((x_padding_dilation + x_input_dilation + x_r_padding_dilation) - kernel_width)/x_stride_dilated + 1;/// This represents total num of times the conv needs to be called

	  WORD32 widthIndexIteration, firstWidthIndexNr, firstWidthIndex;
	  ///Check whether for a given width offset if there does exist a first column/width entry in this sub-matrix; if there are no width entries skip the entire row-offset
	  for(widthIndexIteration=0;widthIndexIteration<x_stride_dilated;widthIndexIteration++)
	  {
		  firstWidthIndexNr = (dilation_w_offset + (widthIndexIteration * dilation_width));
		  firstWidthIndex = firstWidthIndexNr / x_stride;
		  if(firstWidthIndex*x_stride == firstWidthIndexNr)
			  break;
	  }
	  if(widthIndexIteration==x_stride_dilated) //No more iterations for this width offset as the first index does not exist
		  continue;
	  //else if( ((x_padding_dilation + x_input_dilation + x_r_padding_dilation)- widthIndexIteration) < kernel_width) // After identifying the first index value check if there are enough points to convolve;if not break again; There is also no reason to check for higher values of firstIndex further
		//  continue;

	  //// "widthIndexIteration" variable is the first point from where convolution should start in the sub-matrix(polyphase) without accounting for left zero padding
	  //// When zp is consumed on the left and right side of the main matrix this needs to be accounted for
	  //// "widthIndexIteration" can lie after / before left zeropadding consumption point in sub-matrix
	  //// a) If "widthIndexIteration" lies after zero consumption point: adjustZCAndOffsetIndex = widthIndexIteration - zeroPadding
	  //// b) If "widthIndexIteration" lies before zero consumption point: n = ceil( ("zero consumption point" - "widthIndexIteration") / stride_dilation )
	  //// "widthIndexIteration" + n* x_stride_dilated > x_padding_dilation_initial_pad; find n such that this eq holds and substitute back in  <"widthIndexIteration" + n* x_stride_dilated> to find the new offset and later subtract it from  "x_padding_dilation_initial_pad" to get the first point of convolution

	  WORD32 adjustZpAndOffsetIndex;// In the sub-matrix some of the initial left padding values might be consumed by conv_x_left_pad() function and then there is an index offset. The index offset is a value which is oblivious to zero padding or input matrix or so on.
	  /// This is the number of points that needs to be skipped in the sub-matrix for the first convolution to happen in polyphase. There is a chance that conv_x_left_pad() consumed more or less than this offset. In either case the pointer has to be appropriately adjusted for this so as to consume the correct point in convolution
	  //// The variable "adjustZpAndOffsetIndex" is the new offset keeping in mind both the initial offset for sub-matrix and number of points consumed in conv_x_left_pad(). This becomse the new offset even inside circular matrix loading function from where the convolution is to begin
	  if(x_padding_dilation_initial_pad <= widthIndexIteration)
		  adjustZpAndOffsetIndex = widthIndexIteration - x_padding_dilation_initial_pad;
	  else
	  {
		  adjustZpAndOffsetIndex = (  (x_padding_dilation_initial_pad - widthIndexIteration) /  x_stride_dilated  );// This is floor ;
		  adjustZpAndOffsetIndex = adjustZpAndOffsetIndex + (((x_padding_dilation_initial_pad - widthIndexIteration) - (adjustZpAndOffsetIndex*x_stride_dilated))>0);/// ceil implementation
		  adjustZpAndOffsetIndex = widthIndexIteration + (adjustZpAndOffsetIndex * x_stride_dilated);
		  adjustZpAndOffsetIndex = adjustZpAndOffsetIndex - x_padding_dilation_initial_pad;
	  }


	  //// Calculations for out points for this width offset
	  //WORD32 totalPointsParticipatingInConvolution = x_padding_var + input_width + (x_r_pad - (out_width_over_x_r_pad*x_stride));//x_padding
	  WORD32 totalPointsParticipatingInConvolution = x_padding + input_width + (x_r_pad - (out_width_over_x_r_pad*x_stride));//Note:x_padding is added here and not x_padding_var because this is discounted later by sub x_padding_dilation_initial_pad
	  WORD32 pointsParticipatingInConvolutionForThisOffset = ( (totalPointsParticipatingInConvolution)/dilation_width) + (WORD32) ( (((totalPointsParticipatingInConvolution)%dilation_width)-1) >= dilation_w_offset);
	  pointsParticipatingInConvolutionForThisOffset = pointsParticipatingInConvolutionForThisOffset - x_padding_dilation_initial_pad;

	  if(  (pointsParticipatingInConvolutionForThisOffset - adjustZpAndOffsetIndex) < kernel_width) // After identifying the first index value check if there are enough points to convolve;if not break again; There is also no reason to check for higher values of firstIndex further
	  		continue;
	  WORD32 out_points_for_this_xyoffset = ((pointsParticipatingInConvolutionForThisOffset - adjustZpAndOffsetIndex) - kernel_width)/x_stride_dilated + 1;/// This represents total num of times the conv needs to be called

	  for(dilation_h_offset =0; dilation_h_offset<dilation_height; dilation_h_offset++ )
	  {
		  //if( ( dilation_w_offset <= (out_width_part_of_convolution-1)) &&  ( dilation_h_offset <= (out_height_part_of_convolution-1)) )
		  {

			  WORD32 input_padding_consumed =0;
			  WORD32 input_width_consumed = 0;

			  WORD32 y_stride_dilated = y_stride / gcd(y_stride, dilation_height); // This is the new stride value in height dimension
			  ///Check whether for a given height offset if there does exist a height entry in this sub-matrix;
			  ///if there are no width entries skip the entire height-offset
			  WORD32 heightIndexIteration, firstHeightIndexNr,firstHeightIndex ;
			  for(heightIndexIteration=0;heightIndexIteration<y_stride_dilated;heightIndexIteration++)
			  {
				  firstHeightIndexNr = (dilation_h_offset + (heightIndexIteration * dilation_height));
				  firstHeightIndex = firstHeightIndexNr / y_stride;
				  if(firstHeightIndex*y_stride == firstHeightIndexNr)
					  break;
			  }

			  WORD32 heightOfCircMatrix = ((y_padding + input_height + y_b_pad)/dilation_height) + (WORD32) ((((y_padding + input_height + y_b_pad)%dilation_height)-1)>=dilation_h_offset);// Height of circular matrix for a given offset value
			  if(heightIndexIteration==y_stride_dilated) //No more iterations for this height offset as the first index does not exist
				  continue;
			  else if( (heightOfCircMatrix- heightIndexIteration) < kernel_height) // After identifying the first index value check if there are enough points to convolve;if not break again; There is also no reason to check for higher values of firstIndex further
				  continue;

			  /// Initialize circular buffer end/height/size based on the dilation offset
			  xa_nn_dilated_conv2d_std_init_circ_buf(
                  (void*)p_state, (void*)p_kernel,
                   input_height, input_channels,
                   kernel_height_dilation, kernel_width,
                   y_stride, y_padding,
                   out_height, out_channels,
                   dilation_height, dilation_h_offset,
                   PREC_ASYM8S, PREC_ASYM8S); //dilation

#ifdef polyphase_debug
			  p_buff_circ_deb = p_state->cir_buf.p_curr;
#endif
			  WORD32 planesToAdd = (kernel_width - x_stride_dilated);
			  planesToAdd = (planesToAdd>0) ? planesToAdd : 0;
			  //xa_nn_dilated_conv2d_std_load_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, p_state, -input_zero_bias, dilation_height, dilation_h_offset, dilation_width, dilation_w_offset, x_padding, &input_padding_consumed, &input_width_consumed, planesToAdd,1,&circMatrixHeight, widthIndexIteration, x_stride_dilated, heightIndexIteration,y_stride_dilated);
			  xa_nn_dilated_conv2d_std_load_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, (VOID**)&pp_inp, p_state, -input_zero_bias, dilation_height, dilation_h_offset, dilation_width, dilation_w_offset, x_padding, &input_padding_consumed, &input_width_consumed, planesToAdd,1,&circMatrixHeight, adjustZpAndOffsetIndex, x_stride_dilated, heightIndexIteration);

			  ///output index addition corresponding to left padding
			  /*
			  WORD32 left_pad_offset;
			  for(left_pad_offset=out_width_over_x_pad;left_pad_offset<out_width_over_x_pad+dilation_width;left_pad_offset++)
				  if(((left_pad_offset)%dilation_width) == dilation_w_offset)
					  break;
					  */

			  //WORD32 outPointerWidthOffset = x_padding_dilation_initial_pad +

			  WORD32 outPointerHeightOffset = (dilation_h_offset + (heightIndexIteration*dilation_height) ) / y_stride; // In stride =1 case heightIndexIteration = 0// Refer to the PPT slide refering to formula to stich back matrix;last but 2 slide in PPT
			  WORD32 outPointerWidthOffset = (((x_padding_dilation_initial_pad + adjustZpAndOffsetIndex) * dilation_width) + dilation_w_offset) / x_stride;// The two addition terms take us to the point where conv. is going to start in this width_offset. Multiplication with dilation_width translates the same in linear domain. Adding the offset takes it to the right spot in the input matrix. Dividing by stride gives us the output width point
			  //p_out = p_out_base + ( outPointerHeightOffset * out_height_offset) + (left_pad_offset*out_width_offset);//(dilation_w_offset * out_width_offset) + ( (left_pad_offset+out_width_over_x_pad) * out_width_offset);
			  p_out = p_out_base + ( outPointerHeightOffset * out_height_offset) + (outPointerWidthOffset*out_width_offset);//(dilation_w_offset * out_width_offset) + ( (left_pad_offset+out_width_over_x_pad) * out_width_offset);

#ifdef polyphase_debug
			  p_buff_circ_deb = p_state->cir_buf.p_curr;
#endif

			  //out_points_for_this_xyoffset = 0;//To be removed for debugging purpose
			  for(out_iteraions = 0;out_iteraions<out_points_for_this_xyoffset;out_iteraions++)
			  {
				  planesToAdd = x_stride_dilated;
				  if(planesToAdd>kernel_width)
					  planesToAdd = kernel_width;
				  //xa_nn_dilated_conv2d_std_load_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, p_state, -input_zero_bias, dilation_height, dilation_h_offset, dilation_width, dilation_w_offset, x_padding, &input_padding_consumed, &input_width_consumed, planesToAdd,0,&circMatrixHeight, widthIndexIteration, x_stride_dilated, heightIndexIteration,y_stride_dilated);
				  xa_nn_dilated_conv2d_std_load_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, (VOID**)&pp_inp, p_state, -input_zero_bias, dilation_height, dilation_h_offset, dilation_width, dilation_w_offset, x_padding, &input_padding_consumed, &input_width_consumed, planesToAdd,0,&circMatrixHeight, adjustZpAndOffsetIndex, x_stride_dilated, heightIndexIteration);

#ifdef polyphase_debug
			  p_buff_circ_deb = p_state->cir_buf.p_curr;
#endif
				    // Convolution using matXvec with matrix as circular buffer
				    xa_nn_matXvec_sym8sxasym8s_asym8s_circ
				      (p_out /* output */
				       ,p_state->cir_buf.p_curr/* matrix: rows x cols */
				       ,p_state->p_kernel_padded /* vec: cols */
				       ,p_bias /* bias */
				       ,((circMatrixHeight-kernel_height)/y_stride_dilated)+1//out_height /* rows */
				       ,input_channels_pad * kernel_width * kernel_height /* cols */
				       ,input_channels_pad * kernel_width * y_stride_dilated/* row_offset */
				       ,out_channels /* vec_count */
				       ,input_channels_pad * kernel_width * kernel_height /* vec_stride */
				       ,out_channels_offset /* out_col_offset */
				       ,out_height_offset * dilation_height /gcd(y_stride, dilation_height)  /* out_row_offset *//// mul by dilation_height
				       ,input_zero_bias
				       ,p_out_multiplier
				       ,p_out_shift
				       ,out_zero_bias
				      );
			  	  //conv2d_dilation_ptr_reset((void*)p_state, (VOID**)&pp_inp);
				    p_out += (out_width_offset*dilation_width / gcd(x_stride, dilation_width) );//Mul by dilation width
			  }

		  }
	  }
  }
#ifdef polyphase_debug
  writingoutput(p_out_base, out_height, out_width, out_channels );
#endif
  return 0;
}

WORD32 xa_nn_conv2d_std_per_chan_sym8sxasym8s(
    WORD8* __restrict__ p_out,
    const WORD8* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD32* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch)
{
   /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  //XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(UWORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_scratch, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);

  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  WORD32 j;
  WORD32 input_bytewidth = 1;
  VOID *pp_inp = (VOID *)p_inp;

  p_scratch = ALIGNED_ADDR(p_scratch, ALIGNMENT);
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  WORD32 inp_h, inp_w, ker_h, ker_w, x_str, y_str, x_pad, y_pad, out_h, out_w;

  if ((input_height == 1) && (kernel_height == 1) && (out_height == 1))
  {
    inp_h = input_width;
    inp_w = input_height;
    ker_h = kernel_width;
    ker_w = kernel_height;
    x_str = y_stride;
    y_str = x_stride;
    x_pad = y_padding;
    y_pad = x_padding;
    out_h = out_width;
    out_w = out_height;
  }
  else
  {
    inp_h = input_height;
    inp_w = input_width;
    ker_h = kernel_height;
    ker_w = kernel_width;
    x_str = x_stride;
    y_str = y_stride;
    x_pad = x_padding;
    y_pad = y_padding;
    out_h = out_height;
    out_w = out_width;
  }

  xa_nn_conv2d_std_init_state((void*)p_state
      ,(void*)p_kernel
      ,inp_h
      ,input_channels
      ,ker_h
      ,ker_w
      ,y_str
      ,y_pad
      ,out_h
      ,out_channels
      ,PREC_ASYM8S
      ,PREC_SYM8S);

  WORD32 out_channels_offset = out_data_format ? out_h * out_w : 1;
  WORD32 out_height_offset = out_data_format ? out_w : out_w * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;

  WORD32 x_padding_var = x_pad;
  WORD32 input_channels_pad;

#if !ENABLE_PADDING_CONV2D_STD
  input_channels_pad = input_channels;
#else
#if HW_AE_ADDCIRC16X4_XC
  if(input_channels == 1){
    input_channels_pad = 1;
  }
  else
#endif
  {
    input_channels_pad = PADDED_SIZE(input_channels, (ALIGNMENT>>1));
  }
#endif

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
  if(x_padding_var >= ker_w)
  {
    out_width_over_x_pad = conv_x_left_pad(x_pad, ker_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
    x_padding_var -= out_width_over_x_pad * x_str;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = ker_w + (out_w - 1) * x_str - (x_pad + inp_w);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= ker_w)
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_pad, inp_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
  }

  /* When kernel convolves over input region */
  p_out += out_width_over_x_pad * out_width_offset;
  // Initialize circular buffer
  // Determine y-bottom padding
  WORD32 y_b_pad = ker_h + (out_h - 1) * y_str - (y_pad + inp_h);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  conv2d_std_init_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, p_state, -input_zero_bias);

  // Index to padded input width
  WORD32 idx_beg_inp_width_pad = ker_w - x_str;
  idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;


  // Process Loop to compute one output plane [out_h x out_channels] per iteration
  for(j=0;j<out_w-out_width_over_x_pad-out_width_over_x_r_pad;j++)
  {
    // Add x_str x (inp_h x input_channels) new planes to circular buffer
    conv2d_std_update_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state, -input_zero_bias);

    // Update index to input width padded
    idx_beg_inp_width_pad += x_str;

    // Convolution using matXvec with matrix as circular buffer
    xa_nn_matXvec_sym8sxasym8s_asym8s_circ
      (p_out /* output */
       ,p_state->cir_buf.p_curr/* matrix: rows x cols */
       ,p_state->p_kernel_padded /* vec: cols */
       ,p_bias /* bias */
       ,out_h /* rows */
       ,input_channels_pad * ker_w * ker_h /* cols */
       ,input_channels_pad * ker_w * y_str/* row_offset */
       ,out_channels /* vec_count */
       ,input_channels_pad * ker_w * ker_h /* vec_stride */
       ,out_channels_offset /* out_col_offset */
       ,out_height_offset /* out_row_offset */
       ,input_zero_bias
       ,p_out_multiplier
       ,p_out_shift
       ,out_zero_bias
      );

    p_out += out_width_offset;
  }

  return 0;
}

