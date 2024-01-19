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
#include <stddef.h>
#include "xa_nnlib_common.h"

WORD32 xa_nn_matXvec_16x16_16_circ_nb(
  WORD16 * __restrict__ p_out,
  WORD16 * __restrict__ p_mat,
  WORD16 * __restrict__ p_vec,
  WORD16 * __restrict__ p_bias,
  WORD32 rows,
  WORD32 cols,
  WORD32 out_offset,
  WORD32 bias_shift,
  WORD32 acc_shift)
{
  WORD32 row, col;
  ae_int16x4 temp_src1, temp_src2;

  if ((NULL == p_out) || (NULL == p_mat) || (NULL == p_vec))
  {
    return -1;
  }

  if ((0 >= rows ) || (0 >= cols ) || (cols & 0x3))
  {
    return -2;
  }
   
  row = 0;

  if(rows >= 8)
  {
    for (row = 0; row < ( rows & ~(8-1)) ; row+=8)
    {
      ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
      ae_int64 accu1_0, accu1_1, accu1_2, accu1_3, accu1_4, accu1_5, accu1_6, accu1_7;
      ae_int16x4 *p_mat1_0 = (ae_int16x4*)&p_mat[(row+0)*cols]; 
      accu1_0 = p_bias[row+0];            
      accu1_0 = AE_SLAA64S(accu1_0 , bias_shift);
      ae_int16x4 *p_mat1_1 = (ae_int16x4*)&p_mat[(row+1)*cols]; 
      accu1_1 = p_bias[row+1];            
      accu1_1 = AE_SLAA64S(accu1_1 , bias_shift);  
      ae_int16x4 *p_mat1_2 = (ae_int16x4*)&p_mat[(row+2)*cols]; 
      accu1_2 = p_bias[row+2];            
      accu1_2 = AE_SLAA64S(accu1_2 , bias_shift);  
      ae_int16x4 *p_mat1_3 = (ae_int16x4*)&p_mat[(row+3)*cols]; 
      accu1_3 = p_bias[row+3];            
      accu1_3 = AE_SLAA64S(accu1_3 , bias_shift);  
      ae_int16x4 *p_mat1_4 = (ae_int16x4*)&p_mat[(row+4)*cols]; 
      accu1_4 = p_bias[row+4];            
      accu1_4 = AE_SLAA64S(accu1_4 , bias_shift);  
      ae_int16x4 *p_mat1_5 = (ae_int16x4*)&p_mat[(row+5)*cols]; 
      accu1_5 = p_bias[row+5];            
      accu1_5 = AE_SLAA64S(accu1_5 , bias_shift);  
      ae_int16x4 *p_mat1_6 = (ae_int16x4*)&p_mat[(row+6)*cols]; 
      accu1_6 = p_bias[row+6];            
      accu1_6 = AE_SLAA64S(accu1_6 , bias_shift);  
      ae_int16x4 *p_mat1_7 = (ae_int16x4*)&p_mat[(row+7)*cols]; 
      accu1_7 = p_bias[row+7];            
      accu1_7 = AE_SLAA64S(accu1_7 , bias_shift);

      for (col = 0; col < (cols >> 3); col++) 
      {
        ae_int16x4 temp_in1_I, temp_in1; 
    	temp_in1_I = AE_L16X4_I(p_mat1_0, 8); 
    	AE_L16X4_IP(temp_in1, p_mat1_0, 16); 
        AE_L16X4_XC(temp_src1, p_src1, 8); 
        AE_L16X4_XC(temp_src2, p_src1, 8); 
        AE_MULAAAAQ16(accu1_0, temp_src1, temp_in1);
        AE_MULAAAAQ16(accu1_0, temp_src2, temp_in1_I);

        AE_L16X4_IP(temp_in1, p_mat1_1, 8); 
        AE_MULAAAAQ16(accu1_1, temp_src1, temp_in1);
        temp_in1 = AE_L16X4_I(p_mat1_1, 0); 
        p_mat1_1++;
        AE_MULAAAAQ16(accu1_1, temp_src2, temp_in1);
        temp_in1_I = AE_L16X4_I(p_mat1_2, 8); 

        AE_L16X4_IP(temp_in1, p_mat1_2, 16); 
        AE_MULAAAAQ16(accu1_2, temp_src1, temp_in1);
        AE_MULAAAAQ16(accu1_2, temp_src2, temp_in1_I);
        temp_in1_I = AE_L16X4_I(p_mat1_3, 8); 

        AE_L16X4_IP(temp_in1, p_mat1_3, 16); 
        AE_MULAAAAQ16(accu1_3, temp_src1, temp_in1);
        AE_MULAAAAQ16(accu1_3, temp_src2, temp_in1_I);
        temp_in1_I = AE_L16X4_I(p_mat1_4, 8); 

        AE_L16X4_IP(temp_in1, p_mat1_4, 16); 
        AE_MULAAAAQ16(accu1_4, temp_src1, temp_in1);
        AE_MULAAAAQ16(accu1_4, temp_src2, temp_in1_I);
        temp_in1_I = AE_L16X4_I(p_mat1_5, 8); 

        AE_L16X4_IP(temp_in1, p_mat1_5, 16); 
        AE_MULAAAAQ16(accu1_5, temp_src1, temp_in1);
        AE_MULAAAAQ16(accu1_5, temp_src2, temp_in1_I);
        temp_in1_I = AE_L16X4_I(p_mat1_6, 8); 

        AE_L16X4_IP(temp_in1, p_mat1_6, 16); 
        AE_MULAAAAQ16(accu1_6, temp_src1, temp_in1);
        AE_MULAAAAQ16(accu1_6, temp_src2, temp_in1_I);
        temp_in1_I = AE_L16X4_I(p_mat1_7, 8); 

        AE_L16X4_IP(temp_in1, p_mat1_7, 16); 
        AE_MULAAAAQ16(accu1_7, temp_src1, temp_in1);
        AE_MULAAAAQ16(accu1_7, temp_src2, temp_in1_I);
      }
      if((cols & 7) !=0)
      {
        ae_int16x4 temp_in1;
        AE_L16X4_IP(temp_in1, p_mat1_0, 8);
        AE_L16X4_XC(temp_src1, p_src1, 8);
        AE_MULAAAAQ16(accu1_0, temp_src1, temp_in1);
        AE_L16X4_IP(temp_in1, p_mat1_1, 8);
        AE_MULAAAAQ16(accu1_1, temp_src1, temp_in1);
        AE_L16X4_IP(temp_in1, p_mat1_2, 8);
        AE_MULAAAAQ16(accu1_2, temp_src1, temp_in1);
        AE_L16X4_IP(temp_in1, p_mat1_3, 8);
        AE_MULAAAAQ16(accu1_3, temp_src1, temp_in1);
        AE_L16X4_IP(temp_in1, p_mat1_4, 8);
        AE_MULAAAAQ16(accu1_4, temp_src1, temp_in1);
        AE_L16X4_IP(temp_in1, p_mat1_5, 8);
        AE_MULAAAAQ16(accu1_5, temp_src1, temp_in1);
        AE_L16X4_IP(temp_in1, p_mat1_6, 8);
        AE_MULAAAAQ16(accu1_6, temp_src1, temp_in1);
        AE_L16X4_IP(temp_in1, p_mat1_7, 8);
        AE_MULAAAAQ16(accu1_7, temp_src1, temp_in1);
      }

#if !XCHAL_HAVE_HIFI1
      accu1_0 = AE_SLAA64S(accu1_0 , acc_shift);
      p_out[(row+0)*out_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_0),16),-16));  
      accu1_1 = AE_SLAA64S(accu1_1 , acc_shift);
      p_out[(row+1)*out_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_1),16),-16));  
      accu1_2 = AE_SLAA64S(accu1_2 , acc_shift);
      p_out[(row+2)*out_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_2),16),-16));  
      accu1_3 = AE_SLAA64S(accu1_3 , acc_shift);
      p_out[(row+3)*out_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_3),16),-16));  
      accu1_4 = AE_SLAA64S(accu1_4 , acc_shift);
      p_out[(row+4)*out_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_4),16),-16));  
      accu1_5 = AE_SLAA64S(accu1_5 , acc_shift);
      p_out[(row+5)*out_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_5),16),-16));  
      accu1_6 = AE_SLAA64S(accu1_6 , acc_shift);
      p_out[(row+6)*out_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_6),16),-16));  
      accu1_7 = AE_SLAA64S(accu1_7 , acc_shift);
      p_out[(row+7)*out_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_7),16),-16));
#else
      accu1_0 = AE_SLAA64S(accu1_0 , acc_shift);
      ae_int32x2 sat_accu1_0 = AE_ROUND32F64SSYM(accu1_0); 
      p_out[(row+0)*out_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_accu1_0, sat_accu1_0)); 
      accu1_1 = AE_SLAA64S(accu1_1 , acc_shift);
      ae_int32x2 sat_accu1_1 = AE_ROUND32F64SSYM(accu1_1); 
      p_out[(row+1)*out_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_accu1_1, sat_accu1_1));
      accu1_2 = AE_SLAA64S(accu1_2 , acc_shift);
      ae_int32x2 sat_accu1_2 = AE_ROUND32F64SSYM(accu1_2); 
      p_out[(row+2)*out_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_accu1_2, sat_accu1_2));  
      accu1_3 = AE_SLAA64S(accu1_3 , acc_shift);
      ae_int32x2 sat_accu1_3 = AE_ROUND32F64SSYM(accu1_3); 
      p_out[(row+3)*out_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_accu1_3, sat_accu1_3)); 
      accu1_4 = AE_SLAA64S(accu1_4 , acc_shift);
      ae_int32x2 sat_accu1_4 = AE_ROUND32F64SSYM(accu1_4); 
      p_out[(row+4)*out_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_accu1_4, sat_accu1_4)); 
      accu1_5 = AE_SLAA64S(accu1_5 , acc_shift);
      ae_int32x2 sat_accu1_5 = AE_ROUND32F64SSYM(accu1_5); 
      p_out[(row+5)*out_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_accu1_5, sat_accu1_5)); 
      accu1_6 = AE_SLAA64S(accu1_6 , acc_shift);
      ae_int32x2 sat_accu1_6 = AE_ROUND32F64SSYM(accu1_6); 
      p_out[(row+6)*out_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_accu1_6, sat_accu1_6));
      accu1_7 = AE_SLAA64S(accu1_7 , acc_shift);
      ae_int32x2 sat_accu1_7 = AE_ROUND32F64SSYM(accu1_7); 
      p_out[(row+7)*out_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_accu1_7, sat_accu1_7));
#endif
    }
  }
  // Handle remaining rows
  for (; row < rows ; row++)
  {
    ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
    ae_int64 accu1_0;
    ae_int16x4 *p_mat1_0 = (ae_int16x4*)&p_mat[(row+0)*cols]; 
    accu1_0 = p_bias[row+0];            
    accu1_0 = AE_SLAA64S(accu1_0 , bias_shift);
    for (col = 0; col < (cols>>2); col++) {
      ae_int16x4 temp_in1; 
      AE_L16X4_IP(temp_in1, p_mat1_0, 8);
      AE_L16X4_XC(temp_src1, p_src1, 8);
      AE_MULAAAAQ16(accu1_0, temp_src1, temp_in1);
    }
#if !XCHAL_HAVE_HIFI1
    accu1_0 = AE_SLAA64S(accu1_0 , acc_shift);
    p_out[(row+0)*out_offset] = AE_MOVINT16_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_0),16),-16));
#else
    accu1_0 = AE_SLAA64S(accu1_0 , acc_shift);
    ae_int32x2 sat_accu1_0 = AE_ROUND32F64SSYM(accu1_0); 
    p_out[(row+0)*out_offset] = AE_MOVINT16_FROMINT16X4(AE_SAT16X4(sat_accu1_0, sat_accu1_0));
#endif
  }
  return 0;
}

