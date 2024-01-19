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
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_err_chk.h"

#include <xtensa/config/core-isa.h>
#include <stddef.h>

static inline WORD32 xa_nn_memmove_32_1( void *p_dst,
		const void *p_src,
		WORD32 n)
{
  WORD32 MEMCPY_32b_num_elements = n >> 2;
  {
    ae_int32x2 d_inp0, d_inp1;
    ae_int32 * __restrict__ psrc  = (ae_int32 *)p_src;
    ae_int32 * __restrict__ pdest = (ae_int32 *)p_dst;
    ae_valign MEMCPY_32b_s_align = AE_LA64_PP((ae_int32x2 *)psrc);
    ae_valign MEMCPY_32b_d_align = AE_ZALIGN64();
    for (int ii=0; ii < (MEMCPY_32b_num_elements >> 2); ii++) {
      AE_LA32X2_IP(d_inp0, MEMCPY_32b_s_align, (ae_int32x2 *)psrc);
      AE_LA32X2_IP(d_inp1, MEMCPY_32b_s_align, (ae_int32x2 *)psrc);
      AE_SA32X2_IP(d_inp0, MEMCPY_32b_d_align, (ae_int32x2 *)pdest);
      AE_SA32X2_IP(d_inp1, MEMCPY_32b_d_align, (ae_int32x2 *)pdest);
    }
    AE_SA64POS_FP(MEMCPY_32b_d_align, pdest);
    for (int ii = 0; ii<(MEMCPY_32b_num_elements&3); ii++) {
      AE_L32_IP(d_inp0, (ae_int32 *)psrc, sizeof(ae_int32));
      AE_S32_L_IP(d_inp0, (ae_int32 *)pdest, sizeof(ae_int32));
    }
  }
return 0;
}

WORD32 xa_nn_strided_slice_int32(WORD32 * __restrict__ p_out,
		const   WORD32 * __restrict__ p_inp,
		WORD32 start_0, WORD32 stop_0,
		WORD32 start_1, WORD32 stop_1,
		WORD32 start_2, WORD32 stop_2,
		WORD32 start_3, WORD32 stop_3,
		WORD32 start_4, WORD32 stop_4,
		WORD32 stride_0, WORD32 stride_1,
		WORD32 stride_2, WORD32 stride_3, WORD32 stride_4,
		WORD32 dims_1, WORD32 dims_2,
		WORD32 dims_3, WORD32 dims_4)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);

  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD32), -1);

  /* Invalid input checks */
  XA_NNLIB_ARG_CHK_COND(((dims_1 <= 0) || (dims_2 <= 0) || (dims_3 <= 0) || (dims_4 <= 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((stride_0 == 0) || (stride_1 == 0) || (stride_2 == 0) || (stride_3 == 0) || (stride_4 == 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((stride_0 != 1) || (start_0 != 0) || (stop_0 != 1)), -1); /* TFLM reference Op only supports upto 4D cases*/
  /* Below conditions are derived from TFLM ref implementation */
  if(stride_1 > 0)
  {
    XA_NNLIB_ARG_CHK_COND(((start_1 < 0) || (start_1 > dims_1) || (stop_1 < 0) || (stop_1 > dims_1)), -1);
  }
  else
  {
    XA_NNLIB_ARG_CHK_COND(((start_1 < -1) || (start_1 > (dims_1-1)) || (stop_1 < -1) || (stop_1 > (dims_1-1))), -1);
  }
  if(stride_2 > 0)
  {
    XA_NNLIB_ARG_CHK_COND(((start_2 < 0) || (start_2 > dims_2) || (stop_2 < 0) || (stop_2 > dims_2)), -1);
  }
  else
  {
    XA_NNLIB_ARG_CHK_COND(((start_2 < -1) || (start_2 > (dims_2-1)) || (stop_2 < -1) || (stop_2 > (dims_2-1))), -1);
  }
  if(stride_3 > 0)
  {
    XA_NNLIB_ARG_CHK_COND(((start_3 < 0) || (start_3 > dims_3) || (stop_3 < 0) || (stop_3 > dims_3)), -1);
  }
  else
  {
    XA_NNLIB_ARG_CHK_COND(((start_3 < -1) || (start_3 > (dims_3-1)) || (stop_3 < -1) || (stop_3 > (dims_3-1))), -1);
  }
  if(stride_4 > 0)
  {
    XA_NNLIB_ARG_CHK_COND(((start_4 < 0) || (start_4 > dims_4) || (stop_4 < 0) || (stop_4 > dims_4)), -1);
  }
  else
  {
    XA_NNLIB_ARG_CHK_COND(((start_4 < -1) || (start_4 > (dims_4-1)) || (stop_4 < -1) || (stop_4 > (dims_4-1))), -1);
  }

  int i0, i1, i2, i3, i4;
  int a1 = dims_1*dims_2*dims_3*dims_4;
  int a2 = dims_2*dims_3*dims_4;
  int a3 = dims_3*dims_4;
  int a4 = dims_4;
  ae_int32x2 dummy;
  if((stride_0==1)&&(stride_1==1)&&(stride_2==1)&&(stride_3==1)&&(stride_4==1))
  {
    int start_0_a1 = start_0*a1;
    int start_1_a2 = start_1*a2;
    int start_2_a3 = start_2*a3;
    int start_3_a4 = start_3*a4;
    
    int stride_0_a1 = a1*4;
    int stride_1_a2 = a2*4;
    int stride_2_a3 = a3*4;
    int stride_3_a4 = a4*4;
    
    int stop_2_start_2_a3 = (stop_2 - start_2)*a3*4;
    int stop_3_start_3_a4 = (stop_3 - start_3)*a4*4;
    int stop_4_start_4 = (stop_4 - start_4)*4;
    
    WORD32 *p_inp0 = (WORD32*)p_inp+(start_0_a1);
    
    for(i0 = start_0; i0<stop_0; i0++)
    {
      WORD32 *p_inp1 = p_inp0+(start_1_a2);
      for(i1 = start_1; i1<stop_1; i1++)
      {
        WORD32 *p_inp2 = p_inp1+(start_2_a3);
        if ((start_3==0) && (start_4==0) && (stop_3 == dims_3) && (stop_4 == dims_4))
        {
          xa_nn_memmove_32_1((void*)p_out, (void*)p_inp2, stop_2_start_2_a3);
          AE_L32_XP(dummy, (ae_int32 *)p_out, stop_2_start_2_a3); //p_out += ((stop_2 - start_2)*dims_3*dims_4);
        }
        else
        {
          for(i2 = start_2; i2<stop_2; i2++)
          {
            WORD32 *p_inp3 = p_inp2+(start_3_a4);
            if ((start_4==0) && (stop_4 == dims_4))
            {
              xa_nn_memmove_32_1((void*)p_out, (void*)p_inp3, stop_3_start_3_a4);
              AE_L32_XP(dummy, (ae_int32 *)p_out, stop_3_start_3_a4); //p_out+=(stop_3 - start_3)*dims_4;
            }
            else
            {
              for(i3 = start_3; i3<stop_3; i3++)
              {
                WORD32 *p_inp4 = p_inp3+start_4;
                xa_nn_memmove_32_1((void*)p_out, (void*)p_inp4, stop_4_start_4);
                AE_L32_XP(dummy, (ae_int32 *)p_out, stop_4_start_4);
                AE_L32_XP(dummy, (ae_int32 *)p_inp3, stride_3_a4);
              } // for i3
            }
            AE_L32_XP(dummy, (ae_int32 *)p_inp2, stride_2_a3);
          } // for i2
        }
        AE_L32_XP(dummy, (ae_int32 *)p_inp1, stride_1_a2);
      } //for i1
      AE_L32_XP(dummy, (ae_int32 *)p_inp0, stride_0_a1);
    } //for i0
  }
  else
  {
    /* TODO : See if the below ceil function implementation can be optimized. */
    int i0_cnt = ((stop_0 - start_0)/(float)stride_0 + (((stop_0 - start_0)%stride_0)!=0));
    int i1_cnt = ((stop_1 - start_1)/(float)stride_1 + (((stop_1 - start_1)%stride_1)!=0));
    int i2_cnt = ((stop_2 - start_2)/(float)stride_2 + (((stop_2 - start_2)%stride_2)!=0));
    int i3_cnt = ((stop_3 - start_3)/(float)stride_3 + (((stop_3 - start_3)%stride_3)!=0));
    int i4_cnt = ((stop_4 - start_4)/(float)stride_4 + (((stop_4 - start_4)%stride_4)!=0));
    
    i0_cnt = i0_cnt < 0 ? 0 : i0_cnt;
    i1_cnt = i1_cnt < 0 ? 0 : i1_cnt;
    i2_cnt = i2_cnt < 0 ? 0 : i2_cnt;
    i3_cnt = i3_cnt < 0 ? 0 : i3_cnt;
    i4_cnt = i4_cnt < 0 ? 0 : i4_cnt;
    
    int start_0_a1 = start_0*a1;
    int start_1_a2 = start_1*a2;
    int start_2_a3 = start_2*a3;
    int start_3_a4 = start_3*a4;
    
    int stride_0_a1 = stride_0*a1*4;
    int stride_1_a2 = stride_1*a2*4;
    int stride_2_a3 = stride_2*a3*4;
    int stride_3_a4 = stride_3*a4*4;
    
    int stop_2_start_2_a3 = (stop_2-start_2)*a3*4;
    int stop_3_start_3_a4 = (stop_3 - start_3)*a4*4;
    int stop_4_start_4 = (stop_4-start_4)*4;
    
    WORD32 *p_inp0 = (WORD32*)p_inp+(start_0_a1);
    
    for(i0 = 0; i0<i0_cnt; i0++)
    {
      WORD32 *p_inp1 = p_inp0+(start_1_a2);
      for(i1 = 0; i1<i1_cnt; i1++)
      {
        WORD32 *p_inp2 = p_inp1+(start_2_a3);
        if ((stride_2 == 1) && (stride_3 == 1) && (stride_4 == 1) && (start_3==0) && (start_4==0) && (stop_3 == dims_3) && (stop_4 == dims_4))
        {
          xa_nn_memmove_32_1((void*)p_out, (void*)p_inp2, stop_2_start_2_a3);
          AE_L32_XP(dummy, (ae_int32 *)p_out, stop_2_start_2_a3); //p_out += ((stop_2 - start_2)*dims_3*dims_4);
        }
        else
        {
          for(i2 = 0; i2<i2_cnt; i2++)
          {
            WORD32 *p_inp3 = p_inp2+(start_3_a4);
            if ((stride_3 == 1) && (stride_4 == 1) && (start_4==0) && (stop_4 == dims_4))
            {
              xa_nn_memmove_32_1((void*)p_out, (void*)p_inp3, stop_3_start_3_a4);
              AE_L32_XP(dummy, (ae_int32 *)p_out, stop_3_start_3_a4); //p_out+=(stop_3 - start_3)*dims_4;
            }
            else
            {
              for(i3 = 0; i3<i3_cnt; i3++)
              {
                WORD32 *p_inp4 = p_inp3+start_4;
                if(stride_4 == 1)
                {
                  xa_nn_memmove_32_1((void*)p_out, (void*)p_inp4, stop_4_start_4);
                  AE_L32_XP(dummy, (ae_int32 *)p_out, stop_4_start_4);
                }
                else
                {
                  ae_int32x2 d1;
                  for(i4 = 0; i4<i4_cnt; i4++)
                  {
                    AE_L32_XP(d1, (ae_int32*)p_inp4, stride_4*sizeof(WORD32));
                    AE_S32_L_IP(d1, (ae_int32 *)p_out, sizeof(ae_int32));
                  } //for i4
                }
                AE_L32_XP(dummy, (ae_int32 *)p_inp3, stride_3_a4);
              } // for i3
            }
            AE_L32_XP(dummy, (ae_int32 *)p_inp2, stride_2_a3);
          } // for i2
        }
        AE_L32_XP(dummy, (ae_int32 *)p_inp1, stride_1_a2);
      } //for i1
      AE_L32_XP(dummy, (ae_int32 *)p_inp0, stride_0_a1);
    } //for i0
  }
  return 0;
}
