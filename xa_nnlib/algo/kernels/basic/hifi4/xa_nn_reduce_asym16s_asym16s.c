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
#include <string.h>
#include "xa_nn_basic_state.h"
#include "xa_nnlib_common_macros.h"

#define ALIGNMENT_8   8

static WORD32 xa_nn_memset_16(WORD16 *p_dst, WORD16 val, WORD32 n)
{
  int i;
  ae_int16x4 d_inp0 = AE_MOVDA16(val);
  WORD16 *ptr_dst = (WORD16 *)p_dst;
  ae_valign dst_align = AE_ZALIGN64();
  for (i = 0; i < n >> 2 ; i++)
  {
     AE_SA16X4_IP(d_inp0, dst_align, (ae_int16x4 *)ptr_dst);
  }
#if XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= RI9_HWVERSION)
  if((n & 3) != 0)
  {
    AE_SAV16X4_XP(d_inp0, dst_align, (ae_int16x4 *)ptr_dst, (((n & 3) << 1)));
  }
  AE_SA64POS_FP(dst_align, ptr_dst);
#else  
  AE_SA64POS_FP(dst_align, ptr_dst);
  for (i = 0; i < (n & 3) ; i++)
  {
    *ptr_dst++ = val;
  }
#endif  
  return 0;
}

static void vecmax16_inpx3(const WORD16 *p_src1, const WORD16* p_src2, const WORD16* p_src3, WORD16 * /*__restrict__*/ p_dst, int N){
    int i = 0;
    ae_valign align_src1, align_src2, align_src3, align_dst;
    align_src1 = AE_LA64_PP(p_src1);
    align_src2 = AE_LA64_PP(p_src2);
    align_src3 = AE_LA64_PP(p_src3);
    align_dst = AE_ZALIGN64();

    for(i=0; i < (N >> 3); i++)
    {
      ae_int16x4 i1, j1, k1, i2, j2 ,k2;
      xtbool4 b1, b2;
      AE_LA16X4_IP(i1, align_src1, (ae_int16x4 *)p_src1);
      AE_LA16X4_IP(j1, align_src2, (ae_int16x4 *)p_src2);
      AE_LA16X4_IP(k1, align_src3, (ae_int16x4 *)p_src3);
      AE_LA16X4_IP(i2, align_src1, (ae_int16x4 *)p_src1);
      AE_LA16X4_IP(j2, align_src2, (ae_int16x4 *)p_src2);
      AE_LA16X4_IP(k2, align_src3, (ae_int16x4 *)p_src3);
      b1 = AE_LT16(i1, j1);
      b2 = AE_LT16(i2, j2);
      AE_MOVT16X4(i1, j1, b1);
      AE_MOVT16X4(i2, j2, b2);
      b1 = AE_LT16(i1, k1);
      b2 = AE_LT16(i2, k2);
      AE_MOVT16X4(i1, k1, b1);
      AE_MOVT16X4(i2, k2, b2);
      AE_SA16X4_IP(i1, align_dst, (ae_int16x4 *)p_dst);
      AE_SA16X4_IP(i2, align_dst, (ae_int16x4 *)p_dst);
    }
#if XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= RI9_HWVERSION)
    if((N & 7) != 0)
    {
      int num_scalar_ops = (N & 7);
      int num_scalar_ops0 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
      int num_scalar_ops1 = num_scalar_ops > 4 ? num_scalar_ops - 4 : 0;

      ae_int16x4 i1, j1, k1, i2, j2 ,k2;
      xtbool4 b1, b2;
      AE_LAV16X4_XP(i1, align_src1, (ae_int16x4 *)p_src1, ((num_scalar_ops0) << 1));
      AE_LAV16X4_XP(j1, align_src2, (ae_int16x4 *)p_src2, ((num_scalar_ops0) << 1));
      AE_LAV16X4_XP(k1, align_src3, (ae_int16x4 *)p_src3, ((num_scalar_ops0) << 1));
      AE_LAV16X4_XP(i2, align_src1, (ae_int16x4 *)p_src1, ((num_scalar_ops1) << 1));
      AE_LAV16X4_XP(j2, align_src2, (ae_int16x4 *)p_src2, ((num_scalar_ops1) << 1));
      AE_LAV16X4_XP(k2, align_src3, (ae_int16x4 *)p_src3, ((num_scalar_ops1) << 1));
      b1 = AE_LT16(i1, j1);
      b2 = AE_LT16(i2, j2);
      AE_MOVT16X4(i1, j1, b1);
      AE_MOVT16X4(i2, j2, b2);
      b1 = AE_LT16(i1, k1);
      b2 = AE_LT16(i2, k2);
      AE_MOVT16X4(i1, k1, b1);
      AE_MOVT16X4(i2, k2, b2);
      AE_SAV16X4_XP(i1, align_dst, (ae_int16x4 *)p_dst, ((num_scalar_ops0) << 1));
      AE_SAV16X4_XP(i2, align_dst, (ae_int16x4 *)p_dst, ((num_scalar_ops1) << 1));      
    }
    AE_SA64POS_FP(align_dst, p_dst);
#else    
    AE_SA64POS_FP(align_dst, p_dst);
    //Remainder Loop
    for(i = 0; i < (N & 7); i++)
    {
      ae_int32x2 i1, j1, k1, out1;
      i1 = (WORD32) *(p_src1 + i);
      j1 = (WORD32) *(p_src2 + i);
      k1 = (WORD32) *(p_src3 + i);
      out1 = AE_MAX32(i1, j1);
      out1 = AE_MAX32(out1, k1);
      *p_dst++ = (WORD16) AE_MOVAD32_H(out1);
    }
#endif    
}

static void vecmax16_inpx2(const WORD16 *p_src1, const WORD16* p_src2, WORD16 *p_dst, int N){
    int i = 0;
    ae_valign align_src1, align_src2, align_dst;
    align_src1 = AE_LA64_PP(p_src1);
    align_src2 = AE_LA64_PP(p_src2);
    align_dst = AE_ZALIGN64();    
    for(i=0; i < (N >> 3); i++)
    {
      ae_int16x4 i1, j1, i2, j2;
      xtbool4 b1, b2;
      AE_LA16X4_IP(i1, align_src1, (ae_int16x4 *)p_src1);
      AE_LA16X4_IP(j1, align_src2, (ae_int16x4 *)p_src2);
      AE_LA16X4_IP(i2, align_src1, (ae_int16x4 *)p_src1);
      AE_LA16X4_IP(j2, align_src2, (ae_int16x4 *)p_src2);
      b1 = AE_LT16(i1, j1);
      b2 = AE_LT16(i2, j2);
      AE_MOVT16X4(i1, j1, b1);
      AE_MOVT16X4(i2, j2, b2);
      AE_SA16X4_IP(i1, align_dst, (ae_int16x4 *)p_dst);
      AE_SA16X4_IP(i2, align_dst, (ae_int16x4 *)p_dst);
    }
#if XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= RI9_HWVERSION)
    if((N & 7) != 0)
    {
      int num_scalar_ops = (N & 7);
      int num_scalar_ops0 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
      int num_scalar_ops1 = num_scalar_ops > 4 ? num_scalar_ops - 4 : 0;
      
      ae_int16x4 i1, j1, i2, j2;
      xtbool4 b1, b2;
      AE_LAV16X4_XP(i1, align_src1, (ae_int16x4 *)p_src1, (num_scalar_ops0 << 1));
      AE_LAV16X4_XP(j1, align_src2, (ae_int16x4 *)p_src2, (num_scalar_ops0 << 1));
      AE_LAV16X4_XP(i2, align_src1, (ae_int16x4 *)p_src1, (num_scalar_ops1 << 1));
      AE_LAV16X4_XP(j2, align_src2, (ae_int16x4 *)p_src2, (num_scalar_ops1 << 1));
      b1 = AE_LT16(i1, j1);
      b2 = AE_LT16(i2, j2);
      AE_MOVT16X4(i1, j1, b1);
      AE_MOVT16X4(i2, j2, b2);
      AE_SAV16X4_XP(i1, align_dst, (ae_int16x4 *)p_dst, (num_scalar_ops0 << 1));
      AE_SAV16X4_XP(i2, align_dst, (ae_int16x4 *)p_dst, (num_scalar_ops1 << 1));      
    }
    AE_SA64POS_FP(align_dst, p_dst);
#else    
    AE_SA64POS_FP(align_dst, p_dst);
    //Remainder Loop
    for(i = 0; i < (N & 7); i++)
    {
      ae_int32x2 i1, j1, out1;
      i1 = (WORD32) *(p_src1 + i);
      j1 = (WORD32) *(p_src2 + i);
      out1 = AE_MAX32(i1, j1);
      *p_dst++ = (WORD16) AE_MOVAD32_H(out1);
    }
#endif    
}

WORD32 xa_nn_reduce_max_4D_asym16s_asym16s(WORD16 * __restrict__ p_out
                                           ,const WORD32 *const p_out_shape
                                           ,const WORD16 * __restrict__ p_inp
                                           ,const WORD32 *const p_inp_shape
                                           ,const WORD32 * __restrict__ p_axis
                                           ,WORD32 num_out_dims
                                           ,WORD32 num_inp_dims
                                           ,WORD32 num_axis_dims
                                           ,pVOID p_scratch_in)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_axis, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);

  /* Invalid input checks */
  XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND(((num_out_dims <= 0) || (num_out_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND(((num_axis_dims < 0) || (num_axis_dims > 4)), -1);

  int axis_itr = 0, inp_itr = 0, out_itr = 0;
  for(axis_itr=0; axis_itr < num_axis_dims; axis_itr++)
  {
    XA_NNLIB_ARG_CHK_COND(((p_axis[axis_itr] < 0) || (p_axis[axis_itr] > (num_inp_dims - 1))), -1);
  }

  for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[inp_itr] <= 0), -1);
  }

  int out_length = 1;
  for(out_itr=0; out_itr < num_out_dims; out_itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shape[out_itr] <= 0), -1);
    out_length *= p_out_shape[out_itr];
  }

  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_axis, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);

  WORD16 *p_in = (WORD16 *)(p_inp);
  WORD16 *p_scratch = (WORD16 *)(p_scratch_in);

  // Changing order of axis data so that reduce max will be first computed
  // across largest inp shape dim in axis. This is required to
  // minimize the scratch usage.
  int inp_length = 1, p_axis_data[4], inp_shape_max;
  if(num_axis_dims)
  {
    inp_shape_max = p_inp_shape[p_axis[0]];
    axis_itr = 1;
    int max_axis_itr = 0;
    int temp_p_axis_0 = p_axis[0];
    for(axis_itr = 0; axis_itr < num_axis_dims; axis_itr++)
    {
      p_axis_data[axis_itr] = p_axis[axis_itr];
    }
    for(axis_itr = 1; axis_itr < num_axis_dims; axis_itr++)
    {
      if(p_inp_shape[p_axis[axis_itr]] > inp_shape_max)
      {
        inp_shape_max = p_inp_shape[p_axis[axis_itr]];
        max_axis_itr = axis_itr;
      }
    }
    p_axis_data[0] = p_axis_data[max_axis_itr];
    p_axis_data[max_axis_itr] = temp_p_axis_0;

    inp_itr = 0;
    for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
    {
      inp_length *= p_inp_shape[inp_itr];
    }

    xa_nn_memset_16(p_scratch, -32768, (inp_length / inp_shape_max)); //TODO: Alternate approach for memset?
  }

  // Promoting lesser dim tensors to 4D tensors. Also modifying axis
  // data accordingly.
  int p_4D_inp_shape[4] = {1, 1, 1, 1};
  int itr = num_inp_dims - 1;
  int count = 3;
  while(itr >= 0)
  {
    p_4D_inp_shape[count] = p_inp_shape[itr];
    itr--;
    count--;
  }
  for(itr = 0; itr < num_axis_dims; itr++)
  {
    p_axis_data[itr] = p_axis_data[itr] + (4 - num_inp_dims);
  }

  int temp_inp_n = p_4D_inp_shape[0]; 
  int temp_inp_h = p_4D_inp_shape[1]; 
  int temp_inp_w = p_4D_inp_shape[2]; 
  int temp_inp_c = p_4D_inp_shape[3];

  int flag = 0;
  int itr_axis, itr_n, itr_h, itr_w, itr_c;
  WORD16 *p_src1, *p_src2, *p_src3;
  WORD16 * p_dst;

  for(itr_axis=0; itr_axis < num_axis_dims; itr_axis++)
  {
    switch(p_axis_data[itr_axis])
    {
      case 0: {            
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;

        for(itr_n=0; itr_n < (temp_inp_n & ~(2 - 1)); itr_n += 2)
        {
          p_src1 = p_scratch;
          p_src2 = p_in + itr_n * plane_size;
          p_src3 = p_in + (itr_n + 1) * plane_size;
          p_dst  = p_scratch;

          vecmax16_inpx3(p_src1,p_src2,p_src3,p_dst,plane_size);
        }
        if(temp_inp_n & 1)
        {
          p_src1 = p_scratch;
          p_src2 = p_in + itr_n * plane_size;
          p_dst  = p_scratch;
          vecmax16_inpx2(p_src1, p_src2, p_dst, plane_size);
        }
        temp_inp_n = 1;  
        }break;
      case 1: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          p_src1 = p_scratch + (itr_n * wc_plane_size * (!flag)) + (flag * itr_n * plane_size);
          for(itr_h=0; itr_h < (temp_inp_h & ~(2 - 1)); itr_h += 2)
          {
            p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
            p_src3 = p_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size);
            p_dst = p_scratch + (itr_n * wc_plane_size);

            vecmax16_inpx3(p_src1,p_src2,p_src3,p_dst,wc_plane_size);
            p_src1 = p_scratch + (itr_n * wc_plane_size);
          }

          if(temp_inp_h & 1)
          {
            p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
            p_dst = p_scratch + (itr_n * wc_plane_size);

            vecmax16_inpx2(p_src1, p_src2, p_dst, wc_plane_size);
            p_src1 = p_scratch + (itr_n * wc_plane_size);
          }
        }
        temp_inp_h = 1;
        }break;
      case 2:{      
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hc_plane_size = temp_inp_h * temp_inp_c;
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            p_src1 = p_scratch + (((itr_n * hc_plane_size) + itr_h * temp_inp_c) * (!flag)) + (flag)*((itr_n * plane_size) + (itr_h * wc_plane_size));
            for(itr_w=0; itr_w < (temp_inp_w & ~(2 - 1)); itr_w += 2)
            {      
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
              p_src3 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + ((itr_w + 1) * temp_inp_c);
              p_dst = p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c;

              vecmax16_inpx3(p_src1,p_src2,p_src3,p_dst,temp_inp_c);

              p_src1 = p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c);
            }

            if(temp_inp_w & 1)
            {       
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
              p_dst = p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c;

              vecmax16_inpx2(p_src1, p_src2, p_dst, temp_inp_c);
              p_src1 = p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c);
            }
          }
        }
        temp_inp_w = 1;
        }break;
      case 3: {       
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hw_plane_size = temp_inp_h * temp_inp_w;
        int rem_c = (temp_inp_c & 7); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
            {
              p_src1 = p_scratch + (((itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w) * (!flag)) + ((flag) * ((itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w *temp_inp_c)));
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
              p_dst = p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w;
              ae_valign align_src2 = AE_LA64_PP(p_src2);

              for(itr_c=0; itr_c < (temp_inp_c >> 3); itr_c++)
              {
                WORD16 k1, k2, k3;
                ae_int16x4 j1, j2;
                ae_int32x2 out1, out2;
                k1 = AE_L16_I((ae_int16 *)p_src1, 0);
                AE_LA16X4_IP(j1, align_src2, (ae_int16x4 *)p_src2);
                AE_LA16X4_IP(j2, align_src2, (ae_int16x4 *)p_src2);

                k2 = AE_INT16X4_RMAX(j1);
                k3 = AE_INT16X4_RMAX(j2);
                out1 = AE_MAX32(AE_MOVDA32(k2), AE_MOVDA32(k3));
                out2 = AE_MAX32(k1, out1);
                *p_dst = (WORD16) AE_MOVAD32_H(out2);
                p_src1 = p_dst;                
              }

              //Remainder Loop
              #pragma no_unroll
              for(itr_c=0; itr_c < rem_c; itr_c++)
              {
                WORD16 k1, k2;
                ae_int16x4 j1;
                ae_int32x2 out1;
                k1 = AE_L16_I((ae_int16 *)p_src1, 0);
                AE_L16_IP(j1, (ae_int16 *)p_src2, 2);
                k2 = AE_INT16X4_RMAX(j1);
                out1 = AE_MAX32(AE_MOVDA32(k1), AE_MOVDA32(k2));
                *p_dst = (WORD16) AE_MOVAD32_H(out1);
                p_src1 = p_dst;      
              }
            }
          }
        }
        temp_inp_c = 1;
        }break;
      default:
        return -1;
        break;
    }

    p_in = p_scratch;
    flag = 1;
  }
  if(num_axis_dims)
  {
    memcpy(p_out, p_scratch, out_length*sizeof(WORD16)); 
  }
  else
  {
    memcpy(p_out, p_inp, inp_length*sizeof(WORD16)); 
  }

  return 0;
}

static void vecmean16_inpx3(const ae_int32x2 *p_src1, const WORD16* p_src2, const WORD16* p_src3, ae_int32x2 *p_dst, int N){
  int i = 0;
  ae_int32x2 ONE32 = AE_MOVDA32(1);
  ae_valign align_src1, align_dst;
  ae_valign align_src2, align_src3;
  align_src1 = AE_LA64_PP(p_src1);
  align_src2 = AE_LA64_PP(p_src2);
  align_src3 = AE_LA64_PP(p_src3);
  align_dst = AE_ZALIGN64();

  for(i=0; i < (N >> 2); i++)
  {
    ae_int16x4 j1, j2;
    ae_int32x2 j1_32_h, j1_32_l;
    ae_int32x2 j2_32_h, j2_32_l;
    ae_int32x2 wj1_32_h, wj1_32_l;

    ae_int32x2 wout1, wout2;
    AE_LA32X2_IP(wout1, align_src1, p_src1);
    AE_LA32X2_IP(wout2, align_src1, p_src1);

    AE_LA16X4_IP(j1, align_src2, (ae_int16x4 *)p_src2);
    AE_LA16X4_IP(j2, align_src3, (ae_int16x4 *)p_src3);

    j1_32_h = AE_SEXT32X2D16_32(j1);
    j1_32_l = AE_SEXT32X2D16_10(j1);
    j2_32_h = AE_SEXT32X2D16_32(j2);
    j2_32_l = AE_SEXT32X2D16_10(j2);     

    wj1_32_h = AE_ADD32(j1_32_h, j2_32_h);   
    wj1_32_l = AE_ADD32(j1_32_l, j2_32_l);       

    AE_MULAP32X2(wout1, wj1_32_h, ONE32);
    AE_MULAP32X2(wout2, wj1_32_l, ONE32);

    AE_SA32X2_IP(wout1, align_dst, p_dst);
    AE_SA32X2_IP(wout2, align_dst, p_dst);
  }
  AE_SA64POS_FP(align_dst, p_dst); // finalize the stream

  //Remainder Loop
  for(i=0; i < (N & 3); i++)
  {
    ae_int32x2 j1, j2;
    ae_int32x2 wj1;
    ae_int32x2 wout1;
    AE_L32_IP(wout1, (ae_int32 *)p_src1, sizeof(WORD32));
    j1 = (WORD32) *(p_src2 + i);
    j2 = (WORD32) *(p_src3 + i);
    wj1 = AE_ADD32(j1, j2);
    wout1 = AE_ADD32(wout1, wj1);
    AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
  }
}

static void vecmean16_inpx2(const ae_int32x2 *p_src1, const WORD16* p_src2, ae_int32x2 *p_dst, int N){
  ae_int32x2 ONE32 = AE_MOVDA32(1);
  ae_valign align_src1, align_dst;
  ae_valign align_src2;
  align_src1 = AE_LA64_PP(p_src1);
  align_src2 = AE_LA64_PP(p_src2);
  align_dst = AE_ZALIGN64();

  int i = 0;
  for(i=0; i < (N >> 2); i++)
  {
    ae_int16x4 j1;
    ae_int32x2 j1_32_h, j1_32_l;
    ae_int32x2 wout1, wout2;
    AE_LA32X2_IP(wout1, align_src1, p_src1);
    AE_LA32X2_IP(wout2, align_src1, p_src1);

    AE_LA16X4_IP(j1, align_src2, (ae_int16x4 *)p_src2);

    j1_32_h = AE_SEXT32X2D16_32(j1);
    j1_32_l = AE_SEXT32X2D16_10(j1);

    AE_MULAP32X2(wout1, j1_32_h, ONE32);
    AE_MULAP32X2(wout2, j1_32_l, ONE32);

    AE_SA32X2_IP(wout1, align_dst, p_dst);
    AE_SA32X2_IP(wout2, align_dst, p_dst);
  }
  AE_SA64POS_FP(align_dst, p_dst); // finalize the stream

  //Remainder Loop
  for(i=0; i < (N & 3); i++)
  {
    ae_int32x2 j1;
    ae_int32x2 wout1;
    AE_L32_IP(wout1, (ae_int32 *)p_src1, sizeof(WORD32));
    j1 = (WORD32) *(p_src2 + i);
    wout1 = AE_ADD32(wout1, j1);
    AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
  }
}

static void vecmean32_inpx3(const ae_int32x2* p_src1, const ae_int32x2* p_wsrc2, const ae_int32x2* p_wsrc3, ae_int32x2 *p_dst, int N){
  ae_valign align_src1, align_src2, align_src3, align_dst;
  align_src1 = AE_LA64_PP(p_src1);
  align_src2 = AE_LA64_PP(p_wsrc2);
  align_src3 = AE_LA64_PP(p_wsrc3);
  align_dst = AE_ZALIGN64();

  int i = 0;
  for(i=0; i < (N >> 2); i++)
  {
    ae_int32x2 j1, j2, j3, j4;
    ae_int32x2 wj1, wj2;
    ae_int32x2 wout1, wout2;
    AE_LA32X2_IP(wout1, align_src1, p_src1);
    AE_LA32X2_IP(wout2, align_src1, p_src1);
    AE_LA32X2_IP(j1, align_src2, p_wsrc2);
    AE_LA32X2_IP(j2, align_src3, p_wsrc3);
    AE_LA32X2_IP(j3, align_src2, p_wsrc2);
    AE_LA32X2_IP(j4, align_src3, p_wsrc3);
    wj1 = AE_ADD32S(j1, j2);
    wj2 = AE_ADD32S(j3, j4);
    wout1 = AE_ADD32S(wout1, wj1);
    wout2 = AE_ADD32S(wout2, wj2);
    AE_SA32X2_IP(wout1, align_dst, p_dst);
    AE_SA32X2_IP(wout2, align_dst, p_dst);
  }
  AE_SA64POS_FP(align_dst, p_dst); // finalize the stream

  //Remainder Loop
  for(i=0; i < (N & 3); i++)
  {
    ae_int32x2 j1, j2;
    ae_int32x2 wj1;
    ae_int32x2 wout1;
    AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
    AE_L32_IP(j1, (ae_int32 *)p_wsrc2, 4);
    AE_L32_IP(j2, (ae_int32 *)p_wsrc3, 4);
    wj1 = AE_ADD32S(j1, j2);
    wout1 = AE_ADD32S(wout1, wj1);
    AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
  }
}

static void vecmean32_inpx2(const ae_int32x2* p_src1, const ae_int32x2* p_wsrc2, ae_int32x2 *p_dst, int N){
  ae_valign align_src1, align_src2, align_dst;
  align_src1 = AE_LA64_PP(p_src1);
  align_src2 = AE_LA64_PP(p_wsrc2);
  align_dst = AE_ZALIGN64();

  int i = 0;
  for(i=0; i < (N >> 2); i++)
  {
    ae_int32x2 j1, j2;
    ae_int32x2 wout1, wout2;
    AE_LA32X2_IP(wout1, align_src1, p_src1);
    AE_LA32X2_IP(wout2, align_src1, p_src1);
    AE_LA32X2_IP(j1, align_src2, p_wsrc2);
    AE_LA32X2_IP(j2, align_src2, p_wsrc2);
    wout1 = AE_ADD32S(wout1, j1);
    wout2 = AE_ADD32S(wout2, j2);
    AE_SA32X2_IP(wout1, align_dst, p_dst);
    AE_SA32X2_IP(wout2, align_dst, p_dst);
  }
  AE_SA64POS_FP(align_dst, p_dst); // finalize the stream

  //Remainder Loop
  for(i=0; i < (N & 3); i++)
  {
    ae_int32x2 j1;
    ae_int32x2 wout1;
    AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
    AE_L32_IP(j1, (ae_int32 *)p_wsrc2, 4);
    wout1 = AE_ADD32S(wout1, j1);
    AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
  }
}

static inline void xa_nn_reduce_sum_4D_asym16s_asym16s(const WORD16 * __restrict__ p_inp
                                                       ,const WORD32 *const p_4D_inp_shape
                                                       ,const WORD32 * __restrict__ p_axis_data
                                                       ,WORD32 num_inp_dims
                                                       ,WORD32 num_axis_dims
                                                       ,pVOID p_scratch_in)
{
  WORD16 *p_in = (WORD16 *)(p_inp);
  WORD32 *p_scratch = (WORD32 *)(p_scratch_in);

  int temp_inp_n = p_4D_inp_shape[0]; 
  int temp_inp_h = p_4D_inp_shape[1]; 
  int temp_inp_w = p_4D_inp_shape[2]; 
  int temp_inp_c = p_4D_inp_shape[3];

  int itr_axis = 0, itr_n = 0, itr_h = 0, itr_w = 0, itr_c = 0;
  WORD16 *p_src2, *p_src3;
  ae_int32x2 *p_src1;
  ae_int32x2 * p_dst;
  //ae_valign align_dst;
  ae_valign align_src2;
  //align_dst = AE_ZALIGN64();

  int axis_dims_count = num_axis_dims;
  if(axis_dims_count)
  {
    switch(p_axis_data[itr_axis])
    {
      case 0: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        for(itr_n=0; itr_n < (temp_inp_n & ~(2 - 1)); itr_n += 2)
        {
          p_src1 = (ae_int32x2 *)(p_scratch);
          p_src2 = p_in + itr_n * plane_size;
          p_src3 = p_in + (itr_n + 1) * plane_size;
          p_dst  = (ae_int32x2 *)(p_scratch);
          vecmean16_inpx3(p_src1, p_src2, p_src3, p_dst, plane_size);
        }

        if(temp_inp_n & 1)
        {
          p_src1 = (ae_int32x2 *)(p_scratch);
          p_src2 = (p_in + itr_n * plane_size);
          p_dst  = (ae_int32x2 *)(p_scratch);
          vecmean16_inpx2(p_src1, p_src2, p_dst, plane_size);
        }
        temp_inp_n = 1;  
        }break;
      case 1: {     
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size)); 
          for(itr_h=0; itr_h < (temp_inp_h & ~(2 - 1)); itr_h += 2)
          {
            p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
            p_src3 = p_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size);
            p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
            vecmean16_inpx3(p_src1, p_src2, p_src3, p_dst, wc_plane_size);
            p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
          }

          if(temp_inp_h & 1)
          {
            p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
            p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
            vecmean16_inpx2(p_src1, p_src2, p_dst, wc_plane_size);
          }
        }
        temp_inp_h = 1;
        }break;
      case 2:{                    
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hc_plane_size = temp_inp_h * temp_inp_c;
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            p_src1 = (ae_int32x2 *)(p_scratch + (((itr_n * hc_plane_size) + itr_h * temp_inp_c))); 
            for(itr_w=0; itr_w < (temp_inp_w & ~(2 - 1)); itr_w += 2)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
              p_src3 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + ((itr_w + 1) * temp_inp_c);
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              vecmean16_inpx3(p_src1, p_src2, p_src3, p_dst, temp_inp_c);
              p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
            }

            if(temp_inp_w & 1)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              vecmean16_inpx2(p_src1, p_src2, p_dst, temp_inp_c);
            }
          }
        }
        temp_inp_w = 1;
        }break;
      case 3: {                  
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hw_plane_size = temp_inp_h * temp_inp_w;
        int rem_c = (temp_inp_c & 7); 

        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
            {
              p_src1 = (ae_int32x2 *)(p_scratch + (((itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w)));
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w);
              align_src2 = AE_LA64_PP(p_src2);
              ae_int16x4 one_16x4 = AE_MOVDA16(1);
              for(itr_c=0; itr_c < (temp_inp_c >> 3); itr_c++)
              {
                ae_int16x4 j1, j2;
                ae_int32x2 i1, i2;
                ae_int64 out1, out2;
                i1 = AE_L32_I((ae_int32 *)p_src1, 0);
                AE_LA16X4_IP(j1, align_src2, (ae_int16x4 *)p_src2);
                AE_LA16X4_IP(j2, align_src2, (ae_int16x4 *)p_src2);
                out1 = AE_MULZAAAAQ16(j1, one_16x4);
                out2 = AE_MULZAAAAQ16(j2, one_16x4);
                i2 = AE_ADD32S(AE_MOVDA32(AE_MOVINT32_FROMINT64(out1)),AE_MOVDA32(AE_MOVINT32_FROMINT64(out2)));
                i1 = AE_ADD32S(i1, i2);
                AE_S32_L_I(i1, (ae_int32 *)p_dst, 0);
                p_src1 = p_dst;
              }
              //Remainder Loop
              for(itr_c=0; itr_c < rem_c ; itr_c++)
              {
                WORD16 j1;
                ae_int32x2 i1;
                i1 = AE_L32_I((ae_int32 *)p_src1, 0);
                j1 = (WORD32) *(WORD16 *)p_src2;
                p_src2++;
                i1 = AE_ADD32S(i1, AE_MOVDA32(j1));
                AE_S32_L_I(i1, (ae_int32 *)p_dst, 0);
              }
            }
          }
        }
        temp_inp_c = 1;
        }break;
      default:
        break;
    }

    axis_dims_count--;
    itr_axis++;
  }

  while(axis_dims_count)
  {
    ae_valign align_src;
    WORD32 *p_scr_in =(WORD32 *)p_scratch;
    ae_int32x2 *p_wsrc2, *p_wsrc3;
    switch(p_axis_data[itr_axis])
    {
      case 0: {              
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        for(itr_n=1; itr_n < ((temp_inp_n -1) & ~(2 - 1)); itr_n += 2)
        {
          p_src1 = (ae_int32x2 *)(p_scratch);
          p_wsrc2 = (ae_int32x2 *)(p_scr_in + itr_n * plane_size);
          p_wsrc3 = (ae_int32x2 *)(p_scr_in + (itr_n + 1) * plane_size);
          p_dst  = (ae_int32x2 *)(p_scratch);
          vecmean32_inpx3(p_src1, p_wsrc2, p_wsrc3, p_dst, plane_size);
        }

        if((temp_inp_n - 1) & 1)
        {
          p_src1 = (ae_int32x2 *)(p_scratch);
          p_wsrc2 = (ae_int32x2 *)(p_scr_in + itr_n * plane_size);
          p_dst  = (ae_int32x2 *)(p_scratch);
          vecmean32_inpx2(p_src1, p_wsrc2, p_dst, plane_size);
        }
        temp_inp_n = 1;
        }break;
      case 1: {            
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          p_src1 = (ae_int32x2 *)(p_scratch + + (itr_n * plane_size));
          for(itr_h = 1; itr_h < ((temp_inp_h - 1) & ~(2 - 1)); itr_h += 2)
          {
            p_wsrc2 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
            p_wsrc3 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size));
            p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
            vecmean32_inpx3(p_src1, p_wsrc2, p_wsrc3, p_dst, wc_plane_size);
            p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
          }

          if((temp_inp_h - 1) & 1)
          {
            p_wsrc2 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
            p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
            vecmean32_inpx2(p_src1, p_wsrc2, p_dst, plane_size);
          }
        }
        temp_inp_h = 1;
        }break;
      case 2:{                
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hc_plane_size = temp_inp_h * temp_inp_c;
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            p_src1 = (ae_int32x2 *)(p_scratch + ((itr_n * plane_size) + (itr_h * wc_plane_size)));
            for(itr_w = 1; itr_w < ((temp_inp_w - 1) & ~(2 - 1)); itr_w += 2)
            {
              p_wsrc2 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_wsrc3 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + ((itr_w + 1) * temp_inp_c));
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              vecmean32_inpx3(p_src1, p_wsrc2, p_wsrc3, p_dst, temp_inp_c);
              p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
            }

            if((temp_inp_w - 1) & 1)
            {
              p_wsrc2 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              vecmean32_inpx2(p_src1, p_wsrc2, p_dst, temp_inp_c);
            }
          }
        }
        temp_inp_w = 1;
        }break;
      case 3: {              
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hw_plane_size = temp_inp_h * temp_inp_w;
        int rem_c = ((temp_inp_c) & 3); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
            {
              p_wsrc2 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w);
              align_src = AE_LA64_PP(p_wsrc2);
              ae_int32x2 i1 = AE_ZERO32();
              for(itr_c = 0; itr_c < (temp_inp_c >> 2); itr_c++)
              {
                ae_int32x2 j1, j2;
                ae_int32 out1, out2;
                AE_LA32X2_IP(j1, align_src, p_wsrc2);
                AE_LA32X2_IP(j2, align_src, p_wsrc2);
                out1 = AE_INT32X2_RADD(j1);
                out2 = AE_INT32X2_RADD(j2);
                i1 = AE_ADD32S(i1, AE_MOVDA32(out1));
                i1 = AE_ADD32S(i1, AE_MOVDA32(out2));
              }

              //Remainder Loop
              for(itr_c=0; itr_c < rem_c; itr_c++)
              {
                ae_int32x2 j1;
                AE_L32_IP(j1, (ae_int32 *)p_wsrc2, sizeof(WORD32));
                i1 = AE_ADD32S(i1, j1);
              }
              AE_S32_L_I(i1, (ae_int32 *)p_dst, 0);
            }
          }
        }
        temp_inp_c = 1;
        }break;
      default:
        break;
    }
    axis_dims_count--;
    itr_axis++;
  }
}

WORD32 xa_nn_reduce_mean_4D_asym16s_asym16s(WORD16 * __restrict__ p_out
                                            ,const WORD32 *const p_out_shape
                                            ,const WORD16 * __restrict__ p_inp
                                            ,const WORD32 *const p_inp_shape
                                            ,const WORD32 * __restrict__ p_axis
                                            ,WORD32 num_out_dims
                                            ,WORD32 num_inp_dims
                                            ,WORD32 num_axis_dims
                                            ,WORD32 inp_zero_bias
                                            ,WORD32 out_multiplier
                                            ,WORD32 out_shift
                                            ,WORD32 out_zero_bias
                                            ,void * __restrict__ p_scratch_in)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_axis, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);

  /* Invalid input checks */
  XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND(((num_out_dims <= 0) || (num_out_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND(((num_axis_dims < 0) || (num_axis_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND((inp_zero_bias < -32768 || inp_zero_bias > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -32768 || out_zero_bias > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int axis_itr = 0, inp_itr = 0, out_itr = 0;
  int num_elm_in_axis = 1;
  int current, past = -1;
  for(axis_itr=0; axis_itr < num_axis_dims; axis_itr++)
  {
    current = p_axis[axis_itr];
    XA_NNLIB_ARG_CHK_COND(((current < 0) || (current > (num_inp_dims - 1))), -1);
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[current] > 1024), -1);

    /* Avoid calculation in case of repeated axis dims*/
    if(current != past)
    {
      num_elm_in_axis *= p_inp_shape[current];
      past = current;
    }
  }

  for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[inp_itr] <= 0), -1);
  }

  int out_length = 1;
  for(out_itr=0; out_itr < num_out_dims; out_itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shape[out_itr] <= 0), -1);
    out_length *= p_out_shape[out_itr];
  }

  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_axis, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);

  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift < 0 ? 0 : out_shift;
  right_shift = out_shift > 0 ? 0 : -out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  WORD16 *p_in = (WORD16 *)(p_inp);
  WORD32 *p_scratch = (WORD32 *)(ALIGN_PTR(p_scratch_in, ALIGNMENT_8));

  // Changing order of axis data so that reduce max will be first computed
  // across largest inp shape dim in axis. This is required to
  // minimize the scratch usage.
  int inp_length = 1, p_axis_data[4], inp_shape_max;
  if(num_axis_dims)
  {
    inp_shape_max = p_inp_shape[p_axis[0]];
    axis_itr = 1;
    int max_axis_itr = 0;
    int temp_p_axis_0 = p_axis[0];
    for(axis_itr = 0; axis_itr < num_axis_dims; axis_itr++)
    {
      p_axis_data[axis_itr] = p_axis[axis_itr];
    }
    for(axis_itr = 1; axis_itr < num_axis_dims; axis_itr++)
    {
      if(p_inp_shape[p_axis[axis_itr]] > inp_shape_max)
      {
        inp_shape_max = p_inp_shape[p_axis[axis_itr]];
        max_axis_itr = axis_itr;
      }
    }
    p_axis_data[0] = p_axis_data[max_axis_itr];
    p_axis_data[max_axis_itr] = temp_p_axis_0;

    inp_itr = 0;
    for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
    {
      inp_length *= p_inp_shape[inp_itr];
    }

    memset(p_scratch, 0, ((inp_length / inp_shape_max) * sizeof(WORD32))); //TODO: Alternate approach for memset?
  }

  // Promoting lesser dim tensors to 4D tensors. Also modifying axis
  // data accordingly.
  int p_4D_inp_shape[4] = {1, 1, 1, 1};
  int itr = num_inp_dims - 1;
  int count = 3;
  while(itr >= 0)
  {
    p_4D_inp_shape[count] = p_inp_shape[itr];
    itr--;
    count--;
  }
  for(itr = 0; itr < num_axis_dims; itr++)
  {
    p_axis_data[itr] = p_axis_data[itr] + (4 - num_inp_dims);
  }
  ae_valign align_out = AE_ZALIGN64();

  if(num_axis_dims)
  {
    if(num_elm_in_axis > 1)
    { 
      xa_nn_reduce_sum_4D_asym16s_asym16s(p_in,
                                        p_4D_inp_shape,
                                        p_axis_data,
                                        num_inp_dims,
                                        num_axis_dims,
                                        p_scratch);

      xtbool same_quant = (inp_zero_bias == out_zero_bias) && (out_multiplier == 0x40000000) && (out_shift == 1);

      itr = 0;
      ae_int32x2 *p_src1 = (ae_int32x2 *)(p_scratch);

      if(same_quant)
      {
        for(itr = 0; itr < (out_length >> 3); itr++)
        {
          ae_int32x2 temp1, temp2, temp3, temp4;

          temp2 = AE_L32X2_I(p_src1, 8);
          temp3 = AE_L32X2_I(p_src1, 16);
          temp4 = AE_L32X2_I(p_src1, 24);
          AE_L32X2_IP(temp1, p_src1, 32);

          temp1 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(temp1, AE_MOVDA32(-32768)));
          temp2 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(temp2, AE_MOVDA32(-32768)));
          temp3 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(temp3, AE_MOVDA32(-32768)));
          temp4 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(temp4, AE_MOVDA32(-32768)));
          ae_int16x4 out;
          out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(temp1), AE_MOVF16X4_FROMF32X2(temp2));
          AE_SA16X4_IP(out, align_out, (ae_int16x4 *)p_out);
          out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(temp3), AE_MOVF16X4_FROMF32X2(temp4));
          AE_SA16X4_IP(out, align_out, (ae_int16x4 *)p_out);        
        }
        AE_SA64POS_FP(align_out, p_out);

        for(itr = 0; itr < (out_length & 7); itr++)
        {
          ae_int32x2 temp1;
          AE_L32_IP(temp1, (ae_int32 *)p_src1, 4);
          temp1 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(temp1, AE_MOVDA32(-32768)));
          *p_out++ = (WORD16) AE_MOVAD32_H(temp1);
        }
      }
      else
      {
        WORD64 tot_bias = (WORD64)(-inp_zero_bias) * (WORD64)(num_elm_in_axis);
        tot_bias = AE_MIN64(AE_MOVDA32(2147483647), AE_MAX64(tot_bias, AE_MOVDA32(-2147483648)));
        ae_int32x2 total_bias = AE_MOVDA32(AE_MOVINT32_FROMINT64(tot_bias));
        for(itr = 0; itr < (out_length >> 3); itr++)
        {
          ae_int32x2 wout1, wout2, wout3, wout4;
          ae_int32x2 d0_out32, d1_out32, d2_out32, d3_out32;

          wout2 = AE_L32X2_I(p_src1, 8);
          wout3 = AE_L32X2_I(p_src1, 16);
          wout4 = AE_L32X2_I(p_src1, 24);
          AE_L32X2_IP(wout1, p_src1, 32);
          wout1 = AE_ADD32S(wout1, total_bias);
          wout2 = AE_ADD32S(wout2, total_bias);
          wout3 = AE_ADD32S(wout3, total_bias);
          wout4 = AE_ADD32S(wout4, total_bias);
          
          MPY_BY_QUANT_MULT_SLS_X2_OUT32(d0_out32, wout1, out_multiplier, left_shift, right_shift);
          d0_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d0_out32);
          MPY_BY_QUANT_MULT_SLS_X2_OUT32(d1_out32, wout2, out_multiplier, left_shift, right_shift);
          d1_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d1_out32);
          MPY_BY_QUANT_MULT_SLS_X2_OUT32(d2_out32, wout3, out_multiplier, left_shift, right_shift);
          d2_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d2_out32);
          MPY_BY_QUANT_MULT_SLS_X2_OUT32(d3_out32, wout4, out_multiplier, left_shift, right_shift);
          d3_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d3_out32);

          d0_out32 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(d0_out32, AE_MOVDA32(-32768)));
          d1_out32 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(d1_out32, AE_MOVDA32(-32768)));
          d2_out32 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(d2_out32, AE_MOVDA32(-32768)));
          d3_out32 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(d3_out32, AE_MOVDA32(-32768)));

          ae_int16x4 out;
          out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d0_out32), AE_MOVF16X4_FROMF32X2(d1_out32));
          AE_SA16X4_IP(out, align_out, (ae_int16x4 *)p_out);
          out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d2_out32), AE_MOVF16X4_FROMF32X2(d3_out32));
          AE_SA16X4_IP(out, align_out, (ae_int16x4 *)p_out);             

        }
        AE_SA64POS_FP(align_out, p_out);

        for(itr = 0; itr < (out_length & 7); itr++)
        {
          ae_int32x2 wout1;
          ae_int32x2 d0_out32;

          AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
          wout1 = AE_ADD32S(wout1, total_bias);
          
          MPY_BY_QUANT_MULT_SLS_X2_OUT32(d0_out32, wout1, out_multiplier, left_shift, right_shift);
          d0_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d0_out32);

          d0_out32 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(d0_out32, AE_MOVDA32(-32768)));
          *p_out++ = (WORD16) AE_MOVAD32_H(d0_out32);
        }
      }
    }
    else
    {
      xtbool same_quant = (inp_zero_bias == out_zero_bias) && (out_multiplier == 0x40000000) && (out_shift == 1);

      itr = 0;
      ae_valign align_inp = AE_LA64_PP(p_in);

      if(same_quant)
      {
        memcpy(p_out, p_inp, inp_length * sizeof(WORD16)); 
      }
      else
      {
#pragma no_unroll
        for(itr = 0; itr < (out_length >> 2); itr++)
        {
          ae_int16x4 wout1;
          ae_int32x2 d0_out32, d1_out32;
          ae_int32x2 temp1, temp2;
          temp1 = AE_MOVDA32(-inp_zero_bias);
          temp2 = AE_MOVDA32(-inp_zero_bias);
          AE_LA16X4_IP(wout1, align_inp, (ae_int16x4 *)p_in);
          AE_MULA16X4(temp1, temp2, wout1, AE_MOVDA16(1));

          MPY_BY_QUANT_MULT_SLS_X2_OUT32(temp1, temp1, out_multiplier, left_shift, right_shift);
          d0_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), temp1);
          MPY_BY_QUANT_MULT_SLS_X2_OUT32(temp2, temp2, out_multiplier, left_shift, right_shift);
          d1_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), temp2);

          d0_out32 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(d0_out32, AE_MOVDA32(-32768)));
          d1_out32 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(d1_out32, AE_MOVDA32(-32768)));

          ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d0_out32), AE_MOVF16X4_FROMF32X2(d1_out32));
          AE_SA16X4_IP(out, align_out, (ae_int16x4 *)p_out); 
        }
        AE_SA64POS_FP(align_out, p_out);

        //Remainder Loop
        for(itr = 0; itr < (out_length & 3); itr++)
        {
          WORD16 wout1;
          ae_int32x2 d0_out32;
          ae_int32x2 temp1, temp2;
          temp1 = AE_MOVDA32(-inp_zero_bias);
          temp2 = AE_MOVDA32(-inp_zero_bias);
          wout1 = (WORD16) *(p_in + itr);
          wout1 = AE_MOVDA16(wout1);
          AE_MULA16X4(temp1, temp2, wout1, AE_MOVDA16(1));
          MPY_BY_QUANT_MULT_SLS_X2_OUT32(temp1, temp1, out_multiplier, left_shift, right_shift);
          d0_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), temp1);
          d0_out32 = AE_MIN32(AE_MOVDA32(32767), AE_MAX32(d0_out32, AE_MOVDA32(-32768)));
          *p_out++ = (WORD16) AE_MOVAD32_H(d0_out32);
        }
      }
    }
  }
  else
  {
    memcpy(p_out, p_inp, inp_length * sizeof(WORD16)); 
  }

  return 0;
}
