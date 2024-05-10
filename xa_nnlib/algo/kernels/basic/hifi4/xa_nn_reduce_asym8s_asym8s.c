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
#include "xa_nn_basic_state.h"
#include "xa_nnlib_common_macros.h"

#include <string.h>

#define ALIGNMENT_8   8   /* 8 bytes alignment */

#define ALIGNED_SIZE(x, bytes)  (((x)+(bytes-1))&(~(bytes-1)))

#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#define BUS_WIDTH_8 7


WORD32 xa_nn_reduce_getsize_nhwc(WORD32 inp_precision
                                 ,const WORD32 *const p_inp_shape
                                 ,WORD32 num_inp_dims
                                 ,const WORD32 *p_axis
                                 ,WORD32 num_axis_dims
                                 ,WORD32 reduce_ops)
{
    int scratch_bytewidth;
    /*Optimizing scratch requirement by calculating largest input dims in axis*/
    if(num_axis_dims)
    {
      int inp_shape_max = p_inp_shape[p_axis[0]];
      int axis_itr = 1;
      for(axis_itr = 1; axis_itr < num_axis_dims; axis_itr++)
      {
        inp_shape_max = (p_inp_shape[p_axis[axis_itr]] > inp_shape_max) ? p_inp_shape[p_axis[axis_itr]] : inp_shape_max;
      }

      int inp_length = 1, inp_itr = 0;
      for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
      {
        inp_length *= p_inp_shape[inp_itr];
      }

      if(reduce_ops == REDUCE_MAX) /*For Reduce Max*/
      {
        if(inp_precision == -4){
          scratch_bytewidth = sizeof(WORD8);
        }
        else if(inp_precision == -7){
          scratch_bytewidth = sizeof(WORD16);
        }
        
        if(inp_shape_max)
        {
          return (ALIGNED_SIZE(((inp_length / inp_shape_max) * scratch_bytewidth) + (BUS_WIDTH_8), ALIGNMENT_8));
        }
      }
      else if(reduce_ops == REDUCE_MEAN) /*For Reduce Mean*/
      {
        scratch_bytewidth = sizeof(WORD32);
        if(inp_shape_max)
        {
          return (ALIGNED_SIZE(((inp_length / inp_shape_max) * scratch_bytewidth) + (BUS_WIDTH_8), ALIGNMENT_8));
        }
      }
    }

    return 0;
}

#ifndef ENABLE_SCRATCH_SIZE_API_ONLY

#define STORE_8X4_FROM_16X4(out_ptr, val){\
    int o1, o2, o3, o4;\
    o1 = AE_MOVAD16_3(val);\
    o2 = AE_MOVAD16_2(val);\
    o3 = AE_MOVAD16_1(val);\
    o4 = AE_MOVAD16_0(val);\
    *out_ptr++ = (WORD8)o1;\
    *out_ptr++ = (WORD8)o2;\
    *out_ptr++ = (WORD8)o3;\
    *out_ptr++ = (WORD8)o4;\
}

static void vecmax8_inpx3_aligned(const WORD8 *p_src1, const WORD8* p_src2, const WORD8* p_src3, WORD8 *p_dst, int N){
    int i = 0;
#if XCHAL_HAVE_HIFI1
    for(i=0; i < (N >> 3); i++)
    {
      ae_int16x4 i1, j1, k1, i2, j2 ,k2;
      AE_L8X4S_IP(i1, p_src1, 4*sizeof(WORD8));
      AE_L8X4S_IP(j1, p_src2, 4*sizeof(WORD8));
      AE_L8X4S_IP(k1, p_src3, 4*sizeof(WORD8));
      AE_L8X4S_IP(i2, p_src1, 4*sizeof(WORD8));
      AE_L8X4S_IP(j2, p_src2, 4*sizeof(WORD8));
      AE_L8X4S_IP(k2, p_src3, 4*sizeof(WORD8));
      i1 = AE_MAX16(i1, j1);
      i2 = AE_MAX16(i2, j2);
      i1 = AE_MAX16(i1, k1);
      i2 = AE_MAX16(i2, k2);
      STORE_8X4_FROM_16X4(p_dst, i1);
      STORE_8X4_FROM_16X4(p_dst, i2);
     }
#else
    for(i=0; i < (N >> 3); i++)
    {
      ae_int16x4 i1, j1, k1, i2, j2 ,k2;
      xtbool4 b1, b2;
      AE_L8X4F_IP(i1, p_src1, 4*sizeof(WORD8));
      AE_L8X4F_IP(j1, p_src2, 4*sizeof(WORD8));
      AE_L8X4F_IP(k1, p_src3, 4*sizeof(WORD8));
      AE_L8X4F_IP(i2, p_src1, 4*sizeof(WORD8));
      AE_L8X4F_IP(j2, p_src2, 4*sizeof(WORD8));
      AE_L8X4F_IP(k2, p_src3, 4*sizeof(WORD8));
      b1 = AE_LT16(i1, j1);
      b2 = AE_LT16(i2, j2);
      AE_MOVT16X4(i1, j1, b1);
      AE_MOVT16X4(i2, j2, b2);
      b1 = AE_LT16(i1, k1);
      b2 = AE_LT16(i2, k2);
      AE_MOVT16X4(i1, k1, b1);
      AE_MOVT16X4(i2, k2, b2);
      i1 = AE_SRAI16(i1, 8);
      i2 = AE_SRAI16(i2, 8);
      STORE_8X4_FROM_16X4(p_dst, i1);
      STORE_8X4_FROM_16X4(p_dst, i2);
    }
#endif

    //Remainder Loop
    for(i = 0; i < (N & 7); i++)
    {
      ae_int32x2 i1, j1, k1, out1;
      i1 = (WORD32) *(p_src1 + i);
      j1 = (WORD32) *(p_src2 + i);
      k1 = (WORD32) *(p_src3 + i);
      out1 = AE_MAX32(i1, j1);
      out1 = AE_MAX32(out1, k1);
      *p_dst++ = (WORD8) AE_MOVAD32_H(out1);
    }
}
static void vecmax8_inpx2_aligned(const WORD8 *p_src1, const WORD8* p_src2, WORD8 *p_dst, int N){
    int i = 0;
#if XCHAL_HAVE_HIFI1
    for(i=0; i < (N >> 3); i++)
    {
      ae_int16x4 i1, j1, i2, j2;
      AE_L8X4S_IP(i1, p_src1, 4*sizeof(WORD8));
      AE_L8X4S_IP(j1, p_src2, 4*sizeof(WORD8));
      AE_L8X4S_IP(i2, p_src1, 4*sizeof(WORD8));
      AE_L8X4S_IP(j2, p_src2, 4*sizeof(WORD8));
      i1 = AE_MAX16(i1, j1);
      i2 = AE_MAX16(i2, j2);
      STORE_8X4_FROM_16X4(p_dst, i1);
      STORE_8X4_FROM_16X4(p_dst, i2);
    }
#else
    for(i=0; i < (N >> 3); i++)
    {
      ae_int16x4 i1, j1, i2, j2;
      xtbool4 b1, b2;
      AE_L8X4F_IP(i1, p_src1, 4*sizeof(WORD8));
      AE_L8X4F_IP(j1, p_src2, 4*sizeof(WORD8));
      AE_L8X4F_IP(i2, p_src1, 4*sizeof(WORD8));
      AE_L8X4F_IP(j2, p_src2, 4*sizeof(WORD8));
      b1 = AE_LT16(i1, j1);
      b2 = AE_LT16(i2, j2);
      AE_MOVT16X4(i1, j1, b1);
      AE_MOVT16X4(i2, j2, b2);
      i1 = AE_SRAI16(i1, 8);
      i2 = AE_SRAI16(i2, 8);
      STORE_8X4_FROM_16X4(p_dst, i1);
      STORE_8X4_FROM_16X4(p_dst, i2);
    }
#endif

    //Remainder Loop
    for(i = 0; i < (N & 7); i++)
    {
      ae_int32x2 i1, j1, out1;
      i1 = (WORD32) *(p_src1 + i);
      j1 = (WORD32) *(p_src2 + i);
      out1 = AE_MAX32(i1, j1);
      *p_dst++ = (WORD8) AE_MOVAD32_H(out1);
    }
}
#if XCHAL_HAVE_HIFI1
static void vecmax8_inpx3_unaligned(const WORD8 *p_src1, const WORD8* p_src2, const WORD8* p_src3, WORD8 * /*__restrict__*/ p_dst, int N)
{
     int i = 0;
     ae_valign align_src_in1, align_src_in2, align_src_in3, align_out;
     ae_int16x4 i1, j1, k1, i2, j2, k2;
   
     align_src_in1 = AE_LA64_PP(p_src1);
     align_src_in2 = AE_LA64_PP(p_src2);
     align_src_in3 = AE_LA64_PP(p_src3);
     align_out = AE_ZALIGN64();

     for(i=0; i< (N>>3); i++)
     {
      AE_LA8X4S_IP(i1,  align_src_in1,  p_src1);
      AE_LA8X4S_IP(j1,  align_src_in2,  p_src2);
      AE_LA8X4S_IP(k1,  align_src_in3,  p_src3);
      AE_LA8X4S_IP(i2,  align_src_in1,  p_src1);
      AE_LA8X4S_IP(j2,  align_src_in2,  p_src2);
      AE_LA8X4S_IP(k2,  align_src_in3,  p_src3);

      i1 = AE_MAX16(i1, j1);
      i2 = AE_MAX16(i2, j2);
      i1 = AE_MAX16(i1, k1);
      i2 = AE_MAX16(i2, k2);

      AE_SA8X4U_IP(i1, align_out, (ae_int32 *)p_dst);
      AE_SA8X4U_IP(i2, align_out, (ae_int32 *)p_dst);
     }
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    int rem_itr = (N&7);
    if( rem_itr > 4)
    {
      AE_LA8X4S_IP(i1,  align_src_in1,  p_src1);
      AE_LA8X4S_IP(j1,  align_src_in2,  p_src2);
      AE_LA8X4S_IP(k1,  align_src_in3,  p_src3);
      AE_LAV8X4S_XP(i2,  align_src_in1,  (ae_int8x4 *)p_src1, (rem_itr&3));
      AE_LAV8X4S_XP(j2,  align_src_in2,  (ae_int8x4 *)p_src2, (rem_itr&3));
      AE_LAV8X4S_XP(k2,  align_src_in3,  (ae_int8x4 *)p_src3, (rem_itr&3));

      i1 = AE_MAX16(i1, j1);
      i2 = AE_MAX16(i2, j2);
      i1 = AE_MAX16(i1, k1);
      i2 = AE_MAX16(i2, k2);

      AE_SA8X4U_IP(i1, align_out, (ae_int32 *)p_dst);
      AE_SAV8X4U_XP(i2, align_out, (ae_int8x4u *)p_dst, (rem_itr&3));
    }
    else if (rem_itr > 0)
    {
      AE_LAV8X4S_XP(i1,  align_src_in1,  (ae_int8x4 *)p_src1, rem_itr);
      AE_LAV8X4S_XP(j1,  align_src_in2,  (ae_int8x4 *)p_src2, rem_itr);
      AE_LAV8X4S_XP(k1,  align_src_in3,  (ae_int8x4 *)p_src3, rem_itr);

      i1 = AE_MAX16(i1, j1);
      i1 = AE_MAX16(i1, k1);

      AE_SAV8X4U_XP(i1, align_out, (ae_int8x4u *)p_dst, rem_itr);
    }
     AE_SA64POS_FP(align_out, p_dst);
#else
     AE_SA64POS_FP(align_out, p_dst);
    #pragma no_unroll
    for(i = 0; i < (N & 7); i++)
    {
        WORD32 i1, j1, k1, out1;
        i1 = (WORD32) *(p_src1+i);
        j1 = (WORD32) *(p_src2+i);
        k1 = (WORD32) *(p_src3+i);
        out1 = i1>j1 ? i1 : j1;
        out1 = out1>k1 ? out1 : k1;
        *p_dst++ = (WORD8) out1;
    }
#endif
}
#else
static void vecmax8_inpx3_unaligned(const WORD8 *p_src1, const WORD8* p_src2, const WORD8* p_src3, WORD8 * /*__restrict__*/ p_dst, int N){
    int i = 0;
    int Nby8 =  N>>3;
    int remainder_cnt = N&0x7;

    ae_valign align_src_in1, align_src_in2, align_src_in3;

    for(i=0; i < Nby8; i++){
        ae_int24x2 x1, x2, x3;
        ae_int32x2 x03_1, x03_2, x03_3, x14_1, x14_2, x14_3, x25_1, x25_2, x25_3;
        ae_int32x2 out03, out14, out25;

        align_src_in1 = AE_LA64_PP(p_src1);
        align_src_in2 = AE_LA64_PP(p_src2);
        align_src_in3 = AE_LA64_PP(p_src3);

        AE_LA24X2_IP(x1, align_src_in1, p_src1);
        AE_LA24X2_IP(x2, align_src_in2, p_src2);
        AE_LA24X2_IP(x3, align_src_in3, p_src3);

        WORD32 x6_1, x6_2, x6_3, x7_1, x7_2, x7_3, out6, out7;
        x6_1 = *p_src1++;
        x7_1 = *p_src1++;
        x6_2 = *p_src2++;
        x7_2 = *p_src2++;
        x6_3 = *p_src3++;
        x7_3 = *p_src3++;
        out6 = x6_1 > x6_2 ? x6_1 : x6_2;
        out6 = out6 > x6_3 ? out6 : x6_3;
        out7 = x7_1 > x7_2 ? x7_1 : x7_2;
        out7 = out7 > x7_3 ? out7 : x7_3;

        x03_1 = AE_SEXT32((AE_SRAI32(AE_MOVINT32X2_FROMINT24X2(x1), 16)), 7);
        x14_1 = AE_SEXT32((AE_SRAI32(AE_MOVINT32X2_FROMINT24X2(x1), 8)), 7);
        x25_1 = AE_SEXT32(AE_MOVINT32X2_FROMINT24X2(x1), 7);

        x03_2 = AE_SEXT32((AE_SRAI32(AE_MOVINT32X2_FROMINT24X2(x2), 16)), 7);
        x14_2 = AE_SEXT32((AE_SRAI32(AE_MOVINT32X2_FROMINT24X2(x2), 8)), 7);
        x25_2 = AE_SEXT32(AE_MOVINT32X2_FROMINT24X2(x2), 7);

        x03_3 = AE_SEXT32((AE_SRAI32(AE_MOVINT32X2_FROMINT24X2(x3), 16)), 7);
        x14_3 = AE_SEXT32((AE_SRAI32(AE_MOVINT32X2_FROMINT24X2(x3), 8)), 7);
        x25_3 = AE_SEXT32(AE_MOVINT32X2_FROMINT24X2(x3), 7);

        out03 = AE_MAX32(x03_1, x03_2);
        out03 = AE_MAX32(out03, x03_3);
        out14 = AE_MAX32(x14_1, x14_2);
        out14 = AE_MAX32(out14, x14_3);
        out25 = AE_MAX32(x25_1, x25_2);
        out25 = AE_MAX32(out25, x25_3);

        *p_dst++ = (WORD8) AE_MOVAD32_H(out25);
        *p_dst++ = (WORD8) AE_MOVAD32_H(out14);

        *p_dst++ = (WORD8) AE_MOVAD32_H(out03);
        *p_dst++ = (WORD8) AE_MOVAD32_L(out25);

        *p_dst++ = (WORD8) AE_MOVAD32_L(out14);
        *p_dst++ = (WORD8) AE_MOVAD32_L(out03);

        *p_dst++ = (WORD8) out6;
        *p_dst++ = (WORD8) out7;
    }
    #pragma no_unroll
    for(i = 0; i < remainder_cnt; i++){
        WORD32 i1, j1, k1, out1;
        i1 = (WORD32) *(p_src1+i);
        j1 = (WORD32) *(p_src2+i);
        k1 = (WORD32) *(p_src3+i);
        out1 = i1>j1 ? i1 : j1;
        out1 = out1>k1 ? out1 : k1;
        *p_dst++ = (WORD8) out1;
    }
}
#endif

#if XCHAL_HAVE_HIFI1
static void vecmax8_inpx2_unaligned(const WORD8 *p_src1, const WORD8* p_src2, WORD8 *p_dst, int N)
{
     int i = 0;
     ae_valign align_src_in1, align_src_in2,  align_out;
     ae_int16x4 i1, j1, i2, j2;

     align_src_in1 = AE_LA64_PP(p_src1);
     align_src_in2 = AE_LA64_PP(p_src2);
     align_out = AE_ZALIGN64();

     for(i=0; i< (N>>3); i++)
     {
      AE_LA8X4S_IP(i1,  align_src_in1,  p_src1);
      AE_LA8X4S_IP(j1,  align_src_in2,  p_src2);
      AE_LA8X4S_IP(i2,  align_src_in1,  p_src1);
      AE_LA8X4S_IP(j2,  align_src_in2,  p_src2);

      i1 = AE_MAX16(i1, j1);
      i2 = AE_MAX16(i2, j2);

      AE_SA8X4U_IP(i1, align_out, (ae_int32 *)p_dst);
      AE_SA8X4U_IP(i2, align_out, (ae_int32 *)p_dst);
     }
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
    int rem_itr = (N&7);
    if( rem_itr > 4)
    {
      AE_LA8X4S_IP(i1,  align_src_in1,  p_src1);
      AE_LA8X4S_IP(j1,  align_src_in2,  p_src2);
      AE_LAV8X4S_XP(i2,  align_src_in1,  (ae_int8x4 *)p_src1, (rem_itr&3));
      AE_LAV8X4S_XP(j2,  align_src_in2,  (ae_int8x4 *)p_src2, (rem_itr&3));

      i1 = AE_MAX16(i1, j1);
      i2 = AE_MAX16(i2, j2);

      AE_SA8X4U_IP(i1, align_out, (ae_int32 *)p_dst);
      AE_SAV8X4U_XP(i2, align_out, (ae_int8x4u *)p_dst, (rem_itr&3));
    }
    else if (rem_itr > 0)
    {
      AE_LAV8X4S_XP(i1,  align_src_in1,  (ae_int8x4 *)p_src1, rem_itr);
      AE_LAV8X4S_XP(j1,  align_src_in2,  (ae_int8x4 *)p_src2, rem_itr);

      i1 = AE_MAX16(i1, j1);

      AE_SAV8X4U_XP(i1, align_out, (ae_int8x4u *)p_dst, rem_itr);
    }
     AE_SA64POS_FP(align_out, p_dst);
#else
     AE_SA64POS_FP(align_out, p_dst);
    #pragma no_unroll

    for(i = 0; i < (N & 7); i++)
    {
        WORD32 i1, j1, out1;
        i1 = (WORD32) *(p_src1+i);
        j1 = (WORD32) *(p_src2+i);
        out1 = i1>j1 ? i1 : j1;
        *p_dst++ = (WORD8) out1;
    }
#endif
}
#else
static void vecmax8_inpx2_unaligned(const WORD8 *p_src1, const WORD8* p_src2, WORD8 *p_dst, int N){
    int i = 0;

    for(i = 0; i < (N&~3); i+=4)
    {
      UWORD32 i1, j1, i2, j2;
      UWORD32 i3, j3, i4, j4;
      ae_int32x2 iv1, jv1, out1;
      ae_int32x2 iv2, jv2, out2;

      i1 = (UWORD32) *((UWORD8 *)p_src1 + i);
      j1 = (UWORD32) *((UWORD8 *)p_src2 + i);

      i2 = (UWORD32) *((UWORD8 *)p_src1 + i + 1);
      j2 = (UWORD32) *((UWORD8 *)p_src2 + i + 1);

      i3 = (UWORD32) *((UWORD8 *)p_src1 + i + 2);
      j3 = (UWORD32) *((UWORD8 *)p_src2 + i + 2);

      i4 = (UWORD32) *((UWORD8 *)p_src1 + i + 3);
      j4 = (UWORD32) *((UWORD8 *)p_src2 + i + 3);

      iv1 = AE_SEXT32(AE_MOVDA32X2(i1, i2), 7);
      jv1 = AE_SEXT32(AE_MOVDA32X2(j1, j2), 7);

      out1 = AE_MAX32(iv1, jv1);

      iv2 = AE_SEXT32(AE_MOVDA32X2(i3, i4), 7);
      jv2 = AE_SEXT32(AE_MOVDA32X2(j3, j4), 7);

      out2 = AE_MAX32(iv2, jv2);

      *p_dst++ = (WORD8) AE_MOVAD32_H(out1);
      *p_dst++ = (WORD8) AE_MOVAD32_L(out1);

      *p_dst++ = (WORD8) AE_MOVAD32_H(out2);
      *p_dst++ = (WORD8) AE_MOVAD32_L(out2);
    }

    if(N&2)
    {
      WORD32 i1, j1, out1;
      WORD32 i2, j2, out2;
      i1 = (WORD32) *(p_src1 + i);
      j1 = (WORD32) *(p_src2 + i);
      i2 = (WORD32) *(p_src1 + i + 1);
      j2 = (WORD32) *(p_src2 + i + 1);
      out1 = i1>j1 ? i1 : j1;
      out2 = i2>j2 ? i2 : j2;
      *p_dst++ = (WORD8) out1;
      *p_dst++ = (WORD8) out2;
      i+=2;
    }
    if(N&1)
    {
      WORD32 i1, j1, out1;
      i1 = (WORD32) *(p_src1 + i);
      j1 = (WORD32) *(p_src2 + i);
      out1 = i1>j1 ? i1 : j1;
      *p_dst++ = (WORD8) out1;
    }
}
#endif

/*
 * Currently only supports upto 4D input tensors.
 * 1/2/3 D input tensors will be scaled up to 4D.
 * For example, 2x3 -> 1x1x2x3.
 * Currently TFLM reduce max operator requires input and output
 * quantization to be same. Therefore, the kernel does not involve
 * quantization.
 */

WORD32 xa_nn_reduce_max_4D_asym8s_asym8s(WORD8 * __restrict__ p_out
                                        ,const WORD32 *const p_out_shape
                                        ,const WORD8 * __restrict__ p_inp
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_axis, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);

  WORD8 *p_in = (WORD8 *)(p_inp);
  WORD8 *p_scratch = (WORD8 *)(p_scratch_in);

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

    memset(p_scratch, -128, (inp_length / inp_shape_max)); //TODO: Alternate approach for memset?
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
  WORD8 *p_src1, *p_src2, *p_src3;
  WORD8 * p_dst;

  for(itr_axis=0; itr_axis < num_axis_dims; itr_axis++)
  {
    switch(p_axis_data[itr_axis])
    {
      case 0: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        if(((((unsigned)p_in) & 3) == 0) && ((((unsigned)p_scratch) & 3) == 0) && ((plane_size & 3) == 0))
        {
          for(itr_n=0; itr_n < (temp_inp_n & ~(2 - 1)); itr_n += 2)
          {
            p_src1 = p_scratch;
            p_src2 = p_in + itr_n * plane_size;
            p_src3 = p_in + (itr_n + 1) * plane_size;
            p_dst  = p_scratch;

            vecmax8_inpx3_aligned(p_src1,p_src2,p_src3,p_dst,plane_size);
          }

          if(temp_inp_n & 1)
          {
            p_src1 = p_scratch;
            p_src2 = p_in + itr_n * plane_size;
            p_dst  = p_scratch;

            vecmax8_inpx2_aligned(p_src1,p_src2,p_dst,plane_size);
          }
        }
        else
        {
          for(itr_n=0; itr_n < (temp_inp_n & ~(2 - 1)); itr_n += 2)
          {
            p_src1 = p_scratch;
            p_src2 = p_in + itr_n * plane_size;
            p_src3 = p_in + (itr_n + 1) * plane_size;
            p_dst  = p_scratch;

            vecmax8_inpx3_unaligned(p_src1,p_src2,p_src3,p_dst,plane_size);
          }
          if(temp_inp_n & 1)
          {
            p_src1 = p_scratch;
            p_src2 = p_in + itr_n * plane_size;
            p_dst  = p_scratch;
            vecmax8_inpx2_unaligned(p_src1, p_src2, p_dst, plane_size);
          }
        }
        temp_inp_n = 1;  
        }break;
      case 1: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        if(((((unsigned)p_in) & 3) == 0) && ((((unsigned)p_scratch) & 3) == 0) && ((wc_plane_size & 3) == 0))
        {
          for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
          {
            p_src1 = p_scratch + (itr_n * wc_plane_size * (!flag)) + (flag * itr_n * plane_size);
            for(itr_h=0; itr_h < (temp_inp_h & ~(2 - 1)); itr_h += 2)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
              p_src3 = p_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size);
              p_dst = p_scratch + (itr_n * wc_plane_size);

              vecmax8_inpx3_aligned(p_src1,p_src2,p_src3,p_dst,wc_plane_size);
              p_src1 = p_scratch + (itr_n * wc_plane_size);
            }

            if(temp_inp_h & 1)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
              p_dst = p_scratch + (itr_n * wc_plane_size);

              vecmax8_inpx2_aligned(p_src1,p_src2,p_dst,wc_plane_size);
              p_src1 = p_scratch + (itr_n * wc_plane_size);
            }
          }
        }
        else
        {
          for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
          {
            p_src1 = p_scratch + (itr_n * wc_plane_size * (!flag)) + (flag * itr_n * plane_size);
            for(itr_h=0; itr_h < (temp_inp_h & ~(2 - 1)); itr_h += 2)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
              p_src3 = p_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size);
              p_dst = p_scratch + (itr_n * wc_plane_size);

              vecmax8_inpx3_unaligned(p_src1,p_src2,p_src3,p_dst,wc_plane_size);
              p_src1 = p_scratch + (itr_n * wc_plane_size);
            }

            if(temp_inp_h & 1)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
              p_dst = p_scratch + (itr_n * wc_plane_size);

              vecmax8_inpx2_unaligned(p_src1, p_src2, p_dst, wc_plane_size);
              p_src1 = p_scratch + (itr_n * wc_plane_size);
            }
          }
        }
        temp_inp_h = 1;
        }break;
      case 2:{
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hc_plane_size = temp_inp_h * temp_inp_c;
        if(((((unsigned)p_in) & 3) == 0) && ((((unsigned)p_scratch) & 3) == 0) && ((temp_inp_c & 3) == 0))
        {
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

                vecmax8_inpx3_aligned(p_src1,p_src2,p_src3,p_dst,temp_inp_c);
                p_src1 = p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c);
              }

              if(temp_inp_w & 1)
              {
                p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
                p_dst = p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c;

                vecmax8_inpx2_aligned(p_src1,p_src2,p_dst,temp_inp_c);
                p_src1 = p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c);
              }
            }
          }
        }
        else
        {
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

                vecmax8_inpx3_unaligned(p_src1,p_src2,p_src3,p_dst,temp_inp_c);

                p_src1 = p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c);
              }

              if(temp_inp_w & 1)
              {
                p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
                p_dst = p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c;

                vecmax8_inpx2_unaligned(p_src1, p_src2, p_dst, temp_inp_c);
                p_src1 = p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c);
              }
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
        if(((((unsigned)p_in) & 3) == 0) && ((temp_inp_c & 3) == 0))
        {
          for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
          {
            for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
            {
              for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
              {
                p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
                p_dst = p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w;

                ae_int32x2 k1 = AE_MOVDA32(-128);
                for(itr_c=0; itr_c < (temp_inp_c >> 3); itr_c++)
                {
                  WORD16 k2, k3;
                  ae_int16x4 j1, j2;
                  ae_int32x2 out1;
                  AE_L8X4F_IP(j1, p_src2, 4*sizeof(WORD8));
                  AE_L8X4F_IP(j2, p_src2, 4*sizeof(WORD8));
                  k2 = AE_INT16X4_RMAX(j1);
                  k3 = AE_INT16X4_RMAX(j2);
                  out1 = AE_MAX32(AE_MOVDA32(k2), AE_MOVDA32(k3));
                  k1 = AE_MAX32(k1, AE_SRAI32(out1, 8));
                }

                //Remainder Loop
                #pragma no_unroll
                for(itr_c=0; itr_c < rem_c; itr_c++)
                {
                  ae_int32x2 j1;
                  j1 = (WORD32) *(p_src2 + itr_c);
                  k1 = AE_MAX32(k1, j1);
                }
                *p_dst = (WORD8) AE_MOVAD32_H(k1);
              }
            }
          }
        }
        else
        {
          int i = 0;
          for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
          {
            for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
            {
              for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
              {
                p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
                p_dst = p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w;

                ae_int32x2 k1 = AE_MOVDA32(-128);
                for(i=0; i < (temp_inp_c & ~3); i+=4)
                {
                  UWORD32 j1, j2, j3, j4;
                  ae_int32x2 jv1, jv2, out1;
                  j1 = (UWORD32) *((UWORD8 *)p_src2 + i);
                  j2 = (UWORD32) *((UWORD8 *)p_src2 + i + 1);
                  j3 = (UWORD32) *((UWORD8 *)p_src2 + i + 2);
                  j4 = (UWORD32) *((UWORD8 *)p_src2 + i + 3);

                  jv1 = AE_SLAI32(AE_MOVDA32X2(j1, j2), 24);
                  jv2 = AE_SLAI32(AE_MOVDA32X2(j3, j4), 24);
                  out1 = AE_MAX32(jv1, jv2);
                  out1 = AE_INT32X2_RMAX(out1);
                  k1 = AE_MAX32(k1, AE_SRAI32(out1, 24));
                }

                //Remainder Loop
                #pragma no_unroll
                for(; i < temp_inp_c; i++)
                {
                  ae_int32x2 j1;
                  j1 = (WORD32) *(p_src2 + i);
                  k1 = AE_MAX32(k1, j1);
                }
                *p_dst = (WORD8) AE_MOVAD32_H(k1);
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
    xa_nn_memcpy(p_out, p_scratch, out_length); 
  }
  else
  {
    xa_nn_memcpy(p_out, p_inp, inp_length); 
  }

  return 0;
}

static void vecmean8_inpx3_aligned(const ae_int32x2 *p_src1, const WORD8* p_src2, const WORD8* p_src3, ae_int32x2 *p_dst, int N){
  int i = 0;
  ae_int16x4 ONE16 = AE_MOVDA16(1);
  for(i=0; i < (N >> 3); i++)
  {
    ae_int16x4 j1, j2, j3, j4;
    ae_int16x4 wj1, wj2;
    ae_int32x2 wout1, wout2, wout3, wout4;
    AE_L32X2_IP(wout1, p_src1, 2*sizeof(WORD32));
    AE_L32X2_IP(wout2, p_src1, 2*sizeof(WORD32));
    AE_L32X2_IP(wout3, p_src1, 2*sizeof(WORD32));
    AE_L32X2_IP(wout4, p_src1, 2*sizeof(WORD32));
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(j1, p_src2, 4*sizeof(WORD8));
    AE_L8X4S_IP(j2, p_src3, 4*sizeof(WORD8));
    AE_L8X4S_IP(j3, p_src2, 4*sizeof(WORD8));
    AE_L8X4S_IP(j4, p_src3, 4*sizeof(WORD8));
#else
    AE_L8X4F_IP(j1, p_src2, 4*sizeof(WORD8));
    AE_L8X4F_IP(j2, p_src3, 4*sizeof(WORD8));
    AE_L8X4F_IP(j3, p_src2, 4*sizeof(WORD8));
    AE_L8X4F_IP(j4, p_src3, 4*sizeof(WORD8));
    j1 = AE_SRAI16(j1, 8);
    j2 = AE_SRAI16(j2, 8);
    j3 = AE_SRAI16(j3, 8);
    j4 = AE_SRAI16(j4, 8);
#endif
    wj1 = AE_ADD16(j1, j2);
    wj2 = AE_ADD16(j3, j4);
    AE_MULA16X4(wout1, wout2, wj1, ONE16);
    AE_MULA16X4(wout3, wout4, wj2, ONE16);
    AE_S32X2_IP(wout1, p_dst, 2*sizeof(WORD32));
    AE_S32X2_IP(wout2, p_dst, 2*sizeof(WORD32));
    AE_S32X2_IP(wout3, p_dst, 2*sizeof(WORD32));
    AE_S32X2_IP(wout4, p_dst, 2*sizeof(WORD32));
  }
  //Remainder Loop
  for(i=0; i < (N & 7); i++)
  {
    ae_int32x2 j1, j2;
    ae_int32x2 wj1;
    ae_int32x2 wout1;
    AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
    j1 = (WORD32) *(p_src2 + i);
    j2 = (WORD32) *(p_src3 + i);
    wj1 = AE_ADD32(j1, j2);
    wout1 = AE_ADD32(wout1, wj1);
    AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
  }
}
static void vecmean8_inpx2_aligned(const ae_int32x2 *p_src1, const WORD8* p_src2, ae_int32x2 *p_dst, int N){
  int i = 0;
  ae_int16x4 ONE16 = AE_MOVDA16(1);
  for(i=0; i < (N >> 3); i++)
  {
    ae_int16x4 j1, j2;
    ae_int32x2 wout1, wout2, wout3, wout4;
    AE_L32X2_IP(wout1, p_src1, 2*sizeof(WORD32));
    AE_L32X2_IP(wout2, p_src1, 2*sizeof(WORD32));
    AE_L32X2_IP(wout3, p_src1, 2*sizeof(WORD32));
    AE_L32X2_IP(wout4, p_src1, 2*sizeof(WORD32));
#if XCHAL_HAVE_HIFI1
    AE_L8X4S_IP(j1, p_src2, 4*sizeof(WORD8));
    AE_L8X4S_IP(j2, p_src2, 4*sizeof(WORD8));
#else
    AE_L8X4F_IP(j1, p_src2, 4*sizeof(WORD8));
    AE_L8X4F_IP(j2, p_src2, 4*sizeof(WORD8));
    j1 = AE_SRAI16(j1, 8);
    j2 = AE_SRAI16(j2, 8);
#endif
    AE_MULA16X4(wout1, wout2, j1, ONE16);
    AE_MULA16X4(wout3, wout4, j2, ONE16);
    AE_S32X2_IP(wout1, p_dst, 2*sizeof(WORD32));
    AE_S32X2_IP(wout2, p_dst, 2*sizeof(WORD32));
    AE_S32X2_IP(wout3, p_dst, 2*sizeof(WORD32));
    AE_S32X2_IP(wout4, p_dst, 2*sizeof(WORD32));
  }

  //Remainder Loop
  for(i=0; i < (N & 7); i++)
  {
    ae_int32x2 j1;
    ae_int32x2 wout1;
    AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
    j1 = (WORD32) *(p_src2 + i);
    wout1 = AE_ADD32(wout1, j1);
    AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
  }
}
static void vecmean8_inpx3_unaligned(const ae_int32x2 *p_src1, const WORD8* p_src2, const WORD8* p_src3, ae_int32x2 *p_dst, int N){
  int i = 0;
  ae_int16x4 ONE16 = AE_MOVDA16(1);
  ae_valign align_src1, align_dst;
  ALIGN_REGISTER_TYPE align_src2, align_src3;
  align_src1 = AE_LA64_PP(p_src1);
  PRIME_8X4F(p_src2, align_src2);
  PRIME_8X4F(p_src3, align_src3);
  align_dst = AE_ZALIGN64();

  for(i=0; i < (N >> 3); i++)
  {
    ae_int16x4 j1, j2, j3, j4;
    ae_int16x4 wj1, wj2;
    ae_int32x2 wout1, wout2, wout3, wout4;
    AE_LA32X2_IP(wout1, align_src1, p_src1);
    AE_LA32X2_IP(wout2, align_src1, p_src1);
    AE_LA32X2_IP(wout3, align_src1, p_src1);
    AE_LA32X2_IP(wout4, align_src1, p_src1);
#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(j1, align_src2, p_src2);
    AE_LA8X4S_IP(j2, align_src3, p_src3);
    AE_LA8X4S_IP(j3, align_src2, p_src2);
    AE_LA8X4S_IP(j4, align_src3, p_src3);
#else
    AE_LA8X4F_IP(j1, align_src2, p_src2);
    AE_LA8X4F_IP(j2, align_src3, p_src3);
    AE_LA8X4F_IP(j3, align_src2, p_src2);
    AE_LA8X4F_IP(j4, align_src3, p_src3);
    j1 = AE_SRAI16(j1, 8);
    j2 = AE_SRAI16(j2, 8);
    j3 = AE_SRAI16(j3, 8);
    j4 = AE_SRAI16(j4, 8);
#endif
    wj1 = AE_ADD16(j1, j2);
    wj2 = AE_ADD16(j3, j4);
    AE_MULA16X4(wout1, wout2, wj1, ONE16);
    AE_MULA16X4(wout3, wout4, wj2, ONE16);
    AE_SA32X2_IP(wout1, align_dst, p_dst);
    AE_SA32X2_IP(wout2, align_dst, p_dst);
    AE_SA32X2_IP(wout3, align_dst, p_dst);
    AE_SA32X2_IP(wout4, align_dst, p_dst);
  }
  AE_SA64POS_FP(align_dst, p_dst); // finalize the stream

  //Remainder Loop
  for(i=0; i < (N & 7); i++)
  {
    ae_int32x2 j1, j2;
    ae_int32x2 wj1;
    ae_int32x2 wout1;
    AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
    j1 = (WORD32) *(p_src2 + i);
    j2 = (WORD32) *(p_src3 + i);
    wj1 = AE_ADD32(j1, j2);
    wout1 = AE_ADD32(wout1, wj1);
    AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
  }
}
static void vecmean8_inpx2_unaligned(const ae_int32x2 *p_src1, const WORD8* p_src2, ae_int32x2 *p_dst, int N){
  ae_int16x4 ONE16 = AE_MOVDA16(1);
  ae_valign align_src1, align_dst;
  ALIGN_REGISTER_TYPE align_src2;
  align_src1 = AE_LA64_PP(p_src1);
  PRIME_8X4F(p_src2, align_src2);
  align_dst = AE_ZALIGN64();

  int i = 0;
  for(i=0; i < (N >> 3); i++)
  {
    ae_int16x4 j1, j2;
    ae_int32x2 wout1, wout2, wout3, wout4;
    AE_LA32X2_IP(wout1, align_src1, p_src1);
    AE_LA32X2_IP(wout2, align_src1, p_src1);
    AE_LA32X2_IP(wout3, align_src1, p_src1);
    AE_LA32X2_IP(wout4, align_src1, p_src1);
#if XCHAL_HAVE_HIFI1
    AE_LA8X4S_IP(j1, align_src2, p_src2);
    AE_LA8X4S_IP(j2, align_src2, p_src2);
#else
    AE_LA8X4F_IP(j1, align_src2, p_src2);
    AE_LA8X4F_IP(j2, align_src2, p_src2);
    j1 = AE_SRAI16(j1, 8);
    j2 = AE_SRAI16(j2, 8);
#endif
    AE_MULA16X4(wout1, wout2, j1, ONE16);
    AE_MULA16X4(wout3, wout4, j2, ONE16);
    AE_SA32X2_IP(wout1, align_dst, p_dst);
    AE_SA32X2_IP(wout2, align_dst, p_dst);
    AE_SA32X2_IP(wout3, align_dst, p_dst);
    AE_SA32X2_IP(wout4, align_dst, p_dst);
  }
  AE_SA64POS_FP(align_dst, p_dst); // finalize the stream

  //Remainder Loop
  for(i=0; i < (N & 7); i++)
  {
    ae_int32x2 j1;
    ae_int32x2 wout1;
    AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
    j1 = (WORD32) *(p_src2 + i);
    wout1 = AE_ADD32(wout1, j1);
    AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
  }
}
static void vecmean32_inpx3_aligned(const ae_int32x2* p_src1, const ae_int32x2* p_wsrc2, const ae_int32x2* p_wsrc3, ae_int32x2 *p_dst, int N){
  int i = 0;
  for(i=0; i < (N >> 2); i++)
  {
    ae_int32x2 j1, j2, j3, j4;
    ae_int32x2 wj1, wj2;
    ae_int32x2 wout1, wout2;
    AE_L32X2_IP(wout1, p_src1, 2*sizeof(WORD32));
    AE_L32X2_IP(wout2, p_src1, 2*sizeof(WORD32));
    AE_L32X2_IP(j1, p_wsrc2, 2*sizeof(WORD32));
    AE_L32X2_IP(j2, p_wsrc3, 2*sizeof(WORD32));
    AE_L32X2_IP(j3, p_wsrc2, 2*sizeof(WORD32));
    AE_L32X2_IP(j4, p_wsrc3, 2*sizeof(WORD32));
    wj1 = AE_ADD32S(j1, j2);
    wj2 = AE_ADD32S(j3, j4);
    wout1 = AE_ADD32S(wout1, wj1);
    wout2 = AE_ADD32S(wout2, wj2);
    AE_S32X2_IP(wout1, p_dst, 2*sizeof(WORD32));
    AE_S32X2_IP(wout2, p_dst, 2*sizeof(WORD32));
  }

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
static void vecmean32_inpx2_aligned(const ae_int32x2* p_src1, const ae_int32x2* p_wsrc2, ae_int32x2 *p_dst, int N){
  int i = 0;
  for(i=0; i < (N >> 2); i++)
  {
    ae_int32x2 j1, j2;
    ae_int32x2 wout1, wout2;
    AE_L32X2_IP(wout1, p_src1, 2*sizeof(WORD32));
    AE_L32X2_IP(wout2, p_src1, 2*sizeof(WORD32));
    AE_L32X2_IP(j1, p_wsrc2, 2*sizeof(WORD32));
    AE_L32X2_IP(j2, p_wsrc2, 2*sizeof(WORD32));
    wout1 = AE_ADD32S(wout1, j1);
    wout2 = AE_ADD32S(wout2, j2);
    AE_S32X2_IP(wout1, p_dst, 2*sizeof(WORD32));
    AE_S32X2_IP(wout2, p_dst, 2*sizeof(WORD32));
  }

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
static void vecmean32_inpx3_unaligned(const ae_int32x2* p_src1, const ae_int32x2* p_wsrc2, const ae_int32x2* p_wsrc3, ae_int32x2 *p_dst, int N){
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
static void vecmean32_inpx2_unaligned(const ae_int32x2* p_src1, const ae_int32x2* p_wsrc2, ae_int32x2 *p_dst, int N){
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

static inline void xa_nn_reduce_sum_4D_asym8s_asym8s(const WORD8 * __restrict__ p_inp
                                                    ,const WORD32 *const p_4D_inp_shape
                                                    ,const WORD32 * __restrict__ p_axis_data
                                                    ,WORD32 num_axis_dims
                                                    ,pVOID p_scratch_in)
{
  WORD8 *p_in = (WORD8 *)(p_inp);
  WORD32 *p_scratch = (WORD32 *)(p_scratch_in);

  int temp_inp_n = p_4D_inp_shape[0]; 
  int temp_inp_h = p_4D_inp_shape[1]; 
  int temp_inp_w = p_4D_inp_shape[2]; 
  int temp_inp_c = p_4D_inp_shape[3];

  int itr_axis = 0, itr_n = 0, itr_h = 0, itr_w = 0, itr_c = 0;
  WORD8 *p_src2, *p_src3;
  ae_int32x2 *p_src1;
  ae_int32x2 * p_dst;
//  ae_valign align_dst;
  ALIGN_REGISTER_TYPE align_src2;
//  align_dst = AE_ZALIGN64();

  int axis_dims_count = num_axis_dims;
  if(axis_dims_count)
  {
    switch(p_axis_data[itr_axis])
    {
      case 0: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        if(((((unsigned)p_in) & 3) == 0) && ((plane_size & 3) == 0))
        {
          for(itr_n=0; itr_n < (temp_inp_n & ~(2 - 1)); itr_n += 2)
          {
            p_src1 = (ae_int32x2 *)(p_scratch);
            p_src2 = p_in + itr_n * plane_size;
            p_src3 = p_in + (itr_n + 1) * plane_size;
            p_dst  = (ae_int32x2 *)(p_scratch);
            vecmean8_inpx3_aligned(p_src1, p_src2, p_src3, p_dst, plane_size);
          }

          if(temp_inp_n & 1)
          {
            p_src1 = (ae_int32x2 *)(p_scratch);
            p_src2 = (p_in + itr_n * plane_size);
            p_dst  = (ae_int32x2 *)(p_scratch);
            vecmean8_inpx2_aligned(p_src1, p_src2, p_dst, plane_size);
          }
        }
        else
        {
          for(itr_n=0; itr_n < (temp_inp_n & ~(2 - 1)); itr_n += 2)
          {
            p_src1 = (ae_int32x2 *)(p_scratch);
            p_src2 = p_in + itr_n * plane_size;
            p_src3 = p_in + (itr_n + 1) * plane_size;
            p_dst  = (ae_int32x2 *)(p_scratch);
            vecmean8_inpx3_unaligned(p_src1, p_src2, p_src3, p_dst, plane_size);
          }

          if(temp_inp_n & 1)
          {
            p_src1 = (ae_int32x2 *)(p_scratch);
            p_src2 = (p_in + itr_n * plane_size);
            p_dst  = (ae_int32x2 *)(p_scratch);
            vecmean8_inpx2_unaligned(p_src1, p_src2, p_dst, plane_size);
          }
        }
        temp_inp_n = 1;  
        }break;
      case 1: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        if(((((unsigned)p_in) & 3) == 0) && ((((unsigned)p_scratch) & 7) == 0) && ((wc_plane_size & 7) == 0))
        {
          for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
          {
            p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size)); 
            for(itr_h=0; itr_h < (temp_inp_h & ~(2 - 1)); itr_h += 2)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
              p_src3 = p_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size);
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
              vecmean8_inpx3_aligned(p_src1, p_src2, p_src3, p_dst, wc_plane_size);
              p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
            }

            if(temp_inp_h & 1)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
              vecmean8_inpx2_aligned(p_src1, p_src2, p_dst, wc_plane_size);
            }
          }
        }
        else
        {
          for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
          {
            p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size)); 
            for(itr_h=0; itr_h < (temp_inp_h & ~(2 - 1)); itr_h += 2)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
              p_src3 = p_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size);
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
              vecmean8_inpx3_unaligned(p_src1, p_src2, p_src3, p_dst, wc_plane_size);
              p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
            }

            if(temp_inp_h & 1)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
              vecmean8_inpx2_unaligned(p_src1, p_src2, p_dst, wc_plane_size);
            }
          }
        }
        temp_inp_h = 1;
        }break;
      case 2:{
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hc_plane_size = temp_inp_h * temp_inp_c;
        if(((((unsigned)p_in) & 3) == 0) && ((((unsigned)p_scratch) & 7) == 0) && ((temp_inp_c & 7) == 0))
        {
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
                vecmean8_inpx3_aligned(p_src1, p_src2, p_src3, p_dst, temp_inp_c);
                p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
              }

              if(temp_inp_w & 1)
              {
                p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
                p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
                vecmean8_inpx2_aligned(p_src1, p_src2, p_dst, temp_inp_c);
              }
            }
          }
        }
        else
        {
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
                vecmean8_inpx3_unaligned(p_src1, p_src2, p_src3, p_dst, temp_inp_c);
                p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
              }

              if(temp_inp_w & 1)
              {
                p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
                p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
                vecmean8_inpx2_unaligned(p_src1, p_src2, p_dst, temp_inp_c);
              }
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
        if(((((unsigned)p_in) & 3) == 0) && ((temp_inp_c & 3) == 0))
        {
          for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
          {
            for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
            {
              for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
              {
                p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
                p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w);
                ae_int32x2 i1 = AE_ZERO32();
                for(itr_c=0; itr_c < (temp_inp_c >> 3); itr_c++)
                {
                  ae_int16x4 j1, j2;
                  WORD16 out1, out2;
#if XCHAL_HAVE_HIFI1
                  AE_L8X4S_IP(j1, p_src2, 4*sizeof(WORD8));
                  AE_L8X4S_IP(j2, p_src2, 4*sizeof(WORD8));
#else
                  AE_L8X4F_IP(j1, p_src2, 4*sizeof(WORD8));
                  AE_L8X4F_IP(j2, p_src2, 4*sizeof(WORD8)); 
                  j1 = AE_SRAI16(j1, 8);
                  j2 = AE_SRAI16(j2, 8);
#endif
                  out1 = AE_INT16X4_RADD(j1);
                  out2 = AE_INT16X4_RADD(j2);
                  i1 = AE_ADD32S(i1, AE_MOVDA32(out1));
                  i1 = AE_ADD32S(i1, AE_MOVDA32(out2));
                }

                //Remainder Loop
                for(itr_c=0; itr_c < rem_c; itr_c++)
                {
                  ae_int32x2 j1;
                  j1 = (WORD32) *(p_src2 + itr_c);
                  i1 = AE_ADD32S(i1, j1);
                }
                AE_S32_L_I(i1, (ae_int32 *)p_dst, 0);
              }
            }
          }
        }
        else
        {
          for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
          {
            for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
            {
              for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
              {
                p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
                p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w);
                PRIME_8X4F(p_src2, align_src2);
                ae_int32x2 i1 = AE_ZERO32();
                for(itr_c=0; itr_c < (temp_inp_c >> 3); itr_c++)
                {
                  ae_int16x4 j1, j2;
                  WORD16 out1, out2;
#if XCHAL_HAVE_HIFI1
                  AE_LA8X4S_IP(j1, align_src2, p_src2);
                  AE_LA8X4S_IP(j2, align_src2, p_src2);
#else
                  AE_LA8X4F_IP(j1, align_src2, p_src2);
                  AE_LA8X4F_IP(j2, align_src2, p_src2);
                  j1 = AE_SRAI16(j1, 8);
                  j2 = AE_SRAI16(j2, 8);
#endif
                  out1 = AE_INT16X4_RADD(j1);
                  out2 = AE_INT16X4_RADD(j2);
                  i1 = AE_ADD32S(i1, AE_MOVDA32(out1));
                  i1 = AE_ADD32S(i1, AE_MOVDA32(out2));
                }

                //Remainder Loop
                for(itr_c=0; itr_c < rem_c; itr_c++)
                {
                  ae_int32x2 j1;
                  j1 = (WORD32) *(p_src2 + itr_c);
                  i1 = AE_ADD32S(i1, j1);
                }
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
        if(((((unsigned)p_scr_in) & 7) == 0) && ((plane_size & 7) == 0))
        {
          for(itr_n=1; itr_n < ((temp_inp_n -1) & ~(2 - 1)); itr_n += 2)
          {
            p_src1 = (ae_int32x2 *)(p_scratch);
            p_wsrc2 = (ae_int32x2 *)(p_scr_in + itr_n * plane_size);
            p_wsrc3 = (ae_int32x2 *)(p_scr_in + (itr_n + 1) * plane_size);
            p_dst  = (ae_int32x2 *)(p_scratch);
            vecmean32_inpx3_aligned(p_src1, p_wsrc2, p_wsrc3, p_dst, plane_size);
          }

          if((temp_inp_n - 1) & 1)
          {
            p_src1 = (ae_int32x2 *)(p_scratch);
            p_wsrc2 = (ae_int32x2 *)(p_scr_in + itr_n * plane_size);
            p_dst  = (ae_int32x2 *)(p_scratch);
            vecmean32_inpx2_aligned(p_src1, p_wsrc2, p_dst, plane_size);
          }
        }
        else
        {
          for(itr_n=1; itr_n < ((temp_inp_n -1) & ~(2 - 1)); itr_n += 2)
          {
            p_src1 = (ae_int32x2 *)(p_scratch);
            p_wsrc2 = (ae_int32x2 *)(p_scr_in + itr_n * plane_size);
            p_wsrc3 = (ae_int32x2 *)(p_scr_in + (itr_n + 1) * plane_size);
            p_dst  = (ae_int32x2 *)(p_scratch);
            vecmean32_inpx3_unaligned(p_src1, p_wsrc2, p_wsrc3, p_dst, plane_size);
          }

          if((temp_inp_n - 1) & 1)
          {
            p_src1 = (ae_int32x2 *)(p_scratch);
            p_wsrc2 = (ae_int32x2 *)(p_scr_in + itr_n * plane_size);
            p_dst  = (ae_int32x2 *)(p_scratch);
            vecmean32_inpx2_unaligned(p_src1, p_wsrc2, p_dst, plane_size);
          }
        }
        temp_inp_n = 1;
        }break;
      case 1: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        if(((((unsigned)p_scr_in) & 7) == 0) && ((wc_plane_size & 7) == 0))
        {
          for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
          {
            p_src1 = (ae_int32x2 *)(p_scratch + + (itr_n * plane_size));
            for(itr_h = 1; itr_h < ((temp_inp_h - 1) & ~(2 - 1)); itr_h += 2)
            {
              p_wsrc2 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
              p_wsrc3 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size));
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
              vecmean32_inpx3_aligned(p_src1, p_wsrc2, p_wsrc3, p_dst, wc_plane_size);
              p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
            }

            if((temp_inp_h - 1) & 1)
            {
              p_wsrc2 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
              vecmean32_inpx2_aligned(p_src1, p_wsrc2, p_dst, wc_plane_size);
            }
          }
        }
        else
        {
          for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
          {
            p_src1 = (ae_int32x2 *)(p_scratch + + (itr_n * plane_size));
            for(itr_h = 1; itr_h < ((temp_inp_h - 1) & ~(2 - 1)); itr_h += 2)
            {
              p_wsrc2 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
              p_wsrc3 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size));
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
              vecmean32_inpx3_unaligned(p_src1, p_wsrc2, p_wsrc3, p_dst, wc_plane_size);
              p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
            }

            if((temp_inp_h - 1) & 1)
            {
              p_wsrc2 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
              p_dst = (ae_int32x2 *)(p_scratch + (itr_n * wc_plane_size));
              vecmean32_inpx2_unaligned(p_src1, p_wsrc2, p_dst, plane_size);
            }
          }
        }
        temp_inp_h = 1;
        }break;
      case 2:{
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hc_plane_size = temp_inp_h * temp_inp_c;
        if(((((unsigned)p_scr_in) & 7) == 0) && ((temp_inp_c & 7) == 0))
        {
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
                vecmean32_inpx3_aligned(p_src1, p_wsrc2, p_wsrc3, p_dst, temp_inp_c);
                p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
              }

              if((temp_inp_w - 1) & 1)
              {
                p_wsrc2 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
                p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
                vecmean32_inpx2_aligned(p_src1, p_wsrc2, p_dst, temp_inp_c);
              }
            }
          }
        }
        else
        {
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
                vecmean32_inpx3_unaligned(p_src1, p_wsrc2, p_wsrc3, p_dst, temp_inp_c);
                p_src1 = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
              }

              if((temp_inp_w - 1) & 1)
              {
                p_wsrc2 = (ae_int32x2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
                p_dst = (ae_int32x2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
                vecmean32_inpx2_unaligned(p_src1, p_wsrc2, p_dst, temp_inp_c);
              }
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

WORD32 xa_nn_reduce_mean_4D_asym8s_asym8s(WORD8 * __restrict__ p_out
                                        ,const WORD32 *const p_out_shape
                                        ,const WORD8 * __restrict__ p_inp
                                        ,const WORD32 *const p_inp_shape
                                        ,const WORD32 * __restrict__ p_axis
                                        ,WORD32 num_out_dims
                                        ,WORD32 num_inp_dims
                                        ,WORD32 num_axis_dims
                                        ,WORD32 inp_zero_bias
                                        ,WORD32 out_multiplier
                                        ,WORD32 out_shift
                                        ,WORD32 out_zero_bias
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
  XA_NNLIB_ARG_CHK_COND((inp_zero_bias < -128 || inp_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int axis_itr = 0, inp_itr = 0, out_itr = 0;
  int num_elm_in_axis = 1;
  for(axis_itr=0; axis_itr < num_axis_dims; axis_itr++)
  {
    XA_NNLIB_ARG_CHK_COND(((p_axis[axis_itr] < 0) || (p_axis[axis_itr] > (num_inp_dims - 1))), -1);
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[p_axis[axis_itr]] > 1024), -1);
    
    /* Avoid calculation in case of repeated axis dims*/
    if((!axis_itr) || (p_axis[axis_itr] != p_axis[axis_itr-1]))
    {
      num_elm_in_axis *= p_inp_shape[p_axis[axis_itr]];
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
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

  WORD8 *p_in = (WORD8 *)(p_inp);
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
#if XCHAL_HAVE_HIFI1
  ALIGN_REGISTER_TYPE align_out = AE_ZALIGN64();
#endif

  if(num_axis_dims)
  {
    if(num_elm_in_axis > 1)
    { 
      xa_nn_reduce_sum_4D_asym8s_asym8s(p_in,
                                        p_4D_inp_shape,
                                        p_axis_data,
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
#if XCHAL_HAVE_HIFI1
          ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(temp1), AE_MOVF16X4_FROMF32X2(temp2));
          AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_out);
          out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(temp3), AE_MOVF16X4_FROMF32X2(temp4));
          AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_out);    
#else
          STORE_8X4_FROM_32X4(p_out, temp1, temp2);
          STORE_8X4_FROM_32X4(p_out, temp3, temp4);
#endif
        }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_out);
#endif
        for(itr = 0; itr < (out_length & 7); itr++)
        {
          ae_int32x2 temp1;

          AE_L32_IP(temp1, (ae_int32 *)p_src1, 4);
          *p_out++ = (WORD8) AE_MOVAD32_H(temp1);
        }
      }
      else
      {
        ae_int32x2 total_bias = AE_MOVDA32(-inp_zero_bias*num_elm_in_axis);
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

          d0_out32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(d0_out32, AE_MOVDA32(-128)));
          d1_out32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(d1_out32, AE_MOVDA32(-128)));
          d2_out32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(d2_out32, AE_MOVDA32(-128)));
          d3_out32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(d3_out32, AE_MOVDA32(-128)));
#if XCHAL_HAVE_HIFI1
          ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d0_out32), AE_MOVF16X4_FROMF32X2(d1_out32));
          AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_out);
          out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d2_out32), AE_MOVF16X4_FROMF32X2(d3_out32));
          AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_out);    
#else
          STORE_8X4_FROM_32X4(p_out, d0_out32, d1_out32)
          STORE_8X4_FROM_32X4(p_out, d2_out32, d3_out32)
#endif
        }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_out);
#endif
        for(itr = 0; itr < (out_length & 7); itr++)
        {
          ae_int32x2 wout1;
          ae_int32x2 d0_out32;

          AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
          wout1 = AE_ADD32S(wout1, total_bias);
          
          MPY_BY_QUANT_MULT_SLS_X2_OUT32(d0_out32, wout1, out_multiplier, left_shift, right_shift);
          d0_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d0_out32);

          d0_out32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(d0_out32, AE_MOVDA32(-128)));
          *p_out++ = (WORD8) AE_MOVAD32_H(d0_out32);
        }
      }
    }
    else
    {
      xtbool same_quant = (inp_zero_bias == out_zero_bias) && (out_multiplier == 0x40000000) && (out_shift == 1);

      itr = 0;
      ALIGN_REGISTER_TYPE align_inp;
      PRIME_8X4F(p_in, align_inp);

      if(same_quant)
      {
        xa_nn_memcpy(p_out, p_inp, inp_length); 
      }
      else
      {
        ae_int16x4 total_bias = AE_MOVDA16(inp_zero_bias);
#pragma no_unroll
        for(itr = 0; itr < (out_length >> 2); itr++)
        {
          ae_int16x4 wout1;
          ae_int32x2 d0_out32, d1_out32;
          ae_int32x2 temp1, temp2;
#if XCHAL_HAVE_HIFI1
          AE_LA8X4S_IP(wout1, align_inp, p_in);
#else
          AE_LA8X4F_IP(wout1, align_inp, p_in);
          wout1 = AE_SRAI16(wout1, 8);
#endif

          wout1 = AE_SUB16(wout1, total_bias);
         
          AE_MUL16X4(temp1, temp2, wout1, AE_MOVDA16(1));

          MPY_BY_QUANT_MULT_SLS_X2_OUT32(temp1, temp1, out_multiplier, left_shift, right_shift);
          d0_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), temp1);

          MPY_BY_QUANT_MULT_SLS_X2_OUT32(temp2, temp2, out_multiplier, left_shift, right_shift);
          d1_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), temp2);

          d0_out32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(d0_out32, AE_MOVDA32(-128)));
          d1_out32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(d1_out32, AE_MOVDA32(-128)));
#if XCHAL_HAVE_HIFI1
          ae_int16x4 out = AE_SEL16_6420(AE_MOVF16X4_FROMF32X2(d0_out32), AE_MOVF16X4_FROMF32X2(d1_out32));
          AE_SA8X4U_IP(out, align_out, (ae_int32 *)p_out); 
#else
          STORE_8X4_FROM_32X4(p_out, d0_out32, d1_out32)
#endif
        }
#if XCHAL_HAVE_HIFI1
    AE_SA64POS_FP(align_out, p_out);
#endif
        //Remainder Loop
        for(itr = 0; itr < (out_length & 3); itr++)
        {
          WORD16 wout1;
          ae_int32x2 d0_out32;
          ae_int32x2 temp1, temp2;

          wout1 = (WORD16) *(p_in + itr);

          wout1 = AE_MOVDA16(wout1);
          wout1 = AE_SUB16(wout1, total_bias);

          AE_MUL16X4(temp1, temp2, wout1, AE_MOVDA16(1));

          MPY_BY_QUANT_MULT_SLS_X2_OUT32(temp1, temp1, out_multiplier, left_shift, right_shift);
          d0_out32 = AE_ADD32S(AE_MOVDA32(out_zero_bias), temp1);
          d0_out32 = AE_MIN32(AE_MOVDA32(127), AE_MAX32(d0_out32, AE_MOVDA32(-128)));
          *p_out++ = (WORD8) AE_MOVAD32_H(d0_out32);
         
        }
      }

    }
  }
  else
  {
    xa_nn_memcpy(p_out, p_inp, inp_length); 
  }

  return 0;
}
#endif /* #ifndef ENABLE_SCRATCH_SIZE_API_ONLY */
