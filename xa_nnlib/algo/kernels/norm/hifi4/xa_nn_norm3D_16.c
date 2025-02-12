/*******************************************************************************
* Copyright (c) 2018-2025 Cadence Design Systems, Inc.
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

#define  SHRT_MIN -32768
#define  SHRT_MAX 32767
#define UINT_MAX (unsigned)4294967295

#define CALC_NORM_FROM_ACC64(norm64, accum, rshift, max_norm) { \
    ae_int64 round_cnst = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)); \
    ae_int64 temp64 = AE_SLAA64(round_cnst, (rshift-1)); \
    temp64          = AE_ADD64(temp64, accum); \
    temp64          = AE_SRAA64(temp64, rshift); \
    norm64 = AE_MIN64(temp64, AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, max_norm))); \
}

#define APPLY_SCALE_SHIFT_ROUND_ACC64_LL(out64_0, in32x2, scale32, rshift) { \
    ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (rshift-1)); \
    ae_int64 temp64 = round_cnst; \
    AE_MULA32_LL(temp64, in32x2, scale32); \
    out64_0   = AE_SRAA64(temp64, rshift); \
}

#define APPLY_SCALE_SHIFT_ROUND_ACC64X2(out64_0, out64_1, in32x2, scale32, rshift) { \
    ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (rshift-1)); \
    ae_int64 temp64 = round_cnst; \
    AE_MULA32_HH(temp64, in32x2, scale32); \
    out64_0   = AE_SRAA64(temp64, rshift); \
    temp64 = round_cnst; \
    AE_MULA32_LH(temp64, in32x2, scale32); \
    out64_1   = AE_SRAA64(temp64, rshift); \
}

#define CLIP64(inout64, low64, high64) { \
    inout64 = AE_MIN64(inout64, high64); \
    inout64 = AE_MAX64(inout64,  low64); \
}

#ifndef AE_MULFP32X16X2S_H
#define AE_MULFP32X16X2S_H(x1, x2) AE_SEL32_LL( AE_MOVINT32X2_FROMINT64(AE_SRAI64(AE_MUL32X16_H3(x1, x2), 15)),  AE_MOVINT32X2_FROMINT64(AE_SRAI64(AE_MUL32X16_L2(x1, x2), 15)) )
#endif

#ifndef AE_MULFP32X16X2S_L
#define AE_MULFP32X16X2S_L(x1, x2) AE_SEL32_LL( AE_MOVINT32X2_FROMINT64(AE_SRAI64(AE_MUL32X16_H1(x1, x2), 15)),  AE_MOVINT32X2_FROMINT64(AE_SRAI64(AE_MUL32X16_L0(x1, x2), 15)) )
#endif

static inline void internal_norm_vecx2(int64_t * p_acc0, int64_t * p_acc1, const WORD16 *p_vec0, const WORD16 *p_vec1, int cols1)
{
  int c_itr;
  ae_int64 acc0 = 0;  
  ae_int64 acc1 = 0;  
  ae_int16x4 d_vec0, d_vec1;

  if( ( ((unsigned)p_vec0%8) == 0) && ( ((unsigned)p_vec1%8) == 0) ){
    /* Aligned core loop */
    for(c_itr = 0; c_itr < cols1>>2; c_itr++){            
      AE_L16X4_IP(d_vec0, (ae_int16x4 *)p_vec0, 8);
      AE_MULAAAAQ16(acc0, d_vec0, d_vec0);
      AE_L16X4_IP(d_vec1, (ae_int16x4 *)p_vec1, 8);
      AE_MULAAAAQ16(acc1, d_vec1, d_vec1);
    }
  } else {
    ae_valign align_vec0 = AE_LA64_PP(p_vec0);
    ae_valign align_vec1 = AE_LA64_PP(p_vec1);

    for(c_itr = 0; c_itr < cols1>>2; c_itr++){            
      AE_LA16X4_IP(d_vec0, align_vec0, (ae_int16x4 *)p_vec0);
      AE_MULAAAAQ16(acc0, d_vec0, d_vec0);
      AE_LA16X4_IP(d_vec1, align_vec1, (ae_int16x4 *)p_vec1);
      AE_MULAAAAQ16(acc1, d_vec1, d_vec1);
    }
  }

  // remainder loop
  for(c_itr = 0; c_itr < (cols1&0x3); c_itr++){
    AE_L16_IP(d_vec0, (ae_int16 *)p_vec0, 2);
    AE_MULA16_00(acc0, d_vec0, d_vec0);
    AE_L16_IP(d_vec1, (ae_int16 *)p_vec1, 2);
    AE_MULA16_00(acc1, d_vec1, d_vec1);
  }

  *(ae_int64*)p_acc0 = acc0;
  *(ae_int64*)p_acc1 = acc1;
}

static inline void __attribute__((always_inline))  internal_norm_vec(int64_t * p_acc, const WORD16 *p_vec, int cols1)
{
  int c_itr;
  ae_int64 acc = 0;  
  ae_int64 acc1 = 0;  

  ae_int16x4 d_vec0, d_vec1;

  if( ((unsigned)p_vec%8) == 0 ) {
    /* aligned loop */
    for(c_itr = 0; c_itr < cols1>>3; c_itr++){            
      d_vec1 = AE_L16X4_I((ae_int16x4 *)p_vec, 8);
      AE_L16X4_IP(d_vec0, (ae_int16x4 *)p_vec, 16);
      AE_MULAAAAQ16(acc, d_vec0, d_vec0);
      AE_MULAAAAQ16(acc1, d_vec1, d_vec1);
    }
    acc = AE_ADD64(acc, acc1);

    if( (cols1%8) >= 4 ){            
      AE_L16X4_IP(d_vec0, (ae_int16x4 *)p_vec, 8);
      AE_MULAAAAQ16(acc, d_vec0, d_vec0);
    }
  } else {
    ae_valign align_vec = AE_LA64_PP(p_vec);

    for(c_itr = 0; c_itr < cols1>>2; c_itr++){            
      AE_LA16X4_IP(d_vec0, align_vec, (ae_int16x4 *)p_vec);
      AE_MULAAAAQ16(acc, d_vec0, d_vec0);
    }
  }

  // remainder loop
  for(c_itr = 0; c_itr < (cols1&0x3); c_itr++){
    AE_L16_IP(d_vec0, (ae_int16 *)p_vec, 2);
    AE_MULA16_00(acc, d_vec0, d_vec0);
  }

  *(ae_int64*)p_acc = acc;
}


WORD32 xa_nn_norm_calc_3D_16_nhwc(
    UWORD16 * p_outnorm /*Norm data: 2D -> iw*ih, or scalar*/ , 
    WORD8 * p_outnsa /*NSA data: 2D -> iw*ih, or scalar*/ , 
    const WORD16 * p_inp /*3D -> iw*ih*ic */,
    int input_height, int input_width, int input_channels, 
    int accros_depth_flag,
    int out_shift, /*sumSquareShift*/
    const UWORD16 *prsqrt, int rsqrt_table_len /* rsqrt table */)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_outnorm, -1);
  XA_NNLIB_ARG_CHK_PTR(p_outnsa, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(prsqrt, -1);

  /* Pointer Alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_outnorm, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(prsqrt, sizeof(WORD16), -1);

  /* Param Checks*/
  XA_NNLIB_ARG_CHK_COND((out_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((rsqrt_table_len <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((accros_depth_flag != 0) && (accros_depth_flag != 1), -1);

  int out_rshift = -out_shift;
  if(rsqrt_table_len>32768) { rsqrt_table_len = 32768;}

  if(accros_depth_flag == 0) /* Calc norm data for entire 3D input */
  {
    int64_t accum = 0;
    int inp_len = input_height*input_width*input_channels;

    internal_norm_vec(&accum, p_inp, inp_len);

    ae_int64 norm64, norm64_c;
    CALC_NORM_FROM_ACC64(norm64, accum, out_rshift, 0xFFFFFFFF);
    WORD8 nsaShift = 16 - AE_NSAZ32_L(AE_MOVINT32X2_FROMINT64(norm64));
    nsaShift = (nsaShift<0) ? 0 : nsaShift;
    
    CALC_NORM_FROM_ACC64(norm64_c, norm64, nsaShift, rsqrt_table_len-1);
    UWORD16 norm16u = (UWORD16)AE_MOVAD32_L(AE_MOVINT32X2_FROMINT64(norm64_c));
    p_outnorm[0]     = (UWORD16) prsqrt[norm16u];
    p_outnsa[0]      = nsaShift + out_rshift;
  }
  else /* Calc norm data across depth dimension only */
  {
    int ih, iw;
    
    for(ih = 0; ih < input_height; ih++)
    {
      iw = 0;

      for(; iw < (input_width&~0x1); iw+=2)
      {
        int offset = ih*input_width*input_channels + iw*input_channels;
        int offset_1 = ih*input_width*input_channels + (iw+1)*input_channels;
        const WORD16 *p_inp_ch0 = &p_inp[offset];
        const WORD16 *p_inp_ch1 = &p_inp[offset_1];
        int64_t accum0 = 0;
        int64_t accum1 = 0;

        internal_norm_vecx2(&accum0, &accum1, p_inp_ch0, p_inp_ch1, input_channels);

        ae_int64 norm64, norm64_c;
        CALC_NORM_FROM_ACC64(norm64, accum0, out_rshift, 0xFFFFFFFF);
        WORD8 nsaShift = 16 - AE_NSAZ32_L(AE_MOVINT32X2_FROMINT64(norm64));
        nsaShift = (nsaShift<0) ? 0 : nsaShift;
        
        CALC_NORM_FROM_ACC64(norm64_c, norm64, nsaShift, rsqrt_table_len-1);
        UWORD16 norm16u = (UWORD16)AE_MOVAD32_L(AE_MOVINT32X2_FROMINT64(norm64_c));
        p_outnorm[iw + (ih * input_width)]     = (UWORD16) prsqrt[norm16u];
        p_outnsa[iw + (ih * input_width)]      = nsaShift + out_rshift;

        CALC_NORM_FROM_ACC64(norm64, accum1, out_rshift, 0xFFFFFFFF);
        nsaShift = 16 - AE_NSAZ32_L(AE_MOVINT32X2_FROMINT64(norm64));
        nsaShift = (nsaShift<0) ? 0 : nsaShift;
        
        CALC_NORM_FROM_ACC64(norm64_c, norm64, nsaShift, rsqrt_table_len-1);
        norm16u = (UWORD16)AE_MOVAD32_L(AE_MOVINT32X2_FROMINT64(norm64_c));
        p_outnorm[iw + 1 + (ih * input_width)]     = (UWORD16) prsqrt[norm16u];
        p_outnsa[iw + 1 + (ih * input_width)]      = nsaShift + out_rshift;
      }

      for(; iw < input_width; iw++)
      {
        int offset = ih*input_width*input_channels + iw*input_channels;
        const WORD16 *p_inp_ch = &p_inp[offset];
        int64_t accum = 0;

        internal_norm_vec(&accum, p_inp_ch, input_channels);

        ae_int64 norm64, norm64_c;
        CALC_NORM_FROM_ACC64(norm64, accum, out_rshift, 0xFFFFFFFF);
        WORD8 nsaShift = 16 - AE_NSAZ32_L(AE_MOVINT32X2_FROMINT64(norm64));
        nsaShift = (nsaShift<0) ? 0 : nsaShift;
        
        CALC_NORM_FROM_ACC64(norm64_c, norm64, nsaShift, rsqrt_table_len-1);
        UWORD16 norm16u = (UWORD16)AE_MOVAD32_L(AE_MOVINT32X2_FROMINT64(norm64_c));
        p_outnorm[iw + (ih * input_width)]     = (UWORD16) prsqrt[norm16u];
        p_outnsa[iw + (ih * input_width)]      = nsaShift + out_rshift;
      }
    }
  }

  return 0;
}


static inline void __attribute__((always_inline))  internal_apply_1D(
    WORD16 * __restrict__ p_out,
    const WORD16 * __restrict__ p_inp,
    UWORD16 norm_factor,
    int inlen,
    WORD16 * __restrict__ p_out_multiplier,
    int out_multiplier_offset,
    int nsaShift,
    int finalShift)
{
  int ic;

  if(out_multiplier_offset == 0){
    WORD16 out_mul = p_out_multiplier[0];
    WORD16 finalScale = ((nsaShift & 0x1) ? (((int64_t) 46341 * out_mul) >> 15) : out_mul);

    ae_valign align_in = AE_LA64_PP(p_inp);

    ae_int16x4 d_finalScale = AE_MOVDA16(finalScale);
    ae_int32x2 d_norm_factor = AE_MOVDA32(norm_factor);

    ae_int64 out_min = AE_MOV64(SHRT_MIN);
    ae_int64 out_max = AE_MOV64(SHRT_MAX);

    for(ic = 0; ic < (inlen>>2); ic++)
    {
      ae_int16x4 d_in0;
      ae_int32x2 d_inscaled0, d_inscaled1;
      ae_int64 out64_0, out64_1, out64_2, out64_3;

      AE_LA16X4_IP(d_in0, align_in, (ae_int16x4 *)p_inp);
      AE_MUL16X4(d_inscaled0, d_inscaled1, d_in0, d_finalScale);

      APPLY_SCALE_SHIFT_ROUND_ACC64X2(out64_0, out64_1, d_inscaled0, d_norm_factor, finalShift);
      APPLY_SCALE_SHIFT_ROUND_ACC64X2(out64_2, out64_3, d_inscaled1, d_norm_factor, finalShift);
      CLIP64(out64_0, out_min, out_max);
      CLIP64(out64_1, out_min, out_max);
      CLIP64(out64_2, out_min, out_max);
      CLIP64(out64_3, out_min, out_max);

      /* no gain with vectorization, so keep scalar */
      AE_S16_0_IP(AE_MOVINT16X4_FROMINT64(out64_0), (ae_int16 *)p_out, 2);
      AE_S16_0_IP(AE_MOVINT16X4_FROMINT64(out64_1), (ae_int16 *)p_out, 2);
      AE_S16_0_IP(AE_MOVINT16X4_FROMINT64(out64_2), (ae_int16 *)p_out, 2);
      AE_S16_0_IP(AE_MOVINT16X4_FROMINT64(out64_3), (ae_int16 *)p_out, 2);
    }

    for(ic = 0; ic < (inlen&3); ic++)
    {
      ae_int16x4 d_in0;
      ae_int32x2 d_inscaled0, d_inscaled1;
      ae_int64 out64_0;

      AE_L16_IP(d_in0, (ae_int16 *)p_inp, 2);
      AE_MUL16X4(d_inscaled0, d_inscaled1, d_in0, d_finalScale);
      APPLY_SCALE_SHIFT_ROUND_ACC64_LL(out64_0, d_inscaled0, d_norm_factor, finalShift);
      CLIP64(out64_0, out_min, out_max);
      AE_S16_0_IP(AE_MOVINT16X4_FROMINT64(out64_0), (ae_int16 *)p_out, 2);
    }

  } else {
#if 0
    for(ic = 0; ic < inlen; ic++)
    {
      WORD16 val = p_inp[ic];
      WORD16 out_mul = p_out_multiplier[ic];
      WORD16 finalScale = ((nsaShift & 0x1) ? (((int64_t) 46341 * out_mul) >> 15) : out_mul);
      WORD32 in         = (WORD32) val * finalScale;
      p_out[ic] = (WORD16) xaiRoundAndClamp64((int64_t) in * norm_factor, finalShift, SHRT_MIN, SHRT_MAX);
    }
#else
    ae_int16x4 d_nsaShift = AE_MOVDA16(nsaShift);
    ae_int16x4 mask1 = AE_MOVDA16(1);
    mask1 = AE_AND16(d_nsaShift, mask1);
    xtbool4 bool4 = AE_LT16(AE_ZERO16(), mask1);
    ae_int32x2 d_norm_factor = AE_MOVDA32(norm_factor);

    ae_int32x2 scale_cnst = AE_MOVDA32(46341);
    ae_int64 out_min = AE_MOV64(SHRT_MIN);
    ae_int64 out_max = AE_MOV64(SHRT_MAX);

    ae_valign align_in = AE_LA64_PP(p_inp);
    ae_valign align_mu = AE_LA64_PP(p_out_multiplier);

    for(ic = 0; ic < (inlen>>2); ic++)
    {
      ae_int16x4 d_in0;
      ae_int32x2 d_inscaled0, d_inscaled1;
      ae_int64 out64_0, out64_1, out64_2, out64_3;
      
      ae_int16x4 out_mul, d_finalScale;

      AE_LA16X4_IP(d_in0, align_in, (ae_int16x4 *)p_inp);
      AE_LA16X4_IP(out_mul, align_mu, (ae_int16x4 *)p_out_multiplier);

      d_finalScale = out_mul;
      ae_int32x2 mul_scaled0 = AE_MULFP32X16X2S_H(scale_cnst, out_mul);
      ae_int32x2 mul_scaled1 = AE_MULFP32X16X2S_L(scale_cnst, out_mul);
      ae_int16x4 d_finalScale_temp = AE_SAT16X4(mul_scaled0, mul_scaled1);
      AE_MOVT16X4(d_finalScale, d_finalScale_temp, bool4);

      AE_MUL16X4(d_inscaled0, d_inscaled1, d_in0, d_finalScale);
      APPLY_SCALE_SHIFT_ROUND_ACC64X2(out64_0, out64_1, d_inscaled0, d_norm_factor, finalShift);
      APPLY_SCALE_SHIFT_ROUND_ACC64X2(out64_2, out64_3, d_inscaled1, d_norm_factor, finalShift);
      CLIP64(out64_0, out_min, out_max);
      CLIP64(out64_1, out_min, out_max);
      CLIP64(out64_2, out_min, out_max);
      CLIP64(out64_3, out_min, out_max);
      AE_S16_0_IP(AE_MOVINT16X4_FROMINT64(out64_0), (ae_int16 *)p_out, 2);
      AE_S16_0_IP(AE_MOVINT16X4_FROMINT64(out64_1), (ae_int16 *)p_out, 2);
      AE_S16_0_IP(AE_MOVINT16X4_FROMINT64(out64_2), (ae_int16 *)p_out, 2);
      AE_S16_0_IP(AE_MOVINT16X4_FROMINT64(out64_3), (ae_int16 *)p_out, 2);
    }

    for(ic = 0; ic < (inlen&3); ic++)
    {
      ae_int16x4 d_in0;
      ae_int32x2 d_inscaled0, d_inscaled1;
      ae_int64 out64_0;
      
      ae_int16x4 out_mul, d_finalScale;

      AE_L16_IP(out_mul, (ae_int16 *)p_out_multiplier, 2);
      AE_L16_IP(d_in0, (ae_int16 *)p_inp, 2);

      d_finalScale = out_mul;
      ae_int32x2 mul_scaled0 = AE_MULFP32X16X2S_H(scale_cnst, out_mul);
      ae_int32x2 mul_scaled1 = AE_MULFP32X16X2S_L(scale_cnst, out_mul);
      ae_int16x4 d_finalScale_temp = AE_SAT16X4(mul_scaled0, mul_scaled1);
      AE_MOVT16X4(d_finalScale, d_finalScale_temp, bool4);

      AE_MUL16X4(d_inscaled0, d_inscaled1, d_in0, d_finalScale);
      APPLY_SCALE_SHIFT_ROUND_ACC64_LL(out64_0, d_inscaled0, d_norm_factor, finalShift);
      CLIP64(out64_0, out_min, out_max);
      AE_S16_0_IP(AE_MOVINT16X4_FROMINT64(out64_0), (ae_int16 *)p_out, 2);
    }

#endif
  }
  return;
}


WORD32 xa_nn_norm_apply_3D_16_nhwc(
    WORD16 * p_out, 
    const WORD16 * p_inp, /*3D -> iw*ih*ic */
    const UWORD16 *p_inp_normdata,
    const WORD8 *p_inp_nsadata,
    int input_height, int input_width, int input_channels,
    int accross_depth_flag,
    int per_chan_flag,
    WORD16 * p_out_multiplier,
    WORD32 out_shift,
    WORD32 rsqrt_shift
)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_normdata, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_nsadata, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);

  /* Pointer Alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_normdata, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD16), -1);

  /* Param Checks*/
  XA_NNLIB_ARG_CHK_COND((out_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((accross_depth_flag != 0) && (accross_depth_flag != 1), -1);
  XA_NNLIB_ARG_CHK_COND((per_chan_flag != 0) && (per_chan_flag != 1), -1);

  /* positive right shift */
  int out_rshift = -out_shift;

  int out_multiplier_offset = 1;
  if(per_chan_flag == 0) {
    out_multiplier_offset = 0;
  }

  if(accross_depth_flag == 0){
    int ih, iw;
    UWORD16 norm_factor = p_inp_normdata[0];
    WORD8  nsaShift    = p_inp_nsadata[0];
    WORD8  finalShift  = out_rshift + ((nsaShift + 1) >> 1) + rsqrt_shift;
    
    if(out_multiplier_offset == 0 )
    {
      internal_apply_1D(p_out, p_inp, norm_factor, input_channels*input_width*input_height, p_out_multiplier, 0, nsaShift, finalShift);
    }
    else
    {
      for(ih = 0; ih < input_height; ih++)
      {
        for(iw = 0; iw < input_width; iw++)
        {
          int offset = ih*input_width*input_channels + iw*input_channels;
          const WORD16 *p_inp_ch = &p_inp[offset];
          WORD16 *p_out_ch = &p_out[offset];

          internal_apply_1D(p_out_ch, p_inp_ch, norm_factor, input_channels, p_out_multiplier, 1, nsaShift, finalShift);
        }
      }
    }

  } else {
    int ih, iw;
    
    for(ih = 0; ih < input_height; ih++)
    {
      for(iw = 0; iw < input_width; iw++)
      {
        int offset = ih*input_width*input_channels + iw*input_channels;
        const WORD16 *p_inp_ch = &p_inp[offset];
        WORD16 *p_out_ch = &p_out[offset];
        UWORD16 norm_factor = p_inp_normdata[ih*input_width + iw];
        WORD8  nsaShift    = p_inp_nsadata[ih*input_width + iw];

        WORD8 finalShift  = out_rshift + ((nsaShift + 1) >> 1) + rsqrt_shift;

        internal_apply_1D(p_out_ch, p_inp_ch, norm_factor, input_channels, p_out_multiplier, out_multiplier_offset, nsaShift, finalShift);
      }
    }
  }

  return 0;
}

