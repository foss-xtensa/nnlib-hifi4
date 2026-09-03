/*******************************************************************************
* Copyright (c) 2018-2026 Cadence Design Systems, Inc.
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

#define  INT_MAX  2147483647
#define  INT_MIN  (-INT_MAX - 1)

#define SW_MOVDA32(a) AE_MOVDA32X2(a, a)
#define ZERO64  AE_ZERO64()
#define ZERO32   AE_ZERO32()

#define CALC_MPY_SHIFT_ROUND32(index, accum, multiplier, rshift, max_id) { \
    ae_int64 round_cnst = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)); \
    ae_int64 temp64 = AE_SLAA64(round_cnst, (rshift-1)); \
    AE_MULA32X16_L0(temp64, accum, multiplier); \
    index = AE_MOVINT32X2_FROMINT64(AE_SRAA64(temp64, rshift)); \
    index = AE_MIN32(index, AE_MOVDA32(max_id)); \
}

#define AE_SAT32X2_HIFI4(out32x2, inp64_2, inp64_1) \
    out32x2 = AE_TRUNCA32X2F64S(inp64_2, inp64_1, 32);


#ifndef AE_MOVAB2
static inline unsigned char  AE_MOVAB2( xtbool2 b2){
	
	ae_int32x2 d0 = 0;
	ae_int32x2 d1 = 1;	
	AE_MOVT32X2(d0,d1,b2);
	unsigned int low,high;
	low = AE_MOVAD32_L(d0);
	high = AE_MOVAD32_H(d0);
	unsigned char out = (high<<1) | low;
	return out;
}
#endif

#ifndef AE_MULFP32X16X2S_H
#define AE_MULFP32X16X2S_H(x1, x2) AE_SEL32_LL( AE_MOVINT32X2_FROMINT64(AE_SRAI64(AE_MUL32X16_H3(x1, x2), 15)),  AE_MOVINT32X2_FROMINT64(AE_SRAI64(AE_MUL32X16_L2(x1, x2), 15)) )
#endif

#ifndef AE_MULFP32X16X2S_L
#define AE_MULFP32X16X2S_L(x1, x2) AE_SEL32_LL( AE_MOVINT32X2_FROMINT64(AE_SRAI64(AE_MUL32X16_H1(x1, x2), 15)),  AE_MOVINT32X2_FROMINT64(AE_SRAI64(AE_MUL32X16_L0(x1, x2), 15)) )
#endif


WORD32 xa_nn_norm_calc_3D_8_nhwc(
    WORD16 * p_out /*Norm data: 2D -> iw*ih, or scalar*/ , 
    WORD8 * p_outnsa /*NSA data: 2D -> iw*ih, or scalar*/ ,
    const WORD8 * p_inp /*3D -> iw*ih*ic */,
    int input_height, int input_width, int input_channels, 
    int accross_depth_flag,
    int out_shift, /*sumSquareShift*/
    const UWORD16 *prsqrt, int rsqrt_table_len) /* rsqrt table */
{
  /* NULL pointer check */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(prsqrt, -1);
  /* Basic Parameter checks */  
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0 || input_channels <= 0),-1);
  XA_NNLIB_ARG_CHK_COND(rsqrt_table_len <= 0,-1);
  XA_NNLIB_ARG_CHK_COND((out_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((accross_depth_flag != 0) && (accross_depth_flag != 1), -1);
  int out_rshift = -out_shift;
  
  if(accross_depth_flag == 0) /* Calc norm data for entire 3D input */
  {
    WORD32 i;
    WORD32 inp_len = input_height*input_width*input_channels;
    WORD32 lc = inp_len >> 2;
    WORD32 remc = inp_len & 3;
    
    /*ae_int32x2*/ ae_int64 acc_64_1=ZERO64; 
    ae_int16x4 d0, d1;
    ae_int16x4 d_input_val;

    for(i = 0; i < lc; i++)
    {
      const UWORD8 *ptu_inp = (const UWORD8 *)&p_inp[i << 2];
      d0 = AE_MOVDA16X2(ptu_inp[0], ptu_inp[1]);
      d1 = AE_MOVDA16X2(ptu_inp[2], ptu_inp[3]);
      d_input_val = AE_SEL16_5410(d0, d1);
      d_input_val = AE_SRAI16(AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(AE_MOVINT32X2_FROMINT16X4(d_input_val), 8)), 8);
      AE_MULAAAAQ16(acc_64_1, d_input_val, d_input_val);
    }
    if(remc & 3)
    {
      const UWORD8 *ptu_inp = (const UWORD8 *)&p_inp[i << 2];
      UWORD8 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
      if (remc >= 1) b0 = ptu_inp[0];
      if (remc >= 2) b1 = ptu_inp[1];
      if (remc >= 3) b2 = ptu_inp[2];
      
      d0 = AE_MOVDA16X2(b0, b1);
      d1 = AE_MOVDA16X2(b2, b3);
      d_input_val = AE_SEL16_5410(d0, d1);
      d_input_val = AE_SRAI16(AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(AE_MOVINT32X2_FROMINT16X4(d_input_val), 8)), 8);
      AE_MULAAAAQ16(acc_64_1, d_input_val, d_input_val);
    }
    
    ae_f64 acc_64F = AE_SLAA64S((ae_f64)acc_64_1, 32-out_rshift);
    ae_int32x2 acc1 = AE_MOVINT32X2_FROMF32X2(AE_ROUND32X2F64SASYM(acc_64F, acc_64F));
    WORD32 nsaShift = AE_NSAZ32_L(acc1);
    if(AE_MOVAB2(AE_EQ32(acc1,ZERO32)))
    {
      nsaShift = 31;
    }
    nsaShift = 15 - nsaShift + 1;
    nsaShift = (nsaShift<0) ? 0 : nsaShift;
    acc1 = AE_MOVINT32X2_FROMF32X2(AE_SRAA32RS(AE_MOVF32X2_FROMINT32X2(acc1), nsaShift));
    AE_MINMAX32(acc1, AE_MOVDA32(0), AE_MOVDA32(rsqrt_table_len-1));
    p_out[0] = prsqrt[AE_MOVAD32_H(acc1)];
    p_outnsa[0] = nsaShift + out_rshift;

  }

  else /* Calc norm data across depth dimension only */
  {
    WORD32 ihw, ic;
    WORD32 ilc = input_channels >> 2;
    WORD32 iremc = input_channels & 3;
    WORD32 olc = input_height * input_width;
    
    const UWORD8 *ptu_inp = (const UWORD8 *)&p_inp[0];
    for(ihw = 0; ihw < olc; ihw++)
    {
        ae_int64 acc_64_1=ZERO64;
        ae_int16x4 d0, d1;
        ae_int16x4 d_input_val;
        
        for(ic = 0; ic < ilc; ic++)
        {
          d0 = AE_MOVDA16X2(ptu_inp[0], ptu_inp[1]);
          d1 = AE_MOVDA16X2(ptu_inp[2], ptu_inp[3]);
          d_input_val = AE_SEL16_5410(d0, d1);
          d_input_val = AE_SRAI16(AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(AE_MOVINT32X2_FROMINT16X4(d_input_val), 8)), 8);
          AE_MULAAAAQ16(acc_64_1, d_input_val, d_input_val);
          ptu_inp+=4;
        }
        if(iremc)
        {
          UWORD8 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
          if (iremc >= 1) b0 = ptu_inp[0];
          if (iremc >= 2) b1 = ptu_inp[1];
          if (iremc >= 3) b2 = ptu_inp[2];

          d0 = AE_MOVDA16X2(b0, b1);
          d1 = AE_MOVDA16X2(b2, b3);
          d_input_val = AE_SEL16_5410(d0, d1);
          d_input_val = AE_SRAI16(AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(AE_MOVINT32X2_FROMINT16X4(d_input_val), 8)), 8);
          AE_MULAAAAQ16(acc_64_1, d_input_val, d_input_val);
          ptu_inp+=iremc;
        }
    

    ae_f64 acc_64F = AE_SLAA64S((ae_f64)acc_64_1, 32-out_rshift);
    ae_int32x2 acc1 = AE_MOVINT32X2_FROMF32X2(AE_ROUND32X2F64SASYM(acc_64F, acc_64F));

    WORD32 nsaShift = AE_NSAZ32_L(acc1);
    if(AE_MOVAB2(AE_EQ32(acc1,ZERO32)))
    {
      nsaShift = 31;
    }
    nsaShift = 15 - nsaShift + 1;
    nsaShift = (nsaShift<0) ? 0 : nsaShift;
    acc1 = AE_MOVINT32X2_FROMF32X2(AE_SRAA32RS(AE_MOVF32X2_FROMINT32X2(acc1), nsaShift));
    AE_MINMAX32(acc1, AE_MOVDA32(0), AE_MOVDA32(rsqrt_table_len-1));
    p_out[ihw] = prsqrt[AE_MOVAD32_H(acc1)];
    p_outnsa[ihw] = nsaShift + out_rshift;
    }
  }
  return 0;
}

WORD32 xa_nn_norm_apply_3D_8_nhwc(
    WORD8 * p_out, 
    const WORD8 * p_inp, /*3D -> iw*ih*ic */
    WORD16 *p_inp_normdata,
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
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_nsadata, -1);

  /* Pointer Alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_normdata, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_nsadata, sizeof(WORD8), -1);

  /* Param Checks*/
  XA_NNLIB_ARG_CHK_COND((out_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((rsqrt_shift < 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((accross_depth_flag != 0) && (accross_depth_flag != 1), -1);
  XA_NNLIB_ARG_CHK_COND((per_chan_flag != 0) && (per_chan_flag != 1), -1);

  int out_rshift = -out_shift;
  WORD32 inp_ch_lc = input_channels >> 2;
  if(accross_depth_flag == 0)
  {
    int ic, itrc;
    UWORD16 norm_factor = p_inp_normdata[0];
    ae_int32x2 d_norm = SW_MOVDA32(norm_factor);
    WORD8  nsaShift    = p_inp_nsadata[0];
    WORD8 finalShift  = out_rshift + ((nsaShift + 1) >> 1) + rsqrt_shift;

    ae_int16x4 d_inp, d0, d1, d_out_mult, d_mult;
    ae_int64 norm_inp11_1, norm_inp11_2, norm_inp12_1, norm_inp12_2;
    ae_int64 out11_1, out11_2, out12_1, out12_2;
    ae_int32x2 sat_out11, sat_out12;
    ae_int16x4 sat_out;
    ae_int32x2 scaled_inp11, scaled_inp12;

    ae_f32x2 d_scale1 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 d_scale2 = AE_MOVF32X2_FROMINT32X2(ZERO32);

    if(per_chan_flag == 0)
    {
      const ae_int16 * ptr_out_multiplier = (const ae_int16 *)p_out_multiplier;
      AE_L16_IP(d_mult, ptr_out_multiplier, 2);
      WORD32 input_size=input_height*input_width*input_channels;
      WORD32 nsa_mult_factor = (nsaShift & 0x1) ? 46341 : (1<<15);
      ae_int32x2 d_nsa_multiplier = SW_MOVDA32(nsa_mult_factor);

      d_scale1 = AE_MULFP32X16X2S_H(AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult));
      ae_int16x4 d_scale = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale1), AE_MOVINT32X2_FROMF32X2(d_scale1));

      ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));

      for(ic = 0; ic < (input_size>>2); ic++)
      {
        norm_inp11_1 = round_cnst;
        norm_inp11_2 = round_cnst;
        norm_inp12_1 = round_cnst;
        norm_inp12_2 = round_cnst;
        
        itrc = ic << 2;
        const WORD8 *ptu_inp = (const WORD8 *)&p_inp[itrc];
        d0 = AE_MOVDA16X2(ptu_inp[0], ptu_inp[1]);
        d1 = AE_MOVDA16X2(ptu_inp[2], ptu_inp[3]);
        d_inp = AE_SEL16_5410(d0, d1);
        
        AE_MUL16X4(scaled_inp11, scaled_inp12, d_inp, d_scale);
        AE_MIN32(AE_MAX32(scaled_inp11, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));
        AE_MIN32(AE_MAX32(scaled_inp12, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));

        AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
        AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
        AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
        AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);

        out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
        out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
        out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
        out12_2 = AE_SRAA64(norm_inp12_2, finalShift);

        AE_SAT32X2_HIFI4(sat_out11, out11_1, out11_2);
        AE_SAT32X2_HIFI4(sat_out12, out12_1, out12_2);
        
        sat_out = AE_SAT16X4(sat_out11, sat_out12);
        sat_out = AE_MOVINT16X4_FROMF16X4(AE_SLAI16S(AE_MOVF16X4_FROMINT16X4(sat_out), 8));
        sat_out = AE_SRAI16(sat_out, 8);

        p_out[itrc + 0] = (WORD8)(AE_MOVAD16_3(sat_out));
        p_out[itrc + 1] = (WORD8)(AE_MOVAD16_2(sat_out));
        p_out[itrc + 2] = (WORD8)(AE_MOVAD16_1(sat_out));
        p_out[itrc + 3] = (WORD8)(AE_MOVAD16_0(sat_out));
      }
      
      int iremc = input_size & 3;
      if(iremc)
      {
        norm_inp11_1 = round_cnst;
        norm_inp11_2 = round_cnst;
        norm_inp12_1 = round_cnst;
        norm_inp12_2 = round_cnst;

        itrc = input_size & (~3);
        const WORD8 *ptu_inp = (const WORD8 *)&p_inp[itrc];
        WORD8 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        if (iremc >= 1) b0 = ptu_inp[0];
        if (iremc >= 2) b1 = ptu_inp[1];
        if (iremc >= 3) b2 = ptu_inp[2];
        
        d0 = AE_MOVDA16X2(b0, b1);
        d1 = AE_MOVDA16X2(b2, b3);
        d_inp = AE_SEL16_5410(d0, d1);

        AE_MUL16X4(scaled_inp11, scaled_inp12, d_inp, d_scale);
        AE_MIN32(AE_MAX32(scaled_inp11, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));
        AE_MIN32(AE_MAX32(scaled_inp12, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));

        AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
        AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
        AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
        AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);

        out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
        out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
        out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
        out12_2 = AE_SRAA64(norm_inp12_2, finalShift);

        AE_SAT32X2_HIFI4(sat_out11, out11_1, out11_2);
        AE_SAT32X2_HIFI4(sat_out12, out12_1, out12_2);

        sat_out = AE_SAT16X4(sat_out11, sat_out12);
        sat_out = AE_MOVINT16X4_FROMF16X4(AE_SLAI16S(AE_MOVF16X4_FROMINT16X4(sat_out), 8));
        sat_out = AE_SRAI16(sat_out, 8);

        if (iremc >= 1) p_out[itrc + 0] = (WORD8)(AE_MOVAD16_3(sat_out));
        if (iremc >= 2) p_out[itrc + 1] = (WORD8)(AE_MOVAD16_2(sat_out));
        if (iremc >= 3) p_out[itrc + 2] = (WORD8)(AE_MOVAD16_1(sat_out));
    }
  }
    //(accross_depth_flag == 0) && (per_chan_flag == 1)
    else{
    
      ae_int16x4 d_scale;
      WORD32 nsa_mult_factor = (nsaShift & 0x1) ? 46341 : (1<<15);
      ae_int32x2 d_nsa_multiplier = SW_MOVDA32(nsa_mult_factor);

      ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));
      const WORD8 *ptu_inp = (const WORD8 *)&p_inp[0];
      WORD8 *ptu_out = (WORD8 *)&p_out[0];
      WORD32 iremc = input_channels & 3;

      for(int ihw = 0; ihw < input_height * input_width; ihw++)
      {
        ae_int16x4 *ptr_out_mult = (ae_int16x4 *)p_out_multiplier;
        ae_valign a_out_mult = AE_LA64_PP(ptr_out_mult);
        
  //#pragma concurrent
        for(ic = 0; ic < (inp_ch_lc); ic++)
        {
            norm_inp11_1 = round_cnst;
            norm_inp11_2 = round_cnst;
            norm_inp12_1 = round_cnst;
            norm_inp12_2 = round_cnst;
                     
            d0 = AE_MOVDA16X2(ptu_inp[0], ptu_inp[1]);
            d1 = AE_MOVDA16X2(ptu_inp[2], ptu_inp[3]);
            d_inp = AE_SEL16_5410(d0, d1);
            
            AE_LA16X4_IP(d_out_mult, a_out_mult, ptr_out_mult);

            d_scale1 = AE_MULFP32X16X2S_H(AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult));
            d_scale2 = AE_MULFP32X16X2S_L(AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult));
            d_scale = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale1), AE_MOVINT32X2_FROMF32X2(d_scale2));

            AE_MUL16X4(scaled_inp11, scaled_inp12, d_inp, d_scale);
            AE_MIN32(AE_MAX32(scaled_inp11, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX)); 
            AE_MIN32(AE_MAX32(scaled_inp12, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));

            AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
            AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
            AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
            AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);

            out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
            out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
            out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
            out12_2 = AE_SRAA64(norm_inp12_2, finalShift);

            AE_SAT32X2_HIFI4(sat_out11, out11_1, out11_2);
            AE_SAT32X2_HIFI4(sat_out12, out12_1, out12_2);

            sat_out = AE_SAT16X4(sat_out11, sat_out12);
            sat_out = AE_MOVINT16X4_FROMF16X4(AE_SLAI16S(AE_MOVF16X4_FROMINT16X4(sat_out), 8));
            sat_out = AE_SRAI16(sat_out, 8);

            ptu_out[0] = (WORD8)AE_MOVAD16_3(sat_out);
            ptu_out[1] = (WORD8)AE_MOVAD16_2(sat_out);
            ptu_out[2] = (WORD8)AE_MOVAD16_1(sat_out);
            ptu_out[3] = (WORD8)AE_MOVAD16_0(sat_out);

            ptu_inp += 4; ptu_out += 4;
        }
        if(iremc)
        {
          norm_inp11_1 = round_cnst;
          norm_inp11_2 = round_cnst;
          norm_inp12_1 = round_cnst;
          norm_inp12_2 = round_cnst;

          WORD8 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
          WORD16 outMult_b0 = 0, outMult_b1 = 0, outMult_b2 = 0, outMult_b3 = 0;
          WORD16* ptr_out_mult_w = (WORD16*)ptr_out_mult;
          if (iremc >= 1) {
            b0 = ptu_inp[0];
            outMult_b0 = *ptr_out_mult_w;
            ptr_out_mult_w++;
          }
          if (iremc >= 2) {
            b1 = ptu_inp[1];
            outMult_b1 = *ptr_out_mult_w;
            ptr_out_mult_w++;
          }
          if (iremc >= 3) {
            b2 = ptu_inp[2];
            outMult_b2 = *ptr_out_mult_w;
            ptr_out_mult_w++;
          }

          d0 = AE_MOVDA16X2(b0, b1);
          d1 = AE_MOVDA16X2(b2, b3);
          d_inp = AE_SEL16_5410(d0, d1);

          d0 = AE_MOVDA16X2(outMult_b0, outMult_b1);
          d1 = AE_MOVDA16X2(outMult_b2, outMult_b3);
          d_out_mult = AE_SEL16_5410(d0, d1);
          ptr_out_mult = (ae_int16x4 *)ptr_out_mult_w;

          d_scale1 = AE_MULFP32X16X2S_H(AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult));
          d_scale2 = AE_MULFP32X16X2S_L(AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult));
          d_scale = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale1), AE_MOVINT32X2_FROMF32X2(d_scale2));

          AE_MUL16X4(scaled_inp11, scaled_inp12, d_inp, d_scale);
          AE_MIN32(AE_MAX32(scaled_inp11, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX)); 
          AE_MIN32(AE_MAX32(scaled_inp12, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));

          AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
          AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
          AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
          AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);

          out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
          out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
          out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
          out12_2 = AE_SRAA64(norm_inp12_2, finalShift);

          AE_SAT32X2_HIFI4(sat_out11, out11_1, out11_2);
          AE_SAT32X2_HIFI4(sat_out12, out12_1, out12_2);

          sat_out = AE_SAT16X4(sat_out11, sat_out12);
          sat_out = AE_MOVINT16X4_FROMF16X4(AE_SLAI16S(AE_MOVF16X4_FROMINT16X4(sat_out), 8));
          sat_out = AE_SRAI16(sat_out, 8);

          if (iremc >= 1) ptu_out[0] = (WORD8)(AE_MOVAD16_3(sat_out));
          if (iremc >= 2) ptu_out[1] = (WORD8)(AE_MOVAD16_2(sat_out));
          if (iremc >= 3) ptu_out[2] = (WORD8)(AE_MOVAD16_1(sat_out));

          ptu_inp+=iremc; ptu_out+=iremc;
          }
      }
    }
  }
  
  //(accross_depth_flag == 1)
  else{
    int ic;
    const ae_int16 * ptr_inp_normdata = (const ae_int16 *)p_inp_normdata;
    WORD8 * ptr_nsa_shift = (WORD8 *)p_inp_nsadata;

    WORD8 nsaShift, finalShift;
    WORD32 nsa_mult_factor;
    ae_int32x2 d_nsa_multiplier;
    ae_f32x2 d_norm_t;     
    ae_int32x2 d_norm = ZERO32;

    ae_int16x4 d_inp, d0, d1, d_out_mult, d_norm_factor, d_mult, d_scale;
    ae_int64 norm_inp11_1, norm_inp11_2, norm_inp12_1, norm_inp12_2;
    ae_int64 out11_1, out11_2, out12_1, out12_2;
    ae_int32x2 sat_out11, sat_out12;
    ae_int16x4 sat_out;
    ae_int32x2 scaled_inp11, scaled_inp12;

    ae_f32x2 d_scale1 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 d_scale2 = AE_MOVF32X2_FROMINT32X2(ZERO32);

    if(per_chan_flag == 0)
    {
      d_mult = AE_MOVDA16(p_out_multiplier[0]);
      const WORD8 *ptu_inp = (const WORD8 *)&p_inp[0];
      WORD8 *ptu_out = (WORD8 *)&p_out[0];
      WORD32 iremc = input_channels & 3;

    for(int ihw = 0; ihw < input_height * input_width; ihw++)
    {
      AE_L16_IP(d_norm_factor, ptr_inp_normdata, 2);
      d_norm_t = AE_SEXT32X2D16_10(AE_MOVF16X4_FROMINT16X4(d_norm_factor));
      d_norm = AE_MOVINT32X2_FROMF32X2(d_norm_t);
      nsaShift    = *(ptr_nsa_shift++);
      finalShift  = out_rshift + ((nsaShift + 1) >> 1) + rsqrt_shift;
      nsa_mult_factor = (nsaShift & 0x1) ? 46341 : (1<<15);
      d_nsa_multiplier = SW_MOVDA32(nsa_mult_factor);

      d_scale1 = AE_MULFP32X16X2S_H(AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult));
      d_scale2 = AE_MULFP32X16X2S_L(AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult));
      d_scale = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale1), AE_MOVINT32X2_FROMF32X2(d_scale2));

      ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));

//#pragma concurrent
      for(ic = 0; ic < (inp_ch_lc); ic++)
      {
          norm_inp11_1 = round_cnst;
          norm_inp11_2 = round_cnst;
          norm_inp12_1 = round_cnst;
          norm_inp12_2 = round_cnst;
                    
          d0 = AE_MOVDA16X2(ptu_inp[0], ptu_inp[1]);
          d1 = AE_MOVDA16X2(ptu_inp[2], ptu_inp[3]);
          d_inp = AE_SEL16_5410(d0, d1);

          AE_MUL16X4(scaled_inp11, scaled_inp12, d_inp, d_scale);
          AE_MIN32(AE_MAX32(scaled_inp11, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));
          AE_MIN32(AE_MAX32(scaled_inp12, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));

          AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
          AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
          AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
          AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);

          out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
          out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
          out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
          out12_2 = AE_SRAA64(norm_inp12_2, finalShift);

          AE_SAT32X2_HIFI4(sat_out11, out11_1, out11_2);
          AE_SAT32X2_HIFI4(sat_out12, out12_1, out12_2);

          sat_out = AE_SAT16X4(sat_out11, sat_out12);
          sat_out = AE_MOVINT16X4_FROMF16X4(AE_SLAI16S(AE_MOVF16X4_FROMINT16X4(sat_out), 8));
          sat_out = AE_SRAI16(sat_out, 8);

          ptu_out[0] = (WORD8)(AE_MOVAD16_3(sat_out));
          ptu_out[1] = (WORD8)(AE_MOVAD16_2(sat_out));
          ptu_out[2] = (WORD8)(AE_MOVAD16_1(sat_out));
          ptu_out[3] = (WORD8)(AE_MOVAD16_0(sat_out));

          ptu_inp += 4; ptu_out += 4;
      }
      if(iremc){
        norm_inp11_1 = round_cnst;
        norm_inp11_2 = round_cnst;
        norm_inp12_1 = round_cnst;
        norm_inp12_2 = round_cnst;
                  
        WORD8 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        if (iremc >= 1) b0 = ptu_inp[0];
        if (iremc >= 2) b1 = ptu_inp[1];
        if (iremc >= 3) b2 = ptu_inp[2];

        d0 = AE_MOVDA16X2(b0, b1);
        d1 = AE_MOVDA16X2(b2, b3);
        d_inp = AE_SEL16_5410(d0, d1);

        AE_MUL16X4(scaled_inp11, scaled_inp12, d_inp, d_scale);
        AE_MIN32(AE_MAX32(scaled_inp11, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));
        AE_MIN32(AE_MAX32(scaled_inp12, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));

        AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
        AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
        AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
        AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);

        out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
        out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
        out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
        out12_2 = AE_SRAA64(norm_inp12_2, finalShift);

        AE_SAT32X2_HIFI4(sat_out11, out11_1, out11_2);
        AE_SAT32X2_HIFI4(sat_out12, out12_1, out12_2);

        sat_out = AE_SAT16X4(sat_out11, sat_out12);
        sat_out = AE_MOVINT16X4_FROMF16X4(AE_SLAI16S(AE_MOVF16X4_FROMINT16X4(sat_out), 8));
        sat_out = AE_SRAI16(sat_out, 8);

        if (iremc >= 1) ptu_out[0] = (WORD8)(AE_MOVAD16_3(sat_out));
        if (iremc >= 2) ptu_out[1] = (WORD8)(AE_MOVAD16_2(sat_out));
        if (iremc >= 3) ptu_out[2] = (WORD8)(AE_MOVAD16_1(sat_out));

        ptu_inp+=iremc; ptu_out+=iremc;
      }
    }
  }
    //(accross_depth_flag == 1) && (per_chan_flag == 1)
    else{
      const WORD8 *ptu_inp = (const WORD8 *)&p_inp[0];
      WORD8 *ptu_out = (WORD8 *)&p_out[0];
      WORD32 iremc = input_channels & 3;

      for(int ihw = 0; ihw < input_height * input_width; ihw++)
      {
        ae_int16x4 *ptr_out_mult = (ae_int16x4 *)p_out_multiplier;
        ae_valign a_out_mult = AE_LA64_PP(ptr_out_mult);

        AE_L16_IP(d_norm_factor, ptr_inp_normdata, 2);
        d_norm_t = AE_SEXT32X2D16_10(AE_MOVF16X4_FROMINT16X4(d_norm_factor));
        d_norm = AE_MOVINT32X2_FROMF32X2(d_norm_t);
        nsaShift    = *(ptr_nsa_shift++);
        finalShift  = out_rshift + ((nsaShift + 1) >> 1) + rsqrt_shift;
        nsa_mult_factor = (nsaShift & 0x1) ? 46341 : (1<<15);
        d_nsa_multiplier = SW_MOVDA32(nsa_mult_factor);

        ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));

  //#pragma concurrent
        for(ic = 0; ic < (inp_ch_lc); ic++)
        {
            norm_inp11_1 = round_cnst;
            norm_inp11_2 = round_cnst;
            norm_inp12_1 = round_cnst;
            norm_inp12_2 = round_cnst;
                      
            d0 = AE_MOVDA16X2(ptu_inp[0], ptu_inp[1]);
            d1 = AE_MOVDA16X2(ptu_inp[2], ptu_inp[3]);
            d_inp = AE_SEL16_5410(d0, d1);

            AE_LA16X4_IP(d_out_mult, a_out_mult, ptr_out_mult);

            d_scale1 = AE_MULFP32X16X2S_H(AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult));
            d_scale2 = AE_MULFP32X16X2S_L(AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult));
            d_scale = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale1), AE_MOVINT32X2_FROMF32X2(d_scale2));

            AE_MUL16X4(scaled_inp11, scaled_inp12, d_inp, d_scale);
            AE_MIN32(AE_MAX32(scaled_inp11, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));
            AE_MIN32(AE_MAX32(scaled_inp12, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));

            AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
            AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
            AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
            AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);

            out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
            out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
            out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
            out12_2 = AE_SRAA64(norm_inp12_2, finalShift);

            AE_SAT32X2_HIFI4(sat_out11, out11_1, out11_2);
            AE_SAT32X2_HIFI4(sat_out12, out12_1, out12_2);

            sat_out = AE_SAT16X4(sat_out11, sat_out12);
            sat_out = AE_MOVINT16X4_FROMF16X4(AE_SLAI16S(AE_MOVF16X4_FROMINT16X4(sat_out), 8));
            sat_out = AE_SRAI16(sat_out, 8);

            ptu_out[0] = (WORD8)(AE_MOVAD16_3(sat_out));
            ptu_out[1] = (WORD8)(AE_MOVAD16_2(sat_out));
            ptu_out[2] = (WORD8)(AE_MOVAD16_1(sat_out));
            ptu_out[3] = (WORD8)(AE_MOVAD16_0(sat_out));

            ptu_inp += 4; ptu_out += 4;
        }
        if(iremc){
          norm_inp11_1 = round_cnst;
          norm_inp11_2 = round_cnst;
          norm_inp12_1 = round_cnst;
          norm_inp12_2 = round_cnst;
                    
          WORD8 b0 = 0, b1 = 0, b2 = 0, b3 = 0; 
          WORD16 outMult_b0 = 0, outMult_b1 = 0, outMult_b2 = 0, outMult_b3 = 0;
          WORD16* ptr_out_mult_w = (WORD16*)ptr_out_mult;
          if (iremc >= 1) {
            b0 = ptu_inp[0];
            outMult_b0 = *ptr_out_mult_w;
            ptr_out_mult_w++;
          }
          if (iremc >= 2) {
            b1 = ptu_inp[1];
            outMult_b1 = *ptr_out_mult_w;
            ptr_out_mult_w++;
          }
          if (iremc >= 3) {
            b2 = ptu_inp[2];
            outMult_b2 = *ptr_out_mult_w;
            ptr_out_mult_w++;
          }

          d0 = AE_MOVDA16X2(b0, b1);
          d1 = AE_MOVDA16X2(b2, b3);
          d_inp = AE_SEL16_5410(d0, d1);

          d0 = AE_MOVDA16X2(outMult_b0, outMult_b1);
          d1 = AE_MOVDA16X2(outMult_b2, outMult_b3);
          d_out_mult = AE_SEL16_5410(d0, d1);
          ptr_out_mult = (ae_int16x4 *)ptr_out_mult_w;

          d_scale1 = AE_MULFP32X16X2S_H(AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult));
          d_scale2 = AE_MULFP32X16X2S_L(AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult));
          d_scale = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale1), AE_MOVINT32X2_FROMF32X2(d_scale2));

          AE_MUL16X4(scaled_inp11, scaled_inp12, d_inp, d_scale);
          AE_MIN32(AE_MAX32(scaled_inp11, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));
          AE_MIN32(AE_MAX32(scaled_inp12, AE_MOVDA32(INT_MIN)), AE_MOVDA32(INT_MAX));

          AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
          AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
          AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
          AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);

          out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
          out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
          out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
          out12_2 = AE_SRAA64(norm_inp12_2, finalShift);

          AE_SAT32X2_HIFI4(sat_out11, out11_1, out11_2);
          AE_SAT32X2_HIFI4(sat_out12, out12_1, out12_2);

          sat_out = AE_SAT16X4(sat_out11, sat_out12);
          sat_out = AE_MOVINT16X4_FROMF16X4(AE_SLAI16S(AE_MOVF16X4_FROMINT16X4(sat_out), 8));
          sat_out = AE_SRAI16(sat_out, 8);

          if (iremc >= 1) ptu_out[0] = (WORD8)(AE_MOVAD16_3(sat_out));
          if (iremc >= 2) ptu_out[1] = (WORD8)(AE_MOVAD16_2(sat_out));
          if (iremc >= 3) ptu_out[2] = (WORD8)(AE_MOVAD16_1(sat_out));

          ptu_inp+=iremc; ptu_out+=iremc;
        }
      }
  }
}
  return 0;
}

