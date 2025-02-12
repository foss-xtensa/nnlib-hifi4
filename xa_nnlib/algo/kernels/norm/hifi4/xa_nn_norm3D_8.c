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

#define USHRT_MAX  65535
#define  SHRT_MIN -32768
#define  SHRT_MAX 32767

#define  SCHAR_MIN -128
#define  SCHAR_MAX 127

#define CALC_MPY_SHIFT_ROUND32(index, accum, multiplier, rshift, max_id) { \
    ae_int64 round_cnst = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)); \
    ae_int64 temp64 = AE_SLAA64(round_cnst, (rshift-1)); \
    AE_MULA32X16_L0(temp64, accum, multiplier); \
    index = AE_MOVINT32X2_FROMINT64(AE_SRAA64(temp64, rshift)); \
    index = AE_MIN32(index, AE_MOVDA32(max_id)); \
}

#if XCHAL_HAVE_HIFI1
#if XCHAL_HAVE_HIFI1S
static inline void internal_norm_vec(int64_t * p_acc, const WORD8 *p_vec, int cols1)
{
  int c_itr;
  ae_int32x2 acc = 0;  

  ae_int8x8 d_vec0;
  ae_valign align_vec = AE_LA64_PP(p_vec);

  for(c_itr = 0; c_itr < cols1>>3; c_itr++){            
    AE_LA8X8_IP(d_vec0, align_vec, (ae_int8x8 *)p_vec);
    AE_MULAAAAQ8(acc, d_vec0, d_vec0);
  }

  if(cols1&0x07){
    AE_LAV8X8_XP(d_vec0, align_vec, (ae_int8x8 *)p_vec, cols1&0x7);
    AE_MULAAAAQ8(acc, d_vec0, d_vec0);
  }
  acc = AE_ADD32_HL_LH(acc, acc);

  *(ae_int64*)p_acc = (ae_int64)AE_MOVAD32_L(acc);
}
#else
static inline void internal_norm_vec(int64_t * p_acc, const WORD8 *p_vec, int cols1)
{
  int c_itr;
  ae_int64 acc = 0;  

  ae_int16x4 d_vec0;
  ae_valign align_vec = AE_LA64_PP(p_vec);

  for(c_itr = 0; c_itr < cols1>>2; c_itr++){            
    AE_LA8X4S_IP(d_vec0, align_vec, p_vec);
    AE_MULAAAAQ16(acc, d_vec0, d_vec0);
  }

  if(cols1&0x03){
    AE_LAV8X4S_XP(d_vec0, align_vec, (ae_int8x4 *)p_vec, cols1&0x3);
    AE_MULAAAAQ16(acc, d_vec0, d_vec0);
  }

  *(ae_int64*)p_acc = acc;
}
#endif
#else
static inline void internal_norm_vec(int64_t * p_acc, const WORD8 *p_vec, int cols1)
{
  int c_itr;
  ae_int64 acc = 0, acc1 = 0;  
  WORD32 sum_mzb32 = 0;

  // pre-loop
  WORD32 preloop_cnt = (4 - ((unsigned)p_vec-(((unsigned)p_vec)&~0x3))) & 0x03;
  if(preloop_cnt > cols1) { preloop_cnt = 0;}
  cols1 = cols1 - preloop_cnt;

  for(c_itr = 0; c_itr < preloop_cnt; c_itr++){
    WORD8 vecval = *p_vec++;
    sum_mzb32 += vecval*vecval;
  }

  // aligned core loop
  acc = (ae_int64)sum_mzb32;
  acc = AE_SLAI64(acc, 16);
  ae_int16x4 d_vec0, d_vec1;

  for(c_itr = 0; c_itr < cols1>>3; c_itr++){            
    d_vec1 = AE_L8X4F_I(p_vec, 4);                  
    AE_L8X4F_IP(d_vec0, p_vec, 8);
    AE_MULAAAAQ16(acc, d_vec0, d_vec0);
    AE_MULAAAAQ16(acc1, d_vec1, d_vec1);
  }
  acc = AE_ADD64(acc, acc1);

  if( (cols1%8) >= 4) {
    AE_L8X4F_IP(d_vec0, p_vec, 8);
    AE_MULAAAAQ16(acc, d_vec0, d_vec0);
  }
  acc = AE_SRAI64(acc, 16);

  // remainder loop
  sum_mzb32 = AE_MOVINT32X2_FROMINT64(acc);

  for(c_itr = 0; c_itr < (cols1&0x3); c_itr++){
    int vecval = *p_vec++;
    sum_mzb32 += vecval*vecval;
  }

  acc = (ae_int64)sum_mzb32;
  *(ae_int64*)p_acc = acc;
}
#endif

#if XCHAL_HAVE_HIFI1
static inline void internal_norm_max_vecx2(UWORD8 * p_maxval8_0, UWORD8 *p_maxval8_1, int64_t * p_acc0, int64_t *p_acc1, const WORD8 *p_vec0, const WORD8 *p_vec1, int cols1)
{
  int c_itr;
  ae_int64 acc0 = 0, acc1 = 0;
  ae_int16x4 maxval16_0 = 0, maxval16_1 = 0;

  ae_int16x4 d_vec0, d_vec1;

  for(c_itr = 0; c_itr < cols1>>2; c_itr++){            
    AE_L8X4S_IP(d_vec0, p_vec0, 4);       
    AE_L8X4S_IP(d_vec1, p_vec1, 4);       
    AE_MULAAAAQ16(acc0, d_vec0, d_vec0);
    AE_MULAAAAQ16(acc1, d_vec1, d_vec1);
    d_vec0 = AE_ABS16S(d_vec0);
    d_vec1 = AE_ABS16S(d_vec1);

    maxval16_0 = AE_MAX16(maxval16_0, d_vec0);
    maxval16_1 = AE_MAX16(maxval16_1, d_vec1);
  }

  ae_int16x4 out_max = maxval16_0;
  out_max = AE_MAX16(out_max, AE_SEL16_4321(maxval16_0, maxval16_0));
  out_max = AE_MAX16(out_max, AE_SEL16_5432(maxval16_0, maxval16_0));
  out_max = AE_MAX16(out_max, AE_SEL16_6543(maxval16_0, maxval16_0));
  maxval16_0 = out_max;

  out_max = maxval16_1;
  out_max = AE_MAX16(out_max, AE_SEL16_4321(maxval16_1, maxval16_1));
  out_max = AE_MAX16(out_max, AE_SEL16_5432(maxval16_1, maxval16_1));
  out_max = AE_MAX16(out_max, AE_SEL16_6543(maxval16_1, maxval16_1));
  maxval16_1 = out_max;

  *(ae_int64*)p_acc0 = acc0;
  *(ae_int64*)p_acc1 = acc1;
  *p_maxval8_0 = (UWORD8)AE_MOVAD16_3(maxval16_0);
  *p_maxval8_1 = (UWORD8)AE_MOVAD16_3(maxval16_1);
}
#else
static inline void internal_norm_max_vecx2(UWORD8 * p_maxval8_0, UWORD8 *p_maxval8_1, int64_t * p_acc0, int64_t *p_acc1, const WORD8 *p_vec0, const WORD8 *p_vec1, int cols1)
{
  int c_itr;
  ae_int64 acc0 = 0, acc1 = 0;
  ae_int32x2 maxval0 = 0, maxval1 = 0;
  ae_int16x4 maxval16_0 = 0, maxval16_1 = 0;

  ae_int16x4 d_vec0, d_vec1;
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);

  for(c_itr = 0; c_itr < cols1>>2; c_itr++){            
    AE_L8X4F_IP(d_vec0, p_vec0, 4);       
    AE_L8X4F_IP(d_vec1, p_vec1, 4);       
    d_vec0 = AE_SRAI16(d_vec0, 8);
    d_vec1 = AE_SRAI16(d_vec1, 8);
    AE_MULAAAAQ16(acc0, d_vec0, d_vec0);
    AE_MULAAAAQ16(acc1, d_vec1, d_vec1);
    d_vec0 = AE_ABS16S(d_vec0);
    d_vec1 = AE_ABS16S(d_vec1);

    xtbool4 bool4;
    bool4 = AE_LT16(maxval16_0, d_vec0);
    AE_MOVT16X4(maxval16_0, d_vec0, bool4);
    bool4 = AE_LT16(maxval16_1, d_vec1);
    AE_MOVT16X4(maxval16_1, d_vec1, bool4);
  }

  ae_int32x2 maxtemp0, maxtemp1;
  AE_MUL16X4(maxtemp0, maxtemp1, maxval16_0, ONE_16X4);
  maxval0 = AE_MAX32(maxtemp0, maxtemp1);
  maxval0 = AE_MAX32(maxval0, AE_SEL32_LH(maxval0, maxval0));  

  AE_MUL16X4(maxtemp0, maxtemp1, maxval16_1, ONE_16X4);
  maxval1 = AE_MAX32(maxtemp0, maxtemp1);
  maxval1 = AE_MAX32(maxval1, AE_SEL32_LH(maxval1, maxval1));  

  *(ae_int64*)p_acc0 = acc0;
  *(ae_int64*)p_acc1 = acc1;
  *p_maxval8_0 = (UWORD8)AE_MOVAD32_L(maxval0);
  *p_maxval8_1 = (UWORD8)AE_MOVAD32_L(maxval1);
}
#endif

#if XCHAL_HAVE_HIFI1
static inline void internal_norm_max_vec(UWORD8 * p_maxval8, int64_t * p_acc, const WORD8 *p_vec, int cols1)
{
  int c_itr;
  ae_int64 acc = 0;  
  ae_int16x4 maxval=0;
  WORD32 sum_mzb32 = 0;

  acc = (ae_int64)sum_mzb32;
  ae_int16x4 d_vec0;

  ae_valign align_vec = AE_LA64_PP(p_vec);

  for(c_itr = 0; c_itr < cols1>>2; c_itr++){            
    AE_LA8X4S_IP(d_vec0, align_vec, p_vec);
    AE_MULAAAAQ16(acc, d_vec0, d_vec0);
    d_vec0 = AE_ABS16S(d_vec0);
    maxval = AE_MAX16(maxval, d_vec0);
  }
  int rem = cols1&0x03;

  if(rem){
    AE_LAV8X4S_XP(d_vec0, align_vec, (ae_int8x4 *)p_vec, rem);
    AE_MULAAAAQ16(acc, d_vec0, d_vec0);
    d_vec0 = AE_ABS16S(d_vec0);
    maxval = AE_MAX16(maxval, d_vec0);
  }
  
  ae_int16x4 out_max = maxval;
  out_max = AE_MAX16(out_max, AE_SEL16_4321(maxval, maxval));
  out_max = AE_MAX16(out_max, AE_SEL16_5432(maxval, maxval));
  out_max = AE_MAX16(out_max, AE_SEL16_6543(maxval, maxval));

  *(ae_int64*)p_acc = acc;
  *p_maxval8 = (UWORD8)AE_MOVAD16_3(out_max);
}

#else
static inline void internal_norm_max_vec(UWORD8 * p_maxval8, int64_t * p_acc, const WORD8 *p_vec, int cols1)
{
  int c_itr;
  ae_int64 acc = 0;  
  ae_int32x2 maxval=0;
  WORD32 sum_mzb32 = 0;

  // pre-loop
  WORD32 preloop_cnt = (4 - ((unsigned)p_vec-(((unsigned)p_vec)&~0x3))) & 0x03;
  if(preloop_cnt > cols1) { preloop_cnt = 0;}
  cols1 = cols1 - preloop_cnt;

  for(c_itr = 0; c_itr < preloop_cnt; c_itr++){
    WORD8 vecval = *p_vec++;
    sum_mzb32 += vecval*vecval;
    WORD32 absval =  AE_ABS32(AE_MOVDA32(vecval));
    maxval = AE_MAX32(maxval, absval);
  }

  // aligned core loop
  acc = (ae_int64)sum_mzb32;
  ae_int16x4 d_vec0;
  ae_int16x4 ONE_16X4 = AE_MOVDA16(1);

  for(c_itr = 0; c_itr < cols1>>2; c_itr++){            
    ae_int32x2 d_abs0, d_abs1;
    AE_L8X4F_IP(d_vec0, p_vec, 4);                  
    d_vec0 = AE_SRAI16(d_vec0, 8);
    AE_MULAAAAQ16(acc, d_vec0, d_vec0);
    d_vec0 = AE_ABS16S(d_vec0);
    AE_MUL16X4(d_abs0, d_abs1, d_vec0, ONE_16X4);
    maxval = AE_MAX32(maxval, d_abs0);
    maxval = AE_MAX32(maxval, d_abs1);
  }

  // remainder loop
  sum_mzb32 = AE_MOVINT32X2_FROMINT64(acc);

  for(c_itr = 0; c_itr < (cols1&0x3); c_itr++){
    int vecval = *p_vec++;
    sum_mzb32 += vecval*vecval;
    WORD32 absval =  AE_ABS32(AE_MOVDA32(vecval));
    maxval = AE_MAX32(maxval, absval);
  }

  maxval = AE_MAX32(maxval, AE_SEL32_LH(maxval, maxval));  

  acc = (ae_int64)sum_mzb32;
  *(ae_int64*)p_acc = acc;
  *p_maxval8 = (UWORD8)AE_MOVAD32_L(maxval);
}
#endif

WORD32 xa_nn_norm_calc_3D_8_nhwc(
    WORD16 * p_out /*Noram data: 2D -> iw*ih, or scalar*/ , 
    const WORD8 * p_inp /*3D -> iw*ih*ic */,
    int input_height, int input_width, int input_channels, 
    int accross_depth_flag,
    int out_shift, /*sumSquareShift*/
    const UWORD16 *prsqrt, int rsqrt_shift, int rsqrt_table_len, /* rsqrt table */
    const UWORD16 *precip, int recip_shift) /* recip table */
{

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(prsqrt, -1);
  XA_NNLIB_ARG_CHK_PTR(precip, -1);

  /* Pointer Alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(prsqrt, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(precip, sizeof(WORD16), -1);

  /* Param Checks*/
  XA_NNLIB_ARG_CHK_COND((out_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((rsqrt_table_len <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((accross_depth_flag != 0) && (accross_depth_flag != 1), -1);

  /* The out_shit param is passed as left shift. Create right-shift oot_rshift by negating the same. */
  int out_rshift = -out_shift;

  /* Shift checks */
  XA_NNLIB_ARG_CHK_COND((recip_shift-out_rshift < 0), -1);
  XA_NNLIB_ARG_CHK_COND((recip_shift-rsqrt_shift < 0), -1);
  XA_NNLIB_ARG_CHK_COND((recip_shift< 0), -1);

  if(accross_depth_flag == 0) /* Calc norm data for entire 3D input */
  {
    int64_t accum = 0;
    int inp_len = input_height*input_width*input_channels;

    internal_norm_vec(&accum, p_inp, inp_len);

    ae_int64 round_cnst = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1));
    ae_int64 temp64 = AE_SLAA64(round_cnst, (out_rshift-1));
    temp64       = AE_ADD64(temp64, accum);
    ae_int32x2 tableIndex32 = AE_MOVINT32X2_FROMINT64(AE_SRAA64(temp64, out_rshift));
    tableIndex32 = AE_MIN32(tableIndex32, AE_MOVDA32(rsqrt_table_len -1));
    tableIndex32 = AE_MAX32(tableIndex32, AE_MOVDA32(0));
    int tableIndex   = (UWORD16) AE_MOVAD32_L(tableIndex32);
    p_out[0] = prsqrt[tableIndex];
  }
  else /* Calc norm data across depth dimension only */
  {
    int ih, iw;
    
    for(ih = 0; ih < input_height; ih++)
    {
      iw = 0;

      for(; iw < (input_width&~0x01); iw+=2)
      {
        int offset0 = ih*input_width*input_channels + iw*input_channels;
        int offset1 = ih*input_width*input_channels + (iw+1)*input_channels;
        const WORD8 *p_inp_ch0 = &p_inp[offset0];
        const WORD8 *p_inp_ch1 = &p_inp[offset1];
        int64_t accum0 = 0, accum1 = 0;
        UWORD8 maxval0 = 0, maxval1 = 0;

        if(  ( ((unsigned)p_inp_ch0%4) == 0) && ( ((unsigned)p_inp_ch1%4) == 0) && ( (input_channels%4)==0 ) )
        {
          internal_norm_max_vecx2(&maxval0, &maxval1, &accum0, &accum1, p_inp_ch0, p_inp_ch1, input_channels);
        }
        else
        {
          internal_norm_max_vec(&maxval0, &accum0, p_inp_ch0, input_channels);
          internal_norm_max_vec(&maxval1, &accum1, p_inp_ch1, input_channels);
        }
        WORD32 accum32_0 = (WORD32)accum0;
        WORD32 accum32_1 = (WORD32)accum1;

        ae_int32x2 accum32x2_0 = accum32_0;
        ae_int32x2 accum32x2_1 = accum32_1;
        ae_int32x2 norm16u_0, norm16u_1;
        ae_int32x2 tableIndex32_0, tableIndex32_1;

        norm16u_0 = AE_SRAA32RS(accum32x2_0, out_rshift);
        norm16u_0 = AE_MIN32(norm16u_0, AE_MOVDA32(USHRT_MAX));
        norm16u_1 = AE_SRAA32RS(accum32x2_1, out_rshift);
        norm16u_1 = AE_MIN32(norm16u_1, AE_MOVDA32(USHRT_MAX));

        ae_int16x4 recip_dmax0 = AE_MOVDA16(precip[maxval0]);
        ae_int16x4 recip_dmax1 = AE_MOVDA16(precip[maxval1]);

        CALC_MPY_SHIFT_ROUND32(tableIndex32_0, norm16u_0, recip_dmax0, (recip_shift-out_rshift), USHRT_MAX);
        CALC_MPY_SHIFT_ROUND32(tableIndex32_1, norm16u_1, recip_dmax1, (recip_shift-out_rshift), USHRT_MAX);
        ae_int32x2 index_temp0 = tableIndex32_0;
        ae_int32x2 index_temp1 = tableIndex32_1;
        CALC_MPY_SHIFT_ROUND32(tableIndex32_0, index_temp0, recip_dmax0, (recip_shift-rsqrt_shift), (rsqrt_table_len-1));
        CALC_MPY_SHIFT_ROUND32(tableIndex32_1, index_temp1, recip_dmax1, (recip_shift-rsqrt_shift), (rsqrt_table_len-1));
        
        UWORD16 tableIndex0 = (UWORD16)AE_MOVAD32_L(tableIndex32_0);
        ae_int32x2 rsqrtval0 = AE_MOVDA32(prsqrt[tableIndex0]);
        UWORD16 tableIndex1 = (UWORD16)AE_MOVAD32_L(tableIndex32_1);
        ae_int32x2 rsqrtval1 = AE_MOVDA32(prsqrt[tableIndex1]);

        index_temp0 = rsqrtval0;
        index_temp1 = rsqrtval1;
        CALC_MPY_SHIFT_ROUND32(rsqrtval0, index_temp0, recip_dmax0, recip_shift, SHRT_MAX);
        CALC_MPY_SHIFT_ROUND32(rsqrtval1, index_temp1, recip_dmax1, recip_shift, SHRT_MAX);

        rsqrtval0 = AE_MAX32(rsqrtval0, AE_MOVDA32(SHRT_MIN));
        p_out[iw + (ih * input_width)] = (WORD16)AE_MOVAD32_L(rsqrtval0);
        rsqrtval1 = AE_MAX32(rsqrtval1, AE_MOVDA32(SHRT_MIN));
        p_out[iw + 1 + (ih * input_width)] = (WORD16)AE_MOVAD32_L(rsqrtval1);
      }

      for(; iw < input_width; iw++)
      {
        int offset = ih*input_width*input_channels + iw*input_channels;
        const WORD8 *p_inp_ch = &p_inp[offset];
        int64_t accum = 0;
        UWORD8 maxval = 0;

        internal_norm_max_vec(&maxval, &accum, p_inp_ch, input_channels);
        WORD32 accum32 = (WORD32)accum;

        ae_int32x2 accum32x2 = accum32;
        ae_int32x2 norm16u;
        ae_int32x2 tableIndex32;

        norm16u = AE_SRAA32RS(accum32x2, out_rshift);
        norm16u = AE_MIN32(norm16u, AE_MOVDA32(USHRT_MAX));

        ae_int16x4 recip_dmax = AE_MOVDA16(precip[maxval]);

        CALC_MPY_SHIFT_ROUND32(tableIndex32, norm16u, recip_dmax, (recip_shift-out_rshift), USHRT_MAX);
        ae_int32x2 index_temp = tableIndex32;
        CALC_MPY_SHIFT_ROUND32(tableIndex32, index_temp, recip_dmax, (recip_shift-rsqrt_shift), (rsqrt_table_len-1));
        
        UWORD16 tableIndex = (UWORD16)AE_MOVAD32_L(tableIndex32);
        ae_int32x2 rsqrtval = AE_MOVDA32(prsqrt[tableIndex]);

        index_temp = rsqrtval;
        CALC_MPY_SHIFT_ROUND32(rsqrtval, index_temp, recip_dmax, recip_shift, SHRT_MAX);

        rsqrtval = AE_MAX32(rsqrtval, AE_MOVDA32(SHRT_MIN));
        p_out[iw + (ih * input_width)] = (WORD16)AE_MOVAD32_L(rsqrtval);
      }
    }
  }

  return 0;
}

#if XCHAL_HAVE_HIFI1
static inline void __attribute__((always_inline))  internal_apply_1D(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_inp,
    WORD16 norm_factor,
    int inlen,
    WORD16 * __restrict__ p_out_multiplier,
    int out_multiplier_offset,
    int out_rshift,
    int rsqrt_shift)
{
  int ic;

  if(out_multiplier_offset == 0){
    WORD16 multiplier = p_out_multiplier[0];
    ae_int16x4 d_multiplier = AE_MOVDA16(multiplier);

    ae_valign align_in = AE_LA64_PP(p_inp);
    ae_valign align_out = AE_ZALIGN64();

    for(ic = 0; ic < (inlen>>2); ic++) {
      ae_int16x4 val;
      AE_LA8X4S_IP(val, align_in, p_inp);

      ae_int32x2 acc0, acc1;
      AE_MUL16X4(acc0, acc1, val, d_multiplier);
      acc0 = AE_SRAA32RS(acc0, out_rshift);
      acc1 = AE_SRAA32RS(acc1, out_rshift);

      ae_int16x4 acc16 = AE_SAT16X4(acc0, acc1);
      AE_MUL16X4(acc0, acc1, acc16, AE_MOVDA16(norm_factor));
      acc0 = AE_SRAA32RS(acc0, rsqrt_shift);
      acc1 = AE_SRAA32RS(acc1, rsqrt_shift);

      acc16 = AE_SAT16X4(acc0, acc1);
      acc16 = AE_SAT8S(acc16);
    
      AE_SA8X4U_IP(acc16, align_out, (ae_int32 *)p_out);
    }

    int rem = inlen&0x03;

    if(rem) {
      ae_int16x4 val;
      AE_LAV8X4S_XP(val, align_in, (ae_int8x4 *)p_inp, rem);

      ae_int32x2 acc0, acc1;
      AE_MUL16X4(acc0, acc1, val, d_multiplier);
      acc0 = AE_SRAA32RS(acc0, out_rshift);
      acc1 = AE_SRAA32RS(acc1, out_rshift);

      ae_int16x4 acc16 = AE_SAT16X4(acc0, acc1);
      AE_MUL16X4(acc0, acc1, acc16, AE_MOVDA16(norm_factor));
      acc0 = AE_SRAA32RS(acc0, rsqrt_shift);
      acc1 = AE_SRAA32RS(acc1, rsqrt_shift);

      acc16 = AE_SAT16X4(acc0, acc1);
      acc16 = AE_SAT8S(acc16);
    
      AE_SAV8X4U_XP(acc16, align_out, (ae_int8x4u *)p_out, rem);
    }
    AE_SA64POS_FP(align_out, p_out);

  } else {
    
    ae_valign align_m = AE_LA64_PP(p_out_multiplier);
    ae_valign align_in = AE_LA64_PP(p_inp);
    ae_valign align_out = AE_ZALIGN64();

    for(ic = 0; ic < (inlen>>2); ic++) {
      ae_int16x4 multiplier, val;
      AE_LA16X4_IP(multiplier, align_m, (ae_int16x4 *)p_out_multiplier);
      AE_LA8X4S_IP(val, align_in, p_inp);

      ae_int32x2 acc0, acc1;
      AE_MUL16X4(acc0, acc1, val, multiplier);
      acc0 = AE_SRAA32RS(acc0, out_rshift);
      acc1 = AE_SRAA32RS(acc1, out_rshift);

      ae_int16x4 acc16 = AE_SAT16X4(acc0, acc1);
      AE_MUL16X4(acc0, acc1, acc16, AE_MOVDA16(norm_factor));
      acc0 = AE_SRAA32RS(acc0, rsqrt_shift);
      acc1 = AE_SRAA32RS(acc1, rsqrt_shift);

      acc16 = AE_SAT16X4(acc0, acc1);
      acc16 = AE_SAT8S(acc16);
    
      AE_SA8X4U_IP(acc16, align_out, (ae_int32 *)p_out);
    }

    int rem = inlen&0x03;
    if(rem) {
      ae_int16x4 multiplier, val;
      AE_LAV16X4_XP(multiplier, align_m, (ae_int16x4 *)p_out_multiplier, rem*2);
      AE_LAV8X4S_XP(val, align_in, (ae_int8x4 *)p_inp, rem);

      ae_int32x2 acc0, acc1;
      AE_MUL16X4(acc0, acc1, val, multiplier);
      acc0 = AE_SRAA32RS(acc0, out_rshift);
      acc1 = AE_SRAA32RS(acc1, out_rshift);

      ae_int16x4 acc16 = AE_SAT16X4(acc0, acc1);
      AE_MUL16X4(acc0, acc1, acc16, AE_MOVDA16(norm_factor));
      acc0 = AE_SRAA32RS(acc0, rsqrt_shift);
      acc1 = AE_SRAA32RS(acc1, rsqrt_shift);

      acc16 = AE_SAT16X4(acc0, acc1);
      acc16 = AE_SAT8S(acc16);
    
      AE_SAV8X4U_XP(acc16, align_out, (ae_int8x4u *)p_out, rem);
    }
    AE_SA64POS_FP(align_out, p_out);

  }
}

#else
static inline void __attribute__((always_inline))  internal_apply_1D(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_inp,
    WORD16 norm_factor,
    int inlen,
    WORD16 * __restrict__ p_out_multiplier,
    int out_multiplier_offset,
    int out_rshift,
    int rsqrt_shift)
{
  int ic;

  if(out_multiplier_offset == 0){
    WORD16 multiplier = p_out_multiplier[0];
    ae_int16x4 d_multiplier = AE_MOVDA16(multiplier);

    /* preloop */
    WORD32 preloop_cnt = (4 - ((unsigned)p_inp-(((unsigned)p_inp)&~0x3))) & 0x03;
    if(preloop_cnt > inlen) { preloop_cnt = 0;}
    inlen = inlen - preloop_cnt;

    for(ic = 0; ic < preloop_cnt; ic++){
      WORD8 val = *p_inp++;
      ae_int32x2 acc = AE_MOVDA32(val*multiplier);
      //WORD32 temp1 = (int32_t) xaiRoundAndClamp32(acc, out_rshift, SHRT_MIN, SHRT_MAX);
      //*p_out++ = (int8_t) xaiRoundAndClamp32(temp1 * norm_factor, rsqrt_shift, SCHAR_MIN, SCHAR_MAX);
      acc = AE_SRAA32RS(acc, out_rshift);
      ae_int16x4 acc16 = AE_SAT16X4(acc, acc);
      ae_int32x2 dummy;
      AE_MUL16X4(dummy, acc, acc16, AE_MOVDA16(norm_factor));
      acc = AE_SRAA32RS(acc, rsqrt_shift);
      AE_MINMAX32(acc, AE_MOVDA32(SCHAR_MIN), AE_MOVDA32(SCHAR_MAX));
      *p_out++ = (WORD8)AE_MOVAD32_L(acc);
    }

    for(ic = 0; ic < (inlen>>2); ic++) {
      ae_int16x4 val;
      AE_L8X4F_IP(val, p_inp, 4);
      //val = AE_SRAI16(val, 8); // Adjusted in shift below

      ae_int32x2 acc0, acc1;
      AE_MUL16X4(acc0, acc1, val, d_multiplier);
      acc0 = AE_SRAA32RS(acc0, out_rshift+8);
      acc1 = AE_SRAA32RS(acc1, out_rshift+8);

      ae_int16x4 acc16 = AE_SAT16X4(acc0, acc1);
      AE_MUL16X4(acc0, acc1, acc16, AE_MOVDA16(norm_factor));
      acc0 = AE_SRAA32RS(acc0, rsqrt_shift);
      acc1 = AE_SRAA32RS(acc1, rsqrt_shift);
      AE_MINMAX32(acc0, AE_MOVDA32(SCHAR_MIN), AE_MOVDA32(SCHAR_MAX));
      AE_MINMAX32(acc1, AE_MOVDA32(SCHAR_MIN), AE_MOVDA32(SCHAR_MAX));
      *p_out++ = (WORD8)AE_MOVAD32_H(acc0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc0);
      *p_out++ = (WORD8)AE_MOVAD32_H(acc1);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc1);
    }

    for(ic = 0; ic < (inlen&0x3); ic++) {
      WORD8 val = *p_inp++;
      ae_int32x2 acc = AE_MOVDA32(val*multiplier);
      acc = AE_SRAA32RS(acc, out_rshift);
      ae_int16x4 acc16 = AE_SAT16X4(acc, acc);
      ae_int32x2 dummy;
      AE_MUL16X4(dummy, acc, acc16, AE_MOVDA16(norm_factor));
      acc = AE_SRAA32RS(acc, rsqrt_shift);
      AE_MINMAX32(acc, AE_MOVDA32(SCHAR_MIN), AE_MOVDA32(SCHAR_MAX));
      *p_out++ = (WORD8)AE_MOVAD32_L(acc);
    }

  } else {

    /* preloop */
    WORD32 preloop_cnt = (4 - ((unsigned)p_inp-(((unsigned)p_inp)&~0x3))) & 0x03;
    if(preloop_cnt > inlen) { preloop_cnt = 0;}
    inlen = inlen - preloop_cnt;

    for(ic = 0; ic < preloop_cnt; ic++){
      WORD8 val = *p_inp++;
      WORD16 multiplier = *p_out_multiplier++;
      ae_int32x2 acc = AE_MOVDA32(val*multiplier);
      //WORD32 temp1 = (int32_t) xaiRoundAndClamp32(acc, out_rshift, SHRT_MIN, SHRT_MAX);
      //*p_out++ = (int8_t) xaiRoundAndClamp32(temp1 * norm_factor, rsqrt_shift, SCHAR_MIN, SCHAR_MAX);
      acc = AE_SRAA32RS(acc, out_rshift);
      ae_int16x4 acc16 = AE_SAT16X4(acc, acc);
      ae_int32x2 dummy;
      AE_MUL16X4(dummy, acc, acc16, AE_MOVDA16(norm_factor));
      acc = AE_SRAA32RS(acc, rsqrt_shift);
      AE_MINMAX32(acc, AE_MOVDA32(SCHAR_MIN), AE_MOVDA32(SCHAR_MAX));
      *p_out++ = (WORD8)AE_MOVAD32_L(acc);
    }

    ae_valign align_m = AE_LA64_PP(p_out_multiplier);
    for(ic = 0; ic < (inlen>>2); ic++) {
      ae_int16x4 multiplier, val;
      AE_LA16X4_IP(multiplier, align_m, (ae_int16x4 *)p_out_multiplier);
      AE_L8X4F_IP(val, p_inp, 4);
      //val = AE_SRAI16(val, 8); // Adjusted in shift below

      ae_int32x2 acc0, acc1;
      AE_MUL16X4(acc0, acc1, val, multiplier);
      acc0 = AE_SRAA32RS(acc0, out_rshift+8);
      acc1 = AE_SRAA32RS(acc1, out_rshift+8);

      ae_int16x4 acc16 = AE_SAT16X4(acc0, acc1);
      AE_MUL16X4(acc0, acc1, acc16, AE_MOVDA16(norm_factor));
      acc0 = AE_SRAA32RS(acc0, rsqrt_shift);
      acc1 = AE_SRAA32RS(acc1, rsqrt_shift);
      AE_MINMAX32(acc0, AE_MOVDA32(SCHAR_MIN), AE_MOVDA32(SCHAR_MAX));
      AE_MINMAX32(acc1, AE_MOVDA32(SCHAR_MIN), AE_MOVDA32(SCHAR_MAX));
      *p_out++ = (WORD8)AE_MOVAD32_H(acc0);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc0);
      *p_out++ = (WORD8)AE_MOVAD32_H(acc1);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc1);
    }

    for(ic = 0; ic < (inlen&0x3); ic++) {
      WORD16 multiplier = *p_out_multiplier++;
      WORD8 val = *p_inp++;
      ae_int32x2 acc = AE_MOVDA32(val*multiplier);
      acc = AE_SRAA32RS(acc, out_rshift);
      ae_int16x4 acc16 = AE_SAT16X4(acc, acc);
      ae_int32x2 dummy;
      AE_MUL16X4(dummy, acc, acc16, AE_MOVDA16(norm_factor));
      acc = AE_SRAA32RS(acc, rsqrt_shift);
      AE_MINMAX32(acc, AE_MOVDA32(SCHAR_MIN), AE_MOVDA32(SCHAR_MAX));
      *p_out++ = (WORD8)AE_MOVAD32_L(acc);
    }
  }
}
#endif

WORD32 xa_nn_norm_apply_3D_8_nhwc(
    WORD8 * p_out, 
    const WORD8 * p_inp, /*3D -> iw*ih*ic */
    WORD16 *p_inp_normdata,
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

  /* Pointer Alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_normdata, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD16), -1);

  /* Param Checks*/
  XA_NNLIB_ARG_CHK_COND((out_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((rsqrt_shift < 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((accross_depth_flag != 0) && (accross_depth_flag != 1), -1);
  XA_NNLIB_ARG_CHK_COND((per_chan_flag != 0) && (per_chan_flag != 1), -1);

  int out_rshift = -out_shift;

  int out_multiplier_offset = 1;
  if(per_chan_flag == 0) {
    out_multiplier_offset = 0;
  }

  if(accross_depth_flag == 0){
    int ih, iw;
    WORD16 norm_factor = p_inp_normdata[0];

    if(out_multiplier_offset == 0)
    {
      internal_apply_1D( p_out, p_inp, norm_factor, input_height*input_width*input_channels, p_out_multiplier, 0, out_rshift, rsqrt_shift);
    }
    else
    {
      for(ih = 0; ih < input_height; ih++)
      {
        for(iw = 0; iw < input_width; iw++)
        {
          int offset = ih*input_width*input_channels + iw*input_channels;
          const WORD8 *p_inp_ch = &p_inp[offset];
          WORD8 *p_out_ch = &p_out[offset];

          internal_apply_1D( p_out_ch, p_inp_ch, norm_factor, input_channels, p_out_multiplier, 1, out_rshift, rsqrt_shift);
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
        const WORD8 *p_inp_ch = &p_inp[offset];
        WORD8 *p_out_ch = &p_out[offset];
        WORD16 norm_factor = p_inp_normdata[ih*input_width + iw];

        internal_apply_1D( p_out_ch, p_inp_ch, norm_factor, input_channels, p_out_multiplier, out_multiplier_offset, out_rshift, rsqrt_shift);
      }
    }
  }

  return 0;
}

