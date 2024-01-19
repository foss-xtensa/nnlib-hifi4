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
#include "xa_type_def.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2(inp1, inp2, multiplier, left_shift, right_shift) \
{\
  inp1 = AE_SLAA32S(inp1, left_shift); \
  inp1 = AE_MULFP32X2RAS(inp1, AE_NEG32(AE_MOVDA32(multiplier))); \
  inp1 = AE_MULFP32X2RS(inp1, right_shift); \
  inp2 = AE_SLAA32S(inp2, left_shift); \
  inp2 = AE_MULFP32X2RAS(inp2, AE_NEG32(AE_MOVDA32(multiplier))); \
  inp2 = AE_MULFP32X2RS(inp2, right_shift); \
}

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X1(inp, multiplier, left_shift, right_shift) \
{\
  inp = AE_SLAA32S(inp, left_shift); \
  inp = AE_MULFP32X2RAS(inp, AE_NEG32(AE_MOVDA32(multiplier))); \
  inp = AE_MULFP32X2RS(inp, right_shift); \
}

WORD32 xa_nn_vec_leaky_relu_asym16s_asym16s( WORD16 * __restrict__ p_out,
                    const   WORD16 * __restrict__ p_vec,
                            WORD32 inp_zero_bias,
                            WORD32 alpha_multiplier,
                            WORD32 alpha_shift,
                            WORD32 out_multiplier,
                            WORD32 out_shift,
                            WORD32 out_zero_bias,
                            WORD32 vec_length)
{
  int i;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((alpha_shift < -31) || (alpha_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((alpha_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -32768) || (out_zero_bias > 32767)), -1);

  int rem_length = (vec_length & 3);

  WORD16 *p_o = p_out;
  WORD16 *p_v = (WORD16 *)p_vec;

  ae_int32x2 inp_zb = AE_MOVDA32(inp_zero_bias);

  int left_shift  = out_shift<0?0: out_shift;
  int right_shift_orig = out_shift>0?0:-out_shift;
  int right_shift = (0XFFFFFFFF << (31 - right_shift_orig));

  int a_left_shift  = alpha_shift<0?0: alpha_shift;
  int a_right_shift_orig = alpha_shift>0?0:-alpha_shift;
  int a_right_shift = (0XFFFFFFFF << (31 - a_right_shift_orig));

  ae_valign align_src  = AE_LA64_PP((ae_int16x4 *)p_v);
  ae_valign align_dst = AE_ZALIGN64(); // zero alignment reg

#pragma concurrent
  for(i=0; i<(vec_length >> 2); i++)
  {
    ae_int16x4 d_inp0;
    ae_int32x2 d_w0_0, d_w0_1;
    ae_int32x2 d_alpha_w0_0, d_alpha_w0_1;

    AE_LA16X4_IP(d_inp0, align_src, (ae_int16x4 *)p_v);

    d_w0_0 = AE_SEXT32X2D16_32(d_inp0);
    d_w0_1 = AE_SEXT32X2D16_10(d_inp0);

    d_w0_0 = AE_SUB32(d_w0_0, inp_zb);
    d_w0_1 = AE_SUB32(d_w0_1, inp_zb);

    //Checking for input values less than zero
    xtbool2 sel0 = AE_LT32(d_w0_0, AE_ZERO32());
    xtbool2 sel1 = AE_LT32(d_w0_1, AE_ZERO32());

    d_alpha_w0_0 = d_w0_0; d_alpha_w0_1 = d_w0_1;

#if TFLITE_SINGLE_ROUNDING
    (void)right_shift; (void)a_right_shift;
    d_w0_0 = AE_SLAA32S(d_w0_0, left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(d_w0_0, d_w0_0, out_multiplier, -right_shift_orig);
    d_w0_1 = AE_SLAA32S(d_w0_1, left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(d_w0_1, d_w0_1, out_multiplier, -right_shift_orig);

    d_alpha_w0_0 = AE_SLAA32S(d_alpha_w0_0, a_left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(d_alpha_w0_0, d_alpha_w0_0, alpha_multiplier, -a_right_shift_orig);
    d_alpha_w0_1 = AE_SLAA32S(d_alpha_w0_1, a_left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(d_alpha_w0_1, d_alpha_w0_1, alpha_multiplier, -a_right_shift_orig);
#else
    // Multiply with out multiplier for input values >= 0
    MULTIPLYBYQUANTIZEDMULTIPLIER_X2(d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift);

    // Multiply with alpha multiplier for input values < 0
    MULTIPLYBYQUANTIZEDMULTIPLIER_X2(d_alpha_w0_0, d_alpha_w0_1, alpha_multiplier, a_left_shift, a_right_shift);
#endif

    AE_MOVT32X2(d_w0_0, d_alpha_w0_0, sel0);
    AE_MOVT32X2(d_w0_1, d_alpha_w0_1, sel1);

    d_w0_0 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_0);
    d_w0_1 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_1);

    ae_int16x4 out0;
    out0 = AE_SAT16X4(d_w0_0, d_w0_1);

    AE_SA16X4_IP(out0, align_dst, (ae_int16x4 *)p_o);
  }
#if (( XCHAL_HW_VERSION >= RI9_HWVERSION )& (XCHAL_HAVE_HIFI1))
{
    ae_int16x4 d_inp0;
    ae_int32x2 d_w0_0, d_w0_1;
    ae_int32x2 d_alpha_w0_0, d_alpha_w0_1;

    AE_LAV16X4_XP(d_inp0, align_src, (ae_int16x4 *)p_v, (rem_length<<1));

    d_w0_0 = AE_SEXT32X2D16_32(d_inp0);
    d_w0_1 = AE_SEXT32X2D16_10(d_inp0);

    d_w0_0 = AE_SUB32(d_w0_0, inp_zb);
    d_w0_1 = AE_SUB32(d_w0_1, inp_zb);

    //Checking for input values less than zero
    xtbool2 sel0 = AE_LT32(d_w0_0, AE_ZERO32());
    xtbool2 sel1 = AE_LT32(d_w0_1, AE_ZERO32());

    d_alpha_w0_0 = d_w0_0; d_alpha_w0_1 = d_w0_1;

#if TFLITE_SINGLE_ROUNDING
    (void)right_shift; (void)a_right_shift;
    d_w0_0 = AE_SLAA32S(d_w0_0, left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(d_w0_0, d_w0_0, out_multiplier, -right_shift_orig);
    d_w0_1 = AE_SLAA32S(d_w0_1, left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(d_w0_1, d_w0_1, out_multiplier, -right_shift_orig);

    d_alpha_w0_0 = AE_SLAA32S(d_alpha_w0_0, a_left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(d_alpha_w0_0, d_alpha_w0_0, alpha_multiplier, -a_right_shift_orig);
    d_alpha_w0_1 = AE_SLAA32S(d_alpha_w0_1, a_left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(d_alpha_w0_1, d_alpha_w0_1, alpha_multiplier, -a_right_shift_orig);
#else
    // Multiply with out multiplier for input values >= 0
    MULTIPLYBYQUANTIZEDMULTIPLIER_X2(d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift);

    // Multiply with alpha multiplier for input values < 0
    MULTIPLYBYQUANTIZEDMULTIPLIER_X2(d_alpha_w0_0, d_alpha_w0_1, alpha_multiplier, a_left_shift, a_right_shift);
#endif

    AE_MOVT32X2(d_w0_0, d_alpha_w0_0, sel0);
    AE_MOVT32X2(d_w0_1, d_alpha_w0_1, sel1);

    d_w0_0 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_0);
    d_w0_1 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_1);

    ae_int16x4 out0;
    out0 = AE_SAT16X4(d_w0_0, d_w0_1);

    AE_SAV16X4_XP(out0, align_dst, (ae_int16x4 *)p_o, (rem_length<<1));
}
  AE_SA64POS_FP(align_dst, p_o); // finalize the stream
#else
  AE_SA64POS_FP(align_dst, p_o); // finalize the stream

  //remainder loop for 3 elms
  for(i=0; i<rem_length; i++)
  {
    ae_int16x4 d_inp0;
    ae_int32x2 d_w0_0;
    ae_int32x2 d_alpha_w0_0;

    AE_L16_IP(d_inp0, (ae_int16 *)p_v, sizeof(ae_int16));

    d_w0_0 = AE_SEXT32X2D16_10(d_inp0);

    d_w0_0 = AE_SUB32(d_w0_0, inp_zb);

    //Checking for input values less than zero
    xtbool2 sel0 = AE_LT32(d_w0_0, AE_ZERO32());

    d_alpha_w0_0 = d_w0_0;

#if TFLITE_SINGLE_ROUNDING
    d_w0_0 = AE_SLAA32S(d_w0_0, left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(d_w0_0, d_w0_0, out_multiplier, -right_shift_orig);
    d_alpha_w0_0 = AE_SLAA32S(d_alpha_w0_0, a_left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(d_alpha_w0_0, d_alpha_w0_0, alpha_multiplier, -a_right_shift_orig);
#else
    // Multiply with out multiplier for input values >= 0
    MULTIPLYBYQUANTIZEDMULTIPLIER_X1(d_w0_0, out_multiplier, left_shift, right_shift);

    // Multiply with alpha multiplier for input values < 0
    MULTIPLYBYQUANTIZEDMULTIPLIER_X1(d_alpha_w0_0, alpha_multiplier, a_left_shift, a_right_shift);
#endif

    AE_MOVT32X2(d_w0_0, d_alpha_w0_0, sel0);

    d_w0_0 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_0);

    ae_int16x4 out0;
    out0 = AE_SAT16X4(d_w0_0, d_w0_0);

    AE_S16_0_IP(out0, (ae_int16 *)p_o, sizeof(ae_int16));
  }
#endif
  return 0;
}
