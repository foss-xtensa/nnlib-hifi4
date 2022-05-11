/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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
#ifndef __XA_NNLIB_QUANT_MACROS_H__
#define __XA_NNLIB_QUANT_MACROS_H__

#if TFLITE_SINGLE_ROUNDING

#define MPY_BY_QUANT_MULT_X2_OUT32(out, inp, multiplier, l_shift, r_shift) \
{ \
  ae_int64 out64_0, out64_1; \
  out64_0 = AE_MUL32_HH(inp, AE_MOVDA32(multiplier)); \
  out64_1 = AE_MUL32_LL(inp, AE_MOVDA32(multiplier)); \
  out64_0 = AE_SLAA64S(out64_0, 1 + l_shift); \
  out64_1 = AE_SLAA64S(out64_1, 1 + l_shift); \
  out = AE_ROUND32X2F64SASYM(out64_0, out64_1); \
}

#define MPY_BY_QUANT_MULT_SLS_X2_OUT32(out, inp, multiplier, l_shift, r_shift) \
  MPY_BY_QUANT_MULT_X2_OUT32(out, inp, multiplier, l_shift, r_shift)

#define MPY_BY_QUANT_MULT_X2X2_OUT32(out1, out2, inp1, inp2, multiplier, l_shift, r_shift) \
{ \
  ae_int64 out64_0, out64_1, out64_2, out64_3; \
  out64_0 = AE_MUL32_HH(inp1, AE_MOVDA32(multiplier)); \
  out64_1 = AE_MUL32_LL(inp1, AE_MOVDA32(multiplier)); \
  out64_2 = AE_MUL32_HH(inp2, AE_MOVDA32(multiplier)); \
  out64_3 = AE_MUL32_LL(inp2, AE_MOVDA32(multiplier)); \
  out64_0 = AE_SLAA64S(out64_0, 1 + l_shift); \
  out64_1 = AE_SLAA64S(out64_1, 1 + l_shift); \
  out64_2 = AE_SLAA64S(out64_2, 1 + l_shift); \
  out64_3 = AE_SLAA64S(out64_3, 1 + l_shift); \
  out1 = AE_ROUND32X2F64SASYM(out64_0, out64_1); \
  out2 = AE_ROUND32X2F64SASYM(out64_2, out64_3); \
}

#define MPY_BY_QUANT_MULT_X2X2_OUT16(out, inp1, inp2, multiplier, l_shift, r_shift) \
{ \
  ae_int32x2 out32_0, out32_1; \
  MPY_BY_QUANT_MULT_X2X2_OUT32(out32_0, out32_1, inp1, inp2, multiplier, l_shift, r_shift) \
  out = AE_SAT16X4(out32_0, out32_1); \
}

#define MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(out, inp, multiplier, l_shift_0, l_shift_1, r_shift_0, r_shift_1) \
{ \
  ae_int64 out64_0, out64_1; \
  out64_0 = AE_MUL32_HH(inp, multiplier); \
  out64_1 = AE_MUL32_LL(inp, multiplier); \
  out64_0 = AE_SLAA64S(out64_0, 1 + l_shift_0); \
  out64_1 = AE_SLAA64S(out64_1, 1 + l_shift_1); \
  out = AE_ROUND32X2F64SASYM(out64_0, out64_1); \
}

#define MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(y, x, multiplier, lsh) \
  MPY_BY_QUANT_MULT_X2_OUT32(y, x, multiplier, lsh, lsh)

#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(prod, val, multiplier, lsh) \
  MPY_BY_QUANT_MULT_X2_OUT32(prod, val, multiplier, lsh, lsh)

#else /* #if TFLITE_SINGLE_ROUNDING */

#define MPY_BY_QUANT_MULT_X2_OUT32(out, inp, multiplier, l_shift, r_shift) \
  out = AE_SLAA32(inp, l_shift); \
  out = AE_MULFP32X2RAS(out, AE_MOVDA32(multiplier)); \
  out = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(out), r_shift), AE_SRAA64(AE_CVT64F32_L(out), r_shift));

#define MPY_BY_QUANT_MULT_SLS_X2_OUT32(out, inp, multiplier, l_shift, r_shift) \
  out = AE_SLAA32S(inp, l_shift); \
  out = AE_MULFP32X2RAS(out, AE_MOVDA32(multiplier)); \
  out = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(out), r_shift), AE_SRAA64(AE_CVT64F32_L(out), r_shift));

#define MPY_BY_QUANT_MULT_X2X2_OUT16(out, inp1, inp2, multiplier, l_shift, r_shift) \
{ \
  inp1 = AE_SLAA32S(inp1, l_shift); \
  inp1 = AE_MULFP32X2RAS(inp1, AE_NEG32(AE_MOVDA32(multiplier))); \
  inp1 = AE_MULFP32X2RS(inp1, AE_SRAA32(AE_MOVDA32(0x80000000), r_shift)); \
  inp2 = AE_SLAA32S(inp2, l_shift); \
  inp2 = AE_MULFP32X2RAS(inp2, AE_NEG32(AE_MOVDA32(multiplier))); \
  inp2 = AE_MULFP32X2RS(inp2, AE_SRAA32(AE_MOVDA32(0x80000000), r_shift)); \
  out = AE_SAT16X4(inp1, inp2); \
}

#define MPY_BY_QUANT_MULT_PER_CHAN_X2_OUT32(out, inp, multiplier, l_shift_0, l_shift_1, r_shift_0, r_shift_1) \
{ \
  ae_int64 accu1 = AE_MUL32_HH(inp, (1 << l_shift_0)); \
  ae_int64 accu2 = AE_MUL32_LL(inp, (1 << l_shift_1)); \
  out = AE_SEL32_HH(AE_MOVINT32X2_FROMINT64(AE_SLAI64S(accu1, 32)), AE_MOVINT32X2_FROMINT64(AE_SLAI64S(accu2, 32))); \
  out = AE_MULFP32X2RAS(out, multiplier); \
  out = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(out), r_shift_0), AE_SRAA64(AE_CVT64F32_L(out), r_shift_1)); \
}

#define MPY_BY_QUANT_MULT_X2X2_OUT32(out1, out2, inp1, inp2, multiplier, l_shift, r_shift) \
{ \
  ae_int32x2 d_ls = AE_MOVDA32(1<<l_shift); \
  ae_int64 accu1 = AE_MUL32_HL(inp1, d_ls); \
  ae_int64 accu2 = AE_MUL32_LL(inp1, d_ls); \
  ae_int64 accu3 = AE_MUL32_HL(inp2, d_ls); \
  ae_int64 accu4 = AE_MUL32_LL(inp2, d_ls); \
  out1 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu1), AE_MOVINT32X2_FROMINT64(accu2)); \
  out2 = AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(accu3), AE_MOVINT32X2_FROMINT64(accu4)); \
  out1 = AE_MULFP32X2RAS(out1, AE_MOVDA32(multiplier)); \
  out2 = AE_MULFP32X2RAS(out2, AE_MOVDA32(multiplier)); \
  out1 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(out1), r_shift), AE_SRAA64(AE_CVT64F32_L(out1), r_shift)); \
  out2 = AE_ROUND32X2F64SSYM(AE_SRAA64(AE_CVT64F32_H(out2), r_shift), AE_SRAA64(AE_CVT64F32_L(out2), r_shift)); \
}

#define MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(y, x, multiplier, lsh) { \
  y = AE_SLAA32(x, lsh); \
  y = AE_MULFP32X2RAS(y, AE_MOVDA32(multiplier)); \
}

#if XCHAL_HAVE_HIFI1
#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(prod, val, multiplier, lsh) {\
    ae_int64 temp64_h, temp64_l;\
    prod = AE_MULFP32X2RAS(val, multiplier);\
    temp64_h = AE_CVT64F32_H(prod);\
    temp64_l = AE_CVT64F32_L(prod);\
    temp64_h = AE_SLAA64S(temp64_h, lsh);\
    temp64_l = AE_SLAA64S(temp64_l, lsh);\
    prod = AE_ROUND32X2F64SSYM(temp64_h, temp64_l);\
}
#else
#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(prod, val, multiplier, lsh) {\
    ae_int64 temp64_h, temp64_l;\
    prod = AE_MULFP32X2RAS(val, AE_MOVDA32(multiplier));\
    temp64_h = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH((ae_int32x2)prod, AE_ZERO32()));\
    temp64_l = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL((ae_int32x2)prod, AE_ZERO32()));\
    temp64_h = AE_SLAA64S(temp64_h, lsh);\
    temp64_l = AE_SLAA64S(temp64_l, lsh);\
    prod = AE_ROUND32X2F64SSYM(temp64_h, temp64_l);\
}
#endif /* #if XCHAL_HAVE_HIFI1 */

#endif /* #if TFLITE_SINGLE_ROUNDING */

#endif /* #ifndef __XA_NNLIB_QUANT_MACROS_H__ */
