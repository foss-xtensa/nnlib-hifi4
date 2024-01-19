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


#if XCHAL_HAVE_HIFI1
#define MAX_16X4(id1, id0) \
        id1 = AE_MAX16(id1, id0);\

#define MIN_16X4(id1, id0) \
        id1 = AE_MIN16(id1, id0);\

#define LIMIT(out, inp, min, max){\
        out = min;\
        MAX_16X4(out, inp);\
        MIN_16X4(out, max);\
}


#else

#define MAX_16X4(id1, id0) \
        b0 = AE_LT16(id1, id0); \
        AE_MOVT16X4(id1, id0, b0);

#define MIN_16X4(id1, id0) \
        b0 = AE_LT16(id1, id0); \
        AE_MOVF16X4(id1, id0, b0);

#define LIMIT(out, inp, min, max){\
        out = min;\
        MAX_16X4(out, inp);\
        MIN_16X4(out, max);\
}

#endif


#define MAX_WORD16 (int)0x00007fff
#define MIN_WORD16 (int)0xffff8000

/*
 * inp: p_vec: 2 byte aligned input pointer
 * out: p_out: 2 byte aligned input pointer */
#if (( XCHAL_HW_VERSION >= RI9_HWVERSION )& (XCHAL_HAVE_HIFI1))
WORD32 xa_nn_vec_activation_min_max_16_16(WORD16 * __restrict__ p_out,
                                      const  WORD16 * __restrict__ p_vec,
                                      int    activation_min,
                                      int    activation_max,
                                      WORD32 vec_length)
{
    int i;
    ae_int16x4 x, y, min, max;
    ae_valign align_src, align_dst;
#if !XCHAL_HAVE_HIFI1
    xtbool4 b0;
#endif
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

    WORD16 *p_o = p_out;
    WORD16 *p_v = (WORD16 *)p_vec;

    min  = AE_MOVDA16(activation_min);
    max  = AE_MOVDA16(activation_max);

    align_src = AE_LA64_PP((ae_int16x4 *)p_v);
    align_dst = AE_ZALIGN64(); // zero alignment reg

    if((activation_max >= MAX_WORD16) && (activation_min <= MIN_WORD16))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(x, align_src, (ae_int16x4 *)p_v);
            AE_SA16X4_IP(x, align_dst, (ae_int16x4 *)p_o);
        }
        int rem_itr = (vec_length & 3);
        {
            AE_LAV16X4_XP(x, align_src, (ae_int16x4 *)p_v, (rem_itr << 1));
            AE_SAV16X4_XP(x, align_dst, (ae_int16x4 *)p_o, (rem_itr << 1));
        }
        AE_SA64POS_FP(align_dst, p_o); // finalize the stream
    }
    else if((activation_max < MAX_WORD16) && (activation_min <= MIN_WORD16))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(x, align_src, (ae_int16x4 *)p_v);
			
            MIN_16X4(x, max)

            AE_SA16X4_IP(x, align_dst, (ae_int16x4 *)p_o);
        }

        int rem_itr = (vec_length & 3);
        {
            AE_LAV16X4_XP(y, align_src, (ae_int16x4 *)p_v, (rem_itr << 1));

            MIN_16X4(y, max)

            AE_SAV16X4_XP(y, align_dst, (ae_int16x4 *)p_o, (rem_itr << 1));
        }
        AE_SA64POS_FP(align_dst, p_o); // finalize the stream
    }
    else if((activation_max >= MAX_WORD16) && (activation_min > MIN_WORD16))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(y, align_src, (ae_int16x4 *)p_v);

            MAX_16X4(y, min)

            AE_SA16X4_IP(y, align_dst, (ae_int16x4 *)p_o);
        }

        int rem_itr = (vec_length & 3);
        {
            AE_LAV16X4_XP(y, align_src, (ae_int16x4 *)p_v, (rem_itr << 1));

            MAX_16X4(y ,min)

            AE_SAV16X4_XP(y, align_dst, (ae_int16x4 *)p_o, (rem_itr << 1));
        }
        AE_SA64POS_FP(align_dst, p_o); // finalize the stream
    }
    else
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(x, align_src, (ae_int16x4 *)p_v);
			
            LIMIT(y, x, min, max)
			
            AE_SA16X4_IP(y, align_dst, (ae_int16x4 *)p_o);
        }

        int rem_itr = (vec_length & 3);
        {
            AE_LAV16X4_XP(x, align_src, (ae_int16x4 *)p_v, (rem_itr << 1));
			
            LIMIT(y, x, min, max)
			
            AE_SAV16X4_XP(y, align_dst, (ae_int16x4 *)p_o, (rem_itr << 1));
        }
        AE_SA64POS_FP(align_dst, p_o); // finalize the stream

    }

    return 0;
}	
#else
WORD32 xa_nn_vec_activation_min_max_16_16(WORD16 * __restrict__ p_out,
                                      const  WORD16 * __restrict__ p_vec,
                                      int    activation_min,
                                      int    activation_max,
                                      WORD32 vec_length)
{
    int i;
    ae_int16x4 x, y, min, max;
    ae_valign align_src, align_dst;
#if !XCHAL_HAVE_HIFI1
    xtbool4 b0;
#endif

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

    WORD16 *p_o = p_out;
    WORD16 *p_v = (WORD16 *)p_vec;

    min  = AE_MOVDA16(activation_min);
    max  = AE_MOVDA16(activation_max);

    align_src = AE_LA64_PP((ae_int16x4 *)p_v);
    align_dst = AE_ZALIGN64(); // zero alignment reg

    if((activation_max >= MAX_WORD16) && (activation_min <= MIN_WORD16))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(x, align_src, (ae_int16x4 *)p_v);
            AE_SA16X4_IP(x, align_dst, (ae_int16x4 *)p_o);
        }

        AE_SA64POS_FP(align_dst, p_o); // finalize the stream

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L16_IP(x, (ae_int16 *)p_v, sizeof(ae_int16));
            AE_S16_0_IP(x, (ae_int16 *)p_o, sizeof(ae_int16));
        }
    }
    else if((activation_max < MAX_WORD16) && (activation_min <= MIN_WORD16))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(x, align_src, (ae_int16x4 *)p_v);

            //b0 = AE_LT16(x, max);
            //AE_MOVF16X4(x, max, b0);
            MIN_16X4(x, max)

            AE_SA16X4_IP(x, align_dst, (ae_int16x4 *)p_o);
        }

        AE_SA64POS_FP(align_dst, p_o); // finalize the stream

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L16_IP(y, (ae_int16 *)p_v, sizeof(ae_int16));

            //b0 = AE_LT16(y, max);
            //AE_MOVF16X4(y, max, b0);
            MIN_16X4(y, max)

            AE_S16_0_IP(y, (ae_int16 *)p_o, sizeof(ae_int16));
        }
    }
    else if((activation_max >= MAX_WORD16) && (activation_min > MIN_WORD16))
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(y, align_src, (ae_int16x4 *)p_v);

            //b0 = AE_LT16(y, min);
            //AE_MOVT16X4(y, min, b0);
            MAX_16X4(y, min)

            AE_SA16X4_IP(y, align_dst, (ae_int16x4 *)p_o);
        }

        AE_SA64POS_FP(align_dst, p_o); // finalize the stream

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L16_IP(y, (ae_int16 *)p_v, sizeof(ae_int16));

            //b0 = AE_LT16(y, min);
            //AE_MOVT16X4(y, min, b0);
            MAX_16X4(y ,min)

            AE_S16_0_IP(y, (ae_int16 *)p_o, sizeof(ae_int16));
        }
    }
    else
    {
        for(i=0; i<(vec_length >> 2); i++)
        {
            AE_LA16X4_IP(x, align_src, (ae_int16x4 *)p_v);
            LIMIT(y, x, min, max)
            AE_SA16X4_IP(y, align_dst, (ae_int16x4 *)p_o);
        }

        AE_SA64POS_FP(align_dst, p_o); // finalize the stream

        for(i=0; i < (vec_length & 3); i++)
        {
            AE_L16_IP(x, (ae_int16 *)p_v, sizeof(ae_int16));
            LIMIT(y, x, min, max)
            AE_S16_0_IP(y, (ae_int16 *)p_o, sizeof(ae_int16));
        }
    }

    return 0;
}
#endif

/*
 * ReLU 16-bit:
 */
WORD32 xa_nn_vec_relu_16_16(
    WORD16       * __restrict__ p_out,
    const WORD16 * __restrict__ p_vec,
    WORD16       threshold,
    WORD32       vec_length)
{
    xa_nn_vec_activation_min_max_16_16(p_out,
                                      p_vec,
                                      0,
                                      threshold,
                                      vec_length);

    return 0;
}
/*
 * ReLU Standard 16-bit:
 */
WORD32 xa_nn_vec_relu_std_16_16(
    WORD16       * __restrict__ p_out,
    const WORD16 * __restrict__ p_vec,
    WORD32       vec_length)
{

    xa_nn_vec_activation_min_max_16_16(p_out,
                                      p_vec,
                                      0,
                                      MAX_WORD16,
                                      vec_length);
    return 0;
}

#ifndef AE_SEL16_7351
#define AE_SEL16_7351(a,b) AE_SEL16_7520((AE_SEL16_7531((a),(b))), (AE_SEL16_7531((a),(b))))
#endif

#define AE_SLAA16S1(x, shift) AE_SEL16_7351(AE_MOVINT16X4_FROMINT32X2(AE_SLAA32(AE_MOVINT32X2_FROMINT64(AE_SRAI64(AE_MOVINT64_FROMINT16X4(x), 16)), 16+shift)), AE_MOVINT16X4_FROMINT32X2(AE_SLAA32(AE_MOVINT32X2_FROMINT16X4(x), 16+shift)))
#define AE_SLAI16S1(x, shift) AE_SEL16_7351(AE_MOVINT16X4_FROMINT32X2(AE_SLAA32(AE_MOVINT32X2_FROMINT64(AE_SRAI64(AE_MOVINT64_FROMINT16X4(x), 16)), 16+shift)), AE_MOVINT16X4_FROMINT32X2(AE_SLAA32(AE_MOVINT32X2_FROMINT16X4(x), 16+shift)))

#define CONSTANT_TERM             (0x70f6)
#define CONSTANT_1_OVER_3         (0x2aab)
#define CONSTANT_1_OVER_8         (0x1000)
#define Q15_ONE                   (0x7fff)
#define CONSTANT_48_OVER_17       (23130)
#define CONSTANT_NEG_32_OVER_17   (-15420)
#define F2_ONE                    (0x2000)


/* xa_nn_vec_sigmoid_16_16 */
#define ROUNDING_HALF_SUM_16X4(s0, s1, a, b){\
  /*AE_ADDW16(s0, s1, a, b);*/\
  s0 = AE_ADD32S(AE_SEXT32X2D16_32(a), AE_SEXT32X2D16_32(b)); \
  s1 = AE_ADD32S(AE_SEXT32X2D16_10(a), AE_SEXT32X2D16_10(b)); \
  /*AE_MULA16X4(s0, s1, b, ONE);*/ \
  /*s0 = AE_ADD32(AE_SEXT32X2D16_32(a),AE_SEXT32X2D16_32(b));*/ \
  /*s1 = AE_ADD32(AE_SEXT32X2D16_10(a),AE_SEXT32X2D16_10(b));*/ \
  /*AE_MULF2P32X4RS(s0, s1, s0, s1, AE_MOVDA32(1<<(31-1)), AE_MOVDA32(1<<(31-1)));*/\
  s0 = AE_MULFP32X2RS(s0, AE_MOVDA32(1<<(31-1))); \
  s1 = AE_MULFP32X2RS(s1, AE_MOVDA32(1<<(31-1))); \
  /*s0 = AE_SRAA32RS(s0, 1); */\
  /*s1 = AE_SRAA32RS(s1, 1); */\
}

#define EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_16X4(y_out1, inp1)\
{ \
  ae_int16x4 x1_in, x2, x3, x4, x4_by_4, y1, y2, y3, y4, y5, y6;\
  \
  x1_in = AE_ADD16S(inp1, CT_1_BY_8);\
  x2 = AE_MULFP16X4RAS(x1_in, x1_in);\
  x3 = AE_MULFP16X4RAS(x2, x1_in);\
  x4 = AE_MULFP16X4RAS(x2, x2);\
  x4_by_4 = AE_SRAI16R(x4, 2);\
  y1 = AE_ADD16S(x4_by_4, x3);\
  y2 = AE_MULFP16X4RAS(y1, CT_1_BY_3);\
  y3 = AE_ADD16S(y2, x2);\
  y4 = AE_SRAI16R(y3, 1);\
  \
  y5 = AE_ADD16S(x1_in, y4);\
  y6 = AE_MULFP16X4RAS(y5, CT);\
  y_out1 = AE_ADD16S(y6, CT);\
}

#define GEMMLOWP_EXP_BARREL_SHIFTER_16X4(out_1, fract_bits, exponent, FixedPointMultiplier, remainder1)\
{ \
  ae_int16x4 out2, mask1, scale;\
  ae_int16x4 d16_fpm;\
  xtbool4 bit_set;\
  \
  scale = AE_SLAA16S1(ONE, fract_bits + exponent);\
  \
  mask1 = AE_AND16(remainder1, scale);\
  bit_set = AE_LT16(zero16, mask1);\
  \
  d16_fpm = AE_MOVDA16(FixedPointMultiplier); \
  out2 = AE_MULFP16X4RAS(out_1, d16_fpm);\
  AE_MOVT16X4(out_1, out2, bit_set);\
}

#define EXP_INP16_Q12_X4(y1, inp1)\
{ \
  /*xtbool4 b; */\
  ae_int16x4 x_in1, x0, remainder1;\
  ae_int16x4 a_mod_quater_minus_q_1_by_4_first;\
  ae_int16x4 q_1_by_4 = AE_SLAI16S1(ONE, 12-2); /* 1/4 in Q12 */\
  \
  x0 = AE_AND16(inp1, AE_SUB16(q_1_by_4, ONE));\
  a_mod_quater_minus_q_1_by_4_first = AE_SUB16(x0, q_1_by_4);\
  x_in1 = AE_SLAI16S1(a_mod_quater_minus_q_1_by_4_first, 3);\
  \
  EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_16X4(y1, x_in1)\
  \
  remainder1 = AE_SUB16(a_mod_quater_minus_q_1_by_4_first, inp1);\
  \
  GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, 12, -2, (1672461947+32768)/65536, remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, 12, -1, (1302514674+32768)/65536, remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, 12,  0, (790015084+32768)/65536,   remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, 12,  1, (290630308+32768)/65536,   remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, 12,  2, (39332535+32768)/65536,    remainder1);\
  /* Input has only 3 integer bits (0th, 1st and 2nd), no need to check beyond 2nd */ \
  /*GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, 3, 720401,      remainder1);*/\
  \
  /* Input == 0 is handled outside so not needed here
  b = AE_EQ16(inp1, AE_ZERO16());\
  AE_MOVT16X4(y1, AE_MOVDA16(Q15), b); */ \
}

#define ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_16X4(y1, a1){\
  ae_int32x2 t1, t2;\
  ae_int16x4 max16;\
  ae_int16x4 half_den1234, m1, x1, half_denominator_times_x1;\
  ae_int16x4 one_minus_half_denominator_times_x1;\
  ae_int16x4 CT_48_by_17, CT_neg_32_by_17, CT_F2_ONE;\
  int itr_nr;\
  \
  CT_48_by_17 = AE_MOVDA16(CONSTANT_48_OVER_17);\
  CT_neg_32_by_17 = AE_MOVDA16(CONSTANT_NEG_32_OVER_17);\
  CT_F2_ONE = AE_MOVDA16(F2_ONE);\
  \
  max16 = AE_MOVDA16(Q15_ONE); \
  ROUNDING_HALF_SUM_16X4(t1, t2, a1, max16);\
  \
  half_den1234 = AE_SAT16X4(t1, t2);\
  \
  m1 = AE_MULFP16X4RAS(half_den1234, CT_neg_32_by_17);\
  x1 = AE_ADD16S(m1, CT_48_by_17);\
  \
  for(itr_nr=0; itr_nr<3; itr_nr++)\
  { \
    half_denominator_times_x1 = AE_MULFP16X4RAS(x1, half_den1234);\
    one_minus_half_denominator_times_x1 = AE_SUB16S(CT_F2_ONE, half_denominator_times_x1);\
    m1 = AE_MULFP16X4RAS(x1, one_minus_half_denominator_times_x1);\
    /* Shift m1 left by 2, saturate to 16-bit and add to x1 with saturation */\
    /*AE_MULAP16X16X4S(x1, m1, AE_MOVDA16(1<<2));*/ \
    x1 = AE_ADD16S(x1, AE_SLAI16S(m1, 2)); \
  }\
  \
  y1 = AE_SLAI16S(x1, 1);\
  \
}


WORD32 xa_nn_vec_sigmoid_16_16(WORD16 *p_out,         /* result, Q0.15     */
                               const WORD16 *p_vec,   /* input data, Q3.12 */
                               WORD32 vec_length)     /* length of vectors */
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);

#if defined(AE_SIGMOID16X4) && defined(USE_HIFI_ACT_TIE)
  int i;
  ae_int16x4 z10_op, z10;
  ae_valign align_in, align_out;

  const ae_int16x4 *p_in_0  = (const ae_int16x4 *)p_vec;
  ae_int16x4 *p_out_0 = (ae_int16x4 *)p_out;

  align_in = AE_LA64_PP(p_in_0);
  align_out = AE_ZALIGN64();

  WUR_AE_SAR(4);

#pragma concurrent
  for(i=0; i<(vec_length >> 2); i++)
  {
    AE_LA16X4_IP(z10, align_in, p_in_0);

    z10_op = AE_SIGMOID16X4(z10);
    z10_op = AE_SRAI16(z10_op, 1);
    z10_op = AE_AND16(z10_op, AE_MOVDA16(0x7fff));

    AE_SA16X4_IP(z10_op, align_out, p_out_0);
  }
  AE_SA64POS_FP(align_out, p_out_0);

  int rem_itr = vec_length & 3;
  // remainder loop
#pragma concurrent
#pragma loop_count max=3
  for(i = 0; i < rem_itr; i++)
  {
    z10 = ((ae_int16 *)p_in_0)[i];

    z10_op = AE_SIGMOID16X4(z10);
    z10_op = AE_SRAI16(z10_op, 1);
    z10_op = AE_AND16(z10_op, AE_MOVDA16(0x7fff));

    ((ae_int16 *)p_out_0)[i] = z10_op;
  }

#else /* #if defined(AE_SIGMOID16X4) && defined(USE_HIFI_ACT_TIE) */

#if !XCHAL_HAVE_HIFI1
    xtbool4 b0;
#endif
  
  int i;
  ae_int16x4 x3210, y3210;
  ae_int16x4 out0;
  ae_int16x4 zero16;
  ae_int16x4 CT_1_BY_8 = AE_MOVDA16(CONSTANT_1_OVER_8);
  ae_int16x4 CT_1_BY_3 = AE_MOVDA16(CONSTANT_1_OVER_3);
  ae_int16x4 CT = AE_MOVDA16(CONSTANT_TERM);
  ae_int16x4 ONE = AE_MOVDA16(1);
  xtbool4 is_neg0, is_zero0;
  ae_int16x4 exp_y3210, one_half;
  ae_int16x4 z3210;
  ae_valign align_src_hf5, align_dst_hf5;
  ae_valign align_src_hf5_1;
  
  const ae_int16x4 *p_in  = (const ae_int16x4 *)p_vec;
  const ae_int16x4 *p_in1  = (const ae_int16x4 *)p_vec;
  ae_int16x4 *p_o = (ae_int16x4 *)p_out;
  
  align_src_hf5 = AE_LA64_PP(p_in);
  align_src_hf5_1 = AE_LA64_PP(p_in1);
  align_dst_hf5 = AE_ZALIGN64();
  
  zero16 = AE_ZERO16();
  one_half = AE_SLAI16S1(ONE, 14);
  
#pragma concurrent
  for(i=0; i<(vec_length >> 2); i++)
  {
    AE_LA16X4_IP(x3210, align_src_hf5, p_in);
  
    /* Computing Negative value */
    y3210 = AE_NEG16S(x3210);
    MIN_16X4(y3210, x3210);
  
    /* Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x)) */
    EXP_INP16_Q12_X4(exp_y3210, y3210);
  
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_16X4(out0, exp_y3210);
  
    z3210 =  AE_SUB16(AE_MOVDA16(Q15_ONE), out0);
  
    /* This extra load gives a lot of improvement in loop cycles
     * by reducing register pressure
     */
    AE_LA16X4_IP(x3210, align_src_hf5_1, p_in1);
  
    is_neg0 = AE_LT16(x3210, zero16);
  
    AE_MOVT16X4(out0, z3210, is_neg0);
  
    is_zero0 = AE_EQ16(x3210, zero16);
    AE_MOVT16X4(out0, one_half, is_zero0);
  
    AE_SA16X4_IP(out0, align_dst_hf5, p_o);
  }
  AE_SA64POS_FP(align_dst_hf5, p_o);
  
  int rem_itr = (vec_length & 3);
  
  if(rem_itr > 0)
  {
    ae_int16 *p16_in, *p16_out;
    p16_in = (ae_int16 *)p_in;
    p16_out = (ae_int16 *)p_o;
    x3210 = p16_in[0];
    if(rem_itr > 1)
        x3210 = AE_SEL16_6543(x3210, p16_in[1]);
    if(rem_itr > 2)
        x3210 = AE_SEL16_6543(x3210, p16_in[2]);
    
    y3210 = AE_NEG16S(x3210);
    MIN_16X4(y3210, x3210);
    
    /* Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x)) */
    EXP_INP16_Q12_X4(exp_y3210, y3210);
    
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_16X4(y3210, exp_y3210);
    
    z3210 =  AE_SUB16(AE_MOVDA16(Q15_ONE), y3210);
    
    one_half = AE_SLAI16S1(ONE, 14);
    
    is_neg0 = AE_LT16(x3210, zero16);
    is_zero0 = AE_EQ16(x3210, zero16);
    
    AE_MOVT16X4(y3210, z3210, is_neg0);
    AE_MOVT16X4(y3210, one_half, is_zero0);

    if(rem_itr > 2)
    {
      p16_out[2] = y3210;
      y3210 = AE_SEL16_4321(y3210, y3210);
    }
    if(rem_itr > 1)
    {
      p16_out[1] = y3210;
      y3210 = AE_SEL16_4321(y3210, y3210);
    }
    p16_out[0] = y3210;
  }
#endif /* #if defined(AE_SIGMOID16X4) && defined(USE_HIFI_ACT_TIE) */

  return 0;
}

/* xa_nn_vec_tanh_16_16 */
#define EXP_2X_INP16X4(y1, fract_bits, inp1)\
{\
  /* exp(2x) so calculation happens in Q(fract_bits-1) format */ \
  /* xtbool4 b; */\
  ae_int16x4 x_in1, x0, remainder1;\
  ae_int16x4 a_mod_quater_minus_q_1_by_4_first;\
  ae_int16x4 q_1_by_4 = AE_SLAA16S1(ONE, (fract_bits-1)-2); /* 1/4 in Q(fract_bits+1) */\
  \
  x0 = AE_AND16(inp1, AE_SUB16(q_1_by_4, ONE));\
  a_mod_quater_minus_q_1_by_4_first = AE_SUB16(x0, q_1_by_4);\
  x_in1 = AE_SLAA16S1(a_mod_quater_minus_q_1_by_4_first, 15-(fract_bits-1));\
  \
  EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_16X4(y1, x_in1)\
  \
  remainder1 = AE_SUB16(a_mod_quater_minus_q_1_by_4_first, inp1);\
  \
  GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, (fract_bits-1), -2, (1672461947+32768)/65536, remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, (fract_bits-1), -1, (1302514674+32768)/65536, remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, (fract_bits-1),  0, (790015084+32768)/65536,  remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, (fract_bits-1),  1, (290630308+32768)/65536,  remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, (fract_bits-1),  2, (39332535+32768)/65536,   remainder1);\
  GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, (fract_bits-1),  3, (720401+32768)/65536,     remainder1);\
  /* (242+32768)/65536 is 0 so this is not needed in 16-bit implementation */\
  /* GEMMLOWP_EXP_BARREL_SHIFTER_16X4(y1, (fract_bits-1),  4, (242+32768)/65536,        remainder1);*/\
  /* for integet bits greater than 3 in input, 4 here because of exp(2x) */ \
  {\
    ae_int16x4 scale;\
    xtbool4 is_less;\
    scale = AE_SLAA16S(AE_MOVDA16(-1), fract_bits - 1 + 4);\
    is_less = AE_LT16(inp1, scale);\
    AE_MOVT16X4(y1, zero16, is_less);\
  }\
  \
  /* Input == 0 is handled outside so not needed here
  b = AE_EQ16(inp1, AE_ZERO16());\
  AE_MOVT16X4(y1, AE_MOVDA16(Q15), b); */ \
}

/*For Tanh*/
#define ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1(y1, a1){\
  ae_int32x2 t1, t2;\
  ae_int16x4 max16;\
  ae_int16x4 half_den1234, m1, x1, half_denominator_times_x1;\
  ae_int16x4 one_minus_half_denominator_times_x1;\
  ae_int16x4 CT_48_by_17, CT_neg_32_by_17, CT_F2_ONE;\
  int itr_nr;\
  \
  max16 = AE_MOVDA16(Q15_ONE); \
  ROUNDING_HALF_SUM_16X4(t1, t2, a1, max16);\
  \
  half_den1234 = AE_SAT16X4(t1, t2);\
  CT_48_by_17 = AE_MOVDA16(CONSTANT_48_OVER_17);\
  CT_neg_32_by_17 = AE_MOVDA16(CONSTANT_NEG_32_OVER_17);\
  CT_F2_ONE = AE_MOVDA16(F2_ONE);\
  \
  m1 = AE_MULFP16X4RAS(half_den1234, CT_neg_32_by_17);\
  x1 = AE_ADD16S(m1, CT_48_by_17);\
  \
  for(itr_nr=0; itr_nr<3; itr_nr++)\
  {\
    half_denominator_times_x1 = AE_MULFP16X4RAS(x1, half_den1234);\
    one_minus_half_denominator_times_x1 = AE_SUB16S(CT_F2_ONE, half_denominator_times_x1);\
    m1 = AE_MULFP16X4RAS(x1, one_minus_half_denominator_times_x1);\
    /* Shift m1 left by 2, saturate to 16-bit and add to x1 with saturation */\
    /*AE_MULAP16X16X4S(x1, m1, AE_MOVDA16(1<<2));*/ \
    x1 = AE_ADD16S(x1, AE_SLAI16S(m1, 2)); \
  }\
  \
  x1 = AE_SUB16S(x1, CT_F2_ONE);\
  y1 = AE_SLAI16S(x1, 2);\
  \
}

WORD32 xa_nn_vec_tanh_16_16(WORD16 *p_out,
                            const WORD16 *p_vec,
                            WORD32 integer_bits,    /* Number of integer bits to adjust Q-format */
                            WORD32 vec_length)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((integer_bits > 6), -1);

#if defined(AE_TANH16X4) && defined(USE_HIFI_ACT_TIE)
  int i;
  ae_int16x4 z10_op, z10;
  ae_valign align_in, align_out;

  const ae_int16x4 *p_in_0  = (const ae_int16x4 *)p_vec;
  ae_int16x4 *p_out_0 = (ae_int16x4 *)p_out;

  align_in = AE_LA64_PP(p_in_0);
  align_out = AE_ZALIGN64();

  if(integer_bits < 4)
  {
    WUR_AE_SAR(1+integer_bits);

#pragma concurrent
    for(i=0; i<(vec_length >> 2); i++)
    {
      AE_LA16X4_IP(z10, align_in, p_in_0);
      z10_op = AE_TANH16X4(z10);
      AE_SA16X4_IP(z10_op, align_out, p_out_0);
    }
    AE_SA64POS_FP(align_out, p_out_0);
    
    int rem_itr = vec_length & 3;
    // remainder loop
#pragma concurrent
#pragma loop_count max=3
    for(i = 0; i < rem_itr; i++)
    {
      z10 = ((ae_int16 *)p_in_0)[i];
      z10_op = AE_TANH16X4(z10);
      ((ae_int16 *)p_out_0)[i] = z10_op;
    }
  }
  else
  {
    WUR_AE_SAR(4); /* Allowed shift value is [0, 4], Thus, max value of integer bits = 3 */
    int lshift = integer_bits - 3; /* Left shift value before calling tanh to make integer_bits = 3 */
#pragma concurrent
    for(i=0; i<(vec_length >> 2); i++)
    {
      AE_LA16X4_IP(z10, align_in, p_in_0);
      z10 = AE_SLAA16S(z10, lshift);
      z10_op = AE_TANH16X4(z10);
      AE_SA16X4_IP(z10_op, align_out, p_out_0);
    }
    AE_SA64POS_FP(align_out, p_out_0);
    
    int rem_itr = vec_length & 3;
    // remainder loop
#pragma concurrent
#pragma loop_count max=3
    for(i = 0; i < rem_itr; i++)
    {
      z10 = ((ae_int16 *)p_in_0)[i];
      z10 = AE_SLAA16S(z10, lshift);
      z10_op = AE_TANH16X4(z10);
      ((ae_int16 *)p_out_0)[i] = z10_op;
    }
  }
#else /* #if defined(AE_TANH16X4) && defined(USE_HIFI_ACT_TIE) */

#if !XCHAL_HAVE_HIFI1
    xtbool4 b0;
#endif

  int i;
  int fract_bits = 16 - 1 - integer_bits;
  
  ae_int16x4 x0123, y0123, z0123;
  xtbool4 e0123, f0123;
  ae_int16x4 exp_y0123;
  ae_valign align_src_hf5, align_dst_hf5;
  ae_valign align_src_hf5_1;
  ae_int16x4 out1;

  /*Derived variables from constants*/
  ae_int16x4 ONE = AE_MOVDA16(1);
  ae_int16x4 zero16 = AE_ZERO16();
  ae_int16x4 CT_1_BY_8 = AE_MOVDA16(CONSTANT_1_OVER_8);
  ae_int16x4 CT_1_BY_3 = AE_MOVDA16(CONSTANT_1_OVER_3);
  ae_int16x4 CT = AE_MOVDA16(CONSTANT_TERM);

  const ae_int16x4 *p_in  = (const ae_int16x4 *)p_vec;
  const ae_int16x4 *p_in1  = (const ae_int16x4 *)p_vec;
  ae_int16x4 *p_o = (ae_int16x4 *)p_out;
  
  align_src_hf5 = AE_LA64_PP(p_in);
  align_src_hf5_1 = AE_LA64_PP(p_in1);
  align_dst_hf5 = AE_ZALIGN64();

#pragma concurrent
  for(i=0; i < (vec_length >> 2); i++)
  {
    AE_LA16X4_IP(x0123, align_src_hf5, p_in);

    y0123 = AE_NEG16S(x0123);
    MIN_16X4(y0123, x0123);

    EXP_2X_INP16X4(exp_y0123, fract_bits, y0123);

    ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1(z0123, exp_y0123);

    /* This extra load gives a lot of improvement in loop cycles
    * by reducing register pressure
    */
    AE_LA16X4_IP(x0123, align_src_hf5_1, p_in1);
    e0123 = AE_LT16(x0123, zero16);

    out1 = z0123;
    AE_MOVT16X4(out1, AE_NEG16S(z0123), e0123);

    /* Check if input = 0 */
    f0123 = AE_EQ16(x0123, zero16);
    AE_MOVT16X4(out1, AE_ZERO16(), f0123);

    AE_SA16X4_IP(out1, align_dst_hf5, (ae_int16x4 *)p_o);
  }
  AE_SA64POS_FP(align_dst_hf5, p_o);

  int rem_itr = vec_length & 3;
  /* remainder loop */
  if(rem_itr > 0)
  {
    ae_int16 *p16_in, *p16_out;
    p16_in = (ae_int16 *)p_in;
    p16_out = (ae_int16 *)p_o;
    x0123 = p16_in[0];
    if(rem_itr > 1)
        x0123 = AE_SEL16_6543(x0123, p16_in[1]);
    if(rem_itr > 2)
        x0123 = AE_SEL16_6543(x0123, p16_in[2]);

    y0123 = AE_NEG16S(x0123);
    MIN_16X4(y0123, x0123);

    EXP_2X_INP16X4(exp_y0123, fract_bits, y0123);

    ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1(z0123, exp_y0123);

    e0123 = AE_LT16(x0123, zero16);

    out1 = z0123;
    AE_MOVT16X4(out1, AE_NEG16S(z0123), e0123);

    /* Check if input = 0 */
    f0123 = AE_EQ16(x0123, zero16);
    AE_MOVT16X4(out1, AE_ZERO16(), f0123);

    if(rem_itr > 2)
    {
      p16_out[2] = out1;
      out1 = AE_SEL16_4321(out1, out1);
    }
    if(rem_itr > 1)
    {
      p16_out[1] = out1;
      out1 = AE_SEL16_4321(out1, out1);
    }
    p16_out[0] = out1;
  }
#endif /* #if defined(AE_TANH16X4) && defined(USE_HIFI_ACT_TIE) */

  return 0;
}
