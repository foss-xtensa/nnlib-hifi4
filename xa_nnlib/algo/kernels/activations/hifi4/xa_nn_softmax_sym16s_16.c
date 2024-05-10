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
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros.h"

WORD16 exp_lut[513] = {
      2,     2,     2,     2,     2,     2,     2,     2,
      2,     2,     2,     2,     2,     2,     2,     2,
      2,     2,     2,     2,     2,     2,     2,     2,
      2,     2,     2,     3,     3,     3,     3,     3,
      3,     3,     3,     3,     3,     3,     3,     3,
      3,     3,     3,     3,     4,     4,     4,     4,
      4,     4,     4,     4,     4,     4,     4,     4,
      4,     5,     5,     5,     5,     5,     5,     5,
      5,     5,     5,     6,     6,     6,     6,     6,
      6,     6,     6,     7,     7,     7,     7,     7,
      7,     7,     7,     8,     8,     8,     8,     8,
      8,     9,     9,     9,     9,     9,     9,    10,
     10,    10,    10,    10,    11,    11,    11,    11,
     11,    12,    12,    12,    12,    13,    13,    13,
     13,    14,    14,    14,    14,    15,    15,    15,
     16,    16,    16,    17,    17,    17,    18,    18,
     18,    19,    19,    19,    20,    20,    21,    21,
     21,    22,    22,    23,    23,    24,    24,    25,
     25,    26,    26,    27,    27,    28,    28,    29,
     29,    30,    30,    31,    32,    32,    33,    34,
     34,    35,    36,    36,    37,    37,    38,    39,
     40,    40,    42,    42,    43,    44,    45,    45,
     46,    47,    48,    49,    50,    51,    52,    53,
     54,    55,    56,    57,    59,    60,    60,    62,
     63,    65,    65,    67,    68,    69,    71,    73,
     74,    75,    77,    78,    80,    81,    83,    85,
     86,    88,    90,    92,    93,    95,    97,    99,
    101,   103,   105,   107,   109,   112,   114,   116,
    118,   121,   123,   126,   128,   131,   133,   135,
    139,   141,   144,   147,   149,   152,   155,   158,
    162,   165,   168,   171,   174,   178,   181,   185,
    189,   192,   196,   200,   204,   208,   212,   217,
    221,   225,   230,   234,   239,   243,   248,   253,
    258,   263,   268,   273,   279,   284,   290,   296,
    302,   308,   314,   320,   327,   333,   340,   346,
    353,   360,   366,   374,   381,   389,   397,   404,
    413,   421,   429,   437,   446,   455,   464,   473,
    482,   492,   501,   511,   522,   532,   543,   553,
    564,   575,   586,   598,   610,   622,   634,   646,
    659,   672,   685,   699,   713,   727,   741,   756,
    771,   786,   801,   817,   833,   850,   866,   884,
    901,   919,   937,   955,   974,   993,  1013,  1033,
   1053,  1074,  1095,  1117,  1139,  1161,  1184,  1207,
   1232,  1256,  1281,  1306,  1332,  1358,  1385,  1412,
   1440,  1468,  1497,  1527,  1557,  1587,  1619,  1651,
   1683,  1716,  1750,  1785,  1820,  1856,  1892,  1930,
   1968,  2006,  2046,  2087,  2128,  2170,  2212,  2256,
   2300,  2346,  2392,  2439,  2488,  2537,  2587,  2638,
   2690,  2743,  2796,  2852,  2908,  2966,  3024,  3084,
   3145,  3207,  3270,  3334,  3400,  3467,  3535,  3605,
   3677,  3749,  3822,  3898,  3975,  4053,  4133,  4214,
   4297,  4383,  4469,  4557,  4647,  4739,  4833,  4927,
   5024,  5124,  5225,  5328,  5433,  5541,  5649,  5761,
   5875,  5991,  6109,  6230,  6352,  6477,  6605,  6736,
   6868,  7004,  7141,  7282,  7427,  7572,  7722,  7874,
   8030,  8188,  8350,  8514,  8683,  8854,  9028,  9206,
   9387,  9572,  9762,  9954, 10151, 10351, 10555, 10763,
  10976, 11191, 11412, 11637, 11867, 12102, 12341, 12583,
  12831, 13085, 13342, 13606, 13874, 14148, 14427, 14711,
  15002, 15297, 15599, 15907, 16221, 16541, 16867, 17199,
  17539, 17884, 18237, 18597, 18964, 19338, 19719, 20108,
  20505, 20909, 21322, 21742, 22171, 22608, 23054, 23509,
  23973, 24445, 24928, 25419, 25921, 26432, 26953, 27485,
  28027, 28580, 29143, 29718, 30304, 30902, 31512, 32133,
  32767};

WORD16 one_over_one_plus_x_lut[513] = {
  32767, 32704, 32640, 32578, 32514, 32451, 32388, 32326,
  32264, 32202, 32141, 32079, 32018, 31957, 31896, 31835,
  31775, 31715, 31655, 31596, 31537, 31476, 31418, 31359,
  31301, 31242, 31184, 31127, 31069, 31011, 30954, 30897,
  30840, 30784, 30727, 30671, 30615, 30560, 30504, 30449,
  30394, 30339, 30283, 30229, 30175, 30121, 30067, 30013,
  29960, 29906, 29853, 29800, 29746, 29694, 29642, 29589,
  29537, 29486, 29434, 29382, 29331, 29280, 29229, 29177,
  29127, 29076, 29026, 28976, 28926, 28877, 28827, 28777,
  28728, 28679, 28630, 28581, 28532, 28484, 28436, 28388,
  28340, 28292, 28244, 28197, 28150, 28103, 28056, 28008,
  27962, 27915, 27869, 27823, 27777, 27731, 27685, 27640,
  27594, 27549, 27504, 27459, 27413, 27369, 27324, 27280,
  27236, 27192, 27148, 27104, 27060, 27016, 26973, 26930,
  26887, 26844, 26801, 26758, 26715, 26673, 26630, 26588,
  26546, 26504, 26463, 26421, 26380, 26338, 26297, 26255,
  26214, 26174, 26132, 26092, 26051, 26011, 25971, 25931,
  25891, 25851, 25811, 25772, 25732, 25693, 25653, 25614,
  25575, 25536, 25497, 25458, 25420, 25381, 25343, 25305,
  25267, 25229, 25191, 25153, 25116, 25078, 25041, 25003,
  24966, 24928, 24892, 24855, 24818, 24781, 24745, 24709,
  24672, 24636, 24600, 24564, 24528, 24492, 24457, 24421,
  24385, 24350, 24315, 24280, 24245, 24210, 24175, 24140,
  24105, 24070, 24036, 24002, 23967, 23933, 23899, 23865,
  23831, 23798, 23764, 23730, 23697, 23664, 23630, 23597,
  23564, 23530, 23498, 23465, 23432, 23399, 23366, 23334,
  23302, 23269, 23237, 23205, 23173, 23141, 23109, 23077,
  23046, 23014, 22982, 22951, 22920, 22888, 22857, 22826,
  22795, 22764, 22733, 22703, 22672, 22641, 22611, 22580,
  22550, 22520, 22490, 22459, 22429, 22400, 22370, 22340,
  22310, 22281, 22251, 22221, 22192, 22163, 22134, 22104,
  22075, 22046, 22017, 21988, 21959, 21931, 21902, 21874,
  21845, 21817, 21788, 21760, 21732, 21704, 21676, 21648,
  21620, 21592, 21565, 21537, 21509, 21482, 21455, 21427,
  21400, 21372, 21345, 21318, 21291, 21264, 21237, 21210,
  21183, 21157, 21130, 21103, 21077, 21050, 21024, 20998,
  20971, 20945, 20919, 20893, 20867, 20841, 20816, 20790,
  20764, 20738, 20713, 20687, 20662, 20636, 20611, 20586,
  20560, 20535, 20510, 20485, 20460, 20435, 20410, 20385,
  20360, 20336, 20311, 20287, 20262, 20238, 20213, 20189,
  20165, 20141, 20117, 20092, 20068, 20044, 20021, 19997,
  19973, 19949, 19926, 19902, 19878, 19855, 19832, 19808,
  19784, 19762, 19738, 19715, 19692, 19668, 19645, 19622,
  19600, 19577, 19553, 19531, 19508, 19485, 19463, 19440,
  19418, 19395, 19373, 19351, 19328, 19306, 19284, 19262,
  19240, 19218, 19196, 19174, 19152, 19130, 19109, 19087,
  19065, 19044, 19022, 19000, 18979, 18958, 18936, 18915,
  18893, 18872, 18851, 18830, 18809, 18787, 18766, 18745,
  18725, 18704, 18682, 18662, 18641, 18620, 18600, 18579,
  18559, 18538, 18518, 18497, 18477, 18457, 18436, 18416,
  18396, 18376, 18356, 18336, 18316, 18296, 18276, 18256,
  18236, 18216, 18197, 18177, 18157, 18138, 18118, 18099,
  18079, 18059, 18040, 18021, 18001, 17982, 17963, 17944,
  17924, 17905, 17886, 17867, 17848, 17829, 17810, 17791,
  17772, 17754, 17735, 17716, 17697, 17679, 17660, 17641,
  17623, 17604, 17586, 17568, 17549, 17531, 17513, 17494,
  17476, 17458, 17440, 17422, 17404, 17386, 17368, 17350,
  17332, 17314, 17296, 17278, 17261, 17243, 17225, 17208,
  17190, 17172, 17155, 17137, 17120, 17102, 17085, 17067,
  17050, 17033, 17015, 16999, 16981, 16964, 16947, 16930,
  16913, 16895, 16878, 16862, 16845, 16828, 16810, 16794,
  16777, 16760, 16743, 16727, 16710, 16693, 16677, 16660,
  16644, 16627, 16611, 16594, 16578, 16562, 16545, 16529,
  16513, 16497, 16480, 16464, 16448, 16432, 16416, 16400,
  16384};

static inline ae_int16x4 LUTLookUpX4(ae_int16x4 value, WORD16* lut)
{
  ae_int16x4 shifted_value = AE_SRAI16(value, 7);
  ae_int16x4 index = AE_ADD16S(AE_MOVDA16(256), shifted_value);
  ae_int16x4 offset_ls8 = AE_SLAI16S(AE_AND16(value, AE_MOVDA16(0x7f)), 8);

  WORD32 index0, index1, index2, index3;
  index0 = AE_MOVAD16_3(index);
  index1 = AE_MOVAD16_2(index);
  index2 = AE_MOVAD16_1(index);
  index3 = AE_MOVAD16_0(index);

  ae_int16 *p_ae_lut = (ae_int16 *)lut;
  ae_int16x4 base0123 = p_ae_lut[index0];
  base0123 = AE_SEL16_6543(base0123, p_ae_lut[index1]);
  base0123 = AE_SEL16_6543(base0123, p_ae_lut[index2]);
  base0123 = AE_SEL16_6543(base0123, p_ae_lut[index3]);

  ae_int16x4 slope0123 = p_ae_lut[index0 + 1];
  slope0123 = AE_SEL16_6543(slope0123, p_ae_lut[index1 + 1]);
  slope0123 = AE_SEL16_6543(slope0123, p_ae_lut[index2 + 1]);
  slope0123 = AE_SEL16_6543(slope0123, p_ae_lut[index3 + 1]);
  slope0123 = AE_SUB16S(slope0123, base0123);

  ae_int16x4 delta0123;
  delta0123 = AE_MULFP16X4RAS(slope0123, offset_ls8);
  ae_int16x4 result0123 = AE_ADD16S(base0123, delta0123);

  return result0123;
}

static inline ae_int16x4 LUTLookUp(ae_int16x4 value, WORD16* lut)
{
  ae_int16x4 shifted_value = AE_SRAI16(value, 7);
  ae_int16x4 index = AE_ADD16S(AE_MOVDA16(256), shifted_value);
  ae_int16x4 offset_ls8 = AE_SLAI16S(AE_AND16(value, AE_MOVDA16(0x7f)), 8);

  WORD32 index0;
  index0 = AE_MOVAD16_0(index);

  ae_int16 *p_ae_lut = (ae_int16 *)lut;
  ae_int16x4 base = p_ae_lut[index0];

  ae_int16x4 slope = p_ae_lut[index0 + 1];
  slope = AE_SUB16S(slope, base);

  ae_int16x4 delta;
  delta = AE_MULFP16X4RAS(slope, offset_ls8);
  ae_int16x4 result = AE_ADD16S(base, delta);

  return result;
}

// Computes exp(input - max_input)

#define SOFTMAX_CALCULATE_EXP(result, left_shift, right_shift, multiplier, d_inp, max_in_row) \
{ \
  ae_int32x2 input_diff1, input_diff2, scaled_diff1, scaled_diff2; \
  ae_int32x2 sym_scaled_diff1, sym_scaled_diff2; \
  ae_int16x4 d_one16 = AE_MOVDA16(1); \
  AE_MUL16X4(input_diff1, input_diff2, d_inp, d_one16); \
  AE_MULS16X4(input_diff1, input_diff2, max_in_row, d_one16); \
  MPY_BY_QUANT_MULT_SLS_X2_OUT32(scaled_diff1, input_diff1, input_beta_multiplier, left_shift, right_shift); \
  MPY_BY_QUANT_MULT_SLS_X2_OUT32(scaled_diff2, input_diff2, input_beta_multiplier, left_shift, right_shift); \
  ae_int32x2 max_int16s = AE_MOVDA32(32767); \
  sym_scaled_diff1 = AE_ADD32S(scaled_diff1, max_int16s); \
  sym_scaled_diff2 = AE_ADD32S(scaled_diff2, max_int16s); \
  ae_int16x4 sat_sym_shifted_sum = AE_SAT16X4(sym_scaled_diff1, sym_scaled_diff2); \
  result = LUTLookUpX4(sat_sym_shifted_sum, exp_lut); \
} \

/* In Matlab this taking input as uint32_t hence need +1 with AE_NSAZ32_L */
WORD32 count_leading_zeros(ae_int32x2 integer_input)
{
  WORD32 value = AE_MOVDA32(integer_input);
  if(value == 0)
  {
    return 32;
  }
  return AE_NSAZ32_L(integer_input) + 1;
}

WORD32 xa_nn_vec_softmax_sym16s_16( WORD16 * __restrict__ p_out,
                    const   WORD16 * __restrict__ p_vec,
                            WORD32  input_beta_left_shift,
                            WORD32  input_beta_multiplier,
                            WORD32  vec_length)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_beta_left_shift < -31) || (input_beta_left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((input_beta_multiplier < 0), -1);

  // Calculating Max
  ae_int16x4 d_max;
  int i;
  {
    ae_int16x4 d0;
    xtbool4 b4;
    ae_int16x4 *p_inp = (ae_int16x4 *)p_vec;
    ae_valign align_inp = AE_LA64_PP(p_inp);
    d_max = AE_MOVDA16(0x8000);

    for(i = 0; i < (vec_length >> 2); i++)
    {
      AE_LA16X4_IP(d0, align_inp, p_inp);
      b4 = AE_LT16(d_max, d0);
      AE_MOVT16X4(d_max, d0, b4);
    }
    {
      d0 = AE_SEL16_5432(d_max, d_max);
      b4 = AE_LT16(d_max, d0);
      AE_MOVT16X4(d_max, d0, b4);
      d0 = AE_SEL16_6543(d_max, d_max);
      b4 = AE_LT16(d_max, d0);
      AE_MOVT16X4(d_max, d0, b4);
    }

    for(i = 0; i < (vec_length & 3); i++)
    {
      AE_L16_IP(d0, (ae_int16 *)p_inp, sizeof(ae_int16));
      b4 = AE_LT16(d_max, d0);
      AE_MOVT16X4(d_max, d0, b4);
    }
  }

#if TFLITE_SINGLE_ROUNDING
  int left_shift  = input_beta_left_shift;
  int right_shift = input_beta_left_shift;
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int left_shift  = input_beta_left_shift < 0 ? 0 : input_beta_left_shift;
  int right_shift = input_beta_left_shift > 0 ? 0 : -input_beta_left_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  //Compute exp and sum_of_exp
  ae_int32x2 sum_of_exps;
  {
    ae_int16x4 *temp_out = (ae_int16x4 *)p_out;
    ae_int16x4 *p_inp = (ae_int16x4 *)p_vec;
    ae_int16x4 d_inp1;
    ae_valign align_input64 = AE_LA64_PP(p_inp);
    ae_int32x2 acc1, acc2, acc;
    acc1 = AE_ZERO32();
    acc2 = AE_ZERO32();
    ae_valign align_output64 = AE_ZALIGN64();
    ae_int16x4 exp1;
    for(i = 0; i < (vec_length >> 2); i++)
    {
      AE_LA16X4_IP(d_inp1, align_input64, p_inp);
      SOFTMAX_CALCULATE_EXP(exp1, left_shift, right_shift, input_beta_multiplier, d_inp1, d_max);
      AE_SA16X4_IP(exp1, align_output64, temp_out);
      AE_MULA16X4(acc1, acc2, exp1, AE_MOVDA16(1));
    }
    AE_SA64POS_FP(align_output64,(void *)temp_out);

    int rem_length = vec_length & 3;

    {
      d_inp1 = AE_ZERO16();
      ae_int16x4 d_exp_and = d_inp1;
      for(i = 0; i < rem_length; i++)
      {
        ae_int16x4 d_tmp;
        AE_L16_IP(d_tmp, (ae_int16 *)p_inp, 2);
        d_inp1 = AE_SEL16_6543(d_inp1, d_tmp);
        d_exp_and = AE_SEL16_6543(d_exp_and, AE_MOVDA16(0xffff));
      }
      SOFTMAX_CALCULATE_EXP(exp1, left_shift, right_shift, input_beta_multiplier, d_inp1, d_max);
      exp1 = AE_AND16(exp1, d_exp_and);
      AE_MULA16X4(acc1, acc2, exp1, AE_MOVDA16(1));
      ae_int16 *temp_out_ua = (ae_int16 *)temp_out + (rem_length - 1);
      for(i = 0; i < rem_length; i++)
      {
        AE_S16_0_IP(exp1, (ae_int16 *)temp_out_ua, -2);
        exp1 = AE_SEL16_4321(exp1, exp1);
      }
    }

    acc = AE_ADD32S(acc1, acc2);
    sum_of_exps = AE_ADD32S(acc, AE_SEL32_LH(acc, acc));
  }

  // Calculate 1/sum_of_exps
  WORD32 headroom_plus_one = count_leading_zeros(sum_of_exps);
  ae_int32x2 shifted_sum = AE_SRAA32RS(sum_of_exps, 14 - (headroom_plus_one - 1));
  ae_int32x2 plus_one_sym = AE_MOVDA32(-((1<<15) + (1<<16)));
  ae_int32x2 sym_shifted_sum = AE_ADD32S(shifted_sum, plus_one_sym);
  ae_int16x4 sat_sym_shifted_sum = AE_SAT16X4(sym_shifted_sum, sym_shifted_sum);
  ae_int16x4 reciprocal_scale_q015 = LUTLookUp(sat_sym_shifted_sum, one_over_one_plus_x_lut);

  // Compute exp*1/sum_of_exps
  {
    ae_int16x4 *temp_out1 = (ae_int16x4 *)p_out;
    WORD32 right_shift = 31 - headroom_plus_one;
    ae_int16x4 exp1;
    ae_valign exp_align = AE_LA64_PP(temp_out1);
    ae_int32x2 sfmx1, sfmx2;
    ae_int32x2 shifted_sfmx1, shifted_sfmx2;
    ae_int16x4 sfmx12;
    ae_int16x4 *temp_out2 = (ae_int16x4 *)p_out;
    ae_valign align_output = AE_ZALIGN64();
    ae_int32x2 zero32 = AE_ZERO32();

    for(i=0; i<(vec_length >> 2); i++)
    {
      AE_LA16X4_IP(exp1, exp_align, temp_out1);
      AE_MUL16X4(sfmx1, sfmx2, exp1, reciprocal_scale_q015);
      shifted_sfmx1 = AE_SRAA32RS(sfmx1, right_shift);
      shifted_sfmx2 = AE_SRAA32RS(sfmx2, right_shift);
      shifted_sfmx1 = AE_MAX32(shifted_sfmx1, zero32);
      shifted_sfmx2 = AE_MAX32(shifted_sfmx2, zero32);
      sfmx12 = AE_SAT16X4(shifted_sfmx1, shifted_sfmx2);
      AE_SA16X4_IP(sfmx12, align_output, temp_out2);
    }
    AE_SA64POS_FP(align_output, (void *)temp_out2);
    int rem_length = vec_length & 3;
    for(i = 0; i < rem_length; i++)
    {
      AE_L16_IP(exp1, (ae_int16 *)temp_out1, 2);
      AE_MUL16X4(sfmx1, sfmx2, exp1, reciprocal_scale_q015);
      shifted_sfmx1 = AE_SRAA32RS(sfmx1, right_shift);
      shifted_sfmx1 = AE_MAX32(shifted_sfmx1, zero32);
      sfmx12 = AE_SAT16X4(shifted_sfmx1, shifted_sfmx1);
      AE_S16_0_IP(sfmx12, (ae_int16 *)temp_out2, 2);
    }
  }

  return 0;
}
